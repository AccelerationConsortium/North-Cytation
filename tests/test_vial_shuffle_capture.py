"""Hardware-free checks for collection, pairing, collision safety and recovery."""
import json
import logging
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows import vial_shuffle_capture_workflow as workflow
from workflows._vial_shuffle import (
    RACK, apply_transfers, available_slots, eligible_names, plan_shuffle, positions,
    validate_layout,
)
from tools.c110_camera import capture_photo


class Robot:
    def __init__(self):
        self.VIAL_DF = pd.DataFrame([
            dict(vial_index=i, vial_name=f"v{i}", location=RACK, location_index=i,
                 vial_type="8_mL" if i != 9 else "20_mL", capped=i != 8)
            for i in range(10)
        ])
        self.VIAL_POSITIONS = {
            name: dict(rack_present=True, vials_movable=True, rack_size=size)
            for name, size in ((RACK, 12), ("heater", 4), ("photoreactor_array", 1), ("clamp", 1))
        }
        self.GRIPPER_STATUS = None
        self.HELD_PIPET_TYPE = None
        self.logger = logging.getLogger("shuffle_test")
        self.logger.addHandler(logging.NullHandler())
        self.logger.propagate = False
        self.moves = []
        self.homed = False

    def move_vial_to_location(self, name, location, index):
        assert (location, index) not in positions(self.VIAL_DF).values()
        assert name in eligible_names(self.VIAL_DF)
        self.homed = False
        self.moves.append((name, location, index))
        mask = self.VIAL_DF.vial_name == name
        self.VIAL_DF.loc[mask, "location"] = location
        self.VIAL_DF.loc[mask, "location_index"] = index

    def move_home(self):
        self.homed = True


def config(**overrides):
    result = {key: getattr(workflow, key) for key in workflow._CONFIG_KEYS}
    result.update(ITERATIONS=5, MIN_OFF_RACK=1, MAX_OFF_RACK=4, SETTLE_SECONDS=0,
                  RANDOM_SEED=11)
    result.update(overrides)
    return result


class ShuffleTests(unittest.TestCase):
    def test_500_shuffles_preserve_fixed_vials_and_bounds_then_restore(self):
        robot = Robot()
        original = positions(robot.VIAL_DF)
        slots = available_slots(robot)
        names = validate_layout(robot.VIAL_DF, slots)
        rng = random.Random(23)
        for _ in range(500):
            before = positions(robot.VIAL_DF)
            targets = plan_shuffle(before, names, slots, 3, 8, 1, 4, rng)
            self.assertTrue(3 <= len(targets) <= 8)
            self.assertTrue(all(before[name] != slot for name, slot in targets.items()))
            apply_transfers(robot, targets, slots)
            after = positions(robot.VIAL_DF)
            self.assertTrue(1 <= sum(after[name][0] != RACK for name in names) <= 4)
            self.assertEqual(after["v8"], original["v8"])
            self.assertEqual(after["v9"], original["v9"])
        apply_transfers(robot, {name: original[name] for name in names}, slots)
        self.assertEqual(positions(robot.VIAL_DF), original)

    def test_swap_uses_empty_buffer(self):
        robot = Robot()
        slots = available_slots(robot)
        before = positions(robot.VIAL_DF)
        count = apply_transfers(robot, {"v0": before["v1"], "v1": before["v0"]}, slots)
        self.assertEqual(count, 3)
        self.assertEqual(positions(robot.VIAL_DF)["v0"], before["v1"])

    def test_collection_saves_matching_snapshots_then_restores(self):
        robot = Robot()
        original = robot.VIAL_DF.copy(deep=True)
        coordinator = SimpleNamespace(nr_robot=robot, simulate=True, logger=robot.logger)
        attempts = []

        def capture(path, *_):
            self.assertTrue(robot.homed)
            attempts.append(path)
            if len(attempts) == 1:
                raise TimeoutError("transient camera failure")
            # A stand-in stores the state at exposure so we can compare the pair.
            robot.VIAL_DF.to_csv(path, index=False)

        def verifier(photo, csv):
            self.assertEqual(photo.read_text(), csv.read_text())
            return {"matched": int(photo.stem) % 2 == 1, "positions": {}}

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "collection"
            summary = workflow.run_collection(coordinator, config(), output,
                                              capture=capture, verifier=verifier)
            self.assertEqual(summary["images_saved"], 5)
            self.assertEqual(summary["verified"], 5)
            self.assertEqual(summary["matches"], 3)
            self.assertEqual(summary["mismatches"], 2)
            self.assertEqual(len(attempts), 6)
            self.assertEqual(len(list(output.glob("*.jpg"))), 5)
            self.assertEqual(json.loads((output / "summary.json").read_text())["status"], "completed")
            self.assertTrue(summary["restored"])
        pd.testing.assert_frame_equal(robot.VIAL_DF, original)
        self.assertTrue(robot.homed)

    def test_capture_exhaustion_stops_without_more_moves_or_partial_pair(self):
        robot = Robot()
        coordinator = SimpleNamespace(nr_robot=robot, simulate=True, logger=robot.logger)
        states = []

        def capture(path, *_):
            states.append(positions(robot.VIAL_DF))
            path.write_bytes(b"partial image")
            raise TimeoutError("camera unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "collection"
            with self.assertRaises(TimeoutError):
                workflow.run_collection(coordinator, config(), output, capture=capture)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["images_saved"], 0)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(len(states), 3)
            self.assertTrue(all(state == states[0] for state in states))
            self.assertEqual(list(output.glob("*.jpg")), [])
            self.assertFalse((output / "000001.csv").exists())

    def test_verification_errors_are_counted_without_losing_pairs(self):
        robot = Robot()
        coordinator = SimpleNamespace(nr_robot=robot, simulate=True, logger=robot.logger)
        with tempfile.TemporaryDirectory() as temporary:
            result = workflow.run_collection(
                coordinator, config(ITERATIONS=1), Path(temporary) / "run",
                capture=lambda path, *_: path.write_bytes(b"photo"),
                verifier=Mock(side_effect=RuntimeError("model offline")),
            )
        self.assertEqual(result["images_saved"], 1)
        self.assertEqual(result["verified"], 0)
        self.assertEqual(result["verification_errors"], 1)
        self.assertTrue(result["restored"])

    def test_impossible_bounds_rejected_before_motion(self):
        robot = Robot()
        with self.assertRaises(ValueError):
            workflow._validate(config(MIN_OFF_RACK=9, MAX_OFF_RACK=10), robot)
        self.assertEqual(robot.moves, [])

    def test_capture_does_not_call_detection(self):
        camera = SimpleNamespace(C110FrameCapture=Mock())
        camera.C110FrameCapture.return_value.capture_frames.return_value = [object()]
        cv2 = SimpleNamespace(imwrite=Mock(return_value=True))
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(sys.modules, {"camera_frame_capture": camera, "cv2": cv2}):
                with patch("tools.c110_camera.detect_photo") as detect:
                    capture_photo(Path(temporary) / "photo.jpg", temporary)
                    detect.assert_not_called()
                    camera.C110FrameCapture.return_value.capture_frames.assert_called_once_with(count=1, timeout=20)

    def test_simulation_writes_real_images_without_camera_or_slack(self):
        import cv2

        robot = Robot()
        coordinator = SimpleNamespace(nr_robot=robot, simulate=True, logger=robot.logger)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "simulation"
            with patch.object(workflow, "capture_photo") as real_capture:
                with patch.dict(sys.modules, {"slack_agent": SimpleNamespace(safe_send_slack_message=Mock())}):
                    summary = workflow.run_collection(coordinator, config(ITERATIONS=1), output)
                    real_capture.assert_not_called()
                    sys.modules["slack_agent"].safe_send_slack_message.assert_not_called()
            self.assertIsNotNone(cv2.imread(str(output / "000001.jpg")))
            self.assertFalse(summary["verification_enabled"])
            self.assertEqual(summary["images_saved"], 1)

    def test_csv_save_failure_does_not_publish_photo(self):
        robot = Robot()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    workflow._capture_pair(
                        robot, output, 1, config(),
                        lambda path, *_: path.write_bytes(b"captured image"),
                    )
            self.assertFalse((output / "000001.jpg").exists())
            self.assertFalse((output / "000001.csv").exists())

    def test_slack_summary_includes_verification_counts(self):
        coordinator = SimpleNamespace(simulate=False, logger=Mock())
        slack = SimpleNamespace(safe_send_slack_message=Mock(return_value=True))
        summary = dict(status="completed", images_saved=10, requested=10, restored=True,
                       verification_enabled=True, verified=9, matches=8, mismatches=1,
                       verification_errors=1)
        with patch.dict(sys.modules, {"slack_agent": slack}):
            workflow._notify(coordinator, summary)
        message = slack.safe_send_slack_message.call_args.args[0]
        self.assertIn("images=10/10", message)
        self.assertIn("vision matches=8/9", message)
        self.assertIn("verification errors=1", message)

    def test_waits_between_samples_at_home_without_initial_or_final_wait(self):
        robot = Robot()
        coordinator = SimpleNamespace(nr_robot=robot, simulate=False, logger=robot.logger)
        events = []
        now = [0.0]

        def capture(path, *_):
            self.assertTrue(robot.homed)
            events.append("capture")
            now[0] += 120.0
            path.write_bytes(b"photo")

        def sleep(seconds):
            self.assertTrue(robot.homed)
            if seconds > 0:
                events.append(seconds)
                now[0] += seconds

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(workflow.time, "sleep", side_effect=sleep), patch.object(workflow.time, "monotonic", side_effect=lambda: now[0]):
                with patch.object(workflow, "_notify"):
                    summary = workflow.run_collection(
                        coordinator,
                        config(SIMULATE=False, ITERATIONS=3, TARGET_EXPERIMENT_TIME_HOURS=0.144),
                        Path(temporary) / "run", capture=capture,
                    )
        self.assertAlmostEqual(summary["sample_interval_seconds"], 172.8)
        self.assertEqual([event for event in events if event == "capture"], ["capture"] * 3)
        self.assertEqual(events[0], "capture")
        self.assertEqual(events[-1], "capture")
        self.assertAlmostEqual(sum(event for event in events if isinstance(event, float)), 105.6)
        self.assertTrue(all(event <= 60 for event in events if isinstance(event, float)))
        self.assertTrue(summary["restored"])

    def test_simulation_skips_long_wait(self):
        coordinator = SimpleNamespace(simulate=True, logger=Mock())
        with patch.object(workflow.time, "sleep") as sleep:
            workflow._wait_between_samples(coordinator, 24 * 3600 / 500)
            sleep.assert_not_called()
        coordinator.logger.info.assert_called_once()

    def test_invalid_target_lengths_rejected_before_motion(self):
        for length in (-1, float("nan"), float("inf"), True, "24h", 1e308):
            with self.subTest(length=length):
                robot = Robot()
                with self.assertRaises(ValueError):
                    workflow._validate(config(TARGET_EXPERIMENT_TIME_HOURS=length), robot)
                self.assertEqual(robot.moves, [])

    def test_overruns_skip_waits_and_keep_every_sample_on_anchored_schedule(self):
        robot = Robot()
        coordinator = SimpleNamespace(nr_robot=robot, simulate=False, logger=robot.logger)
        now = [0.0]
        durations = iter([120.0, 240.0, 20.0, 20.0])

        def capture(path, *_):
            now[0] += next(durations)
            path.write_bytes(b"photo")

        def sleep(seconds):
            self.assertTrue(robot.homed)
            now[0] += seconds

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(workflow.time, "monotonic", side_effect=lambda: now[0]), patch.object(workflow.time, "sleep", side_effect=sleep), patch.object(workflow, "_notify"):
                summary = workflow.run_collection(
                    coordinator,
                    config(SIMULATE=False, ITERATIONS=4, TARGET_EXPERIMENT_TIME_HOURS=0.192),
                    Path(temporary) / "run", capture=capture,
                )
        self.assertEqual(summary["images_saved"], 4)
        self.assertEqual(summary["status"], "completed")
        for pair, wait in zip(summary["pairs"], [52.8, 0.0, 85.6, 0.0]):
            self.assertAlmostEqual(pair["wait_seconds"], wait)
        self.assertAlmostEqual(summary["pairs"][1]["schedule_lag_seconds"], 67.2)
        self.assertAlmostEqual(summary["elapsed_seconds"], 538.4)
        self.assertTrue(summary["restored"])

    def test_consistently_slow_cycles_finish_after_target_without_dropping_samples(self):
        robot = Robot()
        coordinator = SimpleNamespace(nr_robot=robot, simulate=False, logger=robot.logger)
        now = [0.0]

        def capture(path, *_):
            now[0] += 240.0
            path.write_bytes(b"photo")

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(workflow.time, "monotonic", side_effect=lambda: now[0]), patch.object(workflow.time, "sleep") as sleep, patch.object(workflow, "_notify"):
                summary = workflow.run_collection(
                    coordinator,
                    config(SIMULATE=False, ITERATIONS=3, TARGET_EXPERIMENT_TIME_HOURS=0.144),
                    Path(temporary) / "run", capture=capture,
                )
        self.assertEqual(summary["images_saved"], 3)
        self.assertEqual(summary["elapsed_seconds"], 720.0)
        self.assertGreater(summary["elapsed_seconds"], 0.144 * 3600)
        self.assertTrue(all(pair["wait_seconds"] == 0 for pair in summary["pairs"]))
        self.assertTrue(all(call.args[0] == 0 for call in sleep.call_args_list))
        self.assertTrue(summary["restored"])


if __name__ == "__main__":
    unittest.main()

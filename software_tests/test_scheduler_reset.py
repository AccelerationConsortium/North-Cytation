import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scheduler.reset_runner import run_reset
from scheduler.live_runner import fingerprint


class ResetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.vials = self.folder / "vials.csv"
        self.vials.write_text("vial_index,location\n0,clamp\n")
        self.state = self.folder / "robot.yaml"
        self.state.write_text("gripper_status: Vial\ngripper_vial_index: 0\n")
        self.job = {"vial_file": str(self.vials), "robot_state": str(self.state),
                    "input_hashes": fingerprint([self.vials, self.state])}
        (self.folder / "input.json").write_text(json.dumps(self.job))
        lock = patch("scheduler.reset_runner.scheduler_live_lock", return_value=nullcontext())
        self.lock = lock.start()
        self.addCleanup(lock.stop)
        self.environment = patch.dict("os.environ", {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        output = patch("builtins.print")
        output.start()
        self.addCleanup(output.stop)

    def test_initializes_live_robot_with_selected_vial_file_and_no_second_reset(self):
        robot = Mock()
        robot.GRIPPER_STATUS = robot.GRIPPER_VIAL_INDEX = robot.HELD_PIPET_TYPE = None
        robot.PIPET_FLUID_VIAL_INDEX = None
        robot.PIPET_FLUID_VOLUME = 0
        factory = Mock(return_value=SimpleNamespace(nr_robot=robot))
        result = run_reset(self.folder, factory=factory)
        self.assertTrue(result["completed"], result)
        self.assertEqual(factory.call_args.kwargs["vial_file"], str(self.vials))
        self.assertIs(factory.call_args.kwargs["simulate"], False)
        self.assertIs(factory.call_args.kwargs["initialize_track"], False)
        self.assertIs(factory.call_args.kwargs["initialize_biotek"], False)
        self.assertIs(factory.call_args.kwargs["show_gui"], False)
        robot.reset_after_initialization.assert_not_called()
        robot.save_robot_status.assert_called_once()

    def test_changed_inputs_block_hardware_initialization(self):
        self.vials.write_text("changed")
        factory = Mock()
        result = run_reset(self.folder, factory=factory)
        self.assertFalse(result["completed"])
        factory.assert_not_called()

    def test_cleanup_failure_is_reported(self):
        result = run_reset(self.folder, factory=Mock(side_effect=RuntimeError("cleanup failed")))
        self.assertFalse(result["completed"])
        self.assertIn("cleanup failed", result["exception"])
        self.assertTrue((self.folder / "result.json").is_file())

    def test_live_lock_failure_blocks_initialization(self):
        self.lock.side_effect = RuntimeError("hardware busy")
        factory = Mock()
        self.assertFalse(run_reset(self.folder, factory=factory)["completed"])
        factory.assert_not_called()

    def test_wrong_vial_file_blocks_initialization_even_with_matching_hash(self):
        self.vials.write_text("vial_index,location\n99,clamp\n")
        self.job["input_hashes"] = fingerprint([self.vials, self.state])
        (self.folder / "input.json").write_text(json.dumps(self.job))
        factory = Mock()
        result = run_reset(self.folder, factory=factory)
        self.assertFalse(result["completed"])
        self.assertIn("recorded held vial", result["exception"])
        factory.assert_not_called()

    def test_incomplete_cleanup_cannot_report_success(self):
        robot = SimpleNamespace(GRIPPER_STATUS="Vial")
        result = run_reset(self.folder, factory=Mock(return_value=SimpleNamespace(nr_robot=robot)))
        self.assertFalse(result["completed"])
        self.assertIn("Reset incomplete", result["exception"])

    def test_partial_reset_keeps_current_state_after_interrupt(self):
        before = self.state.read_bytes()

        def interrupted(**kwargs):
            self.state.write_text("gripper_status: null\npipet_fluid_volume: 0\n")
            raise KeyboardInterrupt("interrupted after partial cleanup")

        result = run_reset(self.folder, factory=interrupted)
        self.assertFalse(result["completed"])
        self.assertNotEqual(self.state.read_bytes(), before)

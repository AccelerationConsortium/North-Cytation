import json
import logging
import os
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scheduler import live_runner as runner


class LiveRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "config.yaml"
        self.source.write_text("reviewed config")
        self.vials = self.root / "vials.csv"
        self.vials.write_text("location,location_index\nmain_8mL_rack,0\n")
        self.robot = {"gripper_status": None, "gripper_vial_index": None,
                      "held_pipet_type": None, "pipet_fluid_vial_index": None, "pipet_fluid_volume": 0}
        (self.root / "robot_status.yaml").write_text(yaml.safe_dump(self.robot))
        (self.root / "track_status.yaml").write_text(yaml.safe_dump({"active_wellplate_position": None}))
        self.job = {"workflow": "fixture", "config": {"SIMULATE": False},
                    "input_hashes": runner.fingerprint([self.source, self.vials]),
                    "state_root": str(self.root), "vial_files": [str(self.vials)]}
        self.save_job()
        lock = patch.object(runner, "scheduler_live_lock", return_value=nullcontext())
        lock.start()
        self.addCleanup(lock.stop)

    def save_job(self):
        (self.root / "input.json").write_text(json.dumps(self.job))

    def test_only_ready_retained_plate_passes_live_and_simulated_handoff(self):
        from scheduler.simulation_runner import handoff_issues
        for status, position, allowed in (
            ("second_half_ready", "pipetting_area", True),
            ("in_progress", "pipetting_area", False),
            ("second_half_ready", "cytation_tray", False),
        ):
            with self.subTest(status=status, position=position):
                (self.root / "track_status.yaml").write_text(yaml.safe_dump({
                    "active_wellplate_position": position, "wellplate_type": "96 WELL PLATE",
                }))
                (self.root / "fluorescence_plate.yaml").write_text(yaml.safe_dump(
                    {"status": status, "plate_type": "96 WELL PLATE"}))
                self.assertEqual(not runner.live_state_issues(self.job), allowed)
                self.assertEqual(not handoff_issues(self.root), allowed)

    def test_successful_no_gui_live_call(self):
        execute = Mock()
        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=execute)):
            result = runner.run_job(self.root)
        execute.assert_called_once_with(config={"SIMULATE": False}, show_gui=False)
        self.assertTrue(result["completed"])
        self.assertTrue(result["handoff_ok"])

    def test_logged_error_survives_logger_handler_replacement(self):
        def execute(config, show_gui):
            logger = logging.getLogger("my_logger")
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
            logger.addHandler(logging.NullHandler())
            logger.error("experiment error")

        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=execute)):
            result = runner.run_job(self.root)
        self.assertTrue(result["completed"])
        self.assertFalse(result["handoff_ok"])
        self.assertEqual(result["records"], [{"level": "ERROR", "message": "experiment error"}])

    def test_failure_does_not_get_marked_completed(self):
        execute = Mock(side_effect=RuntimeError("workflow failed"))
        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=execute)):
            result = runner.run_job(self.root)
        self.assertFalse(result["completed"])
        self.assertIn("workflow failed", result["exception"])

    def test_keyboard_interrupt_is_reported_separately_from_failure(self):
        execute = Mock(side_effect=KeyboardInterrupt())
        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=execute)):
            result = runner.run_job(self.root)
        self.assertFalse(result["completed"])
        self.assertTrue(result["interrupted"])
        self.assertIn("KeyboardInterrupt", result["exception"])

    def test_main_preserves_keyboard_interrupt_for_run_logger(self):
        result = {"exception": "KeyboardInterrupt traceback", "interrupted": True}
        with patch.object(sys, "argv", ["live_runner", "--job-folder", str(self.root)]), \
                patch.object(runner, "run_job", return_value=result):
            with self.assertRaises(KeyboardInterrupt):
                runner.main()

    def test_changed_inputs_or_simulation_context_block_import(self):
        self.source.write_text("changed")
        with patch.object(runner.importlib, "import_module") as importer:
            result = runner.run_job(self.root)
            importer.assert_not_called()
        self.assertFalse(result["completed"])
        with patch.dict(os.environ, {runner.SESSION_ENV: "private-session"}), patch.object(runner.importlib, "import_module") as importer:
            result = runner.run_job(self.root)
            importer.assert_not_called()
        self.assertIn("simulation state", result["exception"])

    def test_dirty_end_state_stops_handoff(self):
        def execute(config, show_gui):
            self.robot["held_pipet_type"] = "small_tip"
            (self.root / "robot_status.yaml").write_text(yaml.safe_dump(self.robot))

        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=execute)):
            result = runner.run_job(self.root)
        self.assertTrue(result["completed"])
        self.assertFalse(result["handoff_ok"])
        self.assertIn("held_pipet_type", result["records"][0]["message"])

    def test_no_vial_live_job_keeps_explicit_null(self):
        self.job["config"]["INPUT_VIAL_STATUS_FILE"] = None
        self.job["vial_files"] = []
        self.job["input_hashes"] = runner.fingerprint([self.source])
        self.save_job()
        execute = Mock()
        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=execute)):
            result = runner.run_job(self.root)
        self.assertTrue(result["handoff_ok"])
        self.assertIsNone(execute.call_args.kwargs["config"]["INPUT_VIAL_STATUS_FILE"])

    def prepare_live_chain(self):
        self.job["input_hashes"] = runner.fingerprint([self.source])
        self.job["state_hashes"] = runner.fingerprint([
            self.vials, self.root / "robot_status.yaml", self.root / "track_status.yaml"
        ])
        self.save_job()
        next_folder = self.root / "next"
        next_folder.mkdir()
        next_job = dict(self.job, previous_job_result=str(self.root / "result.json"))
        (next_folder / "input.json").write_text(json.dumps(next_job))
        return next_folder

    def test_shared_vial_updates_carry_forward_between_live_jobs(self):
        next_folder = self.prepare_live_chain()

        def execute(config, show_gui):
            self.vials.write_text("location,location_index,vial_volume\nmain_8mL_rack,0,1.5\n")
            self.robot["pipets_used"] = {"rack": 1}
            (self.root / "robot_status.yaml").write_text(yaml.safe_dump(self.robot))

        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=execute)):
            first = runner.run_job(self.root)
        self.assertTrue(first["handoff_ok"])
        self.assertEqual(first["state_hashes"], runner.fingerprint(self.job["state_hashes"]))
        execute_next = Mock()
        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=execute_next)):
            second = runner.run_job(next_folder)
        self.assertTrue(second["handoff_ok"], second["exception"])
        execute_next.assert_called_once()

    def test_inventory_edit_after_handoff_blocks_next_job(self):
        next_folder = self.prepare_live_chain()
        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=Mock())):
            self.assertTrue(runner.run_job(self.root)["handoff_ok"])
        self.vials.write_text("location,location_index\nmain_8mL_rack,1\n")
        with patch.object(runner.importlib, "import_module") as importer:
            result = runner.run_job(next_folder)
        importer.assert_not_called()
        self.assertIn("Live inventory changed", result["exception"])

    def test_inventory_edit_before_first_job_blocks_import(self):
        self.prepare_live_chain()
        self.vials.write_text("location,location_index\nmain_8mL_rack,1\n")
        with patch.object(runner.importlib, "import_module") as importer:
            result = runner.run_job(self.root)
        importer.assert_not_called()
        self.assertIn("Live inventory changed", result["exception"])

    def test_failed_or_missing_handoff_cannot_authorize_next_job(self):
        next_folder = self.prepare_live_chain()
        for previous in (None, {"completed": False, "handoff_ok": False},
                         {"completed": True, "handoff_ok": True}):
            with self.subTest(previous=previous):
                if previous is not None:
                    (self.root / "result.json").write_text(json.dumps(previous))
                with patch.object(runner.importlib, "import_module") as importer:
                    result = runner.run_job(next_folder)
                importer.assert_not_called()
                self.assertFalse(result["completed"])

    def test_config_edit_between_jobs_still_blocks_import(self):
        next_folder = self.prepare_live_chain()
        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=Mock())):
            self.assertTrue(runner.run_job(self.root)["handoff_ok"])
        self.source.write_text("changed config")
        with patch.object(runner.importlib, "import_module") as importer:
            result = runner.run_job(next_folder)
        importer.assert_not_called()
        self.assertIn("Reviewed inputs changed", result["exception"])

    def test_config_edit_during_job_blocks_handoff(self):
        self.prepare_live_chain()

        def execute(config, show_gui):
            self.source.write_text("changed config")

        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=execute)):
            result = runner.run_job(self.root)
        self.assertFalse(result["handoff_ok"])
        self.assertNotIn("state_hashes", result)
        self.assertIn("Reviewed inputs changed during the live job", result["exception"])

    def test_logged_error_does_not_publish_inventory_handoff(self):
        next_folder = self.prepare_live_chain()

        def execute(config, show_gui):
            logging.getLogger().error("workflow error")

        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=execute)):
            first = runner.run_job(self.root)
        self.assertFalse(first["handoff_ok"])
        self.assertNotIn("state_hashes", first)
        with patch.object(runner.importlib, "import_module") as importer:
            second = runner.run_job(next_folder)
        importer.assert_not_called()
        self.assertIn("successful handoff", second["exception"])


if __name__ == "__main__":
    unittest.main()

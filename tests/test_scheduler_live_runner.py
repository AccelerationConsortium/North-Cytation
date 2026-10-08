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


if __name__ == "__main__":
    unittest.main()
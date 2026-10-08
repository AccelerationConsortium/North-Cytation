import ast
import json
import logging
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml
from scheduler import simulation_state as state
from scheduler import simulation_runner as runner


REPO_ROOT = Path(__file__).resolve().parents[1]


class SimulationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        (self.repo / "workflows").mkdir(parents=True)
        (self.repo / "robot_state").mkdir()
        (self.repo / "workflows" / "test_job.py").touch()
        self.robot_state = {"gripper_status": None, "gripper_vial_index": None,
                            "held_pipet_type": None, "pipet_fluid_vial_index": None,
                            "pipet_fluid_volume": 0, "pipets_used": {"rack": 0}}
        self.track_state = {"active_wellplate_position": None, "num_in_source": 2}
        (self.repo / "robot_state" / "robot_status.yaml").write_text(yaml.safe_dump(self.robot_state))
        (self.repo / "robot_state" / "track_status.yaml").write_text(yaml.safe_dump(self.track_state))
        self.vial = self.repo / "vials.csv"
        self.vial.write_text("vial_name,location,location_index,vial_volume\nvial,clamp,0,2\n")
        self.original = self.vial.read_bytes()
        for item in (patch.object(state, "REPO_ROOT", self.repo),
                     patch.object(state, "STATE_ROOT", Path(self.temp.name) / "private")):
            item.start()
            self.addCleanup(item.stop)
        self.root = state.create_session([self.vial])

    def job(self, job_id):
        folder = self.root / "jobs" / job_id
        folder.mkdir(parents=True)
        (folder / "input.json").write_text(json.dumps({"workflow": "test_job", "config": {
            "SIMULATE": False, "INPUT_VIAL_STATUS_FILE": str(self.vial)}}))
        return folder

    def execute(self, config, show_gui):
        self.assertFalse(show_gui)
        self.assertIs(config["SIMULATE"], True)
        self.assertNotEqual(config["INPUT_VIAL_STATUS_FILE"], str(self.vial))
        logger = logging.getLogger("my_logger")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        state.attach_diagnostics(logger)
        robot = types.SimpleNamespace(simulate=True, VIAL_FILE=config["INPUT_VIAL_STATUS_FILE"])
        track = types.SimpleNamespace(simulate=True)
        state.configure_controller(robot, "robot")
        state.configure_controller(track, "track")

        def save_robot():
            current = yaml.safe_load(Path(robot.ROBOT_STATUS_FILE).read_text())
            current["pipets_used"]["rack"] += 1
            Path(robot.ROBOT_STATUS_FILE).write_text(yaml.safe_dump(current))

        robot.save_robot_status = save_robot
        track.save_track_status = lambda: None
        coordinator = types.SimpleNamespace(nr_robot=robot, nr_track=track, logger=logger,
                                            _workflow_should_continue=True, log_filename="test.log")
        state.register_coordinator(coordinator)
        logger.error("Recoverable simulated error")
        logger.warning("Simulated warning")

    def test_recoverable_errors_reported_and_end_state_carried(self):
        self.job("first")
        self.job("second")
        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=self.execute)):
            first = runner.run_job(self.root, "first")
            second = runner.run_job(self.root, "second")
        self.assertTrue(first["completed"])
        self.assertTrue(second["handoff_ok"])
        self.assertEqual(first["tips_used"], {"rack": 1})
        self.assertEqual([record["level"] for record in first["records"]], ["ERROR", "WARNING"])
        self.assertEqual(len(second["records"]), 2)
        self.assertEqual(yaml.safe_load((self.root / "robot_status.yaml").read_text())["pipets_used"], {"rack": 2})
        self.assertEqual(self.vial.read_bytes(), self.original)
        self.assertNotIn(state.SESSION_ENV, os.environ)

    def test_fatal_exception_reports_incomplete_with_traceback(self):
        self.job("failed")
        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=Mock(side_effect=RuntimeError("failed experiment")))):
            result = runner.run_job(self.root, "failed")
        self.assertFalse(result["completed"])
        self.assertIn("failed experiment", result["exception"])
        self.assertTrue((self.root / "jobs" / "failed" / "result.json").exists())

    def test_input_prompt_becomes_reported_error(self):
        self.job("prompt")

        def execute(config, show_gui):
            input("Refill tips")

        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=execute)):
            result = runner.run_job(self.root, "prompt")
        self.assertFalse(result["completed"])
        self.assertIn("requires operator input", result["exception"])

    def test_dirty_state_blocks_next_handoff(self):
        self.robot_state["held_pipet_type"] = "small_tip"
        (self.root / "robot_status.yaml").write_text(yaml.safe_dump(self.robot_state))
        self.assertTrue(runner.handoff_issues(self.root))

    def test_conflict_override_reports_warnings_but_not_other_state_overrides(self):
        folder = self.job("conflicts")
        specification = json.loads((folder / "input.json").read_text())
        specification["allow_vial_conflicts"] = True
        (folder / "input.json").write_text(json.dumps(specification))
        private_vial = next((self.root / "vials").glob("*.csv"))
        private_vial.write_text("vial_name,location,location_index,vial_volume\none,clamp,0,2\ntwo,clamp,0,2\n")
        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=self.execute)):
            result = runner.run_job(self.root, "conflicts")
        self.assertTrue(result["handoff_ok"])
        self.assertTrue(any(record["level"] == "WARNING" and "test override" in record["message"] for record in result["records"]))
        self.job("held_tip")
        specification = json.loads((folder / "input.json").read_text())
        (self.root / "jobs" / "held_tip" / "input.json").write_text(json.dumps(specification))
        robot = yaml.safe_load((self.root / "robot_status.yaml").read_text())
        robot["held_pipet_type"] = "small_tip"
        (self.root / "robot_status.yaml").write_text(yaml.safe_dump(robot))
        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=self.execute)):
            result = runner.run_job(self.root, "held_tip")
        self.assertFalse(result["handoff_ok"])

    def test_tip_exhaustion_logs_and_continues(self):
        tree = ast.parse((REPO_ROOT / "North_Safe.py").read_text(encoding="utf-8-sig"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "North_Robot")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_pipet")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "North_Safe.py", "exec"), namespace)
        robot = types.SimpleNamespace(HELD_PIPET_TYPE=None, _scheduler_state_root=str(self.root),
                                      PIPET_RACKS={"rack": {"tip_type": "small_tip"}},
                                      PIPETS_USED={"rack": 2}, pause_after_error=Mock(), logger=Mock(),
                                      get_config_parameter=Mock(return_value=2), save_robot_status=Mock(),
                                      _calculate_tip_position=Mock(return_value=0), _move_to_pipet_tip=Mock(),
                                      _perform_pipet_pickup=Mock())
        namespace["get_pipet"](robot, "small_tip")
        robot.pause_after_error.assert_called_once()
        self.assertEqual(robot.PIPETS_USED, {"rack": 1})
        self.assertEqual(robot.HELD_PIPET_TYPE, "small_tip")
        robot._perform_pipet_pickup.assert_called_once()

    def test_reset_marks_totals_unavailable_without_stopping(self):
        self.job("reset")

        def execute(config, show_gui):
            self.execute(config, show_gui)
            logging.getLogger("my_logger").info("Resetting all pipet rack counters to 0 after refill")

        with patch.object(runner.importlib, "import_module", return_value=types.SimpleNamespace(execute=execute)):
            result = runner.run_job(self.root, "reset")
        self.assertTrue(result["completed"])
        self.assertTrue(result["handoff_ok"])
        self.assertEqual(result["tips_used"], "Unavailable after tip counter reset")
        self.assertTrue(any(record["level"] == "ERROR" for record in result["records"]))


if __name__ == "__main__":
    unittest.main()
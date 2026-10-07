import ast
import json
import logging
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scheduler import simulation_state as state


def save_class(class_name, method):
    tree = ast.parse((REPO_ROOT / "North_Safe.py").read_text(encoding="utf-8-sig"))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    names = [method, "_load_state_files"] if class_name == "North_Robot" else [method, "get_track_status"]
    functions = [node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"yaml": yaml, "pd": pd}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "North_Safe.py", "exec"), namespace)
    return type(class_name, (), {name: namespace[name] for name in names})


class SimulationStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        (self.repo / "robot_state").mkdir(parents=True)
        (self.repo / "robot_state" / "robot_status.yaml").write_text("pipets_used: {rack: 0}\n")
        (self.repo / "robot_state" / "track_status.yaml").write_text("num_in_source: 10\n")
        self.vials = []
        for name in ("first", "second"):
            path = self.repo / f"{name}.csv"
            path.write_text(f"vial_index,vial_name,vial_volume\n0,{name},5\n")
            self.vials.append(path)
        self.original = {path: path.read_bytes() for path in [*self.vials, *list((self.repo / "robot_state").glob("*.yaml"))]}
        patches = [patch.object(state, "REPO_ROOT", self.repo),
                   patch.object(state, "STATE_ROOT", Path(self.temp.name) / "private"),
                   patch.dict(os.environ)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        os.environ.pop(state.SESSION_ENV, None)
        self.root = state.create_session(self.vials)
        self.Robot = save_class("North_Robot", "save_robot_status")
        self.Track = save_class("North_Track", "save_track_status")

    def controllers(self, vial_file, tips, plates):
        robot = self.Robot()
        robot.simulate = True
        robot.VIAL_FILE = str(vial_file)
        state.configure_controller(robot, "robot")
        robot.GRIPPER_STATUS = None
        robot.GRIPPER_VIAL_INDEX = None
        robot.HELD_PIPET_TYPE = None
        robot.PIPETS_USED = {"rack": tips}
        robot.PIPET_FLUID_VIAL_INDEX = None
        robot.PIPET_FLUID_VOLUME = 0
        robot.VIAL_DF = pd.read_csv(robot.VIAL_FILE)
        track = self.Track()
        track.simulate = True
        state.configure_controller(track, "track")
        track.NUM_SOURCE = plates
        track.NUM_WASTE = 10 - plates
        track.CURRENT_WP_TYPE = "96 WELL PLATE"
        track.ACTIVE_WELLPLATE_POSITION = None
        track.CURRENT_GRIPPER_LOCATION = "home"
        track.CURRENT_GRIPPER_POSITION = {"x": 0, "z": 0}
        return types.SimpleNamespace(nr_robot=robot, nr_track=track, logger=Mock())

    def assert_live_unchanged(self):
        for path, before in self.original.items():
            self.assertEqual(path.read_bytes(), before)

    def test_end_state_carries_to_next_job_with_separate_vial_files(self):
        first = state.simulated_config(self.root, {"SIMULATE": False, "INPUT_VIAL_STATUS_FILE": str(self.vials[0])})
        second = state.simulated_config(self.root, {"SIMULATE": False, "INPUT_VIAL_STATUS_FILE": str(self.vials[1])})
        with state.workflow_state(self.root, "first_job"):
            coordinator = self.controllers(first["INPUT_VIAL_STATUS_FILE"], 3, 8)
            coordinator.nr_robot.VIAL_DF["vial_volume"] = 4
            state.register_coordinator(coordinator)
        with state.workflow_state(self.root, "second_job"):
            coordinator = self.controllers(second["INPUT_VIAL_STATUS_FILE"], 7, 6)
            self.assertEqual(yaml.safe_load(Path(coordinator.nr_robot.ROBOT_STATUS_FILE).read_text())["pipets_used"], {"rack": 3})
            self.assertEqual(yaml.safe_load(Path(coordinator.nr_track.TRACK_STATUS_FILE).read_text())["num_in_source"], 8)
            state.register_coordinator(coordinator)
        self.assertEqual(pd.read_csv(first["INPUT_VIAL_STATUS_FILE"]).vial_volume.iloc[0], 4)
        self.assertEqual(pd.read_csv(second["INPUT_VIAL_STATUS_FILE"]).vial_volume.iloc[0], 5)
        for job in ("first_job", "second_job"):
            snapshot = self.root / "end_states" / job
            self.assertTrue((snapshot / "robot_status.yaml").exists())
            self.assertTrue(json.loads((snapshot / "result.json").read_text())["completed"])
        self.assert_live_unchanged()

    def test_ordinary_simulation_save_methods_write_nothing(self):
        with state.workflow_state(self.root, "unused"):
            coordinator = self.controllers(state.simulated_config(self.root, {"INPUT_VIAL_STATUS_FILE": str(self.vials[0])})["INPUT_VIAL_STATUS_FILE"], 3, 8)
        robot, track = coordinator.nr_robot, coordinator.nr_track
        del robot._scheduler_state_root
        del track._scheduler_state_root
        robot.VIAL_FILE = str(self.vials[0])
        robot.ROBOT_STATUS_FILE = str(self.repo / "robot_state" / "robot_status.yaml")
        track.TRACK_STATUS_FILE = str(self.repo / "robot_state" / "track_status.yaml")
        robot.save_robot_status()
        track.save_track_status()
        self.assert_live_unchanged()

    def test_reject_live_mode_original_paths_and_changed_session(self):
        config = state.simulated_config(self.root, {"INPUT_VIAL_STATUS_FILE": str(self.vials[0])})
        with state.workflow_state(self.root, "guard"):
            with self.assertRaises(ValueError):
                state.validate_launch(False, config["INPUT_VIAL_STATUS_FILE"])
            with self.assertRaises(ValueError):
                state.validate_launch(True, str(self.vials[0]))
            coordinator = self.controllers(config["INPUT_VIAL_STATUS_FILE"], 3, 8)
            with self.assertRaises(ValueError):
                state.can_save(coordinator.nr_robot, self.vials[0])
        with self.assertRaises(ValueError):
            coordinator.nr_robot.save_robot_status()
        with self.assertRaises(ValueError):
            with state.workflow_state(self.repo, "unsafe"):
                pass
        self.assertNotIn(state.SESSION_ENV, os.environ)
        self.assert_live_unchanged()

    def test_failure_preserves_exception_and_marks_end_snapshot_failed(self):
        config = state.simulated_config(self.root, {"INPUT_VIAL_STATUS_FILE": str(self.vials[0])})
        with self.assertRaisesRegex(RuntimeError, "experiment failed"):
            with state.workflow_state(self.root, "failed"):
                state.register_coordinator(self.controllers(config["INPUT_VIAL_STATUS_FILE"], 3, 8))
                raise RuntimeError("experiment failed")
        result = json.loads((self.root / "end_states" / "failed" / "result.json").read_text())
        self.assertFalse(result["completed"])
        self.assertNotIn(state.SESSION_ENV, os.environ)
        self.assert_live_unchanged()

    def test_new_session_starts_fresh_and_config_is_not_modified(self):
        config = {"SIMULATE": False, "INPUT_VIAL_STATUS_FILE": str(self.vials[0]), "POINTS": [1]}
        simulated = state.simulated_config(self.root, config)
        simulated["POINTS"].append(2)
        self.assertFalse(config["SIMULATE"])
        self.assertEqual(config["POINTS"], [1])
        (self.root / "robot_status.yaml").write_text("pipets_used: {rack: 20}\n")
        fresh = state.create_session(self.vials)
        self.assertEqual(yaml.safe_load((fresh / "robot_status.yaml").read_text())["pipets_used"], {"rack": 0})

    def test_actual_reload_methods_read_private_updated_state(self):
        config = state.simulated_config(self.root, {"INPUT_VIAL_STATUS_FILE": str(self.vials[0])})
        with state.workflow_state(self.root, "reload"):
            coordinator = self.controllers(config["INPUT_VIAL_STATUS_FILE"], 4, 7)
            robot, track = coordinator.nr_robot, coordinator.nr_track
            robot.VIAL_DF["vial_volume"] = 2
            robot.save_robot_status()
            track.save_track_status()
            robot.logger = Mock()
            robot.PIPET_RACKS = {"rack": {}}
            robot.PIPETS_USED = {"rack": 0}
            robot.WELLPLATE_POSITIONS_FILE = str(self.repo / "wellplates.yaml")
            robot._load_yaml_file = lambda path, *args, **kwargs: yaml.safe_load(Path(path).read_text()) if Path(path).is_file() else None
            robot.pause_after_error = Mock(side_effect=AssertionError("Unexpected state load error"))
            robot._load_state_files()
            self.assertEqual(robot.PIPETS_USED, {"rack": 4})
            self.assertEqual(robot.VIAL_DF.vial_volume.iloc[0], 2)
            track.logger = Mock()
            track._load_yaml_file = robot._load_yaml_file
            track.pause_after_error = robot.pause_after_error
            track.NUM_SOURCE = 0
            track.get_track_status()
            self.assertEqual(track.NUM_SOURCE, 7)
        self.assert_live_unchanged()

    def test_default_live_save_behavior_is_unchanged(self):
        config = state.simulated_config(self.root, {"INPUT_VIAL_STATUS_FILE": str(self.vials[0])})
        with state.workflow_state(self.root, "unused_live_fixture"):
            coordinator = self.controllers(config["INPUT_VIAL_STATUS_FILE"], 5, 7)
        robot, track = coordinator.nr_robot, coordinator.nr_track
        del robot._scheduler_state_root
        del track._scheduler_state_root
        robot.simulate = False
        track.simulate = False
        robot.VIAL_FILE = str(self.vials[0])
        robot.ROBOT_STATUS_FILE = str(self.repo / "robot_state" / "robot_status.yaml")
        track.TRACK_STATUS_FILE = str(self.repo / "robot_state" / "track_status.yaml")
        robot.save_robot_status()
        track.save_track_status()
        self.assertEqual(yaml.safe_load(Path(robot.ROBOT_STATUS_FILE).read_text())["pipets_used"], {"rack": 5})
        self.assertEqual(yaml.safe_load(Path(track.TRACK_STATUS_FILE).read_text())["num_in_source"], 7)

    def test_coordinator_rejects_live_mode_before_tracking_or_hardware(self):
        source = REPO_ROOT / "master_usdl_coordinator.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Lash_E")
        constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        tracker = Mock()
        namespace = {"os": os, "sys": sys, "logging": logging, "datetime": datetime,
                     "ConfigManager": None, "experiment_run_logger": types.SimpleNamespace(start_run=tracker)}
        exec(compile(ast.Module(body=[constructor], type_ignores=[]), str(source), "exec"), namespace)
        coordinator = type("Coordinator", (), {"__init__": namespace["__init__"]})
        config = state.simulated_config(self.root, {"INPUT_VIAL_STATUS_FILE": str(self.vials[0])})
        with state.workflow_state(self.root, "reject_live"), patch.dict(sys.modules, {"north": None}):
            with self.assertRaisesRegex(ValueError, "live hardware"):
                coordinator(config["INPUT_VIAL_STATUS_FILE"], simulate=False, show_gui=False,
                            logging_folder=str(self.root / "logs"))
        tracker.assert_not_called()
        self.assertFalse((self.root / "logs").exists())
        self.assert_live_unchanged()


if __name__ == "__main__":
    unittest.main()
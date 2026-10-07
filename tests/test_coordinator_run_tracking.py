import ast
import logging
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]


class CoordinatorRunTrackingTests(unittest.TestCase):
    def exercise_startup(self, initial, confirmed, cancel=False, show_gui=True, hardware_error=False):
        source = REPO_ROOT / "master_usdl_coordinator.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        coordinator = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Lash_E")
        methods = [node for node in coordinator.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"__init__", "update_simulate_flag"}]
        events = []
        values = {"SIMULATE": initial}

        def reload_config(name, namespace, logger):
            namespace["SIMULATE"] = confirmed if "review" in events or not show_gui else initial
            return {"SIMULATE": namespace["SIMULATE"]}

        def review(instance):
            events.append("review")
            self.assertIsNone(instance.log_filename)
            self.assertIsNone(instance._run_tracker)
            self.assertFalse(any(isinstance(handler, logging.FileHandler) for handler in instance.logger.handlers))
            self.assertEqual(list(Path(directory).iterdir()), [])
            instance.logger.info("Review log evidence")
            instance._workflow_should_continue = not cancel

        def track(*args):
            events.append("track")
            self.assertIs(args[1], confirmed)
            self.assertEqual("_simulate" in args[2], confirmed)
            return types.SimpleNamespace(simulate=args[1], log_filename=args[2])

        def hardware(*args, **kwargs):
            events.append("hardware")
            self.assertIn("track", events)
            if hardware_error:
                raise RuntimeError("controller initialization failed")
            return Mock()

        tracker = Mock(side_effect=track)
        namespace = {
            "logging": logging, "os": os, "sys": sys, "datetime": datetime,
            "ConfigManager": types.SimpleNamespace(setup_config_if_missing=Mock(), load_and_update_globals=reload_config),
            "experiment_run_logger": types.SimpleNamespace(start_run=tracker),
        }
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
        constructor = type("TestCoordinator", (), {
            "__init__": namespace["__init__"],
            "update_simulate_flag": namespace["update_simulate_flag"],
            "check_input_status": review,
        })
        logger = logging.getLogger("my_logger")
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
            "north": types.SimpleNamespace(NorthC9=hardware),
            "photoreactor_controller": types.SimpleNamespace(Photoreactor_Controller=hardware),
        }):
            try:
                kwargs = dict(initialize_robot=False, initialize_track=False, initialize_biotek=False,
                              simulate=initial, logging_folder=directory, workflow_globals=values,
                              workflow_name="tracking_test", show_gui=show_gui)
                if hardware_error:
                    with self.assertRaisesRegex(RuntimeError, "controller initialization failed"):
                        constructor("vials.csv", **kwargs)
                    tracker.assert_called_once()
                else:
                    instance = constructor("vials.csv", **kwargs)
                    if cancel:
                        self.assertIsNone(instance._run_tracker)
                        tracker.assert_not_called()
                        self.assertEqual(events, ["review"])
                        self.assertEqual(list(Path(directory).iterdir()), [])
                        self.assertIsNone(instance.log_filename)
                    else:
                        tracker.assert_called_once()
                        self.assertIs(instance.simulate, confirmed)
                        logs = list(Path(directory).glob("*.log"))
                        self.assertEqual(len(logs), 1)
                        self.assertEqual(logs[0].name, instance.log_filename)
                        if show_gui:
                            self.assertEqual(events[:2], ["review", "track"])
                            self.assertNotIn("Review log evidence", logs[0].read_text())
                        self.assertIn(f"Confirmed SIMULATE mode: {confirmed}", logs[0].read_text())
            finally:
                for handler in list(logger.handlers):
                    logger.removeHandler(handler)
                    handler.close()

    def test_simulation_changed_to_live_after_review(self):
        self.exercise_startup(True, False)

    def test_live_changed_to_simulation_after_review(self):
        self.exercise_startup(False, True)

    def test_cancel_creates_no_run_tracker(self):
        self.exercise_startup(True, True, cancel=True)

    def test_no_gui_tracks_before_initialization(self):
        self.exercise_startup(False, False, show_gui=False)

    def test_initialization_failure_occurs_after_tracking_starts(self):
        self.exercise_startup(True, False, hardware_error=True)


if __name__ == "__main__":
    unittest.main()
import ast
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from workflow_config_manager import ConfigManager


REPO_ROOT = Path(__file__).resolve().parents[1]


class WorkflowTemplateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.vials = Path(self.temporary.name) / "vials.csv"
        self.vials.touch()
        self.robot = Mock(spec_set=["VIAL_FILE", "move_home"])
        self.robot.VIAL_FILE = str(self.vials)
        self.temperature = Mock(spec_set=["set_temp", "turn_off_heating", "turn_off_stirring"])
        self.lash = types.SimpleNamespace(
            _workflow_should_continue=True,
            simulate=True,
            logger=Mock(spec_set=["info", "exception", "warning"]),
            nr_robot=self.robot,
            temp_controller=self.temperature,
            grab_new_wellplate=Mock(),
            measure_wellplate=Mock(return_value=None),
            discard_used_wellplate=Mock(),
        )
        def initialize(*args, **kwargs):
            self.lash.workflow_config = ConfigManager.resolve_workflow_config(
                kwargs['workflow_name'], kwargs['workflow_globals'],
                config=kwargs['config'], show_gui=kwargs['show_gui'],
            )
            return self.lash

        self.constructor = Mock(side_effect=initialize)
        with patch.dict(sys.modules, {
            "master_usdl_coordinator": types.SimpleNamespace(Lash_E=self.constructor),
        }):
            self.module = runpy.run_path(str(REPO_ROOT / "workflows" / "workflow_template.py"))
        self.execute = self.module["execute"]
        self.config = {key: self.module[key] for key in self.module["_CONFIG_KEYS"]}
        self.config["INPUT_VIAL_STATUS_FILE"] = str(self.vials)

    def test_explicit_config_runs_complete_simulated_body_without_yaml(self):
        self.execute(self.config, show_gui=False)
        kwargs = self.constructor.call_args.kwargs
        self.assertIs(kwargs["workflow_globals"], self.execute.__globals__)
        self.assertEqual(kwargs["config"], self.config)
        self.assertNotIn("simulate", kwargs)
        self.assertFalse(kwargs["show_gui"])
        self.assertFalse(kwargs["initialize_p2"])
        self.lash.grab_new_wellplate.assert_called_once()
        self.lash.measure_wellplate.assert_called_once_with(
            self.config["MEASUREMENT_PROTOCOL_FILE"], list(range(self.config["PARAM2"]))
        )
        self.lash.discard_used_wellplate.assert_called_once()

    def test_normal_start_uses_gui_edited_parameters_and_filename(self):
        def show_review(*args, **kwargs):
            self.assertTrue(kwargs["show_gui"])
            self.assertEqual(kwargs["workflow_name"], "workflow_template")
            kwargs["workflow_globals"]["TARGET_TEMPERATURE"] = 33.0
            kwargs["workflow_globals"]["PARAM2"] = 4
            self.lash.workflow_config = {**self.config, 'TARGET_TEMPERATURE': 33.0, 'PARAM2': 4}
            return self.lash

        self.constructor.side_effect = show_review
        self.execute()
        self.temperature.set_temp.assert_called_once_with(33.0)
        self.assertEqual(self.lash.measure_wellplate.call_args.args[1], [0, 1, 2, 3])

    def test_explicit_config_with_gui_is_rejected_before_controller_creation(self):
        with self.assertRaisesRegex(ValueError, "show_gui=False"):
            self.execute(self.config)
        self.lash.grab_new_wellplate.assert_not_called()

    def test_supplied_config_is_copied_and_does_not_load_yaml(self):
        namespace = self.execute.__globals__
        keys = namespace["_CONFIG_KEYS"] + ["EXPERIMENT_POINTS"]
        supplied = {**self.config, "EXPERIMENT_POINTS": [1, 2]}
        with patch.dict(namespace, {"_CONFIG_KEYS": keys}), patch.object(
            ConfigManager, 'load_and_update_globals', side_effect=AssertionError('YAML must not load')
        ) as loader:
            confirmed = ConfigManager.resolve_workflow_config('workflow_template', namespace, supplied, False)
        self.assertIsNot(confirmed, supplied)
        confirmed["EXPERIMENT_POINTS"].append(3)
        self.assertEqual(supplied["EXPERIMENT_POINTS"], [1, 2])
        loader.assert_not_called()

    def test_partial_supplied_config_is_rejected_before_hardware(self):
        incomplete = self.config.copy()
        del incomplete["TARGET_TEMPERATURE"]
        with self.assertRaises(KeyError):
            self.execute(incomplete, show_gui=False)
        self.lash.grab_new_wellplate.assert_not_called()

    def test_gui_can_correct_launch_values_before_experiment_validation(self):
        def review(*args, **kwargs):
            self.assertEqual(args, ())
            kwargs["workflow_globals"].update(self.config, PARAM2=4)
            self.lash.workflow_config = {**self.config, 'PARAM2': 4}
            return self.lash

        self.constructor.side_effect = review
        self.execute()
        self.assertEqual(self.lash.measure_wellplate.call_args.args[1], [0, 1, 2, 3])

    def test_cancel_does_not_execute_steps(self):
        self.lash._workflow_should_continue = False
        self.assertIsNone(self.execute(self.config, show_gui=False))
        self.temperature.set_temp.assert_not_called()
        self.lash.grab_new_wellplate.assert_not_called()

    def test_cleanup_failures_do_not_mask_error_or_skip_other_actions(self):
        self.lash.measure_wellplate.side_effect = RuntimeError("reader failed")
        self.temperature.turn_off_heating.side_effect = OSError("heater shutdown failed")
        with self.assertRaisesRegex(RuntimeError, "reader failed"):
            self.execute(self.config, show_gui=False)
        self.temperature.turn_off_stirring.assert_called_once()
        self.robot.move_home.assert_called_once()
        self.lash.logger.exception.assert_any_call("Cleanup failed: turn_off_heating")

    def test_keyboard_interrupt_attempts_cleanup_and_propagates(self):
        self.lash.measure_wellplate.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.execute(self.config, show_gui=False)
        self.robot.move_home.assert_called_once()

    def live_config(self):
        protocol = Path(self.temporary.name) / "measurement.prt"
        protocol.touch()
        self.lash.simulate = False
        self.lash.measure_wellplate.return_value = [1.0]
        return {**self.config, "SIMULATE": False, "MEASUREMENT_PROTOCOL_FILE": str(protocol)}

    def test_simulation_does_not_import_slack(self):
        with patch.dict(sys.modules, {"slack_agent": None}):
            self.execute(self.config, show_gui=False)
        self.lash.logger.warning.assert_not_called()
        self.lash.discard_used_wellplate.assert_called_once()

    def test_live_run_sends_start_and_completion(self):
        config = self.live_config()
        sender = Mock(return_value=True)
        with patch.dict(sys.modules, {"slack_agent": types.SimpleNamespace(safe_send_slack_message=sender)}):
            self.assertEqual(self.execute(config, show_gui=False), [1.0])
        self.assertEqual(sender.call_count, 2)
        self.assertIn("workflow_template started", sender.call_args_list[0].args[0])
        self.assertIn("workflow_template completed successfully", sender.call_args_list[1].args[0])

    def test_live_failure_notifies_after_cleanup_and_preserves_error(self):
        config = self.live_config()
        self.lash.measure_wellplate.side_effect = RuntimeError("reader failed")
        events = []
        self.robot.move_home.side_effect = lambda: events.append("cleanup")

        def send(message):
            events.append(message)
            return True

        with patch.dict(sys.modules, {"slack_agent": types.SimpleNamespace(safe_send_slack_message=send)}):
            with self.assertRaisesRegex(RuntimeError, "reader failed"):
                self.execute(config, show_gui=False)
        self.assertEqual(events[1], "cleanup")
        self.assertIn("failed: reader failed", events[2])
        self.assertFalse(any("completed successfully" in event for event in events))

    def test_slack_failures_are_logged_without_failing_experiment(self):
        config = self.live_config()
        sender = Mock(side_effect=[False, RuntimeError("Slack unavailable")])
        with patch.dict(sys.modules, {"slack_agent": types.SimpleNamespace(safe_send_slack_message=sender)}):
            self.assertEqual(self.execute(config, show_gui=False), [1.0])
        self.assertEqual(self.lash.logger.warning.call_count, 2)
        self.lash.discard_used_wellplate.assert_called_once()

    def test_controller_method_names_exist_in_real_source(self):
        controller_files = {
            "Lash_E": REPO_ROOT / "master_usdl_coordinator.py",
            "North_Robot": REPO_ROOT / "North_Safe.py",
            "North_Temp": REPO_ROOT / "North_Safe.py",
        }
        methods = {}
        for class_name, filename in controller_files.items():
            tree = ast.parse(filename.read_text(encoding="utf-8-sig"))
            definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
            methods[class_name] = {node.name for node in definition.body if isinstance(node, ast.FunctionDef)}
        expected = {
            "Lash_E": ["grab_new_wellplate", "measure_wellplate", "discard_used_wellplate"],
            "North_Robot": ["move_home"],
            "North_Temp": ["set_temp", "turn_off_heating", "turn_off_stirring"],
        }
        for class_name, names in expected.items():
            self.assertTrue(set(names).issubset(methods[class_name]), class_name)


if __name__ == "__main__":
    unittest.main()
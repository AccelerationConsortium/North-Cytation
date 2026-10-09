import ast
import logging
import os
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflow_config_manager import ConfigManager

REPO_ROOT = Path(__file__).resolve().parents[1]


class CoordinatorConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.logs = self.root / 'logs'
        self.original = self.root / 'original.csv'
        self.reviewed = self.root / 'reviewed.csv'
        self.original.touch()
        self.reviewed.touch()
        self.values = {'SIMULATE': True, 'INPUT_VIAL_STATUS_FILE': str(self.original),
                       'PARAMETER': 1, 'POINTS': [1],
                       '_CONFIG_KEYS': ['SIMULATE', 'INPUT_VIAL_STATUS_FILE', 'PARAMETER', 'POINTS']}
        self.tracker = Mock()
        self.hardware = Mock(return_value=Mock())
        self.robot = Mock(side_effect=lambda c9, c8, vial_file, **kwargs: types.SimpleNamespace(VIAL_FILE=vial_file))
        source = REPO_ROOT / 'master_usdl_coordinator.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Lash_E')
        methods = [node for node in definition.body if isinstance(node, ast.FunctionDef)
                   and node.name in {'__init__', 'update_simulate_flag'}]
        namespace = {'logging': logging, 'os': os, 'sys': sys, 'datetime': datetime,
                     'ConfigManager': ConfigManager,
                     'experiment_run_logger': types.SimpleNamespace(start_run=self.tracker),
                     'North_Robot': self.robot, 'North_Spin': Mock()}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
        self.review = Mock()

        def review(instance):
            self.assertIsNone(instance.workflow_config)
            self.assertIsNone(instance._run_tracker)
            self.assertFalse(self.logs.exists())
            self.review(instance)

        self.Coordinator = type('Coordinator', (), {
            '__init__': namespace['__init__'], 'update_simulate_flag': namespace['update_simulate_flag'],
            'check_input_status': review,
        })
        config_patch = patch.object(ConfigManager, 'CONFIG_DIR', str(self.root / 'configs'))
        config_patch.start()
        self.addCleanup(config_patch.stop)
        imports = patch.dict(sys.modules, {
            'north': types.SimpleNamespace(NorthC9=self.hardware),
            'photoreactor_controller': types.SimpleNamespace(Photoreactor_Controller=Mock()),
        })
        imports.start()
        self.addCleanup(imports.stop)
        self.addCleanup(self.close_logs)

    def close_logs(self):
        logger = logging.getLogger('my_logger')
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def start(self, **kwargs):
        return self.Coordinator(initialize_track=False, initialize_biotek=False,
                                logging_folder=str(self.logs), **kwargs)

    def test_gui_changes_mode_vial_and_parameters_before_controllers(self):
        def review(instance):
            self.assertEqual(instance.vial_file, str(self.original))
            path = Path(ConfigManager.CONFIG_DIR) / 'example.yaml'
            selected = yaml.safe_load(path.read_text())
            selected.update(SIMULATE=False, INPUT_VIAL_STATUS_FILE=str(self.reviewed), PARAMETER=7)
            path.write_text(yaml.safe_dump(selected))

        self.review.side_effect = review
        coordinator = self.start(workflow_globals=self.values, workflow_name='example')
        self.assertFalse(coordinator.simulate)
        self.assertEqual(coordinator.workflow_config['PARAMETER'], 7)
        self.assertEqual(coordinator.vial_file, str(self.reviewed))
        self.assertEqual(self.robot.call_args.args[2], str(self.reviewed))
        self.assertIs(self.robot.call_args.kwargs['simulate'], False)
        self.assertIs(self.tracker.call_args.args[1], False)
        self.assertEqual(self.values['PARAMETER'], 7)

    def test_supplied_config_uses_no_yaml_and_is_independent(self):
        config = {key: deepcopy(self.values[key]) for key in self.values['_CONFIG_KEYS']}
        config.update(PARAMETER=9, INPUT_VIAL_STATUS_FILE=None)
        with patch.object(ConfigManager, 'load_and_update_globals', side_effect=AssertionError('No YAML loading')):
            coordinator = self.start(workflow_globals=self.values, workflow_name='example', config=config, show_gui=False)
        self.review.assert_not_called()
        self.assertFalse(Path(ConfigManager.CONFIG_DIR).exists())
        self.assertEqual(coordinator.workflow_config['PARAMETER'], 9)
        self.assertIsNone(coordinator.nr_robot.VIAL_FILE)
        coordinator.workflow_config['POINTS'].append(2)
        self.assertEqual(config['POINTS'], [1])
        self.assertEqual(self.values['POINTS'], [1])

    def test_partial_and_ambiguous_config_fail_before_controllers(self):
        with self.assertRaisesRegex(ValueError, 'show_gui=False'):
            self.start(workflow_globals=self.values, workflow_name='example', config={'SIMULATE': True})
        with self.assertRaisesRegex(KeyError, 'missing keys'):
            self.start(workflow_globals=self.values, workflow_name='example', config={'SIMULATE': True}, show_gui=False)
        self.hardware.assert_not_called()
        self.robot.assert_not_called()
        self.tracker.assert_not_called()

    def test_cancel_creates_no_run_artifacts_or_controllers(self):
        self.review.side_effect = lambda instance: setattr(instance, '_workflow_should_continue', False)
        coordinator = self.start(workflow_globals=self.values, workflow_name='example')
        self.assertIsNone(coordinator.workflow_config)
        self.assertFalse(self.logs.exists())
        self.tracker.assert_not_called()
        self.robot.assert_not_called()

    def test_plain_legacy_constructor_still_uses_explicit_mode_and_vial(self):
        with patch.object(ConfigManager, 'resolve_workflow_config', side_effect=AssertionError('No workflow config expected')):
            coordinator = self.start(vial_file=str(self.original), simulate=True, show_gui=False)
        self.assertTrue(coordinator.simulate)
        self.assertEqual(coordinator.nr_robot.VIAL_FILE, str(self.original))
        self.assertIsNone(coordinator.workflow_config)

    def test_corrupted_yaml_does_not_silently_replace_reviewed_settings(self):
        folder = Path(ConfigManager.CONFIG_DIR)
        folder.mkdir()
        path = folder / 'example.yaml'
        path.write_text('invalid: [')
        with self.assertRaises(yaml.YAMLError):
            self.start(workflow_globals=self.values, workflow_name='example')
        self.robot.assert_not_called()
        self.assertEqual(path.read_text(), 'invalid: [')


if __name__ == '__main__':
    unittest.main()
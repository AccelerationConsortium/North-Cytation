import importlib
import builtins
import io
import logging
import random
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'sdl_pipette_calibration' / 'protocols'))


class NorthProtocolSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(sys.modules, {'master_usdl_coordinator': types.SimpleNamespace(Lash_E=Mock())}):
            cls.module = importlib.import_module('calibration_protocol_northrobot')

    def setUp(self):
        self.protocol = self.module.HardwareCalibrationProtocol()
        self.robot = Mock(spec_set=['home_robot_components', 'move_vial_to_location',
                                  'normalize_vial_index', 'is_vial_pipetable',
                                  '_ensure_vial_accessible_for_pipetting', 'aspirate_from_vial',
                                  'dispense_into_vial', 'get_location', 'c9', 'remove_pipet',
                                  'return_vial_home', 'move_home', 'get_vial_info'])
        self.robot.c9 = Mock(spec_set=['goto', 'network'])
        self.robot.c9.network = Mock(spec_set=['disconnect'])
        self.robot.is_vial_pipetable.return_value = True
        self.lash = types.SimpleNamespace(nr_robot=self.robot, simulate=True)
        self.config = {'experiment': {'liquid': 'water', 'simulate': True,
                                     'volume_targets_ml': [0.02], 'continuous_monitoring': True,
                                     'max_retries_per_measurement': 0, 'adjust_volume': True,
                                     'quality_std_threshold_g': 0.1}}

    def test_borrowed_controller_initialization_and_idempotent_cleanup(self):
        original = deepcopy(self.config)
        with self.protocol.workflow_session(self.lash, self.config, 'water'):
            state = self.protocol.initialize({'experiment': {'liquid': 'water'}})
            self.assertIs(state['lash_e'], self.lash)
            self.assertEqual(state['source_vial'], state['measurement_vial'])
            self.assertFalse(state['swap_enabled'])
            self.assertAlmostEqual(self.protocol.conditioning_volume, 0.024)
            self.assertEqual(self.robot.aspirate_from_vial.call_count, 4)
            self.protocol.wrapup(state)
            self.protocol.wrapup(state)
            self.robot.remove_pipet.assert_called_once()
            self.robot.c9.network.disconnect.assert_not_called()
        self.assertIsNone(self.protocol._workflow_binding)
        self.assertEqual(original, self.config)

    def test_binding_cleared_on_interrupt_and_nested_binding_rejected(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.protocol.workflow_session(self.lash, self.config, 'water'):
                with self.assertRaises(RuntimeError):
                    with self.protocol.workflow_session(self.lash, self.config, 'water'):
                        pass
                raise KeyboardInterrupt()
        self.assertIsNone(self.protocol._workflow_binding)

    def test_standalone_cleanup_still_disconnects(self):
        state = {'lash_e': self.lash, 'measurement_vial': 'water', 'simulate': True}
        self.protocol.wrapup(state)
        self.robot.c9.network.disconnect.assert_called_once()

    def test_compensated_volume_cannot_overflow_tip(self):
        parameters = {'overaspirate_vol': 0.02, 'post_asp_air_vol': 0.0,
                      'pre_asp_air_vol': 0.0}
        with self.assertRaisesRegex(ValueError, 'tip capacity'):
            self.protocol.validate_workflow_capacity(0.19, parameters)
        parameters['overaspirate_vol'] = 0.01
        self.protocol.validate_workflow_capacity(0.19, parameters)
        parameters['pre_asp_air_vol'] = 0.81
        with self.assertRaisesRegex(ValueError, 'maximum safe volume'):
            self.protocol.validate_workflow_capacity(0.19, parameters)


class NorthProtocolMeasurementTests(unittest.TestCase):
    setUpClass = classmethod(NorthProtocolSessionTests.setUpClass.__func__)

    def setUp(self):
        NorthProtocolSessionTests.setUp(self)
        self.lash.logger = Mock(spec_set=logging.Logger)
        self.config['experiment']['random_seed'] = 30
        self.robot.get_vial_info.return_value = 10.0
        self.parameters = {
            'aspirate_speed': 12.0, 'dispense_speed': 10.0,
            'aspirate_wait_time': 0.0, 'dispense_wait_time': 1.5,
            'pre_asp_air_vol': 0.0, 'retract_speed': 5.0,
            'blowout_vol': 0.0, 'post_asp_air_vol': 0.0,
            'post_retract_wait_time': 0.0, 'asp_disp_cycles': 0.0,
            'overaspirate_vol': 0.004,
        }

    def test_required_parameters_rejected_before_robot_calls(self):
        with self.protocol.workflow_session(self.lash, self.config, 'water'):
            state = self.protocol.initialize({'experiment': {'liquid': 'water'}})
            self.robot.reset_mock()
            for nested in (False, True):
                for name in self.parameters:
                    with self.subTest(nested=nested, missing=name):
                        parameters = dict(self.parameters)
                        if nested:
                            parameters = {'overaspirate_vol': parameters.pop('overaspirate_vol'),
                                          'parameters': parameters}
                        source = parameters['parameters'] if nested and name != 'overaspirate_vol' else parameters
                        del source[name]
                        with self.assertRaisesRegex(ValueError, name):
                            self.protocol.measure(state, 0.02, parameters)
            self.assertEqual(self.robot.mock_calls, [])
            self.assertEqual(state['measurement_count'], 0)

    def test_routine_simulation_stays_near_target_and_retains_correction_effect(self):
        self.config['experiment']['volume_targets_ml'] = [0.02, 0.2, 0.55, 0.9]
        with self.protocol.workflow_session(self.lash, self.config, 'water'):
            state = self.protocol.initialize({'experiment': {'liquid': 'water'}})
            for volume in self.config['experiment']['volume_targets_ml']:
                parameters = dict(self.parameters, overaspirate_vol=0.0)
                measurements = self.protocol.measure(state, volume, parameters, replicates=32)
                for measurement in measurements:
                    self.assertLessEqual(abs(measurement['volume'] / volume - 1), .006 + 1e-12)
                parameters['overaspirate_vol'] = volume * .005
                corrected = self.protocol.measure(state, volume, parameters, replicates=32)
                for measurement in corrected:
                    self.assertLessEqual(abs(measurement['volume'] / volume - 1), .001 + 1e-12)

    def test_nonfinite_nonreal_and_bool_parameters_rejected(self):
        with self.protocol.workflow_session(self.lash, self.config, 'water'):
            state = self.protocol.initialize({'experiment': {'liquid': 'water'}})
            self.robot.reset_mock()
            for name in self.parameters:
                for value in (True, False, float('nan'), float('inf'), -float('inf'), '12', None, 1j):
                    with self.subTest(name=name, value=value):
                        parameters = dict(self.parameters, **{name: value})
                        with self.assertRaisesRegex(ValueError, name):
                            self.protocol.measure(state, 0.02, parameters)
            self.assertEqual(self.robot.mock_calls, [])

    def test_simulated_measurement_logs_ascii_and_runs_automation(self):
        with self.protocol.workflow_session(self.lash, self.config, 'water'):
            state = self.protocol.initialize({'experiment': {'liquid': 'water'}})
            self.robot.reset_mock()
            self.robot.get_vial_info.side_effect = RuntimeError('scale \u00b5L \u03bcL \u2713')
            with patch.object(builtins, 'print', side_effect=AssertionError('Bound print')):
                results = self.protocol.measure(state, 0.02, self.parameters, replicates=2)
                self.assertTrue(self.protocol._evaluate_measurement({
                    'pre_stable_count': 1, 'pre_total_count': 1, 'pre_baseline_std': 0.0,
                    'post_stable_count': 1, 'post_total_count': 1, 'post_baseline_std': 0.0,
                }))
            self.assertEqual(self.robot.aspirate_from_vial.call_count, 2)
            self.assertEqual(self.robot.dispense_into_vial.call_count, 2)
            self.robot.aspirate_from_vial.assert_called_with('water', 0.02, parameters=self.parameters)
            self.assertEqual(state['measurement_count'], 2)
            self.assertEqual(results[0]['retract_speed'], 5.0)
            self.assertEqual(results[0]['asp_disp_cycles'], 0.0)
            messages = [call.args[0] for call in self.lash.logger.info.call_args_list]
            self.assertTrue(all(message.isascii() for message in messages))
            self.assertTrue(any('scale uL uL ?' in message for message in messages))
            self.assertTrue(any('Measured:' in message for message in messages))
            self.assertTrue(any('Quality check:' in message for message in messages))

    def test_session_seed_restarts_noise_without_using_global_random(self):
        expected_random = random.Random(30)
        expected = [0.02 * 0.995 + 0.004 + expected_random.uniform(-0.001, 0.001) * 0.02
                    for replicate in range(3)]
        global_state = random.getstate()
        for phase in range(2):
            with self.protocol.workflow_session(self.lash, self.config, 'water'):
                state = self.protocol.initialize({'experiment': {'liquid': 'water'}})
                parameters = dict(self.parameters)
                parameters = {'overaspirate_vol': parameters.pop('overaspirate_vol'),
                              'parameters': parameters}
                with patch.object(random, 'uniform', side_effect=AssertionError('Global noise')):
                    results = self.protocol.measure(state, 0.02, parameters, replicates=3)
                self.assertEqual([result['volume'] for result in results], expected)
        self.assertEqual(random.getstate(), global_state)

    def test_missing_seed_fails_before_simulated_measurement(self):
        del self.config['experiment']['random_seed']
        with self.protocol.workflow_session(self.lash, self.config, 'water'):
            state = self.protocol.initialize({'experiment': {'liquid': 'water'}})
            self.robot.reset_mock()
            with self.assertRaisesRegex(ValueError, 'experiment.random_seed'):
                self.protocol.measure(state, 0.02, self.parameters)
            self.assertEqual(self.robot.mock_calls, [])

    def test_unbound_reporting_and_simulation_preserve_legacy_behavior(self):
        with patch.object(builtins, 'print') as output:
            self.protocol._report('legacy \u00b5L \u2713')
            self.protocol._report()
        self.assertEqual(output.call_args_list[0].args, ('legacy \u00b5L \u2713',))
        self.assertEqual(output.call_args_list[1].args, ('',))
        state = {'lash_e': self.lash, 'source_vial': 'water', 'measurement_vial': 'water',
                 'simulate': True, 'liquid': 'water', 'measurement_count': 0}
        with patch.object(random, 'uniform', return_value=0.001) as noise, \
                patch.object(builtins, 'print'):
            result = self.protocol.measure(state, 0.02, {})[0]
        noise.assert_called_once_with(-0.001, 0.001)
        self.assertAlmostEqual(result['volume'], 0.01992)
        self.assertEqual(result['overaspirate_vol'], 0.0)
        self.lash.logger.info.assert_not_called()

    def test_bound_state_cleanup_retains_controller_after_session_ends(self):
        with self.protocol.workflow_session(self.lash, self.config, 'water'):
            state = self.protocol.initialize({'experiment': {'liquid': 'water'}})
        self.protocol.wrapup(state, skip_physical_cleanup=True)
        self.robot.remove_pipet.assert_not_called()
        self.protocol.wrapup(state)
        self.protocol.wrapup(state)
        self.robot.remove_pipet.assert_called_once()
        self.robot.return_vial_home.assert_called_once_with('water')
        self.robot.move_home.assert_called_once()
        self.robot.c9.network.disconnect.assert_not_called()
        self.assertIs(state['lash_e'], self.lash)

    def test_standalone_initialization_creates_controller_and_runs_setup(self):
        real_open = builtins.open

        def open_existing_config(path, *args, **kwargs):
            if Path(path).name == 'experiment_config.yaml':
                path = ROOT / 'sdl_pipette_calibration' / 'experiment_config.yaml'
            return real_open(path, *args, **kwargs)

        slack = types.SimpleNamespace(send_slack_message=Mock(), safe_send_slack_message=Mock())
        with patch.dict(sys.modules, {'slack_agent': slack}), \
                patch.object(self.module, 'Lash_E', return_value=self.lash) as coordinator, \
                patch.object(builtins, 'open', side_effect=open_existing_config), \
                patch.object(builtins, 'print'):
            state = self.protocol.initialize(self.config)
            self.protocol.wrapup(state)
        coordinator.assert_called_once_with('status/calibration_vials_short.csv', simulate=True,
                                            initialize_biotek=False, show_gui=False)
        self.assertIs(state['lash_e'], self.lash)
        self.assertNotIn('workflow_managed', state)
        self.robot.home_robot_components.assert_called_once()
        self.robot.move_vial_to_location.assert_called_once_with('acetone', 'clamp', 0)
        self.assertEqual(self.robot.aspirate_from_vial.call_count, 4)
        self.assertEqual(self.robot.dispense_into_vial.call_count, 4)
        self.robot.normalize_vial_index.assert_called_once_with('acetone')
        self.robot.c9.goto.assert_called_once_with(self.robot.get_location.return_value)
        self.robot.c9.network.disconnect.assert_called_once()


class WorkflowConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(sys.modules, {'master_usdl_coordinator': types.SimpleNamespace(Lash_E=Mock())}):
            cls.workflow = importlib.import_module('workflows.pipette_calibration_workflow')
        cls.defaults = deepcopy({key: getattr(cls.workflow, key)
                                 for key in cls.workflow._CONFIG_KEYS})

    def setUp(self):
        self.config = deepcopy(self.defaults)
        self.base_path = ROOT / self.config['CALIBRATION_CONFIG_FILE']

    def test_confirmed_mapping_preserves_base_and_independent_validation_volumes(self):
        before = self.base_path.read_bytes()
        self.config.update(LIQUID='water', VOLUME_TARGETS_ML=[0.02],
                           VALIDATION_VOLUMES_ML=[0.015], MIN_GOOD_TRIALS=1,
                           ASPIRATE_SPEED_FIXED=True, ASPIRATE_SPEED_FIXED_VALUE=12,
                           RETRACT_SPEED_FIXED=False, ASP_DISP_CYCLES_FIXED=False)
        original = deepcopy(self.config)
        raw = self.workflow._effective_config(self.config, self.base_path)
        self.assertEqual(raw['experiment']['liquid'], 'water')
        self.assertEqual(raw['experiment']['volume_targets_ml'], [0.02])
        self.assertEqual(raw['validation']['volumes_ml'], [0.015])
        self.assertEqual(raw['optimization']['stopping_criteria']['min_good_trials'], 1)
        self.assertEqual(raw['experiment']['fixed_parameters']['aspirate_speed'], 12)
        self.assertNotIn('retract_speed', raw['experiment']['fixed_parameters'])
        self.assertNotIn('asp_disp_cycles', raw['experiment']['fixed_parameters'])
        self.assertNotIn('overaspirate_vol', raw['experiment']['fixed_parameters'])
        for name, bounds in [('retract_speed', [1, 99]), ('asp_disp_cycles', [0, 5])]:
            self.assertEqual(raw['hardware_parameters'][name]['type'], 'integer')
            self.assertEqual(raw['hardware_parameters'][name]['bounds'], bounds)
        self.assertEqual(raw['experiment']['protocol_override'], 'calibration_protocol_northrobot')
        self.assertEqual(self.config, original)
        self.assertEqual(self.base_path.read_bytes(), before)

    def test_invalid_parameter_bounds_and_flags_rejected(self):
        invalid = [('ASPIRATE_SPEED_FIXED', 1), ('ASPIRATE_SPEED_MIN', float('nan')),
                   ('ASPIRATE_SPEED_MAX', float('inf')),
                   ('ASPIRATE_SPEED_MIN', 30), ('ASPIRATE_SPEED_FIXED_VALUE', 31),
                   ('ASPIRATE_SPEED_MIN', 0), ('DISPENSE_SPEED_MAX', 41),
                   ('RETRACT_SPEED_MIN', 0), ('RETRACT_SPEED_MAX', 100),
                   ('RETRACT_SPEED_MIN', 1.5), ('OVERASPIRATE_VOL_MIN', 0.025),
                   ('OVERASPIRATE_VOL_MAX_FRACTION_OF_TARGET', 0)]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                config = deepcopy(self.config)
                config[key] = value
                with self.assertRaises(ValueError):
                    self.workflow._effective_config(config, self.base_path)

    def test_supplied_config_resolution_is_complete_and_has_no_yaml_io(self):
        from workflow_config_manager import ConfigManager
        self.config['INPUT_VIAL_STATUS_FILE'] = str(ROOT / self.config['INPUT_VIAL_STATUS_FILE'])
        namespace = dict(vars(self.workflow))
        with patch.object(ConfigManager, '_load_config_file', side_effect=AssertionError('YAML load')), \
                patch.object(ConfigManager, '_save_config_file', side_effect=AssertionError('YAML write')):
            resolved = ConfigManager.resolve_workflow_config(
                'pipette_calibration_workflow', namespace, self.config, show_gui=False)
            confirmed = ConfigManager.confirm_workflow_config(
                'pipette_calibration_workflow', namespace, resolved, supplied=True)
        self.assertEqual(confirmed, self.config)
        self.assertIsNot(confirmed, self.config)
        with self.assertRaisesRegex(ValueError, 'show_gui=False'):
            ConfigManager.resolve_workflow_config('pipette_calibration_workflow', namespace, self.config)
        del self.config['MIN_GOOD_TRIALS']
        with self.assertRaisesRegex(KeyError, 'MIN_GOOD_TRIALS'):
            ConfigManager.resolve_workflow_config(
                'pipette_calibration_workflow', namespace, self.config, show_gui=False)


class WorkflowExecutionTests(unittest.TestCase):
    setUpClass = classmethod(WorkflowConfigTests.setUpClass.__func__)

    def setUp(self):
        WorkflowConfigTests.setUp(self)
        self.temporary = tempfile.TemporaryDirectory(prefix='north_calibration_tests_')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source_bytes = self.base_path.read_bytes()
        raw = yaml.safe_load(self.source_bytes)
        raw['output']['base_directory'] = str(self.directory / 'output')
        raw['output']['generate_plots'] = False
        raw['validation']['generate_plots'] = False
        self.base_path = self.directory / 'scientific.yaml'
        self.base_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
        self.base_bytes = self.base_path.read_bytes()
        self.config.update(
            CALIBRATION_CONFIG_FILE=str(self.base_path),
            INPUT_VIAL_STATUS_FILE=str(ROOT / self.config['INPUT_VIAL_STATUS_FILE']),
            LIQUID='water', TARGET_VIAL='water', VOLUME_TARGETS_ML=[0.02],
            VALIDATION_VOLUMES_ML=[0.015], RANDOM_SEED=30,
            MAX_TOTAL_MEASUREMENTS=12, MAX_MEASUREMENTS_FIRST_VOLUME=12,
            NUM_SCREENING_TRIALS=6, MAX_REPLICATES_PER_TRIAL=1,
            TWO_POINT_CALIBRATION_REPLICATES=1, MIN_GOOD_TRIALS=1,
            REPLICATES_PER_VOLUME=1)
        for name in self.workflow._HARDWARE_PARAMETERS:
            self.config[name.upper() + '_FIXED'] = True
        self.robot = Mock(spec_set=[
            'VIAL_FILE', 'home_robot_components', 'move_vial_to_location',
            'normalize_vial_index', 'is_vial_pipetable',
            '_ensure_vial_accessible_for_pipetting', 'aspirate_from_vial',
            'dispense_into_vial', 'get_location', 'c9', 'remove_pipet',
            'return_vial_home', 'move_home', 'get_vial_info'])
        self.robot.VIAL_FILE = self.config['INPUT_VIAL_STATUS_FILE']
        self.robot.normalize_vial_index.return_value = 0
        self.robot.is_vial_pipetable.return_value = True
        self.robot.get_vial_info.return_value = 10.0
        self.robot.c9 = Mock(spec_set=['goto', 'network'])
        self.robot.c9.network = Mock(spec_set=['disconnect'])
        self.controllers = []
        self.experiments = []
        self.validators = []

    def fake_lash(self, *, initialize_biotek, workflow_globals, workflow_name,
                  config, show_gui):
        from workflow_config_manager import ConfigManager
        self.assertFalse(initialize_biotek)
        self.assertEqual(workflow_name, 'pipette_calibration_workflow')
        namespace = dict(workflow_globals)
        launch = ConfigManager.resolve_workflow_config(
            workflow_name, namespace, config=config, show_gui=show_gui)
        confirmed = ConfigManager.confirm_workflow_config(
            workflow_name, namespace, launch, supplied=config is not None)
        controller = types.SimpleNamespace(
            nr_robot=self.robot, simulate=confirmed['SIMULATE'],
            workflow_config=confirmed, _workflow_should_continue=True,
            logger=Mock(spec_set=logging.Logger))
        self.controllers.append(controller)
        return controller

    def test_real_engine_runs_share_singleton_and_exact_csv_handoff(self):
        self.assert_real_run(deepcopy(self.config), show_gui=False)

    def test_execute_rejects_partial_config_and_supplied_config_with_gui(self):
        partial = deepcopy(self.config)
        del partial['MAX_TOTAL_MEASUREMENTS']
        with patch.object(self.workflow, 'Lash_E', side_effect=self.fake_lash):
            with self.assertRaisesRegex(KeyError, 'MAX_TOTAL_MEASUREMENTS'):
                self.workflow.execute(config=partial, show_gui=False)
            with self.assertRaisesRegex(ValueError, 'show_gui=False'):
                self.workflow.execute(config=self.config, show_gui=True)
        self.assertEqual(self.controllers, [])
        self.robot.home_robot_components.assert_not_called()

    def test_invalid_confirmed_measurement_settings_and_budgets_rejected(self):
        invalid = [('SIMULATE', False), ('ADJUST_VOLUME', 1),
                   ('CONTINUOUS_MONITORING', 'true'), ('MAX_RETRIES_PER_MEASUREMENT', -1),
                   ('QUALITY_STD_THRESHOLD_G', 0), ('VOLUME_TARGETS_ML', []),
                   ('VALIDATION_VOLUMES_ML', [0.02, 0.02]),
                   ('MAX_MEASUREMENTS_FIRST_VOLUME', 7), ('MAX_TOTAL_MEASUREMENTS', 11)]
        lash = types.SimpleNamespace(simulate=True, nr_robot=self.robot)
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                config = deepcopy(self.config)
                config[key] = value
                with self.assertRaises(ValueError):
                    self.workflow.validate_experiment(config, lash)

    def test_base_fixed_overaspirate_and_disabled_csv_export_rejected(self):
        for section, key, value in [('experiment', 'fixed_parameters', {'overaspirate_vol': 0.004}),
                                    ('output', 'export_optimal_conditions', False)]:
            with self.subTest(key=key):
                raw = yaml.safe_load(self.base_bytes)
                raw[section][key] = value
                self.base_path.write_text(yaml.safe_dump(raw), encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.workflow._effective_config(self.config, self.base_path)

    def test_gui_confirmed_values_drive_real_engines_and_robot_calls(self):
        from workflow_config_manager import ConfigManager
        launch = deepcopy(self.config)
        self.config.update(TARGET_VIAL='reviewed_water', VOLUME_TARGETS_ML=[0.025],
                           VALIDATION_VOLUMES_ML=[0.012], ASPIRATE_SPEED_FIXED_VALUE=13,
                           ADJUST_VOLUME=False, CONTINUOUS_MONITORING=False,
                           QUALITY_STD_THRESHOLD_G=0.05)
        reviews = iter([launch, deepcopy(self.config)])

        def load(workflow_name, namespace, logger=None, strict=False):
            self.assertTrue(strict)
            namespace.update(next(reviews))

        with patch.object(ConfigManager, 'setup_config_if_missing'), \
                patch.object(ConfigManager, 'load_and_update_globals', side_effect=load) as loader:
            self.assert_real_run(None, show_gui=True)
        self.assertEqual(loader.call_count, 2)

    def assert_real_run(self, config, show_gui):
        from sdl_pipette_calibration import experiment as engine
        from sdl_pipette_calibration import run_validation as validation
        from sdl_pipette_calibration.protocol_loader import load_hardware_protocol
        from workflow_config_manager import ConfigManager
        protocol_module = importlib.import_module('calibration_protocol_northrobot')
        protocol = protocol_module.protocol_instance
        states = []
        echoes = []
        calibration_type = engine.CalibrationExperiment
        validation_type = validation.ValidationRunner
        original_initialize = protocol.initialize
        original_measure = protocol.measure
        original_import = builtins.__import__

        def import_without_slack(name, *args, **kwargs):
            if name == 'slack_agent':
                raise AssertionError('Simulation imported Slack')
            return original_import(name, *args, **kwargs)

        def initialize(config):
            self.robot.c9.network.disconnect.assert_not_called()
            state = original_initialize(config)
            states.append(state)
            return state

        def measure(state, volume, parameters, replicates=1):
            results = original_measure(state, volume, parameters, replicates)
            echoes.extend((state, deepcopy(parameters), result) for result in results)
            return results

        def experiment(config):
            instance = calibration_type(config)
            self.experiments.append(instance)
            return instance

        def validator(config, config_path):
            instance = validation_type(config, config_path)
            self.validators.append(instance)
            return instance

        random_state = random.getstate()
        self.addCleanup(random.setstate, random_state)
        random.seed(self.config['RANDOM_SEED'])
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.workflow, 'Lash_E', side_effect=self.fake_lash))
            stack.enter_context(patch.object(protocol_module, 'Lash_E', side_effect=AssertionError('Second coordinator')))
            stack.enter_context(patch.object(ConfigManager, '_load_config_file', side_effect=AssertionError('Workflow YAML load')))
            stack.enter_context(patch.object(ConfigManager, '_save_config_file', side_effect=AssertionError('Workflow YAML write')))
            stack.enter_context(patch.object(engine, 'CalibrationExperiment', side_effect=experiment))
            stack.enter_context(patch.object(validation, 'ValidationRunner', side_effect=validator))
            stack.enter_context(patch.object(protocol, 'initialize', side_effect=initialize))
            stack.enter_context(patch.object(protocol, 'measure', side_effect=measure))
            constraints = stack.enter_context(patch.object(
                protocol, 'get_parameter_constraints', wraps=protocol.get_parameter_constraints))
            stack.enter_context(patch.object(builtins, '__import__', side_effect=import_without_slack))
            stack.enter_context(redirect_stdout(io.StringIO()))
            result = self.workflow.execute(config=config, show_gui=show_gui)
        self.assertEqual(len(self.controllers), 1)
        self.assertEqual(len(self.experiments), 1)
        self.assertEqual(len(self.validators), 1)
        experiment = self.experiments[0]
        validator = self.validators[0]
        self.assertIsInstance(experiment, calibration_type)
        self.assertIsInstance(validator, validation_type)
        constraints.assert_any_call(self.config['VOLUME_TARGETS_ML'][0])
        self.assertIs(experiment.protocol_module, protocol)
        self.assertIs(validator.protocol.protocol, protocol)
        self.assertIs(load_hardware_protocol('calibration_protocol_northrobot'), protocol)
        self.assertIsNone(protocol._workflow_binding)
        self.assertEqual(len(states), 2)
        for state in states:
            self.assertIs(state['lash_e'], self.controllers[0])
            self.assertEqual(state['source_vial'], self.config['TARGET_VIAL'])
            self.assertEqual(state['measurement_vial'], self.config['TARGET_VIAL'])
            self.assertFalse(state['swap_enabled'])
            self.assertEqual(state['continuous_mass_monitoring'], self.config['CONTINUOUS_MONITORING'])
            self.assertEqual(state['adjust_volume'], self.config['ADJUST_VOLUME'])
            self.assertEqual(state['max_retries_per_measurement'], 0)
            self.assertTrue(state['physical_cleanup_done'])
        self.assertGreater(experiment.total_measurements, 0)
        self.assertLessEqual(experiment.total_measurements, 12)
        self.assertEqual(len(echoes), experiment.total_measurements + 1)
        for state, parameters, echo in echoes:
            hardware = parameters['parameters'] if 'parameters' in parameters else parameters
            for name in self.workflow._HARDWARE_PARAMETERS:
                expected = self.config[name.upper() + '_FIXED_VALUE']
                self.assertEqual(hardware[name], expected)
                self.assertEqual(echo[name], expected)
            self.assertEqual(echo['overaspirate_vol'], parameters['overaspirate_vol'])
            self.assertEqual(echo['measurement_budget_consumed'], 1)
        artifact = Path(experiment.output_dir).resolve() / 'optimal_conditions_water.csv'
        self.assertTrue(artifact.is_file())
        self.assertEqual(result['optimal_conditions_file'], str(artifact))
        self.assertEqual(validator.config._config['validation']['optimal_conditions_file'], str(artifact))
        self.assertEqual(validator.config._config['experiment']['volume_targets_ml'],
                 self.config['VALIDATION_VOLUMES_ML'])
        self.assertEqual([item['volume_target_ml'] for item in
                  result['validation_results']['validation_results']],
                 self.config['VALIDATION_VOLUMES_ML'])
        output = Path(result['output_dir'])
        for filename in ['calibration_effective_config.yaml', 'validation_effective_config.yaml',
                         'workflow_confirmed_config.yaml']:
            self.assertTrue((output / filename).is_file())
        self.assertTrue(output.is_relative_to(self.directory))
        confirmed = yaml.safe_load((output / 'workflow_confirmed_config.yaml').read_text(encoding='utf-8'))
        self.assertEqual(confirmed, self.config)
        calibration = experiment.config._config
        self.assertEqual(calibration['experiment']['volume_targets_ml'], self.config['VOLUME_TARGETS_ML'])
        for key in ['RANDOM_SEED', 'MAX_TOTAL_MEASUREMENTS', 'MAX_MEASUREMENTS_FIRST_VOLUME',
                'NUM_SCREENING_TRIALS', 'MAX_REPLICATES_PER_TRIAL',
                'TWO_POINT_CALIBRATION_REPLICATES', 'QUALITY_STD_THRESHOLD_G']:
            self.assertEqual(calibration['experiment'][key.lower()], self.config[key])
        self.assertEqual(protocol.quality_std_threshold, self.config['QUALITY_STD_THRESHOLD_G'])
        self.robot.c9.network.disconnect.assert_called_once()
        self.assertEqual(self.robot.remove_pipet.call_count, 2)
        self.assertEqual(self.robot.aspirate_from_vial.call_count, len(echoes) + 8)
        self.assertEqual(self.robot.dispense_into_vial.call_count, len(echoes) + 8)
        calls = self.robot.aspirate_from_vial.call_args_list
        self.assertEqual(calls[0].args, (self.config['TARGET_VIAL'],
                                       self.config['VOLUME_TARGETS_ML'][0] * 1.2))
        self.assertEqual(calls[experiment.total_measurements + 4].args,
                         (self.config['TARGET_VIAL'], self.config['VALIDATION_VOLUMES_ML'][0] * 1.2))
        self.assertEqual(self.base_path.read_bytes(), self.base_bytes)
        self.assertEqual((ROOT / self.defaults['CALIBRATION_CONFIG_FILE']).read_bytes(), self.source_bytes)

    def test_calibration_failure_and_interrupt_preserve_original_despite_cleanup_failure(self):
        from sdl_pipette_calibration.experiment import CalibrationExperiment
        from sdl_pipette_calibration import run_validation
        for failure in [RuntimeError('calibration sentinel'), KeyboardInterrupt('interrupted sentinel')]:
            with self.subTest(failure=type(failure).__name__):
                self.robot.reset_mock()
                self.controllers.clear()
                self.robot.remove_pipet.side_effect = RuntimeError('cleanup sentinel')
                with patch.object(self.workflow, 'Lash_E', side_effect=self.fake_lash), \
                        patch.object(CalibrationExperiment, 'run', side_effect=failure), \
                        patch.object(run_validation, 'ValidationRunner') as validator, \
                        patch.object(self.workflow, '_send_workflow_slack') as slack:
                    with self.assertRaises(type(failure)) as raised:
                        self.workflow.execute(config=deepcopy(self.config), show_gui=False)
                self.assertIs(raised.exception, failure)
                validator.assert_not_called()
                self.robot.return_vial_home.assert_called_once_with(self.config['TARGET_VIAL'])
                self.robot.move_home.assert_called_once()
                self.robot.c9.network.disconnect.assert_called_once()
                self.assertTrue(self.controllers[0].logger.exception.called)
                messages = [call.args[1] for call in slack.call_args_list]
                self.assertFalse(any('completed' in message for message in messages))
                self.assertIn('cleanup attempted', messages[-1])
                protocol = importlib.import_module('calibration_protocol_northrobot').protocol_instance
                self.assertIsNone(protocol._workflow_binding)

    def test_standalone_loaders_retain_protocol_selection_without_binding(self):
        from sdl_pipette_calibration.config_manager import ExperimentConfig
        from sdl_pipette_calibration.protocol_loader import create_protocol, load_hardware_protocol
        raw = yaml.safe_load(self.source_bytes)
        raw['experiment'].pop('protocol_override', None)
        protocol = importlib.import_module('calibration_protocol_northrobot').protocol_instance
        self.assertIsNone(protocol._workflow_binding)
        raw['experiment']['simulate'] = False
        live = ExperimentConfig(deepcopy(raw), str(self.base_path))
        wrapper = create_protocol(live, simulate=False)
        self.assertIs(wrapper.protocol, protocol)
        self.assertIs(load_hardware_protocol(live.get_protocol_module()), protocol)
        raw['experiment']['simulate'] = True
        simulated = ExperimentConfig(raw, str(self.base_path))
        simulated_wrapper = create_protocol(simulated, simulate=True)
        self.assertIsNot(simulated_wrapper.protocol, protocol)
        self.assertEqual(simulated.get_protocol_module(), 'calibration_protocol_simulated')
        self.assertIsNone(protocol._workflow_binding)


if __name__ == '__main__':
    unittest.main()

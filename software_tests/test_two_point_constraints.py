"""Two-point measurements must obey the same protocol limits as optimization."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sdl_pipette_calibration.experiment import CalibrationExperiment
from sdl_pipette_calibration.constraint_calibration import ConstraintCalibrator
from sdl_pipette_calibration.parameter_constraints import feasible_interval
from sdl_pipette_calibration.data_structures import (
    CalibrationParameters, HardwareParameters, PipettingParameters,
)


class TwoPointConstraintTests(unittest.TestCase):
    def run_calibration(self, target, baseline, air, capacity, measured):
        engine = object.__new__(CalibrationExperiment)
        engine.config = Mock()
        engine.config.calculate_tolerances_for_volume.return_value = SimpleNamespace(
            precision_tolerance_pct=3, accuracy_tolerance_ul=13.5)
        engine.config.get_two_point_calibration_replicates.return_value = 3
        engine.config.get_liquid_name.return_value = 'test_liquid'
        engine.config.get_parameter_constraints.return_value = []
        engine.protocol_module = Mock()
        engine.protocol_module.get_parameter_constraints.return_value = [
            f'pre_asp_air_vol + overaspirate_vol <= {capacity - target}',
        ]
        engine.current_volume_index = 1
        engine.volume_results = []
        tested = []

        def execute(parameters, volume, *args, **kwargs):
            self.assertLessEqual(volume + air + parameters.overaspirate_vol, capacity + 1e-9)
            tested.append(parameters.overaspirate_vol)
            return SimpleNamespace(
                measurements=[SimpleNamespace(
                    measured_volume_ml=measured + parameters.overaspirate_vol - baseline)],
                analysis=SimpleNamespace(cv_volume_pct=1))

        engine._execute_trial = Mock(side_effect=execute)
        parameters = PipettingParameters(
            CalibrationParameters(baseline), HardwareParameters({'pre_asp_air_vol': air}))
        calibrator = ConstraintCalibrator()
        with patch('sdl_pipette_calibration.constraint_calibration.ConstraintCalibrator', return_value=calibrator), patch.object(
                calibrator, 'calculate_two_point_bounds', wraps=calibrator.calculate_two_point_bounds) as calculate:
            _, update = engine._execute_two_point_measurement(parameters, target)
            calculation = calculate.call_args.kwargs
        self.assertLessEqual(update.max_value, capacity - target - air + 1e-9)
        self.assertLessEqual(update.optimal_overaspirate_ml, update.max_value)
        self.assertGreaterEqual(update.optimal_overaspirate_ml, update.min_value)
        self.assertEqual(calculation['point_1_overaspirate_ml'], tested[0])
        self.assertEqual(calculation['point_2_overaspirate_ml'], tested[1])
        self.assertNotAlmostEqual(tested[0], tested[1])
        self.assertEqual(parameters.overaspirate_vol, baseline)
        return tested

    def test_logged_900_ul_shortfall_is_bounded(self):
        tested = self.run_calibration(.9, .0399, 0, 1, .76093)
        self.assertAlmostEqual(tested[1], .1)

    def test_inherited_air_limits_both_points_and_can_reverse_direction(self):
        tested = self.run_calibration(.9, .0393, .08, 1, .76)
        self.assertAlmostEqual(tested[0], .02)
        self.assertLess(tested[1], tested[0])

    def test_other_hardware_capacity_is_not_limited_to_one_ml(self):
        tested = self.run_calibration(1.8, .04, .1, 2.5, 1.1)
        self.assertAlmostEqual(tested[1], .6)

    def test_already_feasible_probe_is_unchanged(self):
        tested = self.run_calibration(.55, .0399, 0, 1, .48)
        self.assertAlmostEqual(tested[0], .0399)
        self.assertAlmostEqual(tested[1], .1234)

    def test_inherited_measurement_rechecks_its_own_fixed_parameters(self):
        engine = object.__new__(CalibrationExperiment)
        engine.protocol_module = Mock()
        engine.protocol_module.get_parameter_constraints.return_value = [
            'pre_asp_air_vol + overaspirate_vol <= 0.1']
        engine.config = Mock()
        parameters = PipettingParameters(
            CalibrationParameters(.1), HardwareParameters({'pre_asp_air_vol': .08}))
        engine._create_inherited_parameters = Mock(return_value=parameters)
        engine._execute_trial = Mock()
        engine._log_inherited_trial_result = Mock()
        engine._run_inherited_trial(.9, SimpleNamespace(optimal_overaspirate_ml=.1))
        tested = engine._execute_trial.call_args.args[0]
        self.assertAlmostEqual(tested.overaspirate_vol, .02)
        self.assertEqual(parameters.overaspirate_vol, .1)


class GenericConstraintTests(unittest.TestCase):
    def test_coefficients_lower_bounds_and_both_sides(self):
        lower, upper = feasible_interval(
            ['2 * correction + offset <= 0.7', '-correction <= 0.01',
             'correction / 2 >= -0.003', 'correction <= 0.1 + correction / 2'],
            {'offset': .1}, 'correction')
        self.assertAlmostEqual(lower, -.006)
        self.assertAlmostEqual(upper, .2)

    def test_fixed_only_violation_missing_values_and_unsupported_expressions_fail(self):
        cases = [('offset <= 0', {'offset': .1}),
                 ('missing + correction <= 1', {}),
                 ('correction * correction <= 1', {}),
                 ('__import__("os") <= 1', {}),
                 ('correction < 1', {})]
        for constraint, fixed in cases:
            with self.subTest(constraint=constraint), self.assertRaises(ValueError):
                feasible_interval([constraint], fixed, 'correction')

    def test_no_distinct_points_fails_before_any_measurement(self):
        engine = object.__new__(CalibrationExperiment)
        engine.config = Mock()
        engine.config.calculate_tolerances_for_volume.return_value = SimpleNamespace(
            precision_tolerance_pct=3, accuracy_tolerance_ul=1)
        engine.protocol_module = Mock()
        engine.protocol_module.get_parameter_constraints.return_value = [
            'overaspirate_vol <= 0', 'overaspirate_vol >= 0']
        engine._execute_trial = Mock()
        with self.assertRaisesRegex(ValueError, 'two distinct calibration points'):
            engine._execute_two_point_measurement(
                PipettingParameters(CalibrationParameters(0), HardwareParameters({})), .1)
        engine._execute_trial.assert_not_called()


if __name__ == '__main__':
    unittest.main()

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from sdl_pipette_calibration.experiment import CalibrationExperiment
from sdl_pipette_calibration.data_structures import CalibrationParameters, HardwareParameters, PipettingParameters
from sdl_pipette_calibration.run_validation import ValidationRunner
from sdl_pipette_calibration.pipetting_wizard import PipettingWizard


class ProbeComplete(Exception):
    pass


class ConstraintHandoffTests(unittest.TestCase):
    def test_rounded_optimizer_candidate_is_constrained_before_measurement(self):
        engine = object.__new__(CalibrationExperiment)
        engine.protocol_module = Mock()
        engine.protocol_module.get_parameter_constraints.return_value = [
            'overaspirate_vol + pre_asp_air_vol <= 0.1']
        engine._execute_protocol_measurement = Mock(side_effect=ProbeComplete)
        candidate = PipettingParameters(
            CalibrationParameters(.0844), HardwareParameters({'pre_asp_air_vol': .016}))
        with self.assertRaises(ProbeComplete):
            engine._execute_trial(candidate, .9, 'rounded', force_replicates=1)
        tested = engine._execute_protocol_measurement.call_args.args[0]
        self.assertLessEqual(tested.overaspirate_vol + tested.get_hardware_param('pre_asp_air_vol'), .1 + 1e-9)
        self.assertEqual(candidate.overaspirate_vol, .0844)

    def test_transfer_learning_preserves_explicit_fixed_zero(self):
        engine = object.__new__(CalibrationExperiment)
        engine.current_volume_index = 1
        engine.config = Mock()
        engine.config.get_fixed_parameters.return_value = {'pre_asp_air_vol': 0.0}
        engine.config.get_volume_dependent_parameters.return_value = ['pre_asp_air_vol', 'overaspirate_vol']
        engine.config.get_optimizer_backend_subsequent.return_value = 'GPEI'
        engine.volume_results = [SimpleNamespace(optimal_parameters=PipettingParameters(
            CalibrationParameters(.04), HardwareParameters({'pre_asp_air_vol': .227})))]
        engine.protocol_module = Mock()
        with patch('sdl_pipette_calibration.experiment.create_optimizer', side_effect=ProbeComplete) as create:
            with self.assertRaises(ProbeComplete):
                engine._run_optimization_phase(.9, [], 20)
        self.assertEqual(create.call_args.kwargs['fixed_params']['pre_asp_air_vol'], 0)

    def test_compensated_csv_lookup_respects_protocol_capacity(self):
        runner = object.__new__(ValidationRunner)
        runner.logger = Mock()
        runner.pipetting_wizard = PipettingWizard()
        runner.protocol = SimpleNamespace(protocol=Mock())
        runner.protocol.protocol.get_parameter_constraints.return_value = [
            'overaspirate_vol + pre_asp_air_vol <= 0.1']
        data = pd.DataFrame([{'volume_target_ul': 900, 'volume_measured_ml': .828,
                              'calibration_overaspirate_vol': .1,
                              'hardware_parameters_pre_asp_air_vol': 0}])
        parameters = runner._get_parameters_for_volume(.9, data)
        self.assertAlmostEqual(parameters['overaspirate_vol'], .1)
        self.assertEqual(data.iloc[0]['calibration_overaspirate_vol'], .1)

    def test_completed_volume_saves_incremental_csv(self):
        engine = object.__new__(CalibrationExperiment)
        engine.config = Mock()
        engine.config.get_liquid_name.return_value = 'DMSO'
        parameters = PipettingParameters(
            CalibrationParameters(.04), HardwareParameters({'pre_asp_air_vol': 0}))
        trial = SimpleNamespace(analysis=SimpleNamespace(
            mean_volume_ml=.2, absolute_deviation_pct=0, cv_volume_pct=1, mean_duration_s=1))
        engine.volume_results = [SimpleNamespace(
            optimal_parameters=parameters, best_trials=[trial], target_volume_ml=.2,
            trials=[trial], measurement_count=3)]
        with tempfile.TemporaryDirectory() as directory:
            engine.output_dir = Path(directory)
            engine._save_incremental_optimal_conditions()
            data = pd.read_csv(engine.output_dir / 'optimal_conditions_DMSO_incremental.csv')
            self.assertEqual(len(data), 1)
            self.assertAlmostEqual(data.iloc[0]['overaspirate_vol'], .04)


if __name__ == '__main__':
    unittest.main()

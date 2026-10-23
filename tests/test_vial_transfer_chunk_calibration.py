import logging
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from North_Safe import North_Robot
from pipetting_data.pipetting_parameters import PipettingParameters
from pipetting_data.pipetting_wizard import PIPETTING_PARAMETERS, PipettingWizard


class VialTransferChunkCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.robot = object.__new__(North_Robot)
        self.robot.logger = logging.getLogger(__name__)
        self.robot.PIPET_TIPS = {
            'small_tip': {'volume': 0.2},
            'large_tip': {'volume': 1.0},
        }
        self.lookup_volumes = []
        self.lookup_flags = []
        self.overhead = lambda volume: 0.02
        self.real_lookup = PipettingWizard.get_pipetting_parameters
        self.calibrated_max = self.stack.enter_context(patch.object(
            PipettingWizard, 'get_max_calibrated_volume', autospec=True,
            return_value=None))

        def calibration(wizard, liquid, volume, compensate_overvolume=True,
                         tip_type=None):
            self.lookup_volumes.append(volume)
            self.lookup_flags.append(compensate_overvolume)
            if volume > 1.0:
                wizard.logger.warning('Calibration requested above tip capacity')
                return None
            return {'overaspirate_vol': self.overhead(volume),
                    'pre_asp_air_vol': 0.01, 'post_asp_air_vol': 0.01,
                    'aspirate_speed': 17, 'dispense_wait_time': volume}

        self.lookup = self.stack.enter_context(patch.object(
            PipettingWizard, 'get_pipetting_parameters', autospec=True,
            side_effect=calibration))
        self.stack.enter_context(patch.object(
            North_Robot, 'normalize_vial_index', autospec=True,
            side_effect=lambda robot, name: {'source': 0, 'dest': 1}[name]))
        self.aspirate = self.stack.enter_context(patch.object(
            North_Robot, 'aspirate_from_vial', autospec=True))
        self.dispense = self.stack.enter_context(patch.object(
            North_Robot, 'dispense_into_vial', autospec=True, return_value=None))
        self.stack.enter_context(patch.object(
            North_Robot, 'get_vial_in_location', autospec=True, return_value=None))

    def transfer(self, volume, **kwargs):
        return self.robot.dispense_from_vial_into_vial(
            'source', 'dest', volume, remove_tip=False,
            return_vial_home=False, **kwargs)

    def assert_transfers(self, expected_volumes):
        self.assertEqual(self.aspirate.call_count, len(expected_volumes))
        self.assertEqual(self.dispense.call_count, len(expected_volumes))
        for aspirate_call, dispense_call, expected in zip(
                self.aspirate.call_args_list, self.dispense.call_args_list,
                expected_volumes):
            self.assertAlmostEqual(aspirate_call.args[2], round(expected, 3))
            self.assertAlmostEqual(dispense_call.args[2], expected)
            self.assertIs(aspirate_call.kwargs['parameters'],
                          dispense_call.kwargs['parameters'])

    def test_1250ul_calibrates_only_625ul_chunks_without_warning(self):
        with patch.object(self.robot.logger, 'warning') as warning:
            self.transfer(1.25, liquid='water')
        warning.assert_not_called()
        self.assertEqual(self.lookup_volumes, [0.625])
        self.assert_transfers([0.625, 0.625])
        parameters = self.aspirate.call_args.kwargs['parameters']
        self.assertEqual(parameters.overaspirate_vol, 0.02)
        self.assertEqual(parameters.dispense_wait_time, 0.625)

    def test_partial_override_preserves_chunk_calibration_and_flags(self):
        overrides = {'aspirate_speed': 5}
        self.transfer(1.25, liquid='water', parameters=overrides,
                      compensate_overvolume=False)
        parameters = self.aspirate.call_args.kwargs['parameters']
        self.assertEqual(parameters.aspirate_speed, 5)
        self.assertEqual(parameters.overaspirate_vol, 0.02)
        self.assertEqual(self.lookup_flags, [False])
        self.assertEqual(overrides, {'aspirate_speed': 5})

    def test_object_override_keeps_existing_semantics(self):
        overrides = PipettingParameters(aspirate_speed=5, overaspirate_vol=0.03)
        self.transfer(1.25, liquid='water', parameters=overrides)
        parameters = self.aspirate.call_args.kwargs['parameters']
        self.assertEqual(parameters, overrides)
        self.assertIsNot(parameters, overrides)
        self.assert_transfers([0.625, 0.625])

    def test_overhead_recalculated_until_chunk_fits(self):
        self.overhead = lambda volume: 0.1 if volume > 0.5 else 0.6
        self.transfer(0.9, liquid='water')
        self.assertEqual(self.lookup_volumes, [0.9, 0.45, 0.3])
        self.assert_transfers([0.3, 0.3, 0.3])
        self.assertEqual(self.aspirate.call_args.kwargs['parameters'].overaspirate_vol, 0.6)

    def test_impossible_overhead_stops_before_automation(self):
        self.overhead = lambda volume: 1.1
        with self.assertRaisesRegex(ValueError, 'capacity'):
            self.transfer(1.25, liquid='water')
        self.aspirate.assert_not_called()
        self.dispense.assert_not_called()

    def test_specified_tip_capacity_controls_splits(self):
        self.transfer(0.3, liquid='water', specified_tip='small_tip')
        self.assertEqual(self.lookup_volumes, [0.15])
        self.assert_transfers([0.15, 0.15])

    def test_rounded_aspiration_cannot_exceed_capacity(self):
        self.overhead = lambda volume: 0.0132
        self.transfer(0.5, liquid='water', specified_tip='small_tip')
        self.assertEqual(self.lookup_volumes, [0.5 / 3])
        self.assert_transfers([0.5 / 3] * 3)
        for aspirate_call in self.aspirate.call_args_list:
            parameters = aspirate_call.kwargs['parameters']
            total = (aspirate_call.args[2] + parameters.overaspirate_vol +
                     parameters.post_asp_air_vol)
            self.assertLessEqual(total, 0.2)

    def test_no_liquid_uses_defaults_and_splits(self):
        self.transfer(1.25)
        self.calibrated_max.assert_not_called()
        self.lookup.assert_not_called()
        self.assert_transfers([0.625, 0.625])

    def test_large_preparation_respects_liquid_calibration_maximum(self):
        self.calibrated_max.return_value = 0.8
        self.transfer(5.4, liquid='acetone')
        self.assertEqual(self.lookup_volumes, [5.4 / 7])
        self.assert_transfers([5.4 / 7] * 7)

    def test_calibrated_maximum_does_not_replace_tip_capacity(self):
        self.calibrated_max.return_value = 0.8
        self.transfer(0.3, liquid='water', specified_tip='small_tip')
        self.assert_transfers([0.15, 0.15])

    def test_single_transfer_retains_calibration(self):
        self.transfer(0.1, liquid='water')
        self.assertEqual(self.lookup_volumes, [0.1])
        self.assert_transfers([0.1])
        self.assertEqual(self.aspirate.call_args.kwargs['parameters'].aspirate_speed, 17)

    def test_exact_tip_capacity_splits_to_allow_air_gaps(self):
        self.transfer(1.0, liquid='water')
        self.assertEqual(self.lookup_volumes, [0.5])
        self.assert_transfers([0.5, 0.5])

    def test_unknown_specified_tip_stops_before_automation(self):
        with self.assertRaises(KeyError):
            self.transfer(1.25, specified_tip='unknown')
        self.lookup.assert_not_called()
        self.aspirate.assert_not_called()
        self.dispense.assert_not_called()

    def calibration_frame(self, volumes):
        defaults = PipettingParameters()
        frame = pd.DataFrame({
            name: [getattr(defaults, name)] * len(volumes)
            for name in PIPETTING_PARAMETERS
        })
        frame['volume_target'] = volumes
        frame['volume_measured'] = volumes
        frame['overaspirate_vol'] = [0.02, 0.04]
        return frame

    def test_real_wizard_interpolates_chunks_without_total_volume_warning(self):
        frame = self.calibration_frame([400, 800])
        self.calibrated_max.return_value = 0.8
        with patch.object(PipettingWizard, 'get_pipetting_parameters',
                          new=self.real_lookup), \
                patch.object(PipettingWizard, 'find_best_calibration_file',
                             autospec=True, return_value=(Path('water.csv'), frame)), \
                patch.object(self.robot.logger, 'warning') as warning:
            self.transfer(1.25, liquid='water')
        warning.assert_not_called()
        self.assert_transfers([0.625, 0.625])
        self.assertAlmostEqual(
            self.aspirate.call_args.kwargs['parameters'].overaspirate_vol, 0.03125)

    def test_real_wizard_still_warns_for_chunk_below_calibrated_minimum(self):
        frame = self.calibration_frame([700, 800])
        self.calibrated_max.return_value = 0.8
        with patch.object(PipettingWizard, 'get_pipetting_parameters',
                          new=self.real_lookup), \
                patch.object(PipettingWizard, 'find_best_calibration_file',
                             autospec=True, return_value=(Path('water.csv'), frame)), \
                patch.object(self.robot.logger, 'warning') as warning:
            self.transfer(1.25, liquid='water')
        warning.assert_called_once()
        message = warning.call_args.args[0]
        self.assertIn('625.0uL', message)
        self.assertIn('below available range', message)
        self.assertNotIn('1250', message)
        self.assert_transfers([0.625, 0.625])

    def test_real_wizard_avoids_900ul_warning_for_large_preparation(self):
        frame = self.calibration_frame([5, 800])
        self.calibrated_max.return_value = 0.8
        with patch.object(PipettingWizard, 'get_pipetting_parameters',
                          new=self.real_lookup), \
                patch.object(PipettingWizard, 'find_best_calibration_file',
                             autospec=True, return_value=(Path('acetone.csv'), frame)), \
                patch.object(self.robot.logger, 'warning') as warning:
            self.transfer(5.4, liquid='acetone')
        warning.assert_not_called()
        self.assert_transfers([5.4 / 7] * 7)

    def test_nonpositive_volume_does_not_lookup_or_transfer(self):
        for volume in (0, -0.1):
            self.transfer(volume, liquid='water')
        self.lookup.assert_not_called()
        self.aspirate.assert_not_called()
        self.dispense.assert_not_called()


class CalibratedVolumeLimitTests(unittest.TestCase):
    def test_maximum_across_valid_files_is_returned_in_ml(self):
        wizard = PipettingWizard()
        files = [Path('small.csv'), Path('large.csv'), Path('invalid.csv')]
        frames = [pd.DataFrame({'volume_target': [5, 50]}),
                  pd.DataFrame({'volume_target': [5, 800]}), None]
        with patch.object(wizard, 'find_calibration_files', return_value=files), \
                patch.object(wizard, 'load_calibration_data', side_effect=frames):
            self.assertEqual(wizard.get_max_calibrated_volume('water'), 0.8)

    def test_missing_or_invalid_calibration_has_no_limit(self):
        wizard = PipettingWizard()
        with patch.object(wizard, 'find_calibration_files', return_value=[]):
            self.assertIsNone(wizard.get_max_calibrated_volume('unknown'))
        frame = pd.DataFrame({'volume_target': [None, float('inf'), -5, 0, 'bad']})
        with patch.object(wizard, 'find_calibration_files', return_value=[Path('bad.csv')]), \
                patch.object(wizard, 'load_calibration_data', return_value=frame):
            self.assertIsNone(wizard.get_max_calibrated_volume('water'))


if __name__ == '__main__':
    unittest.main()

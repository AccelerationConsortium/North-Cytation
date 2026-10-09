import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from workflows import pipette_calibration_workflow as workflow


class CalibrationPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'optimal_conditions_DMSO.csv'
        self.row = {
            'volume_target_ul': 180.0, 'volume_measured_ml': .18,
            'calibration_overaspirate_vol': -.0016,
            'hardware_parameters_pre_asp_air_vol': 0.0,
            'hardware_parameters_post_asp_air_vol': 0.0,
        }
        self.raw = {
            'hardware_parameters': {'pre_asp_air_vol': {}, 'post_asp_air_vol': {}},
            'experiment': {'fixed_parameters': {}, 'volume_targets_ml': [.18]},
            'validation': {'volumes_ml': [.18]},
        }
        name = 'sdl_pipette_calibration.protocols.calibration_protocol_northrobot'
        self.protocol = importlib.import_module(name).protocol_instance
        patcher = patch.object(workflow, '_PROTOCOL_NAME', name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def preflight(self):
        pd.DataFrame([self.row]).to_csv(self.path, index=False)
        workflow._preflight_artifact(self.path, self.raw)

    def test_negative_exported_correction_reaches_capacity_check(self):
        with patch.object(self.protocol, 'validate_workflow_capacity',
                          wraps=self.protocol.validate_workflow_capacity) as capacity:
            self.preflight()
        capacity.assert_called_once()
        self.assertAlmostEqual(capacity.call_args.args[1]['overaspirate_vol'], -.0016)

    def test_compensation_can_turn_positive_correction_negative(self):
        self.row.update(calibration_overaspirate_vol=.001, volume_measured_ml=.183)
        with patch.object(self.protocol, 'validate_workflow_capacity',
                          wraps=self.protocol.validate_workflow_capacity) as capacity:
            self.preflight()
        self.assertAlmostEqual(capacity.call_args.args[1]['overaspirate_vol'], -.002)

    def test_nonfinite_corrections_are_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                self.row['calibration_overaspirate_vol'] = value
                with self.assertRaisesRegex(ValueError, 'calibration_overaspirate_vol must be a finite'):
                    self.preflight()

    def test_negative_hardware_and_nonpositive_volumes_are_rejected(self):
        for column, value in [('hardware_parameters_pre_asp_air_vol', -.001),
                              ('volume_target_ul', 0), ('volume_measured_ml', 0)]:
            with self.subTest(column=column):
                original = self.row[column]
                self.row[column] = value
                with self.assertRaisesRegex(ValueError, column):
                    self.preflight()
                self.row[column] = original

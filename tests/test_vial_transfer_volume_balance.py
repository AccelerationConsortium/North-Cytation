"""Real aspiration/dispense bookkeeping with hardware calls replaced by mocks."""
import logging
import unittest
from unittest.mock import Mock

import pandas as pd

from North_Safe import North_Robot
from pipetting_data.pipetting_parameters import PipettingParameters


def make_robot():
    robot = object.__new__(North_Robot)
    robot.simulate = True
    robot.logger = logging.getLogger(__name__)
    robot.c9 = Mock()
    robot.dispense_recording = 'off'
    robot.VIAL_DF = pd.DataFrame([
        dict(vial_name='water', location='clamp', location_index=0, vial_volume=8.0),
    ])
    robot.PIPET_FLUID_VOLUME = 0.0
    robot.HELD_PIPET_TYPE = 'small_tip'
    robot.normalize_vial_index = lambda vial: 0
    robot.get_vial_info = lambda vial, field: robot.VIAL_DF.at[0, field]
    robot.get_config_parameter = lambda *args, **kwargs: 8.0
    params = PipettingParameters()
    robot.resolve_pipetting_setup = lambda *args, **kwargs: (params, 'small_tip')
    robot._get_optimized_parameters = lambda *args: params
    robot.plan_vial_transfer = lambda volume, *args: (volume, 1, params, 'small_tip')
    robot._ensure_vial_accessible_for_pipetting = Mock(return_value=True)
    robot.is_vial_pipetable = Mock(return_value=True)
    for method in ('get_tip_if_needed', 'check_if_aspiration_volume_unacceptable',
                   'adjust_pump_speed', 'save_robot_status', 'pipet_aspirate',
                   'pipet_dispense'):
        setattr(robot, method, Mock())
    robot.get_vial_location = Mock(return_value=[0, 0, 0, 0])
    robot.get_aspirate_height = Mock(return_value=0)
    robot.get_height_at_location = Mock(return_value=0)
    robot.adjust_height_based_on_pipet_held = lambda height: height
    robot.get_speed = Mock(return_value=10)
    robot.get_vial_in_location = Mock(return_value=None)
    robot.pause_after_error = Mock(side_effect=AssertionError('Capacity check paused transfer'))
    return robot


class VialTransferVolumeBalanceTests(unittest.TestCase):
    def test_full_vial_self_transfer_preserves_volume(self):
        for volume in (0.18746828616347772, 0.1875, 0.1876):
            for repeats in (1, 3):
                with self.subTest(volume=volume, repeats=repeats):
                    robot = make_robot()
                    params = PipettingParameters()
                    robot.plan_vial_transfer = lambda *args: (volume, repeats, params, 'small_tip')
                    for _ in range(3):
                        robot.dispense_from_vial_into_vial(
                            'water', 'water', volume * repeats, remove_tip=False,
                            return_vial_home=False, record_measurement=False)
                        self.assertAlmostEqual(robot.VIAL_DF.at[0, 'vial_volume'], 8.0, delta=1e-12)
                        self.assertAlmostEqual(robot.PIPET_FLUID_VOLUME, 0.0, delta=1e-12)
                    robot.pause_after_error.assert_not_called()
                    self.assertEqual(robot.pipet_aspirate.call_args.args[0], volume)

    def test_actual_overfill_still_stops_before_dispensing(self):
        robot = make_robot()
        with self.assertRaisesRegex(AssertionError, 'Capacity check'):
            robot.dispense_into_vial('water', 0.000001, record_measurement=False)
        robot.pipet_dispense.assert_not_called()

    def test_float_noise_at_capacity_does_not_stop_dispense(self):
        robot = make_robot()
        robot.VIAL_DF.at[0, 'vial_volume'] = 7.9 + 2e-15
        robot.dispense_into_vial('water', 0.1, record_measurement=False)
        robot.pause_after_error.assert_not_called()
        robot.pipet_dispense.assert_called_once()


if __name__ == '__main__':
    unittest.main()

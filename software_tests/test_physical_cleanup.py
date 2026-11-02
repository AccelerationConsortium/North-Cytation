from unittest.mock import Mock

from North_Safe import North_Robot


def test_physical_cleanup_returns_recorded_liquid_with_air_blowout():
    robot = object.__new__(North_Robot)
    robot.logger = Mock()
    robot.PIPET_FLUID_VIAL_INDEX = 7
    robot.PIPET_FLUID_VOLUME = 0.35
    robot.HELD_PIPET_TYPE = "small_tip"
    robot.GRIPPER_STATUS = None
    robot.c9 = Mock()
    robot.dispense_into_vial = Mock()
    robot.remove_pipet = Mock()
    robot.move_home = Mock()

    robot._perform_physical_cleanup()

    robot.dispense_into_vial.assert_called_once_with(
        7, 0.35, parameters={"blowout_vol": 1.0}
    )
    robot.remove_pipet.assert_called_once_with()
    robot.move_home.assert_called_once_with()


def test_physical_cleanup_does_not_dispense_when_tip_has_no_recorded_liquid():
    robot = object.__new__(North_Robot)
    robot.logger = Mock()
    robot.PIPET_FLUID_VIAL_INDEX = 7
    robot.PIPET_FLUID_VOLUME = 0
    robot.HELD_PIPET_TYPE = None
    robot.GRIPPER_STATUS = None
    robot.dispense_into_vial = Mock()
    robot.move_home = Mock()

    robot._perform_physical_cleanup()

    robot.dispense_into_vial.assert_not_called()
    robot.move_home.assert_called_once_with()

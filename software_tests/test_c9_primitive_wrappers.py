"""The C9 primitive wrappers send exactly the controller commands they replaced."""
import logging
from unittest.mock import MagicMock, call

import pytest

from North_Safe import North_Powder, North_Robot, North_Spin


@pytest.fixture
def robot():
    robot = object.__new__(North_Robot)
    robot.logger = logging.getLogger(__name__)
    robot.simulate = True
    robot.c9 = MagicMock()
    robot.ROBOT_HARDWARE = None  # fall back to robot_state/robot_hardware.yaml
    return robot


def test_vial_gripper_uses_output_7(robot):
    robot.close_vial_gripper()
    robot.open_vial_gripper()
    assert robot.c9.set_output.call_args_list == [call(7, True), call(7, False)]


def test_vial_gripper_reads_output_from_loaded_config(robot):
    robot.ROBOT_HARDWARE = {'robot_pneumatics': {'vial_gripper': 3}}
    robot.close_vial_gripper()
    robot.c9.set_output.assert_called_once_with(3, True)


def test_clamp_wrappers(robot):
    robot.open_clamp()
    robot.close_clamp()
    assert robot.c9.method_calls == [call.open_clamp(), call.close_clamp()]


def test_scale_wrappers_return_controller_values(robot):
    robot.c9.read_scale.return_value = (True, 12.5)
    robot.c9.read_steady_scale.return_value = 12.4
    assert robot.read_scale() == (True, 12.5)
    assert robot.read_steady_scale() == 12.4
    robot.zero_scale()
    robot.c9.zero_scale.assert_called_once_with()


@pytest.mark.parametrize('name, expected', [
    ('home', call(0, 0)),
    ('powder_dispense_mg', call(66.5, 70)),
])
def test_carousel_positions_match_previous_hard_coded_values(robot, name, expected):
    robot.move_carousel_to(name)
    assert robot.c9.move_carousel.call_args == expected


def test_move_home_still_homes_carousel(robot):
    robot.move_home()
    assert robot.c9.move_carousel.call_args == call(0, 0)


def test_centrifuge_outputs():
    spin = North_Spin(MagicMock(), MagicMock(), simulate=True, logger=logging.getLogger(__name__))
    spin.open_lid()
    spin.close_lid()
    spin.turn_on_vacuum()
    spin.turn_off_vacuum()
    assert spin.c9.set_output.call_args_list == [call(11, True), call(11, False), call(10, True), call(10, False)]


def test_powder_mg_dispense_moves_carousel_as_before(monkeypatch):
    powder = North_Powder(MagicMock(), simulate=True, logger=logging.getLogger(__name__))
    powder.p2 = MagicMock()
    monkeypatch.setattr(powder, 'cl_pow_dispense', MagicMock(return_value=5.0))
    assert powder.dispense_powder_mg(5.0) == 5.0
    assert powder.c9.move_carousel.call_args_list == [call(66.5, 70), call(0, 0)]

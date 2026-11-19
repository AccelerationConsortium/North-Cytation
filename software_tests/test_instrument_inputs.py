"""Heater, stirrer, spinner and track inputs are checked against robot_hardware.yaml before sending."""
import logging
from unittest.mock import MagicMock, call

import pytest
import yaml

from North_Safe import North_Temp

HARDWARE = yaml.safe_load(open('robot_state/robot_hardware.yaml', encoding='utf-8'))


@pytest.fixture
def temp():
    return North_Temp(MagicMock(), MagicMock(), simulate=True, logger=logging.getLogger(__name__))


# ---- heater ----

def test_heater_limits_are_in_robot_hardware_yaml():
    assert HARDWARE['heater_limits'] == {'min_c': 0, 'max_c': 250, 'warn_above_c': 100, 'channels': [0, 7]}


def test_set_temp_sends_the_value_unchanged(temp):
    temp.set_temp(40)
    assert temp.t8.mock_calls == [call.set_temp(0, 40), call.enable_channel(0)]


def test_autotune_sends_the_value_unchanged(temp):
    temp.autotune(60, channel=2)
    assert temp.t8.mock_calls == [call.disable_channel(2), call.set_temp(2, 60), call.enable_channel(2),
                                  call.temp_autotune(2, True)]


@pytest.mark.parametrize('method', ['set_temp', 'autotune'])
@pytest.mark.parametrize('target, channel, message', [
    (251, 0, 'outside 0-250 C'), (-1, 0, 'outside 0-250 C'), (float('nan'), 0, 'outside'),
    ('hot', 0, 'not a number'), (None, 0, 'not a number'),
    (40, 8, 'outside 0-7'), (40, -1, 'outside 0-7'), (40, 1.5, 'not a whole number')])
def test_heater_refuses_bad_inputs_before_sending(temp, method, target, channel, message):
    with pytest.raises(ValueError, match=f'refused before moving.*{message}'):
        getattr(temp, method)(target, channel=channel)
    assert temp.t8.mock_calls == []


def test_heater_warns_above_100_c_but_still_heats(temp, caplog):
    with caplog.at_level(logging.WARNING):
        temp.set_temp(150)
    assert 'above 100 C' in caplog.text
    assert call.set_temp(0, 150) in temp.t8.mock_calls


def test_heater_does_not_warn_at_100_c(temp, caplog):
    with caplog.at_level(logging.WARNING):
        temp.set_temp(100)
    assert 'above' not in caplog.text


# ---- stirrer and centrifuge (spin_axis takes motor counts/s) ----

@pytest.fixture
def spin():
    from North_Safe import North_Spin
    return North_Spin(MagicMock(), MagicMock(), simulate=True, logger=logging.getLogger(__name__))


def test_speed_limits_are_the_hardware_max():
    for section in ['stirrer_limits', 'spinner_limits']:
        assert HARDWARE[section] == {'min_speed': 1, 'max_speed': 100000}


def test_stirring_sends_the_speed_unchanged(temp):
    temp.turn_on_stirring()
    temp.turn_on_stirring(speed=1000)
    temp.turn_off_stirring()
    assert temp.c8.spin_axis.call_args_list == [call(1, 10000), call(1, 1000), call(1, 0)]


def test_spinner_sends_the_speed_unchanged(spin):
    spin.set_speed(20000)
    spin.stop_spin()
    assert spin.c8.spin_axis.call_args_list == [call(2, 20000), call(2, 0)]


@pytest.mark.parametrize('speed, message', [(0, 'outside 1-100000'), (100001, 'outside 1-100000'),
                                            (-5, 'outside'), (float('nan'), 'outside'), ('fast', 'not a number')])
def test_stirring_and_spinner_refuse_bad_speeds_before_sending(temp, spin, speed, message):
    with pytest.raises(ValueError, match=f'refused before moving.*{message}'):
        temp.turn_on_stirring(speed)
    with pytest.raises(ValueError, match=f'refused before moving.*{message}'):
        spin.set_speed(speed)
    temp.c8.spin_axis.assert_not_called()
    spin.c8.spin_axis.assert_not_called()


# ---- named move speeds: percent of max velocity (the C9 reads > 100 as counts/s) ----

def test_every_configured_speed_is_within_the_limits():
    limits = HARDWARE['speed_limits']
    assert limits == {'min_percent': 1, 'max_percent': 100}
    track = yaml.safe_load(open('robot_state/track_positions.yaml', encoding='utf-8'))
    for speed in list(HARDWARE['movement_speeds'].values()) + list(track['speeds'].values()):
        assert limits['min_percent'] <= speed <= limits['max_percent']


def _track(speeds):
    from North_Safe import North_Track
    track = object.__new__(North_Track)
    track.logger = logging.getLogger(__name__)
    track.ROBOT_HARDWARE = HARDWARE
    track.TRACK_POSITIONS = {'speeds': speeds}
    return track


def _robot(speeds):
    from North_Safe import North_Robot
    robot = object.__new__(North_Robot)
    robot.logger = logging.getLogger(__name__)
    robot.ROBOT_HARDWARE = {**HARDWARE, 'movement_speeds': speeds}
    robot.get_config_parameter = lambda config, section, key, **kw: robot.ROBOT_HARDWARE[section][key]
    return robot


def test_speeds_are_returned_unchanged():
    assert _track({'default_x': 50}).get_speed('default_x') == 50
    assert _track({}).get_speed('missing') == 50  # unchanged fallback, with a warning
    assert _robot({'retract': 5}).get_speed('retract') == 5


@pytest.mark.parametrize('speed', [0, 101, 5000, -1, float('nan'), 'fast'])
def test_out_of_range_configured_speeds_are_refused_before_moving(speed):
    with pytest.raises(ValueError, match=r"Track move refused before moving: speed 'default_x'.*track_positions.yaml"):
        _track({'default_x': speed}).get_speed('default_x')
    with pytest.raises(ValueError, match=r"Move refused before moving: speed 'retract'.*movement_speeds"):
        _robot({'retract': speed}).get_speed('retract')


# ---- move speeds passed by callers (move_speed / vel): None or 1-100 percent ----

SPEED_METHODS = [('dispense_from_vial_into_vial', ('a', 'b', 0.1)), ('dispense_into_vial', ('a', 0.1)),
                 ('dispense_into_open_vial', ('a', 0.1)), ('return_vial_home', ('a',)),
                 ('drop_off_vial', ('a', 'main_8mL_rack', 0)), ('grab_vial', ('a',)),
                 ('move_vial_to_location', ('a', 'clamp', 0)), ('uncap_clamp_vial', ()),
                 ('recap_clamp_vial', ()), ('goto_location_if_not_there', ('clamp',))]


def _bare_robot():
    from North_Safe import North_Robot
    robot = object.__new__(North_Robot)
    robot.logger = logging.getLogger(__name__)
    robot.ROBOT_HARDWARE = HARDWARE
    robot.c9 = MagicMock()
    return robot


@pytest.mark.parametrize('method, args', SPEED_METHODS)
@pytest.mark.parametrize('speed', [0, 101, 5000, float('nan'), 'fast'])
def test_caller_move_speeds_are_refused_before_moving(method, args, speed):
    robot = _bare_robot()
    with pytest.raises(ValueError, match=f'{method} refused before moving: move_speed'):
        getattr(robot, method)(*args, move_speed=speed)
    assert robot.c9.mock_calls == []


@pytest.mark.parametrize('speed', [0, 101, 'fast'])
def test_move_rel_xyz_refuses_bad_vel_before_moving(speed):
    robot = _bare_robot()
    with pytest.raises(ValueError, match='move_rel_xyz refused before moving: vel'):
        robot.move_rel_xyz(z_distance=10, vel=speed)
    assert robot.c9.mock_calls == []


def test_good_and_default_move_speeds_pass_the_check():
    from North_Safe import _check_optional_speed
    robot = _bare_robot()
    for speed in [None, 1, 10, 100, 7.5]:
        _check_optional_speed(robot, 'x', 'move_speed', speed)


# ---- liquid-handling parameters (after defaults, calibration and overrides are combined) ----

def test_parameter_limits_cover_the_defaults():
    from North_Safe import _check_parameter_limits
    from pipetting_data.pipetting_parameters import PipettingParameters, ReservoirParameters
    robot = _bare_robot()
    _check_parameter_limits(robot, 'Pipetting', PipettingParameters())
    _check_parameter_limits(robot, 'Reservoir dispense', ReservoirParameters())


@pytest.mark.parametrize('field, value, message', [
    ('retract_speed', 150, 'outside 1-100 %'), ('aspirate_speed', 0, 'outside 1-40'),
    ('dispense_speed', 41, 'outside 1-40'), ('aspirate_wait_time', -1, 'outside 0-60 s'),
    ('post_retract_wait_time', 61, 'outside 0-60 s'), ('overaspirate_vol', 1.5, 'outside -1-1 mL'),
    ('pre_asp_air_vol', -0.1, 'outside 0-1 mL'), ('blowout_vol', float('nan'), 'outside'),
    ('asp_disp_cycles', 2.5, 'not a whole number'), ('asp_disp_cycles', 11, 'outside 0-10')])
def test_bad_pipetting_parameters_are_refused(field, value, message):
    from North_Safe import _check_parameter_limits
    from pipetting_data.pipetting_parameters import PipettingParameters
    with pytest.raises(ValueError, match=f'Pipetting refused before moving: {field}.*{message}'):
        _check_parameter_limits(_bare_robot(), 'Pipetting', PipettingParameters(**{field: value}))


def test_none_uses_the_default_and_is_not_checked():
    from North_Safe import _check_parameter_limits
    from pipetting_data.pipetting_parameters import PipettingParameters
    _check_parameter_limits(_bare_robot(), 'Pipetting',
                            PipettingParameters(aspirate_speed=None, dispense_speed=None, retract_speed=None))


def test_resolvers_refuse_bad_overrides_before_moving():
    from pipetting_data.pipetting_parameters import ReservoirParameters
    robot = _bare_robot()
    with pytest.raises(ValueError, match='Reservoir dispense refused before moving: valve_switch_delay'):
        robot._get_optimized_reservoir_parameters(1.0, None, {'valve_switch_delay': 100})
    assert robot._get_optimized_reservoir_parameters(1.0, None, ReservoirParameters(dispense_speed=20)).dispense_speed == 20
    assert robot.c9.mock_calls == []

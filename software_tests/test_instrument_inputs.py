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

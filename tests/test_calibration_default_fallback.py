"""Unavailable calibration permits defaults without bypassing tip limits."""

import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

import North_Safe
from North_Safe import North_Robot
from pipetting_data.pipetting_parameters import PipettingParameters
from pipetting_data.pipetting_wizard import PipettingWizard


@pytest.fixture
def robot():
    instance = object.__new__(North_Robot)
    instance.logger = logging.getLogger(__name__)
    root = Path(__file__).resolve().parents[1]
    instance.PIPET_TIPS = yaml.safe_load((root / 'robot_state/pipet_tips.yaml').read_text())
    return instance


@pytest.mark.parametrize('volume', [.0015, .00118679])
@pytest.mark.parametrize('specified_tip', [None, 'small_tip'])
def test_tiny_water_warns_and_uses_defaults(robot, caplog, volume, specified_tip):
    with caplog.at_level(logging.WARNING):
        parameters, tip = robot.resolve_pipetting_setup(
            volume, 'water', specified_tip=specified_tip)
    assert tip == 'small_tip'
    assert parameters == PipettingParameters()
    assert 'outside the calibrated range' in caplog.text
    assert 'using default pipetting parameters with small_tip' in caplog.text
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


@pytest.mark.parametrize('case', ['no_file', 'wrong_tip', 'one_row'])
def test_wizard_returns_none_for_unavailable_tip_calibration(tmp_path, caplog, case):
    if case != 'no_file':
        pd.DataFrame([dict(volume_target=10, aspirate_speed=5,
                           tip_type='large_tip' if case == 'wrong_tip' else 'small_tip')]
                     ).to_csv(tmp_path / 'optimal_conditions_sample.csv', index=False)
    with caplog.at_level(logging.WARNING):
        result = PipettingWizard(str(tmp_path)).get_pipetting_parameters(
            'sample', .011, tip_type='small_tip')
    assert result is None
    assert caplog.records
    assert all(record.levelno == logging.WARNING for record in caplog.records)


def test_calibrated_large_tip_is_preferred_to_small_tip_defaults(robot, tmp_path, monkeypatch):
    pd.DataFrame([
        dict(volume_target=10, tip_type='small_tip', aspirate_speed=12),
        dict(volume_target=20, tip_type='small_tip', aspirate_speed=12),
        dict(volume_target=160, tip_type='large_tip', aspirate_speed=7),
        dict(volume_target=500, tip_type='large_tip', aspirate_speed=7),
    ]).to_csv(tmp_path / 'optimal_conditions_sample.csv', index=False)
    wizard = PipettingWizard(str(tmp_path))
    monkeypatch.setattr(North_Safe, 'PipettingWizard', lambda logger=None: wizard)
    parameters, tip = robot.resolve_pipetting_setup(.160, 'sample')
    assert tip == 'large_tip'
    assert parameters.aspirate_speed == 7


def test_default_fallback_retains_user_overrides(robot):
    parameters, tip = robot.resolve_pipetting_setup(.0015, 'water', {'aspirate_speed': 3})
    assert tip == 'small_tip'
    assert parameters.aspirate_speed == 3


@pytest.mark.parametrize('volume,overrides,tip', [
    (.0005, None, None),
    (1.0, None, None),
    (.0015, {'post_asp_air_vol': .2}, 'small_tip'),
    (.195, {'overaspirate_vol': .01}, 'small_tip'),
])
def test_fallback_does_not_bypass_minimum_or_capacity(robot, volume, overrides, tip):
    with pytest.raises(ValueError, match='No tip fits'):
        robot.resolve_pipetting_setup(volume, 'water', overrides, specified_tip=tip)


def test_unknown_tip_remains_an_error(robot):
    with pytest.raises(ValueError, match='Unknown pipet tip'):
        robot.resolve_pipetting_setup(.0015, 'water', specified_tip='missing')

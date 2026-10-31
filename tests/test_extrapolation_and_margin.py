"""Edge-row extrapolation and capacity-margin tip preference never add new errors."""

import logging

import pandas as pd
import pytest

import North_Safe
from North_Safe import North_Robot
from pipetting_data.pipetting_wizard import PipettingWizard

TIPS = {
    'small_tip': {'volume': .2, 'min_suggested_volume': .001, 'capacity_margin': .005},
    'large_tip': {'volume': 1.0, 'min_suggested_volume': .15, 'capacity_margin': .010},
}


@pytest.fixture
def calibration(tmp_path):
    # The 150 uL row is deliberately noisy; extrapolation above 180 must ignore it.
    pd.DataFrame([
        dict(volume_target=150, tip_type='small_tip', aspirate_speed=10, overaspirate_vol=-.005),
        dict(volume_target=180, tip_type='small_tip', aspirate_speed=12, overaspirate_vol=.0045),
        dict(volume_target=200, tip_type='large_tip', aspirate_speed=5, overaspirate_vol=.010),
        dict(volume_target=500, tip_type='large_tip', aspirate_speed=7, overaspirate_vol=.012),
    ]).to_csv(tmp_path / 'optimal_conditions_testliquid.csv', index=False)
    return tmp_path


@pytest.fixture
def robot(calibration, monkeypatch):
    monkeypatch.setattr(North_Safe, 'PipettingWizard',
                        lambda logger=None: PipettingWizard(str(calibration), logger=logger))
    instance = object.__new__(North_Robot)
    instance.logger = logging.getLogger(__name__)
    instance.PIPET_TIPS = {name: dict(config) for name, config in TIPS.items()}
    return instance


def test_extrapolation_scales_edge_row_and_ignores_neighbour(calibration):
    wizard = PipettingWizard(str(calibration))
    above = wizard.get_pipetting_parameters('testliquid', .1936, False, tip_type='small_tip')
    assert above['overaspirate_vol'] == pytest.approx(.0045 * 193.6 / 180)
    assert above['aspirate_speed'] == 12
    below = wizard.get_pipetting_parameters('testliquid', .190, False, tip_type='large_tip')
    assert below['overaspirate_vol'] == pytest.approx(.010 * 190 / 200)
    assert below['aspirate_speed'] == 5


def test_extrapolation_fraction(calibration):
    wizard = PipettingWizard(str(calibration))
    assert wizard.extrapolation_fraction('testliquid', .160, 'small_tip') == 0
    assert wizard.extrapolation_fraction('testliquid', .198, 'small_tip') == pytest.approx(.1)
    assert wizard.extrapolation_fraction('testliquid', .190, 'large_tip') == pytest.approx(.05)


def test_slightly_out_of_range_volume_keeps_small_tip_calibration(robot):
    parameters, tip = robot.resolve_pipetting_setup(.182, 'testliquid', compensate_overvolume=False)
    assert tip == 'small_tip'
    assert parameters.overaspirate_vol == pytest.approx(.0045 * 182 / 180)


def test_tight_small_tip_load_moves_to_large_tip_with_room(robot):
    # Small: 192 + 4.8 = 196.8 uL, inside its 5 uL margin. Large fits with room.
    _, tip = robot.resolve_pipetting_setup(.192, 'testliquid', compensate_overvolume=False)
    assert tip == 'large_tip'


def test_tight_fit_is_still_used_when_tip_is_pinned(robot):
    _, tip = robot.resolve_pipetting_setup(.1936, 'testliquid', specified_tip='small_tip',
                                           compensate_overvolume=False)
    assert tip == 'small_tip'


def test_zero_margin_restores_smallest_tip_that_fits(robot):
    for config in robot.PIPET_TIPS.values():
        config['capacity_margin'] = 0
    _, tip = robot.resolve_pipetting_setup(.192, 'testliquid', compensate_overvolume=False)
    assert tip == 'small_tip'


def test_extrapolation_is_reported_once_for_the_chosen_tip(robot, monkeypatch, caplog):
    from pipetting_data import pipetting_wizard
    monkeypatch.setattr(pipetting_wizard, '_REPORTED_EXTRAPOLATIONS', set())
    with caplog.at_level(logging.INFO):
        # 195 uL: the small tip is tried and rejected (tight load), the large tip is used.
        for _ in range(3):
            _, tip = robot.resolve_pipetting_setup(.195, 'testliquid', compensate_overvolume=False)
            robot._get_optimized_parameters(.195, 'testliquid')
    assert tip == 'large_tip'
    messages = [r for r in caplog.records if 'Extrapolating' in r.getMessage()]
    assert len(messages) == 1
    assert messages[0].levelno == logging.INFO and 'below' in messages[0].getMessage()


def test_far_extrapolation_is_still_a_warning(calibration, monkeypatch, caplog):
    from pipetting_data import pipetting_wizard
    monkeypatch.setattr(pipetting_wizard, '_REPORTED_EXTRAPOLATIONS', set())
    with caplog.at_level(logging.INFO):
        PipettingWizard(str(calibration)).get_pipetting_parameters('testliquid', .600, False, tip_type='large_tip')
    assert [r.levelno for r in caplog.records if 'Extrapolating' in r.getMessage()] == [logging.WARNING]

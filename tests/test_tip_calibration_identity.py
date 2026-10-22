"""Tip identity survives import and limits parameter lookup to one tip."""

import sys
import logging
from unittest.mock import Mock
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipetting_data'))

from import_calibration_run import convert
from pipetting_data.pipetting_wizard import PipettingWizard
from North_Safe import North_Robot


def test_wizard_extrapolates_only_with_confirmed_tip_rows(tmp_path):
    pd.DataFrame([
        dict(volume_target=180, tip_type='small_tip', aspirate_speed=14, overaspirate_vol=.002),
        dict(volume_target=195, tip_type='small_tip', aspirate_speed=13, overaspirate_vol=.003),
        dict(volume_target=200, tip_type='large_tip', aspirate_speed=5, overaspirate_vol=.010),
        dict(volume_target=500, tip_type='large_tip', aspirate_speed=7, overaspirate_vol=.020),
    ]).to_csv(tmp_path / 'optimal_conditions_testliquid.csv', index=False)
    wizard = PipettingWizard(str(tmp_path))
    large = wizard.get_pipetting_parameters('testliquid', .195,
                                            compensate_overvolume=False, tip_type='large_tip')
    small = wizard.get_pipetting_parameters('testliquid', .195,
                                            compensate_overvolume=False, tip_type='small_tip')
    assert large['aspirate_speed'] == 5
    assert large['overaspirate_vol'] == pytest.approx(.010 - 5 * .010 / 300)
    assert small['aspirate_speed'] == 13
    assert small['overaspirate_vol'] == .003


def test_converter_preserves_same_volume_on_two_tips(tmp_path):
    source = tmp_path / 'new.csv'
    production = tmp_path / 'optimal_conditions_test.csv'
    pd.DataFrame([dict(volume_target=195, overaspirate_vol=.002)]).to_csv(production, index=False)
    pd.DataFrame([dict(volume_target_ul=195, calibration_overaspirate_vol=.010,
                       tip_type='large_tip')]).to_csv(source, index=False)
    convert(source, production)
    rows = pd.read_csv(production)
    assert set(rows.tip_type) == {'unknown', 'large_tip'}
    assert rows.volume_target.tolist() == [195, 195]


def test_real_195_ul_resolves_tip_and_matching_parameters():
    import yaml

    robot = object.__new__(North_Robot)
    robot.logger = logging.getLogger(__name__)
    robot.PIPET_TIPS = yaml.safe_load((ROOT / 'robot_state/pipet_tips.yaml').read_text())
    wizard = PipettingWizard()
    for liquid, expected_tip in [('water', 'large_tip'), ('SDS', 'small_tip'),
                                 ('DMSO', 'large_tip')]:
        parameters, tip = robot.resolve_pipetting_setup(.195, liquid)
        assert tip == expected_tip
        matching = wizard.get_pipetting_parameters(liquid, .195, tip_type=tip)
        assert parameters.overaspirate_vol == pytest.approx(matching['overaspirate_vol'])
        assert .195 + max(0, parameters.overaspirate_vol) + max(0, parameters.post_asp_air_vol) < robot.PIPET_TIPS[tip]['volume']


def test_liquid_file_lookup_does_not_match_a_different_liquid_name():
    files = PipettingWizard().find_calibration_files('TFA')
    assert [path.name for path in files] == ['optimal_conditions_TFA.csv']


def test_embedded_correction_inserts_same_volume_for_other_tip(tmp_path, monkeypatch):
    from pipetting_data import embedded_calibration_validation as validation
    from pipetting_data.pipetting_parameters import PipettingParameters

    path = tmp_path / 'optimal_conditions_testliquid.csv'
    pd.DataFrame([
        dict(volume_target=195, tip_type='small_tip', aspirate_speed=14,
             overaspirate_vol=.003, volume_measured=194),
        dict(volume_target=200, tip_type='large_tip', aspirate_speed=5,
             overaspirate_vol=.010, volume_measured=200),
        dict(volume_target=500, tip_type='large_tip', aspirate_speed=7,
             overaspirate_vol=.020, volume_measured=500),
    ]).to_csv(path, index=False)
    monkeypatch.setattr(validation, '__file__', str(tmp_path / 'embedded_calibration_validation.py'))
    validation._update_calibration_csv(
        'testliquid', .195, .009, .196, 3, tip_type='large_tip',
        resolved_parameters=PipettingParameters(aspirate_speed=5))
    rows = pd.read_csv(path)
    small = rows[(rows.volume_target == 195) & (rows.tip_type == 'small_tip')].iloc[0]
    large = rows[(rows.volume_target == 195) & (rows.tip_type == 'large_tip')].iloc[0]
    assert small.overaspirate_vol == .003
    assert large.overaspirate_vol == .009
    assert large.volume_measured == 196
    assert large.aspirate_speed == 5


def test_calibration_update_refreshes_compensation_inputs_for_matching_tip(tmp_path, monkeypatch):
    from pipetting_data import embedded_calibration_validation as validation

    path = tmp_path / 'optimal_conditions_testliquid.csv'
    pd.DataFrame([
        dict(volume_target=100, tip_type='small_tip', overaspirate_vol=.004,
             volume_measured=95, measurement_count=2),
        dict(volume_target=100, tip_type='large_tip', overaspirate_vol=.012,
             volume_measured=99, measurement_count=2),
    ]).to_csv(path, index=False)
    monkeypatch.setattr(validation, '__file__', str(tmp_path / 'embedded_calibration_validation.py'))

    validation._update_calibration_csv('testliquid', .1, .006, .102, 3,
                                       tip_type='small_tip')

    rows = pd.read_csv(path)
    small = rows[rows.tip_type == 'small_tip'].iloc[0]
    large = rows[rows.tip_type == 'large_tip'].iloc[0]
    assert small.overaspirate_vol == pytest.approx(.006)
    assert small.volume_measured == pytest.approx(102)
    assert small.measurement_count == 3
    assert large.overaspirate_vol == pytest.approx(.012)
    assert large.volume_measured == pytest.approx(99)
    compensated = PipettingWizard(str(tmp_path)).get_pipetting_parameters(
        'testliquid', .1, compensate_overvolume=True, tip_type='small_tip')
    assert compensated['overaspirate_vol'] == pytest.approx(.004)


def test_calibration_update_adds_missing_measured_column(tmp_path, monkeypatch):
    from pipetting_data import embedded_calibration_validation as validation

    path = tmp_path / 'optimal_conditions_testliquid.csv'
    pd.DataFrame([dict(volume_target=100, tip_type='small_tip',
                       overaspirate_vol=.004)]).to_csv(path, index=False)
    monkeypatch.setattr(validation, '__file__', str(tmp_path / 'embedded_calibration_validation.py'))

    validation._update_calibration_csv('testliquid', .1, .006, .098,
                                       tip_type='small_tip')

    row = pd.read_csv(path).iloc[0]
    assert row.overaspirate_vol == pytest.approx(.006)
    assert row.volume_measured == pytest.approx(98)


def test_calibration_update_requires_fresh_measured_volume(tmp_path, monkeypatch):
    from pipetting_data import embedded_calibration_validation as validation

    path = tmp_path / 'optimal_conditions_testliquid.csv'
    pd.DataFrame([dict(volume_target=100, tip_type='small_tip',
                       overaspirate_vol=.004, volume_measured=95)]).to_csv(path, index=False)
    monkeypatch.setattr(validation, '__file__', str(tmp_path / 'embedded_calibration_validation.py'))

    with pytest.raises(ValueError, match='measured volume'):
        validation._update_calibration_csv('testliquid', .1, .006, float('nan'),
                                           tip_type='small_tip')
    assert pd.read_csv(path).iloc[0].overaspirate_vol == pytest.approx(.004)


def test_calibration_update_reports_write_failure_without_changing_file(tmp_path, monkeypatch):
    from pipetting_data import embedded_calibration_validation as validation

    path = tmp_path / 'optimal_conditions_testliquid.csv'
    pd.DataFrame([dict(volume_target=100, tip_type='small_tip',
                       overaspirate_vol=.004, volume_measured=95)]).to_csv(path, index=False)
    monkeypatch.setattr(validation, '__file__', str(tmp_path / 'embedded_calibration_validation.py'))
    monkeypatch.setattr(validation.os, 'replace', Mock(side_effect=OSError('write failed')))

    with pytest.raises(OSError, match='write failed'):
        validation._update_calibration_csv('testliquid', .1, .006, .098,
                                           tip_type='small_tip')
    row = pd.read_csv(path).iloc[0]
    assert row.overaspirate_vol == pytest.approx(.004)
    assert row.volume_measured == pytest.approx(95)


def test_serial_plate_dispense_reuses_resolved_tip_parameters(monkeypatch):
    import yaml

    robot = object.__new__(North_Robot)
    robot.logger = logging.getLogger(__name__)
    robot.PIPET_TIPS = yaml.safe_load((ROOT / 'robot_state/pipet_tips.yaml').read_text())
    robot.HELD_PIPET_TYPE = None
    robot.normalize_vial_index = Mock(return_value=0)
    robot.get_vial_info = Mock(return_value=10)
    robot.aspirate_from_vial = Mock()
    robot.dispense_into_wellplate = Mock()
    robot.remove_pipet = Mock()
    robot.return_vial_home = Mock()
    plate = pd.DataFrame({'DMSO': [.195]}, index=[0])
    robot._dispense_wellplate_serial(plate, None, 'DMSO', '96 WELL PLATE')
    aspirate = robot.aspirate_from_vial.call_args
    dispense = robot.dispense_into_wellplate.call_args
    assert aspirate.kwargs['specified_tip'] == 'large_tip'
    assert aspirate.kwargs['parameters'] is dispense.kwargs['parameters']

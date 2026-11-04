"""Tracked overaspirate: estimation, damped updates and the lookup overlay."""

import logging

import numpy as np
import pandas as pd
import pytest

from pipetting_data import overaspirate_corrections as corrections
from pipetting_data import overaspirate_estimation as estimation
from pipetting_data.pipetting_wizard import PipettingWizard

STATE = dict(estimate_ul=5.0, variance=1.0, noise=4.0, drift=1.0, sessions=10, source="tracked")


def test_consistent_session_moves_by_the_kalman_gain():
    updated = corrections.decide(STATE, [6.0, 6.2, 5.8, 6.0])
    assert updated["verdict"] == "consistent"
    assert updated["gain"] == pytest.approx(2 / 6)
    assert updated["estimate_ul"] == pytest.approx(5 + 2 / 6)
    assert updated["variance"] < 2


def test_surprise_is_detected_against_the_combined_spread():
    assert not corrections.is_surprising(STATE, [8.0, 8.0])
    assert corrections.is_surprising(STATE, [12.0, 12.0])


def test_confirmed_drift_takes_the_full_damped_step():
    updated = corrections.decide(STATE, [15, 15, 15, 15], [15, 15.1, 14.9, 15])
    assert updated["verdict"] == "drift_confirmed"
    assert updated["estimate_ul"] == pytest.approx(5 + (2 / 6) * 10)


def test_unconfirmed_surprise_is_limited():
    updated = corrections.decide(STATE, [15, 15, 15, 15], [5, 5, 5, 5])
    assert updated["verdict"] == "transient"
    limit = corrections.SURPRISE_SD * np.sqrt(2 + 4)
    assert updated["estimate_ul"] - 5 <= (2 / 6) * limit + 1e-9


def test_confirm_near_the_check_is_drift_even_if_batches_differ_slightly():
    # Live SDS 800 uL, 2026-10-31: check implied ~24.8, confirm (run at 24.8) implied ~25.8.
    state = dict(STATE, estimate_ul=16.0)
    updated = corrections.decide(state, [24.5, 25.0, 24.7, 25.1], [25.6, 25.8, 25.4, 26.0])
    assert updated["verdict"] == "drift_confirmed"


def test_failed_dispenses_are_not_calibration_history():
    history = pd.DataFrame(dict(
        liquid_type='SDS', tip_type='small_tip', target_volume_ml=.05, measured_volume_ml=[.0503, .0003],
        overaspirate_vol=.0055, session_id=['a', 'b'], timestamp=['20261001_100000', '20261002_100000']))
    assert estimation.usable_measurements(history, 'SDS', 'small_tip').session_id.tolist() == ['a']


def test_update_respects_tip_bounds():
    updated = corrections.decide(STATE, [50, 50, 50], [50, 50, 50], upper_ul=8.0)
    assert updated["estimate_ul"] == 8.0


def test_lookup_uses_tracked_value_and_skips_old_compensation(tmp_path, monkeypatch):
    pd.DataFrame([
        dict(volume_target=100, tip_type='small_tip', overaspirate_vol=.004, volume_measured=95),
        dict(volume_target=150, tip_type='small_tip', overaspirate_vol=.005, volume_measured=150),
    ]).to_csv(tmp_path / 'optimal_conditions_testliquid.csv', index=False)
    path = tmp_path / 'overaspirate_corrections.csv'
    monkeypatch.setattr(corrections, 'CORRECTIONS_FILE', path)
    corrections.save_state('testliquid', 'small_tip', 100, dict(STATE, estimate_ul=3.0, verdict='consistent'),
                           base_ul=9.0, session_id='s1')
    wizard = PipettingWizard(str(tmp_path))
    tracked = wizard.get_pipetting_parameters('testliquid', .1, True, tip_type='small_tip')
    assert tracked['overaspirate_vol'] == pytest.approx(.003)
    untracked = wizard.get_pipetting_parameters('testliquid', .15, True, tip_type='small_tip')
    assert untracked['overaspirate_vol'] == pytest.approx(.005)
    corrections.save_state('testliquid', 'small_tip', 100, dict(STATE, estimate_ul=3.5, verdict='consistent'),
                           base_ul=None, session_id='s2')
    stored = pd.read_csv(path)
    assert len(stored) == 1 and stored.base_overaspirate_ul.iloc[0] == 9.0


def test_unreadable_corrections_never_break_lookup(tmp_path):
    bad = tmp_path / 'bad.csv'
    bad.write_bytes(b'\x00\x01not,a\ncsv')
    df = pd.DataFrame([dict(volume_target=100, tip_type='small_tip', overaspirate_vol=.004)])
    result = corrections.apply_corrections(df, 'water', 'small_tip', path=bad, logger=logging.getLogger(__name__))
    assert result.overaspirate_vol.iloc[0] == .004


def test_estimator_matches_settings_with_rounded_speeds():
    history = pd.DataFrame(dict(
        liquid_type='water', tip_type='large_tip', target_volume_ml=.5, measured_volume_ml=[.502, .499],
        overaspirate_vol=.01, retract_speed=10.0, aspirate_speed=8.0, session_id=['a', 'b'],
        timestamp=['20261001_100000', '20261002_100000']))
    rows = estimation.matching_history(history, 500, estimation.row_settings(
        pd.Series(dict(retract_speed=10.26, aspirate_speed=8.0))))
    assert len(rows) == 2
    assert estimation.implied_overaspirate_ul(rows).round(6).tolist() == [8.0, 11.0]


def test_few_sessions_fall_back_to_median():
    sessions = pd.DataFrame(dict(oa_ul=[4.0, 6.0, 5.0], replicates=3, replicate_sd_ul=.5,
                                 when=pd.date_range('2026-10-01', periods=3)))
    result = estimation.estimate_row(sessions)
    assert result["estimate_ul"] == 5.0 and result["method"].startswith("median")


def test_recalibrated_base_row_wins_over_old_correction(tmp_path, monkeypatch):
    calibration = tmp_path / 'optimal_conditions_testliquid.csv'
    pd.DataFrame([dict(volume_target=100, tip_type='small_tip', overaspirate_vol=.004,
                       volume_measured=100)]).to_csv(calibration, index=False)
    path = tmp_path / 'overaspirate_corrections.csv'
    monkeypatch.setattr(corrections, 'CORRECTIONS_FILE', path)
    corrections.save_state('testliquid', 'small_tip', 100, dict(STATE, estimate_ul=3.0, verdict='consistent'),
                           base_ul=4.0, session_id='s1')
    wizard = PipettingWizard(str(tmp_path))
    assert wizard.get_pipetting_parameters('testliquid', .1, True, tip_type='small_tip')['overaspirate_vol'] == pytest.approx(.003)
    # A deliberate recalibration changes the base row: the old correction no longer applies.
    pd.DataFrame([dict(volume_target=100, tip_type='small_tip', overaspirate_vol=.006,
                       volume_measured=100)]).to_csv(calibration, index=False)
    wizard = PipettingWizard(str(tmp_path))
    assert wizard.get_pipetting_parameters('testliquid', .1, True, tip_type='small_tip')['overaspirate_vol'] == pytest.approx(.006)
    state = corrections.prior_state('testliquid', 'small_tip', 100, 6.0, {}, base_ul=6.0)
    assert state['source'] != 'tracked' and state['estimate_ul'] == 6.0

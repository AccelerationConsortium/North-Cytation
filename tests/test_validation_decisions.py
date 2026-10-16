import tempfile
from pathlib import Path
import pandas as pd

from pipetting_data.validation_decisions import annotate, append_csv, read_csv


def measurements(values):
    return pd.DataFrame([dict(target_volume_ml=.1, measured_volume_ml=value / 1000,
                              optimization_stage=stage)
                         for stage, group in enumerate(values, 1) for value in group])


def test_stage_two_can_win_and_improve_without_passing():
    frame, decision = annotate(measurements([[90, 90], [97, 97], [95, 95]]), True)
    assert decision['optimization_stage'] == 2
    assert not decision['within_tolerance']
    assert frame.loc[frame.accepted, 'optimization_stage'].tolist() == [2, 2]


def test_baseline_retained_when_later_stages_are_worse_and_ties_choose_earlier():
    _, decision = annotate(measurements([[90], [80], [70]]), True)
    assert decision['optimization_stage'] == 1
    assert decision['decision_reason'] == 'baseline_retained'
    _, decision = annotate(measurements([[90], [99], [101]]), True)
    assert decision['optimization_stage'] == 2


def test_passed_baseline_and_failure_are_distinct():
    frame, decision = annotate(measurements([[100, 101]]), False)
    assert frame.accepted.all() and decision['within_tolerance']
    frame, decision = annotate(measurements([[90], [95]]), True, failed=True)
    assert not frame.accepted.any()
    assert decision['decision_reason'] == 'optimization_failed'


def test_append_aligns_new_fields_and_old_headers():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'data.csv'
        path.write_text('session,volume\na,0.1\n')
        append_csv(pd.DataFrame([dict(volume=.2, accepted=True, session='b')]), path)
        frame = read_csv(path)
        assert frame.iloc[0].session == 'a' and frame.iloc[0].accepted == ''
        assert frame.iloc[1].session == 'b' and frame.iloc[1].accepted == 'True'


def test_embedded_validation_saves_selected_stage_not_highest(monkeypatch):
    from unittest.mock import MagicMock
    from pipetting_data import embedded_calibration_validation as validation
    from pipetting_data.pipetting_parameters import PipettingParameters
    robot = MagicMock()
    robot.simulate = True
    robot.nr_robot._get_optimized_parameters.return_value = PipettingParameters(overaspirate_vol=.004)
    robot.nr_robot.dispense_from_vial_into_vial.side_effect = [.09, .097, .095]
    saved_measurements, saved_sessions = [], []
    monkeypatch.setattr(validation, '_append_to_master_measurements', lambda rows, kind: saved_measurements.extend(rows))
    monkeypatch.setattr(validation, '_append_to_master_sessions', lambda row, kind: saved_sessions.append(row))
    monkeypatch.setattr(validation, '_create_validation_plots', lambda **kwargs: None)
    monkeypatch.setenv('ROBOT_SIMULATION_MODE', 'TRUE')
    validation.validate_pipetting_accuracy(robot, 'source', 'dest', 'water', [.1], replicates=1,
                                          output_folder=None, adaptive_correction=True)
    assert [row['optimization_stage'] for row in saved_measurements if row['accepted']] == [2]
    assert saved_sessions[0]['optimization_stage'] == 2
    assert saved_sessions[0]['measured_mean_ml'] == .097
    assert not saved_sessions[0]['within_tolerance']


def test_simulation_never_writes_master_csvs(monkeypatch, tmp_path):
    from pipetting_data import embedded_calibration_validation as validation
    monkeypatch.setenv('ROBOT_SIMULATION_MODE', 'TRUE')
    monkeypatch.setattr(validation, 'MASTER_PIPETTING_MEASUREMENTS_DB', str(tmp_path / 'm.csv'))
    monkeypatch.setattr(validation, 'MASTER_PIPETTING_SESSIONS_DB', str(tmp_path / 's.csv'))
    validation._append_to_master_measurements([dict(a=1)], 'pipetting')
    validation._append_to_master_sessions(dict(a=1), 'pipetting')
    assert list(tmp_path.iterdir()) == []


def test_deviant_replicate_is_saved_and_decision_matches_saved_rows(monkeypatch):
    import sys
    from unittest.mock import MagicMock
    from pipetting_data import embedded_calibration_validation as validation
    from pipetting_data.pipetting_parameters import PipettingParameters
    robot = MagicMock()
    robot.simulate = False
    robot.nr_robot._get_optimized_parameters.return_value = PipettingParameters(overaspirate_vol=.004)
    # Stage 1 x2, stage 2 first replicate far off target (stops that stage), stage 3 x2.
    robot.nr_robot.dispense_from_vial_into_vial.side_effect = [.092, .092, .15, .10, .10]
    saved = []
    monkeypatch.setattr(validation, '_append_to_master_measurements', lambda rows, kind: saved.extend(rows))
    monkeypatch.setattr(validation, '_append_to_master_sessions', lambda row, kind: None)
    monkeypatch.setattr(validation, '_create_validation_plots', lambda **kwargs: None)
    monkeypatch.setattr(validation, '_log_parameter_correction', lambda *args, **kwargs: None)
    monkeypatch.setattr(validation, '_update_calibration_csv', lambda *args, **kwargs: None)
    monkeypatch.setattr(validation, '_get_latest_environmental_data', lambda: None)
    monkeypatch.setitem(sys.modules, 'slack_agent', MagicMock())
    monkeypatch.delenv('ROBOT_SIMULATION_MODE', raising=False)
    validation.validate_pipetting_accuracy(robot, 'source', 'dest', 'water', [.1], replicates=2,
                                          output_folder=None, adaptive_correction=True)
    stage_two = [row for row in saved if row['optimization_stage'] == 2]
    assert len(stage_two) == 1 and stage_two[0]['measured_volume_ml'] > .14
    assert sorted({row['optimization_stage'] for row in saved}) == [1, 2, 3]
    assert all(row['measured_at'] and row['mass_difference_g'] for row in saved)

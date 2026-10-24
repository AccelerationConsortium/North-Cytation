import tempfile
from pathlib import Path
import pandas as pd

from pipetting_data.validation_decisions import annotate, append_csv, read_csv
import logging
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest


class RecordingRobot:
    """Hardware-free controller with explicit validation-facing methods."""

    def __init__(self, masses):
        self.masses = iter(masses)
        self.transfers = []
        self.mass_save_requests = []

    def get_vial_info(self, vial, field):
        return "clamp"

    def move_vial_to_location(self, vial, location, index):
        pass

    def remove_pipet(self):
        pass

    def return_vial_home(self, vial):
        pass

    def get_vial_in_location(self, location, index):
        return None

    def _get_optimized_parameters(self, volume, liquid=None, user_parameters=None,
                                  compensate_overvolume=True):
        from pipetting_data.pipetting_parameters import PipettingParameters
        if user_parameters is not None:
            return replace(user_parameters)
        return PipettingParameters(
            aspirate_speed=9, dispense_speed=13, retract_speed=8, blowout_speed=7,
            aspirate_wait_time=.9, dispense_wait_time=1.5, post_retract_wait_time=.2,
            overaspirate_vol=.007 if volume == .1 else .009,
            pre_asp_air_vol=.02, post_asp_air_vol=.003, blowout_vol=.045,
            asp_disp_cycles=2,
        )

    def resolve_pipetting_setup(self, volume, liquid=None, user_parameters=None,
                                specified_tip=None, **kwargs):
        return (self._get_optimized_parameters(volume, liquid, user_parameters),
                specified_tip or ('small_tip' if volume < .2 else 'large_tip'))

    def dispense_from_vial_into_vial(self, source_vial_name, dest_vial_name, volume,
                                    parameters, liquid, remove_tip, use_safe_location,
                                    return_vial_home, compensate_overvolume,
                                     measure_weight,
                                    continuous_mass_monitoring, save_mass_data,
                                    specified_tip=None, record_measurement=True):
        assert record_measurement is False, "Calibration must own its records"
        resolved = self._get_optimized_parameters(volume, liquid, parameters)
        self.transfers.append((volume, asdict(resolved)))
        self.mass_save_requests.append(save_mass_data)
        return next(self.masses), dict(
            pre_stable_count=5, pre_total_count=5, post_stable_count=5,
            post_total_count=5, pre_baseline_std=0., post_baseline_std=0.,
        )


@pytest.fixture
def recording_validation(monkeypatch, tmp_path):
    import sys
    from pipetting_data import embedded_calibration_validation as validation
    monkeypatch.setattr(validation, 'MASTER_PIPETTING_MEASUREMENTS_DB', str(tmp_path / 'measurements.csv'))
    monkeypatch.setattr(validation, 'MASTER_PIPETTING_SESSIONS_DB', str(tmp_path / 'sessions.csv'))
    monkeypatch.setattr(validation, 'PARAMETER_CORRECTIONS_LOG', str(tmp_path / 'corrections.csv'))
    monkeypatch.setattr(validation, '_create_validation_plots', lambda **kwargs: None)
    monkeypatch.setattr(validation, '_update_calibration_csv', lambda *args, **kwargs: None)
    monkeypatch.setattr(validation, '_get_latest_environmental_data',
                        lambda: dict(temp_c=23., humidity_pct=45., pressure_pa=101000.))
    monkeypatch.setitem(sys.modules, 'slack_agent', SimpleNamespace(send_slack_message=lambda message: None))
    return validation


def test_adaptive_correction_rejects_compensated_baseline(recording_validation):
    robot = RecordingRobot([])
    lash = SimpleNamespace(nr_robot=robot, simulate=True, logger=logging.getLogger(__name__))
    with pytest.raises(ValueError, match="uncompensated calibration parameters"):
        recording_validation.validate_pipetting_accuracy(
            lash, 'source', 'dest', 'water', [.1], adaptive_correction=True,
            compensate_overvolume=True,
        )
    assert robot.transfers == []


@pytest.mark.parametrize('adaptive', [False, True])
def test_simulated_validation_with_no_balance_completes_without_saving(
        recording_validation, tmp_path, monkeypatch, adaptive):
    validation = recording_validation

    class NoBalanceRobot(RecordingRobot):
        def dispense_from_vial_into_vial(self, *args, **kwargs):
            mass, info = super().dispense_from_vial_into_vial(*args, **kwargs)
            info.update(pre_stable_count=0, pre_total_count=0,
                        post_stable_count=0, post_total_count=0,
                        pre_baseline_std=float('nan'), post_baseline_std=float('nan'))
            return mass, info

    robot = NoBalanceRobot([None] * 12)
    lash = SimpleNamespace(nr_robot=robot, simulate=True, logger=logging.getLogger(__name__))
    generated = iter([.092, .098, .100])
    monkeypatch.setattr(validation, '_generate_simulated_mass', lambda **kwargs: next(generated))
    result = validation.validate_pipetting_accuracy(
        lash, 'source', 'dest', 'water', [.1], replicates=1,
        output_folder=str(tmp_path / 'raw'), adaptive_correction=adaptive)
    assert result['success']
    assert len(robot.transfers) == (3 if adaptive else 1)
    assert not any(robot.mass_save_requests)
    assert list(tmp_path.iterdir()) == []


def test_real_raw_and_masters_match_actual_stage_parameters(recording_validation, tmp_path):
    validation = recording_validation
    for name in ('measurements.csv', 'sessions.csv'):
        (tmp_path / name).write_text('timestamp,session_id,overaspirate_vol\n', encoding='utf-8')
    robot = RecordingRobot([.092, .098, .100, .2])
    lash = SimpleNamespace(nr_robot=robot, simulate=False, logger=logging.getLogger(__name__))
    result = validation.validate_pipetting_accuracy(
        lash, 'source', 'dest', 'water', [.1, .2], replicates=1,
        output_folder=str(tmp_path / 'raw'), adaptive_correction=True,
    )
    raw = pd.read_csv(result['files']['raw_data'])
    measurements = pd.read_csv(tmp_path / 'measurements.csv')
    sessions = pd.read_csv(tmp_path / 'sessions.csv')
    statistics = pd.read_csv(result['files']['volume_stats'])
    assert len(raw) == len(measurements) == len(robot.transfers) == 4
    for index, (target, actual) in enumerate(robot.transfers):
        assert raw.iloc[index].target_volume_ml == measurements.iloc[index].target_volume_ml == target
        assert raw.iloc[index].measured_volume_ml == measurements.iloc[index].measured_volume_ml
        for field, value in actual.items():
            assert raw.iloc[index][field] == pytest.approx(value), field
            assert measurements.iloc[index][field] == pytest.approx(value), field
    assert raw.optimization_stage.tolist() == [1, 2, 3, 1]
    assert measurements.optimization_stage.tolist() == [1, 2, 3, 1]
    for _, session in sessions.iterrows():
        selected = raw[(raw.target_volume_ml == session.target_volume_ml) & raw.accepted]
        assert session.measured_mean_ml == pytest.approx(selected.measured_volume_ml.mean())
        assert session.num_replicates == len(selected)
        for field in robot.transfers[0][1]:
            assert session[field] == pytest.approx(selected.iloc[0][field]), field
            stat = statistics[statistics.target_volume_ml == session.target_volume_ml].iloc[0]
            assert stat[field] == pytest.approx(session[field]), field
        assert session.temp_c == 23.
    corrections = pd.read_csv(tmp_path / 'corrections.csv')
    assert corrections.iloc[0].old_overaspirate_ml == .007
    assert corrections.iloc[0].new_overaspirate_ml == pytest.approx(raw.iloc[2].overaspirate_vol)
    assert raw.session_id.nunique() == 1
    assert set(raw.session_id) == set(measurements.session_id) == set(sessions.session_id)
    assert not raw.simulate.any() and not measurements.simulate.any() and not sessions.simulate.any()
    assert raw.stability_info.notna().all()
    assert all(robot.mass_save_requests)


def test_simulation_default_saving_creates_no_files(recording_validation, tmp_path, monkeypatch):
    validation = recording_validation
    robot = RecordingRobot([0., 0., 0.])
    generated = iter([.092, .098, .100])
    monkeypatch.setattr(validation, '_generate_simulated_mass', lambda **kwargs: next(generated))
    lash = SimpleNamespace(nr_robot=robot, simulate=True, logger=logging.getLogger(__name__))
    monkeypatch.delenv('ROBOT_SIMULATION_MODE', raising=False)
    validation.validate_pipetting_accuracy(
        lash, 'source', 'dest', 'water', [.1], replicates=1,
        output_folder=str(tmp_path / 'raw'), adaptive_correction=True,
    )
    assert len(robot.transfers) == 3
    assert not any(robot.mass_save_requests)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("blocked_file", ["measurements.csv", "sessions.csv"])
def test_master_save_failure_is_not_success(recording_validation, tmp_path, monkeypatch, blocked_file):
    validation = recording_validation
    robot = RecordingRobot([.1])
    lash = SimpleNamespace(nr_robot=robot, simulate=False, logger=logging.getLogger(__name__))
    append = validation.append_csv
    def fail(frame, path):
        if Path(path).name == blocked_file:
            raise PermissionError("blocked master")
        append(frame, path)
    monkeypatch.setattr(validation, 'append_csv', fail)
    with pytest.raises(PermissionError, match="blocked master"):
        validation.validate_pipetting_accuracy(
            lash, 'source', 'dest', 'water', [.1], replicates=1,
            output_folder=str(tmp_path / 'raw'),
        )
    assert len(list((tmp_path / 'raw').rglob('raw_measurements.csv'))) == 1


@pytest.mark.parametrize("masses, selected_stage", [
    ([.100], 1), ([.092, .100, .098], 2), ([.092, .098, .100], 3),
])
def test_selected_stage_parameters_are_written_to_real_csvs(
        recording_validation, tmp_path, masses, selected_stage):
    validation = recording_validation
    robot = RecordingRobot(masses)
    lash = SimpleNamespace(nr_robot=robot, simulate=False, logger=logging.getLogger(__name__))
    validation.validate_pipetting_accuracy(
        lash, 'source', 'dest', 'water', [.1], replicates=1,
        output_folder=str(tmp_path / 'raw'), adaptive_correction=True,
    )
    session = pd.read_csv(tmp_path / 'sessions.csv').iloc[0]
    assert session.optimization_stage == selected_stage
    assert session.overaspirate_vol == pytest.approx(robot.transfers[selected_stage - 1][1]['overaspirate_vol'])


def test_measurements_survive_correction_save_failure(recording_validation, tmp_path, monkeypatch):
    validation = recording_validation
    robot = RecordingRobot([.092, .098, .100])
    lash = SimpleNamespace(nr_robot=robot, simulate=False, logger=logging.getLogger(__name__))
    def fail(*args):
        raise PermissionError("blocked correction")
    monkeypatch.setattr(validation, 'append_csv', fail)
    with pytest.raises(OSError, match="Could not save parameter correction"):
        validation.validate_pipetting_accuracy(
            lash, 'source', 'dest', 'water', [.1], replicates=1,
            output_folder=str(tmp_path / 'raw'), adaptive_correction=True,
        )
    raw_path, = (tmp_path / 'raw').rglob('raw_measurements.csv')
    raw = pd.read_csv(raw_path)
    assert raw.optimization_stage.tolist() == [1, 2, 3]
    assert raw.overaspirate_vol.tolist() == pytest.approx([p['overaspirate_vol'] for _, p in robot.transfers])


def test_raw_save_failure_stops_before_next_transfer(recording_validation, tmp_path, monkeypatch):
    validation = recording_validation
    robot = RecordingRobot([.1, .1])
    lash = SimpleNamespace(nr_robot=robot, simulate=False, logger=logging.getLogger(__name__))
    def fail(*args):
        raise PermissionError("blocked raw")
    monkeypatch.setattr(validation, 'write_csv', fail)
    with pytest.raises(OSError, match="Could not checkpoint raw"):
        validation.validate_pipetting_accuracy(
            lash, 'source', 'dest', 'water', [.1], replicates=2,
            output_folder=str(tmp_path / 'raw'),
        )
    assert len(robot.transfers) == 1
    assert not (tmp_path / 'measurements.csv').exists()


def test_real_writers_do_not_depend_on_process_global_simulation_flag(
        recording_validation, tmp_path, monkeypatch):
    validation = recording_validation
    monkeypatch.setenv('ROBOT_SIMULATION_MODE', 'TRUE')
    validation._append_to_master_measurements([dict(session_id='real')], simulate=False)
    validation._append_to_master_sessions(dict(session_id='real'), simulate=False)
    assert pd.read_csv(tmp_path / 'measurements.csv').iloc[0].session_id == 'real'
    assert pd.read_csv(tmp_path / 'sessions.csv').iloc[0].session_id == 'real'


@pytest.mark.parametrize("simulate", [False, True])
def test_reservoir_records_resolved_parameters_and_isolates_simulation(
        recording_validation, tmp_path, monkeypatch, simulate):
    from pipetting_data.pipetting_parameters import ReservoirParameters
    validation = recording_validation
    class ReservoirRobot(RecordingRobot):
        def _get_optimized_reservoir_parameters(self, volume, liquid=None, reservoir_params=None):
            return ReservoirParameters(aspirate_speed=8, dispense_speed=9,
                                       overaspirate_vol=.012 if volume == .1 else .015)

        def dispense_into_vial_from_reservoir(
                self, reservoir_index, vial_index, volume, reservoir_params, liquid,
                measure_weight, continuous_mass_monitoring, save_mass_data, record_measurement=True):
            assert record_measurement is False, "Calibration must own its records"
            self.transfers.append((volume, asdict(reservoir_params)))
            return next(self.masses)
    monkeypatch.setattr(validation, 'MASTER_RESERVOIR_MEASUREMENTS_DB', str(tmp_path / 'rm.csv'))
    monkeypatch.setattr(validation, 'MASTER_RESERVOIR_SESSIONS_DB', str(tmp_path / 'rs.csv'))
    robot = ReservoirRobot([.1, .2])
    lash = SimpleNamespace(nr_robot=robot, simulate=simulate, logger=logging.getLogger(__name__))
    result = validation.validate_reservoir_accuracy(
        lash, 1, 'dest', 'water', [.1, .2], replicates=1,
        output_folder=str(tmp_path / 'raw'),
    )
    assert len(robot.transfers) == 2
    if simulate:
        assert list(tmp_path.iterdir()) == []
    else:
        raw = pd.read_csv(result['files']['raw_data'])
        measurements = pd.read_csv(tmp_path / 'rm.csv')
        sessions = pd.read_csv(tmp_path / 'rs.csv')
        for index, (_, parameters) in enumerate(robot.transfers):
            for name, value in parameters.items():
                assert raw.iloc[index][name] == value
                assert measurements.iloc[index][name] == value
                assert sessions.iloc[index][name] == value


def test_recording_controller_methods_match_real_controller_api():
    import ast
    import inspect
    source = Path(__file__).resolve().parents[1] / 'North_Safe.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    controller = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'North_Robot')
    methods = {node.name: node for node in controller.body if isinstance(node, ast.FunctionDef)}
    for name, function in vars(RecordingRobot).items():
        if name.startswith('__'):
            continue
        assert name in methods
        actual = methods[name].args
        actual_names = {arg.arg for arg in actual.args + actual.kwonlyargs}
        if name in ('_get_optimized_parameters', 'dispense_from_vial_into_vial'):
            assert set(inspect.signature(function).parameters) <= actual_names
        else:
            assert len(inspect.signature(function).parameters) <= len(actual.args)


def test_completed_raw_measurement_survives_interruption(recording_validation, tmp_path):
    validation = recording_validation
    robot = RecordingRobot([.1])
    lash = SimpleNamespace(nr_robot=robot, simulate=False, logger=logging.getLogger(__name__))
    with pytest.raises(StopIteration):
        validation.validate_pipetting_accuracy(
            lash, 'source', 'dest', 'water', [.1], replicates=2,
            output_folder=str(tmp_path / 'raw'),
        )
    raw_path, = (tmp_path / 'raw').rglob('raw_measurements.csv')
    raw = pd.read_csv(raw_path)
    assert len(raw) == 1
    assert raw.iloc[0].measured_volume_ml == .1
    assert raw.iloc[0].overaspirate_vol == .007


def test_correction_append_preserves_historical_environment_columns(recording_validation, tmp_path):
    import csv
    validation = recording_validation
    path = tmp_path / 'corrections.csv'
    path.write_text(
        'timestamp,session_id,validation_type,liquid_type,target_volume_ml,target_volume_ul,'
        'old_overaspirate_ml,new_overaspirate_ml,correction_ml,correction_ul\n'
        '20260415_180246,old,iterative_optimization_stage3,water,0.1,100,'
        '0.007,0.008,0.001,1,25,53,99526\n',
        encoding='utf-8',
    )
    validation._log_parameter_correction(
        '20261016_120000', 'water', .1, .007, .009,
        'iterative_optimization_stage2', 'new', SimpleNamespace(simulate=False),
        within_tolerance=True,
    )
    frame = pd.read_csv(path)
    assert frame.iloc[0].session_id == 'old'
    assert frame.iloc[0].temp_c == 25
    assert frame.iloc[0].old_overaspirate_ml == .007
    assert frame.iloc[1].old_overaspirate_ml == .007
    assert frame.iloc[1].new_overaspirate_ml == .009
    assert frame.iloc[1].humidity_pct == 45
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.reader(handle))
    assert all(len(row) == len(rows[0]) for row in rows)


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
    robot.nr_robot.resolve_pipetting_setup.return_value = (
        PipettingParameters(overaspirate_vol=.004), 'small_tip')
    robot.nr_robot.dispense_from_vial_into_vial.side_effect = [.09, .097, .095]
    saved_measurements, saved_sessions = [], []
    monkeypatch.setattr(validation, '_append_to_master_measurements', lambda rows, kind, **kwargs: saved_measurements.extend(rows))
    monkeypatch.setattr(validation, '_append_to_master_sessions', lambda row, kind, **kwargs: saved_sessions.append(row))
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
    validation._append_to_master_measurements([dict(a=1)], 'pipetting', simulate=True)
    validation._append_to_master_sessions(dict(a=1), 'pipetting', simulate=True)
    assert list(tmp_path.iterdir()) == []


def test_deviant_replicate_is_saved_and_decision_matches_saved_rows(monkeypatch):
    import sys
    from unittest.mock import MagicMock
    from pipetting_data import embedded_calibration_validation as validation
    from pipetting_data.pipetting_parameters import PipettingParameters
    robot = MagicMock()
    robot.simulate = False
    robot.nr_robot._get_optimized_parameters.return_value = PipettingParameters(overaspirate_vol=.004)
    robot.nr_robot.resolve_pipetting_setup.return_value = (
        PipettingParameters(overaspirate_vol=.004), 'small_tip')
    # Stage 1 x2, stage 2 first replicate far off target (stops that stage), stage 3 x2.
    robot.nr_robot.dispense_from_vial_into_vial.side_effect = [.092, .092, .15, .10, .10]
    saved = []
    monkeypatch.setattr(validation, '_append_to_master_measurements', lambda rows, kind, **kwargs: saved.extend(rows))
    monkeypatch.setattr(validation, '_append_to_master_sessions', lambda row, kind, **kwargs: None)
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

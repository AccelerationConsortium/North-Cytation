import tempfile
from pathlib import Path
import pandas as pd

from pipetting_data.validation_decisions import append_csv, read_csv
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
    monkeypatch.setattr(validation, '_get_latest_environmental_data',
                        lambda: dict(temp_c=23., humidity_pct=45., pressure_pa=101000.))
    # Never touch the real tracked-correction file from tests.
    from pipetting_data import embedded_liquid_validations, overaspirate_corrections
    monkeypatch.setattr(overaspirate_corrections, 'CORRECTIONS_FILE', tmp_path / 'tracked.csv')
    monkeypatch.setattr(embedded_liquid_validations, 'calibrated_volumes_ul', lambda liquid, tip: [100.0])
    monkeypatch.setitem(sys.modules, 'slack_agent', SimpleNamespace(send_slack_message=lambda message: None))
    return validation


def test_simulated_validation_with_no_balance_completes_without_saving(
        recording_validation, tmp_path, monkeypatch):
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
        output_folder=str(tmp_path / 'raw'))
    assert result['success']
    assert len(robot.transfers) == 1
    assert not any(robot.mass_save_requests)
    assert list(tmp_path.iterdir()) == []


def test_real_raw_and_masters_match_actual_parameters(recording_validation, tmp_path):
    validation = recording_validation
    for name in ('measurements.csv', 'sessions.csv'):
        (tmp_path / name).write_text('timestamp,session_id,overaspirate_vol\n', encoding='utf-8')
    robot = RecordingRobot([.1, .2])
    lash = SimpleNamespace(nr_robot=robot, simulate=False, logger=logging.getLogger(__name__))
    result = validation.validate_pipetting_accuracy(
        lash, 'source', 'dest', 'water', [.1, .2], replicates=1,
        output_folder=str(tmp_path / 'raw'),
    )
    raw = pd.read_csv(result['files']['raw_data'])
    measurements = pd.read_csv(tmp_path / 'measurements.csv')
    sessions = pd.read_csv(tmp_path / 'sessions.csv')
    statistics = pd.read_csv(result['files']['volume_stats'])
    assert len(raw) == len(measurements) == len(robot.transfers) == 2
    for index, (target, actual) in enumerate(robot.transfers):
        assert raw.iloc[index].target_volume_ml == measurements.iloc[index].target_volume_ml == target
        assert raw.iloc[index].measured_volume_ml == measurements.iloc[index].measured_volume_ml
        for field, value in actual.items():
            assert raw.iloc[index][field] == pytest.approx(value), field
            assert measurements.iloc[index][field] == pytest.approx(value), field
    assert raw.optimization_stage.tolist() == [1, 1]
    assert measurements.optimization_stage.tolist() == [1, 1]
    for _, session in sessions.iterrows():
        selected = raw[(raw.target_volume_ml == session.target_volume_ml) & raw.accepted]
        assert session.measured_mean_ml == pytest.approx(selected.measured_volume_ml.mean())
        assert session.num_replicates == len(selected)
        for field in robot.transfers[0][1]:
            assert session[field] == pytest.approx(selected.iloc[0][field]), field
            stat = statistics[statistics.target_volume_ml == session.target_volume_ml].iloc[0]
            assert stat[field] == pytest.approx(session[field]), field
        assert session.temp_c == 23.
    assert raw.session_id.nunique() == 1
    assert set(raw.session_id) == set(measurements.session_id) == set(sessions.session_id)
    assert not raw.simulate.any() and not measurements.simulate.any() and not sessions.simulate.any()
    assert raw.stability_info.notna().all()
    assert all(robot.mass_save_requests)


def test_simulation_default_saving_creates_no_files(recording_validation, tmp_path, monkeypatch):
    validation = recording_validation
    robot = RecordingRobot([0.])
    generated = iter([.1])
    monkeypatch.setattr(validation, '_generate_simulated_mass', lambda **kwargs: next(generated))
    lash = SimpleNamespace(nr_robot=robot, simulate=True, logger=logging.getLogger(__name__))
    monkeypatch.delenv('ROBOT_SIMULATION_MODE', raising=False)
    validation.validate_pipetting_accuracy(
        lash, 'source', 'dest', 'water', [.1], replicates=1,
        output_folder=str(tmp_path / 'raw'),
    )
    assert len(robot.transfers) == 1
    assert not any(robot.mass_save_requests)
    assert list(tmp_path.iterdir()) == []


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


def test_append_aligns_new_fields_and_old_headers():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'data.csv'
        path.write_text('session,volume\na,0.1\n')
        append_csv(pd.DataFrame([dict(volume=.2, accepted=True, session='b')]), path)
        frame = read_csv(path)
        assert frame.iloc[0].session == 'a' and frame.iloc[0].accepted == ''
        assert frame.iloc[1].session == 'b' and frame.iloc[1].accepted == 'True'


def test_simulation_never_writes_master_csvs(monkeypatch, tmp_path):
    from pipetting_data import embedded_calibration_validation as validation
    monkeypatch.setenv('ROBOT_SIMULATION_MODE', 'TRUE')
    monkeypatch.setattr(validation, 'MASTER_PIPETTING_MEASUREMENTS_DB', str(tmp_path / 'm.csv'))
    monkeypatch.setattr(validation, 'MASTER_PIPETTING_SESSIONS_DB', str(tmp_path / 's.csv'))
    validation._append_to_master_measurements([dict(a=1)], 'pipetting', simulate=True)
    validation._append_to_master_sessions(dict(a=1), 'pipetting', simulate=True)
    assert list(tmp_path.iterdir()) == []


class TipFullRobot(RecordingRobot):
    """Raises the capacity error for chosen volumes, after a given number of resolutions."""

    def __init__(self, masses, full_volumes, fail_after=0):
        super().__init__(masses)
        self.full_volumes = set(full_volumes)
        self.fail_after = fail_after
        self.resolutions = 0

    def resolve_pipetting_setup(self, volume, *args, **kwargs):
        self.resolutions += 1
        if volume in self.full_volumes and self.resolutions > self.fail_after:
            raise ValueError(f"No tip fits water at {volume * 1000:g} uL")
        return super().resolve_pipetting_setup(volume, *args, **kwargs)


def test_volume_that_no_longer_fits_is_skipped_before_hardware(recording_validation, tmp_path):
    robot = TipFullRobot([.1], full_volumes=[.2])
    lash = SimpleNamespace(nr_robot=robot, simulate=True, logger=logging.getLogger(__name__))
    result = recording_validation.validate_pipetting_accuracy(
        lash, 'source', 'dest', 'water', [.1, .2], replicates=1,
        output_folder=str(tmp_path / 'raw'), tip_type='small_tip')
    assert result['success']
    assert [volume for volume, _ in robot.transfers] == [.1]


def test_volume_that_stops_fitting_mid_session_is_skipped(recording_validation, tmp_path):
    # Both volumes resolve up front (2 calls); .2 fails when re-resolved after .1 ran.
    robot = TipFullRobot([.1], full_volumes=[.2], fail_after=2)
    lash = SimpleNamespace(nr_robot=robot, simulate=True, logger=logging.getLogger(__name__))
    result = recording_validation.validate_pipetting_accuracy(
        lash, 'source', 'dest', 'water', [.1, .2], replicates=1,
        output_folder=str(tmp_path / 'raw'), tip_type='small_tip')
    assert result['success']
    assert [stat['target_volume_ml'] for stat in result['volume_statistics']] == [.1]


def test_all_volumes_skipped_returns_skipped_result(recording_validation, tmp_path):
    robot = TipFullRobot([], full_volumes=[.2])
    lash = SimpleNamespace(nr_robot=robot, simulate=True, logger=logging.getLogger(__name__))
    result = recording_validation.validate_pipetting_accuracy(
        lash, 'source', 'dest', 'water', [.2], replicates=1,
        output_folder=str(tmp_path / 'raw'), tip_type='small_tip')
    assert result['skipped'] and not result['success']
    assert robot.transfers == []


def test_masters_record_tip_type(recording_validation, tmp_path):
    robot = RecordingRobot([.1, .2])
    lash = SimpleNamespace(nr_robot=robot, simulate=False, logger=logging.getLogger(__name__))
    recording_validation.validate_pipetting_accuracy(
        lash, 'source', 'dest', 'water', [.1, .2], replicates=1,
        output_folder=str(tmp_path / 'raw'))
    measurements = pd.read_csv(tmp_path / 'measurements.csv')
    sessions = pd.read_csv(tmp_path / 'sessions.csv')
    assert measurements.tip_type.tolist() == ['small_tip', 'large_tip']
    assert sessions.tip_type.tolist() == ['small_tip', 'large_tip']


class CheckConfirmRobot(RecordingRobot):
    PIPET_TIPS = {'small_tip': {'volume': .2, 'capacity_margin': .005},
                  'large_tip': {'volume': 1.0, 'capacity_margin': .01}}

    def pause_after_error(self, *args, **kwargs):
        raise AssertionError("check/confirm should not pause here")


@pytest.fixture
def check_confirm(recording_validation, tmp_path, monkeypatch):
    from pipetting_data import embedded_liquid_validations, overaspirate_corrections
    monkeypatch.setattr(overaspirate_corrections, 'CORRECTIONS_FILE', tmp_path / 'tracked.csv')
    monkeypatch.setattr(embedded_liquid_validations, 'calibrated_volumes_ul', lambda liquid, tip: [100.0])
    return recording_validation


def _run_check_confirm(validation, tmp_path, masses):
    robot = CheckConfirmRobot(masses)
    lash = SimpleNamespace(nr_robot=robot, simulate=False, logger=logging.getLogger(__name__))
    result = validation.validate_pipetting_accuracy(
        lash, 'source', 'dest', 'water', [.1], replicates=4, output_folder=str(tmp_path / 'raw'),
        tip_type='small_tip')
    return robot, result


def test_check_consistent_with_history_needs_no_confirm(check_confirm, tmp_path):
    robot, result = _run_check_confirm(check_confirm, tmp_path, [.0998, .1, .1002, .0999])
    assert len(robot.transfers) == 4
    tracked = pd.read_csv(tmp_path / 'tracked.csv')
    assert tracked.last_verdict.tolist() == ['consistent']
    measurements = pd.read_csv(tmp_path / 'measurements.csv')
    assert measurements.accepted.all() and (measurements.decision_reason == 'consistent').all()
    log = pd.read_csv(tmp_path / 'corrections.csv')
    assert log.validation_type.tolist() == ['check_confirm']


def test_surprising_check_is_confirmed_before_a_large_step(check_confirm, tmp_path):
    robot, result = _run_check_confirm(check_confirm, tmp_path, [.092] * 4 + [.0998] * 4)
    assert len(robot.transfers) == 8
    confirm_overaspirate = robot.transfers[-1][1]['overaspirate_vol']
    assert confirm_overaspirate > .007
    tracked = pd.read_csv(tmp_path / 'tracked.csv')
    assert tracked.last_verdict.tolist() == ['drift_confirmed']
    assert tracked.overaspirate_ul.iloc[0] > 7.0
    statistics = result['volume_statistics'][0]
    assert statistics['measured_mean_ml'] < .095  # accuracy reported from the check batch


def test_update_problem_never_stops_validation(check_confirm, tmp_path, monkeypatch):
    from pipetting_data import overaspirate_corrections
    monkeypatch.setattr(overaspirate_corrections, 'decide', lambda *a, **k: 1 / 0)
    robot, result = _run_check_confirm(check_confirm, tmp_path, [.1] * 4)
    assert result['success'] and not (tmp_path / 'tracked.csv').exists()


def test_gross_miss_blocks_the_whole_volume_update_without_pausing(check_confirm, tmp_path, monkeypatch):
    sent = []
    import sys
    monkeypatch.setitem(sys.modules, 'slack_agent', SimpleNamespace(send_slack_message=sent.append))
    # CheckConfirmRobot.pause_after_error raises, so any pause would fail this test.
    robot, result = _run_check_confirm(check_confirm, tmp_path, [.1, .1, .1, .07])
    assert result['success'] and len(robot.transfers) == 4
    assert any('not updated' in message for message in sent)
    assert not (tmp_path / 'tracked.csv').exists()  # no partial use of the clean replicates
    measurements = pd.read_csv(tmp_path / 'measurements.csv')
    assert not measurements.accepted.any()


class ScaleRobot(CheckConfirmRobot):
    """Reports an unstable balance on the chosen (0-based) dispense numbers."""

    def __init__(self, masses, unstable):
        super().__init__(masses)
        self.unstable = set(unstable)

    def dispense_from_vial_into_vial(self, *args, **kwargs):
        mass, info = super().dispense_from_vial_into_vial(*args, **kwargs)
        if len(self.transfers) - 1 in self.unstable:
            info.update(pre_stable_count=0, post_stable_count=0,
                        pre_baseline_std=.13, post_baseline_std=.30)
        return mass, info


def _run_scale(validation, tmp_path, monkeypatch, masses, unstable, volumes):
    sent = []
    import sys
    from pipetting_data import embedded_liquid_validations
    monkeypatch.setitem(sys.modules, 'slack_agent', SimpleNamespace(send_slack_message=sent.append))
    monkeypatch.setattr(embedded_liquid_validations, 'calibrated_volumes_ul',
                        lambda liquid, tip: [v * 1000 for v in volumes])
    robot = ScaleRobot(masses, unstable)
    lash = SimpleNamespace(nr_robot=robot, simulate=False, logger=logging.getLogger(__name__))
    result = validation.validate_pipetting_accuracy(
        lash, 'source', 'dest', 'water', volumes, replicates=4, output_folder=str(tmp_path / 'raw'),
        tip_type='small_tip')
    return robot, result, sent


def test_unstable_scale_is_not_retried_and_stops_the_session(check_confirm, tmp_path, monkeypatch):
    robot, result, sent = _run_scale(check_confirm, tmp_path, monkeypatch, [.1] * 8,
                                     unstable={0}, volumes=[.1, .15])
    assert len(robot.transfers) == 1  # no retries, nothing after the failure
    assert result['success']
    assert sum('Scale unstable' in message for message in sent) == 1
    assert not (tmp_path / 'tracked.csv').exists()


def test_scale_failure_keeps_earlier_updates_and_skips_the_rest(check_confirm, tmp_path, monkeypatch):
    robot, result, sent = _run_scale(check_confirm, tmp_path, monkeypatch,
                                     [.1, .1, .1, .1, .15, .15, .15, .15],
                                     unstable={5}, volumes=[.1, .15, .18])
    # 100 uL: 4 clean replicates -> updated. 150 uL: unstable on its 2nd replicate -> stop.
    assert len(robot.transfers) == 6
    tracked = pd.read_csv(tmp_path / 'tracked.csv')
    assert tracked.volume_target.tolist() == [100.0]
    measurements = pd.read_csv(tmp_path / 'measurements.csv')
    assert measurements[measurements.target_volume_ml == .1].accepted.all()
    assert not measurements[measurements.target_volume_ml == .15].accepted.any()
    assert sorted(measurements.target_volume_ml.unique()) == [.1, .15]  # 180 uL never dispensed


def test_broken_scale_readings_never_trigger_the_near_zero_pause(check_confirm, tmp_path, monkeypatch):
    # Garbage near-zero mass from an unstable scale; ScaleRobot.pause_after_error raises.
    robot, result, sent = _run_scale(check_confirm, tmp_path, monkeypatch, [-.09],
                                     unstable={0}, volumes=[.1])
    assert result['success'] and len(robot.transfers) == 1


def test_there_is_only_one_validation_protocol():
    """Validation must never be able to rewrite base calibration CSVs or pick another mode."""
    import inspect
    from pipetting_data import embedded_calibration_validation as validation
    from pipetting_data.embedded_liquid_validations import run_liquid_validations
    options = set(inspect.signature(validation.validate_pipetting_accuracy).parameters)
    assert not options & {'protocol', 'adaptive_correction', 'compensate_overvolume'}
    assert 'protocol' not in inspect.signature(run_liquid_validations).parameters
    assert not hasattr(validation, '_update_calibration_csv')
    assert not hasattr(validation, '_write_calibration_csv')


def test_environment_reads_last_row_without_loading_the_whole_log(tmp_path, monkeypatch):
    from pipetting_data import embedded_calibration_validation as validation
    log = tmp_path / 'mqtt_log.csv'
    monkeypatch.setattr(validation, 'MQTT_LOG_FILE', str(log))
    assert validation._get_latest_environmental_data() is None  # missing file
    header = 'Timestamp,sht_temp_c,sht_rh,bmp_pa\r\n'
    log.write_text(header, encoding='utf-8')
    assert validation._get_latest_environmental_data() is None  # header only
    rows = ''.join(f'2026-11-04T10:{i % 60:02d},20.0,40.0,100000\r\n' for i in range(5000))
    log.write_text(header + rows + '2026-11-04T11:00,25.4,,99594\r\n', encoding='utf-8')
    assert validation._get_latest_environmental_data() == dict(temp_c=25.4, humidity_pct=None,
                                                               pressure_pa=99594.0)

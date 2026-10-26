"""Pipette recording through real controller methods, without hardware or live CSVs."""
import json
import logging
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

import North_Safe as north
from North_Safe import North_Robot
from pipetting_data import dispense_recording as recording
from pipetting_data.pipetting_parameters import PipettingParameters, ReservoirParameters

REAL_THREAD = threading.Thread
REAL_SLEEP = time.sleep


@pytest.fixture
def robot(monkeypatch, tmp_path):
    robot = object.__new__(North_Robot)
    robot.logger = logging.getLogger(__name__)
    robot.simulate = False
    robot.dispense_recording = 'auto'
    robot.c9 = MagicMock()
    robot.VIAL_DF = pd.DataFrame([
        dict(vial_name='source', location='main_8mL_rack', location_index=0, vial_volume=10., capped=False),
        dict(vial_name='dest', location='clamp', location_index=0, vial_volume=0., capped=False),
    ])
    robot.PIPET_FLUID_VIAL_INDEX = 0
    robot.PIPET_FLUID_VOLUME = 10.
    robot.HELD_PIPET_TYPE = 'small_tip'
    robot.GRIPPER_STATUS = None
    robot.normalize_vial_index = lambda name: {'source': 0, 'dest': 1}.get(name, name)
    robot.get_vial_info = lambda vial, field: robot.VIAL_DF.at[robot.normalize_vial_index(vial), field]
    robot._get_optimized_parameters = lambda volume, liquid, params: params or PipettingParameters()
    robot._get_optimized_reservoir_parameters = lambda volume, liquid, params=None: params or ReservoirParameters()
    robot._ensure_vial_accessible_for_pipetting = MagicMock(return_value=True)
    robot.get_config_parameter = lambda config, key, field, **kw: 2.5 if config == 'pumps' and field == 'volume' else None
    robot.get_vial_location = MagicMock(return_value=[0, 0, 0, 0])
    robot.get_height_at_location = MagicMock(return_value=0)
    robot.adjust_height_based_on_pipet_held = lambda height: height
    robot.get_speed = MagicMock(return_value=10)
    robot.adjust_pump_speed = MagicMock()
    robot.save_robot_status = MagicMock()
    robot.pause_after_error = MagicMock(side_effect=AssertionError('Recording paused execution'))
    robot.move_home = MagicMock()
    robot.is_vial_pipetable = MagicMock(return_value=True)
    robot.recap_clamp_vial = MagicMock()
    robot.return_vial_home = MagicMock()
    robot._create_mass_vs_time_plot = MagicMock(side_effect=AssertionError('Unexpected plot'))
    robot.balance = 10.
    robot.mass_increment = .1
    robot.c9.read_scale.side_effect = lambda: (True, robot.balance)
    robot.pipet_dispense = MagicMock(side_effect=lambda *a, **kw: setattr(robot, 'balance', robot.balance + robot.mass_increment))

    # Drive the actual polling function deterministically at its existing waits.
    robot.pollers = []
    clock = [0.]

    class Poller:
        def __init__(self, target, daemon):
            self.target = target
            closure = dict(zip(target.__code__.co_freevars, target.__closure__))
            self.event = closure['monitoring_active'].cell_contents
            self.joined = False

        def sample(self):
            if not self.event.is_set():
                return
            original = self.event.is_set
            self.event.is_set = MagicMock(side_effect=[True, False])
            try:
                self.target()
            finally:
                self.event.is_set = original

        def start(self):
            robot.pollers.append(self)
            self.sample()

        def join(self, timeout):
            self.joined = True

    def sleep(seconds):
        clock[0] += seconds
        if seconds == 2.:
            for poller in robot.pollers:
                poller.sample()
                poller.sample()

    monkeypatch.setattr(north.threading, 'Thread', Poller)
    monkeypatch.setattr(north.time, 'time', lambda: clock[0])
    monkeypatch.setattr(north.time, 'sleep', sleep)
    monkeypatch.setattr(recording, 'MASTER_MEASUREMENTS', tmp_path / 'measurements.csv')
    from pipetting_data import embedded_calibration_validation as validation
    monkeypatch.setattr(validation, '_get_latest_environmental_data', lambda: None)
    return robot


def rows():
    return pd.read_csv(recording.MASTER_MEASUREMENTS)


def test_records_existing_schema_and_preserves_unweighed_return(robot):
    master = Path(__file__).resolve().parents[1] / 'pipetting_data/master_pipetting_measurements.csv'
    columns = list(pd.read_csv(master, nrows=0).columns)
    pd.DataFrame(columns=columns).to_csv(recording.MASTER_MEASUREMENTS, index=False)
    robot.mass_increment = .0789
    result = robot.dispense_into_vial('dest', .1, liquid='ethanol')
    assert result[0] is None
    assert result[1]['pre_total_count'] == 0
    data = rows()
    assert list(data.columns) == columns
    assert data.measured_volume_ml.iloc[0] == pytest.approx(.1)
    assert data.mass_difference_g.iloc[0] == pytest.approx(.0789)
    assert data.session_id.iloc[0].startswith('routine_')
    assert data.validation_type.iloc[0] == 'vial_to_vial'
    assert data.source_vial.iloc[0] == 'source'
    assert data.destination_vial.iloc[0] == 'dest'
    assert json.loads(data.stability_info.iloc[0])['tip_type'] == 'small_tip'
    assert data.optimization_stage.isna().all()


@pytest.mark.parametrize('change', ['off', 'old_object', 'no_liquid', 'unknown', 'clamp_2', 'rack', 'wrong_index', 'excluded', 'simulate'])
def test_ineligible_dispenses_do_not_measure_or_record(robot, change):
    liquid = 'water'
    if change == 'off':
        robot.dispense_recording = 'off'
    elif change == 'old_object':
        del robot.dispense_recording
    elif change == 'no_liquid':
        liquid = None
    elif change == 'unknown':
        liquid = 'unregistered_liquid'
    elif change in ('clamp_2', 'rack'):
        robot.VIAL_DF.at[1, 'location'] = change
    elif change == 'wrong_index':
        robot.VIAL_DF.at[1, 'location_index'] = 1
    elif change == 'simulate':
        robot.simulate = True
    robot.dispense_into_vial('dest', .1, liquid=liquid, record_measurement=change != 'excluded')
    assert not robot.pollers
    robot.c9.read_scale.assert_not_called()
    assert not recording.MASTER_MEASUREMENTS.exists()
    robot.pipet_dispense.assert_called_once()


def test_eligibility_uses_prepared_destination_without_extra_movement(robot):
    robot.VIAL_DF.at[1, 'location'] = 'rack'
    def prepare(*args, **kwargs):
        robot.VIAL_DF.at[1, 'location'] = 'clamp'
        return True
    robot._ensure_vial_accessible_for_pipetting.side_effect = prepare
    robot.dispense_into_vial('dest', .1, liquid='water')
    assert len(rows()) == 1
    robot._ensure_vial_accessible_for_pipetting.assert_called_once()


def test_balance_failure_is_missing_without_retry_or_pause(robot):
    robot.c9.read_scale.side_effect = OSError('Balance offline')
    robot.dispense_into_vial('dest', .1, liquid='water')
    data = rows()
    assert data.mass_difference_g.isna().all()
    assert data.measured_volume_ml.isna().all()
    assert not data.measurement_acceptable.iloc[0]
    assert 'failure_reason' in json.loads(data.stability_info.iloc[0])
    assert robot.VIAL_DF.at[1, 'vial_volume'] == pytest.approx(.1)
    robot.pipet_dispense.assert_called_once()
    robot.pause_after_error.assert_not_called()


def test_actual_zero_is_not_missing(robot):
    robot.mass_increment = 0.
    robot.dispense_into_vial('dest', .1, liquid='water')
    assert rows().mass_difference_g.iloc[0] == 0.
    assert rows().measurement_acceptable.iloc[0]


def test_existing_unstable_baseline_fallback_is_preserved(robot):
    robot.c9.read_scale.side_effect = lambda: (False, robot.balance)
    mass, info = robot.dispense_into_vial('dest', .1, liquid='water', measure_weight=True, continuous_mass_monitoring=True)
    assert mass == pytest.approx(.1)
    assert info['pre_stable_count'] == 0
    assert rows().mass_difference_g.iloc[0] == pytest.approx(mass)


def test_save_failure_does_not_change_delivery_or_inventory(robot, monkeypatch):
    monkeypatch.setattr(recording, 'append_csv', MagicMock(side_effect=OSError('Disk unavailable')))
    result = robot.dispense_into_vial('dest', .1, liquid='water')
    assert result[0] is None
    robot.pipet_dispense.assert_called_once()
    assert robot.VIAL_DF.at[1, 'vial_volume'] == pytest.approx(.1)
    assert robot.PIPET_FLUID_VOLUME == pytest.approx(9.9)


@pytest.mark.parametrize('error', [OSError('Delivery failed'), KeyboardInterrupt()])
def test_delivery_error_is_preserved_without_a_measurement_row(robot, error):
    robot.pipet_dispense.side_effect = error
    with pytest.raises(type(error)) as raised:
        robot.dispense_into_vial('dest', .1, liquid='water')
    assert raised.value is error
    assert not recording.MASTER_MEASUREMENTS.exists()


def test_polling_start_error_is_not_hidden(robot, monkeypatch):
    monkeypatch.setattr(north.threading, 'Thread', MagicMock(side_effect=RuntimeError('Thread unavailable')))
    with pytest.raises(RuntimeError, match='Thread unavailable'):
        robot.dispense_into_vial('dest', .1, liquid='water')
    robot.pipet_dispense.assert_not_called()
    assert not recording.MASTER_MEASUREMENTS.exists()


def test_explicit_calibration_opt_out_preserves_weighing(robot):
    mass, info = robot.dispense_into_vial('dest', .1, liquid='water', measure_weight=True,
        continuous_mass_monitoring=True, record_measurement=False)
    assert mass == pytest.approx(.1)
    assert info['pre_total_count'] > 0
    assert len(robot.pollers) == 1
    assert not recording.MASTER_MEASUREMENTS.exists()


def test_explicit_traditional_weighing_is_not_replaced(robot):
    robot.c9.read_steady_scale.side_effect = [10., 10.1]
    mass, _ = robot.dispense_into_vial('dest', .1, liquid='water', measure_weight=True)
    assert mass == pytest.approx(.1)
    assert not robot.pollers
    assert robot.c9.read_steady_scale.call_count == 2
    assert rows().mass_difference_g.iloc[0] == pytest.approx(mass)


def test_split_transfer_records_each_chunk_without_changing_total_return(robot):
    robot.plan_vial_transfer = MagicMock(return_value=(.3, 3, PipettingParameters(), 'small_tip'))
    robot.aspirate_from_vial = MagicMock()
    robot.get_vial_in_location = MagicMock(return_value=None)
    result = robot.dispense_from_vial_into_vial('source', 'dest', .9, liquid='water', remove_tip=False, return_vial_home=False)
    assert result[0] == 0
    assert rows().target_volume_ml.tolist() == pytest.approx([.3] * 3)
    assert len(rows()) == 3
    assert rows().session_id.nunique() == 3


def test_actual_parameters_and_unknown_aspiration_fields(robot):
    robot._dispense_aspiration_settings = vars(PipettingParameters(aspirate_speed=7, overaspirate_vol=.02))
    robot.dispense_into_vial('dest', .1, liquid='water', parameters=PipettingParameters(dispense_speed=13))
    robot._dispense_aspiration_settings = None
    robot.dispense_into_vial('dest', .1, liquid='water')
    data = rows()
    assert data.aspirate_speed.iloc[0] == 7
    assert data.overaspirate_vol.iloc[0] == .02
    assert data.dispense_speed.iloc[0] == 13
    assert pd.isna(data.aspirate_speed.iloc[1])


def test_recording_on_matches_off_motion_and_return(robot):
    commands = MagicMock()
    for name in ('c9', 'pipet_dispense', 'adjust_pump_speed', 'save_robot_status', '_ensure_vial_accessible_for_pipetting'):
        commands.attach_mock(getattr(robot, name), name)
    results, traces = [], []
    for policy in ('off', 'auto'):
        robot.dispense_recording = policy
        robot.VIAL_DF.at[1, 'vial_volume'] = 0.
        robot.PIPET_FLUID_VOLUME = 10.
        commands.reset_mock()
        results.append(robot.dispense_into_vial('dest', .1, liquid='water'))
        traces.append([call for call in commands.mock_calls if not call[0].startswith('c9.read')])
    assert traces[0] == traces[1]
    assert results[0] == results[1]


def test_mixing_does_not_record(robot):
    robot.aspirate_from_vial = MagicMock()
    robot.plan_vial_transfer = MagicMock(return_value=(.1, 1, PipettingParameters(), 'small_tip'))
    robot.get_vial_in_location = MagicMock(return_value=None)
    robot.remove_pipet = MagicMock()
    robot.mix_vial('dest', .1, repeats=3, liquid='water')
    assert robot.pipet_dispense.call_count == 3
    assert not robot.pollers
    assert not recording.MASTER_MEASUREMENTS.exists()


def test_reservoir_is_unchanged_and_does_not_record(robot):
    robot.dispense_into_vial_from_reservoir(1, 'dest', 6., liquid='water')
    assert robot.c9.dispense_ml.call_count == 3
    assert not recording.MASTER_MEASUREMENTS.exists()


@pytest.mark.parametrize('density', [0., -1., float('nan'), float('inf')])
def test_invalid_density_does_not_enable_recording(robot, monkeypatch, density):
    monkeypatch.setattr(recording, 'get_liquid', lambda liquid: {'density': density})
    robot.dispense_into_vial('dest', .1, liquid='water')
    assert not robot.pollers
    assert not recording.MASTER_MEASUREMENTS.exists()


def test_real_aspiration_captures_effective_settings_and_raw_aspiration_clears_them(robot):
    params = PipettingParameters(aspirate_speed=None, retract_speed=None, overaspirate_vol=.02)
    robot.resolve_pipetting_setup = lambda *args: (params, 'small_tip')
    robot.get_tip_if_needed = MagicMock()
    robot.check_if_aspiration_volume_unacceptable = MagicMock()
    robot.get_aspirate_height = MagicMock(return_value=0)
    robot.get_tip_dependent_aspirate_speed = MagicMock(return_value=7)
    robot.PIPET_TIPS = {'small_tip': {'volume': .2}}
    robot.aspirate_from_vial('source', .1, parameters=params, liquid='water')
    robot.dispense_into_vial('dest', .1, liquid='water')
    assert rows().aspirate_speed.iloc[0] == 7
    assert rows().retract_speed.iloc[0] == 10
    assert rows().overaspirate_vol.iloc[0] == .02
    assert rows().source_vial.iloc[0] == 'source'
    robot.pipet_aspirate(.01, wait_time=0)
    assert robot._dispense_aspiration_settings is None
    assert robot._dispense_aspiration_source is None


@pytest.mark.parametrize('balance_fails', [False, True])
def test_closed_cap_exchange_matches_with_recording_on_and_off(robot, balance_fails):
    robot.VIAL_DF['cap_type'] = 'closed'
    robot.is_vial_pipetable = lambda vial: not robot.get_vial_info(vial, 'capped')
    robot.is_vial_movable = lambda vial: True
    robot.get_vial_index_from_name = robot.normalize_vial_index
    robot.get_vial_in_location = lambda loc, slot: next(iter(robot.VIAL_DF.index[
        (robot.VIAL_DF.location == loc) & (robot.VIAL_DF.location_index == slot)]), None)

    def return_home(vial, **kwargs):
        robot.VIAL_DF.at[vial, 'location'] = 'rack'
        robot.VIAL_DF.at[vial, 'capped'] = True
        robot.GRIPPER_STATUS = None

    def move(vial, location, location_index, **kwargs):
        robot.VIAL_DF.at[vial, 'location'] = location
        robot.VIAL_DF.at[vial, 'location_index'] = location_index

    def uncap(**kwargs):
        robot.VIAL_DF.at[1, 'capped'] = False
        robot.GRIPPER_STATUS = 'Cap'

    robot.return_vial_home.side_effect = return_home
    robot.move_vial_to_location = MagicMock(side_effect=move)
    robot.uncap_clamp_vial = MagicMock(side_effect=uncap)
    robot._ensure_vial_accessible_for_pipetting = lambda *args, **kwargs: (
        North_Robot._ensure_vial_accessible_for_pipetting(robot, *args, **kwargs))
    if balance_fails:
        robot.c9.read_scale.side_effect = OSError('Balance offline')
    commands = MagicMock()
    for name in ('return_vial_home', 'move_vial_to_location', 'uncap_clamp_vial',
                 'c9', 'pipet_dispense', 'save_robot_status', 'adjust_pump_speed'):
        commands.attach_mock(getattr(robot, name), name)
    traces, states = [], []
    for policy in ('off', 'auto'):
        robot.dispense_recording = policy
        robot.VIAL_DF.loc[:, 'location'] = ['clamp', 'rack']
        robot.VIAL_DF.loc[:, 'location_index'] = [0, 1]
        robot.VIAL_DF.loc[:, 'capped'] = [False, True]
        robot.VIAL_DF.loc[:, 'vial_volume'] = [10., 0.]
        robot.GRIPPER_STATUS = 'Cap'
        robot.PIPET_FLUID_VOLUME = 10.
        commands.reset_mock()
        robot.dispense_into_vial('dest', .1, liquid='water')
        traces.append([call for call in commands.mock_calls if not call[0].startswith('c9.read')])
        states.append((robot.VIAL_DF.copy(), robot.GRIPPER_STATUS, robot.PIPET_FLUID_VOLUME))
    assert traces[0] == traces[1]
    pd.testing.assert_frame_equal(states[0][0], states[1][0])
    assert states[0][1:] == states[1][1:]


def test_invalid_configuration_fails_before_hardware_initialization():
    c9, c8 = MagicMock(), MagicMock()
    with pytest.raises(ValueError, match='dispense_recording'):
        North_Robot(c9, c8, dispense_recording='invalid')
    assert c9.mock_calls == []
    assert c8.mock_calls == []


def test_real_thread_collects_and_stops(robot, monkeypatch):
    monkeypatch.setattr(north.threading, 'Thread', REAL_THREAD)
    monkeypatch.setattr(north.time, 'sleep', REAL_SLEEP)
    monkeypatch.setattr(north.time, 'time', time.monotonic)
    robot.dispense_into_vial('dest', .1, liquid='water')
    assert rows().mass_difference_g.iloc[0] == pytest.approx(.1)
    info = json.loads(rows().stability_info.iloc[0])
    assert info['pre_total_count'] >= 2
    assert info['post_total_count'] >= 2

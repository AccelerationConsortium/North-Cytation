"""Routine recording policy and real controller paths, with hardware mocked."""
import json
import ast
import importlib
import logging
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock
from types import SimpleNamespace

import pandas as pd
import pytest

import North_Safe as north
from North_Safe import North_Robot
from pipetting_data import dispense_recording as recording
from pipetting_data.pipetting_parameters import PipettingParameters, ReservoirParameters

REAL_SLEEP = time.sleep
REAL_THREAD = threading.Thread


@pytest.fixture
def robot(monkeypatch, tmp_path):
    robot = object.__new__(North_Robot)
    robot.logger = logging.getLogger(__name__)
    robot.simulate = False
    robot.dispense_recording = "auto"
    robot.c9 = MagicMock()
    robot.VIAL_DF = pd.DataFrame([
        dict(vial_name="source", location="main_8mL_rack", location_index=0, vial_volume=10., capped=False),
        dict(vial_name="dest", location="clamp", location_index=0, vial_volume=0., capped=False),
    ])
    robot.PIPET_FLUID_VIAL_INDEX = 0
    robot.PIPET_FLUID_VOLUME = 10.
    robot.HELD_PIPET_TYPE = "small_tip"
    robot.GRIPPER_STATUS = None
    robot.normalize_vial_index = lambda name: {"source": 0, "dest": 1}.get(name, name)
    robot.get_vial_info = lambda vial, field: robot.VIAL_DF.at[robot.normalize_vial_index(vial), field]
    robot._get_optimized_parameters = lambda volume, liquid, params: params or PipettingParameters()
    robot._get_optimized_reservoir_parameters = lambda volume, liquid, params=None: params or ReservoirParameters()
    robot._ensure_vial_accessible_for_pipetting = MagicMock(return_value=True)
    robot.get_config_parameter = lambda config, key, field, **kw: 2.5 if config == "pumps" and field == "volume" else None
    robot.get_vial_location = MagicMock(return_value=[0, 0, 0, 0])
    robot.get_height_at_location = MagicMock(return_value=0)
    robot.adjust_height_based_on_pipet_held = lambda height: height
    robot.get_speed = MagicMock(return_value=10)
    robot.adjust_pump_speed = MagicMock()
    robot.pipet_dispense = MagicMock()
    robot.save_robot_status = MagicMock()
    robot.pause_after_error = MagicMock(side_effect=AssertionError("Optional weighing paused the robot"))
    robot.move_home = MagicMock()
    robot.is_vial_pipetable = MagicMock(return_value=True)
    robot.recap_clamp_vial = MagicMock()
    robot.return_vial_home = MagicMock()
    robot._create_mass_vs_time_plot = MagicMock(side_effect=AssertionError("Unexpected plot"))
    monkeypatch.setattr(north.time, "sleep", lambda seconds: None)
    # Never start a physical polling thread; baseline maths is tested separately.
    monkeypatch.setattr(north.threading, "Thread", MagicMock())
    monkeypatch.setattr(recording, "MASTER_MEASUREMENTS", tmp_path / "measurements.csv")
    from pipetting_data import embedded_calibration_validation as validation
    monkeypatch.setattr(validation, "_get_latest_environmental_data", lambda: None)
    return robot


def rows():
    return pd.read_csv(recording.MASTER_MEASUREMENTS)


def successful_weighing(monkeypatch, mass=0.1):
    analysis = MagicMock(return_value=(mass, dict(pre_stable_count=2, post_stable_count=2), 10., 10. + mass))
    monkeypatch.setattr(north, "analyze_baselines", analysis)
    return analysis


def test_existing_schema_and_one_row_per_dispense(robot, monkeypatch):
    successful_weighing(monkeypatch, 0.0789)
    master = Path(__file__).resolve().parents[1] / "pipetting_data/master_pipetting_measurements.csv"
    existing_columns = list(pd.read_csv(master, nrows=0).columns)
    pd.DataFrame(columns=existing_columns).to_csv(recording.MASTER_MEASUREMENTS, index=False)
    for _ in range(2):
        robot.dispense_into_vial("dest", 0.1, liquid="ethanol")
    data = rows()
    assert list(data.columns) == existing_columns
    assert len(data) == 2
    assert data.session_id.nunique() == 2
    assert data.measured_volume_ml.tolist() == pytest.approx([0.1, 0.1])
    assert data.mass_difference_g.tolist() == pytest.approx([0.0789, 0.0789])
    assert data.source_vial.tolist() == ["source", "source"]
    assert data.destination_vial.tolist() == ["dest", "dest"]
    assert data.optimization_stage.isna().all()
    assert robot.pipet_dispense.call_count == 2


@pytest.mark.parametrize("change", ["off", "no_liquid", "unknown", "clamp_2", "rack", "wrong_index", "excluded"])
def test_ineligible_operations_do_not_weigh_or_record(robot, monkeypatch, change):
    analysis = successful_weighing(monkeypatch)
    liquid = "water"
    if change == "off":
        robot.dispense_recording = "off"
    if change == "no_liquid":
        liquid = None
    if change == "unknown":
        liquid = "unregistered_liquid"
    if change in ("clamp_2", "rack"):
        robot.VIAL_DF.at[1, "location"] = change
    if change == "wrong_index":
        robot.VIAL_DF.at[1, "location_index"] = 1
    robot.dispense_into_vial("dest", 0.1, liquid=liquid, record_measurement=change != "excluded")
    analysis.assert_not_called()
    robot.c9.read_steady_scale.assert_not_called()
    assert not recording.MASTER_MEASUREMENTS.exists()
    robot.pipet_dispense.assert_called_once()


def test_failed_weighing_is_empty_and_does_not_pause_or_retry(robot):
    mass, info = robot.dispense_into_vial("dest", 0.1, liquid="water")
    assert mass is None
    assert info["failure_reason"]
    data = rows()
    assert data.measured_volume_ml.isna().all()
    assert data.mass_difference_g.isna().all()
    assert data.accuracy_pct.isna().all()
    assert not data.measurement_acceptable.iloc[0]
    assert json.loads(data.stability_info.iloc[0])["failure_reason"]
    assert robot.VIAL_DF.at[1, "vial_volume"] == pytest.approx(0.1)
    robot.pipet_dispense.assert_called_once()
    robot.pause_after_error.assert_not_called()


def test_true_measured_zero_is_preserved(robot, monkeypatch):
    successful_weighing(monkeypatch, 0.)
    robot.dispense_into_vial("dest", 0.1, liquid="water")
    data = rows()
    assert data.mass_difference_g.iloc[0] == 0.
    assert data.measurement_acceptable.iloc[0]


@pytest.mark.parametrize("reservoir", [False, True])
def test_simulation_cannot_save_even_when_explicitly_requested(robot, monkeypatch, tmp_path, reservoir):
    robot.simulate = True
    monkeypatch.chdir(tmp_path)
    def forbidden_write(*args, **kwargs):
        raise AssertionError("Simulation tried to save measurement data")
    monkeypatch.setattr(pd.DataFrame, "to_csv", forbidden_write)
    kwargs = dict(liquid="water", measure_weight=True, continuous_mass_monitoring=True, save_mass_data=True)
    if reservoir:
        robot.dispense_into_vial_from_reservoir(1, "dest", 0.1, **kwargs)
    else:
        robot.dispense_into_vial("dest", 0.1, **kwargs)
    recording.record_dispense(robot, liquid="water", density=1., target=0.1,
                              mass=0.1, stability={}, parameters=None, destination=1)
    assert list(tmp_path.iterdir()) == []


def test_balance_exception_is_nonfatal_in_traditional_mode(robot):
    robot.c9.read_steady_scale.side_effect = OSError("Balance offline")
    mass, info = robot.dispense_into_vial("dest", 0.1, measure_weight=True, record_measurement=False)
    assert mass == 0.0  # Preserve the numeric contract used by legacy calibration.
    assert info["failure_reason"]
    robot.pipet_dispense.assert_called_once()


def test_recording_error_never_repeats_physical_dispense(robot, monkeypatch):
    successful_weighing(monkeypatch)
    monkeypatch.setattr(recording, "append_csv", MagicMock(side_effect=OSError("Disk unavailable")))
    robot.dispense_into_vial("dest", 0.1, liquid="water")
    robot.pipet_dispense.assert_called_once()
    robot.pause_after_error.assert_not_called()


def test_pipette_chunks_are_individual_rows(robot, monkeypatch):
    successful_weighing(monkeypatch, 0.3)
    robot.plan_vial_transfer = MagicMock(return_value=(0.3, 3, PipettingParameters(), "small_tip"))
    robot.aspirate_from_vial = MagicMock()
    robot.get_vial_in_location = MagicMock(return_value=None)
    robot.dispense_from_vial_into_vial("source", "dest", 0.9, liquid="water",
                                     remove_tip=False, return_vial_home=False)
    assert len(rows()) == 3
    assert rows().target_volume_ml.tolist() == pytest.approx([0.3] * 3)
    assert robot.pipet_dispense.call_count == 3


def test_reservoir_chunks_share_master_and_preserve_total_command(robot, monkeypatch):
    successful_weighing(monkeypatch, 2.)
    robot.dispense_into_vial_from_reservoir(1, "dest", 6., liquid="water",
                                         reservoir_params=ReservoirParameters(overaspirate_vol=0.1))
    data = rows()
    assert len(data) == 3
    assert data.validation_type.tolist() == ["reservoir_dispensing"] * 3
    assert data.reservoir_index.tolist() == [1] * 3
    assert data.target_volume_ml.sum() == pytest.approx(6.)
    assert data.overaspirate_vol.sum() == pytest.approx(0.1)
    assert sum(call.args[1] for call in robot.c9.dispense_ml.call_args_list) == pytest.approx(6.1)
    robot.return_vial_home.assert_called_once()
    robot.recap_clamp_vial.assert_called_once()
    robot.move_home.assert_called_once()


@pytest.mark.parametrize('reservoir', [False, True])
def test_failed_legacy_weighing_keeps_numeric_return(robot, reservoir):
    kwargs = dict(liquid='water', measure_weight=True,
                  continuous_mass_monitoring=True, record_measurement=False)
    if reservoir:
        mass, info = robot.dispense_into_vial_from_reservoir(1, 'dest', .1, **kwargs)
    else:
        mass, info = robot.dispense_into_vial('dest', .1, **kwargs)
    assert mass == 0.0
    assert info['failure_reason']
    assert mass / 1.0 == 0.0  # Existing callers perform this calculation.
    assert not recording.MASTER_MEASUREMENTS.exists()


@pytest.mark.parametrize('simulated', [False, True])
@pytest.mark.parametrize('weighing_succeeds', [False, True])
def test_recording_preserves_reservoir_hardware_sequence(robot, monkeypatch, simulated, weighing_succeeds):
    robot.simulate = simulated
    analysis = successful_weighing(monkeypatch, 2.)
    if not weighing_succeeds:
        analysis.return_value = (None, {'failure_reason': 'Balance offline'}, None, None)
    robot.get_config_parameter = lambda config, key, field, **kw: {
        'volume': 2.5, 'carousel_angle': 90, 'carousel_height': 10,
    }.get(field) if config == 'pumps' else None
    commands = MagicMock()
    for name in ('c9', 'move_home', 'recap_clamp_vial', 'return_vial_home'):
        commands.attach_mock(getattr(robot, name), name)

    def execute(policy):
        commands.reset_mock()
        robot.VIAL_DF.at[1, 'vial_volume'] = 0.
        robot.dispense_recording = policy
        robot.dispense_into_vial_from_reservoir(1, 'dest', 6., liquid='water',
            reservoir_params=ReservoirParameters(overaspirate_vol=.1))
        return [call for call in commands.mock_calls if not call[0].startswith('c9.read')]

    without_recording = execute('off')
    with_recording = execute('auto')
    assert with_recording == without_recording
    robot.recap_clamp_vial.assert_called_once()
    robot.move_home.assert_called_once()
    assert robot.c9.dispense_ml.call_count == 3


def test_conditioning_does_not_pass_liquid():
    source = Path(__file__).resolve().parents[1] / 'workflows/surfactant_grid_adaptive_concentrations.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'condition_tip')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    lash = SimpleNamespace(nr_robot=MagicMock(), logger=MagicMock())
    namespace['condition_tip'](lash, 'water_vial', liquid_type='water')
    for method in (lash.nr_robot.aspirate_from_vial, lash.nr_robot.dispense_into_vial):
        assert method.call_count == 5
        assert all('liquid' not in call.kwargs for call in method.call_args_list)


@pytest.mark.parametrize('module_name,class_name', [
    ('calibration_protocol_northrobot', 'HardwareCalibrationProtocol'),
    ('calibration_protocol_reservoir', 'ReservoirCalibrationProtocol'),
    ('calibration_protocol_heated', 'HeatedCalibrationProtocol'),
])
def test_real_calibration_callers_handle_failed_weighing_without_type_errors(
        robot, monkeypatch, tmp_path, module_name, class_name):
    module = importlib.import_module('sdl_pipette_calibration.protocols.' + module_name)
    protocol = object.__new__(getattr(module, class_name))
    protocol.quality_std_threshold = .001
    protocol._check_and_swap_vials = MagicMock()
    protocol._check_and_maintain_temperature = MagicMock()
    protocol._report = MagicMock()
    robot.aspirate_from_vial = MagicMock()
    robot.remove_pipet = MagicMock()
    robot._get_optimized_parameters = lambda volume, liquid, params: (
        PipettingParameters(**params) if isinstance(params, dict) else params or PipettingParameters())
    robot.c9.read_steady_scale.side_effect = OSError('Balance offline')
    lash = SimpleNamespace(nr_robot=robot, simulate=False, logger=robot.logger)
    state = dict(lash_e=lash, simulate=False, source_vial='source', measurement_vial='dest',
                 liquid='water', measurement_count=0, reservoir_index=1,
                 continuous_mass_monitoring=True, target_temperature_c=25.)
    results = protocol.measure(state, .1, dict(overaspirate_vol=0.), replicates=1)
    assert results[0]['volume'] == 0.0
    assert state['measurement_count'] == 1
    assert list(tmp_path.iterdir()) == []


def test_interruption_during_next_reservoir_baseline_preserves_original_exception(robot, monkeypatch):
    successful_weighing(monkeypatch)
    baseline_waits = []

    def sleep(seconds):
        if seconds == 2.:
            baseline_waits.append(seconds)
            if len(baseline_waits) == 3:
                raise KeyboardInterrupt()

    monkeypatch.setattr(north.time, 'sleep', sleep)
    with pytest.raises(KeyboardInterrupt):
        robot.dispense_into_vial_from_reservoir(1, 'dest', 6., liquid='water')
    robot.c9.dispense_ml.assert_called_once()


def test_calibration_opt_out_keeps_weighing_without_auto_row(robot, monkeypatch):
    analysis = successful_weighing(monkeypatch)
    mass, _ = robot.dispense_into_vial("dest", 0.1, liquid="water", measure_weight=True,
                                     continuous_mass_monitoring=True, record_measurement=False)
    assert mass == 0.1
    analysis.assert_called_once()
    assert not recording.MASTER_MEASUREMENTS.exists()


def test_mixing_never_records(robot, monkeypatch):
    analysis = successful_weighing(monkeypatch)
    robot.aspirate_from_vial = MagicMock()
    robot.plan_vial_transfer = MagicMock(return_value=(0.1, 1, PipettingParameters(), "small_tip"))
    robot.get_vial_in_location = MagicMock(return_value=None)
    robot.remove_pipet = MagicMock()
    robot.mix_vial("dest", 0.1, repeats=3, liquid="water")
    assert robot.pipet_dispense.call_count == 3
    analysis.assert_not_called()
    assert not recording.MASTER_MEASUREMENTS.exists()


def test_missing_chunk_measurement_never_becomes_partial_total(robot, monkeypatch):
    analysis = successful_weighing(monkeypatch)
    analysis.side_effect = [(0.3, {}, 10., 10.3), (None, {"failure_reason": "Balance offline"}, None, None)]
    robot.plan_vial_transfer = MagicMock(return_value=(0.3, 2, PipettingParameters(), "small_tip"))
    robot.aspirate_from_vial = MagicMock()
    robot.get_vial_in_location = MagicMock(return_value=None)
    mass, _ = robot.dispense_from_vial_into_vial("source", "dest", 0.6, liquid="water",
                                               remove_tip=False, return_vial_home=False)
    assert mass is None
    assert len(rows()) == 2
    assert rows().mass_difference_g.isna().tolist() == [False, True]


def test_baselines_use_stable_means_and_never_substitute_unstable_data():
    readings = [dict(phase=phase, mass_g=mass, steady_status=stable)
                for phase, mass, stable in [
                    ("baseline_pre", 10., True), ("baseline_pre", 10.002, True),
                    ("baseline_pre", 9., False),
                    ("baseline_post", 10.101, True), ("baseline_post", 10.101, True)]]
    mass, info, pre, post = recording.analyze_baselines(readings)
    assert mass == pytest.approx(0.1)
    assert pre == pytest.approx(10.001)
    assert post == pytest.approx(10.101)
    assert info["pre_stable_count"] == 2
    for r in readings:
        r["steady_status"] = False
    assert recording.analyze_baselines(readings)[0] is None
    assert recording.analyze_baselines([])[0] is None


def test_actual_aspiration_settings_are_used(robot, monkeypatch):
    successful_weighing(monkeypatch)
    robot._dispense_aspiration_settings = vars(PipettingParameters(aspirate_speed=7, overaspirate_vol=0.02))
    robot.dispense_into_vial("dest", 0.1, liquid="water", parameters=PipettingParameters(dispense_speed=13))
    data = rows()
    assert data.aspirate_speed.iloc[0] == 7
    assert data.overaspirate_vol.iloc[0] == 0.02
    assert data.dispense_speed.iloc[0] == 13


def test_unknown_aspiration_settings_are_not_invented(robot, monkeypatch):
    successful_weighing(monkeypatch)
    robot.dispense_into_vial("dest", 0.1, liquid="water")
    data = rows()
    assert data.aspirate_speed.isna().all()
    assert data.overaspirate_vol.isna().all()
    assert data.dispense_speed.iloc[0] == 10


def test_real_polling_uses_existing_windows_with_mocked_balance(robot, monkeypatch):
    monkeypatch.setattr(north.time, "sleep", REAL_SLEEP)
    monkeypatch.setattr(north.threading, "Thread", REAL_THREAD)
    balance = {"mass": 10.}
    robot.c9.read_scale.side_effect = lambda: (True, balance["mass"])
    robot.pipet_dispense.side_effect = lambda *a, **kw: balance.update(mass=10.1)
    mass, info = robot.dispense_into_vial("dest", 0.1, liquid="water")
    assert mass == pytest.approx(0.1)
    assert info["pre_stable_count"] >= 2
    assert info["post_stable_count"] >= 2
    assert rows().measured_volume_ml.iloc[0] == pytest.approx(0.1)


def test_reservoir_calibration_uses_the_unified_master(robot, monkeypatch):
    from pipetting_data import embedded_calibration_validation as validation
    monkeypatch.setattr(validation, "MASTER_PIPETTING_MEASUREMENTS_DB", str(recording.MASTER_MEASUREMENTS))
    monkeypatch.setattr(validation, "MASTER_RESERVOIR_MEASUREMENTS_DB", str(recording.MASTER_MEASUREMENTS))
    validation._append_to_master_measurements([dict(session_id="cal_pipette", validation_type="vial_to_vial")], simulate=False)
    validation._append_to_master_measurements([dict(session_id="cal_reservoir", validation_type="reservoir_dispensing")], "reservoir", simulate=False)
    before = recording.MASTER_MEASUREMENTS.read_bytes()
    validation._append_to_master_measurements([dict(session_id="simulated")], "reservoir", simulate=True)
    assert recording.MASTER_MEASUREMENTS.read_bytes() == before
    assert rows().session_id.tolist() == ["cal_pipette", "cal_reservoir"]


def test_workflow_liquid_validation_simulates_with_real_dispense_controller(robot, monkeypatch, tmp_path):
    from pipetting_data import embedded_calibration_validation as validation
    from pipetting_data.embedded_liquid_validations import run_liquid_validations
    robot.simulate = True
    robot.aspirate_from_vial = MagicMock()
    robot.remove_pipet = MagicMock()
    robot.get_vial_in_location = MagicMock(return_value=None)
    robot.move_vial_to_location = MagicMock()
    robot.resolve_pipetting_setup = lambda volume, liquid, params=None, **kw: (params or PipettingParameters(), "small_tip")
    robot.plan_vial_transfer = lambda volume, liquid, params, tip, compensate: (volume, 1, params, tip)
    generated = iter([.092, .098, .100])
    monkeypatch.setattr(validation, '_generate_simulated_mass', lambda **kwargs: next(generated))
    save = MagicMock(side_effect=AssertionError("Simulation must not save"))
    monkeypatch.setattr(validation, 'append_csv', save)
    monkeypatch.setattr(recording, 'append_csv', save)
    lash = SimpleNamespace(nr_robot=robot, simulate=True, logger=robot.logger)
    results = run_liquid_validations(lash, [("dest", "water", [100.])], tmp_path / 'output', replicates=1)
    assert results[("dest", "water", "small_tip")]['success']
    assert robot.pipet_dispense.call_count == 6  # Three conditioning and three validation stages.
    save.assert_not_called()
    robot.pause_after_error.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_reservoir_validation_simulates_with_real_dispense_controller(robot, monkeypatch, tmp_path):
    from pipetting_data import embedded_calibration_validation as validation
    robot.simulate = True
    robot.move_vial_to_location = MagicMock()
    save = MagicMock(side_effect=AssertionError("Simulation must not save"))
    monkeypatch.setattr(validation, 'append_csv', save)
    monkeypatch.setattr(recording, 'append_csv', save)
    lash = SimpleNamespace(nr_robot=robot, simulate=True, logger=robot.logger)
    result = validation.validate_reservoir_accuracy(lash, 1, "dest", "water", [.1],
                                                    replicates=1, output_folder=str(tmp_path / 'output'))
    assert result['success']
    robot.c9.dispense_ml.assert_called_once()
    save.assert_not_called()
    assert list(tmp_path.iterdir()) == []

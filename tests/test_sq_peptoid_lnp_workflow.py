"""Check workflow ordering and real fixed-height mixing without hardware."""
import importlib
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from North_Safe import North_Robot, North_Track


def test_shipped_inventory_matches_requested_positions():
    path = Path(__file__).resolve().parents[1] / 'status' / 'SQ_Peptoid_LNP_test_vials.csv'
    inventory = pd.read_csv(path).set_index('vial_name')
    assert inventory.location_index.to_dict() == {'organic_phase': 0, 'aqeuous_phase': 1, 'sample': 2}
    assert inventory.home_location_index.to_dict() == inventory.location_index.to_dict()
    assert inventory.location.eq('main_8mL_rack').all()
    assert inventory.loc[['organic_phase', 'aqeuous_phase'], 'vial_volume'].ge(.6).all()
    assert inventory.loc['sample', 'vial_volume'] == 0.


@pytest.fixture
def setup(monkeypatch, tmp_path):
    robot = MagicMock(spec=North_Robot)
    robot.VIAL_FILE = str(tmp_path / 'vials.csv')
    robot.VIAL_DF = pd.DataFrame([
        dict(vial_name='organic_phase', location='main_8mL_rack', location_index=0, vial_volume=1.),
        dict(vial_name='aqeuous_phase', location='main_8mL_rack', location_index=1, vial_volume=1.),
        dict(vial_name='sample', location='main_8mL_rack', location_index=2, vial_volume=0.),
    ])
    robot.VIAL_DF.to_csv(robot.VIAL_FILE, index=False)
    robot.get_vial_info.side_effect = lambda name, field: robot.VIAL_DF.set_index('vial_name').at[name, field]
    robot.WELLPLATES = {'96 WELL PLATE': {'num_wells': 96, 'max_volume_per_well': .3}}
    robot.PIPET_TIPS = {'large_tip': {'volume': 1.}}
    track = MagicMock(spec=North_Track)
    from robot_state.plate_inventory import PlateInventory
    import yaml
    root = Path(__file__).resolve().parents[1]
    track.inventory = PlateInventory({"wellplate_type": "96 WELL PLATE", "num_in_source": 1},
        yaml.safe_load((root / "robot_state/wellplates.yaml").read_text()),
        yaml.safe_load((root / "robot_state/track_positions.yaml").read_text()))
    track.NUM_SOURCE = 1
    track.ACTIVE_WELLPLATE_POSITION = None
    lash = SimpleNamespace(nr_robot=robot, nr_track=track, simulate=True,
        logger=logging.getLogger(__name__), _workflow_should_continue=True,
        grab_new_wellplate=MagicMock(), measure_wellplate=MagicMock(return_value=None))
    constructor = MagicMock(return_value=lash)
    monkeypatch.setitem(sys.modules, 'master_usdl_coordinator', SimpleNamespace(Lash_E=constructor))
    name = 'workflows.SQ_Peptoid_LNP_test'
    monkeypatch.delitem(sys.modules, name, raising=False)
    workflow = importlib.import_module(name)
    config = {key: getattr(workflow, key) for key in workflow._CONFIG_KEYS}
    config['INPUT_VIAL_STATUS_FILE'] = robot.VIAL_FILE
    config['OUTPUT_DIRECTORY'] = str(tmp_path / 'output')
    lash.workflow_config = config
    calls = MagicMock()
    calls.attach_mock(robot, 'robot')
    calls.attach_mock(lash.grab_new_wellplate, 'grab_new_wellplate')
    calls.attach_mock(lash.measure_wellplate, 'measure_wellplate')
    return workflow, lash, constructor, calls, tmp_path


def test_six_steps_in_order_and_simulation_does_not_save(setup):
    workflow, lash, constructor, calls, root = setup
    assert workflow.execute(lash.workflow_config, show_gui=False) is None
    kwargs = constructor.call_args.kwargs
    assert kwargs['config'] is lash.workflow_config
    assert kwargs['workflow_name'] == 'SQ_Peptoid_LNP_test'
    assert kwargs['workflow_globals'] is workflow.__dict__
    assert 'simulate' not in kwargs
    robot = lash.nr_robot
    transfers = robot.dispense_from_vial_into_vial.call_args_list
    assert [call.args for call in transfers] == [('organic_phase', 'sample', .6), ('aqeuous_phase', 'sample', .6)]
    assert transfers[1].kwargs['remove_tip'] is False
    assert transfers[1].kwargs['return_vial_home'] is False
    aspiration = robot.aspirate_from_vial.call_args_list
    assert aspiration[0].args == ('sample', .4)
    assert aspiration[0].kwargs['move_up'] is False
    assert aspiration[0].kwargs['parameters'].asp_disp_cycles == 2
    robot.dispense_into_vial.assert_called_once()
    assert robot.dispense_into_vial.call_args.kwargs['initial_move'] is False
    assert robot.dispense_into_vial.call_args.kwargs['record_measurement'] is False
    assert [call.args for call in aspiration[1:]] == [('sample', .15)] * 3
    assert [call.args for call in robot.dispense_into_wellplate.call_args_list] == [([i], [.15]) for i in range(3)]
    lash.grab_new_wellplate.assert_called_once()
    lash.measure_wellplate.assert_called_once_with(r'C:\Protocols\SQ_Peroxide.prt', [0, 1, 2], plate_type='96 WELL PLATE')
    sequence = [call[0] for call in calls.mock_calls]
    start = sequence.index('robot.dispense_from_vial_into_vial')
    assert sequence[start:] == [
        'robot.dispense_from_vial_into_vial', 'robot.dispense_from_vial_into_vial',
        'robot.aspirate_from_vial', 'robot.dispense_into_vial', 'robot.remove_pipet',
        'grab_new_wellplate', 'robot.aspirate_from_vial', 'robot.dispense_into_wellplate',
        'robot.aspirate_from_vial', 'robot.dispense_into_wellplate',
        'robot.aspirate_from_vial', 'robot.dispense_into_wellplate',
        'robot.remove_pipet', 'measure_wellplate',
    ]
    assert not (root / 'output').exists()


def test_gui_confirmed_values_drive_execution(setup):
    workflow, lash, _, _, _ = setup
    lash.workflow_config = {**lash.workflow_config, 'PHASE_VOLUME_UL': 300., 'MIX_VOLUME_UL': 200.,
        'MIX_CYCLES': 5, 'ALIQUOT_VOLUME_UL': 100., 'WELL_INDICES': [12, 13, 14]}
    workflow.execute()
    assert [call.args[2] for call in lash.nr_robot.dispense_from_vial_into_vial.call_args_list] == [.3, .3]
    mixing = lash.nr_robot.aspirate_from_vial.call_args_list[0]
    assert mixing.args[1] == .2
    assert mixing.kwargs['parameters'].asp_disp_cycles == 4
    assert lash.measure_wellplate.call_args.args[1] == [12, 13, 14]


def test_cancelled_startup_does_not_execute(setup):
    workflow, lash, _, _, _ = setup
    lash._workflow_should_continue = False
    assert workflow.execute() is None
    lash.nr_robot.home_robot_components.assert_not_called()
    lash.grab_new_wellplate.assert_not_called()


@pytest.mark.parametrize('change', ['same_position', 'not_enough_liquid', 'occupied_plate_area', 'duplicate_well'])
def test_invalid_setup_fails_before_experiment_motion(setup, change):
    workflow, lash, _, _, _ = setup
    if change == 'same_position':
        lash.nr_robot.VIAL_DF.at[1, 'location_index'] = 2
    elif change == 'not_enough_liquid':
        lash.nr_robot.VIAL_DF.at[0, 'vial_volume'] = .1
    elif change == 'occupied_plate_area':
        lash.nr_track.inventory.add("pipetting_area", "96 WELL PLATE")
    else:
        lash.workflow_config['WELL_INDICES'] = [0, 0, 1]
    with pytest.raises(ValueError):
        workflow.execute()
    lash.nr_robot.home_robot_components.assert_not_called()
    lash.nr_robot.dispense_from_vial_into_vial.assert_not_called()


def test_live_read_is_saved_without_sending_notifications(setup, monkeypatch):
    workflow, lash, _, _, root = setup
    protocol = root / 'test.prt'
    protocol.touch()
    lash.simulate = False
    lash.workflow_config = {**lash.workflow_config, 'SIMULATE': False, 'MEASUREMENT_PROTOCOL_FILE': str(protocol)}
    monkeypatch.setattr(workflow, '_send_workflow_slack', MagicMock())
    expected = pd.DataFrame({'signal': [1., 2., 3.]}, index=[0, 1, 2])
    lash.measure_wellplate.return_value = expected
    pd.testing.assert_frame_equal(workflow.execute(), expected)
    paths = list((root / 'output').glob('SQ_Peptoid_LNP_test_*.csv'))
    assert len(paths) == 1
    pd.testing.assert_frame_equal(pd.read_csv(paths[0], index_col=0), expected)


def test_real_mixing_executes_exactly_three_cycles_without_intermediate_arm_motion(setup):
    workflow, lash, _, _, _ = setup
    workflow.execute()
    aspiration = lash.nr_robot.aspirate_from_vial.call_args_list[0]
    dispense = lash.nr_robot.dispense_into_vial.call_args
    robot = object.__new__(North_Robot)
    robot.simulate = True
    robot.logger = logging.getLogger(__name__)
    robot.c9 = MagicMock()
    robot.dispense_recording = 'off'
    robot.VIAL_DF = pd.DataFrame([dict(vial_name='sample', location='main_8mL_rack',
        location_index=2, vial_volume=1.2, capped=False)])
    robot.normalize_vial_index = lambda vial: 0
    robot.get_vial_info = lambda vial, field: robot.VIAL_DF.at[0, field]
    robot.resolve_pipetting_setup = lambda amount, liquid, params, tip: (params, tip)
    robot._get_optimized_parameters = lambda amount, liquid, params: params
    robot._ensure_vial_accessible_for_pipetting = MagicMock(return_value=True)
    robot.is_vial_pipetable = MagicMock(return_value=True)
    robot.get_tip_if_needed = MagicMock()
    robot.check_if_aspiration_volume_unacceptable = MagicMock()
    robot.get_aspirate_height = MagicMock(return_value=5)
    robot.get_height_at_location = MagicMock(return_value=10)
    robot.adjust_height_based_on_pipet_held = lambda height: height
    robot.get_vial_location = MagicMock(return_value=[0, 0, 0, 0])
    robot.get_speed = MagicMock(return_value=10)
    robot.get_config_parameter = MagicMock(return_value=None)
    robot.adjust_pump_speed = MagicMock()
    robot.save_robot_status = MagicMock()
    robot.PIPET_FLUID_VOLUME = 0.
    robot.PIPET_FLUID_VIAL_INDEX = None
    robot.HELD_PIPET_TYPE = 'large_tip'
    robot.aspirate_from_vial(*aspiration.args, **aspiration.kwargs)
    robot.dispense_into_vial(*dispense.args, **dispense.kwargs)
    pump_calls = [call for call in robot.c9.mock_calls if call[0] in ('aspirate_ml', 'dispense_ml')]
    assert [call[0] for call in pump_calls] == ['aspirate_ml', 'dispense_ml'] * 3
    assert [call.args for call in pump_calls] == [(0, .4)] * 6
    robot.c9.move_z.assert_called_once_with(5)
    commands = robot.c9.mock_calls
    first = commands.index(pump_calls[0])
    assert [call[0] for call in commands[first:]] == ['aspirate_ml', 'dispense_ml'] * 3
    assert robot.VIAL_DF.at[0, 'vial_volume'] == pytest.approx(1.2)
    assert robot.PIPET_FLUID_VOLUME == pytest.approx(0.)

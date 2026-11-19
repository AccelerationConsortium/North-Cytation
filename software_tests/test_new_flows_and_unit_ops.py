"""New North_Robot unit operations and flows call the expected building blocks."""
import logging
from unittest.mock import MagicMock, call

import pytest
import yaml

from North_Safe import North_Robot


def test_small_tip_positions_come_from_vial_positions_yaml():
    robot = object.__new__(North_Robot)
    robot.VIAL_POSITIONS = yaml.safe_load(open('robot_state/vial_positions.yaml', encoding='utf-8'))
    assert robot.get_small_tip_staging_positions() == [
        ('main_8mL_rack', 47), ('main_8mL_rack', 46), ('main_8mL_rack', 45),
        ('heater', 2), ('heater', 5), ('heater', 8), ('main_8mL_rack', 44), ('main_8mL_rack', 43)]
    assert robot.get_small_tip_safe_indices('main_8mL_rack') == [43, 44, 45, 46, 47]
    assert robot.get_small_tip_safe_indices('heater') == [2, 5, 8]
    assert robot.get_small_tip_safe_indices('large_vial_rack') == []


@pytest.fixture
def robot():
    robot = object.__new__(North_Robot)
    robot.VIAL_POSITIONS = yaml.safe_load(open('robot_state/vial_positions.yaml', encoding='utf-8'))
    robot.logger = logging.getLogger(__name__)
    robot.simulate = True
    robot.c9 = MagicMock()
    open_8mL = dict(capped=True, cap_type='open', vial_type='8_mL')
    robot.vials = {
        'stock': dict(vial_name='stock', vial_volume=6.0, location='main_8mL_rack', location_index=5,
                      home_location='main_8mL_rack', home_location_index=5, **open_8mL),
        'source': dict(vial_volume=3.0, location='main_8mL_rack', location_index=6,
                       home_location='main_8mL_rack', home_location_index=6, **open_8mL),
        'dest': dict(vial_volume=0.0, location='main_8mL_rack', location_index=7,
                     home_location='main_8mL_rack', home_location_index=7, **open_8mL),
    }
    robot.normalize_vial_index = lambda name: name
    robot.get_vial_info = lambda vial, field: robot.vials[vial][field]
    robot.get_vial_in_location = MagicMock(return_value=None)
    for name in ['aspirate_from_vial', 'dispense_into_vial', 'move_vial_to_location', 'return_vial_home',
                 'dispense_into_vial_from_reservoir', 'dispense_from_vial_into_vial', 'grab_vial',
                 'drop_off_vial', 'recap_clamp_vial', 'get_safe_height', 'get_config_parameter']:
        setattr(robot, name, MagicMock())
    robot.GRIPPER_VIAL_INDEX = None
    robot.GRIPPER_STATUS = None
    return robot


def test_condition_tip_cycles_aspirate_and_dispense(robot):
    robot.condition_tip('stock', volume_per_cycle_mL=0.05, cycles=3)
    assert robot.aspirate_from_vial.call_count == 3
    assert robot.dispense_into_vial.call_count == 3
    assert robot.aspirate_from_vial.call_args[0][:2] == ('stock', 0.05)


def test_stage_vials_uses_safest_free_positions_in_order(robot):
    robot.get_vial_in_location.side_effect = lambda loc, idx: 'other' if (loc, idx) == ('clamp', 0) else None
    staged = robot.stage_vials(['stock', 'source'])
    assert staged == {'stock': ('main_8mL_rack', 47), 'source': ('main_8mL_rack', 46)}
    assert robot.move_vial_to_location.call_args_list == [
        call('stock', 'main_8mL_rack', 47), call('source', 'main_8mL_rack', 46)]


def test_stage_vials_leaves_vial_in_place_when_no_position_free(robot):
    staged = robot.stage_vials(['stock'], safe_positions=[])
    assert staged == {}
    robot.move_vial_to_location.assert_not_called()


def test_return_staged_vials_skips_vials_already_home(robot):
    robot.vials['source']['location_index'] = 47
    robot.return_staged_vials(['stock', 'source'])
    robot.return_vial_home.assert_called_once_with('source')


def test_refill_vial_from_reservoir_tops_up_to_target(robot):
    added = robot.refill_vial('stock', 8.0, reservoir=1)
    assert added == pytest.approx(2.0)
    robot.dispense_into_vial_from_reservoir.assert_called_once()
    assert robot.dispense_into_vial_from_reservoir.call_args[0] == (1, 'stock', pytest.approx(2.0))


def test_refill_vial_from_source_vial_is_limited_by_source_volume(robot):
    added = robot.refill_vial('dest', 8.0, source_vial='source')
    assert added == pytest.approx(3.0)
    assert robot.dispense_from_vial_into_vial.call_args[0] == ('source', 'dest', pytest.approx(3.0))


def test_refill_vial_skips_small_additions(robot):
    assert robot.refill_vial('stock', 6.5, reservoir=1, min_addition_mL=1.0) == 0.0
    robot.dispense_into_vial_from_reservoir.assert_not_called()


def test_refill_vial_without_source_is_not_implemented_yet(robot):
    with pytest.raises(NotImplementedError):
        robot.refill_vial('stock', 8.0)


def test_create_substock_runs_reservoir_first_then_pipetting_then_vortex(robot):
    robot.vortex_vial = MagicMock()
    total = robot.create_substock('dest', [
        {'source_vial': 'stock', 'volume_mL': 0.5, 'liquid': 'SDS'},
        {'reservoir': 1, 'volume_mL': 4.5},
    ], vortex_time=5)
    assert total == pytest.approx(5.0)
    robot.dispense_into_vial_from_reservoir.assert_called_once()
    assert robot.dispense_from_vial_into_vial.call_args[0] == ('stock', 'dest', 0.5)
    robot.vortex_vial.assert_called_once_with('dest', 5)


def test_create_substock_rejects_ambiguous_component(robot):
    with pytest.raises(ValueError):
        robot.create_substock('dest', [{'source_vial': 'stock', 'reservoir': 1, 'volume_mL': 1.0}])


def test_vortex_vial_returns_vial_home_by_default(robot):
    robot.vials['stock']['location'] = 'clamp'
    robot.vials['stock']['location_index'] = 0
    robot.get_config_parameter.return_value = 0
    robot.vortex_vial('stock', 2)
    robot.drop_off_vial.assert_called_once_with('stock', 'main_8mL_rack', 5)


# ---- dispense_from_vials_into_wellplate (flow) and tip conditioning ----
import pandas as pd

from pipetting_data.liquid_registry import load_liquids


def test_condition_tip_setting_matches_refill_pipets_for_every_liquid():
    liquids = load_liquids()
    assert all(props['condition_tip'] == props['refill_pipets'] for props in liquids.values())
    assert liquids['glycerol']['condition_tip'] is True
    assert liquids['water']['condition_tip'] is False


@pytest.fixture
def plate_robot(robot):
    robot.PIPET_TIPS = {'small_tip': {'volume': 0.25}, 'large_tip': {'volume': 1.0}}
    robot.HELD_PIPET_TYPE = 'large_tip'
    robot._resolve_pipetting_plate = MagicMock(return_value='96 WELL PLATE')
    from pathlib import Path
    robot.WELLPLATES = yaml.safe_load((Path(__file__).resolve().parents[1] / 'robot_state' / 'wellplates.yaml').read_text())
    robot.resolve_pipetting_setup = MagicMock(
        side_effect=lambda volume, liquid, params: (f'params_{volume}', 'small_tip' if volume < 0.05 else 'large_tip'))
    robot.is_vial_movable = MagicMock(return_value=True)
    robot.is_vial_pipetable = MagicMock(return_value=True)  # Open vials unless a test says otherwise
    robot.dispense_into_wellplate = MagicMock()
    robot.remove_pipet = MagicMock()
    robot.condition_tip = MagicMock()
    robot.pause_after_error = MagicMock(side_effect=AssertionError('paused'))
    robot.return_vial_home.side_effect = lambda vial, **kw: robot.vials[vial].update(
        location=robot.vials[vial]['home_location'], location_index=robot.vials[vial]['home_location_index'])
    return robot


def test_wellplate_flow_groups_by_tip_and_uses_fresh_tip_per_vial(plate_robot):
    plate = pd.DataFrame({'stock': [0.02, 0.2, 0.01, 0.3], 'source': [0.1, 0.0, 0.0, 0.0]}, index=[0, 1, 2, 3])
    plate_robot.dispense_from_vials_into_wellplate(plate, liquid='water')
    doses = [(c.args[0], c.args[1], c.kwargs['specified_tip']) for c in plate_robot.aspirate_from_vial.call_args_list]
    # stock: small tip group first (largest dose first), then large tip group; then source
    assert doses == [('stock', 0.02, 'small_tip'), ('stock', 0.01, 'small_tip'),
                     ('stock', 0.3, 'large_tip'), ('stock', 0.2, 'large_tip'), ('source', 0.1, 'large_tip')]
    assert plate_robot.remove_pipet.call_count == 3  # Held tip dropped before staging, then one per vial
    plate_robot.condition_tip.assert_not_called()   # water has condition_tip: false


def test_wellplate_flow_stages_emptiest_small_tip_vial_first_and_returns_home(plate_robot):
    plate_robot.vials['source']['vial_volume'] = 1.0   # Will be emptiest after dispensing
    # dest goes into its own well: a 96-well plate holds 0.3 mL per well
    plate = pd.DataFrame({'stock': [0.02, 0.0], 'source': [0.03, 0.0], 'dest': [0.0, 0.25]}, index=[0, 1])
    plate_robot.vials['dest']['vial_volume'] = 5.0
    def move(vial, location, index):
        plate_robot.vials[vial].update(location=location, location_index=index)
    plate_robot.move_vial_to_location.side_effect = move
    plate_robot.dispense_from_vials_into_wellplate(plate, liquid='water')
    # Only small-tip vials are staged; the emptiest-after-dispensing one gets the safest position
    assert plate_robot.move_vial_to_location.call_args_list == [
        call('source', 'main_8mL_rack', 47), call('stock', 'main_8mL_rack', 46)]
    # Every vial goes home right after its own dispense, in plan order
    assert [c.args[0] for c in plate_robot.return_vial_home.call_args_list] == ['stock', 'source', 'dest']


def test_wellplate_flow_skips_staging_for_vials_already_safe_or_immovable(plate_robot):
    plate_robot.vials['stock']['location_index'] = 46  # Already small-tip safe
    plate_robot.is_vial_movable = MagicMock(side_effect=lambda vial: vial != 'source')
    plate = pd.DataFrame({'stock': [0.02], 'source': [0.03]}, index=[0])
    plate_robot.dispense_from_vials_into_wellplate(plate, liquid='water')
    plate_robot.move_vial_to_location.assert_not_called()


def test_wellplate_flow_stages_closed_vial_to_rack_0_5_and_open_vial_to_small_tip_spot(plate_robot):
    # 'stock' is closed (any tip) -> rack 0-5; 'source' is open with the small tip -> 47. No tip held.
    plate_robot.is_vial_pipetable = MagicMock(side_effect=lambda vial: vial != 'stock')
    plate_robot.vials['stock']['location_index'] = 30
    events = []
    def move(vial, location, index):
        events.append((vial, location, index, plate_robot.HELD_PIPET_TYPE))
        plate_robot.vials[vial].update(location=location, location_index=index)
    plate_robot.move_vial_to_location.side_effect = move
    plate_robot.remove_pipet.side_effect = lambda: setattr(plate_robot, 'HELD_PIPET_TYPE', None)
    plate = pd.DataFrame({'stock': [0.2], 'source': [0.03]}, index=[0])
    plate_robot.dispense_from_vials_into_wellplate(plate, liquid='water')
    assert sorted(events) == [('source', 'main_8mL_rack', 47, None), ('stock', 'main_8mL_rack', 0, None)]
    assert [c.args[0] for c in plate_robot.return_vial_home.call_args_list] == ['stock', 'source']
    assert plate_robot.STAGED_POSITIONS == {}  # Cleared once the vials are home


def test_wellplate_flow_leaves_closed_vial_in_rack_0_5_where_it_is(plate_robot):
    plate_robot.is_vial_pipetable = MagicMock(return_value=False)  # stock is closed, at rack 5
    plate = pd.DataFrame({'stock': [0.3]}, index=[0])
    plate_robot.dispense_from_vials_into_wellplate(plate, liquid='water')
    plate_robot.move_vial_to_location.assert_not_called()


# ---- stage_vials memory and the clamp swap ----

def test_staged_vial_leaving_clamp_goes_back_to_its_staged_spot(robot):
    robot.STAGED_POSITIONS = {'source': ('main_8mL_rack', 2)}
    robot.GRIPPER_STATUS = 'Cap'
    robot._return_from_clamp('source')
    robot.recap_clamp_vial.assert_called_once()
    robot.move_vial_to_location.assert_called_once_with('source', 'main_8mL_rack', 2, move_speed=None)
    robot.return_vial_home.assert_not_called()


def test_unstaged_vial_leaving_clamp_goes_home(robot):
    robot._return_from_clamp('dest')
    robot.return_vial_home.assert_called_once_with('dest', move_speed=None)


def test_pipet_safe_move_positions_are_rack_0_to_5(robot):
    assert robot.get_pipet_safe_move_positions() == [('main_8mL_rack', i) for i in range(6)]


# ---- dispense_from_vial_into_vial (flow): staging ----

@pytest.fixture
def v2v_robot(robot):
    del robot.dispense_from_vial_into_vial  # Use the real flow
    robot.HELD_PIPET_TYPE = None
    robot.plan_vial_transfer = MagicMock(return_value=(0.5, 1, 'params', 'large_tip'))
    robot.dispense_into_vial = MagicMock(return_value=(0.0, None))
    robot.remove_pipet = MagicMock()
    def move(vial, location, index, **kwargs):
        robot.vials[vial].update(location=location, location_index=index)
    robot.move_vial_to_location.side_effect = move
    robot.return_vial_home.side_effect = lambda vial, **kw: robot.vials[vial].update(
        location=robot.vials[vial]['home_location'], location_index=robot.vials[vial]['home_location_index'])
    for name in ('source', 'dest'):
        robot.vials[name].update(cap_type='closed', location_index=30 + len(name),
                                 home_location_index=30 + len(name))
    return robot


def test_v2v_closed_to_closed_stages_both_to_rack_0_5_then_returns_them(v2v_robot):
    v2v_robot.dispense_from_vial_into_vial('source', 'dest', 0.5, liquid='water')
    assert v2v_robot.move_vial_to_location.call_args_list[:2] == [
        call('source', 'main_8mL_rack', 0), call('dest', 'main_8mL_rack', 1)]
    assert {c.args[0] for c in v2v_robot.return_vial_home.call_args_list} == {'source', 'dest'}
    assert v2v_robot.STAGED_POSITIONS == {}


def test_v2v_open_small_tip_source_goes_to_small_tip_spot_and_open_dest_stays(v2v_robot):
    v2v_robot.plan_vial_transfer.return_value = (0.05, 1, 'params', 'small_tip')
    for name in ('source', 'dest'):
        v2v_robot.vials[name]['cap_type'] = 'open'
    v2v_robot.dispense_from_vial_into_vial('source', 'dest', 0.05, liquid='water')
    assert v2v_robot.move_vial_to_location.call_args_list[0] == call('source', 'main_8mL_rack', 47)
    assert all(c.args[0] != 'dest' for c in v2v_robot.move_vial_to_location.call_args_list)


def test_v2v_vials_already_in_place_are_not_moved(v2v_robot):
    v2v_robot.vials['source'].update(location_index=3, home_location_index=3)
    v2v_robot.vials['dest'].update(location_index=4, home_location_index=4)
    v2v_robot.dispense_from_vial_into_vial('source', 'dest', 0.5, liquid='water')
    v2v_robot.move_vial_to_location.assert_not_called()


def test_wellplate_flow_conditions_once_per_tip_group_for_flagged_liquid(plate_robot):
    plate = pd.DataFrame({'stock': [0.02, 0.04, 0.3]}, index=[0, 1, 2])
    plate_robot.vials['stock']['location_index'] = 46
    plate_robot.dispense_from_vials_into_wellplate(plate, liquid='glycerol')
    assert plate_robot.condition_tip.call_args_list == [
        call('stock', volume_per_cycle_mL=0.04, specified_tip='small_tip'),
        call('stock', volume_per_cycle_mL=0.3, specified_tip='large_tip')]


def test_wellplate_flow_takes_a_liquid_per_vial(plate_robot):
    plate = pd.DataFrame({'stock': [0.1, 0.0], 'source': [0.0, 0.2]}, index=[0, 1])
    plate_robot.dispense_from_vials_into_wellplate(plate, liquid={'stock': 'glycerol', 'source': 'water'})
    used = [(c.args[0], c.kwargs['liquid']) for c in plate_robot.aspirate_from_vial.call_args_list]
    assert used == [('stock', 'glycerol'), ('source', 'water')]
    plate_robot.condition_tip.assert_called_once()  # only glycerol conditions its tip
    with pytest.raises(ValueError, match="No liquid given for vial"):
        plate_robot.dispense_from_vials_into_wellplate(plate, liquid={'stock': 'glycerol'})


@pytest.mark.parametrize('value, message', [(float('nan'), 'volume missing'), (-0.1, 'negative volume'),
                                            ('lots', 'is not a volume in mL')])
def test_wellplate_flow_refuses_bad_volumes_before_moving(plate_robot, value, message):
    plate = pd.DataFrame({'stock': [0.1, value]}, index=[0, 1], dtype=object)
    with pytest.raises(ValueError, match=message):
        plate_robot.dispense_from_vials_into_wellplate(plate, liquid='water')
    plate_robot.aspirate_from_vial.assert_not_called()


def test_wellplate_flow_refuses_an_overfull_well_before_moving(plate_robot):
    # 96-well plates hold 0.3 mL per well (wellplates.yaml); 0.2 + 0.15 into well 1 is too much
    plate = pd.DataFrame({'stock': [0.1, 0.2], 'source': [0.0, 0.15]}, index=[0, 1])
    with pytest.raises(ValueError, match=r"well 1: 0.35 mL planned"):
        plate_robot.dispense_from_vials_into_wellplate(plate, liquid='water')
    plate_robot.aspirate_from_vial.assert_not_called()


def test_wellplate_flow_has_no_strategy_argument(plate_robot):
    with pytest.raises(TypeError):
        plate_robot.dispense_from_vials_into_wellplate(pd.DataFrame({'stock': [0.1]}), strategy='serial')


@pytest.mark.parametrize('liquid, held, expected', [('glycerol', None, True), ('water', None, False),
                                                     ('glycerol', 'large_tip', False)])
def test_vial_to_vial_conditions_fresh_tip_for_flagged_liquid(robot, liquid, held, expected):
    robot.HELD_PIPET_TYPE = held
    robot.condition_tip = MagicMock()
    robot.plan_vial_transfer = MagicMock(return_value=(0.5, 1, 'params', 'large_tip'))
    robot.dispense_into_vial = MagicMock(return_value=(0.0, None))
    robot.remove_pipet = MagicMock()
    North_Robot.dispense_from_vial_into_vial(robot, 'stock', 'dest', 0.5, liquid=liquid)
    assert robot.condition_tip.called is expected
    if expected:
        robot.condition_tip.assert_called_once_with('stock', volume_per_cycle_mL=0.5, specified_tip='large_tip')


# ---- dispense_into_wellplate: inputs checked before moving ----

@pytest.fixture
def well_robot(plate_robot):
    from types import SimpleNamespace
    del plate_robot.dispense_into_wellplate  # use the real unit operation
    plate_robot.PIPET_FLUID_VOLUME = 0.2
    plate_robot.select_wellplate_grid = MagicMock(return_value={i: f'well_{i}' for i in range(96)})
    plate_robot._get_optimized_parameters = MagicMock(return_value=SimpleNamespace(
        dispense_wait_time=0, blowout_vol=0, dispense_speed=10,
        overaspirate_vol=0, pre_asp_air_vol=0, post_asp_air_vol=0))
    for name in ['adjust_pump_speed', 'get_height_at_location', 'adjust_height_based_on_pipet_held',
                 'get_speed', 'pipet_dispense', '_track_dispense', '_well_container_id',
                 'save_robot_status']:
        setattr(plate_robot, name, MagicMock())
    plate_robot._contents_store = MagicMock(return_value=None)
    return plate_robot


def test_single_well_dispense_still_works(well_robot, caplog):
    caplog.set_level(logging.INFO)
    assert well_robot.dispense_into_wellplate([5], [0.2], liquid='water') is True
    well_robot.pipet_dispense.assert_called_once()
    assert 'Pipetting plate: 96 WELL PLATE (from the plate inventory)' in caplog.text
    assert 'tip is tracked' not in caplog.text
    assert well_robot.PIPET_FLUID_VOLUME == pytest.approx(0.0)


def test_multi_well_dispense_skips_zero(well_robot):
    well_robot.dispense_into_wellplate([0, 1, 2], [0.1, 0.0, 0.1])
    assert well_robot.pipet_dispense.call_count == 2


@pytest.mark.parametrize('wells, volumes, message', [
    ([0, 1], [0.1], '2 wells but 1 volumes'),
    ([0], [float('nan')], 'volume missing'),
    ([0], [-0.05], 'negative volume'),
    ([0], ['lots'], 'is not a volume in mL'),
    ([0], [0.35], r'well 0: 0.35 mL planned'),            # 96-well plate: 0.3 mL per well
    ([3, 3], [0.2, 0.15], r'well 3: 0.35 mL planned'),    # same well twice adds up
])
def test_wellplate_dispense_refuses_bad_inputs_before_moving(well_robot, wells, volumes, message):
    with pytest.raises(ValueError, match=message):
        well_robot.dispense_into_wellplate(wells, volumes)
    well_robot.c9.goto_xy_safe.assert_not_called()
    well_robot.pipet_dispense.assert_not_called()


def test_wellplate_dispense_counts_what_is_already_in_the_well(well_robot):
    well_robot._well_container_id = MagicMock(side_effect=lambda well: f'plate1:{well}')
    store = MagicMock()
    store.get.side_effect = lambda cid: {'volume_mL': 0.25} if cid == 'plate1:4' else None
    well_robot._contents_store = MagicMock(return_value=store)
    with pytest.raises(ValueError, match=r'0.1 mL planned \+ 0.25 mL already in it'):
        well_robot.dispense_into_wellplate([4], [0.1])
    well_robot.dispense_into_wellplate([5], [0.1])  # an empty well is fine


def test_wellplate_dispense_warns_but_runs_when_tip_holds_less(well_robot, caplog):
    caplog.set_level(logging.INFO)
    well_robot.PIPET_FLUID_VOLUME = 0.05
    well_robot.dispense_into_wellplate([0], [0.1], well_plate_type='96 WELL PLATE')
    assert 'tip is tracked as holding 0.050 mL' in caplog.text
    assert '(as given)' in caplog.text
    well_robot.pipet_dispense.assert_called_once()


# ---- vortex_vial: time and speed checked against robot_hardware.yaml before moving ----

def test_vortex_limits_are_in_robot_hardware_yaml():
    limits = yaml.safe_load(open('robot_state/robot_hardware.yaml', encoding='utf-8'))['vortex_limits']
    assert (limits['min_time_s'], limits['max_time_s'], limits['min_speed'], limits['max_speed']) == (1, 60, 1, 99)


def test_vortex_passes_values_to_the_motor_unchanged(robot, caplog):
    caplog.set_level(logging.INFO)
    robot.get_config_parameter.return_value = 0
    robot.vortex_vial('stock', 5)
    robot.c9.move_axis.assert_called_once_with(0, 1000 * 5 * 70, vel=70, accel=10000)
    assert 'for 5 s at speed 70' in caplog.text


@pytest.mark.parametrize('time_s, speed, message', [
    (0, 70, 'vortex_time 0 is outside 1-60 s'),
    (-5, 70, 'vortex_time -5 is outside'),
    (500, 70, 'vortex_time 500 is outside'),
    (float('nan'), 70, 'vortex_time nan is outside'),
    (None, 70, 'vortex_time None is not a number'),
    (5, 0, 'vortex_speed 0 is outside 1-99'),
    (5, 100, 'vortex_speed 100 is outside'),
])
def test_vortex_refuses_bad_time_or_speed_before_moving(robot, time_s, speed, message):
    with pytest.raises(ValueError, match=message):
        robot.vortex_vial('stock', time_s, vortex_speed=speed)
    robot.grab_vial.assert_not_called()
    robot.c9.move_axis.assert_not_called()

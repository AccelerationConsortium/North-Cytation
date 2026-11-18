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
        ('clamp', 0), ('main_8mL_rack', 47), ('main_8mL_rack', 46), ('main_8mL_rack', 45),
        ('clamp_2', 0), ('main_8mL_rack', 44), ('main_8mL_rack', 43)]
    assert robot.get_small_tip_safe_indices('main_8mL_rack') == [43, 44, 45, 46, 47]
    assert robot.get_small_tip_safe_indices('heater') == [2]
    assert robot.get_small_tip_safe_indices('large_vial_rack') == []


@pytest.fixture
def robot():
    robot = object.__new__(North_Robot)
    robot.VIAL_POSITIONS = yaml.safe_load(open('robot_state/vial_positions.yaml', encoding='utf-8'))
    robot.logger = logging.getLogger(__name__)
    robot.simulate = True
    robot.c9 = MagicMock()
    robot.vials = {
        'stock': dict(vial_name='stock', vial_volume=6.0, location='main_8mL_rack', location_index=5,
                      home_location='main_8mL_rack', home_location_index=5),
        'source': dict(vial_volume=3.0, location='main_8mL_rack', location_index=6,
                       home_location='main_8mL_rack', home_location_index=6),
        'dest': dict(vial_volume=0.0, location='main_8mL_rack', location_index=7,
                     home_location='main_8mL_rack', home_location_index=7),
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
    robot.dispense_into_wellplate = MagicMock()
    robot.remove_pipet = MagicMock()
    robot.condition_tip = MagicMock()
    robot.pause_after_error = MagicMock(side_effect=AssertionError('paused'))
    return robot


def test_wellplate_flow_groups_by_tip_and_uses_fresh_tip_per_vial(plate_robot):
    plate = pd.DataFrame({'stock': [0.02, 0.2, 0.01, 0.3], 'source': [0.1, 0.0, 0.0, 0.0]}, index=[0, 1, 2, 3])
    plate_robot.dispense_from_vials_into_wellplate(plate, liquid='water')
    doses = [(c.args[0], c.args[1], c.kwargs['specified_tip']) for c in plate_robot.aspirate_from_vial.call_args_list]
    # stock: small tip group first (largest dose first), then large tip group; then source
    assert doses == [('stock', 0.02, 'small_tip'), ('stock', 0.01, 'small_tip'),
                     ('stock', 0.3, 'large_tip'), ('stock', 0.2, 'large_tip'), ('source', 0.1, 'large_tip')]
    assert plate_robot.remove_pipet.call_count == 2  # Fresh tip for each vial
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
    # Only small-tip vials are staged; the emptiest-after-dispensing one gets the safest position (clamp)
    assert plate_robot.move_vial_to_location.call_args_list == [
        call('source', 'clamp', 0), call('stock', 'main_8mL_rack', 47)]
    # Unstaged vial goes home after its dispense; staged vials are returned at the end
    assert [c.args[0] for c in plate_robot.return_vial_home.call_args_list] == ['dest', 'source', 'stock']


def test_wellplate_flow_skips_staging_for_vials_already_safe_or_immovable(plate_robot):
    plate_robot.vials['stock']['location_index'] = 46  # Already small-tip safe
    plate_robot.is_vial_movable = MagicMock(side_effect=lambda vial: vial != 'source')
    plate = pd.DataFrame({'stock': [0.02], 'source': [0.03]}, index=[0])
    plate_robot.dispense_from_vials_into_wellplate(plate, liquid='water')
    plate_robot.move_vial_to_location.assert_not_called()


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

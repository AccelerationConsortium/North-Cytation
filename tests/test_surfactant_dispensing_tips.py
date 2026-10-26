"""The dispensing helper follows robot resolution through conditioning and dosing."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

from North_Safe import North_Robot


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dispensing():
    path = ROOT / 'workflows/surfactant_grid_adaptive_concentrations.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'validate_and_convert_recipe_volumes', 'condition_tip',
             'dispense_component_to_wellplate'}
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(pd=pd)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('liquid,volumes,expected', [
    ('water', [10, 180, 195], ['small_tip', 'small_tip', 'large_tip']),
    ('SDS', [10, 195, 210], ['small_tip', 'small_tip', 'large_tip']),
])
def test_groups_and_conditions_using_real_robot_resolution(dispensing, liquid, volumes, expected):
    robot = object.__new__(North_Robot)
    robot.logger = logging.getLogger(__name__)
    robot.PIPET_TIPS = yaml.safe_load((ROOT / 'robot_state/pipet_tips.yaml').read_text())
    controller = SimpleNamespace(nr_robot=robot, logger=robot.logger)
    events = []

    def aspirate(vial, volume, parameters=None, liquid=None, specified_tip=None):
        _, tip = robot.resolve_pipetting_setup(volume, liquid, parameters, specified_tip)
        events.append(('condition' if parameters is not None else 'dose', tip, volume))

    robot.aspirate_from_vial = Mock(side_effect=aspirate)
    robot.dispense_into_vial = Mock()
    robot.dispense_into_wellplate = Mock()
    frame = pd.DataFrame({'wellplate_index': range(len(volumes)), 'water_volume_ul': volumes})
    dispensing['dispense_component_to_wellplate'](
        controller, frame, 'source', liquid, 'water_volume_ul',
    )

    doses = {round(volume * 1000): tip for kind, tip, volume in events if kind == 'dose'}
    assert [doses[volume] for volume in volumes] == expected
    conditioned_tip = None
    conditioning_cycles = {}
    for kind, tip, volume in events:
        if kind == 'condition':
            conditioned_tip = tip
            conditioning_cycles[tip] = conditioning_cycles.get(tip, 0) + 1
        else:
            assert tip == conditioned_tip
    assert conditioning_cycles == {'small_tip': 5, 'large_tip': 5}
    assert robot.dispense_into_wellplate.call_count == len(volumes)


def test_unresolvable_dose_fails_before_any_aspiration(dispensing):
    robot = SimpleNamespace(resolve_pipetting_setup=Mock(side_effect=ValueError('no tip')),
                            aspirate_from_vial=Mock())
    controller = SimpleNamespace(nr_robot=robot, logger=Mock())
    frame = pd.DataFrame({'wellplate_index': [0], 'water_volume_ul': [1.5]})
    with pytest.raises(ValueError, match='no tip'):
        dispensing['dispense_component_to_wellplate'](
            controller, frame, 'water', 'water', 'water_volume_ul',
        )
    robot.aspirate_from_vial.assert_not_called()


def test_conditioning_failure_stops_dispensing(dispensing):
    robot = SimpleNamespace(
        resolve_pipetting_setup=Mock(return_value=(None, 'small_tip')),
        PIPET_TIPS={'small_tip': {'volume': .2}},
        aspirate_from_vial=Mock(side_effect=ValueError('conditioning failed')),
        dispense_into_wellplate=Mock(),
    )
    controller = SimpleNamespace(nr_robot=robot, logger=Mock())
    frame = pd.DataFrame({'wellplate_index': [0], 'water_volume_ul': [10]})
    with pytest.raises(ValueError, match='conditioning failed'):
        dispensing['dispense_component_to_wellplate'](
            controller, frame, 'water', 'water', 'water_volume_ul',
        )
    robot.dispense_into_wellplate.assert_not_called()

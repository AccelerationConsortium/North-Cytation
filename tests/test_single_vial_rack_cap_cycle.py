"""Check the single-vial route and cap handling without operating hardware."""

from pathlib import Path
import runpy
import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def cycle():
    root = Path(__file__).resolve().parents[1]
    lash = MagicMock(_workflow_should_continue=True)
    with patch.dict(sys.modules, {"master_usdl_coordinator": types.SimpleNamespace(
            Lash_E=MagicMock(return_value=lash))}):
        workflow = runpy.run_path(str(root / "tests" / "single_vial_rack_cap_cycle.py"))
    lash.workflow_config = {key: workflow[key] for key in workflow["_CONFIG_KEYS"]}
    robot = lash.nr_robot
    rows = pd.read_csv(workflow["INPUT_VIAL_STATUS_FILE"])
    robot.VIAL_DF = rows
    robot.VIAL_POSITIONS = {"main_8mL_rack": {"rack_size": 48, "rack_present": True}}
    robot.VIAL_POSITIONS.update({
        location: {"rack_size": size, "rack_present": True, "vials_movable": True,
                   "vial_positions_in_Locator": "configured_positions"}
        for location, size in (("photoreactor_array", 1), ("clamp_2", 1), ("heater", 12))
    })
    robot.GRIPPER_STATUS = None
    robot.HELD_PIPET_TYPE = None
    robot.get_vial_in_location.side_effect = lambda location, slot: (
        0 if (rows.at[0, "location"], rows.at[0, "location_index"]) == (location, slot) else None)
    events = []

    def move(name, location, slot):
        assert name == "cycle_vial"
        assert rows.at[0, "capped"] and robot.GRIPPER_STATUS is None
        assert location != "main_8mL_rack" or slot != 42
        events.append(("move", rows.at[0, "location"], int(rows.at[0, "location_index"]), location, slot))
        rows.at[0, "location"] = location
        rows.at[0, "location_index"] = slot

    def uncap():
        assert rows.at[0, "location"] == "clamp" and rows.at[0, "capped"]
        rows.at[0, "capped"] = False
        robot.GRIPPER_STATUS = "Cap"
        events.append("uncap")

    def recap():
        assert rows.at[0, "location"] == "clamp" and not rows.at[0, "capped"]
        assert robot.GRIPPER_STATUS == "Cap"
        rows.at[0, "capped"] = True
        robot.GRIPPER_STATUS = None
        events.append("recap")

    robot.move_vial_to_location.side_effect = move
    robot.uncap_clamp_vial.side_effect = uncap
    robot.recap_clamp_vial.side_effect = recap
    robot.move_home.side_effect = lambda: events.append(("home", robot.GRIPPER_STATUS))
    return workflow, lash, events


def test_single_vial_visits_rack_and_all_auxiliary_positions_and_wraps(cycle):
    workflow, lash, events = cycle
    lash.workflow_config["MAX_CYCLES"] = 2
    assert workflow["execute"](show_gui=False) == 2
    positions = [("main_8mL_rack", slot) for slot in range(48) if slot != 42]
    positions += [("photoreactor_array", 0), ("clamp_2", 0)]
    positions += [("heater", slot) for slot in range(12)]
    expected = [("home", None)]
    for _ in range(2):
        for source, destination in zip(positions, positions[1:] + positions[:1]):
            expected.extend([
                ("move", *source, "clamp", 0),
                "uncap", ("home", "Cap"), "recap",
                ("move", "clamp", 0, *destination),
                ("home", None),
            ])
    assert events == expected
    assert lash.nr_robot.VIAL_DF.at[0, "location_index"] == 0
    assert lash.nr_robot.VIAL_DF.at[0, "capped"]


@pytest.mark.parametrize("location", ["photoreactor_array", "clamp_2", "heater"])
def test_rejects_unavailable_station_before_test_motion(cycle, location):
    workflow, lash, events = cycle
    lash.nr_robot.VIAL_POSITIONS[location]["rack_present"] = False
    with pytest.raises(ValueError, match=location):
        workflow["execute"](show_gui=False)
    assert events == []


def test_route_uses_configured_heater_size(cycle):
    workflow, lash, _ = cycle
    lash.nr_robot.VIAL_POSITIONS["heater"]["rack_size"] = 4
    route = workflow["_build_route"](lash.nr_robot)
    assert [position for location, position in route if location == "heater"] == [0, 1, 2, 3]


@pytest.mark.parametrize("change", ["extra_vial", "wrong_slot", "uncapped", "tip", "held_vial"])
def test_rejects_invalid_setup_before_test_motion(cycle, change):
    workflow, lash, events = cycle
    robot = lash.nr_robot
    if change == "extra_vial":
        robot.VIAL_DF = pd.concat([robot.VIAL_DF, robot.VIAL_DF], ignore_index=True)
    elif change == "wrong_slot":
        robot.VIAL_DF.at[0, "location_index"] = 1
    elif change == "uncapped":
        robot.VIAL_DF.at[0, "capped"] = False
    elif change == "tip":
        robot.HELD_PIPET_TYPE = "large_tip"
    else:
        robot.GRIPPER_STATUS = "Vial"
    with pytest.raises(ValueError):
        workflow["execute"](show_gui=False)
    assert events == []


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_continuous_mode_stops_without_recovery_motion(cycle, error):
    workflow, lash, events = cycle
    lash.workflow_config["MAX_CYCLES"] = 0
    lash.nr_robot.recap_clamp_vial.side_effect = error
    with pytest.raises(error):
        workflow["execute"](show_gui=False)
    assert events[-1] == ("home", "Cap")
    assert lash.nr_robot.GRIPPER_STATUS == "Cap"
    assert lash.nr_robot.VIAL_DF.at[0, "location"] == "clamp"
    lash.nr_robot._perform_physical_cleanup.assert_not_called()

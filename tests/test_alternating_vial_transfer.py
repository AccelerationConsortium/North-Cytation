"""Exercise real clamp exchanges and movement safety without hardware."""
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from North_Safe import North_Robot
from pipetting_data.pipetting_parameters import PipettingParameters
from pipetting_data.pipetting_wizard import PipettingWizard


@pytest.fixture
def transfer_setup():
    lash = MagicMock(_workflow_should_continue=True)
    coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash))
    with patch.dict(sys.modules, {"master_usdl_coordinator": coordinator}):
        workflow = runpy.run_path(str(Path(__file__).with_name("alternating_vial_transfer.py")))
    lash.workflow_config = {key: workflow[key] for key in workflow["_CONFIG_KEYS"]}
    lash.workflow_config.update(SIMULATE=True, MAX_CYCLES=2)
    robot = lash.nr_robot
    robot.VIAL_DF = pd.DataFrame([
        dict(vial_index=i, vial_name=name, location="main_8mL_rack",
             location_index=slot, home_location="main_8mL_rack",
             home_location_index=slot, capped=True, cap_type="closed", vial_type="8_mL")
        for i, name, slot in ((0, "vial_a", 9), (1, "vial_b", 10))
    ])
    rows = robot.VIAL_DF
    robot.normalize_vial_index.side_effect = lambda vial: (
        int(rows.loc[rows.vial_name == vial, "vial_index"].iloc[0])
        if isinstance(vial, str) else vial)
    robot.get_vial_index_from_name.side_effect = robot.normalize_vial_index.side_effect
    robot.get_vial_info.side_effect = lambda vial, column: rows.at[robot.normalize_vial_index(vial), column]
    robot.get_vial_in_location.side_effect = lambda location, slot: next(
        iter(rows.loc[(rows.location == location) & (rows.location_index == slot), "vial_index"]), None)
    for name in ("is_vial_movable", "is_vial_pipetable", "return_vial_home",
                 "plan_vial_transfer", "resolve_pipetting_setup",
                 "is_vial_movement_safe_with_pipet", "is_location_safe_for_vial_movement_with_pipet"):
        method = getattr(North_Robot, name)
        getattr(robot, name).side_effect = lambda *args, _method=method, **kw: _method(robot, *args, **kw)
    robot.GRIPPER_STATUS = None
    robot.HELD_PIPET_TYPE = None

    def move(vial, location, location_index, **kwargs):
        if robot.HELD_PIPET_TYPE is not None:
            assert robot.is_vial_movement_safe_with_pipet(vial, location, location_index), (
                f"Cannot move vial {vial} to {location}[{location_index}] while holding pipet")
        index = robot.normalize_vial_index(vial)
        if robot.GRIPPER_STATUS == "Cap":
            assert rows.at[index, "location"] == "clamp"
            robot.recap_clamp_vial()
        rows.at[index, "location"] = location
        rows.at[index, "location_index"] = location_index

    def cap(capped, **kwargs):
        rows.at[robot.get_vial_in_location("clamp", 0), "capped"] = capped
        robot.GRIPPER_STATUS = None if capped else "Cap"

    def aspirate(vial, *args, **kwargs):
        robot.HELD_PIPET_TYPE = "large_tip"
        North_Robot._ensure_vial_accessible_for_pipetting(robot, vial)

    robot.move_vial_to_location.side_effect = move
    robot.uncap_clamp_vial.side_effect = lambda **kwargs: cap(False)
    robot.recap_clamp_vial.side_effect = lambda **kwargs: cap(True)
    robot.remove_pipet.side_effect = lambda: setattr(robot, "HELD_PIPET_TYPE", None)
    robot.PIPET_TIPS = {"large_tip": {"volume": 1.0}}
    robot._get_optimized_parameters.return_value = PipettingParameters()
    robot.aspirate_from_vial.side_effect = aspirate
    robot.dispense_into_vial.side_effect = lambda vial, *args, **kwargs: (
        North_Robot._ensure_vial_accessible_for_pipetting(robot, vial), None)[1]
    robot.dispense_from_vial_into_vial.side_effect = lambda *args, **kwargs: (
        North_Robot.dispense_from_vial_into_vial(robot, *args, **kwargs))
    with patch.object(PipettingWizard, "get_max_calibrated_volume", return_value=None):
        yield workflow, lash


@pytest.mark.parametrize("volume", [0.1, 2.5])
def test_closed_cap_transfers_stage_safe_homes_then_return(transfer_setup, volume):
    workflow, lash = transfer_setup
    lash.workflow_config["TRANSFER_VOLUME_ML"] = volume
    assert workflow["execute"](config=lash.workflow_config, show_gui=False) == 2
    robot = lash.nr_robot
    calls = robot.dispense_from_vial_into_vial.call_args_list
    assert [(c.kwargs["source_vial_name"], c.kwargs["dest_vial_name"]) for c in calls] == [
        ("vial_a", "vial_b"), ("vial_b", "vial_a")]
    assert list(robot.VIAL_DF.location_index) == [9, 10]
    assert list(robot.VIAL_DF.home_location_index) == [9, 10]
    assert robot.HELD_PIPET_TYPE is None
    assert robot.GRIPPER_STATUS is None


def test_interrupt_cleanup_keeps_safe_homes_until_tip_is_cleared(transfer_setup):
    workflow, lash = transfer_setup
    robot = lash.nr_robot

    def interrupted_dispense(vial, *args, **kwargs):
        raise KeyboardInterrupt

    def cleanup():
        # Reproduce the interrupted liquid-return path using the real clamp
        # helper: source is still clamped and tip contains its liquid.
        assert list(robot.VIAL_DF.home_location_index) == [0, 1]
        North_Robot._ensure_vial_accessible_for_pipetting(robot, "vial_a")
        robot.remove_pipet()
        robot.return_vial_home("vial_a")

    robot.dispense_into_vial.side_effect = interrupted_dispense
    robot._perform_physical_cleanup.side_effect = cleanup
    with pytest.raises(KeyboardInterrupt):
        workflow["execute"](config=lash.workflow_config, show_gui=False)
    lash.nr_track.reset_after_initialization.assert_called_once_with()
    robot._perform_physical_cleanup.assert_called_once_with()
    assert list(robot.VIAL_DF.home_location_index) == [9, 10]
    assert robot.HELD_PIPET_TYPE is None

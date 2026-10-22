"""Verify rack coverage, cap sequencing and cleanup without hardware."""
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest


@pytest.fixture
def rack_test():
    lash = MagicMock(_workflow_should_continue=True)
    root = Path(__file__).resolve().parents[1]
    with patch.dict(sys.modules, {"master_usdl_coordinator": types.SimpleNamespace(
            Lash_E=MagicMock(return_value=lash))}):
        workflow = runpy.run_path(str(root / "tests" / "rack_cap_cycle.py"))
    lash.workflow_config = {key: workflow[key] for key in workflow["_CONFIG_KEYS"]}
    lash.nr_robot.VIAL_DF = pd.read_csv(root / "status" / "rack_cap_cycle_vials.csv")
    lash.nr_robot.get_vial_in_location.return_value = None
    lash.nr_robot.HELD_PIPET_TYPE = None
    return workflow, lash


def test_visits_all_31_vials_in_order_for_each_sweep(rack_test):
    workflow, lash = rack_test
    lash.workflow_config["MAX_CYCLES"] = 2
    assert workflow["execute"](config=lash.workflow_config, show_gui=False) == 2
    expected = [call.get_vial_in_location("clamp", 0)]
    for _ in range(2):
        for position in range(31):
            name = f"vial_{position:02d}"
            expected.extend([
                call.move_vial_to_location(name, "clamp", 0),
                call.uncap_clamp_vial(),
                call.recap_clamp_vial(),
                call.return_vial_home(name),
            ])
    expected.append(call.move_home())
    assert lash.nr_robot.mock_calls == expected
    rows = lash.nr_robot.VIAL_DF
    assert set(rows.location_index) == set(range(31))
    assert all(rows.capped) and all(rows.cap_type == "closed")


@pytest.mark.parametrize("operation", [
    "move_vial_to_location", "uncap_clamp_vial", "recap_clamp_vial", "return_vial_home",
])
def test_interrupt_runs_shared_cleanup(rack_test, operation):
    workflow, lash = rack_test
    lash.workflow_config["MAX_CYCLES"] = 0
    getattr(lash.nr_robot, operation).side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        workflow["execute"](config=lash.workflow_config, show_gui=False)
    lash.nr_track.reset_after_initialization.assert_called_once_with()
    lash.nr_robot._perform_physical_cleanup.assert_called_once_with()


@pytest.mark.parametrize("bad_location", [31, 42, 48])
def test_rejects_unreachable_or_outside_rack_home_before_moving(rack_test, bad_location):
    workflow, lash = rack_test
    lash.nr_robot.VIAL_DF.at[0, "home_location_index"] = bad_location
    with pytest.raises(ValueError):
        workflow["execute"](config=lash.workflow_config, show_gui=False)
    lash.nr_robot.move_vial_to_location.assert_not_called()
    lash.nr_robot._perform_physical_cleanup.assert_not_called()

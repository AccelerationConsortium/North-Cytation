"""Lash_E.get_module_state reports each resource's saved state, read-only."""
from types import SimpleNamespace

import pandas as pd
import pytest

from master_usdl_coordinator import Lash_E


class FakeInventory:
    def __init__(self, **positions):
        self.data = {"positions": {"source_stack": [], "waste_stack": [], "pipetting_area": [],
                                   "cytation_tray": [], "gripper": None, **positions},
                     "pending_transfer": None}

    @property
    def positions(self):
        return self.data["positions"]

    def contents(self, location):
        value = self.positions[location]
        return list(value) if isinstance(value, list) else ([] if value is None else [value])


def vials(*rows):
    df = pd.DataFrame(rows, columns=["vial_index", "vial_name", "location", "location_index",
                                     "home_location", "home_location_index"])
    df.index = df["vial_index"].values
    return df


@pytest.fixture
def lash():
    lash = object.__new__(Lash_E)
    lash.nr_robot = SimpleNamespace(
        GRIPPER_STATUS=None, GRIPPER_VIAL_INDEX=None, HELD_PIPET_TYPE=None,
        PIPET_FLUID_VOLUME=0.0, PIPET_FLUID_VIAL_INDEX=None,
        VIAL_DF=vials((0, "stock", "main_8mL_rack", 5, "main_8mL_rack", 5),
                      (1, "sample", "main_8mL_rack", 6.0, "main_8mL_rack", 6)))
    lash.nr_track = SimpleNamespace(inventory=FakeInventory(), CURRENT_GRIPPER_LOCATION="origin")
    return lash


def test_everything_home_and_parked_is_ready(lash):
    state = lash.get_module_state()
    for name in ("robot_arm", "clamp", "clamp_2", "track"):
        assert state[name]["status"] == "ready", (name, state[name])
    assert state["cytation"]["status"] == "idle"
    assert all(entry["in_use_by"] is None for entry in state.values())


def test_robot_arm_reports_cap_tip_and_vials_not_home(lash):
    robot = lash.nr_robot
    robot.VIAL_DF.loc[0, ["location", "location_index"]] = ["clamp", 0]
    robot.GRIPPER_STATUS, robot.GRIPPER_VIAL_INDEX = "Cap", 0
    robot.HELD_PIPET_TYPE, robot.PIPET_FLUID_VOLUME, robot.PIPET_FLUID_VIAL_INDEX = "large_tip", 0.2, 0
    state = lash.get_module_state()
    arm = state["robot_arm"]
    assert arm["status"] == "not_ready"
    assert (arm["gripper_holding"], arm["gripper_item_id"]) == ("Cap", "stock")
    assert (arm["tip_type"], arm["tip_liquid_mL"], arm["tip_source_vial"]) == ("large_tip", 0.2, "stock")
    assert arm["vials_not_home"] == ["stock"]
    assert state["clamp"]["status"] == "not_ready" and state["clamp"]["vial"] == "stock"


def test_heater_lists_its_vials_but_running_state_is_unknown(lash):
    lash.nr_robot.VIAL_DF.loc[0, ["location", "location_index"]] = ["heater", 2]
    lash.nr_robot.VIAL_DF.loc[1, ["location", "location_index"]] = ["heater", 3]
    heater = lash.get_module_state()["heater"]
    assert heater["vials"] == ["stock", "sample"]
    assert heater["status"] == "unknown"


def test_track_not_ready_when_holding_unparked_or_mid_move(lash):
    lash.nr_track.inventory.positions["gripper"] = "plate-1"
    lash.nr_track.inventory.data["pending_transfer"] = {"object_id": "plate-1"}
    lash.nr_track.CURRENT_GRIPPER_LOCATION = "cytation_tray"
    track = lash.get_module_state()["track"]
    assert track["status"] == "not_ready"
    assert len(track["reasons"]) == 3 and track["gripper_plate"] == "plate-1"


def test_plate_on_deck_is_fine_but_plate_in_cytation_is_reported(lash):
    lash.nr_track.inventory.positions["pipetting_area"] = ["plate-1"]
    lash.nr_track.inventory.positions["cytation_tray"] = ["plate-2"]
    state = lash.get_module_state()
    assert state["track"]["status"] == "ready"
    assert state["cytation"]["status"] == "has_plate" and state["cytation"]["plate"] == "plate-2"


def test_untracked_resources_are_unknown_placeholders(lash):
    state = lash.get_module_state()
    for name in ("spin_coater", "carousel", "powder_dispenser", "reservoir_pumps", "camera"):
        assert (state[name]["status"], state[name]["source"]) == ("unknown", "not_tracked")


def test_works_without_track_or_vial_file(lash):
    lash.nr_track = None
    lash.nr_robot.VIAL_DF = None
    state = lash.get_module_state()
    assert state["track"]["status"] == "unknown"
    assert state["robot_arm"]["status"] == "ready"

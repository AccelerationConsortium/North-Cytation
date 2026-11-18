"""safe_stop is the one cleanup after any error or interrupt; the old helper delegates to it."""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pandas as pd
import pytest

from master_usdl_coordinator import Lash_E
from North_Safe import North_Robot
from workflows.helpers.interrupt_cleanup import cleanup_after_keyboard_interrupt


@pytest.fixture
def lash():
    lash = object.__new__(Lash_E)
    lash.simulate = True
    lash.logger = Mock()
    lash.temp_controller = Mock()
    lash.photoreactor = Mock()
    lash.spinner = Mock()
    lash.nr_track = Mock()
    lash.nr_track.inventory.positions = {"gripper": None}
    lash.nr_track.inventory.data = {"pending_transfer": None}
    lash.nr_robot = Mock(GRIPPER_STATUS=None, HELD_PIPET_TYPE=None)
    lash.nr_robot.return_all_vials_home.return_value = ([], [])
    return lash


def test_helper_delegates_to_safe_stop():
    lash = Mock()
    cleanup_after_keyboard_interrupt(lash)
    lash.safe_stop.assert_called_once_with("workflow failed or was interrupted")


def test_safe_stop_turns_off_instruments_then_track_then_robot_then_vials(lash):
    events = []
    lash.temp_controller.turn_off_heating.side_effect = lambda c: events.append(f"heater {c}")
    lash.photoreactor.turn_off_reactor_led.side_effect = lambda r: events.append(f"led {r}")
    lash.spinner.stop_spin.side_effect = lambda: events.append("spin")
    lash.nr_track.reset_after_initialization.side_effect = lambda: events.append("track")
    lash.nr_robot._perform_physical_cleanup.side_effect = lambda: events.append("robot")
    lash.nr_robot.return_all_vials_home.side_effect = lambda: events.append("vials") or ([], [])
    report = lash.safe_stop("test")
    # Heater channels and reactor numbers come from safe_stop in robot_hardware.yaml
    assert events == ["heater 0", "led 0", "spin", "track", "robot", "vials"]
    lash.photoreactor.turn_off_reactor_fan.assert_called_once_with(0)
    lash.temp_controller.turn_off_stirring.assert_called_once_with()
    assert report["failed"] == [] and report["left_for_operator"] == []


@pytest.mark.parametrize("error", [RuntimeError("track failed"), KeyboardInterrupt()])
def test_later_steps_still_run_when_one_fails(lash, error):
    lash.nr_track.reset_after_initialization.side_effect = error
    report = lash.safe_stop()
    lash.nr_robot._perform_physical_cleanup.assert_called_once_with()
    lash.nr_robot.return_all_vials_home.assert_called_once_with()
    assert report["failed"] == ["track to origin"]


def test_track_holding_a_plate_is_left_alone_and_reported(lash):
    lash.nr_track.inventory.positions["gripper"] = "plate-1"
    report = lash.safe_stop()
    lash.nr_track.reset_after_initialization.assert_not_called()
    assert any("plate-1" in item for item in report["left_for_operator"])


def test_vials_that_cannot_be_moved_are_reported(lash):
    lash.nr_robot.return_all_vials_home.return_value = (["a"], [("b", "not movable")])
    report = lash.safe_stop()
    assert report["vials_returned"] == ["a"]
    assert "vial b not home (not movable)" in report["left_for_operator"]


def test_missing_instruments_are_skipped(lash):
    lash.temp_controller = None
    lash.nr_track = None
    report = lash.safe_stop()
    lash.nr_robot._perform_physical_cleanup.assert_called_once_with()
    assert report["failed"] == []


def test_return_all_vials_home_moves_movable_vials_one_by_one():
    robot = object.__new__(North_Robot)
    robot.logger = logging.getLogger(__name__)
    robot.VIAL_DF = pd.DataFrame({
        "vial_name": ["home", "away", "uncapped", "broken"],
        "location": ["rack", "clamp", "clamp_2", "heater"], "location_index": [1, 0, 0, 2],
        "home_location": ["rack"] * 4, "home_location_index": [1, 2, 3, 4],
        "capped": [True, True, False, True]})
    robot.get_vial_info = lambda i, field: robot.VIAL_DF.loc[i, field]
    robot.is_vial_movable = lambda i: bool(robot.VIAL_DF.loc[i, "capped"])
    robot.return_vial_home = MagicMock(side_effect=lambda i: (_ for _ in ()).throw(RuntimeError("stuck")) if i == 3 else None)
    returned, left = robot.return_all_vials_home()
    assert returned == ["away"]
    assert left[0] == ("uncapped", "not movable") and left[1][0] == "broken"

"""Startup and end-of-workflow cleanup: return stray vials, warn, never fail; clear held-vial errors."""
import contextlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import yaml

from master_usdl_coordinator import Lash_E
from North_Safe import North_Robot


def robot_with_vials(rows=None):
    robot = object.__new__(North_Robot)
    robot.logger = Mock()
    robot.simulate = True
    robot.VIAL_DF = pd.DataFrame(rows or [
        {"vial_name": "home", "location": "rack", "location_index": 1, "home_location": "rack", "home_location_index": 1, "capped": True},
        {"vial_name": "away", "location": "clamp", "location_index": 0, "home_location": "rack", "home_location_index": 2, "capped": True},
    ])
    robot.get_vial_info = lambda i, field: robot.VIAL_DF.loc[i, field]
    robot.is_vial_movable = lambda i: bool(robot.VIAL_DF.loc[i, "capped"])
    robot.return_vial_home = Mock()
    robot.move_home = Mock()
    return robot


# --- Startup ---------------------------------------------------------------

def test_startup_returns_stray_vials_and_warns():
    robot = robot_with_vials()
    robot._return_stray_vials_at_startup()
    robot.return_vial_home.assert_called_once_with(1)
    robot.move_home.assert_called_once_with()
    assert "away" in robot.logger.warning.call_args.args[0]


def test_startup_with_everything_home_does_nothing():
    robot = robot_with_vials([{"vial_name": "home", "location": "rack", "location_index": 1,
                               "home_location": "rack", "home_location_index": 1, "capped": True}])
    robot._return_stray_vials_at_startup()
    robot.return_vial_home.assert_not_called()
    robot.logger.warning.assert_not_called()


def test_startup_never_fails_because_of_the_vial_check():
    robot = robot_with_vials()
    robot.return_vial_home.side_effect = None
    robot.get_vial_info = Mock(side_effect=KeyError("location"))
    robot._return_stray_vials_at_startup()  # no exception
    assert "could not check" in robot.logger.warning.call_args.args[0]


def test_startup_runs_the_vial_check_after_cleanup():
    robot = object.__new__(North_Robot)
    robot.logger, robot.c9 = Mock(), Mock()
    events = []
    for name in ("get_speed", "open_clamp", "zero_scale", "open_vial_gripper", "load_pumps", "home_robot_components"):
        setattr(robot, name, Mock())
    robot._perform_physical_cleanup = Mock(side_effect=lambda: events.append("cleanup"))
    robot._return_stray_vials_at_startup = Mock(side_effect=lambda: events.append("vials"))
    robot.reset_after_initialization()
    assert events == ["cleanup", "vials"]


# --- Held vial or cap with missing data ------------------------------------

def held(status, index):
    robot = object.__new__(North_Robot)
    robot.logger = Mock()
    robot.PIPET_FLUID_VIAL_INDEX, robot.PIPET_FLUID_VOLUME, robot.HELD_PIPET_TYPE = None, 0, None
    robot.GRIPPER_STATUS, robot.GRIPPER_VIAL_INDEX = status, index
    for name in ("recap_clamp_vial", "return_vial_home", "drop_off_vial", "move_home"):
        setattr(robot, name, Mock())
    robot.get_vial_info = Mock(return_value=None)
    return robot


@pytest.mark.parametrize("index", [None, 3])
def test_held_vial_without_number_or_home_gives_a_clear_error(index):
    robot = held("Vial", index)
    with pytest.raises(RuntimeError, match="Check the gripper by hand") as raised:
        robot._perform_physical_cleanup()
    assert raised.value.error_code == "HELD_VIAL_UNKNOWN"
    robot.drop_off_vial.assert_not_called()


def test_held_cap_without_number_recaps_and_leaves_return_to_vial_check():
    robot = held("Cap", None)
    robot._perform_physical_cleanup()
    robot.recap_clamp_vial.assert_called_once_with()
    robot.return_vial_home.assert_not_called()


def test_unknown_held_vial_is_not_retried_with_homing():
    robot = object.__new__(North_Robot)
    robot.logger, robot.c9 = Mock(), Mock()
    for name in ("get_speed", "open_clamp", "zero_scale", "open_vial_gripper", "load_pumps",
                 "home_robot_components", "_return_stray_vials_at_startup"):
        setattr(robot, name, Mock())
    error = RuntimeError("unknown held vial")
    error.error_code = "HELD_VIAL_UNKNOWN"
    robot._perform_physical_cleanup = Mock(side_effect=error)
    with pytest.raises(RuntimeError, match="unknown held vial"):
        robot.reset_after_initialization()
    robot.home_robot_components.assert_not_called()


# --- End of workflow --------------------------------------------------------

def lash_with(robot):
    lash = object.__new__(Lash_E)
    lash.logger, lash.nr_robot, lash.workflow_name = Mock(), robot, "demo_workflow"
    return lash


def test_end_of_workflow_returns_left_vials_with_a_warning_naming_the_workflow():
    lash = lash_with(robot_with_vials())
    returned, left = lash.tidy_after_workflow()
    assert returned == ["away"] and left == []
    message = lash.logger.warning.call_args.args[0]
    assert "demo_workflow" in message and "away" in message


def test_end_of_workflow_with_everything_home_logs_nothing():
    robot = robot_with_vials([{"vial_name": "home", "location": "rack", "location_index": 1,
                               "home_location": "rack", "home_location_index": 1, "capped": True}])
    lash = lash_with(robot)
    lash.tidy_after_workflow()
    lash.logger.warning.assert_not_called()


def test_end_of_workflow_never_raises():
    robot = robot_with_vials()
    robot.return_all_vials_home = Mock(side_effect=RuntimeError("no connection"))
    lash = lash_with(robot)
    assert lash.tidy_after_workflow() == ([], [])
    assert "could not check" in lash.logger.warning.call_args.args[0]


def test_live_runner_tidies_only_the_coordinator_from_this_job():
    from scheduler import live_runner
    stale, fresh = Mock(), Mock()
    fake_module = SimpleNamespace(last_coordinator=stale)
    with patch.dict(sys.modules, {"master_usdl_coordinator": fake_module}):
        fake_module.last_coordinator = None  # what run_job does before execute()
        live_runner._tidy_after_workflow()
        fake_module.last_coordinator = fresh  # the workflow created one
        live_runner._tidy_after_workflow()
    stale.tidy_after_workflow.assert_not_called()
    fresh.tidy_after_workflow.assert_called_once_with()


# --- Reset button -------------------------------------------------------------

def test_reset_refuses_a_held_vial_with_no_number(tmp_path):
    from scheduler import reset_runner
    status = tmp_path / "robot_status.yaml"
    status.write_text(yaml.safe_dump({"gripper_status": "Vial", "gripper_vial_index": None,
                                      "pipet_fluid_vial_index": None}))
    (tmp_path / "input.json").write_text(json.dumps({"vial_file": None, "robot_state": str(status), "input_hashes": {}}))
    factory = Mock()
    with patch.object(reset_runner, "fingerprint", return_value={}), \
            patch.object(reset_runner, "scheduler_live_lock", return_value=contextlib.nullcontext()):
        result = reset_runner.run_reset(tmp_path, factory=factory)
    assert not result["completed"] and "no vial number is saved" in result["exception"]
    factory.assert_not_called()

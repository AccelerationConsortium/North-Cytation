"""Fault injection: a simulated motor fault at chosen points of a real simulated run.

Runs hardware_tests/rack_cap_cycle.py (rack -> clamp -> uncap -> recap -> home) on two
vials with a real Lash_E in a scheduler simulation session, makes the simulated C9 fault
on move N, then checks the whole chain: the error is classified, safe_stop ran and
recorded what it could not do, and a restart ("power cycle") leaves the robot tidy.
"""
import os
import runpy
from pathlib import Path

import pandas as pd
import pytest

from experiment_tracking import experiment_run_logger
from experiment_tracking.error_records import classify
import master_usdl_coordinator
from scheduler import simulation_state as state
from scheduler.fault_injection import FAULT_ENV, inject_fault, requested_fault, simulated_c9

ROOT = Path(__file__).resolve().parents[1]
MOVES_IN_RUN = 18  # moves in a clean two-vial run (checked below)


@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path / "private")
    real_start = experiment_run_logger.start_run

    def private_start(*args, **kwargs):  # keep run tracking out of the real CSV
        tracker = real_start(*args, **kwargs)
        tracker.csv_path = str(tmp_path / "experiment_runs.csv")
        return tracker
    monkeypatch.setattr(experiment_run_logger, "start_run", private_start)
    vials = tmp_path / "two_vials.csv"
    pd.read_csv(ROOT / "status" / "rack_cap_cycle_vials.csv").iloc[:2].to_csv(vials, index=False)
    root = state.create_session([str(vials)])
    return root, next((root / "vials").glob("*.csv")), tmp_path


def run_rack_cycle(session, label):
    root, vials, tmp_path = session
    namespace = runpy.run_path(str(ROOT / "hardware_tests" / "rack_cap_cycle.py"))
    created = []

    def make(**kwargs):
        lash = master_usdl_coordinator.Lash_E(logging_folder=str(tmp_path / "logs"), **kwargs)
        created.append(lash)
        return lash
    namespace["execute"].__globals__["Lash_E"] = make
    with state.workflow_state(root, label):
        namespace["execute"](config={"SIMULATE": True, "INPUT_VIAL_STATUS_FILE": str(vials), "MAX_CYCLES": 1},
                             show_gui=False)
    return created[0]


def restart(session, label):
    """A fresh start (the 'power cycle'); returns the new robot after startup cleanup."""
    root, vials, tmp_path = session
    with state.workflow_state(root, label):
        lash = master_usdl_coordinator.Lash_E(vial_file=str(vials), simulate=True, initialize_biotek=False,
                                              show_gui=False, logging_folder=str(tmp_path / "logs"))
    return lash.nr_robot


def vials_away(vials):
    df = pd.read_csv(vials)
    return df[(df.location != df.home_location) | (df.location_index != df.home_location_index)].vial_name.tolist()


def test_no_fault_unless_asked(monkeypatch):
    monkeypatch.delenv(FAULT_ENV, raising=False)
    assert requested_fault() is None
    assert not isinstance(simulated_c9().fault_state, dict)  # a plain stand-in, as before


def test_bad_fault_setting_is_rejected(monkeypatch):
    monkeypatch.setenv(FAULT_ENV, "MOTOR:0")
    with pytest.raises(ValueError, match="C9_MOTOR_FAULT:3"):
        requested_fault()


def test_controller_stays_down_after_the_fault():
    with inject_fault("C9_MOTOR_FAULT", after_moves=2):
        c9 = simulated_c9()
    c9.goto_safe("a")  # move 1 is fine
    for command in (lambda: c9.move_z(1), lambda: c9.open_gripper(), lambda: c9.set_output(1, True)):
        with pytest.raises(RuntimeError) as raised:
            command()
        assert classify(raised.value)["code"] == "C9_MOTOR_FAULT"


def test_clean_run_has_the_expected_number_of_moves(session):
    with inject_fault("C9_MOTOR_FAULT", after_moves=10_000):  # counts moves, never faults
        lash = run_rack_cycle(session, "count")
    assert lash.nr_robot.c9.fault_state["moves"] == MOVES_IN_RUN


# Startup (1, 3), vial in gripper (5, 11), cap in gripper with vial in clamp (7, 14),
# vial on its way home (16), and the final move home (18)
@pytest.mark.parametrize("move", [1, 3, 5, 7, 11, 14, 16, 18])
def test_motor_fault_at_move_then_restart_leaves_robot_tidy(session, move):
    root, vials, _ = session
    with inject_fault("C9_MOTOR_FAULT", after_moves=move):
        with pytest.raises(RuntimeError) as raised:
            run_rack_cycle(session, f"fault_{move}")
    record = classify(raised.value)
    assert record["code"] == "C9_MOTOR_FAULT" and record["device_code"] == "IO/21"
    assert "fault_injection" not in record["step"]  # the step is in our code

    lash = master_usdl_coordinator.last_coordinator
    report = lash._run_tracker.safe_stop_report
    if report is not None:  # a fault during startup happens before the workflow's cleanup is set up
        assert report["error_code"] == "C9_MOTOR_FAULT"
        assert report["failed"]  # its moves fail too: the controller is down until the power cycle

    robot = restart(session, f"restart_{move}")
    assert robot.GRIPPER_STATUS is None and robot.HELD_PIPET_TYPE is None
    assert vials_away(vials) == []


def test_fault_just_after_uncapping_saves_which_vial_the_cap_belongs_to(session, monkeypatch):
    """Move 7 is the lift right after the first uncap. The saved status used to say
    "Cap" with no vial number there, and the next start crashed with KeyError: None."""
    import yaml
    from North_Safe import North_Robot
    root, vials, _ = session
    with inject_fault("C9_MOTOR_FAULT", after_moves=7):
        with pytest.raises(RuntimeError):
            run_rack_cycle(session, "fault_after_uncap")
    status = yaml.safe_load((root / "robot_status.yaml").read_text())
    assert status["gripper_status"] == "Cap" and status["gripper_vial_index"] == 0
    # The held-cap cleanup alone now returns the vial (no need for the startup vial return)
    monkeypatch.setattr(North_Robot, "_return_stray_vials_at_startup", lambda self: None)
    robot = restart(session, "restart_after_uncap")
    assert robot.GRIPPER_STATUS is None and vials_away(vials) == []

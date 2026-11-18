"""Error records add information only: classify known faults, never raise, never change behaviour."""
import csv
import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import yaml

from experiment_tracking import experiment_run_logger
from experiment_tracking.error_records import classify, load_codes
from master_usdl_coordinator import Lash_E
from North_Safe import North_Robot


class FakeC9Errors:
    """Shape of north_c9.C9Errors: .source, .error and a readable str."""
    def __init__(self, source, error, text):
        self.source, self.error, self.text = source, error, text

    def __str__(self):
        return self.text


def motor_fault():
    return RuntimeError(FakeC9Errors("IO", 21, "IO: MOTOR FAULT: A critical error has occured on a motor driver"))


def raised(error, wrapper=None):
    try:
        try:
            raise error
        except BaseException:
            if wrapper is None:
                raise
            raise wrapper
    except BaseException as final:
        return final


def test_codes_file_loads():
    assert {"C9_MOTOR_FAULT", "C9_HARD_ESTOP", "CYTATION_READ_ERROR", "PYTHON_EXCEPTION"} <= set(load_codes())


def test_c9_motor_fault_uses_the_sdk_code():
    record = classify(raised(motor_fault()))
    assert (record["code"], record["kind"], record["device_code"]) == ("C9_MOTOR_FAULT", "hardware_fault", "IO/21")
    assert record["slack"] is True and "Power-cycle" in record["what_to_do"]


def test_motor_fault_hidden_by_our_wrapper_is_still_found():
    record = classify(raised(motor_fault(), RuntimeError("Substock preparation failed: SDS_dilution_2")))
    assert record["code"] == "C9_MOTOR_FAULT" and "MOTOR FAULT" in record["message"]


def test_motor_fault_only_in_message_text_is_found():
    error = raised(RuntimeError("Failed to initialize measurement protocol: IO: MOTOR FAULT: restart"))
    assert classify(error)["code"] == "C9_MOTOR_FAULT"


@pytest.mark.parametrize("error, code, slack", [
    (RuntimeError(FakeC9Errors("IO", 8, "IO: SYSTEM ERROR: Hard E-Stop triggered")), "C9_HARD_ESTOP", False),
    (KeyboardInterrupt(), "OPERATOR_STOP", False),
    (KeyError(None), "PYTHON_EXCEPTION", True),
    (RuntimeError("Plate read error encountered (Gen5: lid missing)"), "CYTATION_READ_ERROR", True),
    (ValueError("something new"), "UNKNOWN", True),
])
def test_known_errors_and_unknown(error, code, slack):
    record = classify(raised(error))
    assert record["code"] == code and record["slack"] is slack


def test_code_set_by_our_code_wins():
    error = EOFError("EOF when reading a line")
    error.error_code = "CYTATION_NOT_CONNECTED"
    assert classify(raised(error))["kind"] == "setup_problem"


def test_classify_never_raises_on_a_broken_codes_list():
    assert classify(raised(motor_fault()), codes={"BROKEN": None})["code"] == "UNKNOWN"


def test_failed_run_writes_yaml_record_and_csv_columns(tmp_path):
    logger = logging.getLogger("test_error_records")
    tracker = experiment_run_logger.start_run("demo", False, "demo.log", logger, str(tmp_path / "logs"))
    tracker.csv_path = str(tmp_path / "experiment_runs.csv")
    tracker.safe_stop_report = {"done": ["arm home"], "failed": [], "left_for_operator": ["vial a not home"]}
    error = raised(motor_fault())
    with patch.object(sys, "last_value", error, create=True), \
            patch.object(sys, "last_traceback", error.__traceback__, create=True):
        experiment_run_logger.finalize_run(tracker)
    row = list(csv.DictReader(open(tmp_path / "experiment_runs.csv")))[-1]
    assert (row["status"], row["error_kind"], row["error_code"]) == ("failed", "hardware_fault", "C9_MOTOR_FAULT")
    record = yaml.safe_load(open(tmp_path / "logs" / "experiment_run_errors" / f"{tracker.run_id}.yaml"))
    assert record["run_id"] == tracker.run_id and record["device_code"] == "IO/21"
    assert record["state_after_error"]["left_for_operator"] == ["vial a not home"]
    assert (tmp_path / "logs" / "experiment_run_errors" / f"{tracker.run_id}.txt").exists()


def test_completed_run_leaves_error_columns_blank(tmp_path):
    tracker = experiment_run_logger.start_run("demo", True, "demo.log", logging.getLogger("t"), str(tmp_path / "logs"))
    tracker.csv_path = str(tmp_path / "experiment_runs.csv")
    experiment_run_logger.finalize_run(tracker)
    row = list(csv.DictReader(open(tmp_path / "experiment_runs.csv")))[-1]
    assert row["error_kind"] == "" and row["error_code"] == ""


def robot_with_pump(failure):
    robot = object.__new__(North_Robot)
    robot.logger = Mock()
    robot.simulate = True
    robot.c9 = Mock()
    robot.c9.dispense_ml.side_effect = failure
    robot.c9.aspirate_ml.side_effect = failure
    robot.get_config_parameter = Mock(return_value=3000)
    return robot


@pytest.mark.parametrize("position, warns", [(1499, False), (1400, True)])
def test_a_dispense_past_empty_empties_the_pump(position, warns):
    """The pump-position check that replaced the old catch-all (merged from the lab PC, 2026-11-18):
    0.5 mL from a 1 mL pump is 1500 counts; one count short is rounding (the '-1' case), more warns."""
    robot = robot_with_pump(None)
    robot.simulate = False
    robot.c9.pumps = {0: {"pos": position, "volume": 1.0}}
    robot.pipet_dispense(0.5)
    robot.c9.move_pump.assert_called_once_with(0, 0)
    robot.c9.dispense_ml.assert_not_called()
    assert robot.logger.warning.called is warns


def test_aspirate_fallback_is_unchanged_but_logs_the_real_error():
    robot = robot_with_pump(RuntimeError("Pump volume too full to aspirate"))
    robot.pipet_aspirate(0.5, wait_time=0)
    robot.c9.move_pump.assert_called_once_with(0, 3000)
    assert "too full" in robot.logger.warning.call_args_list[0].args[0]


@pytest.fixture
def live_lash():
    lash = object.__new__(Lash_E)
    lash.simulate = False
    lash.logger = Mock()
    lash.temp_controller = lash.photoreactor = lash.spinner = lash.nr_track = None
    lash.nr_robot = Mock(GRIPPER_STATUS=None, HELD_PIPET_TYPE=None)
    lash.nr_robot.return_all_vials_home.return_value = ([], [])
    return lash


def safe_stop_during(lash, error):
    sender = Mock()
    with patch.dict(sys.modules, {"slack_agent": SimpleNamespace(send_slack_message=sender)}):
        try:
            raise error
        except BaseException:
            report = lash.safe_stop("test")
    return report, sender


def test_safe_stop_skips_slack_after_ctrl_c(live_lash):
    report, sender = safe_stop_during(live_lash, KeyboardInterrupt())
    sender.assert_not_called()
    assert report["error_code"] == "OPERATOR_STOP"


def test_safe_stop_tells_operator_what_to_do_after_a_motor_fault(live_lash):
    report, sender = safe_stop_during(live_lash, motor_fault())
    assert "C9_MOTOR_FAULT" in sender.call_args.args[0] and "Power-cycle" in sender.call_args.args[0]


def test_safe_stop_called_directly_still_sends_slack(live_lash):
    sender = Mock()
    with patch.dict(sys.modules, {"slack_agent": SimpleNamespace(send_slack_message=sender)}):
        report = live_lash.safe_stop("manual")
    sender.assert_called_once()
    assert "error_code" not in report

"""Execute one approved live job; no simulation state or execution deadline."""

import argparse
import csv
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import traceback

import yaml

from scheduler.simulation_runner import Diagnostics
from scheduler import simulation_state as state
from scheduler.simulation_state import SESSION_ENV
from scheduler.interrupt import watch_for_interrupt
from scheduler.workflow_modules import workflow_module
from experiment_tracking.experiment_run_logger import EXPERIMENT_ID_ENV


def fingerprint(paths):
    return {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}


def changed_inputs(hashes):
    current = fingerprint([path for path in hashes if Path(path).is_file()])
    return [Path(path).name for path in hashes if current.get(path) != hashes[path]]


def reviewed_live_state(job):
    """Use the reviewed start state, then each successful live handoff."""
    expected = job["state_hashes"]
    previous_path = job.get("previous_job_result")
    if previous_path:
        previous = json.loads(Path(previous_path).read_text(encoding="utf-8"))
        if (previous.get("completed") is not True or previous.get("handoff_ok") is not True
                or previous.get("exception")
                or any(record["level"] in {"ERROR", "CRITICAL"} for record in previous["records"])):
            raise ValueError("Previous live job did not return a successful handoff.")
        expected = previous["state_hashes"]
        if set(expected) != set(job["state_hashes"]):
            raise ValueError("Previous live handoff does not cover this queue's inventory.")
    return expected


def live_state_issues(job):
    root = Path(job["state_root"])
    robot = yaml.safe_load((root / "robot_status.yaml").read_text())
    track = yaml.safe_load((root / "track_status.yaml").read_text())
    issues = []
    for key in ("gripper_status", "gripper_vial_index", "held_pipet_type", "pipet_fluid_vial_index"):
        if robot[key] is not None and robot[key] != "None":
            issues.append(f"Robot requires review: {key}={robot[key]}")
    volume = robot["pipet_fluid_volume"]
    if volume is not None and float(volume) != 0:
        issues.append("Liquid remains in the pipet")
    position = track.get("active_wellplate_position")
    from scheduler.plate_state import operation_issues
    issues.extend(operation_issues(track))
    from workflows.helpers.two_run_plate import ready_for_second_run, read_marker
    if position is not None and position != "None" and not ready_for_second_run(track, read_marker(root)):
        issues.append(f"Wellplate remains at {position}")
    occupied = set()
    for filename in set(job["vial_files"]):
        with Path(filename).open(newline="", encoding="utf-8-sig") as stream:
            for record in csv.DictReader(stream):
                key = record["location"], Decimal(record["location_index"])
                if key in occupied:
                    issues.append(f"Current vial-position conflict at {key}")
                occupied.add(key)
    return issues


@contextmanager
def scheduler_live_lock():
    """Prevent overlapping scheduler children; unrelated launchers are not covered."""
    import msvcrt
    path = Path(__file__).resolve().parent / "state" / "live.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("Another scheduler live job already owns the hardware launch lock.") from error
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class Capture(logging.Filter):
    def __init__(self, diagnostics):
        super().__init__()
        self.diagnostics = diagnostics

    def filter(self, record):
        self.diagnostics.handle(record)
        return True


def notify(message):
    """Best-effort Slack update for the external experiment planner; never fails the job."""
    logger = logging.getLogger("my_logger")
    try:
        from slack_agent import safe_send_slack_message
        sent = safe_send_slack_message(message)
    except Exception as error:
        logger.warning(f"Slack update not sent ({error}): {message}")
        return
    if not sent:
        logger.warning(f"Slack update not sent: {message}")


def run_job(folder):
    folder = Path(folder)
    result = {"completed": False, "handoff_ok": False, "records": [], "exception": None, "interrupted": False}
    diagnostics = Diagnostics()
    logger = logging.getLogger("my_logger")
    capture = Capture(diagnostics)
    root_logger = logging.getLogger()
    logger.addFilter(capture)
    root_logger.addHandler(diagnostics)
    experiment_id = None
    try:
        if os.environ.get(SESSION_ENV):
            raise RuntimeError("Live execution cannot use simulation state.")
        job = json.loads((folder / "input.json").read_text(encoding="utf-8"))
        config = job["config"]
        if config["SIMULATE"] is not False:
            raise ValueError("Live config must explicitly select SIMULATE=False.")
        name = job["workflow"]
        experiment_id = result["experiment_id"] = job["experiment_id"]
        os.environ[EXPERIMENT_ID_ENV] = experiment_id
        module_name = workflow_module(name, state.REPO_ROOT)
        with scheduler_live_lock():
            changed = changed_inputs(job["input_hashes"])
            if changed:
                raise ValueError("Reviewed inputs changed; simulate again before running: " + ", ".join(changed))
            if "state_hashes" in job:
                expected = reviewed_live_state(job)
                actual = fingerprint(expected)
                changed = [path for path in expected if actual[path] != expected[path]]
                if changed:
                    raise ValueError("Live inventory changed since review or the previous handoff: " + "; ".join(changed))
            issues = live_state_issues(job)
            if issues:
                raise ValueError("Live starting state requires review: " + "; ".join(issues))
            module = importlib.import_module(module_name)
            notify(f"North experiment {experiment_id} started | workflow={name}")
            with watch_for_interrupt(folder):
                module.execute(config=config, show_gui=False)
            issues = live_state_issues(job)
            # This job already ran with the code it imported at start; the next job's
            # pre-run check blocks it until the queue is simulated again.
            changed = changed_inputs(job["input_hashes"])
            if changed:
                logger.warning("Files edited during this run (simulate again before the next run): " + ", ".join(changed))
            result["completed"] = True
            for issue in issues:
                result["records"].append({"level": "ERROR", "message": f"Live handoff blocked: {issue}"})
            result["handoff_ok"] = not issues and not any(record["level"] in {"ERROR", "CRITICAL"} for record in diagnostics.records)
            if result["handoff_ok"] and "state_hashes" in job:
                result["state_hashes"] = fingerprint(job["state_hashes"])
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                result["log_file"] = handler.baseFilename
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        result["interrupted"] = isinstance(error, KeyboardInterrupt)
        result["exception"] = traceback.format_exc()
        result["records"].append({"level": "ERROR", "message": result["exception"]})
        error_line = traceback.format_exception_only(type(error), error)[-1].strip()
    finally:
        os.environ.pop(EXPERIMENT_ID_ENV, None)
        result["records"] = diagnostics.records + result["records"]
        logger.removeFilter(capture)
        root_logger.removeHandler(diagnostics)
        temp = folder / "result.tmp"
        temp.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temp.replace(folder / "result.json")
    # Sent after result.json exists so the planner can read the outcome; jobs whose
    # input.json could not be read have no experiment_id and are not reported.
    if experiment_id is not None:
        if result["completed"]:
            notify(f"North experiment {experiment_id} finished | workflow={name}")
        elif not result["interrupted"]:
            notify(f"North experiment {experiment_id} failed | workflow={name} | error={error_line}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-folder", required=True)
    args = parser.parse_args()
    result = run_job(args.job_folder)
    if result["exception"]:
        if result["interrupted"]:
            # Preserve Ctrl+C classification in experiment_run_logger's atexit hook.
            raise KeyboardInterrupt(result["exception"])
        # Re-raise so this becomes a genuinely unhandled exception in this process -
        # run_job() caught it to still write result.json, but experiment_run_logger's
        # atexit hook needs an unhandled exception to tag the run "failed" instead of "completed".
        raise RuntimeError(result["exception"])
    raise SystemExit(0 if result["completed"] and result["handoff_ok"] else 1)


if __name__ == "__main__":
    main()

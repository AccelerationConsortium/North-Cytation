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
from scheduler.simulation_state import SESSION_ENV


def fingerprint(paths):
    return {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}


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
    position = track["active_wellplate_position"]
    if position is not None and position != "None":
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


def run_job(folder):
    folder = Path(folder)
    result = {"completed": False, "handoff_ok": False, "records": [], "exception": None}
    diagnostics = Diagnostics()
    logger = logging.getLogger("my_logger")
    capture = Capture(diagnostics)
    root_logger = logging.getLogger()
    logger.addFilter(capture)
    root_logger.addHandler(diagnostics)
    try:
        if os.environ.get(SESSION_ENV):
            raise RuntimeError("Live execution cannot use simulation state.")
        job = json.loads((folder / "input.json").read_text(encoding="utf-8"))
        if fingerprint(job["input_hashes"]) != job["input_hashes"]:
            raise ValueError("Reviewed inputs changed; simulate again before running.")
        config = job["config"]
        if config["SIMULATE"] is not False:
            raise ValueError("Live config must explicitly select SIMULATE=False.")
        name = job["workflow"]
        if Path(name).name != name or name.startswith("_"):
            raise ValueError("Invalid workflow name")
        with scheduler_live_lock():
            issues = live_state_issues(job)
            if issues:
                raise ValueError("Live starting state requires review: " + "; ".join(issues))
            module = importlib.import_module(f"workflows.{name}")
            module.execute(config=config, show_gui=False)
            issues = live_state_issues(job)
        result["completed"] = True
        for issue in issues:
            result["records"].append({"level": "ERROR", "message": f"Live handoff blocked: {issue}"})
        result["handoff_ok"] = not issues and not any(record["level"] in {"ERROR", "CRITICAL"} for record in diagnostics.records)
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                result["log_file"] = handler.baseFilename
    except (Exception, KeyboardInterrupt, SystemExit):
        result["exception"] = traceback.format_exc()
        result["records"].append({"level": "ERROR", "message": result["exception"]})
    finally:
        result["records"] = diagnostics.records + result["records"]
        logger.removeFilter(capture)
        root_logger.removeHandler(diagnostics)
        temp = folder / "result.tmp"
        temp.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temp.replace(folder / "result.json")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-folder", required=True)
    args = parser.parse_args()
    result = run_job(args.job_folder)
    if result["exception"]:
        # Re-raise so this becomes a genuinely unhandled exception in this process -
        # run_job() caught it to still write result.json, but experiment_run_logger's
        # atexit hook needs an unhandled exception to tag the run "failed" instead of "completed".
        raise RuntimeError(result["exception"])
    raise SystemExit(0 if result["completed"] and result["handoff_ok"] else 1)


if __name__ == "__main__":
    main()
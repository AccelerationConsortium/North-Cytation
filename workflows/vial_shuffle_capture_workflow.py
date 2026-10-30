"""Collect indexed C110 photographs and matching robot-tracked vial CSVs.

Run from the repository root: python workflows/vial_shuffle_capture_workflow.py
Configuration uses the normal workflow YAML/operator review. See
docs/vial_shuffle_capture.md for output and future verification contracts.
"""
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys
import time

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.c110_camera import capture_photo
from workflows._vial_shuffle import (
    RACK, apply_transfers, available_slots, plan_shuffle, positions, validate_layout,
)

SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/sample_input_vials.csv"
ITERATIONS = 10
TARGET_EXPERIMENT_TIME_HOURS = 0.0  # Target cycle spacing includes work; 0 runs without pacing
MIN_VIALS_MOVED = 3
MAX_VIALS_MOVED = 8
MIN_OFF_RACK = 0
MAX_OFF_RACK = 14
CAMERA_FOLDER = r"C:\Users\Imaging Controller\Desktop\camera-control"
CAPTURE_TIMEOUT_S = 20
CAPTURE_ATTEMPTS = 3
SETTLE_SECONDS = 1.0
RANDOM_SEED = None

_WORKFLOW_NAME = Path(__file__).stem
_CONFIG_KEYS = [
    "SIMULATE", "INPUT_VIAL_STATUS_FILE", "ITERATIONS", "TARGET_EXPERIMENT_TIME_HOURS", "MIN_VIALS_MOVED",
    "MAX_VIALS_MOVED", "MIN_OFF_RACK", "MAX_OFF_RACK", "CAMERA_FOLDER",
    "CAPTURE_TIMEOUT_S", "CAPTURE_ATTEMPTS", "SETTLE_SECONDS", "RANDOM_SEED",
]


def _validate(config, robot):
    for key in ("ITERATIONS", "MIN_VIALS_MOVED", "MAX_VIALS_MOVED", "CAPTURE_ATTEMPTS"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("MIN_OFF_RACK", "MAX_OFF_RACK"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for low, high in (("MIN_VIALS_MOVED", "MAX_VIALS_MOVED"), ("MIN_OFF_RACK", "MAX_OFF_RACK")):
        if config[low] > config[high]:
            raise ValueError(f"{low} must not exceed {high}")
    for key in ("CAPTURE_TIMEOUT_S", "SETTLE_SECONDS", "TARGET_EXPERIMENT_TIME_HOURS"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number")
    if config["CAPTURE_TIMEOUT_S"] == 0:
        raise ValueError("CAPTURE_TIMEOUT_S must be positive")
    if not math.isfinite(config["TARGET_EXPERIMENT_TIME_HOURS"] * 3600):
        raise ValueError("TARGET_EXPERIMENT_TIME_HOURS is too large to convert to seconds")
    if robot.HELD_PIPET_TYPE is not None or robot.GRIPPER_STATUS is not None:
        raise ValueError("Shuffle requires an empty gripper and no held pipette")
    slots = available_slots(robot)
    names = validate_layout(robot.VIAL_DF, slots)
    current = positions(robot.VIAL_DF)
    fixed = {slot for name, slot in current.items() if name not in names}
    off_capacity = sum(slot[0] != RACK and slot not in fixed for slot in slots)
    if config["MIN_OFF_RACK"] > min(len(names), off_capacity):
        raise ValueError("MIN_OFF_RACK exceeds available vials or off-rack capacity")
    return slots, names


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _capture_pair(robot, directory, index, config, capture):
    """Stage the pair before publishing it; the manifest counts only complete pairs."""
    stem = f"{index:06d}"
    photo = directory / f"{stem}.jpg"
    csv = directory / f"{stem}.csv"
    staging = directory / ".staging"
    staging.mkdir(exist_ok=True)
    staged_photo, staged_csv = staging / photo.name, staging / csv.name
    robot.move_home()
    if not config["SIMULATE"]:
        time.sleep(config["SETTLE_SECONDS"])
    snapshot = robot.VIAL_DF.copy(deep=True)
    for attempt in range(1, config["CAPTURE_ATTEMPTS"] + 1):
        try:
            capture(staged_photo, config["CAMERA_FOLDER"], config["CAPTURE_TIMEOUT_S"])
            if not staged_photo.is_file() or staged_photo.stat().st_size == 0:
                raise OSError("Camera did not save a nonempty image")
            break
        except Exception:
            staged_photo.unlink(missing_ok=True)
            robot.logger.warning("Capture attempt %s failed", attempt, exc_info=True)
            if attempt == config["CAPTURE_ATTEMPTS"]:
                raise
    try:
        snapshot.to_csv(staged_csv, index=False)
        staged_csv.replace(csv)
        staged_photo.replace(photo)
    except BaseException:
        photo.unlink(missing_ok=True)
        csv.unlink(missing_ok=True)
        raise
    return photo, csv


def _simulated_capture(path, _camera_folder, _timeout):
    """Copy a labeled fixture; simulation never opens a camera or image library."""
    fixture = Path(__file__).with_name("_simulated_vial_capture.jpg")
    path.write_bytes(fixture.read_bytes())


def _notify(lash_e, summary):
    message = (
        f"Vial shuffle capture {summary['status']} | "
        f"images={summary['images_saved']}/{summary['requested']} | "
        f"original positions restored={summary['restored']}"
    )
    if summary["verification_enabled"]:
        message += (f" | vision matches={summary['matches']}/{summary['verified']}"
                    f" | mismatches={summary['mismatches']}"
                    f" | verification errors={summary['verification_errors']}")
    else:
        message += " | vision verification disabled"
    lash_e.logger.info(message)
    if lash_e.simulate:
        return
    try:
        import slack_agent
        if not slack_agent.safe_send_slack_message(message):
            lash_e.logger.warning("End-of-run Slack summary was not delivered")
    except Exception:
        lash_e.logger.warning("End-of-run Slack summary failed", exc_info=True)


def _wait_between_samples(lash_e, seconds):
    if seconds <= 0:
        return
    if lash_e.simulate:
        lash_e.logger.info("SIMULATION: Would wait %.3f seconds before next shuffle", seconds)
        return
    lash_e.logger.info("Waiting %.3f seconds before next shuffle (robot at home)", seconds)
    # Short sleeps keep long waits interruptible without changing the total delay.
    deadline = time.monotonic() + seconds
    remaining = seconds
    while remaining > 0:
        duration = min(60.0, remaining)
        time.sleep(duration)
        remaining = deadline - time.monotonic()


def run_collection(lash_e, config, directory, *, capture=None, verifier=None):
    """Run with an initialized coordinator; verifier(photo, csv) returns a result dict.

    A future verifier must return {"matched": bool, ...JSON-serializable evidence}.
    No detection service is contacted unless a verifier is explicitly supplied.
    """
    robot = lash_e.nr_robot
    slots, names = _validate(config, robot)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    capture = capture or (_simulated_capture if lash_e.simulate else capture_photo)
    original = {name: slot for name, slot in positions(robot.VIAL_DF).items() if name in names}
    robot.VIAL_DF.to_csv(directory / "initial_state.csv", index=False)
    _write_json(directory / "config.json", config)
    rng = random.Random(config["RANDOM_SEED"])
    interval_seconds = config["TARGET_EXPERIMENT_TIME_HOURS"] * 3600 / config["ITERATIONS"]
    summary = dict(status="running", requested=config["ITERATIONS"], images_saved=0,
                   target_experiment_time_hours=config["TARGET_EXPERIMENT_TIME_HOURS"],
                   sample_interval_seconds=interval_seconds,
                   verification_enabled=verifier is not None, verified=0, matches=0,
                   mismatches=0, verification_errors=0, restored=False, pairs=[])
    started = time.monotonic()
    try:
        for index in range(1, config["ITERATIONS"] + 1):
            cycle_started = time.monotonic()
            targets = plan_shuffle(
                positions(robot.VIAL_DF), names, slots, config["MIN_VIALS_MOVED"],
                config["MAX_VIALS_MOVED"], config["MIN_OFF_RACK"],
                config["MAX_OFF_RACK"], rng,
            )
            transfers = apply_transfers(robot, targets, slots)
            photo, csv = _capture_pair(robot, directory, index, config, capture)
            summary["images_saved"] += 1
            pair = dict(index=index, photo=photo.name, vial_csv=csv.name,
                        vials_moved=len(targets), transfers=transfers,
                        off_rack=sum(positions(robot.VIAL_DF)[name][0] != RACK for name in names))
            summary["pairs"].append(pair)
            _write_json(directory / "summary.json", summary)
            if verifier is not None:
                try:
                    result = verifier(photo, csv)
                    if not isinstance(result, dict) or type(result.get("matched")) is not bool:
                        raise ValueError("Verifier must return a mapping with a boolean 'matched'")
                    _write_json(directory / f"{index:06d}.verification.json", result)
                    pair["matched"] = result["matched"]
                    summary["verified"] += 1
                    summary["matches" if result["matched"] else "mismatches"] += 1
                except Exception as error:
                    summary["verification_errors"] += 1
                    pair["verification_error"] = str(error)
                    robot.logger.warning("Verification failed for %s", photo.name, exc_info=True)
            cycle_finished = time.monotonic()
            pair["cycle_seconds"] = cycle_finished - cycle_started
            # Live deadlines stay anchored to the first shuffle, avoiding drift.
            # Simulation skips waits, so only estimate each cycle's remaining gap.
            deadline = (cycle_started if lash_e.simulate else started + (index - 1) * interval_seconds) + interval_seconds
            remaining = max(0.0, deadline - cycle_finished) if interval_seconds else 0.0
            pair["schedule_lag_seconds"] = max(0.0, cycle_finished - deadline) if interval_seconds else 0.0
            pair["wait_seconds"] = remaining if index < config["ITERATIONS"] else 0.0
            if pair["schedule_lag_seconds"] and not lash_e.simulate:
                robot.logger.warning(
                    "Sample %s finished %.3f seconds behind target; continuing without dropping samples",
                    index, pair["schedule_lag_seconds"],
                )
            _write_json(directory / "summary.json", summary)
            if index < config["ITERATIONS"]:
                # Include manifest-save time when calculating the actual live wait.
                wait = max(0.0, deadline - time.monotonic()) if not lash_e.simulate and interval_seconds else remaining
                _wait_between_samples(lash_e, wait)
        apply_transfers(robot, original, slots)
        robot.move_home()
        summary["restored"] = True
        summary["status"] = "completed"
    except BaseException as error:
        summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary["error"] = str(error)
        # A failed motion may leave a vial held or its position uncertain. Preserve
        # tracked state for recovery instead of attempting blind return movements.
        robot.logger.exception("Collection stopped; inspect tracked state before recovery")
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        try:
            robot.VIAL_DF.to_csv(directory / "final_state.csv", index=False)
            _write_json(directory / "summary.json", summary)
        finally:
            _notify(lash_e, summary)
    return summary


def execute(config=None, show_gui=True, *, verifier=None):
    from master_usdl_coordinator import Lash_E

    lash_e = Lash_E(
        initialize_biotek=False, initialize_track=False,
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
    )
    if not lash_e._workflow_should_continue:
        return None
    confirmed = lash_e.workflow_config
    if confirmed["SIMULATE"] != lash_e.simulate:
        raise ValueError("Workflow simulation mode differs from controller mode")
    if Path(confirmed["INPUT_VIAL_STATUS_FILE"]).resolve() != Path(lash_e.nr_robot.VIAL_FILE).resolve():
        raise ValueError("Vial input changed during review; restart the workflow")
    mode = "SIMULATION" if lash_e.simulate else "LIVE"
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{mode}"
    directory = _ROOT / "output" / _WORKFLOW_NAME / run_name
    return run_collection(lash_e, confirmed, directory, verifier=verifier)


if __name__ == "__main__":
    execute()

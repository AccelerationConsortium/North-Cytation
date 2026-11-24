"""Collect indexed C110 photographs and matching robot-tracked vial CSVs.

Run from the repository root: python workflows/vial_shuffle_capture_workflow.py
Configuration uses the normal workflow YAML/operator review. See
docs/workflows/vial_shuffle_capture.md for output and future verification contracts.
"""
from datetime import datetime, timedelta
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
from workflows.helpers._vial_shuffle import (
    RACK as _RACK, apply_transfers, available_slots, plan_shuffle, positions, validate_layout,
)

SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/sample_input_vials.csv"
ITERATIONS = 10
TARGET_EXPERIMENT_TIME_HOURS = 0.0  # Target cycle spacing includes work; 0 runs without pacing
RUN_UNTIL = ""  # "08:00" (next time the clock reads that) or "2026-11-22 08:00"; "" = run ITERATIONS cycles
MAX_RUN_HOURS = 0.0  # Stop starting cycles after this long; 0 = no limit
RESTORE_RESERVE_MINUTES = 10.0  # Extra time kept free before the deadline for restoring positions
MIN_VIALS_MOVED = 3
MAX_VIALS_MOVED = 8
MIN_OFF_RACK = 0
MAX_OFF_RACK = 14
CAMERA_FOLDER = r"C:\Users\Imaging Controller\Desktop\camera-control"
CAPTURE_TIMEOUT_S = 20
CAPTURE_ATTEMPTS = 3
SETTLE_SECONDS = 1.0
RANDOM_SEED = None

# Simulations exercise the shuffle and timing sequence without writing photos or manifests.
_SAVE_SIMULATION_OUTPUT = False

_WORKFLOW_NAME = Path(__file__).stem
_CONFIG_KEYS = [
    "SIMULATE", "INPUT_VIAL_STATUS_FILE", "ITERATIONS", "TARGET_EXPERIMENT_TIME_HOURS",
    "RUN_UNTIL", "MAX_RUN_HOURS", "RESTORE_RESERVE_MINUTES", "MIN_VIALS_MOVED",
    "MAX_VIALS_MOVED", "MIN_OFF_RACK", "MAX_OFF_RACK", "CAMERA_FOLDER",
    "CAPTURE_TIMEOUT_S", "CAPTURE_ATTEMPTS", "SETTLE_SECONDS", "RANDOM_SEED",
]

_CONFIG_DOCS = {
    "SIMULATE": {"description": "Run the shuffle sequence without controlling hardware or saving capture outputs.", "unit": None, "affects": []},
    "INPUT_VIAL_STATUS_FILE": {"description": "CSV describing the vial positions to shuffle.", "unit": None, "affects": ["vials"], "shares": "vial_layout"},
    "ITERATIONS": {"description": "Number of shuffle-and-capture cycles to run (the most cycles when RUN_UNTIL or MAX_RUN_HOURS is set).", "unit": None, "affects": ["duration"]},
    "RUN_UNTIL": {"description": "Run as many cycles as fit before this time, then restore positions: a clock time such as 08:00 (its next occurrence) or a date and time. Empty runs ITERATIONS cycles.", "unit": None, "affects": ["duration"]},
    "MAX_RUN_HOURS": {"description": "Run as many cycles as fit in this many hours, then restore positions. 0 means no limit.", "unit": "h", "affects": ["duration"]},
    "RESTORE_RESERVE_MINUTES": {"description": "Extra time kept free before the deadline, beyond the estimated restore time.", "unit": "min", "affects": ["duration"]},
    "TARGET_EXPERIMENT_TIME_HOURS": {"description": "Target spacing for the complete sequence of shuffle cycles.", "unit": "h", "affects": ["duration"]},
    "MIN_VIALS_MOVED": {"description": "Minimum number of vials moved during each shuffle.", "unit": None, "affects": ["duration", "vials"]},
    "MAX_VIALS_MOVED": {"description": "Maximum number of vials moved during each shuffle.", "unit": None, "affects": ["duration", "vials"]},
    "MIN_OFF_RACK": {"description": "Minimum number of shuffled vials placed outside the rack.", "unit": None, "affects": ["duration", "vials"]},
    "MAX_OFF_RACK": {"description": "Maximum number of shuffled vials placed outside the rack.", "unit": None, "affects": ["duration", "vials"]},
    "CAMERA_FOLDER": {"description": "Folder path passed to the camera capture tool.", "unit": None, "affects": []},
    "CAPTURE_TIMEOUT_S": {"description": "Maximum time allowed for one camera capture attempt.", "unit": "s", "affects": ["duration"]},
    "CAPTURE_ATTEMPTS": {"description": "Maximum number of attempts for each camera capture.", "unit": None, "affects": ["duration"]},
    "SETTLE_SECONDS": {"description": "Live-run delay after the robot moves home and before capture.", "unit": "s", "affects": ["duration"]},
    "RANDOM_SEED": {"description": "Seed used to select shuffle destinations reproducibly.", "unit": None, "affects": ["vials"]},
}

_PRODUCT_VIALS = []


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
    for key in ("MAX_RUN_HOURS", "RESTORE_RESERVE_MINUTES"):
        value = config.get(key, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number")
    if not isinstance(config.get("RUN_UNTIL", ""), str):
        raise ValueError("RUN_UNTIL must be text such as 08:00 or 2026-11-22 08:00")
    if config.get("RUN_UNTIL", "").strip():
        _deadline_from(config["RUN_UNTIL"], datetime.now())
    if _deadline_mode(config) and config["TARGET_EXPERIMENT_TIME_HOURS"]:
        raise ValueError("Use RUN_UNTIL/MAX_RUN_HOURS or TARGET_EXPERIMENT_TIME_HOURS, not both")
    if robot.HELD_PIPET_TYPE is not None or robot.GRIPPER_STATUS is not None:
        raise ValueError("Shuffle requires an empty gripper and no held pipette")
    slots = available_slots(robot)
    names = validate_layout(robot.VIAL_DF, slots)
    current = positions(robot.VIAL_DF)
    fixed = {slot for name, slot in current.items() if name not in names}
    off_capacity = sum(slot[0] != _RACK and slot not in fixed for slot in slots)
    if config["MIN_OFF_RACK"] > min(len(names), off_capacity):
        raise ValueError("MIN_OFF_RACK exceeds available vials or off-rack capacity")
    return slots, names


def _deadline_mode(config):
    return bool(config.get("RUN_UNTIL", "").strip()) or bool(config.get("MAX_RUN_HOURS", 0))


def _deadline_from(text, now):
    """The wall-clock deadline RUN_UNTIL names: "HH:MM" is its next occurrence after now."""
    text = text.strip()
    try:
        clock = datetime.strptime(text, "%H:%M")
    except ValueError:
        try:
            when = datetime.fromisoformat(text)
        except ValueError:
            raise ValueError("RUN_UNTIL must look like 08:00 or 2026-11-22 08:00") from None
        if when.tzinfo is not None:
            when = when.astimezone().replace(tzinfo=None)
        if when <= now:
            raise ValueError(f"RUN_UNTIL {text} is already past")
        return when
    when = now.replace(hour=clock.hour, minute=clock.minute, second=0, microsecond=0)
    return when if when > now else when + timedelta(days=1)


def _seconds_left(config, started_wall, started, now):
    """Seconds until the earliest of RUN_UNTIL and MAX_RUN_HOURS (monotonic clock), or None."""
    limits = []
    if config.get("RUN_UNTIL", "").strip():
        limits.append((_deadline_from(config["RUN_UNTIL"], started_wall) - started_wall).total_seconds())
    if config.get("MAX_RUN_HOURS", 0):
        limits.append(config["MAX_RUN_HOURS"] * 3600)
    return min(limits) - (now - started) if limits else None


def _restore_estimate(pairs, original, current):
    """Seconds to put displaced vials back: their number times the slowest seen seconds per moved vial."""
    displaced = sum(current.get(name) != slot for name, slot in original.items())
    per_vial = max((pair["move_seconds"] / pair["vials_moved"] for pair in pairs
                    if pair.get("vials_moved") and pair.get("move_seconds") is not None), default=0.0)
    return displaced * per_vial


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    # Windows denies the replace while another process (e.g. a virus scan) briefly holds the target.
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


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
    requested = "until deadline" if "run_until" in summary else summary["requested"]
    message = (
        f"Vial shuffle capture {summary['status']} | "
        f"images={summary['images_saved']}/{requested} | "
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
    save = not lash_e.simulate or _SAVE_SIMULATION_OUTPUT
    directory = Path(directory)
    capture = capture or (_simulated_capture if lash_e.simulate else capture_photo)
    original = {name: slot for name, slot in positions(robot.VIAL_DF).items() if name in names}
    if save:
        directory.mkdir(parents=True, exist_ok=False)
        robot.VIAL_DF.to_csv(directory / "initial_state.csv", index=False)
        _write_json(directory / "config.json", config)
    rng = random.Random(config["RANDOM_SEED"])
    interval_seconds = config["TARGET_EXPERIMENT_TIME_HOURS"] * 3600 / config["ITERATIONS"]
    deadline_mode = _deadline_mode(config)
    # Simulation has no real clock to run against: check the sequence with a few cycles.
    cycles = min(config["ITERATIONS"], 3) if deadline_mode and lash_e.simulate else config["ITERATIONS"]
    started_wall = datetime.now()
    summary = dict(status="running", requested=config["ITERATIONS"], images_saved=0,
                   target_experiment_time_hours=config["TARGET_EXPERIMENT_TIME_HOURS"],
                   sample_interval_seconds=interval_seconds,
                   verification_enabled=verifier is not None, verified=0, matches=0,
                   mismatches=0, verification_errors=0, restored=False, pairs=[])
    if deadline_mode:
        summary.update(run_until=config.get("RUN_UNTIL", ""), max_run_hours=config.get("MAX_RUN_HOURS", 0),
                       stopped_for_deadline=False)
    started = time.monotonic()
    try:
        for index in range(1, cycles + 1):
            if deadline_mode and summary["pairs"] and not lash_e.simulate:
                now = time.monotonic()
                left = _seconds_left(config, started_wall, started, now)
                next_cycle = max(pair["cycle_seconds"] for pair in summary["pairs"])
                restore = _restore_estimate(summary["pairs"], original, positions(robot.VIAL_DF))
                reserve = config.get("RESTORE_RESERVE_MINUTES", 0) * 60
                if next_cycle + restore + reserve > left:
                    summary["stopped_for_deadline"] = True
                    robot.logger.info(
                        "Stopping after %s cycles: next cycle (~%.0f s) plus restoring (~%.0f s) and %.0f s "
                        "reserve would pass the deadline (%.0f s left)", index - 1, next_cycle, restore,
                        reserve, left)
                    break
            cycle_started = time.monotonic()
            targets = plan_shuffle(
                positions(robot.VIAL_DF), names, slots, config["MIN_VIALS_MOVED"],
                config["MAX_VIALS_MOVED"], config["MIN_OFF_RACK"],
                config["MAX_OFF_RACK"], rng,
            )
            transfers = apply_transfers(robot, targets, slots)
            moved_at = time.monotonic()
            if save:
                photo, csv = _capture_pair(robot, directory, index, config, capture)
                summary["images_saved"] += 1
            else:
                robot.move_home()
                photo = csv = None
            pair = dict(index=index, photo=photo and photo.name, vial_csv=csv and csv.name,
                        vials_moved=len(targets), transfers=transfers,
                        off_rack=sum(positions(robot.VIAL_DF)[name][0] != _RACK for name in names),
                        move_seconds=moved_at - cycle_started)
            summary["pairs"].append(pair)
            if verifier is not None and save:
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
            pair["wait_seconds"] = remaining if index < cycles else 0.0
            if pair["schedule_lag_seconds"] and not lash_e.simulate:
                robot.logger.warning(
                    "Sample %s finished %.3f seconds behind target; continuing without dropping samples",
                    index, pair["schedule_lag_seconds"],
                )
            if not lash_e.simulate:
                # Progress record for long live runs; simulation writes once at the end.
                _write_json(directory / "summary.json", summary)
            if index < cycles:
                # Include manifest-save time when calculating the actual live wait.
                wait = max(0.0, deadline - time.monotonic()) if not lash_e.simulate and interval_seconds else remaining
                _wait_between_samples(lash_e, wait)
        restore_started = time.monotonic()
        apply_transfers(robot, original, slots)
        robot.move_home()
        summary["restore_seconds"] = time.monotonic() - restore_started
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
            if save:
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

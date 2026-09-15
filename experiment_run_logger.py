"""Automatic per-run experiment logging, wired in by Lash_E.__init__ - no workflow file changes needed.

Design summary (see /memories/session/plan.md for the full history of this design):
- Exactly one fully-populated row is written to logs/experiment_runs.csv per run, at the end.
- Crash detection uses an invisible marker file (logs/.active_runs/{run_id}.txt), never a
  partially-filled CSV row, so every CSV column is always meaningful.
- No global hooks (no sys.excepthook replacement, no SIGINT handler) - only atexit plus a
  passive read of sys.last_value/sys.last_traceback, which Python itself already populates
  for any uncaught exception (including Ctrl-C's KeyboardInterrupt).
"""
import atexit
import csv
import ctypes
import getpass
import inspect
import os
import sys
import traceback
import uuid
from datetime import datetime

# Files that are internal plumbing, never the actual workflow - skip over them when walking
# the call stack to find whichever file really constructed Lash_E(...).
_INTERNAL_CALLER_FILENAMES = {"experiment_run_logger.py", "master_usdl_coordinator.py"}

CSV_FIELDS = [
    "run_id", "workflow_name", "simulate", "datetime_started", "datetime_stopped",
    "status", "duration_sec", "log_filename", "pid", "user",
]

DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


class RunTracker:
    def __init__(self, run_id, workflow_name, simulate, log_filename, datetime_started, marker_path,
                 csv_path, error_dir, logger):
        self.run_id = run_id
        self.workflow_name = workflow_name
        self.simulate = simulate
        self.log_filename = log_filename
        self.datetime_started = datetime_started
        self.marker_path = marker_path
        self.csv_path = csv_path
        self.error_dir = error_dir
        self.logger = logger
        self.pid = os.getpid()
        self.user = getpass.getuser()
        self.cancelled = False


def _pid_is_running(pid):
    """Best-effort Windows PID liveness check using stdlib ctypes (no psutil dependency)."""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    try:
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    except (AttributeError, OSError, ValueError):
        # Not on Windows, or pid unparseable - assume dead so a stuck marker still gets healed.
        return False


def _detect_caller_filename():
    """Walk the call stack to find the first frame outside this module and
    master_usdl_coordinator.py - i.e. the actual file whose code constructed Lash_E(...),
    regardless of how the overall process was launched (direct script, GUI, batch driver, etc.).
    Falls back to the process entry point if something unexpected prevents stack inspection.
    """
    try:
        for frame_info in inspect.stack():
            if os.path.basename(frame_info.filename) not in _INTERNAL_CALLER_FILENAMES:
                return os.path.basename(frame_info.filename)
    except Exception:
        pass
    return os.path.basename(sys.argv[0])


def _read_marker(marker_path):
    fields = {}
    with open(marker_path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                key, _, value = line.rstrip("\n").partition("=")
                fields[key] = value
    return fields


def _write_csv_row(csv_path, row):
    file_exists = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _heal_orphaned_runs(csv_path, marker_dir):
    if not os.path.isdir(marker_dir):
        return
    now = datetime.now()
    for marker_name in os.listdir(marker_dir):
        marker_path = os.path.join(marker_dir, marker_name)
        try:
            fields = _read_marker(marker_path)
        except OSError:
            continue

        pid = fields.get("pid")
        if pid and _pid_is_running(pid):
            # Still legitimately running in another process - not a crash, leave it alone.
            continue

        started = fields.get("datetime_started", now.strftime(DATETIME_FMT))
        try:
            started_dt = datetime.strptime(started, DATETIME_FMT)
            duration_sec = (now - started_dt).total_seconds()
        except ValueError:
            duration_sec = ""

        row = {
            "run_id": fields.get("run_id", os.path.splitext(marker_name)[0]),
            "workflow_name": fields.get("workflow_name", ""),
            "simulate": fields.get("simulate", ""),
            "datetime_started": started,
            "datetime_stopped": now.strftime(DATETIME_FMT),
            "status": "interrupted_unknown_crash",
            "duration_sec": duration_sec,
            "log_filename": fields.get("log_filename", ""),
            "pid": pid or "",
            "user": fields.get("user", ""),
        }
        _write_csv_row(csv_path, row)
        try:
            os.remove(marker_path)
        except OSError:
            pass


def start_run(workflow_name, simulate, log_filename, logger, logging_folder):
    """Register this run for tracking. Call once, early in Lash_E.__init__.

    logging_folder must be the same folder Lash_E already writes its per-run .log file to,
    so experiment_runs.csv/marker/error files always land next to it regardless of the
    caller's current working directory (both are resolved the same way relative to cwd).
    """
    csv_path = os.path.join(logging_folder, "experiment_runs.csv")
    marker_dir = os.path.join(logging_folder, ".active_runs")
    error_dir = os.path.join(logging_folder, "experiment_run_errors")

    # Clear any stale exception state left by an earlier, unrelated Lash_E instantiation in
    # this same process, so it can't be misattributed to this run at finalize time.
    for attr in ("last_value", "last_type", "last_traceback", "last_exc"):
        if hasattr(sys, attr):
            try:
                delattr(sys, attr)
            except AttributeError:
                pass

    _heal_orphaned_runs(csv_path, marker_dir)

    run_id = uuid.uuid4().hex[:8]
    workflow_name = workflow_name or _detect_caller_filename()
    datetime_started = datetime.now().strftime(DATETIME_FMT)

    os.makedirs(marker_dir, exist_ok=True)
    marker_path = os.path.join(marker_dir, f"{run_id}.txt")
    tracker = RunTracker(run_id, workflow_name, simulate, log_filename, datetime_started, marker_path,
                         csv_path, error_dir, logger)

    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(f"run_id={run_id}\n")
        f.write(f"datetime_started={datetime_started}\n")
        f.write(f"workflow_name={workflow_name}\n")
        f.write(f"simulate={simulate}\n")
        f.write(f"pid={tracker.pid}\n")
        f.write(f"user={tracker.user}\n")
        f.write(f"log_filename={log_filename}\n")
        f.flush()
        os.fsync(f.fileno())

    atexit.register(finalize_run, tracker)
    return tracker


def cancel_run(tracker):
    """Call when a run is aborted before it really started (e.g. GUI cancel) - no CSV row wanted."""
    tracker.cancelled = True
    try:
        os.remove(tracker.marker_path)
    except OSError:
        pass


def finalize_run(tracker):
    """Registered with atexit - writes this run's single CSV row and clears its marker."""
    if tracker.cancelled:
        return
    if not os.path.exists(tracker.marker_path):
        # Already finalized (e.g. called more than once) - the marker's removal is the guard.
        return
    try:
        last_exc = getattr(sys, "last_exc", None)  # Python 3.12+
        last_value = last_exc if last_exc is not None else getattr(sys, "last_value", None)

        if last_value is None:
            status = "completed"
        elif isinstance(last_value, KeyboardInterrupt):
            status = "interrupted_by_user"
        else:
            status = "failed"
            os.makedirs(tracker.error_dir, exist_ok=True)
            last_tb = getattr(sys, "last_traceback", None)
            tb_text = "".join(traceback.format_exception(type(last_value), last_value, last_tb))
            with open(os.path.join(tracker.error_dir, f"{tracker.run_id}.txt"), "w", encoding="utf-8") as f:
                f.write(tb_text)

        datetime_stopped = datetime.now()
        started_dt = datetime.strptime(tracker.datetime_started, DATETIME_FMT)
        duration_sec = (datetime_stopped - started_dt).total_seconds()

        row = {
            "run_id": tracker.run_id,
            "workflow_name": tracker.workflow_name,
            "simulate": tracker.simulate,
            "datetime_started": tracker.datetime_started,
            "datetime_stopped": datetime_stopped.strftime(DATETIME_FMT),
            "status": status,
            "duration_sec": duration_sec,
            "log_filename": tracker.log_filename,
            "pid": tracker.pid,
            "user": tracker.user,
        }
        # Remove the marker before writing so a concurrent/duplicate finalize call sees it gone.
        try:
            os.remove(tracker.marker_path)
        except OSError:
            pass
        _write_csv_row(tracker.csv_path, row)
    except Exception as e:
        # Never let logging failures crash the interpreter shutdown - just surface them.
        tracker.logger.error(f"experiment_run_logger failed to finalize run {tracker.run_id}: {e}")

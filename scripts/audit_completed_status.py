"""One-off audit: check 'completed' non-simulated, non-backfilled CSV rows against their
full log file content for signs the run didn't actually complete cleanly."""
import csv
import os
from datetime import datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LOGS_DIR = os.path.join(ROOT, "logs")
CSV_PATH = os.path.join(ROOT, "experiment_tracking", "experiment_runs.csv")

MARKERS = [
    "Traceback (most recent call last)", "KeyboardInterrupt", "logger.exception",
    "Workflow failed", "CRITICAL", "paused for", "pause_after_error", "SIMULATE flag updated",
]
CLEAN_WORDS = ["complete", "completed successfully", "finished"]

DT_FMT = "%Y-%m-%d %H:%M:%S"


def parse_log_timestamp(line):
    try:
        ts_part = line.split(" - ", 1)[0]
        ts_part = ts_part.split(",")[0]
        return datetime.strptime(ts_part, DT_FMT)
    except Exception:
        return None


def main():
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    filtered = [r for r in rows if r["status"] == "completed" and r["simulate"] == "False"
                and not r["status"].endswith("_backfilled")]

    print(f"Total completed/non-simulated/non-backfilled rows: {len(filtered)}")
    print("-" * 120)

    flagged_marker = 0
    flagged_gap = 0
    flagged_not_clean = 0
    missing_logs = 0

    for row in filtered:
        log_path = os.path.join(LOGS_DIR, row["log_filename"])
        if not os.path.isfile(log_path):
            missing_logs += 1
            print(f"{row['run_id']} | {row['workflow_name']} | {row['log_filename']} | LOG MISSING")
            continue

        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        found = [m for m in MARKERS if m.lower() in text.lower()]
        lines = [l for l in text.splitlines() if l.strip()]
        last_line = lines[-1] if lines else ""
        last_ts = parse_log_timestamp(last_line)

        gap = "n/a"
        if last_ts is not None:
            try:
                stopped = datetime.strptime(row["datetime_stopped"], DT_FMT)
                gap = round((stopped - last_ts).total_seconds(), 1)
            except Exception:
                gap = "n/a"

        clean = any(w in last_line.lower() for w in CLEAN_WORDS)

        if found:
            flagged_marker += 1
        if isinstance(gap, (int, float)) and gap > 120:
            flagged_gap += 1
        if not clean:
            flagged_not_clean += 1

        print(f"{row['run_id']} | {row['workflow_name']} | {row['log_filename']}")
        print(f"  markers={found or 'none'} | gap_sec={gap} | clean_ending={clean}")
        print(f"  last_line={last_line[:150]}")

    print("-" * 120)
    print(f"Checked: {len(filtered)}")
    print(f"Missing logs: {missing_logs}")
    print(f"Rows with any marker found: {flagged_marker}")
    print(f"Rows with gap_sec > 120: {flagged_gap}")
    print(f"Rows without a clean-looking ending line: {flagged_not_clean}")


if __name__ == "__main__":
    main()

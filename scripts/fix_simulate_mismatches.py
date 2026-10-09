"""Correct experiment_runs.csv `simulate` values using each run's own log file.

Root cause: before the ordering fix in master_usdl_coordinator.py, start_run() could be
called with the pre-GUI-confirmation `simulate` value, and the GUI's actual correction
(logged as "SIMULATE flag updated: <old> -> <new>") happened afterward. So some historical
CSV rows record the wrong (pre-correction) simulate state. The log file itself is the
authoritative source - it's the last "SIMULATE flag updated" line (if any) in the run's own
log file, not the filename's "_simulate" suffix (which is also derived from the possibly-wrong
pre-correction value) and not the CSV's existing value.

Rows with a "_backfilled" status are skipped entirely - those were heuristically reconstructed
from old logs with no live tracking, and are out of scope here per explicit user direction.
"""
import csv
import os
import re

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiment_tracking", "experiment_runs.csv")

UPDATE_RE = re.compile(r"SIMULATE flag updated:\s*(True|False)\s*->\s*(True|False)")


def true_simulate_value(log_filename):
    """Return the final 'True'/'False' string from the last SIMULATE flag update in the log,
    or None if the log is missing or never had its simulate flag corrected."""
    path = os.path.join(LOGS_DIR, log_filename)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    matches = UPDATE_RE.findall(text)
    if not matches:
        return None
    return matches[-1][1]  # final value from the last toggle


def main():
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    changed = []
    missing_logs = 0
    for row in rows:
        if row["status"].endswith("_backfilled"):
            continue
        corrected = true_simulate_value(row["log_filename"])
        if corrected is None:
            continue
        if row["simulate"] != corrected:
            changed.append((row["run_id"], row["workflow_name"], row["log_filename"], row["simulate"], corrected))
            row["simulate"] = corrected

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rows corrected: {len(changed)}")
    for run_id, workflow_name, log_filename, old, new in changed:
        print(f"  {run_id} ({workflow_name}, {log_filename}): {old} -> {new}")


if __name__ == "__main__":
    main()

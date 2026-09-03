"""One-time backfill of historical experiment logs into logs/experiment_runs.csv.

Scope (per investigation + explicit user decisions):
- Only modern-format, non-simulated `experiment_log*.log` files (has per-line timestamps).
- Old no-timestamp `.txt`-era logs (~271 files) are excluded entirely - nothing reliable is
  recoverable from them (no stop time, no duration, no workflow name).
- workflow_name resolution, in priority order:
    1. "Detected workflow name: X" line (GUI-launched runs) -> use X directly.
    2. "Loaded vial data from <path>" line -> map the CSV's basename to a workflow via
       CSV_TO_WORKFLOW below. A few CSV names are shared by more than one workflow script;
       per user direction, we just pick the most likely single owner rather than leaving
       these unresolved (see AMBIGUOUS_CSV_NOTES).
    3. Otherwise -> "unknown".
- status is only asserted when the log content actually supports it - a Python traceback or
  an explicit "Workflow complete and wellplate discarded" line. Everything else (the vast
  majority - most logs just stop abruptly with no completion/failure signal at all) is
  labeled "unknown_backfilled" rather than guessed, per the repo's no-silent-defaults rule.
- pid/user are always blank - these were never recorded in any historical log.
- Safe to re-run: skips any log_filename already present in experiment_runs.csv.
"""
import csv
import os
import re
import uuid

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiment_tracking", "experiment_runs.csv")

CSV_FIELDS = [
    "run_id", "workflow_name", "simulate", "datetime_started", "datetime_stopped",
    "status", "duration_sec", "log_filename", "pid", "user",
]

TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}")
DETECTED_NAME_RE = re.compile(r"Detected workflow name:\s*(\S+)")
VIAL_LOAD_RE = re.compile(r"Loaded vial data from\s*(\S+)")
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"

# Ambiguous CSV names (shared by >1 workflow script) - documented, not hidden:
# - amine_protonation_vials.csv: shared by amine_protonation_workflow.py and
#   amine_protonation_workflow_SPmodified.py -> defaulted to the base workflow.
# - microgel_inputs.csv: shared by microgel_workflow.py and nanoparticle_workflow.py ->
#   defaulted to microgel_workflow.
# - mof_synthesis_vials.csv: shared by mof_synthesis_workflow.py, _concurrent, _multiple ->
#   defaulted to the base workflow.
# - surfactant_grid_vials_expanded.csv: shared by create_surfactant_substocks.py,
#   surfactant_grid_adaptive_concentrations.py, and a fallback default in
#   enhanced_SP_arm_position_program_v1.1...py -> defaulted to
#   surfactant_grid_adaptive_concentrations.py per explicit user direction (it's "a real one").
CSV_TO_WORKFLOW = {
    "amine_protonation_vials.csv": "amine_protonation_workflow",
    "color_matching_vials.csv": "color_matching",
    "color_mixing_vials.csv": "color_mixing",
    "surfactant_grid_vials_expanded.csv": "surfactant_grid_adaptive_concentrations",
    "degradation_vial_status.csv": "Degradation_serena",
    "calibration_vials.csv": "glycerol_dispense_baseline",
    "ilya_input_vials.csv": "ilya_workflow_v2",
    "ilya_input.csv": "ilya_workflow",
    "microgel_inputs.csv": "microgel_workflow",
    "mof_synthesis_vials.csv": "mof_synthesis_workflow",
    "peroxide_assay_vial_status.csv": "peroxide_serena_v2",
    "polymer_dye_vials.csv": "polymer_dye_kinetics_workflow",
    "polymer_phospholipid_vials.csv": "polymer_phospholipid_turbidity_assay",
    "surfactant_4d_SDS_NaLS_TTAB_DTAB_vials.csv": "run_4d_algorithm_comparison",
    "sample_input_vials.csv": "sample_workflow_v2",
    "oilsands_vials.csv": "sands_workflow_SP",
    "surfactant_multidim_vials.csv": "surfactant_multidimensional_workflow",
    "zif8_bsa_vials.csv": "zif8_bsa_workflow",
}

COMPLETE_RE = re.compile(r"Workflow complete and wellplate discarded")
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")


def load_existing_log_filenames():
    if not os.path.isfile(CSV_PATH):
        return set()
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        return {row["log_filename"] for row in csv.DictReader(f)}


def resolve_workflow_name(lines):
    for line in lines:
        m = DETECTED_NAME_RE.search(line)
        if m:
            return m.group(1)
    for line in lines:
        m = VIAL_LOAD_RE.search(line)
        if m:
            csv_basename = os.path.basename(m.group(1).replace("\\", "/"))
            return CSV_TO_WORKFLOW.get(csv_basename, "unknown")
    return "unknown"


def resolve_status(lines):
    for line in lines:
        if TRACEBACK_RE.search(line):
            return "failed_backfilled"
    for line in lines:
        if COMPLETE_RE.search(line):
            return "completed_backfilled"
    return "unknown_backfilled"


def process_file(path, filename):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    started = stopped = None
    for line in lines:
        m = TIMESTAMP_RE.match(line)
        if m:
            if started is None:
                started = m.group(1)
            stopped = m.group(1)

    if started is None:
        return None  # no parseable timestamps at all - shouldn't happen for *.log, skip defensively

    from datetime import datetime
    duration_sec = (datetime.strptime(stopped, DATETIME_FMT) - datetime.strptime(started, DATETIME_FMT)).total_seconds()

    return {
        "run_id": uuid.uuid4().hex[:8],
        "workflow_name": resolve_workflow_name(lines),
        "simulate": False,
        "datetime_started": started,
        "datetime_stopped": stopped,
        "status": resolve_status(lines),
        "duration_sec": duration_sec,
        "log_filename": filename,
        "pid": "",
        "user": "",
    }


def main():
    existing = load_existing_log_filenames()
    candidates = [
        f for f in os.listdir(LOGS_DIR)
        if f.endswith(".log") and "simulate" not in f and f not in existing
    ]

    rows = []
    skipped_no_timestamp = []
    for filename in sorted(candidates):
        row = process_file(os.path.join(LOGS_DIR, filename), filename)
        if row is None:
            skipped_no_timestamp.append(filename)
        else:
            rows.append(row)

    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Summary
    from collections import Counter
    wf_counts = Counter(r["workflow_name"] for r in rows)
    status_counts = Counter(r["status"] for r in rows)

    print(f"Candidates found: {len(candidates)}")
    print(f"Rows written: {len(rows)}")
    print(f"Skipped (no parseable timestamp): {len(skipped_no_timestamp)}")
    print()
    print("Workflow name breakdown:")
    for name, count in wf_counts.most_common():
        print(f"  {name}: {count}")
    print()
    print("Status breakdown:")
    for name, count in status_counts.most_common():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()

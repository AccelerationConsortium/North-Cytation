import csv, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LOGS_DIR = os.path.join(ROOT, "logs")
CSV_PATH = os.path.join(ROOT, "experiment_tracking", "experiment_runs.csv")

with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

filtered = [r for r in rows if r["status"] == "completed" and r["simulate"] == "False"]

print(f"Total rows: {len(filtered)}")
for row in filtered:
    path = os.path.join(LOGS_DIR, row["log_filename"])
    if not os.path.isfile(path):
        print(f"{row['run_id']} | MISSING")
        continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    found_idx = None
    for i, l in enumerate(lines):
        if "User selected: Abort Workflow" in l:
            found_idx = i
            break
    if found_idx is None:
        print(f"{row['run_id']} | abort_found=False | total_lines={len(lines)}")
    else:
        after = len(lines) - found_idx - 1
        print(f"{row['run_id']} | abort_found=True | line {found_idx+1}/{len(lines)} | lines_after={after}")

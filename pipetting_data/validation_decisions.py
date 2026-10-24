"""Shared, documented selection rules for embedded pipetting validation."""
from pathlib import Path
import csv
import os
import threading

import pandas as pd

_APPEND_LOCK = threading.Lock()

DECISION_COLUMNS = ["optimization_stage", "accepted", "adaptive_correction",
                    "within_tolerance", "tolerance_ul", "decision_reason", "decision_source"]


def tolerance_ul(volume_ml):
    volume = volume_ml * 1000
    pct = 1 if 200 <= volume <= 1000 else 2 if 60 <= volume <= 199 else 3 if 20 <= volume <= 59 else 5 if 2.5 <= volume <= 19 else 10
    return volume * pct / 100


def annotate(frame, adaptive, failed=False, source="recorded"):
    """One session/target group. Select by mean error, preserving all replicates."""
    result = frame.copy()
    target = float(result.target_volume_ml.iloc[0])
    tolerance = tolerance_ul(target)
    means = result.groupby("optimization_stage").measured_volume_ml.mean()
    errors = (means * 1000 - target * 1000).abs()
    if 1 not in means or means.isna().any():
        raise ValueError("Cannot decide without a finite baseline and stage means")
    chosen = 1
    reason = "baseline_passed" if errors.loc[1] <= tolerance else "baseline_retained"
    if adaptive and set(means.index) == {1, 2, 3} and not failed:
        best = min(errors.index, key=lambda stage: (errors.loc[stage], stage))
        if errors.loc[best] <= tolerance or (best != 1 and errors.loc[best] < errors.loc[1]):
            chosen = best
            reason = "best_stage_selected" if best != 1 else "baseline_passed"
    if failed:
        reason = "optimization_failed"
    result["accepted"] = (result.optimization_stage == chosen) & (not failed)
    result["adaptive_correction"] = adaptive
    result["within_tolerance"] = result.optimization_stage.map(errors <= tolerance)
    result["tolerance_ul"] = tolerance
    result["decision_reason"] = reason
    result["decision_source"] = source
    summary = {name: result.loc[result.optimization_stage == chosen, name].iloc[0] for name in DECISION_COLUMNS}
    return result, summary


def read_csv(path):
    """Read the historical correction header with its three unnamed environment columns."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    header, records = rows[0], rows[1:]
    width = max([len(header)] + [len(row) for row in records])
    if len(header) == 10 and width == 13 and "old_overaspirate_ml" in header:
        header += ["temp_c", "humidity_pct", "pressure_pa"]
    if any(len(row) > len(header) for row in records):
        raise ValueError(f"Unrecognized CSV width: {path}")
    return pd.DataFrame([row + [""] * (len(header) - len(row)) for row in records], columns=header)


def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def append_csv(frame, path):
    """Align by column name; adding fields must never corrupt an old CSV header."""
    with _APPEND_LOCK:
        path = Path(path)
        if path.exists():
            old = read_csv(path)
            frame = pd.concat([old, frame], ignore_index=True)
        write_csv(frame, path)

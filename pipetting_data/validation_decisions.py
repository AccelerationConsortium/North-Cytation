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


def annotate_check_confirm(frame, outcome, source="check_confirm"):
    """Check/confirm session: every replicate used for the update is accepted.

    Statistics come from the check batch (stage 1): what the workflow actually dispensed
    at the overaspirate in use, not a best-of-several selection.
    """
    result = frame.copy()
    target = float(result.target_volume_ml.iloc[0])
    tolerance = tolerance_ul(target)
    means = result.groupby("optimization_stage").measured_volume_ml.mean()
    errors = (means * 1000 - target * 1000).abs()
    used = (result["used_for_update"] if "used_for_update" in result else pd.Series(True, index=result.index))
    result["accepted"] = used.fillna(True).astype(bool)
    result["adaptive_correction"] = False
    result["within_tolerance"] = result.optimization_stage.map(errors <= tolerance)
    result["tolerance_ul"] = tolerance
    result["decision_reason"] = outcome["verdict"] if outcome else "not_updated"
    result["decision_source"] = source
    check = result[result.optimization_stage == 1]
    summary = {name: check[name].iloc[0] for name in DECISION_COLUMNS}
    summary["accepted"] = True
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

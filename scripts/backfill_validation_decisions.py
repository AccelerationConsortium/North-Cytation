"""Backfill accepted stages using preserved replicate order and recorded corrections.

Requires a previously verified backup directory. Produces a per-volume audit report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import linregress

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipetting_data.validation_decisions import DECISION_COLUMNS, annotate, read_csv, write_csv


def migrate(root, backup):
    paths = [root / "pipetting_data" / name for name in (
        "master_pipetting_measurements.csv", "master_pipetting_sessions.csv", "parameter_corrections.csv")]
    manifest = json.loads((backup / "manifest.json").read_text())
    for path in paths:
        rel = str(path.relative_to(root))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest[rel] or digest != hashlib.sha256((backup / rel).read_bytes()).hexdigest():
            raise ValueError(f"Backup verification failed or data changed: {path}")
    measurements, sessions, corrections = [read_csv(path) for path in paths]
    original = measurements.copy()
    for frame in (measurements, sessions):
        for name in DECISION_COLUMNS:
            frame[name] = ""
    for name in DECISION_COLUMNS + ["selected_stage"]:
        corrections[name] = ""
    numeric = original.copy()
    for key in ("target_volume_ml", "measured_volume_ml", "replicate_number", "overaspirate_vol"):
        numeric[key] = pd.to_numeric(numeric[key], errors="coerce")
    reports, decisions, selected_groups = [], {}, {}
    for key, group in numeric.groupby(["session_id", "liquid_type", "target_volume_ml"], sort=False):
        session, liquid, volume = key
        issue = ""
        source = "reconstructed_replicate_order"
        blocks = (group.replicate_number.diff().fillna(1) <= 0).cumsum() + 1
        sizes = group.groupby(blocks).size()
        valid_blocks = all(part.replicate_number.tolist() == list(range(1, len(part) + 1))
                           for _, part in group.groupby(blocks))
        if len(sizes) not in (1, 3) or not valid_blocks or (len(sizes) == 3 and sizes.nunique() != 1):
            issue = "Cannot establish complete stage blocks"
        elif not np.isfinite(group[["target_volume_ml", "measured_volume_ml", "overaspirate_vol"]].to_numpy()).all():
            issue = "Invalid measurements"
        if issue:
            reports.append(dict(session_id=session, liquid=liquid, target_ml=volume, status="unresolved", detail=issue))
            continue
        staged = group.copy()
        staged["optimization_stage"] = blocks.to_numpy()
        adaptive = len(sizes) == 3
        annotated, decision = annotate(staged, adaptive, source=source)
        matching = corrections[(corrections.session_id == session) & (corrections.liquid_type == liquid) &
                               np.isclose(pd.to_numeric(corrections.target_volume_ml), volume, rtol=0, atol=1e-10)]
        for index, correction in matching.iterrows():
            stage = re.search(r"stage([123])", correction.validation_type)
            if not stage:
                continue
            selected = 1 if correction.validation_type.endswith("_rejected") else int(stage[1])
            expected_oa = float(correction.old_overaspirate_ml if correction.validation_type.endswith("_rejected")
                                else correction.new_overaspirate_ml)
            chosen = staged[staged.optimization_stage == selected]
            if selected != decision["optimization_stage"] or chosen.empty or not np.isclose(
                    chosen.overaspirate_vol.to_numpy(), expected_oa, rtol=1e-6, atol=1e-8).all():
                issue = "Recorded correction conflicts with reconstructed stages/parameters"
                break
            source = "reconstructed_correction_confirmed"
        if issue:
            reports.append(dict(session_id=session, liquid=liquid, target_ml=volume, status="unresolved", detail=issue))
            continue
        annotated["decision_source"] = source
        decision["decision_source"] = source
        # One block establishes the reported result, but cannot establish whether adjustment was enabled.
        if not adaptive:
            annotated["adaptive_correction"] = ""
            decision["adaptive_correction"] = ""
        measurements.loc[annotated.index, DECISION_COLUMNS] = annotated[DECISION_COLUMNS].astype(object)
        decisions[key] = decision
        selected_groups[key] = annotated[annotated.accepted == True]
        reports.append(dict(session_id=session, liquid=liquid, target_ml=volume, status="resolved",
                            selected_stage=int(decision["optimization_stage"]), source=source,
                            within_tolerance=bool(decision["within_tolerance"])))
        for index, correction in matching.iterrows():
            stage = re.search(r"stage([123])", correction.validation_type)
            corrections.loc[index, "selected_stage"] = decision["optimization_stage"]
            corrections.loc[index, "decision_source"] = source
            corrections.loc[index, "adaptive_correction"] = True
            corrections.loc[index, "tolerance_ul"] = decision["tolerance_ul"]
            corrections.loc[index, "decision_reason"] = decision["decision_reason"]
            if stage:
                attempted = int(stage[1])
                corrections.loc[index, "optimization_stage"] = attempted
                corrections.loc[index, "accepted"] = not correction.validation_type.endswith("_rejected")
                corrections.loc[index, "within_tolerance"] = bool(annotated[annotated.optimization_stage == attempted].within_tolerance.iloc[0])
    for index, row in sessions.iterrows():
        key = (row.session_id, row.liquid_type, float(row.target_volume_ml))
        if key not in decisions:
            continue
        chosen = selected_groups[key]
        mean = chosen.measured_volume_ml.mean()
        std = chosen.measured_volume_ml.std() if len(chosen) > 1 else 0.0
        values = dict(decisions[key], measured_mean_ml=mean, measured_std_ml=std,
                      num_replicates=len(chosen), accuracy_pct=(mean / key[2] - 1) * 100,
                      precision_cv_pct=std / mean * 100 if mean > 0 else 0)
        from pipetting_data.pipetting_parameters import PipettingParameters
        for name in vars(PipettingParameters()):
            if name in chosen:
                values[name] = chosen.iloc[0][name]
        for name, value in values.items():
            sessions.loc[index, name] = value
    # Overall regression is coherent only when every volume for that session is resolved.
    for sid, group in sessions.groupby("session_id"):
        keys = [(sid, row.liquid_type, float(row.target_volume_ml)) for _, row in group.iterrows()]
        if all(key in selected_groups for key in keys):
            chosen = pd.concat([selected_groups[key] for key in keys])
            if chosen.target_volume_ml.nunique() > 1:
                fit = linregress(chosen.target_volume_ml, chosen.measured_volume_ml)
                values = [fit.rvalue ** 2, fit.slope, fit.intercept]
            else:
                values = [1., 1., 0.]
            sessions.loc[group.index, ["r_squared", "linear_slope", "linear_intercept"]] = values
        else:
            sessions.loc[group.index, ["r_squared", "linear_slope", "linear_intercept"]] = ""
    # Preserve every original measurement field, including original replicate ordering.
    pd.testing.assert_frame_equal(measurements[original.columns], original)
    for frame, path in zip((measurements, sessions, corrections), paths):
        write_csv(frame, path)
        check = read_csv(path)
        assert len(check) == len(frame) and list(check.columns) == list(frame.columns)
    report = pd.DataFrame(reports)
    report.to_csv(backup / "backfill_report.csv", index=False)
    print(report.status.value_counts().to_dict())
    print("Measurement rows preserved:", len(measurements), "Session rows:", len(sessions), "Corrections:", len(corrections))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", required=True, type=Path)
    args = parser.parse_args()
    migrate(Path(__file__).resolve().parents[1], args.backup.resolve())

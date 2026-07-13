"""
Regenerate two_point_combined_raw_data.csv and two_point_combined_trial_data.csv
with computed fields added:
  raw:   measurement_id, deviation_individual_pct
  trial: volume_measured_ul, volume_measured_ml, deviation_pct, precision_cv_pct
Hardware parameters left as empty columns (to be recovered separately).
"""
import pandas as pd
import numpy as np
from pathlib import Path

FOLDER = Path("output/two_point_focus_glycerol_25_50_75_100")
DETAIL_GLOB = "two_point_series_demo_details_*.csv"

POINT_TO_TRIAL = {"point_1": 1, "point_2": 2, "point_3_validation": 3}

HW_COLS = [
    "hardware_parameters_aspirate_speed",
    "hardware_parameters_dispense_speed",
    "hardware_parameters_aspirate_wait_time",
    "hardware_parameters_dispense_wait_time",
    "hardware_parameters_pre_asp_air_vol",
    "hardware_parameters_post_asp_air_vol",
    "hardware_parameters_blowout_vol",
    "hardware_parameters_asp_disp_cycles",
    "hardware_parameters_post_retract_wait_time",
    "hardware_parameters_retract_speed",
]

# ── Load and tag all detail files ─────────────────────────────────────────────
frames = []
for f in sorted(FOLDER.glob(DETAIL_GLOB)):
    ts = f.stem.replace("two_point_series_demo_details_", "")
    summary_f = FOLDER / f"two_point_series_demo_summary_{ts}.csv"
    df = pd.read_csv(f)
    df["source_timestamp"] = ts
    df["run_id"] = f"two_point_{ts}"
    df["source_detail_file"] = f.name
    df["source_summary_file"] = summary_f.name if summary_f.exists() else ""
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)

# ── Build raw measurements (replicate-level) ──────────────────────────────────
raw["trial_id"] = raw["point"].map(POINT_TO_TRIAL)
raw["replicate_id"] = raw["replicate"]
raw["liquid"] = raw["liquid_name"]
raw["strategy"] = "two_point_series"
raw["volume_target_ul"] = raw["target_volume_uL"]
raw["volume_target_ml"] = raw["target_volume_uL"] / 1000.0
raw["volume_measured_ul"] = raw["measured_volume_uL"]
raw["volume_measured_ml"] = raw["measured_volume_uL"] / 1000.0
raw["duration_s"] = raw["elapsed_s"]
raw["calibration_overaspirate_vol"] = raw["overaspirate_uL"] / 1000.0

# Computable: individual deviation %
raw["deviation_individual_pct"] = (
    (raw["volume_measured_ul"] - raw["volume_target_ul"]).abs() / raw["volume_target_ul"] * 100.0
)

# Sequential measurement_id
raw.insert(0, "measurement_id", range(1, len(raw) + 1))

# Empty hardware columns
for col in HW_COLS:
    raw[col] = ""

# metadata columns (start/end not available; replicate maps to replicate_id)
raw["metadata_replicate"] = raw["replicate_id"]
raw["metadata_start_time"] = raw["timestamp"]
raw["metadata_end_time"] = ""

# Final column order: optimizer-style fields first, then 2pt originals
RAW_COLS = [
    "measurement_id", "run_id", "source_detail_file", "source_timestamp",
    "trial_id", "strategy", "liquid", "label", "vial_name",
    "volume_target_ul", "volume_target_ml",
    "volume_measured_ul", "volume_measured_ml",
    "duration_s", "timestamp",
    "replicate_id", "deviation_individual_pct",
    "calibration_overaspirate_vol",
] + HW_COLS + [
    "metadata_replicate", "metadata_start_time", "metadata_end_time",
    "density_g_mL",
]

raw_out = raw[[c for c in RAW_COLS if c in raw.columns]]
raw_out.to_csv(FOLDER / "two_point_combined_raw_data.csv", index=False)
print(f"Wrote raw: {len(raw_out)} rows, {len(raw_out.columns)} columns")

# ── Build trial data (grouped by run + liquid + target volume + trial_id) ──────
grp = raw.groupby(["run_id", "source_timestamp", "source_detail_file", "source_summary_file",
                    "liquid", "label", "vial_name", "volume_target_ul", "trial_id", "point"],
                   sort=False)

trial_rows = []
for keys, g in grp:
    vols = g["volume_measured_ul"]
    mean_vol = vols.mean()
    std_vol = vols.std(ddof=1) if len(vols) > 1 else float("nan")
    cv_pct = (std_vol / mean_vol * 100.0) if mean_vol > 0 else float("nan")
    dev_pct = abs(mean_vol - g["volume_target_ul"].iloc[0]) / g["volume_target_ul"].iloc[0] * 100.0

    row = {
        "run_id": g["run_id"].iloc[0],
        "source_timestamp": g["source_timestamp"].iloc[0],
        "source_detail_file": g["source_detail_file"].iloc[0],
        "source_summary_file": g["source_summary_file"].iloc[0],
        "strategy": "two_point_series",
        "trial_id": int(g["trial_id"].iloc[0]),
        "point": g["point"].iloc[0],
        "liquid": g["liquid"].iloc[0],
        "label": g["label"].iloc[0],
        "vial_name": g["vial_name"].iloc[0],
        "volume_target_ul": g["volume_target_ul"].iloc[0],
        "volume_target_ml": round(g["volume_target_ul"].iloc[0] / 1000.0, 9),
        "calibration_overaspirate_vol": round(g["calibration_overaspirate_vol"].mean(), 9),
        "measurement_count": len(g),
        "volume_measured_ul": round(mean_vol, 6),
        "volume_measured_ml": round(mean_vol / 1000.0, 9),
        "volume_measured_ul_std": round(std_vol, 6) if not np.isnan(std_vol) else "",
        "duration_mean_s": round(g["duration_s"].mean(), 6),
        "deviation_pct": round(dev_pct, 6),
        "precision_cv_pct": round(cv_pct, 6) if not np.isnan(cv_pct) else "",
    }
    for col in HW_COLS:
        row[col] = ""
    trial_rows.append(row)

trial_out = pd.DataFrame(trial_rows).sort_values(
    ["source_timestamp", "liquid", "volume_target_ul", "trial_id"]
)
trial_out.to_csv(FOLDER / "two_point_combined_trial_data.csv", index=False)
print(f"Wrote trial: {len(trial_out)} rows, {len(trial_out.columns)} columns")
print("Done.")

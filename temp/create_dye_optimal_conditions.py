"""
1. Copies existing optimal_conditions_ethanol.csv, optimal_conditions_glycerol_small.csv,
   optimal_conditions_water_complete.csv to pipetting_data/pre_june10_backup/

2. Creates new files for the DYE variants using P3 results from the most recent
   two-point series run (20260610_111558):
     optimal_conditions_ethanol_dye.csv
     optimal_conditions_glycerol_dye.csv
     optimal_conditions_water_dye.csv

   Hardware params are taken from the corresponding liquid's trial_results.csv
   (same params used as baseline in the two-point demo).
   overaspirate_vol = optimal_overaspirate_uL / 1000  (mL)
   volume_measured  = point3_mean_uL
"""

import shutil
import csv
import statistics
from pathlib import Path

import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent
PIP_DIR     = ROOT / "pipetting_data"
BACKUP_DIR  = PIP_DIR / "pre_june10_backup"
SUMMARY_CSV = ROOT / "output/two_point_series_demo_summary_20260610_111558.csv"
DETAILS_CSV = ROOT / "output/two_point_series_demo_details_20260610_111558.csv"

TRIAL_RESULTS = {
    "glycerol": ROOT / "calibration_modular_v2/output/run_1780513054_glycerol/trial_results.csv",
    "water":    ROOT / "calibration_modular_v2/output/run_1779739005_water/trial_results.csv",
    "ethanol":  ROOT / "calibration_modular_v2/output/run_1780412080_ethanol/trial_results.csv",
}

EXISTING_FILES = {
    "ethanol":  PIP_DIR / "optimal_conditions_ethanol.csv",
    "glycerol": PIP_DIR / "optimal_conditions_glycerol_small.csv",
    "water":    PIP_DIR / "optimal_conditions_water_complete.csv",
}

OUTPUT_COLS = [
    "aspirate_speed", "dispense_speed", "aspirate_wait_time", "blowout_vol",
    "post_asp_air_vol", "pre_asp_air_vol", "dispense_wait_time", "asp_disp_cycles",
    "overaspirate_vol", "volume_target", "post_retract_wait_time", "retract_speed",
    "volume_measured", "volume_target_ul", "time", "average_deviation",
    "variability", "trials_count", "status", "measurement_count",
]

# ── 1. Backup ─────────────────────────────────────────────────────────────────
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
for name, src in EXISTING_FILES.items():
    dst = BACKUP_DIR / src.name
    shutil.copy2(src, dst)
    print(f"Backed up: {src.name} -> {BACKUP_DIR.name}/")

# ── 2. Load summary and details ───────────────────────────────────────────────
summary = pd.read_csv(SUMMARY_CSV)
details = pd.read_csv(DETAILS_CSV)

# ── 3. Helper: best hardware params from trial_results ────────────────────────
def _get_hw_params(liquid_name: str) -> dict:
    df = pd.read_csv(TRIAL_RESULTS[liquid_name])
    for col in ["deviation_pct", "precision_cv_pct", "duration_mean_s", "measurement_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["measurement_count"] >= 2]

    # Desirability scoring (same as two_point_series_calibration_demo)
    tol = 3.0
    t_range = max(df["duration_mean_s"].max() - df["duration_mean_s"].min(), 1.0)
    df["d_acc"]  = 1.0 / (1.0 + (df["deviation_pct"]   / tol) ** 2)
    df["d_prec"] = 1.0 / (1.0 + (df["precision_cv_pct"] / tol) ** 2)
    df["d_time"] = (df["duration_mean_s"].max() - df["duration_mean_s"]) / t_range
    df["score"]  = 0.4 * df["d_acc"] + 0.5 * df["d_prec"] + 0.1 * df["d_time"]
    best = df.loc[df["score"].idxmax()]

    hw_cols = ["aspirate_speed", "dispense_speed", "aspirate_wait_time", "blowout_vol",
               "post_asp_air_vol", "pre_asp_air_vol", "dispense_wait_time",
               "asp_disp_cycles", "post_retract_wait_time", "retract_speed"]
    params = {}
    for col in hw_cols:
        for prefix in ("", "hardware_parameters_"):
            if prefix + col in best.index:
                params[col] = float(best[prefix + col])
                break
    return params

# ── 4. Create dye files ───────────────────────────────────────────────────────
DYE_LABELS = {
    "glycerol": "glycerol_dye",
    "water":    "water_dye",
    "ethanol":  "ethanol_dye",
}

for liquid, dye_label in DYE_LABELS.items():
    hw = _get_hw_params(liquid)
    grp = summary[summary["label"] == dye_label].sort_values("target_volume_uL")

    rows = []
    for _, r in grp.iterrows():
        target_ul = float(r["target_volume_uL"])
        opt_ov_ml = float(r["optimal_overaspirate_uL"]) / 1000.0
        p3_mean   = float(r["point3_mean_uL"])
        dev_pct   = float(r["point3_deviation_pct"])

        # P3 reps from details for timing and variability
        p3_reps = details[
            (details["label"] == dye_label) &
            (details["target_volume_uL"] == target_ul) &
            (details["point"] == "point_3_validation")
        ]
        time_s  = p3_reps["elapsed_s"].mean() if not p3_reps.empty else 0.0
        if len(p3_reps) >= 2:
            vols    = p3_reps["measured_volume_uL"].tolist()
            variability = statistics.stdev(vols) / statistics.mean(vols) * 100
        else:
            variability = 0.0

        rows.append({
            "aspirate_speed":       hw.get("aspirate_speed", ""),
            "dispense_speed":       hw.get("dispense_speed", ""),
            "aspirate_wait_time":   hw.get("aspirate_wait_time", ""),
            "blowout_vol":          hw.get("blowout_vol", ""),
            "post_asp_air_vol":     hw.get("post_asp_air_vol", ""),
            "pre_asp_air_vol":      hw.get("pre_asp_air_vol", ""),
            "dispense_wait_time":   hw.get("dispense_wait_time", ""),
            "asp_disp_cycles":      int(hw.get("asp_disp_cycles", 0)),
            "overaspirate_vol":     round(opt_ov_ml, 10),
            "volume_target":        target_ul,
            "post_retract_wait_time": hw.get("post_retract_wait_time", 0.0),
            "retract_speed":        hw.get("retract_speed", 5.0),
            "volume_measured":      round(p3_mean, 6),
            "volume_target_ul":     target_ul,
            "time":                 round(time_s, 6),
            "average_deviation":    round(dev_pct, 6),
            "variability":          round(variability, 6),
            "trials_count":         1,
            "status":               "success",
            "measurement_count":    len(p3_reps) if not p3_reps.empty else 3,
        })

    out_path = PIP_DIR / f"optimal_conditions_{dye_label}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created: {out_path.name}  ({len(rows)} rows)")

print("\nDone.")

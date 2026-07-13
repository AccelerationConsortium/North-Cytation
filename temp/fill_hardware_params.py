"""
1. Runs the exact desirability scoring logic from two_point_series_calibration_demo.py
   against each SBT trial_results.csv.
2. Saves optimal_conditions.csv to the focus folder.
3. Fills hardware_parameters_* columns in both combined CSVs.

Liquid name mapping (combined CSV 'liquid' column -> SBT folder label):
    glycerol      -> glycerol_SBT   (top-level)
    water         -> water_SBT
    ethanol       -> ethanol_SBT
    DMSO          -> dmso_SBT
    PVA_DMSO      -> pva_dmso_SBT
    agar_water_4% -> alginate_SBT

Defaults for any parameter not found in SBT data:
    retract_speed = 5.0, all others = 0.0
"""
import pandas as pd
from pathlib import Path

BASE = Path("calibration_modular_v2/output/calibration_results_summary")
FOCUS = Path("output/two_point_focus_glycerol_25_50_75_100")

SBT_FOLDERS = {
    "glycerol":      BASE / "glycerol_SBT",
    "water":         BASE / "water_SBT",
    "ethanol":       BASE / "ethanol_SBT",
    "DMSO":          BASE / "dmso_SBT",
    "PVA_DMSO":      BASE / "pva_dmso_SBT",
    "agar_water_4%": BASE / "alginate_SBT",
}

HW_PARAM_COLS = [
    "aspirate_speed", "dispense_speed", "aspirate_wait_time", "dispense_wait_time",
    "pre_asp_air_vol", "post_asp_air_vol", "blowout_vol", "asp_disp_cycles",
    "post_retract_wait_time", "retract_speed",
]

DEFAULTS = {col: 0.0 for col in HW_PARAM_COLS}
DEFAULTS["retract_speed"] = 5.0

OVERASPIRATE_CANDIDATES = ("overaspirate_vol", "calibration_overaspirate_vol", "param_overaspirate_vol")


def _desirability(metric: float, tolerance: float, s: float = 2.0) -> float:
    return 1.0 / (1.0 + (metric / tolerance) ** s)


def extract_best_params(df: pd.DataFrame, tolerance_pct: float = 3.0) -> dict:
    df = df.copy()
    for col in ["deviation_pct", "precision_cv_pct", "duration_mean_s", "measurement_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["measurement_count"] >= 2].reset_index(drop=True)
    if df.empty:
        return {}

    t_min = df["duration_mean_s"].min()
    t_max = df["duration_mean_s"].max()
    t_range = max(t_max - t_min, 1.0)

    df["d_acc"]  = df["deviation_pct"].apply(lambda x: _desirability(x, tolerance_pct))
    df["d_prec"] = df["precision_cv_pct"].apply(lambda x: _desirability(x, tolerance_pct))
    df["d_time"] = (t_max - df["duration_mean_s"]) / t_range
    df["composite"] = 0.4 * df["d_acc"] + 0.5 * df["d_prec"] + 0.1 * df["d_time"]

    best = df.loc[df["composite"].idxmax()]

    params = dict(DEFAULTS)  # start with defaults

    for col in HW_PARAM_COLS:
        for variant in (col, f"hardware_parameters_{col}", f"param_{col}"):
            if variant in best.index and pd.notna(best[variant]):
                params[col] = float(best[variant])
                break

    for ov_col in OVERASPIRATE_CANDIDATES:
        if ov_col in best.index and pd.notna(best[ov_col]):
            params["overaspirate_vol"] = float(best[ov_col])
            break

    params["composite_score"] = float(best["composite"])
    params["deviation_pct"]   = float(best["deviation_pct"])
    params["precision_cv_pct"] = float(best["precision_cv_pct"])
    params["duration_mean_s"]  = float(best["duration_mean_s"])
    return params


# ── Step 1: Extract best params per liquid ───────────────────────────────────
optimal = {}
for liquid, folder in SBT_FOLDERS.items():
    tr = folder / "trial_results.csv"
    df = pd.read_csv(tr)
    params = extract_best_params(df)
    optimal[liquid] = params
    print(f"{liquid}: composite={params.get('composite_score', 'N/A'):.4f}  "
          f"overasp={params.get('overaspirate_vol', 'N/A'):.6f}  "
          f"asp_speed={params.get('aspirate_speed', 'N/A')}")

# ── Step 2: Save optimal_conditions.csv ──────────────────────────────────────
rows = []
for liquid, params in optimal.items():
    row = {"liquid": liquid}
    row.update(params)
    rows.append(row)

opt_df = pd.DataFrame(rows)
opt_path = FOCUS / "optimal_conditions.csv"
opt_df.to_csv(opt_path, index=False)
print(f"\nWrote optimal_conditions: {opt_path} ({len(opt_df)} rows)")

# ── Step 3: Fill hardware_parameters_* in combined CSVs ──────────────────────
HW_PREFIX_COLS = [f"hardware_parameters_{c}" for c in HW_PARAM_COLS]

for fname in ("two_point_combined_raw_data.csv", "two_point_combined_trial_data.csv"):
    path = FOCUS / fname
    df = pd.read_csv(path)

    for col in HW_PREFIX_COLS:
        if col not in df.columns:
            df[col] = float("nan")

    for liquid, params in optimal.items():
        mask = df["liquid"] == liquid
        if mask.sum() == 0:
            print(f"  WARNING: no rows for liquid '{liquid}' in {fname}")
            continue
        for col in HW_PARAM_COLS:
            df.loc[mask, f"hardware_parameters_{col}"] = params.get(col, DEFAULTS[col])

    df.to_csv(path, index=False)
    filled = df["hardware_parameters_aspirate_speed"].notna().sum()
    print(f"Filled {filled}/{len(df)} rows in {fname}")

print("\nDone.")

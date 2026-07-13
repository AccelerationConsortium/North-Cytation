"""
Re-runs the exact _extract_best_trial_parameters() logic from two_point_series_calibration_demo.py
against the SBT trial_results.csv files to recover hardware parameters used in two-point runs.
"""
import pandas as pd
from pathlib import Path

BASE = Path("calibration_modular_v2/output/calibration_results_summary")

SBT_FOLDERS = {
    "alginate":      BASE / "alginate_SBT",
    "dmso":          BASE / "dmso_SBT",
    "ethanol":       BASE / "ethanol_SBT",
    "glycerol":      BASE / "glycerol_SBT",
    "glycerol_early": BASE / "glycerol_nowaittime_dripping/glycerol_SBT",
    "pva_dmso":      BASE / "pva_dmso_SBT",
    "water":         BASE / "water_SBT",
}

PARAM_COLS = [
    "aspirate_speed", "dispense_speed", "aspirate_wait_time", "dispense_wait_time",
    "pre_asp_air_vol", "post_asp_air_vol", "blowout_vol", "asp_disp_cycles",
    "post_retract_wait_time", "retract_speed",
]

OVERASPIRATE_CANDIDATES = ("overaspirate_vol", "calibration_overaspirate_vol", "param_overaspirate_vol")


def _desirability(metric: float, tolerance: float, s: float = 2.0) -> float:
    return 1.0 / (1.0 + (metric / tolerance) ** s)


def extract_best_params(df: pd.DataFrame, tolerance_pct: float = 3.0):
    df = df.copy()
    for col in ["deviation_pct", "precision_cv_pct", "duration_mean_s", "measurement_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["measurement_count"] >= 2].reset_index(drop=True)
    if df.empty:
        return None, None

    t_min = df["duration_mean_s"].min()
    t_max = df["duration_mean_s"].max()
    t_range = max(t_max - t_min, 1.0)

    df["d_acc"]  = df["deviation_pct"].apply(lambda x: _desirability(x, tolerance_pct))
    df["d_prec"] = df["precision_cv_pct"].apply(lambda x: _desirability(x, tolerance_pct))
    df["d_time"] = (t_max - df["duration_mean_s"]) / t_range
    df["composite"] = 0.4 * df["d_acc"] + 0.5 * df["d_prec"] + 0.1 * df["d_time"]

    best = df.loc[df["composite"].idxmax()]

    params = {}
    for col in PARAM_COLS:
        for variant in (col, f"hardware_parameters_{col}", f"param_{col}"):
            if variant in best.index and pd.notna(best[variant]):
                params[col] = float(best[variant])
                break

    for ov_col in OVERASPIRATE_CANDIDATES:
        if ov_col in best.index and pd.notna(best[ov_col]):
            params["overaspirate_vol"] = float(best[ov_col])
            break

    return params, best


print(f"{'='*70}")
for name, folder in SBT_FOLDERS.items():
    tr = folder / "trial_results.csv"
    if not tr.exists():
        print(f"{name}: trial_results.csv NOT FOUND at {tr}")
        continue

    df = pd.read_csv(tr)
    print(f"\n{name} ({len(df)} trials total)")
    print(f"  Columns: {list(df.columns)}")

    params, best_row = extract_best_params(df)
    if params is None:
        print("  No valid trials (measurement_count>=2)")
        continue

    print(f"  Best trial composite={best_row['composite']:.4f} | "
          f"dev={best_row['deviation_pct']:.3f}% | "
          f"cv={best_row['precision_cv_pct']:.3f}% | "
          f"time={best_row['duration_mean_s']:.2f}s")
    print("  Extracted parameters:")
    for k, v in sorted(params.items()):
        print(f"    {k}: {v}")

print(f"\n{'='*70}")

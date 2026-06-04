"""Verify the two-point calibration algorithm math on the latest summary CSV."""
import pandas as pd
import numpy as np
import glob
import os

files = sorted(glob.glob("output/two_point_series_demo_summary_*.csv"))
if not files:
    raise FileNotFoundError("No summary files found")
latest = files[-1]
print(f"Using: {latest}\n")
summ = pd.read_csv(latest)

errors = []

for _, row in summ.iterrows():
    target = row["target_volume_uL"]
    p1_ov = row["point1_overaspirate_uL"]
    p1_mean = row["point1_mean_uL"]
    tol_pct = row["tolerance_pct"]
    tol_buf = row["tolerance_buffer_uL"]
    spread = row["spread_uL"]
    direction = row["point2_direction"]
    p2_ov = row["point2_overaspirate_uL"]
    p2_mean = row["point2_mean_uL"]
    slope_stored = row["slope_mL_per_mL"]
    opt_ov_stored = row["optimal_overaspirate_uL"]

    shortfall = target - p1_mean

    # 1. tolerance_buffer = target * tol_pct / 100
    expected_buf = target * tol_pct / 100.0
    if not np.isclose(expected_buf, tol_buf, atol=1e-6):
        errors.append(f"  FAIL tolerance_buffer {row['label']} {target}uL: expected {expected_buf:.4f} got {tol_buf:.4f}")

    # 2. spread = max(abs(shortfall) + tol_buf, 2.0)
    expected_spread = max(abs(shortfall) + tol_buf, 2.0)
    if not np.isclose(expected_spread, spread, atol=1e-6):
        errors.append(f"  FAIL spread {row['label']} {target}uL: expected {expected_spread:.4f} got {spread:.4f}")

    # 3. direction
    expected_dir = "increased" if shortfall > 0 else "decreased"
    if expected_dir != direction:
        errors.append(f"  FAIL direction {row['label']} {target}uL: expected {expected_dir} got {direction}")

    # 4. P2 overaspirate = P1 +/- spread
    expected_p2_ov = p1_ov + spread if direction == "increased" else p1_ov - spread
    expected_p2_ov = max(-10.0, expected_p2_ov)
    if not np.isclose(expected_p2_ov, p2_ov, atol=1e-4):
        errors.append(f"  FAIL P2_ov {row['label']} {target}uL: expected {expected_p2_ov:.4f} got {p2_ov:.4f}")

    # 5. slope = (P2_mean - P1_mean) / (P2_ov - P1_ov)
    ov_diff = p2_ov - p1_ov
    if abs(ov_diff) > 1e-9:
        expected_slope = (p2_mean - p1_mean) / ov_diff
        if not np.isclose(expected_slope, slope_stored, atol=1e-4):
            errors.append(f"  FAIL slope {row['label']} {target}uL: expected {expected_slope:.6f} got {slope_stored:.6f}")

    # 6. optimal_ov = P1_ov + (target - P1_mean) / slope
    if abs(slope_stored) > 1e-9:
        expected_opt = p1_ov + (target - p1_mean) / slope_stored
        expected_opt = max(-10.0, expected_opt)
        if not np.isclose(expected_opt, opt_ov_stored, atol=1e-4):
            errors.append(f"  FAIL opt_ov {row['label']} {target}uL: expected {expected_opt:.4f} got {opt_ov_stored:.4f}")

    # 7. Sanity: does optimal_ov linearly predict target?
    predicted_at_opt = p1_mean + slope_stored * (opt_ov_stored - p1_ov)
    miss = predicted_at_opt - target
    print(f"  {row['label']:15s} {target:5.0f}uL | P1={p1_mean:6.2f} P2={p2_mean:6.2f} | slope={slope_stored:.4f} | opt_ov={opt_ov_stored:6.2f}uL | predicted@opt={predicted_at_opt:6.2f}uL (miss={miss:+.4f}uL)")

print()
if errors:
    print(f"FAILURES ({len(errors)}):")
    for e in errors:
        print(e)
else:
    print(f"ALL {len(summ)} rows PASS - algorithm math is correct.")

# Point 3 convergence check
if "point3_mean_uL" in summ.columns:
    print("\n=== POINT 3 CONVERGENCE (did optimal overaspirate move closer to target?) ===")
    converged = 0
    for _, row in summ.iterrows():
        target = row["target_volume_uL"]
        p1_dev = abs(row["point1_mean_uL"] - target)
        p3_dev = abs(row["point3_mean_uL"] - target)
        tol_pct = row["tolerance_pct"]
        within_tol = row["point3_deviation_pct"] <= tol_pct
        marker = "PASS" if p3_dev < p1_dev else "WORSE"
        tol_marker = " [IN TOL]" if within_tol else ""
        if p3_dev < p1_dev:
            converged += 1
        print(f"  {marker}{tol_marker:10s} {row['label']:15s} {target:5.0f}uL | P1 dev={p1_dev:5.2f}uL -> P3 dev={p3_dev:5.2f}uL | P3={row['point3_mean_uL']:.2f}uL (tol={tol_pct}%)")
    print(f"\nConverged: {converged}/{len(summ)} rows ({100*converged/len(summ):.0f}%)")
else:
    print("\nNOTE: No point3_mean_uL column - re-run two_point_series_calibration_demo.py to get P3 validation data.")

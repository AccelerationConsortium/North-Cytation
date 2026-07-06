"""
Recover PVA_DMSO 25uL and 50uL data from the 2026-06-02 hardware run log.
Creates summary and details CSVs in the output directory.
"""
import csv
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DENSITY = 1.110  # g/mL for PVA_DMSO

# --- Raw measurements from log ---
# Format: (overaspirate_uL, measured_volume_uL, elapsed_s, timestamp)
data = {
    25.0: {
        "point_1": [
            (15.9, 32.34, 39.40, "2026-06-02T17:50:38"),
            (15.9, 32.09, 37.00, "2026-06-02T17:51:15"),
            (15.9, 31.53, 37.40, "2026-06-02T17:51:52"),
        ],
        "point_2": [
            (8.162012, 25.50, 38.00, "2026-06-02T17:52:30"),
            (8.162012, 24.77, 37.00, "2026-06-02T17:53:08"),
            (8.162012, 25.50, 37.60, "2026-06-02T17:53:45"),
        ],
        "point_3": [
            (7.868645, 25.41, 37.00, "2026-06-02T17:54:22"),
            (7.868645, 25.23, 36.80, "2026-06-02T17:55:00"),
            (7.868645, 25.68, 35.80, "2026-06-02T17:55:37"),
        ],
    },
    50.0: {
        "point_1": [
            (15.9, 50.63, 39.80, "2026-06-02T17:57:20"),
            (15.9, 53.33, 37.80, "2026-06-02T17:57:58"),
            (15.9, 52.52, 38.40, "2026-06-02T17:58:36"),
        ],
        "point_2": [
            (12.237838, 49.37, 38.40, "2026-06-02T17:59:15"),
            (12.237838, 48.92, 38.60, "2026-06-02T17:59:53"),
            (12.237838, 47.84, 38.40, "2026-06-02T18:00:32"),
        ],
        "point_3": [
            (13.607168, 50.27, 38.40, "2026-06-02T18:01:11"),
            (13.607168, 50.09, 37.60, "2026-06-02T18:01:50"),
            (13.607168, 49.28, 37.40, "2026-06-02T18:02:27"),
        ],
    },
}

TOLERANCES = {25.0: (3.0, 0.75), 50.0: (3.0, 1.5)}

detail_fields = [
    "label", "liquid_name", "vial_name", "target_volume_uL", "point", "replicate",
    "overaspirate_uL", "measured_volume_uL", "elapsed_s", "density_g_mL", "timestamp",
]
summary_fields = [
    "label", "liquid_name", "vial_name", "target_volume_uL", "replicates_per_point",
    "point1_overaspirate_uL", "point1_mean_uL", "point1_shortfall_uL",
    "tolerance_pct", "tolerance_buffer_uL", "spread_uL", "point2_direction",
    "point2_overaspirate_uL", "point2_mean_uL", "slope_mL_per_mL",
    "optimal_overaspirate_uL", "point3_mean_uL", "point3_deviation_pct", "delta_equation",
]

detail_rows = []
summary_rows = []

for vol_ul, points in data.items():
    tol_pct, tol_buf = TOLERANCES[vol_ul]

    p1_reps = points["point_1"]
    p2_reps = points["point_2"]
    p3_reps = points["point_3"]

    p1_ov = p1_reps[0][0]
    p2_ov = p2_reps[0][0]
    p3_ov = p3_reps[0][0]

    p1_mean = sum(r[1] for r in p1_reps) / len(p1_reps)
    p2_mean = sum(r[1] for r in p2_reps) / len(p2_reps)
    p3_mean = sum(r[1] for r in p3_reps) / len(p3_reps)

    shortfall = vol_ul - p1_mean
    spread = max(abs(shortfall) + tol_buf, 2.0)
    direction = "increased" if shortfall > 0 else "decreased"

    slope = (p2_mean - p1_mean) / (p2_ov - p1_ov)
    optimal_ov = p2_ov + (vol_ul - p2_mean) / slope
    p3_dev = abs(p3_mean - vol_ul) / vol_ul * 100

    for point_name, reps in [("point_1", p1_reps), ("point_2", p2_reps), ("point_3", p3_reps)]:
        for i, (ov, meas, elapsed, ts) in enumerate(reps, start=1):
            detail_rows.append({
                "label": "PVA_dmso",
                "liquid_name": "PVA_DMSO",
                "vial_name": "PVA_DMSO",
                "target_volume_uL": vol_ul,
                "point": point_name,
                "replicate": i,
                "overaspirate_uL": ov,
                "measured_volume_uL": meas,
                "elapsed_s": elapsed,
                "density_g_mL": DENSITY,
                "timestamp": ts,
            })

    summary_rows.append({
        "label": "PVA_dmso",
        "liquid_name": "PVA_DMSO",
        "vial_name": "PVA_DMSO",
        "target_volume_uL": vol_ul,
        "replicates_per_point": 3,
        "point1_overaspirate_uL": p1_ov,
        "point1_mean_uL": round(p1_mean, 6),
        "point1_shortfall_uL": round(shortfall, 6),
        "tolerance_pct": tol_pct,
        "tolerance_buffer_uL": tol_buf,
        "spread_uL": round(spread, 6),
        "point2_direction": direction,
        "point2_overaspirate_uL": p2_ov,
        "point2_mean_uL": round(p2_mean, 6),
        "slope_mL_per_mL": round(slope, 6),
        "optimal_overaspirate_uL": round(optimal_ov, 6),
        "point3_mean_uL": round(p3_mean, 6),
        "point3_deviation_pct": round(p3_dev, 6),
        "delta_equation": "spread_ul=max(abs(shortfall_ul)+tolerance_buffer_ul,2.0)",
    })

detail_path = OUTPUT_DIR / "two_point_series_demo_details_recovered_PVA_DMSO_20260602.csv"
summary_path = OUTPUT_DIR / "two_point_series_demo_summary_recovered_PVA_DMSO_20260602.csv"

with open(detail_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=detail_fields)
    w.writeheader()
    w.writerows(detail_rows)

with open(summary_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=summary_fields)
    w.writeheader()
    w.writerows(summary_rows)

print(f"Saved details: {detail_path}")
print(f"Saved summary: {summary_path}")
print()
for row in summary_rows:
    print(f"  {row['target_volume_uL']}uL | P1={row['point1_mean_uL']:.2f}uL | P2={row['point2_mean_uL']:.2f}uL | "
          f"opt_ov={row['optimal_overaspirate_uL']:.2f}uL | P3={row['point3_mean_uL']:.2f}uL (dev={row['point3_deviation_pct']:.2f}%)")

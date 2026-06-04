"""
Visualize the 3-step two-point overaspirate calibration process.

For each liquid x volume:
  - P1 and P2 are plotted as (overaspirate, mean_measured) dots
  - A dashed interpolation line is drawn through P1 and P2
  - P3 (validation at optimal overaspirate) is shown as a star
  - Target volume is shown as a horizontal line

Gives an intuitive view of how the correction works.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SUMMARIES = [
    "output/two_point_series_demo_summary_20260602_183433.csv",
    "output/two_point_series_demo_summary_20260604_121212.csv",
    "output/two_point_series_demo_summary_20260604_155945.csv",
]
DETAILS_FILES = [
    "output/two_point_series_demo_details_20260602_183433.csv",
    "output/two_point_series_demo_details_20260604_121212.csv",
    "output/two_point_series_demo_details_20260604_155945.csv",
]

OUT_DIR = Path("output/two_point_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

summ = pd.concat([pd.read_csv(p) for p in SUMMARIES], ignore_index=True)
det  = pd.concat([pd.read_csv(p) for p in DETAILS_FILES], ignore_index=True)

# Normalise liquid names
_remap = {"agar_water_4%": "alginate"}
summ["liquid_name"] = summ["liquid_name"].replace(_remap)
det["liquid_name"]  = det["liquid_name"].replace(_remap)

COLORS = {
    "PVA_DMSO":  "#e07b39",
    "DMSO":      "#5b8fd4",
    "water":     "#4caf7d",
    "ethanol":   "#b05fc9",
    "glycerol":  "#c0392b",
    "alginate":  "#7d5a9e",
}

liquids   = list(summ["liquid_name"].unique())
volumes   = sorted(summ["target_volume_uL"].unique())
n_liq     = len(liquids)
n_vol     = len(volumes)

fig, axes = plt.subplots(n_liq, n_vol, figsize=(4.5 * n_vol, 3.8 * n_liq),
                         sharex=False, sharey=False)
if n_liq == 1:
    axes = axes[np.newaxis, :]
if n_vol == 1:
    axes = axes[:, np.newaxis]

for row, liq in enumerate(liquids):
    c = COLORS.get(liq, "#555555")
    s_liq = summ[summ["liquid_name"] == liq]
    d_liq = det[det["liquid_name"] == liq]

    for col, vol in enumerate(volumes):
        ax = axes[row, col]

        s_row = s_liq[s_liq["target_volume_uL"] == vol]
        if s_row.empty:
            ax.set_visible(False)
            continue
        s_row = s_row.iloc[0]

        p1_ov   = s_row["point1_overaspirate_uL"]
        p1_mean = s_row["point1_mean_uL"]
        p2_ov   = s_row["point2_overaspirate_uL"]
        p2_mean = s_row["point2_mean_uL"]
        p3_ov   = s_row["optimal_overaspirate_uL"]
        p3_mean = s_row["point3_mean_uL"]
        target  = s_row["target_volume_uL"]

        # individual replicate scatter
        for pt_label, marker, alpha, zorder in [
            ("point_1",           "o", 0.35, 2),
            ("point_2",           "s", 0.35, 2),
            ("point_3_validation","*", 0.45, 4),
        ]:
            reps = d_liq[(d_liq["target_volume_uL"] == vol) & (d_liq["point"] == pt_label)]
            if reps.empty:
                continue
            ms = 60 if marker != "*" else 140
            ax.scatter(reps["overaspirate_uL"], reps["measured_volume_uL"],
                       color=c, marker=marker, s=ms, alpha=alpha, zorder=zorder)

        # interpolation line extended slightly beyond P1-P2 range
        ov_lo = min(p1_ov, p2_ov)
        ov_hi = max(p1_ov, p2_ov)
        pad   = (ov_hi - ov_lo) * 0.4 if ov_hi > ov_lo else 1.0
        x_line = np.linspace(ov_lo - pad, ov_hi + pad, 50)
        if abs(p2_ov - p1_ov) > 1e-6:
            slope = (p2_mean - p1_mean) / (p2_ov - p1_ov)
            y_line = p1_mean + slope * (x_line - p1_ov)
            ax.plot(x_line, y_line, "--", color=c, lw=1.2, alpha=0.5, zorder=1)

        # mean markers (larger, filled)
        ax.scatter([p1_ov], [p1_mean], color=c, marker="o", s=120, zorder=5,
                   edgecolors="k", linewidths=0.8, label="P1 mean")
        ax.scatter([p2_ov], [p2_mean], color=c, marker="s", s=120, zorder=5,
                   edgecolors="k", linewidths=0.8, label="P2 mean")
        ax.scatter([p3_ov], [p3_mean], color=c, marker="*", s=280, zorder=6,
                   edgecolors="k", linewidths=0.6, label="P3 (optimal)")

        # label each mean
        for ov, mean, lbl in [(p1_ov, p1_mean, "P1"), (p2_ov, p2_mean, "P2"), (p3_ov, p3_mean, "P3")]:
            ax.annotate(lbl, (ov, mean), textcoords="offset points",
                        xytext=(5, 5), fontsize=7.5, color=c, fontweight="bold")

        # target line
        x_all = [p1_ov, p2_ov, p3_ov]
        x_span = max(x_all) - min(x_all)
        pad2 = max(x_span * 0.5, 2.0)
        ax.axhline(target, color="k", linestyle=":", lw=1.2, alpha=0.6, zorder=0)
        ax.text(min(x_all) - pad2 * 0.9, target, f"target\n{target:.0f}uL",
                fontsize=7, va="center", color="k", alpha=0.7)

        # deviation annotation at P3
        dev = p3_mean - target
        ax.annotate(f"Δ={dev:+.2f}uL\n({s_row['point3_deviation_pct']:.2f}%)",
                    (p3_ov, p3_mean), textcoords="offset points",
                    xytext=(8, -22), fontsize=7.5, color="k",
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.5))

        ax.set_xlabel("Overaspirate (uL)", fontsize=8)
        ax.set_ylabel("Measured Volume (uL)", fontsize=8)
        ax.set_title(f"{liq} — {vol:.0f}uL", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

        if col == 0 and row == 0:
            ax.legend(fontsize=7, loc="upper left")

fig.suptitle(
    "Two-Point Overaspirate Calibration: Steps P1 → P2 → P3\n"
    "○=P1 baseline  ■=P2 bracket  ★=P3 validation  ···=target",
    fontsize=12, fontweight="bold"
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = OUT_DIR / "two_point_steps_visualization.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

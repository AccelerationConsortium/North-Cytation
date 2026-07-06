"""
Compare two-point calibration datasets:
  OLD  : water/ethanol from 20260602_183433  (25/50/75/100 uL)
         glycerol       from 20260604_121212  (25/50/75/100 uL)
  NEW  : glycerol/glycerol_dye/water_dye/ethanol_dye from 20260610_111558  (70/100/150 uL)
         water          from 20260610_124837  (70/100/150 uL)
         ethanol        from 20260610_133418  (70/100/150 uL)

Plots (saved to output/two_point_analysis/):
  1. p3_accuracy_new6.png        -- P3 validation accuracy for all 6 new liquids
  2. overaspirate_vs_volume_<liquid>.png  -- dye vs plain per base liquid (2 stacked subplots)
  3. crossover_100uL.png         -- 100 uL comparison: old / new-plain / new-dye

Derived columns added to all dataframes:
  excess_uL       = point3_mean_uL - target_volume_uL
  corrected_ov_uL = optimal_overaspirate_uL - excess_uL
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ── File paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

OLD_FILES = {
    "20260602_183433": ROOT / "output/two_point_series_demo_summary_20260602_183433.csv",
    "20260604_121212": ROOT / "output/two_point_series_demo_summary_20260604_121212.csv",
}
NEW_FILES = {
    "20260610_111558": ROOT / "output/two_point_series_demo_summary_20260610_111558.csv",
    "20260610_124837": ROOT / "output/two_point_series_demo_summary_20260610_124837.csv",
    "20260610_133418": ROOT / "output/two_point_series_demo_summary_20260610_133418.csv",
}

OUT_DIR = ROOT / "output/two_point_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Load & tag data ────────────────────────────────────────────────────────────
def _load(path, dataset_tag):
    df = pd.read_csv(path)
    df["dataset"] = dataset_tag
    return df

old_df = pd.concat([_load(p, "old") for p in OLD_FILES.values()], ignore_index=True)
new_df = pd.concat([_load(p, "new") for p in NEW_FILES.values()], ignore_index=True)

# Ethanol file has one bad DMSO row (ethanol 70, ~13.6% dev) — keep last occurrence per (label, target)
new_df = (new_df
          .sort_index()
          .drop_duplicates(subset=["label", "target_volume_uL"], keep="last"))

# Filter old to glycerol/water/ethanol only (drop PVA_DMSO, DMSO)
old_df = old_df[old_df["liquid_name"].isin(["glycerol", "water", "ethanol"])].copy()

# ── Derived columns ───────────────────────────────────────────────────────────
for df in (old_df, new_df):
    df["excess_uL"] = df["point3_mean_uL"] - df["target_volume_uL"]
    df["corrected_ov_uL"] = df["optimal_overaspirate_uL"] - df["excess_uL"]

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "glycerol":     "#c0392b",
    "glycerol_dye": "#e8857d",
    "water":        "#2980b9",
    "water_dye":    "#85c1e9",
    "ethanol":      "#27ae60",
    "ethanol_dye":  "#82e0aa",
}

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — P3 accuracy for the 6 new liquids
# ─────────────────────────────────────────────────────────────────────────────
NEW_LABELS_ORDER = ["glycerol_dye", "water_dye", "ethanol_dye", "glycerol", "water", "ethanol"]
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for ax, lbl in zip(axes, NEW_LABELS_ORDER):
    grp = new_df[new_df["label"] == lbl].sort_values("target_volume_uL")
    if grp.empty:
        ax.set_visible(False)
        continue

    t = grp["target_volume_uL"].values
    m = grp["point3_mean_uL"].values
    devs = grp["point3_deviation_pct"].values
    color = COLORS.get(lbl, "#555555")

    lim = (min(t.min(), m.min()) * 0.92, max(t.max(), m.max()) * 1.06)
    ax.plot(lim, lim, "k--", lw=1, alpha=0.35, label="ideal")
    ax.scatter(t, m, color=color, s=90, zorder=5)
    for ti, mi, di in zip(t, m, devs):
        ax.annotate(f"{di:.2f}%", (ti, mi), textcoords="offset points",
                    xytext=(6, 4), fontsize=8.5, color=color)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Target (uL)"); ax.set_ylabel("P3 Measured (uL)")
    ax.set_title(lbl, fontsize=11, fontweight="bold", color=color)
    ax.legend(fontsize=8)

fig.suptitle("Two-Point Calibration — P3 Validation Accuracy (New Run, 70/100/150 uL)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
p1 = OUT_DIR / f"p3_accuracy_new6_{TIMESTAMP}.png"
fig.savefig(p1, dpi=150)
plt.close(fig)
print(f"Saved: {p1}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Overaspirate vs target volume: dye vs plain (new data only)
#          3 figures (one per base liquid), each with 2 stacked subplots
# ─────────────────────────────────────────────────────────────────────────────
BASE_LIQUIDS = ["glycerol", "water", "ethanol"]

for base in BASE_LIQUIDS:
    plain_lbl = base
    dye_lbl   = f"{base}_dye"

    plain = new_df[new_df["label"] == plain_lbl].sort_values("target_volume_uL")
    dye   = new_df[new_df["label"] == dye_lbl].sort_values("target_volume_uL")

    if plain.empty and dye.empty:
        continue

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 9), sharex=True)
    fig.suptitle(f"{base.capitalize()} — Overaspirate vs Target Volume (new run)",
                 fontsize=13, fontweight="bold")

    for subset, lbl, color, marker in [
        (plain, plain_lbl, COLORS[base],           "o"),
        (dye,   dye_lbl,   COLORS[f"{base}_dye"],  "s"),
    ]:
        if subset.empty:
            continue
        t  = subset["target_volume_uL"].values
        ov = subset["optimal_overaspirate_uL"].values
        co = subset["corrected_ov_uL"].values

        ax_top.plot(t, ov, color=color, marker=marker, lw=1.8, ms=8, label=lbl)
        ax_bot.plot(t, co, color=color, marker=marker, lw=1.8, ms=8, label=lbl)

    ax_top.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax_bot.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax_top.set_ylabel("Optimal overaspirate (uL)")
    ax_bot.set_ylabel("Corrected overaspirate\n(optimal_ov - excess, uL)")
    ax_bot.set_xlabel("Target volume (uL)")
    ax_top.set_title("Optimal overaspirate", fontsize=10)
    ax_bot.set_title("Corrected overaspirate  [removes P3 residual error]", fontsize=10)
    ax_top.legend(); ax_bot.legend()
    fig.tight_layout()

    p2 = OUT_DIR / f"overaspirate_vs_volume_{base}_{TIMESTAMP}.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved: {p2}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — 100 uL crossover: old / new-plain / new-dye
# ─────────────────────────────────────────────────────────────────────────────
TARGET_VOL = 100.0

old_100 = old_df[old_df["target_volume_uL"] == TARGET_VOL]
new_100 = new_df[new_df["target_volume_uL"] == TARGET_VOL]

rows = []
for base in BASE_LIQUIDS:
    # old
    r = old_100[old_100["liquid_name"] == base]
    if not r.empty:
        rows.append({"liquid": base, "source": "old (25-100uL run)",
                     "optimal_ov": r.iloc[0]["optimal_overaspirate_uL"],
                     "corrected_ov": r.iloc[0]["corrected_ov_uL"]})
    # new plain
    r = new_100[new_100["label"] == base]
    if not r.empty:
        rows.append({"liquid": base, "source": "new plain (70-150uL run)",
                     "optimal_ov": r.iloc[0]["optimal_overaspirate_uL"],
                     "corrected_ov": r.iloc[0]["corrected_ov_uL"]})
    # new dye
    r = new_100[new_100["label"] == f"{base}_dye"]
    if not r.empty:
        rows.append({"liquid": base, "source": f"new {base}_dye (70-150uL run)",
                     "optimal_ov": r.iloc[0]["optimal_overaspirate_uL"],
                     "corrected_ov": r.iloc[0]["corrected_ov_uL"]})

cdf = pd.DataFrame(rows)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle(f"100 uL Crossover Comparison — old vs new-plain vs new-dye",
             fontsize=13, fontweight="bold")

source_colors = {
    "old (25-100uL run)":      "#555555",
    "new plain (70-150uL run)":"#2c7bb6",
    "new glycerol_dye (70-150uL run)": COLORS["glycerol_dye"],
    "new water_dye (70-150uL run)":    COLORS["water_dye"],
    "new ethanol_dye (70-150uL run)":  COLORS["ethanol_dye"],
}

x = np.arange(len(BASE_LIQUIDS))
width = 0.25
source_order = ["old (25-100uL run)", "new plain (70-150uL run)"]
source_order += [f"new {b}_dye (70-150uL run)" for b in BASE_LIQUIDS]

for ax, metric, title in [
    (ax1, "optimal_ov",   "Optimal overaspirate at 100 uL"),
    (ax2, "corrected_ov", "Corrected overaspirate at 100 uL\n(optimal_ov - P3 excess)"),
]:
    offset = -(len(BASE_LIQUIDS) - 1) * width / 2
    plotted_sources = []
    for i, base in enumerate(BASE_LIQUIDS):
        sub = cdf[cdf["liquid"] == base].reset_index(drop=True)
        for j, row in sub.iterrows():
            src = row["source"]
            color = source_colors.get(src, "#aaaaaa")
            bar = ax.bar(i + j * width - width, row[metric],
                         width * 0.85, color=color, alpha=0.85,
                         label=src if src not in plotted_sources else "")
            if src not in plotted_sources:
                plotted_sources.append(src)
            ax.text(i + j * width - width, row[metric] + 0.15,
                    f"{row[metric]:.1f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(x - width / 2)
    ax.set_xticklabels([b.capitalize() for b in BASE_LIQUIDS])
    ax.set_ylabel("Overaspirate (uL)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper left")

fig.tight_layout()
p3 = OUT_DIR / f"crossover_100uL_{TIMESTAMP}.png"
fig.savefig(p3, dpi=150)
plt.close(fig)
print(f"Saved: {p3}")

print("\nDone. All plots saved to:", OUT_DIR)

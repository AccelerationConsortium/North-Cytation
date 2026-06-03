"""
Analyze two-point calibration results:
1. P3 accuracy per liquid (target vs measured, R2)
2. P1 50uL drift vs best SBT trial for PVA_DMSO
Saves plots to output/two_point_analysis/
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SUMMARY = "output/two_point_series_demo_summary_20260602_183433.csv"
DETAILS = "output/two_point_series_demo_details_20260602_183433.csv"

OUT_DIR = Path("output/two_point_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# SBT best trial results for 50uL comparison
SBT_TRIAL_RESULTS = {
    "PVA_DMSO": "calibration_modular_v2/output/run_1779906029_PVA_DMSO/trial_results.csv",
    "DMSO":     "calibration_modular_v2/output/run_1779912579_DMSO/trial_results.csv",
    "water":    "calibration_modular_v2/output/run_1779739005_water/trial_results.csv",
    "ethanol":  "calibration_modular_v2/output/run_1780412080_ethanol/trial_results.csv",
}

df = pd.read_csv(SUMMARY)
det = pd.read_csv(DETAILS)

# ── 1. P3 calibration accuracy ──────────────────────────────────────────────
print("=" * 65)
print("P3 CALIBRATION ACCURACY  (P3 = validation at optimal overaspirate)")
print("=" * 65)
print(f"{'Liquid':<14} {'Target':>8} {'P3 Mean':>10} {'Error':>9} {'Dev%':>8}")
print("-" * 55)
for _, r in df.iterrows():
    err = r["point3_mean_uL"] - r["target_volume_uL"]
    print(f"{r['liquid_name']:<14} {r['target_volume_uL']:>8.1f}uL "
          f"{r['point3_mean_uL']:>9.2f}uL {err:>+8.2f}uL {r['point3_deviation_pct']:>7.2f}%")

print()
print("R2 per liquid (P3 measured vs target, across all volumes):")
for liq, grp in df.groupby("liquid_name"):
    t = grp["target_volume_uL"].values
    m = grp["point3_mean_uL"].values
    ss_res = np.sum((m - t) ** 2)
    ss_tot = np.sum((m - np.mean(m)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    mae = np.mean(np.abs(m - t))
    max_dev = grp["point3_deviation_pct"].max()
    print(f"  {liq:<14}  R2={r2:.6f}  MAE={mae:.2f}uL  max_dev={max_dev:.2f}%")

# ── 2. P1 50uL drift vs SBT best trial ─────────────────────────────────────
print()
print("=" * 65)
print("P1 50uL DRIFT  (current run P1 mean vs SBT best trial mean)")
print("This compares pipetting at the original overaspirate (no correction)")
print("=" * 65)

p1_50 = det[(det["point"] == "point_1") & (det["target_volume_uL"] == 50.0)]

for liq in df["liquid_name"].unique():
    rows = p1_50[p1_50["liquid_name"] == liq]
    if rows.empty:
        continue

    current_mean = rows["measured_volume_uL"].mean()
    current_std  = rows["measured_volume_uL"].std()
    current_ov   = rows["overaspirate_uL"].iloc[0]

    sbt_path = SBT_TRIAL_RESULTS.get(liq)
    if sbt_path:
        try:
            trials = pd.read_csv(sbt_path)
            # best trial = lowest absolute deviation at 50uL with >=2 measurements
            t50 = trials[(trials["volume_target_ul"] == 50.0) & (trials["measurement_count"] >= 2)]
            if not t50.empty:
                best = t50.loc[t50["deviation_pct"].abs().idxmin()]
                best_mean_ul = best["volume_measured_ul"]
                best_ov_ul   = best["calibration_overaspirate_vol"] * 1000
                drift = current_mean - best_mean_ul
                print(f"\n  {liq}")
                print(f"    SBT best   : {best_mean_ul:.2f}uL  (ov={best_ov_ul:.2f}uL, dev={best['deviation_pct']:.2f}%)")
                print(f"    P1 current : {current_mean:.2f}uL ± {current_std:.2f}uL  (ov={current_ov:.2f}uL)")
                print(f"    Drift      : {drift:+.2f}uL ({drift/best_mean_ul*100:+.2f}%)")
            else:
                print(f"\n  {liq}: no 50uL trials in SBT results")
        except Exception as e:
            print(f"\n  {liq}: could not load SBT results ({e})")
    else:
        print(f"\n  {liq}: no SBT path configured")
        print(f"    P1 current : {current_mean:.2f}uL +/- {current_std:.2f}uL  (ov={current_ov:.2f}uL)")

# ── 3. R2 plot: target vs P3 measured, one panel per liquid ─────────────────
liquids = df["liquid_name"].unique()
colors = {"PVA_DMSO": "#e07b39", "DMSO": "#5b8fd4", "water": "#4caf7d", "ethanol": "#b05fc9"}

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()

for ax, liq in zip(axes, liquids):
    grp = df[df["liquid_name"] == liq].sort_values("target_volume_uL")
    t = grp["target_volume_uL"].values
    m = grp["point3_mean_uL"].values
    devs = grp["point3_deviation_pct"].values

    ss_res = np.sum((m - t) ** 2)
    ss_tot = np.sum((m - np.mean(m)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    c = colors.get(liq, "#555555")
    ax.plot([t.min(), t.max()], [t.min(), t.max()], "k--", lw=1, alpha=0.4, label="ideal")
    ax.scatter(t, m, color=c, s=80, zorder=5)
    for ti, mi, di in zip(t, m, devs):
        ax.annotate(f"{di:.2f}%", (ti, mi), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=c)

    ax.set_xlabel("Target (uL)")
    ax.set_ylabel("P3 Measured (uL)")
    ax.set_title(f"{liq}  |  R²={r2:.5f}")
    ax.legend(fontsize=8)

fig.suptitle("Two-Point Calibration: P3 Validation Accuracy", fontsize=13, fontweight="bold")
fig.tight_layout()
p = OUT_DIR / "p3_accuracy_r2.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"\nSaved: {p}")

# ── 4. Drift bar chart ───────────────────────────────────────────────────────
drift_data = []
p1_50 = det[(det["point"] == "point_1") & (det["target_volume_uL"] == 50.0)]

for liq in liquids:
    rows = p1_50[p1_50["liquid_name"] == liq]
    if rows.empty:
        continue
    current_mean = rows["measured_volume_uL"].mean()
    current_std  = rows["measured_volume_uL"].std()
    sbt_path = SBT_TRIAL_RESULTS.get(liq)
    if not sbt_path:
        continue
    try:
        trials = pd.read_csv(sbt_path)
        t50 = trials[(trials["volume_target_ul"] == 50.0) & (trials["measurement_count"] >= 2)]
        if t50.empty:
            continue
        best = t50.loc[t50["deviation_pct"].abs().idxmin()]
        drift_data.append({
            "liquid": liq,
            "sbt_mean": best["volume_measured_ul"],
            "p1_mean": current_mean,
            "p1_std": current_std,
            "drift_ul": current_mean - best["volume_measured_ul"],
        })
    except Exception:
        pass

if drift_data:
    dd = pd.DataFrame(drift_data)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(dd))
    w = 0.35
    bars_sbt = ax.bar(x - w/2, dd["sbt_mean"], w, label="SBT best trial", color="#aac4e8", edgecolor="k", linewidth=0.7)
    bars_p1  = ax.bar(x + w/2, dd["p1_mean"],  w, label="P1 today (same ov)", color="#f4a96a", edgecolor="k", linewidth=0.7,
                      yerr=dd["p1_std"], capsize=4)
    ax.axhline(50, color="k", linestyle="--", lw=1, alpha=0.5, label="target 50uL")

    for bar, row in zip(bars_p1, dd.itertuples()):
        drift = row.drift_ul
        ax.annotate(f"{drift:+.2f}uL\n({drift/row.sbt_mean*100:+.1f}%)",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8,
                    color="darkred" if abs(drift) > 2 else "darkgreen")

    ax.set_xticks(x)
    ax.set_xticklabels(dd["liquid"])
    ax.set_ylabel("Measured volume (uL)")
    ax.set_title("50uL P1 Drift: Current Run vs SBT Best Trial\n(same overaspirate, no two-point correction)")
    ax.legend()
    fig.tight_layout()
    p = OUT_DIR / "p1_50uL_drift.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")

# ── 5. Deviation % heatmap across volumes and liquids ───────────────────────
pivot = df.pivot(index="liquid_name", columns="target_volume_uL", values="point3_deviation_pct")
tol_map = {25.0: 3.0, 50.0: 3.0, 75.0: 2.0, 100.0: 2.0, 150.0: 2.0}

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=4)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{int(v)}uL" for v in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
for i, liq in enumerate(pivot.index):
    for j, vol in enumerate(pivot.columns):
        val = pivot.loc[liq, vol]
        tol = tol_map.get(vol, 2.0)
        marker = "OK" if val <= tol else "FAIL"
        ax.text(j, i, f"{val:.2f}%\n{marker}", ha="center", va="center", fontsize=8,
                color="white" if val > 2.5 else "black")
plt.colorbar(im, ax=ax, label="P3 deviation %")
ax.set_title("P3 Deviation % by Liquid and Volume")
fig.tight_layout()
p = OUT_DIR / "p3_deviation_heatmap.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

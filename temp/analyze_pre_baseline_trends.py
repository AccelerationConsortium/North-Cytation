import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

THRESHOLD = 0.003
CAMPAIGNS = ["200uL", "1000uL"]
BASE = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign"


def analyze_pre_baseline(trace_path):
    """Returns (slope_g_per_s, net_change_g, monotonicity_frac, n_points) for baseline_pre phase."""
    try:
        mdf = pd.read_csv(trace_path)
        pre = mdf[mdf["phase"] == "baseline_pre"]["mass_g"].values
        if len(pre) < 3:
            return None
        t = np.arange(len(pre))
        slope = np.polyfit(t, pre, 1)[0]
        net = pre[-1] - pre[0]
        diffs = np.diff(pre)
        mono = np.sum(diffs > 0) / len(diffs)
        return slope, net, mono, len(pre)
    except Exception:
        return None


results = []

for c in CAMPAIGNS:
    df = pd.read_csv(os.path.join(BASE, c, "incremental_results.csv"))
    ok = df[df["status"] == "ok"].copy()
    ok["date"] = ok["timestamp"].astype(str).str[:8]
    flagged = ok[(ok["pre_baseline_std"] > THRESHOLD) | (ok["post_baseline_std"] > THRESHOLD)].copy()
    mass_dir = os.path.join(BASE, c, "mass_time_data")

    for _, row in flagged.iterrows():
        fpath = os.path.join(mass_dir, str(row["mass_data_file"]))
        res = analyze_pre_baseline(fpath)
        if res is None:
            continue
        slope, net, mono, n = res
        results.append({
            "campaign": c,
            "date": row["date"],
            "row_index": int(row["row_index"]),
            "pre_std": row["pre_baseline_std"],
            "slope_g_per_sample": slope,
            "net_change_g": net,
            "monotonicity": mono,
            "n_pre_points": n,
            "mass_data_file": row["mass_data_file"],
        })

rdf = pd.DataFrame(results)

# Classify: "trending_up" = net > 5mg AND monotonicity >= 0.6
rdf["pattern"] = "noisy"
rdf.loc[(rdf["net_change_g"] > 0.005) & (rdf["monotonicity"] >= 0.6), "pattern"] = "trending_up"
rdf.loc[(rdf["net_change_g"] > 0.005) & (rdf["monotonicity"] < 0.6), "pattern"] = "spike_then_flat"

print("=== Pattern summary ===")
print(rdf.groupby(["campaign", "pattern"]).size().to_string())

print("\n=== By date (trending_up count vs total flagged) ===")
for c in CAMPAIGNS:
    sub = rdf[rdf["campaign"] == c]
    by_date = sub.groupby("date")["pattern"].apply(lambda x: (x == "trending_up").sum())
    total = sub.groupby("date").size()
    print(f"\n{c}:")
    for d in sorted(sub["date"].unique()):
        up = by_date.get(d, 0)
        tot = total.get(d, 0)
        print(f"  {d}: {up}/{tot} trending_up")

# Save the classified list
out_path = os.path.join(BASE, "flagged_unstable_classified.csv")
rdf.to_csv(out_path, index=False)
print(f"\nSaved classified list to {out_path}")

# ---- Plot sample traces: up to 16 trending_up examples ----
trending = rdf[rdf["pattern"] == "trending_up"].head(16)
fig, axes = plt.subplots(4, 4, figsize=(16, 10))
axes = axes.flatten()

for i, (_, row) in enumerate(trending.iterrows()):
    c = row["campaign"]
    mass_dir = os.path.join(BASE, c, "mass_time_data")
    fpath = os.path.join(mass_dir, row["mass_data_file"])
    try:
        mdf = pd.read_csv(fpath)
        pre = mdf[mdf["phase"] == "baseline_pre"]
        ax = axes[i]
        ax.plot(pre["time_relative"].values, pre["mass_g"].values, "b.-", markersize=4)
        ax.set_title(f"{c} idx={row['row_index']}\n{row['date']} net={row['net_change_g']*1000:.1f}mg", fontsize=7)
        ax.set_xlabel("t (s)", fontsize=6)
        ax.set_ylabel("g", fontsize=6)
        ax.tick_params(labelsize=6)
    except Exception:
        axes[i].set_visible(False)

for j in range(i + 1, 16):
    axes[j].set_visible(False)

plt.suptitle("Sample trending_up pre-baselines (drip signature)", fontsize=11)
plt.tight_layout()
plot_path = os.path.join(BASE, "flagged_trending_up_samples.png")
plt.savefig(plot_path, dpi=130)
print(f"Saved sample plot to {plot_path}")
plt.show()

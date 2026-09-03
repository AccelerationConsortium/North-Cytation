import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign"
THRESHOLD = 0.003

param_cols = ["aspirate_speed", "dispense_speed", "aspirate_wait_time", "dispense_wait_time",
              "pre_asp_air_vol_uL", "blowout_vol_uL", "overaspirate_vol_uL"]

# Load classified list from earlier analysis
classified = pd.read_csv(os.path.join(BASE, "flagged_unstable_classified.csv"))
trending_up = set(zip(classified[classified["pattern"] == "trending_up"]["campaign"],
                      classified[classified["pattern"] == "trending_up"]["row_index"].astype(int)))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, c in enumerate(["200uL", "1000uL"]):
    df = pd.read_csv(os.path.join(BASE, c, "incremental_results.csv"))
    ok = df[df["status"] == "ok"].copy()
    ok["row_index"] = ok["row_index"].astype(int)
    ok["is_trending_up"] = ok["row_index"].apply(lambda r: (c, r) in trending_up)

    # Assign condition group via merge with source file
    src = pd.read_csv(rf"C:\Users\Imaging Controller\Desktop\utoronto_demo\inputs\Glycerin_Sobol_Parameters_{c}.csv")
    src["row_index"] = src.index
    ok = ok.merge(src[param_cols + ["row_index"]].rename(columns={p: p+"_src" for p in param_cols}),
                  on="row_index", how="left")

    # Group by condition (param values from source)
    src_param_cols = [p + "_src" for p in param_cols]
    ok["condition_id"] = ok.groupby(src_param_cols).ngroup()

    # For each condition: compute within-group stats
    results = []
    for cid, grp in ok.groupby("condition_id"):
        acc = grp["accuracy_pct"].values
        has_flagged = grp["is_trending_up"].any()
        n_flagged = grp["is_trending_up"].sum()
        flagged_vals = grp.loc[grp["is_trending_up"], "accuracy_pct"].values
        clean_vals = grp.loc[~grp["is_trending_up"], "accuracy_pct"].values

        spread = acc.max() - acc.min()
        median_acc = np.median(acc)

        # Is the flagged rep the highest?
        flagged_is_max = False
        flagged_excess = np.nan
        if has_flagged and len(clean_vals) > 0:
            flagged_is_max = flagged_vals.max() == acc.max()
            flagged_excess = flagged_vals.max() - np.median(clean_vals)

        results.append({
            "condition_id": cid,
            "has_flagged": has_flagged,
            "n_flagged": n_flagged,
            "spread": spread,
            "median_acc": median_acc,
            "flagged_is_max": flagged_is_max,
            "flagged_excess": flagged_excess,
            "n_reps": len(grp),
        })

    rdf = pd.DataFrame(results)
    full_reps = rdf[rdf["n_reps"] == 5]  # only complete 5-rep conditions

    flagged_conds = full_reps[full_reps["has_flagged"]]
    clean_conds = full_reps[~full_reps["has_flagged"]]

    print(f"=== {c} (5-rep conditions only) ===")
    print(f"Conditions with >=1 trending_up rep: {len(flagged_conds)}")
    print(f"Conditions with no flagged reps: {len(clean_conds)}")
    print(f"Flagged conditions - median within-group spread: {flagged_conds['spread'].median():.2f}%")
    print(f"Clean conditions   - median within-group spread: {clean_conds['spread'].median():.2f}%")
    pct_is_max = flagged_conds["flagged_is_max"].mean() * 100
    print(f"Flagged rep is highest of 5: {pct_is_max:.0f}% of flagged conditions")
    print(f"Median excess of flagged rep over clean siblings median: {flagged_conds['flagged_excess'].median():.2f}%")
    print()

    # Plot 1: spread distribution
    ax1 = axes[i][0]
    ax1.hist(clean_conds["spread"], bins=60, alpha=0.6, color="steelblue", label="clean", density=True)
    ax1.hist(flagged_conds["spread"], bins=60, alpha=0.6, color="tomato", label="has flagged rep", density=True)
    ax1.set_title(f"{c} - Within-group spread (max-min accuracy)")
    ax1.set_xlabel("Spread (%)")
    ax1.legend()

    # Plot 2: flagged excess over clean siblings
    ax2 = axes[i][1]
    excess = flagged_conds["flagged_excess"].dropna()
    ax2.hist(excess, bins=50, color="tomato", alpha=0.8)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1)
    ax2.axvline(excess.median(), color="red", linestyle="-", linewidth=1.5,
                label=f"Median: {excess.median():.1f}%")
    ax2.set_title(f"{c} - Flagged rep excess over clean sibling median")
    ax2.set_xlabel("Excess accuracy (%)")
    ax2.legend()

plt.tight_layout()
out = os.path.join(BASE, "replicate_spread_analysis.png")
plt.savefig(out, dpi=140)
print(f"Saved to {out}")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign"
THRESHOLD = 0.003
BAD_DATES = {"20260506", "20260507", "20260508", "20260513", "20260514",
             "20260522", "20260608", "20260619", "20260709", "20260728"}

param_cols = ["aspirate_speed", "dispense_speed", "aspirate_wait_time", "dispense_wait_time",
              "pre_asp_air_vol_uL", "blowout_vol_uL", "overaspirate_vol_uL"]

classified = pd.read_csv(os.path.join(BASE, "flagged_unstable_classified.csv"))
trending_up = set(zip(classified[classified["pattern"] == "trending_up"]["campaign"],
                      classified[classified["pattern"] == "trending_up"]["row_index"].astype(int)))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for i, c in enumerate(["200uL", "1000uL"]):
    df = pd.read_csv(os.path.join(BASE, c, "incremental_results.csv"))
    ok = df[df["status"] == "ok"].copy()
    ok["row_index"] = ok["row_index"].astype(int)
    ok["date"] = ok["timestamp"].astype(str).str[:8]
    ok["is_trending_up"] = ok["row_index"].apply(lambda r: (c, r) in trending_up)
    ok["is_bad_day"] = ok["date"].isin(BAD_DATES)

    # Label each row
    def label(row):
        if row["is_trending_up"]:
            return "flagged"
        elif row["is_bad_day"]:
            return "bad_day_unflagged"
        else:
            return "clean"
    ok["group"] = ok.apply(label, axis=1)

    # Merge condition IDs from source
    src = pd.read_csv(rf"C:\Users\Imaging Controller\Desktop\utoronto_demo\inputs\Glycerin_Sobol_Parameters_{c}.csv")
    src["row_index"] = src.index
    ok = ok.merge(src[param_cols + ["row_index"]].rename(columns={p: p+"_src" for p in param_cols}),
                  on="row_index", how="left")
    src_param_cols = [p + "_src" for p in param_cols]
    ok["condition_id"] = ok.groupby(src_param_cols).ngroup()

    # For each row, compute its rank within its condition group (1=lowest, 5=highest)
    ok["rank_in_group"] = ok.groupby("condition_id")["accuracy_pct"].rank(method="average")
    ok["excess_over_median"] = ok.groupby("condition_id")["accuracy_pct"].transform(
        lambda x: x - x.median()
    )

    group_counts = ok["group"].value_counts()
    print(f"=== {c} ===")
    print(group_counts.to_string())

    for grp in ["flagged", "bad_day_unflagged", "clean"]:
        sub = ok[ok["group"] == grp]
        pct_highest = (sub["rank_in_group"] >= 4.5).mean() * 100  # rank 5 of 5
        median_excess = sub["excess_over_median"].median()
        print(f"  {grp:25s}: n={len(sub):5d}, is_highest={pct_highest:.0f}%, median_excess={median_excess:+.2f}%")
    print()

    # Plot rank distributions
    colors = {"flagged": "tomato", "bad_day_unflagged": "orange", "clean": "steelblue"}
    for j, grp in enumerate(["flagged", "bad_day_unflagged", "clean"]):
        ax = axes[i][j]
        sub = ok[ok["group"] == grp]["rank_in_group"]
        ax.hist(sub, bins=[0.5,1.5,2.5,3.5,4.5,5.5], color=colors[grp], alpha=0.85, edgecolor="white")
        ax.set_title(f"{c} - {grp}\n(n={len(sub)})", fontsize=9)
        ax.set_xlabel("Rank within condition (5=highest)")
        ax.set_ylabel("Count")
        # Add expected flat line for uniform distribution
        expected = len(sub) / 5
        ax.axhline(expected, color="black", linestyle="--", linewidth=1, label="uniform")
        ax.set_xticks([1,2,3,4,5])
        ax.legend(fontsize=7)

plt.suptitle("Rank within 5-replicate group by measurement category", fontsize=12)
plt.tight_layout()
out = os.path.join(BASE, "rank_distribution_by_group.png")
plt.savefig(out, dpi=140)
print(f"Saved to {out}")
plt.show()

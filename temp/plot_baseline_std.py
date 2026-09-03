import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Scale Baseline Stability - Pre & Post Dispense STD", fontsize=14)

for i, c in enumerate(["200uL", "1000uL"]):
    path = rf"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign\{c}\incremental_results.csv"
    df = pd.read_csv(path)
    # Use only ok rows for clean distribution
    ok = df[df["status"] == "ok"]

    pre = ok["pre_baseline_std"].dropna()
    post = ok["post_baseline_std"].dropna()

    ax_pre = axes[i][0]
    ax_post = axes[i][1]

    ax_pre.hist(pre, bins=80, color="steelblue", edgecolor="none", alpha=0.85)
    ax_pre.set_title(f"{c} - Pre-dispense STD")
    ax_pre.set_xlabel("STD (g)")
    ax_pre.set_ylabel("Count")
    ax_pre.axvline(pre.median(), color="red", linestyle="--", linewidth=1, label=f"Median: {pre.median():.4f}")
    ax_pre.axvline(pre.quantile(0.95), color="orange", linestyle="--", linewidth=1, label=f"95th pct: {pre.quantile(0.95):.4f}")
    ax_pre.legend(fontsize=8)

    ax_post.hist(post, bins=80, color="seagreen", edgecolor="none", alpha=0.85)
    ax_post.set_title(f"{c} - Post-dispense STD")
    ax_post.set_xlabel("STD (g)")
    ax_post.set_ylabel("Count")
    ax_post.axvline(post.median(), color="red", linestyle="--", linewidth=1, label=f"Median: {post.median():.4f}")
    ax_post.axvline(post.quantile(0.95), color="orange", linestyle="--", linewidth=1, label=f"95th pct: {post.quantile(0.95):.4f}")
    ax_post.legend(fontsize=8)

    print(f"{c} pre_baseline_std  - median: {pre.median():.5f}, 95th: {pre.quantile(0.95):.5f}, max: {pre.max():.5f}")
    print(f"{c} post_baseline_std - median: {post.median():.5f}, 95th: {post.quantile(0.95):.5f}, max: {post.max():.5f}")

plt.tight_layout()
out = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign\baseline_std_histograms.png"
plt.savefig(out, dpi=150)
print(f"\nSaved to {out}")
plt.show()

"""Offline boundary-percentile sensitivity sweep. No GP fitting needed."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from recommenders.synthetic_nd import build_landscape
from recommenders.systematic_compare_nd import grid_for_d, grad_mag_nd
import glob, os

TOP_FRACS = {
    "turbidity_600": [0.03, 0.05, 0.10, 0.15, 0.20],
    "ratio":         [0.05, 0.10, 0.20, 0.30, 0.40],
}
NEAR_R = 0.06
RUNS = [
    (3, "baseline_3d_384_b14_pool50k"),
    (4, "baseline_4d_384_b14_pool50k"),
]

def build_cache(d, landscape):
    n = grid_for_d(d)
    g = np.linspace(0, 1, n)
    mesh = np.meshgrid(*([g] * d), indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    Y = landscape.evaluate(pts)
    F1 = Y[:, 0].reshape(*([n] * d))
    F2 = Y[:, 1].reshape(*([n] * d))
    cache = {}
    for name, F, fracs in [("turbidity_600", F1, TOP_FRACS["turbidity_600"]),
                            ("ratio",         F2, TOP_FRACS["ratio"])]:
        G = grad_mag_nd(F)
        cache[name] = {}
        for frac in fracs:
            thresh = np.percentile(G, 100 * (1 - frac))
            mask = G >= thresh
            bnd_pts = np.stack([m[mask] for m in mesh], axis=1)
            cache[name][frac] = {"mask": mask, "bnd_pts": bnd_pts, "ng": n}
    return cache

results = []
for dim, tag in RUNS:
    print(f"Building {dim}D cache...")
    cache = build_cache(dim, build_landscape(dim))
    input_cols = [f"x{i+1}" for i in range(dim)]
    base = f"recommenders/test_outputs/{tag}"
    for fpath in sorted(glob.glob(f"{base}/data_*.csv")):
        fname = os.path.basename(fpath)
        parts = fname.replace("data_", "").replace(".csv", "").rsplit("_seed", 1)
        rec = parts[0].replace("_sobol27", "").replace("_sobol64", "")
        seed = int(parts[1])
        picks = pd.read_csv(fpath)[input_cols].values
        for out_name, fracs_cache in cache.items():
            for frac, c in fracs_cache.items():
                ng = c["ng"]
                idx = tuple(
                    np.clip((picks[:, i] * (ng - 1)).round().astype(int), 0, ng - 1)
                    for i in range(dim)
                )
                hit = float(c["mask"][idx].mean())
                d_bnd, _ = cKDTree(picks).query(c["bnd_pts"], k=1)
                recall = float((d_bnd <= NEAR_R).mean())
                results.append({"rec": rec, "dim": dim, "output": out_name,
                                 "top_frac": frac, "hit_rate": hit, "recall": recall})

df = pd.DataFrame(results)
g = df.groupby(["rec", "dim", "output", "top_frac"])[["hit_rate", "recall"]].mean().reset_index()

colors = {"BayesianContrast": "tab:blue", "Simplex": "tab:orange", "Random": "tab:red"}
labels = {"BayesianContrast": "Bayesian", "Simplex": "Simplex", "Random": "Random"}
out_labels = {"turbidity_600": "Turbidity", "ratio": "Ratio"}
metric_labels = {
    "hit_rate": "Hit Rate",
    "recall": f"Surface Recall (r={NEAR_R})",
}
current_frac = {"turbidity_600": 10, "ratio": 30}

fig, axs = plt.subplots(2, 4, figsize=(16, 8))
col_spec = [(3, "hit_rate"), (3, "recall"), (4, "hit_rate"), (4, "recall")]

for row, out in enumerate(["turbidity_600", "ratio"]):
    for col, (dim, metric) in enumerate(col_spec):
        ax = axs[row][col]
        sub = g[(g.output == out) & (g.dim == dim)]
        for rec, color in colors.items():
            rg = sub[sub.rec == rec].sort_values("top_frac")
            ax.plot(rg["top_frac"] * 100, rg[metric], "o-", color=color,
                    label=labels[rec], lw=2, markersize=5)
            if metric == "hit_rate":
                # hit_rate == top_frac for random by construction — show diagonal
                pass
        ax.axvline(current_frac[out], color="gray", lw=1.2, ls=":",
                   alpha=0.8, label=f"current ({current_frac[out]}%)")
        ax.set_xlabel("Boundary top_frac (%)", fontsize=9)
        if col in (0, 2):
            ax.set_ylabel(metric_labels[metric], fontsize=9)
        ax.set_title(f"{dim}D — {out_labels[out]}\n{metric_labels[metric]}",
                     fontsize=9, fontweight="bold")
        ax.grid(alpha=0.3)
        if metric == "hit_rate":
            ax.set_ylim(0, 1.05)
        else:
            yvals = sub[metric].values
            ymin = max(0.0, float(yvals.min()) - 0.05)
            ymax = min(1.0, float(yvals.max()) + 0.05)
            if ymax - ymin < 0.1:
                pad = 0.05
                ymin = max(0.0, ymin - pad)
                ymax = min(1.0, ymax + pad)
            ax.set_ylim(ymin, ymax)
        if row == 0 and col == 0:
            ax.legend(fontsize=8)

fig.suptitle(
    "Sensitivity to Boundary Percentile Definition\n"
    "(384 picks, batch 14, 3 seeds, recall radius=0.06)",
    fontsize=12, fontweight="bold",
)
fig.tight_layout()
out_path = "recommenders/test_outputs/boundary_percentile_sensitivity.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Wrote {out_path}")

# Extra recall-only figure with tighter axes for reporting readability.
fig2, axs2 = plt.subplots(1, 4, figsize=(16, 4))
recall_cols = [("turbidity_600", 3), ("turbidity_600", 4), ("ratio", 3), ("ratio", 4)]
for i, (out, dim) in enumerate(recall_cols):
    ax = axs2[i]
    sub = g[(g.output == out) & (g.dim == dim)]
    for rec, color in colors.items():
        rg = sub[sub.rec == rec].sort_values("top_frac")
        ax.plot(rg["top_frac"] * 100, rg["recall"], "o-", color=color,
                label=labels[rec], lw=2, markersize=5)
    ax.axvline(current_frac[out], color="gray", lw=1.2, ls=":",
               alpha=0.8, label=f"current ({current_frac[out]}%)")
    yvals = sub["recall"].values
    ymin = max(0.0, float(yvals.min()) - 0.05)
    ymax = min(1.0, float(yvals.max()) + 0.05)
    if ymax - ymin < 0.1:
        pad = 0.05
        ymin = max(0.0, ymin - pad)
        ymax = min(1.0, ymax + pad)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Boundary top_frac (%)", fontsize=9)
    if i == 0:
        ax.set_ylabel(f"Surface Recall (r={NEAR_R})", fontsize=9)
    ax.set_title(f"{dim}D — {out_labels[out]}", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

fig2.suptitle("Boundary Percentile Sensitivity (Recall Only, Zoomed)",
              fontsize=12, fontweight="bold")
fig2.tight_layout()
out_path2 = "recommenders/test_outputs/boundary_percentile_sensitivity_recall_zoom.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Wrote {out_path2}")

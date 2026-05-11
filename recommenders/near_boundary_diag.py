"""
Diagnostic: for each pick, measure the distance to the nearest
boundary cell (top-X% of |grad y|) for both outputs. Reports how many
picks are 'near' the boundary at several radius thresholds.

Usage:
  python -m recommenders.near_boundary_diag
  python -m recommenders.near_boundary_diag --folder 3d_corner_spike_grid3
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recommenders.metrics import TOP_FRAC, define_boundary_3d
from recommenders.synthetic_3d_visualize import f1_turbidity, f2_ratio


GRID_N = 50
INPUT_COLS = ["x1", "x2", "x3"]
OUT_BASE = os.path.join(os.path.dirname(__file__), "test_outputs")
RADII = [0.02, 0.04, 0.06, 0.08, 0.10]


def gt_grids(n=GRID_N):
    g = np.linspace(0.0, 1.0, n)
    X1, X2, X3 = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)
    F1 = f1_turbidity(pts).reshape(n, n, n)
    F2 = f2_ratio(pts).reshape(n, n, n)
    return (g, g, g), F1, F2


def boundary_points(F, axes, top_frac):
    mask, _ = define_boundary_3d(F, axes, top_frac)
    g1, g2, g3 = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.stack([g1[mask], g2[mask], g3[mask]], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default="3d_corner_spike_grid3")
    args = p.parse_args()
    folder = os.path.join(OUT_BASE, args.folder)
    print(f"Folder: {folder}")

    axes, F_turb, F_ratio = gt_grids(GRID_N)
    bnd_pts = {
        "turbidity_600": boundary_points(F_turb, axes, TOP_FRAC["turbidity_600"]),
        "ratio":         boundary_points(F_ratio, axes, TOP_FRAC["ratio"]),
    }
    bnd_trees = {k: cKDTree(v) for k, v in bnd_pts.items()}
    print(f"Boundary cell counts: "
          f"turb={len(bnd_pts['turbidity_600'])}, "
          f"ratio={len(bnd_pts['ratio'])}")

    rows = []
    csv_files = [f for f in os.listdir(folder)
                 if f.startswith("all_data_") and f.endswith(".csv")]
    if not csv_files:
        print(f"No all_data_*.csv files found in {folder}")
        return

    fig, axs = plt.subplots(len(csv_files), 2,
                            figsize=(11, 3.2 * len(csv_files)),
                            squeeze=False)
    for r, fn in enumerate(sorted(csv_files)):
        label = fn[len("all_data_"):-len(".csv")]
        df = pd.read_csv(os.path.join(folder, fn))
        X = df[INPUT_COLS].values
        for c, out_name in enumerate(["turbidity_600", "ratio"]):
            d, _ = bnd_trees[out_name].query(X, k=1)
            ax = axs[r, c]
            ax.hist(d, bins=30, color="steelblue")
            for R in RADII:
                ax.axvline(R, ls="--", c="k", alpha=0.4, lw=0.7)
            within_counts = {R: int((d <= R).sum()) for R in RADII}
            ax.set_title(
                f"{label} -> {out_name}\n"
                f"n_within: " + " ".join(
                    f"R={R}:{within_counts[R]}" for R in RADII
                ),
                fontsize=8,
            )
            ax.set_xlabel("dist to nearest boundary cell")
            ax.set_ylabel("# picks")
            row = {"recommender": label, "output": out_name,
                   "n_picks": int(len(d)),
                   "median_dist": float(np.median(d)),
                   "mean_dist": float(d.mean()),
                   "min_dist": float(d.min())}
            for R in RADII:
                row[f"n_within_{R}"] = within_counts[R]
                row[f"frac_within_{R}"] = within_counts[R] / len(d)
            rows.append(row)

    fig.suptitle(f"Distance from picks to nearest boundary cell ({args.folder})",
                 fontsize=11)
    fig.tight_layout()
    out_png = os.path.join(folder, "near_boundary_diag.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_png}")

    df_summary = pd.DataFrame(rows)
    out_csv = os.path.join(folder, "near_boundary_summary.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()

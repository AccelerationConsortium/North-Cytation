I do """Generate 3D pick visualizations from systematic_compare data CSVs.

For each (recommender, init, seed), produces:
  - 4-view 3D scatter of picks colored by iteration, overlaid on
    boundary-cell point cloud (one figure per output)

Run after recommenders.systematic_compare has produced data_*.csv.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from recommenders.metrics import TOP_FRAC, define_boundary_3d
from recommenders.synthetic_3d_visualize import f1_turbidity, f2_ratio
from scipy.spatial import cKDTree


GRID_N = 50
INPUT_COLS = ["x1", "x2", "x3"]
OUTPUTS = ["turbidity_600", "ratio"]
VIEWS = [(25, 35), (25, 125), (25, 215), (60, 35)]
NEAR_R = 0.04
FOLDER = os.path.join(os.path.dirname(__file__), "test_outputs",
                      "systematic_compare")


def build_boundary():
    g = np.linspace(0, 1, GRID_N)
    X1, X2, X3 = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)
    F = {
        "turbidity_600": f1_turbidity(pts).reshape(GRID_N, GRID_N, GRID_N),
        "ratio":         f2_ratio(pts).reshape(GRID_N, GRID_N, GRID_N),
    }
    bnd = {}
    for name, Ff in F.items():
        m, _ = define_boundary_3d(Ff, (g, g, g), TOP_FRAC[name])
        bp = np.stack([X1[m], X2[m], X3[m]], axis=1)
        bnd[name] = {"pts": bp, "mask": m, "tree": cKDTree(bp)}
    return bnd


def plot_picks(data, bnd, label, init, seed, out_name, out_path,
               mode="all"):
    """mode: 'all' | 'on' (exact cell) | 'near' (within NEAR_R)."""
    picks = data[INPUT_COLS].values
    iters = data["iteration"].values
    bnd_pts = bnd["pts"]
    if mode == "on":
        ng = bnd["mask"].shape[0]
        ix = np.clip((picks[:, 0] * (ng - 1)).round().astype(int),
                     0, ng - 1)
        iy = np.clip((picks[:, 1] * (ng - 1)).round().astype(int),
                     0, ng - 1)
        iz = np.clip((picks[:, 2] * (ng - 1)).round().astype(int),
                     0, ng - 1)
        keep = bnd["mask"][ix, iy, iz]
        marker_size = 60
        tag = f"on-boundary ({keep.sum()}/{len(keep)})"
    elif mode == "near":
        d, _ = bnd["tree"].query(picks, k=1)
        keep = d <= NEAR_R
        marker_size = 55
        tag = f"within R={NEAR_R} ({keep.sum()}/{len(keep)})"
    else:
        keep = np.ones(len(picks), dtype=bool)
        marker_size = 22
        tag = f"all picks ({len(picks)})"
    picks = picks[keep]
    iters = iters[keep]

    if len(bnd_pts) > 4000:
        sel = np.random.default_rng(0).choice(len(bnd_pts), 4000,
                                              replace=False)
        bnd_pts = bnd_pts[sel]
    fig = plt.figure(figsize=(13, 11))
    sc = None
    for k, (elev, azim) in enumerate(VIEWS):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        ax.scatter(bnd_pts[:, 0], bnd_pts[:, 1], bnd_pts[:, 2],
                   s=2, c="lightgray", alpha=0.18,
                   label=f"{out_name} boundary")
        if len(picks) > 0:
            sc = ax.scatter(picks[:, 0], picks[:, 1], picks[:, 2],
                            c=iters, cmap="autumn", s=marker_size,
                            edgecolor="black", linewidth=0.4,
                            label="picks")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
        ax.set_title(f"elev={elev} azim={azim}")
        ax.view_init(elev=elev, azim=azim)
        if k == 0:
            ax.legend(loc="upper left", fontsize=8)
    if sc is not None:
        cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
        fig.colorbar(sc, cax=cbar_ax, label="iteration")
    fig.suptitle(f"{label} | {init} seed{seed} | {out_name} | {tag}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    bnd = build_boundary()
    files = sorted(glob.glob(os.path.join(FOLDER, "data_*.csv")))
    print(f"Found {len(files)} run files")
    pat = re.compile(r"data_(?P<rec>[^_]+(?:_[A-Z]+)?)_"
                     r"(?P<init>[a-z0-9]+)_seed(?P<seed>\d+)\.csv")
    for f in files:
        m = pat.search(os.path.basename(f))
        if not m:
            print(f"  skip (name): {f}")
            continue
        rec, init, seed = m.group("rec"), m.group("init"), m.group("seed")
        data = pd.read_csv(f)
        for out_name in OUTPUTS:
            for mode, suffix in [("all", ""), ("on", "_onbnd"),
                                 ("near", "_near")]:
                out_path = os.path.join(
                    FOLDER,
                    f"picks3d_{rec}_{init}_seed{seed}_"
                    f"{out_name}{suffix}.png")
                plot_picks(data, bnd[out_name], rec, init, seed,
                           out_name, out_path, mode=mode)


if __name__ == "__main__":
    main()

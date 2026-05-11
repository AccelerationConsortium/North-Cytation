"""
Boundary-definition diagnostic.

For each requested 2D dataset, evaluate ground truth on a fine grid and
visualize candidate boundary masks at several |grad y| percentiles.
The goal is to eyeball what percentile (if any) corresponds to "the
boundary" we want algorithms to trace.

No picks, no algorithms, no metrics. Just visualization.

Boundary-presence rule (per output): if max(y) - min(y) < MIN_RANGE
the output is treated as having no boundary and panels are annotated.

Datasets supported:
  circle2d                    synthetic, smooth circle r=0.3
  CHAPS_BDDAC, BDDAC_NaLS,    real-data GP posteriors (must already be
  DSS_BZT, DSS_CTAB,          built; see build_ground_truth_gp.py)
  SDS_TTAB

Run:
  python -m recommenders.boundary_diagnostic
  python -m recommenders.boundary_diagnostic --names circle2d,DSS_BZT
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from recommenders.build_ground_truth_gp import load_ground_truth_gp
from recommenders.test_gradient_transition_recommender import circle2d_outputs


N_GRID = 150
PERCENTILES = [30, 20, 10, 5, 2]   # top X% of |grad y|
MIN_RANGE = 0.1                    # below this => "no boundary"
OUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs",
                       "boundary_diag")

REAL_NAMES = ["CHAPS_BDDAC", "BDDAC_NaLS", "DSS_BZT",
              "DSS_CTAB", "SDS_TTAB"]
SYNTH_NAMES = ["circle2d"]
ALL_NAMES = SYNTH_NAMES + REAL_NAMES


# -----------------------------------------------------------------
# Ground-truth field producers. All return (X, Y, fields, axis_labels).
# -----------------------------------------------------------------

def gt_circle2d():
    g = np.linspace(0.0, 1.0, N_GRID)
    G0, G1 = np.meshgrid(g, g, indexing="ij")
    R = np.empty_like(G0)
    T = np.empty_like(G0)
    for i in range(N_GRID):
        for j in range(N_GRID):
            r, t = circle2d_outputs(np.array([G0[i, j], G1[i, j]]))
            R[i, j] = r
            T[i, j] = t
    return G0, G1, {"ratio": R, "turbidity_600": T}, ("x0", "x1")


def gt_real(name):
    models, bundle = load_ground_truth_gp(name)
    lo, hi = bundle["log_bounds"]
    g_a = np.linspace(lo[0], hi[0], N_GRID)
    g_b = np.linspace(lo[1], hi[1], N_GRID)
    G_A, G_B = np.meshgrid(g_a, g_b, indexing="ij")
    pts = torch.tensor(np.stack([G_A.ravel(), G_B.ravel()], axis=1),
                       dtype=torch.double)
    fields = {}
    output_names = ["ratio", "turbidity_600"]
    with torch.no_grad():
        for m, name_out in zip(models, output_names):
            mu = m.posterior(pts).mean.squeeze(-1).numpy()
            fields[name_out] = mu.reshape(N_GRID, N_GRID)
    return G_A, G_B, fields, ("log10(surf_A_mm)", "log10(surf_B_mm)")


def get_ground_truth(name):
    if name == "circle2d":
        return gt_circle2d()
    if name in REAL_NAMES:
        return gt_real(name)
    raise ValueError(f"Unknown dataset: {name}")


# -----------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------

def grad_magnitude(F, X, Y):
    """Central-difference |grad F| using grid spacing in display coords."""
    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]
    gx, gy = np.gradient(F, dx, dy)
    return np.sqrt(gx ** 2 + gy ** 2)


def plot_dataset(name, out_path):
    X, Y, fields, (xlabel, ylabel) = get_ground_truth(name)
    output_names = list(fields.keys())
    n_rows = len(output_names)
    n_cols = len(PERCENTILES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.4 * n_cols, 3.4 * n_rows),
                             squeeze=False)
    for r, out in enumerate(output_names):
        F = fields[out]
        rng = float(F.max() - F.min())
        has_boundary = rng >= MIN_RANGE
        if has_boundary:
            G = grad_magnitude(F, X, Y)
        for c, pct in enumerate(PERCENTILES):
            ax = axes[r, c]
            im = ax.pcolormesh(X, Y, F, cmap="viridis", shading="auto")
            if has_boundary:
                thresh = np.percentile(G, 100 - pct)
                mask = G >= thresh
                xs = X[mask]
                ys = Y[mask]
                ax.scatter(xs, ys, s=2, c="red", alpha=0.6,
                           edgecolors="none")
                title = (f"{out} | top {pct}% |grad|\n"
                         f"range={rng:.3f}")
            else:
                title = (f"{out} | NO BOUNDARY\n"
                         f"range={rng:.3f} < {MIN_RANGE}")
            ax.set_title(title, fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel(xlabel, fontsize=8)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Boundary diagnostic: {name}  (grid {N_GRID}x{N_GRID})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", default=",".join(ALL_NAMES),
                   help="Comma-separated dataset names. "
                        f"Available: {','.join(ALL_NAMES)}")
    args = p.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    for name in names:
        try:
            plot_dataset(name, os.path.join(OUT_DIR, f"{name}.png"))
        except FileNotFoundError as e:
            print(f"SKIP {name}: GP bundle not found ({e}). "
                  f"Build with: python -m recommenders.build_ground_truth_gp "
                  f"--csv <path> --name {name}")


if __name__ == "__main__":
    main()

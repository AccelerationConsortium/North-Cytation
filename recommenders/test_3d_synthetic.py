"""
3D head-to-head test: BayesianContrast vs GradientUCB on synthetic
3-input/2-output boundary problems.

Tests both algorithms' ability to:
  (a) HIT the boundary  (frac_in_top_quartile_grad)
  (b) COVER the boundary uniformly (max & median gap from boundary cells
      to the nearest pick; nearest-neighbor spacing among on-boundary picks)

Boundaries
----------
- sphere3d : ||x - 0.5||_2 = 0.3                (closed convex surface)
- saddle3d : x[2] = 0.5 + 0.5*(x[0]^2 - x[1]^2) (open curved surface)

Each boundary defines TWO outputs (a smoothed step on the signed distance
with two different sharpnesses) so the recommenders see a multi-output
problem analogous to (ratio, turbidity).

Usage
-----
  python -m recommenders.test_3d_synthetic              # both surfaces
  python -m recommenders.test_3d_synthetic --boundary sphere3d
  python -m recommenders.test_3d_synthetic --reduce max
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import cKDTree

from recommenders.bayesian_transition_recommender import (
    BayesianTransitionRecommender,
)
from recommenders.gradient_transition_recommender import (
    GradientTransitionRecommender,
)


SEED = 0
Q_BATCH = 8
N_ITERATIONS = 16
N_INIT_PER_DIM = 4         # 4^3 = 64 initial points
GRID_N = 50                # 50^3 = 125k cells for coverage metric
INPUT_COLS = ["x1", "x2", "x3"]
OUTPUT_COLS = ["f1", "f2"]
OUT_ROOT = os.path.join(os.path.dirname(__file__), "test_outputs", "3d")


# ------------------------------------------------------------------ #
# Synthetic boundaries (signed distance)
# ------------------------------------------------------------------ #

def sphere3d(X):
    """h(x) = ||x - 0.5|| - 0.3. h<0 inside sphere, h>0 outside."""
    X = np.atleast_2d(X)
    return np.linalg.norm(X - 0.5, axis=1) - 0.3


def saddle3d(X):
    """h(x) = x[2] - (0.5 + 0.5*(x[0]^2 - x[1]^2)). h>0 above surface."""
    X = np.atleast_2d(X)
    return X[:, 2] - (0.5 + 0.5 * (X[:, 0] ** 2 - X[:, 1] ** 2))


BOUNDARIES = {"sphere3d": sphere3d, "saddle3d": saddle3d}


def evaluate(X, boundary_func, eps1=0.04, eps2=0.10):
    """Two-output smoothed step on signed distance.

    f1: sharp transition (steepness 1/eps1)
    f2: softer transition (steepness 1/eps2)
    """
    h = boundary_func(X)
    f1 = 1.0 / (1.0 + np.exp(-h / eps1))
    f2 = 1.0 / (1.0 + np.exp(-h / eps2))
    return np.stack([f1, f2], axis=1)


# ------------------------------------------------------------------ #
# Ground-truth gradient field on a regular 3D grid
# ------------------------------------------------------------------ #

def ground_truth_grid(boundary_func, n=GRID_N):
    """Compute |grad f1| on a regular n^3 grid in [0,1]^3.

    Uses central differences. Returns (G, GRAD_MAG, in_top_mask).
    """
    g = np.linspace(0, 1, n)
    X1, X2, X3 = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)
    Y = evaluate(pts, boundary_func)[:, 0].reshape(n, n, n)  # f1
    g1, g2, g3 = np.gradient(Y)
    GRAD_MAG = np.sqrt(g1 ** 2 + g2 ** 2 + g3 ** 2)
    thresh = np.quantile(GRAD_MAG, 0.75)
    return g, GRAD_MAG, GRAD_MAG >= thresh


# ------------------------------------------------------------------ #
# Initial design + recommender runner
# ------------------------------------------------------------------ #

def initial_grid(boundary_func, n=N_INIT_PER_DIM):
    g = np.linspace(0.05, 0.95, n)  # avoid corners exactly
    X1, X2, X3 = np.meshgrid(g, g, g, indexing="ij")
    X = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)
    Y = evaluate(X, boundary_func)
    df = pd.DataFrame({
        "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2],
        "f1": Y[:, 0], "f2": Y[:, 1],
        "well_type": "experiment", "iteration": 0,
    })
    return df


def run_recommender(rec, boundary_func, label):
    data = initial_grid(boundary_func)
    for it in range(1, N_ITERATIONS + 1):
        print(f"\n=== {label} iter {it}/{N_ITERATIONS} ===")
        recs = rec.get_recommendations(data, n_points=Q_BATCH, iteration=it)
        X_new = recs[INPUT_COLS].values
        Y_new = evaluate(X_new, boundary_func)
        new = recs.copy()
        new["f1"] = Y_new[:, 0]
        new["f2"] = Y_new[:, 1]
        new["well_type"] = "experiment"
        new["iteration"] = it
        data = pd.concat([data, new], ignore_index=True)
    return data


# ------------------------------------------------------------------ #
# Coverage metrics in 3D
# ------------------------------------------------------------------ #

def coverage_metrics(data, GRAD_MAG, n=GRID_N):
    """3D coverage metrics in [0,1]^3."""
    picks = data[INPUT_COLS].values  # already in [0,1]^3
    # boundary cells
    g = np.linspace(0, 1, n)
    X1, X2, X3 = np.meshgrid(g, g, g, indexing="ij")
    thresh = np.quantile(GRAD_MAG, 0.75)
    bnd_mask = GRAD_MAG >= thresh
    bnd_pts = np.stack([X1[bnd_mask], X2[bnd_mask], X3[bnd_mask]], axis=1)

    # which picks land in the top-quartile region?
    ix = np.clip((picks * (n - 1)).astype(int), 0, n - 1)
    in_top = GRAD_MAG[ix[:, 0], ix[:, 1], ix[:, 2]] >= thresh

    # gap: nearest-pick distance for every boundary cell
    pick_tree = cKDTree(picks)
    nearest_dist, _ = pick_tree.query(bnd_pts, k=1)

    metrics = {
        "n_total": int(len(data)),
        "frac_in_top_quartile_grad": float(in_top.mean()),
        "n_picks_on_boundary": int(in_top.sum()),
        "boundary_coverage_max": float(nearest_dist.max()),
        "boundary_coverage_median": float(np.median(nearest_dist)),
    }
    on_bnd = picks[in_top]
    if len(on_bnd) >= 2:
        nn, _ = cKDTree(on_bnd).query(on_bnd, k=2)
        nn_dists = nn[:, 1]
        metrics["on_boundary_min_pairwise"] = float(nn_dists.min())
        metrics["on_boundary_median_nn"] = float(np.median(nn_dists))
    else:
        metrics["on_boundary_min_pairwise"] = float("nan")
        metrics["on_boundary_median_nn"] = float("nan")
    return metrics


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #

def plot_3d_picks(data, GRAD_MAG, label, title, out_path, n=GRID_N):
    """3D scatter of picks + boundary-cell scatter (top-quartile |grad|)."""
    g = np.linspace(0, 1, n)
    X1, X2, X3 = np.meshgrid(g, g, g, indexing="ij")
    thresh = np.quantile(GRAD_MAG, 0.75)
    bnd_mask = GRAD_MAG >= thresh
    # subsample boundary cells for plotting
    bnd_pts = np.stack([X1[bnd_mask], X2[bnd_mask], X3[bnd_mask]], axis=1)
    if len(bnd_pts) > 5000:
        idx = np.random.default_rng(0).choice(len(bnd_pts), 5000,
                                              replace=False)
        bnd_pts = bnd_pts[idx]

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(bnd_pts[:, 0], bnd_pts[:, 1], bnd_pts[:, 2],
               c="lightgray", s=2, alpha=0.15, label="true boundary cells")
    picks = data[INPUT_COLS].values
    sc = ax.scatter(picks[:, 0], picks[:, 1], picks[:, 2],
                    c=data["iteration"].values, cmap="autumn",
                    s=30, edgecolor="black", linewidth=0.4, label="picks")
    plt.colorbar(sc, ax=ax, label="iteration", shrink=0.6)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_slices(data, GRAD_MAG, label, title, out_path, n=GRID_N):
    """Three orthogonal mid-plane slices through |grad| with picks overlaid
    (only picks within +/- 0.1 of the slice plane shown on each)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    g = np.linspace(0, 1, n)
    mid = n // 2
    picks = data[INPUT_COLS].values
    iters = data["iteration"].values

    slices = [
        (GRAD_MAG[mid, :, :], 0, "x1=0.5", ("x2", "x3")),
        (GRAD_MAG[:, mid, :], 1, "x2=0.5", ("x1", "x3")),
        (GRAD_MAG[:, :, mid], 2, "x3=0.5", ("x1", "x2")),
    ]
    for ax, (S, axis, slabel, axlabels) in zip(axes, slices):
        im = ax.pcolormesh(g, g, S.T, cmap="inferno", shading="auto")
        plt.colorbar(im, ax=ax, label="|grad f1|", shrink=0.85)
        # picks within +-0.1 of slice plane
        near = np.abs(picks[:, axis] - 0.5) < 0.1
        if near.any():
            other = [i for i in range(3) if i != axis]
            ax.scatter(picks[near, other[0]], picks[near, other[1]],
                       c=iters[near], cmap="autumn",
                       s=28, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(axlabels[0]); ax.set_ylabel(axlabels[1])
        ax.set_title(f"{slabel} ({near.sum()} picks)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def run_one_boundary(name, reduce_mode):
    print(f"\n{'#' * 60}")
    print(f"# 3D BOUNDARY: {name}  (reduce={reduce_mode})")
    print(f"{'#' * 60}")
    boundary_func = BOUNDARIES[name]
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    g, GRAD_MAG, _ = ground_truth_grid(boundary_func)

    summary = []
    for label, build in [
        ("BayesianContrast", lambda: BayesianTransitionRecommender(
            input_columns=INPUT_COLS, output_columns=OUTPUT_COLS,
            log_transform_inputs=False,
        )),
        ("GradientUCB", lambda: GradientTransitionRecommender(
            input_columns=INPUT_COLS, output_columns=OUTPUT_COLS,
            log_transform_inputs=False,
            multi_output_reduce=reduce_mode,
        )),
    ]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        rec = build()
        torch.manual_seed(SEED); np.random.seed(SEED)
        data = run_recommender(rec, boundary_func, label)
        data.to_csv(os.path.join(out_dir, f"all_data_{label}.csv"),
                    index=False)
        m = coverage_metrics(data, GRAD_MAG)
        m["recommender"] = label
        summary.append(m)
        plot_3d_picks(
            data, GRAD_MAG, label,
            title=f"{name}: {label}\n"
                  f"hit={m['frac_in_top_quartile_grad']:.2f}  "
                  f"max gap={m['boundary_coverage_max']:.3f}  "
                  f"med gap={m['boundary_coverage_median']:.3f}",
            out_path=os.path.join(out_dir, f"3d_picks_{label}.png"),
        )
        plot_slices(
            data, GRAD_MAG, label,
            title=f"{name}: {label} (mid-plane slices)",
            out_path=os.path.join(out_dir, f"slices_{label}.png"),
        )

    summary_df = pd.DataFrame(summary)[
        ["recommender", "n_total", "frac_in_top_quartile_grad",
         "n_picks_on_boundary",
         "boundary_coverage_max", "boundary_coverage_median",
         "on_boundary_median_nn", "on_boundary_min_pairwise"]]
    summary_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"),
                      index=False)
    print(f"\n=== {name} coverage summary ===")
    print(summary_df.to_string(index=False))
    return summary_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boundary", default="all",
                   choices=["all"] + list(BOUNDARIES.keys()))
    p.add_argument("--reduce", default="standardized_sum",
                   choices=["sum", "max", "standardized_sum"])
    args = p.parse_args()
    targets = list(BOUNDARIES.keys()) if args.boundary == "all" \
        else [args.boundary]
    all_summaries = []
    for name in targets:
        all_summaries.append(run_one_boundary(name, args.reduce))
    if len(all_summaries) > 1:
        print("\n\n### Combined summary ###")
        combined = pd.concat(
            [df.assign(boundary=n) for n, df in zip(targets, all_summaries)],
            ignore_index=True)
        combined.to_csv(os.path.join(OUT_ROOT, "all_metrics_summary.csv"),
                        index=False)
        print(combined.to_string(index=False))


if __name__ == "__main__":
    main()

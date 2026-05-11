"""
3D head-to-head test on the corner_spike_cube synthetic dataset.

Inputs : x1, x2, x3 in [0, 1]
Outputs:
  turbidity_600 - sharp cone spike from corner (0,0,0), value 0.04..1.0
  ratio         - smooth rounded cube at (0.5,0.5,0.5), value 0.70..0.90

Recommenders compared (no Triangle in 3D):
  BayesianContrast, GradientUCB

Init: 3x3x3 grid (27 points), then N_ITERATIONS x Q_BATCH per recommender.
Metrics: recommenders.metrics 3D variants.
Plots:
  - 4-angle 3D scatter of picks colored by iteration, with
    boundary-cell point cloud as background (one figure per recommender,
    per output)
  - 3 mid-plane slice grid showing |grad y| with picks-near-slice overlaid

Usage:
  python -m recommenders.test_3d_corner_spike
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

from recommenders.bayesian_transition_recommender import (
    BayesianTransitionRecommender,
)
from recommenders.gradient_transition_recommender import (
    GradientTransitionRecommender,
)
from recommenders.levelset_transition_recommender import (
    LevelSetTransitionRecommender,
)
from recommenders.delaunay_simplex_recommender import (
    DelaunaySimplexTransitionRecommender,
)
from recommenders.metrics import (
    TOP_FRAC, define_boundary_3d, boundary_metrics_3d, clumping_metrics,
    grad_magnitude_3d,
)
from recommenders.synthetic_3d_visualize import (
    f1_turbidity, f2_ratio,
    SPIKE_AXIS, SPIKE_R0, SPIKE_LENGTH,
)


SEED = 0
Q_BATCH = 8
N_ITERATIONS = 16                  # default; overridden when --n-points is set
GRID_N = 50                        # 50^3 ground-truth grid for metrics
NEAR_R = 0.04                      # tolerance for surface recall/precision
INPUT_COLS = ["x1", "x2", "x3"]
OUTPUT_COLS = ["turbidity_600", "ratio"]
OUT_BASE = os.path.join(os.path.dirname(__file__), "test_outputs")


def evaluate(X):
    """Two-output simulator. X: (N,3) in [0,1]."""
    return np.stack([f1_turbidity(X), f2_ratio(X)], axis=1)


def initial_design(init_mode):
    """Build the initial design.

    init_mode:
      'grid3'  -> 3x3x3 = 27 points (linspace 0.05..0.95)
      'grid4'  -> 4x4x4 = 64 points (linspace 0.05..0.95)
      'sobol'  -> 27 quasi-random Sobol points (matches grid3 count)
      'sobol64'-> 64 Sobol points (matches grid4 count)
    """
    if init_mode in ("grid3", "grid4"):
        n = 3 if init_mode == "grid3" else 4
        g = np.linspace(0.05, 0.95, n)
        X1, X2, X3 = np.meshgrid(g, g, g, indexing="ij")
        X = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)
    elif init_mode in ("sobol", "sobol64"):
        from scipy.stats.qmc import Sobol
        n_pts = 27 if init_mode == "sobol" else 64
        sampler = Sobol(d=3, scramble=True, seed=SEED)
        X = sampler.random(n_pts)
    else:
        raise ValueError(f"unknown init_mode={init_mode}")
    Y = evaluate(X)
    return pd.DataFrame({
        "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2],
        "turbidity_600": Y[:, 0], "ratio": Y[:, 1],
        "well_type": "experiment", "iteration": 0,
    })


def run_recommender(rec, label, init_mode, n_iters):
    data = initial_design(init_mode)
    for it in range(1, n_iters + 1):
        print(f"=== {label} iter {it}/{n_iters} ===")
        recs = rec.get_recommendations(data, n_points=Q_BATCH, iteration=it)
        X_new = recs[INPUT_COLS].values
        Y_new = evaluate(X_new)
        new = recs.copy()
        new["turbidity_600"] = Y_new[:, 0]
        new["ratio"] = Y_new[:, 1]
        new["well_type"] = "experiment"
        new["iteration"] = it
        data = pd.concat([data, new], ignore_index=True)
    return data


# -----------------------------------------------------------------
# Per-iteration metrics (cheap; reuses cached boundary KD-trees)
# -----------------------------------------------------------------

def _boundary_pts(F, axes, top_frac):
    mask, _ = define_boundary_3d(F, axes, top_frac)
    g1, g2, g3 = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.stack([g1[mask], g2[mask], g3[mask]], axis=1), mask


def compute_iter_metrics(data, axes, fields, label, near_r=NEAR_R):
    """For each iteration t, compute cumulative metrics on picks where
    iteration <= t. Returns long-format rows.
    """
    from scipy.spatial import cKDTree
    rows = []
    bnd_cache = {}
    for name, F in fields.items():
        bnd_pts, mask = _boundary_pts(F, axes, TOP_FRAC[name])
        bnd_cache[name] = (bnd_pts, mask, cKDTree(bnd_pts))

    iters_sorted = sorted(data["iteration"].unique())
    for t in iters_sorted:
        sub = data[data["iteration"] <= t]
        picks = sub[INPUT_COLS].values
        n = len(picks)
        for name, F in fields.items():
            bnd_pts, mask, tree = bnd_cache[name]
            # surface precision: fraction of picks within near_r of surface
            d_pick, _ = tree.query(picks, k=1)
            surf_prec = float((d_pick <= near_r).mean())
            # surface recall: fraction of surface cells with a pick within R
            tree_picks = cKDTree(picks)
            d_bnd, _ = tree_picks.query(bnd_pts, k=1)
            surf_recall = float((d_bnd <= near_r).mean())
            # hit_rate: picks landing exactly in a boundary cell
            ng = len(axes[0])
            ix = np.clip((picks[:, 0] * (ng - 1)).round().astype(int),
                         0, ng - 1)
            iy = np.clip((picks[:, 1] * (ng - 1)).round().astype(int),
                         0, ng - 1)
            iz = np.clip((picks[:, 2] * (ng - 1)).round().astype(int),
                         0, ng - 1)
            hit_rate = float(mask[ix, iy, iz].mean())
            # max F achieved
            F_pick = (f1_turbidity(picks) if name == "turbidity_600"
                      else f2_ratio(picks))
            max_F = float(F_pick.max())
            rows.append({
                "recommender": label, "iteration": int(t),
                "n_points": int(n), "output": name,
                "surf_recall": surf_recall,
                "surf_precision": surf_prec,
                "hit_rate": hit_rate,
                "max_F": max_F,
            })
    return rows


def plot_per_iter(df_iter, out_path):
    """Lines: x=n_points, y=metric, one line per recommender, one panel
    per (output, metric)."""
    metrics = [
        ("surf_recall", "surface recall (frac of surface cells with\n"
                        f"pick within R={NEAR_R})"),
        ("surf_precision", f"surface precision (frac picks within R={NEAR_R})"),
        ("hit_rate", "hit_rate (picks in boundary cell)"),
        ("max_F", "max F achieved"),
    ]
    outputs = sorted(df_iter["output"].unique())
    fig, axs = plt.subplots(len(metrics), len(outputs),
                            figsize=(5.2 * len(outputs), 3.6 * len(metrics)),
                            squeeze=False)
    recs = sorted(df_iter["recommender"].unique())
    cmap = plt.get_cmap("tab10")
    for i, (mkey, mlabel) in enumerate(metrics):
        for j, out_name in enumerate(outputs):
            ax = axs[i][j]
            for k, rec in enumerate(recs):
                sub = df_iter[(df_iter["recommender"] == rec) &
                              (df_iter["output"] == out_name)]
                ax.plot(sub["n_points"], sub[mkey],
                        marker="o", ms=3, lw=1.4,
                        color=cmap(k), label=rec)
            ax.set_xlabel("n_points")
            ax.set_ylabel(mlabel)
            ax.set_title(f"{out_name}: {mkey}")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


# -----------------------------------------------------------------
# Ground-truth grid + metrics
# -----------------------------------------------------------------

def gt_grids(n=GRID_N):
    g = np.linspace(0.0, 1.0, n)
    X1, X2, X3 = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)
    F1 = f1_turbidity(pts).reshape(n, n, n)
    F2 = f2_ratio(pts).reshape(n, n, n)
    return (g, g, g), F1, F2


def compute_metrics(data, axes, fields, label):
    """fields = {output_name: F_grid (n,n,n)}. Returns long-format rows."""
    from scipy.spatial import cKDTree
    picks = data[INPUT_COLS].values
    rows = []
    for out_name, F in fields.items():
        mask, has_bnd = define_boundary_3d(F, axes, TOP_FRAC[out_name])
        row = {
            "recommender": label, "output": out_name,
            "n_picks_total": int(len(data)),
            "has_boundary": bool(has_bnd),
            "y_range": float(F.max() - F.min()),
        }
        if has_bnd:
            row.update(boundary_metrics_3d(picks, mask, axes))
            # surface metrics @ NEAR_R
            bnd_pts, _ = _boundary_pts(F, axes, TOP_FRAC[out_name])
            tree_b = cKDTree(bnd_pts)
            d_pick, _ = tree_b.query(picks, k=1)
            row["surf_precision"] = float((d_pick <= NEAR_R).mean())
            tree_p = cKDTree(picks)
            d_bnd, _ = tree_p.query(bnd_pts, k=1)
            row["surf_recall"] = float((d_bnd <= NEAR_R).mean())
        else:
            row.update({"hit_rate": float("nan"),
                        "coverage_auc": float("nan"),
                        "n_boundary_cells": 0,
                        "surf_precision": float("nan"),
                        "surf_recall": float("nan")})
        rows.append(row)
    cm = clumping_metrics(picks, d=3)
    rows.append({"recommender": label, "output": "_all",
                 "n_picks_total": int(len(data)),
                 "has_boundary": True, "y_range": float("nan"),
                 "hit_rate": float("nan"),
                 "coverage_auc": float("nan"),
                 "n_boundary_cells": 0,
                 **cm})
    return rows


# -----------------------------------------------------------------
# Plots
# -----------------------------------------------------------------

VIEWS = [(25, 35), (25, 125), (25, 215), (60, 35)]


def plot_3d_multi_angle(data, F, top_frac, axes, label, out_name, out_path,
                        on_boundary_only=False, near_radius=None):
    """Per-output figure: 2x2 grid of 3D scatters at different viewpoints.

    on_boundary_only : bool
        If True, only picks landing in the boundary mask (or within
        near_radius of it, if given) are drawn.
    near_radius : float or None
        Tolerance distance (in normalized [0,1]^d) for the
        "near-boundary" plot. When set, picks within this distance of
        ANY boundary cell count as on-boundary. If None, exact-cell
        match is used.
    """
    mask, _ = define_boundary_3d(F, axes, top_frac)
    g1, g2, g3 = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    bnd_pts = np.stack([g1[mask], g2[mask], g3[mask]], axis=1)

    picks = data[INPUT_COLS].values
    iters = data["iteration"].values

    if on_boundary_only:
        if near_radius is not None:
            from scipy.spatial import cKDTree
            tree = cKDTree(bnd_pts)
            dists, _ = tree.query(picks, k=1)
            on = dists <= near_radius
            tag = (f" - within R={near_radius:g} of boundary "
                   f"({on.sum()}/{len(on)})")
        else:
            n = len(axes[0])
            ix = np.clip((picks[:, 0] * (n - 1)).round().astype(int),
                         0, n - 1)
            iy = np.clip((picks[:, 1] * (n - 1)).round().astype(int),
                         0, n - 1)
            iz = np.clip((picks[:, 2] * (n - 1)).round().astype(int),
                         0, n - 1)
            on = mask[ix, iy, iz]
            tag = f" - on-boundary picks only ({on.sum()}/{len(on)})"
        picks = picks[on]
        iters = iters[on]
        marker_size = 55
        title_extra = tag
    else:
        marker_size = 28
        title_extra = ""

    # Subsample boundary cloud for plotting only
    if len(bnd_pts) > 4000:
        sel = np.random.default_rng(0).choice(len(bnd_pts), 4000,
                                              replace=False)
        bnd_pts = bnd_pts[sel]

    fig = plt.figure(figsize=(14, 12))
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
        ax.set_title(f"view elev={elev} azim={azim}")
        ax.view_init(elev=elev, azim=azim)
        if k == 0:
            ax.legend(loc="upper left", fontsize=8)
    if sc is not None:
        cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
        fig.colorbar(sc, cax=cbar_ax, label="iteration")
    fig.suptitle(f"{label}: {out_name} boundary "
                 f"(top {int(top_frac*100)}% |grad|){title_extra}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_slices(data, F, axes, label, out_name, out_path):
    """3 mid-plane slices of |grad F| with picks within +/-0.1 of plane."""
    G = grad_magnitude_3d(F, axes)
    n = len(axes[0])
    mid = n // 2
    g = axes[0]
    picks = data[INPUT_COLS].values
    iters = data["iteration"].values

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    slices = [
        (G[mid, :, :], 0, "x1=0.5", ("x2", "x3")),
        (G[:, mid, :], 1, "x2=0.5", ("x1", "x3")),
        (G[:, :, mid], 2, "x3=0.5", ("x1", "x2")),
    ]
    for ax, (S, axis, slab, axlabels) in zip(axs, slices):
        im = ax.pcolormesh(g, g, S.T, cmap="inferno", shading="auto")
        plt.colorbar(im, ax=ax, label=f"|grad {out_name}|", shrink=0.85)
        near = np.abs(picks[:, axis] - 0.5) < 0.1
        if near.any():
            other = [i for i in range(3) if i != axis]
            ax.scatter(picks[near, other[0]], picks[near, other[1]],
                       c=iters[near], cmap="autumn", s=30,
                       edgecolor="white", linewidth=0.5)
        ax.set_xlabel(axlabels[0]); ax.set_ylabel(axlabels[1])
        ax.set_title(f"{slab} ({near.sum()} picks)")
    fig.suptitle(f"{label}: mid-plane slices of |grad {out_name}|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reduce", default="standardized_sum",
                   choices=["sum", "max", "standardized_sum"])
    p.add_argument("--init", default="grid3",
                   choices=["grid3", "grid4", "sobol", "sobol64"],
                   help="Initial design: 3x3x3 grid (27), 4x4x4 grid (64), "
                        "or scrambled Sobol (27 / 64).")
    p.add_argument("--n-points", type=int, default=350,
                   help="Total points budget per recommender (init + iters).")
    args = p.parse_args()

    n_init = {"grid3": 27, "grid4": 64, "sobol": 27, "sobol64": 64}[args.init]
    n_iters = max(0, int(np.ceil((args.n_points - n_init) / Q_BATCH)))
    actual_total = n_init + n_iters * Q_BATCH
    print(f"Budget: {args.n_points} -> {n_init} init + {n_iters} x {Q_BATCH} "
          f"= {actual_total} points")

    tag = args.init + (f"_n{actual_total}" if args.n_points != 155 else "")
    out_root = os.path.join(OUT_BASE, f"3d_corner_spike_{tag}")
    os.makedirs(out_root, exist_ok=True)
    print(f"Init mode: {args.init}\nOutput dir: {out_root}")
    print("Building 3D ground-truth grids...")
    axes, F_turb, F_ratio = gt_grids(GRID_N)
    fields = {"turbidity_600": F_turb, "ratio": F_ratio}
    print(f"  turbidity_600: {F_turb.min():.3f}..{F_turb.max():.3f}")
    print(f"  ratio        : {F_ratio.min():.3f}..{F_ratio.max():.3f}")

    summary_rows = []
    iter_rows = []
    for label, build in [
        ("BayesianContrast", lambda: BayesianTransitionRecommender(
            input_columns=INPUT_COLS, output_columns=OUTPUT_COLS,
            log_transform_inputs=False,
            explore_beta=0.0,
        )),
        ("BayesianContrast_UCB", lambda: BayesianTransitionRecommender(
            input_columns=INPUT_COLS, output_columns=OUTPUT_COLS,
            log_transform_inputs=False,
            explore_beta=1.0,
        )),
        ("GradientUCB", lambda: GradientTransitionRecommender(
            input_columns=INPUT_COLS, output_columns=OUTPUT_COLS,
            log_transform_inputs=False,
            multi_output_reduce=args.reduce,
        )),
        ("LevelSet", lambda: LevelSetTransitionRecommender(
            input_columns=INPUT_COLS, output_columns=OUTPUT_COLS,
            log_transform_inputs=False,
        )),
        ("Simplex", lambda: DelaunaySimplexTransitionRecommender(
            input_columns=INPUT_COLS, output_columns=OUTPUT_COLS,
            log_transform_inputs=False,
        )),
    ]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        rec = build()
        torch.manual_seed(SEED); np.random.seed(SEED)
        data = run_recommender(rec, label, args.init, n_iters)
        data.to_csv(os.path.join(out_root, f"all_data_{label}.csv"),
                    index=False)

        rows = compute_metrics(data, axes, fields, label)
        summary_rows.extend(rows)
        iter_rows.extend(compute_iter_metrics(data, axes, fields, label))

        for out_name, F in fields.items():
            plot_3d_multi_angle(
                data, F, TOP_FRAC[out_name], axes, label, out_name,
                os.path.join(out_root,
                             f"3d_picks_{label}_{out_name}.png"))
            plot_3d_multi_angle(
                data, F, TOP_FRAC[out_name], axes, label, out_name,
                os.path.join(out_root,
                             f"3d_picks_onbnd_{label}_{out_name}.png"),
                on_boundary_only=True)
            plot_3d_multi_angle(
                data, F, TOP_FRAC[out_name], axes, label, out_name,
                os.path.join(out_root,
                             f"3d_picks_near_{label}_{out_name}.png"),
                on_boundary_only=True, near_radius=0.04)
            plot_slices(
                data, F, axes, label, out_name,
                os.path.join(out_root,
                             f"slices_{label}_{out_name}.png"))

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(out_root, "metrics_summary.csv"), index=False)
    print("\n=== metrics_summary.csv ===")
    with pd.option_context("display.width", 200,
                           "display.max_columns", None):
        print(df.to_string(index=False))

    df_iter = pd.DataFrame(iter_rows)
    df_iter.to_csv(os.path.join(out_root, "metrics_per_iter.csv"),
                   index=False)
    plot_per_iter(df_iter, os.path.join(out_root, "per_iter_compare.png"))


if __name__ == "__main__":
    main()

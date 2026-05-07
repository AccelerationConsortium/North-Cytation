"""
Run BayesianTransitionRecommender vs GradientTransitionRecommender against
a ground-truth GP fit to a real experiment dataset (built by
build_ground_truth_gp.py).

Outputs to recommenders/test_outputs/real_<name>/:
  ground_truth.png          # ratio + turbidity heatmaps + |grad| heatmap
  all_data_<rec>.csv
  metrics_<rec>.csv
  2d_exploration_<rec>.png  # picks colored by iteration over true ratio gradient
  side_by_side.png          # both recommenders, both true outputs

Run:
  python -m recommenders.test_real_ground_truth --name CHAPS_BDDAC
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from recommenders.bayesian_transition_recommender import BayesianTransitionRecommender
from recommenders.gradient_transition_recommender import GradientTransitionRecommender
from recommenders.delaunay_triangle_recommender import DelaunayTriangleRecommender
from recommenders.build_ground_truth_gp import (
    load_ground_truth_gp, simulate_from_gp,
)
from recommenders.test_gradient_transition_recommender import (
    Q_BATCH, N_ITERATIONS, N_INIT_GRID, SEED,
    simulate_surfactant_measurements,
)


SYNTHETIC_NAMES = {"surfactant2d"}
# log10(mM) bounds for the analytical surfactant simulator. Same range
# as the real workflow: 0.01 mM to 22.5 mM (B) / 11.25 mM (A).
SURFACTANT2D_LOG_BOUNDS = (np.array([np.log10(0.01), np.log10(0.01)]),
                           np.array([np.log10(11.25), np.log10(22.5)]))


OUT_ROOT = os.path.join(os.path.dirname(__file__), "test_outputs")


class TriangleAdapter:
    """Make DelaunayTriangleRecommender match the (rec.get_recommendations,
    rec.get_metrics_df) interface used by run_recommender."""
    def __init__(self, **kwargs):
        self.rec = DelaunayTriangleRecommender(**kwargs)

    def get_recommendations(self, data, n_points, iteration=None):
        return self.rec.get_recommendations(
            data, n_points=n_points, create_visualization=False,
        )

    def get_metrics_df(self):
        return pd.DataFrame()


# =================================================================
# Ground-truth queries on a regular grid (for plotting and metrics)
# =================================================================

def gt_grid(name, n_grid=120):
    """Posterior-mean ratio + turbidity, plus |grad ratio|, on a regular
    n_grid x n_grid grid in log10 concentration space.
    Returns (A_mm_grid, B_mm_grid, RATIO, TURB, GRAD_MAG, lo, hi).
    Supports real datasets (saved GP bundles) and the synthetic
    'surfactant2d' simulator.
    """
    if name in SYNTHETIC_NAMES:
        return _gt_grid_synthetic(name, n_grid)
    models, bundle = load_ground_truth_gp(name)
    lo, hi = bundle["log_bounds"]            # (2,) each
    g_a = np.linspace(lo[0], hi[0], n_grid)
    g_b = np.linspace(lo[1], hi[1], n_grid)
    G_A, G_B = np.meshgrid(g_a, g_b, indexing="ij")
    pts = torch.tensor(np.stack([G_A.ravel(), G_B.ravel()], axis=1),
                       dtype=torch.double, requires_grad=True)
    # ratio + turbidity
    means = []
    grads = []
    for m in models:
        post = m.posterior(pts)
        mu = post.mean.squeeze(-1)
        means.append(mu.detach().numpy().reshape(n_grid, n_grid))
        g, = torch.autograd.grad(mu.sum(), pts, retain_graph=True,
                                 create_graph=False)
        grads.append(g.detach().numpy())
    RATIO = means[0]
    TURB = means[1]
    grad_ratio = grads[0].reshape(n_grid, n_grid, 2)
    GRAD_MAG = np.linalg.norm(grad_ratio, axis=-1)
    return 10 ** G_A, 10 ** G_B, RATIO, TURB, GRAD_MAG, lo, hi


def _gt_grid_synthetic(name, n_grid):
    if name != "surfactant2d":
        raise ValueError(f"unknown synthetic source {name}")
    lo, hi = SURFACTANT2D_LOG_BOUNDS
    g_a = np.linspace(lo[0], hi[0], n_grid)
    g_b = np.linspace(lo[1], hi[1], n_grid)
    G_A, G_B = np.meshgrid(g_a, g_b, indexing="ij")
    A_mm = 10 ** G_A
    B_mm = 10 ** G_B
    RATIO = np.empty_like(A_mm)
    TURB = np.empty_like(A_mm)
    for i in range(n_grid):
        for j in range(n_grid):
            r, t = simulate_surfactant_measurements(
                float(A_mm[i, j]), float(B_mm[i, j]), add_noise=False)
            RATIO[i, j] = r
            TURB[i, j] = t
    # central-diff gradient of ratio in log space
    dx = G_A[1, 0] - G_A[0, 0]
    dy = G_B[0, 1] - G_B[0, 0]
    gx, gy = np.gradient(RATIO, dx, dy)
    GRAD_MAG = np.sqrt(gx ** 2 + gy ** 2)
    return A_mm, B_mm, RATIO, TURB, GRAD_MAG, lo, hi


def plot_ground_truth(name, out_path):
    A, B, RATIO, TURB, GRAD_MAG, lo, hi = gt_grid(name)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["ratio (GP posterior mean)",
              "turbidity_600 (GP posterior mean)",
              "||grad ratio|| (in log-conc space)"]
    fields = [RATIO, TURB, GRAD_MAG]
    cmaps = ["viridis", "magma", "inferno"]
    for ax, F, title, cm in zip(axes, fields, titles, cmaps):
        im = ax.pcolormesh(np.log10(A), np.log10(B), F, cmap=cm,
                           shading="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("log10(surf_A_mm)")
        ax.set_ylabel("log10(surf_B_mm)")
        ax.set_title(title)
    fig.suptitle(f"Ground-truth GP from real data: {name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return RATIO, TURB, GRAD_MAG, lo, hi


# =================================================================
# Test runner
# =================================================================

def make_initial_grid(lo, hi, n_grid=N_INIT_GRID):
    """5x5 grid in mM concentration, log-spaced inside fitted bounds."""
    g_a = np.logspace(lo[0], hi[0], n_grid)
    g_b = np.logspace(lo[1], hi[1], n_grid)
    G_A, G_B = np.meshgrid(g_a, g_b, indexing="ij")
    return np.stack([G_A.ravel(), G_B.ravel()], axis=1)  # (n*n, 2)


def evaluate_simulator(coords_mm, name):
    rs, ts = [], []
    if name in SYNTHETIC_NAMES:
        for a, b in coords_mm:
            r, t = simulate_surfactant_measurements(
                float(a), float(b), add_noise=False)
            rs.append(r); ts.append(t)
    else:
        for a, b in coords_mm:
            r, t = simulate_from_gp(a, b, name)
            rs.append(r); ts.append(t)
    return np.array(rs), np.array(ts)


def run_recommender(rec, name, lo, hi, label):
    cols = ["surf_A_mm", "surf_B_mm"]
    init = make_initial_grid(lo, hi)
    rs, ts = evaluate_simulator(init, name)
    data = pd.DataFrame({
        cols[0]: init[:, 0], cols[1]: init[:, 1],
        "ratio": rs, "turbidity_600": ts,
        "well_type": "experiment", "iteration": 0,
    })
    for it in range(1, N_ITERATIONS + 1):
        print(f"\n=== {label} iter {it}/{N_ITERATIONS} ===")
        recs = rec.get_recommendations(data, n_points=Q_BATCH, iteration=it)
        coords = recs[cols].values
        rs, ts = evaluate_simulator(coords, name)
        new = recs.copy()
        new["ratio"] = rs
        new["turbidity_600"] = ts
        new["well_type"] = "experiment"
        new["iteration"] = it
        data = pd.concat([data, new], ignore_index=True)
    return data, rec.get_metrics_df()


def compute_all_metrics(data, RATIO, TURB, lo, hi, label):
    """Per-output boundary metrics + boundary-agnostic clumping.
    Returns list of per-output metric dicts plus one row per recommender for
    the clumping component (output='_all').
    """
    from recommenders.metrics import (
        define_boundary, boundary_metrics, clumping_metrics, TOP_FRAC,
    )
    log_a = np.log10(np.clip(data["surf_A_mm"].values, 10**lo[0], 10**hi[0]))
    log_b = np.log10(np.clip(data["surf_B_mm"].values, 10**lo[1], 10**hi[1]))
    nx = (log_a - lo[0]) / (hi[0] - lo[0])
    ny = (log_b - lo[1]) / (hi[1] - lo[1])
    picks = np.stack([nx, ny], axis=1)

    # Normalized grid
    n = RATIO.shape[0]
    g = np.linspace(0.0, 1.0, n)
    GX, GY = np.meshgrid(g, g, indexing="ij")

    rows = []
    for out_name, F in [("ratio", RATIO), ("turbidity_600", TURB)]:
        mask, has_bnd = define_boundary(F, GX, GY, TOP_FRAC[out_name])
        row = {"recommender": label, "output": out_name,
               "n_picks_total": int(len(data)),
               "has_boundary": has_bnd,
               "y_range": float(F.max() - F.min())}
        if has_bnd:
            row.update(boundary_metrics(picks, mask, GX, GY))
        rows.append(row)

    clump = clumping_metrics(picks)
    rows.append({"recommender": label, "output": "_all",
                 "n_picks_total": int(len(data)),
                 "has_boundary": True,
                 **clump})
    return rows


def plot_picks_over_field(data, A, B, FIELD, label, title, out_path,
                          field_name="ratio"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(np.log10(A), np.log10(B), FIELD, cmap="viridis",
                       shading="auto", alpha=0.7)
    plt.colorbar(im, ax=ax, label=field_name)
    log_a = np.log10(np.clip(data["surf_A_mm"].values, A.min(), A.max()))
    log_b = np.log10(np.clip(data["surf_B_mm"].values, B.min(), B.max()))
    sc = ax.scatter(log_a, log_b, c=data["iteration"].values, cmap="autumn",
                    s=35, edgecolor="white", linewidth=0.6)
    plt.colorbar(sc, ax=ax, label="iteration")
    ax.set_xlabel("log10(surf_A_mm)"); ax.set_ylabel("log10(surf_B_mm)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="CHAPS_BDDAC")
    p.add_argument("--reduce", default="standardized_sum",
                   choices=["sum", "max", "standardized_sum"],
                   help="multi-output reduction for GradientUCB")
    args = p.parse_args()

    name = args.name
    out_dir = os.path.join(OUT_ROOT, f"real_{name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n### Ground-truth surface: {name}")
    RATIO, TURB, GRAD_MAG, lo, hi = plot_ground_truth(
        name, os.path.join(out_dir, "ground_truth.png"))
    A, B, _, _, _, _, _ = gt_grid(name)

    summary = []
    results = {}
    for label, build in [
        ("BayesianContrast", lambda: BayesianTransitionRecommender(
            input_columns=["surf_A_mm", "surf_B_mm"],
            output_columns=["ratio", "turbidity_600"],
            log_transform_inputs=True,
        )),
        ("GradientUCB", lambda: GradientTransitionRecommender(
            input_columns=["surf_A_mm", "surf_B_mm"],
            output_columns=["ratio", "turbidity_600"],
            log_transform_inputs=True,
            multi_output_reduce=args.reduce,
        )),
        ("Triangle", lambda: TriangleAdapter(
            input_columns=["surf_A_mm", "surf_B_mm"],
            output_columns=["ratio", "turbidity_600"],
            log_transform_inputs=True,
        )),
    ]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        rec = build()
        torch.manual_seed(SEED); np.random.seed(SEED)
        data, metrics = run_recommender(rec, name, lo, hi, label)
        data.to_csv(os.path.join(out_dir, f"all_data_{label}.csv"), index=False)
        metrics.to_csv(os.path.join(out_dir, f"metrics_{label}.csv"), index=False)
        plot_picks_over_field(
            data, A, B, RATIO, label,
            title=f"{name}: {label} picks over true ratio",
            out_path=os.path.join(out_dir, f"2d_exploration_{label}.png"),
            field_name="ratio",
        )
        m_rows = compute_all_metrics(data, RATIO, TURB, lo, hi, label)
        summary.extend(m_rows)
        results[label] = data

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"),
                      index=False)
    print("\nMetrics summary:")
    print("  Per-output boundary metrics + boundary-agnostic clumping ('_all').")
    print("  hit_rate          frac of picks landing on the boundary mask")
    print("  coverage_auc      area under recall-vs-radius curve (higher better)")
    print("  cv_nn             clumpiness of NN distances (uniform~0.52)")
    print("  clumping_ratio    median NN / (1/N)^(1/d). ~1 = grid-like")
    print(summary_df.to_string(index=False))

    # Side-by-side: each recommender, ratio + grad-magnitude background
    labels_for_plot = ["BayesianContrast", "GradientUCB", "Triangle"]
    fig, axes = plt.subplots(2, len(labels_for_plot),
                             figsize=(7 * len(labels_for_plot), 12))
    for col, label in enumerate(labels_for_plot):
        data = results[label]
        log_a = np.log10(np.clip(data["surf_A_mm"].values, A.min(), A.max()))
        log_b = np.log10(np.clip(data["surf_B_mm"].values, B.min(), B.max()))
        for row, (FIELD, fname, cm) in enumerate(
                [(RATIO, "true ratio", "viridis"),
                 (GRAD_MAG, "||grad ratio||", "inferno")]):
            ax = axes[row, col]
            im = ax.pcolormesh(np.log10(A), np.log10(B), FIELD, cmap=cm,
                               shading="auto", alpha=0.7)
            plt.colorbar(im, ax=ax, label=fname, shrink=0.8)
            sc = ax.scatter(log_a, log_b, c=data["iteration"].values,
                            cmap="autumn", s=25, edgecolor="white",
                            linewidth=0.4)
            # Pick out the ratio-row metrics for this recommender
            r_row = next(s for s in summary
                         if s["recommender"] == label
                         and s.get("output") == "ratio")
            c_row = next(s for s in summary
                         if s["recommender"] == label
                         and s.get("output") == "_all")
            ax.set_title(
                f"{label}\n"
                f"ratio: hit={r_row.get('hit_rate', float('nan')):.2f}  "
                f"AUC={r_row.get('coverage_auc', float('nan')):.3f}\n"
                f"cv_nn={c_row['cv_nn']:.2f}  "
                f"clump={c_row['clumping_ratio']:.2f}"
            )
            ax.set_xlabel("log10(surf_A_mm)")
            ax.set_ylabel("log10(surf_B_mm)")
    fig.suptitle(f"Real-data ground truth: {name}  "
                 f"(GradientUCB reduce={args.reduce})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "side_by_side.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {os.path.join(out_dir, 'side_by_side.png')}")


if __name__ == "__main__":
    main()

"""
Compare the three multi-output reduction modes for GradientTransitionRecommender
on the surfactant2d simulator. Produces:

  recommenders/test_outputs/multi_output_compare/
    2d_exploration_sum.png
    2d_exploration_max.png
    2d_exploration_standardized_sum.png
    side_by_side.png
    metrics_summary.csv

Run:
  python -m recommenders.test_multi_output_reduce
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from recommenders.gradient_transition_recommender import GradientTransitionRecommender
from recommenders.test_gradient_transition_recommender import (
    TESTS, _make_initial_dataset, _evaluate_recommendations,
    _plot_2d_exploration, _normalize_mM, surfactant2d_boundary,
    _surfactant_turbidity_boundary, MIN_CONC, MAX_CONC,
    Q_BATCH, N_ITERATIONS, N_INIT_GRID, SEED,
)


REDUCE_MODES = ["sum", "max", "standardized_sum"]
TEST_NAME = "surfactant2d"
OUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs",
                       "multi_output_compare")


def run_one(mode):
    cfg = TESTS[TEST_NAME]
    torch.manual_seed(SEED); np.random.seed(SEED)
    rec = GradientTransitionRecommender(
        input_columns=cfg["input_columns"],
        output_columns=["ratio", "turbidity_600"],
        log_transform_inputs=cfg["log_transform"],
        multi_output_reduce=mode,
    )
    data = _make_initial_dataset(TEST_NAME, cfg["input_columns"],
                                 n_init_grid=N_INIT_GRID)
    data["iteration"] = 0
    for it in range(1, N_ITERATIONS + 1):
        print(f"\n=== mode={mode} iter {it}/{N_ITERATIONS} ===")
        recs = rec.get_recommendations(
            data, n_points=Q_BATCH, iteration=it,
            boundary_func=cfg["boundary_func"],
        )
        new_rows = _evaluate_recommendations(recs, TEST_NAME, cfg["input_columns"])
        new_rows["iteration"] = it
        data = pd.concat([data, new_rows], ignore_index=True)
    return data, rec.get_metrics_df()


def _coverage_metrics(data, input_columns):
    """Boundary-coverage metrics: fraction of picks within 0.05 of EITHER true
    boundary in normalized log space, plus separate counts per boundary."""
    nx, ny = _normalize_mM(data[input_columns[0]].values,
                           data[input_columns[1]].values)
    h_ratio = np.array([surfactant2d_boundary([a, b])
                        for a, b in zip(nx, ny)])
    h_turb = np.array([_surfactant_turbidity_boundary([a, b])
                       for a, b in zip(nx, ny)])
    eps = 0.05
    near_ratio = np.abs(h_ratio) < eps
    near_turb = np.abs(h_turb) < eps
    near_either = near_ratio | near_turb
    return {
        "n_total": int(len(data)),
        "near_ratio": int(near_ratio.sum()),
        "near_turb": int(near_turb.sum()),
        "near_either": int(near_either.sum()),
        "frac_near_either": float(near_either.mean()),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = TESTS[TEST_NAME]

    results = {}
    summary_rows = []
    for mode in REDUCE_MODES:
        data, _ = run_one(mode)
        results[mode] = data
        data.to_csv(os.path.join(OUT_DIR, f"all_data_{mode}.csv"), index=False)
        out = os.path.join(OUT_DIR, f"2d_exploration_{mode}.png")
        _plot_2d_exploration(data, TEST_NAME, cfg["input_columns"],
                             title=f"surfactant2d: reduce={mode}",
                             out_path=out)
        m = _coverage_metrics(data, cfg["input_columns"])
        m["mode"] = mode
        summary_rows.append(m)

    summary = pd.DataFrame(summary_rows)[
        ["mode", "n_total", "near_ratio", "near_turb",
         "near_either", "frac_near_either"]]
    summary.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)
    print("\nCoverage summary:")
    print(summary.to_string(index=False))

    # Side-by-side composite
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    n_grid = 200
    g = np.linspace(0.0, 1.0, n_grid)
    G0, G1 = np.meshgrid(g, g)
    H_ratio = np.zeros_like(G0); H_turb = np.zeros_like(G0)
    for i in range(n_grid):
        for j in range(n_grid):
            xn = [G0[i, j], G1[i, j]]
            H_ratio[i, j] = surfactant2d_boundary(xn)
            H_turb[i, j] = _surfactant_turbidity_boundary(xn)

    for ax, mode in zip(axes, REDUCE_MODES):
        data = results[mode]
        ax.contour(G0, G1, H_ratio, levels=[0.0], colors="k",
                   linewidths=2, linestyles="--")
        ax.contour(G0, G1, H_turb, levels=[0.0], colors="red",
                   linewidths=2, linestyles=":")
        nx, ny = _normalize_mM(data[cfg["input_columns"][0]].values,
                               data[cfg["input_columns"][1]].values)
        sc = ax.scatter(nx, ny, c=data["iteration"].values, cmap="viridis",
                        s=30, edgecolor="white", linewidth=0.4)
        m = next(r for r in summary_rows if r["mode"] == mode)
        ax.set_title(f"{mode}\nnear ratio: {m['near_ratio']}, "
                     f"turb: {m['near_turb']}, either: {m['near_either']}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("surf_A_mm (norm)"); ax.set_ylabel("surf_B_mm (norm)")
    fig.colorbar(sc, ax=axes, label="iteration", shrink=0.8)
    fig.suptitle("GradientUCB: multi-output reduction comparison "
                 "(surfactant2d, 5x5 grid + 21 batches of 8)")
    fig.savefig(os.path.join(OUT_DIR, "side_by_side.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {os.path.join(OUT_DIR, 'side_by_side.png')}")


if __name__ == "__main__":
    main()

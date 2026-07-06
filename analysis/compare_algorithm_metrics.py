"""Compare per-iteration metrics across multiple recommender algorithms.

Recomputes metrics (including IoU) directly from results_final.csv so that
all existing simulation runs can be compared without re-running workflows.

Usage (from repo root):
    python analysis/compare_algorithm_metrics.py

Output:
    output/algorithm_comparison/algorithm_metrics_comparison.png
    output/algorithm_comparison/algorithm_metrics_combined.csv
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.iteration_metrics import _TITLES, _SUBTITLES, _YLABELS, compute_iteration_metrics

ALGORITHMS = ['bayesian', 'gradient', 'levelset', 'triangle', 'sobol']

# Colour per algorithm — colourblind-safe palette
COLOURS = {
    'bayesian': '#2166ac',   # blue
    'gradient': '#d6604d',   # red-orange
    'levelset': '#4dac26',   # green
    'triangle': '#8e44ad',   # purple
    'sobol':    '#f4a582',   # peach (baseline)
}

PLOT_COLS = ['turb_rmse', 'ratio_rmse', 'iou_ratio_0_81']


def find_latest_results_csv(algorithm, base='output/simulated_surfactant_grid'):
    """Find the most recent results_final.csv for a given algorithm."""
    pattern = os.path.join(base, f'multidim_3D_*_{algorithm}_*', 'results_final.csv')
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.dirname(p))
    return matches[-1]


def load_all_metrics(algorithms=ALGORITHMS, base='output/simulated_surfactant_grid',
                     surfactants=None):
    """Recompute per-iteration metrics (including IoU) from results_final.csv.

    Uses the noise-free synthetic function as ground truth, so IoU is properly
    defined against a fixed boundary rather than the measured data itself.
    Existing simulation runs do not need to be re-run.
    """
    if surfactants is None:
        surfactants = ['SDS', 'TTAB', 'DTAB']

    frames = []
    for alg in algorithms:
        csv = find_latest_results_csv(alg, base)
        if csv is None:
            print(f"  WARNING: no results_final.csv found for '{alg}' — skipping")
            continue
        print(f"  Computing metrics for {alg} ...")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df = compute_iteration_metrics(csv, surfactants, '', simulate=True)
        df['algorithm'] = alg
        df['source_folder'] = os.path.dirname(csv)
        frames.append(df)
        print(f"    {len(df)} iterations from {os.path.dirname(csv)}")
    if not frames:
        raise FileNotFoundError(
            "No results_final.csv found for any algorithm. "
            "Run the workflow for each algorithm first."
        )
    return pd.concat(frames, ignore_index=True)


def plot_comparison(combined_df, output_folder):
    """Comparison: RMSE and IoU time-series panels only."""
    n_ts = sum(1 for c in PLOT_COLS if c in combined_df.columns)
    fig, axes_grid = plt.subplots(1, n_ts, figsize=(5.5 * n_ts, 6))
    if n_ts == 1:
        axes_grid = [axes_grid]
    fig.patch.set_facecolor('#f8f8f8')

    algorithms_present = combined_df['algorithm'].unique()

    for ax, col in zip(axes_grid[:n_ts], [c for c in PLOT_COLS if c in combined_df.columns]):
        if col not in combined_df.columns:
            ax.set_visible(False)
            continue

        for alg in ALGORITHMS:
            if alg not in algorithms_present:
                continue
            sub = combined_df[combined_df['algorithm'] == alg].sort_values('iteration')
            if sub[col].isna().all():
                continue
            ax.plot(
                sub['iteration'], sub[col],
                'o-', lw=2, ms=4,
                color=COLOURS.get(alg, '#888888'),
                label=alg,
                alpha=0.85,
            )

        ax.set_facecolor('white')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_xlabel('Cumulative iterations', fontsize=9)
        ax.set_ylabel(_YLABELS.get(col, col), fontsize=9)
        ax.set_title(_TITLES.get(col, col), fontsize=11, fontweight='bold', pad=18)
        ax.text(
            0.5, 1.01,
            _SUBTITLES.get(col, ''),
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=7, style='italic', color='#555555',
            linespacing=1.4,
        )

    # Shared legend
    legend_handles = [
        Line2D([0], [0], color=COLOURS.get(a, '#888888'), lw=2, marker='o',
               markersize=5, label=a)
        for a in ALGORITHMS if a in algorithms_present
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(legend_handles),
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.suptitle(
        'Algorithm Comparison — Per-Iteration RMSE and IoU Metrics',
        fontsize=14, fontweight='bold', y=0.995,
    )

    os.makedirs(output_folder, exist_ok=True)
    png_path = os.path.join(output_folder, 'algorithm_metrics_comparison.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved comparison plot: {png_path}")
    return png_path


if __name__ == '__main__':
    SURFACTANTS = ['SDS', 'TTAB', 'DTAB']
    BASE = 'output/simulated_surfactant_grid'

    print("Computing iteration metrics for all algorithms...")
    combined = load_all_metrics(surfactants=SURFACTANTS, base=BASE)

    out_folder = os.path.join('output', 'algorithm_comparison')
    csv_out = os.path.join(out_folder, 'algorithm_metrics_combined.csv')
    os.makedirs(out_folder, exist_ok=True)
    combined.to_csv(csv_out, index=False)
    print(f"Combined CSV saved: {csv_out}")

    plot_comparison(combined, out_folder)

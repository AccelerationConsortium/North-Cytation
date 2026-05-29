"""Compare per-iteration metrics across multiple recommender algorithms.

Finds the most recent run folder per algorithm, loads iteration_metrics.csv,
and plots all algorithms on the same 8-panel figure.

Usage (from repo root):
    python analysis/compare_algorithm_metrics.py

Output:
    output/algorithm_comparison/algorithm_metrics_comparison.png
    output/algorithm_comparison/algorithm_metrics_combined.csv
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.iteration_metrics import _TITLES, _SUBTITLES, _YLABELS

ALGORITHMS = ['bayesian', 'gradient', 'levelset', 'triangle', 'sobol']

# Colour per algorithm — colourblind-safe palette
COLOURS = {
    'bayesian': '#2166ac',   # blue
    'gradient': '#d6604d',   # red-orange
    'levelset': '#4dac26',   # green
    'triangle': '#8e44ad',   # purple
    'sobol':    '#f4a582',   # peach (baseline)
}

PLOT_COLS = ['turb_hit_rate', 'n_cloud_measured', 'turb_rmse', 'ratio_rmse']


def find_latest_results_csv(algorithm, base='output/simulated_surfactant_grid'):
    """Find the most recent results_final.csv for a given algorithm."""
    pattern = os.path.join(base, f'multidim_3D_*_{algorithm}_*', 'results_final.csv')
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.dirname(p))
    return matches[-1]


def compute_nn_distances(results_csv, surfactants):
    """Return normalised NN distances for all picks in results_final.csv."""
    from analysis.plot_3d_interactive import compute_gp_grids
    df = pd.read_csv(results_csv)
    conc_cols = [f'{s}_conc_mm' for s in surfactants]
    _, _, _, log10_bounds = compute_gp_grids(results_csv, surfactants=surfactants)
    span = log10_bounds[:, 1] - log10_bounds[:, 0]
    span = np.where(span < 1e-12, 1.0, span)
    log_c = np.log10(np.clip(df[conc_cols].values.astype(float), 1e-9, None))
    picks_norm = np.clip((log_c - log10_bounds[:, 0]) / span, 0.0, 1.0)
    nn_d, _ = cKDTree(picks_norm).query(picks_norm, k=2)
    return nn_d[:, 1]


def find_latest_metrics_csv(algorithm, base='output/simulated_surfactant_grid'):
    """Find the most recent iteration_metrics.csv for a given algorithm."""
    pattern = os.path.join(base, f'multidim_3D_*_{algorithm}_*', 'iteration_metrics.csv')
    matches = glob.glob(pattern)
    if not matches:
        return None
    # Sort by folder timestamp (last component of parent path)
    matches.sort(key=lambda p: os.path.dirname(p))
    return matches[-1]


def load_all_metrics(algorithms=ALGORITHMS, base='output/simulated_surfactant_grid'):
    """Load iteration_metrics.csv for each algorithm, tag with algorithm name."""
    frames = []
    for alg in algorithms:
        csv = find_latest_metrics_csv(alg, base)
        if csv is None:
            print(f"  WARNING: no iteration_metrics.csv found for '{alg}' — skipping")
            continue
        df = pd.read_csv(csv)
        df['algorithm'] = alg
        df['source_folder'] = os.path.dirname(csv)
        frames.append(df)
        print(f"  Loaded {alg}: {len(df)} iterations from {os.path.dirname(csv)}")
    if not frames:
        raise FileNotFoundError(
            "No iteration_metrics.csv found for any algorithm. "
            "Run the workflow for each algorithm first."
        )
    return pd.concat(frames, ignore_index=True)


def plot_comparison(combined_df, output_folder, nn_distances=None):
    """1x5 comparison: 4 time-series panels + 1 NN-distance histogram panel."""
    n_ts = sum(1 for c in PLOT_COLS if c in combined_df.columns)
    n_panels = n_ts + (1 if nn_distances else 0)
    fig, axes_grid = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 6))
    if n_panels == 1:
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

    # --- Histogram panel: NN distances ---
    ax_hist = axes_grid[n_ts] if nn_distances else None
    ax_hist.set_facecolor('white')
    if nn_distances:
        all_vals = np.concatenate(list(nn_distances.values()))
        bins = np.linspace(0, np.percentile(all_vals, 99), 40)
        for alg in ALGORITHMS:
            if alg not in nn_distances:
                continue
            ax_hist.hist(
                nn_distances[alg], bins=bins,
                color=COLOURS.get(alg, '#888888'),
                alpha=0.45, density=True, label=alg,
            )
        ax_hist.set_xlabel('Nearest-neighbour distance (normalised)', fontsize=9)
        ax_hist.set_ylabel('Density', fontsize=9)
        ax_hist.set_title('NN Distance Distribution', fontsize=11,
                          fontweight='bold', pad=18)
        ax_hist.text(
            0.5, 1.01,
            'Distribution of distances to nearest neighbour (all 288 final picks).\n'
            'Narrow peak near right = uniform; long left tail = dense clusters + large gaps',
            transform=ax_hist.transAxes,
            ha='center', va='bottom', fontsize=7, style='italic',
            color='#555555', linespacing=1.4,
        )
        ax_hist.grid(True, alpha=0.2, linestyle='--')
    else:
        ax_hist.set_visible(False)

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
        'Algorithm Comparison — Per-Iteration Exploration Metrics',
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

    print("Loading iteration metrics for all algorithms...")
    combined = load_all_metrics()

    out_folder = os.path.join('output', 'algorithm_comparison')
    csv_out = os.path.join(out_folder, 'algorithm_metrics_combined.csv')
    os.makedirs(out_folder, exist_ok=True)
    combined.to_csv(csv_out, index=False)
    print(f"Combined CSV saved: {csv_out}")

    # Check whether RMSE columns exist (requires simulate=True recompute)
    has_rmse = 'turb_rmse' in combined.columns and not combined['turb_rmse'].isna().all()
    if not has_rmse:
        print("RMSE columns not found — recomputing metrics with simulate=True...")
        from analysis.iteration_metrics import compute_iteration_metrics, save_iteration_metrics
        import warnings
        frames = []
        for alg in ALGORITHMS:
            csv = find_latest_results_csv(alg, BASE)
            if csv is None:
                continue
            folder = os.path.dirname(csv)
            print(f"  {alg}...", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                df = compute_iteration_metrics(csv, SURFACTANTS, folder, simulate=True)
            save_iteration_metrics(df, folder)
            df['algorithm'] = alg
            frames.append(df)
            print(f"    turb_rmse final={df.turb_rmse.iloc[-1]:.4f}  ratio_rmse final={df.ratio_rmse.iloc[-1]:.4f}")
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(csv_out, index=False)
        print(f"Updated CSV saved: {csv_out}")

    print("Computing NN distance distributions...")
    import warnings
    nn_dists = {}
    for alg in ALGORITHMS:
        rcsv = find_latest_results_csv(alg, BASE)
        if rcsv is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            nn_dists[alg] = compute_nn_distances(rcsv, SURFACTANTS)
        print(f"  {alg}: median NN = {np.median(nn_dists[alg]):.4f}")

    plot_comparison(combined, out_folder, nn_distances=nn_dists)

"""compare_algorithm_iou.py

Which algorithm finds the ratio and turbidity boundaries fastest?

Metric — Boundary Coverage (median nearest-pick distance):
  - Ground truth boundary: cells of a fixed N^d grid where the noise-free
    synthetic value is within a narrow band of the threshold.
  - At each iteration: for every GT boundary cell, find the distance to the
    nearest accumulated pick (in normalised [0,1]^d log-concentration space).
  - Report the MEDIAN of those distances. Lower = better.
    Algorithms that target the boundary drop this fast; uniform Sobol stays high.
  - Also report 90th-percentile distance (boundary coverage tail).

X-axis: cumulative number of measured wells.

Usage (from repo root):
    python analysis/compare_algorithm_iou.py
    python analysis/compare_algorithm_iou.py SDS NaLS TTAB DTAB

Output:
    output/algorithm_comparison/boundary_coverage_SDS_TTAB_DTAB.png
    output/algorithm_comparison/boundary_coverage_SDS_TTAB_DTAB.csv

Optional positional args: surfactant names (default: SDS TTAB DTAB).
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
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.surfactant_grid_adaptive_concentrations import SURFACTANT_LIBRARY

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SURFACTANTS = ['SDS', 'TTAB', 'DTAB']
BASE = 'output/simulated_surfactant_grid'
OUT_DIR = os.path.join('output', 'algorithm_comparison')

ALGORITHMS = ['bayesian', 'gradient', 'levelset', 'triangle', 'sobol']
COLOURS = {
    'bayesian': '#2166ac',
    'gradient': '#d6604d',
    'levelset': '#4dac26',
    'triangle': '#8e44ad',
    'sobol':    '#f4a582',
}

RATIO_THRESHOLD     = 0.81    # boundary: sub-CMC (high ratio) vs micellar (low ratio)
TURBIDITY_THRESHOLD = 0.10    # boundary: clear vs precipitated
SIGMA_LOG           = 0.35    # catanionic band width (must match simulate_measurements_nd)
GRID_N              = 20      # grid points per axis -> 20^3 = 8000 grid cells
# Boundary band widths — cells within this distance of the threshold are "on the boundary".
# GT ratio range [0.70, 0.85]; GT turb range [0.04, 1.0].
RATIO_BAND      = 0.01   # +-0.01 around 0.81 -> ratio in [0.80, 0.82]
TURBIDITY_BAND  = 0.02   # +-0.02 around 0.10 -> turbidity in [0.08, 0.12]


# ---------------------------------------------------------------------------
# Noise-free synthetic function (matches simulate_measurements_nd exactly)
# ---------------------------------------------------------------------------

def _simulate_noisefree(concs_mm):
    """Returns (turbidity, ratio) with no noise."""
    sum_anionic  = 1e-10
    sum_cationic = 1e-10
    for s, c in concs_mm.items():
        cat = SURFACTANT_LIBRARY[s].get('category', 'nonionic')
        if cat == 'anionic':
            sum_anionic  += c
        elif cat == 'cationic':
            sum_cationic += c

    log_imbalance = np.log10(sum_anionic / sum_cationic)
    turb_peak = np.exp(-(log_imbalance / SIGMA_LOG) ** 2)
    total_conc = sum_anionic + sum_cationic
    conc_gate  = 1.0 / (1.0 + np.exp(-4.0 * (np.log10(total_conc) - np.log10(0.5))))
    turbidity  = float(np.clip(0.04 + (1.0 - 0.04) * turb_peak * conc_gate, 0.02, 1.0))

    mixed_cmc = sum(c / SURFACTANT_LIBRARY[s]['cmc_mm'] for s, c in concs_mm.items())
    ratio_drop = 1.0 / (1.0 + np.exp(-5.0 * np.log10(max(mixed_cmc, 1e-9))))
    ratio      = float(np.clip(0.85 - (0.85 - 0.70) * ratio_drop, 0.70, 0.95))
    return turbidity, ratio


# ---------------------------------------------------------------------------
# Build fixed ground-truth grid
# ---------------------------------------------------------------------------

def build_ground_truth_grid(surfactants, log10_bounds, n=GRID_N):
    """Evaluate noise-free synthetic function on a regular n^d grid.

    Returns:
        grid_norm      (n^d, d) float — normalised [0,1] coordinates
        ratio_bnd_pts  (m, d)   float — grid_norm rows that are on ratio boundary
        turb_bnd_pts   (k, d)   float — grid_norm rows that are on turbidity boundary
    """
    d = len(surfactants)
    axes_norm = [np.linspace(0, 1, n) for _ in range(d)]
    mesh = np.meshgrid(*axes_norm, indexing='ij')
    grid_norm = np.column_stack([m.ravel() for m in mesh])

    span = log10_bounds[:, 1] - log10_bounds[:, 0]

    turb_gt  = np.empty(len(grid_norm))
    ratio_gt = np.empty(len(grid_norm))

    for i, pt in enumerate(grid_norm):
        log_c = log10_bounds[:, 0] + pt * span
        concs = dict(zip(surfactants, 10.0 ** log_c))
        turb_gt[i], ratio_gt[i] = _simulate_noisefree(concs)

    ratio_bnd = np.abs(ratio_gt - RATIO_THRESHOLD)    <= RATIO_BAND
    turb_bnd  = np.abs(turb_gt  - TURBIDITY_THRESHOLD) <= TURBIDITY_BAND

    return grid_norm, grid_norm[ratio_bnd], grid_norm[turb_bnd]


# ---------------------------------------------------------------------------
# Boundary coverage metric
# ---------------------------------------------------------------------------

def boundary_coverage(picks_norm, bnd_pts):
    """For each GT boundary cell, find distance to nearest pick.

    Returns (median_dist, p90_dist) in normalised [0,1]^d space.
    Lower is better — boundary-targeting algorithms should drop this fast.
    """
    if len(picks_norm) == 0 or len(bnd_pts) == 0:
        return float('nan'), float('nan')
    tree = cKDTree(picks_norm)
    dists, _ = tree.query(bnd_pts, k=1)
    return float(np.median(dists)), float(np.percentile(dists, 90))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def get_log10_bounds(results_csv, surfactants):
    df = pd.read_csv(results_csv)
    conc_cols = [f'{s}_conc_mm' for s in surfactants]
    exp = df[df['well_type'] == 'experiment'].copy()
    exp = exp[exp[conc_cols].gt(0).all(axis=1)]
    return np.array([
        [np.log10(exp[c].min()), np.log10(exp[c].max())]
        for c in conc_cols
    ])


def compute_coverage_over_iterations(results_csv, surfactants,
                                     ratio_bnd_pts, turb_bnd_pts,
                                     log10_bounds):
    """Replay accumulated picks; compute boundary coverage at each iteration.

    Returns DataFrame: [n_measurements, iteration,
                        ratio_median_dist, ratio_p90_dist,
                        turb_median_dist,  turb_p90_dist]
    """
    df = pd.read_csv(results_csv)
    conc_cols = [f'{s}_conc_mm' for s in surfactants]
    exp = df[df['well_type'] == 'experiment'].copy()
    exp = exp[exp[conc_cols].gt(0).all(axis=1)].reset_index(drop=True)

    if 'iteration' not in exp.columns:
        raise ValueError("'iteration' column missing from results_final.csv")

    span = log10_bounds[:, 1] - log10_bounds[:, 0]
    span = np.where(span < 1e-12, 1.0, span)

    def normalize(df_subset):
        concs = df_subset[conc_cols].values.astype(float)
        log_c = np.log10(np.clip(concs, 1e-9, None))
        return np.clip((log_c - log10_bounds[:, 0]) / span, 0.0, 1.0)

    rows = []
    for iter_val in sorted(exp['iteration'].unique()):
        subset = exp[exp['iteration'] <= iter_val]
        if len(subset) < 2:
            continue
        picks_norm = normalize(subset)
        r_med, r_p90 = boundary_coverage(picks_norm, ratio_bnd_pts)
        t_med, t_p90 = boundary_coverage(picks_norm, turb_bnd_pts)
        rows.append({
            'n_measurements':   int(len(subset)),
            'iteration':        int(iter_val),
            'ratio_median_dist': r_med,
            'ratio_p90_dist':    r_p90,
            'turb_median_dist':  t_med,
            'turb_p90_dist':     t_p90,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Find latest results_final.csv per algorithm
# ---------------------------------------------------------------------------

def find_latest_results_csv(algorithm, base=BASE, surfactant_tag='SDS_TTAB_DTAB'):
    pattern = os.path.join(
        base, f'multidim_3D_{surfactant_tag}_{algorithm}_*', 'results_final.csv'
    )
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.dirname(p))
    return matches[-1]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_coverage_comparison(combined_df, output_folder, surfactants,
                              out_filename='boundary_coverage.png'):
    """Two-panel figure: median nearest-pick distance to GT boundary.

    Lower = better. Algorithms that target the boundary drop faster than Sobol.
    Shaded band = median to 90th percentile.
    """
    algs_present = [a for a in ALGORITHMS if a in combined_df['algorithm'].unique()]

    fig, (ax_r, ax_t) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#f8f8f8')

    panel_specs = [
        (ax_r, 'ratio_median_dist', 'ratio_p90_dist',
         f'Ratio Boundary Coverage\n(band: ratio = {RATIO_THRESHOLD} +/- {RATIO_BAND})',
         'Median distance to nearest pick  [normalised]'),
        (ax_t, 'turb_median_dist', 'turb_p90_dist',
         f'Turbidity Boundary Coverage\n(band: turbidity = {TURBIDITY_THRESHOLD} +/- {TURBIDITY_BAND})',
         'Median distance to nearest pick  [normalised]'),
    ]

    for ax, col_med, col_p90, title, ylabel in panel_specs:
        for alg in algs_present:
            sub = combined_df[combined_df['algorithm'] == alg].sort_values('n_measurements')
            if sub[col_med].isna().all():
                continue
            colour = COLOURS.get(alg, '#888888')
            ax.plot(sub['n_measurements'], sub[col_med],
                    'o-', lw=2, ms=4, color=colour, label=alg, alpha=0.9)
            ax.fill_between(sub['n_measurements'], sub[col_med], sub[col_p90],
                            color=colour, alpha=0.12)

        ax.set_facecolor('white')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_xlabel('Cumulative measurements', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(bottom=0)

    legend_handles = [
        Line2D([0], [0], color=COLOURS.get(a, '#888888'), lw=2, marker='o',
               markersize=5, label=a)
        for a in algs_present
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(legend_handles), fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        'Algorithm Comparison — Boundary Coverage vs Cumulative Measurements\n'
        f'Surfactants: {", ".join(surfactants)}  |  {len(surfactants)}D simulation  '
        f'|  lower = faster boundary discovery',
        fontsize=11, fontweight='bold', y=1.01,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1.0])

    os.makedirs(output_folder, exist_ok=True)
    png_path = os.path.join(output_folder, out_filename)
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) > 1:
        SURFACTANTS = [s.strip() for s in sys.argv[1:]]
        print(f"CLI override: SURFACTANTS = {SURFACTANTS}")

    surfactant_tag = '_'.join(SURFACTANTS)
    nd = len(SURFACTANTS)
    OUT_DIR_RUN = os.path.join('output', 'algorithm_comparison')

    print(f"Building {GRID_N}^{nd} ground-truth boundary for {SURFACTANTS} ...")

    sample_csv = (find_latest_results_csv('bayesian', surfactant_tag=surfactant_tag)
                  or find_latest_results_csv('sobol', surfactant_tag=surfactant_tag))
    if sample_csv is None:
        raise FileNotFoundError(
            f"No results_final.csv found under '{BASE}' for surfactants {SURFACTANTS}."
        )

    log10_bounds = get_log10_bounds(sample_csv, SURFACTANTS)
    print(f"  Concentration bounds (log10 mM):")
    for s, b in zip(SURFACTANTS, log10_bounds):
        print(f"    {s}: [{b[0]:.2f}, {b[1]:.2f}]")

    grid_norm, ratio_bnd_pts, turb_bnd_pts = build_ground_truth_grid(SURFACTANTS, log10_bounds)
    print(f"  Ratio boundary cells:     {len(ratio_bnd_pts)}/{len(grid_norm)} "
          f"({100*len(ratio_bnd_pts)/len(grid_norm):.1f}%)")
    print(f"  Turbidity boundary cells: {len(turb_bnd_pts)}/{len(grid_norm)} "
          f"({100*len(turb_bnd_pts)/len(grid_norm):.1f}%)")

    frames = []
    for alg in ALGORITHMS:
        csv = find_latest_results_csv(alg, surfactant_tag=surfactant_tag)
        if csv is None:
            print(f"  WARNING: no data for '{alg}' — skipping")
            continue
        print(f"  Processing {alg} ...")
        df = compute_coverage_over_iterations(
            csv, SURFACTANTS, ratio_bnd_pts, turb_bnd_pts, log10_bounds
        )
        df['algorithm'] = alg
        frames.append(df)
        r_med = df['ratio_median_dist'].iloc[-1]
        t_med = df['turb_median_dist'].iloc[-1]
        print(f"    {len(df)} steps | final ratio dist={r_med:.4f} | turb dist={t_med:.4f}")

    if not frames:
        raise RuntimeError("No data found for any algorithm.")

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(OUT_DIR_RUN, exist_ok=True)
    csv_out = os.path.join(OUT_DIR_RUN, f'boundary_coverage_{surfactant_tag}.csv')
    combined.to_csv(csv_out, index=False)
    print(f"Saved data: {csv_out}")

    plot_coverage_comparison(combined, OUT_DIR_RUN, SURFACTANTS,
                             out_filename=f'boundary_coverage_{surfactant_tag}.png')
    print("Done.")

"""Per-iteration cumulative metrics for evaluating N-D surfactant recommenders.

Three metrics, computed cumulatively at each iteration:

  turb_hit_rate      fraction of picks landing on the GP-predicted 0.1 OD turbidity shell
  n_cloud_measured   cumulative count of directly-measured wells with turbidity_600 > 0.1

Simulation-only metrics (simulate=True, requires ground-truth simulator):
  turb_rmse          RMSE of the algorithm's GP turbidity vs ground truth on a fixed test set
  ratio_rmse         RMSE of the algorithm's GP ratio vs ground truth on a fixed test set
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommenders.metrics import define_boundary_levelset_3d
from analysis.plot_3d_interactive import compute_gp_grids
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

def _simulate_ground_truth(concs_mm, surfactant_library):
    """Noise-free version of simulate_measurements_nd for test-set evaluation."""
    SIGMA_LOG = 0.35
    sum_anionic = 1e-10
    sum_cationic = 1e-10
    for s, c in concs_mm.items():
        cat = surfactant_library[s].get('category', 'nonionic')
        if cat == 'anionic':
            sum_anionic += c
        elif cat == 'cationic':
            sum_cationic += c
    log_imbalance = np.log10(sum_anionic / sum_cationic)
    turb_peak = np.exp(-(log_imbalance / SIGMA_LOG) ** 2)
    total_conc = sum_anionic + sum_cationic
    conc_gate = 1.0 / (1.0 + np.exp(-4.0 * (np.log10(total_conc) - np.log10(0.5))))
    turbidity = float(np.clip(0.04 + (1.0 - 0.04) * turb_peak * conc_gate, 0.02, 1.0))
    mixed_cmc = sum(c / surfactant_library[s]['cmc_mm'] for s, c in concs_mm.items())
    ratio_drop = 1.0 / (1.0 + np.exp(-5.0 * np.log10(max(mixed_cmc, 1e-9))))
    ratio = float(np.clip(0.85 - (0.85 - 0.70) * ratio_drop, 0.70, 0.95))
    return turbidity, ratio


def _build_test_set(surfactants, log10_bounds, span, n=500):
    """Generate a fixed feasible test set in normalised [0,1]^3 space.

    Returns (test_norm, test_turb_true, test_ratio_true) using the
    noise-free simulator as ground truth.
    """
    from scipy.stats.qmc import Sobol as _Sobol
    from workflows.surfactant_grid_adaptive_concentrations import SURFACTANT_LIBRARY

    seq = _Sobol(d=len(surfactants), scramble=True, seed=7)
    raw = seq.random(4096)
    log_c = log10_bounds[:, 0] + raw * span
    concs = 10.0 ** log_c
    # Feasibility: sum of concentrations <= 2x max single-surfactant concentration
    # (proxy for simplex budget — good enough for a test set)
    budget = concs.max(axis=1).max() * len(surfactants) * 0.6
    feas = concs.sum(axis=1) <= budget
    test_concs = concs[feas][:n]
    test_norm = raw[feas][:n]

    turb_true = np.array([
        _simulate_ground_truth(dict(zip(surfactants, row)), SURFACTANT_LIBRARY)[0]
        for row in test_concs
    ])
    ratio_true = np.array([
        _simulate_ground_truth(dict(zip(surfactants, row)), SURFACTANT_LIBRARY)[1]
        for row in test_concs
    ])
    return test_norm, turb_true, ratio_true


def _fit_gp_predict(train_norm, y_train, test_norm, log_target=False):
    """Fit a lightweight GP on train_norm -> y_train, predict on test_norm."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_norm)
    X_test = scaler.transform(test_norm)
    if log_target:
        y_fit = np.log(np.clip(y_train, 1e-6, None))
    else:
        y_fit = y_train
    kernel = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                  normalize_y=True)
    gp.fit(X_train, y_fit)
    y_pred = gp.predict(X_test)
    if log_target:
        y_pred = np.exp(y_pred)
    return y_pred


TURBIDITY_THRESHOLD = 0.1


def compute_iteration_metrics(csv_path, surfactants, output_folder, simulate=False):
    """Compute per-iteration cumulative metrics from results_final.csv.

    Parameters
    ----------
    csv_path      : path to results_final.csv (must have an 'iteration' column)
    surfactants   : list of exactly 3 surfactant names (e.g. ['SDS', 'TTAB', 'DTAB'])
    output_folder : unused; kept for API consistency
    simulate      : if True, also compute turb_rmse and ratio_rmse against the
                    noise-free ground-truth simulator (simulation runs only)

    Returns
    -------
    pd.DataFrame with one row per iteration.
    Columns: iteration, n_picks, turb_hit_rate, n_cloud_measured
             [+ turb_rmse, ratio_rmse if simulate=True]
    """
    df = pd.read_csv(csv_path)

    if 'iteration' not in df.columns:
        raise ValueError(
            "'iteration' column missing from results_final.csv. "
            "Re-run the workflow with the updated code to generate this column."
        )

    conc_cols = [f"{s}_conc_mm" for s in surfactants]

    # --- GP grid (turbidity only — for hit_rate) ---
    turb_grid, _, axes, log10_bounds = compute_gp_grids(csv_path, surfactants)

    # --- Turbidity boundary: level-set shell at 0.1 OD ---
    turb_mask, turb_ok = define_boundary_levelset_3d(turb_grid, TURBIDITY_THRESHOLD)

    # --- Normalize picks into [0,1]^3 ---
    span = log10_bounds[:, 1] - log10_bounds[:, 0]
    span = np.where(span < 1e-12, 1.0, span)

    # --- Ground-truth test set (simulation only) ---
    if simulate:
        test_norm, test_turb_true, test_ratio_true = _build_test_set(
            surfactants, log10_bounds, span
        )

    def normalize_picks(rows_df):
        concs = rows_df[conc_cols].values.astype(float)
        log_concs = np.log10(np.clip(concs, 1e-9, None))
        return np.clip((log_concs - log10_bounds[:, 0]) / span, 0.0, 1.0)

    # --- Per-iteration cumulative metrics ---
    max_iter = int(df['iteration'].max())
    rows = []

    for i in range(0, max_iter + 1):
        subset = df[df['iteration'] <= i]
        if len(subset) == 0:
            continue

        picks_norm = normalize_picks(subset)
        row = {'iteration': i, 'n_picks': len(subset)}

        n_grid = len(axes[0])

        # Turbidity hit rate: fraction of picks on the 0.1 OD shell
        if turb_ok:
            ix = np.clip((picks_norm[:, 0] * (n_grid - 1)).round().astype(int), 0, n_grid - 1)
            iy = np.clip((picks_norm[:, 1] * (n_grid - 1)).round().astype(int), 0, n_grid - 1)
            iz = np.clip((picks_norm[:, 2] * (n_grid - 1)).round().astype(int), 0, n_grid - 1)
            row['turb_hit_rate'] = float(turb_mask[ix, iy, iz].mean())
        else:
            row['turb_hit_rate'] = float('nan')

        # Directly measured turbid wells (no GP)
        row['n_cloud_measured'] = int((subset['turbidity_600'] > TURBIDITY_THRESHOLD).sum())

        # Simulation RMSE: how accurately does this algorithm's GP predict truth?
        if simulate and len(picks_norm) >= 4:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                turb_pred = _fit_gp_predict(
                    picks_norm,
                    subset['turbidity_600'].values.astype(float),
                    test_norm, log_target=True,
                )
                ratio_pred = _fit_gp_predict(
                    picks_norm,
                    subset['ratio'].values.astype(float),
                    test_norm, log_target=False,
                )
            row['turb_rmse'] = float(np.sqrt(np.mean((turb_pred - test_turb_true) ** 2)))
            row['ratio_rmse'] = float(np.sqrt(np.mean((ratio_pred - test_ratio_true) ** 2)))
        elif simulate:
            row['turb_rmse'] = float('nan')
            row['ratio_rmse'] = float('nan')

        rows.append(row)

    return pd.DataFrame(rows)


# Subtitle text for each metric panel (italicised in the plot)
_SUBTITLES = {
    'turb_hit_rate': (
        "Fraction of cumulative picks landing on the\n"
        "GP-predicted turbidity = 0.1 OD surface"
    ),
    'n_cloud_measured': (
        "Count of measured wells with turbidity_600 > 0.1\n"
        "(directly measured, no GP)"
    ),
    'turb_rmse': (
        "RMSE of the algorithm's GP turbidity vs noise-free ground truth\n"
        "on a fixed 500-point test set. Lower = better model. Simulation only."
    ),
    'ratio_rmse': (
        "RMSE of the algorithm's GP ratio vs noise-free ground truth\n"
        "on a fixed 500-point test set. Lower = better model. Simulation only."
    ),
}

_TITLES = {
    'turb_hit_rate':    'Turbidity Surface Hit Rate',
    'n_cloud_measured': 'Measured Turbid Wells',
    'turb_rmse':        'Turbidity Model RMSE',
    'ratio_rmse':       'Ratio Model RMSE',
}

_YLABELS = {
    'turb_hit_rate':    'Fraction of picks on surface',
    'n_cloud_measured': 'Cumulative count',
    'turb_rmse':        'RMSE (OD600)',
    'ratio_rmse':       'RMSE (ratio)',
}


def save_iteration_metrics(metrics_df, output_folder):
    """Save metrics DataFrame to CSV and a 6-panel PNG plot.

    Returns (csv_path, png_path).
    """
    csv_path = os.path.join(output_folder, "iteration_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)

    cols = [c for c in ['turb_hit_rate', 'n_cloud_measured', 'turb_rmse', 'ratio_rmse']
            if c in metrics_df.columns]
    n_cols = len(cols)
    fig, axes_grid = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    fig.patch.set_facecolor('#f8f8f8')

    for ax, col in zip(axes_grid.flatten(), cols):
        if col not in metrics_df.columns:
            ax.set_visible(False)
            continue

        ax.plot(metrics_df['iteration'], metrics_df[col],
                'o-', lw=2, ms=5, color='#2c6fad')
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xlabel('Cumulative iterations', fontsize=9)
        ax.set_ylabel(_YLABELS[col], fontsize=9)

        # Bold title
        ax.set_title(_TITLES[col], fontsize=11, fontweight='bold', pad=16)

        # Italicised subtitle below the title, inside the axes
        ax.text(
            0.5, 1.01,
            _SUBTITLES[col],
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=7.5,
            style='italic',
            color='#444444',
            linespacing=1.4,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle('Per-Iteration Exploration Metrics', fontsize=13,
                 fontweight='bold', y=0.995)

    png_path = os.path.join(output_folder, "iteration_metrics.png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    return csv_path, png_path


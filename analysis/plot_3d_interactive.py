"""
Standalone interactive 3D scatter + isosurface plots for N-D surfactant results.

Usage:
    python analysis/plot_3d_interactive.py path/to/results_final.csv

Produces three HTML files:
  interactive_turbidity_3d.html   - scatter colored by turbidity
  interactive_ratio_3d.html       - scatter colored by ratio
  isosurface_turbidity_3d.html    - semi-transparent surface at TURBIDITY_THRESHOLD
                                    showing the boundary between clear and turbid,
                                    with raw scatter underneath
"""

import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

TURBIDITY_THRESHOLD = 0.1    # boundary surface level
GRID_N = 30                  # points per axis for GP prediction grid (30^3 = 27k)


def compute_gp_grids(csv_path, surfactants=None, grid_n=GRID_N):
    """Fit GPs to turbidity and ratio and predict on a normalized [0,1]^3 grid.

    Used by analysis/iteration_metrics.py for retrospective boundary analysis.

    Parameters
    ----------
    csv_path    : path to results_final.csv
    surfactants : list of 3 surfactant names; auto-detected from columns if None
    grid_n      : points per axis (default: GRID_N = 30)

    Returns
    -------
    turb_grid    : (grid_n, grid_n, grid_n) GP-predicted turbidity
    ratio_grid   : (grid_n, grid_n, grid_n) GP-predicted ratio
    axes         : tuple of 3 np.linspace(0, 1, grid_n) arrays (normalized [0,1])
    log10_bounds : np.ndarray shape (3, 2) — log10 [min, max] per surfactant axis.
                   Normalize picks via:
                     (log10(conc) - log10_bounds[:, 0]) /
                     (log10_bounds[:, 1] - log10_bounds[:, 0])
    """
    df = pd.read_csv(csv_path)
    if 'well_type' in df.columns:
        df = df[df['well_type'] == 'experiment'].copy()
    if surfactants is None:
        conc_cols = [c for c in df.columns
                     if c.endswith("_conc_mm") and "substock" not in c
                     and not c.startswith("water")]
        surfactants = [c.replace("_conc_mm", "") for c in conc_cols]
    else:
        conc_cols = [f"{s}_conc_mm" for s in surfactants]

    if len(surfactants) != 3:
        raise ValueError(f"compute_gp_grids requires exactly 3 surfactants, got {surfactants}")

    log_pts = np.column_stack([
        np.log10(df[c].values.astype(float)) for c in conc_cols
    ])
    turb = df['turbidity_600'].values.astype(float)
    ratio = df['ratio'].values.astype(float)

    # Log10 bounds used by callers to normalize picks into [0, 1]
    log10_min = log_pts.min(axis=0)   # shape (3,)
    log10_max = log_pts.max(axis=0)   # shape (3,)
    log10_bounds = np.column_stack([log10_min, log10_max])  # shape (3, 2)

    # Normalized [0, 1] axes (what metrics.py expects)
    axes = tuple(np.linspace(0.0, 1.0, grid_n) for _ in range(3))

    # Map normalized grid back to log10 space for GP prediction
    ax_log = [np.linspace(log10_min[i], log10_max[i], grid_n) for i in range(3)]
    Xg, Yg, Zg = np.meshgrid(ax_log[0], ax_log[1], ax_log[2], indexing='ij')
    grid_pts_log = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])

    scaler = StandardScaler()
    pts_scaled = scaler.fit_transform(log_pts)
    grid_pts_scaled = scaler.transform(grid_pts_log)

    # Turbidity GP (log-transformed target for better Gaussian behaviour)
    log_turb = np.log(np.clip(turb, 1e-6, None))
    kernel_t = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.05)
    gp_turb = GaussianProcessRegressor(kernel=kernel_t, n_restarts_optimizer=3, normalize_y=True)
    gp_turb.fit(pts_scaled, log_turb)
    turb_grid = np.exp(gp_turb.predict(grid_pts_scaled)).reshape(grid_n, grid_n, grid_n)

    # Ratio GP (linear-space target — ratio is well-behaved ~0.7-0.9)
    kernel_r = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.001)
    gp_ratio = GaussianProcessRegressor(kernel=kernel_r, n_restarts_optimizer=3, normalize_y=True)
    gp_ratio.fit(pts_scaled, ratio)
    ratio_grid = gp_ratio.predict(grid_pts_scaled).reshape(grid_n, grid_n, grid_n)

    return turb_grid, ratio_grid, axes, log10_bounds


def plot_3d_interactive(csv_path, output_dir=None):
    df = pd.read_csv(csv_path)
    if 'well_type' in df.columns:
        df = df[df['well_type'] == 'experiment'].copy()

    # Detect surfactant concentration columns (exclude substock columns)
    conc_cols = [c for c in df.columns if c.endswith("_conc_mm") and "substock" not in c and not c.startswith("water")]
    surfactants = [c.replace("_conc_mm", "") for c in conc_cols]

    if len(surfactants) != 3:
        print(f"This script expects exactly 3 surfactants, found: {surfactants}")
        print("Plotting first 3 only." if len(surfactants) > 3 else "Cannot plot.")
        surfactants = surfactants[:3]
        conc_cols = conc_cols[:3]

    s0, s1, s2 = surfactants
    x = np.log10(df[conc_cols[0]].values.astype(float))
    y = np.log10(df[conc_cols[1]].values.astype(float))
    z = np.log10(df[conc_cols[2]].values.astype(float))

    hover_text = [
        f"{s0}: {df[conc_cols[0]].iloc[i]:.3f} mM<br>"
        f"{s1}: {df[conc_cols[1]].iloc[i]:.3f} mM<br>"
        f"{s2}: {df[conc_cols[2]].iloc[i]:.3f} mM<br>"
        f"Turbidity: {df['turbidity_600'].iloc[i]:.4f}<br>"
        f"Ratio: {df['ratio'].iloc[i]:.4f}"
        for i in range(len(df))
    ]

    axis_labels = dict(
        xaxis_title=f"log10({s0} mM)",
        yaxis_title=f"log10({s1} mM)",
        zaxis_title=f"log10({s2} mM)",
    )

    # ---- Figure 1: colored by turbidity ----
    fig_turb = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=df['turbidity_600'].values,
            colorscale='RdYlGn_r',   # red = high turbidity (precipitate)
            colorbar=dict(title='Turbidity 600'),
            opacity=0.85,
            cmin=float(df['turbidity_600'].quantile(0.02)),
            cmax=float(df['turbidity_600'].quantile(0.98)),
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
    ))
    fig_turb.update_layout(
        title=f'3D Turbidity — {s0} / {s1} / {s2}  ({len(df)} wells)',
        scene=axis_labels,
        width=900, height=700,
    )

    # ---- Figure 2: colored by ratio ----
    fig_ratio = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=df['ratio'].values,
            colorscale='RdYlGn',      # green = high ratio (below CMC)
            colorbar=dict(title='I373/I384 Ratio'),
            opacity=0.85,
            cmin=float(df['ratio'].quantile(0.02)),
            cmax=float(df['ratio'].quantile(0.98)),
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
    ))
    fig_ratio.update_layout(
        title=f'3D Ratio (I373/I384) — {s0} / {s1} / {s2}  ({len(df)} wells)',
        scene=axis_labels,
        width=900, height=700,
    )

    # Save
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)

    path_turb = os.path.join(output_dir, "interactive_turbidity_3d.html")
    path_ratio = os.path.join(output_dir, "interactive_ratio_3d.html")

    fig_turb.write_html(path_turb)
    fig_ratio.write_html(path_ratio)

    print(f"Saved: {path_turb}")
    print(f"Saved: {path_ratio}")

    return path_turb, path_ratio


def plot_isosurface(csv_path, threshold=TURBIDITY_THRESHOLD, output_dir=None):
    """Render the turbidity=threshold isosurface in 3D log-concentration space.

    Interpolates scattered measurements onto a regular grid, then renders the
    surface where turbidity == threshold as a semi-transparent mesh. Raw data
    points are shown underneath, colored by their measured turbidity value.
    """
    df = pd.read_csv(csv_path)
    if 'well_type' in df.columns:
        df = df[df['well_type'] == 'experiment'].copy()

    conc_cols = [c for c in df.columns if c.endswith("_conc_mm") and "substock" not in c and not c.startswith("water")]
    surfactants = [c.replace("_conc_mm", "") for c in conc_cols]

    if len(surfactants) != 3:
        print(f"Isosurface requires exactly 3 surfactants, found: {surfactants}")
        return None

    s0, s1, s2 = surfactants
    x = np.log10(df[conc_cols[0]].values.astype(float))
    y = np.log10(df[conc_cols[1]].values.astype(float))
    z = np.log10(df[conc_cols[2]].values.astype(float))
    turb = df['turbidity_600'].values.astype(float)
    # Log-transform turbidity for coloring and cloud
    turb_log = np.log10(np.clip(turb, 0.04, None))

    # Fit a GP to log-turbidity in log10-concentration space.
    # Matern(nu=1.5) matches what the recommenders use.
    # WhiteKernel absorbs measurement noise so the surface isn't pinned to outliers.
    pts = np.column_stack([x, y, z])
    log_turb = np.log10(np.clip(turb, 0.04, None))  # log10 for visual scaling

    scaler = StandardScaler()
    pts_scaled = scaler.fit_transform(pts)

    kernel = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.05)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
    print("  Fitting GP...")
    gp.fit(pts_scaled, log_turb)

    # Predict on regular grid
    xi = np.linspace(x.min(), x.max(), GRID_N)
    yi = np.linspace(y.min(), y.max(), GRID_N)
    zi = np.linspace(z.min(), z.max(), GRID_N)
    Xg, Yg, Zg = np.meshgrid(xi, yi, zi, indexing='ij')
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
    grid_pts_scaled = scaler.transform(grid_pts)

    print("  Predicting on grid...")
    log_turb_pred, log_turb_std = gp.predict(grid_pts_scaled, return_std=True)
    turb_grid_log = log_turb_pred.reshape(GRID_N, GRID_N, GRID_N)
    # For on/off cloud, binary mask above threshold in log space
    threshold_log = np.log10(max(threshold, 0.04))
    cloud_mask = (turb_grid_log >= threshold_log).astype(float)

    hover_text = [
        f"{s0}: {df[conc_cols[0]].iloc[i]:.3f} mM<br>"
        f"{s1}: {df[conc_cols[1]].iloc[i]:.3f} mM<br>"
        f"{s2}: {df[conc_cols[2]].iloc[i]:.3f} mM<br>"
        f"Turbidity: {turb[i]:.4f}"
        for i in range(len(df))
    ]

    fig = go.Figure()

    # Volumetric cloud: binary on/off mask in log-turbidity space
    fig.add_trace(go.Volume(
        x=Xg.ravel(), y=Yg.ravel(), z=Zg.ravel(),
        value=cloud_mask.ravel(),
        isomin=0.5,  # binary mask: 1=above threshold, 0=below
        isomax=1.0,
        opacity=0.18,  # more visible boundary
        surface_count=1,
        colorscale=[[0.0, 'rgba(255,0,0,0.0)'], [1.0, 'rgba(255,0,0,0.7)']],
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='GP turbid region',
    ))

    # Raw scatter underneath, colored by log-turbidity
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=turb_log,
            colorscale='RdYlGn_r',
            colorbar=dict(
                title='log10(Turbidity)',
                tickvals=[-1.4, -1.1, -0.7, -0.4, 0, 0.3, 0.7, 1.0],
                ticktext=['0.04', '0.08', '0.2', '0.4', '1', '2', '5', '10']
            ),
            opacity=0.7,
            cmin=np.log10(0.04),
            cmax=float(np.percentile(turb_log, 98)),
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name='Measured points',
    ))

    fig.update_layout(
        title=(
            f'GP Turbidity Cloud (threshold={threshold}) — '
            f'{s0} / {s1} / {s2}  ({len(df)} wells)<br>'
            '<sup>Red cloud = GP-predicted turbid region. Denser = more turbid. Points = measured data.</sup>'
        ),
        legend=dict(x=0.01, y=0.99),
        scene=dict(
            xaxis_title=f'log10({s0} mM)',
            yaxis_title=f'log10({s1} mM)',
            zaxis_title=f'log10({s2} mM)',
        ),
        width=950, height=750,
    )

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    path = os.path.join(output_dir, f"isosurface_turbidity_3d.html")
    fig.write_html(path)
    print(f"Saved: {path}")
    return path


def plot_ratio_phases(csv_path, output_dir=None):
    """3D phase map for ratio (I373/I384).

    Fits a GP to the ratio data, predicts on a regular 3D grid, then renders
    the entire concentration volume color-coded by phase:

      Blue   = micellar  (ratio below transition — surfactant aggregated)
      Red    = transition (ratio changing fastest — the mixed-CMC surface)
      Green  = sub-CMC   (ratio above transition — surfactant dissolved)

    The red band automatically lands where the gradient is steepest, without
    needing manual thresholds. Opacity is uniform so both phases and the
    transition sheet are visible simultaneously.
    """
    df = pd.read_csv(csv_path)
    if 'well_type' in df.columns:
        df = df[df['well_type'] == 'experiment'].copy()

    conc_cols = [c for c in df.columns if c.endswith("_conc_mm")
                 and "substock" not in c and not c.startswith("water")]
    surfactants = [c.replace("_conc_mm", "") for c in conc_cols]

    if len(surfactants) != 3:
        print(f"plot_ratio_phases requires exactly 3 surfactants, found: {surfactants}")
        return None

    s0, s1, s2 = surfactants
    x = np.log10(df[conc_cols[0]].values.astype(float))
    y = np.log10(df[conc_cols[1]].values.astype(float))
    z = np.log10(df[conc_cols[2]].values.astype(float))
    ratio = df['ratio'].values.astype(float)

    # Fit GP to ratio (linear space — ratio is already well-behaved 0.7–0.9)
    pts = np.column_stack([x, y, z])
    scaler = StandardScaler()
    pts_scaled = scaler.fit_transform(pts)

    kernel = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.001)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
    print("  Fitting GP to ratio...")
    gp.fit(pts_scaled, ratio)

    # Predict on regular grid
    xi = np.linspace(x.min(), x.max(), GRID_N)
    yi = np.linspace(y.min(), y.max(), GRID_N)
    zi = np.linspace(z.min(), z.max(), GRID_N)
    Xg, Yg, Zg = np.meshgrid(xi, yi, zi, indexing='ij')
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])

    print("  Predicting ratio on grid...")
    ratio_grid = gp.predict(scaler.transform(grid_pts)).reshape(GRID_N, GRID_N, GRID_N)

    # Compute gradient magnitude: where the ratio is changing fastest = CMC surface.
    # np.gradient returns d/dx, d/dy, d/dz in grid-index space; we normalise by
    # axis spacing so the magnitude is in ratio-units per log10(mM).
    dx = (xi[-1] - xi[0]) / (GRID_N - 1)
    dy = (yi[-1] - yi[0]) / (GRID_N - 1)
    dz = (zi[-1] - zi[0]) / (GRID_N - 1)
    gr, gc, gs = np.gradient(ratio_grid, dx, dy, dz)
    grad_mag = np.sqrt(gr**2 + gc**2 + gs**2).ravel()

    # Normalise to [0,1] for volume rendering; clip top 2% to suppress edge noise
    gm_min = float(np.percentile(grad_mag, 2))
    gm_max = float(np.percentile(grad_mag, 98))
    grad_norm = np.clip((grad_mag - gm_min) / (gm_max - gm_min + 1e-12), 0, 1)

    # Raw ratio at each grid point (for scatter coloring)
    ratio_flat = ratio_grid.ravel()
    r_low  = float(np.percentile(ratio, 5))
    r_high = float(np.percentile(ratio, 95))

    # Scatter colorscale: still use ratio value for the raw dots so you can
    # see which side of the transition each measured point is on.
    scatter_colorscale = [
        [0.0, 'rgb(30,100,200)'],   # blue  = below CMC (micellar)
        [0.5, 'rgb(200,200,200)'],  # grey  = mid
        [1.0, 'rgb(20,120,40)'],    # green = above CMC (sub-CMC)
    ]

    hover_text = [
        f"{s0}: {df[conc_cols[0]].iloc[i]:.3f} mM<br>"
        f"{s1}: {df[conc_cols[1]].iloc[i]:.3f} mM<br>"
        f"{s2}: {df[conc_cols[2]].iloc[i]:.3f} mM<br>"
        f"Ratio: {ratio[i]:.4f}"
        for i in range(len(df))
    ]

    fig = go.Figure()

    # Gradient magnitude volume — bright orange/red where ratio is changing fastest.
    # The CMC transition surface self-selects as the bright region; no thresholds needed.
    fig.add_trace(go.Volume(
        x=Xg.ravel(), y=Yg.ravel(), z=Zg.ravel(),
        value=grad_norm,
        isomin=0.15,   # ignore low-gradient (flat) regions
        isomax=1.0,
        opacity=0.07,
        surface_count=20,
        colorscale=[
            [0.0, 'rgba(255,240,200,0)'],   # transparent — flat region
            [0.3, 'rgb(255,180,50)'],        # yellow — moderate gradient
            [0.7, 'rgb(220,60,20)'],         # orange-red — steep gradient
            [1.0, 'rgb(160,0,0)'],           # deep red — sharpest transition
        ],
        colorbar=dict(title='|∇ratio|<br>(norm.)'),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='CMC transition (gradient)',
    ))

    # Raw scatter colored by ratio value — shows which phase each point is in
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=ratio,
            colorscale=scatter_colorscale,
            cmin=r_low, cmax=r_high,
            opacity=0.9,
            line=dict(width=0.5, color='black'),
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name='Measured points (ratio)',
    ))

    fig.update_layout(
        title=(
            f'3D CMC Transition Surface — {s0} / {s1} / {s2}  ({len(df)} wells)<br>'
            '<sup>Red/orange cloud = where ratio changes fastest (CMC surface). '
            'Dots: blue = micellar, green = sub-CMC.</sup>'
        ),
        legend=dict(x=0.01, y=0.99),
        scene=dict(
            xaxis_title=f'log10({s0} mM)',
            yaxis_title=f'log10({s1} mM)',
            zaxis_title=f'log10({s2} mM)',
        ),
        width=950, height=750,
    )

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    path = os.path.join(output_dir, "ratio_phases_3d.html")
    fig.write_html(path)
    print(f"Saved: {path}")
    return path


def plot_ratio_isosurfaces(csv_path, control_csv_path=None, output_dir=None):
    """Render two ratio isosurfaces defined by CMC control fits.

    high-boundary = min(A2) - 0.005
    low-boundary  = max(A1) + 0.005

    Looks for control CMC rows (control_type starting with 'cmc_1d_{surf}_')
    in `control_csv_path` (falls back to results_after_initial_grid.csv in
    the same folder as `csv_path`, or to `csv_path` itself).
    """
    # Read main data
    df = pd.read_csv(csv_path)
    if 'well_type' in df.columns:
        df = df[df['well_type'] == 'experiment'].copy()

    conc_cols = [c for c in df.columns if c.endswith("_conc_mm") and "substock" not in c and not c.startswith("water")]
    surfactants = [c.replace("_conc_mm", "") for c in conc_cols]
    if len(surfactants) != 3:
        print(f"plot_ratio_isosurfaces requires exactly 3 surfactants, found: {surfactants}")
        return None

    # Find control CSV
    if control_csv_path is None:
        candidate = os.path.join(os.path.dirname(csv_path), "results_after_initial_grid.csv")
        if os.path.exists(candidate):
            control_csv_path = candidate
        else:
            control_csv_path = csv_path

    try:
        control_df = pd.read_csv(control_csv_path)
    except Exception:
        control_df = pd.DataFrame()

    # Compute A1/A2 per surfactant using existing control fitting routine
    A1_list = []
    A2_list = []
    try:
        from analysis.control_cmc_analysis import fit_cmc_curve
    except Exception:
        fit_cmc_curve = None

    for s in surfactants:
        conc_col = f"{s}_conc_mm"
        if not control_df.empty and 'control_type' in control_df.columns:
            mask = control_df['control_type'].astype(str).str.contains(f'cmc_1d_{s}', na=False)
            ctrl = control_df[mask].copy()
        else:
            ctrl = pd.DataFrame()

        if len(ctrl) >= 4 and fit_cmc_curve is not None and conc_col in ctrl.columns:
            cmc_est, r2, popt, xdata = fit_cmc_curve(ctrl[conc_col].values, ctrl['ratio'].values)
            if popt is not None:
                A1_list.append(popt[0])
                A2_list.append(popt[1])
                continue

        # Fallback: use percentiles of control rows (if present) or overall data
        if len(ctrl) > 0 and conc_col in ctrl.columns:
            A1_list.append(float(np.percentile(ctrl['ratio'].values, 95)))
            A2_list.append(float(np.percentile(ctrl['ratio'].values, 5)))
        else:
            A1_list.append(float(np.percentile(df['ratio'].values, 95)))
            A2_list.append(float(np.percentile(df['ratio'].values, 5)))

    # Corrected boundaries per user: low = min(A1) - 0.005, high = max(A2) + 0.005
    low_boundary = min(A1_list) - 0.005
    high_boundary = max(A2_list) + 0.005

    # Fit GP to ratio (reuse logic from plot_ratio_phases)
    s0, s1, s2 = surfactants
    x = np.log10(df[f"{s0}_conc_mm"].values.astype(float))
    y = np.log10(df[f"{s1}_conc_mm"].values.astype(float))
    z = np.log10(df[f"{s2}_conc_mm"].values.astype(float))
    ratio = df['ratio'].values.astype(float)

    pts = np.column_stack([x, y, z])
    scaler = StandardScaler()
    pts_scaled = scaler.fit_transform(pts)

    kernel = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.001)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
    print("  Fitting GP to ratio for isosurfaces...")
    gp.fit(pts_scaled, ratio)

    xi = np.linspace(x.min(), x.max(), GRID_N)
    yi = np.linspace(y.min(), y.max(), GRID_N)
    zi = np.linspace(z.min(), z.max(), GRID_N)
    Xg, Yg, Zg = np.meshgrid(xi, yi, zi, indexing='ij')
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])

    print("  Predicting ratio on grid for isosurfaces...")
    ratio_grid = gp.predict(scaler.transform(grid_pts)).reshape(GRID_N, GRID_N, GRID_N)

    fig = go.Figure()

    # Low-boundary isosurface (from min(A1)-0.005) — colored blue
    fig.add_trace(go.Isosurface(
        x=Xg.ravel(), y=Yg.ravel(), z=Zg.ravel(),
        value=ratio_grid.ravel(),
        isomin=low_boundary, isomax=low_boundary,
        surface_count=1,
        opacity=0.35,
        colorscale=[[0.0, 'rgb(30,120,200)'], [1.0, 'rgb(30,120,200)']],
        name=f'Low-boundary (min(A1)-0.005 = {low_boundary:.3f})',
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))

    # High-boundary isosurface (from max(A2)+0.005) — colored red
    fig.add_trace(go.Isosurface(
        x=Xg.ravel(), y=Yg.ravel(), z=Zg.ravel(),
        value=ratio_grid.ravel(),
        isomin=high_boundary, isomax=high_boundary,
        surface_count=1,
        opacity=0.4,
        colorscale=[[0.0, 'rgb(200,30,30)'], [1.0, 'rgb(200,30,30)']],
        name=f'High-boundary (max(A2)+0.005 = {high_boundary:.3f})',
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))

    # Raw scatter colored by ratio for context
    hover_text = [
        f"{s0}: {df[f'{s0}_conc_mm'].iloc[i]:.3f} mM<br>"
        f"{s1}: {df[f'{s1}_conc_mm'].iloc[i]:.3f} mM<br>"
        f"{s2}: {df[f'{s2}_conc_mm'].iloc[i]:.3f} mM<br>"
        f"Ratio: {ratio[i]:.4f}"
        for i in range(len(df))
    ]

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=4, color=ratio, colorscale='RdYlGn', colorbar=dict(title='Ratio'), opacity=0.9,
                    cmin=float(np.percentile(ratio, 2)), cmax=float(np.percentile(ratio, 98))),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name='Measured points (ratio)',
    ))

    fig.update_layout(
        title=(f'Ratio Isosurfaces — {s0} / {s1} / {s2}  ({len(df)} wells)'),
        scene=dict(xaxis_title=f'log10({s0} mM)', yaxis_title=f'log10({s1} mM)', zaxis_title=f'log10({s2} mM)'),
        width=950, height=750,
    )

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    path = os.path.join(output_dir, "ratio_isosurfaces_3d.html")
    fig.write_html(path)
    print(f"Saved: {path}")
    return path


def plot_init_feasible_overlay_3d(results_csv_or_df, surfactants,
                                  boundary_points, output_dir=None,
                                  feasible_config=None, grid_n=32,
                                  dropped_points=None):
    """Plot 3D feasible-region overlay with init picks highlighted.

    Args:
        results_csv_or_df: CSV path or DataFrame containing measured data
        surfactants: ordered list of three surfactant names
        boundary_points: list of dicts with concentration keys for surfactants.
            Used as fallback visualization when feasible_config is not provided.
        output_dir: directory for HTML output
        feasible_config: optional dict with keys stock_concs, well_volume_ul,
            surfactant_budget_ul, min_conc_mm, and optional max_conc_multiplier.
            When provided, render a true budget-feasibility isosurface from a
            3D grid (consistent with 2D mask-based visualization).
        grid_n: points per axis for 3D feasibility grid.
    """
    if len(surfactants) != 3:
        print(f"plot_init_feasible_overlay_3d requires 3 surfactants, got {surfactants}")
        return None

    if isinstance(results_csv_or_df, str):
        df = pd.read_csv(results_csv_or_df)
        if output_dir is None:
            output_dir = os.path.dirname(results_csv_or_df)
    else:
        df = results_csv_or_df.copy()

    if output_dir is None:
        output_dir = os.getcwd()

    if 'well_type' in df.columns:
        df = df[df['well_type'] == 'experiment'].copy()
    if len(df) == 0:
        print("plot_init_feasible_overlay_3d: no experiment rows")
        return None

    if 'iteration' in df.columns:
        init_df = df[df['iteration'] == 0].copy()
    else:
        init_df = df.copy()
    if len(init_df) == 0:
        print("plot_init_feasible_overlay_3d: no iteration==0 rows")
        return None

    s0, s1, s2 = surfactants
    c0 = f"{s0}_conc_mm"
    c1 = f"{s1}_conc_mm"
    c2 = f"{s2}_conc_mm"

    ix = np.log10(init_df[c0].values.astype(float))
    iy = np.log10(init_df[c1].values.astype(float))
    iz = np.log10(init_df[c2].values.astype(float))

    fig = go.Figure()

    if feasible_config is not None:
        stock_concs = feasible_config["stock_concs"]
        well_volume_ul = float(feasible_config["well_volume_ul"])
        budget_ul = float(feasible_config["surfactant_budget_ul"])
        min_conc_mm = float(feasible_config["min_conc_mm"])
        max_conc_multiplier = float(feasible_config.get("max_conc_multiplier", 1.0))

        a0 = well_volume_ul / float(stock_concs[s0])
        a1 = well_volume_ul / float(stock_concs[s1])
        a2 = well_volume_ul / float(stock_concs[s2])

        # Axis maxima with other surfactants pinned at minimum concentration.
        x_max = max((budget_ul - a1 * min_conc_mm - a2 * min_conc_mm) / a0, min_conc_mm)
        y_max = max((budget_ul - a0 * min_conc_mm - a2 * min_conc_mm) / a1, min_conc_mm)
        z_max = max((budget_ul - a0 * min_conc_mm - a1 * min_conc_mm) / a2, min_conc_mm)

        x_max *= max_conc_multiplier
        y_max *= max_conc_multiplier
        z_max *= max_conc_multiplier

        lx = np.linspace(np.log10(min_conc_mm), np.log10(x_max), grid_n)
        ly = np.linspace(np.log10(min_conc_mm), np.log10(y_max), grid_n)
        lz = np.linspace(np.log10(min_conc_mm), np.log10(z_max), grid_n)
        LX, LY, LZ = np.meshgrid(lx, ly, lz, indexing='ij')
        X = 10 ** LX
        Y = 10 ** LY
        Z = 10 ** LZ

        feasible = (a0 * X + a1 * Y + a2 * Z) <= budget_ul
        values = feasible.astype(float)

        fig.add_trace(go.Isosurface(
            x=LX.ravel(),
            y=LY.ravel(),
            z=LZ.ravel(),
            value=values.ravel(),
            isomin=0.5,
            isomax=0.5,
            surface_count=1,
            opacity=0.18,
            colorscale=[[0.0, 'royalblue'], [1.0, 'royalblue']],
            caps=dict(x_show=False, y_show=False, z_show=False),
            name='Budget-feasible envelope',
            hoverinfo='skip',
            showscale=False,
        ))
    else:
        bdf = pd.DataFrame(boundary_points)
        if len(bdf) == 0:
            print("plot_init_feasible_overlay_3d: no boundary points")
            return None

        def _col(df_like, surf, conc_col):
            if conc_col in df_like.columns:
                return conc_col
            return surf

        b0 = _col(bdf, s0, c0)
        b1 = _col(bdf, s1, c1)
        b2 = _col(bdf, s2, c2)

        bx = np.log10(bdf[b0].values.astype(float))
        by = np.log10(bdf[b1].values.astype(float))
        bz = np.log10(bdf[b2].values.astype(float))

        fig.add_trace(go.Mesh3d(
            x=bx, y=by, z=bz,
            alphahull=0,
            color='royalblue',
            opacity=0.18,
            name='Geometric feasible envelope',
            hoverinfo='skip',
            showscale=False,
        ))

    n_total_init = len(init_df)
    n_boundary_init = 0
    n_sobol_init = 0
    ddf = pd.DataFrame(dropped_points) if dropped_points is not None else pd.DataFrame()

    if "_init_source_type" in init_df.columns:
        boundary_mask = (init_df["_init_source_type"] == "boundary").values
        sobol_mask = (init_df["_init_source_type"] == "sobol").values
        n_boundary_init = int(np.sum(boundary_mask))
        n_sobol_init = int(np.sum(sobol_mask))

        if n_boundary_init > 0:
            fig.add_trace(go.Scatter3d(
                x=ix[boundary_mask], y=iy[boundary_mask], z=iz[boundary_mask],
                mode='markers',
                marker=dict(size=7, color='#2c5aa0', symbol='diamond-open', opacity=0.95,
                            line=dict(color='white', width=1.0)),
                name=f'Init boundary picks (n={n_boundary_init})',
            ))

        if n_sobol_init > 0:
            fig.add_trace(go.Scatter3d(
                x=ix[sobol_mask], y=iy[sobol_mask], z=iz[sobol_mask],
                mode='markers',
                marker=dict(size=6, color='#d62728', symbol='square', opacity=0.95,
                            line=dict(color='white', width=0.8)),
                name=f'Init sobol picks (n={n_sobol_init})',
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=ix, y=iy, z=iz,
            mode='markers',
            marker=dict(size=6, color='#d62728', symbol='diamond', opacity=0.95,
                        line=dict(color='white', width=0.8)),
            name=f'Init picks (iteration 0, n={n_total_init})',
        ))

    if len(ddf) > 0:
        def _col(df_like, surf, conc_col):
            if conc_col in df_like.columns:
                return conc_col
            return surf

        d0 = _col(ddf, s0, c0)
        d1 = _col(ddf, s1, c1)
        d2 = _col(ddf, s2, c2)
        dx = np.log10(ddf[d0].values.astype(float))
        dy = np.log10(ddf[d1].values.astype(float))
        dz = np.log10(ddf[d2].values.astype(float))

        fig.add_trace(go.Scatter3d(
            x=dx, y=dy, z=dz,
            mode='markers',
            marker=dict(size=4, color='#111111', symbol='x', opacity=0.85,
                        line=dict(color='white', width=0.6)),
            name=f'Dropped init candidates (n={len(ddf)})',
        ))

    fig.update_layout(
        title=(
            f'Feasible Region vs Init Picks — {s0} / {s1} / {s2}<br>'
            f'<sup>init total={n_total_init}, boundary={n_boundary_init}, sobol={n_sobol_init}, dropped={len(ddf)}</sup>'
        ),
        scene=dict(
            xaxis_title=f'log10({s0} mM)',
            yaxis_title=f'log10({s1} mM)',
            zaxis_title=f'log10({s2} mM)',
        ),
        width=950,
        height=760,
        legend=dict(x=0.01, y=0.99),
    )

    path = os.path.join(output_dir, "init_feasible_overlay_3d.html")
    fig.write_html(path)
    print(f"Saved: {path}")
    return path



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis/plot_3d_interactive.py path/to/results_final.csv")
        sys.exit(1)
    csv_path = sys.argv[1]
    #plot_3d_interactive(csv_path)
    plot_isosurface(csv_path)
    #plot_ratio_phases(csv_path)

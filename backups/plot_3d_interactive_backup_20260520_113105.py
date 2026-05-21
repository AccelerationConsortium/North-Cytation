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


def plot_3d_interactive(csv_path, output_dir=None):
    df = pd.read_csv(csv_path)

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

    # Fit a GP to log-turbidity in log10-concentration space.
    # Matern(nu=1.5) matches what the recommenders use.
    # WhiteKernel absorbs measurement noise so the surface isn't pinned to outliers.
    pts = np.column_stack([x, y, z])
    log_turb = np.log(np.clip(turb, 1e-6, None))  # log-space is more Gaussian

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
    turb_grid = np.exp(log_turb_pred)            # back to linear turbidity
    turb_grid_upper = np.exp(log_turb_pred + log_turb_std)  # +1 sigma shell

    hover_text = [
        f"{s0}: {df[conc_cols[0]].iloc[i]:.3f} mM<br>"
        f"{s1}: {df[conc_cols[1]].iloc[i]:.3f} mM<br>"
        f"{s2}: {df[conc_cols[2]].iloc[i]:.3f} mM<br>"
        f"Turbidity: {turb[i]:.4f}"
        for i in range(len(df))
    ]

    fig = go.Figure()

    # Volumetric cloud: show the full region where GP predicts turbidity > threshold.
    # go.Volume maps opacity to value — points deep inside the turbid zone are opaque,
    # points near the boundary are translucent, points below threshold are invisible.
    # Clip values below threshold to 0 so only the turbid region renders.
    turb_clipped = np.where(turb_grid >= threshold, turb_grid, 0.0)
    fig.add_trace(go.Volume(
        x=Xg.ravel(), y=Yg.ravel(), z=Zg.ravel(),
        value=turb_clipped,
        isomin=threshold,
        isomax=float(turb_clipped.max()),
        opacity=0.08,           # low per-voxel opacity; cloud emerges from accumulation
        surface_count=15,       # more slices = denser cloud
        colorscale='Reds',
        colorbar=dict(title='GP turbidity'),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='GP turbid region',
    ))

    # Raw scatter underneath, colored by turbidity
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=turb,
            colorscale='RdYlGn_r',
            colorbar=dict(title='Turbidity'),
            opacity=0.7,
            cmin=float(np.percentile(turb, 2)),
            cmax=float(np.percentile(turb, 98)),
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



    if len(sys.argv) < 2:
        print("Usage: python analysis/plot_3d_interactive.py path/to/results_final.csv")
        sys.exit(1)
    plot_3d_interactive(sys.argv[1])
    plot_isosurface(sys.argv[1])

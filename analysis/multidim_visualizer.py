"""
Pairwise marginal contour maps for N-D surfactant experiments.

For each pair of surfactants, all other concentration axes are averaged out,
producing a 2D heatmap comparable to the 2D contour plots.  Works for any
N in [2, 5].  Called automatically at the end of run_multidim_workflow() but
can also be run standalone on any results CSV.

Usage (standalone):
    python analysis/multidim_visualizer.py path/to/results_final.csv
"""

import sys
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import griddata


# ── helpers ──────────────────────────────────────────────────────────────────

def _pairwise_grid(df, surf_x, surf_y, metric, grid_n=40):
    """Interpolate metric onto a 2-D log-spaced grid for (surf_x, surf_y),
    marginalising over all other concentration axes by taking the mean of
    the nearest points in bins."""
    col_x = f"{surf_x}_conc_mm"
    col_y = f"{surf_y}_conc_mm"

    x = np.log10(df[col_x].values.astype(float))
    y = np.log10(df[col_y].values.astype(float))
    z = df[metric].values.astype(float)

    xi = np.linspace(x.min(), x.max(), grid_n)
    yi = np.linspace(y.min(), y.max(), grid_n)
    Xi, Yi = np.meshgrid(xi, yi)

    # linear then nearest fallback for edge NaNs
    Zi = griddata((x, y), z, (Xi, Yi), method="linear")
    Zi_nn = griddata((x, y), z, (Xi, Yi), method="nearest")
    Zi = np.where(np.isnan(Zi), Zi_nn, Zi)
    return Xi, Yi, Zi


def _tick_labels(log_vals):
    """Convert log10 tick positions to readable mM strings (scientific notation for small values)."""
    vals = 10 ** log_vals
    labels = []
    for v in vals:
        if v >= 0.1:
            labels.append(f"{v:.1f}")
        else:
            # Use scientific notation for very small values
            labels.append(f"{v:.1e}")
    return labels


def _plot_one_panel(ax, Xi, Yi, Zi, surf_x, surf_y, metric, vmin, vmax, cmap):
    """Draw a single contour panel."""
    try:
        levels = np.linspace(vmin, vmax, 14)
        cs = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, alpha=0.85)
        ax.contour(Xi, Yi, Zi, levels=levels, colors="k", linewidths=0.3, alpha=0.25)
    except Exception:
        cs = ax.pcolormesh(Xi, Yi, Zi, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")

    # Axis labels with nice tick formatting
    xt = np.linspace(Xi.min(), Xi.max(), 5)
    yt = np.linspace(Yi.min(), Yi.max(), 5)
    ax.set_xticks(xt)
    ax.set_xticklabels(_tick_labels(xt), fontsize=7)
    ax.set_yticks(yt)
    ax.set_yticklabels(_tick_labels(yt), fontsize=7)
    ax.set_xlabel(f"{surf_x} (mM)", fontsize=8)
    ax.set_ylabel(f"{surf_y} (mM)", fontsize=8)
    ax.set_title(f"{surf_x} vs {surf_y}", fontsize=9)
    return cs


# ── main public function ──────────────────────────────────────────────────────

def plot_pairwise_maps(results_csv_or_df, surfactants, output_folder,
                       metrics=("ratio", "turbidity_600"), grid_n=40):
    """Generate one figure per metric, each containing a pairwise heatmap grid.

    Args:
        results_csv_or_df: path to CSV or DataFrame
        surfactants: list of surfactant names (e.g. ['SDS','TTAB','CTAB','DTAB'])
        output_folder: where to save PNGs
        metrics: tuple of column names to plot
        grid_n: interpolation grid resolution per axis
    """
    if isinstance(results_csv_or_df, str):
        df = pd.read_csv(results_csv_or_df)
    else:
        df = results_csv_or_df.copy()

    # Keep only experiment rows with valid measurements
    df = df[df["well_type"] == "experiment"].copy()
    for m in metrics:
        df = df[df[m].notna()]
    if len(df) == 0:
        print("multidim_visualizer: no valid experiment rows — skipping plots")
        return {}

    pairs = list(itertools.combinations(surfactants, 2))
    n_pairs = len(pairs)

    # Layout: up to 3 columns
    n_cols = min(3, n_pairs)
    n_rows = int(np.ceil(n_pairs / n_cols))

    saved = {}
    metric_labels = {"ratio": "I373/I384 ratio", "turbidity_600": "Turbidity 600nm"}
    cmaps = {"ratio": "RdYlGn", "turbidity_600": "viridis"}

    for metric in metrics:
        if metric not in df.columns:
            continue

        vmin = float(df[metric].quantile(0.02))
        vmax = float(df[metric].quantile(0.98))
        if vmin >= vmax:
            vmin = float(df[metric].min())
            vmax = float(df[metric].max())

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.5 * n_cols, 4 * n_rows),
                                 squeeze=False)
        fig.suptitle(
            f"{metric_labels.get(metric, metric)}  —  pairwise marginal maps\n"
            f"({len(df)} wells, {len(surfactants)}D: {', '.join(surfactants)})",
            fontsize=11, fontweight="bold", y=1.01
        )

        cs_last = None
        for idx, (sx, sy) in enumerate(pairs):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            try:
                Xi, Yi, Zi = _pairwise_grid(df, sx, sy, metric, grid_n)
                cs_last = _plot_one_panel(ax, Xi, Yi, Zi, sx, sy, metric,
                                          vmin, vmax, cmaps.get(metric, "viridis"))

                # Scatter data points with small log-space jitter so stacked
                # duplicates (same concentration from repeated grid visits)
                # spread apart visually.  Jitter is ~1.5% of the axis range —
                # invisible at the data scale but separates overlapping dots.
                rng = np.random.default_rng(seed=42)
                lx = np.log10(df[f"{sx}_conc_mm"].values)
                ly = np.log10(df[f"{sy}_conc_mm"].values)
                jitter_scale = 0.015 * max(lx.max() - lx.min(),
                                           ly.max() - ly.min(), 0.1)
                lx_j = lx + rng.uniform(-jitter_scale, jitter_scale, len(lx))
                ly_j = ly + rng.uniform(-jitter_scale, jitter_scale, len(ly))
                ax.scatter(
                    lx_j, ly_j,
                    c="black", s=18, edgecolors="white", linewidths=0.7,
                    zorder=5, alpha=0.85
                )
            except Exception as e:
                ax.set_title(f"{sx} vs {sy}\n(error: {e})", fontsize=8)

        # Hide unused panels
        for idx in range(n_pairs, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)

        # Shared colorbar
        if cs_last is not None:
            fig.colorbar(cs_last, ax=axes.ravel().tolist(),
                         label=metric_labels.get(metric, metric),
                         shrink=0.7, pad=0.02)

        plt.tight_layout()
        fname = os.path.join(output_folder, f"pairwise_{metric}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        saved[metric] = fname
        print(f"  Saved: {fname}")

    # 1-D marginals — one figure, one row per surfactant, one col per metric
    _plot_marginals(df, surfactants, metrics, metric_labels, output_folder, saved)

    return saved


def _plot_marginals(df, surfactants, metrics, metric_labels, output_folder, saved):
    """One row per surfactant, one column per metric. Quick sanity check."""
    n_s = len(surfactants)
    n_m = len([m for m in metrics if m in df.columns])
    if n_m == 0:
        return

    fig, axes = plt.subplots(n_s, n_m,
                             figsize=(4 * n_m, 3 * n_s),
                             squeeze=False)
    fig.suptitle("1-D marginals (mean +/- std across all other axes)",
                 fontsize=11, fontweight="bold")

    valid_metrics = [m for m in metrics if m in df.columns]
    for si, surf in enumerate(surfactants):
        col_c = f"{surf}_conc_mm"
        for mi, metric in enumerate(valid_metrics):
            ax = axes[si][mi]
            concs = np.log10(df[col_c].values.astype(float))
            vals = df[metric].values.astype(float)
            # Bin into ~8 quantile bins
            bins = np.percentile(concs, np.linspace(0, 100, 9))
            bins = np.unique(bins)
            if len(bins) < 2:
                ax.scatter(concs, vals, s=10, alpha=0.5)
            else:
                bin_idx = np.digitize(concs, bins) - 1
                bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
                centers, means, stds = [], [], []
                for b in range(len(bins) - 1):
                    mask = bin_idx == b
                    if mask.sum() > 0:
                        centers.append((bins[b] + bins[b + 1]) / 2)
                        means.append(vals[mask].mean())
                        stds.append(vals[mask].std())
                centers, means, stds = map(np.array, [centers, means, stds])
                ax.plot(centers, means, "o-", ms=5, lw=1.5)
                ax.fill_between(centers, means - stds, means + stds,
                                alpha=0.25)
                xt = np.linspace(concs.min(), concs.max(), 5)
                ax.set_xticks(xt)
                ax.set_xticklabels(_tick_labels(xt), fontsize=7)

            ax.set_xlabel(f"{surf} (mM)", fontsize=8)
            ax.set_ylabel(metric_labels.get(metric, metric), fontsize=8)
            if si == 0:
                ax.set_title(metric_labels.get(metric, metric), fontsize=9)

    plt.tight_layout()
    fname = os.path.join(output_folder, "marginals.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved["marginals"] = fname
    print(f"  Saved: {fname}")


def plot_pairwise_feasible_overlay(results_csv_or_df, surfactants, boundary_points,
                                   output_folder, show_all_experiment=True,
                                   feasible_config=None, achievable_fn=None,
                                   grid_n=140, logger=None, dropped_points=None):
    """Plot pairwise feasible-region overlays with init picks highlighted.

    Args:
        results_csv_or_df: path to CSV or DataFrame with experiment results
        surfactants: list of surfactant names
        boundary_points: list of dicts {surfactant: conc_mm} on geometric boundary
        output_folder: where to save PNG
        show_all_experiment: if True, draw all experiment points as faint context
        feasible_config: dict with keys stock_concs, well_volume_ul,
            surfactant_budget_ul, min_conc_mm. If provided, draw a filled
            budget-feasible region mask from the same equation used by is_feasible().
        achievable_fn: optional callable(point_dict)->bool. If provided, draw an
            additional filled region for points that are operationally achievable
            with current substocks/source-selection constraints.
        grid_n: log-grid resolution per pairwise panel for mask evaluation.
        logger: optional logger for diagnostic output.
    """
    if isinstance(results_csv_or_df, str):
        df = pd.read_csv(results_csv_or_df)
    else:
        df = results_csv_or_df.copy()

    if "well_type" in df.columns:
        df = df[df["well_type"] == "experiment"].copy()

    if len(df) == 0:
        print("multidim_visualizer overlay: no experiment rows — skipping")
        return None

    if "iteration" in df.columns:
        init_df = df[df["iteration"] == 0].copy()
    else:
        init_df = df.copy()

    if len(init_df) == 0:
        print("multidim_visualizer overlay: no iteration==0 rows — skipping")
        return None

    bdf = pd.DataFrame(boundary_points) if boundary_points is not None else pd.DataFrame()

    pairs = list(itertools.combinations(surfactants, 2))
    n_pairs = len(pairs)
    n_cols = min(3, n_pairs)
    n_rows = int(np.ceil(n_pairs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.8 * n_cols, 4.2 * n_rows),
                             squeeze=False)
    ddf = pd.DataFrame(dropped_points) if dropped_points is not None else pd.DataFrame()

    fig.suptitle(
        f"Feasible Region vs Initialization Picks\n"
        f"({len(init_df)} init wells, {len(df)} experiment wells total, {len(ddf)} dropped init candidates)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    def _pairwise_budget_mask(sx, sy):
        if feasible_config is None:
            return None
        stock_concs = feasible_config["stock_concs"]
        well_volume_ul = float(feasible_config["well_volume_ul"])
        budget_ul = float(feasible_config["surfactant_budget_ul"])
        min_conc_mm = float(feasible_config["min_conc_mm"])
        max_conc_multiplier = float(feasible_config.get("max_conc_multiplier", 1.0))

        ax_coef = well_volume_ul / float(stock_concs[sx])
        ay_coef = well_volume_ul / float(stock_concs[sy])
        bg_vol = sum(
            min_conc_mm * (well_volume_ul / float(stock_concs[s]))
            for s in surfactants if s not in (sx, sy)
        )
        budget_rem = budget_ul - bg_vol
        if budget_rem <= 0:
            return None

        x_max = max((budget_rem - ay_coef * min_conc_mm) / ax_coef, min_conc_mm)
        y_max = max((budget_rem - ax_coef * min_conc_mm) / ay_coef, min_conc_mm)
        
        # Apply multiplier to smooth high-end boundaries
        x_max = x_max * max_conc_multiplier
        y_max = y_max * max_conc_multiplier
        
        if x_max <= min_conc_mm or y_max <= min_conc_mm:
            return None

        lx = np.linspace(np.log10(min_conc_mm), np.log10(x_max), grid_n)
        ly = np.linspace(np.log10(min_conc_mm), np.log10(y_max), grid_n)
        LX, LY = np.meshgrid(lx, ly, indexing="ij")
        X = 10 ** LX
        Y = 10 ** LY
        feasible = (ax_coef * X + ay_coef * Y) <= budget_rem

        achievable = None
        if callable(achievable_fn):
            achievable = np.zeros_like(feasible, dtype=bool)
            base_point = {s: min_conc_mm for s in surfactants}
            candidate_idx = np.argwhere(feasible)
            for ii, jj in candidate_idx:
                pt = dict(base_point)
                pt[sx] = float(X[ii, jj])
                pt[sy] = float(Y[ii, jj])
                achievable[ii, jj] = bool(achievable_fn(pt))
            
            # Diagnostic logging: report white dot density
            n_feasible = np.sum(feasible)
            n_achievable = np.sum(achievable)
            n_white_dots = n_feasible - n_achievable
            pct_white = 100.0 * n_white_dots / n_feasible if n_feasible > 0 else 0.0
            msg = f"[{sx} vs {sy}] grid={grid_n}: {n_feasible} feasible, {n_achievable} achievable, {n_white_dots} white dots ({pct_white:.1f}%)"
            if logger:
                logger.info(f"  Feasibility diagnostic: {msg}")
            else:
                print(msg)
            
            # Spatial analysis of white dots: which edges do they cluster on?
            if logger and n_white_dots > 0:
                white_dot_indices = np.argwhere((feasible) & (~achievable))
                # Find min/max indices (spatial edges)
                min_ii = np.min(white_dot_indices[:, 0])
                max_ii = np.max(white_dot_indices[:, 0])
                min_jj = np.min(white_dot_indices[:, 1])
                max_jj = np.max(white_dot_indices[:, 1])
                
                # Count dots on edges vs interior
                on_low_i_edge = np.sum(white_dot_indices[:, 0] == min_ii)
                on_high_i_edge = np.sum(white_dot_indices[:, 0] == max_ii)
                on_low_j_edge = np.sum(white_dot_indices[:, 1] == min_jj)
                on_high_j_edge = np.sum(white_dot_indices[:, 1] == max_jj)
                on_any_edge = len(np.unique(np.concatenate([
                    white_dot_indices[white_dot_indices[:, 0] == min_ii],
                    white_dot_indices[white_dot_indices[:, 0] == max_ii],
                    white_dot_indices[white_dot_indices[:, 1] == min_jj],
                    white_dot_indices[white_dot_indices[:, 1] == max_jj],
                ], axis=0), axis=0))
                interior = n_white_dots - on_any_edge
                
                logger.info(f"  White dot spatial distribution:")
                logger.info(f"    On edges: {on_any_edge}/{n_white_dots} ({100*on_any_edge/n_white_dots:.0f}%)")
                logger.info(f"    Interior: {interior}/{n_white_dots} ({100*interior/n_white_dots:.0f}%)")
                logger.info(f"    Edge breakdown: low-{sx}={on_low_i_edge}, high-{sx}={on_high_i_edge}, low-{sy}={on_low_j_edge}, high-{sy}={on_high_j_edge}")
            
            # Sample white dots and log details about why they fail
            if logger and n_white_dots > 0:
                white_dot_indices = np.argwhere((feasible) & (~achievable))
                n_sample = min(3, len(white_dot_indices))  # Sample up to 3 white dots
                if n_sample > 0:
                    logger.info(f"  Sampling {n_sample} white dots to diagnose why they're infeasible:")
                    for sample_idx in np.random.choice(len(white_dot_indices), n_sample, replace=False):
                        ii, jj = white_dot_indices[sample_idx]
                        pt = dict(base_point)
                        pt[sx] = float(X[ii, jj])
                        pt[sy] = float(Y[ii, jj])
                        logger.info(f"    White dot: {sx}={pt[sx]:.4f} mM, {sy}={pt[sy]:.4f} mM")
                        # Log the attempt to see what fails
                        logger.info(f"      achievable_fn returned: False (source selection failed)")

        return {
            "LX": LX,
            "LY": LY,
            "feasible": feasible,
            "achievable": achievable,
        }

    for idx, (sx, sy) in enumerate(pairs):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        col_x = f"{sx}_conc_mm"
        col_y = f"{sy}_conc_mm"

        if col_x not in init_df.columns or col_y not in init_df.columns:
            ax.set_title(f"{sx} vs {sy}\n(missing concentration columns)", fontsize=8)
            continue

        mask_pack = _pairwise_budget_mask(sx, sy)
        bx = np.array([])
        by = np.array([])
        if len(bdf) > 0:
            bx = np.log10(bdf[col_x if col_x in bdf.columns else sx].values.astype(float))
            by = np.log10(bdf[col_y if col_y in bdf.columns else sy].values.astype(float))

        dx = np.array([])
        dy = np.array([])
        if len(ddf) > 0:
            dx = np.log10(ddf[col_x if col_x in ddf.columns else sx].values.astype(float))
            dy = np.log10(ddf[col_y if col_y in ddf.columns else sy].values.astype(float))

        if show_all_experiment:
            ex = np.log10(df[col_x].values.astype(float))
            ey = np.log10(df[col_y].values.astype(float))
            ax.scatter(ex, ey, c="#9aa0a6", s=14, alpha=0.35, linewidths=0,
                       zorder=2, label="All experiment picks")

        if mask_pack is not None:
            LX = mask_pack["LX"]
            LY = mask_pack["LY"]
            feasible = mask_pack["feasible"]
            achievable = mask_pack["achievable"]

            # Fill ONLY the achievable region (what can actually be pipetted).
            # White background shows infeasible regions naturally.
            # If achievable_fn was provided, use it. Otherwise fall back to budget-feasible.
            if achievable is not None:
                ax.contourf(
                    LX, LY, achievable.astype(float),
                    levels=[0.5, 1.5], colors=["#1f77b4"], alpha=0.70, zorder=1,
                )
            else:
                ax.contourf(
                    LX, LY, feasible.astype(float),
                    levels=[0.5, 1.5], colors=["#1f77b4"], alpha=0.70, zorder=1,
                )

        if len(bx) > 0:
            ax.scatter(bx, by, c="#4e79a7", s=16, alpha=0.55, linewidths=0,
                       zorder=3, label="Boundary lattice samples")

        # Plot init picks, separating by source type if available
        ix = np.log10(init_df[col_x].values.astype(float))
        iy = np.log10(init_df[col_y].values.astype(float))
        
        # Check if source type metadata is present
        if "_init_source_type" in init_df.columns:
            # Separate boundary and sobol points
            boundary_mask = init_df["_init_source_type"] == "boundary"
            sobol_mask = init_df["_init_source_type"] == "sobol"
            
            # Plot boundary points as hollow diamonds (blue)
            if boundary_mask.any():
                ax.scatter(ix[boundary_mask], iy[boundary_mask], c="#2c5aa0", s=48, marker="D",
                           edgecolors="white", linewidths=1.0, alpha=0.85, facecolors="none",
                           zorder=5, label="Init picks: boundary")
            
            # Plot sobol points as filled triangles (red)
            if sobol_mask.any():
                ax.scatter(ix[sobol_mask], iy[sobol_mask], c="#d62728", s=38, marker="^",
                           edgecolors="white", linewidths=0.8, alpha=0.95,
                           zorder=5, label="Init picks: sobol")
        else:
            # Fallback: all init picks as red triangles
            ax.scatter(ix, iy, c="#d62728", s=38, marker="^",
                       edgecolors="white", linewidths=0.8, alpha=0.95,
                       zorder=5, label="Init picks (iteration 0)")

        # Overlay dropped init candidates for diagnostics.
        if len(dx) > 0:
            ax.scatter(dx, dy, c="#111111", s=24, marker="x",
                       linewidths=0.8, alpha=0.75, zorder=6,
                       label="Dropped init candidates")

        x_min = min(ix.min(), np.log10(float(feasible_config["min_conc_mm"])) if feasible_config else ix.min())
        y_min = min(iy.min(), np.log10(float(feasible_config["min_conc_mm"])) if feasible_config else iy.min())
        x_max = ix.max()
        y_max = iy.max()
        if len(bx) > 0:
            x_min = min(x_min, bx.min())
            y_min = min(y_min, by.min())
            x_max = max(x_max, bx.max())
            y_max = max(y_max, by.max())
        if mask_pack is not None:
            x_min = min(x_min, mask_pack["LX"].min())
            y_min = min(y_min, mask_pack["LY"].min())
            x_max = max(x_max, mask_pack["LX"].max())
            y_max = max(y_max, mask_pack["LY"].max())

        xt = np.linspace(x_min, x_max, 5)
        yt = np.linspace(y_min, y_max, 5)
        ax.set_xticks(xt)
        ax.set_xticklabels(_tick_labels(xt), fontsize=7)
        ax.set_yticks(yt)
        ax.set_yticklabels(_tick_labels(yt), fontsize=7)
        ax.set_xlabel(f"{sx} (mM)", fontsize=8)
        ax.set_ylabel(f"{sy} (mM)", fontsize=8)
        ax.set_title(f"{sx} vs {sy}", fontsize=9)

    for idx in range(n_pairs, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    legend_handles = []
    if feasible_config is not None:
        legend_handles.append(Patch(facecolor="#f8a29f", edgecolor="#c0392b", alpha=0.32,
                                    label="Budget-feasible region"))
    if callable(achievable_fn):
        legend_handles.append(Patch(facecolor="#d33f32", edgecolor="#7f1d1d", alpha=0.22,
                                    label="Substock-achievable region"))
    if show_all_experiment:
        legend_handles.append(Line2D([0], [0], marker="o", linestyle="None",
                                     markerfacecolor="#9aa0a6", markeredgecolor="#9aa0a6",
                                     markersize=6, alpha=0.7,
                                     label="All experiment picks"))
    if len(bdf) > 0:
        legend_handles.append(Line2D([0], [0], marker="o", linestyle="None",
                                     markerfacecolor="#4e79a7", markeredgecolor="#4e79a7",
                                     markersize=6, alpha=0.7,
                                     label="Boundary lattice samples"))
    if len(ddf) > 0:
        legend_handles.append(Line2D([0], [0], marker="x", linestyle="None",
                                     markerfacecolor="#111111", markeredgecolor="#111111",
                                     markersize=6, alpha=0.8,
                                     label="Dropped init candidates"))
    legend_handles.append(Line2D([0], [0], marker="^", linestyle="None",
                                 markerfacecolor="#d62728", markeredgecolor="white",
                                 markersize=7, label="Init picks (iteration 0)"))
    fig.legend(legend_handles, [h.get_label() for h in legend_handles],
               loc="upper right", fontsize=8)

    plt.tight_layout()
    fname = os.path.join(output_folder, "pairwise_feasible_init_overlay.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return fname


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python multidim_visualizer.py results_final.csv [SDS TTAB CTAB DTAB]")
        sys.exit(1)

    csv_path = sys.argv[1]
    df_check = pd.read_csv(csv_path, nrows=1)

    # Auto-detect surfactants from column names if not supplied
    if len(sys.argv) > 2:
        surfs = sys.argv[2:]
    else:
        surfs = [c.replace("_conc_mm", "") for c in df_check.columns if c.endswith("_conc_mm")]
        print(f"Auto-detected surfactants: {surfs}")

    out = os.path.dirname(csv_path)
    plot_pairwise_maps(csv_path, surfs, out)

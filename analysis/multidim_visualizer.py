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
    """Convert log10 tick positions to readable mM strings."""
    vals = 10 ** log_vals
    labels = []
    for v in vals:
        if v >= 1:
            labels.append(f"{v:.1f}")
        elif v >= 0.1:
            labels.append(f"{v:.2f}")
        else:
            labels.append(f"{v:.3f}")
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

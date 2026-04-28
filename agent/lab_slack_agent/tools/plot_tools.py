"""
plot_tools.py — Tools for generating and retrieving experiment figures.

All plots are saved as PNG files under:
    outputs/plots/{run_id}/

Placeholder implementations use matplotlib with synthetic data so the
full upload workflow can be exercised without real experiment files.

TODO: Replace placeholder generators with real analysis visualisations.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required for server-side use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIR

logger = logging.getLogger(__name__)

# Canonical plot type names used as file stems
STANDARD_PLOT_TYPES = ["calibration", "residual", "pareto", "anomaly"]


# ─────────────────────────────────────────────────────────────────────────────
# Standard plot generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_standard_plots(
    df: Optional[pd.DataFrame],
    output_dir: str,
    run_id: str = "unknown",
) -> List[str]:
    """
    Generate the standard set of diagnostic plots for a run.

    Default figures produced:
      - calibration_curve.png
      - residual_plot.png
      - pareto_plot.png
      - anomaly_plot.png

    Args:
        df:         Experiment DataFrame (may be None — placeholders used).
        output_dir: Directory to write PNG files into.
        run_id:     Used for plot titles.

    Returns:
        List of absolute file path strings for each PNG created.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_paths: List[str] = []

    generators = {
        "calibration_curve": _plot_calibration,
        "residual_plot":     _plot_residuals,
        "pareto_plot":       _plot_pareto,
        "anomaly_plot":      _plot_anomaly,
    }

    for name, fn in generators.items():
        path = out / f"{name}.png"
        try:
            fn(df=df, save_path=str(path), run_id=run_id)
            plot_paths.append(str(path))
            logger.info(f"Saved plot: {path}")
        except Exception as exc:
            logger.warning(f"Failed to generate {name}: {exc}")

    return plot_paths


def select_best_figures(
    plot_paths: List[str],
    max_figures: int = 4,
) -> List[str]:
    """
    Select the most important figures to upload to Slack.

    Priority order: calibration → residual → pareto → anomaly → others.
    Returns up to max_figures paths.
    """
    priority = ["calibration", "residual", "pareto", "anomaly"]

    # Sort existing paths by priority, then alphabetically
    def _sort_key(p: str) -> int:
        name = Path(p).stem.lower()
        for i, keyword in enumerate(priority):
            if keyword in name:
                return i
        return len(priority)

    sorted_paths = sorted(plot_paths, key=_sort_key)
    return sorted_paths[:max_figures]


# ─────────────────────────────────────────────────────────────────────────────
# On-demand plot retrieval / generation
# ─────────────────────────────────────────────────────────────────────────────

def get_existing_plot(run_id: str, plot_type: str) -> Optional[str]:
    """
    Return the path to an existing plot file if it exists on disk.

    Looks in outputs/plots/{run_id}/ for a file whose stem contains plot_type.

    Returns the path string, or None if not found.
    """
    run_dir = PLOTS_DIR / run_id
    if not run_dir.exists():
        return None

    for png in run_dir.glob("*.png"):
        if plot_type.lower() in png.stem.lower():
            logger.info(f"Found existing plot: {png}")
            return str(png)

    return None


def generate_requested_plot(run_id: str, plot_type: str) -> str:
    """
    Generate a single on-demand plot and return its file path.

    Raises FileNotFoundError if the plot type is not supported.
    """
    out_dir = PLOTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    generators = {
        "calibration":  _plot_calibration,
        "residual":     _plot_residuals,
        "pareto":       _plot_pareto,
        "anomaly":      _plot_anomaly,
        "failed_wells": _plot_anomaly,  # alias
        "scatter":      _plot_pareto,   # alias
    }

    fn = generators.get(plot_type)
    if fn is None:
        raise FileNotFoundError(
            f"No generator for plot type '{plot_type}'. "
            f"Supported types: {list(generators)}"
        )

    save_path = str(out_dir / f"{plot_type}.png")
    fn(df=None, save_path=save_path, run_id=run_id)
    logger.info(f"Generated on-demand plot: {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder plot generators
# ─────────────────────────────────────────────────────────────────────────────

def _plot_calibration(
    df: Optional[pd.DataFrame],
    save_path: str,
    run_id: str = "run",
) -> None:
    """Placeholder calibration curve: signal vs known concentration."""
    rng = np.random.default_rng(42)
    x = np.linspace(0, 1, 10)
    y = 1.8 * x + 0.05 + rng.normal(0, 0.03, len(x))
    fit = np.polyfit(x, y, 1)
    y_fit = np.polyval(fit, x)
    r2 = 1 - np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, color="steelblue", zorder=3, label="Standards")
    ax.plot(x, y_fit, color="tomato", label=f"Fit (R²={r2:.4f})")
    ax.set_xlabel("Concentration (mM)")
    ax.set_ylabel("Absorbance (450 nm)")
    ax.set_title(f"Calibration Curve — {run_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_residuals(
    df: Optional[pd.DataFrame],
    save_path: str,
    run_id: str = "run",
) -> None:
    """Placeholder residual plot: predicted vs residual."""
    rng = np.random.default_rng(7)
    predicted = np.linspace(0.1, 1.5, 50)
    residuals = rng.normal(0, 0.05, 50)
    # Inject one outlier for visual interest
    residuals[20] = 0.25

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(predicted, residuals, color="steelblue", alpha=0.7)
    ax.axhline(0, color="tomato", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual (observed - predicted)")
    ax.set_title(f"Residual Plot — {run_id}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_pareto(
    df: Optional[pd.DataFrame],
    save_path: str,
    run_id: str = "run",
) -> None:
    """Placeholder Pareto front plot for a two-objective problem."""
    rng = np.random.default_rng(13)
    n = 80
    obj1 = rng.uniform(0, 1, n)
    obj2 = 1 - obj1 + rng.normal(0, 0.1, n)
    obj2 = np.clip(obj2, 0, None)

    # Identify a rough Pareto front
    pareto_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        dominated = any(
            (obj1[j] <= obj1[i] and obj2[j] <= obj2[i] and (obj1[j] < obj1[i] or obj2[j] < obj2[i]))
            for j in range(n)
        )
        pareto_mask[i] = not dominated

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(obj1[~pareto_mask], obj2[~pareto_mask], color="lightsteelblue",
               alpha=0.5, label="Dominated")
    ax.scatter(obj1[pareto_mask], obj2[pareto_mask], color="tomato",
               zorder=3, label="Pareto front")
    ax.set_xlabel("Objective 1 (minimise)")
    ax.set_ylabel("Objective 2 (minimise)")
    ax.set_title(f"Pareto Front — {run_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_anomaly(
    df: Optional[pd.DataFrame],
    save_path: str,
    run_id: str = "run",
) -> None:
    """Placeholder anomaly / failed-well plate map."""
    rng = np.random.default_rng(99)
    plate = rng.uniform(0.5, 1.5, (8, 12))
    # Inject anomalous wells
    plate[2, 4] = 3.2
    plate[5, 9] = 0.02
    plate[1, 11] = 2.9

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(plate, cmap="RdYlGn_r", vmin=0, vmax=3, aspect="auto")
    plt.colorbar(im, ax=ax, label="Absorbance")
    ax.set_xticks(range(12))
    ax.set_xticklabels([str(i + 1) for i in range(12)])
    ax.set_yticks(range(8))
    ax.set_yticklabels(list("ABCDEFGH"))
    ax.set_title(f"Anomaly / Failed Wells — {run_id}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

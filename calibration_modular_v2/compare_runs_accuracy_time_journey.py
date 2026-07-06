"""
Accuracy vs Time journey plot for two calibration runs.

Each trial is shown as a dot in (mean dispense time, |accuracy deviation %|) space.
Dots are colored by trial index (light = early, dark = late) to show progression.
A line connects the "best composite so far" points in that space, showing the
trajectory (journey) of improvement for each run.

Usage: edit RUN_DIRS, RUN_LABELS, and OUTPUT_NAME, then run.
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Configuration ─────────────────────────────────────────────────────────────
RUNS_BY_LIQUID = [
    {
        "liquid": "Glycerol",
        "output": "glycerol_accuracy_time_journey.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1780513054_glycerol",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1780521190_glycerol",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1780592810_glycerol",
        ],
        "labels": ["SOBOL+BAYESIAN+TOOLS", "SOBOL", "SOBOL+BAYESIAN"],
    },
    {
        "liquid": "Water",
        "output": "water_accuracy_time_journey.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779220789_water",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779224764_water",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779739005_water",
        ],
        "labels": ["SOBOL+BAYESIAN", "SOBOL", "SOBOL+BAYESIAN+TOOLS"],
    },
    {
        "liquid": "Alginate",
        "output": "alginate_accuracy_time_journey.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779813169_agar_water_4%",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779820046_agar_water_4%",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779897239_agar_water_4%",
        ],
        "labels": ["SOBOL+BAYESIAN+TOOLS", "SOBOL+BAYESIAN", "SOBOL"],
    },
    {
        "liquid": "PVA+DMSO",
        "output": "pva_dmso_accuracy_time_journey.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779290791_PVA_DMSO",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779471577_PVA_DMSO",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779906029_PVA_DMSO",
        ],
        "labels": ["SOBOL+BAYESIAN", "SOBOL", "SOBOL+BAYESIAN+TOOLS"],
    },
    {
        "liquid": "DMSO",
        "output": "dmso_accuracy_time_journey.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779208775_DMSO",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779212375_DMSO",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779912579_DMSO",
        ],
        "labels": ["SOBOL+BAYESIAN", "SOBOL", "SOBOL+BAYESIAN+TOOLS"],
    },
    {
        "liquid": "Ethanol",
        "output": "ethanol_accuracy_time_journey.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1780412080_ethanol",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1780417119_ethanol",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1780420806_ethanol",
        ],
        "labels": ["SOBOL+BAYESIAN+TOOLS", "SOBOL+BAYESIAN", "SOBOL"],
    },
]

PALETTE = [
    ("#FF5722", "#FFCCBC"),   # (dark, light) for SOBOL+BAYESIAN
    ("#2196F3", "#BBDEFB"),   # (dark, light) for SOBOL
    ("#4CAF50", "#C8E6C9"),   # (dark, light) for SOBOL+BAYESIAN+TOOLS
]
COMPARISONS_DIR = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\comparisons"
# ──────────────────────────────────────────────────────────────────────────────


def load_df(run_dir: str) -> pd.DataFrame:
    df = pd.read_csv(f"{run_dir}/trial_results.csv")
    for col in ["deviation_pct", "precision_cv_pct", "duration_mean_s", "measurement_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trial_num"] = df["trial_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("trial_num").reset_index(drop=True)
    # Filter out single-measurement trials: CV is undefined with n=1, distorts precision scoring
    df = df[df["measurement_count"] >= 2].reset_index(drop=True)
    # Recompute cumulative measurements after filtering
    df["cumulative_measurements"] = df["measurement_count"].cumsum()
    return df


def _desirability(metric: float, tolerance: float, s: float = 2.0) -> float:
    """Soft desirability: 1.0=perfect, 0.5=at tolerance, >0 beyond tolerance."""
    return 1.0 / (1.0 + (metric / tolerance) ** s)


def _get_tolerance_pct(run_dir: str) -> float:
    """Read volume target from experiment_config_used.yaml and return the matching tolerance %."""
    cfg_path = os.path.join(run_dir, "experiment_config_used.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    vol_ml = cfg["experiment"]["volume_targets_ml"][0]
    vol_ul = vol_ml * 1000
    for vr in cfg["tolerances"]["volume_ranges"]:
        if vr["volume_min_ul"] <= vol_ul <= vr["volume_max_ul"]:
            return float(vr["tolerance_pct"])
    return 3.0  # fallback


def running_best_composite(df: pd.DataFrame, run_dir: str) -> pd.DataFrame:
    """Return rows that set a new best desirability at the moment they were seen.

    Desirability: d=1/(1+(metric/tolerance)^2), higher=better.
    Time normalized within this run's population.
    Weights: accuracy=0.4, precision=0.5, time=0.1.
    """
    tol = _get_tolerance_pct(run_dir)
    t_min = df["duration_mean_s"].min()
    t_max = df["duration_mean_s"].max()
    t_range = max(t_max - t_min, 1.0)

    df = df.copy()
    df["_score"] = (
        0.4 * df["deviation_pct"].apply(lambda x: _desirability(x, tol)) +
        0.5 * df["precision_cv_pct"].apply(lambda x: _desirability(x, tol)) +
        0.1 * (t_max - df["duration_mean_s"]) / t_range
    )

    best_score = -np.inf
    best_rows = []
    for _, row in df.iterrows():
        if row["_score"] > best_score:
            best_score = row["_score"]
            best_rows.append(row)
    return pd.DataFrame(best_rows)


def plot_journey(runs: list[tuple[str, pd.DataFrame, str]], liquid_name: str, output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(
        f"Accuracy vs Time -- Optimization Journey ({liquid_name})\n"
        "Each dot = one trial  |  Line = best solution found so far",
        fontsize=13, fontweight="bold"
    )

    journey_points = []

    for idx, ((label, df, run_dir), (color_dark, color_light)) in enumerate(zip(runs, PALETTE)):
        # Colour each dot by trial order: light (early) -> dark (late)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"cmap_{label}", [color_light, color_dark]
        )
        norm = plt.Normalize(vmin=1, vmax=max(df["trial_num"]))
        colors = [cmap(norm(t)) for t in df["trial_num"]]

        # All trial dots: x = duration, y = accuracy deviation
        ax.scatter(
            df["duration_mean_s"], df["deviation_pct"],
            c=colors, s=55, zorder=3, alpha=0.85, linewidths=0
        )

        # Journey line through best-composite points
        best = running_best_composite(df, run_dir)
        journey_points.append(best[["duration_mean_s", "deviation_pct"]].copy())
        ax.plot(
            best["duration_mean_s"], best["deviation_pct"],
            color=color_dark, linewidth=2.2, zorder=4,
            marker="D", markersize=7, markerfacecolor="white",
            markeredgecolor=color_dark, markeredgewidth=2,
            label=label
        )

        # Annotate trial numbers on journey waypoints (only on every Nth waypoint to avoid clutter)
        waypoints = best.iloc[::max(1, len(best)//3)]  # Sample ~3 waypoints
        for _, row in waypoints.iterrows():
            ax.annotate(
                f"T{int(row['trial_num'])}",
                xy=(row["duration_mean_s"], row["deviation_pct"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=7, color=color_dark, fontweight="bold"
            )

        # Mark the final best with a star
        final = best.iloc[-1]
        ax.scatter(
            final["duration_mean_s"], final["deviation_pct"],
            marker="*", s=260, color=color_dark, zorder=5,
            edgecolors="white", linewidths=0.8
        )

    # Colourbar to show trial progression (for reference only)
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(vmin=1, vmax=max(df["trial_num"].max() for _, df, _ in runs)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Trial number (dot shade)", fontsize=9)

    ax.set_xlabel("Mean dispense time (s)", fontsize=11)
    ax.set_ylabel("|Accuracy deviation| (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="upper right")

    # Auto-zoom: trim high-end outliers while preserving journey visibility.
    all_times = pd.concat([df["duration_mean_s"] for _, df, _ in runs])
    all_devs = pd.concat([df["deviation_pct"] for _, df, _ in runs])
    journey_df = pd.concat(journey_points, ignore_index=True)

    x_max = max(all_times.quantile(0.95), journey_df["duration_mean_s"].max())

    # Y-limits come from points inside the kept x-range so low-time/high-deviation
    # points remain visible even when trimming very slow outliers.
    in_zoom_x = all_times <= x_max
    y_trimmed = all_devs[in_zoom_x]
    y_min = min(y_trimmed.min(), journey_df["deviation_pct"].min())
    y_max = max(y_trimmed.quantile(0.95), journey_df["deviation_pct"].max())

    # Derive x_min from data that remains visible after y-trimming, while
    # always keeping journey points in view.
    in_zoom_y = all_devs <= y_max
    x_trimmed = all_times[in_zoom_y]
    x_min = min(x_trimmed.min(), journey_df["duration_mean_s"].min())

    ax.set_xlim(left=x_min, right=x_max * 1.05)
    ax.set_ylim(bottom=max(0, y_min * 0.95), top=y_max * 1.10)

    # Dashed reference lines at 5% accuracy and 30s as rough targets
    ax.axhline(5.0,  color="grey", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(30.0, color="grey", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(30.5, ax.get_ylim()[1] * 0.97, "30 s", fontsize=8, color="grey", va="top")
    ax.text(ax.get_xlim()[0] * 1.01, 5.3, "5 %", fontsize=8, color="grey")

    plt.tight_layout()
    os.makedirs(COMPARISONS_DIR, exist_ok=True)
    out_path = os.path.join(COMPARISONS_DIR, output_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")
    plt.close()


def main():
    for cfg in RUNS_BY_LIQUID:
        runs = []
        for run_dir, label in zip(cfg["dirs"], cfg["labels"]):
            df = load_df(run_dir)
            print(f"  {label}: {len(df)} trials")
            runs.append((label, df, run_dir))
        plot_journey(runs, cfg["liquid"], cfg["output"])


main()

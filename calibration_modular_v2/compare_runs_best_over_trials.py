"""
Compare calibration runs: best metric achieved over cumulative number of trials.

Supports 2-way, 3-way, or N-way comparisons.

Plots 4 panels:
  - Best accuracy (|deviation %|) over # trials
  - Best precision (CV %) over # trials
  - Best time (mean duration s) over # trials
  - Best cross-run composite score over # trials
    (SDL-normalized using the combined population of all runs so scores are
     directly comparable — unlike the per-run composite stored in the CSV)

Usage: edit RUNS_BY_LIQUID and COLORS, then run.
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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

# ── Configuration ─────────────────────────────────────────────────────────────
RUNS_BY_LIQUID = [
    {
        "liquid": "Glycerol",
        "output": "glycerol_best_over_trials.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1780513054_glycerol",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1780521190_glycerol",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1780592810_glycerol",
        ],
        "labels": ["SOBOL+BAYESIAN+TOOLS", "SOBOL", "SOBOL+BAYESIAN"],
    },
    {
        "liquid": "Water",
        "output": "water_best_over_trials.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779220789_water",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779224764_water",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779739005_water",
        ],
        "labels": ["SOBOL+BAYESIAN", "SOBOL", "SOBOL+BAYESIAN+TOOLS"],
    },
    {
        "liquid": "Alginate",
        "output": "alginate_best_over_trials.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779813169_agar_water_4%",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779820046_agar_water_4%",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779897239_agar_water_4%",
        ],
        "labels": ["SOBOL+BAYESIAN+TOOLS", "SOBOL+BAYESIAN", "SOBOL"],
    },
    {
        "liquid": "PVA+DMSO",
        "output": "pva_dmso_best_over_trials.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779290791_PVA_DMSO",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779471577_PVA_DMSO",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779906029_PVA_DMSO",
        ],
        "labels": ["SOBOL+BAYESIAN", "SOBOL", "SOBOL+BAYESIAN+TOOLS"],
    },
    {
        "liquid": "DMSO",
        "output": "dmso_best_over_trials.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779208775_DMSO",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779212375_DMSO",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779912579_DMSO",
        ],
        "labels": ["SOBOL+BAYESIAN", "SOBOL", "SOBOL+BAYESIAN+TOOLS"],
    },
    {
        "liquid": "Ethanol",
        "output": "ethanol_best_over_trials.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1780412080_ethanol",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1780417119_ethanol",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1780420806_ethanol",
        ],
        "labels": ["SOBOL+BAYESIAN+TOOLS", "SOBOL+BAYESIAN", "SOBOL"],
    },
]

COLORS = [
    "#FF5722",   # orange-red for SOBOL+BAYESIAN
    "#2196F3",   # blue for SOBOL
    "#4CAF50",   # green for SOBOL+BAYESIAN+TOOLS
]
COMPARISONS_DIR = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\comparisons"


def load_trial_results(run_dir: str) -> pd.DataFrame:
    csv_path = f"{run_dir}/trial_results.csv"
    df = pd.read_csv(csv_path)
    for col in ["deviation_pct", "precision_cv_pct", "duration_mean_s", "measurement_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trial_num"] = df["trial_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("trial_num").reset_index(drop=True)
    # Filter out single-measurement trials: CV is undefined with n=1, distorts precision scoring
    df = df[df["measurement_count"] >= 2].reset_index(drop=True)
    # Recompute cumulative measurements after filtering
    df["cumulative_measurements"] = df["measurement_count"].cumsum()
    return df


def add_cross_run_composite(runs: list[tuple[str, pd.DataFrame, str]]) -> None:
    """Recompute composite desirability score using per-run tolerance from config.

    Desirability: d=1/(1+(metric/tolerance)^2), higher=better.
    Time normalized across the combined population (no fixed reference).
    Weights: accuracy=0.4, precision=0.5, time=0.1.
    """
    all_dfs = [df for _, df, _ in runs]
    combined = pd.concat(all_dfs, ignore_index=True)
    t_min = combined["duration_mean_s"].min()
    t_max = combined["duration_mean_s"].max()
    t_range = max(t_max - t_min, 1.0)

    for _, df, run_dir in runs:
        tol = _get_tolerance_pct(run_dir)
        df["cross_composite"] = (
            0.4 * df["deviation_pct"].apply(lambda x: _desirability(x, tol)) +
            0.5 * df["precision_cv_pct"].apply(lambda x: _desirability(x, tol)) +
            0.1 * (t_max - df["duration_mean_s"]) / t_range
        )


def best_over_trials(series: pd.Series, lower_is_better: bool = True) -> pd.Series:
    """Running best (cumulative min or max) across trials."""
    if lower_is_better:
        return series.cummin()
    return series.cummax()


def plot_comparison(runs: list[tuple[str, pd.DataFrame, str]], liquid_name: str) -> None:
    metrics = [
        ("deviation_pct",    "Best Accuracy\n(|Deviation %|, lower = better)",              True),
        ("precision_cv_pct", "Best Precision\n(CV %, lower = better)",                      True),
        ("duration_mean_s",  "Best Time\n(Mean duration s, lower = better)",                True),
        ("cross_composite",  "Best Composite Score\n(desirability, higher = better)", False),
    ]

    n_runs = len(runs)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    fig.suptitle(f"Best Metric Achieved Over Measurements -- {liquid_name} ({n_runs}-way Comparison)", 
                 fontsize=14, fontweight="bold")

    for ax, (col, title, lower_is_better) in zip(axes, metrics):
        for (label, df, _), color in zip(runs, COLORS[:len(runs)]):
            # Cap at 96 measurements
            df_capped = df[df["cumulative_measurements"] <= 96]
            x = df_capped["cumulative_measurements"]
            y_best = best_over_trials(df_capped[col], lower_is_better)
            ax.plot(x, y_best, color=color, linewidth=2, label=label, marker="o",
                    markersize=3, markevery=4)
            ax.fill_between(x, y_best, alpha=0.10, color=color)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Cumulative Measurements", fontsize=9)
        ax.set_xlim(0, 100)  # Show up to 100 for context
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    # Annotate final best values
    for ax, (col, _, lower_is_better) in zip(axes, metrics):
        for (label, df, _), color in zip(runs, COLORS[:len(runs)]):
            df_capped = df[df["cumulative_measurements"] <= 96]
            if len(df_capped) > 0:
                final_best = best_over_trials(df_capped[col], lower_is_better).iloc[-1]
                ax.annotate(f"{final_best:.2f}",
                            xy=(df_capped["cumulative_measurements"].iloc[-1], final_best),
                            xytext=(4, 0), textcoords="offset points",
                            fontsize=7, color=color, va="center")

    plt.tight_layout()
    os.makedirs(COMPARISONS_DIR, exist_ok=True)
    out_path = os.path.join(COMPARISONS_DIR, liquid_name.lower().replace(" ", "_") + "_best_over_measurements.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")
    plt.close()


def main():
    for cfg in RUNS_BY_LIQUID:
        runs = []
        for run_dir, label in zip(cfg["dirs"], cfg["labels"]):
            try:
                df = load_trial_results(run_dir)
                print(f"  {label}: {len(df)} trials, {df['measurement_count'].sum():.0f} total measurements")
                runs.append((label, df, run_dir))
            except FileNotFoundError as e:
                print(f"  WARNING: Skipping {label} ({run_dir}) - file not found: {e}")
                continue

        if len(runs) > 0:
            add_cross_run_composite(runs)
            plot_comparison(runs, cfg["liquid"])
        else:
            print(f"  ERROR: No valid runs for {cfg['liquid']}")


main()

"""
Compare two calibration runs: best metric achieved over cumulative number of trials.

Plots 4 panels:
  - Best accuracy (|deviation %|) over # trials
  - Best precision (CV %) over # trials
  - Best time (mean duration s) over # trials
  - Best cross-run composite score over # trials
    (SDL-normalized using the combined population of both runs so scores are
     directly comparable — unlike the per-run composite stored in the CSV)

Usage: edit RUN_DIRS and RUN_LABELS, then run.
"""

import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ─────────────────────────────────────────────────────────────
RUN_DIRS = [
    r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779237313_glycerol",
    r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779282234_glycerol",
]
RUN_LABELS = [
    "SOBOL+BAYESIAN",
    "SOBOL",
]
COLORS = ["#FF5722", "#2196F3"]   # orange-red for SOBOL+BAYESIAN, blue for SOBOL
# ──────────────────────────────────────────────────────────────────────────────


def load_trial_results(run_dir: str) -> pd.DataFrame:
    csv_path = f"{run_dir}/trial_results.csv"
    df = pd.read_csv(csv_path)
    for col in ["deviation_pct", "precision_cv_pct", "duration_mean_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trial_num"] = df["trial_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("trial_num").reset_index(drop=True)
    return df


def add_cross_run_composite(runs: list[tuple[str, pd.DataFrame]]) -> None:
    """Recompute composite score using the combined population of all runs.

    The per-run SDL composite stored in the CSV is normalized against each
    run's own population, making cross-run comparison meaningless.  Here we
    pool every trial from every run, compute combined stds, and re-score so
    both runs are on the same scale.
    """
    all_dfs = [df for _, df in runs]
    combined = pd.concat(all_dfs, ignore_index=True)

    acc_std  = max(combined["deviation_pct"].std(),    0.1)
    prec_std = max(combined["precision_cv_pct"].std(), 0.1)
    time_std = max(combined["duration_mean_s"].std(),  1.0)

    # Same weights as the system (equal weighting — adjust if needed)
    acc_w, prec_w, time_w = 1.0, 1.0, 1.0

    for _, df in runs:
        df["cross_composite"] = (
            acc_w  * df["deviation_pct"]    / acc_std  * 100 +
            prec_w * df["precision_cv_pct"] / prec_std * 100 +
            time_w * df["duration_mean_s"]  / time_std * 100
        )


def best_over_trials(series: pd.Series, lower_is_better: bool = True) -> pd.Series:
    """Running best (cumulative min or max) across trials."""
    if lower_is_better:
        return series.cummin()
    return series.cummax()


def plot_comparison(runs: list[tuple[str, pd.DataFrame]]) -> None:
    metrics = [
        ("deviation_pct",    "Best Accuracy\n(|Deviation %|, lower = better)",              True),
        ("precision_cv_pct", "Best Precision\n(CV %, lower = better)",                      True),
        ("duration_mean_s",  "Best Time\n(Mean duration s, lower = better)",                True),
        ("cross_composite",  "Best Composite Score\n(cross-run normalized, lower = better)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    fig.suptitle("Best Metric Achieved Over Trials — Two Run Comparison", fontsize=14, fontweight="bold")

    for ax, (col, title, lower_is_better) in zip(axes, metrics):
        for label, color, (_, df) in zip(RUN_LABELS, COLORS, runs):
            x = df["trial_num"]
            y_best = best_over_trials(df[col], lower_is_better)
            ax.plot(x, y_best, color=color, linewidth=2, label=label, marker="o",
                    markersize=3, markevery=4)
            ax.fill_between(x, y_best, alpha=0.10, color=color)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("# Trials", fontsize=9)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    # Annotate final best values
    for ax, (col, _, lower_is_better) in zip(axes, metrics):
        for label, color, (_, df) in zip(RUN_LABELS, COLORS, runs):
            final_best = best_over_trials(df[col], lower_is_better).iloc[-1]
            ax.annotate(f"{final_best:.2f}",
                        xy=(df["trial_num"].iloc[-1], final_best),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=color, va="center")

    plt.tight_layout()
    import os
    comparisons_dir = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\comparisons"
    os.makedirs(comparisons_dir, exist_ok=True)
    out_path = os.path.join(comparisons_dir, "glycerol_run_comparison_best_over_trials.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")
    plt.show()


def main():
    runs = []
    for run_dir, label in zip(RUN_DIRS, RUN_LABELS):
        df = load_trial_results(run_dir)
        print(f"{label}: {len(df)} trials, {df['measurement_count'].sum():.0f} total measurements")
        runs.append((label, df))

    add_cross_run_composite(runs)
    plot_comparison(runs)


main()

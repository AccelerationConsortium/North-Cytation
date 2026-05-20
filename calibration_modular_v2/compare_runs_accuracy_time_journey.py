"""
Accuracy vs Time journey plot for two calibration runs.

Each trial is shown as a dot in (mean dispense time, |accuracy deviation %|) space.
Dots are colored by trial index (light = early, dark = late) to show progression.
A line connects the "best composite so far" points in that space, showing the
trajectory (journey) of improvement for each run.

Usage: edit RUN_DIRS, RUN_LABELS, and OUTPUT_NAME, then run.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Configuration ─────────────────────────────────────────────────────────────
RUNS_BY_LIQUID = [
    {
        "liquid": "DMSO",
        "output": "dmso_accuracy_time_journey.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779208775_DMSO",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779212375_DMSO",
        ],
        "labels": ["SOBOL+BAYESIAN", "SOBOL"],
    },
    {
        "liquid": "Water",
        "output": "water_accuracy_time_journey.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779220789_water",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779224764_water",
        ],
        "labels": ["SOBOL+BAYESIAN", "SOBOL"],
    },
    {
        "liquid": "Glycerol",
        "output": "glycerol_accuracy_time_journey.png",
        "dirs":   [
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779237313_glycerol",
            r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1779282234_glycerol",
        ],
        "labels": ["SOBOL+BAYESIAN", "SOBOL"],
    },
]

PALETTE = [
    ("#FF5722", "#FFCCBC"),   # (dark, light) for SOBOL+BAYESIAN
    ("#2196F3", "#BBDEFB"),   # (dark, light) for SOBOL
]
COMPARISONS_DIR = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\comparisons"
# ──────────────────────────────────────────────────────────────────────────────


def load_df(run_dir: str) -> pd.DataFrame:
    df = pd.read_csv(f"{run_dir}/trial_results.csv")
    for col in ["deviation_pct", "duration_mean_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trial_num"] = df["trial_id"].str.extract(r"(\d+)").astype(int)
    return df.sort_values("trial_num").reset_index(drop=True)


def running_best_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that were the best composite point (min accuracy+time sum,
    cross-normalized) at the moment they were seen."""
    # Simple equal-weight composite on normalised axes so both runs share the same scale
    acc_std  = max(df["deviation_pct"].std(),   0.1)
    time_std = max(df["duration_mean_s"].std(), 1.0)
    df = df.copy()
    df["_score"] = df["deviation_pct"] / acc_std + df["duration_mean_s"] / time_std

    best_score = np.inf
    best_rows = []
    for _, row in df.iterrows():
        if row["_score"] < best_score:
            best_score = row["_score"]
            best_rows.append(row)
    return pd.DataFrame(best_rows)


def plot_journey(runs: list[tuple[str, pd.DataFrame]], liquid_name: str, output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(
        f"Accuracy vs Time — Optimization Journey ({liquid_name})\n"
        "Each dot = one trial  |  Line = best solution found so far",
        fontsize=13, fontweight="bold"
    )

    n_trials = max(len(df) for _, df in runs)

    for (label, df), (color_dark, color_light) in zip(runs, PALETTE):
        # Colour each dot by trial order: light (early) -> dark (late)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"cmap_{label}", [color_light, color_dark]
        )
        norm = plt.Normalize(vmin=1, vmax=n_trials)
        colors = [cmap(norm(t)) for t in df["trial_num"]]

        # All trial dots
        ax.scatter(
            df["duration_mean_s"], df["deviation_pct"],
            c=colors, s=55, zorder=3, alpha=0.85, linewidths=0
        )

        # Journey line through best-composite points
        best = running_best_composite(df)
        ax.plot(
            best["duration_mean_s"], best["deviation_pct"],
            color=color_dark, linewidth=2.2, zorder=4,
            marker="D", markersize=7, markerfacecolor="white",
            markeredgecolor=color_dark, markeredgewidth=2,
            label=label
        )

        # Annotate trial numbers on journey waypoints
        for _, row in best.iterrows():
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

    # Colourbar to show trial progression
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(vmin=1, vmax=n_trials))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Trial number (dot shade)", fontsize=9)

    ax.set_xlabel("Mean dispense time (s)", fontsize=11)
    ax.set_ylabel("|Accuracy deviation| (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="upper right")

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
            runs.append((label, df))
        plot_journey(runs, cfg["liquid"], cfg["output"])


main()

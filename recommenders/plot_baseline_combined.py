"""
Combined baseline visualization: 4 methods (Bayesian, Simplex, Sobol,
Random) on three landscapes (2D, 3D, 4D). Reads metrics_per_iter.csv
from baseline_2d, baseline_3d, baseline_4d and produces:

  baseline_combined_<output>.png  (rows = metric, cols = dim)

Usage:
  python -m recommenders.plot_baseline_combined
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = os.path.join(os.path.dirname(__file__), "test_outputs")
DIMS = [2, 3, 4]
METRICS = ["hit_rate", "surf_recall", "surf_precision"]
OUTPUTS = ["turbidity_600", "ratio"]
COLORS = {
    "BayesianContrast": "tab:blue",
    "Simplex": "tab:orange",
    "Sobol": "tab:green",
    "Random": "tab:red",
}


def main():
    dfs = {}
    for d in DIMS:
        path = os.path.join(BASE, f"baseline_{d}d", "metrics_per_iter.csv")
        if not os.path.exists(path):
            print(f"missing {path}, skipping d={d}")
            continue
        dfs[d] = pd.read_csv(path)

    for output in OUTPUTS:
        fig, axs = plt.subplots(len(METRICS), len(DIMS),
                                figsize=(5.5 * len(DIMS), 3.6 * len(METRICS)),
                                squeeze=False, sharex="col")
        for j, d in enumerate(DIMS):
            if d not in dfs:
                continue
            df = dfs[d]
            for i, m in enumerate(METRICS):
                ax = axs[i][j]
                for rec, color in COLORS.items():
                    sub = df[(df["recommender"] == rec) &
                             (df["output"] == output)]
                    if sub.empty:
                        continue
                    g = sub.groupby("n_picks")[m].agg(["mean", "std"])
                    ax.plot(g.index, g["mean"], label=rec, color=color, lw=1.6)
                    ax.fill_between(g.index, g["mean"] - g["std"],
                                    g["mean"] + g["std"], color=color,
                                    alpha=0.15)
                ax.set_xlabel("n_picks")
                ax.set_ylabel(m)
                if i == 0:
                    ax.set_title(f"d={d} | {output}")
                ax.grid(alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend(fontsize=9, loc="best")
        fig.tight_layout()
        out_path = os.path.join(BASE, f"baseline_combined_{output}.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

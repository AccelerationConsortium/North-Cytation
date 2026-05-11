"""Rescore systematic_compare CSVs with two boundary definitions side-by-side.

For each output:
  - "grad" = top-TOP_FRAC[output] of |grad F|       (current default)
  - "thresh" = F >= THRESH[output]                  (feature-vs-background)

Metrics per (recommender, init, seed, output, def, iteration):
  hit_rate, surf_recall, surf_precision

Outputs:
  - rescored_per_iter.csv
  - rescored_summary_final.csv
  - trajectories_thresh.png  (only the threshold-based panels)
"""

import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from recommenders.metrics import TOP_FRAC, define_boundary_3d
from recommenders.synthetic_3d_visualize import f1_turbidity, f2_ratio


GRID_N = 50
NEAR_R = 0.04
INPUT_COLS = ["x1", "x2", "x3"]
OUTPUTS = ["turbidity_600", "ratio"]
FOLDER = os.path.join(os.path.dirname(__file__), "test_outputs",
                      "systematic_compare")

# threshold-based definitions
# turbidity: above-background (cone interior), since baseline ~0.04
# ratio: middle of transition band, since baseline shifts in real expts
THRESH = {"turbidity_600": (">=", 0.10),
          "ratio":         ("band", (0.75, 0.85))}


def build_boundaries():
    g = np.linspace(0, 1, GRID_N)
    X1, X2, X3 = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)
    F = {
        "turbidity_600": f1_turbidity(pts).reshape(GRID_N, GRID_N, GRID_N),
        "ratio":         f2_ratio(pts).reshape(GRID_N, GRID_N, GRID_N),
    }
    out = {}
    for name, Ff in F.items():
        # gradient-based
        m_g, _ = define_boundary_3d(Ff, (g, g, g), TOP_FRAC[name])
        bp_g = np.stack([X1[m_g], X2[m_g], X3[m_g]], axis=1)
        # threshold-based
        op, val = THRESH[name]
        if op == ">=":
            m_t = Ff >= val
        elif op == "band":
            lo, hi = val
            m_t = (Ff >= lo) & (Ff <= hi)
        bp_t = np.stack([X1[m_t], X2[m_t], X3[m_t]], axis=1)
        out[name] = {
            "grad":   {"mask": m_g, "pts": bp_g, "tree": cKDTree(bp_g),
                       "n": int(m_g.sum())},
            "thresh": {"mask": m_t, "pts": bp_t, "tree": cKDTree(bp_t),
                       "n": int(m_t.sum())},
        }
    return out


def metrics_for(picks, b):
    ng = b["mask"].shape[0]
    ix = np.clip((picks[:, 0] * (ng - 1)).round().astype(int), 0, ng - 1)
    iy = np.clip((picks[:, 1] * (ng - 1)).round().astype(int), 0, ng - 1)
    iz = np.clip((picks[:, 2] * (ng - 1)).round().astype(int), 0, ng - 1)
    hit = float(b["mask"][ix, iy, iz].mean())
    d_pick, _ = b["tree"].query(picks, k=1)
    prec = float((d_pick <= NEAR_R).mean())
    d_bnd, _ = cKDTree(picks).query(b["pts"], k=1)
    rec = float((d_bnd <= NEAR_R).mean())
    return {"hit_rate": hit, "surf_precision": prec, "surf_recall": rec}


def main():
    bnd = build_boundaries()
    print("Boundary cell counts:")
    for name, defs in bnd.items():
        print(f"  {name}: grad={defs['grad']['n']}, "
              f"thresh={defs['thresh']['n']}")

    files = sorted(glob.glob(os.path.join(FOLDER, "data_*.csv")))
    pat = re.compile(r"data_(?P<rec>[^_]+(?:_[A-Z]+)?)_"
                     r"(?P<init>[a-z0-9]+)_seed(?P<seed>\d+)\.csv")
    rows = []
    for f in files:
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        rec, init, seed = (m.group("rec"), m.group("init"),
                           int(m.group("seed")))
        data = pd.read_csv(f)
        iters = sorted(data["iteration"].unique())
        for t in iters:
            sub = data[data["iteration"] <= t]
            picks = sub[INPUT_COLS].values
            n = len(picks)
            for out_name in OUTPUTS:
                for defn in ["grad", "thresh"]:
                    mv = metrics_for(picks, bnd[out_name][defn])
                    rows.append({
                        "recommender": rec, "init": init, "seed": seed,
                        "iteration": int(t), "n_picks": int(n),
                        "output": out_name, "boundary_def": defn,
                        **mv,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(FOLDER, "rescored_per_iter.csv"), index=False)

    # final-step summary
    final_n = df["n_picks"].max()
    fin = df[df["n_picks"] == final_n]
    summ = (fin.groupby(["recommender", "init", "output", "boundary_def"])
            [["hit_rate", "surf_recall", "surf_precision"]]
            .agg(["mean", "std"]).round(4))
    summ.to_csv(os.path.join(FOLDER, "rescored_summary_final.csv"))
    print("\n=== rescored summary @ final n ===")
    with pd.option_context("display.width", 240,
                           "display.max_columns", None):
        print(summ)

    # plot trajectories: 3 metrics x 2 outputs, only the thresh def
    metrics = ["hit_rate", "surf_recall", "surf_precision"]
    cmap = {"BayesianContrast": "tab:blue", "Simplex": "tab:orange"}
    ls = {"grid3": "-", "sobol27": "--"}
    for defn in ["grad", "thresh"]:
        sub = df[df["boundary_def"] == defn]
        fig, axs = plt.subplots(len(metrics), len(OUTPUTS),
                                figsize=(6 * len(OUTPUTS),
                                         3.6 * len(metrics)),
                                squeeze=False)
        for i, mk in enumerate(metrics):
            for j, on in enumerate(OUTPUTS):
                ax = axs[i][j]
                for r in sorted(sub["recommender"].unique()):
                    for ini in sorted(sub["init"].unique()):
                        s = sub[(sub["recommender"] == r) &
                                (sub["init"] == ini) &
                                (sub["output"] == on)]
                        g = s.groupby("n_picks")[mk].agg(["mean", "std"])
                        ax.plot(g.index, g["mean"],
                                label=f"{r}/{ini}", color=cmap[r],
                                lw=1.6, ls=ls[ini])
                        ax.fill_between(g.index, g["mean"] - g["std"],
                                        g["mean"] + g["std"],
                                        color=cmap[r], alpha=0.12)
                ax.set_xlabel("n_picks"); ax.set_ylabel(mk)
                ax.set_title(f"{on}: {mk} ({defn})")
                ax.grid(alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        out = os.path.join(FOLDER, f"trajectories_{defn}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

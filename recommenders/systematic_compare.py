"""
Systematic head-to-head: BayesianContrast vs Simplex on 3D corner-spike-cube.

Tracks 4 metrics per output, per iteration:
  - hit_rate           : fraction of picks in a boundary cell
  - coverage_auc       : AUC of (boundary cells found) vs n_picks
  - surf_recall        : fraction of boundary cells with a pick within R
  - surf_precision     : fraction of picks within R of any boundary cell

Boundary cell = top TOP_FRAC[output] of |grad F| over the 50^3 grid
  (turbidity 0.30, ratio 0.10 -- unchanged).

Multiple seeds, multiple inits. Trajectory plots + final-table CSV.

Usage:
  python -m recommenders.systematic_compare --n-points 200 --seeds 3
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from recommenders.bayesian_transition_recommender import (
    BayesianTransitionRecommender,
)
from recommenders.delaunay_simplex_recommender import (
    DelaunaySimplexTransitionRecommender,
)
from recommenders.metrics import TOP_FRAC, define_boundary_3d
from recommenders.synthetic_3d_visualize import f1_turbidity, f2_ratio


Q_BATCH = 8
GRID_N = 50
NEAR_R = 0.04
INPUT_COLS = ["x1", "x2", "x3"]
OUTPUT_COLS = ["turbidity_600", "ratio"]
OUT_BASE = os.path.join(os.path.dirname(__file__), "test_outputs")


# --- ground truth + boundary cache --------------------------------

def build_boundary_cache():
    g = np.linspace(0.0, 1.0, GRID_N)
    X1, X2, X3 = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)
    fields = {
        "turbidity_600": f1_turbidity(pts).reshape(GRID_N, GRID_N, GRID_N),
        "ratio":         f2_ratio(pts).reshape(GRID_N, GRID_N, GRID_N),
    }
    cache = {}
    for name, F in fields.items():
        mask, _ = define_boundary_3d(F, (g, g, g), TOP_FRAC[name])
        bnd_pts = np.stack([X1[mask], X2[mask], X3[mask]], axis=1)
        cache[name] = {
            "F": F,
            "mask": mask,
            "bnd_pts": bnd_pts,
            "tree": cKDTree(bnd_pts),
            "n_bnd": int(mask.sum()),
        }
    return cache, (g, g, g)


# --- evaluator + init ---------------------------------------------

def evaluate(X):
    return np.stack([f1_turbidity(X), f2_ratio(X)], axis=1)


def initial_design(init_mode, seed):
    # Grid: gridK = K^3 evenly spaced from 0.05..0.95
    # Sobol: sobolN = N scrambled Sobol points in [0,1]^3 (interior)
    if init_mode.startswith("grid"):
        k = int(init_mode[4:])
        g = np.linspace(0.05, 0.95, k)
        X1, X2, X3 = np.meshgrid(g, g, g, indexing="ij")
        X = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)
    elif init_mode.startswith("sobol"):
        from scipy.stats.qmc import Sobol
        n = int(init_mode[5:])
        X = Sobol(d=3, scramble=True, seed=seed).random(n)
    else:
        raise ValueError(init_mode)
    Y = evaluate(X)
    return pd.DataFrame({
        "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2],
        "turbidity_600": Y[:, 0], "ratio": Y[:, 1],
        "well_type": "experiment", "iteration": 0,
    })


# --- per-iter metrics ---------------------------------------------

def metrics_at_step(picks, cache):
    """Compute the 4 metrics for the cumulative pick set."""
    out = {}
    n = picks.shape[0]
    for name, c in cache.items():
        mask = c["mask"]
        # hit_rate
        ng = mask.shape[0]
        ix = np.clip((picks[:, 0] * (ng - 1)).round().astype(int), 0, ng - 1)
        iy = np.clip((picks[:, 1] * (ng - 1)).round().astype(int), 0, ng - 1)
        iz = np.clip((picks[:, 2] * (ng - 1)).round().astype(int), 0, ng - 1)
        hit = float(mask[ix, iy, iz].mean())
        # surf_precision: picks within R of boundary
        d_pick, _ = c["tree"].query(picks, k=1)
        prec = float((d_pick <= NEAR_R).mean())
        # surf_recall: boundary cells within R of a pick
        d_bnd, _ = cKDTree(picks).query(c["bnd_pts"], k=1)
        recall = float((d_bnd <= NEAR_R).mean())
        out[name] = {"hit_rate": hit, "surf_precision": prec,
                     "surf_recall": recall}
    return out


def coverage_auc_from_history(history, cache, name):
    """history = list of (n_picks, recall) for a given output.
    AUC normalized to [0, 1] x [0, 1] -- area under recall vs n_picks
    curve, normalized by max n_picks.
    """
    arr = np.array([[h["n_picks"], h[name]["surf_recall"]] for h in history])
    if len(arr) < 2:
        return float("nan")
    x = arr[:, 0] / arr[:, 0].max()
    y = arr[:, 1]
    return float(np.trapz(y, x))


# --- runner -------------------------------------------------------

def make_rec(label):
    if label == "BayesianContrast":
        return BayesianTransitionRecommender(
            input_columns=INPUT_COLS, output_columns=OUTPUT_COLS,
            log_transform_inputs=False, explore_beta=0.0,
        )
    if label == "Simplex":
        return DelaunaySimplexTransitionRecommender(
            input_columns=INPUT_COLS, output_columns=OUTPUT_COLS,
            log_transform_inputs=False,
        )
    raise ValueError(label)


def run_one(label, init_mode, seed, n_iters, cache):
    torch.manual_seed(seed); np.random.seed(seed)
    rec = make_rec(label)
    torch.manual_seed(seed); np.random.seed(seed)
    data = initial_design(init_mode, seed)

    history = []
    # snapshot t=0
    picks0 = data[INPUT_COLS].values
    snap = metrics_at_step(picks0, cache)
    history.append({"iteration": 0, "n_picks": len(picks0), **snap})

    for it in range(1, n_iters + 1):
        recs = rec.get_recommendations(data, n_points=Q_BATCH, iteration=it)
        X_new = recs[INPUT_COLS].values
        Y_new = evaluate(X_new)
        new = recs.copy()
        new["turbidity_600"] = Y_new[:, 0]
        new["ratio"] = Y_new[:, 1]
        new["well_type"] = "experiment"
        new["iteration"] = it
        data = pd.concat([data, new], ignore_index=True)

        picks = data[INPUT_COLS].values
        snap = metrics_at_step(picks, cache)
        history.append({"iteration": it, "n_picks": len(picks), **snap})

    # flatten history to long DataFrame
    rows = []
    for h in history:
        for out_name, m in [(k, h[k]) for k in OUTPUT_COLS]:
            rows.append({
                "recommender": label, "init": init_mode, "seed": seed,
                "iteration": h["iteration"], "n_picks": h["n_picks"],
                "output": out_name,
                "hit_rate": m["hit_rate"],
                "surf_precision": m["surf_precision"],
                "surf_recall": m["surf_recall"],
            })
    df_iter = pd.DataFrame(rows)

    # add coverage_auc per output (one value per run, broadcast)
    for out_name in OUTPUT_COLS:
        sub = df_iter[df_iter["output"] == out_name]
        x = sub["n_picks"].values / sub["n_picks"].max()
        y = sub["surf_recall"].values
        auc = float(np.trapz(y, x))
        df_iter.loc[df_iter["output"] == out_name, "coverage_auc"] = auc

    return df_iter, data


# --- plots --------------------------------------------------------

def plot_trajectories(df_all, out_path):
    """Lines vs n_picks. Rows = metric, cols = output. Mean +- std across seeds."""
    metrics = ["hit_rate", "surf_recall", "surf_precision"]
    outputs = OUTPUT_COLS
    inits = sorted(df_all["init"].unique())
    recs = sorted(df_all["recommender"].unique())
    cmap = {"BayesianContrast": "tab:blue", "Simplex": "tab:orange"}
    style_for_family = {"grid": "-", "sobol": "--"}
    # marker per init size so we can tell 27/64/125 apart
    size_marker = {27: "o", 64: "s", 125: "^"}

    def init_style(init):
        fam = "grid" if init.startswith("grid") else "sobol"
        if fam == "grid":
            n = int(init[4:]) ** 3
        else:
            n = int(init[5:])
        return style_for_family[fam], size_marker.get(n, "x")

    fig, axs = plt.subplots(len(metrics), len(outputs),
                            figsize=(6.0 * len(outputs), 3.6 * len(metrics)),
                            squeeze=False)
    for i, mkey in enumerate(metrics):
        for j, out_name in enumerate(outputs):
            ax = axs[i][j]
            for rec in recs:
                for init in inits:
                    ls, mk = init_style(init)
                    sub = df_all[(df_all["recommender"] == rec) &
                                 (df_all["init"] == init) &
                                 (df_all["output"] == out_name)]
                    g = sub.groupby("n_picks")[mkey].agg(["mean", "std"])
                    ax.plot(g.index, g["mean"], label=f"{rec} / {init}",
                            color=cmap[rec], lw=1.4,
                            linestyle=ls, marker=mk, markersize=4,
                            markevery=max(1, len(g.index) // 8))
                    ax.fill_between(g.index, g["mean"] - g["std"],
                                    g["mean"] + g["std"],
                                    color=cmap[rec], alpha=0.08)
            ax.set_xlabel("n_picks")
            ax.set_ylabel(mkey)
            ax.set_title(f"{out_name}: {mkey}")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


# --- main ---------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-points", type=int, default=200)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--inits", nargs="+", default=["grid3", "sobol27"])
    p.add_argument("--recs", nargs="+",
                   default=["BayesianContrast", "Simplex"])
    p.add_argument("--out", default="systematic_compare")
    args = p.parse_args()

    out_root = os.path.join(OUT_BASE, args.out)
    os.makedirs(out_root, exist_ok=True)
    print(f"Output: {out_root}")

    print("Building boundary cache...")
    cache, axes = build_boundary_cache()
    for n, c in cache.items():
        print(f"  {n}: {c['n_bnd']} boundary cells")

    all_iter_dfs = []
    for init in args.inits:
        # Determine n_init from init mode label
        if init.startswith("grid"):
            n_init = int(init[4:]) ** 3
        elif init.startswith("sobol"):
            n_init = int(init[5:])
        else:
            raise ValueError(init)
        # Cap total budget at args.n_points; floor iterations so we don't overshoot.
        n_iters = max(0, (args.n_points - n_init) // Q_BATCH)
        actual = n_init + n_iters * Q_BATCH
        print(f"\n[{init}] budget {args.n_points} -> "
              f"{n_init} + {n_iters}x{Q_BATCH} = {actual} pts")
        for seed in range(args.seeds):
            for rec in args.recs:
                print(f"  rec={rec} seed={seed} ...")
                df_iter, data = run_one(rec, init, seed, n_iters, cache)
                all_iter_dfs.append(df_iter)
                fname = f"data_{rec}_{init}_seed{seed}.csv"
                data.to_csv(os.path.join(out_root, fname), index=False)

    df_all = pd.concat(all_iter_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(out_root, "metrics_per_iter.csv"),
                  index=False)

    # final-step table: take last n_picks PER (recommender, init, seed)
    final = (df_all.sort_values("n_picks")
             .groupby(["recommender", "init", "seed", "output"])
             .tail(1))
    summary = (final.groupby(["recommender", "init", "output"])
               [["hit_rate", "surf_recall", "surf_precision",
                 "coverage_auc"]]
               .agg(["mean", "std"])
               .round(4))
    summary.to_csv(os.path.join(out_root, "summary_final.csv"))
    print("\n=== summary @ final n_picks ===")
    with pd.option_context("display.width", 220,
                           "display.max_columns", None):
        print(summary)

    plot_trajectories(df_all,
                      os.path.join(out_root, "trajectories.png"))


if __name__ == "__main__":
    main()

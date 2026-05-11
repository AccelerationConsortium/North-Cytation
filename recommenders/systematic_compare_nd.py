"""
D-dimensional version of systematic_compare.py.

Generalizes the BayesianContrast vs Simplex comparison to arbitrary d.
Default d=3 reproduces systematic_compare; d=4 enables the 4D test.

Usage:
  python -m recommenders.systematic_compare_nd --d 4 --n-points 200 \
      --seeds 3 --inits sobol125 --recs BayesianContrast Simplex \
      --out compare_4d
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
from recommenders.metrics import TOP_FRAC, MIN_RANGE
from recommenders.synthetic_nd import build_landscape


DEFAULT_Q_BATCH = 8
DEFAULT_NEAR_R = 0.04
OUTPUT_COLS = ["turbidity_600", "ratio"]
OUT_BASE = os.path.join(os.path.dirname(__file__), "test_outputs")


def grid_for_d(d):
    """Choose ground truth grid resolution per dimension to keep memory
    sane. 50^3 = 125k, 25^4 = 390k, 12^5 = 248k."""
    return {2: 200, 3: 50, 4: 25, 5: 12}.get(d, max(8, int(1e6 ** (1.0 / d))))


def grad_mag_nd(F):
    grads = np.gradient(F)
    return np.sqrt(sum(g ** 2 for g in grads))


def build_boundary_cache(d, landscape):
    n = grid_for_d(d)
    g = np.linspace(0.0, 1.0, n)
    mesh = np.meshgrid(*([g] * d), indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    print(f"  ground truth grid: {n}^{d} = {n ** d:,} cells")
    F1 = landscape.f1(pts).reshape(*([n] * d))
    F2 = landscape.f2(pts).reshape(*([n] * d))
    fields = {"turbidity_600": F1, "ratio": F2}
    cache = {}
    for name, F in fields.items():
        rng = float(F.max() - F.min())
        if rng < MIN_RANGE:
            raise RuntimeError(f"{name}: range {rng} too small")
        G = grad_mag_nd(F)
        thresh = np.percentile(G, 100.0 * (1.0 - TOP_FRAC[name]))
        mask = G >= thresh
        bnd_pts = np.stack([m[mask] for m in mesh], axis=1)
        cache[name] = {
            "F": F, "mask": mask, "bnd_pts": bnd_pts,
            "tree": cKDTree(bnd_pts),
            "n_bnd": int(mask.sum()),
            "ng": n,
        }
        print(f"  {name}: {mask.sum():,} boundary cells ({100*mask.mean():.2f}% of grid)")
    return cache


def initial_design(init_mode, seed, d, landscape):
    if init_mode.startswith("grid"):
        k = int(init_mode[4:])
        g = np.linspace(0.05, 0.95, k)
        mesh = np.meshgrid(*([g] * d), indexing="ij")
        X = np.stack([m.ravel() for m in mesh], axis=1)
    elif init_mode.startswith("sobol"):
        from scipy.stats.qmc import Sobol
        n = int(init_mode[5:])
        X = Sobol(d=d, scramble=True, seed=seed).random(n)
    elif init_mode.startswith("random"):
        n = int(init_mode[6:])
        X = np.random.default_rng(seed).uniform(0, 1, (n, d))
    else:
        raise ValueError(init_mode)
    Y = landscape.evaluate(X)
    cols = {f"x{i+1}": X[:, i] for i in range(d)}
    cols["turbidity_600"] = Y[:, 0]
    cols["ratio"] = Y[:, 1]
    cols["well_type"] = "experiment"
    cols["iteration"] = 0
    return pd.DataFrame(cols)


def metrics_at_step(picks, cache, d, near_r):
    out = {}
    for name, c in cache.items():
        ng = c["ng"]
        idx = tuple(np.clip((picks[:, i] * (ng - 1)).round().astype(int),
                            0, ng - 1) for i in range(d))
        hit = float(c["mask"][idx].mean())
        d_pick, _ = c["tree"].query(picks, k=1)
        prec = float((d_pick <= near_r).mean())
        d_bnd, _ = cKDTree(picks).query(c["bnd_pts"], k=1)
        recall = float((d_bnd <= near_r).mean())
        out[name] = {"hit_rate": hit, "surf_precision": prec,
                     "surf_recall": recall}
    return out


class _NonAdaptiveRec:
    """Baseline that ignores observed data: each call returns fresh
    space-filling/random points. Mimics the recommender API just enough.
    """
    def __init__(self, input_cols, mode, d, seed):
        self.input_cols = input_cols
        self.mode = mode  # 'random' or 'sobol'
        self.d = d
        if mode == "sobol":
            from scipy.stats.qmc import Sobol
            self._sobol = Sobol(d=d, scramble=True, seed=seed)
        else:
            self._rng = np.random.default_rng(seed)

    def get_recommendations(self, data_df, n_points, iteration=None):
        if self.mode == "sobol":
            X = self._sobol.random(n_points)
        else:
            X = self._rng.uniform(0, 1, (n_points, self.d))
        return pd.DataFrame({c: X[:, i] for i, c in enumerate(self.input_cols)})


def make_rec(label, input_cols, explore_beta=0.0, candidate_pool=None,
             d=None, seed=0):
    if label == "BayesianContrast":
        kwargs = {}
        if candidate_pool is not None:
            kwargs["candidate_pool"] = candidate_pool
        return BayesianTransitionRecommender(
            input_columns=input_cols, output_columns=OUTPUT_COLS,
            log_transform_inputs=False, explore_beta=explore_beta,
            **kwargs,
        )
    if label == "Simplex":
        return DelaunaySimplexTransitionRecommender(
            input_columns=input_cols, output_columns=OUTPUT_COLS,
            log_transform_inputs=False,
        )
    if label == "Random":
        return _NonAdaptiveRec(input_cols, "random", d, seed)
    if label == "Sobol":
        return _NonAdaptiveRec(input_cols, "sobol", d, seed)
    raise ValueError(label)


def run_one(label, init_mode, seed, n_iters, cache, d, landscape,
            q_batch=DEFAULT_Q_BATCH, near_r=DEFAULT_NEAR_R,
            candidate_pool=None,
            explore_beta=0.0):
    input_cols = [f"x{i+1}" for i in range(d)]
    torch.manual_seed(seed); np.random.seed(seed)
    rec = make_rec(label, input_cols, explore_beta=explore_beta,
                   candidate_pool=candidate_pool,
                   d=d, seed=seed)
    torch.manual_seed(seed); np.random.seed(seed)
    data = initial_design(init_mode, seed, d, landscape)

    history = []
    picks0 = data[input_cols].values
    snap = metrics_at_step(picks0, cache, d, near_r)
    history.append({"iteration": 0, "n_picks": len(picks0), **snap})

    for it in range(1, n_iters + 1):
        recs = rec.get_recommendations(data, n_points=q_batch, iteration=it)
        X_new = recs[input_cols].values
        Y_new = landscape.evaluate(X_new)
        new = recs.copy()
        new["turbidity_600"] = Y_new[:, 0]
        new["ratio"] = Y_new[:, 1]
        new["well_type"] = "experiment"
        new["iteration"] = it
        data = pd.concat([data, new], ignore_index=True)

        picks = data[input_cols].values
        snap = metrics_at_step(picks, cache, d, near_r)
        history.append({"iteration": it, "n_picks": len(picks), **snap})

    rows = []
    for h in history:
        for out_name in OUTPUT_COLS:
            m = h[out_name]
            rows.append({
                "recommender": label, "init": init_mode, "seed": seed,
                "iteration": h["iteration"], "n_picks": h["n_picks"],
                "output": out_name,
                "hit_rate": m["hit_rate"],
                "surf_precision": m["surf_precision"],
                "surf_recall": m["surf_recall"],
            })
    df_iter = pd.DataFrame(rows)
    for out_name in OUTPUT_COLS:
        sub = df_iter[df_iter["output"] == out_name]
        x = sub["n_picks"].values / sub["n_picks"].max()
        y = sub["surf_recall"].values
        auc = float(np.trapz(y, x))
        df_iter.loc[df_iter["output"] == out_name, "coverage_auc"] = auc
    return df_iter, data


def plot_trajectories(df_all, out_path):
    metrics = ["hit_rate", "surf_recall", "surf_precision"]
    inits = sorted(df_all["init"].unique())
    recs = sorted(df_all["recommender"].unique())
    _default_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                       "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
    cmap = {r: _default_colors[i % len(_default_colors)]
            for i, r in enumerate(sorted(df_all["recommender"].unique()))}
    style_for_family = {"grid": "-", "sobol": "--"}
    size_marker = {27: "o", 64: "s", 125: "^", 81: "D", 16: "v"}

    def init_style(init):
        fam = "grid" if init.startswith("grid") else "sobol"
        if fam == "grid":
            k = int(init[4:])
            n = k ** (df_all.attrs.get("d", 3))
        else:
            n = int(init[5:])
        return style_for_family[fam], size_marker.get(n, "x")

    fig, axs = plt.subplots(len(metrics), len(OUTPUT_COLS),
                            figsize=(6.0 * len(OUTPUT_COLS),
                                     3.6 * len(metrics)),
                            squeeze=False)
    for i, mkey in enumerate(metrics):
        for j, out_name in enumerate(OUTPUT_COLS):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--n-points", type=int, default=200)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--inits", nargs="+",
                   default=["grid3", "sobol27"])
    p.add_argument("--recs", nargs="+",
                   default=["BayesianContrast", "Simplex"])
    p.add_argument("--out", type=str, default="compare_nd")
    p.add_argument("--q-batch", type=int, default=DEFAULT_Q_BATCH,
                   help="Points proposed per recommender iteration.")
    p.add_argument("--near-r", type=float, default=DEFAULT_NEAR_R,
                   help="Distance threshold for surf_precision/surf_recall.")
    p.add_argument("--candidate-pool", type=int, default=50000,
                   help="Bayesian candidate pool size per iteration.")
    p.add_argument("--explore-beta", type=float, default=0.0,
                   help="Bayesian only: UCB exploration weight.")
    args = p.parse_args()

    out_root = os.path.join(OUT_BASE, args.out)
    os.makedirs(out_root, exist_ok=True)

    print(f"Building landscape (d={args.d})...")
    landscape = build_landscape(args.d)
    print("Building boundary cache...")
    cache = build_boundary_cache(args.d, landscape)

    all_iter_dfs = []
    for init in args.inits:
        if init.startswith("grid"):
            n_init = int(init[4:]) ** args.d
        elif init.startswith("sobol"):
            n_init = int(init[5:])
        elif init.startswith("random"):
            n_init = int(init[6:])
        else:
            raise ValueError(init)
        n_iters = max(0, (args.n_points - n_init) // args.q_batch)
        actual = n_init + n_iters * args.q_batch
        print(f"\n[{init}] budget {args.n_points} -> "
              f"{n_init} + {n_iters}x{args.q_batch} = {actual} pts"
              f" (near_r={args.near_r})")
        for seed in range(args.seeds):
            for rec in args.recs:
                print(f"  rec={rec} seed={seed} ...")
                df_iter, data = run_one(
                    rec, init, seed, n_iters, cache, args.d, landscape,
                                        q_batch=args.q_batch, near_r=args.near_r,
                    candidate_pool=args.candidate_pool,
                    explore_beta=args.explore_beta)
                all_iter_dfs.append(df_iter)
                fname = f"data_{rec}_{init}_seed{seed}.csv"
                data.to_csv(os.path.join(out_root, fname), index=False)

    df_all = pd.concat(all_iter_dfs, ignore_index=True)
    df_all.attrs["d"] = args.d
    df_all.to_csv(os.path.join(out_root, "metrics_per_iter.csv"),
                  index=False)

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

    plot_trajectories(df_all, os.path.join(out_root, "trajectories.png"))


if __name__ == "__main__":
    main()

"""Quick comparison: compute consistent metrics for grid3_n155 and grid4."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from recommenders.synthetic_3d_visualize import (
    f1_turbidity, f2_ratio, SPIKE_AXIS, SPIKE_R0, SPIKE_LENGTH,
)
from recommenders.metrics import TOP_FRAC, define_boundary_3d

NEAR_R = 0.04
g = np.linspace(0, 1, 50)
X1, X2, X3 = np.meshgrid(g, g, g, indexing="ij")
pts = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)
F = {
    "turbidity_600": f1_turbidity(pts).reshape(50, 50, 50),
    "ratio": f2_ratio(pts).reshape(50, 50, 50),
}
bnd = {}
for n, Ff in F.items():
    m, _ = define_boundary_3d(Ff, (g, g, g), TOP_FRAC[n])
    bp = np.stack([X1[m], X2[m], X3[m]], axis=1)
    bnd[n] = (bp, m, cKDTree(bp))

RECS = ["BayesianContrast", "BayesianContrast_UCB",
        "GradientUCB", "LevelSet", "Simplex"]

for folder, label in [
    ("3d_corner_spike_grid3_n155", "grid3 (init 27 -> n=155)"),
    ("3d_corner_spike_grid4",      "grid4 (init 64 -> n=192)"),
]:
    print(f"\n=== {label} ===")
    print(f"{'recommender':22s} {'out':13s} {'srf_rec':>7s} "
          f"{'srf_prec':>8s} {'hit':>6s} {'maxF':>6s} "
          f"{'in_cone':>8s} {'tip':>4s} {'max_s':>6s}")
    for rec in RECS:
        path = f"recommenders/test_outputs/{folder}/all_data_{rec}.csv"
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"{rec:22s} (missing)")
            continue
        X = df[["x1", "x2", "x3"]].values
        s = X @ SPIKE_AXIS
        perp = np.linalg.norm(X - np.outer(s, SPIKE_AXIS), axis=1)
        rA = SPIKE_R0 * np.clip(1 - s / SPIKE_LENGTH, 0, 1)
        inside = (s >= 0) & (s <= SPIKE_LENGTH) & (perp <= rA)
        tip = inside & (s > 0.7 * SPIKE_LENGTH)
        ms = s[inside].max() if inside.any() else 0.0
        for n in ["turbidity_600", "ratio"]:
            bp, m, tree = bnd[n]
            d_pick, _ = tree.query(X, k=1)
            prec = float((d_pick <= NEAR_R).mean())
            d_b, _ = cKDTree(X).query(bp, k=1)
            srec = float((d_b <= NEAR_R).mean())
            ix = np.clip((X[:, 0] * 49).round().astype(int), 0, 49)
            iy = np.clip((X[:, 1] * 49).round().astype(int), 0, 49)
            iz = np.clip((X[:, 2] * 49).round().astype(int), 0, 49)
            hit = float(m[ix, iy, iz].mean())
            Fp = (f1_turbidity(X) if n == "turbidity_600"
                  else f2_ratio(X))
            in_str = f"{int(inside.sum())}" if n == "turbidity_600" else ""
            tip_str = f"{int(tip.sum())}" if n == "turbidity_600" else ""
            ms_str = f"{ms:.3f}" if n == "turbidity_600" else ""
            print(f"{rec:22s} {n:13s} {srec:7.3f} {prec:8.3f} "
                  f"{hit:6.3f} {Fp.max():6.3f} {in_str:>8s} "
                  f"{tip_str:>4s} {ms_str:>6s}")

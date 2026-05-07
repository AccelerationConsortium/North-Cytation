"""
Build a Gaussian Process "ground truth" simulator from a real iterative
surfactant experiment CSV, save it to disk, and provide a query function
the test harness can call as a drop-in replacement for the analytical
`simulate_surfactant_measurements` simulator.

Why
---
The synthetic surfactant simulator we used in Phase 4 has a known ratio +
turbidity structure that the contrast recommender happens to match well.
To stress-test the gradient recommender on a more realistic surface, we
fit a 2-output GP to a real experiment dataset and use its posterior mean
as a smooth ground truth. We can then run BOTH recommenders against this
GP-emulated surface and compare boundary coverage.

Defaults
--------
- Source CSV: DSS / CTAB experiment, 169 experimental points.
- Inputs: surf_A_conc_mm, surf_B_conc_mm (log10-transformed for the GP)
- Outputs: ratio, turbidity_600
- GP: BoTorch SingleTaskGP per output, RBF + ARD, MLE fit via fit_gpytorch_mll
- Pickle saved under recommenders/test_outputs/ground_truth_gp/

Usage
-----
  python -m recommenders.build_ground_truth_gp           # fit + save default
  python -m recommenders.build_ground_truth_gp --csv ... # fit + save custom
"""

import argparse
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import torch

import gpytorch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP


DEFAULT_CSV = (
    r"C:\Users\Owen\Documents\GitHub\surfactant-treasure-mapping"
    r"\surfactant-treasure-mapping\2D Experiments"
    r"\surfactant_grid_DSS_CTAB_Mar_25_Experiment_20260325_141400"
    r"\iterative_experiment_results.csv"
)
DEFAULT_NAME = "DSS_CTAB"
OUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs",
                       "ground_truth_gp")


def _load_experiment(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "well_type" not in df.columns:
        raise KeyError(f"{csv_path}: missing 'well_type' column")
    exp = df[df["well_type"] == "experiment"].copy()
    needed = ["surf_A_conc_mm", "surf_B_conc_mm", "ratio", "turbidity_600"]
    missing = [c for c in needed if c not in exp.columns]
    if missing:
        raise KeyError(f"{csv_path}: missing columns {missing}")
    exp = exp.dropna(subset=needed)
    if len(exp) < 20:
        raise ValueError(f"{csv_path}: only {len(exp)} usable rows")
    return exp.reset_index(drop=True)


def fit_ground_truth_gp(csv_path: str = DEFAULT_CSV,
                        name: str = DEFAULT_NAME):
    """Fit per-output SingleTaskGP on log10(conc) -> {ratio, turbidity_600}.
    Returns the saved bundle dict.
    """
    exp = _load_experiment(csv_path)
    print(f"Loaded {len(exp)} experimental rows from {csv_path}")

    X_raw = exp[["surf_A_conc_mm", "surf_B_conc_mm"]].values.astype(np.float64)
    X_log = np.log10(X_raw)                       # GP fit in log space
    Y = exp[["ratio", "turbidity_600"]].values.astype(np.float64)

    bounds = np.stack([X_log.min(axis=0), X_log.max(axis=0)], axis=0)  # (2, 2)
    print(f"  log10 bounds (A, B):\n{bounds}")

    X_t = torch.tensor(X_log, dtype=torch.double)
    Y_t = torch.tensor(Y, dtype=torch.double)

    models = []
    for i, col in enumerate(["ratio", "turbidity_600"]):
        m = SingleTaskGP(X_t, Y_t[:, i:i+1])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(m.likelihood, m)
        fit_gpytorch_mll(mll)
        cov = m.covar_module
        base = cov.base_kernel if hasattr(cov, "base_kernel") else cov
        os_ = float(cov.outputscale.detach()) if hasattr(cov, "outputscale") else 1.0
        ls = base.lengthscale.detach().squeeze().tolist()
        ns = float(m.likelihood.noise.detach().mean())
        print(f"  {col}: lengthscale={ls}, outputscale={os_:.4f}, "
              f"noise={ns:.4e}")
        models.append(m)

    bundle = {
        "name": name,
        "csv_path": csv_path,
        "X_log_train": X_log,
        "Y_train": Y,
        "log_bounds": bounds,
        "model_state_dicts": [m.state_dict() for m in models],
        "n_train": len(exp),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved ground-truth GP bundle -> {out_path}")
    return bundle


def load_ground_truth_gp(name: str = DEFAULT_NAME):
    """Re-instantiate per-output SingleTaskGPs from a saved bundle."""
    path = os.path.join(OUT_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    X_t = torch.tensor(bundle["X_log_train"], dtype=torch.double)
    Y_t = torch.tensor(bundle["Y_train"], dtype=torch.double)
    models = []
    for i, sd in enumerate(bundle["model_state_dicts"]):
        m = SingleTaskGP(X_t, Y_t[:, i:i+1])
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models, bundle


_CACHE = {}


def simulate_from_gp(surf_a_mm: float, surf_b_mm: float,
                     name: str = DEFAULT_NAME) -> Tuple[float, float]:
    """Drop-in replacement for simulate_surfactant_measurements that returns
    posterior-mean (ratio, turbidity_600) from a saved ground-truth GP.

    The GP was fit on log10 concentrations, so this clamps queries inside
    the fitted log10 range to avoid wild extrapolation.
    """
    if name not in _CACHE:
        _CACHE[name] = load_ground_truth_gp(name)
    models, bundle = _CACHE[name]
    log_a = np.log10(max(surf_a_mm, 1e-6))
    log_b = np.log10(max(surf_b_mm, 1e-6))
    lo, hi = bundle["log_bounds"]
    log_a = float(np.clip(log_a, lo[0], hi[0]))
    log_b = float(np.clip(log_b, lo[1], hi[1]))
    x = torch.tensor([[log_a, log_b]], dtype=torch.double)
    with torch.no_grad():
        means = [float(m.posterior(x).mean.squeeze()) for m in models]
    return means[0], means[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--name", default=DEFAULT_NAME)
    args = p.parse_args()
    fit_ground_truth_gp(args.csv, args.name)
    # quick sanity check
    a_mid = 10 ** ((np.log10(0.01) + np.log10(11.25)) / 2)
    b_mid = 10 ** ((np.log10(0.01) + np.log10(2.25)) / 2)
    r, t = simulate_from_gp(a_mid, b_mid, args.name)
    print(f"\nSanity check at ({a_mid:.4f}, {b_mid:.4f}) mM: "
          f"ratio={r:.4f}, turbidity={t:.4f}")


if __name__ == "__main__":
    main()

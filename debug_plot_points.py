"""
Plot raw experiment points from a results CSV, no interpolation.
Usage: edit CSV_PATH and SURFACTANTS below, then run.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else r"output\simulated_surfactant_grid\multidim_2D_SDS_TTAB_multidim_v1_20260519_144159\results_final.csv"
INIT_CSV = sys.argv[2] if len(sys.argv) > 2 else r"output\simulated_surfactant_grid\multidim_2D_SDS_TTAB_multidim_v1_20260519_144159\results_after_initial_grid.csv"
SURFACTANTS = ["SDS", "TTAB"]

df   = pd.read_csv(CSV_PATH)
init = pd.read_csv(INIT_CSV)
n_init = len(init)

x_col = f"{SURFACTANTS[0]}_conc_mm"
y_col = f"{SURFACTANTS[1]}_conc_mm"

init_pts = df.iloc[:n_init]
iter_pts = df.iloc[n_init:]

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(init_pts[x_col], init_pts[y_col], c="orange", s=30, zorder=3,
           label=f"init grid ({n_init} pts)", edgecolors="k", linewidths=0.5)
ax.scatter(iter_pts[x_col], iter_pts[y_col], c="steelblue", s=18, alpha=0.8,
           label=f"sobol iterations ({len(iter_pts)} pts)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(f"{SURFACTANTS[0]} (mM)")
ax.set_ylabel(f"{SURFACTANTS[1]} (mM)")
ax.set_title(f"Raw points — no interpolation\n{len(df)} total wells")
ax.legend()
plt.tight_layout()
plt.savefig("debug_raw_points.png", dpi=150)
plt.show()
print(f"Init: {n_init} pts, Iterations: {len(iter_pts)} pts")
print(f"SDS  range: {df[x_col].min():.3g} - {df[x_col].max():.3g} mM")
print(f"TTAB range: {df[y_col].min():.3g} - {df[y_col].max():.3g} mM")

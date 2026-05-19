"""Test simplex init point counts for 2D, 3D, 4D by patching SURFACTANTS."""
import sys, importlib, types
sys.path.insert(0, '.')

from workflows import surfactant_multidimensional_workflow as wf
from workflows.surfactant_multidimensional_workflow import (
    SURFACTANT_LIBRARY, WELL_VOLUME_UL, SURFACTANT_BUDGET_UL,
)
import numpy as np

def total_vol(pt, surfactants):
    return sum(pt[s] * WELL_VOLUME_UL / SURFACTANT_LIBRARY[s]["stock_conc"]
               for s in surfactants)

for surfs in [["SDS", "DTAB"], ["SDS", "DSS", "DTAB"], ["SDS", "DSS", "TTAB", "DTAB"]]:
    n = len(surfs)
    # Temporarily override module-level variables for this test
    orig_surfs = wf.SURFACTANTS
    wf.SURFACTANTS = surfs
    wf.n_surfactants = n
    wf.simplex_max_conc_mm = {
        s: SURFACTANT_LIBRARY[s]["stock_conc"] * (SURFACTANT_BUDGET_UL / WELL_VOLUME_UL)
        for s in surfs
    }

    pts = wf.generate_simplex_init()
    infeas = [p for p in pts if not wf.is_feasible(p)]
    vols = [total_vol(p, surfs) for p in pts]
    cmcs = {s: SURFACTANT_LIBRARY[s]["cmc_mm"] for s in surfs}
    vmax = {s: wf.simplex_max_conc_mm[s] for s in surfs}

    print(f"\n=== {n}D: {surfs} ===")
    print(f"  Init points: {len(pts)}  (infeasible: {len(infeas)})")
    print(f"  Max total vol: {max(vols):.2f} uL")
    print(f"  Simplex maxima: {vmax}")
    print(f"  CMCs:          {cmcs}")
    # Check each surf's CMC is reachable
    for s in surfs:
        max_c = max(p[s] for p in pts)
        cmc = cmcs[s]
        print(f"  {s}: max_in_init={max_c:.2f} mM, CMC={cmc} mM, reachable={max_c >= cmc}")

    # Restore
    wf.SURFACTANTS = orig_surfs
    wf.n_surfactants = len(orig_surfs)
    wf.simplex_max_conc_mm = {
        s: SURFACTANT_LIBRARY[s]["stock_conc"] * (SURFACTANT_BUDGET_UL / WELL_VOLUME_UL)
        for s in orig_surfs
    }

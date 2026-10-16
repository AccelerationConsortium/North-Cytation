"""run_4d_algorithm_comparison.py

Runs the multidimensional workflow in simulation for all 5 recommender
algorithms on a 4D surfactant system (SDS + NaLS anionic, TTAB + DTAB cationic).

Executes algorithms sequentially. Each run saves results to:
    output/simulated_surfactant_grid/multidim_4D_SDS_NaLS_TTAB_DTAB_{algorithm}_*

Usage (from repo root):
    python workflows/run_4d_algorithm_comparison.py

To run a single algorithm:
    python workflows/run_4d_algorithm_comparison.py bayesian
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALGORITHMS = ['sobol', 'triangle', 'bayesian', 'gradient', 'levelset']
SURFACTANTS_4D = ['SDS', 'NaLS', 'TTAB', 'DTAB']   # 2 anionic + 2 cationic

# If a single algorithm name is passed as argument, run only that one
if len(sys.argv) > 1:
    alg_arg = sys.argv[1].strip().lower()
    if alg_arg not in ALGORITHMS:
        print(f"Unknown algorithm '{alg_arg}'. Choose from: {ALGORITHMS}")
        sys.exit(1)
    ALGORITHMS = [alg_arg]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import workflows.surfactant_multidimensional_workflow as wf
from master_usdl_coordinator import Lash_E

INPUT_VIAL_FILE = "../utoronto_demo/status/surfactant_4d_SDS_NaLS_TTAB_DTAB_vials.csv"

for alg in ALGORITHMS:
    print(f"\n{'='*70}")
    print(f"  Running algorithm: {alg.upper()}  |  Surfactants: {SURFACTANTS_4D}")
    print(f"{'='*70}\n")

    # Patch module-level config before calling run()
    wf.SURFACTANTS      = SURFACTANTS_4D
    wf.RECOMMENDER_TYPE = alg
    wf.SIMULATE         = True
    wf.VALIDATE_LIQUIDS = False

    lash_e = Lash_E(INPUT_VIAL_FILE, simulate=True, show_gui=False)

    try:
        wf.run_multidim_workflow(lash_e)
    except Exception as e:
        print(f"\n  ERROR in {alg}: {e}")
        import traceback
        traceback.print_exc()
        print(f"  Continuing to next algorithm...\n")
        continue

    # Clean up between runs so state doesn't bleed over
    del lash_e

print("\nAll algorithms complete.")
print("Run  python analysis/compare_algorithm_iou.py  with SURFACTANTS=['SDS','NaLS','TTAB','DTAB']")
print("and  surfactant_tag='SDS_NaLS_TTAB_DTAB'  to compare results.")

"""
Standalone debug script that replicates the liquid validation step.
Use this to diagnose scale measurement issues without modifying any existing code.

Mirrors exactly what validate_pipetting_system() does in surfactant_grid_adaptive_concentrations.py
when called from surfactant_multidimensional_workflow.py.

Tunable parameters are at the top - adjust freely.
"""

import sys
import os
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy

# ── Configuration ────────────────────────────────────────────────────────────
VIAL_FILE       = "status/surfactant_multidim_vials.csv"
SOURCE_VIAL     = "water"
DEST_VIAL       = "water"       # same as source, matching the real workflow
LIQUID_TYPE     = "water"
VOLUMES_ML      = [0.01]       # 50 uL - the failing volume  [REAL WORKFLOW: [0.02, 0.05, 0.1, 0.15]]
REPLICATES      = 1             # single rep to keep it fast   [REAL WORKFLOW: 3]
SWITCH_PIPET    = False
CONDITION_TIP   = True
CONDITIONING_UL = 150
ADAPTIVE        = True         # [CHANGED: True in real workflow] disabled to bypass correction logic
OUTPUT_FOLDER   = "output/debug_validation"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

lash_e = Lash_E(VIAL_FILE, simulate=False, initialize_biotek=False)

print(f"\nSource vial location : {lash_e.nr_robot.get_vial_info(SOURCE_VIAL, 'location')}")
print(f"Dest vial location   : {lash_e.nr_robot.get_vial_info(DEST_VIAL, 'location')}")
print(f"Volumes to test      : {[v*1000 for v in VOLUMES_ML]} uL")
print(f"Replicates           : {REPLICATES}")
print()

results = validate_pipetting_accuracy(
    lash_e=lash_e,
    source_vial=SOURCE_VIAL,
    destination_vial=DEST_VIAL,
    liquid_type=LIQUID_TYPE,
    volumes_ml=VOLUMES_ML,
    replicates=REPLICATES,
    output_folder=OUTPUT_FOLDER,
    switch_pipet=SWITCH_PIPET,
    save_raw_data=True,          # always save so we can inspect the CSV
    condition_tip_enabled=CONDITION_TIP,
    conditioning_volume_ul=CONDITIONING_UL,
    adaptive_correction=ADAPTIVE,
    compensate_overvolume=True,  # [CHANGED: True in real workflow] bypass compensation that produces negative overaspirate from bad 60uL CSV row
)

print("\n=== Results ===")
print(f"R^2            : {results.get('r_squared', 'N/A')}")
print(f"Mean accuracy  : {results.get('mean_accuracy_pct', 'N/A')} %")
print(f"\nMass data CSVs saved to: output/mass_measurements/")
print("Check those files to see pre/post baseline readings and steady_status values.")

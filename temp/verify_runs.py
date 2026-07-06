import yaml
from pathlib import Path

base = Path("calibration_modular_v2/output")
runs = [
    ("run_1779237313_glycerol", "glycerol"),
    ("run_1779282234_glycerol", "glycerol"),
    ("run_1779731396_glycerol", "glycerol"),
    ("run_1779220789_water", "water"),
    ("run_1779224764_water", "water"),
    ("run_1779739005_water", "water"),
    ("run_1779813169_agar_water_4%", "alginate"),
    ("run_1779820046_agar_water_4%", "alginate"),
    ("run_1779897239_agar_water_4%", "alginate"),
    ("run_1779290791_PVA_DMSO", "PVA_DMSO"),
    ("run_1779471577_PVA_DMSO", "PVA_DMSO"),
    ("run_1779906029_PVA_DMSO", "PVA_DMSO"),
    ("run_1779208775_DMSO", "DMSO"),
    ("run_1779212375_DMSO", "DMSO"),
    ("run_1779912579_DMSO", "DMSO"),
    ("run_1780412080_ethanol", "ethanol"),
    ("run_1780417119_ethanol", "ethanol"),
    ("run_1780420806_ethanol", "ethanol"),
]

for r, liquid in runs:
    cfg = base / r / "experiment_config_used.yaml"
    if not cfg.exists():
        print(f"MISSING  {r}")
        continue
    with open(cfg) as f:
        c = yaml.safe_load(f)
    scr = c["experiment"]["num_screening_trials"]
    tp = c["experiment"].get("two_point_calibration_replicates", "N/A")
    if scr == 32:
        tag = "SOBOL"
    elif tp == 3:
        tag = "SBT  "
    else:
        tag = "SB   "
    print(f"{tag}  {r}  scr={scr} twopoint={tp}")

"""Measure 6 liquids with HardwareCalibrationProtocol: 50uL x3 replicates using real protocol infrastructure.

Uses calibration_protocol_northrobot.py directly:
- Tip conditioning applied per liquid refill_pipets setting
- Tip swapping for viscous liquids (glycerol, PVA_DMSO, agar_water_4%)
- Mass measurements converted to volume using protocol density values

Usage:
  python workflows/calibration_vials_short_mass_validation.py          # Real hardware
  python workflows/calibration_vials_short_mass_validation.py --simulate   # Simulation mode
"""

import argparse
import csv
import logging
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration_modular_v2.calibration_protocol_northrobot import HardwareCalibrationProtocol, LIQUIDS
from pipetting_data.pipetting_parameters import PipettingParameters

SIMULATE = False
TARGET_VOLUME_ML = 0.05
REPLICATES = 3
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
CONFIG_FILE = Path(__file__).resolve().parent.parent / "calibration_modular_v2" / "experiment_config.yaml"
HARDWARE_CONFIG_FILE = Path(__file__).resolve().parent.parent / "calibration_modular_v2" / "north_robot_hardware.yaml"
VIAL_STATUS_FILE = Path(__file__).resolve().parent.parent / "status" / "calibration_vials_short.csv"

# Liquids to measure - using exact names from protocol's LIQUIDS dict
LIQUID_SERIES: List[Dict[str, str]] = [
    {"label": "glycerol", "liquid_name": "glycerol", "vial_name": "glycerol"},
    {"label": "alginate_4pct", "liquid_name": "agar_water_4%", "vial_name": "agar_water_4%"},
    {"label": "PVA_dmso", "liquid_name": "PVA_DMSO", "vial_name": "PVA_DMSO"},
    {"label": "dmso", "liquid_name": "DMSO", "vial_name": "DMSO"},
    {"label": "water", "liquid_name": "water", "vial_name": "water"},
    {"label": "ethanol", "liquid_name": "ethanol", "vial_name": "ethanol"},
]


def _create_protocol_config(liquid_name: str, simulate: bool, vial_name: str) -> Dict[str, Any]:
    """Create protocol configuration dict for a specific liquid."""
    # Load base config
    with open(CONFIG_FILE, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Update for this liquid
    cfg['experiment']['liquid'] = liquid_name
    cfg['experiment']['volume_targets_ml'] = [TARGET_VOLUME_ML]
    cfg['experiment']['simulate'] = simulate
    
    # Update hardware config with correct vial name
    with open(HARDWARE_CONFIG_FILE, 'r') as f:
        hw_cfg = yaml.safe_load(f)
    hw_cfg['vials']['source_vial'] = vial_name
    hw_cfg['vials']['measurement_vial'] = vial_name
    with open(HARDWARE_CONFIG_FILE, 'w') as f:
        yaml.dump(hw_cfg, f, default_flow_style=False, sort_keys=False)
    
    return cfg


def run_calibration_vials_short_mass_validation(simulate: bool = False) -> None:
    global SIMULATE
    if simulate:
        SIMULATE = simulate
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("calibration_vials_short_mass_validation")

    logger.info("Starting calibration vials measurement using HardwareCalibrationProtocol")
    logger.info("Target volume: %.1fuL, replicates: %d, simulate=%s\n", 
                TARGET_VOLUME_ML * 1000, REPLICATES, SIMULATE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    protocol = HardwareCalibrationProtocol()

    try:
        for entry in LIQUID_SERIES:
            label = entry["label"]
            liquid_name = entry["liquid_name"]
            vial_name = entry["vial_name"]
            
            # Verify liquid exists in protocol's LIQUIDS dict
            if liquid_name not in LIQUIDS:
                raise ValueError(f"Liquid '{liquid_name}' not found in protocol LIQUIDS dict. Available: {list(LIQUIDS.keys())}")
            
            density_g_ml = LIQUIDS[liquid_name]['density']
            logger.info("Measuring %s (density=%.2f g/mL)...", liquid_name, density_g_ml)
            
            # Create config for this liquid and update hardware files
            cfg = _create_protocol_config(liquid_name, SIMULATE, vial_name)
            
            # Initialize protocol for this liquid
            state = protocol.initialize(cfg)
            
            # Measure 3 replicates using protocol.measure()
            params = PipettingParameters(
                aspirate_speed=10,
                dispense_speed=10,
                aspirate_wait_time=0.0,
                dispense_wait_time=1.5,
                overaspirate_vol=0.0,
            )
            
            measurement_results = protocol.measure(state, TARGET_VOLUME_ML, params, replicates=REPLICATES)
            logger.info("  Got %d measurements from protocol", len(measurement_results))
            
            # Convert protocol results to output format
            for meas in measurement_results:
                result = {
                    "label": label,
                    "liquid_name": liquid_name,
                    "vial_name": vial_name,
                    "replicate": meas['replicate'],
                    "target_volume_uL": TARGET_VOLUME_ML * 1000,
                    "measured_volume_uL": meas['volume'] * 1000,
                    "density_g_mL": density_g_ml,
                    "elapsed_s": meas['elapsed_s'],
                    "timestamp": meas.get('start_time', datetime.now().isoformat()),
                }
                all_results.append(result)
                logger.info("    Rep %d: %.2fuL (%.1fs)", meas['replicate'], meas['volume'] * 1000, meas['elapsed_s'])
            
            # Cleanup this measurement
            protocol.wrapup(state)
        
        # Save all results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"calibration_vials_short_mass_validation_{timestamp}.csv"
        
        fieldnames = [
            "label",
            "liquid_name",
            "vial_name",
            "replicate",
            "target_volume_uL",
            "measured_volume_uL",
            "density_g_mL",
            "elapsed_s",
            "timestamp",
        ]
        
        with open(output_path, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
        
        logger.info("Saved %d results to %s\n", len(all_results), output_path)
        logger.info("Workflow complete")

    except Exception as e:
        logger.error("Workflow failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure 6 calibration liquids x3 replicates using real protocol")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
    args = parser.parse_args()
    
    run_calibration_vials_short_mass_validation(simulate=args.simulate)

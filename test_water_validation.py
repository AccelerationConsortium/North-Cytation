#!/usr/bin/env python3
"""
Water Validation Test - 80, 130, 180 μL volumes
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from master_usdl_coordinator import Lash_E
from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy

# Configuration
SIMULATE = False
INPUT_VIAL_STATUS_FILE = "status/calibration_vials_short.csv"

# Separate volumes by tip type
LARGE_TIP_VOLUMES_UL = [200]  # Large tip volumes (>= 200 uL)
SMALL_TIP_VOLUMES_UL = [150, 100, 70]  # Small tip volumes (< 200 uL)

REPLICATES = 5
LIQUID_TYPE = "water"
SOURCE_VIAL = "water"
DESTINATION_VIAL = "water"


def main():
    print(f"Water Validation Test - Large Tip: {LARGE_TIP_VOLUMES_UL} μL, Small Tip: {SMALL_TIP_VOLUMES_UL} μL")
    print(f"Simulation: {SIMULATE}")
    
    lash_e = None
    try:
        # Initialize
        lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
        lash_e.nr_robot.home_robot_components()
        
        # Create base output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_folder = f"output/water_validation_{timestamp}"
        os.makedirs(base_output_folder, exist_ok=True)
        print(f"Base output: {base_output_folder}")
        
        results = {}
        
        # 1. Large Tip Calibration (200+ μL volumes)
        if LARGE_TIP_VOLUMES_UL:
            print(f"\n--- LARGE TIP CALIBRATION ---")
            print(f"Testing volumes: {LARGE_TIP_VOLUMES_UL} μL")
            
            large_tip_output = os.path.join(base_output_folder, "large_tip_calibration")
            os.makedirs(large_tip_output, exist_ok=True)
            
            large_tip_volumes_ml = [vol / 1000.0 for vol in LARGE_TIP_VOLUMES_UL]
            
            results['large_tip'] = validate_pipetting_accuracy(
                lash_e=lash_e,
                source_vial=SOURCE_VIAL,
                destination_vial=DESTINATION_VIAL,
                liquid_type=LIQUID_TYPE,
                volumes_ml=large_tip_volumes_ml,
                replicates=REPLICATES,
                output_folder=large_tip_output,
                plot_title=f"Water Validation - Large Tip ({', '.join(map(str, LARGE_TIP_VOLUMES_UL))} μL)",
                save_raw_data=True,
                compensate_overvolume=True,
                condition_tip_enabled=True,
                conditioning_volume_ul=max(max(LARGE_TIP_VOLUMES_UL), 300) if LARGE_TIP_VOLUMES_UL else 300,
                quality_std_threshold=0.0005,
                adaptive_correction=True
            )
            r_squared = results['large_tip'].get('r_squared', 'N/A')
            print(f"Large tip calibration complete - R²: {r_squared}")

            lash_e.nr_robot.remove_pipet()
        
        # 2. Small Tip Calibration (< 200 μL volumes)  
        if SMALL_TIP_VOLUMES_UL:
            print(f"\n--- SMALL TIP CALIBRATION ---")
            print(f"Testing volumes: {SMALL_TIP_VOLUMES_UL} μL")
            
            small_tip_output = os.path.join(base_output_folder, "small_tip_calibration")
            os.makedirs(small_tip_output, exist_ok=True)
            
            small_tip_volumes_ml = [vol / 1000.0 for vol in SMALL_TIP_VOLUMES_UL]
            
            results['small_tip'] = validate_pipetting_accuracy(
                lash_e=lash_e,
                source_vial=SOURCE_VIAL,
                destination_vial=DESTINATION_VIAL,
                liquid_type=LIQUID_TYPE,
                volumes_ml=small_tip_volumes_ml,
                replicates=REPLICATES,
                output_folder=small_tip_output,
                plot_title=f"Water Validation - Small Tip ({', '.join(map(str, SMALL_TIP_VOLUMES_UL))} μL)",
                save_raw_data=True,
                compensate_overvolume=True,
                condition_tip_enabled=True,
                conditioning_volume_ul=max(SMALL_TIP_VOLUMES_UL) if SMALL_TIP_VOLUMES_UL else 150,
                quality_std_threshold=0.0005,
                adaptive_correction=True
            )
            lash_e.nr_robot.remove_pipet()
            r_squared = results['small_tip'].get('r_squared', 'N/A')
            print(f"Small tip calibration complete - R²: {r_squared}")
        
        lash_e.nr_robot.move_home()
        
        # Summary
        print(f"\n--- VALIDATION SUMMARY ---")
        for tip_type, result in results.items():
            r_squared = result.get('r_squared', 'N/A')
            accuracy = result.get('mean_accuracy_pct', 0)
            print(f"{tip_type.upper()}: R² = {r_squared}, Accuracy = {accuracy:.1f}%")
        
        print("All validations complete")
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        # Return vials home
        if lash_e is not None:
            try:
                lash_e.nr_robot.return_vial_home(SOURCE_VIAL)
                lash_e.nr_robot.return_vial_home(DESTINATION_VIAL)
            except:
                pass


if __name__ == "__main__":
    main()
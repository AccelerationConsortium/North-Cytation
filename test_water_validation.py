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
TEST_VOLUMES_UL = [180,130,80]
TEST_VOLUMES_ML = [vol / 1000.0 for vol in TEST_VOLUMES_UL]
REPLICATES = 5
LIQUID_TYPE = "water"
SOURCE_VIAL = "liquid_source_0"
DESTINATION_VIAL = "liquid_source_0"


def main():
    print(f"Water Validation Test - {', '.join(map(str, TEST_VOLUMES_UL))} μL")
    print(f"Simulation: {SIMULATE}")
    
    lash_e = None
    try:
        # Initialize
        lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
        lash_e.nr_robot.home_robot_components()
        
        # Create output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"output/water_validation_{timestamp}"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output: {output_folder}")
        
        # Run validation
        results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial=SOURCE_VIAL,
            destination_vial=DESTINATION_VIAL,
            liquid_type=LIQUID_TYPE,
            volumes_ml=TEST_VOLUMES_ML,
            replicates=REPLICATES,
            output_folder=output_folder,
            plot_title=f"Water Validation - {', '.join(map(str, TEST_VOLUMES_UL))} μL",
            save_raw_data=True,
            compensate_overvolume=True,
            condition_tip_enabled=True,
            conditioning_volume_ul=200,
            quality_std_threshold=0.0005  # Apply strict quality control (0.5mg threshold)
        )
        
        print("Validation complete")
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
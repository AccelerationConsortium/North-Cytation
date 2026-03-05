#!/usr/bin/env python3
"""
Simple Dispense Test Workflow
Dispenses 0.005 mL into a specified number of wells in a wellplate.
"""

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy

# Configuration
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/calibration_vials_short.csv"
SIMULATE = False  # Set to False for hardware
NUMBER_OF_WELLS = 50  # How many wells to dispense into
DISPENSE_VOLUME_ML = 0.005  # Volume to dispense per well
SOURCE_VIAL = "liquid_source_0"  # Which vial to dispense from

def run_simple_dispense():
    """Run the simple dispense workflow."""
    
    print("="*50)
    print("SIMPLE DISPENSE TEST WORKFLOW")
    print(f"Dispensing {DISPENSE_VOLUME_ML*1000:.1f} µL into {NUMBER_OF_WELLS} wells")
    print(f"Source: {SOURCE_VIAL}")
    print(f"Mode: {'SIMULATION' if SIMULATE else 'HARDWARE'}")
    print("="*50)
    
    # Initialize robot
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
    
    try:
        # DMSO validation - 50 replicates of 0.005 mL
        print("\n" + "="*50)
        print("DMSO VALIDATION PHASE")
        print("Testing 50 replicates of 0.005 mL DMSO dispensing")
        print("="*50)
        
        dmso_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial=SOURCE_VIAL,
            destination_vial=SOURCE_VIAL, 
            liquid_type='DMSO',
            volumes_ml=[0.005],  # Same volume as main workflow
            replicates=50,
            output_folder="output",
            switch_pipet=False,
            save_raw_data=not SIMULATE,
            condition_tip_enabled=True,
            conditioning_volume_ul=25
        )
        
        print(f"DMSO Validation Results: R²={dmso_results['r_squared']:.3f}, Accuracy={dmso_results['mean_accuracy_pct']:.1f}%")
        print("="*50)
        
          
        # Dispense into sequential wells (0, 1, 2, ...)
        well_indices = list(range(NUMBER_OF_WELLS))
        volumes = [DISPENSE_VOLUME_ML] * NUMBER_OF_WELLS
        
        # Single aspirate, multiple dispense for efficiency
        total_volume = DISPENSE_VOLUME_ML * NUMBER_OF_WELLS
        
        print(f"\nAspirating {total_volume*1000:.1f} µL from {SOURCE_VIAL}")
        lash_e.nr_robot.aspirate_from_vial(SOURCE_VIAL, total_volume, liquid="DMSO")
        
        print(f"Dispensing into wells {well_indices[0]} to {well_indices[-1]}")
        lash_e.nr_robot.dispense_into_wellplate(
            dest_wp_num_array=well_indices,
            amount_mL_array=volumes,
            liquid="DMSO"
        )
        
        # Clean up
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.move_home()
        
        print("\n" + "="*50)
        print("COMPLETE WORKFLOW FINISHED!")
        print(f"VALIDATION: 50 DMSO replicates at 5.0 µL each")
        print(f"MAIN: {NUMBER_OF_WELLS} dispenses at {DISPENSE_VOLUME_ML*1000:.1f} µL each")
        print("="*50)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        lash_e.logger.error(f"Workflow failed: {e}")
        
    finally:
        # Always move home
        try:
            lash_e.nr_robot.move_home()
        except:
            pass

if __name__ == "__main__":
    run_simple_dispense()
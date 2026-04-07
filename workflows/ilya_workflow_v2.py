#!/usr/bin/env python3
"""
Ilya Workflow V2 - Updated to Modern API
========================================

Updates from original:
- Uses string vial names instead of indices
- Separates aspirate_from_vial and dispense_into_wellplate calls  
- Adds liquid= parameter for optimized pipetting
- Includes embedded validation patterns
- Uses modern workflow structure patterns

Author: Updated for modern North robotics API
Date: March 15, 2026
"""

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Import embedded calibration validation
from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy

# ========================================
# CONFIGURATION SECTION
# ========================================

# File paths
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/ilya_input_vials.csv"  # Updated extension
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Ilya_Measurement.prt"  
INSTRUCTIONS_FILE = "../utoronto_demo/status/ilya_input.csv"

# Workflow settings
SIMULATE = False  # Set to False for hardware execution
N_ROUNDS = 3  # Number of rounds to execute
WELLS_PER_ROUND = 48  # Wells per round (48 for rounds 1-2, alternative: 96 for full plate)

# Volume settings (in uL)
MIN_PIPETTE_VOLUME_UL = 10.0   # Minimum volume for accurate pipetting
MAX_PIPETTE_VOLUME_UL = 200.0  # Maximum volume per pipetting operation  

# Conditioning volumes by tip type
SMALL_TIP_CONDITIONING_UL = 150.0  # Conditioning volume for small tips
LARGE_TIP_CONDITIONING_UL = 250.0  # Conditioning volume for large tips
SMALL_TIP_MAX_UL = 150.0  # Maximum volume for small tips

# ========================================
# VALIDATION FUNCTIONS  
# ========================================

def move_lid_to_wellplate(lash_e):
    lash_e.nr_track.grab_wellplate_from_location('lid_storage_96', wellplate_type='96_wellplate_lid', waypoint_locations=['cytation_safe_area'])
    lash_e.nr_track.release_wellplate_in_location('pipetting_area', wellplate_type='96_wellplate_lid')

def remove_lid_from_wellplate(lash_e):
    lash_e.nr_track.grab_wellplate_from_location('pipetting_area', wellplate_type='96_wellplate_lid', waypoint_locations=['cytation_safe_area'])
    lash_e.nr_track.release_wellplate_in_location('lid_storage_96', wellplate_type='96_wellplate_lid')


def validate_recipe_data(input_data):
    """Validate the recipe data for consistency and pipettable volumes."""
    print("\n=== RECIPE DATA VALIDATION ===")
    
    # Check for required columns
    required_columns = ['water_dye', 'water', 'glycerol_dye', 'glycerol', 'ethanol_dye', 'ethanol']
    missing_columns = [col for col in required_columns if col not in input_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert all volumes to numeric
    for col in required_columns:
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
        
        # Check for invalid values
        invalid_count = input_data[col].isna().sum()
        if invalid_count > 0:
            print(f"⚠️  Column {col}: {invalid_count} invalid values converted to 0")
            input_data[col] = input_data[col].fillna(0)
    
    # Validate volume ranges
    total_wells = len(input_data)
    wells_with_volumes = 0
    
    for idx, row in input_data.iterrows():
        total_volume = row[required_columns].sum()
        if total_volume > 0:
            wells_with_volumes += 1
            
            # Check individual volumes are pipettable
            for col in required_columns:
                volume = row[col]
                if 0 < volume < MIN_PIPETTE_VOLUME_UL:
                    print(f"⚠️  Well {idx}: {col} volume {volume}uL below minimum {MIN_PIPETTE_VOLUME_UL}uL")
                elif volume > MAX_PIPETTE_VOLUME_UL:
                    print(f"⚠️  Well {idx}: {col} volume {volume}uL above maximum {MAX_PIPETTE_VOLUME_UL}uL")
    
    print(f"✓ Recipe validation complete: {wells_with_volumes}/{total_wells} wells have volumes")
    return input_data

def condition_tip_for_liquid(lash_e, vial_name, liquid_type='water', max_volume_ul=None):
    """Condition pipette tip by aspirating and dispensing from source vial multiple times."""
    # Determine tip type and conditioning volume based on expected volumes
    if max_volume_ul and max_volume_ul > SMALL_TIP_MAX_UL:
        conditioning_volume_ul = LARGE_TIP_CONDITIONING_UL
        tip_type = "large"
    else:
        conditioning_volume_ul = SMALL_TIP_CONDITIONING_UL  
        tip_type = "small"
    
    # Standard conditioning: 5 cycles (like other workflows)
    cycles = 5
    volume_per_cycle_ml = conditioning_volume_ul / 1000
    
    print(f"  Conditioning {tip_type} tip with {vial_name}: {cycles} cycles of {conditioning_volume_ul}uL (liquid={liquid_type})")
    
    try:
        for cycle in range(cycles):
            # Aspirate conditioning volume
            lash_e.nr_robot.aspirate_from_vial(
                source_vial_name=vial_name,
                amount_mL=volume_per_cycle_ml,
                liquid=liquid_type,
                track_height=True
            )
            
            # Dispense back to same vial
            lash_e.nr_robot.dispense_into_vial(
                dest_vial_name=vial_name,
                amount_mL=volume_per_cycle_ml,
                liquid=liquid_type
            )
        
        print(f"  Tip conditioning complete for {vial_name}")
        
    except Exception as e:
        print(f"  Warning: Tip conditioning failed for {vial_name}: {e}")

def dispense_component_to_wells(lash_e, component_df, vial_name, liquid_type, well_range):
    """
    Dispense a specific component from vial to multiple wellplate positions.
    Uses modern separate aspirate/dispense pattern with liquid parameter.
    Includes vial movement from storage to working position and back.
    """
    if len(component_df) == 0:
        print(f"  No wells need {vial_name}")
        return
        
    print(f"  Dispensing {vial_name} to {len(component_df)} wells (liquid={liquid_type})")
    
    # Move vial from storage to working position
    print(f"    Moving {vial_name} to working position...")
    lash_e.nr_robot.move_vial_to_location(vial_name, "main_8mL_rack", 43)
    
    # Tip strategy: condition ethanol/water, change tip for glycerol every time
    if liquid_type == 'glycerol':
        print(f"    Using fresh tips for glycerol (no conditioning)")
    else:
        # Condition tip for ethanol and water only
        max_volume_ml = component_df[vial_name].max()  # Use actual column name
        max_volume_ul = max_volume_ml * 1000  # Convert mL to uL for tip sizing
        condition_tip_for_liquid(lash_e, vial_name, liquid_type, max_volume_ul)
    
    # Sort wells by volume (descending) for better pipetting efficiency
    component_df_sorted = component_df.sort_values(by=vial_name, ascending=False)  # Use actual column name
    
    for well_idx, row in component_df_sorted.iterrows():
        # Get volume directly from the vial column (already converted to mL)
        volume_ml = row[vial_name]  # This is in mL (after /1000 conversion)
        volume_ul = volume_ml * 1000  # Convert to uL for display and checking
        
        if volume_ul < MIN_PIPETTE_VOLUME_UL:
            print(f"    Skipping well {well_idx}: volume {volume_ul:.1f}uL below minimum {MIN_PIPETTE_VOLUME_UL}uL")
            continue
            
        try:
            print(f"    Well {well_idx}: {volume_ul:.1f}uL")
            
            # Aspirate from vial with liquid-specific parameters
            lash_e.nr_robot.aspirate_from_vial(
                source_vial_name=vial_name,
                amount_mL=volume_ml,  # Use mL for robot calls
                liquid=liquid_type,
            )
            
            # Dispense into wellplate at specific position
            lash_e.nr_robot.dispense_into_wellplate(
                dest_wp_num_array=[well_idx],
                amount_mL_array=[volume_ml],  # Use mL for robot calls
                liquid=liquid_type
            )
            
            # For glycerol: remove tip after each well  
            if liquid_type == 'glycerol':
                lash_e.nr_robot.remove_pipet()
            
        except Exception as e:
            print(f"    ERROR dispensing to well {well_idx}: {e}")
            if not SIMULATE:
                raise
    
    # Move vial back to storage position
    lash_e.nr_robot.remove_pipet()
    print(f"    Moving {vial_name} back to storage...")
    lash_e.nr_robot.return_vial_home(vial_name)

# ========================================
# CALIBRATION FUNCTION
# ========================================

def run_embedded_calibration(lash_e):
    """Run embedded calibration for dye solutions using validate_pipetting_accuracy.
    
    Args:
        lash_e: Lash_E coordinator instance
        
    Returns:
        dict: Calibration results for each dye solution
    """
    print(f"\nRunning embedded calibration...")
    cal_results = {}
    
    # Calibrate all liquid components using embedded validation
    dye_calibrations = [
        
        ("ethanol_dye", "ethanol"),
        ("water_dye", "water"),
        ("glycerol_dye", "glycerol")
    ]
    
    for source_vial, liquid_type in dye_calibrations:
        print(f"Calibrating {source_vial} (liquid={liquid_type})")
        
        # Add tip conditioning for water and ethanol, but not glycerol
        use_tip_conditioning = liquid_type in ['water', 'ethanol']
        
        try:
            if use_tip_conditioning:
                result = validate_pipetting_accuracy(
                    lash_e=lash_e,
                    source_vial=source_vial,
                    destination_vial=source_vial,  # Self-calibration
                    liquid_type=liquid_type,
                    volumes_ml=[0.070, 0.100, 0.150, 0.200],  # 70, 100, 150, 200 μL
                    replicates=3,
                    condition_tip_enabled=True,
                    conditioning_volume_ul=150,
                    adaptive_correction=True
                )
            else:
                # Glycerol: no tip conditioning, but use fresh tips for each measurement
                # Enable adaptive correction for better parameter tuning with viscous liquids
                result = validate_pipetting_accuracy(
                    lash_e=lash_e,
                    source_vial=source_vial,
                    destination_vial=source_vial,  # Self-calibration
                    liquid_type=liquid_type,
                    volumes_ml=[0.070, 0.100, 0.150, 0.200],  # 70, 100, 150, 200 μL
                    replicates=3,
                    switch_pipet=True,  # Fresh tip for each measurement (like real workflow)
                    adaptive_correction=True  # Enable dynamic parameter tuning for viscous glycerol
                )
            cal_results[source_vial] = result
            print(f"✓ {source_vial} calibration complete")
        except Exception as e:
            print(f"⚠️ {source_vial} calibration failed: {e}")
            cal_results[source_vial] = {"error": str(e)}
    
    return cal_results


# ========================================
# MAIN WORKFLOW FUNCTION
# ========================================

def ilya_workflow_v2():
   
    # Load recipe data (keep in uL for validation)
    input_data = pd.read_csv(INSTRUCTIONS_FILE, sep=',', index_col="Well")
    print(f"✓ Loaded recipe data: {len(input_data)} wells")
    print("Recipe data preview:")
    print(input_data.head())
    
    # Initialize Lash_E coordinator
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
    
    # Load and display vial information
    vial_data = pd.read_csv(INPUT_VIAL_STATUS_FILE)
    print(f"\n=== VIAL SETUP ===")
    print(vial_data[['vial_name', 'location', 'location_index', 'vial_volume', 'notes']].to_string(index=False))
    
    RUN_VALIDATION = True  # Set to False to skip embedded calibration validation

    # Run embedded calibration before main workflow
    if RUN_VALIDATION:
        cal_results = run_embedded_calibration(lash_e)
        input("Pausing...")
             
    # Validate recipe data (before converting to mL)
    input_data = validate_recipe_data(input_data)
    
    # Convert to mL after validation
    input_data = input_data / 1000
    

    for round_num in range(1, N_ROUNDS + 1):
        print(f"\n--- ROUND {round_num}/{N_ROUNDS} ---")
        
        # Set well range for this round
        if round_num == 1:
            wells = range(0, 48)
        elif round_num == 2:
            wells = range(48, 96)
        else:
            wells = range(0, 48)  # Round 3: repeat wells 0-47
        
        # Wellplate management
        if round_num == 1:
            print(f"Getting new wellplate for round {round_num}")
            lash_e.nr_track.get_new_wellplate()
        elif round_num == 2:
            print(f"Using same wellplate for round {round_num}")
        else:  # round 3
            print(f"Discarding wellplate and getting new one for round {round_num}")
            lash_e.nr_track.discard_wellplate()
            lash_e.nr_track.get_new_wellplate()
        
        # Set well indices for this round
        # Always use the same CSV data (wells 0-47) but map to different wellplate positions
        csv_wells = range(0, 48)  # Always use CSV wells 0-47
        round_data = input_data[input_data.index.isin(csv_wells)].copy()
        
        # Map CSV well indices to actual wellplate positions
        if round_num == 2:
            # For round 2, map CSV wells 0-47 to wellplate wells 48-95
            wellplate_mapping = {i: i + 48 for i in range(48)}
            round_data.index = [wellplate_mapping[i] for i in round_data.index]
        # Rounds 1 and 3 use wells 0-47 as-is
        
        print(f"Round {round_num} data shape: {round_data.shape}")
        
        # Process each liquid component - vial name = column name
        liquid_components = [
            ('water_dye', 'water'),
            ('water', 'water'),
            ('glycerol_dye', 'glycerol'),
            ('glycerol', 'glycerol'),  
            ('ethanol_dye', 'ethanol'),
            ('ethanol', 'ethanol')
        ]
        
        for vial_name, liquid_type in liquid_components:
            print(f"\n  Processing {vial_name} (liquid={liquid_type})")
            
            # Filter to wells that need this component
            component_df = round_data[round_data[vial_name] > 0].copy()
            
            if len(component_df) > 0:
                print(f"    Found {len(component_df)} wells needing {vial_name}")
                print("    Wells and volumes:")
                for well_idx, row in component_df.iterrows():
                    volume_ml = row[vial_name]  # Get from actual column name
                    volume_ul = volume_ml * 1000  # Convert to uL for display
                    print(f"      Well {well_idx}: {volume_ul:.1f}uL")
                
                # Dispense to wells using modern API
                dispense_component_to_wells(lash_e, component_df, vial_name, liquid_type, wells)
            else:
                print(f"    No wells need {vial_name}")
        
        print(f"\nRound {round_num} dispensing complete!")
        
        # Place lid before measurement
        print("Placing wellplate lid for measurement...")
        move_lid_to_wellplate(lash_e)
        
        # Measure wellplate after dispensing
        wells_list = list(wells)
        print(f"Measuring wellplate: wells {wells_list[0]}-{wells_list[-1]}")
        try:
            measurement_data = lash_e.measure_wellplate(
                protocol_file_path=MEASUREMENT_PROTOCOL_FILE,
                wells_to_measure=wells_list
            )
            print(f"✓ Measurement complete for round {round_num}")
            
            # Save measurement data if valid
            if measurement_data is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ilya_measurement_round{round_num}_{timestamp}.csv"
                filepath = os.path.join("output", filename)
                os.makedirs("output", exist_ok=True)
                measurement_data.to_csv(filepath)
                print(f"✓ Saved: {filename}")
            else:
                print(f"⚠️  No measurement data returned for round {round_num}")
            
        except Exception as e:
            print(f"⚠️  Measurement failed for round {round_num}: {e}")
        
        # Remove lid after measurement
        print("Removing wellplate lid after measurement...")
        remove_lid_from_wellplate(lash_e)
        print("✓ Lid removed successfully")
        
        if round_num < N_ROUNDS:
            if not SIMULATE:
                input("Press Enter when ready for next round (or Ctrl+C to stop)...")
            else:
                print("Simulation: Moving to next round...")
    
    # Step 6: Workflow completion
    print("\n" + "="*50)
    print("WORKFLOW COMPLETE!")
    print("="*50)
    print(f"Successfully completed {N_ROUNDS} rounds")
    print(f"Total wells processed: {N_ROUNDS * WELLS_PER_ROUND}")
    
    # Display vial information at completion
    print("\n=== VIAL INFORMATION ===")
    # Display with all columns visible (no truncation)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(lash_e.nr_robot.VIAL_DF.to_string(index=False))
    # Reset display options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width') 
    pd.reset_option('display.max_colwidth')

# ========================================
# EXECUTION SECTION
# ========================================

if __name__ == "__main__":
    try:
        ilya_workflow_v2()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
    except Exception as e:
        print(f"\n\nWorkflow failed with error: {e}")
        raise
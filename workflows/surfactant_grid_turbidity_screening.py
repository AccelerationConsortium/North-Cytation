"""
Surfactant Grid Turbidity Screening Workflow
Systematic dilution grid of two surfactants with turbidity measurements.
"""
import sys
sys.path.append("../utoronto_demo")
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from master_usdl_coordinator import Lash_E

# WORKFLOW CONSTANTS
SURFACTANT_A_NAME = "CTAB"
SURFACTANT_B_NAME = "SDS"
SURFACTANT_A_CONC_MM = 50.0  # mM stock concentration
SURFACTANT_B_CONC_MM = 50.0   # mM stock concentration

# Grid parametersany ch
MIN_CONC_LOG = -4  # 10^-8 mM minimum
MAX_CONC_LOG = 1  # 10^-3 mM maximum
LOG_STEP = 1       # 10^-1 step size
N_REPLICATES = 2
WELL_VOLUME_UL = 200  # μL per well
MAX_WELLS = 96 #Wellplate size
FINAL_SUBSTOCK_VOLUME = 6  # mL final volume for each dilution
MEASUREMENT_INTERVAL = 12  # Measure every N wells

# File paths
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/surfactant_grid_vials.csv"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Quick_Measurement.prt"

def calculate_grid_concentrations():
    """Calculate concentration grid points for both surfactants."""
    log_range = np.arange(MIN_CONC_LOG, MAX_CONC_LOG + LOG_STEP, LOG_STEP, dtype=float)
    concentrations = 10.0 ** log_range  # Convert log to actual concentrations
    return concentrations

def calculate_total_wells_needed():
    """Calculate total wells needed for the grid."""
    concs = calculate_grid_concentrations()
    grid_size = len(concs) ** 2  # n x n grid
    total_wells = grid_size * N_REPLICATES
    return total_wells, len(concs)

def create_dilution_series(lash_e, surfactant_vial, target_concs_mm, stock_conc_mm):
    """
    Create systematic serial dilutions of a surfactant.
    Uses 10-fold serial dilutions to avoid pipetting very small volumes.
    Prepares dilutions at 2x target concentrations since they will be mixed 1:1 in wells.
    
    Returns:
        list: Names of created dilution vials in concentration order
    """
    print(f"Creating serial dilution series for {surfactant_vial}")
    
    # Prepare dilutions at 2x target concentrations (since mixing 1:1 halves concentrations)
    dilution_concs = [conc * 2 for conc in target_concs_mm]
    
    # Sort concentrations from highest to lowest for serial dilution
    sorted_concs = sorted(dilution_concs, reverse=True)
    dilution_vials = []
    
    for i, target_conc in enumerate(sorted_concs):
        dilution_vial = f"dilution_{surfactant_vial}_{i}"
        dilution_vials.append(dilution_vial)
        
        if i == 0:
            # First dilution: dilute from stock
            dilution_factor = stock_conc_mm / target_conc
            stock_volume = FINAL_SUBSTOCK_VOLUME / dilution_factor
            water_volume = FINAL_SUBSTOCK_VOLUME - stock_volume
            
            print(f"  Creating {target_conc:.2e} mM (2x = {target_conc/2:.2e} mM final) from stock (dilution factor: {dilution_factor:.1f})")
            
            # Add water first, then stock
            if water_volume > 0:
                lash_e.nr_robot.dispense_into_vial_from_reservoir(
                    reservoir_index=0, vial_index=dilution_vial, volume=water_volume
                )
            
            lash_e.nr_robot.dispense_from_vial_into_vial(
                source_vial_name=surfactant_vial, 
                dest_vial_name=dilution_vial, 
                volume=stock_volume,
                liquid='water'
            )
        else:
            # Subsequent dilutions: use 10-fold serial dilution from previous
            previous_conc = sorted_concs[i-1]
            current_conc = target_conc
            
            # Should be 10-fold dilution (10^LOG_STEP)
            expected_dilution_factor = 10 ** LOG_STEP
            actual_dilution_factor = previous_conc / current_conc
            
            if abs(actual_dilution_factor - expected_dilution_factor) < 0.1:
                # Standard 10-fold dilution: 1 part previous + 9 parts water
                previous_volume = FINAL_SUBSTOCK_VOLUME / expected_dilution_factor
                water_volume = FINAL_SUBSTOCK_VOLUME - previous_volume
                
                print(f"  Creating {current_conc:.2e} mM (2x = {current_conc/2:.2e} mM final) from previous dilution (10-fold)")
                
                # Add water first, then previous dilution
                lash_e.nr_robot.dispense_into_vial_from_reservoir(
                    reservoir_index=1, vial_index=dilution_vial, volume=water_volume
                )
                
                previous_vial = dilution_vials[i-1]
                lash_e.nr_robot.dispense_from_vial_into_vial(
                    source_vial_name=previous_vial, 
                    dest_vial_name=dilution_vial, 
                    volume=previous_volume,
                    liquid='water'
                )
            else:
                # Non-standard dilution: calculate from previous
                dilution_factor = actual_dilution_factor
                previous_volume = FINAL_SUBSTOCK_VOLUME / dilution_factor
                water_volume = FINAL_SUBSTOCK_VOLUME - previous_volume
                
                print(f"  Creating {current_conc:.2e} mM (2x = {current_conc/2:.2e} mM final) from previous (factor: {dilution_factor:.1f})")
                
                lash_e.nr_robot.dispense_into_vial_from_reservoir(
                    reservoir_index=1, vial_index=dilution_vial, volume=water_volume
                )
                
                previous_vial = dilution_vials[i-1]
                lash_e.nr_robot.dispense_from_vial_into_vial(
                    source_vial_name=previous_vial, 
                    dest_vial_name=dilution_vial, 
                    volume=previous_volume,
                    liquid='water'
                )
        
        # Vortex to mix
        lash_e.nr_robot.vortex_vial(vial_name=dilution_vial, vortex_time=3)
    
    # Return vials in original concentration order (not sorted)
    # Create mapping from 2x concentration to vial name, then map back to target concentrations
    conc_to_vial = dict(zip(sorted_concs, dilution_vials))
    ordered_vials = [conc_to_vial[conc * 2] for conc in target_concs_mm]
    
    return ordered_vials

def check_measurement_interval(lash_e, current_well, wellplate_data, total_wells_added):
    """
    Check if it's time to measure based on MEASUREMENT_INTERVAL.
    
    Args:
        lash_e: Lash_E controller
        current_well: Current well number within plate (0-based)
        wellplate_data: Dict tracking wellplate info
        total_wells_added: Total wells added across all plates
        
    Returns:
        updated_wellplate_data
    """
    # Check if we should measure (every MEASUREMENT_INTERVAL wells)
    if total_wells_added > 0 and total_wells_added % MEASUREMENT_INTERVAL == 0:
        # Determine which wells to measure since last measurement
        last_measured = wellplate_data.get('last_measured_well', -1)
        wells_to_measure = list(range(last_measured + 1, current_well + 1))
        
        if wells_to_measure:
            print(f"Interval measurement: plate {wellplate_data['current_plate']}, wells {wells_to_measure[0]}-{wells_to_measure[-1]} ({len(wells_to_measure)} wells)")
            measurement_data = measure_turbidity(lash_e, wells_to_measure)
            
            # Store measurement data
            wellplate_data['measurements'].append({
                'plate_number': wellplate_data['current_plate'],
                'wells_measured': wells_to_measure,
                'measurement_type': 'interval',
                'data': measurement_data
            })
            
            wellplate_data['last_measured_well'] = current_well
    
    return wellplate_data

def manage_wellplate_switching(lash_e, current_well, wellplate_data):
    """
    Handle wellplate switching when MAX_WELLS is reached.
    """
    if current_well >= MAX_WELLS:
        # Measure any remaining unmeasured wells on current plate
        last_measured = wellplate_data.get('last_measured_well', -1)
        remaining_wells = list(range(last_measured + 1, wellplate_data['wells_used']))
        
        if remaining_wells:
            print(f"Final measurement for plate {wellplate_data['current_plate']}: wells {remaining_wells[0]}-{remaining_wells[-1]}")
            measurement_data = measure_turbidity(lash_e, remaining_wells)
            
            wellplate_data['measurements'].append({
                'plate_number': wellplate_data['current_plate'],
                'wells_measured': remaining_wells,
                'measurement_type': 'final',
                'data': measurement_data
            })
        
        # Discard current wellplate and get new one
        lash_e.discard_used_wellplate()
        lash_e.grab_new_wellplate()
        
        # Update tracking for new plate
        wellplate_data['current_plate'] += 1
        wellplate_data['wells_used'] = 0
        wellplate_data['last_measured_well'] = -1  # Reset for new plate
        
        print(f"Switched to wellplate {wellplate_data['current_plate']}")
        return 0, wellplate_data
    else:
        wellplate_data['wells_used'] = max(wellplate_data['wells_used'], current_well + 1)
        return current_well, wellplate_data

def pipette_grid_to_wellplate(lash_e, concs_a, concs_b, dilution_vials_a, dilution_vials_b):
    """Pipette concentration grid into wellplate(s) with interval measurements."""
    well_counter = 0
    well_map = []
    total_wells_added = 0
    
    # Initialize wellplate tracking
    wellplate_data = {
        'current_plate': 1,
        'wells_used': 0,
        'measurements': [],
        'last_measured_well': -1
    }
    
    for i, conc_a in enumerate(concs_a):
        for j, conc_b in enumerate(concs_b):
            for rep in range(N_REPLICATES):
                # Check if we need to switch wellplates
                well_counter, wellplate_data = manage_wellplate_switching(lash_e, well_counter, wellplate_data)
                
                # Get actual dilution vial names
                dilution_a_vial = dilution_vials_a[i]
                dilution_b_vial = dilution_vials_b[j]
                
                # Pipette volume from each dilution
                volume_each = WELL_VOLUME_UL / 2000  # Convert μL to mL and split
                
                # Aspirate both solutions
                lash_e.nr_robot.aspirate_from_vial(dilution_a_vial, volume_each, liquid='water')
                lash_e.nr_robot.aspirate_from_vial(dilution_b_vial, volume_each, liquid='water')
                
                # Dispense into well
                lash_e.nr_robot.dispense_into_wellplate(
                    dest_wp_num_array=[well_counter], 
                    amount_mL_array=[WELL_VOLUME_UL / 1000],  # Convert μL to mL
                    liquid='water'
                )
                
                # Record well information
                well_map.append({
                    'well': well_counter,
                    'plate': wellplate_data['current_plate'],
                    'surfactant_a': SURFACTANT_A_NAME,
                    'surfactant_b': SURFACTANT_B_NAME,
                    'conc_a_mm': conc_a,
                    'conc_b_mm': conc_b,
                    'replicate': rep + 1,
                    'vial_a': dilution_a_vial,
                    'vial_b': dilution_b_vial
                })
                
                well_counter += 1
                total_wells_added += 1
                
                # Check if it's time for interval measurement
                wellplate_data = check_measurement_interval(lash_e, well_counter - 1, wellplate_data, total_wells_added)
    
    lash_e.nr_robot.remove_pipet()
    
    # Final measurement of any remaining unmeasured wells
    if wellplate_data['wells_used'] > 0:
        last_measured = wellplate_data.get('last_measured_well', -1)
        remaining_wells = list(range(last_measured + 1, wellplate_data['wells_used']))
        
        if remaining_wells:
            print(f"Final measurement for plate {wellplate_data['current_plate']}: wells {remaining_wells[0]}-{remaining_wells[-1]}")
            final_measurement = measure_turbidity(lash_e, remaining_wells)
            
            wellplate_data['measurements'].append({
                'plate_number': wellplate_data['current_plate'],
                'wells_measured': remaining_wells,
                'measurement_type': 'final',
                'data': final_measurement
            })
    
    return well_map, wellplate_data

def combine_measurement_data(well_map, wellplate_data):
    """
    Combine measurement data from all intervals into a single DataFrame.
    
    Args:
        well_map: List of well information dictionaries
        wellplate_data: Dictionary containing measurement data from all intervals
        
    Returns:
        pd.DataFrame: Combined results with turbidity measurements
    """
    print("\nCombining measurement data from all intervals...")
    combined_results = []
    
    for measurement in wellplate_data['measurements']:
        plate_num = measurement['plate_number']
        measurement_data = measurement['data']
        wells_measured = measurement['wells_measured']
        
        # Handle simulation mode where measurement_data might be None
        if measurement_data is None:
            measurement_data = {'turbidity': [0.5] * len(wells_measured)}  # Mock data
        
        # Extract turbidity values for measured wells
        for well_idx in wells_measured:
            if well_idx < len(well_map):
                well_info = well_map[well_idx]
                # Find turbidity value for this well
                if well_idx < len(measurement_data.get('turbidity', [])):
                    turbidity = measurement_data['turbidity'][well_idx]
                    
                    combined_results.append({
                        'plate': plate_num,
                        'well': well_idx,
                        'surfactant_a': well_info['surfactant_a'],
                        'surfactant_b': well_info['surfactant_b'],
                        'conc_a_mm': well_info['conc_a_mm'],
                        'conc_b_mm': well_info['conc_b_mm'],
                        'replicate': well_info['replicate'],
                        'turbidity': turbidity,
                        'measurement_type': measurement.get('measurement_type', 'interval')
                    })
    
    print(f"Combined {len(combined_results)} measurements from {len(wellplate_data['measurements'])} intervals")
    return pd.DataFrame(combined_results)

def create_output_folder(simulate=True):
    """
    Create timestamped output folder for results.
    
    Args:
        simulate: If True, skip folder creation
        
    Returns:
        str: Output folder path or None if simulating
    """
    if simulate:
        print("Simulation mode: skipping output folder creation")
        return None
    
    # Create timestamped folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workflow_name = "surfactant_grid_turbidity_screening"
    folder_name = f"{workflow_name}_{timestamp}"
    
    # Create output path
    output_dir = os.path.join("output", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created output folder: {output_dir}")
    return output_dir

def save_results(results_df, well_map, wellplate_data, all_measurements, concentrations, simulate=True):
    """
    Save results and metadata to timestamped output folder.
    
    Args:
        results_df: Combined measurement results DataFrame
        well_map: List of well information dictionaries
        wellplate_data: Dictionary containing wellplate tracking info
        all_measurements: List of all measurement intervals
        concentrations: Array of concentration values
        simulate: If True, skip saving
    """
    output_folder = create_output_folder(simulate)
    if not output_folder:
        return
    
    results_file = os.path.join(output_folder, f"results_{SURFACTANT_A_NAME}_{SURFACTANT_B_NAME}.csv")
    metadata_file = os.path.join(output_folder, "experiment_metadata.json")
    
    # Save results
    results_df.to_csv(results_file, index=False)
    
    # Save experiment metadata
    metadata = {
        'workflow': 'surfactant_grid_turbidity_screening',
        'surfactant_a': SURFACTANT_A_NAME,
        'surfactant_b': SURFACTANT_B_NAME,
        'stock_conc_a_mm': SURFACTANT_A_CONC_MM,
        'stock_conc_b_mm': SURFACTANT_B_CONC_MM,
        'grid_points': len(concentrations),
        'replicates': N_REPLICATES,
        'total_wells': len(well_map),
        'measurement_interval': MEASUREMENT_INTERVAL,
        'wellplates_used': wellplate_data['current_plate'],
        'measurement_intervals': len(all_measurements),
        'concentration_range_mm': [float(concentrations[0]), float(concentrations[-1])],
        'well_volume_ul': WELL_VOLUME_UL
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print(f"Metadata saved to: {metadata_file}")

def measure_turbidity(lash_e, well_indices):
    """Measure turbidity in specified wells."""
    print("Measuring turbidity...")
    data = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, well_indices)
    return data

def surfactant_grid_screening(simulate=True):
    """
    Main workflow for surfactant grid turbidity screening.
    
    Args:
        simulate (bool): Run in simulation mode
    """
    print("=== Surfactant Grid Turbidity Screening ===")
    print(f"Surfactant A: {SURFACTANT_A_NAME} ({SURFACTANT_A_CONC_MM} mM stock)")
    print(f"Surfactant B: {SURFACTANT_B_NAME} ({SURFACTANT_B_CONC_MM} mM stock)")
    
    # Calculate experimental parameters
    concentrations = calculate_grid_concentrations()
    total_wells, grid_dimension = calculate_total_wells_needed()
    
    print(f"Grid: {grid_dimension}x{grid_dimension} = {grid_dimension**2} conditions")
    print(f"Replicates: {N_REPLICATES}")
    print(f"Total wells needed: {total_wells}")
    print(f"Measurement interval: every {MEASUREMENT_INTERVAL} wells")
    print(f"Concentration range: {concentrations[0]:.2e} to {concentrations[-1]:.2e} mM")
    
    num_wellplates_needed = (total_wells + MAX_WELLS - 1) // MAX_WELLS  # Ceiling division
    print(f"Wellplates needed: {num_wellplates_needed}")
    
    # 1. Initialize workstation
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
    
    # 2. Check input files and get new wellplate
    lash_e.nr_robot.check_input_file()
    lash_e.nr_track.check_input_file()
    lash_e.grab_new_wellplate()
    
    # 3. Create dilution series for both surfactants
    dilution_vials_a = create_dilution_series(lash_e, "surfactant_a", concentrations, SURFACTANT_A_CONC_MM)
    dilution_vials_b = create_dilution_series(lash_e, "surfactant_b", concentrations, SURFACTANT_B_CONC_MM)
    
    # 4. Pipette grid into wellplate(s) with interval measurements
    well_map, wellplate_data = pipette_grid_to_wellplate(lash_e, concentrations, concentrations, dilution_vials_a, dilution_vials_b)
    
    # 5. Combine measurement data from all intervals
    results_df = combine_measurement_data(well_map, wellplate_data)
    all_measurements = wellplate_data['measurements']
    
    # 6. Save results to timestamped output folder
    save_results(results_df, well_map, wellplate_data, all_measurements, concentrations, simulate)
    
    print(f"Experiment completed using {wellplate_data['current_plate']} wellplates")
    print(f"Total measurement datasets: {len(all_measurements)}")
    
    # 7. Final cleanup (last wellplate already measured in pipette_grid_to_wellplate)
    lash_e.discard_used_wellplate()
    lash_e.nr_robot.move_home()
    
    print("=== Workflow Complete ===")
    return results_df, all_measurements

if __name__ == "__main__":
    """
    Run the surfactant grid screening workflow.
    Set simulate=False when ready to run with actual hardware.
    """
    results, measurements = surfactant_grid_screening(simulate=True)
    print(f"Collected data for {len(results)} wells")
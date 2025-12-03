"""
Surfactant Grid Turbidity Screening Workflow
Systematic dilution grid of two surfactants with turbidity measurements.

DATA PROTECTION FEATURES:
- Raw Cytation data is immediately backed up to output/cytation_raw_backups/ (preserves complete original data)
- Processed measurement data is backed up to output/measurement_backups/ after each interval
- If processing fails, use recover_raw_cytation_data() and recover_from_measurement_backups() functions

RECOVERY USAGE:
  # Recover original Cytation data if processing failed:
  raw_data_list = recover_raw_cytation_data()
  
  # Recover processed measurements from crashed workflow:  
  measurements = recover_from_measurement_backups()
"""
import sys

from sympy import use
sys.path.append("../utoronto_demo")
import pandas as pd
import numpy as np
import os
import json
import glob
from datetime import datetime
from master_usdl_coordinator import Lash_E

# Surfactant library with stock concentrations (from cmc_exp_new.py)
SURFACTANT_LIBRARY = {
    "SDS": {
        "full_name": "Sodium Dodecyl Sulfate",
        "category": "anionic",
        "stock_conc": 50,  # mM
    },
    "NaDC": {
        "full_name": "Sodium Docusate", 
        "category": "anionic",
        "stock_conc": 25,  # mM
    },
    "NaC": {
        "full_name": "Sodium Cholate",
        "category": "anionic", 
        "stock_conc": 50,  # mM
    },
    "CTAB": {
        "full_name": "Hexadecyltrimethylammonium Bromide",
        "category": "cationic",
        "stock_conc": 5,  # mM
    },
    "DTAB": {
        "full_name": "Dodecyltrimethylammonium Bromide",
        "category": "cationic",
        "stock_conc": 50,  # mM
    },
    "TTAB": {
        "full_name": "Tetradecyltrimethylammonium Bromide", 
        "category": "cationic",
        "stock_conc": 50,  # mM
    }
}

# WORKFLOW CONSTANTS
SIMULATE = False  # Set to False for actual hardware execution

# Pump configuration:
# Pump 0 = Pipetting pump (no reservoir, used for aspirate/dispense)
# Pump 1 = Water reservoir pump (carousel angle 45°, height 70)

SURFACTANT_A_NAME = "TTAB"
SURFACTANT_B_NAME = "SDS"
SURFACTANT_A_CONC_MM = 50.0  # mM stock concentration
SURFACTANT_B_CONC_MM = 50.0   # mM stock concentration

# Grid parametersany ch
MIN_CONC_LOG = -4  # 10^-7 mM minimum
MAX_CONC_LOG = 1  # 10^-2 mM maximum
LOG_STEP = 1       # 10^-1 step size
N_REPLICATES = 1
WELL_VOLUME_UL = 200  # μL per well
MAX_WELLS = 96 #Wellplate size
FINAL_SUBSTOCK_VOLUME = 6  # mL final volume for each dilution
MEASUREMENT_INTERVAL = 12  # Measure every N wells

# File paths
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/surfactant_grid_vials_expanded.csv"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\CMC_Absorbance_96.prt"

def calculate_grid_concentrations():
    """Calculate concentration grid points for both surfactants."""
    log_range = np.arange(MIN_CONC_LOG, MAX_CONC_LOG + LOG_STEP, LOG_STEP, dtype=float)
    concentrations = 10.0 ** log_range  # Convert log to actual concentrations
    return concentrations

def load_substock_tracking():
    """Create simple in-memory tracking for substocks."""
    tracking = {}
    concentrations = calculate_grid_concentrations()
    
    for surfactant in SURFACTANT_LIBRARY.keys():
        tracking[surfactant] = {
            "stock_available": True,
            "dilutions_created": set()  # Set of concentration values that have been created
        }
    
    return tracking

def save_substock_tracking(tracking_data, output_folder):
    """Save substock tracking data to output folder."""
    tracking_file = os.path.join(output_folder, "substock_tracking.json")
    # Convert sets to lists for JSON serialization
    json_data = {}
    for surf, data in tracking_data.items():
        json_data[surf] = {
            "stock_available": data["stock_available"],
            "dilutions_created": list(data["dilutions_created"])
        }
    
    with open(tracking_file, 'w') as f:
        json.dump(json_data, f, indent=2)

def load_wellplate_state():
    """Create simple in-memory wellplate tracking."""
    return {
        "current_plate": 1,
        "wells_used": 0,
        "last_measured_well": -1,
        "total_plates_used": 0
    }

def save_wellplate_state(state_data):
    """Wellplate state doesn't need to be saved - North system handles it."""
    pass  # No file saving needed

def get_achievable_concentrations(surfactant_name, target_concentrations):
    """
    Get which concentrations are achievable for a surfactant.
    
    Args:
        surfactant_name: Name of surfactant
        target_concentrations: List of target concentrations
        
    Returns:
        list: Achievable concentrations (None for non-achievable)
    """
    stock_conc = SURFACTANT_LIBRARY[surfactant_name]["stock_conc"]
    
    achievable = []
    for target in target_concentrations:
        # Target is final concentration after 1:1 mixing
        # So substock needs to be 2x higher
        required_substock = target * 2
        
        if required_substock <= stock_conc:
            achievable.append(target)
        else:
            achievable.append(None)
    
    return achievable

def check_or_create_substocks(lash_e, surfactant_name, target_concentrations, tracking):
    """
    Check if substocks exist for a surfactant, create missing ones.
    
    Args:
        lash_e: Lash_E coordinator
        surfactant_name: Name of surfactant
        target_concentrations: List of target concentrations
        tracking: In-memory tracking dict
        
    Returns:
        tuple: (dilution_vials, dilution_steps, achievable_concentrations)
    """
    print(f"Checking substocks for {surfactant_name}...")
    
    # Get achievable concentrations
    achievable_concs = get_achievable_concentrations(surfactant_name, target_concentrations)
    
    dilution_vials = []
    dilution_steps = []
    
    stock_conc = SURFACTANT_LIBRARY[surfactant_name]["stock_conc"]
    
    for i, (target_conc, achievable_conc) in enumerate(zip(target_concentrations, achievable_concs)):
        vial_name = f"{surfactant_name}_dilution_{i}"
        dilution_vials.append(vial_name)
        
        if achievable_conc is None:
            # Not achievable
            dilution_steps.append({
                'vial_name': vial_name,
                'target_conc_mm': target_conc,
                'achievable': False,
                'created': False,
                'reason': f'Stock concentration ({stock_conc} mM) too low'
            })
            continue
        
        # Check if already exists
        if target_conc in tracking[surfactant_name]["dilutions_created"]:
            print(f"  {vial_name}: Already exists, reusing")
            dilution_steps.append({
                'vial_name': vial_name,
                'target_conc_mm': target_conc,
                'final_conc_mm': achievable_conc,
                'achievable': True,
                'created': True,
                'reused': True
            })
            continue
        
        # Need to create this dilution
        print(f"  {vial_name}: Creating dilution for {target_conc:.2e} mM...")
        
        # Calculate dilution (same logic as original)
        required_substock_conc = target_conc * 2
        
        if i == 0:
            # From stock
            source_vial = f"{surfactant_name}_stock"
            source_conc = stock_conc
            dilution_factor = stock_conc / required_substock_conc
        else:
            # Try from previous dilution or stock
            prev_achievable = achievable_concs[i-1]
            if prev_achievable and (prev_achievable * 2) > required_substock_conc:
                source_vial = f"{surfactant_name}_dilution_{i-1}"
                source_conc = prev_achievable * 2
                dilution_factor = source_conc / required_substock_conc
            else:
                # Go back to stock
                source_vial = f"{surfactant_name}_stock"
                source_conc = stock_conc
                dilution_factor = stock_conc / required_substock_conc
        
        # Calculate volumes
        stock_volume = FINAL_SUBSTOCK_VOLUME / dilution_factor  # mL
        water_volume = FINAL_SUBSTOCK_VOLUME - stock_volume     # mL
        
        print(f"    From {source_vial} ({source_conc:.2e} mM), {dilution_factor:.1f}x dilution")
        print(f"    Volumes: {stock_volume:.3f} mL + {water_volume:.3f} mL = {FINAL_SUBSTOCK_VOLUME} mL")
        
        # Perform dilution
        if not lash_e.simulate:
            lash_e.nr_robot.move_vial_to_location(source_vial, "clamp", 0)
            lash_e.nr_robot.aspirate(stock_volume, 0)
            lash_e.nr_robot.move_vial_to_location(vial_name, "clamp", 0)
            lash_e.nr_robot.dispense(stock_volume, 0)
            
            # Add water
            lash_e.nr_robot.move_vial_to_location("water", "clamp", 0)
            lash_e.nr_robot.aspirate(water_volume, 1)
            lash_e.nr_robot.move_vial_to_location(vial_name, "clamp", 0)
            lash_e.nr_robot.dispense(water_volume, 1)
        
        # Update tracking
        tracking[surfactant_name]["dilutions_created"].add(target_conc)
        
        # Record step
        dilution_steps.append({
            'vial_name': vial_name,
            'target_conc_mm': target_conc,
            'final_conc_mm': achievable_conc,
            'substock_conc_mm': required_substock_conc,
            'source_vial': source_vial,
            'source_conc_mm': source_conc,
            'dilution_factor': dilution_factor,
            'stock_volume_ul': stock_volume * 1000,
            'water_volume_ul': water_volume * 1000,
            'total_volume_ml': FINAL_SUBSTOCK_VOLUME,
            'achievable': True,
            'created': True,
            'reused': False
        })
    
    return dilution_vials, dilution_steps, achievable_concs

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
        tuple: (list of dilution vial names, list of dilution step details)
    """
    print(f"Creating serial dilution series for {surfactant_vial}")
    
    # Prepare dilutions at 2x target concentrations (since mixing 1:1 halves concentrations)
    dilution_concs = [conc * 2 for conc in target_concs_mm]
    
    # Sort concentrations from highest to lowest for serial dilution
    sorted_concs = sorted(dilution_concs, reverse=True)
    dilution_vials = []
    dilution_steps = []  # Track all dilution details
    
    for i, target_conc in enumerate(sorted_concs):
        dilution_vial = f"dilution_{surfactant_vial}_{i}"
        dilution_vials.append(dilution_vial)
        
        if i == 0:
            # First dilution: dilute from stock
            dilution_factor = stock_conc_mm / target_conc
            stock_volume = FINAL_SUBSTOCK_VOLUME / dilution_factor
            water_volume = FINAL_SUBSTOCK_VOLUME - stock_volume
            
            print(f"  Creating {target_conc:.2e} mM (2x = {target_conc/2:.2e} mM final) from stock (dilution factor: {dilution_factor:.1f})")
            
            # Record dilution step
            dilution_steps.append({
                'vial_name': dilution_vial,
                'target_conc_mm': target_conc,
                'final_conc_mm': target_conc / 2,  # After 1:1 mixing
                'source': 'stock',
                'source_vial': surfactant_vial,
                'source_conc_mm': stock_conc_mm,
                'dilution_factor': dilution_factor,
                'stock_volume_ml': round(stock_volume, 4),
                'water_volume_ml': round(water_volume, 4),
                'total_volume_ml': FINAL_SUBSTOCK_VOLUME,
                'stock_volume_ul': round(stock_volume * 1000, 1),
                'water_volume_ul': round(water_volume * 1000, 1)
            })
            
            # Add water first, then stock
            if water_volume > 0:
                lash_e.nr_robot.dispense_into_vial_from_reservoir(
                    reservoir_index=1, vial_index=dilution_vial, volume=water_volume, return_home=False
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
                
                # Record dilution step
                dilution_steps.append({
                    'vial_name': dilution_vial,
                    'target_conc_mm': current_conc,
                    'final_conc_mm': current_conc / 2,  # After 1:1 mixing
                    'source': 'serial_dilution',
                    'source_vial': dilution_vials[i-1],
                    'source_conc_mm': previous_conc,
                    'dilution_factor': expected_dilution_factor,
                    'stock_volume_ml': round(previous_volume, 4),
                    'water_volume_ml': round(water_volume, 4),
                    'total_volume_ml': FINAL_SUBSTOCK_VOLUME,
                    'stock_volume_ul': round(previous_volume * 1000, 1),
                    'water_volume_ul': round(water_volume * 1000, 1)
                })
                
                # Add water first, then previous dilution
                lash_e.nr_robot.dispense_into_vial_from_reservoir(
                    reservoir_index=1, vial_index=dilution_vial, volume=water_volume, return_home=False  # Pump 1 = water reservoir
                )
                
                previous_vial = dilution_vials[i-1]
                lash_e.nr_robot.dispense_from_vial_into_vial(
                    source_vial_name=previous_vial, 
                    dest_vial_name=dilution_vial, 
                    volume=previous_volume,
                    liquid='water')
            else:
                # Non-standard dilution: calculate from previous
                dilution_factor = actual_dilution_factor
                previous_volume = FINAL_SUBSTOCK_VOLUME / dilution_factor
                water_volume = FINAL_SUBSTOCK_VOLUME - previous_volume
                
                print(f"  Creating {current_conc:.2e} mM (2x = {current_conc/2:.2e} mM final) from previous (factor: {dilution_factor:.1f})")
                
                # Record dilution step
                dilution_steps.append({
                    'vial_name': dilution_vial,
                    'target_conc_mm': current_conc,
                    'final_conc_mm': current_conc / 2,  # After 1:1 mixing
                    'source': 'custom_dilution',
                    'source_vial': dilution_vials[i-1],
                    'source_conc_mm': previous_conc,
                    'dilution_factor': dilution_factor,
                    'stock_volume_ml': round(previous_volume, 4),
                    'water_volume_ml': round(water_volume, 4),
                    'total_volume_ml': FINAL_SUBSTOCK_VOLUME,
                    'stock_volume_ul': round(previous_volume * 1000, 1),
                    'water_volume_ul': round(water_volume * 1000, 1)
                })
                
                lash_e.nr_robot.dispense_into_vial_from_reservoir(
                    reservoir_index=1, vial_index=dilution_vial, volume=water_volume, return_home=False
                )
                
                previous_vial = dilution_vials[i-1]
                lash_e.nr_robot.dispense_from_vial_into_vial(
                    source_vial_name=previous_vial, 
                    dest_vial_name=dilution_vial, 
                    volume=previous_volume,
                    liquid='water'
                )
        
        # Vortex to mix
        lash_e.nr_robot.vortex_vial(vial_name=dilution_vial, vortex_time=8, vortex_speed=80)
    
    # FIXED: Return vials in decreasing concentration order to match physical reality
    # Physical dilution creates: vial_0 = highest, vial_1 = 2nd highest, etc.
    # This matches the actual serial dilution process (highest → lowest)
    return dilution_vials, dilution_steps

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
            measurement_entry = {
                'plate_number': wellplate_data['current_plate'],
                'wells_measured': wells_to_measure,
                'measurement_type': 'interval',
                'data': measurement_data,
                'timestamp': datetime.now().isoformat()
            }
            wellplate_data['measurements'].append(measurement_entry)
            
            # BACKUP: Save raw measurement data immediately to prevent data loss
            backup_raw_measurement_data(measurement_entry, wellplate_data['current_plate'], wells_to_measure)
            
            wellplate_data['last_measured_well'] = current_well
    
    return wellplate_data

def backup_raw_measurement_data(measurement_entry, plate_number, wells_measured):
    """
    Immediately backup raw measurement data to prevent data loss if processing crashes.
    
    Args:
        measurement_entry: Dictionary containing measurement data
        plate_number: Current plate number
        wells_measured: List of wells that were measured
    """
    try:
        # Create backup directory
        backup_dir = os.path.join("output", "measurement_backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create unique backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        backup_filename = f"raw_measurement_plate{plate_number}_wells{wells_measured[0]}-{wells_measured[-1]}_{timestamp}.json"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Save raw measurement data
        backup_data = {
            'measurement_entry': measurement_entry,
            'plate_number': plate_number,
            'wells_measured': wells_measured,
            'backup_timestamp': timestamp,
            'workflow': 'surfactant_grid_turbidity_screening'
        }
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)  # default=str handles datetime objects
        
        print(f"✓ Backed up measurement data to: {backup_path}")
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to backup measurement data: {e}")
        # Don't crash the workflow if backup fails

def backup_raw_cytation_data(raw_data, well_indices):
    """
    Save the raw Cytation DataFrame immediately after measurement to prevent data loss.
    This preserves the complete original data before any processing attempts.
    
    Args:
        raw_data: Raw DataFrame from lash_e.measure_wellplate()
        well_indices: List of well indices that were measured
    """
    try:
        # Create backup directory for raw Cytation data
        backup_dir = os.path.join("output", "cytation_raw_backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create unique backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        backup_filename = f"raw_cytation_wells{well_indices[0]}-{well_indices[-1]}_{timestamp}.pkl"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Save raw DataFrame using pickle to preserve structure perfectly
        raw_data.to_pickle(backup_path)
        
        # Also save as CSV for human readability (may lose some structure)
        csv_path = backup_path.replace('.pkl', '.csv')
        try:
            raw_data.to_csv(csv_path)
        except Exception:
            # MultiIndex columns might cause CSV issues, try flattening
            try:
                flat_data = raw_data.copy()
                if hasattr(raw_data.columns, 'levels'):
                    flat_data.columns = ['_'.join(map(str, col)).strip() for col in raw_data.columns]
                flat_data.to_csv(csv_path)
            except Exception:
                # If CSV fails entirely, just save the pickle
                pass
        
        print(f"✅ Raw Cytation data backed up to: {backup_path}")
        if os.path.exists(csv_path):
            print(f"   Human-readable version: {csv_path}")
        
    except Exception as e:
        print(f"⚠️ CRITICAL: Failed to backup raw Cytation data: {e}")
        print(f"   This could result in permanent data loss if processing fails!")

def recover_from_measurement_backups(backup_dir="output/measurement_backups"):
    """
    Recover measurement data from backup files if workflow crashed.
    
    Args:
        backup_dir: Directory containing backup measurement files
        
    Returns:
        list: List of recovered measurement entries
    """
    if not os.path.exists(backup_dir):
        print(f"No backup directory found at {backup_dir}")
        return []
    
    backup_files = glob.glob(os.path.join(backup_dir, "raw_measurement_*.json"))
    if not backup_files:
        print("No backup files found")
        return []
    
    print(f"Found {len(backup_files)} backup measurement files")
    recovered_measurements = []
    
    for backup_file in sorted(backup_files):
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            measurement_entry = backup_data['measurement_entry']
            recovered_measurements.append(measurement_entry)
            print(f"✓ Recovered measurement from: {os.path.basename(backup_file)}")
            
        except Exception as e:
            print(f"⚠️ Failed to recover from {backup_file}: {e}")
    
    print(f"Successfully recovered {len(recovered_measurements)} measurements")
    return recovered_measurements

def recover_raw_cytation_data(backup_dir="output/cytation_raw_backups"):
    """
    Recover raw Cytation DataFrame files if processing failed and data needs to be reanalyzed.
    
    Args:
        backup_dir: Directory containing raw Cytation backup files
        
    Returns:
        list: List of tuples (filepath, DataFrame) for all recovered raw data
    """
    if not os.path.exists(backup_dir):
        print(f"No raw Cytation backup directory found at {backup_dir}")
        return []
    
    backup_files = glob.glob(os.path.join(backup_dir, "raw_cytation_*.pkl"))
    if not backup_files:
        print("No raw Cytation backup files found")
        return []
    
    recovered_data = []
    print(f"Found {len(backup_files)} raw Cytation backup files:")
    
    for backup_file in sorted(backup_files):
        try:
            raw_df = pd.read_pickle(backup_file)
            recovered_data.append((backup_file, raw_df))
            
            basename = os.path.basename(backup_file)
            print(f"  ✅ Recovered: {basename}")
            print(f"     Shape: {raw_df.shape}, Columns: {len(raw_df.columns)}")
            
        except Exception as e:
            print(f"  ❌ Failed to recover {backup_file}: {e}")
    
    print(f"Successfully recovered {len(recovered_data)} raw Cytation datasets")
    return recovered_data

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
            
            measurement_entry = {
                'plate_number': wellplate_data['current_plate'],
                'wells_measured': remaining_wells,
                'measurement_type': 'final',
                'data': measurement_data,
                'timestamp': datetime.now().isoformat()
            }
            wellplate_data['measurements'].append(measurement_entry)
            
            # BACKUP: Save raw measurement data immediately
            backup_raw_measurement_data(measurement_entry, wellplate_data['current_plate'], remaining_wells)
        
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

def pipette_grid_to_wellplate(lash_e, concs_a, concs_b, dilution_vials_a, dilution_vials_b, surfactant_a_name, surfactant_b_name):
    """Pipette concentration grid into wellplate(s) with batched measurements for tip efficiency."""
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
    
    # Generate all well requirements first
    all_well_requirements = []
    for i, conc_a in enumerate(concs_a):
        for j, conc_b in enumerate(concs_b):
            for rep in range(N_REPLICATES):
                all_well_requirements.append({
                    'conc_a': concs_a[i],
                    'conc_b': concs_b[j], 
                    'conc_a_idx': i,
                    'conc_b_idx': j,
                    'dilution_a_vial': dilution_vials_a[i],
                    'dilution_b_vial': dilution_vials_b[j],
                    'replicate': rep + 1
                })
    
    # Process wells in batches of MEASUREMENT_INTERVAL
    for batch_start in range(0, len(all_well_requirements), MEASUREMENT_INTERVAL):
        batch_end = min(batch_start + MEASUREMENT_INTERVAL, len(all_well_requirements))
        current_batch = all_well_requirements[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//MEASUREMENT_INTERVAL + 1}: wells {well_counter}-{well_counter + len(current_batch) - 1}")
        
        # Check wellplate capacity and switch if needed
        wells_in_batch = []
        for req in current_batch:
            # Check if we need to switch wellplates
            well_counter, wellplate_data = manage_wellplate_switching(lash_e, well_counter, wellplate_data)
            wells_in_batch.append(well_counter)
            well_counter += 1
        
        # PHASE 1: Add all surfactant A solutions (grouped by vial to minimize movements)
        # Group by vial to aspirate multiple wells from same vial without moving it back
        from collections import defaultdict
        vial_a_groups = defaultdict(list)
        for batch_idx, req in enumerate(current_batch):
            actual_well = wells_in_batch[batch_idx]
            vial_a_groups[req['dilution_a_vial']].append((batch_idx, actual_well, req))
        
        # Sort vial groups by concentration (low→high to prevent contamination)
        sorted_vial_groups_a = sorted(vial_a_groups.items(), 
                                     key=lambda x: current_batch[x[1][0][0]]['conc_a'])
        
        print(f"  Phase 1: Adding surfactant A (low→high concentration, grouped by vial)")
        for vial_name, well_list in sorted_vial_groups_a:
            print(f"    Using vial {vial_name} for {len(well_list)} wells")
            for batch_idx, actual_well, req in well_list:
                volume_each = WELL_VOLUME_UL / 2000  # Convert μL to mL
                
                # Use safe location (handles vial movement properly)
                lash_e.nr_robot.aspirate_from_vial(req['dilution_a_vial'], volume_each, liquid='water', use_safe_location=True)
                lash_e.nr_robot.dispense_into_wellplate(
                    dest_wp_num_array=[actual_well], 
                    amount_mL_array=[volume_each],
                    liquid='water'
                )
            # Return vial home and remove tip after each vial to allow safe movement
            lash_e.nr_robot.remove_pipet()  # Remove tip to allow next vial movement
            lash_e.nr_robot.return_vial_home(vial_name)
            
        
        # PHASE 2: Add all surfactant B solutions (grouped by vial to minimize movements)
        vial_b_groups = defaultdict(list)
        for batch_idx, req in enumerate(current_batch):
            actual_well = wells_in_batch[batch_idx]
            vial_b_groups[req['dilution_b_vial']].append((batch_idx, actual_well, req))
        
        # Sort vial groups by concentration (low→high to prevent contamination)
        sorted_vial_groups_b = sorted(vial_b_groups.items(), 
                                     key=lambda x: current_batch[x[1][0][0]]['conc_b'])
        
        print(f"  Phase 2: Adding surfactant B (low→high concentration, grouped by vial)")
        for vial_name, well_list in sorted_vial_groups_b:
            print(f"    Using vial {vial_name} for {len(well_list)} wells")
            for batch_idx, actual_well, req in well_list:
                volume_each = WELL_VOLUME_UL / 2000  # Convert μL to mL
                
                # Use safe location (handles vial movement properly)
                lash_e.nr_robot.aspirate_from_vial(req['dilution_b_vial'], volume_each, liquid='water', use_safe_location=True)
                lash_e.nr_robot.dispense_into_wellplate(
                    dest_wp_num_array=[actual_well], 
                    amount_mL_array=[volume_each],
                    liquid='water'
                )
            # Remove tip first, then return vial home to allow safe movement
            lash_e.nr_robot.remove_pipet()  # Remove tip to allow next vial movement
            lash_e.nr_robot.return_vial_home(vial_name)
        
        # Record well information for this batch
        for i, req in enumerate(current_batch):
            actual_well = wells_in_batch[i]
            well_map.append({
                'well': actual_well,
                'plate': wellplate_data['current_plate'],
                'surfactant_a': surfactant_a_name,
                'surfactant_b': surfactant_b_name,
                'conc_a_mm': req['conc_a'],
                'conc_b_mm': req['conc_b'],
                'replicate': req['replicate'],
                'vial_a': req['dilution_a_vial'],
                'vial_b': req['dilution_b_vial']
            })
        
        # PHASE 3: Measure the completed batch
        print(f"  Phase 3: Measuring batch wells {wells_in_batch[0]}-{wells_in_batch[-1]}")
        measurement_data = measure_turbidity(lash_e, wells_in_batch)
        
        # Store measurement data
        measurement_entry = {
            'plate_number': wellplate_data['current_plate'],
            'wells_measured': wells_in_batch,
            'measurement_type': 'batch',
            'data': measurement_data,
            'timestamp': datetime.now().isoformat()
        }
        wellplate_data['measurements'].append(measurement_entry)
        
        # BACKUP: Save raw measurement data immediately
        backup_raw_measurement_data(measurement_entry, wellplate_data['current_plate'], wells_in_batch)
        
        # Update tracking
        wellplate_data['last_measured_well'] = wells_in_batch[-1]
        total_wells_added += len(current_batch)
        
        print(f"  ✓ Completed batch with {len(current_batch)} wells using 2 tips")
    
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
        for batch_idx, well_idx in enumerate(wells_measured):
            if well_idx < len(well_map):
                well_info = well_map[well_idx]
                # Find turbidity value for this well using batch position, not absolute well index
                if batch_idx < len(measurement_data.get('turbidity', [])):
                    turbidity = measurement_data['turbidity'][batch_idx]
                    
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
        simulate: If True, add "SIMULATION" prefix to folder name
        
    Returns:
        str: Output folder path
    """
    # Create timestamped folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_type = "surfactant_grid_turbidity"
    
    if simulate:
        folder_name = f"SIMULATION_{experiment_type}_{timestamp}"
        print("Simulation mode: creating output folder with SIMULATION prefix")
    else:
        folder_name = f"{experiment_type}_{timestamp}"
    
    # Create output path with experiment-specific subfolder
    output_dir = os.path.join("output", "surfactant_grid_experiments", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created output folder: {output_dir}")
    return output_dir

def save_results(results_df, well_map, wellplate_data, all_measurements, concentrations, dilution_vials_a, dilution_vials_b, dilution_steps_a, dilution_steps_b, surfactant_a_name, surfactant_b_name, simulate=True, output_folder=None):
    """
    Save results and metadata to output folder.
    """
    if output_folder is None:
        output_folder = create_output_folder(simulate)
    
    results_file = os.path.join(output_folder, f"results_{surfactant_a_name}_{surfactant_b_name}.csv")
    metadata_file = os.path.join(output_folder, "experiment_metadata.json")
    dilutions_file = os.path.join(output_folder, "dilution_series.json")
    dilution_report_file = os.path.join(output_folder, "dilution_report.txt")
    
    # Save results
    results_df.to_csv(results_file, index=False)
    
    # Create readable dilution report
    with open(dilution_report_file, 'w') as f:
        f.write("SURFACTANT GRID DILUTION SERIES REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        if simulate:
            f.write("NOTE: This is a SIMULATION - no actual pipetting was performed\n\n")
        
        # Surfactant A dilutions
        surf_a_info = SURFACTANT_LIBRARY[surfactant_a_name]
        f.write(f"SURFACTANT A: {surfactant_a_name}\n")
        f.write(f"Stock concentration: {surf_a_info['stock_conc']} mM\n")
        f.write("-" * 40 + "\n")
        for i, step in enumerate(dilution_steps_a):
            f.write(f"Step {i+1}: {step['vial_name']}\n")
            f.write(f"  Target concentration: {step['target_conc_mm']:.2e} mM (dilution stock)\n")
            f.write(f"  Final concentration: {step['final_conc_mm']:.2e} mM (after 1:1 mixing)\n")
            f.write(f"  Source: {step['source_vial']} ({step['source_conc_mm']:.2e} mM)\n")
            f.write(f"  Dilution factor: {step['dilution_factor']:.1f}x\n")
            f.write(f"  Volumes added:\n")
            f.write(f"    - Source solution: {step['stock_volume_ul']} uL\n")
            f.write(f"    - Water: {step['water_volume_ul']} uL\n")
            f.write(f"    - Total final volume: {step['total_volume_ml']} mL\n")
            f.write("\n")
        
        f.write("\n" + "=" * 50 + "\n\n")
        
        # Surfactant B dilutions
        surf_b_info = SURFACTANT_LIBRARY[surfactant_b_name]
        f.write(f"SURFACTANT B: {surfactant_b_name}\n")
        f.write(f"Stock concentration: {surf_b_info['stock_conc']} mM\n")
        f.write("-" * 40 + "\n")
        for i, step in enumerate(dilution_steps_b):
            f.write(f"Step {i+1}: {step['vial_name']}\n")
            f.write(f"  Target concentration: {step['target_conc_mm']:.2e} mM (dilution stock)\n")
            f.write(f"  Final concentration: {step['final_conc_mm']:.2e} mM (after 1:1 mixing)\n")
            f.write(f"  Source: {step['source_vial']} ({step['source_conc_mm']:.2e} mM)\n")
            f.write(f"  Dilution factor: {step['dilution_factor']:.1f}x\n")
            f.write(f"  Volumes added:\n")
            f.write(f"    - Source solution: {step['stock_volume_ul']} uL\n")
            f.write(f"    - Water: {step['water_volume_ul']} uL\n")
            f.write(f"    - Total final volume: {step['total_volume_ml']} mL\n")
            f.write("\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("WELL PLATE MIXING:\n")
        f.write(f"Each well contains {WELL_VOLUME_UL/2} uL of surfactant A dilution\n")
        f.write(f"                 + {WELL_VOLUME_UL/2} uL of surfactant B dilution\n")
        f.write(f"                 = {WELL_VOLUME_UL} uL total per well\n")
    
    # Save dilution series information (JSON for programmatic access)
    surf_a_info = SURFACTANT_LIBRARY[surfactant_a_name]
    surf_b_info = SURFACTANT_LIBRARY[surfactant_b_name]
    
    dilution_info = {
        'surfactant_a': {
            'name': surfactant_a_name,
            'stock_conc_mm': surf_a_info['stock_conc'],
            'dilution_steps': dilution_steps_a
        },
        'surfactant_b': {
            'name': surfactant_b_name,
            'stock_conc_mm': surf_b_info['stock_conc'],
            'dilution_steps': dilution_steps_b
        },
        'well_mixing': {
            'volume_each_ul': WELL_VOLUME_UL / 2,
            'total_volume_ul': WELL_VOLUME_UL
        }
    }
    
    with open(dilutions_file, 'w') as f:
        json.dump(dilution_info, f, indent=2)
    
    # Save experiment metadata
    metadata = {
        'workflow': 'surfactant_grid_turbidity_screening',
        'simulation_mode': simulate,
        'surfactant_a': surfactant_a_name,
        'surfactant_b': surfactant_b_name,
        'stock_conc_a_mm': surf_a_info['stock_conc'],
        'stock_conc_b_mm': surf_b_info['stock_conc'],
        'grid_points': len(concentrations),
        'replicates': N_REPLICATES,
        'total_wells': len(well_map),
        'measurement_interval': MEASUREMENT_INTERVAL,
        'wellplates_used': wellplate_data['current_plate'],
        'measurement_intervals': len(all_measurements),
        'concentration_range_mm': [float(concentrations[0]), float(concentrations[-1])],
        'well_volume_ul': WELL_VOLUME_UL
    }
    
    if simulate:
        metadata['note'] = "This is simulated data - no actual hardware was used"
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print(f"Dilution report saved to: {dilution_report_file}")
    print(f"Dilution series info saved to: {dilutions_file}")
    print(f"Metadata saved to: {metadata_file}")

def measure_turbidity(lash_e, well_indices):
    """
    Measure turbidity in specified wells using Cytation plate reader.
    
    Args:
        lash_e: Lash_E coordinator instance
        well_indices: List of well indices to measure
        
    Returns:
        dict: Processed measurement data with 'turbidity' values, or None if simulation
    """
    print(f"Measuring turbidity in wells {well_indices}...")
    
    # Get raw measurement data from Cytation
    raw_data = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, well_indices)
    
    # CRITICAL: Save raw Cytation data immediately to prevent loss
    if raw_data is not None:
        backup_raw_cytation_data(raw_data, well_indices)
    
    # Handle simulation mode
    if raw_data is None:
        return None
    
    # Process the complex DataFrame structure from Cytation
    try:
        # Extract absorbance data (assuming turbidity protocol uses absorbance)
        processed_data = extract_turbidity_values(raw_data, well_indices)
        return processed_data
    
    except Exception as e:
        print(f"⚠️  CRITICAL: Failed to process measurement data: {e}")
        print(f"Raw data structure: {type(raw_data)}")
        if hasattr(raw_data, 'columns'):
            print(f"Raw data columns: {raw_data.columns.tolist()}")
        
        print(f"🔄 Raw Cytation data was saved to backup before processing failed")
        print(f"   You can recover the data from output/cytation_raw_backups/")
        
        # Fallback: return mock data to prevent workflow crash
        print(f"⚠️  Using fallback mock data to continue workflow...")
        return {'turbidity': [0.5] * len(well_indices)}

def extract_turbidity_values(raw_data, well_indices):
    """
    Extract turbidity values from Cytation measurement DataFrame.
    
    Args:
        raw_data: DataFrame returned from lash_e.measure_wellplate()
        well_indices: List of well indices that were measured
        
    Returns:
        dict: {'turbidity': [value1, value2, ...]} aligned with well_indices
    """
    if raw_data is None or raw_data.empty:
        return {'turbidity': [0.5] * len(well_indices)}
    
    # Handle MultiIndex columns structure from Cytation
    if hasattr(raw_data.columns, 'levels'):
        # MultiIndex structure: (replicate_protocol, wavelength/measurement)
        
        # Find the first available measurement column
        # Common patterns: '600' (absorbance), wavelength values, etc.
        measurement_values = []
        
        for well_idx in well_indices:
            try:
                # Try to get measurement from first available column
                # Use first top-level column group and first measurement column
                top_level = raw_data.columns.levels[0][0]  # First replicate/protocol
                sub_columns = raw_data[top_level].columns
                
                # Find a numeric measurement column (wavelength or measurement)
                measurement_col = None
                for col in sub_columns:
                    # Look for numeric columns that aren't metadata
                    if str(col).replace('.', '').isdigit() or col in ['600', 'absorbance', 'turbidity']:
                        measurement_col = col
                        break
                
                if measurement_col is not None:
                    # Extract value for this well
                    if well_idx < len(raw_data):
                        value = raw_data[top_level][measurement_col].iloc[well_idx]
                        measurement_values.append(float(value))
                    else:
                        measurement_values.append(0.5)  # Default fallback
                else:
                    measurement_values.append(0.5)  # No measurement column found
                    
            except (IndexError, ValueError, KeyError) as e:
                print(f"Warning: Could not extract value for well {well_idx}: {e}")
                measurement_values.append(0.5)  # Fallback value
    
    else:
        # Simple DataFrame structure
        # Try to find a measurement column
        measurement_col = None
        for col in raw_data.columns:
            if str(col).replace('.', '').isdigit() or col.lower() in ['absorbance', 'turbidity', '600']:
                measurement_col = col
                break
        
        if measurement_col is not None:
            measurement_values = []
            for well_idx in well_indices:
                try:
                    # Convert well index to well name (0->A1, 1->A2, 12->B1, etc.)
                    row = well_idx // 12  # 0=A, 1=B, 2=C, etc.
                    col_num = (well_idx % 12) + 1  # 1-12
                    well_name = f"{chr(ord('A') + row)}{col_num}"
                    
                    # Get value for this well
                    if well_name in raw_data.index:
                        value = raw_data.loc[well_name, measurement_col]
                        measurement_values.append(float(value))
                    else:
                        print(f"Warning: Well {well_name} (index {well_idx}) not found in data")
                        measurement_values.append(0.5)  # Fallback
                        
                except (IndexError, ValueError, KeyError) as e:
                    print(f"Warning: Could not extract value for well {well_idx}: {e}")
                    measurement_values.append(0.5)  # Fallback value
        else:
            print(f"Warning: No measurement column found. Available columns: {raw_data.columns.tolist()}")
            measurement_values = [0.5] * len(well_indices)
    
    # Ensure we have the right number of values
    while len(measurement_values) < len(well_indices):
        measurement_values.append(0.5)
    
    return {'turbidity': measurement_values[:len(well_indices)]}

def surfactant_grid_screening(surfactant_a_name="TTAB", surfactant_b_name="SDS", simulate=True, output_folder=None):
    """
    Main workflow for surfactant grid turbidity screening.
    
    Args:
        surfactant_a_name (str): Name of first surfactant (cationic)
        surfactant_b_name (str): Name of second surfactant (anionic)
        simulate (bool): Run in simulation mode
        output_folder (str): Optional output folder path
    """
    print("=== Surfactant Grid Turbidity Screening ===")
    
    # Get surfactant info from library
    surf_a_info = SURFACTANT_LIBRARY[surfactant_a_name]
    surf_b_info = SURFACTANT_LIBRARY[surfactant_b_name]
    
    print(f"Surfactant A: {surfactant_a_name} ({surf_a_info['full_name']}, {surf_a_info['stock_conc']} mM stock)")
    print(f"Surfactant B: {surfactant_b_name} ({surf_b_info['full_name']}, {surf_b_info['stock_conc']} mM stock)")
    
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
    
    # Initialize tracking
    substock_tracking = load_substock_tracking()
    
    # 1. Initialize workstation
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
    
    # 2. Move robot components to home position
    print("Moving robot components to home positions...")
    lash_e.nr_robot.move_home()
    lash_e.nr_track.origin()
    lash_e.nr_robot.home_robot_components()
    
    # 3. Check input files and get new wellplate
    lash_e.nr_robot.check_input_file()
    lash_e.nr_track.check_input_file()
    lash_e.grab_new_wellplate()

    lash_e.nr_robot.prime_reservoir_line(1, 'water')
    
    # 4. Create or reuse dilution series for both surfactants
    dilution_vials_a, dilution_steps_a, achievable_a = check_or_create_substocks(lash_e, surfactant_a_name, concentrations, substock_tracking)
    dilution_vials_b, dilution_steps_b, achievable_b = check_or_create_substocks(lash_e, surfactant_b_name, concentrations, substock_tracking)
    
    # 5. Pipette grid into wellplate(s) with interval measurements
    well_map, wellplate_data = pipette_grid_to_wellplate(lash_e, achievable_a, achievable_b, dilution_vials_a, dilution_vials_b, surfactant_a_name, surfactant_b_name)
    
    # 6. Combine measurement data from all intervals
    results_df = combine_measurement_data(well_map, wellplate_data)
    all_measurements = wellplate_data['measurements']
    
    # 7. Create output folder if not provided
    if output_folder is None:
        output_folder = create_output_folder(simulate)
    
    # 8. Save substock tracking to output folder
    save_substock_tracking(substock_tracking, output_folder)
    
    # 9. Save results to output folder  
    save_results(results_df, well_map, wellplate_data, all_measurements, concentrations, dilution_vials_a, dilution_vials_b, dilution_steps_a, dilution_steps_b, surfactant_a_name, surfactant_b_name, simulate, output_folder)
    
    print(f"Experiment completed using {wellplate_data['current_plate']} wellplates")
    print(f"Total measurement datasets: {len(all_measurements)}")
    
    # 10. Final cleanup (last wellplate already measured in pipette_grid_to_wellplate)
    if wellplate_data['current_plate'] > 0:
        lash_e.discard_used_wellplate()
    lash_e.nr_robot.move_home()
    
    print("=== Workflow Complete ===")
    return results_df, all_measurements, wellplate_data['current_plate']

def run_all_surfactant_combinations(simulate=True):
    """
    Run screening for all anionic/cationic surfactant combinations.
    
    Args:
        simulate (bool): Run in simulation mode
        
    Returns:
        dict: Results for each surfactant combination
    """
    print("="*80)
    print("COMPREHENSIVE SURFACTANT SCREENING - ALL COMBINATIONS")
    print("="*80)
    
    # Get surfactants by category
    anionic_surfactants = [name for name, info in SURFACTANT_LIBRARY.items() if info['category'] == 'anionic']
    cationic_surfactants = [name for name, info in SURFACTANT_LIBRARY.items() if info['category'] == 'cationic']
    
    print(f"Anionic surfactants: {anionic_surfactants}")
    print(f"Cationic surfactants: {cationic_surfactants}")
    
    # Calculate total experiments and wells
    total_combinations = len(anionic_surfactants) * len(cationic_surfactants)
    wells_per_experiment = calculate_total_wells_needed()[0]
    total_wells = total_combinations * wells_per_experiment
    total_plates_needed = (total_wells + MAX_WELLS - 1) // MAX_WELLS
    
    print(f"Total combinations: {total_combinations}")
    print(f"Wells per experiment: {wells_per_experiment}")
    print(f"Total wells needed: {total_wells}")
    print(f"Estimated wellplates needed: {total_plates_needed}")
    
    # Create main output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output = f"output/comprehensive_surfactant_screening_{timestamp}"
    if simulate:
        main_output += "_SIMULATION"
    os.makedirs(main_output, exist_ok=True)
    
    # Initialize tracking
    global_substock_tracking = load_substock_tracking()
    current_plate = 1
    wells_used_total = 0
    all_results = {}
    
    print(f"\\nStarting comprehensive screening...")
    print(f"Results will be saved to: {main_output}")
    
    for i, anionic in enumerate(anionic_surfactants):
        for j, cationic in enumerate(cationic_surfactants):
            combo_num = i * len(cationic_surfactants) + j + 1
            print(f"\\n{'='*60}")
            print(f"COMBINATION {combo_num}/{total_combinations}: {anionic} + {cationic}")
            print(f"{'='*60}")
            
            # Create subfolder for this combination
            combo_folder = os.path.join(main_output, f"{combo_num:02d}_{anionic}_{cationic}")
            os.makedirs(combo_folder, exist_ok=True)
            
            try:
                # Run the experiment for this combination
                results_df, measurements, plates_used = surfactant_grid_screening(
                    surfactant_a_name=cationic,  # Cationic first
                    surfactant_b_name=anionic,   # Anionic second  
                    simulate=simulate,
                    output_folder=combo_folder
                )
                
                all_results[f"{anionic}_{cationic}"] = {
                    'anionic': anionic,
                    'cationic': cationic,
                    'results_df': results_df,
                    'measurements': measurements,
                    'plates_used': plates_used,
                    'wells_count': len(results_df),
                    'output_folder': combo_folder
                }
                
                wells_used_total += len(results_df)
                current_plate += plates_used
                
                print(f"✅ Completed {anionic} + {cationic}: {len(results_df)} wells, {plates_used} plates")
                
            except Exception as e:
                print(f"❌ Failed {anionic} + {cationic}: {e}")
                all_results[f"{anionic}_{cationic}"] = {
                    'anionic': anionic,
                    'cationic': cationic,
                    'error': str(e),
                    'completed': False
                }
    
    # Save global tracking and summary
    save_substock_tracking(global_substock_tracking, main_output)
    
    # Create summary report
    summary_file = os.path.join(main_output, "comprehensive_screening_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("COMPREHENSIVE SURFACTANT SCREENING SUMMARY\\n")
        f.write("=" * 50 + "\\n\\n")
        
        if simulate:
            f.write("NOTE: This was a SIMULATION - no actual hardware was used\\n\\n")
        
        f.write(f"Total combinations tested: {total_combinations}\\n")
        f.write(f"Total wells used: {wells_used_total}\\n")
        f.write(f"Total wellplates used: {current_plate - 1}\\n\\n")
        
        f.write("SURFACTANT COMBINATIONS:\\n")
        f.write("-" * 30 + "\\n")
        
        for combo_name, result in all_results.items():
            if 'error' in result:
                f.write(f"{result['anionic']} + {result['cationic']}: FAILED - {result['error']}\\n")
            else:
                f.write(f"{result['anionic']} + {result['cationic']}: {result['wells_count']} wells, {result['plates_used']} plates\\n")
    
    print(f"\\n{'='*80}")
    print("COMPREHENSIVE SCREENING COMPLETE")
    print(f"Total wells used: {wells_used_total}")
    print(f"Total wellplates used: {current_plate - 1}")
    print(f"Results saved to: {main_output}")
    print(f"{'='*80}")
    
    return all_results

if __name__ == "__main__":
    """
    Run surfactant grid screening.
    
    Options:
    1. Single experiment: Specify surfactant names
    2. Comprehensive: Run all anionic/cationic combinations
    """
    
    # Choose experiment mode
    RUN_COMPREHENSIVE = True  # Set to False for single experiment
    
    if RUN_COMPREHENSIVE:
        # Run all 9 combinations (3 anionic × 3 cationic)
        print("Running comprehensive screening for all surfactant combinations...")
        all_results = run_all_surfactant_combinations(simulate=SIMULATE)
        
        successful = sum(1 for r in all_results.values() if 'error' not in r)
        print(f"\\nCompleted: {successful}/{len(all_results)} combinations successful")
        
    else:
        # Run single experiment
        SURFACTANT_A = "TTAB"  # Cationic
        SURFACTANT_B = "SDS"   # Anionic
        
        print(f"Running single experiment: {SURFACTANT_A} + {SURFACTANT_B}")
        results, measurements, plates_used = surfactant_grid_screening(
            surfactant_a_name=SURFACTANT_A, 
            surfactant_b_name=SURFACTANT_B, 
            simulate=SIMULATE
        )
        print(f"Single experiment completed: {len(results)} wells, {plates_used} plates")
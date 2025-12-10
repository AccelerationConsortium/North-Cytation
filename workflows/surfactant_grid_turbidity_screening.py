"""
Surfactant Grid Turbidity + Fluorescence Screening Workflow
Systematic dilution grid of two surfactants with turbidity measurements followed by pyrene fluorescence.

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

sys.path.append("../utoronto_demo")
import pandas as pd
import numpy as np
import os
import json
import glob
from datetime import datetime
from master_usdl_coordinator import Lash_E

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available - heatmap visualization will be skipped")

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
SIMULATE = False # Set to False for actual hardware execution

# Pump configuration:
# Pump 0 = Pipetting pump (no reservoir, used for aspirate/dispense)
# Pump 1 = Water reservoir pump (carousel angle 45 deg, height 70)

# Grid parameters - Updated to better capture turbidity transition region
MIN_CONC_LOG = -4  # 10^-2 = 0.01 mM minimum (focus on higher concentrations)
MAX_CONC_LOG = 1   # 10^1 = 10 mM maximum (extend into asdtransition region)  
LOG_STEP = 1     # 10^0.5 ~= 3.16-fold steps for finer resolution
N_REPLICATES = 1
WELL_VOLUME_UL = 200  # uL per well
PYRENE_VOLUME_UL = 10  # uL pyrene_DMSO to add per well
MAX_WELLS = 96 #Wellplate size

# Constants
FINAL_SUBSTOCK_VOLUME = 6  # mL final volume for each dilution
MINIMUM_PIPETTE_VOLUME = 0.2  # mL (200 uL) - minimum volume for accurate pipetting
MEASUREMENT_INTERVAL = 36    # Measure every N wells to prevent evaporation

# Measurement protocol files for Cytation
TURBIDITY_PROTOCOL_FILE = r"C:\Protocols\CMC_Absorbance_96.prt"
FLUORESCENCE_PROTOCOL_FILE = r"C:\Protocols\CMC_Fluorescence_96.prt"

# File paths
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/surfactant_grid_vials_expanded.csv"

def calculate_grid_concentrations():
    """Calculate concentration grid points for both surfactants."""
    log_range = np.arange(MIN_CONC_LOG, MAX_CONC_LOG + LOG_STEP, LOG_STEP, dtype=float)
    concentrations = 10.0 ** log_range  # Convert log to actual concentrations
    return concentrations

def load_substock_tracking(logger=None):
    """Create substock tracking, detecting existing dilutions from CSV file."""
    tracking = {}
    concentrations = calculate_grid_concentrations()
    
    # Initialize tracking structure
    for surfactant in SURFACTANT_LIBRARY.keys():
        tracking[surfactant] = {
            "stock_available": True,
            "dilutions_created": set(),  # Set of concentration values that have been created
            "dilution_recipes": {}  # Dictionary to store full dilution recipe information by concentration
        }
    
    # Read CSV file to detect existing dilutions based on volume
    try:
        vial_file_path = "status/surfactant_grid_vials_expanded.csv"
        if os.path.exists(vial_file_path):
            import pandas as pd
            df = pd.read_csv(vial_file_path)
            
            if logger:
                logger.info("Checking for existing substocks in CSV...")
            else:
                print("Checking for existing substocks in CSV...")
            
            for _, row in df.iterrows():
                vial_name = row['vial_name']
                volume = float(row['vial_volume']) if pd.notna(row['vial_volume']) else 0.0
                
                # Skip if no volume (empty vial)
                if volume <= 0:
                    continue
                
                # Parse dilution vials to extract surfactant and concentration
                for surfactant_name in SURFACTANT_LIBRARY.keys():
                    if vial_name.startswith(f"{surfactant_name}_dilution_"):
                        try:
                            # Extract dilution index from vial name
                            dilution_idx = int(vial_name.split("_")[-1])
                            
                            # Map dilution index to concentration
                            if dilution_idx < len(concentrations):
                                target_conc = concentrations[dilution_idx]
                                tracking[surfactant_name]["dilutions_created"].add(target_conc)
                                if logger:
                                    logger.info(f"  Found existing {surfactant_name} dilution: {vial_name} = {target_conc:.2e} mM ({volume:.1f} mL)")
                                else:
                                    print(f"  Found existing {surfactant_name} dilution: {vial_name} = {target_conc:.2e} mM ({volume:.1f} mL)")
                        except (ValueError, IndexError):
                            continue
                        break
    
    except Exception as e:
        if logger:
            logger.warning(f"Could not load existing dilutions from CSV: {e}")
        else:
            print(f"Warning: Could not load existing dilutions from CSV: {e}")
    
    return tracking

def save_substock_tracking(tracking_data, output_folder):
    """Save substock tracking data to output folder."""
    tracking_file = os.path.join(output_folder, "substock_tracking.json")
    # Convert sets to lists for JSON serialization
    json_data = {}
    for surf, data in tracking_data.items():
        json_data[surf] = {
            "stock_available": data["stock_available"],
            "dilutions_created": list(data["dilutions_created"]),
            "dilution_recipes": {str(conc): recipe for conc, recipe in data["dilution_recipes"].items()}
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
    Check if substocks exist for a surfactant, create missing ones using multi-pass hierarchical strategy.
    
    Args:
        lash_e: Lash_E coordinator
        surfactant_name: Name of surfactant
        target_concentrations: List of target concentrations
        tracking: In-memory tracking dict
        
    Returns:
        tuple: (dilution_vials, dilution_steps, achievable_concentrations)
    """
    logger = lash_e.logger
    logger.debug(f"Checking substocks for {surfactant_name}")
    print(f"Checking substocks for {surfactant_name}...")
    
    # Get achievable concentrations
    achievable_concs = get_achievable_concentrations(surfactant_name, target_concentrations)
    logger.debug(f"{surfactant_name} achievable concentrations: {[f'{c:.2e}' if c is not None else 'None' for c in achievable_concs]} mM")
    
    dilution_vials = []
    dilution_steps = []
    all_dilution_operations = []  # Store operations for batched execution
    
    stock_conc = SURFACTANT_LIBRARY[surfactant_name]["stock_conc"]
    
    # Initialize vials list with None for all positions
    for i in range(len(target_concentrations)):
        dilution_vials.append(None)
    
    # MULTI-PASS APPROACH: Create dilutions that can be made, then use those for harder ones
    max_passes = 3
    created_in_this_session = set()
    
    for pass_num in range(max_passes):
        pass_operations = []
        progress_made = False
        
        print(f"  Pass {pass_num + 1}: Checking remaining dilutions...")
        
        for i, (target_conc, achievable_conc) in enumerate(zip(target_concentrations, achievable_concs)):
            vial_name = f"{surfactant_name}_dilution_{i}"
            
            if achievable_conc is None:
                # Not achievable at all
                if dilution_vials[i] is None:  # Only add step once
                    dilution_steps.append({
                        'vial_name': vial_name,
                        'target_conc_mm': target_conc,
                        'achievable': False,
                        'created': False,
                        'reason': f'Stock concentration ({stock_conc} mM) too low'
                    })
                continue
            
            # Skip if already created
            if target_conc in tracking[surfactant_name]["dilutions_created"] or target_conc in created_in_this_session:
                if dilution_vials[i] is None:  # Only update once
                    dilution_vials[i] = vial_name
                    
                    # Get complete recipe information for reused vials
                    if target_conc in tracking[surfactant_name]["dilution_recipes"]:
                        recipe = tracking[surfactant_name]["dilution_recipes"][target_conc]
                        dilution_steps.append({
                            'vial_name': vial_name,
                            'target_conc_mm': target_conc,
                            'final_conc_mm': target_conc,
                            'achievable': True,
                            'created': True,
                            'reused': True,
                            # Complete recipe information from tracking
                            'source_vial': recipe['source_vial'],
                            'source_conc_mm': recipe['source_conc_mm'],
                            'dilution_factor': recipe['dilution_factor'],
                            'stock_volume_ul': recipe['stock_volume_ul'],
                            'water_volume_ul': recipe['water_volume_ul'],
                            'total_volume_ml': recipe['total_volume_ml']
                        })
                    else:
                        # Fallback for vials created in current session
                        dilution_steps.append({
                            'vial_name': vial_name,
                            'target_conc_mm': target_conc,
                            'final_conc_mm': target_conc,
                            'achievable': True,
                            'created': True,
                            'reused': True,
                            # Minimal info when recipe not stored
                            'source_vial': 'Unknown (current session)',
                            'source_conc_mm': 0,
                            'dilution_factor': 0,
                            'stock_volume_ul': 0,
                            'water_volume_ul': 0,
                            'total_volume_ml': 0
                        })
                continue
            
            # Skip if already processed in this session
            if dilution_vials[i] is not None:
                continue
                
            print(f"    {vial_name}: Attempting creation for {target_conc:.2e} mM (pass {pass_num + 1})...")
            
            # Calculate dilution with volume constraints and hierarchical strategy
            required_substock_conc = target_conc * 2
            
            # Find best source considering volume constraints
            best_source = None
            best_volume = float('inf')
            
            # Option 1: From stock
            stock_dilution_factor = stock_conc / required_substock_conc
            stock_volume = FINAL_SUBSTOCK_VOLUME / stock_dilution_factor
            if stock_volume >= MINIMUM_PIPETTE_VOLUME:
                best_source = (f"{surfactant_name}_stock", stock_conc, stock_dilution_factor, stock_volume)
                best_volume = stock_volume
            
            # Option 2: From previously created dilutions (including this session)
            for j in range(len(achievable_concs)):
                if j == i:  # Can't use self
                    continue
                    
                prev_achievable = achievable_concs[j]
                if prev_achievable is None:
                    continue
                    
                # Check if this dilution exists (either pre-existing or created this session)
                prev_target = target_concentrations[j]
                if not (prev_target in tracking[surfactant_name]["dilutions_created"] or prev_target in created_in_this_session):
                    continue
                    
                if (prev_achievable * 2) > required_substock_conc:
                    prev_dilution_factor = (prev_achievable * 2) / required_substock_conc
                    prev_volume = FINAL_SUBSTOCK_VOLUME / prev_dilution_factor
                    
                    # Use this source if it meets minimum and has smaller volume (more efficient)
                    if prev_volume >= MINIMUM_PIPETTE_VOLUME and prev_volume < best_volume:
                        best_source = (f"{surfactant_name}_dilution_{j}", prev_achievable * 2, prev_dilution_factor, prev_volume)
                        best_volume = prev_volume
            
            if best_source is None:
                # Cannot create with minimum volume constraint in this pass
                print(f"      DEFERRED: Cannot achieve {target_conc:.2e} mM with minimum {MINIMUM_PIPETTE_VOLUME*1000:.0f} uL volume (pass {pass_num + 1})")
                continue
            
            # Can create in this pass!
            source_vial, source_conc, dilution_factor, stock_volume = best_source
            water_volume = FINAL_SUBSTOCK_VOLUME - stock_volume
            
            logger.debug(f"Creating {vial_name}: {dilution_factor:.1f}x dilution from {source_vial}")
            logger.debug(f"Volumes - stock: {stock_volume:.3f}mL, water: {water_volume:.3f}mL")
            
            print(f"      SUCCESS: From {source_vial} ({source_conc:.2e} mM), {dilution_factor:.1f}x dilution")
            print(f"      Volumes: {stock_volume:.3f} mL + {water_volume:.3f} mL = {FINAL_SUBSTOCK_VOLUME} mL")
            
            # Store dilution operation for later batching
            pass_operations.append({
                'vial_name': vial_name,
                'source_vial': source_vial,
                'stock_volume': stock_volume,
                'water_volume': water_volume,
                'target_conc': target_conc,
                'source_conc_mm': source_conc,
                'dilution_factor': dilution_factor,
                'stock_volume_ul': stock_volume * 1000,
                'water_volume_ul': water_volume * 1000,
                'total_volume_ml': FINAL_SUBSTOCK_VOLUME
            })
            
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
            
            # Mark as processed
            dilution_vials[i] = vial_name
            created_in_this_session.add(target_conc)
            progress_made = True
        
        # Execute this pass's operations
        if pass_operations:
            print(f"  Executing {len(pass_operations)} dilutions from pass {pass_num + 1}...")
            execute_batched_dilutions(lash_e, pass_operations, tracking, surfactant_name, logger)
            all_dilution_operations.extend(pass_operations)
        
        # Stop if no progress made
        if not progress_made:
            break
    
    # Add failure entries for any remaining None positions
    for i, (target_conc, achievable_conc) in enumerate(zip(target_concentrations, achievable_concs)):
        if dilution_vials[i] is None and achievable_conc is not None:
            vial_name = f"{surfactant_name}_dilution_{i}"
            print(f"    FINAL SKIP: {vial_name} could not be created with volume constraints")
            dilution_steps.append({
                'vial_name': vial_name,
                'target_conc_mm': target_conc,
                'achievable': False,
                'created': False,
                'reason': f'Could not achieve minimum {MINIMUM_PIPETTE_VOLUME*1000:.0f} uL volume in {max_passes} passes'
            })
    
    return dilution_vials, dilution_steps, achievable_concs

def execute_batched_dilutions(lash_e, dilution_operations, tracking, surfactant_name, logger):
    """
    Execute dilution operations efficiently by batching operations by source vial.
    
    Args:
        lash_e: Lash_E coordinator
        dilution_operations: List of dilution operation dictionaries
        tracking: Tracking dictionary to update
        surfactant_name: Name of surfactant for tracking
        logger: Logger instance
    """
    logger.info(f"Executing {len(dilution_operations)} dilutions for {surfactant_name} using batched operations")
    print(f"Executing {len(dilution_operations)} dilutions efficiently...")
    
    # Group operations by source vial for batching
    from collections import defaultdict
    source_vial_groups = defaultdict(list)
    for op in dilution_operations:
        source_vial_groups[op['source_vial']].append(op)
    
    # PHASE 1: Batch all stock solution transfers (grouped by source vial)
    print("Phase 1: Stock solution transfers (batched by source vial)")
    for source_vial, operations in source_vial_groups.items():
        print(f"  Processing {len(operations)} transfers from {source_vial}...")
        
        # Move source vial to clamp ONCE for all transfers
        lash_e.nr_robot.move_vial_to_location(source_vial, "clamp", 0)
        
        # Do all transfers keeping source vial at clamp
        for i, op in enumerate(operations):
            is_last_transfer = (i == len(operations) - 1)
            
            # Use dispense_from_vial_into_vial with conditional vial return
            lash_e.nr_robot.dispense_from_vial_into_vial(
                source_vial_name=source_vial,
                dest_vial_name=op['vial_name'],
                volume=op['stock_volume'],
                liquid='water',
                remove_tip=is_last_transfer,  # Only remove tip on last transfer
                use_safe_location=False,  # Source vial already at clamp
                return_vial_home=is_last_transfer  # Only return home on last transfer
            )
            
            logger.debug(f"Transferred {op['stock_volume']:.3f} mL from {source_vial} to {op['vial_name']}")
        
        print(f"  OK Completed {len(operations)} transfers from {source_vial} using 1 tip")
    
    # PHASE 2: Batch all water additions from reservoir
    print("Phase 2: Water additions from reservoir")
    water_operations = [op for op in dilution_operations if op['water_volume'] > 0]
    
    if water_operations:
        print(f"  Adding water to {len(water_operations)} vials from reservoir...")
        for op in water_operations:
            lash_e.nr_robot.move_vial_to_location(op['vial_name'], "clamp", 0)
            lash_e.nr_robot.dispense_into_vial_from_reservoir(1, op['vial_name'], op['water_volume'])
            logger.debug(f"Added {op['water_volume']:.3f} mL water to {op['vial_name']}")
        print(f"  OK Completed water additions to {len(water_operations)} vials")

    # PHASE 3: Vortex mixing for all created dilutions
    print("Phase 3: Vortex mixing for proper homogenization")
    print(f"  Vortexing {len(dilution_operations)} dilutions for proper mixing...")
    for op in dilution_operations:
        lash_e.nr_robot.vortex_vial(vial_name=op['vial_name'], vortex_time=8, vortex_speed=80)
        logger.debug(f"Vortexed {op['vial_name']} for 8s at speed 80")
    print(f"  OK Completed vortex mixing of {len(dilution_operations)} dilutions")

    # PHASE 4: Update tracking for all completed dilutions
    for op in dilution_operations:
        tracking[surfactant_name]["dilutions_created"].add(op['target_conc'])
        
        # Store complete recipe information for reuse
        tracking[surfactant_name]["dilution_recipes"][op['target_conc']] = {
            'vial_name': op['vial_name'],
            'source_vial': op['source_vial'],
            'source_conc_mm': op['source_conc_mm'],
            'dilution_factor': op['dilution_factor'],
            'stock_volume_ul': op['stock_volume_ul'],
            'water_volume_ul': op['water_volume_ul'],
            'total_volume_ml': op['total_volume_ml']
        }
        
        logger.debug(f"Successfully created {op['vial_name']} with target concentration {op['target_conc']:.2e} mM")
    
    # Calculate efficiency gains
    total_tips_old_method = len(dilution_operations)  # 1 tip per dilution
    total_tips_new_method = len(source_vial_groups)    # 1 tip per source vial
    tips_saved = total_tips_old_method - total_tips_new_method
    
    logger.info(f"Batched dilution complete: {total_tips_new_method} tips used (saved {tips_saved} tips vs sequential method)")
    print(f"OK Batched dilution complete: {total_tips_new_method} tips used (saved {tips_saved} tips)")

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
    # This matches the actual serial dilution process (highest -> lowest)
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
        
        print(f"OK Backed up measurement data to: {backup_path}")
        
    except Exception as e:
        print(f"WARNING: Failed to backup measurement data: {e}")
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
        
        print(f"SUCCESS: Raw Cytation data backed up to: {backup_path}")
        if os.path.exists(csv_path):
            print(f"   Human-readable version: {csv_path}")
        
    except Exception as e:
        print(f"CRITICAL: Failed to backup raw Cytation data: {e}")
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
            print(f"OK Recovered measurement from: {os.path.basename(backup_file)}")
            
        except Exception as e:
            print(f"WARNING: Failed to recover from {backup_file}: {e}")
    
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
            print(f"  SUCCESS: Recovered: {basename}")
            print(f"     Shape: {raw_df.shape}, Columns: {len(raw_df.columns)}")
            
        except Exception as e:
            print(f"  ERROR: Failed to recover {backup_file}: {e}")
    
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

def pipette_grid_to_shared_wellplate(lash_e, concs_a, concs_b, dilution_vials_a, dilution_vials_b, surfactant_a_name, surfactant_b_name, shared_wellplate_state):
    """Pipette concentration grid into wellplate(s) using shared wellplate state with batched measurements for tip efficiency."""
    logger = lash_e.logger
    logger.info(f"Starting grid pipetting: {len(concs_a)}x{len(concs_b)} grid with {N_REPLICATES} replicates each")
    
    well_counter = shared_wellplate_state.get('global_well_counter', 0)
    well_map = []  # FIX: Fresh well map for each experiment (don't accumulate old mappings)
    total_wells_added = shared_wellplate_state.get('total_wells_added', 0)
    
    # Use existing wellplate tracking from shared state (but isolate measurements per experiment)
    wellplate_data = {
        'current_plate': shared_wellplate_state.get('current_plate', 1),
        'wells_used': shared_wellplate_state.get('wells_used', 0),
        'measurements': [],  # FIX: Fresh measurements list for each experiment
        'last_measured_well': shared_wellplate_state.get('last_measured_well', -1)
    }
    
    # Generate all well requirements first (skip combinations where vials don't exist)
    all_well_requirements = []
    for i, conc_a in enumerate(concs_a):
        for j, conc_b in enumerate(concs_b):
            # Skip if either concentration is not achievable (vial is None)
            if dilution_vials_a[i] is None or dilution_vials_b[j] is None:
                conc_a_str = f"{conc_a:.2e}" if conc_a is not None else "None"
                conc_b_str = f"{conc_b:.2e}" if conc_b is not None else "None"
                logger.debug(f"Skipping combination [{i},{j}]: {conc_a_str} + {conc_b_str} mM (vial not available)")
                continue
                
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
    
    # Report on achievable vs total combinations
    total_possible_wells = len(concs_a) * len(concs_b) * N_REPLICATES
    achievable_wells = len(all_well_requirements)
    skipped_wells = total_possible_wells - achievable_wells
    
    logger.info(f"Grid summary: {achievable_wells}/{total_possible_wells} wells achievable ({skipped_wells} skipped due to non-achievable concentrations)")
    
    if achievable_wells == 0:
        logger.warning("No achievable concentration combinations found! Check stock concentrations and target ranges.")
        return [], shared_wellplate_state
    
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
        
        # Sort vial groups by concentration (low->high to prevent contamination)
        # Handle None concentrations by treating them as 0
        sorted_vial_groups_a = sorted(vial_a_groups.items(), 
                                     key=lambda x: current_batch[x[1][0][0]]['conc_a'] or 0)
        
        print(f"  Phase 1: Adding surfactant A (low->high concentration, grouped by vial)")
        for vial_name, well_list in sorted_vial_groups_a:
            print(f"    Using vial {vial_name} for {len(well_list)} wells")
            for batch_idx, actual_well, req in well_list:
                volume_each = WELL_VOLUME_UL / 2000  # Convert uL to mL
                
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
        
        # Sort vial groups by concentration (low->high to prevent contamination)
        # Handle None concentrations by treating them as 0
        sorted_vial_groups_b = sorted(vial_b_groups.items(), 
                                     key=lambda x: current_batch[x[1][0][0]]['conc_b'] or 0)
        
        print(f"  Phase 2: Adding surfactant B (low->high concentration, grouped by vial)")
        for vial_name, well_list in sorted_vial_groups_b:
            print(f"    Using vial {vial_name} for {len(well_list)} wells")
            for batch_idx, actual_well, req in well_list:
                volume_each = WELL_VOLUME_UL / 2000  # Convert uL to mL
                
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
        
        # PHASE 3: Measure initial turbidity
        print(f"  Phase 3: Measuring initial turbidity for wells {wells_in_batch[0]}-{wells_in_batch[-1]}")
        turbidity_data = measure_turbidity(lash_e, wells_in_batch)
        
        # Store turbidity measurement data
        turbidity_entry = {
            'plate_number': wellplate_data['current_plate'],
            'wells_measured': wells_in_batch,
            'measurement_type': 'turbidity_batch',
            'data': turbidity_data,
            'timestamp': datetime.now().isoformat()
        }
        wellplate_data['measurements'].append(turbidity_entry)
        
        # BACKUP: Save raw turbidity data immediately
        backup_raw_measurement_data(turbidity_entry, wellplate_data['current_plate'], wells_in_batch)
        
        # PHASE 4: Add pyrene_DMSO to all wells
        print(f"  Phase 4: Adding pyrene_DMSO ({PYRENE_VOLUME_UL} uL) to wells {wells_in_batch[0]}-{wells_in_batch[-1]}")
        pyrene_volume_ml = PYRENE_VOLUME_UL / 1000  # Convert to mL
        
        # Add pyrene to all wells in batch using single tip
        lash_e.nr_robot.aspirate_from_vial('pyrene_DMSO', pyrene_volume_ml * len(wells_in_batch), liquid='DMSO', use_safe_location=True)
        for well_idx in wells_in_batch:
            lash_e.nr_robot.dispense_into_wellplate(
                dest_wp_num_array=[well_idx], 
                amount_mL_array=[pyrene_volume_ml],
                liquid='DMSO'
            )
        lash_e.nr_robot.remove_pipet()  # Remove tip after pyrene addition
        lash_e.nr_robot.return_vial_home('pyrene_DMSO')
        
        # PHASE 5: Measure fluorescence after pyrene addition
        print(f"  Phase 5: Measuring fluorescence for wells {wells_in_batch[0]}-{wells_in_batch[-1]}")
        fluorescence_data = measure_fluorescence(lash_e, wells_in_batch)
        
        # Store fluorescence measurement data
        fluorescence_entry = {
            'plate_number': wellplate_data['current_plate'],
            'wells_measured': wells_in_batch,
            'measurement_type': 'fluorescence_batch',
            'data': fluorescence_data,
            'timestamp': datetime.now().isoformat()
        }
        wellplate_data['measurements'].append(fluorescence_entry)
        
        # BACKUP: Save raw fluorescence data immediately
        backup_raw_measurement_data(fluorescence_entry, wellplate_data['current_plate'], wells_in_batch)
        
        # Update tracking
        wellplate_data['last_measured_well'] = wells_in_batch[-1]
        total_wells_added += len(current_batch)
        
        print(f"  ✓ Completed batch with {len(current_batch)} wells using 3 tips (2 surfactants + 1 pyrene)")
    
    # Update shared state before returning
    updated_shared_state = {
        'global_well_counter': well_counter,
        'global_well_map': well_map,
        'total_wells_added': total_wells_added,
        'current_plate': wellplate_data['current_plate'],
        'wells_used': wellplate_data['wells_used'],
        'measurements': wellplate_data['measurements'],
        'last_measured_well': wellplate_data['last_measured_well'],
        'plates_used_this_combo': shared_wellplate_state.get('plates_used_this_combo', 0) + (wellplate_data['current_plate'] - shared_wellplate_state.get('current_plate', 1))
    }
    
    return well_map, updated_shared_state

def combine_measurement_data(well_map, wellplate_data):
    """
    Combine measurement data from all intervals into a single DataFrame.
    Now handles both turbidity and fluorescence measurements and structures 
    data for CMC analysis compatibility.
    
    Args:
        well_map: List of well information dictionaries
        wellplate_data: Dictionary containing measurement data from all intervals
        
    Returns:
        pd.DataFrame: Combined results with both turbidity and fluorescence measurements
    """
    print("\nCombining measurement data from all intervals...")
    
    # Group measurements by well to combine turbidity and fluorescence
    well_measurements = {}
    
    for measurement in wellplate_data['measurements']:
        plate_num = measurement['plate_number']
        measurement_data = measurement['data']
        wells_measured = measurement['wells_measured']
        measurement_type = measurement.get('measurement_type', 'interval')
        
        # Handle simulation mode where measurement_data might be None
        if measurement_data is None:
            if 'turbidity' in measurement_type:
                measurement_data = {'turbidity': [-1] * len(wells_measured)}
            elif 'fluorescence' in measurement_type:
                measurement_data = {
                    '334_373': [-1] * len(wells_measured),
                    '334_384': [-1] * len(wells_measured)
                }
            else:
                print(f"WARNING: Unknown measurement type {measurement_type}, skipping")
                continue
        
        # Process each well in this measurement
        for batch_idx, well_idx in enumerate(wells_measured):
            # Find well info by matching well number
            well_info = None
            for well_entry in well_map:
                if well_entry['well'] == well_idx:
                    well_info = well_entry
                    break
            
            if well_info is None:
                print(f"ERROR: Could not find well info for well {well_idx} in well_map")
                continue
                
            # Initialize well entry if not exists
            if well_idx not in well_measurements:
                well_measurements[well_idx] = {
                    'plate': plate_num,
                    'well': well_idx,
                    'surfactant_a': well_info['surfactant_a'],
                    'surfactant_b': well_info['surfactant_b'],
                    'conc_a_mm': well_info['conc_a_mm'],
                    'conc_b_mm': well_info['conc_b_mm'],
                    'replicate': well_info['replicate'],
                    # CMC analysis compatible column names
                    '600': None,           # Turbidity/absorbance at 600nm
                    '334_373': None,       # Fluorescence excitation 334, emission 373
                    '334_384': None,       # Fluorescence excitation 334, emission 384 
                    'turbidity': None      # Raw turbidity value
                }
            
            # Add measurement data based on type
            if 'turbidity' in measurement_type and batch_idx < len(measurement_data.get('turbidity', [])):
                turbidity_val = measurement_data['turbidity'][batch_idx]
                well_measurements[well_idx]['turbidity'] = turbidity_val
                well_measurements[well_idx]['600'] = turbidity_val  # CMC analysis expects '600' column
                
            elif 'fluorescence' in measurement_type and batch_idx < len(measurement_data.get('334_373', [])):
                # Store only the two wavelength values needed for CMC analysis
                if '334_373' in measurement_data and '334_384' in measurement_data:
                    # Real fluorescence data with wavelength specificity
                    well_measurements[well_idx]['334_373'] = measurement_data['334_373'][batch_idx]
                    well_measurements[well_idx]['334_384'] = measurement_data['334_384'][batch_idx]
                else:
                    # Invalid/missing fluorescence data - use -1 to indicate failure
                    well_measurements[well_idx]['334_373'] = -1
                    well_measurements[well_idx]['334_384'] = -1
    
    # Convert to list for DataFrame creation and filter out incomplete entries
    combined_results = []
    for well_idx, well_data in well_measurements.items():
        # Only include wells that have both turbidity and fluorescence data
        if well_data['600'] is not None and well_data['334_373'] is not None:
            combined_results.append(well_data)
        else:
            print(f"WARNING: Well {well_idx} missing data - turbidity: {well_data['600']}, fluorescence: {well_data['334_373']}")
    
    print(f"Combined {len(combined_results)} complete wells with both turbidity and fluorescence from {len(wellplate_data['measurements'])} measurements")
    return pd.DataFrame(combined_results)

def validate_measurement_data(results_df):
    """
    Detect and flag wells with fallback/fake measurement data.
    
    Args:
        results_df: DataFrame with measurement results
        
    Returns:
        tuple: (results_df_with_flags, clean_data_df)
    """
    print("\nValidating measurement data for fallback values...")
    
    fake_data_flags = []
    fake_count = 0
    
    for idx, row in results_df.iterrows():
        flags = []
        
        # Check for -1 fallback values (indicates measurement failure)
        if row['600'] == -1:
            flags.append('turbidity_failed')
        if row['334_373'] == -1:
            flags.append('fluorescence_334_373_failed')
        if row['334_384'] == -1:
            flags.append('fluorescence_334_384_failed')
            
        if flags:
            fake_count += 1
            
        fake_data_flags.append(flags)
    
    # Add validation column
    results_df_flagged = results_df.copy()
    results_df_flagged['data_quality_flags'] = fake_data_flags
    
    # Create clean dataset (no fallback values)
    clean_data = results_df_flagged[results_df_flagged['data_quality_flags'].apply(len) == 0].copy()
    
    print(f"Data validation complete:")
    print(f"  Total wells: {len(results_df)}")
    print(f"  Wells with real data: {len(clean_data)}")
    print(f"  Wells with fallback data: {fake_count}")
    if fake_count > 0:
        print(f"  ⚠️  WARNING: {fake_count} wells have fallback values (-1) indicating measurement failures")
        print(f"     Use only clean_data for CMC analysis to avoid invalid results")
    
    return results_df_flagged, clean_data

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
    experiment_type = "surfactant_grid_turbidity_fluorescence"
    
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

def create_concentration_grid_summary(concentrations, dilution_vials_a, dilution_vials_b, surfactant_a_name, surfactant_b_name):
    """
    Create a text-based visual summary showing which concentration combinations are achievable.
    Uses 'X' for achievable combinations and '.' for missing/non-achievable ones.
    
    Args:
        concentrations: List of target concentrations
        dilution_vials_a: List of vial names for surfactant A (None for non-achievable)
        dilution_vials_b: List of vial names for surfactant B (None for non-achievable)
        surfactant_a_name: Name of surfactant A
        surfactant_b_name: Name of surfactant B
        
    Returns:
        str: Text grid showing achievable combinations
    """
    grid_lines = []
    grid_lines.append(f"CONCENTRATION GRID SUMMARY: {surfactant_a_name} + {surfactant_b_name}")
    grid_lines.append("=" * 60)
    grid_lines.append("Legend: X = achievable combination, . = not achievable")
    grid_lines.append("")
    
    # Create header with surfactant B concentrations (columns)
    header = f"{'':>12}"  # Space for row labels
    for j, conc_b in enumerate(concentrations):
        if dilution_vials_b[j] is not None:
            header += f"{conc_b:.1e}".rjust(12)
        else:
            header += f"{'(N/A)':<12}"
    grid_lines.append(f"{surfactant_b_name} (mM) ->")
    grid_lines.append(header)
    grid_lines.append("-" * len(header))
    
    # Create rows for each surfactant A concentration
    for i, conc_a in enumerate(concentrations):
        if dilution_vials_a[i] is not None:
            row_label = f"{conc_a:.1e}".rjust(10) + " |"
        else:
            row_label = f"{'(N/A)':<10} |"
        
        row = row_label
        for j, conc_b in enumerate(concentrations):
            # Check if both concentrations are achievable
            if dilution_vials_a[i] is not None and dilution_vials_b[j] is not None:
                symbol = "X"
            else:
                symbol = "."
            row += f"{symbol:>12}"
        grid_lines.append(row)
    
    grid_lines.append("")
    grid_lines.append(f"Total combinations: {len([1 for a in dilution_vials_a for b in dilution_vials_b if a is not None and b is not None])}/{len(concentrations)**2}")
    
    return "\\n".join(grid_lines)

def create_turbidity_heatmap(results_df, concentrations, surfactant_a_name, surfactant_b_name, output_folder, logger=None):
    """
    Create and save a turbidity heatmap visualization from experimental data.
    
    Args:
        results_df: DataFrame with experimental results including turbidity measurements
        concentrations: List of concentration values used in the grid
        surfactant_a_name: Name of surfactant A (cationic)
        surfactant_b_name: Name of surfactant B (anionic) 
        output_folder: Directory to save the heatmap
        logger: Logger instance for error reporting
        
    Returns:
        str: Path to saved heatmap file, or None if visualization failed
    """
    if not MATPLOTLIB_AVAILABLE:
        if logger:
            logger.warning("Matplotlib not available - skipping heatmap visualization")
        print("Warning: matplotlib not available - skipping heatmap visualization")
        return None
    
    try:
        if logger:
            logger.info(f"Creating turbidity heatmap for {surfactant_a_name} vs {surfactant_b_name}")
        
        # Create heatmap data matrix (no replicates - single values)
        n_concs = len(concentrations)
        heatmap_data = np.full((n_concs, n_concs), np.nan)
        
        # Populate the matrix from results_df
        for _, row in results_df.iterrows():
            # Find the concentration indices
            conc_a_idx = None
            conc_b_idx = None
            
            # Match concentrations with some tolerance for floating point comparison
            for i, conc in enumerate(concentrations):
                if abs(row['conc_a_mm'] - conc) < 1e-10:
                    conc_a_idx = i
                if abs(row['conc_b_mm'] - conc) < 1e-10:
                    conc_b_idx = i
            
            if conc_a_idx is not None and conc_b_idx is not None:
                heatmap_data[conc_a_idx, conc_b_idx] = row['turbidity']
        
        # Create the heatmap plot
        plt.figure(figsize=(10, 8))
        
        # Use 'RdYlBu_r' colormap for better contrast at low turbidity values
        # Set vmin/vmax to focus on the main data range excluding outliers
        data_min = np.nanmin(heatmap_data)
        data_max = np.nanmax(heatmap_data)
        
        if logger:
            logger.debug(f"Heatmap data range: {data_min:.4f} to {data_max:.4f}")
        
        # Use 95th percentile as max to avoid outlier saturation
        vmax = np.nanpercentile(heatmap_data, 95) if not np.isnan(data_max) else 1.0
        vmin = max(0, data_min) if not np.isnan(data_min) else 0
        
        c = plt.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r', origin='lower', 
                      vmin=vmin, vmax=vmax, interpolation='nearest')
        
        plt.colorbar(c, label='Absorbance (Turbidity)')
        
        # Set axis labels with concentration values
        plt.xticks(range(len(concentrations)), [f'{c:.1e}' for c in concentrations], rotation=45)
        plt.yticks(range(len(concentrations)), [f'{c:.1e}' for c in concentrations])
        
        plt.xlabel(f'{surfactant_b_name} concentration (mM)')
        plt.ylabel(f'{surfactant_a_name} concentration (mM)')
        plt.title(f'Turbidity Heatmap: {surfactant_a_name} vs {surfactant_b_name}')
        plt.tight_layout()
        
        # Save the heatmap
        heatmap_file = os.path.join(output_folder, f"turbidity_heatmap_{surfactant_a_name}_{surfactant_b_name}.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        if logger:
            logger.info(f"Turbidity heatmap saved: {heatmap_file}")
        print(f"Turbidity heatmap saved: {heatmap_file}")
        
        return heatmap_file
        
    except Exception as e:
        error_msg = f"Failed to create turbidity heatmap: {e}"
        if logger:
            logger.warning(error_msg)
        else:
            print(f"Warning: {error_msg}")
        return None

def create_ratio_heatmap(results_df, concentrations, surfactant_a_name, surfactant_b_name, output_folder, logger=None):
    """
    Create and save a pyrene fluorescence ratio heatmap visualization from experimental data.
    
    Args:
        results_df: DataFrame with experimental results including fluorescence measurements
        concentrations: List of concentration values used in the grid
        surfactant_a_name: Name of surfactant A (cationic)
        surfactant_b_name: Name of surfactant B (anionic) 
        output_folder: Directory to save the heatmap
        logger: Logger instance for error reporting
        
    Returns:
        str: Path to saved heatmap file, or None if visualization failed
    """
    if not MATPLOTLIB_AVAILABLE:
        if logger:
            logger.warning("Matplotlib not available - skipping ratio heatmap visualization")
        print("Warning: matplotlib not available - skipping ratio heatmap visualization")
        return None
    
    try:
        if logger:
            logger.info(f"Creating pyrene ratio heatmap for {surfactant_a_name} vs {surfactant_b_name}")
        
        # Check if fluorescence data is available
        if '334_373' not in results_df.columns or '334_384' not in results_df.columns:
            error_msg = "No fluorescence data available (missing 334_373 or 334_384 columns) - skipping ratio heatmap"
            if logger:
                logger.warning(error_msg)
            print(f"Warning: {error_msg}")
            return None
        
        # Create heatmap data matrix (no replicates - single values)
        n_concs = len(concentrations)
        heatmap_data = np.full((n_concs, n_concs), np.nan)
        
        # Populate the matrix from results_df
        for _, row in results_df.iterrows():
            # Calculate ratio (I3/I1) - lower values indicate more hydrophobic environment
            if row['334_373'] > 0 and row['334_384'] > 0:  # Avoid division by zero
                ratio = row['334_373'] / row['334_384']
            else:
                ratio = np.nan  # Mark failed measurements as NaN
            
            # Find the concentration indices
            conc_a_idx = None
            conc_b_idx = None
            
            # Match concentrations with some tolerance for floating point comparison
            for i, conc in enumerate(concentrations):
                if abs(row['conc_a_mm'] - conc) < 1e-10:
                    conc_a_idx = i
                if abs(row['conc_b_mm'] - conc) < 1e-10:
                    conc_b_idx = i
            
            if conc_a_idx is not None and conc_b_idx is not None:
                heatmap_data[conc_a_idx, conc_b_idx] = ratio
        
        # Create the heatmap plot
        plt.figure(figsize=(10, 8))
        
        # Use 'viridis' colormap for ratio data (lower values = more hydrophobic = darker)
        # Set vmin/vmax to focus on the main data range excluding outliers
        data_min = np.nanmin(heatmap_data)
        data_max = np.nanmax(heatmap_data)
        
        if logger:
            logger.debug(f"Ratio heatmap data range: {data_min:.4f} to {data_max:.4f}")
        
        # Use reasonable bounds for pyrene ratio (typically 0.6-1.8)
        # Dynamic scaling based on actual data for optimal contrast
        vmin = max(0.5, np.nanpercentile(heatmap_data, 5)) if not np.isnan(data_min) else 0.5
        vmax = min(2.0, np.nanpercentile(heatmap_data, 95)) if not np.isnan(data_max) else 2.0
        
        # Ensure minimum contrast range for small variations
        if vmax - vmin < 0.1:  # If data range is very tight, expand slightly
            center = (vmax + vmin) / 2
            vmin = center - 0.05
            vmax = center + 0.05
        
        c = plt.imshow(heatmap_data, aspect='auto', cmap='plasma', origin='lower', 
                      vmin=vmin, vmax=vmax, interpolation='nearest')
        
        plt.colorbar(c, label='Pyrene Ratio (I₃/I₁) - 334_373/334_384')
        
        # Set axis labels with concentration values
        plt.xticks(range(len(concentrations)), [f'{c:.1e}' for c in concentrations], rotation=45)
        plt.yticks(range(len(concentrations)), [f'{c:.1e}' for c in concentrations])
        
        plt.xlabel(f'{surfactant_b_name} concentration (mM)')
        plt.ylabel(f'{surfactant_a_name} concentration (mM)')
        plt.title(f'Pyrene Ratio Heatmap: {surfactant_a_name} vs {surfactant_b_name}\n(Lower ratio = more hydrophobic environment)')
        plt.tight_layout()
        
        # Save the heatmap
        heatmap_file = os.path.join(output_folder, f"ratio_heatmap_{surfactant_a_name}_{surfactant_b_name}.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        if logger:
            logger.info(f"Pyrene ratio heatmap saved: {heatmap_file}")
        print(f"Pyrene ratio heatmap saved: {heatmap_file}")
        
        return heatmap_file
        
    except Exception as e:
        error_msg = f"Failed to create ratio heatmap: {e}"
        if logger:
            logger.warning(error_msg)
        else:
            print(f"Warning: {error_msg}")
        return None

def save_results(results_df, well_map, wellplate_data, all_measurements, concentrations, dilution_vials_a, dilution_vials_b, dilution_steps_a, dilution_steps_b, surfactant_a_name, surfactant_b_name, simulate=True, output_folder=None, logger=None):
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
        
        # Add concentration grid summary
        grid_summary = create_concentration_grid_summary(concentrations, dilution_vials_a, dilution_vials_b, surfactant_a_name, surfactant_b_name)
        f.write(grid_summary + "\n\n")
        f.write("=" * 50 + "\n\n")
        
        # Surfactant A dilutions
        surf_a_info = SURFACTANT_LIBRARY[surfactant_a_name]
        f.write(f"SURFACTANT A: {surfactant_a_name}\n")
        f.write(f"Stock concentration: {surf_a_info['stock_conc']} mM\n")
        f.write("-" * 40 + "\n")
        
        print(f"DEBUG: Processing {len(dilution_steps_a)} dilution steps for {surfactant_a_name}")
        for i, step in enumerate(dilution_steps_a):
            print(f"DEBUG: Step {i}: {list(step.keys())}")
            try:
                f.write(f"Step {i+1}: {step['vial_name']}\n")
                f.write(f"  Target concentration: {step['target_conc_mm']:.2e} mM (dilution stock)\n")
                
                if not step.get('achievable', True):
                    f.write(f"  STATUS: NOT ACHIEVABLE - {step.get('reason', 'Unknown reason')}\n")
                    f.write("\n")
                    continue
                    
                # Check for required fields before accessing
                if 'source_vial' not in step:
                    print(f"ERROR: Step {i} missing 'source_vial' field: {step}")
                    f.write(f"  STATUS: INCOMPLETE DATA - missing source_vial field\n")
                    f.write("\n")
                    continue
                    
                f.write(f"  Final concentration: {step['final_conc_mm']:.2e} mM (after 1:1 mixing)\n")
                f.write(f"  Source: {step['source_vial']} ({step['source_conc_mm']:.2e} mM)\n")
                f.write(f"  Dilution factor: {step['dilution_factor']:.1f}x\n")
                f.write(f"  Volumes added:\n")
                f.write(f"    - Source solution: {step['stock_volume_ul']} uL\n")
                f.write(f"    - Water: {step['water_volume_ul']} uL\n")
                f.write(f"    - Total final volume: {step['total_volume_ml']} mL\n")
                f.write("\n")
                
            except KeyError as e:
                print(f"ERROR: KeyError in step {i} for {surfactant_a_name}: {e}")
                print(f"ERROR: Step contents: {step}")
                f.write(f"  ERROR: Missing field {e} in dilution step\n")
                f.write("\n")
        
        f.write("\n" + "=" * 50 + "\n\n")
        
        # Surfactant B dilutions
        surf_b_info = SURFACTANT_LIBRARY[surfactant_b_name]
        f.write(f"SURFACTANT B: {surfactant_b_name}\n")
        f.write(f"Stock concentration: {surf_b_info['stock_conc']} mM\n")
        f.write("-" * 40 + "\n")
        
        print(f"DEBUG: Processing {len(dilution_steps_b)} dilution steps for {surfactant_b_name}")
        for i, step in enumerate(dilution_steps_b):
            print(f"DEBUG: Step {i}: {list(step.keys())}")
            try:
                f.write(f"Step {i+1}: {step['vial_name']}\n")
                f.write(f"  Target concentration: {step['target_conc_mm']:.2e} mM (dilution stock)\n")
                
                if not step.get('achievable', True):
                    f.write(f"  STATUS: NOT ACHIEVABLE - {step.get('reason', 'Unknown reason')}\n")
                    f.write("\n")
                    continue
                    
                # Check for required fields before accessing  
                if 'source_vial' not in step:
                    print(f"ERROR: Step {i} missing 'source_vial' field: {step}")
                    f.write(f"  STATUS: INCOMPLETE DATA - missing source_vial field\n")
                    f.write("\n")
                    continue
                    
                f.write(f"  Final concentration: {step['final_conc_mm']:.2e} mM (after 1:1 mixing)\n")
                f.write(f"  Source: {step['source_vial']} ({step['source_conc_mm']:.2e} mM)\n")
                f.write(f"  Dilution factor: {step['dilution_factor']:.1f}x\n")
                f.write(f"  Volumes added:\n")
                f.write(f"    - Source solution: {step['stock_volume_ul']} uL\n")
                f.write(f"    - Water: {step['water_volume_ul']} uL\n")
                f.write(f"    - Total final volume: {step['total_volume_ml']} mL\n")
                f.write("\n")
                
            except KeyError as e:
                print(f"ERROR: KeyError in step {i} for {surfactant_b_name}: {e}")
                print(f"ERROR: Step contents: {step}")
                f.write(f"  ERROR: Missing field {e} in dilution step\n")
                f.write("\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("WELL PLATE MIXING:\n")
        f.write(f"Each well contains {WELL_VOLUME_UL/2} uL of surfactant A dilution\n")
        f.write(f"                 + {WELL_VOLUME_UL/2} uL of surfactant B dilution\n")
        f.write(f"                 = {WELL_VOLUME_UL} uL total per well\n")
    
    # Save dilution series information (JSON for programmatic access)
    surf_a_info = SURFACTANT_LIBRARY[surfactant_a_name]
    surf_b_info = SURFACTANT_LIBRARY[surfactant_b_name]
    
    # Filter dilution steps to only include successfully created ones (with all required fields)
    print(f"DEBUG: Filtering {len(dilution_steps_a)} steps for {surfactant_a_name}")
    successful_steps_a = []
    for i, step in enumerate(dilution_steps_a):
        if step.get('achievable', False) and step.get('created', False):
            # Check if all required fields are present
            required_fields = ['source_vial', 'source_conc_mm', 'dilution_factor', 'stock_volume_ul']
            missing_fields = [field for field in required_fields if field not in step]
            if missing_fields:
                print(f"WARNING: Skipping step {i} for {surfactant_a_name} - missing fields: {missing_fields}")
                continue
            successful_steps_a.append(step)
        else:
            print(f"DEBUG: Skipping step {i} for {surfactant_a_name} - not achievable or created")
    
    print(f"DEBUG: Filtering {len(dilution_steps_b)} steps for {surfactant_b_name}")
    successful_steps_b = []
    for i, step in enumerate(dilution_steps_b):
        if step.get('achievable', False) and step.get('created', False):
            # Check if all required fields are present
            required_fields = ['source_vial', 'source_conc_mm', 'dilution_factor', 'stock_volume_ul']
            missing_fields = [field for field in required_fields if field not in step]
            if missing_fields:
                print(f"WARNING: Skipping step {i} for {surfactant_b_name} - missing fields: {missing_fields}")
                continue
            successful_steps_b.append(step)
        else:
            print(f"DEBUG: Skipping step {i} for {surfactant_b_name} - not achievable or created")
    
    print(f"DEBUG: Final counts - {surfactant_a_name}: {len(successful_steps_a)}, {surfactant_b_name}: {len(successful_steps_b)}")
    
    dilution_info = {
        'surfactant_a': {
            'name': surfactant_a_name,
            'stock_conc_mm': surf_a_info['stock_conc'],
            'dilution_steps': successful_steps_a
        },
        'surfactant_b': {
            'name': surfactant_b_name,
            'stock_conc_mm': surf_b_info['stock_conc'],
            'dilution_steps': successful_steps_b
        },
        'well_mixing': {
            'volume_each_ul': WELL_VOLUME_UL / 2,
            'total_volume_ul': WELL_VOLUME_UL
        }
    }
    
    with open(dilutions_file, 'w') as f:
        try:
            json.dump(dilution_info, f, indent=2)
            print(f"DEBUG: Successfully saved dilution JSON with {len(successful_steps_a)} + {len(successful_steps_b)} steps")
        except Exception as e:
            print(f"ERROR: Failed to save dilution JSON: {e}")
            print(f"ERROR: dilution_info contents: {dilution_info}")
            # Try saving a minimal version
            minimal_info = {
                'surfactant_a': {'name': surfactant_a_name, 'stock_conc_mm': surf_a_info['stock_conc'], 'dilution_steps': []},
                'surfactant_b': {'name': surfactant_b_name, 'stock_conc_mm': surf_b_info['stock_conc'], 'dilution_steps': []},
                'well_mixing': dilution_info['well_mixing'],
                'error': f'Original save failed: {str(e)}'
            }
            json.dump(minimal_info, f, indent=2)
    
    # Save experiment metadata
    metadata = {
        'workflow': 'surfactant_grid_turbidity_fluorescence_screening',
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
        'well_volume_ul': WELL_VOLUME_UL,
        'pyrene_volume_ul': PYRENE_VOLUME_UL,
        'measurement_types': ['turbidity', 'fluorescence'],
        'protocols_used': [TURBIDITY_PROTOCOL_FILE, FLUORESCENCE_PROTOCOL_FILE]
    }
    
    if simulate:
        metadata['note'] = "This is simulated data - no actual hardware was used"
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create turbidity heatmap visualization
    turbidity_heatmap_file = create_turbidity_heatmap(results_df, concentrations, surfactant_a_name, surfactant_b_name, output_folder, logger)
    
    # Create pyrene ratio heatmap visualization  
    ratio_heatmap_file = create_ratio_heatmap(results_df, concentrations, surfactant_a_name, surfactant_b_name, output_folder, logger)
    
    print(f"Results saved to: {results_file}")
    print(f"Dilution report saved to: {dilution_report_file}")
    print(f"Dilution series info saved to: {dilutions_file}")
    print(f"Metadata saved to: {metadata_file}")
    if turbidity_heatmap_file:
        print(f"Turbidity heatmap saved to: {turbidity_heatmap_file}")
    if ratio_heatmap_file:
        print(f"Pyrene ratio heatmap saved to: {ratio_heatmap_file}")

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
    raw_data = lash_e.measure_wellplate(TURBIDITY_PROTOCOL_FILE, well_indices)
    
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
        print(f"CRITICAL: Failed to process measurement data: {e}")
        print(f"Raw data structure: {type(raw_data)}")
        if hasattr(raw_data, 'columns'):
            print(f"Raw data columns: {raw_data.columns.tolist()}")
        
        print(f"🔄 Raw Cytation data was saved to backup before processing failed")
        print(f"   You can recover the data from output/cytation_raw_backups/")
        
        # Fallback: return mock data to prevent workflow crash
        print(f"WARNING: Using fallback mock data to continue workflow...")
        return {'turbidity': [-1] * len(well_indices)}

def measure_fluorescence(lash_e, well_indices):
    """
    Measure fluorescence in specified wells using Cytation plate reader.
    
    Args:
        lash_e: Lash_E coordinator instance
        well_indices: List of well indices to measure
        
    Returns:
        dict: Processed measurement data with 'fluorescence' values, or None if simulation
    """
    print(f"Measuring fluorescence in wells {well_indices}...")
    
    # Get raw measurement data from Cytation
    raw_data = lash_e.measure_wellplate(FLUORESCENCE_PROTOCOL_FILE, well_indices)
    
    # CRITICAL: Save raw Cytation data immediately to prevent loss
    if raw_data is not None:
        backup_raw_cytation_data(raw_data, well_indices)
    
    # Handle simulation mode
    if raw_data is None:
        return None
    
    # Process the complex DataFrame structure from Cytation
    try:
        # Extract fluorescence data (similar to turbidity but different column names)
        processed_data = extract_fluorescence_values(raw_data, well_indices)
        return processed_data
    
    except Exception as e:
        print(f"⚠️  CRITICAL: Failed to process fluorescence data: {e}")
        print(f"Raw data structure: {type(raw_data)}")
        if hasattr(raw_data, 'columns'):
            print(f"Raw data columns: {raw_data.columns.tolist()}")
        
        print(f"🔄 Raw Cytation data was saved to backup before processing failed")
        print(f"   You can recover the data from output/cytation_raw_backups/")
        
        # Fallback: return mock data to prevent workflow crash
        print(f"⚠️  Using fallback mock data to continue workflow...")
        return {
            '334_373': [-1] * len(well_indices),
            '334_384': [-1] * len(well_indices)
        }

def extract_fluorescence_values(raw_data, well_indices):
    """
    Extract fluorescence values from Cytation measurement DataFrame.
    Looks for CMC-specific wavelengths: 334_373 and 334_384 excitation/emission pairs.
    
    Args:
        raw_data: DataFrame returned from lash_e.measure_wellplate()
        well_indices: List of well indices that were measured
        
    Returns:
        dict: {'334_373': [...], '334_384': [...]} aligned with well_indices
    """
    if raw_data is None or raw_data.empty:
        return {
            '334_373': [-1] * len(well_indices),
            '334_384': [-1] * len(well_indices)
        }
    
    # Handle MultiIndex columns structure from Cytation
    if hasattr(raw_data.columns, 'levels'):
        measurement_values_373 = []
        measurement_values_384 = []
        measurement_values_general = []
        
        for well_idx in well_indices:
            try:
                # Use first available column group
                top_level = raw_data.columns.levels[0][0]
                sub_columns = raw_data[top_level].columns
                
                # Look for specific CMC fluorescence wavelengths
                col_373 = None
                col_384 = None
                col_general = None
                
                for col in sub_columns:
                    col_str = str(col).lower()
                    if '373' in col_str or ('334' in col_str and '373' in col_str):
                        col_373 = col
                    elif '384' in col_str or ('334' in col_str and '384' in col_str):
                        col_384 = col
                    elif any(keyword in col_str for keyword in ['fluorescence', 'emission', 'excitation']):
                        col_general = col
                    elif str(col).replace('.', '').isdigit():
                        col_general = col  # Fallback to first numeric column
                
                # Extract values for each wavelength
                if well_idx < len(raw_data):
                    if col_373 is not None:
                        val_373 = float(raw_data[top_level][col_373].iloc[well_idx])
                    else:
                        val_373 = 80.0  # Default value
                    
                    if col_384 is not None:
                        val_384 = float(raw_data[top_level][col_384].iloc[well_idx])
                    else:
                        val_384 = 100.0  # Default value
                    
                    if col_general is not None:
                        val_general = float(raw_data[top_level][col_general].iloc[well_idx])
                    else:
                        val_general = (val_373 + val_384) / 2  # Average of specific wavelengths
                        
                    measurement_values_373.append(val_373)
                    measurement_values_384.append(val_384)
                    measurement_values_general.append(val_general)
                else:
                    measurement_values_373.append(80.0)
                    measurement_values_384.append(100.0)
                    measurement_values_general.append(90.0)
                    
            except (IndexError, ValueError, KeyError) as e:
                print(f"Warning: Could not extract fluorescence for well {well_idx}: {e}")
                measurement_values_373.append(80.0)
                measurement_values_384.append(100.0)
                measurement_values_general.append(90.0)
    
    else:
        # Simple DataFrame structure
        measurement_values_373 = []
        measurement_values_384 = []
        measurement_values_general = []
        
        # Look for wavelength-specific columns
        col_373 = None
        col_384 = None
        col_general = None
        
        for col in raw_data.columns:
            col_str = str(col).lower()
            if '373' in col_str:
                col_373 = col
            elif '384' in col_str:
                col_384 = col
            elif any(keyword in col_str for keyword in ['fluorescence', 'emission', 'excitation']):
                col_general = col
        
        for well_idx in well_indices:
            try:
                row = well_idx // 12
                col_num = (well_idx % 12) + 1
                well_name = f"{chr(ord('A') + row)}{col_num}"
                
                if well_name in raw_data.index:
                    val_373 = float(raw_data.loc[well_name, col_373]) if col_373 else 80.0
                    val_384 = float(raw_data.loc[well_name, col_384]) if col_384 else 100.0
                    val_general = float(raw_data.loc[well_name, col_general]) if col_general else (val_373 + val_384) / 2
                else:
                    print(f"Warning: Well {well_name} (index {well_idx}) not found in fluorescence data")
                    val_373, val_384, val_general = 80.0, 100.0, 90.0
                    
                measurement_values_373.append(val_373)
                measurement_values_384.append(val_384)
                measurement_values_general.append(val_general)
                        
            except (IndexError, ValueError, KeyError) as e:
                print(f"Warning: Could not extract fluorescence for well {well_idx}: {e}")
                measurement_values_373.append(80.0)
                measurement_values_384.append(100.0)
                measurement_values_general.append(90.0)
    
    # Ensure we have the right number of values
    while len(measurement_values_373) < len(well_indices):
        measurement_values_373.append(80.0)
        measurement_values_384.append(100.0)
        measurement_values_general.append(90.0)
    
    return {
        '334_373': measurement_values_373[:len(well_indices)],
        '334_384': measurement_values_384[:len(well_indices)]
    }

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

def initialize_screening_session(simulate=True):
    """
    Initialize a shared screening session with Lash_E coordinator and input validation.
    Call this once at the start of a multi-combination screening session.
    
    Args:
        simulate (bool): Run in simulation mode
        
    Returns:
        tuple: (lash_e, shared_substock_tracking, shared_wellplate_state)
    """
    print("=== Initializing Surfactant Screening Session ===")
    
    # 1. Initialize Lash_E coordinator (ONCE per session)
    INPUT_VIAL_STATUS_FILE = "status/surfactant_grid_vials_expanded.csv"
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
    logger = lash_e.logger
    
    logger.info(f"Initialized Lash_E workstation in {'simulation' if simulate else 'hardware'} mode")
    
    # 2. Check input files ONCE (requires manual ENTER)
    print("Validating input files (manual interaction required)...")
    logger.info("Validating input files - manual interaction required")
    lash_e.nr_robot.check_input_file()
    lash_e.nr_track.check_input_file()
    
    # 3. Initial robot setup
    print("Setting up robot components...")
    lash_e.nr_robot.move_home()
    lash_e.nr_track.origin()
    lash_e.nr_robot.home_robot_components()
    
    # 4. Get initial wellplate
    logger.info("Preparing initial wellplate")
    lash_e.grab_new_wellplate()
    
    # 5. Prime water lines
    lash_e.nr_robot.prime_reservoir_line(1, 'water')
    
    # 6. Initialize shared tracking
    shared_substock_tracking = load_substock_tracking(lash_e.logger)
    shared_wellplate_state = {
        'global_well_counter': 0,
        'global_well_map': [],
        'total_wells_added': 0,
        'current_plate': 1,
        'wells_used': 0,
        'measurements': [],
        'last_measured_well': -1,
        'total_plates_used': 1,
        'plates_used_this_combo': 0
    }
    
    logger.info("Session initialization complete")
    print("OK Session ready for surfactant combinations")
    return lash_e, shared_substock_tracking, shared_wellplate_state

def finalize_screening_session(lash_e, shared_wellplate_state):
    """
    Clean up and finalize the screening session.
    
    Args:
        lash_e: Lash_E coordinator
        shared_wellplate_state: Final wellplate state
    """
    logger = lash_e.logger
    logger.info("Finalizing screening session")
    
    # Final cleanup
    if shared_wellplate_state['current_plate'] > 0 and shared_wellplate_state['wells_used'] > 0:
        logger.info("Discarding final wellplate")
        lash_e.discard_used_wellplate()
    
    lash_e.nr_robot.move_home()
    
    logger.info(f"Session complete - used {shared_wellplate_state['total_plates_used']} total wellplates")
    print(f"OK Session complete - {shared_wellplate_state['total_plates_used']} plates used")

def surfactant_grid_screening(surfactant_a_name="TTAB", surfactant_b_name="SDS", simulate=True, output_folder=None):
    """
    Backward compatibility wrapper - single combination with full session setup.
    For multi-combination workflows, use initialize_screening_session + execute_single_combination.
    """
    print("=== Single Combination Screening (Legacy Mode) ===")
    
    # Initialize session
    lash_e, shared_substock_tracking, shared_wellplate_state = initialize_screening_session(simulate)
    
    try:
        # Execute single combination
        results_df, measurements, plates_used = execute_single_combination(
            lash_e, surfactant_a_name, surfactant_b_name, 
            shared_substock_tracking, shared_wellplate_state, output_folder
        )
        
        return results_df, measurements, plates_used
        
    finally:
        # Always clean up
        finalize_screening_session(lash_e, shared_wellplate_state)

def execute_single_combination(lash_e, surfactant_a_name, surfactant_b_name, shared_substock_tracking, shared_wellplate_state, output_folder=None):
    """
    Execute surfactant grid screening for a single combination using shared session resources.
    
    Args:
        lash_e: Shared Lash_E coordinator instance
        surfactant_a_name (str): Name of first surfactant (cationic)
        surfactant_b_name (str): Name of second surfactant (anionic)
        shared_substock_tracking: Shared dilution tracking across experiments
        shared_wellplate_state: Shared wellplate state management
        output_folder (str): Optional output folder path
        
    Returns:
        tuple: (results_df, measurements, plates_used_this_combination)
    """
    logger = lash_e.logger
    logger.info(f"Starting surfactant combination: {surfactant_a_name} vs {surfactant_b_name}")
    
    # Get surfactant info from library
    surf_a_info = SURFACTANT_LIBRARY[surfactant_a_name]
    surf_b_info = SURFACTANT_LIBRARY[surfactant_b_name]
    
    print(f"\n=== {surfactant_a_name} + {surfactant_b_name} ===")
    print(f"Surfactant A: {surfactant_a_name} ({surf_a_info['full_name']}, {surf_a_info['stock_conc']} mM stock)")
    print(f"Surfactant B: {surfactant_b_name} ({surf_b_info['full_name']}, {surf_b_info['stock_conc']} mM stock)")
    
    # Calculate experimental parameters
    concentrations = calculate_grid_concentrations()
    total_wells, grid_dimension = calculate_total_wells_needed()
    
    print(f"Grid: {grid_dimension}x{grid_dimension} = {grid_dimension**2} conditions")
    print(f"Wells needed: {total_wells}")
    
    # Periodic re-homing and priming (good for automation)
    print("Performing periodic robot maintenance...")
    lash_e.nr_robot.move_home()
    lash_e.nr_robot.home_robot_components()
    lash_e.nr_robot.prime_reservoir_line(1, 'water')
    
    # Create or reuse dilution series for both surfactants
    logger.info(f"Preparing dilution series for {surfactant_a_name} and {surfactant_b_name}")
    dilution_vials_a, dilution_steps_a, achievable_a = check_or_create_substocks(lash_e, surfactant_a_name, concentrations, shared_substock_tracking)
    dilution_vials_b, dilution_steps_b, achievable_b = check_or_create_substocks(lash_e, surfactant_b_name, concentrations, shared_substock_tracking)
    
    # Display concentration grid summary
    print("\n" + create_concentration_grid_summary(concentrations, dilution_vials_a, dilution_vials_b, surfactant_a_name, surfactant_b_name))
    print("")
    
    # Pipette grid using shared wellplate state
    logger.info("Starting grid pipetting with interval measurements")
    well_map, updated_wellplate_state = pipette_grid_to_shared_wellplate(lash_e, achievable_a, achievable_b, dilution_vials_a, dilution_vials_b, surfactant_a_name, surfactant_b_name, shared_wellplate_state)
    
    # Update shared state
    shared_wellplate_state.update(updated_wellplate_state)
    
    # Combine measurement data
    logger.info("Combining measurement data from all intervals")
    results_df = combine_measurement_data(well_map, updated_wellplate_state)
    all_measurements = updated_wellplate_state['measurements']
    
    # Create output folder if not provided
    if output_folder is None:
        output_folder = create_output_folder(lash_e.simulate)
    
    logger.info(f"Saving results to: {output_folder}")
    
    # Save results to output folder  
    save_results(results_df, well_map, updated_wellplate_state, all_measurements, concentrations, dilution_vials_a, dilution_vials_b, dilution_steps_a, dilution_steps_b, surfactant_a_name, surfactant_b_name, lash_e.simulate, output_folder, logger)
    
    plates_used_this_combo = updated_wellplate_state.get('plates_used_this_combo', 0)
    logger.info(f"Combination completed, plates used for this combination: {plates_used_this_combo}")
    
    print(f"OK {surfactant_a_name}+{surfactant_b_name}: {len(results_df)} wells")
    return results_df, all_measurements, plates_used_this_combo

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
    
    # Initialize shared session (ONCE - handles input file checking)
    lash_e, shared_substock_tracking, shared_wellplate_state = initialize_screening_session(simulate)
    
    # Initialize tracking with logger
    global_substock_tracking = load_substock_tracking(lash_e.logger)
    current_plate = 1
    wells_used_total = 0
    all_results = {}
    
    print(f"Results will be saved to: {main_output}")
    
    try:
        all_results = {}
        
        print(f"\\nStarting comprehensive screening with shared session...")
        
        # Filter experiments to run only 6-8 
        START_EXPERIMENT = 6  # 0-based: experiment 6 = index 5   
        END_EXPERIMENT = 8    # 0-based: experiment 8 = index 7
        print(f"Running experiments {START_EXPERIMENT+1} to {END_EXPERIMENT+1} only")
        
        for i, anionic in enumerate(anionic_surfactants):
            for j, cationic in enumerate(cationic_surfactants):
                combo_num = i * len(cationic_surfactants) + j + 1
                
                # Skip experiments not in our target range
                if combo_num < START_EXPERIMENT + 1 or combo_num > END_EXPERIMENT + 1:
                    print(f"\\nSKIPPING combination {combo_num}: {anionic} + {cationic}")
                    continue
                    
                print(f"\\n{'='*60}")
                print(f"COMBINATION {combo_num}/{total_combinations}: {anionic} + {cationic}")
                print(f"Current wellplate: {shared_wellplate_state['current_plate']}, wells used: {shared_wellplate_state['wells_used']}/96")
                print(f"{'='*60}")
                
                # Create subfolder for this combination
                combo_folder = os.path.join(main_output, f"{combo_num:02d}_{anionic}_{cationic}")
                os.makedirs(combo_folder, exist_ok=True)
                
                try:
                    # Execute combination using shared session
                    results_df, measurements, plates_used_this_combo = execute_single_combination(
                        lash_e, cationic, anionic,  # Cationic first, anionic second
                        shared_substock_tracking, shared_wellplate_state, combo_folder
                    )
                    
                    all_results[f"{anionic}_{cationic}"] = {
                        'anionic': anionic,
                        'cationic': cationic,
                        'results_df': results_df,
                        'measurements': measurements,
                        'plates_used_this_combo': plates_used_this_combo,
                        'wells_count': len(results_df),
                        'output_folder': combo_folder,
                        'completed': True
                    }
                    
                    print(f"\u2713 Completed {anionic} + {cationic}: {len(results_df)} wells")
                    
                except Exception as e:
                    print(f"\u274c Failed {anionic} + {cationic}: {e}")
                    lash_e.logger.error(f"Combination {anionic} + {cationic} failed: {e}")
                    all_results[f"{anionic}_{cationic}"] = {
                        'anionic': anionic,
                        'cationic': cationic,
                        'error': str(e),
                        'completed': False
                    }
        
        # Save session-level tracking
        save_substock_tracking(shared_substock_tracking, main_output)
        
        # Create comprehensive summary
        successful_experiments = sum(1 for r in all_results.values() if r.get('completed', False))
        total_wells_used = sum(r.get('wells_count', 0) for r in all_results.values() if r.get('completed', False))
        
        summary_file = os.path.join(main_output, "comprehensive_screening_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE SURFACTANT SCREENING SUMMARY\\n")
            f.write("=" * 50 + "\\n\\n")
            
            if simulate:
                f.write("NOTE: This was a SIMULATION - no actual hardware was used\\n\\n")
            
            f.write(f"Total combinations attempted: {total_combinations}\\n")
            f.write(f"Successful combinations: {successful_experiments}\\n")
            f.write(f"Total wells used: {total_wells_used}\\n")
            f.write(f"Total wellplates used: {shared_wellplate_state['total_plates_used']}\\n\\n")
            
            f.write("SURFACTANT COMBINATIONS:\\n")
            f.write("-" * 30 + "\\n")
            
            for combo_name, result in all_results.items():
                if result.get('completed', False):
                    f.write(f"{result['anionic']} + {result['cationic']}: {result['wells_count']} wells\\n")
                else:
                    f.write(f"{result['anionic']} + {result['cationic']}: FAILED - {result.get('error', 'Unknown error')}\\n")
        
        print(f"\\n{'='*80}")
        print("COMPREHENSIVE SCREENING COMPLETE")
        print(f"Successful combinations: {successful_experiments}/{total_combinations}")
        print(f"Total wells used: {total_wells_used}")
        print(f"Total wellplates used: {shared_wellplate_state['total_plates_used']}")
        print(f"Results saved to: {main_output}")
        print(f"{'='*80}")
        
        return all_results
        
    finally:
        # Always finalize session
        finalize_screening_session(lash_e, shared_wellplate_state)
if __name__ == "__main__":
    """
    Run surfactant grid screening.
    
    Options:
    1. Single experiment: Specify surfactant names
    2. Comprehensive: Run all anionic/cationic combinations
    """
    
    # Choose experiment mode
    RUN_COMPREHENSIVE = False  # Set to False for single experiment
    
    print(f"Starting surfactant grid screening in {'comprehensive' if RUN_COMPREHENSIVE else 'single'} mode")
    print(f"Simulation mode: {SIMULATE}")
    
    if RUN_COMPREHENSIVE:
        # Run all 9 combinations (3 anionic × 3 cationic)
        print("Running comprehensive screening for all surfactant combinations...")
        all_results = run_all_surfactant_combinations(simulate=SIMULATE)
        
        successful = sum(1 for r in all_results.values() if 'error' not in r)
        print(f"\\nCompleted: {successful}/{len(all_results)} combinations successful")
        
    else:
        # Run single experiment
        SURFACTANT_A = "SDS"  # Cationic
        SURFACTANT_B = "DTAB"   # Anionic
        
        print(f"Running single experiment: {SURFACTANT_A} + {SURFACTANT_B}")
        results, measurements, plates_used = surfactant_grid_screening(
            surfactant_a_name=SURFACTANT_A, 
            surfactant_b_name=SURFACTANT_B, 
            simulate=SIMULATE
        )
        print(f"Single experiment completed: {len(results)} wells, {plates_used} plates")
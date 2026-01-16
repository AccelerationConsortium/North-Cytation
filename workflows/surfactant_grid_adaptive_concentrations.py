"""
Surfactant Grid Turbidity + Fluorescence Screening Workflow - Adaptive Concentrations
Systematic dilution grid of two surfactants with adaptive concentration ranges based on stock concentrations.

UPDATED CONCENTRATION APPROACH:
- Uses adaptive concentration ranges: min_conc = 10^-4 mM, max_conc = stock_conc * (allocated_volume / well_volume)
- Logarithmic spacing with fixed number of concentrations (9 by default)
- Each surfactant gets its own optimized concentration range based on volume constraints

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

# ================================================================================
# IMPORTS AND DEPENDENCIES
# ================================================================================

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

# ================================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# ================================================================================

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
SIMULATE = True # Set to False for actual hardware execution
VALIDATE_LIQUIDS = True # Set to False to skip pipetting validation during initialization

# Pump configuration:
# Pump 0 = Pipetting pump (no reservoir, used for aspirate/dispense)
# Pump 1 = Water reservoir pump (carousel angle 45 deg, height 70)

# Adaptive grid parameters - concentration ranges adapt to stock concentrations
MIN_CONC = 10**-4  # 0.0001 mM minimum concentration for all surfactants
NUMBER_CONCENTRATIONS = 9  # Number of concentration steps in logarithmic grid
N_REPLICATES = 1
WELL_VOLUME_UL = 200  # uL per well
PYRENE_VOLUME_UL = 5  # uL pyrene_DMSO to add per well

# Buffer addition settings
ADD_BUFFER = False  # Set to False to skip buffer addition
BUFFER_VOLUME_UL = 20  # uL buffer to add per well
BUFFER_OPTIONS = ['MES', 'HEPES', 'CAPS']  # Available buffers
SELECTED_BUFFER = 'HEPES'  # Choose from BUFFER_OPTIONS

# Volume calculation with buffer compensation
# Always reserve space for buffer and pyrene to maintain consistent concentration ranges
# This ensures max concentrations are always the same regardless of ADD_BUFFER setting
EFFECTIVE_SURFACTANT_VOLUME = (WELL_VOLUME_UL - BUFFER_VOLUME_UL ) / 2  # Always reserve space
print(f"Effective surfactant volume: {EFFECTIVE_SURFACTANT_VOLUME} µL per surfactant (reserves space for buffer+pyrene)")

# CRITICAL: Concentration correction factor for buffer dilution
# When buffer is added, stock concentrations must be higher to compensate for dilution
# This ensures final concentrations match intended values
CONCENTRATION_CORRECTION_FACTOR = WELL_VOLUME_UL / (2 * EFFECTIVE_SURFACTANT_VOLUME)
print(f"Concentration correction factor: {CONCENTRATION_CORRECTION_FACTOR:.3f} (buffer={ADD_BUFFER}, buffer_vol={BUFFER_VOLUME_UL if ADD_BUFFER else 0}uL)")
MAX_WELLS = 96 #Wellplate size

# Constants
FINAL_SUBSTOCK_VOLUME = 6  # mL final volume for each dilution
MINIMUM_PIPETTE_VOLUME = 0.2  # mL (200 uL) - minimum volume for accurate pipetting
MEASUREMENT_INTERVAL = 96    # Measure every N wells to prevent evaporation

# Measurement protocol files for Cytation
TURBIDITY_PROTOCOL_FILE = r"C:\Protocols\CMC_Absorbance_96.prt"
FLUORESCENCE_PROTOCOL_FILE = r"C:\Protocols\CMC_Fluorescence_96.prt"

# File paths
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/surfactant_grid_vials_expanded.csv"



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
            'workflow': 'surfactant_grid_adaptive_concentrations'
        }
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)  # default=str handles datetime objects
        
        print(f"OK Backed up measurement data to: {backup_path}")
        
    except Exception as e:
        print(f"WARNING: Failed to backup measurement data: {e}")
        # Don't crash the workflow if backup fails
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
            'workflow': 'surfactant_grid_adaptive_concentrations'
        }
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)  # default=str handles datetime objects
        
        print(f"OK Backed up measurement data to: {backup_path}")
        
    except Exception as e:
        print(f"WARNING: Failed to backup measurement data: {e}")
        # Don't crash the workflow if backup fails

def condition_tip(lash_e, vial_name, conditioning_volume_ul=100):
    """Condition a pipette tip by aspirating and dispensing into source vial multiple times
    
    Args:
        lash_e: Lash_E robot controller
        vial_name: Name of vial to condition tip with
        conditioning_volume_ul: Total volume for conditioning (default 100 µL)
    """
    try:
        # Calculate volume per conditioning cycle (5 cycles total)
        cycles = 5
        volume_per_cycle_ul = conditioning_volume_ul / cycles
        volume_per_cycle_ml = volume_per_cycle_ul / 1000
        
        print(f"    Conditioning tip with {vial_name}: {cycles} cycles of {volume_per_cycle_ul:.1f}µL")
        
        for cycle in range(cycles):
            # Aspirate from vial (with return_home=False to keep tip position)
            lash_e.nr_robot.aspirate_from_vial(
                vial_name, volume_per_cycle_ml, 
                liquid='water', use_safe_location=True, return_home=False
            )
            # Dispense back into same vial (with return_home=False to keep tip position)
            lash_e.nr_robot.dispense_into_vial(
                vial_name, volume_per_cycle_ml, 
                liquid='water', return_home=False
            )
        
        print(f"    Tip conditioning complete for {vial_name}")
        
    except Exception as e:
        print(f"    Warning: Could not condition tip with {vial_name}: {e}")

def get_pipette_usage_breakdown(lash_e):
    """Get detailed pipette usage breakdown by tip type from North_Robot.PIPETS_USED property."""
    try:
        # Access the actual PIPETS_USED property from North_Robot
        pipette_usage = lash_e.nr_robot.PIPETS_USED
        
        large_tips = (pipette_usage.get('large_tip_rack_1', 0) + 
                     pipette_usage.get('large_tip_rack_2', 0))
        small_tips = (pipette_usage.get('small_tip_rack_1', 0) + 
                     pipette_usage.get('small_tip_rack_2', 0))
        
        return {
            'large_tips': large_tips,
            'small_tips': small_tips,
            'total': large_tips + small_tips
        }
    except Exception as e:
        print(f"  Warning: Could not read pipette status: {e}")
        # Fallback to manual count if something fails
        manual_count = getattr(lash_e, 'pipette_count', 0)
        return {
            'large_tips': manual_count,
            'small_tips': 0,
            'total': manual_count
        }

def track_pipette_usage(lash_e, operation_description="pipette operation"):
    """Increment pipette counter (simulation only) and optionally log the operation."""
    if hasattr(lash_e, 'simulate') and lash_e.simulate:
        # Only manually count in simulation mode
        lash_e.pipette_count += 1
        if not getattr(lash_e, 'quiet_mode', False):
            print(f"    [Simulated Pipette {lash_e.pipette_count}]: {operation_description}")
    # In hardware mode, the robot automatically tracks usage in robot_status.yaml

# ================================================================================
# SECTION 1: SMART SUBSTOCK MANAGEMENT
# ================================================================================

def optimize_vial_positions_for_pipetting(lash_e, plan_a, plan_b, surfactant_a_name, surfactant_b_name):
    """
    Move all stock and substock vials to optimal positions (end of rack) before pipetting.
    Pipette from low to high concentration, then return vials home.
    
    Args:
        lash_e: Robot coordinator
        plan_a, plan_b: Smart dilution plans containing vial names
        surfactant_a_name, surfactant_b_name: Surfactant names
    
    Returns:
        dict: Mapping of vial names to their temporary positions for restoration
    """
    print(f"\n" + "="*60)
    print(f"OPTIMIZING VIAL POSITIONS FOR PIPETTING")
    print(f"="*60)
    
    # Collect all vials that will be used for pipetting
    surfactant_a_vials = set()
    surfactant_b_vials = set()
    
    # Get stock vials
    surfactant_a_vials.add(f"{surfactant_a_name}_stock")
    surfactant_b_vials.add(f"{surfactant_b_name}_stock")
    
    # Get substock vials from plans
    for recipe in plan_a['recipes']:
        surfactant_a_vials.add(recipe['vial_name'])
    for recipe in plan_b['recipes']:
        surfactant_b_vials.add(recipe['vial_name'])
    
    print(f"  Surfactant A vials to optimize: {sorted(surfactant_a_vials)}")
    print(f"  Surfactant B vials to optimize: {sorted(surfactant_b_vials)}")
    
    # Find available positions at end of rack (best for pipetting access)
    # Rack has 48 positions (0-47), use positions 40-47 for temporary staging
    OPTIMAL_POSITIONS = list(range(40, 48))  # Positions 40-47 (8 positions)
    
    # Track original positions for restoration
    original_positions = {}
    position_counter = 0
    
    # Move surfactant A vials first (will be pipetted first)
    for vial_name in sorted(surfactant_a_vials):
        if position_counter >= len(OPTIMAL_POSITIONS):
            print(f"  Warning: Ran out of optimal positions, leaving {vial_name} in original location")
            break
            
        try:
            # Get current position
            current_location = lash_e.nr_robot.get_vial_info(vial_name, 'location')
            current_index = lash_e.nr_robot.get_vial_info(vial_name, 'location_index')
            
            # Store original position
            original_positions[vial_name] = {
                'location': current_location,
                'location_index': current_index
            }
            
            # Move to optimal position
            optimal_position = OPTIMAL_POSITIONS[position_counter]
            print(f"  Moving {vial_name}: {current_location}[{current_index}] -> main_8mL_rack[{optimal_position}]")
            lash_e.nr_robot.move_vial_to_location(vial_name, 'main_8mL_rack', optimal_position)
            position_counter += 1
            
        except Exception as e:
            print(f"  Warning: Could not move {vial_name}: {e}")
    
    # Move surfactant B vials  
    for vial_name in sorted(surfactant_b_vials):
        if position_counter >= len(OPTIMAL_POSITIONS):
            print(f"  Warning: Ran out of optimal positions, leaving {vial_name} in original location")
            break
            
        try:
            # Get current position
            current_location = lash_e.nr_robot.get_vial_info(vial_name, 'location')
            current_index = lash_e.nr_robot.get_vial_info(vial_name, 'location_index')
            
            # Store original position
            original_positions[vial_name] = {
                'location': current_location,
                'location_index': current_index
            }
            
            # Move to optimal position
            optimal_position = OPTIMAL_POSITIONS[position_counter]
            print(f"  Moving {vial_name}: {current_location}[{current_index}] -> main_8mL_rack[{optimal_position}]")
            lash_e.nr_robot.move_vial_to_location(vial_name, 'main_8mL_rack', optimal_position)
            position_counter += 1
            
        except Exception as e:
            print(f"  Warning: Could not move {vial_name}: {e}")
    
    print(f"  Successfully positioned {len(original_positions)} vials in optimal locations")
    print(f"  Ready for low->high concentration pipetting")
    
    return original_positions

def restore_vial_positions(lash_e, original_positions):
    """
    Restore all vials to their original home positions after pipetting.
    
    Args:
        lash_e: Robot coordinator
        original_positions: Dictionary mapping vial names to their original positions
    """
    print(f"\n" + "="*60)
    print(f"RESTORING VIAL POSITIONS AFTER PIPETTING")
    print(f"="*60)
    
    for vial_name, position_info in original_positions.items():
        try:
            print(f"  Restoring {vial_name} -> {position_info['location']}[{position_info['location_index']}]")
            lash_e.nr_robot.move_vial_to_location(
                vial_name, 
                position_info['location'], 
                position_info['location_index']
            )
        except Exception as e:
            print(f"  Warning: Could not restore {vial_name}: {e}")
    
    print(f"  Successfully restored {len(original_positions)} vials to home positions")

class SurfactantSubstockTracker:
    """Track surfactant substock vial contents and find optimal dilution strategies."""
    
    def __init__(self):
        self.substocks = {}  # vial_name: {'surfactant_name': str, 'concentration_mm': float, 'volume_ml': float}
        self.next_available_substock = 0
        self.min_pipette_volume_ul = 10  # Minimum pipetting volume in µL
        self.max_surfactant_volume_ul = EFFECTIVE_SURFACTANT_VOLUME  # Max volume per surfactant per well
    
    def find_best_solution_for_concentration(self, surfactant_name, target_conc_mm):
        """
        Find best available solution (stock or substock) for achieving target concentration.
        Returns solution dict or None if no suitable solution exists within pipetting limits.
        """
        options = []
        max_volume_ml = self.max_surfactant_volume_ul / 1000
        
        # Check stock solution first
        if surfactant_name in SURFACTANT_LIBRARY:
            stock_conc = SURFACTANT_LIBRARY[surfactant_name]["stock_conc"]
            # Calculate volume needed for target final concentration in 200 µL well
            # Final_conc = (stock_conc * vol_needed) / 200 µL
            # Therefore: vol_needed = (target_conc * 200) / stock_conc
            vol_needed_ul = (target_conc_mm * WELL_VOLUME_UL) / stock_conc
            vol_needed_ml = vol_needed_ul / 1000
            
            if vol_needed_ul >= self.min_pipette_volume_ul and vol_needed_ul <= self.max_surfactant_volume_ul:
                options.append({
                    'vial_name': f"{surfactant_name}_stock",
                    'concentration_mm': stock_conc,
                    'volume_needed_ml': vol_needed_ml,
                    'volume_needed_ul': vol_needed_ul,
                    'is_stock': True
                })
        
        # Check existing substocks
        for vial_name, contents in self.substocks.items():
            if (contents['surfactant_name'] == surfactant_name and 
                contents['volume_ml'] > 0):
                
                # Calculate volume needed for target final concentration in 200 µL well
                vol_needed_ul = (target_conc_mm * WELL_VOLUME_UL) / contents['concentration_mm']
                vol_needed_ml = vol_needed_ul / 1000
                
                if (vol_needed_ul >= self.min_pipette_volume_ul and 
                    vol_needed_ul <= self.max_surfactant_volume_ul and
                    contents['volume_ml'] >= vol_needed_ml):
                    
                    options.append({
                        'vial_name': vial_name,
                        'concentration_mm': contents['concentration_mm'],
                        'volume_needed_ml': vol_needed_ml,
                        'volume_needed_ul': vol_needed_ul,
                        'is_stock': False
                    })
        
        if not options:
            return None
        
        # Return most concentrated option that's still pipettable (uses least volume)
        options.sort(key=lambda x: x['concentration_mm'], reverse=True)
        return options[0]
    
    def calculate_optimal_substock_concentration(self, surfactant_name, target_conc_mm):
        """
        Calculate optimal substock concentration for a target.
        Aims for 15-20 µL pipetting volumes for reusability.
        """
        max_volume_ml = self.max_surfactant_volume_ul / 1000
        
        # Target 15-20 µL volumes for efficient, reusable solutions
        target_volume_ul = 17.5
        # Calculate concentration needed to achieve target final concentration with target volume
        # target_conc_mm = (optimal_conc * target_volume_ul) / 200
        # Therefore: optimal_conc = (target_conc_mm * 200) / target_volume_ul
        optimal_conc = (target_conc_mm * WELL_VOLUME_UL) / target_volume_ul
        
        # Ensure we don't go below minimum pipette volume
        min_volume_ul = self.min_pipette_volume_ul
        max_conc_absolute = (target_conc_mm * WELL_VOLUME_UL) / min_volume_ul
        
        # CRITICAL: Substock cannot be more concentrated than the stock!
        stock_conc = SURFACTANT_LIBRARY[surfactant_name]['stock_conc']
        max_substock_conc = stock_conc * (EFFECTIVE_SURFACTANT_VOLUME / WELL_VOLUME_UL)  # 90/200 = 0.45
        
        # Use the most restrictive constraint
        final_conc = min(optimal_conc, max_conc_absolute, max_substock_conc)
        
        # Round to nice numbers
        def round_to_nice_concentration(value):
            if value <= 0:
                return 0
            log_val = np.log10(value)
            magnitude = 10 ** np.floor(log_val)
            normalized = value / magnitude
            
            if normalized <= 1.0:
                nice_normalized = 1.0
            elif normalized <= 2.0:
                nice_normalized = 2.0
            elif normalized <= 3.0:
                nice_normalized = 3.0
            elif normalized <= 5.0:
                nice_normalized = 3.0
            else:
                nice_normalized = 5.0
                
            return nice_normalized * magnitude
        
        # Use 80% for safety margin
        safe_conc = final_conc * 0.8
        return round_to_nice_concentration(safe_conc)
    
    def add_substock(self, surfactant_name, concentration_mm, volume_ml=6.0):
        """Add a new substock to tracking."""
        vial_name = f"{surfactant_name}_dilution_{self.next_available_substock}"
        self.substocks[vial_name] = {
            'surfactant_name': surfactant_name,
            'concentration_mm': concentration_mm,
            'volume_ml': volume_ml
        }
        self.next_available_substock += 1
        return vial_name

def calculate_smart_dilution_plan(surfactant_name, target_concentrations_mm):
    """
    Calculate optimal dilution strategy for a surfactant across all target concentrations.
    Returns plan showing which solutions to create and how to use them.
    """
    tracker = SurfactantSubstockTracker()
    plan = {'substocks_needed': [], 'concentration_map': {}}
    
    print(f"\n=== Analyzing dilution strategy for {surfactant_name} ===")
    print(f"Target concentrations: {[f'{c:.2e}' for c in target_concentrations_mm]} mM")
    
    for target_conc in target_concentrations_mm:
        # Try to find existing solution
        solution = tracker.find_best_solution_for_concentration(surfactant_name, target_conc)
        
        if solution:
            # Found existing solution
            plan['concentration_map'][target_conc] = solution
            print(f"  {target_conc:.2e} mM: Use {solution['vial_name']} ({solution['volume_needed_ul']:.1f} µL)")
        else:
            # Need new substock
            optimal_conc = tracker.calculate_optimal_substock_concentration(surfactant_name, target_conc)
            vial_name = tracker.add_substock(surfactant_name, optimal_conc)
            
            # Now find solution using new substock
            solution = tracker.find_best_solution_for_concentration(surfactant_name, target_conc)
            if solution:
                plan['concentration_map'][target_conc] = solution
                plan['substocks_needed'].append({
                    'vial_name': vial_name,
                    'concentration_mm': optimal_conc,
                    'needed_for': [target_conc]
                })
                print(f"  {target_conc:.2e} mM: CREATE {vial_name} at {optimal_conc:.2e} mM → use {solution['volume_needed_ul']:.1f} µL")
            else:
                print(f"  {target_conc:.2e} mM: *** CANNOT ACHIEVE ***")
    
    return plan, tracker

def calculate_dilution_recipes(plan_a, plan_b, surfactant_a_name, surfactant_b_name):
    """
    Calculate the exact dilution recipes for each substock that needs to be created.
    Uses serial dilutions when direct dilution would require volumes < 200 µL.
    """
    MIN_SUBSTOCK_VOLUME_UL = 200  # Minimum volume for accurate substock preparation
    FINAL_SUBSTOCK_VOLUME_ML = 6.0
    
    recipes = []
    
    print(f"\n=== SUBSTOCK DILUTION RECIPES ===")
    print("For each substock showing: source + volume → target_concentration")
    print(f"(Using minimum {MIN_SUBSTOCK_VOLUME_UL}µL pipetting volumes for accuracy)")
    
    # Process substocks for both surfactants
    for plan, surfactant_name in [(plan_a, surfactant_a_name), (plan_b, surfactant_b_name)]:
        if plan['substocks_needed']:
            print(f"\n{surfactant_name} substocks:")
            
            # Get stock concentration
            stock_conc = SURFACTANT_LIBRARY[surfactant_name]["stock_conc"]
            
            # Sort substocks by concentration (highest first for cascade approach)
            substocks = sorted(plan['substocks_needed'], 
                             key=lambda x: x['concentration_mm'], reverse=True)
            
            created_substocks = {}  # Track what we've created: {conc: vial_name}
            created_substocks[stock_conc] = f"{surfactant_name}_stock"  # Stock is available
            
            for substock in substocks:
                target_conc = substock['concentration_mm']
                vial_name = substock['vial_name']
                
                # Find best source (stock or existing substock)
                best_source = None
                min_volume_needed = float('inf')
                
                for source_conc, source_name in created_substocks.items():
                    if source_conc > target_conc:  # Can only dilute down
                        # Match automation logic: apply concentration correction factor
                        required_stock_conc = target_conc * CONCENTRATION_CORRECTION_FACTOR
                        dilution_factor = source_conc / required_stock_conc
                        source_volume_ml = FINAL_SUBSTOCK_VOLUME_ML / dilution_factor
                        source_volume_ul = source_volume_ml * 1000
                        
                        if source_volume_ul >= MIN_SUBSTOCK_VOLUME_UL:
                            if source_volume_ul < min_volume_needed:
                                min_volume_needed = source_volume_ul
                                best_source = {
                                    'name': source_name,
                                    'conc': source_conc,
                                    'volume_ul': source_volume_ul,
                                    'volume_ml': source_volume_ml
                                }
                
                if best_source:
                    # Match automation logic: apply concentration correction factor
                    required_stock_conc = target_conc * CONCENTRATION_CORRECTION_FACTOR
                    dilution_factor = best_source['conc'] / required_stock_conc
                    
                    source_volume_ml = FINAL_SUBSTOCK_VOLUME_ML / dilution_factor
                    source_volume_ul = source_volume_ml * 1000
                    water_volume_ml = FINAL_SUBSTOCK_VOLUME_ML - source_volume_ml
                    water_volume_ul = water_volume_ml * 1000
                    
                    recipes.append({
                        'Vial_Name': vial_name,
                        'Surfactant': surfactant_name,
                        'Target_Conc_mM': target_conc,
                        'Source_Vial': best_source['name'],
                        'Source_Conc_mM': best_source['conc'],
                        'Source_Volume_mL': source_volume_ml,
                        'Source_Volume_uL': source_volume_ul,
                        'Water_Volume_mL': water_volume_ml,
                        'Water_Volume_uL': water_volume_ul,
                        'Final_Volume_mL': FINAL_SUBSTOCK_VOLUME_ML,
                        'Dilution_Factor': dilution_factor
                    })
                    
                    # Add to available sources
                    created_substocks[target_conc] = vial_name
                    
                    source_type = "stock" if best_source['name'].endswith('_stock') else "dilution"
                    print(f"  {vial_name}: {source_volume_ul:.0f}µL {best_source['name']} + {water_volume_ul:.0f}µL water → {target_conc:.2e} mM")
                    print(f"    (Dilution factor: {dilution_factor:.1f}x from {best_source['conc']:.2e} mM {source_type}, corrected for {CONCENTRATION_CORRECTION_FACTOR:.2f}x buffer factor)")
                
                else:
                    # Cannot make with current minimum volume - needs intermediate
                    print(f"  {vial_name}: *** NEEDS INTERMEDIATE DILUTION - too dilute for {MIN_SUBSTOCK_VOLUME_UL}µL minimum ***")
                    
                    # Suggest intermediate concentration
                    max_possible_dilution = FINAL_SUBSTOCK_VOLUME_ML / (MIN_SUBSTOCK_VOLUME_UL / 1000)
                    intermediate_conc = stock_conc / max_possible_dilution
                    print(f"    Suggestion: Create intermediate at ~{intermediate_conc:.2e} mM first")
    
    if not recipes:
        print("No substocks needed - using only stock solutions!")
    
    return recipes

def create_substocks_from_recipes(lash_e, recipes):
    """
    Create physical substocks according to the calculated recipes.
    Checks for existing substocks (volume > 0) before creating new ones.
    """
    logger = lash_e.logger
    logger.info(f"Checking for existing substocks and creating {len(recipes)} new ones as needed")
    
    # First, check for existing substocks
    existing_substocks = {}
    try:
        vial_file_path = "status/surfactant_grid_vials_expanded.csv"
        if os.path.exists(vial_file_path):
            import pandas as pd
            df = pd.read_csv(vial_file_path)
            
            for _, row in df.iterrows():
                vial_name = row['vial_name']
                volume = float(row['vial_volume']) if pd.notna(row['vial_volume']) else 0.0
                
                if volume > 0:
                    existing_substocks[vial_name] = volume
                    logger.info(f"Found existing substock: {vial_name} ({volume:.1f} mL)")
    except Exception as e:
        logger.warning(f"Could not check existing substocks: {e}")
    
    created_substocks = []
    skipped_count = 0
    
    for recipe in recipes:
        vial_name = recipe['Vial_Name']
        surfactant = recipe['Surfactant']
        target_conc = recipe['Target_Conc_mM']
        source_vial = recipe['Source_Vial']
        source_volume_ml = recipe['Source_Volume_mL'] 
        water_volume_ml = recipe['Water_Volume_mL']
        
        # Check if this substock already exists with sufficient volume
        if vial_name in existing_substocks and existing_substocks[vial_name] > 0:
            logger.info(f"Skipping {vial_name}: already exists with {existing_substocks[vial_name]:.1f} mL")
            created_substocks.append({
                'vial_name': vial_name,
                'concentration_mm': target_conc,
                'volume_ml': existing_substocks[vial_name],
                'created': False,  # Not newly created
                'existed': True
            })
            skipped_count += 1
            continue
        
        logger.info(f"Creating {vial_name}: {target_conc:.2e} mM")
        logger.info(f"  Recipe: {source_volume_ml*1000:.0f}µL {source_vial} + {water_volume_ml*1000:.0f}µL water")
        
        # Always call robot functions - Lash_E handles simulation internally
        try:
            # Add water first if needed
            if water_volume_ml > 0:
                lash_e.nr_robot.dispense_into_vial_from_reservoir(
                    reservoir_index=1, vial_index=vial_name, 
                    volume=water_volume_ml, return_home=False
                )
            
            # Add source solution
            lash_e.nr_robot.dispense_from_vial_into_vial(
                source_vial_name=source_vial, 
                dest_vial_name=vial_name, 
                volume=source_volume_ml,
                liquid='water'
            )
            
            # Vortex to mix
            lash_e.nr_robot.vortex_vial(vial_name=vial_name, vortex_time=8, vortex_speed=80)
            
            created_substocks.append({
                'vial_name': vial_name,
                'concentration_mm': target_conc,
                'volume_ml': recipe['Final_Volume_mL'],
                'created': True,
                'existed': False
            })
            
            logger.info(f"  + Successfully created {vial_name}")
            
        except Exception as e:
            logger.error(f"  - Failed to create {vial_name}: {str(e)}")
            created_substocks.append({
                'vial_name': vial_name,
                'concentration_mm': target_conc,
                'volume_ml': 0,
                'created': False,
                'existed': False,
                'error': str(e)
            })
    
    newly_created = len([s for s in created_substocks if s['created'] and not s.get('existed', False)])
    total_available = len([s for s in created_substocks if (s['created'] or s.get('existed', False))])
    
    logger.info(f"Substock summary: {newly_created} newly created, {skipped_count} already existed, {total_available} total available")
    return created_substocks

def verify_concentration_calculations_and_export(plan_a, plan_b, surfactant_a_name, surfactant_b_name, target_concs_a, target_concs_b):
    """
    Verify that the dilution plan will actually produce target concentrations.
    Export detailed calculations to CSV for review.
    """
    import pandas as pd
    
    print(f"\n=== CONCENTRATION VERIFICATION ===")
    
    verification_data = []
    
    # Verify each concentration for both surfactants
    for i, target_conc in enumerate(target_concs_a):
        if target_conc in plan_a['concentration_map']:
            solution = plan_a['concentration_map'][target_conc]
            
            # Calculate final concentration in 200 µL well
            solution_conc_mm = solution['concentration_mm'] 
            solution_volume_ul = solution['volume_needed_ul']
            
            # Final concentration calculation: (solution_conc * solution_volume) / total_well_volume
            calculated_final_conc = (solution_conc_mm * solution_volume_ul) / WELL_VOLUME_UL
            
            # Water volume for this surfactant
            water_volume_ul = EFFECTIVE_SURFACTANT_VOLUME - solution_volume_ul
            
            # Calculate error
            error_percent = ((calculated_final_conc - target_conc) / target_conc) * 100 if target_conc > 0 else 0
            
            verification_data.append({
                'Surfactant': surfactant_a_name,
                'Target_Conc_mM': target_conc,
                'Solution_Vial': solution['vial_name'],
                'Solution_Conc_mM': solution_conc_mm,
                'Solution_Volume_uL': solution_volume_ul,
                'Water_Volume_uL': water_volume_ul,
                'Total_Surfactant_Volume_uL': solution_volume_ul + water_volume_ul,
                'Calculated_Final_Conc_mM': calculated_final_conc,
                'Error_Percent': error_percent,
                'Is_Stock': solution['is_stock']
            })
    
    for i, target_conc in enumerate(target_concs_b):
        if target_conc in plan_b['concentration_map']:
            solution = plan_b['concentration_map'][target_conc]
            
            # Calculate final concentration in 200 µL well
            solution_conc_mm = solution['concentration_mm'] 
            solution_volume_ul = solution['volume_needed_ul']
            
            # Final concentration calculation
            calculated_final_conc = (solution_conc_mm * solution_volume_ul) / WELL_VOLUME_UL
            
            # Water volume for this surfactant
            water_volume_ul = EFFECTIVE_SURFACTANT_VOLUME - solution_volume_ul
            
            # Calculate error
            error_percent = ((calculated_final_conc - target_conc) / target_conc) * 100 if target_conc > 0 else 0
            
            verification_data.append({
                'Surfactant': surfactant_b_name,
                'Target_Conc_mM': target_conc,
                'Solution_Vial': solution['vial_name'],
                'Solution_Conc_mM': solution_conc_mm,
                'Solution_Volume_uL': solution_volume_ul,
                'Water_Volume_uL': water_volume_ul,
                'Total_Surfactant_Volume_uL': solution_volume_ul + water_volume_ul,
                'Calculated_Final_Conc_mM': calculated_final_conc,
                'Error_Percent': error_percent,
                'Is_Stock': solution['is_stock']
            })
    
    # Create DataFrame and export
    df = pd.DataFrame(verification_data)
    
    # Add well composition summary
    well_summary = pd.DataFrame([{
        'Component': 'Surfactant A',
        'Volume_uL': EFFECTIVE_SURFACTANT_VOLUME,
        'Notes': 'Solution + Water'
    }, {
        'Component': 'Surfactant B', 
        'Volume_uL': EFFECTIVE_SURFACTANT_VOLUME,
        'Notes': 'Solution + Water'
    }, {
        'Component': 'Buffer',
        'Volume_uL': BUFFER_VOLUME_UL if ADD_BUFFER else 0,
        'Notes': SELECTED_BUFFER if ADD_BUFFER else 'None'
    }, {
        'Component': 'Pyrene',
        'Volume_uL': PYRENE_VOLUME_UL,
        'Notes': 'Added later'
    }, {
        'Component': 'TOTAL',
        'Volume_uL': WELL_VOLUME_UL,
        'Notes': 'Target well volume'
    }])
    
    # Export to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conc_file = f"output/concentration_verification_{timestamp}.csv"
    well_file = f"output/well_composition_{timestamp}.csv"
    
    os.makedirs("output", exist_ok=True)
    df.to_csv(conc_file, index=False)
    well_summary.to_csv(well_file, index=False)
    
    # Print verification summary
    max_error = df['Error_Percent'].abs().max()
    avg_error = df['Error_Percent'].abs().mean()
    
    print(f"Verification complete:")
    print(f"  Maximum error: {max_error:.2f}%")
    print(f"  Average error: {avg_error:.2f}%")
    print(f"  Exported detailed calculations to: {conc_file}")
    print(f"  Exported well composition to: {well_file}")
    
    # Show any large errors
    large_errors = df[df['Error_Percent'].abs() > 5]
    if len(large_errors) > 0:
        print(f"  WARNING: {len(large_errors)} concentrations have >5% error!")
        for _, row in large_errors.iterrows():
            print(f"    {row['Surfactant']} {row['Target_Conc_mM']:.2e} mM: {row['Error_Percent']:.1f}% error")
    else:
        print(f"  + All concentration errors < 5%")
    
    return df, well_summary

# ================================================================================
# SECTION 2: CONCENTRATION AND GRID CALCULATION FUNCTIONS
# ================================================================================

#DESCRIPTION: Calculate adaptive logarithmic concentration grid for surfactant dilution series
#RATING: 9/10 - Adaptive approach that optimizes concentration range for each surfactant
def calculate_grid_concentrations(surfactant_name=None):
    """
    Calculate adaptive concentration grid points for surfactants.
    Each surfactant gets its own optimized range: min_conc to stock_conc/2.
    
    Args:
        surfactant_name: Name of surfactant (if None, uses generic range)
        
    Returns:
        numpy.array: Concentration values in mM
    """
    if surfactant_name and surfactant_name in SURFACTANT_LIBRARY:
        stock_conc = SURFACTANT_LIBRARY[surfactant_name]["stock_conc"]
        # Maximum concentration is limited by volume allocation in the well
        # Max achievable: stock_conc * (allocated_volume / total_well_volume)
        max_conc = stock_conc * (EFFECTIVE_SURFACTANT_VOLUME / WELL_VOLUME_UL)
    else:
        # Generic range if no surfactant specified
        max_conc = 25  # Generic max for calculations
    
    # Create logarithmic spacing from MIN_CONC to max_conc
    # np.logspace(start, stop, num) where start and stop are log10 values
    log_min = np.log10(MIN_CONC)
    log_max = np.log10(max_conc)
    
    concentrations = np.logspace(log_min, log_max, NUMBER_CONCENTRATIONS)
    
    print(f"Adaptive grid for {surfactant_name or 'generic'}: {MIN_CONC:.1e} to {max_conc:.1f} mM ({NUMBER_CONCENTRATIONS} steps)")
    print(f"  Concentrations: {[f'{c:.3e}' for c in concentrations]}")
    
    return concentrations

#DESCRIPTION: Calculate concentration grids for both surfactants with adaptive ranges
#RATING: 10/10 - Clean interface for dual-surfactant experiments
def calculate_dual_surfactant_grids(surfactant_a_name, surfactant_b_name):
    """
    Calculate optimized concentration grids for both surfactants.
    
    Args:
        surfactant_a_name: Name of first surfactant
        surfactant_b_name: Name of second surfactant
        
    Returns:
        tuple: (concentrations_a, concentrations_b)
    """
    concs_a = calculate_grid_concentrations(surfactant_a_name)
    concs_b = calculate_grid_concentrations(surfactant_b_name)
    
    print(f"\nAdaptive concentration grid summary:")
    print(f"  {surfactant_a_name}: {len(concs_a)} concentrations from {concs_a[0]:.3e} to {concs_a[-1]:.3e} mM")
    print(f"  {surfactant_b_name}: {len(concs_b)} concentrations from {concs_b[0]:.3e} to {concs_b[-1]:.3e} mM")
    print(f"  Total combinations: {len(concs_a)} × {len(concs_b)} = {len(concs_a) * len(concs_b)} wells (× {N_REPLICATES} replicates = {len(concs_a) * len(concs_b) * N_REPLICATES} total wells)")
    
    return concs_a, concs_b

#DESCRIPTION: Load existing substock dilutions from CSV and track what's already prepared
#RATING: 7/10 - Comprehensive but complex parsing logic, handles edge cases well  
def load_substock_tracking(logger=None):
    """Create substock tracking, detecting existing dilutions from CSV file."""
    tracking = {}
    
    # Initialize tracking structure
    for surfactant in SURFACTANT_LIBRARY.keys():
        tracking[surfactant] = {
            "stock_available": True,
            "dilutions_created": set(),  # Set of concentration values that have been created
            "dilution_recipes": {}  # Dictionary to store full dilution recipe information by concentration
        }
        
        # Calculate adaptive concentrations for this specific surfactant
        concentrations = calculate_grid_concentrations(surfactant)
        
        # Read CSV file to detect existing dilutions based on volume
        try:
            vial_file_path = "status/surfactant_grid_vials_expanded.csv"
            if os.path.exists(vial_file_path):
                import pandas as pd
                df = pd.read_csv(vial_file_path)
                
                if logger:
                    logger.info(f"Checking for existing {surfactant} substocks in CSV...")
                else:
                    print(f"Checking for existing {surfactant} substocks in CSV...")
                
                for _, row in df.iterrows():
                    vial_name = row['vial_name']
                    volume = float(row['vial_volume']) if pd.notna(row['vial_volume']) else 0.0
                    
                    # Skip if no volume (empty vial)
                    if volume <= 0:
                        continue
                    
                    # Parse dilution vials to extract surfactant and concentration
                    if vial_name.startswith(f"{surfactant}_dilution_"):
                        try:
                            # Extract dilution index from vial name
                            dilution_idx = int(vial_name.split("_")[-1])
                            
                            # Map dilution index to concentration using this surfactant's grid
                            if dilution_idx < len(concentrations):
                                target_conc = concentrations[dilution_idx]
                                tracking[surfactant]["dilutions_created"].add(target_conc)
                                if logger:
                                    logger.info(f"  Found existing {surfactant} dilution: {vial_name} = {target_conc:.2e} mM ({volume:.1f} mL)")
                                else:
                                    print(f"  Found existing {surfactant} dilution: {vial_name} = {target_conc:.2e} mM ({volume:.1f} mL)")
                        except (ValueError, IndexError):
                            continue
        
        except Exception as e:
            if logger:
                logger.warning(f"Could not load existing {surfactant} dilutions from CSV: {e}")
            else:
                print(f"Warning: Could not load existing {surfactant} dilutions from CSV: {e}")
    
    return tracking

# Continue with all the other functions from the original file...
# Let me add the key functions that need to be updated for the adaptive approach

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
        if target is None:
            achievable.append(None)
            continue
        
        # Apply concentration correction factor for buffer dilution
        required_stock_conc = target * CONCENTRATION_CORRECTION_FACTOR
        
        # Check if this concentration is achievable given stock concentration and dilution limits  
        dilution_factor = stock_conc / required_stock_conc
        if dilution_factor >= 1.0:  # Only check that we're not trying to concentrate (dilution_factor >= 1)
            achievable.append(target)
        else:
            achievable.append(None)
    
    return achievable

def create_concentration_grid_summary(concs_a, concs_b, dilution_vials_a, dilution_vials_b, surfactant_a_name, surfactant_b_name):
    """
    Create a text-based visual summary showing which concentration combinations are achievable.
    Updated to handle different concentration arrays for each surfactant.
    
    Args:
        concs_a: Concentrations for surfactant A
        concs_b: Concentrations for surfactant B
        dilution_vials_a: List of vial names for surfactant A (None for non-achievable)
        dilution_vials_b: List of vial names for surfactant B (None for non-achievable)
        surfactant_a_name: Name of surfactant A
        surfactant_b_name: Name of surfactant B
        
    Returns:
        str: Text grid showing achievable combinations
    """
    grid_lines = []
    grid_lines.append(f"ADAPTIVE CONCENTRATION GRID SUMMARY: {surfactant_a_name} + {surfactant_b_name}")
    grid_lines.append("=" * 70)
    grid_lines.append("Legend: X = achievable combination, . = not achievable")
    grid_lines.append(f"Grid size: {len(concs_a)} × {len(concs_b)} = {len(concs_a) * len(concs_b)} combinations")
    grid_lines.append("")
    
    # Surfactant concentration ranges
    grid_lines.append(f"{surfactant_a_name}: {concs_a[0]:.1e} to {concs_a[-1]:.1e} mM (stock: {SURFACTANT_LIBRARY[surfactant_a_name]['stock_conc']} mM)")
    grid_lines.append(f"{surfactant_b_name}: {concs_b[0]:.1e} to {concs_b[-1]:.1e} mM (stock: {SURFACTANT_LIBRARY[surfactant_b_name]['stock_conc']} mM)")
    grid_lines.append("")
    
    # Create header with surfactant B concentrations (columns)
    header = f"{'':>12}"  # Space for row labels
    for j, conc_b in enumerate(concs_b):
        if dilution_vials_b[j] is not None:
            header += f"{conc_b:.1e}".rjust(12)
        else:
            header += f"{'(N/A)':<12}"
    grid_lines.append(f"{surfactant_b_name} (mM) ->")
    grid_lines.append(header)
    grid_lines.append("-" * len(header))
    
    # Create rows for each surfactant A concentration
    for i, conc_a in enumerate(concs_a):
        if dilution_vials_a[i] is not None:
            row_label = f"{conc_a:.1e}".rjust(10) + " |"
        else:
            row_label = f"{'(N/A)':<10} |"
        
        row = row_label
        for j, conc_b in enumerate(concs_b):
            # Check if both concentrations are achievable
            if dilution_vials_a[i] is not None and dilution_vials_b[j] is not None:
                symbol = "X"
            else:
                symbol = "."
            row += f"{symbol:>12}"
        grid_lines.append(row)
    
    grid_lines.append("")
    achievable_count = len([1 for a in dilution_vials_a for b in dilution_vials_b if a is not None and b is not None])
    total_count = len(concs_a) * len(concs_b)
    grid_lines.append(f"Achievable combinations: {achievable_count}/{total_count}")
    
def create_control_wells(surfactant_a_name, surfactant_b_name, position_prefix="start"):
    """
    Create quality control wells for start/end of experiment.
    Returns list of control well requirements.
    """
    controls = []
    
    # Control 1: Surfactant A stock only (200 μL)
    controls.append({
        'control_type': f'{position_prefix}_control_surfactant_A',
        'description': f'{surfactant_a_name} stock (200 μL)',
        'dilution_a_vial': f'{surfactant_a_name}_stock',
        'dilution_b_vial': None,
        'volume_a_ul': 200,
        'volume_b_ul': 0,
        'conc_a': None,  # Full stock concentration
        'conc_b': 0,
        'replicate': 1,
        'is_control': True
    })
    
    # Control 2: Surfactant B stock only (200 μL)  
    controls.append({
        'control_type': f'{position_prefix}_control_surfactant_B',
        'description': f'{surfactant_b_name} stock (200 μL)',
        'dilution_a_vial': None,
        'dilution_b_vial': f'{surfactant_b_name}_stock',
        'volume_a_ul': 0,
        'volume_b_ul': 200,
        'conc_a': 0,
        'conc_b': None,  # Full stock concentration
        'replicate': 1,
        'is_control': True
    })
    
    # Control 3: Buffer only (if using buffer)
    if ADD_BUFFER:
        controls.append({
            'control_type': f'{position_prefix}_control_buffer',
            'description': f'Buffer ({SELECTED_BUFFER}, 200 μL)',
            'dilution_a_vial': None,
            'dilution_b_vial': None,
            'volume_a_ul': 0,
            'volume_b_ul': 0,
            'conc_a': 0,
            'conc_b': 0,
            'buffer_only': True,
            'replicate': 1,
            'is_control': True
        })
    
    # Control 4: Water only (200 μL)
    controls.append({
        'control_type': f'{position_prefix}_control_water',
        'description': 'Water blank (200 μL)',
        'dilution_a_vial': None,
        'dilution_b_vial': None,
        'volume_a_ul': 0,
        'volume_b_ul': 0,
        'conc_a': 0,
        'conc_b': 0,
        'water_only': True,
        'replicate': 1,
        'is_control': True
    })
    
    return controls

# Now let me add the controls to the main pipetting function



# Now let me add the complete corrected pipetting function
def pipette_grid_to_shared_wellplate(lash_e, concs_a, concs_b, plan_a, plan_b, surfactant_a_name, surfactant_b_name, shared_wellplate_state):
    """Pipette concentration grid into wellplate(s) with CORRECTED phase order for proper buffer timing."""
    logger = lash_e.logger
    logger.info(f"Starting grid pipetting: {len(concs_a)}x{len(concs_b)} grid with {N_REPLICATES} replicates each")
    
    well_counter = shared_wellplate_state.get('global_well_counter', 0)
    well_map = []  
    total_wells_added = shared_wellplate_state.get('total_wells_added', 0)
    
    # Use existing wellplate tracking from shared state
    wellplate_data = {
        'current_plate': shared_wellplate_state.get('current_plate', 1),
        'wells_used': shared_wellplate_state.get('wells_used', 0),
        'measurements': [],  
        'last_measured_well': shared_wellplate_state.get('last_measured_well', -1)
    }
    
    # Generate all well requirements using smart dilution plans
    
    # START: Add control wells at beginning
    start_controls = create_control_wells(surfactant_a_name, surfactant_b_name, "start")
    all_well_requirements = start_controls.copy()
    
    # MIDDLE: Add grid experiment wells  
    grid_wells = []
    for i, conc_a in enumerate(concs_a):
        for j, conc_b in enumerate(concs_b):
            # Check if both concentrations are achievable using the plans
            solution_a = plan_a['concentration_map'].get(conc_a)
            solution_b = plan_b['concentration_map'].get(conc_b)
            
            if solution_a is None or solution_b is None:
                conc_a_str = f"{conc_a:.2e}" if conc_a is not None else "None"
                conc_b_str = f"{conc_b:.2e}" if conc_b is not None else "None"
                logger.debug(f"Skipping combination [{i},{j}]: {conc_a_str} + {conc_b_str} mM (no solution available)")
                continue
                
            for rep in range(N_REPLICATES):
                grid_wells.append({
                    'conc_a': concs_a[i],
                    'conc_b': concs_b[j], 
                    'conc_a_idx': i,
                    'conc_b_idx': j,
                    'dilution_a_vial': solution_a['vial_name'],
                    'dilution_b_vial': solution_b['vial_name'],
                    'volume_a_ul': solution_a['volume_needed_ul'],
                    'volume_b_ul': solution_b['volume_needed_ul'],
                    'replicate': rep + 1,
                    'is_control': False
                })
    
    all_well_requirements.extend(grid_wells)
    
    # END: Add control wells at end
    end_controls = create_control_wells(surfactant_a_name, surfactant_b_name, "end")
    all_well_requirements.extend(end_controls)
    
    # Report on achievable vs total combinations
    total_possible_wells = len(concs_a) * len(concs_b) * N_REPLICATES
    achievable_grid_wells = len(grid_wells)
    skipped_wells = total_possible_wells - achievable_grid_wells
    total_controls = len(start_controls) + len(end_controls)
    total_wells_with_controls = achievable_grid_wells + total_controls
    
    logger.info(f"Grid summary: {achievable_grid_wells}/{total_possible_wells} grid wells achievable ({skipped_wells} skipped)")
    logger.info(f"Controls: {len(start_controls)} start + {len(end_controls)} end = {total_controls} control wells")
    logger.info(f"Total wells: {total_wells_with_controls} (grid + controls)")
    
    if achievable_grid_wells == 0:
        logger.warning("No achievable concentration combinations found! Check stock concentrations and target ranges.")
        # Still continue with controls only
        if total_controls == 0:
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
        
        # PHASE 1: Add all surfactant A solutions (with integrated vial optimization)
        print(f"  Phase 1: Adding surfactant A (including controls)")
        
        from collections import defaultdict
        vial_a_groups = defaultdict(list)
        control_wells_a = defaultdict(list)  # Track control wells separately
        
        # Track actual dispensed volumes for each well
        actual_volumes_dispensed = {}  # {well_idx: {'volume_a': X, 'volume_b': Y}}
        
        for batch_idx, req in enumerate(current_batch):
            actual_well = wells_in_batch[batch_idx]
            
            # Initialize tracking for this well
            actual_volumes_dispensed[actual_well] = {
                'volume_a_ul': req['volume_a_ul'],
                'volume_b_ul': req['volume_b_ul'],
                'volume_a_dispensed': 0,
                'volume_b_dispensed': 0
            }
            
            # Group by vial type (regular vs control)
            if req.get('is_control', False):
                if req['volume_a_ul'] > 0 and req['dilution_a_vial']:
                    control_wells_a[req['dilution_a_vial']].append((batch_idx, actual_well, req))
            else:
                # Regular grid wells
                if req['volume_a_ul'] > 0:
                    vial_a_groups[req['dilution_a_vial']].append((batch_idx, actual_well, req))
        
        # Handle control wells first (they use stock concentrations)
        print(f"    Processing {sum(len(wells) for wells in control_wells_a.values())} control wells for surfactant A")
        for vial_name, well_list in control_wells_a.items():
            for batch_idx, actual_well, req in well_list:
                volume_each_ml = req['volume_a_ul'] / 1000
                print(f"      Control well {actual_well}: {req['volume_a_ul']}μL {vial_name}")
                
                lash_e.nr_robot.aspirate_from_vial(vial_name, volume_each_ml, liquid='water', use_safe_location=True)
                lash_e.nr_robot.dispense_into_wellplate(
                    dest_wp_num_array=[actual_well], 
                    amount_mL_array=[volume_each_ml],
                    liquid='water'
                )
                lash_e.nr_robot.remove_pipet()
                
                actual_volumes_dispensed[actual_well]['volume_a_dispensed'] = req['volume_a_ul']
        
        # Handle regular grid wells (existing logic)
        print(f"    Processing {sum(len(wells) for wells in vial_a_groups.values())} grid wells for surfactant A")
        print(f"    Moving surfactant A vials to optimal positions for better access...")
        
        # Move surfactant A vials to optimal positions (40-47)
        vial_a_original_positions = {}
        optimal_position = 40
        unique_vials_a = list(vial_a_groups.keys())
        
        for vial_name in unique_vials_a:
            try:
                # Store original position
                current_location = lash_e.nr_robot.get_vial_info(vial_name, 'location')
                current_index = lash_e.nr_robot.get_vial_info(vial_name, 'location_index')
                vial_a_original_positions[vial_name] = {'location': current_location, 'location_index': current_index}
                
                # Move to optimal position
                lash_e.nr_robot.move_vial_to_location(vial_name, 'main_8mL_rack', optimal_position)
                print(f"      Moved {vial_name} to position {optimal_position}")
                optimal_position += 1
            except Exception as e:
                print(f"      Warning: Could not move {vial_name}: {e}")
        
        # Sort vial groups by concentration (low->high to prevent contamination)
        sorted_vial_groups_a = sorted(vial_a_groups.items(), 
                                     key=lambda x: current_batch[x[1][0][0]]['conc_a'] or 0)
        
        # Pipette surfactant A
        surfactant_a_tip_conditioned = False
        for vial_name, well_list in sorted_vial_groups_a:
            print(f"    Using vial {vial_name} for {len(well_list)} wells")
            
            # Condition tip on first surfactant A vial
            if not surfactant_a_tip_conditioned:
                condition_tip(lash_e, vial_name)
                surfactant_a_tip_conditioned = True
            
            for batch_idx, actual_well, req in well_list:
                volume_each_ml = req['volume_a_ul'] / 1000  # Convert µL to mL
                
                # Track actual dispensed volume
                actual_volumes_dispensed[actual_well]['volume_a_dispensed'] = req['volume_a_ul']
                
                # Use safe location (handles vial movement properly)
                lash_e.nr_robot.aspirate_from_vial(req['dilution_a_vial'], volume_each_ml, liquid='water', use_safe_location=True)
                lash_e.nr_robot.dispense_into_wellplate(
                    dest_wp_num_array=[actual_well], 
                    amount_mL_array=[volume_each_ml],
                    liquid='water'
                )
            # Return vial home and remove tip after each vial to allow safe movement
        lash_e.nr_robot.remove_pipet()
        
        # Return surfactant A vials to original positions
        print(f"    Returning surfactant A vials to home positions...")
        for vial_name, position_info in vial_a_original_positions.items():
            try:
                lash_e.nr_robot.move_vial_to_location(vial_name, position_info['location'], position_info['location_index'])
                print(f"      Restored {vial_name} to home position")
            except Exception as e:
                print(f"      Warning: Could not restore {vial_name}: {e}")
            
        # PHASE 2: Add all surfactant B solutions (including controls)
        print(f"  Phase 2: Adding surfactant B (including controls)")
        
        vial_b_groups = defaultdict(list)
        control_wells_b = defaultdict(list)  # Track control wells separately
        
        for batch_idx, req in enumerate(current_batch):
            actual_well = wells_in_batch[batch_idx]
            
            # Group by vial type (regular vs control)
            if req.get('is_control', False):
                if req['volume_b_ul'] > 0 and req['dilution_b_vial']:
                    control_wells_b[req['dilution_b_vial']].append((batch_idx, actual_well, req))
            else:
                # Regular grid wells
                if req['volume_b_ul'] > 0:
                    vial_b_groups[req['dilution_b_vial']].append((batch_idx, actual_well, req))
        
        # Handle control wells first (they use stock concentrations)
        print(f"    Processing {sum(len(wells) for wells in control_wells_b.values())} control wells for surfactant B")
        for vial_name, well_list in control_wells_b.items():
            for batch_idx, actual_well, req in well_list:
                volume_each_ml = req['volume_b_ul'] / 1000
                print(f"      Control well {actual_well}: {req['volume_b_ul']}μL {vial_name}")
                
                lash_e.nr_robot.aspirate_from_vial(vial_name, volume_each_ml, liquid='water', use_safe_location=True)
                lash_e.nr_robot.dispense_into_wellplate(
                    dest_wp_num_array=[actual_well], 
                    amount_mL_array=[volume_each_ml],
                    liquid='water'
                )
                lash_e.nr_robot.remove_pipet()
                
                actual_volumes_dispensed[actual_well]['volume_b_dispensed'] = req['volume_b_ul']
        
        # Handle regular grid wells (existing logic)
        print(f"    Processing {sum(len(wells) for wells in vial_b_groups.values())} grid wells for surfactant B")
        print(f"    Moving surfactant B vials to optimal positions for better access...")
        
        # Move surfactant B vials to optimal positions (40-47)
        vial_b_original_positions = {}
        optimal_position = 40
        unique_vials_b = list(vial_b_groups.keys())
        
        for vial_name in unique_vials_b:
            try:
                # Store original position
                current_location = lash_e.nr_robot.get_vial_info(vial_name, 'location')
                current_index = lash_e.nr_robot.get_vial_info(vial_name, 'location_index')
                vial_b_original_positions[vial_name] = {'location': current_location, 'location_index': current_index}
                
                # Move to optimal position
                lash_e.nr_robot.move_vial_to_location(vial_name, 'main_8mL_rack', optimal_position)
                print(f"      Moved {vial_name} to position {optimal_position}")
                optimal_position += 1
            except Exception as e:
                print(f"      Warning: Could not move {vial_name}: {e}")
        
        # Sort vial groups by concentration (low->high to prevent contamination)
        sorted_vial_groups_b = sorted(vial_b_groups.items(), 
                                     key=lambda x: current_batch[x[1][0][0]]['conc_b'] or 0)
        
        # Pipette surfactant B
        surfactant_b_tip_conditioned = False
        for vial_name, well_list in sorted_vial_groups_b:
            print(f"    Using vial {vial_name} for {len(well_list)} wells")
            
            # Condition tip on first surfactant B vial
            if not surfactant_b_tip_conditioned:
                condition_tip(lash_e, vial_name)
                surfactant_b_tip_conditioned = True
            
            for batch_idx, actual_well, req in well_list:
                # FIXED: Use calculated volume from smart dilution plan, not fixed volume
                volume_each_ml = req['volume_b_ul'] / 1000  # Convert µL to mL (was incorrectly using EFFECTIVE_SURFACTANT_VOLUME)
                
                # Track actual dispensed volume
                actual_volumes_dispensed[actual_well]['volume_b_dispensed'] = req['volume_b_ul']
                
                # Use safe location (handles vial movement properly)
                lash_e.nr_robot.aspirate_from_vial(req['dilution_b_vial'], volume_each_ml, liquid='water', use_safe_location=True)
                lash_e.nr_robot.dispense_into_wellplate(
                    dest_wp_num_array=[actual_well], 
                    amount_mL_array=[volume_each_ml],
                    liquid='water'
                )
        # Remove tip first, then return vial home to allow safe movement
        lash_e.nr_robot.remove_pipet()
        
        # Return surfactant B vials to original positions
        print(f"    Returning surfactant B vials to home positions...")
        for vial_name, position_info in vial_b_original_positions.items():
            try:
                lash_e.nr_robot.move_vial_to_location(vial_name, position_info['location'], position_info['location_index'])
                print(f"      Restored {vial_name} to home position")
            except Exception as e:
                print(f"      Warning: Could not restore {vial_name}: {e}")
        
        # PHASE 3: Add remaining water to fill wells (including control wells)
        print(f"  Phase 3: Adding water (controls and grid wells)")
        
        # Move water vial to optimal position for efficient access
        try:
            water_original_location = lash_e.nr_robot.get_vial_info('water', 'location')
            water_original_index = lash_e.nr_robot.get_vial_info('water', 'location_index')
            optimal_position = 46  # Use rack position for optimal access
            lash_e.nr_robot.move_vial_to_location('water', 'main_8mL_rack', optimal_position)
            print(f"    Moved water vial from {water_original_location} to position {optimal_position}")
        except Exception as e:
            print(f"    Warning: Could not move water vial to optimal position: {e}")
            water_original_location = None
        
        # Calculate water needed for each well individually
        for well_idx in wells_in_batch:
            batch_idx = wells_in_batch.index(well_idx)
            req = current_batch[batch_idx]
            volume_tracking = actual_volumes_dispensed[well_idx]
            
            # Handle special control wells
            if req.get('is_control', False):
                if req.get('water_only', False):
                    # Water-only control: add full 200 μL water
                    print(f"    Control well {well_idx}: 200μL water (water blank)")
                    lash_e.nr_robot.aspirate_from_vial('water', 0.2, liquid='water', use_safe_location=True)
                    lash_e.nr_robot.dispense_into_wellplate(
                        dest_wp_num_array=[well_idx], 
                        amount_mL_array=[0.2],
                        liquid='water'
                    )
                    # No pipet removal - keep for back-and-forth efficiency
                    continue
                elif req.get('buffer_only', False):
                    # Buffer-only control: no water needed here (buffer added in Phase 4)
                    print(f"    Control well {well_idx}: No water (buffer-only control)")
                    continue
                else:
                    # Surfactant control: no additional water needed (already 200 μL)
                    print(f"    Control well {well_idx}: No additional water needed")
                    continue
            
            # Regular grid wells: calculate water needed
            total_dispensed_ul = (volume_tracking['volume_a_dispensed'] + 
                                volume_tracking['volume_b_dispensed'])
            
            # Calculate remaining volume needed (excluding pyrene which comes later)
            target_volume_before_pyrene = WELL_VOLUME_UL - PYRENE_VOLUME_UL
            # Only exclude buffer volume if buffer will actually be added
            if ADD_BUFFER:
                target_volume_before_buffer = target_volume_before_pyrene - BUFFER_VOLUME_UL
            else:
                # Include buffer volume as water when buffer is disabled
                target_volume_before_buffer = target_volume_before_pyrene
            
            # Calculate water needed for this specific well
            water_needed_ul = target_volume_before_buffer - total_dispensed_ul
            
            if water_needed_ul > 0:
                water_volume_ml = water_needed_ul / 1000
                print(f"    Well {well_idx}: {total_dispensed_ul:.1f}µL surfactants + {water_needed_ul:.1f}µL water = {total_dispensed_ul + water_needed_ul:.1f}µL")
                
                # Add water to this specific well
                lash_e.nr_robot.aspirate_from_vial('water', water_volume_ml, liquid='water', use_safe_location=True)
                lash_e.nr_robot.dispense_into_wellplate(
                    dest_wp_num_array=[well_idx], 
                    amount_mL_array=[water_volume_ml],
                    liquid='water'
                )
                # No pipet removal - keep for back-and-forth efficiency
            
            elif water_needed_ul < 0:
                print(f"    WARNING: Well {well_idx} has {abs(water_needed_ul):.1f}µL excess volume!")
            else:
                print(f"    Well {well_idx}: No additional water needed")
        
        # Return water vial to original position after all wells processed
        lash_e.nr_robot.remove_pipet()
        if water_original_location:
            try:
                lash_e.nr_robot.move_vial_to_location('water', water_original_location, water_original_index)
                print(f"    Restored water vial to original position {water_original_location}")
            except Exception as e:
                print(f"    Warning: Could not restore water vial position: {e}")
        else:
            lash_e.nr_robot.return_vial_home('water')

        # PHASE 4: Add buffer (if enabled and available)
        if ADD_BUFFER:
            print(f"  Phase 4: Adding {SELECTED_BUFFER} buffer (using small tips, back-and-forth)")
            buffer_volume_ml = BUFFER_VOLUME_UL / 1000  # Convert to mL
            
            # Validate buffer vial exists
            buffer_found = False
            try:
                lash_e.nr_robot.get_vial_info(SELECTED_BUFFER, 'location')
                buffer_found = True
            except:
                logger.warning(f"Buffer vial '{SELECTED_BUFFER}' not found in vial status")
                print(f"Warning: Buffer vial '{SELECTED_BUFFER}' not found - continuing without buffer")
            
            if buffer_found:
                # Move buffer vial to optimal position before dispensing
                try:
                    buffer_original_location = lash_e.nr_robot.get_vial_info(SELECTED_BUFFER, 'location')
                    buffer_original_index = lash_e.nr_robot.get_vial_info(SELECTED_BUFFER, 'location_index')
                    print(f"    Moving {SELECTED_BUFFER} to optimal position for small-tip access...")
                    lash_e.nr_robot.move_vial_to_location(SELECTED_BUFFER, 'main_8mL_rack', 47)  # Last position
                except Exception as e:
                    print(f"    Warning: Could not move {SELECTED_BUFFER} vial: {e}")
                    buffer_original_location = None
                
                # Condition tip with buffer before first use
                condition_tip(lash_e, SELECTED_BUFFER)
                
                # Separate regular wells from buffer-only controls
                regular_wells = []
                buffer_only_wells = []
                
                for i, req in enumerate(current_batch):
                    well_idx = wells_in_batch[i]
                    if req.get('is_control', False) and req.get('buffer_only', False):
                        buffer_only_wells.append(well_idx)
                    else:
                        regular_wells.append(well_idx)
                
                # Add regular buffer amount to regular wells using back-and-forth with small tips
                if regular_wells:
                    print(f"    Adding {BUFFER_VOLUME_UL}μL buffer to {len(regular_wells)} regular wells (back-and-forth)")
                    for well_idx in regular_wells:
                        lash_e.nr_robot.aspirate_from_vial(SELECTED_BUFFER, buffer_volume_ml, liquid='water', use_safe_location=True)
                        lash_e.nr_robot.dispense_into_wellplate(
                            dest_wp_num_array=[well_idx], 
                            amount_mL_array=[buffer_volume_ml],
                            liquid='water'
                        )
                        # No pipet removal - keep for back-and-forth efficiency
                
                # Add full 200μL buffer to buffer-only control wells using back-and-forth with small tips
                if buffer_only_wells:
                    print(f"    Adding 200μL buffer to {len(buffer_only_wells)} buffer-only control wells (back-and-forth)")
                    for well_idx in buffer_only_wells:
                        lash_e.nr_robot.aspirate_from_vial(SELECTED_BUFFER, 0.2, liquid='water', use_safe_location=True)
                        lash_e.nr_robot.dispense_into_wellplate(
                            dest_wp_num_array=[well_idx], 
                            amount_mL_array=[0.2],
                            liquid='water'
                        )
                        # No pipet removal - keep for back-and-forth efficiency
                        print(f"      Buffer control well {well_idx}: 200μL {SELECTED_BUFFER}")
                
                # Remove pipet after completing all buffer dispensing
                lash_e.nr_robot.remove_pipet()
                
                # Return buffer vial to home position
                if buffer_original_location:
                    try:
                        lash_e.nr_robot.move_vial_to_location(SELECTED_BUFFER, buffer_original_location, buffer_original_index)
                        print(f"    Returned {SELECTED_BUFFER} to home position")
                    except Exception as e:
                        print(f"    Warning: Could not return {SELECTED_BUFFER}: {e}")
                else:
                    lash_e.nr_robot.return_vial_home(SELECTED_BUFFER)
        
        # Record well information for this batch
        for i, req in enumerate(current_batch):
            actual_well = wells_in_batch[i]
            well_map.append({
                'well': actual_well,
                'plate': wellplate_data['current_plate'],
                'surfactant_a': surfactant_a_name,
                'surfactant_b': surfactant_b_name,
                'conc_a_mm': req.get('conc_a'),
                'conc_b_mm': req.get('conc_b'),
                'replicate': req.get('replicate', 1),
                'vial_a': req.get('dilution_a_vial'),
                'vial_b': req.get('dilution_b_vial'),
                'is_control': req.get('is_control', False),
                'control_type': req.get('control_type', 'experiment')
            })
        
        # PHASE 5: Measure turbidity AFTER buffer addition (CORRECTED ORDER)
        print(f"  Phase 5: Measuring turbidity for wells {wells_in_batch[0]}-{wells_in_batch[-1]} (with buffer present)")
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
        
        # PHASE 6: Add pyrene_DMSO to all wells (using small tips, back-and-forth)
        print(f"  Phase 6: Adding pyrene_DMSO ({PYRENE_VOLUME_UL}μL) using small tips, back-and-forth")
        pyrene_volume_ml = PYRENE_VOLUME_UL / 1000  # Convert to mL
        
        # Move pyrene_DMSO vial to optimal position before dispensing
        try:
            pyrene_original_location = lash_e.nr_robot.get_vial_info('pyrene_DMSO', 'location')
            pyrene_original_index = lash_e.nr_robot.get_vial_info('pyrene_DMSO', 'location_index')
            print(f"    Moving pyrene_DMSO to optimal position for small-tip access...")
            lash_e.nr_robot.move_vial_to_location('pyrene_DMSO', 'main_8mL_rack', 46)  # Second-to-last position
        except Exception as e:
            print(f"    Warning: Could not move pyrene_DMSO vial: {e}")
            pyrene_original_location = None
        
        # Condition tip with DMSO before first use
        condition_tip(lash_e, 'pyrene_DMSO')
        
        # Add pyrene to each well individually using back-and-forth with small tips
        print(f"    Dispensing to {len(wells_in_batch)} wells individually for accuracy...")
        for well_idx in wells_in_batch:
            lash_e.nr_robot.aspirate_from_vial('pyrene_DMSO', pyrene_volume_ml, liquid='DMSO', use_safe_location=True)
            lash_e.nr_robot.dispense_into_wellplate(
                dest_wp_num_array=[well_idx], 
                amount_mL_array=[pyrene_volume_ml],
                liquid='DMSO'
            )
            # No pipet removal - keep for back-and-forth efficiency
        
        # Remove pipet after completing all pyrene dispensing
        lash_e.nr_robot.remove_pipet()
        
        # Return pyrene_DMSO vial to home position
        if pyrene_original_location:
            try:
                lash_e.nr_robot.move_vial_to_location('pyrene_DMSO', pyrene_original_location, pyrene_original_index)
                print(f"    Returned pyrene_DMSO to home position")
            except Exception as e:
                print(f"    Warning: Could not return pyrene_DMSO: {e}")
        else:
            lash_e.nr_robot.return_vial_home('pyrene_DMSO')
        
        # PHASE 7: Measure fluorescence after pyrene addition
        print(f"  Phase 7: Measuring fluorescence for wells {wells_in_batch[0]}-{wells_in_batch[-1]}")
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
        
        print(f"  + Completed batch with {len(current_batch)} wells using optimal tip sequence")
    
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

# Add stubs for the required measurement and helper functions
def manage_wellplate_switching(lash_e, current_well, wellplate_data):
    """Handle wellplate switching when MAX_WELLS is reached."""
    if current_well >= MAX_WELLS:
        # Switch to new plate logic would go here
        wellplate_data['current_plate'] += 1
        wellplate_data['wells_used'] = 0
        wellplate_data['last_measured_well'] = -1
        return 0, wellplate_data
    else:
        wellplate_data['wells_used'] = max(wellplate_data['wells_used'], current_well + 1)
        return current_well, wellplate_data

def measure_turbidity(lash_e, well_indices):
    """Measure turbidity using Cytation plate reader."""
    print(f"Measuring turbidity in wells {well_indices}...")
    # In real implementation, this calls lash_e.measure_wellplate()
    # For now, return mock data
    return {'turbidity': [0.5] * len(well_indices)}

def measure_fluorescence(lash_e, well_indices):
    """Measure fluorescence using Cytation plate reader."""
    print(f"Measuring fluorescence in wells {well_indices}...")
    # In real implementation, this calls lash_e.measure_wellplate()
    # For now, return mock data
    return {
        '334_373': [80.0] * len(well_indices),
        '334_384': [100.0] * len(well_indices)
    }

# ================================================================================
# SECTION 6: FULL WORKFLOW EXECUTION WITH LASH_E INTEGRATION
# ================================================================================

def create_output_folder(simulate=True):
    """Create timestamped output folder for experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "simulated" if simulate else "hardware"
    folder_name = f"adaptive_surfactant_screening_{mode}_{timestamp}"
    output_folder = os.path.join("output", folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def check_or_create_substocks(lash_e, surfactant_name, target_concentrations, tracking):
    """
    Simplified version: Check if substocks exist, create missing ones.
    For the adaptive workflow, this creates real dilution series.
    """
    logger = lash_e.logger
    logger.info(f"Preparing substocks for {surfactant_name}")
    
    dilution_vials = []
    dilution_steps = []
    achievable_concentrations = get_achievable_concentrations(surfactant_name, target_concentrations)
    
    for i, (target_conc, achievable_conc) in enumerate(zip(target_concentrations, achievable_concentrations)):
        vial_name = f"{surfactant_name}_dilution_{i}"
        
        if achievable_conc is not None:
            dilution_vials.append(vial_name)
            
            # Record dilution step
            dilution_steps.append({
                'vial_name': vial_name,
                'target_conc_mm': target_conc,
                'final_conc_mm': achievable_conc,
                'achievable': True,
                'created': True
            })
            
            if not lash_e.simulate:
                # Real hardware dilution creation
                logger.info(f"Creating physical dilution: {vial_name} = {achievable_conc:.2e} mM")
                # Create the actual dilution using robot
                stock_conc = SURFACTANT_LIBRARY[surfactant_name]["stock_conc"]
                required_stock_conc = achievable_conc * CONCENTRATION_CORRECTION_FACTOR
                dilution_factor = stock_conc / required_stock_conc
                
                stock_volume = FINAL_SUBSTOCK_VOLUME / dilution_factor  # mL
                water_volume = FINAL_SUBSTOCK_VOLUME - stock_volume  # mL
                
                # Add water first
                if water_volume > 0:
                    lash_e.nr_robot.dispense_into_vial_from_reservoir(
                        reservoir_index=1, vial_index=vial_name, volume=water_volume, return_home=False
                    )
                
                # Add stock solution
                lash_e.nr_robot.dispense_from_vial_into_vial(
                    source_vial_name=f"{surfactant_name}_stock", 
                    dest_vial_name=vial_name, 
                    volume=stock_volume,
                    liquid='water'
                )
                
                # Vortex to mix
                lash_e.nr_robot.vortex_vial(vial_name=vial_name, vortex_time=8, vortex_speed=80)
            else:
                logger.info(f"Simulated dilution: {vial_name} = {achievable_conc:.2e} mM")
        else:
            dilution_vials.append(None)
            
            dilution_steps.append({
                'vial_name': vial_name,
                'target_conc_mm': target_conc,
                'achievable': False,
                'created': False,
                'reason': 'Stock concentration too low'
            })
    
    return dilution_vials, dilution_steps, achievable_concentrations

def execute_adaptive_surfactant_screening(surfactant_a_name="SDS", surfactant_b_name="DTAB", simulate=True):
    """
    Execute the complete surfactant grid screening workflow using adaptive concentrations.
    
    Args:
        surfactant_a_name: Name of first surfactant (cationic)
        surfactant_b_name: Name of second surfactant (anionic) 
        simulate: Run in simulation mode
        
    Returns:
        dict: Results including well_map, measurements, and concentrations used
    """
    print("="*80)
    print("ADAPTIVE SURFACTANT GRID SCREENING WORKFLOW")
    print("="*80)
    print(f"Surfactants: {surfactant_a_name} + {surfactant_b_name}")
    print(f"Mode: {'SIMULATION' if simulate else 'HARDWARE'}")
    print("")
    
    # STEP 1: Initialize Lash_E coordinator
    print("Step 1: Initializing Lash_E coordinator...")
    print(f"  Mode: {'Simulation' if simulate else 'Hardware'}")
    
    # Initialize Lash_E with built-in simulation support
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
    
    # Validate system state
    print("  Validating robot and track status...")
    lash_e.nr_robot.check_input_file()
    lash_e.nr_track.check_input_file()
    
    # Validate pipetting capability if enabled
    if VALIDATE_LIQUIDS:
        #TODO add validation
        None
    
    # Robot is ready - no initial positioning needed
    print("  Robot ready for operations")
    
    # Get fresh wellplate
    print("  Getting fresh wellplate...")
    lash_e.nr_track.get_new_wellplate()
    
    print("+ Lash_E initialization complete")
    print("")
    
    # STEP 2: Calculate adaptive concentrations
    print("Step 2: Calculating adaptive concentration grids...")
    concs_a, concs_b = calculate_dual_surfactant_grids(surfactant_a_name, surfactant_b_name)
    
    # Check achievability
    achievable_a = get_achievable_concentrations(surfactant_a_name, concs_a)
    achievable_b = get_achievable_concentrations(surfactant_b_name, concs_b)
    
    print(f"Concentration analysis:")
    print(f"  {surfactant_a_name}: {len([x for x in achievable_a if x])}/{len(achievable_a)} concentrations achievable")
    print(f"  {surfactant_b_name}: {len([x for x in achievable_b if x])}/{len(achievable_b)} concentrations achievable")
    print("")
    
    # STEP 3: Calculate smart dilution strategies
    print("Step 3: Calculating optimal dilution strategies...")
    
    # Filter to only achievable concentrations
    achievable_concs_a = [c for c in achievable_a if c is not None]
    achievable_concs_b = [c for c in achievable_b if c is not None]
    
    # Calculate smart dilution plans
    plan_a, tracker_a = calculate_smart_dilution_plan(surfactant_a_name, achievable_concs_a)
    plan_b, tracker_b = calculate_smart_dilution_plan(surfactant_b_name, achievable_concs_b)
    
    # Create dilution vial mappings for compatibility with legacy grid functions
    dilution_vials_a = []
    for conc in concs_a:
        if conc in plan_a['concentration_map']:
            dilution_vials_a.append(plan_a['concentration_map'][conc]['vial_name'])
        else:
            dilution_vials_a.append(None)
    
    dilution_vials_b = []
    for conc in concs_b:
        if conc in plan_b['concentration_map']:
            dilution_vials_b.append(plan_b['concentration_map'][conc]['vial_name'])
        else:
            dilution_vials_b.append(None)
    
    # Show summary of dilution strategy
    print(f"\n=== DILUTION STRATEGY SUMMARY ===")
    print(f"{surfactant_a_name}: {len(plan_a['substocks_needed'])} substocks needed for {len(achievable_concs_a)} concentrations")
    print(f"{surfactant_b_name}: {len(plan_b['substocks_needed'])} substocks needed for {len(achievable_concs_b)} concentrations")
    
    print(f"\nSubstocks to create:")
    for substock in plan_a['substocks_needed']:
        print(f"  {substock['vial_name']}: {substock['concentration_mm']:.2e} mM")
    for substock in plan_b['substocks_needed']:
        print(f"  {substock['vial_name']}: {substock['concentration_mm']:.2e} mM")
    
    # Calculate and display dilution recipes
    dilution_recipes = calculate_dilution_recipes(plan_a, plan_b, surfactant_a_name, surfactant_b_name)
    
    # Show detailed pipetting plan
    print(f"\n=== DETAILED PIPETTING PLAN ===")
    print(f"For each target concentration, showing: vial_to_use (volume_to_pipette)")
    print(f"\n{surfactant_a_name} concentrations:")
    for target_conc, solution in plan_a['concentration_map'].items():
        # Water fills the remaining volume in this surfactant's allocation
        water_for_this_surfactant = EFFECTIVE_SURFACTANT_VOLUME - solution['volume_needed_ul']
        print(f"  {target_conc:.2e} mM: {solution['vial_name']} ({solution['volume_needed_ul']:.1f}µL) + {water_for_this_surfactant:.1f}µL water")
    
    print(f"\n{surfactant_b_name} concentrations:")
    for target_conc, solution in plan_b['concentration_map'].items():
        # Water fills the remaining volume in this surfactant's allocation
        water_for_this_surfactant = EFFECTIVE_SURFACTANT_VOLUME - solution['volume_needed_ul']
        print(f"  {target_conc:.2e} mM: {solution['vial_name']} ({solution['volume_needed_ul']:.1f}µL) + {water_for_this_surfactant:.1f}µL water")
    
    print(f"\n=== WELL COMPOSITION BREAKDOWN ===")
    print(f"Total well volume: {WELL_VOLUME_UL} µL")
    print(f"  Surfactant A allocation: {EFFECTIVE_SURFACTANT_VOLUME:.1f} µL (solution + water)")
    print(f"  Surfactant B allocation: {EFFECTIVE_SURFACTANT_VOLUME:.1f} µL (solution + water)")
    if ADD_BUFFER:
        print(f"  Buffer: {BUFFER_VOLUME_UL} µL")
    print(f"  Pyrene (added later): {PYRENE_VOLUME_UL} µL")
    remaining = WELL_VOLUME_UL - (2 * EFFECTIVE_SURFACTANT_VOLUME) - (BUFFER_VOLUME_UL if ADD_BUFFER else 0) - PYRENE_VOLUME_UL
    if remaining > 0:
        print(f"  Additional water: {remaining:.1f} µL")
    
    # STEP 3b: Verify concentration calculations and export to CSV
    print(f"\n" + "="*50)
    verification_df, well_composition_df = verify_concentration_calculations_and_export(
        plan_a, plan_b, surfactant_a_name, surfactant_b_name, achievable_concs_a, achievable_concs_b
    )
    
    # Calculate water usage from verification dataframe
    total_water_ul = verification_df['Water_Volume_uL'].sum()
    total_wells = len(achievable_concs_a) * len(achievable_concs_b) * N_REPLICATES
    avg_water_per_solution = total_water_ul / len(verification_df) if len(verification_df) > 0 else 0
    
    print(f"\n" + "="*50)
    print(f"WATER USAGE SUMMARY (from actual calculations)")
    print(f"="*50)
    print(f"Total water for substock dilutions: {total_water_ul:.1f} µL ({total_water_ul/1000:.2f} mL)")
    print(f"Average water per surfactant solution: {avg_water_per_solution:.1f} µL")
    print(f"Wells to be created: {total_wells} wells")
    print(f"Additional water for well filling: ~{total_wells * 85:.0f} µL (~{total_wells * 85/1000:.1f} mL)")
    total_estimated = (total_water_ul + total_wells * 85) / 1000
    print(f"TOTAL ESTIMATED WATER USAGE: ~{total_estimated:.1f} mL")
    print(f"="*50)
    
    print("\n" + "="*50)
    response = input("Does this dilution plan look correct? Press Enter to proceed with automation, or 'q' to quit: ")
    if response.lower() == 'q':
        print("Workflow cancelled by user.")
        return None
    print("Proceeding with automated execution...")
    print("")
    
    # STEP 4: Create physical substocks using smart recipes
    print("Step 4: Creating physical substocks from calculated recipes...")
    
    # Create substocks in the correct order (recipes are already ordered for dependencies)
    created_substocks = create_substocks_from_recipes(lash_e, dilution_recipes)
    
    # Report results
    successful_substocks = len([s for s in created_substocks if s['created']])
    total_substocks = len(created_substocks)
    print(f"+ Substock creation complete: {successful_substocks}/{total_substocks} successful")
    
    if successful_substocks < total_substocks:
        failed_substocks = [s for s in created_substocks if not s['created']]
        print(f"Failed substocks:")
        for substock in failed_substocks:
            error_msg = substock.get('error', 'Unknown error')
            print(f"  {substock['vial_name']}: {error_msg}")
    
    print("")
    
    # STEP 5: Show concentration grid summary
    print("Step 5: Concentration Grid Summary")
    # Create simple summary from smart planning data
    total_combinations = len(concs_a) * len(concs_b)
    achievable_combinations = len(plan_a['concentration_map']) * len(plan_b['concentration_map'])
    print(f"Grid analysis:")
    print(f"  {surfactant_a_name}: {len(plan_a['concentration_map'])}/{len(concs_a)} concentrations achievable")
    print(f"  {surfactant_b_name}: {len(plan_b['concentration_map'])}/{len(concs_b)} concentrations achievable")
    print(f"  Total combinations: {achievable_combinations}/{total_combinations} achievable")
    print("")
    print("")
    
    # STEP 6: Execute pipetting workflow with integrated vial optimization
    print("Step 6: Executing pipetting workflow (with integrated vial positioning)...")
    shared_wellplate_state = {
        'global_well_counter': 0,
        'global_well_map': [],
        'total_wells_added': 0,
        'current_plate': 1,
        'wells_used': 0,
        'measurements': [],
        'last_measured_well': -1
    }
    
    well_map, updated_state = pipette_grid_to_shared_wellplate(
        lash_e, concs_a, concs_b, plan_a, plan_b, 
        surfactant_a_name, surfactant_b_name, shared_wellplate_state
    )
    
    print(f"+ Pipetting workflow completed!")
    print(f"  Wells filled: {len(well_map)}")
    print(f"  Measurements taken: {len(updated_state.get('measurements', []))}")
    print("")
    
    # STEP 7: Create output folder and save results
    print("Step 7: Saving results...")
    output_folder = create_output_folder(simulate)
    
    # Save well mapping data
    if well_map:
        well_map_file = os.path.join(output_folder, "well_mapping.csv")
        pd.DataFrame(well_map).to_csv(well_map_file, index=False)
        print(f"  Well mapping saved to: {well_map_file}")
    
    # Save dilution information
    dilution_info = {
        'surfactant_a': {
            'name': surfactant_a_name,
            'concentrations': concs_a.tolist(),
            'dilution_plan': plan_a
        },
        'surfactant_b': {
            'name': surfactant_b_name,
            'concentrations': concs_b.tolist(),
            'dilution_plan': plan_b
        }
    }
    
    dilution_file = os.path.join(output_folder, "dilution_info.json")
    with open(dilution_file, 'w') as f:
        json.dump(dilution_info, f, indent=2, default=str)
    print(f"  Dilution info saved to: {dilution_file}")
    
    print(f"+ Results saved to: {output_folder}")
    
    # Get actual pipette usage breakdown (real from hardware or simulated count)
    pipette_breakdown = get_pipette_usage_breakdown(lash_e)
    print(f"+ Pipette tips used: {pipette_breakdown['large_tips']} large, {pipette_breakdown['small_tips']} small (total: {pipette_breakdown['total']}) ({'simulated' if simulate else 'actual'})")
    print("")
    
    # Show the corrected phase order being used
    print("+ CORRECTED phase order used:")
    print("  1. Add Surfactant A")
    print("  2. Add Surfactant B") 
    print("  3. Add Buffer (BEFORE turbidity measurement)")
    print("  4. Measure Turbidity (with buffer present)")
    print("  5. Add Pyrene")
    print("  6. Measure Fluorescence")
    print("")
    
    # Return comprehensive results
    return {
        'surfactant_a': surfactant_a_name,
        'surfactant_b': surfactant_b_name,
        'concentrations_a': concs_a.tolist(),
        'concentrations_b': concs_b.tolist(),
        'achievable_a': [x for x in achievable_a if x is not None],
        'achievable_b': [x for x in achievable_b if x is not None],
        'well_map': well_map,
        'measurements': updated_state.get('measurements', []),
        'total_wells': len(well_map),
        'plates_used': updated_state.get('current_plate', 1),
        'output_folder': output_folder,
        'dilution_vials_a': dilution_vials_a,
        'dilution_vials_b': dilution_vials_b,
        'pipette_breakdown': get_pipette_usage_breakdown(lash_e),
        'simulation': simulate,
        'workflow_complete': True
    }

if __name__ == "__main__":
    """
    Run the adaptive surfactant grid screening workflow.
    """
    print("Starting adaptive surfactant grid screening...")
    print("")
    
    # Choose experiment mode
    RUN_FULL_WORKFLOW = True  # Set to True to run the complete workflow
    
    if RUN_FULL_WORKFLOW:
        # Execute the complete workflow with adaptive concentrations
        results = execute_adaptive_surfactant_screening(
            surfactant_a_name="SDS", 
            surfactant_b_name="DTAB", 
            simulate=SIMULATE
        )
        
        if results and results['workflow_complete']:
            print("="*80)
            print("WORKFLOW COMPLETE!")
            print("="*80)
            print(f"+ Surfactants: {results['surfactant_a']} + {results['surfactant_b']}")
            print(f"+ Wells processed: {results['total_wells']}")
            print(f"+ Plates used: {results['plates_used']}")
            breakdown = results['pipette_breakdown']
            print(f"+ Pipette tips: {breakdown['large_tips']} large, {breakdown['small_tips']} small (total: {breakdown['total']})")
            print(f"+ Measurements: {len(results['measurements'])} intervals")
            print(f"+ Mode: {'Simulation' if results['simulation'] else 'Hardware'}")
            print(f"+ Results saved to: {results['output_folder']}")
    else:
        # Just test the concentration calculation (original behavior)
        print("Testing adaptive concentration grid approach...")
        
        # Test with different surfactants
        surfactants_to_test = [("SDS", "DTAB"), ("CTAB", "NaC"), ("TTAB", "NaDC")]
        
        for surf_a, surf_b in surfactants_to_test:
            print(f"\\n=== Testing {surf_a} + {surf_b} ===")
            concs_a, concs_b = calculate_dual_surfactant_grids(surf_a, surf_b)
            
            # Test achievability
            achievable_a = get_achievable_concentrations(surf_a, concs_a)
            achievable_b = get_achievable_concentrations(surf_b, concs_b)
            
            # Mock vial arrays for grid summary
            vials_a = [f"{surf_a}_dilution_{i}" if ach else None for i, ach in enumerate(achievable_a)]
            vials_b = [f"{surf_b}_dilution_{i}" if ach else None for i, ach in enumerate(achievable_b)]
            
            summary = create_concentration_grid_summary(concs_a, concs_b, vials_a, vials_b, surf_a, surf_b)
            print(summary)
"""
Surfactant Grid Turbidity + Fluorescence Screening Workflow - Adaptive Concentrations
Systematic dilution grid of two surfactants with adaptive concentration ranges based on stock concentrations.

UPDATED CONCENTRATION APPROACH:
- Uses adaptive concentration ranges: min_conc = 10^-4 mM, max_conc = stock_conc * (allocated_volume / well_volume)
- Logarithmic spacing with fixed number of concentrations (9 by default)
- Each surfactant gets its own optimized concentration range based on volume constraints

VALIDATION MODES:
- Set VALIDATE_LIQUIDS=True to run validation alongside full experiment
- Set VALIDATION_ONLY=True to run only pipetting validation and skip experiment (great for testing)
- Both modes save validation results to experiment_name/calibration_validation/

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

from re import S
import sys


sys.path.append("../utoronto_demo")
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from master_usdl_coordinator import Lash_E
import slack_agent

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

SURFACTANT_A = "SDS"
SURFACTANT_B = "TTAB"

# WORKFLOW CONSTANTS
SIMULATE = False # Set to False for actual hardware execution
VALIDATE_LIQUIDS = True # Set to False to skip pipetting validation during initialization
VALIDATION_ONLY = False # Set to True to run only validations and skip full experiment

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
#lash_e.logger.info(f"Effective surfactant volume: {EFFECTIVE_SURFACTANT_VOLUME} uL per surfactant (reserves space for buffer+pyrene)")

# CRITICAL: Concentration correction factor for buffer dilution
# When buffer is added, stock concentrations must be higher to compensate for dilution
# This ensures final concentrations match intended values
CONCENTRATION_CORRECTION_FACTOR = WELL_VOLUME_UL / (2 * EFFECTIVE_SURFACTANT_VOLUME)
#lash_e.logger.info(f"Concentration correction factor: {CONCENTRATION_CORRECTION_FACTOR:.3f} (buffer={ADD_BUFFER}, buffer_vol={BUFFER_VOLUME_UL if ADD_BUFFER else 0}uL)")
MAX_WELLS = 96 #Wellplate size

# Constants
FINAL_SUBSTOCK_VOLUME = 6  # mL final volume for each dilution
MINIMUM_PIPETTE_VOLUME = 0.2  # mL (200 uL) - minimum volume for accurate pipetting
MEASUREMENT_INTERVAL = 96    # Measure every N wells to prevent evaporation

# Measurement protocol files for Cytation
TURBIDITY_PROTOCOL_FILE = r"C:\Protocols\CMC_Absorbance_96.prt"
FLUORESCENCE_PROTOCOL_FILE = r"C:\Protocols\CMC_Fluorescence_96.prt"

# Workflow control flags
CREATE_WELLPLATE = True  # Set to True to create wellplate, False to skip to measurements only

# File paths
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/surfactant_grid_vials_expanded.csv"



def backup_raw_measurement_data(lash_e, measurement_entry, plate_number, wells_measured, experiment_name=None):
    """
    Immediately backup raw measurement data to prevent data loss if processing crashes.
    
    Args:
        lash_e: Lash_E coordinator for logging
        measurement_entry: Dictionary containing measurement data
        plate_number: Current plate number
        wells_measured: List of wells that were measured
        experiment_name: Name of current experiment for folder organization
    """
    try:
        # Create backup directory within experiment folder
        if experiment_name:
            backup_dir = os.path.join("output", experiment_name, "measurement_backups")
        else:
            backup_dir = os.path.join("output", "measurement_backups")  # Fallback
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create unique backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        backup_filename = f"raw_measurement_plate{plate_number}_wells{wells_measured[0]}-{wells_measured[-1]}_{timestamp}.json"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Skip file saving during simulation
        if hasattr(lash_e, 'simulate') and lash_e.simulate:
            lash_e.logger.info(f"    [SIMULATED] Would save backup: {backup_filename}")
            return
        
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
        
        lash_e.logger.info(f"OK Backed up measurement data to: {backup_path}")
        
    except Exception as e:
        lash_e.logger.info(f"WARNING: Failed to backup measurement data: {e}")
        # Don't crash the workflow if backup fails

def condition_tip(lash_e, vial_name, conditioning_volume_ul=100, liquid_type='water'):
    """Condition a pipette tip by aspirating and dispensing into source vial multiple times
    
    Args:
        lash_e: Lash_E robot controller
        vial_name: Name of vial to condition tip with
        conditioning_volume_ul: Total volume for conditioning (default 100 uL)
        liquid_type: Type of liquid for pipetting parameters ('water', 'DMSO', etc.)
    """
    try:
        # Calculate volume per conditioning cycle (5 cycles total)
        cycles = 5
        volume_per_cycle_ul = conditioning_volume_ul
        volume_per_cycle_ml = volume_per_cycle_ul / 1000
        
        lash_e.logger.info(f"    Conditioning tip with {vial_name}: {cycles} cycles of {volume_per_cycle_ul:.1f}uL")
        
        for cycle in range(cycles):
            # Aspirate from vial 
            lash_e.nr_robot.aspirate_from_vial(vial_name, volume_per_cycle_ml, liquid=liquid_type)
            # Dispense back into same vial
            lash_e.nr_robot.dispense_into_vial(vial_name, volume_per_cycle_ml, liquid=liquid_type)
        
        lash_e.logger.info(f"    Tip conditioning complete for {vial_name}")
        
    except Exception as e:
        lash_e.logger.info(f"    Warning: Could not condition tip with {vial_name}: {e}")

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
        lash_e.logger.info(f"  Warning: Could not read pipette status: {e}")
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
            lash_e.logger.info(f"    [Simulated Pipette {lash_e.pipette_count}]: {operation_description}")
    # In hardware mode, the robot automatically tracks usage in robot_status.yaml

def fill_water_vial(lash_e, vial_name):
    """
    Fill a water vial to maximum capacity (8mL) by moving it to reservoir,
    calculating needed volume, dispensing from reservoir, and returning home.
    
    Args:
        lash_e: The Lash_E coordinator instance
        vial_name (str): Name of water vial to fill ('water' or 'water_2')
    """
    # Get current volume and vial capacity
    current_volume_ml = lash_e.nr_robot.get_vial_info(vial_name, 'vial_volume')
    max_volume_ml = 8
    
    # Calculate volume needed to fill to max capacity
    fill_volume_ml = max_volume_ml - current_volume_ml
    
    if fill_volume_ml <= 0.1:  # Already nearly full (within 100uL)
        lash_e.logger.info(f"    Water vial '{vial_name}' already full ({current_volume_ml:.2f}mL), skipping fill")
        return
        
    lash_e.logger.info(f"    Filling water vial '{vial_name}': {current_volume_ml:.2f}mL -> {max_volume_ml:.2f}mL (adding {fill_volume_ml:.2f}mL)")
    
    # Fill vial from reservoir
    lash_e.nr_robot.dispense_into_vial_from_reservoir(1, vial_name, fill_volume_ml)
    
    
    lash_e.logger.info(f"    Water vial '{vial_name}' filled successfully to {max_volume_ml:.2f}mL")

# ================================================================================
# SECTION 1: SMART SUBSTOCK MANAGEMENT
# ================================================================================

class SurfactantSubstockTracker:
    """Track surfactant substock vial contents and find optimal dilution strategies."""
    
    def __init__(self):
        self.substocks = {}  # vial_name: {'surfactant_name': str, 'concentration_mm': float, 'volume_ml': float}
        self.next_available_substock = 0
        self.min_pipette_volume_ul = 10  # Minimum pipetting volume in uL
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
            # Calculate volume needed for target final concentration in 200 uL well
            # Final_conc = (stock_conc * vol_needed) / 200 uL
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
                
                # Calculate volume needed for target final concentration in 200 uL well
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
        Aims for 15-20 uL pipetting volumes for reusability.
        """
        max_volume_ml = self.max_surfactant_volume_ul / 1000
        
        # Target 15-20 uL volumes for efficient, reusable solutions
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

def calculate_smart_dilution_plan(lash_e, surfactant_name, target_concentrations_mm):
    """
    Calculate optimal dilution strategy for a surfactant across all target concentrations.
    Returns plan showing which solutions to create and how to use them.
    """
    tracker = SurfactantSubstockTracker()
    plan = {'substocks_needed': [], 'concentration_map': {}}
    
    lash_e.logger.info(f"\n=== Analyzing dilution strategy for {surfactant_name} ===")
    lash_e.logger.info(f"Target concentrations: {[f'{c:.2e}' for c in target_concentrations_mm]} mM")
    
    for target_conc in target_concentrations_mm:
        # Try to find existing solution
        solution = tracker.find_best_solution_for_concentration(surfactant_name, target_conc)
        
        if solution:
            # Found existing solution
            plan['concentration_map'][target_conc] = solution
            lash_e.logger.info(f"  {target_conc:.2e} mM: Use {solution['vial_name']} ({solution['volume_needed_ul']:.1f} uL)")
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
                print(f"  {target_conc:.2e} mM: CREATE {vial_name} at {optimal_conc:.2e} mM -> use {solution['volume_needed_ul']:.1f} uL")
            else:
                print(f"  {target_conc:.2e} mM: *** CANNOT ACHIEVE ***")
    
    return plan, tracker

def calculate_dilution_recipes(lash_e, plan_a, plan_b, surfactant_a_name, surfactant_b_name):
    """
    Calculate the exact dilution recipes for each substock that needs to be created.
    Uses serial dilutions when direct dilution would require volumes < 200 uL.
    """
    MIN_SUBSTOCK_VOLUME_UL = 200  # Minimum volume for accurate substock preparation
    FINAL_SUBSTOCK_VOLUME_ML = 6.0
    
    recipes = []
    
    lash_e.logger.info(f"\n=== SUBSTOCK DILUTION RECIPES ===")
    lash_e.logger.info("For each substock showing: source + volume -> target_concentration")
    lash_e.logger.info(f"(Using minimum {MIN_SUBSTOCK_VOLUME_UL}uL pipetting volumes for accuracy)")
    
    # Process substocks for both surfactants
    for plan, surfactant_name in [(plan_a, surfactant_a_name), (plan_b, surfactant_b_name)]:
        if plan['substocks_needed']:
            lash_e.logger.info(f"\n{surfactant_name} substocks:")
            
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
                        # Simple dilution: no correction factor for substock creation
                        dilution_factor = source_conc / target_conc
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
                    # Simple dilution: no correction factor for substock creation
                    dilution_factor = best_source['conc'] / target_conc
                    
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
                    lash_e.logger.info(f"  {vial_name}: {source_volume_ul:.0f}uL {best_source['name']} + {water_volume_ul:.0f}uL water -> {target_conc:.2e} mM")
                    lash_e.logger.info(f"    (Dilution factor: {dilution_factor:.1f}x from {best_source['conc']:.2e} mM {source_type})")
                
                else:
                    # Cannot make with current minimum volume - needs intermediate
                    lash_e.logger.info(f"  {vial_name}: *** NEEDS INTERMEDIATE DILUTION - too dilute for {MIN_SUBSTOCK_VOLUME_UL}uL minimum ***")
                    
                    # Suggest intermediate concentration
                    max_possible_dilution = FINAL_SUBSTOCK_VOLUME_ML / (MIN_SUBSTOCK_VOLUME_UL / 1000)
                    intermediate_conc = stock_conc / max_possible_dilution
                    lash_e.logger.info(f"    Suggestion: Create intermediate at ~{intermediate_conc:.2e} mM first")
    
    if not recipes:
        lash_e.logger.info("No substocks needed - using only stock solutions!")
    
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
        logger.info(f"  Recipe: {source_volume_ml*1000:.0f}uL {source_vial} + {water_volume_ml*1000:.0f}uL water")
        
        # Always call robot functions - Lash_E handles simulation internally
        try:
            # Add source solution
            lash_e.nr_robot.dispense_from_vial_into_vial(
                source_vial_name=source_vial, 
                dest_vial_name=vial_name, 
                volume=source_volume_ml,
                liquid='water'
            )

            # Add water first if needed
            if water_volume_ml > 0:
                lash_e.nr_robot.dispense_into_vial_from_reservoir(
                    reservoir_index=1, vial_index=vial_name, 
                    volume=water_volume_ml, return_home=False
                )
            

            
            # Vortex to mix
            lash_e.nr_robot.vortex_vial(vial_name=vial_name, vortex_time=8, vortex_speed=80)
            lash_e.nr_robot.return_vial_home(vial_name=vial_name)
            

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


# ================================================================================
# SECTION 2: CONCENTRATION AND GRID CALCULATION FUNCTIONS
# ================================================================================

#DESCRIPTION: Calculate adaptive logarithmic concentration grid for surfactant dilution series
#RATING: 9/10 - Adaptive approach that optimizes concentration range for each surfactant
def calculate_grid_concentrations(lash_e, surfactant_name=None):
    """
    Calculate adaptive concentration grid points for surfactants.
    Each surfactant gets its own optimized range: min_conc to stock_conc/2.
    
    Args:
        lash_e: Lash_E coordinator for logging
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
    
    lash_e.logger.info(f"Adaptive grid for {surfactant_name or 'generic'}: {MIN_CONC:.1e} to {max_conc:.1f} mM ({NUMBER_CONCENTRATIONS} steps)")
    lash_e.logger.info(f"  Concentrations: {[f'{c:.3e}' for c in concentrations]}")
    
    return concentrations

#DESCRIPTION: Calculate concentration grids for both surfactants with adaptive ranges
#RATING: 10/10 - Clean interface for dual-surfactant experiments
def calculate_dual_surfactant_grids(lash_e, surfactant_a_name, surfactant_b_name):
    """
    Calculate optimized concentration grids for both surfactants.
    
    Args:
        lash_e: Lash_E coordinator for logging
        surfactant_a_name: Name of first surfactant
        surfactant_b_name: Name of second surfactant
        
    Returns:
        tuple: (concentrations_a, concentrations_b)
    """
    concs_a = calculate_grid_concentrations(lash_e, surfactant_a_name)
    concs_b = calculate_grid_concentrations(lash_e, surfactant_b_name)
    
    lash_e.logger.info(f"\nAdaptive concentration grid summary:")
    lash_e.logger.info(f"  {surfactant_a_name}: {len(concs_a)} concentrations from {concs_a[0]:.3e} to {concs_a[-1]:.3e} mM")
    lash_e.logger.info(f"  {surfactant_b_name}: {len(concs_b)} concentrations from {concs_b[0]:.3e} to {concs_b[-1]:.3e} mM")
    lash_e.logger.info(f"  Total combinations: {len(concs_a)} x {len(concs_b)} = {len(concs_a) * len(concs_b)} wells (x {N_REPLICATES} replicates = {len(concs_a) * len(concs_b) * N_REPLICATES} total wells)")
    
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
                    logger.info(f"Checking for existing {surfactant} substocks in CSV...")
                
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
                                    logger.info(f"  Found existing {surfactant} dilution: {vial_name} = {target_conc:.2e} mM ({volume:.1f} mL)")
                        except (ValueError, IndexError):
                            continue
        
        except Exception as e:
            if logger:
                logger.warning(f"Could not load existing {surfactant} dilutions from CSV: {e}")
            else:
                logger.info(f"Warning: Could not load existing {surfactant} dilutions from CSV: {e}")
    
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
    
def create_control_wells(surfactant_a_name, surfactant_b_name, position_prefix="start"):
    """
    Create quality control wells for start/end of experiment.
    Returns list of control well requirements.
    """
    controls = []
    
    # Control 1: Surfactant A stock only (200 uL)
    controls.append({
        'control_type': f'{position_prefix}_control_surfactant_A',
        'description': f'{surfactant_a_name} stock (200 uL)',
        'dilution_a_vial': f'{surfactant_a_name}_stock',
        'dilution_b_vial': None,
        'volume_a_ul': 200,
        'volume_b_ul': 0,
        'replicate': 1,
        'is_control': True
    })
    
    # Control 2: Surfactant B stock only (200 uL)  
    controls.append({
        'control_type': f'{position_prefix}_control_surfactant_B',
        'description': f'{surfactant_b_name} stock (200 uL)',
        'dilution_a_vial': None,
        'dilution_b_vial': f'{surfactant_b_name}_stock',
        'volume_a_ul': 0,
        'volume_b_ul': 200,
        'replicate': 1,
        'is_control': True
    })
    
    # Control 3: Buffer only (if using buffer)
    if ADD_BUFFER:
        controls.append({
            'control_type': f'{position_prefix}_control_buffer',
            'description': f'{SELECTED_BUFFER} buffer (200 uL)',
            'dilution_a_vial': None,
            'dilution_b_vial': None,
            'volume_a_ul': 0,
            'volume_b_ul': 0,
            'buffer_only': True,
            'replicate': 1,
            'is_control': True
        })
    
    # Control 4: Water only (200 uL)
    controls.append({
        'control_type': f'{position_prefix}_control_water',
        'description': 'Water blank (200 uL)',
        'dilution_a_vial': None,
        'dilution_b_vial': None,
        'volume_a_ul': 0,
        'volume_b_ul': 0,
        'water_only': True,
        'replicate': 1,
        'is_control': True
    })
    
    return controls
    
def measure_wellplate_turbidity(lash_e, wells_in_batch, wellplate_data):
    """Measure turbidity for a batch of wells and save data."""
    from datetime import datetime
    import os
    
    lash_e.logger.info(f"  Measuring turbidity for wells {wells_in_batch[0]}-{wells_in_batch[-1]} (with buffer present)")
    turbidity_data = measure_turbidity(lash_e, wells_in_batch)
    
    # Save raw turbidity data to CSV immediately (skip in simulation)
    turbidity_filename = None
    if turbidity_data is not None and not lash_e.simulate:
        experiment_name = getattr(lash_e, 'current_experiment_name', 'unknown_experiment')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        turbidity_filename = f"turbidity_plate{wellplate_data['current_plate']}_wells{wells_in_batch[0]}-{wells_in_batch[-1]}_{timestamp}.csv"
        turbidity_path = os.path.join("output", experiment_name, "measurement_backups", turbidity_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(turbidity_path), exist_ok=True)
        
        # Save actual DataFrame to CSV
        turbidity_data.to_csv(turbidity_path, index=True)
        lash_e.logger.info(f"    Saved raw turbidity data: {turbidity_filename}")
    
    # Store turbidity measurement data (keep for compatibility)
    turbidity_entry = {
        'plate_number': wellplate_data['current_plate'],
        'wells_measured': wells_in_batch,
        'measurement_type': 'turbidity_batch',
        'data_file': turbidity_filename,
        'timestamp': datetime.now().isoformat()
    }
    wellplate_data['measurements'].append(turbidity_entry)
    
    return turbidity_entry

def measure_wellplate_fluorescence(lash_e, wells_in_batch, wellplate_data):
    """Measure fluorescence for a batch of wells and save data."""
    from datetime import datetime
    import os
    
    lash_e.logger.info(f"  Measuring fluorescence for wells {wells_in_batch[0]}-{wells_in_batch[-1]}")
    fluorescence_data = measure_fluorescence(lash_e, wells_in_batch)
    
    # Save raw fluorescence data to CSV immediately (skip in simulation)
    fluorescence_filename = None
    if fluorescence_data is not None and not lash_e.simulate:
        experiment_name = getattr(lash_e, 'current_experiment_name', 'unknown_experiment')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fluorescence_filename = f"fluorescence_plate{wellplate_data['current_plate']}_wells{wells_in_batch[0]}-{wells_in_batch[-1]}_{timestamp}.csv"
        fluorescence_path = os.path.join("output", experiment_name, "measurement_backups", fluorescence_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(fluorescence_path), exist_ok=True)
        
        # Save actual DataFrame to CSV
        fluorescence_data.to_csv(fluorescence_path, index=True)
        lash_e.logger.info(f"    Saved raw fluorescence data: {fluorescence_filename}")
    
    # Store fluorescence measurement data (keep for compatibility)
    fluorescence_entry = {
        'plate_number': wellplate_data['current_plate'],
        'wells_measured': wells_in_batch,
        'measurement_type': 'fluorescence_batch', 
        'data_file': fluorescence_filename,
        'timestamp': datetime.now().isoformat()
    }
    wellplate_data['measurements'].append(fluorescence_entry)
    
    return fluorescence_entry

def dispense_component_to_wellplate(lash_e, batch_df, vial_name, liquid_type, volume_column):
    """
    Unified dispensing method for any liquid component.
    
    Args:
        lash_e: Robot coordinator
        batch_df: DataFrame with well recipes for this batch
        vial_name: Name of source vial (e.g., 'water', 'SDS_1.0mM', 'pyrene_DMSO')
        liquid_type: Type of liquid for pipetting parameters ('water', 'DMSO')
        volume_column: Column name for volume (e.g., 'surf_A_volume_ul', 'water_volume_ul')
    """
    logger = lash_e.logger
    
    # Get component name for logging
    component_name = volume_column.replace('_volume_ul', '').replace('_', ' ').title()
    
    # Filter to wells that need this specific component/substock
    if volume_column == 'surf_A_volume_ul':
        # For surfactant A, also check that this is the correct substock
        wells_needing_component = batch_df[
            (batch_df[volume_column] > 0) & 
            (batch_df['substock_A_name'] == vial_name)
        ].copy()
    elif volume_column == 'surf_B_volume_ul':
        # For surfactant B, also check that this is the correct substock  
        wells_needing_component = batch_df[
            (batch_df[volume_column] > 0) & 
            (batch_df['substock_B_name'] == vial_name)
        ].copy()
    else:
        # For other components (water, buffer), use original logic
        wells_needing_component = batch_df[batch_df[volume_column] > 0].copy()
    
    if len(wells_needing_component) == 0:
        logger.info(f"  {component_name}: No wells need this component from {vial_name}")
        return
    
    logger.info(f"  {component_name}: Dispensing from {vial_name} to {len(wells_needing_component)} wells")
    
    # Dispense to each well individually
    for _, row in wells_needing_component.iterrows():
        well_idx = row['wellplate_index']
        volume_ul = row[volume_column]
        volume_ml = volume_ul / 1000
        
        logger.info(f"    Well {well_idx}: {volume_ul:.1f}uL from {vial_name}")
        
        # Robot actions (Lash_E handles simulation internally)
        lash_e.nr_robot.aspirate_from_vial(vial_name, volume_ml, liquid=liquid_type)
        lash_e.nr_robot.dispense_into_wellplate(
            dest_wp_num_array=[well_idx], 
            amount_mL_array=[volume_ml],
            liquid=liquid_type
        )
           
    logger.info(f"    {component_name}: Dispensing complete")

def position_surfactant_vials_by_concentration(lash_e, vial_names, batch_df, vial_type):
    """
    Cute little method to sort surfactant vials by concentration and move to safe positions.
    Prevents contamination by going dilute → concentrated.
    
    Args:
        lash_e: Robot coordinator
        vial_names: List of vial names (e.g., ['SDS_1.0mM', 'SDS_5.0mM'])
        batch_df: DataFrame with concentration info
        vial_type: 'A' or 'B' for logging and column selection
        
    Returns:
        list: Vial names sorted by concentration (dilute first)
    """
    if len(vial_names) == 0:
        return []
        
    logger = lash_e.logger
    safe_positions = [36, 43, 44, 45, 46, 47, 'clamp']  # Safe spots in order
    
    # Get concentration for each vial from the DataFrame
    vial_concentrations = []
    conc_column = f'substock_{vial_type}_conc_mm'
    name_column = f'substock_{vial_type}_name'
    
    for vial_name in vial_names:
        # Find the concentration for this vial
        vial_rows = batch_df[batch_df[name_column] == vial_name]
        if len(vial_rows) > 0:
            concentration = vial_rows.iloc[0][conc_column]
            vial_concentrations.append((vial_name, concentration))
        else:
            # Fallback - assume concentration from vial name if possible
            try:
                if '_' in vial_name and 'mM' in vial_name:
                    conc_str = vial_name.split('_')[-1].replace('mM', '')
                    concentration = float(conc_str)
                    vial_concentrations.append((vial_name, concentration))
                else:
                    # Can't determine concentration, put at end
                    vial_concentrations.append((vial_name, float('inf')))
            except:
                vial_concentrations.append((vial_name, float('inf')))
    
    # Sort by concentration (dilute first)
    vial_concentrations.sort(key=lambda x: x[1])
    sorted_vials = [vial for vial, conc in vial_concentrations]
    
    logger.info(f"  Positioning surfactant {vial_type} vials by concentration (dilute -> concentrated):")
    
    # Move vials to safe positions in concentration order
    for i, vial_name in enumerate(sorted_vials):
        if i < len(safe_positions):
            position = safe_positions[i]
            concentration = vial_concentrations[i][1]
            if position == 'clamp':
                logger.info(f"    {vial_name} ({concentration:.2f}mM) -> clamp")
                lash_e.nr_robot.move_vial_to_location(vial_name, 'clamp', 0)
            else:
                logger.info(f"    {vial_name} ({concentration:.2f}mM) -> main_8mL_rack[{position}]")
                lash_e.nr_robot.move_vial_to_location(vial_name, 'main_8mL_rack', position)
        else:
            logger.warning(f"    {vial_name}: No safe position available (too many vials)")
    
    return sorted_vials

def return_surfactant_vials_home(lash_e, vial_names, vial_type):
    """Return surfactant vials to home positions after dispensing."""
    if len(vial_names) == 0:
        return
        
    logger = lash_e.logger
    logger.info(f"  Returning surfactant {vial_type} vials to home positions:")
    
    for vial_name in vial_names:
        logger.info(f"    {vial_name} -> home")
        lash_e.nr_robot.return_vial_home(vial_name)

def position_water_vial_safely(lash_e, vial_name):
    """Position water vial at safe location for dispensing."""
    logger = lash_e.logger
    safe_position = 36  # Use first safe position for water
    
    logger.info(f"  Positioning {vial_name} at main_8mL_rack[{safe_position}] (safe position)")
    lash_e.nr_robot.move_vial_to_location(vial_name, 'main_8mL_rack', safe_position)

def return_water_vial_home(lash_e, vial_name):
    """Return water vial to home position after dispensing."""
    logger = lash_e.logger
    logger.info(f"  Returning {vial_name} -> home")
    lash_e.nr_robot.return_vial_home(vial_name)


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
    """Measure turbidity using Cytation plate reader with predefined protocol."""
    lash_e.logger.info(f"Measuring turbidity in wells {well_indices} using protocol {TURBIDITY_PROTOCOL_FILE}...")
    
    if not lash_e.simulate:
        # Use the predefined turbidity protocol
        turbidity_data = lash_e.measure_wellplate(
            protocol_file_path=TURBIDITY_PROTOCOL_FILE,
            wells_to_measure=well_indices,
            plate_type="96 WELL PLATE"
        )
        
        # Fix CSV format: skip first row so second row becomes headers
        if hasattr(turbidity_data, 'iloc') and len(turbidity_data) > 1:
            # Drop the first row and reset headers
            turbidity_data = turbidity_data.iloc[1:].copy()
            turbidity_data.columns = turbidity_data.iloc[0]  # Use first row as column names
            turbidity_data = turbidity_data.iloc[1:].copy()   # Drop the header row from data
            turbidity_data.index.name = None  # Clean up index name
            
        lash_e.logger.info(f"Successfully measured turbidity for {len(well_indices)} wells")
        return turbidity_data

    else:
        lash_e.logger.info("Simulation mode - returning mock turbidity data")
        return {'turbidity': [0.5] * len(well_indices)}

def measure_fluorescence(lash_e, well_indices):
    """Measure fluorescence using Cytation plate reader with predefined protocol."""
    lash_e.logger.info(f"Measuring fluorescence in wells {well_indices} using protocol {FLUORESCENCE_PROTOCOL_FILE}...")
    
    if not lash_e.simulate:
        # Use the predefined fluorescence protocol
        fluorescence_data = lash_e.measure_wellplate(
            protocol_file_path=FLUORESCENCE_PROTOCOL_FILE,
            wells_to_measure=well_indices,
            plate_type="96 WELL PLATE"
        )
        
        # Fix CSV format: skip first row so second row becomes headers
        if hasattr(fluorescence_data, 'iloc') and len(fluorescence_data) > 1:
            # Drop the first row and reset headers
            fluorescence_data = fluorescence_data.iloc[1:].copy()
            fluorescence_data.columns = fluorescence_data.iloc[0]  # Use first row as column names  
            fluorescence_data = fluorescence_data.iloc[1:].copy()   # Drop the header row from data
            fluorescence_data.index.name = None  # Clean up index name
            
        lash_e.logger.info(f"Successfully measured fluorescence for {len(well_indices)} wells")
        return fluorescence_data

    else:
        lash_e.logger.info("Simulation mode - returning mock fluorescence data")
        return {
            '334_373': [80.0] * len(well_indices),
            '334_384': [100.0] * len(well_indices)
        }

# ================================================================================
# SECTION 6: FULL WORKFLOW EXECUTION WITH LASH_E INTEGRATION
# ================================================================================

def create_output_folder(simulate=True, experiment_name=None):
    """Create experiment-specific output folder."""
    if experiment_name:
        output_folder = os.path.join("output", experiment_name)
    else:
        # Fallback to old naming scheme
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "simulated" if simulate else "hardware"
        folder_name = f"adaptive_surfactant_screening_{mode}_{timestamp}"
        output_folder = os.path.join("output", folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def create_experiment_folder_structure(experiment_name):
    """
    Create organized folder structure for experiment data.
    
    Folder organization:
    output/{experiment_name}/
    ├── calibration_validation/     # Pipetting validation results
    ├── measurement_backups/        # Raw measurement data backups  
    ├── substocks/                  # Substock preparation logs
    ├── analysis/                   # Data analysis outputs
    └── logs/                       # Workflow execution logs
    
    Args:
        experiment_name: Name of the experiment (e.g., surfactant_grid_SDS_DTAB_20240203_143022)
        
    Returns:
        dict: Paths to all created subdirectories
    """
    base_folder = os.path.join("output", experiment_name)
    
    subfolders = {
        'base': base_folder,
        'validation': os.path.join(base_folder, "calibration_validation"),
        'measurement_backups': os.path.join(base_folder, "measurement_backups"), 
        'substocks': os.path.join(base_folder, "substocks"),
        'analysis': os.path.join(base_folder, "analysis"),
        'logs': os.path.join(base_folder, "logs")
    }
    
    # Create all subdirectories
    for folder_path in subfolders.values():
        os.makedirs(folder_path, exist_ok=True)
    
    return subfolders


def create_well_recipe_from_control(control, well_index, surfactant_a_name, surfactant_b_name):
    """Convert control well specification to well recipe DataFrame row."""
    # Start with base recipe structure using None for not-applicable values
    recipe = {
        'wellplate_index': well_index,
        'well_type': 'control',
        'control_type': control['control_type'],
        'surf_A': surfactant_a_name,
        'surf_B': surfactant_b_name,
        'surf_A_conc_mm': None,  # Will be set if applicable
        'surf_B_conc_mm': None,  # Will be set if applicable  
        'substock_A_name': None,
        'substock_A_conc_mm': None,
        'surf_A_volume_ul': 0.0,
        'substock_B_name': None, 
        'substock_B_conc_mm': None,
        'surf_B_volume_ul': 0.0,
        'water_volume_ul': 0.0,
        'buffer_volume_ul': BUFFER_VOLUME_UL if ADD_BUFFER else 0.0,
        'buffer_used': SELECTED_BUFFER if ADD_BUFFER else None,
        'pyrene_volume_ul': PYRENE_VOLUME_UL,  # Add pyrene to all wells
        'replicate': control['replicate']
    }
    
    # Handle different control types
    if control.get('water_only', False):
        # Water-only control: 200 µL water
        recipe['water_volume_ul'] = 200.0
        recipe['control_type'] = 'water_blank'
        
    elif control.get('buffer_only', False):
        # Buffer-only control: 200 µL buffer  
        recipe['buffer_volume_ul'] = 200.0
        recipe['water_volume_ul'] = 0.0
        recipe['control_type'] = 'buffer_blank'
        
    elif control['volume_a_ul'] > 0:
        # Surfactant A control: 200 µL surfactant A stock
        recipe['surf_A_conc_mm'] = SURFACTANT_LIBRARY[surfactant_a_name]['stock_conc']
        recipe['substock_A_name'] = control['dilution_a_vial']
        recipe['substock_A_conc_mm'] = SURFACTANT_LIBRARY[surfactant_a_name]['stock_conc'] 
        recipe['surf_A_volume_ul'] = control['volume_a_ul']
        recipe['water_volume_ul'] = 0.0
        recipe['control_type'] = 'surfactant_A_stock'
        # surf_B stays None (not applicable)
        
    elif control['volume_b_ul'] > 0:
        # Surfactant B control: 200 µL surfactant B stock
        recipe['surf_B_conc_mm'] = SURFACTANT_LIBRARY[surfactant_b_name]['stock_conc']
        recipe['substock_B_name'] = control['dilution_b_vial']
        recipe['substock_B_conc_mm'] = SURFACTANT_LIBRARY[surfactant_b_name]['stock_conc']
        recipe['surf_B_volume_ul'] = control['volume_b_ul']
        recipe['water_volume_ul'] = 0.0
        recipe['control_type'] = 'surfactant_B_stock'
        # surf_A stays None (not applicable)
    
    return recipe

def create_well_recipe_from_concentrations(conc_a, conc_b, plan_a, plan_b, well_index, surfactant_a_name, surfactant_b_name, replicate):
    """Convert target concentrations to well recipe DataFrame row using dilution plans."""
    
    # Get solutions from plans (this uses the working calculation logic)
    solution_a = plan_a['concentration_map'].get(conc_a)
    solution_b = plan_b['concentration_map'].get(conc_b)
    
    if not solution_a or not solution_b:
        raise ValueError(f"No solution found for concentrations {conc_a:.2e} + {conc_b:.2e} mM")
    
    # Calculate remaining water volume
    total_surfactant_volume = solution_a['volume_needed_ul'] + solution_b['volume_needed_ul']
    target_volume_before_buffer = WELL_VOLUME_UL - (BUFFER_VOLUME_UL if ADD_BUFFER else 0)
    water_volume = target_volume_before_buffer - total_surfactant_volume
    
    recipe = {
        'wellplate_index': well_index,
        'well_type': 'experiment',
        'control_type': 'experiment',
        'surf_A': surfactant_a_name,
        'surf_B': surfactant_b_name,
        'surf_A_conc_mm': conc_a,
        'surf_B_conc_mm': conc_b,
        'substock_A_name': solution_a['vial_name'],
        'substock_A_conc_mm': solution_a['concentration_mm'],
        'surf_A_volume_ul': solution_a['volume_needed_ul'],
        'substock_B_name': solution_b['vial_name'],
        'substock_B_conc_mm': solution_b['concentration_mm'],
        'surf_B_volume_ul': solution_b['volume_needed_ul'],
        'water_volume_ul': max(0, water_volume),  # Ensure non-negative
        'buffer_volume_ul': BUFFER_VOLUME_UL if ADD_BUFFER else 0.0,
        'buffer_used': SELECTED_BUFFER if ADD_BUFFER else None,
        'pyrene_volume_ul': PYRENE_VOLUME_UL,  # Add pyrene to all wells
        'replicate': replicate
    }
    
    return recipe

def create_complete_experiment_plan(lash_e, surfactant_a_name, surfactant_b_name, experiment_name):
    """
    Create complete experiment plan with simplified, clear data structure.
    Returns: experiment_plan dict with surfactants, stock_solutions_needed, and well_recipes_df
    """
    lash_e.logger.info("Step 2: Creating complete experiment plan...")
    
    # Step 1: Calculate concentration grids (keep existing working logic)
    lash_e.logger.info("  Calculating adaptive concentration grids...")
    concs_a, concs_b = calculate_dual_surfactant_grids(lash_e, surfactant_a_name, surfactant_b_name)
    
    # Step 2: Check achievability and create smart dilution plans (keep existing working logic)
    achievable_a = get_achievable_concentrations(surfactant_a_name, concs_a)
    achievable_b = get_achievable_concentrations(surfactant_b_name, concs_b)
    achievable_concs_a = [c for c in achievable_a if c is not None]
    achievable_concs_b = [c for c in achievable_b if c is not None]
    
    lash_e.logger.info(f"  {surfactant_a_name}: {len(achievable_concs_a)}/{len(concs_a)} concentrations achievable")
    lash_e.logger.info(f"  {surfactant_b_name}: {len(achievable_concs_b)}/{len(concs_b)} concentrations achievable")
    
    # Step 3: Calculate smart dilution plans (keep existing working logic)
    lash_e.logger.info("  Calculating optimal dilution strategies...")
    plan_a, tracker_a = calculate_smart_dilution_plan(lash_e, surfactant_a_name, achievable_concs_a)
    plan_b, tracker_b = calculate_smart_dilution_plan(lash_e, surfactant_b_name, achievable_concs_b)
    
    # Step 4: Create stock solutions list with dilution recipes
    stock_solutions_needed = []
    
    # Calculate dilution recipes to get the "how to make" details
    dilution_recipes = calculate_dilution_recipes(lash_e, plan_a, plan_b, surfactant_a_name, surfactant_b_name)
    
    # Create lookup for recipe details
    recipe_lookup = {recipe['Vial_Name']: recipe for recipe in dilution_recipes}
    
    # Add substocks for surfactant A
    for substock in plan_a['substocks_needed']:
        recipe_details = recipe_lookup.get(substock['vial_name'], {})
        stock_solutions_needed.append({
            'vial_name': substock['vial_name'],
            'surfactant': surfactant_a_name,
            'target_concentration_mm': substock['concentration_mm'],
            'needed_for_concentrations': ', '.join([f"{c:.2e}" for c in substock['needed_for']]),  # Remove brackets
            'source_vial': recipe_details.get('Source_Vial', 'Unknown'),
            'source_concentration_mm': recipe_details.get('Source_Conc_mM', 'Unknown'),
            'source_volume_ml': recipe_details.get('Source_Volume_mL', 'Unknown'),
            'water_volume_ml': recipe_details.get('Water_Volume_mL', 'Unknown'),
            'final_volume_ml': recipe_details.get('Final_Volume_mL', 6.0),
            'dilution_factor': recipe_details.get('Dilution_Factor', 'Unknown')
        })
    
    # Add substocks for surfactant B  
    for substock in plan_b['substocks_needed']:
        recipe_details = recipe_lookup.get(substock['vial_name'], {})
        stock_solutions_needed.append({
            'vial_name': substock['vial_name'],
            'surfactant': surfactant_b_name,
            'target_concentration_mm': substock['concentration_mm'],
            'needed_for_concentrations': ', '.join([f"{c:.2e}" for c in substock['needed_for']]),  # Remove brackets
            'source_vial': recipe_details.get('Source_Vial', 'Unknown'),
            'source_concentration_mm': recipe_details.get('Source_Conc_mM', 'Unknown'),
            'source_volume_ml': recipe_details.get('Source_Volume_mL', 'Unknown'),
            'water_volume_ml': recipe_details.get('Water_Volume_mL', 'Unknown'),
            'final_volume_ml': recipe_details.get('Final_Volume_mL', 6.0),
            'dilution_factor': recipe_details.get('Dilution_Factor', 'Unknown')
        })
    
    # Step 5: Create complete well-by-well recipes DataFrame
    lash_e.logger.info("  Creating complete well recipes...")
    well_recipes = []
    well_index = 0
    
    # Add start control wells
    start_controls = create_control_wells(surfactant_a_name, surfactant_b_name, "start")
    for control in start_controls:
        well_recipe = create_well_recipe_from_control(control, well_index, surfactant_a_name, surfactant_b_name)
        well_recipes.append(well_recipe)
        well_index += 1
    
    # Add grid experiment wells
    for conc_a in achievable_concs_a:
        for conc_b in achievable_concs_b:
            for replicate in range(N_REPLICATES):
                well_recipe = create_well_recipe_from_concentrations(
                    conc_a, conc_b, plan_a, plan_b, well_index, 
                    surfactant_a_name, surfactant_b_name, replicate + 1
                )
                well_recipes.append(well_recipe)
                well_index += 1
    
    # Add end control wells
    end_controls = create_control_wells(surfactant_a_name, surfactant_b_name, "end")
    for control in end_controls:
        well_recipe = create_well_recipe_from_control(control, well_index, surfactant_a_name, surfactant_b_name)
        well_recipes.append(well_recipe)
        well_index += 1
    
    # Convert to DataFrame
    import pandas as pd
    well_recipes_df = pd.DataFrame(well_recipes)
    
    # Step 6: Create experiment plan structure
    experiment_plan = {
        'surfactants': {
            'A': surfactant_a_name,
            'B': surfactant_b_name
        },
        'stock_solutions_needed': stock_solutions_needed,
        'well_recipes_df': well_recipes_df
    }
    
    # Step 7: Export to experiment folder (always save planning CSVs, even in simulation)
    experiment_output_folder = os.path.join("output", experiment_name)
    os.makedirs(experiment_output_folder, exist_ok=True)
    
    # Save well recipes to CSV - directly in experiment folder  
    recipes_csv_path = os.path.join(experiment_output_folder, "experiment_plan_well_recipes.csv")
    well_recipes_df.to_csv(recipes_csv_path, index=False)
    lash_e.logger.info(f"  Saved experiment plan: {recipes_csv_path}")
    
    # Save stock solutions to CSV - directly in experiment folder
    stocks_csv_path = os.path.join(experiment_output_folder, "experiment_plan_stock_solutions.csv")
    stocks_df = pd.DataFrame(stock_solutions_needed)
    stocks_df.to_csv(stocks_csv_path, index=False)
    lash_e.logger.info(f"  Saved stock solutions plan: {stocks_csv_path}")
    
    lash_e.logger.info(f"+ Planning complete: {len(well_recipes)} total wells ({len(achievable_concs_a)} x {len(achievable_concs_b)} grid + {len(start_controls) + len(end_controls)} controls)")
    
    return experiment_plan


def setup_experiment_environment(lash_e, surfactant_a_name, surfactant_b_name, simulate):
    """Initialize experiment environment: create folders, set name, log header."""
    # Create experiment name with timestamp
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"surfactant_grid_{surfactant_a_name}_{surfactant_b_name}_{experiment_timestamp}"
    lash_e.current_experiment_name = experiment_name  # Store for access by other functions
    
    # Create organized experiment folder structure
    experiment_folders = create_experiment_folder_structure(experiment_name)
    experiment_output_folder = experiment_folders['base']
    
    # Log experiment details
    lash_e.logger.info(f"Experiment: {experiment_name}")
    lash_e.logger.info(f"Output folder: {experiment_output_folder}")
    lash_e.logger.info(f"Organized subfolders: validation, measurement_backups, substocks, analysis, logs")
    
    # Log workflow header
    lash_e.logger.info("="*80)
    lash_e.logger.info("ADAPTIVE SURFACTANT GRID SCREENING WORKFLOW")
    lash_e.logger.info("="*80)
    lash_e.logger.info(f"Surfactants: {surfactant_a_name} + {surfactant_b_name}")
    lash_e.logger.info(f"Mode: {'SIMULATION' if simulate else 'HARDWARE'}")
    lash_e.logger.info("")
    
    return experiment_output_folder, experiment_name

def validate_pipetting_system(lash_e, experiment_output_folder):
    """Perform comprehensive pipetting validation tests for all liquid types."""
    if VALIDATION_ONLY:
        lash_e.logger.info("  VALIDATION-ONLY MODE: Running comprehensive pipetting validation...")
    else:
        lash_e.logger.info("  Validating pipetting capability using embedded validation...")
    
    # Use the already-created experiment output folder
    lash_e.logger.info(f"  Validation data will be saved to: {experiment_output_folder}/calibration_validation/")
    
    try:
        # Import embedded validation functions
        from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy
        
        # Define test volumes for different liquid types
        dmso_test_volume = [0.005]  # 5 uL in mL
        
        validation_results = {}
        
        # Test 1: Water validation
        lash_e.logger.info("    Validating water pipetting (10-900 uL)...")
        # Split into two separate tests as requested
        
        # Test 1a: Small water volumes with conditioning
        small_volumes = [0.02,0.05,0.1]
        lash_e.logger.info("      Testing small water volumes (10-100 uL) with conditioning...")
        
        small_water_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial='water',
            destination_vial='water',  
            liquid_type='water',
            volumes_ml=small_volumes,
            replicates=3,
            output_folder=experiment_output_folder,
            switch_pipet=False,
            save_raw_data=not (hasattr(lash_e, 'simulate') and lash_e.simulate),
            condition_tip_enabled=True,
            conditioning_volume_ul=150
        )
        validation_results['water_small'] = small_water_results
        lash_e.logger.info(f"        Small water: R^2={small_water_results['r_squared']:.3f}, Accuracy={small_water_results['mean_accuracy_pct']:.1f}%")
        
        # Test 1b: Large water volumes with conditioning
        large_volumes = [0.2, 0.5, 0.9]
        lash_e.logger.info("      Testing large water volumes (200-900 uL) with conditioning...")
        
        large_water_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial='water',
            destination_vial='water',
            liquid_type='water',
            volumes_ml=large_volumes,
            replicates=3,
            output_folder=experiment_output_folder,
            switch_pipet=False,
            save_raw_data=not (hasattr(lash_e, 'simulate') and lash_e.simulate),
            condition_tip_enabled=True,
            conditioning_volume_ul=900
        )
        validation_results['water_large'] = large_water_results
        lash_e.logger.info(f"        Large water: R^2={large_water_results['r_squared']:.3f}, Accuracy={large_water_results['mean_accuracy_pct']:.1f}%")
        
        # Test 2: DMSO validation  
        lash_e.logger.info("    Validating DMSO pipetting (5 uL) with conditioning...")
        
        dmso_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial='pyrene_DMSO',
            destination_vial='pyrene_DMSO',
            liquid_type='DMSO',
            volumes_ml=dmso_test_volume,
            replicates=5,
            output_folder=experiment_output_folder,
            switch_pipet=False,
            save_raw_data=not (hasattr(lash_e, 'simulate') and lash_e.simulate),
            condition_tip_enabled=True,
            conditioning_volume_ul=25
        )
        validation_results['dmso'] = dmso_results
        lash_e.logger.info(f"      DMSO validation: R^2={dmso_results['r_squared']:.3f}, Accuracy={dmso_results['mean_accuracy_pct']:.1f}%")
        
        # Test 3a: Surfactant A stock validation - Small volumes (small tips)
        surfactant_a_stock = f"{SURFACTANT_A}_stock"
        lash_e.logger.info(f"    Validating {surfactant_a_stock} pipetting - Small volumes (10-100 uL) with conditioning...")
        
        surf_a_small_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial=surfactant_a_stock,
            destination_vial=surfactant_a_stock,
            liquid_type='water',  # Aqueous surfactant solution
            volumes_ml=small_volumes,  # Small volumes: 0.01, 0.05, 0.1 mL
            replicates=3,
            output_folder=experiment_output_folder,
            switch_pipet=False,
            save_raw_data=not (hasattr(lash_e, 'simulate') and lash_e.simulate),
            condition_tip_enabled=True,
            conditioning_volume_ul=100
        )
        validation_results['surfactant_a_small'] = surf_a_small_results
        lash_e.logger.info(f"        Small {surfactant_a_stock}: R^2={surf_a_small_results['r_squared']:.3f}, Accuracy={surf_a_small_results['mean_accuracy_pct']:.1f}%")
        
        # Test 3b: Surfactant A stock validation - Large volumes (large tips)
        lash_e.logger.info(f"    Validating {surfactant_a_stock} pipetting - Large volumes (200-900 uL) with conditioning...")
        
        surf_a_large_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial=surfactant_a_stock,
            destination_vial=surfactant_a_stock,
            liquid_type='water',  # Aqueous surfactant solution
            volumes_ml=large_volumes,  # Large volumes: 0.2, 0.5, 0.9 mL
            replicates=3,
            output_folder=experiment_output_folder,
            switch_pipet=False,
            save_raw_data=not (hasattr(lash_e, 'simulate') and lash_e.simulate),
            condition_tip_enabled=True,
            conditioning_volume_ul=800
        )
        validation_results['surfactant_a_large'] = surf_a_large_results
        lash_e.logger.info(f"        Large {surfactant_a_stock}: R^2={surf_a_large_results['r_squared']:.3f}, Accuracy={surf_a_large_results['mean_accuracy_pct']:.1f}%")
        
        # Test 4a: Surfactant B stock validation - Small volumes (small tips)
        surfactant_b_stock = f"{SURFACTANT_B}_stock"
        lash_e.logger.info(f"    Validating {surfactant_b_stock} pipetting - Small volumes (10-100 uL) with conditioning...")
        
        surf_b_small_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial=surfactant_b_stock,
            destination_vial=surfactant_b_stock,
            liquid_type='water',  # Aqueous surfactant solution
            volumes_ml=small_volumes,  # Small volumes: 0.01, 0.05, 0.1 mL
            replicates=1,
            output_folder=experiment_output_folder,
            switch_pipet=False,
            save_raw_data=not (hasattr(lash_e, 'simulate') and lash_e.simulate),
            condition_tip_enabled=True,
            conditioning_volume_ul=150
        )
        validation_results['surfactant_b_small'] = surf_b_small_results
        lash_e.logger.info(f"        Small {surfactant_b_stock}: R^2={surf_b_small_results['r_squared']:.3f}, Accuracy={surf_b_small_results['mean_accuracy_pct']:.1f}%")
        
        # Test 4b: Surfactant B stock validation - Large volumes (large tips)
        lash_e.logger.info(f"    Validating {surfactant_b_stock} pipetting - Large volumes (200-900 uL) with conditioning...")
        
        surf_b_large_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial=surfactant_b_stock,
            destination_vial=surfactant_b_stock,
            liquid_type='water',  # Aqueous surfactant solution
            volumes_ml=large_volumes,  # Large volumes: 0.2, 0.5, 0.9 mL
            replicates=1,
            output_folder=experiment_output_folder,
            switch_pipet=False,
            save_raw_data=not (hasattr(lash_e, 'simulate') and lash_e.simulate),
            condition_tip_enabled=True,
            conditioning_volume_ul=800
        )
        validation_results['surfactant_b_large'] = surf_b_large_results
        lash_e.logger.info(f"        Large {surfactant_b_stock}: R^2={surf_b_large_results['r_squared']:.3f}, Accuracy={surf_b_large_results['mean_accuracy_pct']:.1f}%")
        
        # Overall validation summary
        all_r_squared = [r['r_squared'] for r in validation_results.values()]
        all_accuracy = [r['mean_accuracy_pct'] for r in validation_results.values()]
        avg_r_squared = sum(all_r_squared) / len(all_r_squared)
        avg_accuracy = sum(all_accuracy) / len(all_accuracy)
        
        lash_e.logger.info(f"    All pipetting validations COMPLETE:")
        lash_e.logger.info(f"      Average R^2: {avg_r_squared:.3f}")
        lash_e.logger.info(f"      Average accuracy: {avg_accuracy:.1f}%")
        lash_e.logger.info(f"      Results saved to: {experiment_output_folder}/calibration_validation/")
        lash_e.logger.info("")
        lash_e.logger.info("="*60)
        lash_e.logger.info("PIPETTING VALIDATION COMPLETE - REVIEW RESULTS")
        lash_e.logger.info("="*60)
        lash_e.logger.info("")
        
        # Early exit for validation-only mode
        if VALIDATION_ONLY:
            lash_e.logger.info("="*60)
            lash_e.logger.info("VALIDATION-ONLY MODE: Exiting after validation completion")
            lash_e.logger.info("="*60)
            return {
                'validation_only': True,
                'validation_results': validation_results,
                'workflow_complete': True
            }
        
        return validation_results
        
    except ImportError as e:
        lash_e.logger.info(f"    Could not import embedded validation: {e}")
        lash_e.logger.info("    Skipping validation (validation system not available)...")
        return None
    except Exception as e:
        lash_e.logger.info(f"    Pipetting validation FAILED: {e}")
        lash_e.logger.info("    Continuing with workflow (validation failure non-critical)...")     
        return None

def create_consolidated_measurement_csv(lash_e, well_map, updated_state, experiment_output_folder):
    """Create consolidated CSV with well mapping, concentrations, and measurement data."""
    import pandas as pd
    import os
    import glob
    
    lash_e.logger.info("Creating consolidated measurement CSV...")
    
    # Create base dataframe from well_map (if available) or generate for existing wells
    if well_map:
        # Use actual well mapping from experiment
        df_base = pd.DataFrame(well_map)
    else:
        # Create basic mapping for wells measured (when CREATE_WELLPLATE = False)
        measurements = updated_state.get('measurements', [])
        if measurements:
            # Get wells that were actually measured
            measured_wells = set()
            for measurement in measurements:
                measured_wells.update(measurement.get('wells_measured', []))
            
            # Create basic mapping
            df_base = pd.DataFrame([
                {
                    'well': well,
                    'plate': updated_state.get('current_plate', 1),
                    'surfactant_a': 'unknown',
                    'surfactant_b': 'unknown', 
                    'conc_a_mm': None,
                    'conc_b_mm': None,
                    'replicate': 1,
                    'is_control': False,
                    'control_type': 'unknown'
                } for well in sorted(measured_wells)
            ])
        else:
            lash_e.logger.warning("No measurement data found for consolidated CSV")
            return None
    
    # Load measurement data from CSV files
    measurement_backups_dir = os.path.join(experiment_output_folder, "measurement_backups")
    
    if os.path.exists(measurement_backups_dir):
        # Find turbidity CSV files
        turbidity_files = glob.glob(os.path.join(measurement_backups_dir, "turbidity_*.csv"))
        fluorescence_files = glob.glob(os.path.join(measurement_backups_dir, "fluorescence_*.csv"))
        
        # Load and combine turbidity data
        turbidity_df = None
        if turbidity_files:
            turbidity_dfs = []
            for file in turbidity_files:
                try:
                    df = pd.read_csv(file, index_col=0)
                    turbidity_dfs.append(df)
                except Exception as e:
                    lash_e.logger.warning(f"Could not load turbidity file {file}: {e}")
            
            if turbidity_dfs:
                turbidity_df = pd.concat(turbidity_dfs, ignore_index=False)
                # Convert index to well column for merging
                turbidity_df = turbidity_df.reset_index()
                turbidity_df = turbidity_df.rename(columns={'index': 'well_position'})
        
        # Load and combine fluorescence data
        fluorescence_df = None
        if fluorescence_files:
            fluorescence_dfs = []
            for file in fluorescence_files:
                try:
                    # Read CSV with special handling for Cytation format
                    temp_df = pd.read_csv(file)
                    
                    # Check if this is Cytation format (wavelength info in row 2)
                    if len(temp_df) > 0 and temp_df.iloc[0].dropna().tolist():
                        # This is Cytation format - wavelength info is in row 1 (0-indexed)
                        wavelengths = temp_df.iloc[0].tolist()  # ['334_373', '334_384']
                        
                        # Create proper column names mapping
                        new_columns = {}
                        for i, wl in enumerate(wavelengths):
                            if pd.notna(wl) and str(wl) in ['334_373', '334_384']:
                                original_col = temp_df.columns[i]
                                new_columns[original_col] = f"fluorescence_{wl}"
                        
                        # Remove the wavelength row
                        temp_df = temp_df.iloc[1:].copy()  # Remove wavelength row
                        temp_df = temp_df.rename(columns=new_columns)
                        temp_df = temp_df.reset_index()
                        temp_df = temp_df.rename(columns={'index': 'well_position'})
                    else:
                        # Standard CSV format
                        temp_df = pd.read_csv(file, index_col=0)
                        temp_df = temp_df.reset_index()
                        temp_df = temp_df.rename(columns={'index': 'well_position'})
                    
                    fluorescence_dfs.append(temp_df)
                except Exception as e:
                    lash_e.logger.warning(f"Could not load fluorescence file {file}: {e}")
            
            if fluorescence_dfs:
                fluorescence_df = pd.concat(fluorescence_dfs, ignore_index=True)
                lash_e.logger.info(f"  Loaded {len(fluorescence_df)} fluorescence measurements")
        
        # Convert well numbers to well positions for merging (A1, A2, etc.)
        def well_number_to_position(well_number):
            """Convert well number (0-based) to well position (A1, A2, etc.)"""
            row = well_number // 12  # 12 columns per row
            col = well_number % 12
            return f"{chr(65 + row)}{col + 1}"
        
        df_base['well_position'] = df_base['well'].apply(well_number_to_position)
        
        # Merge measurement data
        df_consolidated = df_base.copy()
        
        if turbidity_df is not None:
            df_consolidated = df_consolidated.merge(turbidity_df, on='well_position', how='left', suffixes=('', '_turb'))
            lash_e.logger.info(f"  Merged turbidity data: {len(turbidity_df)} measurements")
        
        if fluorescence_df is not None:
            df_consolidated = df_consolidated.merge(fluorescence_df, on='well_position', how='left', suffixes=('', '_fluor'))
            lash_e.logger.info(f"  Merged fluorescence data: {len(fluorescence_df)} measurements")
        
        # Clean up column names - handle potential column name variations
        # Check what columns we actually have after merge
        actual_columns = list(df_consolidated.columns)
        lash_e.logger.info(f"  Columns after merge: {actual_columns}")
        
        # Columns should already have proper names from CSV reading
        # Just ensure turbidity column is named correctly if present
        column_mapping = {}
        for col in actual_columns:
            if 'Absorbance' in col or '600' in col:
                column_mapping[col] = 'turbidity_600'
        
        if column_mapping:
            df_consolidated = df_consolidated.rename(columns=column_mapping)
            lash_e.logger.info(f"  Renamed columns: {column_mapping}")
        
        # Calculate fluorescence ratio (same as cmc_shared)
        if 'fluorescence_334_373' in df_consolidated.columns and 'fluorescence_334_384' in df_consolidated.columns:
            df_consolidated['ratio'] = df_consolidated['fluorescence_334_373'] / df_consolidated['fluorescence_334_384']
            lash_e.logger.info("  Calculated fluorescence ratio (334_373/334_384)")
        
        # Reorder columns for readability
        column_order = ['well', 'well_position', 'plate', 'surfactant_a', 'surfactant_b', 
                       'conc_a_mm', 'conc_b_mm', 'replicate', 'is_control', 'control_type',
                       'turbidity_600', 'fluorescence_334_373', 'fluorescence_334_384', 'ratio']
        
        available_columns = [col for col in column_order if col in df_consolidated.columns]
        df_consolidated = df_consolidated[available_columns]
        
        # Save consolidated CSV
        consolidated_path = os.path.join(experiment_output_folder, "consolidated_measurements.csv")
        df_consolidated.to_csv(consolidated_path, index=False)
        
        lash_e.logger.info(f"  Saved consolidated CSV: {consolidated_path}")
        lash_e.logger.info(f"  Total rows: {len(df_consolidated)}")
        lash_e.logger.info(f"  Columns: {list(df_consolidated.columns)}")
        
        return df_consolidated
    
    else:
        lash_e.logger.warning(f"Measurement backups directory not found: {measurement_backups_dir}")
        return None

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
    # Initialize Lash_E FIRST so logger is available from the start
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
    lash_e.logger.info("Step 1: Initializing Lash_E coordinator...")
    lash_e.logger.info(f"  Mode: {'Simulation' if simulate else 'Hardware'}")
    
    # Setup experiment environment and folders
    experiment_output_folder, experiment_name = setup_experiment_environment(lash_e, surfactant_a_name, surfactant_b_name, simulate)
  
    # Validate system state
    lash_e.logger.info("  Validating robot and track status...")
    lash_e.nr_robot.check_input_file()
    lash_e.nr_track.check_input_file()

    
    # Home robot to ensure clean starting position
    lash_e.logger.info("  Homing robot to ensure clean starting position...")
    lash_e.nr_robot.home_robot_components()  # Ensure robot is homed before checking vials
    
    # Fill water vials to maximum capacity before workflow
    lash_e.logger.info("  Ensuring water vials are full before workflow...")
    fill_water_vial(lash_e, "water")
    fill_water_vial(lash_e, "water_2")
    
    # Validate pipetting capability if enabled or if in validation-only mode
    if VALIDATE_LIQUIDS or VALIDATION_ONLY:
        validation_results = validate_pipetting_system(lash_e, experiment_output_folder)
        
        # Early exit for validation-only mode
        if validation_results and validation_results.get('validation_only'):
            return validation_results
    else:
        lash_e.logger.info("  Pipetting validation disabled (set VALIDATE_LIQUIDS=True or VALIDATION_ONLY=True to enable)")
    
    # Robot is ready - no initial positioning needed
    lash_e.logger.info("  Robot ready for operations")
    
    # Get fresh wellplate only if creating new wellplate
    if CREATE_WELLPLATE:
        lash_e.logger.info("  Getting fresh wellplate...")
        lash_e.nr_track.get_new_wellplate()
    else:
        lash_e.logger.info("  Using existing wellplate (CREATE_WELLPLATE = False)")
    
    lash_e.logger.info("+ Lash_E initialization complete")
    lash_e.logger.info("")
    
    # STEP 2: Create complete experiment plan with simplified structure
    experiment_plan = create_complete_experiment_plan(lash_e, surfactant_a_name, surfactant_b_name, experiment_name)
    
    # Extract the well recipes DataFrame
    well_recipes_df = experiment_plan['well_recipes_df']
    stock_solutions_needed = experiment_plan['stock_solutions_needed']
    
    lash_e.logger.info(f"+ Experiment plan created: {len(well_recipes_df)} wells planned")
    lash_e.logger.info(f"+ Stock solutions needed: {len(stock_solutions_needed)}")   
    
    # STEP 3: Create physical substocks BEFORE dispensing
    lash_e.logger.info("Step 4: Creating physical substocks from calculated recipes...")
    
    # Get dilution recipes from the experiment plan
    dilution_recipes = []
    for stock in stock_solutions_needed:
        if stock['source_vial'] != 'Unknown':  # Only include valid recipes
            dilution_recipes.append({
                'Vial_Name': stock['vial_name'],
                'Surfactant': stock['surfactant'],
                'Target_Conc_mM': stock['target_concentration_mm'],
                'Source_Vial': stock['source_vial'],
                'Source_Conc_mM': stock['source_concentration_mm'],
                'Source_Volume_mL': stock['source_volume_ml'],
                'Water_Volume_mL': stock['water_volume_ml'],
                'Final_Volume_mL': stock['final_volume_ml']
            })
    
    # Sort by concentration (highest first) for correct creation order
    dilution_recipes.sort(key=lambda x: x['Target_Conc_mM'], reverse=True)
    
    # Create substocks in the correct order (highest concentration first)
    if dilution_recipes:
        created_substocks = create_substocks_from_recipes(lash_e, dilution_recipes)
        newly_created = [s for s in created_substocks if s['created']]
        already_existing = [s for s in created_substocks if not s['created'] and not s.get('error')]
        actually_failed = [s for s in created_substocks if s.get('error')]
        
        lash_e.logger.info(f"+ Substock creation complete: {len(newly_created)} newly created, {len(already_existing)} already existed, {len(actually_failed)} failed")
        
        if already_existing:
            lash_e.logger.info(f"Substocks already existed (OK):")
            for substock in already_existing:
                lash_e.logger.info(f"  {substock['vial_name']}: already available")
        
        if actually_failed:
            lash_e.logger.info(f"Failed substocks (ERRORS):")
            for substock in actually_failed:
                error_msg = substock.get('error', 'Unknown error')
                lash_e.logger.info(f"  {substock['vial_name']}: {error_msg}")
    else:
        lash_e.logger.info("+ No substocks needed (using only stock solutions)")
    
    # STEP 5: Execute dispensing and measurements in clear phases
    lash_e.logger.info("Step 5: Executing dispensing and measurements...")
    
    # Initialize measurement columns in the DataFrame
    well_recipes_df['turbidity_600'] = None
    well_recipes_df['fluorescence_334_373'] = None 
    well_recipes_df['fluorescence_334_384'] = None
    well_recipes_df['ratio'] = None
    
    if not CREATE_WELLPLATE:
        lash_e.logger.info("CREATE_WELLPLATE = False: Skipping dispensing")
    else:
        # Process in batches to match measurement intervals
        total_wells = len(well_recipes_df)
        lash_e.logger.info(f"Total wells to process: {total_wells}")
        
        for batch_start in range(0, total_wells, MEASUREMENT_INTERVAL):
            batch_end = min(batch_start + MEASUREMENT_INTERVAL, total_wells)
            batch_df = well_recipes_df.iloc[batch_start:batch_end]
            
            lash_e.logger.info(f"\nProcessing batch {batch_start//MEASUREMENT_INTERVAL + 1}: wells {batch_start}-{batch_end-1}")
            
            # DISPENSING PHASE: Use unified dispensing for each component with proper vial positioning
            # Get unique vial names needed for this batch
            surf_a_vials = batch_df[batch_df['surf_A_volume_ul'] > 0]['substock_A_name'].dropna().unique()
            surf_b_vials = batch_df[batch_df['surf_B_volume_ul'] > 0]['substock_B_name'].dropna().unique()
            
            # Position surfactant A vials by concentration (dilute → concentrated)
            if len(surf_a_vials) > 0:
                sorted_surf_a_vials = position_surfactant_vials_by_concentration(lash_e, surf_a_vials, batch_df, 'A')
                
                # Dispense surfactant A substocks in concentration order
                for surf_a_vial in sorted_surf_a_vials:
                    dispense_component_to_wellplate(lash_e, batch_df, surf_a_vial, 'water', 'surf_A_volume_ul')
                
                # Return surfactant A vials to home
                lash_e.nr_robot.remove_pipet()
                return_surfactant_vials_home(lash_e, sorted_surf_a_vials, 'A')
            
            # Dispense water - split between two water vials to prevent running out
            water_wells = batch_df[batch_df['water_volume_ul'] > 0]
            if len(water_wells) > 0:
                # Split water dispensing in half
                mid_point = len(water_wells) // 2
                water_batch_1 = water_wells.iloc[:mid_point]
                water_batch_2 = water_wells.iloc[mid_point:]

                lash_e.nr_robot.move_vial_to_location('water', 'main_8mL_rack', 44)
                lash_e.nr_robot.move_vial_to_location('water_2', 'main_8mL_rack', 45)

                # Dispense first half with water vial
                if len(water_batch_1) > 0:
                    dispense_component_to_wellplate(lash_e, water_batch_1, 'water', 'water', 'water_volume_ul')
                    
                # Dispense second half with water_2 vial
                if len(water_batch_2) > 0:
                    dispense_component_to_wellplate(lash_e, water_batch_2, 'water_2', 'water', 'water_volume_ul')

                lash_e.nr_robot.remove_pipet()
                return_water_vial_home(lash_e, 'water')
                return_water_vial_home(lash_e, 'water_2')
            
            # Dispense buffer with safe positioning
            if ADD_BUFFER:
                lash_e.logger.info(f"  Positioning {SELECTED_BUFFER} buffer at clamp (safe position)")
                lash_e.nr_robot.move_vial_to_location(SELECTED_BUFFER, 'clamp', 0)
                dispense_component_to_wellplate(lash_e, batch_df, SELECTED_BUFFER, 'water', 'buffer_volume_ul')
                lash_e.nr_robot.remove_pipet()
                lash_e.nr_robot.return_vial_home(SELECTED_BUFFER)
            
            # Position surfactant B vials by concentration (dilute → concentrated)
            if len(surf_b_vials) > 0:
                sorted_surf_b_vials = position_surfactant_vials_by_concentration(lash_e, surf_b_vials, batch_df, 'B')
                
                # Dispense surfactant B substocks in concentration order
                for surf_b_vial in sorted_surf_b_vials:
                    dispense_component_to_wellplate(lash_e, batch_df, surf_b_vial, 'water', 'surf_B_volume_ul')
                
                # Return surfactant B vials to home
                lash_e.nr_robot.remove_pipet()
                return_surfactant_vials_home(lash_e, sorted_surf_b_vials, 'B')
            
            # MEASUREMENT PHASE: Turbidity first
            wells_in_batch = batch_df['wellplate_index'].tolist()
            lash_e.logger.info("  Measuring turbidity...")
            turbidity_data = measure_turbidity(lash_e, wells_in_batch)
            
            # Add turbidity data to DataFrame by order
            if turbidity_data is not None:
                if not lash_e.simulate:
                    # Hardware mode - turbidity_data is a DataFrame, extract values in order
                    if hasattr(turbidity_data, 'values') and len(turbidity_data) > 0:
                        # Get the turbidity values in order (first column with numeric data)
                        turbidity_values = turbidity_data.iloc[:, 0].values  # First data column
                        for i, well_idx in enumerate(wells_in_batch):
                            if i < len(turbidity_values):
                                well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'turbidity_600'] = turbidity_values[i]
                else:
                    # Simulation mode - simple values
                    for i, well_idx in enumerate(wells_in_batch):
                        well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'turbidity_600'] = 0.5
            
            # Add DMSO with safe positioning then measure fluorescence
            lash_e.logger.info("  Positioning pyrene_DMSO at clamp (safe position)")
            lash_e.nr_robot.move_vial_to_location('pyrene_DMSO', 'clamp', 0)
            dispense_component_to_wellplate(lash_e, batch_df, 'pyrene_DMSO', 'DMSO', 'pyrene_volume_ul')
            lash_e.nr_robot.remove_pipet()
            lash_e.nr_robot.return_vial_home('pyrene_DMSO')
            lash_e.logger.info("  Measuring fluorescence...")
            fluorescence_data = measure_fluorescence(lash_e, wells_in_batch)
            
            # Add fluorescence data to DataFrame by order
            if fluorescence_data is not None:
                if not lash_e.simulate:
                    # Hardware mode - fluorescence_data is a DataFrame, extract values in order
                    if hasattr(fluorescence_data, 'values') and len(fluorescence_data) > 0:
                        # Get both fluorescence columns in order
                        val_373_list = fluorescence_data['334_373'].values if '334_373' in fluorescence_data.columns else []
                        val_384_list = fluorescence_data['334_384'].values if '334_384' in fluorescence_data.columns else []
                        
                        for i, well_idx in enumerate(wells_in_batch):
                            if i < len(val_373_list) and i < len(val_384_list):
                                val_373 = val_373_list[i]
                                val_384 = val_384_list[i]
                                ratio = val_373 / val_384 if val_384 != 0 else None
                                
                                well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'fluorescence_334_373'] = val_373
                                well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'fluorescence_334_384'] = val_384
                                well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'ratio'] = ratio
                else:
                    # Simulation mode - simple values
                    for i, well_idx in enumerate(wells_in_batch):
                        val_373 = 80.0
                        val_384 = 100.0
                        ratio = val_373 / val_384
                        well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'fluorescence_334_373'] = val_373
                        well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'fluorescence_334_384'] = val_384
                        well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'ratio'] = ratio
    

    
    # STEP 6: Save results to experiment folder
    lash_e.logger.info("Step 6: Saving results...")
    output_folder = experiment_output_folder
    
    # Save the complete well recipes with measurements
    final_results_path = os.path.join(output_folder, "complete_experiment_results.csv")
    well_recipes_df.to_csv(final_results_path, index=False)
    lash_e.logger.info(f"  Complete results saved to: {final_results_path}")
    
    lash_e.logger.info(f"+ Results saved to: {output_folder}")
    
    # Get actual pipette usage breakdown 
    pipette_breakdown = get_pipette_usage_breakdown(lash_e)
    lash_e.logger.info(f"+ Pipette tips used: {pipette_breakdown['large_tips']} large, {pipette_breakdown['small_tips']} small (total: {pipette_breakdown['total']}) ({'simulated' if simulate else 'actual'})")
    
    # Summary statistics
    experiment_wells = well_recipes_df[well_recipes_df['well_type'] == 'experiment']
    control_wells = well_recipes_df[well_recipes_df['well_type'] == 'control']
    measured_wells = well_recipes_df[well_recipes_df['turbidity_600'].notna()]
    
    lash_e.logger.info("\n" + "="*60)
    lash_e.logger.info("EXPERIMENT COMPLETE - SUMMARY")
    lash_e.logger.info("="*60)
    lash_e.logger.info(f"Surfactants: {surfactant_a_name} + {surfactant_b_name}")
    lash_e.logger.info(f"Wells: {len(experiment_wells)} experiment + {len(control_wells)} control = {len(well_recipes_df)} total")
    lash_e.logger.info(f"Measurements: {len(measured_wells)} wells measured")
    lash_e.logger.info(f"Mode: {'SIMULATION' if simulate else 'HARDWARE'}")
    lash_e.logger.info(f"Results: {final_results_path}")
    lash_e.logger.info("="*60)
    
    # Display final DataFrame for verification
    if not well_recipes_df.empty:
        lash_e.logger.info("DataFrame sample with measurements:")
        sample_cols = ['wellplate_index', 'surf_A_conc_mm', 'surf_B_conc_mm', 'turbidity_600', 'fluorescence_334_373', 'fluorescence_334_384', 'ratio']
        existing_cols = [col for col in sample_cols if col in well_recipes_df.columns]
        sample_df = well_recipes_df[existing_cols].head(10)
        lash_e.logger.info(f"\n{sample_df.to_string()}")
    
    # Return clean results with just the essential data
    return {
        'surfactant_a': surfactant_a_name,
        'surfactant_b': surfactant_b_name,
        'well_recipes_df': well_recipes_df,  # Complete DataFrame with measurements
        'experiment_plan': experiment_plan,
        'total_wells': len(well_recipes_df),
        'measured_wells': len(measured_wells),
        'output_folder': output_folder,
        'pipette_breakdown': pipette_breakdown,
        'simulation': simulate,
        'workflow_complete': True
    }


if __name__ == "__main__":
    """
    Run the adaptive surfactant grid screening workflow.
    """
    
    # TEST PLANNING ONLY (comment out to run full workflow)
    # test_planning_only(SURFACTANT_A, SURFACTANT_B)
    
    # FULL WORKFLOW 
    print("Starting adaptive surfactant grid screening...")
    if not SIMULATE:
        slack_agent.send_slack_message("Starting adaptive surfactant grid screening workflow...")

    results = execute_adaptive_surfactant_screening(
        surfactant_a_name=SURFACTANT_A, 
        surfactant_b_name=SURFACTANT_B, 
        simulate=SIMULATE
    )
    # if results and results['workflow_complete']:
    #     print("="*80)
    #     print("WORKFLOW COMPLETE!")
    #     print("="*80)
    #     print(f"+ Surfactants: {results['surfactant_a']} + {results['surfactant_b']}")
    #     print(f"+ Wells processed: {results['total_wells']}")
    #     print(f"+ Plates used: {results['plates_used']}")
    #     breakdown = results['pipette_breakdown']
    #     print(f"+ Pipette tips: {breakdown['large_tips']} large, {breakdown['small_tips']} small (total: {breakdown['total']})")
    #     print(f"+ Measurements: {len(results['measurements'])} intervals")
    #     print(f"+ Mode: {'Simulation' if results['simulation'] else 'Hardware'}")
    #     print(f"+ Results saved to: {results['output_folder']}")

    # if not SIMULATE:
    #     slack_agent.send_slack_message("Completed adaptive surfactant grid screening workflow...")
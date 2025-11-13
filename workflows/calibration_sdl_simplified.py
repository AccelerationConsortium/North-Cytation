# calibration_sdl_simplified.py
"""
Simplified calibration workflow that eliminates dynamic cutoff and cascading precision tests.

Workflow:
1. External Data Loading OR Screening (SOBOL/LLM exploration)
2. Overaspirate calibration (same as modular)
3. 3-objective optimization (deviation, variability, time)
4. Simple stopping: 60 measurements OR 6 "GOOD" parameter sets
5. Best candidate selection: rank by accuracy → precision → time
6. Single precision test

First volume: optimize all parameters
Subsequent volumes: selective optimization (volume-dependent parameters only)

EXTERNAL DATA INTEGRATION:
- Can load pre-existing calibration data from CSV files
- Automatically replaces screening phase when external data available
- Supports volume and liquid filtering for targeted data selection
- Falls back to traditional screening if no external data found
- Expected CSV format: volume, aspirate_speed, dispense_speed, deviation, time, etc.
"""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
from sympy import false

# Add paths for imports
sys.path.append("../North-Cytation")

# Import base functionality
from calibration_sdl_base import (
    pipet_and_measure_simulated, pipet_and_measure, strip_tuples, save_analysis,
    LIQUIDS, set_vial_management
)
import calibration_sdl_base as base_module
from master_usdl_coordinator import Lash_E

# Import optimizers
try:
    import recommenders.pipetting_optimizer_3objectives as optimizer_3obj
    OPTIMIZER_3OBJ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import 3-objectives optimizer: {e}")
    optimizer_3obj = None
    OPTIMIZER_3OBJ_AVAILABLE = False

try:
    import recommenders.pipetting_optimizer_single_objective as optimizer_single
    OPTIMIZER_SINGLE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import single-objective optimizer: {e}")
    optimizer_single = None
    OPTIMIZER_SINGLE_AVAILABLE = False

try:
    import recommenders.pipeting_optimizer_v2 as recommender_v2  # Legacy fallback
    RECOMMENDER_V2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import v2 optimizer: {e}")
    recommender_v2 = None
    RECOMMENDER_V2_AVAILABLE = False

# Conditionally import slack_agent and LLM optimizer
try:
    import slack_agent
    SLACK_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import slack_agent: {e}")
    slack_agent = None
    SLACK_AVAILABLE = False

try:
    import recommenders.llm_optimizer as llm_opt
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import LLM optimizer: {e}")
    llm_opt = None
    LLM_AVAILABLE = False

# --- DEFAULT EXPERIMENT CONFIG ---
DEFAULT_LIQUID = "glycerol"
DEFAULT_SIMULATE = False
DEFAULT_SEED = 7
DEFAULT_VOLUMES = [0.05, 0.025, 0.01]  # mL
DEFAULT_INPUT_VIAL_STATUS_FILE = "status/calibration_vials_short.csv"

# External data loading configuration
DEFAULT_EXTERNAL_DATA_PATH = None  # Path to external calibration data CSV
DEFAULT_USE_EXTERNAL_DATA = False  # Enable external data loading
DEFAULT_EXTERNAL_DATA_VOLUME_FILTER = None  # Filter external data by volume (mL), None = use all
DEFAULT_EXTERNAL_DATA_LIQUID_FILTER = None  # Filter external data by liquid, None = use all

# Measurement amount hyperparameters
DEFAULT_MAX_MEASUREMENTS = 96  # Total measurements for entire calibration
DEFAULT_MAX_MEASUREMENTS_INITIAL_VOLUME = 60  # Maximum measurements for first volume
DEFAULT_MIN_GOOD_PARAMETER_SETS = 6  # Minimum "good" parameter sets before stopping
DEFAULT_PARAMETER_SETS_PER_RECOMMENDATION = 1  # Number of parameter sets requested per optimization round
DEFAULT_PRECISION_MEASUREMENTS = 3
DEFAULT_INITIAL_PARAMETER_SETS = 5

# Active learning settings
DEFAULT_USE_LLM_FOR_SCREENING = False
DEFAULT_BAYESIAN_MODEL_TYPE = 'qNEHVI'  # Use multi-objective optimization for first volume
DEFAULT_BAYESIAN_MODEL_TYPE_SUBSEQUENT = 'qLogEI'  # Single-objective (deviation only) for subsequent volumes

# Overaspirate configuration
DEFAULT_OVERASPIRATE_BASE_UL = 5.0
DEFAULT_OVERASPIRATE_SCALING_PERCENT = 5.0
DEFAULT_AUTO_CALIBRATE_OVERVOLUME = True
DEFAULT_OVERVOLUME_CALIBRATION_BUFFER_UL = 5.0  # Fixed: Was 2.0, should be 5.0μL
DEFAULT_OVERVOLUME_MAX_BASE_UL = 50.0
DEFAULT_OVERVOLUME_MAX_PERCENT = 100.0

# Adaptive measurement hyperparameters
DEFAULT_ADAPTIVE_DEVIATION_THRESHOLD = 10.0  # % threshold for running additional replicates
DEFAULT_ADAPTIVE_PENALTY_VARIABILITY = 100.0  # Penalty value for high deviation trials

# Ranking system weights
DEFAULT_ACCURACY_WEIGHT = 0.5  # Weight for accuracy in composite scoring
DEFAULT_PRECISION_WEIGHT = 0.4  # Weight for precision in composite scoring  
DEFAULT_TIME_WEIGHT = 0.1  # Weight for time in composite scoring

# Optimizer objective thresholds (higher thresholds = more gradient, more forgiving)
DEFAULT_OPTIMIZER_DEVIATION_THRESHOLD = 50.0    # Deviation threshold for optimizer (%)
DEFAULT_OPTIMIZER_VARIABILITY_THRESHOLD = 25.0  # Variability threshold for optimizer (%) - increased from 10%
DEFAULT_OPTIMIZER_TIME_THRESHOLD = 120.0         # Time threshold for optimizer (seconds)

# Simulation tolerance multipliers (defaults - can be overridden by environment variables)
DEFAULT_SIM_DEV_MULTIPLIER = 2.0  # Simulation tolerance multiplier for deviation
DEFAULT_SIM_VAR_MULTIPLIER = 2.0  # Simulation tolerance multiplier for variation

# Fixed parameters configuration - parameters to hold constant during optimization
DEFAULT_FIXED_PARAMETERS = {}  # Dict of parameter_name: value to fix during optimization

# Volume tolerance ranges
VOLUME_TOLERANCE_RANGES = [
    {'min_ul': 200, 'max_ul': 1000, 'tolerance_pct': 1.0, 'name': 'large_volume'},
    {'min_ul': 60,  'max_ul': 200,  'tolerance_pct': 2.0, 'name': 'medium_large_volume'},
    {'min_ul': 20,  'max_ul': 60,   'tolerance_pct': 3.0, 'name': 'medium_volume'},
    {'min_ul': 1,   'max_ul': 20,   'tolerance_pct': 5.0, 'name': 'small_volume'},
    {'min_ul': 0,   'max_ul': 1,    'tolerance_pct': 10.0, 'name': 'micro_volume'},
]

# --- RUNTIME CONFIG (MUTABLE) ---
LIQUID = DEFAULT_LIQUID
SIMULATE = DEFAULT_SIMULATE
SEED = DEFAULT_SEED
INITIAL_PARAMETER_SETS = DEFAULT_INITIAL_PARAMETER_SETS
PARAMETER_SETS_PER_RECOMMENDATION = DEFAULT_PARAMETER_SETS_PER_RECOMMENDATION
# REPLICATES = DEFAULT_REPLICATES  # DEPRECATED: Replaced by PRECISION_MEASUREMENTS system
PRECISION_MEASUREMENTS = DEFAULT_PRECISION_MEASUREMENTS
VOLUMES = DEFAULT_VOLUMES.copy()
MAX_MEASUREMENTS = DEFAULT_MAX_MEASUREMENTS
MAX_MEASUREMENTS_INITIAL_VOLUME = DEFAULT_MAX_MEASUREMENTS_INITIAL_VOLUME
INPUT_VIAL_STATUS_FILE = DEFAULT_INPUT_VIAL_STATUS_FILE
FIXED_PARAMETERS = DEFAULT_FIXED_PARAMETERS.copy()  # Parameters to fix during optimization

# Simplified stopping criteria  
MIN_GOOD_PARAMETER_SETS = DEFAULT_MIN_GOOD_PARAMETER_SETS

USE_LLM_FOR_SCREENING = DEFAULT_USE_LLM_FOR_SCREENING
BAYESIAN_MODEL_TYPE = DEFAULT_BAYESIAN_MODEL_TYPE
BAYESIAN_MODEL_TYPE_SUBSEQUENT = DEFAULT_BAYESIAN_MODEL_TYPE_SUBSEQUENT

# Overaspirate config
OVERASPIRATE_BASE_UL = DEFAULT_OVERASPIRATE_BASE_UL
OVERASPIRATE_SCALING_PERCENT = DEFAULT_OVERASPIRATE_SCALING_PERCENT
AUTO_CALIBRATE_OVERVOLUME = DEFAULT_AUTO_CALIBRATE_OVERVOLUME
OVERVOLUME_CALIBRATION_BUFFER_UL = DEFAULT_OVERVOLUME_CALIBRATION_BUFFER_UL
OVERVOLUME_MAX_BASE_UL = DEFAULT_OVERVOLUME_MAX_BASE_UL
OVERVOLUME_MAX_PERCENT = DEFAULT_OVERVOLUME_MAX_PERCENT

# Adaptive measurement config
ADAPTIVE_DEVIATION_THRESHOLD = DEFAULT_ADAPTIVE_DEVIATION_THRESHOLD
ADAPTIVE_PENALTY_VARIABILITY = DEFAULT_ADAPTIVE_PENALTY_VARIABILITY

# Ranking system weights
ACCURACY_WEIGHT = DEFAULT_ACCURACY_WEIGHT
PRECISION_WEIGHT = DEFAULT_PRECISION_WEIGHT
TIME_WEIGHT = DEFAULT_TIME_WEIGHT

# Optimizer objective thresholds
OPTIMIZER_DEVIATION_THRESHOLD = DEFAULT_OPTIMIZER_DEVIATION_THRESHOLD
OPTIMIZER_VARIABILITY_THRESHOLD = DEFAULT_OPTIMIZER_VARIABILITY_THRESHOLD
OPTIMIZER_TIME_THRESHOLD = DEFAULT_OPTIMIZER_TIME_THRESHOLD

# Simulation tolerance multipliers
SIM_DEV_MULTIPLIER = DEFAULT_SIM_DEV_MULTIPLIER
SIM_VAR_MULTIPLIER = DEFAULT_SIM_VAR_MULTIPLIER

# Selective parameter optimization
USE_SELECTIVE_OPTIMIZATION = True
VOLUME_DEPENDENT_PARAMS = ["blowout_vol", "overaspirate_vol"]

# Transfer learning configuration
DEFAULT_USE_TRANSFER_LEARNING = True  # Enable cross-volume parameter transfer
USE_TRANSFER_LEARNING = DEFAULT_USE_TRANSFER_LEARNING

# External data loading (runtime variables)
EXTERNAL_DATA_PATH = DEFAULT_EXTERNAL_DATA_PATH
USE_EXTERNAL_DATA = DEFAULT_USE_EXTERNAL_DATA
EXTERNAL_DATA_VOLUME_FILTER = DEFAULT_EXTERNAL_DATA_VOLUME_FILTER
EXTERNAL_DATA_LIQUID_FILTER = DEFAULT_EXTERNAL_DATA_LIQUID_FILTER

# Global measurement counter and volume-specific calibrations
global_measurement_count = 0
volume_overaspirate_calibrations = {}  # Store volume-specific overaspirate calibrations
global_ax_client = None  # Global optimizer for transfer learning

def pipet_and_measure_tracked(*args, **kwargs):
    """
    Wrapper around pipet_and_measure that automatically tracks measurement count.
    Every call to this function represents one actual measurement/data point.
    ENFORCES HARD BUDGET LIMIT - returns None if budget exceeded.
    """
    global global_measurement_count
    
    # HARD BUDGET ENFORCEMENT: Check before making any measurement
    if global_measurement_count >= MAX_MEASUREMENTS:
        print(f"🛑 HARD BUDGET LIMIT: Skipping measurement {global_measurement_count + 1}/{MAX_MEASUREMENTS}")
        return None  # Return None instead of raising exception
    
    # VIAL MANAGEMENT: Check vials before every pipetting operation
    # Extract lash_e from arguments (always first parameter)
    if len(args) > 0:
        lash_e = args[0]
        
        # Get current vial assignments from vial management config
        if base_module._VIAL_MANAGEMENT_CONFIG_OVERRIDE:
            cfg = {**base_module.VIAL_MANAGEMENT_DEFAULTS, **base_module._VIAL_MANAGEMENT_CONFIG_OVERRIDE}
            current_measurement_vial = cfg.get('measurement_vial', args[2] if len(args) > 2 else "measurement_vial_0")
        else:
            current_measurement_vial = args[2] if len(args) > 2 else "measurement_vial_0"
            
        state = {"measurement_vial_name": current_measurement_vial}
        
        # Apply vial management and get correct liquid source
        try:
            liquid_source = get_liquid_source_with_vial_management(lash_e, state)
            
            # Update both source and measurement vials if they changed
            source_changed = len(args) >= 2 and liquid_source != args[1]
            measurement_changed = len(args) >= 3 and state["measurement_vial_name"] != args[2]
            
            if source_changed or measurement_changed:
                # Update parameters to match vial management changes
                args_list = list(args)
                
                if source_changed:
                    args_list[1] = liquid_source
                    
                if measurement_changed:
                    args_list[2] = state["measurement_vial_name"]
                    
                args = tuple(args_list)
        except Exception as e:
            print(f"⚠️ Vial management error: {e}")
    
    # Call the original function
    result = pipet_and_measure(*args, **kwargs)
    
    # Increment the global measurement counter
    global_measurement_count += 1
    
    # Log if we're approaching the limit
    if global_measurement_count >= MAX_MEASUREMENTS - 2:
        print(f"   ⚠️  Budget warning: {global_measurement_count}/{MAX_MEASUREMENTS} measurements used")
    
    return result

def get_volume_measurement_count(start_count):
    """Get the number of measurements used for this volume since start_count."""
    global global_measurement_count
    return global_measurement_count - start_count

def reset_global_measurement_count():
    """Reset the global measurement counter (for new experiments)."""
    global global_measurement_count
    global_measurement_count = 0

def extract_performance_metrics(all_results, volume_ml, best_params, raw_measurements=None, best_candidate=None):
    """
    Extract key performance metrics for a volume from the best candidate.
    
    Args:
        best_candidate: The actual ranked best candidate (contains all the data we need)
    
    Returns dict with volume_target, volume_measured, average_deviation, variability, time,
    and tolerance check results.
    """
    target_ul = volume_ml * 1000  # Convert to μL
    
    # Use the ranked best candidate data directly (this is the gold standard!)
    if best_candidate is not None:
        # Extract the actual measured performance from candidate
        measured_volume_ml = best_candidate.get('measured_volume')
        measured_ul = measured_volume_ml * 1000 if measured_volume_ml is not None else None
        
        # Get the actual performance metrics calculated during ranking
        actual_deviation_pct = best_candidate.get('raw_accuracy')  # Already in %
        actual_variability = best_candidate.get('raw_precision')   # Already in %
        actual_time = best_candidate.get('raw_time')
        
        # Get tolerance check results from quality evaluation
        quality_eval = best_candidate.get('quality_evaluation', {})
        accuracy_tolerance_met = quality_eval.get('accuracy_ok')
        precision_tolerance_met = quality_eval.get('precision_ok')
        
        return {
            'volume_target': target_ul,
            'volume_measured': measured_ul,
            'average_deviation': actual_deviation_pct,
            'variability': actual_variability,
            'time': actual_time,
            'accuracy_tolerance_met': accuracy_tolerance_met,
            'precision_tolerance_met': precision_tolerance_met
        }
    
    # If no candidate provided, try to find the best trial from all_results for this volume
    if all_results:
        # Find trials for this volume that match the best_params
        volume_trials = [r for r in all_results if r.get('volume') == volume_ml]
        
        if volume_trials:
            # Find the trial that matches the best_params most closely
            best_trial = None
            for trial in volume_trials:
                # Check if this trial has the same parameters as best_params
                params_match = True
                for param in ALL_PARAMS:
                    if param in best_params and param in trial:
                        if abs(trial[param] - best_params[param]) > 1e-6:  # Small tolerance for float comparison
                            params_match = False
                            break
                
                if params_match:
                    best_trial = trial
                    break
            
            # If we found a matching trial, use its data
            if best_trial:
                measured_volume_ml = best_trial.get('measured_volume', 0)
                measured_ul = measured_volume_ml * 1000 if measured_volume_ml is not None else None
                
                return {
                    'volume_target': target_ul,
                    'volume_measured': measured_ul,
                    'average_deviation': best_trial.get('deviation'),
                    'variability': best_trial.get('variability'),
                    'time': best_trial.get('time'),
                    'accuracy_tolerance_met': None,  # Will need to be calculated elsewhere
                    'precision_tolerance_met': None   # Will need to be calculated elsewhere
                }
    
    # Final fallback: return empty metrics (don't try to reconstruct)
    return {
        'volume_target': target_ul,
        'volume_measured': None,
        'average_deviation': None,
        'variability': None,
        'time': None,
        'accuracy_tolerance_met': None,
        'precision_tolerance_met': None
    }
ALL_PARAMS = ["aspirate_speed", "dispense_speed", "aspirate_wait_time", "dispense_wait_time", 
              "retract_speed", "blowout_vol", "post_asp_air_vol", "overaspirate_vol"]

# Global state
EXPERIMENT_INDEX = 0
_CACHED_LASH_E = None
RETAIN_PIPET_BETWEEN_EXPERIMENTS = False

# Configure autosave
DEFAULT_LOCAL_AUTOSAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'calibration_runs'))
DEFAULT_HARDWARE_AUTOSAVE_DIR = 'C:\\Users\\Imaging Controller\\Desktop\\Calibration_SDL_Output\\New_Method'

BASE_AUTOSAVE_DIR = os.environ.get('CALIBRATION_AUTOSAVE_DIR')
if BASE_AUTOSAVE_DIR is None:
    if SIMULATE:
        BASE_AUTOSAVE_DIR = DEFAULT_LOCAL_AUTOSAVE_DIR
        os.makedirs(BASE_AUTOSAVE_DIR, exist_ok=True)
    else:
        try:
            if os.path.exists(os.path.dirname(DEFAULT_HARDWARE_AUTOSAVE_DIR)):
                test_dir = os.path.join(DEFAULT_HARDWARE_AUTOSAVE_DIR, 'test_write_access')
                os.makedirs(test_dir, exist_ok=True)
                os.rmdir(test_dir)
                BASE_AUTOSAVE_DIR = DEFAULT_HARDWARE_AUTOSAVE_DIR
            else:
                raise PermissionError("Hardware directory not accessible")
        except (PermissionError, OSError):
            BASE_AUTOSAVE_DIR = DEFAULT_LOCAL_AUTOSAVE_DIR
            os.makedirs(BASE_AUTOSAVE_DIR, exist_ok=True)
else:
    os.makedirs(BASE_AUTOSAVE_DIR, exist_ok=True)

print(f"[info] Using autosave directory: {BASE_AUTOSAVE_DIR}")

# --- HELPER FUNCTIONS ---

def reset_config_to_defaults():
    """Reset all configuration variables to defaults."""
    global LIQUID, SIMULATE, SEED, INITIAL_PARAMETER_SETS, PARAMETER_SETS_PER_RECOMMENDATION
    global PRECISION_MEASUREMENTS, VOLUMES, MAX_MEASUREMENTS, MAX_MEASUREMENTS_INITIAL_VOLUME, INPUT_VIAL_STATUS_FILE
    global MAX_MEASUREMENTS, MIN_GOOD_PARAMETER_SETS, USE_LLM_FOR_SCREENING, BAYESIAN_MODEL_TYPE
    global OVERASPIRATE_BASE_UL, OVERASPIRATE_SCALING_PERCENT, AUTO_CALIBRATE_OVERVOLUME
    global OVERVOLUME_CALIBRATION_BUFFER_UL, OVERVOLUME_MAX_BASE_UL, OVERVOLUME_MAX_PERCENT
    global ADAPTIVE_DEVIATION_THRESHOLD, ADAPTIVE_PENALTY_VARIABILITY
    global ACCURACY_WEIGHT, PRECISION_WEIGHT, TIME_WEIGHT, SIM_DEV_MULTIPLIER, SIM_VAR_MULTIPLIER
    global BAYESIAN_MODEL_TYPE_SUBSEQUENT, OPTIMIZER_DEVIATION_THRESHOLD, OPTIMIZER_VARIABILITY_THRESHOLD, OPTIMIZER_TIME_THRESHOLD
    global USE_TRANSFER_LEARNING, global_ax_client
    global EXTERNAL_DATA_PATH, USE_EXTERNAL_DATA, EXTERNAL_DATA_VOLUME_FILTER, EXTERNAL_DATA_LIQUID_FILTER, FIXED_PARAMETERS
    
    print("🔄 Resetting configuration to default values...")
    
    LIQUID = DEFAULT_LIQUID
    SIMULATE = DEFAULT_SIMULATE
    SEED = DEFAULT_SEED
    INITIAL_PARAMETER_SETS = DEFAULT_INITIAL_PARAMETER_SETS
    PARAMETER_SETS_PER_RECOMMENDATION = DEFAULT_PARAMETER_SETS_PER_RECOMMENDATION
    # REPLICATES = DEFAULT_REPLICATES  # DEPRECATED: Replaced by PRECISION_MEASUREMENTS system
    PRECISION_MEASUREMENTS = DEFAULT_PRECISION_MEASUREMENTS
    VOLUMES = DEFAULT_VOLUMES.copy()
    MAX_MEASUREMENTS = DEFAULT_MAX_MEASUREMENTS
    MAX_MEASUREMENTS_INITIAL_VOLUME = DEFAULT_MAX_MEASUREMENTS_INITIAL_VOLUME
    INPUT_VIAL_STATUS_FILE = DEFAULT_INPUT_VIAL_STATUS_FILE
    MAX_MEASUREMENTS = DEFAULT_MAX_MEASUREMENTS
    MIN_GOOD_PARAMETER_SETS = DEFAULT_MIN_GOOD_PARAMETER_SETS
    USE_LLM_FOR_SCREENING = DEFAULT_USE_LLM_FOR_SCREENING
    BAYESIAN_MODEL_TYPE = DEFAULT_BAYESIAN_MODEL_TYPE
    BAYESIAN_MODEL_TYPE_SUBSEQUENT = DEFAULT_BAYESIAN_MODEL_TYPE_SUBSEQUENT
    OVERASPIRATE_BASE_UL = DEFAULT_OVERASPIRATE_BASE_UL
    OVERASPIRATE_SCALING_PERCENT = DEFAULT_OVERASPIRATE_SCALING_PERCENT
    AUTO_CALIBRATE_OVERVOLUME = DEFAULT_AUTO_CALIBRATE_OVERVOLUME
    OVERVOLUME_CALIBRATION_BUFFER_UL = DEFAULT_OVERVOLUME_CALIBRATION_BUFFER_UL
    OVERVOLUME_MAX_BASE_UL = DEFAULT_OVERVOLUME_MAX_BASE_UL
    OVERVOLUME_MAX_PERCENT = DEFAULT_OVERVOLUME_MAX_PERCENT
    ADAPTIVE_DEVIATION_THRESHOLD = DEFAULT_ADAPTIVE_DEVIATION_THRESHOLD
    ADAPTIVE_PENALTY_VARIABILITY = DEFAULT_ADAPTIVE_PENALTY_VARIABILITY
    ACCURACY_WEIGHT = DEFAULT_ACCURACY_WEIGHT
    PRECISION_WEIGHT = DEFAULT_PRECISION_WEIGHT
    TIME_WEIGHT = DEFAULT_TIME_WEIGHT
    SIM_DEV_MULTIPLIER = DEFAULT_SIM_DEV_MULTIPLIER
    SIM_VAR_MULTIPLIER = DEFAULT_SIM_VAR_MULTIPLIER
    BAYESIAN_MODEL_TYPE_SUBSEQUENT = DEFAULT_BAYESIAN_MODEL_TYPE_SUBSEQUENT
    OPTIMIZER_DEVIATION_THRESHOLD = DEFAULT_OPTIMIZER_DEVIATION_THRESHOLD
    OPTIMIZER_VARIABILITY_THRESHOLD = DEFAULT_OPTIMIZER_VARIABILITY_THRESHOLD
    OPTIMIZER_TIME_THRESHOLD = DEFAULT_OPTIMIZER_TIME_THRESHOLD
    USE_TRANSFER_LEARNING = DEFAULT_USE_TRANSFER_LEARNING
    EXTERNAL_DATA_PATH = DEFAULT_EXTERNAL_DATA_PATH
    USE_EXTERNAL_DATA = DEFAULT_USE_EXTERNAL_DATA
    EXTERNAL_DATA_VOLUME_FILTER = DEFAULT_EXTERNAL_DATA_VOLUME_FILTER
    EXTERNAL_DATA_LIQUID_FILTER = DEFAULT_EXTERNAL_DATA_LIQUID_FILTER
    FIXED_PARAMETERS = DEFAULT_FIXED_PARAMETERS.copy()
    global_ax_client = None
    
    print("✅ Configuration reset complete")

def get_current_config_summary():
    """Print current configuration summary."""
    print("📋 CURRENT EXPERIMENT CONFIG:")
    print(f"   Liquid: {LIQUID}")
    print(f"   Simulate: {SIMULATE}")
    print(f"   Volumes: {[f'{v*1000:.0f}uL' for v in VOLUMES]}")
    print(f"   Max measurements for initial volume: {MAX_MEASUREMENTS_INITIAL_VOLUME}")
    print(f"   Min good parameter sets per volume: {MIN_GOOD_PARAMETER_SETS}")
    print(f"   Global measurement limit: {MAX_MEASUREMENTS} (entire calibration)")
    print(f"   Precision measurements: {PRECISION_MEASUREMENTS}")
    print(f"   Initial parameter sets: {INITIAL_PARAMETER_SETS}")
    print(f"   LLM screening: {USE_LLM_FOR_SCREENING}")
    print(f"   Bayesian model (1st vol): {BAYESIAN_MODEL_TYPE}")
    print(f"   Bayesian model (2nd+ vol): {BAYESIAN_MODEL_TYPE_SUBSEQUENT}")
    print(f"   Transfer learning: {'✅ ENABLED' if USE_TRANSFER_LEARNING else '❌ DISABLED'}")
    print(f"   External data: {'✅ ENABLED' if USE_EXTERNAL_DATA else '❌ DISABLED'}")
    if USE_EXTERNAL_DATA and EXTERNAL_DATA_PATH:
        print(f"     Path: {EXTERNAL_DATA_PATH}")
        if EXTERNAL_DATA_VOLUME_FILTER:
            print(f"     Volume filter: {EXTERNAL_DATA_VOLUME_FILTER*1000:.0f}μL")
        if EXTERNAL_DATA_LIQUID_FILTER:
            print(f"     Liquid filter: {EXTERNAL_DATA_LIQUID_FILTER}")
    print(f"   Adaptive threshold: {ADAPTIVE_DEVIATION_THRESHOLD}% deviation")
    print(f"   Ranking weights: Acc={ACCURACY_WEIGHT}, Prec={PRECISION_WEIGHT}, Time={TIME_WEIGHT}")
    if FIXED_PARAMETERS:
        print(f"   Fixed parameters: {FIXED_PARAMETERS}")
    if SIMULATE:
        print(f"   Simulation multipliers: Dev={SIM_DEV_MULTIPLIER}x, Var={SIM_VAR_MULTIPLIER}x")

def get_volume_dependent_tolerances(volume_ml):
    """Calculate volume-dependent tolerances."""
    volume_ul = volume_ml * 1000
    
    # Find appropriate tolerance range
    tolerance_pct = None
    range_name = 'unknown'
    
    for vol_range in VOLUME_TOLERANCE_RANGES:
        if vol_range['min_ul'] <= volume_ul < vol_range['max_ul']:
            tolerance_pct = vol_range['tolerance_pct']
            range_name = vol_range['name']
            break
    
    if tolerance_pct is None:
        tolerance_pct = 10.0
        range_name = 'fallback'
    
    base_tolerance_ul = volume_ul * (tolerance_pct / 100.0)
    
    # Apply simulation tolerance multipliers if needed
    if SIMULATE:
        # Use environment variables first, then global hyperparameters as fallback
        try:
            dev_multiplier = float(os.environ.get('SIM_DEV_MULTIPLIER', str(SIM_DEV_MULTIPLIER)))
            var_multiplier = float(os.environ.get('SIM_VAR_MULTIPLIER', str(SIM_VAR_MULTIPLIER)))
        except ValueError:
            dev_multiplier, var_multiplier = SIM_DEV_MULTIPLIER, SIM_VAR_MULTIPLIER
        
        deviation_ul = base_tolerance_ul * dev_multiplier
        variation_ul = base_tolerance_ul * var_multiplier
    else:
        deviation_ul = base_tolerance_ul
        variation_ul = base_tolerance_ul
    
    return {
        'deviation_ul': deviation_ul,
        'variation_ul': variation_ul,
        'tolerance_percent': tolerance_pct,
        'range_name': range_name
    }

def get_max_overaspirate_ul(volume_ml):
    """Calculate maximum overaspirate volume."""
    volume_ul = volume_ml * 1000
    scaling_volume = volume_ul * (OVERASPIRATE_SCALING_PERCENT / 100.0)
    max_overaspirate = OVERASPIRATE_BASE_UL + scaling_volume
    
    min_overaspirate = 1.0
    if max_overaspirate < min_overaspirate:
        max_overaspirate = min_overaspirate
    
    return max_overaspirate

def debug_ax_constraints(ax_client, label="", autosave_raw_path=None):
    """Debug function to check what constraints Ax actually has."""
    debug_lines = []
    
    try:
        experiment = ax_client.experiment
        search_space = experiment.search_space
        
        print(f"🔍 AX CONSTRAINTS CHECK {label}:")
        debug_lines.append(f"AX CONSTRAINTS CHECK {label}")
        
        # Debug: show all parameters
        print(f"   All parameters: {list(search_space.parameters.keys())}")
        debug_lines.append(f"All parameters: {list(search_space.parameters.keys())}")
        
        for param_name, param in search_space.parameters.items():
            if param_name == "overaspirate_vol":
                print(f"   Found parameter '{param_name}': type = {type(param)}")
                debug_lines.append(f"Found parameter '{param_name}': type = {type(param)}")
                print(f"   Parameter attributes: {dir(param)}")
                
                # For RangeParameter, use lower and upper attributes
                if hasattr(param, 'lower') and hasattr(param, 'upper'):
                    lower_ml = param.lower
                    upper_ml = param.upper
                    lower_ul = lower_ml * 1000  # Convert mL to μL
                    upper_ul = upper_ml * 1000  # Convert mL to μL
                    print(f"   Parameter '{param_name}': range = [{lower_ml:.6f}, {upper_ml:.6f}] mL")
                    print(f"   → In μL: [{lower_ul:.6f}, {upper_ul:.6f}]μL")
                    print(f"   ✅ CONSTRAINT ACTIVE: Ax will limit overaspirate_vol to {upper_ul:.1f}μL")
                    
                    debug_lines.append(f"Parameter '{param_name}': range = [{lower_ml:.6f}, {upper_ml:.6f}] mL")
                    debug_lines.append(f"In μL: [{lower_ul:.6f}, {upper_ul:.6f}]μL")
                    debug_lines.append(f"✅ CONSTRAINT ACTIVE: Ax will limit overaspirate_vol to {upper_ul:.1f}μL")
                else:
                    print(f"   Parameter '{param_name}': No lower/upper bounds found")
                    debug_lines.append(f"Parameter '{param_name}': No lower/upper bounds found")
                break
        else:
            print(f"   ⚠️  'overaspirate_vol' parameter not found!")
            debug_lines.append(f"⚠️  'overaspirate_vol' parameter not found!")
        
        # Check outcome constraints too
        outcome_constraints = experiment.optimization_config.outcome_constraints if experiment.optimization_config else []
        if outcome_constraints:
            print(f"   Outcome constraints: {len(outcome_constraints)} found")
            debug_lines.append(f"Outcome constraints: {len(outcome_constraints)} found")
        else:
            print(f"   No outcome constraints found")
            debug_lines.append(f"No outcome constraints found")
            
    except Exception as e:
        print(f"   ❌ Error checking Ax constraints: {e}")
        debug_lines.append(f"❌ Error checking Ax constraints: {e}")
    
    print()
    
    # Log to constraint log file if path provided
    if autosave_raw_path and debug_lines:
        try:
            log_file = os.path.join(os.path.dirname(autosave_raw_path), "constraint_log.txt")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n[{timestamp}] AX CONSTRAINT VERIFICATION {label}\n")
                for line in debug_lines:
                    f.write(f"{line}\n")
        except Exception as e:
            print(f"   ⚠️  Could not write to constraint log: {e}")

def calculate_measurements_per_volume(global_measurement_count, volumes_remaining):
    """
    Calculate deterministic measurement allocation for remaining volumes.
    
    Args:
        global_measurement_count: Total measurements used so far
        volumes_remaining: Number of volumes left to process
        
    Returns:
        int: Measurements allocated per remaining volume
    """
    if volumes_remaining <= 0:
        return 0
        
    measurements_remaining = MAX_MEASUREMENTS - global_measurement_count
    measurements_per_volume = measurements_remaining // volumes_remaining
    
    # Ensure we don't allocate negative measurements
    return max(0, measurements_per_volume)

def initialize_experiment():
    """Initialize experiment with cached lash_e."""
    global _CACHED_LASH_E
    DENSITY_LIQUID = LIQUIDS[LIQUID]["density"]
    NEW_PIPET_EACH_TIME_SET = LIQUIDS[LIQUID]["refill_pipets"]
    state = {
        "measurement_vial_index": 0, 
        "measurement_vial_name": "measurement_vial_0",
        "total_measurements": 0
    }

    if _CACHED_LASH_E is None:
        print("Creating new Lash_E controller...")
        _CACHED_LASH_E = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False)
        try:
            _CACHED_LASH_E.nr_robot.check_input_file()
        except Exception as e:
            _CACHED_LASH_E.logger.warning(f"Initial check_input_file failed: {e}")
    else:
        print("Reusing existing Lash_E controller...")
        _CACHED_LASH_E.logger.info(f"Reusing Lash_E controller for experiment index {EXPERIMENT_INDEX}")

    try:
        _CACHED_LASH_E.nr_robot.move_vial_to_location("measurement_vial_0", "clamp", 0)
    except Exception as e:
        _CACHED_LASH_E.logger.warning(f"Could not move measurement vial: {e}")

    return _CACHED_LASH_E, DENSITY_LIQUID, NEW_PIPET_EACH_TIME_SET, state

def get_tip_volume_for_volume(lash_e, volume):
    """Get tip volume capacity for given volume."""
    try:
        tip_type = lash_e.nr_robot.select_pipet_tip(volume)
        tip_config = lash_e.nr_robot.get_config_parameter('pipet_tips', tip_type, None, error_on_missing=False)
        if tip_config:
            return tip_config.get('volume', 1.0)
        else:
            return 1.0 if volume > 0.25 else 0.25
    except Exception as e:
        print(f"Warning: Could not determine tip volume for {volume} mL: {e}")
        return 1.0 if volume > 0.25 else 0.25

def create_transfer_learning_optimizer():
    """Create a global optimizer for transfer learning across all volumes."""
    global global_ax_client
    
    if not USE_TRANSFER_LEARNING:
        return None
        
    print(f"\n🌐 CREATING TRANSFER LEARNING OPTIMIZER")
    print(f"   Volumes: {[f'{v*1000:.0f}μL' for v in VOLUMES]}")
    print(f"   Benefits: Cross-volume parameter learning")
    
    # Calculate volume bounds from all volumes we'll test
    min_volume = min(VOLUMES)
    max_volume = max(VOLUMES)
    volume_bounds = [min_volume, max_volume]
    
    # Use maximum overaspirate constraint across all volumes
    max_overaspirate_across_volumes = max([get_max_overaspirate_ul(v) for v in VOLUMES])
    
    # Create transfer learning optimizer
    global_ax_client = optimizer_3obj.create_model(
        seed=SEED,
        num_initial_recs=0,  # We'll load data as we go
        bayesian_batch_size=PARAMETER_SETS_PER_RECOMMENDATION,
        volume=None,  # No fixed volume
        tip_volume=1.0,  # Use largest tip
        model_type=BAYESIAN_MODEL_TYPE,
        optimize_params=ALL_PARAMS,  # Will include volume parameter internally
        fixed_params={},
        simulate=SIMULATE,
        max_overaspirate_ul=max_overaspirate_across_volumes,
        transfer_learning=True,
        volume_bounds=volume_bounds
    )
    
    print(f"   ✅ Global optimizer created with volume range {min_volume*1000:.0f}-{max_volume*1000:.0f}μL")
    return global_ax_client

# --- ADAPTIVE VARIABILITY MEASUREMENT ---

def run_adaptive_measurement(lash_e, liquid_source, measurement_vial, volume, params, 
                            expected_mass, expected_time, simulate, autosave_raw_path, 
                            raw_measurements, liquid, new_pipet_each_time_set, trial_type,
                            deviation_threshold=None):
    """
    Run adaptive measurement with conditional replicates based on initial deviation.
    
    Args:
        deviation_threshold: Threshold for running additional replicates (uses global ADAPTIVE_DEVIATION_THRESHOLD if None)
        
    Returns:
        dict: {
            'deviation': average deviation,
            'variability': calculated variability or penalty,
            'time': average time,
            'replicate_count': number of replicates run,
            'all_measurements': list of individual measurement results
        }
    """
    
    # Use global hyperparameter if not specified
    if deviation_threshold is None:
        deviation_threshold = ADAPTIVE_DEVIATION_THRESHOLD
    
    print(f"   🔬 Adaptive measurement (threshold: {deviation_threshold}%)...")
    
    all_measurements = []
    all_deviations = []
    all_times = []
    
    # Step 1: Run initial single measurement
    print(f"      Initial measurement...", end=" ")
    
    result = pipet_and_measure_tracked(lash_e, liquid_source, measurement_vial, 
                                      volume, params, expected_mass, expected_time, 
                                      1, simulate, autosave_raw_path, raw_measurements, 
                                      liquid, new_pipet_each_time_set, trial_type)
    
    # Check if budget was exceeded
    if result is None:
        print("🛑 Budget exhausted during initial measurement - returning penalty result")
        return {
            'deviation': 100.0,  # Penalty deviation
            'variability': ADAPTIVE_PENALTY_VARIABILITY,
            'time': expected_time,
            'replicate_count': 0,
            'all_measurements': [],
            'all_deviations': [],
            'all_times': []
        }
    
    initial_deviation = result['deviation']  # pipet_and_measure always returns deviation
    initial_time = result['time']  # pipet_and_measure always returns time
    initial_measured_volume = result.get('measured_volume', 0)  # pipet_and_measure now returns measured_volume
    
    all_deviations.append(initial_deviation)
    all_times.append(initial_time)
    all_measurements.append(initial_measured_volume)
    
    print(f"{initial_deviation:.1f}% deviation | Target: {volume*1000:.1f}μL → Measured: {initial_measured_volume*1000:.1f}μL")
    
    # Step 2: Decision based on initial deviation
    if initial_deviation > deviation_threshold:
        # Poor accuracy - don't waste additional replicates
        print(f"      🚫 High deviation (>{deviation_threshold}%) - using penalty variability")
        variability = ADAPTIVE_PENALTY_VARIABILITY  # Penalty value
        replicate_count = 1
        
        avg_deviation = initial_deviation
        avg_time = initial_time
        
    else:
        # Good accuracy - run additional replicates based on PRECISION_MEASUREMENTS setting
        additional_replicates = PRECISION_MEASUREMENTS - 1  # Already ran 1 initial
        total_replicates = PRECISION_MEASUREMENTS
        print(f"      ✅ Good deviation (≤{deviation_threshold}%) - running {additional_replicates} additional replicates...")
        
        for i in range(additional_replicates):  # Run additional replicates 
            replicate_num = i + 2  # Replicate 2, 3, etc.
            print(f"         Replicate {replicate_num}/{total_replicates}...", end=" ")
            
            # Need to get fresh liquid source for each replicate
            check_if_measurement_vial_full(lash_e, {"measurement_vial_name": measurement_vial, "measurement_vial_index": 0})
            
            result = pipet_and_measure_tracked(lash_e, liquid_source, measurement_vial, 
                                              volume, params, expected_mass, expected_time, 
                                              1, simulate, autosave_raw_path, raw_measurements, 
                                              liquid, new_pipet_each_time_set, trial_type)
            
            # Check if budget was exceeded during replicates
            if result is None:
                print("🛑 Budget exhausted during replicate - stopping early")
                break
            
            deviation = result['deviation']  # pipet_and_measure always returns deviation
            time_taken = result['time']  # pipet_and_measure always returns time
            measured_volume = result.get('measured_volume', 0)  # pipet_and_measure now returns measured_volume
            
            all_deviations.append(deviation)
            all_times.append(time_taken)
            all_measurements.append(measured_volume)
            
            print(f"Target: {volume*1000:.1f}μL → Measured: {measured_volume*1000:.1f}μL ({deviation:+.1f}% dev)")
        
        replicate_count = total_replicates
        
        # Calculate averages from all replicates
        avg_deviation = np.mean(all_deviations)
        avg_time = np.mean(all_times)
        
        # Calculate variability from volume measurements
        if len(all_measurements) >= 2:
            max_vol = max(all_measurements)
            min_vol = min(all_measurements)
            avg_vol = np.mean(all_measurements)
            
            # Use range-based variability: (max_vol - min_vol) / (2 * avg_vol) * 100
            # Safeguard against negative measurements causing impossible negative variability
            if avg_vol > 0 and max_vol >= min_vol:
                variability = (max_vol - min_vol) / (2 * avg_vol) * 100
                variability = max(0.0, variability)  # Ensure non-negative
            else:
                print(f"⚠️  WARNING: Invalid measurement data for variability - avg_vol: {avg_vol}, min: {min_vol}, max: {max_vol}")
                print(f"    Setting variability to penalty value ({ADAPTIVE_PENALTY_VARIABILITY}%) due to invalid measurements")
                variability = ADAPTIVE_PENALTY_VARIABILITY
            
            print(f"      📊 Final averages: Target: {volume:.1f}μL → Avg Measured: {avg_vol:.1f}μL ({avg_deviation:+.1f}% dev, {variability:.1f}% var, {avg_time:.1f}s)")
        else:
            variability = 0.0  # Single measurement somehow
    
    # Calculate average measured volume
    avg_measured_volume = np.mean(all_measurements) if all_measurements else 0
    
    return {
        'deviation': avg_deviation,
        'variability': variability,
        'time': avg_time,
        'replicate_count': replicate_count,
        'all_measurements': all_measurements,
        'all_deviations': all_deviations,
        'all_times': all_times,
        'measured_volume': avg_measured_volume
    }

# --- GOOD TRIAL EVALUATION ---

def evaluate_trial_quality(trial_results, volume_ml, tolerances):
    """
    Evaluate if a trial is "GOOD" based on accuracy and precision criteria.
    
    A trial is GOOD if:
    1. Accuracy (deviation) is within tolerance
    2. Precision (max_vol - min_vol)/(2 * target_vol) ≤ tolerance_percent
    
    Args:
        trial_results: Dict with keys 'deviation', 'variability', 'time', and raw measurements
        volume_ml: Target volume in mL
        tolerances: Dict with 'deviation_ul', 'variation_ul', 'tolerance_percent'
    
    Returns:
        dict: {'is_good': bool, 'accuracy_ok': bool, 'precision_ok': bool, 'precision_value': float}
    """
    
    # Check accuracy (deviation)
    if 'deviation' not in trial_results:
        raise ValueError(f"Trial results missing 'deviation' field: {trial_results.keys()}")
    deviation_pct = trial_results['deviation']
    volume_ul = volume_ml * 1000
    deviation_ul = (deviation_pct / 100.0) * volume_ul
    accuracy_ok = deviation_ul <= tolerances['deviation_ul']
    
    # Check precision - need raw measurements for this
    precision_ok = False
    precision_value = ADAPTIVE_PENALTY_VARIABILITY  # Default penalty value
    
    if 'raw_measurements' in trial_results and len(trial_results['raw_measurements']) > 1:
        measurements = [m * 1000 for m in trial_results['raw_measurements']]  # Convert to uL
        max_vol = max(measurements)
        min_vol = min(measurements)
        target_vol = volume_ml * 1000  # Target in uL
        
        # Calculate precision: (max_vol - min_vol) / (2 * target_vol) 
        precision_value = (max_vol - min_vol) / (2 * target_vol) * 100  # As percentage
        precision_ok = precision_value <= tolerances['tolerance_percent']
    elif 'variability' in trial_results and trial_results['variability'] is not None:
        # Fallback: use variability if available (coefficient of variation)
        precision_value = trial_results['variability']
        precision_ok = precision_value <= tolerances['tolerance_percent']
    else:
        # No precision data available - treat as poor precision
        precision_value = ADAPTIVE_PENALTY_VARIABILITY
        precision_ok = False
    
    is_good = accuracy_ok and precision_ok
    
    return {
        'is_good': is_good,
        'accuracy_ok': accuracy_ok,
        'precision_ok': precision_ok,
        'precision_value': precision_value,
        'accuracy_deviation_ul': deviation_ul,
        'accuracy_tolerance_ul': tolerances['deviation_ul']
    }

# --- RANKING SYSTEM (MODULAR) ---

def rank_candidates_by_priority(candidates, volume_ml, tolerances):
    """
    Rank candidates using data-driven weighted scoring approach.
    
    Uses normalization based on actual candidate pool ranges and weighted composite scoring:
    - Accuracy: 50% weight
    - Precision: 40% weight  
    - Time: 10% weight
    
    Args:
        candidates: List of trial result dicts
        volume_ml: Target volume in mL
        tolerances: Dict with tolerance thresholds
        
    Returns:
        list: Ranked candidates (best first)
    """
    
    if not candidates:
        return []
    
    # Filter candidates to only include those with replicate measurements
    # Exclude single measurements with penalty variability (incomplete experiments)
    replicated_candidates = []
    for candidate in candidates:
        replicate_count = candidate.get('replicate_count', 1)
        variability = candidate.get('variability', ADAPTIVE_PENALTY_VARIABILITY)
        
        # Only include candidates with multiple replicates OR non-penalty variability
        if replicate_count > 1 or variability != ADAPTIVE_PENALTY_VARIABILITY:
            replicated_candidates.append(candidate)
    
    if not replicated_candidates:
        print("   ⚠️  No replicated candidates found - using all candidates as fallback")
        replicated_candidates = candidates
    else:
        excluded_count = len(candidates) - len(replicated_candidates)
        if excluded_count > 0:
            print(f"   🔍 Excluding {excluded_count} single-measurement candidates from ranking")
    
    # Evaluate each replicated candidate and collect raw metrics
    evaluated_candidates = []
    raw_accuracies = []
    raw_precisions = []
    raw_times = []
    
    for candidate in replicated_candidates:
        quality = evaluate_trial_quality(candidate, volume_ml, tolerances)
        
        # Collect raw metrics for normalization - fail fast if data is missing
        if 'deviation' not in candidate:
            raise ValueError(f"Candidate missing 'deviation' field: {candidate.keys()}")
        if 'time' not in candidate:
            raise ValueError(f"Candidate missing 'time' field: {candidate.keys()}")
            
        accuracy = candidate['deviation']  # % deviation
        precision = quality['precision_value']  # % precision
        time = candidate['time']  # seconds
        
        raw_accuracies.append(accuracy)
        raw_precisions.append(precision)
        raw_times.append(time)
        
        candidate_with_metrics = candidate.copy()
        candidate_with_metrics.update({
            'quality_evaluation': quality,
            'raw_accuracy': accuracy,
            'raw_precision': precision,
            'raw_time': time
        })
        evaluated_candidates.append(candidate_with_metrics)
    
    # Calculate normalization using standard deviation (compresses small differences, preserves large ones)
    import statistics
    
    acc_mean = statistics.mean(raw_accuracies)
    prec_mean = statistics.mean(raw_precisions) 
    time_mean = statistics.mean(raw_times)
    
    # Use standard deviation for normalization (with minimum threshold to prevent division by zero)
    acc_std = max(statistics.stdev(raw_accuracies) if len(raw_accuracies) > 1 else 0.1, 0.1)
    prec_std = max(statistics.stdev(raw_precisions) if len(raw_precisions) > 1 else 0.1, 0.1)
    time_std = max(statistics.stdev(raw_times) if len(raw_times) > 1 else 1.0, 1.0)
    
    # Calculate normalized scores using direct performance scoring (lower raw value = lower score)
    for candidate in evaluated_candidates:
        # Compare to zero instead of mean - lower raw values get better (lower) scores
        acc_score = candidate['raw_accuracy'] / acc_std * 100
        prec_score = candidate['raw_precision'] / prec_std * 100  
        time_score = candidate['raw_time'] / time_std * 100
        
        # Weighted composite score (lower is better)
        composite_score = ACCURACY_WEIGHT * acc_score + PRECISION_WEIGHT * prec_score + TIME_WEIGHT * time_score
        
        candidate.update({
            'accuracy_score': acc_score,
            'precision_score': prec_score,
            'time_score': time_score,
            'composite_score': composite_score
        })
    
    # Sort by composite score (lower is better)
    ranked_candidates = sorted(evaluated_candidates, key=lambda x: x['composite_score'])
    
    # Log ranking results with composite scores
    print(f"🏆 CANDIDATE RANKING (best first):")
    print(f"   Data stats: Acc μ={acc_mean:.1f}%±{acc_std:.1f}, Prec μ={prec_mean:.1f}%±{prec_std:.1f}, Time μ={time_mean:.1f}s±{time_std:.1f}")
    for i, candidate in enumerate(ranked_candidates[:5]):  # Show top 5
        quality = candidate['quality_evaluation']
        print(f"   #{i+1}: Score={candidate['composite_score']:.1f} "
              f"(Acc={candidate['accuracy_score']:.1f}, Prec={candidate['precision_score']:.1f}, Time={candidate['time_score']:.1f}) "
              f"→ Dev={candidate['raw_accuracy']:.1f}%, Prec={candidate['raw_precision']:.1f}%, Time={candidate['raw_time']:.1f}s, "
              f"Good={quality['is_good']}")
    
    return ranked_candidates

# --- SIMPLIFIED STOPPING CRITERIA ---

def check_stopping_criteria(all_results, volume_ml, tolerances):
    """
    Check if we should stop optimization based on simplified criteria:
    1. Reached MAX_MEASUREMENTS_INITIAL_VOLUME total individual measurements for first volume, OR
    2. Have MIN_GOOD_TRIALS "good" parameter sets (6)
    
    Note: MAX_MEASUREMENTS_INITIAL_VOLUME counts individual measurements for first volume, MIN_GOOD_TRIALS counts parameter sets
    
    Returns:
        dict: {'should_stop': bool, 'reason': str, 'good_trials': int, 'total_trials': int}
    """
    
    # Filter ALL first volume results (screening + optimization, exclude precision tests)
    first_volume_results = [r for r in all_results 
                           if r.get('volume') == volume_ml 
                           and r.get('strategy') not in ['PRECISION_TEST', 'PRECISION']]
    
    # Count individual measurements and good parameter sets separately
    total_measurements = 0
    good_parameter_sets = 0
    total_parameter_sets = len(first_volume_results)
    
    for result in first_volume_results:
        # Count individual measurements (replicates)
        replicate_count = result.get('replicate_count', 1)
        total_measurements += replicate_count
        
        # Check if this parameter set produced good results
        quality = evaluate_trial_quality(result, volume_ml, tolerances)
        if quality['is_good']:
            good_parameter_sets += 1
    
    # Check stopping criteria
    if total_measurements >= MAX_MEASUREMENTS_INITIAL_VOLUME:
        return {
            'should_stop': True,
            'reason': f'Reached maximum individual measurements for initial volume ({MAX_MEASUREMENTS_INITIAL_VOLUME})',
            'good_trials': good_parameter_sets,
            'total_trials': total_parameter_sets
        }
    elif good_parameter_sets >= MIN_GOOD_PARAMETER_SETS:
        return {
            'should_stop': True,
            'reason': f'Reached minimum good parameter sets ({MIN_GOOD_PARAMETER_SETS})',
            'good_trials': good_parameter_sets,
            'total_trials': total_parameter_sets
        }
    else:
        return {
            'should_stop': False,
            'reason': f'Need {MIN_GOOD_PARAMETER_SETS - good_parameter_sets} more good parameter sets or {MAX_MEASUREMENTS_INITIAL_VOLUME - total_measurements} more measurements',
            'good_trials': good_parameter_sets,
            'total_trials': total_parameter_sets
        }

# --- LLM SUGGESTIONS (FROM MODULAR) ---

def get_llm_suggestions(ax_client, n, all_results, volume=None, liquid=None):
    """Get LLM-based parameter suggestions."""
    # Use different configs for initial exploration vs optimization
    if not all_results:
        # Initial exploration: no existing data to analyze
        config_path = os.path.abspath("recommenders/calibration_initial_config.json")
        print("   🎯 Using initial exploration config - systematic space coverage")
    else:
        # Optimization: analyze existing data for improvements
        config_path = os.path.abspath("recommenders/calibration_unified_config.json")
        print("   📊 Using optimization config - data-driven improvements")
    
    # Load config and sync parameter ranges with actual Ax search space
    optimizer = llm_opt.LLMOptimizer(backend="ollama", ollama_model="online_server")
    config = optimizer.load_config(config_path)
    
    # Update config with current experimental context
    if volume is not None:
        config['experimental_setup']['target_volume_ul'] = volume * 1000  # Convert mL to μL
        print(f"   📏 Target volume: {volume*1000:.0f}μL")
    
    if liquid is not None:
        config['experimental_setup']['current_liquid'] = liquid
        print(f"   🧪 Setting current_liquid to: {liquid}")
        
        # Debug material properties lookup
        material_props = config.get('material_properties', {})
        print(f"   🔍 Available materials in config: {list(material_props.keys())}")
        print(f"   🔍 Looking for liquid: '{liquid.lower()}'")
        
        # For initial exploration, add liquid context to system message if available in config
        if not all_results and liquid.lower() in material_props:
            liquid_info = material_props[liquid.lower()]
            liquid_context = f"LIQUID CONTEXT: You are optimizing for {liquid.upper()}. {liquid_info.get('description', '')} Focus: {liquid_info.get('focus', '')}"
            
            # Insert liquid-specific guidance at the beginning of system message
            original_message = config['system_message']
            enhanced_message = [liquid_context, ""] + original_message
            config['system_message'] = enhanced_message
            print(f"   🧪 Enhanced config for {liquid}")
            print(f"   📝 Added liquid context: {liquid_context}")
        elif liquid.lower() not in material_props:
            print(f"   ⚠️  No material properties found for '{liquid}' in config")
        else:
            print(f"   ℹ️  Skipping liquid context (not initial exploration)")
    else:
        print(f"   ⚠️  No liquid specified")
    
    # Update config parameters with actual search space bounds from Ax client
    if hasattr(ax_client, 'experiment') and hasattr(ax_client.experiment, 'search_space'):
        print("   🔧 Syncing LLM config with Ax search space bounds...")
        for param_name, param in ax_client.experiment.search_space.parameters.items():
            if param_name in config['parameters']:
                if hasattr(param, 'lower') and hasattr(param, 'upper'):
                    old_range = config['parameters'][param_name]['range']
                    new_range = [param.lower, param.upper]
                    config['parameters'][param_name]['range'] = new_range
                    if old_range != new_range:
                        print(f"     📝 {param_name}: {old_range} → {new_range}")
    
    # Prepare input data
    llm_input_file = os.path.join("output", "temp_llm_input.csv")
    os.makedirs("output", exist_ok=True)
    
    if not all_results:
        # Create empty DataFrame with expected columns for initial exploration
        # Include the current liquid type so LLM knows what material to optimize for
        empty_df = pd.DataFrame(columns=['liquid', 'aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 
                                        'dispense_wait_time', 'retract_speed', 'blowout_vol', 
                                        'post_asp_air_vol', 'overaspirate_vol', 'deviation', 'time', 'trial_type'])
        
        # Add a single row with the current liquid type for material properties detection
        if liquid:
            empty_df.loc[0] = [liquid] + [None] * (len(empty_df.columns) - 1)
            print(f"   🧪 Added liquid type '{liquid}' to LLM input for material properties detection")
        
        empty_df.to_csv(llm_input_file, index=False)
    else:
        pd.DataFrame(all_results).to_csv(llm_input_file, index=False)
    
    config["batch_size"] = n
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_output_file = os.path.join("output", f"llm_recommendations_{timestamp}.csv")
    result = optimizer.optimize(llm_input_file, config, llm_output_file)
    all_llm_recs = result.get('recommendations', [])
    suggestions = []
    for i, llm_params in enumerate(all_llm_recs[:n]):
        if llm_params:
            expected_params = set(ax_client.experiment.search_space.parameters.keys())
            filtered_params = {}
            
            # Apply proper type casting and bounds enforcement based on Ax search space parameter types
            for k, v in llm_params.items():
                if k in expected_params:
                    ax_param = ax_client.experiment.search_space.parameters[k]
                    if hasattr(ax_param, 'parameter_type'):
                        # Cast to appropriate type based on Ax parameter type
                        if ax_param.parameter_type.name == 'INT':
                            casted_value = int(round(float(v)))
                        elif ax_param.parameter_type.name == 'FLOAT':
                            casted_value = float(v)
                        else:
                            casted_value = v
                        
                        # Enforce bounds for RangeParameter types
                        if hasattr(ax_param, 'lower') and hasattr(ax_param, 'upper'):
                            original_value = casted_value
                            casted_value = max(ax_param.lower, min(ax_param.upper, casted_value))
                            if casted_value != original_value:
                                print(f"     🔧 Clamped {k}: {original_value} → {casted_value} (bounds: [{ax_param.lower}, {ax_param.upper}])")
                        
                        filtered_params[k] = casted_value
                    else:
                        filtered_params[k] = v
            
            print(f"   LLM suggestion {i+1}: {filtered_params}")
            params, trial_index = ax_client.attach_trial(filtered_params)
            suggestions.append((params, trial_index))
    return suggestions

# --- EXTERNAL DATA LOADING ---

def load_external_calibration_data(volume, liquid, data_path=None, volume_filter=None, liquid_filter=None):
    """
    Load and filter external calibration data to replace screening phase.
    
    Args:
        volume: Target volume for calibration (mL)
        liquid: Target liquid for calibration  
        data_path: Path to external CSV file (uses EXTERNAL_DATA_PATH if None)
        volume_filter: Volume to filter by (uses EXTERNAL_DATA_VOLUME_FILTER if None)
        liquid_filter: Liquid to filter by (uses EXTERNAL_DATA_LIQUID_FILTER if None)
    
    Returns:
        list: List of result dicts compatible with screening_results format, or empty list if no data
    """
    
    # Use global config if not specified
    if data_path is None:
        data_path = EXTERNAL_DATA_PATH
    if volume_filter is None:
        volume_filter = EXTERNAL_DATA_VOLUME_FILTER  
    if liquid_filter is None:
        liquid_filter = EXTERNAL_DATA_LIQUID_FILTER
    
    # Return empty if external data is disabled or path not specified
    if not USE_EXTERNAL_DATA or not data_path or not os.path.exists(data_path):
        if USE_EXTERNAL_DATA:
            print(f"   ⚠️  External data enabled but file not found: {data_path}")
        return []
    
    try:
        print(f"🗂️  LOADING EXTERNAL CALIBRATION DATA")
        print(f"   📁 Source: {data_path}")
        
        # Load the CSV file
        df = pd.read_csv(data_path)
        print(f"   📊 Loaded {len(df)} total records")
        
        # Apply filters
        filtered_df = df.copy()
        
        # Filter by volume if specified
        if volume_filter is not None:
            volume_tolerance = 0.001  # Allow 1μL tolerance for volume matching
            filtered_df = filtered_df[abs(filtered_df.get('volume', 0) - volume_filter) <= volume_tolerance]
            print(f"   🔍 Volume filter ({volume_filter*1000:.0f}μL): {len(filtered_df)} records")
        
        # Filter by liquid if specified  
        if liquid_filter is not None:
            liquid_col = 'liquid' if 'liquid' in filtered_df.columns else None
            if liquid_col:
                filtered_df = filtered_df[filtered_df[liquid_col].str.lower() == liquid_filter.lower()]
                print(f"   🧪 Liquid filter ({liquid_filter}): {len(filtered_df)} records")
        
        # If no specific filters, use current experiment volume and liquid
        if volume_filter is None and liquid_filter is None:
            # Filter by current volume (with tolerance)
            volume_tolerance = 0.001
            filtered_df = filtered_df[abs(filtered_df.get('volume', 0) - volume) <= volume_tolerance]
            print(f"   🎯 Auto-filter by current volume ({volume*1000:.0f}μL): {len(filtered_df)} records")
            
            # Filter by current liquid if column exists
            if 'liquid' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['liquid'].str.lower() == liquid.lower()]
                print(f"   🧪 Auto-filter by current liquid ({liquid}): {len(filtered_df)} records")
        
        if len(filtered_df) == 0:
            print(f"   ❌ No data remaining after filtering")
            return []
        
        # Convert to screening_results format
        external_results = []
        required_columns = ['aspirate_speed', 'dispense_speed', 'deviation', 'time']
        optional_columns = ['aspirate_wait_time', 'dispense_wait_time', 'retract_speed', 
                           'blowout_vol', 'post_asp_air_vol', 'overaspirate_vol', 'variability',
                           'measured_volume', 'calculated_volume']
        
        for idx, row in filtered_df.iterrows():
            # Check for required columns
            missing_required = [col for col in required_columns if col not in row or pd.isna(row[col])]
            if missing_required:
                print(f"   ⚠️  Skipping row {idx}: missing required columns {missing_required}")
                continue
            
            # CRITICAL: External data MUST have actual measured volume - never reconstruct from deviation
            measured_volume = None
            if 'measured_volume' in row and not pd.isna(row['measured_volume']):
                measured_volume = float(row['measured_volume'])
            elif 'calculated_volume' in row and not pd.isna(row['calculated_volume']):
                measured_volume = float(row['calculated_volume'])
            else:
                # NEVER reconstruct measured volume from deviation - this makes wrong assumptions
                print(f"   ❌ ERROR: External data row {idx} missing both 'measured_volume' and 'calculated_volume' - skipping")
                print(f"   💡 External data must contain actual measured volumes, not just deviation percentages")
                continue  # Skip this row entirely
            
            # Build result dict  
            result = {
                "volume": volume,  # Use target volume
                "deviation": float(row['deviation']),
                "time": float(row['time']),
                "variability": float(row.get('variability', ADAPTIVE_PENALTY_VARIABILITY)),  # Use penalty if missing
                "strategy": "EXTERNAL_DATA",
                "liquid": liquid,  # Use target liquid
                "time_reported": datetime.now().isoformat(),
                "trial_index": f"ext_{idx}",  # Unique identifier
                "replicate_count": int(row.get('replicate_count', 1)),  # Default to 1
                "raw_measurements": [],  # Empty for external data - no individual measurements available
                "measured_volume": measured_volume  # CRITICAL: Always provide measured_volume for external data
            }
            
            # Add all parameter columns
            for col in ['aspirate_speed', 'dispense_speed'] + optional_columns:
                if col in row and not pd.isna(row[col]):
                    # Apply appropriate type conversion
                    if col in ['aspirate_speed', 'dispense_speed']:
                        result[col] = int(row[col])
                    else:
                        result[col] = float(row[col])
            
            external_results.append(result)
        
        print(f"   ✅ Successfully loaded {len(external_results)} external calibration records")
        print(f"   📈 Performance range: {min(r['deviation'] for r in external_results):.1f}-{max(r['deviation'] for r in external_results):.1f}% deviation")
        
        return external_results
        
    except Exception as e:
        print(f"   ❌ Error loading external data: {e}")
        return []

def load_external_data_or_run_screening(ax_client, lash_e, state, volume, expected_mass, expected_time, 
                                       autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set):
    """
    Attempt to load external data first, fall back to screening if no external data available.
    
    Returns:
        list: Results in screening_results format (either external or from screening)
    """
    
    # Try to load external data first
    external_results = load_external_calibration_data(volume, liquid)
    
    if external_results:
        print(f"   📂 Using {len(external_results)} external data records (screening skipped)")
        return external_results
    else:
        # Fall back to traditional screening
        print(f"   🔍 No external data available, running traditional screening")
        return run_screening_phase(ax_client, lash_e, state, volume, expected_mass, expected_time, 
                                  autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set)

# --- SCREENING PHASE (REUSE FROM MODULAR) ---

def run_screening_phase(ax_client, lash_e, state, volume, expected_mass, expected_time, 
                       autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set):
    """Run initial screening phase with SOBOL or LLM suggestions."""
    
    print(f"\n🔍 SCREENING PHASE: {INITIAL_PARAMETER_SETS} initial parameter sets...")
    
    screening_results = []
    llm_suggestions = []
    
    # Get all LLM suggestions upfront if using LLM
    if USE_LLM_FOR_SCREENING and LLM_AVAILABLE:
        print(f"   🤖 Requesting {INITIAL_PARAMETER_SETS} LLM suggestions...")
        llm_suggestions = get_llm_suggestions(ax_client, INITIAL_PARAMETER_SETS, screening_results, volume, liquid)
        print(f"   📝 Received {len(llm_suggestions)} LLM suggestions")
    
    for i in range(INITIAL_PARAMETER_SETS):
        print(f"   Screening trial {i+1}/{INITIAL_PARAMETER_SETS}...")
        
        if USE_LLM_FOR_SCREENING and LLM_AVAILABLE and i < len(llm_suggestions):
            # Use pre-generated LLM suggestion
            params, trial_index = llm_suggestions[i]
            print(f"     Using LLM suggestion {i+1}")
        else:
            # Get Ax suggestion (SOBOL) as fallback
            params, trial_index = ax_client.get_next_trial()
            print(f"     Using Ax/SOBOL suggestion {i+1}")
        
        # Enforce fixed parameters - override any values Ax suggested
        for param_name, fixed_value in FIXED_PARAMETERS.items():
            if param_name in params:
                params[param_name] = fixed_value
            else:
                params[param_name] = fixed_value
        
        # Run adaptive measurement (conditional replicates)
        liquid_source = get_liquid_source_with_vial_management(lash_e, state)
        adaptive_result = run_adaptive_measurement(
            lash_e, liquid_source, state["measurement_vial_name"], 
            volume, params, expected_mass, expected_time, 
            SIMULATE, autosave_raw_path, raw_measurements, 
            liquid, new_pipet_each_time_set, "SCREENING"
        )
        
        # Add result to model (3-objective) - using averaged values for good trials
        model_result = {
            "deviation": adaptive_result['deviation'],
            "variability": adaptive_result['variability'],
            "time": adaptive_result['time']
        }
        optimizer_3obj.add_result(ax_client, trial_index, model_result)
        
        # Store full result for later analysis
        full_result = dict(params)
        full_result.update({
            "volume": volume,
            "deviation": adaptive_result['deviation'],
            "variability": adaptive_result['variability'],
            "time": adaptive_result['time'],
            "trial_index": trial_index,
            "strategy": "SCREENING",
            "liquid": liquid,
            "time_reported": datetime.now().isoformat(),
            "replicate_count": adaptive_result['replicate_count'],
            "raw_measurements": adaptive_result['all_measurements'],
            "measured_volume": adaptive_result.get('measured_volume', 0)  # CRITICAL: Store measured volume for optimal conditions reporting
        })
        screening_results.append(full_result)
        
        # Display target vs measured volume
        target_ul = volume * 1000  # Convert to μL
        measured_ul = adaptive_result.get('measured_volume', 0) * 1000  # Convert to μL
        print(f"      → {adaptive_result['deviation']:.1f}% dev, {adaptive_result['variability']:.1f}% var, {adaptive_result['time']:.1f}s | Target: {target_ul:.1f}μL → Measured: {measured_ul:.1f}μL [{adaptive_result['replicate_count']} reps]")
    
    print(f"   ✅ Screening complete: {len(screening_results)} trials")
    return screening_results

# --- OVERASPIRATE CALIBRATION (REUSE FROM MODULAR) ---

def calculate_first_volume_constraint(best_candidate, volume, autosave_raw_path=None):
    """
    Calculate overaspirate constraint for first volume based on screening shortfall.
    
    Args:
        best_candidate: Best screening candidate with deviation and parameters
        volume: Target volume in mL
        autosave_raw_path: Path to save constraint log file
        
    Returns:
        tuple: (min_overaspirate_ml, max_overaspirate_ml) constraint bounds
    """
    # Calculate shortfall from the screening result using actual measured volume
    target_volume_ul = volume * 1000
    
    # Use actual average measured volume if available, otherwise fall back to deviation calculation
    raw_measurements = best_candidate.get('raw_measurements', [])
    if raw_measurements:
        # Use actual average measured volume (handles both over- and under-delivery)
        avg_measured_volume_ml = np.mean(raw_measurements)  # raw_measurements are in mL
        measured_volume_ul = avg_measured_volume_ml * 1000  # Convert to μL
        print(f"   📏 Using actual measured volume: {measured_volume_ul:.1f}μL from {len(raw_measurements)} measurements")
    else:
        # Fallback: calculate from deviation (assuming under-delivery)
        deviation_pct = best_candidate.get('deviation', 0)
        measured_volume_ul = target_volume_ul * (1 - deviation_pct / 100)
        print(f"   📏 Using deviation-calculated volume: {measured_volume_ul:.1f}μL (deviation: {deviation_pct:.1f}%)")
    
    shortfall_ul = target_volume_ul - measured_volume_ul  # Positive = under-delivery, Negative = over-delivery
    
    # Get existing overaspirate from screening parameters
    existing_overaspirate_ul = best_candidate.get('overaspirate_vol', 0) * 1000
    
    # Calculate constraint: existing + shortfall + buffer
    # Logic: Need baseline amount already tried + additional to cover shortfall + safety buffer
    # Note: shortfall can be negative (over-delivery), which would reduce total overaspirate needed
    raw_max_overaspirate_ul = existing_overaspirate_ul + shortfall_ul + OVERVOLUME_CALIBRATION_BUFFER_UL
    print(f"   📊 Constraint calc: {existing_overaspirate_ul:.1f} + {shortfall_ul:.1f} + {OVERVOLUME_CALIBRATION_BUFFER_UL:.1f} = {raw_max_overaspirate_ul:.1f}μL")
    
    # Apply consistent constraint logic (same as subsequent volumes)
    if raw_max_overaspirate_ul < 0:
        # Negative overaspirate means we need LESS volume than target
        # Set constraint to allow negative overaspirate values down to the calculated minimum
        min_overaspirate_ul = raw_max_overaspirate_ul  # e.g., -6.2μL
        max_overaspirate_ul = OVERVOLUME_CALIBRATION_BUFFER_UL  # Use configured buffer as max
        print(f"   🎯 Using NEGATIVE overaspirate range for first volume: [{min_overaspirate_ul:.1f}μL, {max_overaspirate_ul:.1f}μL]")
        print(f"   🔍 DEBUG: Screening suggests under-aspiration needed")
    elif raw_max_overaspirate_ul < OVERVOLUME_CALIBRATION_BUFFER_UL:
        # Positive but very small overaspirate - ensure at least buffer range
        min_overaspirate_ul = 0.0
        max_overaspirate_ul = OVERVOLUME_CALIBRATION_BUFFER_UL  # Use configured buffer
        print(f"   🎯 Using minimum buffer range for first volume: [{min_overaspirate_ul:.1f}μL, {max_overaspirate_ul:.1f}μL]")
        print(f"   🔍 DEBUG: Small calculated value ({raw_max_overaspirate_ul:.1f}μL) increased to configured buffer")
    else:
        # Normal positive overaspirate
        min_overaspirate_ul = 0.0
        max_overaspirate_ul = raw_max_overaspirate_ul
        print(f"   🎯 Using calculated overaspirate constraint for first volume: [{min_overaspirate_ul:.1f}μL, {max_overaspirate_ul:.1f}μL]")
    
    max_overaspirate_ml = max_overaspirate_ul / 1000  # Convert back to mL
    min_overaspirate_ml = min_overaspirate_ul / 1000  # Convert back to mL
    
    print(f"   📊 First volume constraint calculation:")
    if shortfall_ul >= 0:
        print(f"     Target: {target_volume_ul:.1f}μL, Measured: {measured_volume_ul:.1f}μL → Under-delivery: {shortfall_ul:.1f}μL")
    else:
        print(f"     Target: {target_volume_ul:.1f}μL, Measured: {measured_volume_ul:.1f}μL → Over-delivery: {abs(shortfall_ul):.1f}μL")
    print(f"     Existing overaspirate: {existing_overaspirate_ul:.1f}μL")
    print(f"     Buffer: {OVERVOLUME_CALIBRATION_BUFFER_UL:.1f}μL")
    print(f"     → Overaspirate constraint range: [{min_overaspirate_ul:.1f}μL, {max_overaspirate_ul:.1f}μL] ([{min_overaspirate_ml:.4f}mL, {max_overaspirate_ml:.4f}mL])")
    
    # Log constraint calculation to file
    if autosave_raw_path:
        log_file = os.path.join(os.path.dirname(autosave_raw_path), "constraint_log.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[{timestamp}] FIRST VOLUME CONSTRAINT CALCULATION\n")
            f.write(f"Volume: {volume*1000:.1f}μL\n")
            f.write(f"Target: {target_volume_ul:.1f}μL, Measured: {measured_volume_ul:.1f}μL\n")
            if shortfall_ul >= 0:
                f.write(f"Under-delivery: {shortfall_ul:.1f}μL\n")
            else:
                f.write(f"Over-delivery: {abs(shortfall_ul):.1f}μL\n")
            f.write(f"Existing overaspirate: {existing_overaspirate_ul:.1f}μL\n")
            f.write(f"Buffer: {OVERVOLUME_CALIBRATION_BUFFER_UL:.1f}μL\n")
            f.write(f"Overaspirate constraint range: [{min_overaspirate_ul:.1f}μL, {max_overaspirate_ul:.1f}μL]\n")
            f.write(f"Min overaspirate constraint: {min_overaspirate_ul:.1f}μL ({min_overaspirate_ml:.4f}mL)\n")
            f.write(f"Max overaspirate constraint: {max_overaspirate_ul:.1f}μL ({max_overaspirate_ml:.4f}mL)\n")
    
    return min_overaspirate_ml, max_overaspirate_ml

def calibrate_overvolume_post_optimization(optimized_params, remaining_volumes, lash_e, state, 
                                         autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set):
    """
    Calibrate overvolume parameters using final optimized parameters from first volume.
    Tests optimized parameters on remaining volumes and calculates volume-specific shortfalls.
    
    Args:
        optimized_params: Final optimized parameters from first volume
        remaining_volumes: List of volumes to test (excluding first volume)
        
    Returns:
        dict: {volume_ml: {'guess_ml': X, 'max_ml': Y}} for each remaining volume
    """
    import statistics  # For calculating precision from replicates
    
    if not remaining_volumes:
        print("⚠️  No remaining volumes for post-optimization overaspirate calibration")
        return {}
    
    print(f"\n🔬 POST-OPTIMIZATION OVERASPIRATE CALIBRATION")
    print(f"   Testing optimized parameters on {len(remaining_volumes)} volumes...")
    
    volume_calibrations = {}
    
    for volume in remaining_volumes:
        # Check budget before each measurement
        if global_measurement_count >= MAX_MEASUREMENTS:
            print(f"🛑 BUDGET EXHAUSTED: Cannot continue overaspirate calibration")
            break
            
        print(f"   🧪 Testing {volume*1000:.0f}μL...", end=" ")
        
        expected_mass = volume * LIQUIDS[liquid]["density"]
        expected_time = volume * 10.146 + 9.5813
        
        check_if_measurement_vial_full(lash_e, state)
        liquid_source = get_liquid_source_with_vial_management(lash_e, state)
        
        # Multiple measurements using optimized parameters for better reliability
        print(f"   📏 Running {PRECISION_MEASUREMENTS} replicate measurements for overaspirate calibration...")
        all_measurements = []
        all_times = []
        
        for rep in range(PRECISION_MEASUREMENTS):
            print(f"      Replicate {rep+1}/{PRECISION_MEASUREMENTS}...", end=" ")
            
            result = pipet_and_measure_tracked(lash_e, liquid_source, state["measurement_vial_name"], 
                                              volume, optimized_params, expected_mass, expected_time, 
                                              1, SIMULATE, autosave_raw_path, raw_measurements, 
                                              liquid, new_pipet_each_time_set, "POST_OPT_OVERVOLUME_ASSAY")
            
            # Get actual measured volume from this replicate
            if raw_measurements:
                actual_mass = raw_measurements[-1]['mass']
                actual_volume_ml = actual_mass / LIQUIDS[liquid]["density"]
                all_measurements.append(actual_volume_ml)
                all_times.append(result.get('time', 0))
                
                # Show individual result
                measured_ul = actual_volume_ml * 1000
                target_ul = volume * 1000
                deviation_pct = abs(measured_ul - target_ul) / target_ul * 100
                print(f"{measured_ul:.1f}μL ({deviation_pct:.1f}% dev)")
            else:
                all_measurements.append(volume)  # Fallback
                all_times.append(result.get('time', 0))
                print(f"sim")
        
        # Calculate averages from all replicates
        if all_measurements:
            actual_volume_ml = sum(all_measurements) / len(all_measurements)
            avg_time = sum(all_times) / len(all_times)
            
            # Calculate precision from replicates
            if len(all_measurements) > 1:
                # Safeguard against negative measurements causing invalid standard deviation
                if all(m >= 0 for m in all_measurements) and actual_volume_ml > 0:
                    volume_std = statistics.stdev(all_measurements)
                    precision_pct = (volume_std / actual_volume_ml) * 100
                    precision_pct = max(0.0, precision_pct)  # Ensure non-negative
                else:
                    print(f"⚠️  WARNING: Invalid measurements for precision - negative values detected: {all_measurements}")
                    print(f"    Setting precision to penalty value (100%) due to invalid measurements")
                    precision_pct = 100.0  # High penalty for invalid data
            else:
                precision_pct = 0
                
            print(f"   📊 Average: {actual_volume_ml*1000:.1f}μL ±{precision_pct:.1f}% ({len(all_measurements)} reps)")
        else:
            actual_volume_ml = volume  # Fallback
        
        # Calculate shortfall and overaspirate adjustments
        target_volume_ul = volume * 1000
        measured_volume_ul = actual_volume_ml * 1000
        shortfall_ul = target_volume_ul - measured_volume_ul
        
        existing_overaspirate_ul = optimized_params.get('overaspirate_vol', 0) * 1000
        
        # Calculate guess (no buffer) and max constraint (with buffer)
        guess_overaspirate_ul = existing_overaspirate_ul + shortfall_ul
        max_overaspirate_ul = guess_overaspirate_ul + OVERVOLUME_CALIBRATION_BUFFER_UL
        
        # Convert to mL and store
        volume_calibrations[volume] = {
            'guess_ml': guess_overaspirate_ul / 1000,
            'max_ml': max_overaspirate_ul / 1000,
            'shortfall_ul': shortfall_ul,
            'measured_volume_ul': measured_volume_ul
        }
        
        print(f"{measured_volume_ul:.1f}μL measured (shortfall: {shortfall_ul:+.1f}μL)")
        print(f"     → Guess: {guess_overaspirate_ul:.1f}μL, Max: {max_overaspirate_ul:.1f}μL")
        
        # Log subsequent volume constraint to file
        if autosave_raw_path:
            log_file = os.path.join(os.path.dirname(autosave_raw_path), "constraint_log.txt")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n[{timestamp}] SUBSEQUENT VOLUME CONSTRAINT CALCULATION\n")
                f.write(f"Volume: {volume*1000:.1f}μL\n")
                f.write(f"Target: {target_volume_ul:.1f}μL, Measured: {measured_volume_ul:.1f}μL\n")
                f.write(f"Shortfall: {shortfall_ul:+.1f}μL\n")
                f.write(f"Existing overaspirate: {existing_overaspirate_ul:.1f}μL\n")
                f.write(f"Buffer: {OVERVOLUME_CALIBRATION_BUFFER_UL:.1f}μL\n")
                f.write(f"Guess overaspirate: {guess_overaspirate_ul:.1f}μL\n")
                f.write(f"Max overaspirate constraint: {max_overaspirate_ul:.1f}μL\n")
    
    print(f"   ✅ Post-optimization overaspirate calibration complete for {len(volume_calibrations)} volumes")
    return volume_calibrations

def get_liquid_source_with_vial_management(lash_e, state, minimum_volume=2.0):
    """Get liquid source with vial management."""
    try:
        # Apply vial management first
        if base_module._VIAL_MANAGEMENT_MODE_OVERRIDE and base_module._VIAL_MANAGEMENT_MODE_OVERRIDE.lower() != "legacy":
            base_module.manage_vials(lash_e, state)
            
            # Get current source from config
            if base_module._VIAL_MANAGEMENT_CONFIG_OVERRIDE:
                cfg = {**base_module.VIAL_MANAGEMENT_DEFAULTS, **base_module._VIAL_MANAGEMENT_CONFIG_OVERRIDE}
            else:
                cfg = base_module.VIAL_MANAGEMENT_DEFAULTS
            
            current_source = cfg.get('source_vial', 'liquid_source_0')
            lash_e.logger.info(f"[vial-mgmt] Using current source vial: {current_source}")
            return current_source
    except Exception as e:
        lash_e.logger.info(f"[vial-mgmt] pre-pipetting management skipped: {e}")
    
    # Legacy mode with smart source switching
    return get_next_available_liquid_source(lash_e, state, minimum_volume)

def get_next_available_liquid_source(lash_e, state, minimum_volume=2.0):
    """Legacy mode: Find next available liquid source with enough volume."""
    # Initialize source index if not exists
    if "current_source_index" not in state:
        state["current_source_index"] = 0
    
    # Try current source first
    current_index = state["current_source_index"]
    current_source = f"liquid_source_{current_index}"
    
    try:
        current_vol = lash_e.nr_robot.get_vial_info(current_source, "vial_volume")
        if current_vol is not None and current_vol >= minimum_volume:
            # Current source is still good
            return current_source
        else:
            print(f"[legacy] {current_source} volume ({current_vol:.1f}mL) below minimum ({minimum_volume:.1f}mL)")
    except Exception as e:
        print(f"[legacy] Could not check {current_source} volume: {e}")
    
    # Current source is low, try to find next available source
    max_sources_to_check = 10  # Reasonable limit to avoid infinite loop
    for i in range(max_sources_to_check):
        next_index = current_index + 1 + i
        next_source = f"liquid_source_{next_index}"
        
        try:
            next_vol = lash_e.nr_robot.get_vial_info(next_source, "vial_volume")
            if next_vol is not None and next_vol >= minimum_volume:
                # Found good source, switch to it
                state["current_source_index"] = next_index
                print(f"[legacy] Switching to {next_source} (volume: {next_vol:.1f}mL)")
                return next_source
        except Exception as e:
            # Source doesn't exist or error accessing it
            continue
    
    # No good sources found, stick with current (will likely cause error downstream)
    print(f"[legacy] Warning: No liquid sources with ≥{minimum_volume:.1f}mL found, using {current_source}")
    return current_source

def check_if_measurement_vial_full(lash_e, state):
    """Check and handle full measurement vial."""
    current_vial = state["measurement_vial_name"]
    vol = lash_e.nr_robot.get_vial_info(current_vial, "vial_volume")
    if vol > 7.0:
        try:
            # Check if vial management is active
            if base_module._VIAL_MANAGEMENT_MODE_OVERRIDE and base_module._VIAL_MANAGEMENT_MODE_OVERRIDE.lower() != "legacy":
                print(f"[vial-mgmt] Measurement vial {current_vial} full ({vol:.1f}mL) - vial management will handle emptying")
                return
        except Exception as e:
            print(f"[vial-mgmt] Could not check vial management status: {e}")
        
        # Legacy behavior: switch to next measurement vial
        print(f"[legacy] Measurement vial {current_vial} full ({vol:.1f}mL) - switching to next vial")
        if not RETAIN_PIPET_BETWEEN_EXPERIMENTS:
            lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.return_vial_home(current_vial)
        
        # Ensure measurement_vial_index exists (robustness fix)
        if "measurement_vial_index" not in state:
            print(f"[legacy] Warning: measurement_vial_index missing from state, initializing to 0")
            state["measurement_vial_index"] = 0
        
        state["measurement_vial_index"] += 1
        new_vial_name = f"measurement_vial_{state['measurement_vial_index']}"
        state["measurement_vial_name"] = new_vial_name
        lash_e.logger.info(f"[info] Switching to new measurement vial: {new_vial_name}")
        lash_e.nr_robot.move_vial_to_location(new_vial_name, "clamp", 0)

# --- OVERASPIRATE CALIBRATION (FROM MODULAR) ---

def calibrate_overvolume_parameters(screening_candidates, remaining_volumes, lash_e, state, 
                                  autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set, 
                                  criteria, autosave_dir=None):
    """
    Automatically calibrate overvolume base and scaling parameters using multi-volume testing.
    
    Args:
        screening_candidates: List of screening results with parameters and performance
        remaining_volumes: List of volumes to test (excluding first volume)
        lash_e, state, etc.: Standard experiment parameters
        criteria: Criteria dict with accuracy thresholds
        
    Returns:
        tuple: (new_base_ul, new_scaling_percent, calibration_data) or (None, None, None) if failed
    """
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    
    if not screening_candidates or not remaining_volumes:
        print("⚠️  OVERVOLUME CALIBRATION: No candidates or volumes to test - skipping")
        return None, None, None
    
    # Step 1: Select best candidate using ranking system
    print(f"\n🔬 OVERVOLUME CALIBRATION: Selecting best candidate from {len(screening_candidates)} screening results...")
    
    # Use our existing ranking system to select the best candidate
    first_volume = VOLUMES[0]  # The volume that was screened
    tolerances = get_volume_dependent_tolerances(first_volume)
    
    # Rank the screening candidates 
    ranked_candidates = rank_candidates_by_priority(screening_candidates, first_volume, tolerances)
    best_candidate = ranked_candidates[0]
    
    # Extract parameters from the best candidate
    best_params = {k: v for k, v in best_candidate.items() if k in ALL_PARAMS}
    
    print(f"   ✅ Selected candidate: {best_candidate['accuracy_score']:.1f}% deviation, {best_candidate['time_score']:.1f}s")
    print(f"   📋 Testing these parameters on {len(remaining_volumes)} additional volumes...")
    
    # Step 2: Start with the first volume's data from the selected candidate
    calibration_data = []
    
    # Add the first volume's measurement data
    first_volume_ul = first_volume * 1000  # Convert to uL
    deviation_pct = best_candidate['deviation']
    
    # Use actual measured volume if available, otherwise calculate from deviation
    raw_measurements = best_candidate.get('raw_measurements', [])
    if raw_measurements:
        # Use actual average measured volume (handles both over- and under-delivery)
        avg_measured_volume_ml = np.mean(raw_measurements)  # raw_measurements are in mL
        measured_volume_ul = avg_measured_volume_ml * 1000  # Convert to μL
    else:
        # Fallback: calculate from deviation (assuming under-delivery)
        # Deviation = |target - measured| / target * 100 (absolute deviation)
        # For under-delivery (most common): measured = target * (1 - deviation/100)
        measured_volume_ul = first_volume_ul * (1 - deviation_pct / 100)
    
    calibration_data.append({
        'volume_set': first_volume_ul,
        'volume_measured': measured_volume_ul,
        'deviation_pct': deviation_pct,
        'existing_overaspirate_ul': best_params.get('overaspirate_vol', 0) * 1000  # Convert mL to uL
    })
    existing_overaspirate_first = best_params.get('overaspirate_vol', 0) * 1000
    print(f"   📊 Including first volume: {first_volume_ul:.0f}uL → {measured_volume_ul:.1f}uL ({deviation_pct:.1f}% dev, had {existing_overaspirate_first:.1f}uL overaspirate)")
    
    # Step 3: Test best parameters on remaining volumes  
    for volume in remaining_volumes:
        # Check budget before each overaspirate calibration measurement
        if global_measurement_count >= MAX_MEASUREMENTS:
            print(f"🛑 BUDGET EXHAUSTED: Cannot continue overaspirate calibration")
            print(f"   Used {global_measurement_count}/{MAX_MEASUREMENTS} measurements")
            break
            
        print(f"   🧪 Testing {volume*1000:.0f}uL...", end=" ")
        
        expected_mass = volume * LIQUIDS[liquid]["density"]
        expected_time = volume * 10.146 + 9.5813
        
        check_if_measurement_vial_full(lash_e, state)
        liquid_source = get_liquid_source_with_vial_management(lash_e, state)
        
        # Single measurement (no replicates as specified)
        result = pipet_and_measure_tracked(lash_e, liquid_source, state["measurement_vial_name"], 
                                          volume, best_params, expected_mass, expected_time, 
                                          1, SIMULATE, autosave_raw_path, raw_measurements, 
                                          liquid, new_pipet_each_time_set, "OVERVOLUME_ASSAY")
        
        # Get actual measured volume from raw_measurements
        if raw_measurements:
            actual_mass = raw_measurements[-1]['mass']
            actual_volume = actual_mass / LIQUIDS[liquid]["density"]  # Convert back to mL
        else:
            actual_volume = volume  # Fallback
        
        calibration_data.append({
            'volume_set': volume * 1000,      # Convert to uL for easier math
            'volume_measured': actual_volume * 1000,  # Convert to uL
            'deviation_pct': result.get('deviation', 0),
            'existing_overaspirate_ul': best_params.get('overaspirate_vol', 0) * 1000  # Store existing overaspirate in uL
        })
        
        print(f"{actual_volume*1000:.1f}uL measured ({result.get('deviation', 0):.1f}% dev)")
    
    # Step 4: Fit line to shortfalls and calculate overaspirate parameters
    try:
        # Calculate shortfalls (how much we're STILL under-delivering despite existing overaspirate)  
        x_data = np.array([d['volume_set'] for d in calibration_data]).reshape(-1, 1)
        shortfalls = np.array([d['volume_set'] - d['volume_measured'] for d in calibration_data])
        existing_overaspirates = np.array([d['existing_overaspirate_ul'] for d in calibration_data])
        
        print(f"   📊 Analyzing shortfalls from {len(calibration_data)} data points with existing overaspirate:")
        for i, d in enumerate(calibration_data):
            shortfall = d['volume_set'] - d['volume_measured'] 
            existing_over = d['existing_overaspirate_ul']
            print(f"     {d['volume_set']:.0f}uL → {d['volume_measured']:.1f}uL (shortfall: {shortfall:.1f}uL, had {existing_over:.1f}uL overaspirate)")
        
        # Fit line to shortfalls: shortfall = slope * volume + intercept
        model = LinearRegression()
        model.fit(x_data, shortfalls)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        print(f"   📈 Shortfall fit: additional_shortfall = {slope:.4f} * volume + {intercept:.2f}")
        
        # Calculate the average existing overaspirate to add to our formula
        avg_existing_overaspirate = np.mean(existing_overaspirates)
        print(f"   📊 Average existing overaspirate: {avg_existing_overaspirate:.1f}uL")
        
        # Calculate total overaspirate needed (existing + additional + buffer)
        min_overaspirate = 2.0  # Minimum 2uL to prevent crashes
        
        # Total formula: total_overaspirate = avg_existing + shortfall_from_line + buffer  
        new_base_ul = avg_existing_overaspirate + intercept + OVERVOLUME_CALIBRATION_BUFFER_UL
        new_scaling_percent = slope * 100  # Convert slope to percentage
        
        # Ensure minimum overaspirate for all volumes by adjusting base if needed
        smallest_volume = min([d['volume_set'] for d in calibration_data])
        min_overaspirate_at_smallest = new_base_ul + (new_scaling_percent/100) * smallest_volume
        
        if min_overaspirate_at_smallest < min_overaspirate:
            adjustment = min_overaspirate - min_overaspirate_at_smallest
            new_base_ul += adjustment
            print(f"   🔧 Adjusted base by +{adjustment:.1f}uL to ensure minimum {min_overaspirate:.1f}uL overaspirate")
        
        print(f"   🎯 Calibrated formula: overaspirate = {new_base_ul:.1f}uL + {new_scaling_percent:.1f}% * volume")
        
        # Apply safety bounds
        if new_base_ul > OVERVOLUME_MAX_BASE_UL:
            print(f"   ⚠️  Base {new_base_ul:.1f}uL exceeds limit {OVERVOLUME_MAX_BASE_UL:.1f}uL - capping")
            new_base_ul = OVERVOLUME_MAX_BASE_UL
            
        if new_scaling_percent > OVERVOLUME_MAX_PERCENT:
            print(f"   ⚠️  Scaling {new_scaling_percent:.1f}% exceeds limit {OVERVOLUME_MAX_PERCENT:.1f}% - capping")
            new_scaling_percent = OVERVOLUME_MAX_PERCENT
            
        if new_base_ul < 0:
            print(f"   ⚠️  Negative base {new_base_ul:.1f}uL - setting to 0")
            new_base_ul = 0
            
        if new_scaling_percent < 0:
            print(f"   ⚠️  Negative scaling {new_scaling_percent:.1f}% - setting to 0")
            new_scaling_percent = 0
        
        # Safety check: ensure at least some overaspirate capability
        if new_base_ul == 0 and new_scaling_percent == 0:
            print(f"   ⚠️  Both base and scaling are 0 - setting minimum base to 1uL to maintain optimization capability")
            new_base_ul = 1.0
        
        print(f"   ✅ Final calibrated values: base = {new_base_ul:.1f}uL, scaling = {new_scaling_percent:.1f}%")
        
        # Store raw shortfall coefficients for reporting
        for point in calibration_data:
            point['slope'] = slope  
            point['intercept'] = intercept  
        
        # Optional: Generate calibration plot (skip for now to keep simplified)
        print(f"   📊 Calibration plot generation skipped in simplified version")
        
        return new_base_ul, new_scaling_percent, calibration_data
        
    except Exception as e:
        print(f"   ❌ Calibration failed: {e}")
        return None, None, None

# --- MAIN WORKFLOW FUNCTIONS ---

def optimize_first_volume(volume, lash_e, state, autosave_raw_path, raw_measurements, 
                         liquid, new_pipet_each_time_set, all_results):
    """
    Optimize the first volume with full parameter optimization.
    
    Workflow:
    1. Screening phase
    2. Overaspirate calibration (if enabled)
    3. 3-objective optimization with simplified stopping
    4. Best candidate selection and precision test
    """
    
    print(f"\n🎯 OPTIMIZING FIRST VOLUME: {volume*1000:.0f}μL")
    
    # Calculate tolerances and expected values
    tolerances = get_volume_dependent_tolerances(volume)
    expected_mass = volume * LIQUIDS[liquid]["density"]
    expected_time = volume * 10.146 + 9.5813  # Simple time model
    tip_volume = get_tip_volume_for_volume(lash_e, volume)
    max_overaspirate_ul = get_max_overaspirate_ul(volume)
    
    print(f"   🔧 INITIAL CONSTRAINT: max_overaspirate_ul={max_overaspirate_ul:.6f}μL (from get_max_overaspirate_ul)")
    print(f"   Target tolerances: ±{tolerances['deviation_ul']:.1f}μL deviation, {tolerances['tolerance_percent']:.1f}% precision")
    
    # Check if optimizer is available
    if not OPTIMIZER_3OBJ_AVAILABLE:
        print("❌ 3-objectives optimizer not available - cannot proceed with first volume optimization")
        return False, None, None
    
    # Determine initial recommendations based on external data availability
    external_data_preview = load_external_calibration_data(volume, liquid)
    if external_data_preview:
        initial_recs = 0  # No SOBOL needed - we have external data
        print(f"   🗂️  External data available ({len(external_data_preview)} records) - skipping SOBOL initialization")
    else:
        initial_recs = INITIAL_PARAMETER_SETS  # Use SOBOL as usual
        print(f"   🎲 No external data - will use {initial_recs} SOBOL initial recommendations")
    
    # Create 3-objective optimizer for all parameters
    # Remove fixed parameters from optimization list
    optimize_params = [param for param in ALL_PARAMS if param not in FIXED_PARAMETERS]
    ax_client = optimizer_3obj.create_model(
        seed=SEED,
        num_initial_recs=initial_recs,
        bayesian_batch_size=PARAMETER_SETS_PER_RECOMMENDATION,
        volume=volume,
        tip_volume=tip_volume,
        model_type=BAYESIAN_MODEL_TYPE,
        optimize_params=optimize_params,  # Optimize non-fixed parameters only
        fixed_params=FIXED_PARAMETERS,
        simulate=SIMULATE,
        min_overaspirate_ul=0.0,  # Default min constraint for initial optimizer
        max_overaspirate_ul=max_overaspirate_ul
    )
    debug_ax_constraints(ax_client, "(INITIAL OPTIMIZER)", autosave_raw_path)
    
    # Phase 1: External Data Loading or Screening
    screening_results = load_external_data_or_run_screening(ax_client, lash_e, state, volume, expected_mass, 
                                                          expected_time, autosave_raw_path, raw_measurements, 
                                                          liquid, new_pipet_each_time_set)
    all_results.extend(screening_results)
    
    # Phase 2: Calculate first volume constraint based on screening shortfall
    print(f"\n🎯 CALCULATING FIRST VOLUME CONSTRAINT...")
    
    # Select best screening candidate using ranking system
    ranked_candidates = rank_candidates_by_priority(screening_results, volume, tolerances)
    best_candidate = ranked_candidates[0] if ranked_candidates else screening_results[0]
    print(f"   🏆 Selected best screening candidate: {best_candidate.get('deviation', 0):.1f}% deviation")
    
    min_overaspirate_ml_updated, max_overaspirate_ml_updated = calculate_first_volume_constraint(best_candidate, volume, autosave_raw_path)
    min_overaspirate_ul_updated = min_overaspirate_ml_updated * 1000  # Convert to μL for display and optimizer
    max_overaspirate_ul_updated = max_overaspirate_ml_updated * 1000  # Convert to μL for display and optimizer
    print(f"   ✅ Updated overaspirate constraint range: [{min_overaspirate_ul_updated:.1f}μL, {max_overaspirate_ul_updated:.1f}μL]")
    print(f"   🔍 DEBUG: mL range = [{min_overaspirate_ml_updated:.6f}mL, {max_overaspirate_ml_updated:.6f}mL], μL range = [{min_overaspirate_ul_updated:.6f}μL, {max_overaspirate_ul_updated:.6f}μL]")
    
    # Recreate ax_client with updated constraint if it changed  
    if abs(max_overaspirate_ul_updated - max_overaspirate_ul) > 0.01:  # Only recreate if meaningful change (compare in μL)
        print(f"   🔄 Recreating optimizer with updated constraint ({max_overaspirate_ul:.1f} → {max_overaspirate_ul_updated:.1f}μL)...")
        print(f"   🔧 CONSTRAINT DEBUG: About to pass min/max_overaspirate_ul=[{min_overaspirate_ul_updated:.6f}μL, {max_overaspirate_ul_updated:.6f}μL] to create_model")
        # Remove fixed parameters from optimization list
        optimize_params = [param for param in ALL_PARAMS if param not in FIXED_PARAMETERS]
        ax_client = optimizer_3obj.create_model(
            seed=SEED,
            num_initial_recs=0,  # No initial SOBOL since we already have screening data
            bayesian_batch_size=PARAMETER_SETS_PER_RECOMMENDATION,
            volume=volume,
            tip_volume=tip_volume,
            model_type=BAYESIAN_MODEL_TYPE,
            optimize_params=optimize_params,
            fixed_params=FIXED_PARAMETERS,
            simulate=SIMULATE,
            min_overaspirate_ul=min_overaspirate_ul_updated,  # Pass min bound
            max_overaspirate_ul=max_overaspirate_ul_updated   # Pass max bound
        )
        
        # Load screening results into the new optimizer
        optimizer_3obj.load_previous_data_into_model(ax_client, screening_results)
        print(f"   ✅ Loaded {len(screening_results)} screening results into updated optimizer")
        debug_ax_constraints(ax_client, "(AFTER UPDATE)", autosave_raw_path)
    else:
        print(f"   ✅ Constraint unchanged, keeping existing optimizer")
        debug_ax_constraints(ax_client, "(NO UPDATE)", autosave_raw_path)
    
    # Phase 3: 3-objective optimization with simplified stopping
    print(f"\n⚙️  3-OBJECTIVE OPTIMIZATION...")
    optimization_trial_count = 0
    
    while True:
        # HARD BUDGET CHECK: Stop immediately if we've hit the global limit
        if global_measurement_count >= MAX_MEASUREMENTS:
            print(f"   🛑 HARD BUDGET LIMIT REACHED: {global_measurement_count}/{MAX_MEASUREMENTS} measurements")
            break
            
        # Check stopping criteria
        stopping_result = check_stopping_criteria(all_results, volume, tolerances)
        print(f"   📊 Status: {stopping_result['total_trials']} trials, {stopping_result['good_trials']} good")
        
        if stopping_result['should_stop']:
            print(f"   🛑 STOPPING: {stopping_result['reason']}")
            break
        else:
            print(f"   🔄 CONTINUING: {stopping_result['reason']}")
        
        # Get next suggestion
        if optimization_trial_count == 0:  # Only debug on first optimization trial to avoid spam
            debug_ax_constraints(ax_client, "(BEFORE FIRST SUGGESTION)", autosave_raw_path)
        
        params, trial_index = optimizer_3obj.get_suggestions(ax_client, volume, n=1)[0]
        optimization_trial_count += 1
        
        # Enforce fixed parameters - override any values optimizer suggested
        for param_name, fixed_value in FIXED_PARAMETERS.items():
            if param_name in params:
                params[param_name] = fixed_value
        
        print(f"   Optimization trial {optimization_trial_count}...")
        if optimization_trial_count == 1:  # Show the first suggested parameters to verify constraint application
            overaspirate_suggested_ul = params.get('overaspirate_vol', 0) * 1000  # Convert to μL
            print(f"   🔍 FIRST SUGGESTION: overaspirate_vol = {overaspirate_suggested_ul:.3f}μL")
        
        # Run adaptive measurement (conditional replicates)
        check_if_measurement_vial_full(lash_e, state)
        liquid_source = get_liquid_source_with_vial_management(lash_e, state)
        
        adaptive_result = run_adaptive_measurement(
            lash_e, liquid_source, state["measurement_vial_name"], 
            volume, params, expected_mass, expected_time, 
            SIMULATE, autosave_raw_path, raw_measurements, 
            liquid, new_pipet_each_time_set, "OPTIMIZATION"
        )
        
        # Add result to model (3-objective) - using averaged values for good trials
        model_result = {
            "deviation": adaptive_result['deviation'],
            "variability": adaptive_result['variability'],
            "time": adaptive_result['time']
        }
        optimizer_3obj.add_result(ax_client, trial_index, model_result)
        
        # Store full result
        full_result = dict(params)
        full_result.update({
            "volume": volume,
            "deviation": adaptive_result['deviation'],
            "variability": adaptive_result['variability'],
            "time": adaptive_result['time'],
            "trial_index": trial_index,
            "strategy": f"OPTIMIZATION_{optimization_trial_count}",
            "liquid": liquid,
            "time_reported": datetime.now().isoformat(),
            "replicate_count": adaptive_result['replicate_count'],
            "raw_measurements": adaptive_result['all_measurements'],
            "measured_volume": adaptive_result.get('measured_volume', 0)  # CRITICAL: Store measured volume for optimal conditions reporting
        })
        all_results.append(full_result)
        
        quality = evaluate_trial_quality(full_result, volume, tolerances)
        quality_status = "✅ GOOD" if quality['is_good'] else "❌ needs improvement"
        
        # Display target vs measured volume
        target_ul = volume * 1000  # Convert to μL  
        measured_ul = adaptive_result.get('measured_volume', 0) * 1000  # Convert to μL
        print(f"      → {adaptive_result['deviation']:.1f}% dev, {adaptive_result['variability']:.1f}% var, {adaptive_result['time']:.1f}s | Target: {target_ul:.1f}μL → Measured: {measured_ul:.1f}μL ({quality_status}) [{adaptive_result['replicate_count']} reps]")
    
    # Phase 4: Select best candidate and run precision test
    print(f"\n🏆 SELECTING BEST CANDIDATE...")
    
    # Get all first volume trials (screening + optimization) for ranking
    first_volume_trials = [r for r in all_results 
                          if r.get('volume') == volume 
                          and r.get('strategy') in ['SCREENING'] or r.get('strategy', '').startswith('OPTIMIZATION')]
    
    if not first_volume_trials:
        print("   ❌ No trials found for ranking!")
        return False, None, None
    
    print(f"   🔍 Ranking {len(first_volume_trials)} total trials (screening + optimization)")
    
    # Rank candidates
    ranked_candidates = rank_candidates_by_priority(first_volume_trials, volume, tolerances)
    best_candidate = ranked_candidates[0]
    
    print(f"   🎯 Selected best candidate:")
    print(f"      Accuracy: {best_candidate['accuracy_score']:.1f}% deviation")
    print(f"      Precision: {best_candidate['precision_score']:.1f}% variability")
    print(f"      Time: {best_candidate['time_score']:.1f}s")
    print(f"      Quality: {'✅ GOOD' if best_candidate['quality_evaluation']['is_good'] else '❌ Not good'}")
    
    # Phase 5: Check if best candidate meets tolerance
    quality = best_candidate['quality_evaluation']
    tolerance_met = quality['is_good']
    
    best_params = {k: v for k, v in best_candidate.items() if k in ALL_PARAMS}
    
    if tolerance_met:
        print(f"\n✅ FIRST VOLUME OPTIMIZATION COMPLETE!")
        print(f"   Selected parameters meet tolerance requirements")
        print(f"   Parameters will be used as baseline for subsequent volumes")
        return True, best_params, best_candidate
    else:
        print(f"\n🔶 FIRST VOLUME PARTIAL SUCCESS!")
        print(f"   Best parameters found but do not meet strict tolerance")
        print(f"   Accuracy: {quality['accuracy_deviation_ul']:.2f}μL > {quality['accuracy_tolerance_ul']:.2f}μL tolerance")
        
        # Calculate rescue overaspirate constraints based on best candidate shortfall
        min_rescue_overaspirate_ml, max_rescue_overaspirate_ml = calculate_first_volume_constraint(best_candidate, volume)
        rescue_overaspirate_constraint = max_rescue_overaspirate_ml  # Use the max constraint for rescue
        rescue_overaspirate_ul = rescue_overaspirate_constraint * 1000  # Convert to μL
        
        # Store as volume-specific calibration for rescue optimization
        global volume_overaspirate_calibrations  
        volume_overaspirate_calibrations = {
            volume: {
                'guess_ml': rescue_overaspirate_constraint,
                'max_ml': rescue_overaspirate_constraint,  # Use same value for both
                'shortfall_ul': 0,  # Placeholder
                'measured_volume_ul': 0  # Placeholder
            }
        }
        
        print(f"\n🎯 RESCUE OVERASPIRATE CONSTRAINT: {rescue_overaspirate_ul:.1f}μL (based on best candidate shortfall)")
        
        # Try rescue optimization with volume-dependent parameter refinement
        print(f"\n🚑 RESCUE OPTIMIZATION: Attempting targeted parameter refinement...")
        
        # Calculate rescue budget (fair share of remaining measurements)
        volumes_remaining = len(VOLUMES) - 1  # Excluding current volume
        measurements_remaining = MAX_MEASUREMENTS - global_measurement_count
        rescue_budget = max(5, measurements_remaining // (volumes_remaining + 1)) if volumes_remaining > 0 else measurements_remaining
        rescue_budget = min(rescue_budget, measurements_remaining)  # Cap at what's available
        
        print(f"   📊 Rescue budget: {rescue_budget} measurements")
        
        if rescue_budget >= 2:  # Need at least 2 measurements for meaningful rescue attempt
            # Run rescue optimization using subsequent volume logic
            rescue_success, rescue_params, rescue_status = optimize_subsequent_volume_budget_aware(
                volume, lash_e, state, autosave_raw_path, raw_measurements,
                LIQUID, new_pipet_each_time_set, all_results, best_params, rescue_budget
            )
            
            # Find rescue trials for comparison
            rescue_trials = [r for r in all_results 
                           if r.get('volume') == volume and r.get('strategy') == 'INHERITED_TEST']
            
            # Rank original vs rescue candidates to pick the best overall
            all_candidates = [best_candidate] + rescue_trials
            ranked_comparison = rank_candidates_by_priority(all_candidates, volume, tolerances)
            final_best = ranked_comparison[0]
            
            # Extract final parameters
            final_params = {k: v for k, v in final_best.items() if k in ALL_PARAMS}
            
            # Check if final result meets tolerance
            final_quality = final_best['quality_evaluation']
            final_tolerance_met = final_quality['is_good']
            
            # Log what happened
            if final_best in rescue_trials:
                if final_tolerance_met:
                    print(f"   ✅ RESCUE SUCCESSFUL: Found parameters meeting tolerance!")
                else:
                    print(f"   � RESCUE IMPROVED: Better than original but still not meeting tolerance")
            else:
                print(f"   � ORIGINAL BETTER: Keeping original parameters")
            
            return final_tolerance_met, final_params, final_best
        else:
            print(f"   ⚠️  Insufficient budget for rescue attempt ({rescue_budget} measurements)")
            print(f"   📊 Using original best parameters as baseline for subsequent volumes")
            return False, best_params, best_candidate


def optimize_subsequent_volume_budget_aware(volume, lash_e, state, autosave_raw_path, raw_measurements, 
                                           liquid, new_pipet_each_time_set, all_results, successful_params, measurements_budget):
    """
    Budget-aware optimization for subsequent volumes using unified screening approach.
    
    Process:
    1. Test inherited parameters (1 measurement, with replicates if good)
    2. Continue optimization within budget using existing adaptive screening logic
    3. Always return best parameters found, even if tolerance not met
    
    Args:
        successful_params: Parameters from first successful volume to use as baseline
        measurements_budget: Maximum measurements allowed for this volume
        
    Returns:
        tuple: (tolerance_met, best_params, status)
               status: 'success' if tolerance met, 'partial_success' if best effort
    """
    
    print(f"\n🎯 BUDGET-AWARE OPTIMIZATION: {volume*1000:.0f}μL")
    print(f"   Budget: {measurements_budget} measurements")
    print(f"   Starting with inherited parameters from first volume")
    
    # Calculate tolerances and expected values
    tolerances = get_volume_dependent_tolerances(volume)
    expected_mass = volume * LIQUIDS[liquid]["density"]
    expected_time = volume * 10.146 + 9.5813
    deviation_threshold = ADAPTIVE_DEVIATION_THRESHOLD
    
    # Track measurements used for this volume (starting count)
    volume_start_count = global_measurement_count
    volume_results = []  # Results for this volume only
    
    # Prepare inherited parameters with volume-specific overaspirate if available
    test_params = successful_params.copy()
    # DEBUG: Show what calibration data we have
    print(f"   🔍 DEBUG: Available calibrations: {list(volume_overaspirate_calibrations.keys())}")
    print(f"   🔍 DEBUG: Looking for volume: {volume} (mL)")
    
    # Use volume in mL for lookup (that's how calibration data is stored)
    if volume in volume_overaspirate_calibrations:
        old_overaspirate = test_params.get('overaspirate_vol', 0) * 1000  # Convert to μL
        calib_data = volume_overaspirate_calibrations[volume]
        new_overaspirate_ml = calib_data['guess_ml']
        test_params['overaspirate_vol'] = new_overaspirate_ml  # Already in mL
        print(f"   🎯 Using calibrated overaspirate guess: {old_overaspirate:.1f}μL → {new_overaspirate_ml*1000:.1f}μL")
        print(f"   🔍 DEBUG: Calibration data: guess={calib_data['guess_ml']*1000:.1f}μL, max={calib_data['max_ml']*1000:.1f}μL")
    else:
        print(f"   🎯 Using inherited overaspirate: {test_params.get('overaspirate_vol', 0)*1000:.1f}μL")
        print(f"   🔍 DEBUG: No calibration found for {volume} mL")

    # Test inherited parameters first
    print(f"   Testing inherited parameters...")
    check_if_measurement_vial_full(lash_e, state)
    liquid_source = get_liquid_source_with_vial_management(lash_e, state)
    
    inherited_result = pipet_and_measure_tracked(lash_e, liquid_source, state["measurement_vial_name"], 
                                               volume, test_params, expected_mass, expected_time, 
                                               1, SIMULATE, autosave_raw_path, raw_measurements, 
                                               liquid, new_pipet_each_time_set, "INHERITED_TEST")
    
    deviation = inherited_result.get('deviation', float('inf'))
    print(f"   Inherited result: {deviation:.1f}% deviation")
    
    # Use existing adaptive testing logic for the inherited parameters
    current_measurements = get_volume_measurement_count(volume_start_count)
    if deviation <= deviation_threshold and current_measurements + PRECISION_MEASUREMENTS - 1 <= measurements_budget:
        # Good result - run additional replicates using existing logic
        additional_replicates = PRECISION_MEASUREMENTS - 1
        print(f"   ✅ Good deviation - running {additional_replicates} additional replicates...")
        
        all_deviations = [deviation]
        all_times = [inherited_result.get('time', 0)]
        all_measurements = []
        
        # Extract measurement data
        if raw_measurements:
            actual_mass = raw_measurements[-1]['mass']
            liquid_density = LIQUIDS[liquid]["density"]
            actual_volume = actual_mass / liquid_density
            all_measurements.append(actual_volume)
        
        for i in range(additional_replicates):
            replicate_num = i + 2
            print(f"      Replicate {replicate_num}/{PRECISION_MEASUREMENTS}...", end=" ")
            
            check_if_measurement_vial_full(lash_e, {"measurement_vial_name": state["measurement_vial_name"], "measurement_vial_index": state.get("measurement_vial_index", 0)})
            
            result = pipet_and_measure_tracked(lash_e, liquid_source, state["measurement_vial_name"], 
                                              volume, test_params, expected_mass, expected_time, 
                                              1, SIMULATE, autosave_raw_path, raw_measurements, 
                                              liquid, new_pipet_each_time_set, "INHERITED_TEST")
            
            rep_deviation = result['deviation']
            all_deviations.append(rep_deviation)
            all_times.append(result.get('time', 0))
            
            if raw_measurements:
                actual_mass = raw_measurements[-1]['mass']
                actual_volume = actual_mass / liquid_density
                all_measurements.append(actual_volume)
            
            print(f"{rep_deviation:.1f}% dev")
        
        # Calculate final metrics using existing logic
        avg_deviation = np.mean(all_deviations)
        avg_time = np.mean(all_times)
        avg_measured_volume = np.mean(all_measurements) if all_measurements else 0
        
        if len(all_measurements) > 1:
            # Safeguard against negative measurements causing invalid variability
            if all(m >= 0 for m in all_measurements) and np.mean(all_measurements) > 0:
                volume_std = np.std(all_measurements)
                variability = volume_std / np.mean(all_measurements) * 100
                variability = max(0.0, variability)  # Ensure non-negative
            else:
                print(f"⚠️  WARNING: Invalid measurements for variability - negative values detected: {all_measurements}")
                print(f"    Setting variability to penalty value ({ADAPTIVE_PENALTY_VARIABILITY}%) due to invalid measurements")
                variability = ADAPTIVE_PENALTY_VARIABILITY
        else:
            variability = ADAPTIVE_PENALTY_VARIABILITY
        
        # Create comprehensive result
        inherited_comprehensive_result = {
            'volume': volume,
            'deviation': avg_deviation,
            'time': avg_time,
            'variability': variability,
            'replicate_count': PRECISION_MEASUREMENTS,
            'strategy': 'INHERITED_TEST',
            'measured_volume': avg_measured_volume,  # CRITICAL: Store measured volume for optimal conditions reporting
            **test_params
        }
        volume_results.append(inherited_comprehensive_result)
        
        # Check if inherited parameters meet tolerance - convert percentage to μL units
        avg_deviation_ul = (avg_deviation / 100.0) * volume * 1000  # Convert % to μL
        variability_ul = (variability / 100.0) * volume * 1000 if variability != ADAPTIVE_PENALTY_VARIABILITY else variability  # Convert % to μL unless penalty
        
        tolerance_met = (avg_deviation_ul <= tolerances['deviation_ul'] and 
                        variability_ul <= tolerances['variation_ul'])
        
        print(f"   📊 Tolerance check: {avg_deviation_ul:.2f}μL ≤ {tolerances['deviation_ul']:.2f}μL dev, {variability_ul:.2f}μL ≤ {tolerances['variation_ul']:.2f}μL var → {'✅ PASS' if tolerance_met else '❌ FAIL'}")
        
        if tolerance_met:
            print(f"   ✅ INHERITED PARAMETERS SUCCESSFUL: {avg_deviation:.1f}% dev, {variability:.1f}% var")
            
            # Add the successful inherited test result to all_results so CSV can find it
            all_results.append(inherited_comprehensive_result)
            
            return True, test_params, 'success'
    else:
        # Poor result or insufficient budget for replicates - use penalty variability
        measured_volume = inherited_result.get('measured_volume', 0)  # Get measured volume from single test
        inherited_comprehensive_result = {
            'volume': volume,
            'deviation': deviation,
            'time': inherited_result.get('time', 0),
            'variability': ADAPTIVE_PENALTY_VARIABILITY,
            'replicate_count': 1,
            'strategy': 'INHERITED_TEST',
            'measured_volume': measured_volume,  # CRITICAL: Store measured volume for optimal conditions reporting
            **test_params
        }
        volume_results.append(inherited_comprehensive_result)
        print(f"   ❌ Inherited parameters need improvement: {deviation:.1f}% deviation")
    
    # Continue optimization within remaining budget using existing screening approach
    current_measurements = get_volume_measurement_count(volume_start_count)
    remaining_budget = measurements_budget - current_measurements
    print(f"   Continuing optimization with {remaining_budget} measurements remaining...")
    
    if remaining_budget > 0:
        # Use the existing optimization logic but with budget constraint
        # This reuses the proven adaptive testing approach from the initial volume
        optimization_results = run_budget_constrained_optimization(
            volume, lash_e, state, autosave_raw_path, raw_measurements,
            liquid, new_pipet_each_time_set, test_params, remaining_budget, tolerances, all_results
        )
        volume_results.extend(optimization_results)
    
    # Always select best parameters from all tested candidates
    if volume_results:
        # Use existing ranking logic to find best parameters
        ranked_results = rank_candidates_by_priority(volume_results, volume, tolerances)
        best_result = ranked_results[0] if ranked_results else volume_results[0]
        best_params = {k: v for k, v in best_result.items() 
                      if k not in ['volume', 'deviation', 'time', 'variability', 'replicate_count', 'strategy']}
        
        # Check if we met tolerance - convert percentage to μL units for proper comparison
        best_deviation_ul = (best_result['deviation'] / 100.0) * volume * 1000  # Convert % to μL
        best_variability_ul = (best_result['variability'] / 100.0) * volume * 1000 if best_result['variability'] != ADAPTIVE_PENALTY_VARIABILITY else best_result['variability']
        
        tolerance_met = (best_deviation_ul <= tolerances['deviation_ul'] and 
                        best_variability_ul <= tolerances['variation_ul'])
        
        print(f"   📊 Final tolerance check: {best_deviation_ul:.2f}μL ≤ {tolerances['deviation_ul']:.2f}μL dev, {best_variability_ul:.2f}μL ≤ {tolerances['variation_ul']:.2f}μL var → {'✅ SUCCESS' if tolerance_met else '❌ PARTIAL'}")
        
        status = 'success' if tolerance_met else 'partial_success'
        
        total_measurements_used = get_volume_measurement_count(volume_start_count)
        print(f"   📊 VOLUME COMPLETE: Used {total_measurements_used}/{measurements_budget} measurements")
        print(f"   📈 Best result: {best_result['deviation']:.1f}% dev, {best_result['variability']:.1f}% var")
        
        # Add all results to global results list
        for result in volume_results:
            all_results.append(result)
        
        return tolerance_met, best_params, status
    else:
        print(f"   ❌ No valid results obtained")
        return False, successful_params, 'failed'

def run_budget_constrained_optimization(volume, lash_e, state, autosave_raw_path, raw_measurements,
                                       liquid, new_pipet_each_time_set, successful_params, budget, tolerances, all_results):
    """
    Run optimization within measurement budget using configurable optimization type.
    
    - Single-objective: deviation only (if BAYESIAN_MODEL_TYPE_SUBSEQUENT is qLogEI/qEI)
    - Multi-objective: deviation + variability (if BAYESIAN_MODEL_TYPE_SUBSEQUENT is qNEHVI)
    
    Only optimizes volume-dependent parameters, keeps others fixed from successful_params.
    """
    if budget <= 0:
        return []
    
    # Choose optimizer based on model type
    use_single_objective = BAYESIAN_MODEL_TYPE_SUBSEQUENT in ['qLogEI', 'qEI']
        
    print(f"   Running budget-constrained optimization: {budget} measurements available")
    if use_single_objective:
        print(f"   Using single-objective optimization (deviation only) with {BAYESIAN_MODEL_TYPE_SUBSEQUENT}")
    else:
        print(f"   Using 2-objective optimization (deviation + variability) with {BAYESIAN_MODEL_TYPE_SUBSEQUENT}")
    
    # Calculate expected values
    expected_mass = volume * LIQUIDS[liquid]["density"]
    expected_time = volume * 10.146 + 9.5813
    
    # Get volume-dependent parameters to optimize  
    volume_dependent_params = ['overaspirate_vol', 'blowout_vol']
    fixed_params = {k: v for k, v in successful_params.items() 
                   if k not in volume_dependent_params}
    # Merge with global fixed parameters (global FIXED_PARAMETERS take priority)
    fixed_params.update(FIXED_PARAMETERS)
    
    if use_single_objective and not OPTIMIZER_SINGLE_AVAILABLE:
        print("   ⚠️  Single-objective optimizer not available - using inherited parameters")
        return []
    elif not use_single_objective and not OPTIMIZER_3OBJ_AVAILABLE:
        print("   ⚠️  Multi-objective optimizer not available - using inherited parameters") 
        return []
    
    # Create appropriate optimizer
    tip_volume = get_tip_volume_for_volume(lash_e, volume)
    
    # Use volume-specific overaspirate calibration if available, otherwise default
    global volume_overaspirate_calibrations
    print(f"   🔍 DEBUG: Optimization constraint lookup - volume={volume} mL, available calibrations: {list(volume_overaspirate_calibrations.keys())}")
    
    # Use volume in mL for lookup (that's how calibration data is stored)
    if volume in volume_overaspirate_calibrations:
        calib_data = volume_overaspirate_calibrations[volume]
        raw_max_overaspirate_ul = calib_data['max_ml'] * 1000  # Convert to μL
        
        # Handle negative overaspirate constraints properly
        if raw_max_overaspirate_ul < 0:
            # Negative overaspirate means we need LESS volume than target
            # Set constraint to allow negative overaspirate values down to the calibrated minimum
            min_overaspirate_ul = raw_max_overaspirate_ul  # e.g., -6.2μL
            max_overaspirate_ul = OVERVOLUME_CALIBRATION_BUFFER_UL  # Use configured buffer as max
            print(f"   🎯 Using calibrated NEGATIVE overaspirate range: [{min_overaspirate_ul:.1f}μL, {max_overaspirate_ul:.1f}μL]")
            print(f"   🔍 DEBUG: Calibration suggests under-aspiration - guess={calib_data['guess_ml']*1000:.1f}μL, max={raw_max_overaspirate_ul:.1f}μL")
            print(f"   📏 Range spans {max_overaspirate_ul - min_overaspirate_ul:.1f}μL to give optimizer meaningful choices")
        elif raw_max_overaspirate_ul < OVERVOLUME_CALIBRATION_BUFFER_UL:
            # Positive but very small overaspirate - ensure at least buffer range
            min_overaspirate_ul = 0.0
            max_overaspirate_ul = OVERVOLUME_CALIBRATION_BUFFER_UL  # Use configured buffer
            print(f"   🎯 Using minimum buffer overaspirate range: [{min_overaspirate_ul:.1f}μL, {max_overaspirate_ul:.1f}μL]")
            print(f"   🔍 DEBUG: Small calibrated value ({raw_max_overaspirate_ul:.1f}μL) increased to configured buffer ({OVERVOLUME_CALIBRATION_BUFFER_UL:.1f}μL)")
        else:
            # Positive overaspirate - normal case
            min_overaspirate_ul = 0.0
            max_overaspirate_ul = raw_max_overaspirate_ul
            print(f"   🎯 Using calibrated overaspirate constraint: {max_overaspirate_ul:.1f}μL")
            print(f"   🔍 DEBUG: From calibration - guess={calib_data['guess_ml']*1000:.1f}μL, max={calib_data['max_ml']*1000:.1f}μL")
    else:
        # Default case - always positive
        min_overaspirate_ul = 0.0
        max_overaspirate_ul = get_max_overaspirate_ul(volume)
        print(f"   🎯 Using default overaspirate constraint: {max_overaspirate_ul:.1f}μL")
        print(f"   🔍 DEBUG: No calibration data found for volume {volume} mL")
    
    try:
        if use_single_objective:
            # Single-objective optimization (deviation only)
            ax_client = optimizer_single.create_model(
                seed=SEED,
                num_initial_recs=min(3, budget // 2),  # Use half budget for initial exploration
                bayesian_batch_size=1,
                volume=volume,
                tip_volume=tip_volume,
                model_type=BAYESIAN_MODEL_TYPE_SUBSEQUENT,  # qLogEI or qEI
                optimize_params=volume_dependent_params,
                fixed_params=fixed_params,
                simulate=SIMULATE,
                min_overaspirate_ul=min_overaspirate_ul,
                max_overaspirate_ul=max_overaspirate_ul
            )
        else:
            # Multi-objective optimization (deviation + variability)
            ax_client = optimizer_3obj.create_model(
                seed=SEED,
                num_initial_recs=min(3, budget // 2),  # Use half budget for initial exploration
                bayesian_batch_size=1,
                volume=volume,
                tip_volume=tip_volume,
                model_type=BAYESIAN_MODEL_TYPE_SUBSEQUENT,
                optimize_params=volume_dependent_params,
                fixed_params=fixed_params,
                simulate=SIMULATE,
                min_overaspirate_ul=min_overaspirate_ul,
                max_overaspirate_ul=max_overaspirate_ul
            )
    except Exception as e:
        print(f"   ⚠️  Could not create optimizer: {e}")
        return []
    
    optimization_results = []
    optimization_start_count = global_measurement_count
    optimization_trial_count = 0
    
    # Run optimization within budget
    while get_volume_measurement_count(optimization_start_count) < budget:
        try:
            params, trial_index = optimizer_3obj.get_suggestions(ax_client, volume, n=1)[0]
            optimization_trial_count += 1
            
            # Enforce fixed parameters - override any values optimizer suggested
            for param_name, fixed_value in FIXED_PARAMETERS.items():
                if param_name in params:
                    params[param_name] = fixed_value
            
            print(f"      Optimization trial {optimization_trial_count}...", end=" ")
            
            # Run adaptive measurement (may use multiple measurements if result is good)
            check_if_measurement_vial_full(lash_e, state)
            liquid_source = get_liquid_source_with_vial_management(lash_e, state)
            
            # Estimate measurements this trial might use
            current_measurements = get_volume_measurement_count(optimization_start_count)
            estimated_measurements = 1  # At minimum, we'll use 1
            if current_measurements + PRECISION_MEASUREMENTS <= budget:
                # We have budget for replicates if the result is good
                max_possible_measurements = PRECISION_MEASUREMENTS
            else:
                # Limited budget - only single measurements
                max_possible_measurements = 1
            
            if current_measurements + estimated_measurements > budget:
                print("budget exhausted")
                break
                
            adaptive_result = run_adaptive_measurement(
                lash_e, liquid_source, state["measurement_vial_name"], 
                volume, params, expected_mass, expected_time, 
                SIMULATE, autosave_raw_path, raw_measurements, 
                liquid, new_pipet_each_time_set, f"OPTIMIZATION_{optimization_trial_count}"
            )
            
            # Track actual measurements used (automatically handled by pipet_and_measure_tracked)
            actual_measurements_used = adaptive_result.get('replicate_count', 1)
            
            print(f"{adaptive_result['deviation']:.1f}% dev ({actual_measurements_used} meas)")
            
            # Add result to optimizer 
            if use_single_objective:
                # Single-objective: only deviation matters
                model_result = {"deviation": adaptive_result['deviation']}
                optimizer_single.add_result(ax_client, trial_index, model_result)
            else:
                # Multi-objective: deviation + variability, time set to minimal impact
                model_result = {
                    "deviation": adaptive_result['deviation'],
                    "variability": adaptive_result['variability'], 
                    "time": 30.0  # Fixed neutral time since we're not optimizing for it
                }
                optimizer_3obj.add_result(ax_client, trial_index, model_result)
            
            # Store result for ranking
            full_result = dict(params)
            full_result.update({
                "volume": volume,
                "deviation": adaptive_result['deviation'],
                "variability": adaptive_result['variability'],
                "time": adaptive_result['time'],
                "measured_volume": adaptive_result['measured_volume'],  # CRITICAL: Include measured volume
                "replicate_count": actual_measurements_used,
                "strategy": f"OPTIMIZATION_{optimization_trial_count}"
            })
            optimization_results.append(full_result)
            
            # Early termination if we found good enough result - convert percentage to μL units
            deviation_ul = (adaptive_result['deviation'] / 100.0) * volume * 1000  # Convert % to μL
            variability_ul = (adaptive_result['variability'] / 100.0) * volume * 1000 if adaptive_result['variability'] != ADAPTIVE_PENALTY_VARIABILITY else adaptive_result['variability']
            
            tolerance_met = (deviation_ul <= tolerances['deviation_ul'] and 
                           variability_ul <= tolerances['variation_ul'])
            
            print(f"      📊 Early stop check: {deviation_ul:.2f}μL ≤ {tolerances['deviation_ul']:.2f}μL dev, {variability_ul:.2f}μL ≤ {tolerances['variation_ul']:.2f}μL var → {'✅ STOP' if tolerance_met else '❌ CONTINUE'}")
            
            if tolerance_met:
                print(f"   ✅ Found parameters within tolerance: {adaptive_result['deviation']:.1f}% dev")
                break
                
        except Exception as e:
            print(f"optimization error: {e}")
            break
    
    total_optimization_measurements = get_volume_measurement_count(optimization_start_count)
    print(f"   📊 Optimization complete: {total_optimization_measurements}/{budget} measurements used")
    return optimization_results

def generate_experimental_summary(all_results, optimal_conditions, raw_measurements, timestamp, liquid, autosave_dir):
    """
    Generate comprehensive experimental summary like the modular version.
    Saves to both console and text file.
    """
    
    # Build report lines for both console and file output
    report_lines = []
    report_lines_console = []  # Separate for console (with emojis)
    
    report_lines.append("EXPERIMENT SUMMARY:")  # No emoji for file
    report_lines_console.append("📋 EXPERIMENT SUMMARY:")  # Emoji for console
    
    # Count trials by phase (unique parameter sets)
    screening_trials = len([r for r in all_results if r.get('strategy') == 'SCREENING'])
    optimization_trials = len([r for r in all_results if r.get('strategy', '').startswith('OPTIMIZATION')])
    precision_trials = len([r for r in all_results if r.get('strategy') == 'PRECISION_TEST'])
    
    # Count measurements from raw data by trial_type if available
    total_measurements = len(raw_measurements) if raw_measurements else sum(r.get('replicate_count', 1) for r in all_results)
    
    # Count measurements by phase from raw data
    screening_measurements = 0
    optimization_measurements = 0  
    precision_measurements = 0
    overvolume_measurements = 0
    
    if raw_measurements:
        for measurement in raw_measurements:
            trial_type = measurement.get('trial_type', '').upper()
            if 'SCREENING' in trial_type:
                screening_measurements += 1
            elif 'OPTIMIZATION' in trial_type:
                optimization_measurements += 1
            elif 'PRECISION' in trial_type:
                precision_measurements += 1
            elif 'OVERVOLUME' in trial_type:
                overvolume_measurements += 1
    
    total_trials = len(all_results)
    
    report_lines.append(f"   • Total trials: {total_trials}")
    report_lines.append(f"   • Total measurements: {total_measurements}")
    report_lines.append(f"   • Phase breakdown:")
    report_lines.append(f"     - Screening: {screening_trials} trials, {screening_measurements} measurements")
    report_lines.append(f"     - Optimization: {optimization_trials} trials, {optimization_measurements} measurements") 
    report_lines.append(f"     - Precision tests: {precision_trials} trials, {precision_measurements} measurements")
    if overvolume_measurements > 0:
        report_lines.append(f"     - Overvolume assay: {overvolume_measurements} measurements")
    
    # Copy to console version
    report_lines_console.extend(report_lines[1:])
    
    # Volume completion status - include both success and partial_success as completed
    completed_volumes = [oc for oc in optimal_conditions if oc.get('status') in ['success', 'partial_success']]
    failed_volumes = [oc for oc in optimal_conditions if oc.get('status') == 'failed']
    
    report_lines.append(f"   • Volumes completed: {len(completed_volumes)}/{len(optimal_conditions)}")
    
    for volume_result in optimal_conditions:
        volume_ul = volume_result.get('volume_ul', 0)
        status = volume_result.get('status', 'unknown')
        
        if status in ['success', 'partial_success']:
            # Find performance metrics from results
            volume_ml = volume_result.get('volume_ml', 0)
            volume_trials = [r for r in all_results if r.get('volume') == volume_ml]
            
            # Get best performance - try precision tests first, then fall back to best optimization result
            deviation = None
            time_per_trial = None
            
            # Find precision test results for this volume
            precision_results = [r for r in all_results 
                               if r.get('volume') == volume_ml and r.get('strategy') == 'PRECISION_TEST']
            
            if precision_results:
                # Calculate average performance from precision tests
                avg_deviation = sum(r.get('deviation', 0) for r in precision_results) / len(precision_results)
                avg_time = sum(r.get('time', 0) for r in precision_results) / len(precision_results)
                deviation = f"{avg_deviation:.1f}%"
                time_per_trial = f"{avg_time:.0f}s"
            else:
                # Fallback: use the selected best parameters from optimal_conditions
                # Find the best result that was selected for this volume
                volume_optimization_results = [r for r in all_results 
                                             if r.get('volume') == volume_ml 
                                             and r.get('strategy') in ['SCREENING'] or r.get('strategy', '').startswith('OPTIMIZATION')]
                
                if volume_optimization_results:
                    # Use the parameters that are stored in optimal_conditions (these come from the best ranked candidate)
                    # Find the result that matches the stored parameters
                    best_result = None
                    for param_key in ['aspirate_speed', 'dispense_speed']:  # Check a few key parameters to find match
                        if param_key in volume_result:
                            target_value = volume_result[param_key]
                            for result in volume_optimization_results:
                                if result.get(param_key) == target_value:
                                    best_result = result
                                    break
                            if best_result:
                                break
                    
                    # If we found the matching result, use its performance
                    if best_result:
                        deviation = f"{best_result.get('deviation', 0):.1f}%"
                        time_per_trial = f"{best_result.get('time', 0):.0f}s"
                    else:
                        # Final fallback: use average of all optimization results for this volume
                        avg_deviation = sum(r.get('deviation', 0) for r in volume_optimization_results) / len(volume_optimization_results)
                        avg_time = sum(r.get('time', 0) for r in volume_optimization_results) / len(volume_optimization_results)
                        deviation = f"{avg_deviation:.1f}%"
                        time_per_trial = f"{avg_time:.0f}s"
                else:
                    deviation = "N/A"
                    time_per_trial = "N/A"
            
            if status == 'success':
                report_lines.append(f"     ✅ {volume_ul:.0f}μL: {len(volume_trials)} trials, {deviation} accuracy, {time_per_trial}/trial")
            else:  # partial_success
                report_lines.append(f"     ⚡ {volume_ul:.0f}μL: {len(volume_trials)} trials, {deviation} accuracy, {time_per_trial}/trial (optimized within budget)")
        else:
            volume_trials = [r for r in all_results if r.get('volume') == volume_result.get('volume_ml', 0)]
            report_lines.append(f"     ❌ {volume_ul:.0f}μL: {len(volume_trials)} trials, failed optimization")
    
    # Performance summary for completed volumes
    avg_accuracy = None
    avg_time = None
    
    if completed_volumes:
        # Calculate overall averages from precision test results
        all_precision_results = [r for r in all_results if r.get('strategy') == 'PRECISION_TEST']
        
        if all_precision_results:
            avg_accuracy = sum(r.get('deviation', 0) for r in all_precision_results) / len(all_precision_results)
            avg_time = sum(r.get('time', 0) for r in all_precision_results) / len(all_precision_results)
            report_lines.append(f"   • Overall performance: {avg_accuracy:.1f}% avg accuracy, {avg_time:.0f}s avg time")
        
        # Show hyperparameters used
        report_lines.append(f"   • Hyperparameters:")
        report_lines.append(f"     - Adaptive threshold: {ADAPTIVE_DEVIATION_THRESHOLD}%")
        report_lines.append(f"     - Ranking weights: Acc={ACCURACY_WEIGHT}, Prec={PRECISION_WEIGHT}, Time={TIME_WEIGHT}")
        report_lines.append(f"     - Precision measurements: {PRECISION_MEASUREMENTS}")
        if SIMULATE:
            report_lines.append(f"     - Sim multipliers: Dev={SIM_DEV_MULTIPLIER}x, Var={SIM_VAR_MULTIPLIER}x")
    
    # Timing information
    report_lines.append(f"   • Experiment: {timestamp}")
    report_lines.append(f"   • Liquid: {liquid}")
    report_lines.append(f"   • Results saved: {autosave_dir}")
    
    # Print to console (with emojis)
    print(f"\n" + "\n".join(report_lines_console))
    
    # Save to text file (without emojis to avoid encoding issues)
    report_file_path = os.path.join(autosave_dir, "experiment_summary.txt")
    try:
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
            f.write("\n")
        print(f"📄 Experiment summary saved to: {report_file_path}")
    except Exception as e:
        print(f"⚠️  Could not save experiment summary to file: {e}")
    
    return {
        'total_trials': total_trials,
        'total_measurements': total_measurements,
        'completed_volumes': len(completed_volumes),
        'failed_volumes': len(failed_volumes),
        'avg_accuracy': avg_accuracy if completed_volumes and all_precision_results else None,
        'avg_time': avg_time if completed_volumes and all_precision_results else None,
        'report_file_path': report_file_path
    }

# --- MAIN WORKFLOW ---

def cleanup_robot_and_vials(lash_e, simulate=False):
    """
    Clean up robot state at end of workflow:
    - Return any vial in clamp to home position
    - Remove pipet tip
    - Origin the robot
    """
    try:
        print(f"\n🧹 CLEANUP: Starting robot and vial cleanup...")
        
        if not simulate:
            # Remove any pipet tip
            try:
                print(f"   🗑️  Removing pipet tip...")
                lash_e.nr_robot.remove_pipet()
            except Exception as e:
                print(f"   ⚠️  Warning: Could not remove pipet: {e}")

            # Check if there's a vial in the clamp and return it home
            try:
                clamp_vial = lash_e.nr_robot.get_vial_in_location('clamp', 0)
                if clamp_vial is not None:
                    vial_name = lash_e.nr_robot.get_vial_info(clamp_vial, 'vial_name')
                    print(f"   🏠 Returning vial '{vial_name}' from clamp to home position...")
                    lash_e.nr_robot.return_vial_home(clamp_vial)
                else:
                    print(f"   ✅ No vial in clamp - clamp is clear")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not return clamp vial home: {e}")
            
            # Origin the robot (return to safe home position)
            try:
                print(f"   🏠 Originating robot to home position...")
                lash_e.nr_robot.origin()
                print(f"   ✅ Robot cleanup complete!")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not origin robot: {e}")
        else:
            print(f"   🎭 SIMULATION: Skipping physical robot cleanup")
            print(f"   ✅ Cleanup complete!")
            
    except Exception as e:
        print(f"   ❌ Cleanup failed: {e}")
        print(f"   ⚠️  Manual cleanup may be required!")

def run_simplified_calibration_workflow(vial_mode="legacy", input_vial_status_file=None, **config_overrides):
    """
    Main simplified calibration workflow.
    
    Args:
        vial_mode: Vial management mode ('legacy', 'maintain', 'swap', 'single')
        input_vial_status_file: Path to vial status CSV file (overrides automatic selection)
        **config_overrides: Configuration parameters to override
    """
    
    # Reset config and measurement counter
    reset_config_to_defaults()
    reset_global_measurement_count()
    
    # Select appropriate vial file based on mode (if not explicitly provided)
    if input_vial_status_file is None:
        if vial_mode in ["swap", "maintain"]:
            input_vial_status_file = "status/calibration_vials_overnight.csv"
            print(f"   🧪 Auto-selected vial file for {vial_mode} mode: {input_vial_status_file}")
        else:  # legacy, single
            input_vial_status_file = "status/calibration_vials_short.csv"
            print(f"   🧪 Auto-selected vial file for {vial_mode} mode: {input_vial_status_file}")
    else:
        print(f"   🧪 Using specified vial file: {input_vial_status_file}")
    
    # Override the global vial file setting
    globals()['INPUT_VIAL_STATUS_FILE'] = input_vial_status_file
    
    for key, value in config_overrides.items():
        if key.upper() in globals():
            globals()[key.upper()] = value
            print(f"   🔧 Override: {key} = {value}")

    
    get_current_config_summary()
    
    # Initialize experiment
    lash_e, density_liquid, new_pipet_each_time_set, state = initialize_experiment()
    
    # Set vial management mode with error handling
    if vial_mode != "legacy":
        try:
            set_vial_management(mode=vial_mode)
            print(f"   🧪 Vial management: {vial_mode}")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not set vial management to {vial_mode}: {e}")
            print(f"   🧪 Falling back to legacy vial management")
            vial_mode = "legacy"
    
    # Setup autosave
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simulate_suffix = "_simulate" if SIMULATE else ""
    experiment_name = f"calibration_simplified_{LIQUID}_{timestamp}{simulate_suffix}"
    autosave_dir = os.path.join(BASE_AUTOSAVE_DIR, experiment_name)
    os.makedirs(autosave_dir, exist_ok=True)
    
    autosave_raw_path = os.path.join(autosave_dir, "raw_measurements.csv")
    raw_measurements = []
    all_results = []
    optimal_conditions = []
    
    print(f"📁 Saving results to: {autosave_dir}")
    print(f"🔢 Global limit: {MAX_MEASUREMENTS} total measurements for entire calibration process")
    
    # Global measurement tracking - reference the global variable
    global global_measurement_count
    
    # Process volumes
    successful_params = None
    
    for volume_index, volume in enumerate(VOLUMES):
        # Check global measurement limit before starting each volume
        wells_remaining = MAX_MEASUREMENTS - global_measurement_count
        min_wells_needed = PRECISION_MEASUREMENTS + 1  # At least 1 optimization + precision test
        
        if wells_remaining < min_wells_needed:
            print(f"\n⚠️  SKIPPING volume {volume*1000:.0f}μL: Not enough wells remaining")
            print(f"   Need ≥{min_wells_needed} wells, have {wells_remaining} remaining")
            break
        # HARD BUDGET ENFORCEMENT: Check if we've exceeded the global limit
        if global_measurement_count >= MAX_MEASUREMENTS:
            print(f"\n🛑 HARD BUDGET LIMIT REACHED: {global_measurement_count}/{MAX_MEASUREMENTS} measurements used")
            print(f"   Skipping remaining volumes to enforce strict budget")
            break
            
        print(f"\n{'='*60}")
        print(f"VOLUME {volume_index + 1}/{len(VOLUMES)}: {volume*1000:.0f}μL")
        print(f"📊 Global measurements used: {global_measurement_count}/{MAX_MEASUREMENTS}")
        print(f"{'='*60}")
        
        if volume_index == 0:
            # First volume: full optimization
            success, best_params, best_candidate = optimize_first_volume(
                volume, lash_e, state, autosave_raw_path, raw_measurements,
                LIQUID, new_pipet_each_time_set, all_results
            )
            
            # Always continue with best_params found, regardless of tolerance met
            successful_params = best_params
            
            # Extract performance metrics using the actual ranked best candidate
            performance = extract_performance_metrics(all_results, volume, best_params, raw_measurements, best_candidate)
            
            # Determine status based on success flag
            status = 'success' if success else 'partial_success'
            
            # Extract ONLY the pipetting parameters (filter out scoring junk)
            pipetting_params = {k: v for k, v in best_params.items() if k in ALL_PARAMS}
            
            optimal_conditions.append({
                # Key performance metrics first
                'volume_target': performance['volume_target'],
                'volume_measured': performance['volume_measured'], 
                'average_deviation': performance['average_deviation'],
                'variability': performance['variability'],
                'time': performance['time'],
                
                # Parameters
                **pipetting_params,
                
                # Metadata
                'volume_ml': volume,
                'volume_ul': volume * 1000,
                'status': status,
                
                # Tolerance checks (clear boolean columns) - LAST
                'accuracy_tolerance_met': performance['accuracy_tolerance_met'],
                'precision_tolerance_met': performance['precision_tolerance_met']
            })
            
            if success:
                print(f"✅ VOLUME {volume*1000:.0f}μL COMPLETED SUCCESSFULLY")
            else:
                print(f"🔶 VOLUME {volume*1000:.0f}μL PARTIAL SUCCESS - best effort within tolerance limits")
            
            # Post-optimization overaspirate assay for subsequent volumes (if any)
            if len(VOLUMES) > 1:
                print(f"\n🔬 RUNNING POST-OPTIMIZATION OVERASPIRATE ASSAY...")
                remaining_volumes = VOLUMES[1:]  # Skip first volume
                volume_calibrations = calibrate_overvolume_post_optimization(
                    best_params, remaining_volumes, lash_e, state, autosave_raw_path, 
                    raw_measurements, LIQUID, new_pipet_each_time_set
                )
                
                # Store volume-specific calibrations for subsequent optimization seeding
                global volume_overaspirate_calibrations
                volume_overaspirate_calibrations = volume_calibrations
                print(f"   ✅ Calibrated overaspirate for {len(volume_calibrations)} subsequent volumes")
                
        else:
            # Subsequent volumes: budget-aware optimization
            if successful_params is None:
                print(f"❌ No successful parameters from first volume - cannot continue")
                break
                
            # Calculate measurement budget for this volume
            volumes_remaining = len(VOLUMES) - volume_index
            measurements_budget = calculate_measurements_per_volume(global_measurement_count, volumes_remaining)
            print(f"📊 Budget for this volume: {measurements_budget} measurements")
            
            if measurements_budget < 2:  # Need at least 1 inherited test + 1 optimization
                print(f"⚠️  SKIPPING volume {volume*1000:.0f}μL: Insufficient budget ({measurements_budget} measurements)")
                optimal_conditions.append({
                    # Key performance metrics first (all None for failed)
                    'volume_target': volume * 1000,
                    'volume_measured': None,
                    'average_deviation': None,
                    'variability': None,
                    'time': None,
                    
                    # Tolerance checks (None for failed)
                    'accuracy_tolerance_met': None,
                    'precision_tolerance_met': None,
                    
                    # No parameters for failed volumes
                    
                    # Metadata
                    'volume_ml': volume,
                    'volume_ul': volume * 1000,
                    'status': 'failed',
                    'reason': 'insufficient_budget'
                })
                continue
                
            result = optimize_subsequent_volume_budget_aware(
                volume, lash_e, state, autosave_raw_path, raw_measurements,
                LIQUID, new_pipet_each_time_set, all_results, successful_params, measurements_budget
            )
            success, best_params, status = result
            
            # Extract performance metrics using actual precision test measurements
            performance = extract_performance_metrics(all_results, volume, best_params, raw_measurements)
            
            # Extract ONLY the pipetting parameters (filter out scoring junk)
            pipetting_params = {k: v for k, v in best_params.items() if k in ALL_PARAMS}
            
            optimal_conditions.append({
                # Key performance metrics first
                'volume_target': performance['volume_target'],
                'volume_measured': performance['volume_measured'],
                'average_deviation': performance['average_deviation'], 
                'variability': performance['variability'],
                'time': performance['time'],
                
                # Parameters
                **pipetting_params,
                
                # Metadata
                'volume_ml': volume,
                'volume_ul': volume * 1000,
                'status': status,  # 'success' or 'partial_success'
                
                # Tolerance checks (clear boolean columns) - LAST
                'accuracy_tolerance_met': performance['accuracy_tolerance_met'],
                'precision_tolerance_met': performance['precision_tolerance_met']
            })
            
            if success:
                print(f"✅ VOLUME {volume*1000:.0f}μL COMPLETED SUCCESSFULLY")
            else:
                if status == 'partial_success':
                    print(f"🔶 VOLUME {volume*1000:.0f}μL PARTIAL SUCCESS - best effort within budget")
                else:
                    print(f"❌ VOLUME {volume*1000:.0f}μL FAILED")
        
        # Report actual measurement count (tracked automatically by pipet_and_measure_tracked)
        volume_measurements = len([r for r in all_results if r.get('volume') == volume])
        print(f"📊 Added {volume_measurements} trials for this volume")
        print(f"📊 Total measurements used: {global_measurement_count}/{MAX_MEASUREMENTS}")
        
        # Check if we're approaching the global limit
        if global_measurement_count >= MAX_MEASUREMENTS - PRECISION_MEASUREMENTS:
            print(f"🛑 STOPPING: Approaching global measurement limit ({MAX_MEASUREMENTS})")
            print(f"   Used {global_measurement_count} actual measurements, need {PRECISION_MEASUREMENTS} reserved for precision tests")
            break
    
    # Save final results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    try:
        # Save results dataframes
        results_df = pd.DataFrame(all_results)
        raw_df = pd.DataFrame(raw_measurements)
        optimal_df = pd.DataFrame(optimal_conditions)
        
        # Save to CSV
        results_df.to_csv(os.path.join(autosave_dir, "experiment_summary.csv"), index=False)
        raw_df.to_csv(os.path.join(autosave_dir, "raw_measurements.csv"), index=False)
        optimal_df.to_csv(os.path.join(autosave_dir, "optimal_conditions.csv"), index=False)
        
        # Save configuration with all hyperparameters
        config_summary = {
            'liquid': LIQUID,
            'simulate': SIMULATE,
            'volumes': VOLUMES,
            'max_measurements': MAX_MEASUREMENTS,
            'min_good_parameter_sets': MIN_GOOD_PARAMETER_SETS,
            'precision_measurements': PRECISION_MEASUREMENTS,
            'adaptive_deviation_threshold': ADAPTIVE_DEVIATION_THRESHOLD,
            'adaptive_penalty_variability': ADAPTIVE_PENALTY_VARIABILITY,
            'accuracy_weight': ACCURACY_WEIGHT,
            'precision_weight': PRECISION_WEIGHT,
            'time_weight': TIME_WEIGHT,
            'sim_dev_multiplier': SIM_DEV_MULTIPLIER,
            'sim_var_multiplier': SIM_VAR_MULTIPLIER,
            'vial_mode': vial_mode,
            'timestamp': timestamp,
            'workflow_type': 'simplified',
            **config_overrides
        }
        
        with open(os.path.join(autosave_dir, "experiment_config.yaml"), 'w') as f:
            yaml.dump(config_summary, f, default_flow_style=False)
        
        print(f"✅ Results saved to: {autosave_dir}")
        print(f"   📊 {len(all_results)} total trials")
        print(f"   🎯 {len([c for c in optimal_conditions if c['status'] == 'success'])} successful volumes")
        
        # Generate experimental summary
        summary_stats = generate_experimental_summary(
            all_results, optimal_conditions, raw_measurements, 
            timestamp, LIQUID, autosave_dir
        )
        
        # Optional: Run analysis if available
        if base_module.ANALYZER_AVAILABLE:
            print(f"📈 Running analysis...")
            save_analysis(results_df, raw_df, autosave_dir, 
                         include_shap=True, include_scatter=True,
                         optimal_conditions=optimal_conditions)
        
    except Exception as e:
        print(f"⚠️  Error saving results: {e}")
    
    # Send completion Slack message (only for real experiments)
    if not SIMULATE and SLACK_AVAILABLE:
        try:
            # Calculate summary stats
            successful_vols = [c for c in optimal_conditions if c['status'] == 'success']
            partial_vols = [c for c in optimal_conditions if c['status'] == 'partial_success']
            total_vols = len(optimal_conditions)
            success_rate = (len(successful_vols) + len(partial_vols)) / total_vols * 100 if total_vols > 0 else 0
            
            # Build results table from optimal_conditions
            results_table = "📊 CALIBRATION RESULTS:\n"
            results_table += "Volume(μL) | Measured(μL) | Deviation(%) | Variability(%) | Time(s)\n"
            results_table += "-----------|--------------|--------------|----------------|--------\n"
            
            for condition in optimal_conditions:
                vol_target = condition.get('volume_ul', condition.get('volume_target', 'N/A'))
                vol_measured = condition.get('volume_measured', 'N/A')
                deviation = condition.get('average_deviation', 'N/A')
                variability = condition.get('variability', 'N/A')  
                time_val = condition.get('time', 'N/A')
                
                # Format values
                if vol_measured != 'N/A':
                    vol_measured = f"{vol_measured:.2f}"
                if deviation != 'N/A':
                    deviation = f"{deviation:.2f}"
                if variability != 'N/A':
                    variability = f"{variability:.2f}"
                if time_val != 'N/A':
                    time_val = f"{time_val:.2f}"
                    
                results_table += f"{vol_target:>9} | {vol_measured:>11} | {deviation:>11} | {variability:>13} | {time_val:>6}\n"
            
            slack_msg = (
                f"🎯 Simplified calibration with {LIQUID.upper()} FINISHED\n"
                f"✅ Success rate: {success_rate:.1f}% ({len(successful_vols + partial_vols)}/{total_vols} volumes)\n"
                f"🎯 Completed: {len(successful_vols)} success, {len(partial_vols)} partial\n\n"
                f"{results_table}"
            )
            slack_agent.send_slack_message(slack_msg)
        except Exception as e:
            print(f"Warning: Failed to send completion Slack message: {e}")
    
    # CLEANUP: Return vials to home, remove pipet, origin robot
    cleanup_robot_and_vials(lash_e, simulate=SIMULATE)
    
    print(f"\n🎉 SIMPLIFIED CALIBRATION WORKFLOW COMPLETE!")
    return optimal_conditions, autosave_dir

# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    # Example 1: Traditional calibration with screening (no external data)
    # print("🎯 SIMPLIFIED CALIBRATION WORKFLOW - TRADITIONAL MODE")
    # print("   Running standard calibration with screening phase\n")
    
    # optimal_conditions, save_dir = run_simplified_calibration_workflow(
    #     vial_mode="legacy",
    #     liquid="glycerol",
    #     simulate=True,
    #     volumes=[0.05, 0.025, 0.1],  # Test with 3 volumes
    #     use_llm_for_screening=True
    # )
    
    # Example 2: Fixed parameters for water - fast parameters
    # print("\n⚡ FIXED PARAMETERS EXPERIMENT - Water with fast parameters")
    # print("   Fixing wait times to 0 and post-aspirate air volume for water\n")
    
    # optimal_conditions_water, save_dir_water = run_simplified_calibration_workflow(
    #     vial_mode="legacy",
    #     liquid="water",
    #     simulate=True,
    #     volumes=[0.05, 0.025, 0.1],  # Test with 3 volumes
    #     use_LLM_for_screening = True,
    #     # Fix timing parameters for speed and post-aspirate air volume
    #     fixed_parameters={
    #         'post_asp_air_vol': 0.05
    #     }
    # )
    
    # Example 3: Fixed parameters for glycerol - just post-aspirate air volume
    print("\n🔧 FIXED PARAMETERS EXPERIMENT - Glycerol with fixed air volume")
    print("   Fixing only post-aspirate air volume for glycerol\n")
    
    try:
        optimal_conditions_water, save_dir_water = run_simplified_calibration_workflow(
            vial_mode="swap",
            liquid="water",
            simulate=False,
            use_LLM_for_screening=True,
            # volumes=[0.05, 0.025, 0.1],  # Test with 3 volumes
            #volumes=[0.05, 0.025, 0.1, 0.01, 0.005, 0.2, 0.5, 0.8],
            volumes=[0.01, 0.025, 0.005],
            min_good_parameter_sets=5,  # Instead of 6
            precision_measurements=5,    # Instead of 3 replicates
            max_measurements=250,        # Instead of 96 total trials
            first_volume_max_measurements=150,  # Max for first volume
            # No fixed_parameters for full optimization
        )
        print(f"\n✅ Workflow completed successfully!")
        print(f"📊 Results saved to: {save_dir_water}")
        
    except Exception as e:
        print(f"\n❌ WORKFLOW ERROR: {e}")
        print(f"🧹 Attempting emergency cleanup...")
        
        # Emergency cleanup - create minimal lash_e for cleanup if needed
        try:
            from master_usdl_coordinator import Lash_E
            emergency_lash_e = Lash_E("status/calibration_vials_short.csv", simulate=False, initialize_biotek=False)
            cleanup_robot_and_vials(emergency_lash_e, simulate=False)
        except Exception as cleanup_error:
            print(f"⚠️  Emergency cleanup failed: {cleanup_error}")
            print(f"🔧 Manual robot cleanup may be required!")
        
        raise  # Re-raise the original error
    
    # Example 4: Hot start experiment using unified dataset
    # print("\n🔥 HOT START EXPERIMENT - Using unified dataset for faster convergence")
    # print("   This experiment loads prior knowledge from previous calibration data\n")
    
    # optimal_conditions_hot, save_dir_hot = run_simplified_calibration_workflow(
    #     vial_mode="legacy",
    #     liquid="water",  # Match the dataset
    #     simulate=True,
    #     volumes=[0.05, 0.025, 0.1],  # Test with 3 volumes
    #     # Hot start configuration - load prior knowledge
    #     use_external_data=True,
    #     external_data_path="pipetting_data/unified_dataset_water.csv",
    #     external_data_liquid_filter="water"
    #     # This should converge much faster than cold start!
    # )

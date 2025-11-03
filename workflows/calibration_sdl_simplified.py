# calibration_sdl_simplified.py
"""
Simplified calibration workflow that eliminates dynamic cutoff and cascading precision tests.

Workflow:
1. Screening (SOBOL/LLM exploration) 
2. Overaspirate calibration (same as modular)
3. 3-objective optimization (deviation, variability, time)
4. Simple stopping: 60 measurements OR 6 "GOOD" parameter sets
5. Best candidate selection: rank by accuracy → precision → time
6. Single precision test

First volume: optimize all parameters
Subsequent volumes: selective optimization (volume-dependent parameters only)
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
DEFAULT_OVERVOLUME_CALIBRATION_BUFFER_UL = 2.0
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

# Global measurement counter and volume-specific calibrations
global_measurement_count = 0
volume_overaspirate_calibrations = {}  # Store volume-specific overaspirate calibrations

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

def extract_performance_metrics(all_results, volume_ml, best_params, raw_measurements=None):
    """
    Extract key performance metrics for a volume from actual precision test measurements.
    
    Returns dict with volume_target, volume_measured, average_deviation, variability, time,
    and tolerance check results.
    """
    # Find all results for this volume
    volume_results = [r for r in all_results if r.get('volume') == volume_ml]
    
    if not volume_results:
        return {
            'volume_target': volume_ml * 1000,  # Convert to μL
            'volume_measured': None,
            'average_deviation': None,
            'variability': None,
            'time': None,
            'accuracy_tolerance_met': None,
            'precision_tolerance_met': None
        }
    
    # Find the best result that matches the selected parameters
    best_result = None
    for result in volume_results:
        # Check if this result matches the selected parameters (compare a few key params)
        matches = all(result.get(param) == best_params.get(param) 
                     for param in ['aspirate_speed', 'dispense_speed', 'overaspirate_vol'] 
                     if param in best_params and param in result)
        if matches:
            best_result = result
            break
    
    # Fallback: use the result with best deviation if no exact match
    if best_result is None and volume_results:
        best_result = min(volume_results, key=lambda r: r.get('deviation', float('inf')))
    
    if best_result is None:
        return {
            'volume_target': volume_ml * 1000,
            'volume_measured': None,
            'average_deviation': None,
            'variability': None,
            'time': None,
            'accuracy_tolerance_met': None,
            'precision_tolerance_met': None
        }
    
    # CRITICAL FIX: Use actual precision test measurements, not calculated estimates!
    target_ul = volume_ml * 1000
    measured_ul = None
    actual_deviation_pct = None
    actual_variability = None
    
    # Find precision test measurements from raw_measurements
    if raw_measurements:
        # Look for precision test measurements for this volume
        # These could be INHERITED_TEST, PRECISION, or similar trial types
        precision_measurements = [
            m for m in raw_measurements 
            if (m.get('volume') == volume_ml and 
                m.get('trial_type') in ['INHERITED_TEST', 'PRECISION', 'PRECISION_TEST'])
        ]
        
        if precision_measurements:
            # Extract actual measured volumes from precision tests
            measured_volumes_ml = [m.get('calculated_volume', 0) for m in precision_measurements]
            measured_volumes_ul = [v * 1000 for v in measured_volumes_ml]  # Convert to μL
            
            # Calculate actual statistics from real measurements
            measured_ul = sum(measured_volumes_ul) / len(measured_volumes_ul)  # Average
            
            # Calculate actual deviation from target
            actual_deviation_pct = abs(measured_ul - target_ul) / target_ul * 100
            
            # Calculate actual variability (CV as percentage)
            if len(measured_volumes_ul) > 1:
                std_ul = np.std(measured_volumes_ul)
                actual_variability = (std_ul / measured_ul) * 100
            else:
                actual_variability = 0
    
    # Fallback to optimization result estimates if no precision data found
    if measured_ul is None:
        deviation_pct = best_result.get('deviation', 0)
        measured_ul = target_ul * (1 - deviation_pct / 100)  # Old calculation as fallback
        actual_deviation_pct = deviation_pct
        actual_variability = best_result.get('variability', None)
    
    # Calculate tolerance checks using actual measurements
    tolerances = get_volume_dependent_tolerances(volume_ml)
    
    # Check accuracy tolerance using actual deviation
    if actual_deviation_pct is not None:
        deviation_ul = (actual_deviation_pct / 100.0) * target_ul
        accuracy_tolerance_met = deviation_ul <= tolerances['deviation_ul']
    else:
        accuracy_tolerance_met = None
    
    # Check precision tolerance using actual variability
    if actual_variability is not None and actual_variability != ADAPTIVE_PENALTY_VARIABILITY:
        # Convert variability percentage to μL
        variability_ul = (actual_variability / 100.0) * target_ul
        precision_tolerance_met = variability_ul <= tolerances['variation_ul']
    else:
        precision_tolerance_met = False  # No valid precision data or penalty value
    
    return {
        'volume_target': target_ul,
        'volume_measured': measured_ul,
        'average_deviation': actual_deviation_pct,
        'variability': actual_variability,
        'time': best_result.get('time', None),
        'accuracy_tolerance_met': accuracy_tolerance_met,
        'precision_tolerance_met': precision_tolerance_met
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
    print(f"   Adaptive threshold: {ADAPTIVE_DEVIATION_THRESHOLD}% deviation")
    print(f"   Ranking weights: Acc={ACCURACY_WEIGHT}, Prec={PRECISION_WEIGHT}, Time={TIME_WEIGHT}")
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
    state = {"measurement_vial_index": 0, "measurement_vial_name": "measurement_vial_0"}

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
    
    all_deviations.append(initial_deviation)
    all_times.append(initial_time)
    
    # Extract actual measurement volume for variability calculation
    if raw_measurements:
        actual_mass = raw_measurements[-1]['mass']
        liquid_density = LIQUIDS[liquid]["density"]
        actual_volume = actual_mass / liquid_density
        all_measurements.append(actual_volume)
    else:
        # Fallback for simulation
        liquid_density = LIQUIDS[liquid]["density"]
        actual_volume = expected_mass / liquid_density
        all_measurements.append(actual_volume)
    
    print(f"{initial_deviation:.1f}% deviation")
    
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
            check_if_measurement_vial_full(lash_e, {"measurement_vial_name": measurement_vial})
            
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
            
            all_deviations.append(deviation)
            all_times.append(time_taken)
            
            # Extract measurement volume
            if raw_measurements:
                actual_mass = raw_measurements[-1]['mass']
                actual_volume = actual_mass / liquid_density
                all_measurements.append(actual_volume)
            else:
                actual_volume = expected_mass / liquid_density
                all_measurements.append(actual_volume)
            
            print(f"{deviation:.1f}% dev")
        
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
            variability = (max_vol - min_vol) / (2 * avg_vol) * 100
            
            print(f"      📊 Final averages: {avg_deviation:.1f}% dev, {variability:.1f}% var, {avg_time:.1f}s")
        else:
            variability = 0.0  # Single measurement somehow
    
    return {
        'deviation': avg_deviation,
        'variability': variability,
        'time': avg_time,
        'replicate_count': replicate_count,
        'all_measurements': all_measurements,
        'all_deviations': all_deviations,
        'all_times': all_times
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
    
    # Evaluate each candidate and collect raw metrics
    evaluated_candidates = []
    raw_accuracies = []
    raw_precisions = []
    raw_times = []
    
    for candidate in candidates:
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
    
    # Calculate normalization ranges from actual data
    acc_min, acc_max = min(raw_accuracies), max(raw_accuracies)
    prec_min, prec_max = min(raw_precisions), max(raw_precisions)
    time_min, time_max = min(raw_times), max(raw_times)
    
    # Ensure minimum ranges to prevent division by zero
    acc_range = max(acc_max - acc_min, 0.1)
    prec_range = max(prec_max - prec_min, 0.1)
    time_range = max(time_max - time_min, 1.0)
    
    # Calculate normalized scores and composite scores
    for candidate in evaluated_candidates:
        # Normalize to 0-100 scale (0 = best performer, 100 = worst performer)
        acc_score = (candidate['raw_accuracy'] - acc_min) / acc_range * 100
        prec_score = (candidate['raw_precision'] - prec_min) / prec_range * 100  
        time_score = (candidate['raw_time'] - time_min) / time_range * 100
        
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
    print(f"   Data ranges: Acc {acc_min:.1f}-{acc_max:.1f}%, Prec {prec_min:.1f}-{prec_max:.1f}%, Time {time_min:.1f}-{time_max:.1f}s")
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
            'reason': f'Need {MIN_GOOD_PARAMETER_SETS - good_parameter_sets} more good parameter sets or {MAX_MEASUREMENTS - total_measurements} more measurements',
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
    optimizer = llm_opt.LLMOptimizer()
    config = optimizer.load_config(config_path)
    
    # Update config with current experimental context
    if volume is not None:
        config['experimental_setup']['target_volume_ul'] = volume * 1000  # Convert mL to μL
        print(f"   📏 Target volume: {volume*1000:.0f}μL")
    
    if liquid is not None:
        config['experimental_setup']['current_liquid'] = liquid
        # For initial exploration, add liquid context to system message if available in config
        if not all_results and liquid.lower() in config.get('material_properties', {}):
            liquid_info = config['material_properties'][liquid.lower()]
            liquid_context = f"LIQUID CONTEXT: You are optimizing for {liquid.upper()}. {liquid_info.get('exploration_notes', '')} {liquid_info.get('recommended_focus', '')}"
            
            # Insert liquid-specific guidance at the beginning of system message
            original_message = config['system_message']
            enhanced_message = [liquid_context, ""] + original_message
            config['system_message'] = enhanced_message
            print(f"   🧪 Enhanced config for {liquid}")
        elif liquid.lower() not in config.get('material_properties', {}):
            print(f"   ⚠️  No material properties found for {liquid} in config")
    
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
        empty_df = pd.DataFrame(columns=['liquid', 'aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 
                                        'dispense_wait_time', 'retract_speed', 'blowout_vol', 
                                        'post_asp_air_vol', 'overaspirate_vol', 'deviation', 'time', 'trial_type'])
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
            
            # Apply proper type casting based on Ax search space parameter types
            for k, v in llm_params.items():
                if k in expected_params:
                    ax_param = ax_client.experiment.search_space.parameters[k]
                    if hasattr(ax_param, 'parameter_type'):
                        # Cast to appropriate type based on Ax parameter type
                        if ax_param.parameter_type.name == 'INT':
                            filtered_params[k] = int(round(float(v)))
                        elif ax_param.parameter_type.name == 'FLOAT':
                            filtered_params[k] = float(v)
                        else:
                            filtered_params[k] = v
                    else:
                        filtered_params[k] = v
            
            print(f"   LLM suggestion {i+1}: {filtered_params}")
            params, trial_index = ax_client.attach_trial(filtered_params)
            suggestions.append((params, trial_index))
    return suggestions

# --- SCREENING PHASE (REUSE FROM MODULAR) ---

def run_screening_phase(ax_client, lash_e, state, volume, expected_mass, expected_time, 
                       autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set):
    """Run initial screening phase with SOBOL or LLM suggestions."""
    
    print(f"\n🔍 SCREENING PHASE: {INITIAL_PARAMETER_SETS} initial parameter sets...")
    
    screening_results = []
    
    for i in range(INITIAL_PARAMETER_SETS):
        print(f"   Screening trial {i+1}/{INITIAL_PARAMETER_SETS}...")
        
        if USE_LLM_FOR_SCREENING and LLM_AVAILABLE:
            # Get LLM suggestion
            suggestions = get_llm_suggestions(ax_client, 1, screening_results, volume, liquid)
            if suggestions:
                params, trial_index = suggestions[0]
            else:
                # Fallback to Ax suggestion
                params, trial_index = ax_client.get_next_trial()
        else:
            # Get Ax suggestion (SOBOL)
            params, trial_index = ax_client.get_next_trial()
        
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
            "raw_measurements": adaptive_result['all_measurements']
        })
        screening_results.append(full_result)
        
        print(f"      → {adaptive_result['deviation']:.1f}% deviation, {adaptive_result['variability']:.1f}% variability, {adaptive_result['time']:.1f}s ({adaptive_result['replicate_count']} reps)")
    
    print(f"   ✅ Screening complete: {len(screening_results)} trials")
    return screening_results

# --- OVERASPIRATE CALIBRATION (REUSE FROM MODULAR) ---

def calculate_first_volume_constraint(best_candidate, volume):
    """
    Calculate overaspirate constraint for first volume based on screening shortfall.
    
    Args:
        best_candidate: Best screening candidate with deviation and parameters
        volume: Target volume in mL
        
    Returns:
        max_overaspirate_ml: Upper constraint for overaspirate_vol parameter
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
    # Note: shortfall can be negative (over-delivery), which would reduce total overaspirate needed
    max_overaspirate_ul = existing_overaspirate_ul + shortfall_ul + OVERVOLUME_CALIBRATION_BUFFER_UL
    
    # Ensure minimum constraint (prevent negative overaspirate)
    min_overaspirate_ul = 1.0  # Minimum 1μL overaspirate
    max_overaspirate_ul = max(max_overaspirate_ul, min_overaspirate_ul)
    max_overaspirate_ml = max_overaspirate_ul / 1000  # Convert back to mL
    
    print(f"   📊 First volume constraint calculation:")
    if shortfall_ul >= 0:
        print(f"     Target: {target_volume_ul:.1f}μL, Measured: {measured_volume_ul:.1f}μL → Under-delivery: {shortfall_ul:.1f}μL")
    else:
        print(f"     Target: {target_volume_ul:.1f}μL, Measured: {measured_volume_ul:.1f}μL → Over-delivery: {abs(shortfall_ul):.1f}μL")
    print(f"     Existing overaspirate: {existing_overaspirate_ul:.1f}μL")
    print(f"     Buffer: {OVERVOLUME_CALIBRATION_BUFFER_UL:.1f}μL")
    print(f"     → Max overaspirate constraint: {max_overaspirate_ul:.1f}μL ({max_overaspirate_ml:.4f}mL)")
    
    return max_overaspirate_ml

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
        
        # Single measurement using optimized parameters
        result = pipet_and_measure_tracked(lash_e, liquid_source, state["measurement_vial_name"], 
                                          volume, optimized_params, expected_mass, expected_time, 
                                          1, SIMULATE, autosave_raw_path, raw_measurements, 
                                          liquid, new_pipet_each_time_set, "POST_OPT_OVERVOLUME_ASSAY")
        
        # Get actual measured volume from raw_measurements
        if raw_measurements:
            actual_mass = raw_measurements[-1]['mass']
            actual_volume_ml = actual_mass / LIQUIDS[liquid]["density"]
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
    
    # Fallback to default source
    return "liquid_source_0"

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
    
    print(f"   Target tolerances: ±{tolerances['deviation_ul']:.1f}μL deviation, {tolerances['tolerance_percent']:.1f}% precision")
    
    # Check if optimizer is available
    if not OPTIMIZER_3OBJ_AVAILABLE:
        print("❌ 3-objectives optimizer not available - cannot proceed with first volume optimization")
        return False, None
    
    # Create 3-objective optimizer for all parameters
    ax_client = optimizer_3obj.create_model(
        seed=SEED,
        num_initial_recs=INITIAL_PARAMETER_SETS,
        bayesian_batch_size=PARAMETER_SETS_PER_RECOMMENDATION,
        volume=volume,
        tip_volume=tip_volume,
        model_type=BAYESIAN_MODEL_TYPE,
        optimize_params=ALL_PARAMS,  # Optimize all parameters for first volume
        fixed_params={},
        simulate=SIMULATE,
        max_overaspirate_ul=max_overaspirate_ul
    )
    
    # Phase 1: Screening
    screening_results = run_screening_phase(ax_client, lash_e, state, volume, expected_mass, 
                                          expected_time, autosave_raw_path, raw_measurements, 
                                          liquid, new_pipet_each_time_set)
    all_results.extend(screening_results)
    
    # Phase 2: Calculate first volume constraint based on screening shortfall
    print(f"\n🎯 CALCULATING FIRST VOLUME CONSTRAINT...")
    
    # Select best screening candidate using ranking system
    ranked_candidates = rank_candidates_by_priority(screening_results, volume, tolerances)
    best_candidate = ranked_candidates[0] if ranked_candidates else screening_results[0]
    print(f"   🏆 Selected best screening candidate: {best_candidate.get('deviation', 0):.1f}% deviation")
    
    max_overaspirate_ml_updated = calculate_first_volume_constraint(best_candidate, volume)
    max_overaspirate_ul_updated = max_overaspirate_ml_updated * 1000  # Convert to μL for display and optimizer
    print(f"   ✅ Updated max overaspirate constraint: {max_overaspirate_ul_updated:.1f}μL")
    print(f"   🔍 DEBUG: mL value = {max_overaspirate_ml_updated:.6f}mL, μL value = {max_overaspirate_ul_updated:.6f}μL")
    
    # Recreate ax_client with updated constraint if it changed  
    if abs(max_overaspirate_ul_updated - max_overaspirate_ul) > 0.01:  # Only recreate if meaningful change (compare in μL)
        print(f"   🔄 Recreating optimizer with updated constraint ({max_overaspirate_ul:.1f} → {max_overaspirate_ul_updated:.1f}μL)...")
        ax_client = optimizer_3obj.create_model(
            seed=SEED,
            num_initial_recs=0,  # No initial SOBOL since we already have screening data
            bayesian_batch_size=PARAMETER_SETS_PER_RECOMMENDATION,
            volume=volume,
            tip_volume=tip_volume,
            model_type=BAYESIAN_MODEL_TYPE,
            optimize_params=ALL_PARAMS,
            fixed_params={},
            simulate=SIMULATE,
            max_overaspirate_ul=max_overaspirate_ul_updated  # Pass in μL as expected
        )
        
        # Load screening results into the new optimizer
        optimizer_3obj.load_previous_data_into_model(ax_client, screening_results)
        print(f"   ✅ Loaded {len(screening_results)} screening results into updated optimizer")
    else:
        print(f"   ✅ Constraint unchanged, keeping existing optimizer")
    
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
        params, trial_index = optimizer_3obj.get_suggestions(ax_client, volume, n=1)[0]
        optimization_trial_count += 1
        
        print(f"   Optimization trial {optimization_trial_count}...")
        
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
            "raw_measurements": adaptive_result['all_measurements']
        })
        all_results.append(full_result)
        
        quality = evaluate_trial_quality(full_result, volume, tolerances)
        quality_status = "✅ GOOD" if quality['is_good'] else "❌ needs improvement"
        print(f"      → {adaptive_result['deviation']:.1f}% dev, {adaptive_result['variability']:.1f}% var, {adaptive_result['time']:.1f}s ({quality_status}) [{adaptive_result['replicate_count']} reps]")
    
    # Phase 4: Select best candidate and run precision test
    print(f"\n🏆 SELECTING BEST CANDIDATE...")
    
    # Get all first volume trials (screening + optimization) for ranking
    first_volume_trials = [r for r in all_results 
                          if r.get('volume') == volume 
                          and r.get('strategy') in ['SCREENING'] or r.get('strategy', '').startswith('OPTIMIZATION')]
    
    if not first_volume_trials:
        print("   ❌ No trials found for ranking!")
        return False, None
    
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
        return True, best_params
    else:
        print(f"\n🔶 FIRST VOLUME PARTIAL SUCCESS!")
        print(f"   Best parameters found but do not meet strict tolerance")
        print(f"   Accuracy: {quality['accuracy_deviation_ul']:.2f}μL > {quality['accuracy_tolerance_ul']:.2f}μL tolerance")
        print(f"   Parameters will still be used as baseline for subsequent volumes")
        return False, best_params  # Return False to indicate partial success


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
    volume_ul = volume * 1000  # Convert to μL for lookup
    if volume_ul in volume_overaspirate_calibrations:
        old_overaspirate = test_params.get('overaspirate_vol', 0) * 1000  # Convert to μL
        new_overaspirate_ul = volume_overaspirate_calibrations[volume_ul]['guess_overaspirate_ul']
        test_params['overaspirate_vol'] = new_overaspirate_ul / 1000  # Convert back to mL
        print(f"   🎯 Using calibrated overaspirate guess: {old_overaspirate:.1f}μL → {new_overaspirate_ul:.1f}μL")
    else:
        print(f"   🎯 Using inherited overaspirate: {test_params.get('overaspirate_vol', 0)*1000:.1f}μL")

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
            
            check_if_measurement_vial_full(lash_e, {"measurement_vial_name": state["measurement_vial_name"]})
            
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
        
        if len(all_measurements) > 1:
            volume_std = np.std(all_measurements)
            variability = volume_std / np.mean(all_measurements) * 100
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
        inherited_comprehensive_result = {
            'volume': volume,
            'deviation': deviation,
            'time': inherited_result.get('time', 0),
            'variability': ADAPTIVE_PENALTY_VARIABILITY,
            'replicate_count': 1,
            'strategy': 'INHERITED_TEST',
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
    volume_ul = volume * 1000  # Convert to μL for lookup
    if volume_ul in volume_overaspirate_calibrations:
        max_overaspirate_ul = volume_overaspirate_calibrations[volume_ul]['max_overaspirate_ul']
        print(f"   🎯 Using calibrated overaspirate constraint: {max_overaspirate_ul:.1f}μL")
    else:
        max_overaspirate_ul = get_max_overaspirate_ul(volume)
        print(f"   🎯 Using default overaspirate constraint: {max_overaspirate_ul:.1f}μL")
    
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

def run_simplified_calibration_workflow(vial_mode="legacy", **config_overrides):
    """
    Main simplified calibration workflow.
    
    Args:
        vial_mode: Vial management mode ('legacy', 'maintain', 'swap', 'single')
        **config_overrides: Configuration parameters to override
    """
    
    # Reset config and measurement counter
    reset_config_to_defaults()
    reset_global_measurement_count()
    
    for key, value in config_overrides.items():
        if key.upper() in globals():
            globals()[key.upper()] = value
            print(f"   🔧 Override: {key} = {value}")

    
    get_current_config_summary()
    
    # Initialize experiment
    lash_e, density_liquid, new_pipet_each_time_set, state = initialize_experiment()
    
    # Set vial management mode
    if vial_mode != "legacy":
        set_vial_management(mode=vial_mode)
        print(f"   🧪 Vial management: {vial_mode}")
    
    # Setup autosave
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"calibration_simplified_{LIQUID}_{timestamp}"
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
            success, best_params = optimize_first_volume(
                volume, lash_e, state, autosave_raw_path, raw_measurements,
                LIQUID, new_pipet_each_time_set, all_results
            )
            
            # Always continue with best_params found, regardless of tolerance met
            successful_params = best_params
            
            # Extract performance metrics using actual precision test measurements
            performance = extract_performance_metrics(all_results, volume, best_params, raw_measurements)
            
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
                
            success, best_params, status = optimize_subsequent_volume_budget_aware(
                volume, lash_e, state, autosave_raw_path, raw_measurements,
                LIQUID, new_pipet_each_time_set, all_results, successful_params, measurements_budget
            )
            
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
    
    print(f"\n🎉 SIMPLIFIED CALIBRATION WORKFLOW COMPLETE!")
    return optimal_conditions, autosave_dir

# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    # Single calibration experiment with moderate tolerance
    print("🎯 SIMPLIFIED CALIBRATION WORKFLOW")
    print("   Running with 1.5x tolerance multiplier (moderate challenge)")
    print("   Testing deterministic budget allocation with 3 volumes\n")
    
    optimal_conditions, save_dir = run_simplified_calibration_workflow(
        vial_mode="legacy",
        liquid="glycerol",
        simulate=False,
        volumes=[0.05, 0.025, 0.1]  # Test with 3 volumes
    )
    
    # Analyze results
    successful_volumes = [c for c in optimal_conditions if c['status'] == 'success']
    partial_volumes = [c for c in optimal_conditions if c['status'] == 'partial_success']
    failed_volumes = [c for c in optimal_conditions if c['status'] == 'failed']
    
    print(f"\n📊 EXPERIMENT RESULTS:")
    print(f"   🎯 Success Rate: {len(successful_volumes + partial_volumes)}/{len(optimal_conditions)} volumes")
    print(f"   ✅ Successful: {len(successful_volumes)} volumes")
    print(f"   ⚡ Partial Success: {len(partial_volumes)} volumes") 
    print(f"   ❌ Failed: {len(failed_volumes)} volumes")
    print(f"   📁 Results saved to: {save_dir}")
    
    overall_success_rate = (len(successful_volumes) + len(partial_volumes)) / len(optimal_conditions) * 100
    print(f"   📈 Overall Success Rate: {overall_success_rate:.1f}%")

# End of simplified calibration workflow
"""
                'error': str(e),
                'success_rate': 0.0
            }
            all_experiment_results.append(experiment_summary)
    
    # Final comprehensive analysis
    print(f"\n{'='*80}")
    print("🏆 CALIBRATION ACCURACY CHALLENGE COMPLETE!")
    print(f"{'='*80}")
    
    print(f"\n� EXPERIMENT SERIES SUMMARY:")
    print(f"{'Exp':<4} {'Name':<12} {'Dev/Var':<8} {'Success':<8} {'Partial':<8} {'Failed':<8} {'Rate':<8}")
    print(f"{'-'*60}")
    
    for result in all_experiment_results:
        if 'error' not in result:
            print(f"{result['experiment']:<4} {result['name']:<12} {result['dev_multiplier']:<8} "
                  f"{result['successful_volumes']:<8} {result['partial_volumes']:<8} "
                  f"{result['failed_volumes']:<8} {result['success_rate']:<8.1f}%")
        else:
            print(f"{result['experiment']:<4} {result['name']:<12} {result['dev_multiplier']:<8} ERROR")
    
    # Trend analysis
    successful_experiments = [r for r in all_experiment_results if 'error' not in r]
    if len(successful_experiments) > 1:
        print(f"\n🔍 TREND ANALYSIS:")
        
        # Success rate trend
        rates = [r['success_rate'] for r in successful_experiments]
        multipliers = [r['dev_multiplier'] for r in successful_experiments]
        
        print(f"   📈 Success Rate Trend:")
        for i, (mult, rate) in enumerate(zip(multipliers, rates)):
            trend = ""
            if i > 0:
                change = rate - rates[i-1]
                trend = f" ({change:+.1f}%)"
            print(f"      {mult}x multiplier: {rate:.1f}% success{trend}")
        
        # Demonstrate system resilience
        min_success_rate = min(rates)
        print(f"\n🛡️  SYSTEM RESILIENCE DEMONSTRATED:")
        print(f"   • Minimum success rate: {min_success_rate:.1f}% (even under extreme conditions)")
        print(f"   • Budget allocation ensures completion across all tolerance levels")
        print(f"   • Deterministic resource management prevents workflow failure")
        
        if min_success_rate > 0:
            print(f"   ⭐ SUCCESS: System completed calibration under all tested conditions!")
        else:
            print(f"   ⚠️  Some conditions caused complete failure - system limits reached")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   • Deterministic budget allocation ensures workflow completion")
    print(f"   • Success rates may decrease with stricter tolerances (expected)")
    print(f"   • System gracefully degrades to 'partial_success' under extreme conditions")
    print(f"   • Resource management prevents measurement budget exhaustion")
    print(f"   \n📝 This demonstrates the robustness of our budget-aware optimization approach!")
"""
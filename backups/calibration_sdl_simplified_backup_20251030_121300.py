# calibration_sdl_simplified.py
"""
Simplified calibration workflow that eliminates dynamic cutoff and cascading precision tests.

Workflow:
1. Screening (SOBOL/LLM exploration) 
2. Overaspirate calibration (same as modular)
3. 3-objective optimization (deviation, variability, time)
4. Simple stopping: 60 trials OR 6 "GOOD" trials
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
DEFAULT_INITIAL_SUGGESTIONS = 5
DEFAULT_BATCH_SIZE = 1
DEFAULT_REPLICATES = 1
DEFAULT_PRECISION_REPLICATES = 4
DEFAULT_VOLUMES = [0.05, 0.025, 0.01]  # mL
DEFAULT_MAX_WELLS = 96
DEFAULT_INPUT_VIAL_STATUS_FILE = "status/calibration_vials_short.csv"

# Simplified stopping criteria
DEFAULT_MAX_MEASUREMENTS = 60  # Maximum individual measurements before stopping
DEFAULT_MIN_GOOD_TRIALS = 6  # Minimum "good" parameter sets before stopping

# LLM configuration
DEFAULT_USE_LLM_FOR_SCREENING = False
DEFAULT_BAYESIAN_MODEL_TYPE = 'qNEHVI'  # Use multi-objective optimization

# Overaspirate configuration
DEFAULT_OVERASPIRATE_BASE_UL = 5.0
DEFAULT_OVERASPIRATE_SCALING_PERCENT = 5.0
DEFAULT_AUTO_CALIBRATE_OVERVOLUME = True
DEFAULT_OVERVOLUME_CALIBRATION_BUFFER_UL = 2.0
DEFAULT_OVERVOLUME_MAX_BASE_UL = 50.0
DEFAULT_OVERVOLUME_MAX_PERCENT = 100.0

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
INITIAL_SUGGESTIONS = DEFAULT_INITIAL_SUGGESTIONS
BATCH_SIZE = DEFAULT_BATCH_SIZE
REPLICATES = DEFAULT_REPLICATES
PRECISION_REPLICATES = DEFAULT_PRECISION_REPLICATES
VOLUMES = DEFAULT_VOLUMES.copy()
MAX_WELLS = DEFAULT_MAX_WELLS
INPUT_VIAL_STATUS_FILE = DEFAULT_INPUT_VIAL_STATUS_FILE

# Simplified stopping criteria  
MAX_MEASUREMENTS = DEFAULT_MAX_MEASUREMENTS
MIN_GOOD_TRIALS = DEFAULT_MIN_GOOD_TRIALS

USE_LLM_FOR_SCREENING = DEFAULT_USE_LLM_FOR_SCREENING
BAYESIAN_MODEL_TYPE = DEFAULT_BAYESIAN_MODEL_TYPE

# Overaspirate config
OVERASPIRATE_BASE_UL = DEFAULT_OVERASPIRATE_BASE_UL
OVERASPIRATE_SCALING_PERCENT = DEFAULT_OVERASPIRATE_SCALING_PERCENT
AUTO_CALIBRATE_OVERVOLUME = DEFAULT_AUTO_CALIBRATE_OVERVOLUME
OVERVOLUME_CALIBRATION_BUFFER_UL = DEFAULT_OVERVOLUME_CALIBRATION_BUFFER_UL
OVERVOLUME_MAX_BASE_UL = DEFAULT_OVERVOLUME_MAX_BASE_UL
OVERVOLUME_MAX_PERCENT = DEFAULT_OVERVOLUME_MAX_PERCENT

# Selective parameter optimization
USE_SELECTIVE_OPTIMIZATION = True
VOLUME_DEPENDENT_PARAMS = ["blowout_vol", "overaspirate_vol"]
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
    global LIQUID, SIMULATE, SEED, INITIAL_SUGGESTIONS, BATCH_SIZE, REPLICATES
    global PRECISION_REPLICATES, VOLUMES, MAX_WELLS, INPUT_VIAL_STATUS_FILE
    global MAX_MEASUREMENTS, MIN_GOOD_TRIALS, USE_LLM_FOR_SCREENING, BAYESIAN_MODEL_TYPE
    global OVERASPIRATE_BASE_UL, OVERASPIRATE_SCALING_PERCENT, AUTO_CALIBRATE_OVERVOLUME
    global OVERVOLUME_CALIBRATION_BUFFER_UL, OVERVOLUME_MAX_BASE_UL, OVERVOLUME_MAX_PERCENT
    
    print("🔄 Resetting configuration to default values...")
    
    LIQUID = DEFAULT_LIQUID
    SIMULATE = DEFAULT_SIMULATE
    SEED = DEFAULT_SEED
    INITIAL_SUGGESTIONS = DEFAULT_INITIAL_SUGGESTIONS
    BATCH_SIZE = DEFAULT_BATCH_SIZE
    REPLICATES = DEFAULT_REPLICATES
    PRECISION_REPLICATES = DEFAULT_PRECISION_REPLICATES
    VOLUMES = DEFAULT_VOLUMES.copy()
    MAX_WELLS = DEFAULT_MAX_WELLS
    INPUT_VIAL_STATUS_FILE = DEFAULT_INPUT_VIAL_STATUS_FILE
    MAX_MEASUREMENTS = DEFAULT_MAX_MEASUREMENTS
    MIN_GOOD_TRIALS = DEFAULT_MIN_GOOD_TRIALS
    USE_LLM_FOR_SCREENING = DEFAULT_USE_LLM_FOR_SCREENING
    BAYESIAN_MODEL_TYPE = DEFAULT_BAYESIAN_MODEL_TYPE
    OVERASPIRATE_BASE_UL = DEFAULT_OVERASPIRATE_BASE_UL
    OVERASPIRATE_SCALING_PERCENT = DEFAULT_OVERASPIRATE_SCALING_PERCENT
    AUTO_CALIBRATE_OVERVOLUME = DEFAULT_AUTO_CALIBRATE_OVERVOLUME
    OVERVOLUME_CALIBRATION_BUFFER_UL = DEFAULT_OVERVOLUME_CALIBRATION_BUFFER_UL
    OVERVOLUME_MAX_BASE_UL = DEFAULT_OVERVOLUME_MAX_BASE_UL
    OVERVOLUME_MAX_PERCENT = DEFAULT_OVERVOLUME_MAX_PERCENT
    
    print("✅ Configuration reset complete")

def get_current_config_summary():
    """Print current configuration summary."""
    print("📋 CURRENT EXPERIMENT CONFIG:")
    print(f"   Liquid: {LIQUID}")
    print(f"   Simulate: {SIMULATE}")
    print(f"   Volumes: {[f'{v*1000:.0f}uL' for v in VOLUMES]}")
    print(f"   Max measurements per volume: {MAX_MEASUREMENTS}")
    print(f"   Min good trials per volume: {MIN_GOOD_TRIALS}")
    print(f"   Global measurement limit: {MAX_WELLS} (entire calibration)")
    print(f"   Precision replicates: {PRECISION_REPLICATES}")
    print(f"   Initial suggestions: {INITIAL_SUGGESTIONS}")
    print(f"   LLM screening: {USE_LLM_FOR_SCREENING}")
    print(f"   Bayesian model: {BAYESIAN_MODEL_TYPE}")

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
        try:
            dev_multiplier = float(os.environ.get('SIM_DEV_MULTIPLIER', '2.0'))
            var_multiplier = float(os.environ.get('SIM_VAR_MULTIPLIER', '2.0'))
        except ValueError:
            dev_multiplier, var_multiplier = 2.0, 2.0
        
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
                            deviation_threshold=10.0):
    """
    Run adaptive measurement with conditional replicates based on initial deviation.
    
    Args:
        deviation_threshold: Threshold for running additional replicates (default 10%)
        
    Returns:
        dict: {
            'deviation': average deviation,
            'variability': calculated variability or penalty,
            'time': average time,
            'replicate_count': number of replicates run,
            'all_measurements': list of individual measurement results
        }
    """
    
    print(f"   🔬 Adaptive measurement (threshold: {deviation_threshold}%)...")
    
    all_measurements = []
    all_deviations = []
    all_times = []
    
    # Step 1: Run initial single measurement
    print(f"      Initial measurement...", end=" ")
    
    result = pipet_and_measure(lash_e, liquid_source, measurement_vial, 
                             volume, params, expected_mass, expected_time, 
                             1, simulate, autosave_raw_path, raw_measurements, 
                             liquid, new_pipet_each_time_set, trial_type)
    
    initial_deviation = result.get('deviation', 100.0)
    initial_time = result.get('time', 0.0)
    
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
        variability = 100.0  # Penalty value
        replicate_count = 1
        
        avg_deviation = initial_deviation
        avg_time = initial_time
        
    else:
        # Good accuracy - run 2 additional replicates
        print(f"      ✅ Good deviation (≤{deviation_threshold}%) - running 2 additional replicates...")
        
        for i in range(2):  # Run 2 more replicates (total of 3)
            replicate_num = i + 2  # Replicate 2 and 3
            print(f"         Replicate {replicate_num}/3...", end=" ")
            
            # Need to get fresh liquid source for each replicate
            check_if_measurement_vial_full(lash_e, {"measurement_vial_name": measurement_vial})
            
            result = pipet_and_measure(lash_e, liquid_source, measurement_vial, 
                                     volume, params, expected_mass, expected_time, 
                                     1, simulate, autosave_raw_path, raw_measurements, 
                                     liquid, new_pipet_each_time_set, trial_type)
            
            deviation = result.get('deviation', 100.0)
            time_taken = result.get('time', 0.0)
            
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
        
        replicate_count = 3
        
        # Calculate averages from all 3 measurements
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
    deviation_pct = trial_results.get('deviation', 100.0)  # Default to high value if missing
    volume_ul = volume_ml * 1000
    deviation_ul = (deviation_pct / 100.0) * volume_ul
    accuracy_ok = deviation_ul <= tolerances['deviation_ul']
    
    # Check precision - need raw measurements for this
    precision_ok = False
    precision_value = 100.0  # Default to high value
    
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
        precision_value = 100.0
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
        
        # Collect raw metrics for normalization
        accuracy = candidate.get('deviation', 100.0)  # % deviation
        precision = quality['precision_value']  # % precision
        time = candidate.get('time', 120.0)  # seconds
        
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
        composite_score = 0.5 * acc_score + 0.4 * prec_score + 0.1 * time_score
        
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
    1. Reached MAX_MEASUREMENTS total individual measurements (60), OR
    2. Have MIN_GOOD_TRIALS "good" parameter sets (6)
    
    Note: MAX_MEASUREMENTS counts individual measurements, MIN_GOOD_TRIALS counts parameter sets
    
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
    if total_measurements >= MAX_MEASUREMENTS:
        return {
            'should_stop': True,
            'reason': f'Reached maximum individual measurements ({MAX_MEASUREMENTS})',
            'good_trials': good_parameter_sets,
            'total_trials': total_parameter_sets
        }
    elif good_parameter_sets >= MIN_GOOD_TRIALS:
        return {
            'should_stop': True,
            'reason': f'Reached minimum good parameter sets ({MIN_GOOD_TRIALS})',
            'good_trials': good_parameter_sets,
            'total_trials': total_parameter_sets
        }
    else:
        return {
            'should_stop': False,
            'reason': f'Need {MIN_GOOD_TRIALS - good_parameter_sets} more good parameter sets or {MAX_MEASUREMENTS - total_measurements} more measurements',
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
    
    print(f"\n🔍 SCREENING PHASE: {INITIAL_SUGGESTIONS} initial suggestions...")
    
    screening_results = []
    
    for i in range(INITIAL_SUGGESTIONS):
        print(f"   Screening trial {i+1}/{INITIAL_SUGGESTIONS}...")
        
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
    
    # Calculate measured volume from deviation
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
        print(f"   🧪 Testing {volume*1000:.0f}uL...", end=" ")
        
        expected_mass = volume * LIQUIDS[liquid]["density"]
        expected_time = volume * 10.146 + 9.5813
        
        check_if_measurement_vial_full(lash_e, state)
        liquid_source = get_liquid_source_with_vial_management(lash_e, state)
        
        # Single measurement (no replicates as specified)
        result = pipet_and_measure(lash_e, liquid_source, state["measurement_vial_name"], 
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
        num_initial_recs=INITIAL_SUGGESTIONS,
        bayesian_batch_size=BATCH_SIZE,
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
    
    # Phase 2: Overaspirate calibration (if enabled)
    if AUTO_CALIBRATE_OVERVOLUME and len(VOLUMES) > 1:
        print(f"\n🔬 OVERASPIRATE CALIBRATION...")
        remaining_volumes = VOLUMES[1:]  # Skip first volume
        criteria = {'max_deviation_ul': tolerances['deviation_ul']}
        
        new_base_ul, new_scaling_percent, calibration_data = calibrate_overvolume_parameters(
            screening_results, remaining_volumes, lash_e, state, autosave_raw_path, 
            raw_measurements, liquid, new_pipet_each_time_set, criteria
        )
        
        if new_base_ul is not None:
            global OVERASPIRATE_BASE_UL, OVERASPIRATE_SCALING_PERCENT
            old_base = OVERASPIRATE_BASE_UL
            old_scaling = OVERASPIRATE_SCALING_PERCENT
            OVERASPIRATE_BASE_UL = new_base_ul
            OVERASPIRATE_SCALING_PERCENT = new_scaling_percent
            print(f"   📈 Updated overaspirate parameters: {old_base:.1f}μL + {old_scaling:.1f}% → {new_base_ul:.1f}μL + {new_scaling_percent:.1f}%")
            
            # CRITICAL: Recreate ax_client with updated overaspirate bounds
            print(f"   🔄 Recreating optimizer with updated overaspirate bounds...")
            max_overaspirate_ul_updated = get_max_overaspirate_ul(volume)  # This now uses updated parameters
            
            # Load existing screening data into new optimizer
            ax_client = optimizer_3obj.create_model(
                seed=SEED,
                num_initial_recs=0,  # No initial SOBOL since we already have screening data
                bayesian_batch_size=BATCH_SIZE,
                volume=volume,
                tip_volume=tip_volume,
                model_type=BAYESIAN_MODEL_TYPE,
                optimize_params=ALL_PARAMS,
                fixed_params={},
                simulate=SIMULATE,
                max_overaspirate_ul=max_overaspirate_ul_updated  # Uses updated bounds!
            )
            
            # Load screening results into the new optimizer
            optimizer_3obj.load_previous_data_into_model(ax_client, screening_results)
            print(f"   ✅ Loaded {len(screening_results)} screening results into updated optimizer")
    
    # Phase 3: 3-objective optimization with simplified stopping
    print(f"\n⚙️  3-OBJECTIVE OPTIMIZATION...")
    optimization_trial_count = 0
    
    while True:
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
    
    # Phase 5: Return best candidate (no precision test for first volume)
    print(f"\n✅ FIRST VOLUME OPTIMIZATION COMPLETE!")
    print(f"   Selected parameters will be used as baseline for subsequent volumes")
    
    best_params = {k: v for k, v in best_candidate.items() if k in ALL_PARAMS}
    return True, best_params

def run_precision_test(lash_e, state, best_params, volume, expected_mass, expected_time, 
                      autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set, 
                      max_variation_ul, all_results=None):
    """
    Run precision test with multiple replicates.
    
    Args:
        max_variation_ul: Maximum allowed variation in microliters
        
    Returns:
        tuple: (passed, measurements, times)
    """
    
    print(f"🎯 PRECISION TEST: Testing with {PRECISION_REPLICATES} replicates...")
    
    target_volume = volume  # mL
    variation_range = max_variation_ul / 1000  # Convert uL to mL
    min_acceptable = target_volume - variation_range
    max_acceptable = target_volume + variation_range
    
    print(f"   Target: {target_volume*1000:.0f}μL, Range: {min_acceptable*1000:.0f}μL - {max_acceptable*1000:.0f}μL (±{max_variation_ul:.0f}μL)")
    
    measurements = []
    times = []
    precision_start_idx = len(raw_measurements)
    
    for i in range(PRECISION_REPLICATES):
        print(f"   Replicate {i+1}/{PRECISION_REPLICATES}...", end=" ")
        check_if_measurement_vial_full(lash_e, state)
        
        liquid_source = get_liquid_source_with_vial_management(lash_e, state)
        result = pipet_and_measure(lash_e, liquid_source, state["measurement_vial_name"], 
                                 volume, best_params, expected_mass, expected_time, 
                                 1, SIMULATE, autosave_raw_path, raw_measurements, 
                                 liquid, new_pipet_each_time_set, "PRECISION")
        
        # Fix replicate number in raw measurements
        if raw_measurements and len(raw_measurements) > precision_start_idx:
            raw_measurements[-1]['replicate'] = i
        
        # Extract measurement
        if raw_measurements:
            actual_mass = raw_measurements[-1]['mass']
            liquid_density = LIQUIDS[liquid]["density"]
            actual_volume = actual_mass / liquid_density
            measurements.append(actual_volume)
        else:
            liquid_density = LIQUIDS[liquid]["density"]
            actual_volume = expected_mass / liquid_density
            measurements.append(actual_volume)
        
        times.append(result.get('time', 0))
        current_volume = measurements[-1]
        
        # Early stopping if outside range
        if current_volume < min_acceptable or current_volume > max_acceptable:
            print(f"❌ FAILED ({current_volume*1000:.0f}μL outside range)")
            return False, measurements, times[:len(measurements)]
        else:
            print(f"✅ {current_volume*1000:.0f}μL")
        
        # Add to all_results for tracking
        if all_results is not None:
            precision_result = dict(best_params)
            precision_result.update({
                "volume": volume,
                "deviation": result.get('deviation', 0),
                "time": result.get('time', 0),
                "trial_type": "PRECISION",
                "strategy": "PRECISION_TEST",
                "liquid": liquid,
                "trial_index": f"precision_{i+1}",
                "time_reported": datetime.now().isoformat(),
                "precision_replicate": i+1
            })
            all_results.append(precision_result)
    
    # All measurements passed
    mean_volume = np.mean(measurements)
    std_volume = np.std(measurements)
    cv_percent = (std_volume / mean_volume) * 100
    
    print(f"   ✅ PRECISION TEST PASSED: Mean {mean_volume*1000:.0f}μL ± {std_volume*1000:.1f}μL (CV: {cv_percent:.1f}%)")
    return True, measurements, times

def optimize_subsequent_volume(volume, lash_e, state, autosave_raw_path, raw_measurements, 
                              liquid, new_pipet_each_time_set, all_results, successful_params):
    """
    Optimize subsequent volumes following the modular approach:
    1. Start with precision test using best parameters from first volume
    2. If precision test fails, optimize only volume-dependent parameters for deviation only
    3. Once within tolerance, do precision test again
    4. Continue until precision test passes
    
    Args:
        successful_params: Parameters from first successful volume to use as baseline
    """
    
    print(f"\n🎯 OPTIMIZING SUBSEQUENT VOLUME: {volume*1000:.0f}μL")
    print(f"   Starting with precision test using best parameters from first volume")
    
    # Calculate tolerances and expected values
    tolerances = get_volume_dependent_tolerances(volume)
    expected_mass = volume * LIQUIDS[liquid]["density"]
    expected_time = volume * 10.146 + 9.5813
    
    # Get volume-dependent parameters to optimize (if needed)
    # Only include parameters that exist in the 3-objective optimizer
    volume_dependent_params = ['overaspirate_vol', 'blowout_vol']
    fixed_params = {k: v for k, v in successful_params.items() 
                   if k not in volume_dependent_params}
    
    candidate_params = successful_params.copy()
    max_attempts = 3  # Maximum optimization + precision test cycles
    
    for attempt in range(max_attempts):
        print(f"\n🎯 PRECISION TEST (attempt {attempt + 1}): Testing candidate parameters...")
        
        # Run precision test with current candidate parameters
        precision_passed, precision_measurements, precision_times = run_precision_test(
            lash_e, state, candidate_params, volume, expected_mass, expected_time,
            autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set,
            tolerances['variation_ul'], all_results
        )
        
        if precision_passed:
            print(f"   ✅ PRECISION TEST PASSED: Found reliable parameters")
            return True, candidate_params
        else:
            print(f"   ❌ PRECISION TEST FAILED: Need to optimize further")
            
            if attempt < max_attempts - 1:  # Don't optimize on last attempt
                print(f"🔍 OPTIMIZATION: Improving parameters (deviation-only)")
                
                # Create proper single-objective optimizer for deviation only
                tip_volume = get_tip_volume_for_volume(lash_e, volume)
                max_overaspirate_ul = get_max_overaspirate_ul(volume)
                
                # Check if single-objective optimizer is available
                if not OPTIMIZER_SINGLE_AVAILABLE:
                    print("   ❌ Single-objective optimizer not available - cannot optimize further")
                    break
                    
                ax_client = optimizer_single.create_model(
                    seed=SEED,
                    num_initial_recs=3,
                    bayesian_batch_size=1,
                    volume=volume,
                    tip_volume=tip_volume,
                    model_type='qLogEI',  # Use LogEI for single-objective
                    optimize_params=volume_dependent_params,
                    fixed_params=fixed_params,
                    simulate=SIMULATE,
                    max_overaspirate_ul=max_overaspirate_ul
                )
                
                # Do a few optimization trials
                optimization_trials = 0
                max_optimization_trials = 5
                
                while optimization_trials < max_optimization_trials:
                    params, trial_index = optimizer_single.get_suggestions(ax_client, volume, n=1)[0]
                    
                    optimization_trials += 1
                    print(f"   Optimization trial {optimization_trials}...")
                    
                    # Single measurement (no replicates during optimization)
                    check_if_measurement_vial_full(lash_e, state)
                    liquid_source = get_liquid_source_with_vial_management(lash_e, state)
                    
                    result = pipet_and_measure(lash_e, liquid_source, state["measurement_vial_name"], 
                                             volume, params, expected_mass, expected_time, 
                                             1, SIMULATE, autosave_raw_path, raw_measurements, 
                                             liquid, new_pipet_each_time_set, "OPTIMIZATION")
                    
                    # Add result to single-objective model (deviation only)
                    optimizer_single.add_result(ax_client, trial_index, result)
                    
                    # Store result
                    full_result = dict(params)
                    full_result.update({
                        "volume": volume,
                        "deviation": result.get('deviation', 0),
                        "time": result.get('time', 0),
                        "replicate_count": 1,
                        "trial_index": trial_index,
                        "strategy": f"OPTIMIZATION_{optimization_trials}",
                        "liquid": liquid,
                        "time_reported": datetime.now().isoformat()
                    })
                    all_results.append(full_result)
                    
                    print(f"      Dev: {result.get('deviation', 0):.1f}%")
                    
                    # Check if within tolerance for accuracy
                    if result.get('deviation', float('inf')) <= tolerances['tolerance_percent']:
                        print(f"   ✅ Found improved parameters: {result.get('deviation', 0):.1f}% deviation")
                        candidate_params = params
                        break
                
                if optimization_trials >= max_optimization_trials:
                    print(f"   ⚠️  Optimization complete after {max_optimization_trials} trials")
    
    print(f"   ❌ Failed to achieve reliable parameters after {max_attempts} attempts")
    return False, None

# --- MAIN WORKFLOW ---

def run_simplified_calibration_workflow(vial_mode="legacy", **config_overrides):
    """
    Main simplified calibration workflow.
    
    Args:
        vial_mode: Vial management mode ('legacy', 'maintain', 'swap', 'single')
        **config_overrides: Configuration parameters to override
    """
    
    # Reset config and apply overrides
    reset_config_to_defaults()
    
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
    print(f"🔢 Global limit: {MAX_WELLS} total measurements for entire calibration process")
    
    # Global measurement tracking
    global_measurement_count = 0
    
    # Process volumes
    successful_params = None
    
    for volume_index, volume in enumerate(VOLUMES):
        # Check global measurement limit before starting each volume
        wells_remaining = MAX_WELLS - global_measurement_count
        min_wells_needed = PRECISION_REPLICATES + 1  # At least 1 optimization + precision test
        
        if wells_remaining < min_wells_needed:
            print(f"\n⚠️  SKIPPING volume {volume*1000:.0f}μL: Not enough wells remaining")
            print(f"   Need ≥{min_wells_needed} wells, have {wells_remaining} remaining")
            break
        print(f"\n{'='*60}")
        print(f"VOLUME {volume_index + 1}/{len(VOLUMES)}: {volume*1000:.0f}μL")
        print(f"📊 Global measurements used: {global_measurement_count}/{MAX_WELLS}")
        print(f"{'='*60}")
        
        if volume_index == 0:
            # First volume: full optimization
            success, best_params = optimize_first_volume(
                volume, lash_e, state, autosave_raw_path, raw_measurements,
                LIQUID, new_pipet_each_time_set, all_results
            )
            
            if success:
                successful_params = best_params
                optimal_conditions.append({
                    'volume_ml': volume,
                    'volume_ul': volume * 1000,
                    **best_params,
                    'status': 'success'
                })
                print(f"✅ VOLUME {volume*1000:.0f}μL COMPLETED SUCCESSFULLY")
            else:
                print(f"❌ VOLUME {volume*1000:.0f}μL FAILED - stopping workflow")
                break
                
        else:
            # Subsequent volumes: selective optimization
            if successful_params is None:
                print(f"❌ No successful parameters from first volume - cannot continue")
                break
                
            success, best_params = optimize_subsequent_volume(
                volume, lash_e, state, autosave_raw_path, raw_measurements,
                LIQUID, new_pipet_each_time_set, all_results, successful_params
            )
            
            if success:
                optimal_conditions.append({
                    'volume_ml': volume,
                    'volume_ul': volume * 1000,
                    **best_params,
                    'status': 'success'
                })
                print(f"✅ VOLUME {volume*1000:.0f}μL COMPLETED SUCCESSFULLY")
            else:
                print(f"❌ VOLUME {volume*1000:.0f}μL FAILED")
                optimal_conditions.append({
                    'volume_ml': volume,
                    'volume_ul': volume * 1000,
                    'status': 'failed'
                })
        
        # Update global measurement count
        volume_measurements = len([r for r in all_results if r.get('volume') == volume])
        global_measurement_count += volume_measurements
        print(f"📊 Added {volume_measurements} measurements for this volume")
        print(f"📊 Total measurements used: {global_measurement_count}/{MAX_WELLS}")
        
        # Check if we're approaching the global limit
        if global_measurement_count >= MAX_WELLS - PRECISION_REPLICATES:
            print(f"🛑 STOPPING: Approaching global measurement limit ({MAX_WELLS})")
            print(f"   Used {global_measurement_count} measurements, need {PRECISION_REPLICATES} reserved for precision tests")
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
        
        # Save configuration
        config_summary = {
            'liquid': LIQUID,
            'simulate': SIMULATE,
            'volumes': VOLUMES,
            'max_measurements': MAX_MEASUREMENTS,
            'min_good_trials': MIN_GOOD_TRIALS,
            'precision_replicates': PRECISION_REPLICATES,
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
        
        # Optional: Run analysis if available
        if base_module.ANALYZER_AVAILABLE:
            print(f"📈 Running analysis...")
            save_analysis(results_df, raw_df, autosave_dir, 
                         include_shap=True, include_scatter=True,
                         optimal_conditions=optimal_conditions)
        
    except Exception as e:
        print(f"⚠️  Error saving results: {e}")
    
    print(f"\n🎉 SIMPLIFIED CALIBRATION WORKFLOW COMPLETE!")
    return optimal_conditions, autosave_dir

# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    # Example: Run simplified calibration with glycerol
    optimal_conditions, save_dir = run_simplified_calibration_workflow(
        vial_mode="maintain",
        liquid="glycerol",
        simulate=True,
        volumes=[0.05, 0.025, 0.1],  # Test with 3 volumes
        max_measurements=60,  # Per-volume measurement limit (was 20, increased for adequate optimization)
        min_good_trials=6  # Good parameter sets per volume
    )
    
    print(f"\n📋 FINAL RESULTS:")
    for condition in optimal_conditions:
        if condition['status'] == 'success':
            print(f"   ✅ {condition['volume_ul']:.0f}μL: {condition.get('deviation', 'N/A')}% deviation")
        else:
            print(f"   ❌ {condition['volume_ul']:.0f}μL: Failed")
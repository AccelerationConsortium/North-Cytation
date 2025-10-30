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
import recommenders.pipetting_optimizer_3objectives as optimizer_3obj
import recommenders.pipeting_optimizer_v2 as recommender_v2  # For subsequent volumes

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
DEFAULT_MAX_TRIALS = 60  # Maximum trials before stopping
DEFAULT_MIN_GOOD_TRIALS = 6  # Minimum "good" trials before stopping

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
MAX_TRIALS = DEFAULT_MAX_TRIALS
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
    global MAX_TRIALS, MIN_GOOD_TRIALS, USE_LLM_FOR_SCREENING, BAYESIAN_MODEL_TYPE
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
    MAX_TRIALS = DEFAULT_MAX_TRIALS
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
    print(f"   Max trials: {MAX_TRIALS}")
    print(f"   Min good trials: {MIN_GOOD_TRIALS}")
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
    elif 'variability' in trial_results:
        # Fallback: use variability if available (coefficient of variation)
        precision_value = trial_results['variability']
        precision_ok = precision_value <= tolerances['tolerance_percent']
    
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
    Rank candidates by priority: 1st accuracy → 2nd precision → 3rd time.
    
    This function is modular and can be easily modified later.
    
    Args:
        candidates: List of trial result dicts
        volume_ml: Target volume in mL
        tolerances: Dict with tolerance thresholds
        
    Returns:
        list: Ranked candidates (best first)
    """
    
    # Evaluate each candidate and add ranking metrics
    evaluated_candidates = []
    for candidate in candidates:
        quality = evaluate_trial_quality(candidate, volume_ml, tolerances)
        
        # Create ranking metrics (lower is better for all)
        accuracy_score = candidate.get('deviation', 100.0)  # Lower deviation is better
        precision_score = quality['precision_value']  # Lower precision range is better
        time_score = candidate.get('time', 1000.0)  # Lower time is better
        
        candidate_with_scores = candidate.copy()
        candidate_with_scores.update({
            'quality_evaluation': quality,
            'accuracy_score': accuracy_score,
            'precision_score': precision_score,
            'time_score': time_score
        })
        evaluated_candidates.append(candidate_with_scores)
    
    # Sort by lexicographic priority: accuracy → precision → time
    def ranking_key(candidate):
        return (candidate['accuracy_score'], candidate['precision_score'], candidate['time_score'])
    
    ranked_candidates = sorted(evaluated_candidates, key=ranking_key)
    
    # Log ranking results
    print(f"🏆 CANDIDATE RANKING (best first):")
    for i, candidate in enumerate(ranked_candidates[:5]):  # Show top 5
        quality = candidate['quality_evaluation']
        print(f"   #{i+1}: Dev={candidate['accuracy_score']:.1f}%, "
              f"Prec={candidate['precision_score']:.1f}%, "
              f"Time={candidate['time_score']:.1f}s, "
              f"Good={quality['is_good']}")
    
    return ranked_candidates

# --- SIMPLIFIED STOPPING CRITERIA ---

def check_stopping_criteria(all_results, volume_ml, tolerances):
    """
    Check if we should stop optimization based on simplified criteria:
    1. Reached MAX_TRIALS total trials, OR
    2. Have MIN_GOOD_TRIALS "good" trials
    
    Returns:
        dict: {'should_stop': bool, 'reason': str, 'good_trials': int, 'total_trials': int}
    """
    
    # Filter optimization results (exclude screening and precision tests)
    optimization_results = [r for r in all_results if r.get('strategy', '').startswith('OPT')]
    total_trials = len(optimization_results)
    
    # Count good trials
    good_trials = 0
    for result in optimization_results:
        quality = evaluate_trial_quality(result, volume_ml, tolerances)
        if quality['is_good']:
            good_trials += 1
    
    # Check stopping criteria
    if total_trials >= MAX_TRIALS:
        return {
            'should_stop': True,
            'reason': f'Reached maximum trials ({MAX_TRIALS})',
            'good_trials': good_trials,
            'total_trials': total_trials
        }
    elif good_trials >= MIN_GOOD_TRIALS:
        return {
            'should_stop': True,
            'reason': f'Reached minimum good trials ({MIN_GOOD_TRIALS})',
            'good_trials': good_trials,
            'total_trials': total_trials
        }
    else:
        return {
            'should_stop': False,
            'reason': f'Need {MIN_GOOD_TRIALS - good_trials} more good trials or {MAX_TRIALS - total_trials} more total trials',
            'good_trials': good_trials,
            'total_trials': total_trials
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
        
        # Run trial with single replicate
        liquid_source = get_liquid_source_with_vial_management(lash_e, state)
        result = pipet_and_measure(lash_e, liquid_source, state["measurement_vial_name"], 
                                 volume, params, expected_mass, expected_time, 
                                 REPLICATES, SIMULATE, autosave_raw_path, raw_measurements, 
                                 liquid, new_pipet_each_time_set, "SCREENING")
        
        # Add result to model (3-objective)
        model_result = {
            "deviation": result.get('deviation', 0),
            "variability": None,  # Single replicate, no variability
            "time": result.get('time', 0)
        }
        optimizer_3obj.add_result(ax_client, trial_index, model_result)
        
        # Store full result for later analysis
        full_result = dict(params)
        full_result.update({
            "volume": volume,
            "deviation": result.get('deviation', 0),
            "time": result.get('time', 0),
            "trial_index": trial_index,
            "strategy": "SCREENING",
            "liquid": liquid,
            "time_reported": datetime.now().isoformat()
        })
        screening_results.append(full_result)
        
        print(f"      → {result.get('deviation', 0):.1f}% deviation, {result.get('time', 0):.1f}s")
    
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

# --- PLACEHOLDER FOR OVERASPIRATE CALIBRATION ---
# TODO: Import calibrate_overvolume_parameters from modular or reimplement

def calibrate_overvolume_parameters(screening_candidates, remaining_volumes, lash_e, state, 
                                  autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set, 
                                  criteria, autosave_dir=None):
    """
    Placeholder for overaspirate calibration.
    TODO: Import from modular or reimplement simplified version.
    """
    print("⚠️  Overaspirate calibration not yet implemented in simplified version")
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
            OVERASPIRATE_BASE_UL = new_base_ul
            OVERASPIRATE_SCALING_PERCENT = new_scaling_percent
            print(f"   📈 Updated overaspirate parameters: {new_base_ul:.1f}μL + {new_scaling_percent:.1f}%")
    
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
        
        # Run trial with multiple replicates to get variability
        check_if_measurement_vial_full(lash_e, state)
        liquid_source = get_liquid_source_with_vial_management(lash_e, state)
        
        result = pipet_and_measure(lash_e, liquid_source, state["measurement_vial_name"], 
                                 volume, params, expected_mass, expected_time, 
                                 REPLICATES, SIMULATE, autosave_raw_path, raw_measurements, 
                                 liquid, new_pipet_each_time_set, "OPTIMIZATION")
        
        # Calculate variability from recent measurements if multiple replicates
        variability = None
        raw_measurements_for_trial = []
        if REPLICATES > 1:
            # Get last REPLICATES measurements for this trial
            recent_measurements = raw_measurements[-REPLICATES:]
            volumes = [m['calculated_volume'] for m in recent_measurements]
            raw_measurements_for_trial = volumes
            
            if len(volumes) > 1:
                mean_vol = np.mean(volumes)
                std_vol = np.std(volumes)
                variability = (std_vol / mean_vol) * 100  # CV as percentage
        
        # Add result to model
        model_result = {
            "deviation": result.get('deviation', 0),
            "variability": variability,
            "time": result.get('time', 0)
        }
        optimizer_3obj.add_result(ax_client, trial_index, model_result)
        
        # Store full result
        full_result = dict(params)
        full_result.update({
            "volume": volume,
            "deviation": result.get('deviation', 0),
            "variability": variability,
            "time": result.get('time', 0),
            "trial_index": trial_index,
            "strategy": f"OPTIMIZATION_{optimization_trial_count}",
            "liquid": liquid,
            "time_reported": datetime.now().isoformat(),
            "raw_measurements": raw_measurements_for_trial
        })
        all_results.append(full_result)
        
        quality = evaluate_trial_quality(full_result, volume, tolerances)
        quality_status = "✅ GOOD" if quality['is_good'] else "❌ needs improvement"
        print(f"      → {result.get('deviation', 0):.1f}% dev, {variability:.1f}% var, {result.get('time', 0):.1f}s ({quality_status})")
    
    # Phase 4: Select best candidate and run precision test
    print(f"\n🏆 SELECTING BEST CANDIDATE...")
    
    # Get optimization trials only for ranking
    optimization_trials = [r for r in all_results if r.get('strategy', '').startswith('OPTIMIZATION')]
    
    if not optimization_trials:
        print("   ❌ No optimization trials found!")
        return False, None
    
    # Rank candidates
    ranked_candidates = rank_candidates_by_priority(optimization_trials, volume, tolerances)
    best_candidate = ranked_candidates[0]
    
    print(f"   🎯 Selected best candidate:")
    print(f"      Accuracy: {best_candidate['accuracy_score']:.1f}% deviation")
    print(f"      Precision: {best_candidate['precision_score']:.1f}% variability")
    print(f"      Time: {best_candidate['time_score']:.1f}s")
    print(f"      Quality: {'✅ GOOD' if best_candidate['quality_evaluation']['is_good'] else '❌ Not good'}")
    
    # Phase 5: Single precision test
    print(f"\n🎯 PRECISION TEST...")
    
    best_params = {k: v for k, v in best_candidate.items() if k in ALL_PARAMS}
    precision_passed, precision_measurements, precision_times = run_precision_test(
        lash_e, state, best_params, volume, expected_mass, expected_time,
        autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set,
        tolerances['variation_ul'], all_results
    )
    
    if precision_passed:
        print(f"   ✅ PRECISION TEST PASSED!")
        return True, best_params
    else:
        print(f"   ❌ PRECISION TEST FAILED!")
        return False, None

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
    Optimize subsequent volumes with selective parameter optimization.
    
    Args:
        successful_params: Parameters from first successful volume to use as baseline
    """
    
    print(f"\n🎯 OPTIMIZING SUBSEQUENT VOLUME: {volume*1000:.0f}μL")
    print(f"   Using selective optimization (volume-dependent parameters only)")
    
    # Calculate tolerances and expected values
    tolerances = get_volume_dependent_tolerances(volume)
    expected_mass = volume * LIQUIDS[liquid]["density"]
    expected_time = volume * 10.146 + 9.5813
    tip_volume = get_tip_volume_for_volume(lash_e, volume)
    max_overaspirate_ul = get_max_overaspirate_ul(volume)
    
    print(f"   Target tolerances: ±{tolerances['deviation_ul']:.1f}μL deviation, {tolerances['tolerance_percent']:.1f}% precision")
    
    # Use v2 optimizer for subsequent volumes (2-objective: deviation + time)
    fixed_params = {k: v for k, v in successful_params.items() if k not in VOLUME_DEPENDENT_PARAMS}
    optimize_params = VOLUME_DEPENDENT_PARAMS
    
    print(f"   🔒 Fixed parameters: {list(fixed_params.keys())}")
    print(f"   ⚙️  Optimizing parameters: {optimize_params}")
    
    # Create 2-objective optimizer (deviation + time)
    ax_client = recommender_v2.create_model(
        seed=SEED,
        num_initial_recs=INITIAL_SUGGESTIONS,
        bayesian_batch_size=BATCH_SIZE,
        volume=volume,
        tip_volume=tip_volume,
        model_type=BAYESIAN_MODEL_TYPE,
        optimize_params=optimize_params,
        fixed_params=fixed_params,
        simulate=SIMULATE,
        max_overaspirate_ul=max_overaspirate_ul
    )
    
    # Optimization loop with simplified stopping
    optimization_trial_count = 0
    
    while True:
        # Check stopping criteria (use same function but different thresholds for subsequent volumes)
        stopping_result = check_stopping_criteria(all_results, volume, tolerances)
        print(f"   📊 Status: {stopping_result['total_trials']} trials, {stopping_result['good_trials']} good")
        
        if stopping_result['should_stop']:
            print(f"   🛑 STOPPING: {stopping_result['reason']}")
            break
        else:
            print(f"   🔄 CONTINUING: {stopping_result['reason']}")
        
        # Get next suggestion
        params, trial_index = recommender_v2.get_suggestions(ax_client, volume, n=1)[0]
        optimization_trial_count += 1
        
        print(f"   Optimization trial {optimization_trial_count}...")
        
        # Run trial
        check_if_measurement_vial_full(lash_e, state)
        liquid_source = get_liquid_source_with_vial_management(lash_e, state)
        
        result = pipet_and_measure(lash_e, liquid_source, state["measurement_vial_name"], 
                                 volume, params, expected_mass, expected_time, 
                                 REPLICATES, SIMULATE, autosave_raw_path, raw_measurements, 
                                 liquid, new_pipet_each_time_set, "OPTIMIZATION")
        
        # Add result to model (2-objective)
        model_result = {
            "deviation": result.get('deviation', 0),
            "time": result.get('time', 0)
        }
        recommender_v2.add_result(ax_client, trial_index, model_result)
        
        # Store full result
        full_result = dict(params)
        full_result.update({
            "volume": volume,
            "deviation": result.get('deviation', 0),
            "time": result.get('time', 0),
            "trial_index": trial_index,
            "strategy": f"OPTIMIZATION_{optimization_trial_count}",
            "liquid": liquid,
            "time_reported": datetime.now().isoformat()
        })
        all_results.append(full_result)
        
        quality = evaluate_trial_quality(full_result, volume, tolerances)
        quality_status = "✅ GOOD" if quality['is_good'] else "❌ needs improvement"
        print(f"      → {result.get('deviation', 0):.1f}% dev, {result.get('time', 0):.1f}s ({quality_status})")
    
    # Select best candidate and run precision test
    print(f"\n🏆 SELECTING BEST CANDIDATE...")
    
    optimization_trials = [r for r in all_results if r.get('strategy', '').startswith('OPTIMIZATION') and r.get('volume') == volume]
    
    if not optimization_trials:
        print("   ❌ No optimization trials found!")
        return False, None
    
    # For subsequent volumes, simpler ranking (just by deviation since we only have 2 objectives)
    best_candidate = min(optimization_trials, key=lambda x: x['deviation'])
    
    print(f"   🎯 Selected best candidate: {best_candidate['deviation']:.1f}% deviation, {best_candidate['time']:.1f}s")
    
    # Single precision test
    print(f"\n🎯 PRECISION TEST...")
    
    best_params = {k: v for k, v in best_candidate.items() if k in ALL_PARAMS}
    precision_passed, precision_measurements, precision_times = run_precision_test(
        lash_e, state, best_params, volume, expected_mass, expected_time,
        autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set,
        tolerances['variation_ul'], all_results
    )
    
    if precision_passed:
        print(f"   ✅ PRECISION TEST PASSED!")
        return True, best_params
    else:
        print(f"   ❌ PRECISION TEST FAILED!")
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
    
    # Process volumes
    successful_params = None
    
    for volume_index, volume in enumerate(VOLUMES):
        print(f"\n{'='*60}")
        print(f"VOLUME {volume_index + 1}/{len(VOLUMES)}: {volume*1000:.0f}μL")
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
            'max_trials': MAX_TRIALS,
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
        volumes=[0.05, 0.025],  # Test with 2 volumes
        max_trials=20,  # Smaller for testing
        min_good_trials=3
    )
    
    print(f"\n📋 FINAL RESULTS:")
    for condition in optimal_conditions:
        if condition['status'] == 'success':
            print(f"   ✅ {condition['volume_ul']:.0f}μL: {condition.get('deviation', 'N/A')}% deviation")
        else:
            print(f"   ❌ {condition['volume_ul']:.0f}μL: Failed")
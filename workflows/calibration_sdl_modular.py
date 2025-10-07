# calibration_sdl_modular.py
import sys
import os

sys.path.append("../North-Cytation")

from calibration_sdl_base import *
from master_usdl_coordinator import Lash_E
import recommenders.pipeting_optimizer_v2 as recommender_v2
import recommenders.pipetting_optimizer_v3 as recommender_v3
from datetime import datetime
import pandas as pd
import numpy as np

# Conditionally import slack_agent
try:
    import slack_agent
    SLACK_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import slack_agent: {e}")
    print("Slack notifications will be disabled.")
    slack_agent = None
    SLACK_AVAILABLE = False

# Conditionally import LLM optimizer
try:
    import recommenders.llm_optimizer as llm_opt
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import LLM optimizer: {e}")
    print("LLM optimization will be disabled.")
    llm_opt = None
    LLM_AVAILABLE = False

# --- Experiment Config ---
LIQUID = "glycerol"
SIMULATE = True
SEED = 7
INITIAL_SUGGESTIONS = 5  # replaces SOBOL_CYCLES_PER_VOLUME
BATCH_SIZE = 1
REPLICATES = 1  # for optimization
PRECISION_REPLICATES = 5
VOLUMES = [0.05, 0.025, 0.1] #Small tip
#VOLUMES = [0.3, 0.5, 1.0] # Large tip
MAX_WELLS = 96
INPUT_VIAL_STATUS_FILE = "status/calibration_vials_short.csv"


# --- LLM Configuration ---
# Two independent LLM settings for different phases:
# 1. SCREENING: Initial exploration of parameter space (first volume only)
#    - USE_LLM_FOR_SCREENING = True: Use LLM instead of SOBOL for initial suggestions
#    - USE_LLM_FOR_SCREENING = False: Use SOBOL for initial suggestions (default)
# 
# 2. OPTIMIZATION: Finding better parameters when current ones don't meet criteria
#    - USE_LLM_FOR_OPTIMIZATION = True: Use LLM instead of Bayesian optimization
#    - USE_LLM_FOR_OPTIMIZATION = False: Use Bayesian optimization (default)
USE_LLM_FOR_SCREENING = False     # LLM vs SOBOL for initial exploration (first volume)
USE_LLM_FOR_OPTIMIZATION = False  # LLM vs Bayesian for optimization loops

# --- Bayesian Model Configuration ---
# Controls which Bayesian acquisition function to use for optimization
# NOTE: This is only used when USE_LLM_FOR_OPTIMIZATION = False
# Options: 'qEI' (Expected Improvement), 'qLogEI' (Log Expected Improvement), 'qNEHVI' (Noisy Expected Hypervolume Improvement)
BAYESIAN_MODEL_TYPE = 'qEI'  # Default Bayesian acquisition function

if SIMULATE:
    DEFAULT_LOCAL_AUTOSAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'calibration_runs'))
    os.makedirs(DEFAULT_LOCAL_AUTOSAVE_DIR, exist_ok=True)
    BASE_AUTOSAVE_DIR = os.environ.get('CALIBRATION_AUTOSAVE_DIR', DEFAULT_LOCAL_AUTOSAVE_DIR)
    print(f"[info] Using BASE_AUTOSAVE_DIR={BASE_AUTOSAVE_DIR}")
else:
    BASE_AUTOSAVE_DIR='C:\\Users\\Imaging Controller\\Desktop\\Calibration_SDL_Output\\New_Method'

# Criteria (For real life testing) - Base tolerances with volume-dependent scaling
BASE_DEVIATION_UL = 1.0  # Base ±1 μL absolute deviation for optimization acceptance  
BASE_TIME_SECONDS = 20  # Base time in seconds
BASE_VARIATION_UL = 2.0  # Base ±2 μL absolute variation for precision test

max_overvolume_percent = 0.2  # 20% extra volume to account for pipetting error

# Volume scaling factors (per 100 μL above baseline)
DEVIATION_SCALING_FACTOR = 0.2  # +0.2 μL per 100 μL (1μL->2μL for 500μL)
TIME_SCALING_FACTOR = 1.0  # +1 second per 100 μL  
VARIATION_SCALING_FACTOR = 0.2  # +0.2 μL per 100 μL (2μL->3μL for 500μL)

# Selective parameter optimization config
USE_SELECTIVE_OPTIMIZATION = True  # Enable selective parameter optimization
USE_HISTORICAL_DATA_FOR_OPTIMIZATION = True  # Load data from previous volumes into optimizer
VOLUME_DEPENDENT_PARAMS = ["blowout_vol", "overaspirate_vol"]  # Parameters to optimize for each volume
ALL_PARAMS = ["aspirate_speed", "dispense_speed", "aspirate_wait_time", "dispense_wait_time", 
              "retract_speed", "blowout_vol", "post_asp_air_vol", "overaspirate_vol"]

# --- Helper Methods ---
def get_volume_dependent_tolerances(volume_ml):
    """Calculate volume-dependent tolerances based on scaling factors."""
    volume_ul = volume_ml * 1000  # Convert to μL
    volume_excess_ul = max(0, volume_ul - 100)  # Only scale above 100μL baseline
    scaling_factor = volume_excess_ul / 100  # Per 100μL scaling
    
    deviation_ul = BASE_DEVIATION_UL + (DEVIATION_SCALING_FACTOR * scaling_factor)
    variation_ul = BASE_VARIATION_UL + (VARIATION_SCALING_FACTOR * scaling_factor)

    # In simulation we relax tolerances so runs "pass" more often to view plots.
    # Strategy: enforce a minimum relative window for deviation/variation.
    if SIMULATE:
        # Allow environment override for quick tuning without code edits
        try:
            min_rel_dev = float(os.environ.get('SIM_MIN_REL_DEV', '0.08'))  # default 8%
            min_rel_var = float(os.environ.get('SIM_MIN_REL_VAR', '0.10'))  # default 10%
        except ValueError:
            min_rel_dev, min_rel_var = 0.8, 0.10
        deviation_ul = max(deviation_ul, volume_ul * min_rel_dev)
        variation_ul = max(variation_ul, volume_ul * min_rel_var)

    tolerances = {
        'deviation_ul': deviation_ul,
        'time_seconds': BASE_TIME_SECONDS + (TIME_SCALING_FACTOR * scaling_factor),
        'variation_ul': variation_ul
    }
    
    return tolerances

def initialize_experiment():
    DENSITY_LIQUID = LIQUIDS[LIQUID]["density"]
    NEW_PIPET_EACH_TIME_SET = LIQUIDS[LIQUID]["refill_pipets"]
    state = {"measurement_vial_index": 0, "measurement_vial_name": "measurement_vial_0"}
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False)
    lash_e.nr_robot.check_input_file()
    lash_e.nr_robot.move_vial_to_location("measurement_vial_0", "clamp", 0)
    return lash_e, DENSITY_LIQUID, NEW_PIPET_EACH_TIME_SET, state

def get_tip_volume_for_volume(lash_e, volume):
    """Get the tip volume capacity for a given pipetting volume"""
    try:
        # Use North_Safe's tip selection logic to get the appropriate tip
        tip_type = lash_e.nr_robot.select_pipet_tip(volume)
        tip_config = lash_e.nr_robot.get_config_parameter('pipet_tips', tip_type, None, error_on_missing=False)
        if tip_config:
            return tip_config.get('volume', 1.0)  # Default to 1.0 mL if not found
        else:
            # Fallback logic based on volume ranges
            if volume <= 0.25:
                return 0.25  # small_tip
            else:
                return 1.0   # large_tip
    except Exception as e:
        print(f"Warning: Could not determine tip volume for {volume} mL: {e}")
        # Safe fallback
        return 1.0 if volume > 0.25 else 0.25

def get_recommender():
    """Get the appropriate recommender module based on configuration"""
    if USE_SELECTIVE_OPTIMIZATION:
        return recommender_v3
    else:
        return recommender_v2

def get_optimization_config(volume_index, completed_volumes, all_results):
    """
    Get optimization configuration for a given volume.
    
    Args:
        volume_index: Index of current volume (0 for first volume)
        completed_volumes: List of (volume, params) pairs that passed precision test
        all_results: All optimization results from previous volumes
        
    Returns:
        optimize_params: List of parameters to optimize
        fixed_params: Dict of parameters to keep fixed
    """
    if not USE_SELECTIVE_OPTIMIZATION:
        # Use all parameters (v2 behavior)
        return ALL_PARAMS, {}
    
    # Once we have ANY successful volume (passed precision test), 
    # we only optimize volume-dependent parameters for all remaining volumes
    if completed_volumes:
        # Use parameters from the most recent successful volume
        _, last_successful_params = completed_volumes[-1]
        fixed_params = {k: v for k, v in last_successful_params.items() 
                      if k not in VOLUME_DEPENDENT_PARAMS}
        print(f"   Using parameters from successful volume {completed_volumes[-1][0]*1000:.0f}μL")
        print(f"   🔒 FIXED: {list(fixed_params.keys())}")
        return VOLUME_DEPENDENT_PARAMS, fixed_params
    
    # If no successful volumes yet, optimize all parameters
    else:
        print(f"   No successful volumes yet - optimizing all parameters")
        return ALL_PARAMS, {}

def load_previous_data_into_model(ax_client, all_results):
    """Load previous experimental results into a new ax_client model"""
    if not all_results:
        return
    
    # Define parameter columns that ax_client expects
    parameter_columns = [
        'aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 
        'dispense_wait_time', 'retract_speed', 'blowout_vol', 
        'post_asp_air_vol', 'overaspirate_vol'
    ]
    
    # Define outcome columns (only deviation and time for v3 optimizer)
    outcome_columns = ['deviation', 'time']
    
    print(f"Loading {len(all_results)} existing trials into new model...")
    
    # Add each result as a completed trial (only optimization trials, not precision tests)
    optimization_results = [r for r in all_results if r.get('strategy') != 'PRECISION_TEST']
    print(f"  Loading {len(optimization_results)} optimization trials (excluding {len(all_results) - len(optimization_results)} precision test measurements)")
    
    for result in optimization_results:
        try:
            # Extract parameters
            parameters = {}
            for param in parameter_columns:
                if param in result:
                    # Convert to appropriate type
                    if param in ['aspirate_speed', 'dispense_speed']:
                        parameters[param] = int(result[param])
                    else:
                        parameters[param] = float(result[param])
            
            # Skip if we don't have all required parameters
            if len(parameters) != len(parameter_columns):
                print(f"  Skipping result missing parameters: {set(parameter_columns) - set(parameters.keys())}")
                continue
            
            # Extract outcomes
            raw_data = {}
            for col in outcome_columns:
                if col in result and result[col] is not None:
                    raw_data[col] = (float(result[col]), 0.0)  # (mean, sem)
            
            # Skip if we don't have all required outcomes
            if len(raw_data) != len(outcome_columns):
                print(f"  Skipping result missing outcomes: {set(outcome_columns) - set(raw_data.keys())}")
                continue
            
            # Attach trial to get trial index
            parameterization, trial_index = ax_client.attach_trial(parameters)
            
            # Complete the trial with the outcomes
            ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
            
        except Exception as e:
            print(f"  Error loading result into model: {e}")
            continue
    
    print(f"Successfully loaded previous data into new model")

def get_initial_suggestions(ax_client, method, n, volume, expected_mass, expected_time, lash_e, state, autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set, all_results, autosave_summary_path):
    suggestions = []
    if method == 'LLM' and LLM_AVAILABLE:
        # Use LLM for initial suggestions
        suggestions = get_llm_suggestions(ax_client, n, all_results)
    else:
        # Try to get n suggestions, but handle search space exhaustion gracefully
        for i in range(n):
            try:
                params, trial_index = ax_client.get_next_trial()
                suggestions.append((params, trial_index))
            except Exception as e:
                print(f"  Warning: Could not generate suggestion {i+1}/{n}: {e}")
                print(f"  Search space may be exhausted. Continuing with {len(suggestions)} suggestions.")
                break
    for i, (params, trial_index) in enumerate(suggestions):
        check_if_measurement_vial_full(lash_e, state)
        result = pipet_and_measure(lash_e, 'liquid_source', state["measurement_vial_name"], volume, params, expected_mass, expected_time, REPLICATES, SIMULATE, autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set)
        
        # Get the most recent measurement for display
        recent_mass = raw_measurements[-1]['mass'] if raw_measurements else expected_mass
        recent_volume = raw_measurements[-1]['calculated_volume'] if raw_measurements else volume
        
        # Show result with pass/fail status
        current_trial = len(all_results) + 1
        print(f"  Initial trial {i+1}/{len(suggestions)} (Trial {current_trial}): {recent_mass:.4f}g → {recent_volume*1000:.1f}μL, deviation={result.get('deviation', 'N/A'):.2f}%, time={result.get('time', 'N/A'):.1f}s")
        
        get_recommender().add_result(ax_client, trial_index, result)
        result.update(params)
        result.update({"volume": volume, "trial_index": trial_index, "strategy": method, "liquid": liquid, "time_reported": datetime.now().isoformat()})
        result = strip_tuples(result)
        all_results.append(result)
        if not SIMULATE:
            pd.DataFrame([result]).to_csv(autosave_summary_path, mode='a', index=False, header=not os.path.exists(autosave_summary_path))

def get_llm_suggestions(ax_client, n, all_results):
    # This is a simplified version; you may want to add existing data
    config_path = os.path.abspath("recommenders/calibration_unified_config.json")
    llm_input_file = os.path.join("output", "temp_llm_input.csv")
    os.makedirs("output", exist_ok=True)
    pd.DataFrame(all_results).to_csv(llm_input_file, index=False)
    optimizer = llm_opt.LLMOptimizer()
    config = optimizer.load_config(config_path)
    config["batch_size"] = n
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_output_file = os.path.join("output", f"llm_recommendations_{timestamp}.csv")
    result = optimizer.optimize(llm_input_file, config, llm_output_file)
    all_llm_recs = result.get('recommendations', [])
    suggestions = []
    for i, llm_params in enumerate(all_llm_recs[:n]):
        if llm_params:
            expected_params = set(ax_client.experiment.search_space.parameters.keys())
            filtered_params = {k: v for k, v in llm_params.items() if k in expected_params}
            params, trial_index = ax_client.attach_trial(filtered_params)
            suggestions.append((params, trial_index))
    return suggestions

def optimization_loop(ax_client, method, batch_size, volume, expected_mass, expected_time, lash_e, state, autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set, all_results, autosave_summary_path, criteria):
    while not check_optimization_criteria(all_results, criteria):
        if method == 'LLM' and LLM_AVAILABLE:
            suggestions = get_llm_suggestions(ax_client, batch_size, all_results)
        else:
            try:
                suggestions = get_recommender().get_suggestions(ax_client, volume, n=batch_size)
            except Exception as e:
                print(f"Error getting suggestions: {e}")
                suggestions = []
        for params, trial_index in suggestions:
            check_if_measurement_vial_full(lash_e, state)
            result = pipet_and_measure(lash_e, 'liquid_source', state["measurement_vial_name"], volume, params, expected_mass, expected_time, REPLICATES, SIMULATE, autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set)
            get_recommender().add_result(ax_client, trial_index, result)
            result.update(params)
            result.update({"volume": volume, "trial_index": trial_index, "strategy": method, "liquid": liquid, "time_reported": datetime.now().isoformat()})
            result = strip_tuples(result)
            all_results.append(result)
            if not SIMULATE:
                pd.DataFrame([result]).to_csv(autosave_summary_path, mode='a', index=False, header=not os.path.exists(autosave_summary_path))

def check_optimization_criteria(all_results, criteria):
    # Check if we have results that meet both deviation and time criteria
    # ONLY look at optimization trials, NOT precision test measurements
    if not all_results:
        return False
    
    # Filter out precision test measurements - only check optimization trials
    optimization_results = [r for r in all_results if r.get('strategy') != 'PRECISION_TEST']
    
    if not optimization_results:
        return False
        
    df = pd.DataFrame(optimization_results)
    if 'deviation' in df.columns and 'time' in df.columns and 'volume' in df.columns:
        # deviation is % already. Compute absolute μL deviation for original logic
        absolute_deviation_ul = (df['deviation'] / 100) * (df['volume'] * 1000)

        if SIMULATE:
            # In simulation: allow pass if percent deviation <= dynamic_pct OR absolute deviation <= specified μL
            # Also relax time constraint to 1.25 * max_time
            max_pct = 20.0  # default percent threshold
            try:
                max_pct = float(os.environ.get('SIM_MAX_DEV_PCT', '20'))
            except ValueError:
                pass
            relaxed_time = criteria['max_time'] * 1.25
            meets_criteria = ((df['deviation'] <= max_pct) | (absolute_deviation_ul <= criteria['max_deviation_ul'])) & (df['time'] <= relaxed_time)
        else:
            meets_criteria = (absolute_deviation_ul <= criteria['max_deviation_ul']) & (df['time'] <= criteria['max_time'])

        if not meets_criteria.any():
            # Debug first few rows
            sample = df.head(3)[['volume','deviation','time']].to_dict(orient='records')
            print(f"[criteria-debug] No trial meets criteria yet. Example rows: {sample} | criteria={criteria} | SIMULATE={SIMULATE}")
        return meets_criteria.any()
    return False

def run_precision_test(lash_e, state, best_params, volume, expected_mass, expected_time, autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set, max_variation_ul, all_results=None):
    print(f"🎯 PRECISION TEST: Testing best parameters with {PRECISION_REPLICATES} replicates...")
    
    # Calculate acceptable range around target volume using absolute tolerance
    target_volume = volume  # mL
    variation_range = max_variation_ul / 1000  # Convert μL to mL
    min_acceptable = target_volume - variation_range
    max_acceptable = target_volume + variation_range
    
    print(f"   Target: {target_volume*1000:.0f}μL, Range: {min_acceptable*1000:.0f}μL - {max_acceptable*1000:.0f}μL (±{max_variation_ul:.0f}μL)")
    
    measurements = []
    deviations = []
    times = []  # Capture timing data
    
    # Store the starting point in raw_measurements to track precision replicate numbers
    precision_start_idx = len(raw_measurements)
    
    for i in range(PRECISION_REPLICATES):
        print(f"   Replicate {i+1}/{PRECISION_REPLICATES}...", end=" ")
        check_if_measurement_vial_full(lash_e, state)
        
        # Get single measurement result
        result = pipet_and_measure(lash_e, 'liquid_source', state["measurement_vial_name"], volume, best_params, expected_mass, expected_time, 1, SIMULATE, autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set)
        
        # Fix the replicate number in the raw_measurements entry that was just added
        if raw_measurements and len(raw_measurements) > precision_start_idx:
            raw_measurements[-1]['replicate'] = i  # Set correct replicate number (0-5)
        
        # Extract the actual measurement from raw_measurements (last entry)
        if raw_measurements:
            actual_mass = raw_measurements[-1]['mass']
            # Convert mass to volume using correct liquid density - FAIL if not found
            if liquid not in LIQUIDS:
                raise ValueError(f"Unknown liquid '{liquid}' - must be one of: {list(LIQUIDS.keys())}")
            if "density" not in LIQUIDS[liquid]:
                raise ValueError(f"No density specified for liquid '{liquid}' in LIQUIDS dictionary")
            liquid_density = LIQUIDS[liquid]["density"]
            actual_volume = actual_mass / liquid_density
            measurements.append(actual_volume)
        else:
            # Fallback for simulation - convert expected mass to volume using correct density
            if liquid not in LIQUIDS:
                raise ValueError(f"Unknown liquid '{liquid}' - must be one of: {list(LIQUIDS.keys())}")
            if "density" not in LIQUIDS[liquid]:
                raise ValueError(f"No density specified for liquid '{liquid}' in LIQUIDS dictionary")
            liquid_density = LIQUIDS[liquid]["density"]
            actual_volume = expected_mass / liquid_density
            measurements.append(actual_volume)
        
        deviation = result.get('deviation', 0)
        deviations.append(deviation)
        
        # Capture timing data
        time_taken = result.get('time', 0)
        times.append(time_taken)
        # Debug timing data
        if i == 0:  # Only print on first replicate to avoid spam
            print(f"   Debug: time_taken={time_taken}, result keys={list(result.keys())}")
        
        current_volume = measurements[-1]
        
        # Early stopping check: if current measurement is outside acceptable range
        if current_volume < min_acceptable or current_volume > max_acceptable:
            print(f"❌ FAILED ({current_volume*1000:.0f}μL outside range)")
            print(f"   Precision test FAILED after {len(measurements)} replicates")
            return False, measurements, times[:len(measurements)]
        else:
            print(f"✅ {current_volume*1000:.0f}μL")
        
        # Add precision test measurement to all_results for tracking
        if all_results is not None:
            precision_result = dict(best_params)  # Copy best parameters
            precision_result.update({
                "volume": volume,
                "deviation": deviation,
                "time": result.get('time', 0),
                "variability": result.get('variability', 0),
                "simulated_mass": raw_measurements[-1]['mass'] if raw_measurements else expected_mass,
                "strategy": "PRECISION_TEST",
                "liquid": liquid,
                "trial_index": f"precision_{i+1}",
                "time_reported": datetime.now().isoformat(),
                "precision_replicate": i+1,
                "target_volume": target_volume,
                "acceptable_range": f"{min_acceptable*1000:.1f}-{max_acceptable*1000:.1f}μL"
            })
            all_results.append(precision_result)
    
    # If we reach here, all measurements were within the acceptable range
    mean_volume = np.mean(measurements)
    std_volume = np.std(measurements)
    cv_percent = (std_volume / mean_volume) * 100  # Coefficient of variation
    
    print(f"   ✅ PRECISION TEST PASSED: Mean {mean_volume*1000:.0f}μL ± {std_volume*1000:.1f}μL (CV: {cv_percent:.1f}%)")
    
    return True, measurements, times

def get_fallback_suggestions(ax_client, all_results, volume, n):
    """
    Fallback when search space is exhausted - reuse best parameters from previous trials
    """
    suggestions = []
    if not all_results:
        return suggestions
    
    # Get best performing parameters from all trials
    df = pd.DataFrame(all_results)
    if 'deviation' in df.columns:
        best_results = df.nsmallest(min(n*3, len(df)), 'deviation')  # Get more than needed
        
        param_keys = set(ax_client.experiment.search_space.parameters.keys())
        
        for _, result in best_results.head(n).iterrows():
            try:
                # Extract parameter values
                params = {k: result[k] for k in param_keys if k in result}
                if params:
                    params, trial_index = ax_client.attach_trial(params)
                    suggestions.append((params, trial_index))
                    if len(suggestions) >= n:
                        break
            except Exception as e:
                print(f"  Error creating fallback suggestion: {e}")
                continue
    
    return suggestions

def check_if_measurement_vial_full(lash_e, state):
    current_vial = state["measurement_vial_name"]
    vol = lash_e.nr_robot.get_vial_info(current_vial, "vial_volume")
    if vol > 7.0:
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.return_vial_home(current_vial)
        state["measurement_vial_index"] += 1
        new_vial_name = f"measurement_vial_{state['measurement_vial_index']}"
        state["measurement_vial_name"] = new_vial_name
        lash_e.logger.info(f"[info] Switching to new measurement vial: {new_vial_name}")
        lash_e.nr_robot.move_vial_to_location(new_vial_name, "clamp", 0)

def params_are_blacklisted(params, blacklisted_params, tolerance=1e-6):
    """Check if parameters are in the blacklist (failed precision test)"""
    param_keys = set(params.keys())
    for blacklisted in blacklisted_params:
        blacklisted_keys = set(blacklisted.keys())
        if param_keys == blacklisted_keys:
            # Check if all parameter values match within tolerance
            match = True
            for key in param_keys:
                if abs(params[key] - blacklisted[key]) > tolerance:
                    match = False
                    break
            if match:
                return True
    return False

def save_optimal_conditions(optimal_conditions, filepath):
    """Save the optimal conditions to CSV with just essential information"""
    if optimal_conditions:
        df = pd.DataFrame(optimal_conditions)
        df.to_csv(filepath, index=False)
        print(f"✅ Saved optimal conditions to: {filepath}")

def save_experiment_config(autosave_dir, new_pipet_each_time_set=None):
    """Save the experiment configuration to a file for reference"""
    config = {
        'experiment_type': 'calibration_sdl_modular',
        'timestamp': datetime.now().isoformat(),
        'liquid': LIQUID,
        'simulate': SIMULATE,
        'seed': SEED,
        'initial_suggestions': INITIAL_SUGGESTIONS,
        'batch_size': BATCH_SIZE,
        'replicates': REPLICATES,
        'precision_replicates': PRECISION_REPLICATES,
        'volumes': VOLUMES,
        'max_wells': MAX_WELLS,
        'max_overvolume_percent': max_overvolume_percent,
        'new_pipet_each_time_set': new_pipet_each_time_set,
        'base_deviation_ul': BASE_DEVIATION_UL,
        'base_time_seconds': BASE_TIME_SECONDS,
        'base_variation_ul': BASE_VARIATION_UL,
        'deviation_scaling_factor': DEVIATION_SCALING_FACTOR,
        'time_scaling_factor': TIME_SCALING_FACTOR,
        'variation_scaling_factor': VARIATION_SCALING_FACTOR,
        'use_selective_optimization': USE_SELECTIVE_OPTIMIZATION,
        'use_historical_data_for_optimization': USE_HISTORICAL_DATA_FOR_OPTIMIZATION,
        'volume_dependent_params': VOLUME_DEPENDENT_PARAMS,
        'all_params': ALL_PARAMS,
        'use_llm_for_screening': USE_LLM_FOR_SCREENING,
        'use_llm_for_optimization': USE_LLM_FOR_OPTIMIZATION,
        'bayesian_model_type': BAYESIAN_MODEL_TYPE,
        'input_vial_status_file': INPUT_VIAL_STATUS_FILE,
        'slack_available': SLACK_AVAILABLE,
        'llm_available': LLM_AVAILABLE
    }
    
    config_path = os.path.join(autosave_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        import json
        json.dump(config, f, indent=2, default=str)
    
    print(f"✅ Saved experiment config to: {config_path}")
    return config_path

def main():
    lash_e, DENSITY_LIQUID, NEW_PIPET_EACH_TIME_SET, state = initialize_experiment()
    
    # LLM Control Variables - Two separate settings for two different phases
    use_llm_for_screening = USE_LLM_FOR_SCREENING     # Use LLM instead of SOBOL for initial exploration (first volume only)
    use_llm_for_optimization = USE_LLM_FOR_OPTIMIZATION  # Use LLM instead of Bayesian for optimization loops
    
    # Bayesian model type for ax_client (separate from screening method)
    bayesian_model_type = BAYESIAN_MODEL_TYPE  # qEI, qLogEI, qNEHVI, etc.
    
    # Show optimization configuration
    print(f"\n🤖 OPTIMIZATION CONFIGURATION:")
    print(f"   📊 Initial Screening (first volume): {'LLM' if use_llm_for_screening else 'SOBOL'}")
    print(f"   🔍 Optimization Loops: {'LLM' if use_llm_for_optimization else f'Bayesian ({bayesian_model_type})'}")
    print(f"   🧠 Bayesian Model Type: {bayesian_model_type}")
    print(f"   🔧 LLM Available: {LLM_AVAILABLE}")
    if (use_llm_for_screening or use_llm_for_optimization) and not LLM_AVAILABLE:
        print(f"   ⚠️  WARNING: LLM requested but not available - will fallback to traditional methods")
    
    if not SIMULATE and SLACK_AVAILABLE:
        slack_agent.send_slack_message(f"Starting new modular calibration experiment with {LIQUID}")
    
    # Track results and optimal conditions
    completed_volumes = []  # (volume, params) pairs that passed precision test
    all_results = []  # All optimization trials (not precision test measurements)
    raw_measurements = []  # All individual measurements
    optimal_conditions = []  # Final optimal conditions for each volume
    blacklisted_params = []  # Parameters that failed precision tests
    # Note: criteria will be calculated per volume using get_volume_dependent_tolerances()
    trial_count = 0
    
    # Create single output directory for entire experiment
    simulate_suffix = "_simulate" if SIMULATE else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S" + f"_{LIQUID}{simulate_suffix}")
    autosave_dir = os.path.join(BASE_AUTOSAVE_DIR, timestamp)
    os.makedirs(autosave_dir, exist_ok=True)
    autosave_summary_path = os.path.join(autosave_dir, "experiment_summary_autosave.csv")
    autosave_raw_path = os.path.join(autosave_dir, "raw_replicate_data_autosave.csv")
    optimal_conditions_path = os.path.join(autosave_dir, "optimal_conditions.csv")
    
    # Save experiment configuration
    save_experiment_config(autosave_dir, NEW_PIPET_EACH_TIME_SET)
    
    for volume_index, volume in enumerate(VOLUMES):
        print(f"\n{'='*60}")
        print(f"🧪 VOLUME: {volume*1000:.0f}μL")
        print(f"{'='*60}")
        
        # Calculate volume-dependent tolerances
        tolerances = get_volume_dependent_tolerances(volume)
        criteria = {
            'max_deviation_ul': tolerances['deviation_ul'],
            'max_time': tolerances['time_seconds']
        }
        
        # Get optimization configuration for selective parameter optimization
        optimize_params, fixed_params = get_optimization_config(volume_index, completed_volumes, all_results)
        
        # DEBUG: Always show what we're optimizing
        print(f"🔧 OPTIMIZING: {len(optimize_params)} parameters: {optimize_params}")
        if fixed_params:
            print(f"� FIXED: {len(fixed_params)} parameters: {list(fixed_params.keys())}")
            print(f"   Fixed values: {fixed_params}")
        else:
            print(f"🔒 FIXED: No parameters fixed")
        print(f"   USE_SELECTIVE_OPTIMIZATION = {USE_SELECTIVE_OPTIMIZATION}")
        print(f"   completed_volumes = {len(completed_volumes)}")
        print(f"   volume_index = {volume_index}")
        
        # Check if we have enough wells remaining
        min_trials_needed = PRECISION_REPLICATES  # At minimum we need precision test wells
        if trial_count + min_trials_needed > MAX_WELLS:
            print(f"⚠️  SKIPPING: Not enough wells remaining (need ≥{min_trials_needed}, have {MAX_WELLS - trial_count})")
            break
        
        expected_mass = volume * DENSITY_LIQUID
        expected_time = volume * 10.146 + 9.5813
        
        # Create ax_client for this volume
        tip_volume = get_tip_volume_for_volume(lash_e, volume)
        print(f"Using tip volume: {tip_volume} mL for pipetting volume: {volume} mL")
        
        volume_completed = False
        candidate_params = None
        candidate_trial_number = None  # Track which trial the candidate came from
        
        # Always create ax_client with the correct selective optimization configuration
        ax_client = get_recommender().create_model(SEED, INITIAL_SUGGESTIONS, bayesian_batch_size=BATCH_SIZE, 
                                           volume=volume, tip_volume=tip_volume, model_type=bayesian_model_type, 
                                           optimize_params=optimize_params, fixed_params=fixed_params, simulate=SIMULATE,
                                           max_overvolume_percent=max_overvolume_percent)
        
        # Step 1: Determine starting candidate
        if len(completed_volumes) > 0:
            # Use the last successful volume's parameters as starting point
            last_volume, last_params = completed_volumes[-1]
            print(f"🔄 TESTING PREVIOUS SUCCESS: Using parameters from {last_volume*1000:.0f}μL volume")
            candidate_params = last_params
            
            # Find the original trial number for these parameters
            last_optimal = optimal_conditions[-1] if optimal_conditions else {}
            candidate_trial_number = last_optimal.get('trial_number', 'unknown')
        else:
            # First volume - need SOBOL/LLM initial exploration
            screening_method = "LLM" if use_llm_for_screening else "SOBOL"
            print(f"🎲 INITIAL SCREENING: Running {INITIAL_SUGGESTIONS} {screening_method} conditions...")
            
            # Run initial suggestions and collect all results
            screening_candidates = []  # Store all acceptable candidates
            
            # Generate all screening suggestions upfront
            screening_suggestions = []
            if use_llm_for_screening and LLM_AVAILABLE:
                # Use LLM for screening suggestions
                screening_suggestions = get_llm_suggestions(ax_client, INITIAL_SUGGESTIONS, all_results)
            else:
                # Use SOBOL for screening suggestions
                for i in range(INITIAL_SUGGESTIONS):
                    try:
                        params, trial_index = ax_client.get_next_trial()
                        screening_suggestions.append((params, trial_index))
                    except Exception as e:
                        print(f"   Could not generate screening trial {i+1}: {e}")
                        break
            
            for i, (params, trial_index) in enumerate(screening_suggestions):
                if trial_count >= MAX_WELLS - PRECISION_REPLICATES:
                    break
                    
                check_if_measurement_vial_full(lash_e, state)
                result = pipet_and_measure(lash_e, 'liquid_source', state["measurement_vial_name"], volume, params, 
                                         expected_mass, expected_time, REPLICATES, SIMULATE, autosave_raw_path, 
                                         raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET)
                
                # Get the most recent measurement for display
                recent_mass = raw_measurements[-1]['mass'] if raw_measurements else expected_mass
                recent_volume = raw_measurements[-1]['calculated_volume'] if raw_measurements else volume
                
                # Check if this trial meets criteria - convert % deviation to absolute μL
                deviation_pct = result.get('deviation', float('inf'))
                absolute_deviation_ul = (deviation_pct / 100) * (volume * 1000)  # Convert to μL
                meets_criteria = (absolute_deviation_ul <= criteria['max_deviation_ul'] and 
                                result.get('time', float('inf')) <= criteria['max_time'])
                status = "✅ CANDIDATE" if meets_criteria else "❌ reject"
                print(f"   Trial {i+1}/{INITIAL_SUGGESTIONS}: {recent_mass:.4f}g → {recent_volume*1000:.1f}μL, {result.get('deviation', 'N/A'):.1f}% dev, {result.get('time', 'N/A'):.0f}s - {status}")
                
                get_recommender().add_result(ax_client, trial_index, result)
                result.update(params)
                result.update({"volume": volume, "trial_index": trial_index, "strategy": screening_method, "liquid": LIQUID, "time_reported": datetime.now().isoformat()})
                result = strip_tuples(result)
                all_results.append(result)
                if not SIMULATE:
                    pd.DataFrame([result]).to_csv(autosave_summary_path, mode='a', index=False, header=not os.path.exists(autosave_summary_path))
                
                trial_count += 1
                
                # Collect acceptable candidates
                if meets_criteria:
                    candidate_info = {
                        'params': params.copy(),
                        'deviation': result.get('deviation', float('inf')),
                        'time': result.get('time', float('inf')),
                        'score': result.get('deviation', float('inf')),  # Use deviation as primary ranking criterion
                        'trial_number': trial_count  # Track which trial this came from
                    }
                    screening_candidates.append(candidate_info)
            
            # Choose the BEST candidate from acceptable ones
            if screening_candidates:
                # Sort by deviation (lower is better), then by time if tied
                best_candidate = min(screening_candidates, key=lambda x: (x['deviation'], x['time']))
                candidate_params = best_candidate['params']
                candidate_trial_number = best_candidate['trial_number']
                print(f"   ✅ Selected BEST candidate from trial #{candidate_trial_number}: {best_candidate['deviation']:.1f}% deviation, {best_candidate['time']:.0f}s")
                print(f"   📊 {len(screening_candidates)}/{INITIAL_SUGGESTIONS} {screening_method} trials met criteria")
            else:
                print(f"   ❌ No {screening_method} trials met criteria - will need optimization")
            
            print(f"   ✅ Completed {INITIAL_SUGGESTIONS} {screening_method} conditions")
        
        # Step 2: Optimization loop until we have a successful precision test
        while not volume_completed and trial_count < MAX_WELLS - PRECISION_REPLICATES:
            
            # If we don't have a candidate yet, we need to optimize
            if candidate_params is None:
                print(f"🔍 OPTIMIZATION: Finding acceptable parameters (target: ≤{criteria['max_deviation_ul']:.0f}μL deviation, ≤{criteria['max_time']:.0f}s)")
                
                # Only load historical data from OTHER volumes, not current volume data (which is already in ax_client)
                if all_results and USE_HISTORICAL_DATA_FOR_OPTIMIZATION:
                    # Filter out results from current volume to avoid double-loading
                    historical_results = [r for r in all_results if r.get('volume') != volume]
                    if historical_results:
                        print(f"Loading {len(historical_results)} trials from previous volumes (excluding {len(all_results) - len(historical_results)} from current volume)")
                        load_previous_data_into_model(ax_client, historical_results)
                    else:
                        print(f"No historical data from other volumes to load")
                elif all_results and not USE_HISTORICAL_DATA_FOR_OPTIMIZATION:
                    print(f"Skipping historical data loading (USE_HISTORICAL_DATA_FOR_OPTIMIZATION = False)")
                    print(f"Each volume will optimize independently without using data from previous volumes")
                
                # Get suggestions and test them
                optimization_found = False
                while not optimization_found and trial_count < MAX_WELLS - PRECISION_REPLICATES:
                    try:
                        if use_llm_for_optimization and LLM_AVAILABLE:
                            # Use LLM for optimization suggestions
                            suggestions = get_llm_suggestions(ax_client, BATCH_SIZE, all_results)
                        else:
                            # Use Bayesian optimization
                            suggestions = get_recommender().get_suggestions(ax_client, volume, n=BATCH_SIZE)
                    except Exception:
                        suggestions = get_fallback_suggestions(ax_client, all_results, volume, BATCH_SIZE)
                    
                    if not suggestions:
                        print("   ❌ No more suggestions available")
                        break
                    
                    for params, trial_index in suggestions:
                        if trial_count >= MAX_WELLS - PRECISION_REPLICATES:
                            break
                            
                        # Skip if parameters are blacklisted
                        if params_are_blacklisted(params, blacklisted_params):
                            print(f"   ⚫ SKIPPING: Parameters blacklisted (failed previous precision test)")
                            continue
                        
                        check_if_measurement_vial_full(lash_e, state)
                        result = pipet_and_measure(lash_e, 'liquid_source', state["measurement_vial_name"], volume, params, 
                                                 expected_mass, expected_time, REPLICATES, SIMULATE, autosave_raw_path, 
                                                 raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET)
                        
                        # Get the most recent measurement for display
                        recent_mass = raw_measurements[-1]['mass'] if raw_measurements else expected_mass
                        recent_volume = raw_measurements[-1]['calculated_volume'] if raw_measurements else volume
                        
                        # Check if this trial meets criteria - convert % deviation to absolute μL
                        deviation_pct = result.get('deviation', float('inf'))
                        absolute_deviation_ul = (deviation_pct / 100) * (volume * 1000)  # Convert to μL
                        meets_criteria = (absolute_deviation_ul <= criteria['max_deviation_ul'] and 
                                        result.get('time', float('inf')) <= criteria['max_time'])
                        status = "✅ CANDIDATE" if meets_criteria else "❌ reject"
                        print(f"   Optimization trial: {recent_mass:.4f}g → {recent_volume*1000:.1f}μL, {result.get('deviation', 'N/A'):.1f}% dev, {result.get('time', 'N/A'):.0f}s - {status}")
                        
                        get_recommender().add_result(ax_client, trial_index, result)
                        result.update(params)
                        optimization_strategy = "LLM" if use_llm_for_optimization else "Bayesian"
                        result.update({"volume": volume, "trial_index": trial_index, "strategy": optimization_strategy, "liquid": LIQUID, "time_reported": datetime.now().isoformat()})
                        result = strip_tuples(result)
                        all_results.append(result)
                        if not SIMULATE:
                            pd.DataFrame([result]).to_csv(autosave_summary_path, mode='a', index=False, header=not os.path.exists(autosave_summary_path))
                        
                        trial_count += 1
                        
                        if meets_criteria:
                            candidate_params = params
                            candidate_trial_number = trial_count
                            optimization_found = True
                            print(f"   ✅ FOUND ACCEPTABLE CANDIDATE from trial #{trial_count}!")
                            break
                
                if candidate_params is None:
                    print(f"❌ Could not find acceptable parameters for {volume*1000:.0f}μL within well limit")
                    break  # Move to next volume
            
            # Step 3: Precision test
            print(f"\n🎯 PRECISION TEST: Testing candidate parameters...")
            
            if trial_count + PRECISION_REPLICATES > MAX_WELLS:
                print(f"⚠️ Not enough wells remaining for precision test ({MAX_WELLS - trial_count} left)")
                break
            
            passed, precision_measurements, precision_times = run_precision_test(lash_e, state, candidate_params, volume, expected_mass, expected_time, autosave_raw_path, raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET, tolerances['variation_ul'])
            trial_count += len(precision_measurements)
            
            if passed:
                # SUCCESS! 
                completed_volumes.append((volume, candidate_params))
                
                # Calculate actual performance metrics from precision test measurements
                avg_obtained_volume = np.mean(precision_measurements) if precision_measurements else volume
                
                # Calculate deviation: average difference from target volume in %
                deviations = [abs(measurement - volume) / volume * 100 for measurement in precision_measurements]
                avg_deviation_percent = np.mean(deviations)
                
                # Calculate variability: standard deviation of measurements in %
                std_volume = np.std(precision_measurements)
                variability_percent = (std_volume / volume) * 100  # CV as percentage
                
                # Calculate average time from precision test measurements
                avg_time_seconds = np.mean(precision_times) if precision_times and len(precision_times) > 0 else None
                
                optimal_condition = {
                    'target_volume_mL': volume,
                    'average_obtained_volume_mL': avg_obtained_volume,
                    'deviation_percent': avg_deviation_percent,  # Average deviation from target in %
                    'time_seconds': avg_time_seconds,            # Average time from precision test
                    'variability_percent': variability_percent, # Standard deviation as % of target
                    'trial_number': candidate_trial_number,     # Actual trial/well number where condition was discovered
                    'precision_replicates': len(precision_measurements),
                    **candidate_params  # Add all parameter values
                }
                
                optimal_conditions.append(optimal_condition)
                save_optimal_conditions(optimal_conditions, optimal_conditions_path)
                
                print(f"\n🎉 VOLUME {volume*1000:.0f}μL: ✅ COMPLETED")
                print(f"   Precision test PASSED - all {len(precision_measurements)} replicates within ±{tolerances['variation_ul']:.0f}μL range")
                volume_completed = True
                
            else:
                # FAILED! Blacklist these parameters and try again
                print(f"\n❌ PRECISION TEST FAILED - blacklisting parameters and finding new candidate")
                blacklisted_params.append(candidate_params.copy())
                candidate_params = None  # Force new optimization
        
        if not volume_completed:
            print(f"\n⚠️  VOLUME {volume*1000:.0f}μL: Could not complete within well limit ({MAX_WELLS - trial_count} wells remaining)")
        
        print(f"Wells used: {trial_count}/{MAX_WELLS}")
        
        if trial_count >= MAX_WELLS:
            print(f"Reached maximum wells ({MAX_WELLS}), stopping experiment.")
            break
    # Wrap up
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.return_vial_home(state["measurement_vial_name"])
    lash_e.nr_robot.move_home()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE - {LIQUID.upper()} CALIBRATION")
    print(f"{'='*60}")
    print(f"Total trials used: {trial_count}/{MAX_WELLS}")
    print(f"Volumes attempted: {len(VOLUMES)}")
    print(f"Volumes completed: {len(completed_volumes)}")
    print(f"Parameters blacklisted: {len(blacklisted_params)} (failed precision tests)")
    
    if completed_volumes:
        print(f"\n✅ VOLUMES THAT PASSED PRECISION TEST:")
        for i, (vol, params) in enumerate(completed_volumes):
            optimal = optimal_conditions[i] if i < len(optimal_conditions) else {}
            deviation = optimal.get('deviation_percent')
            time_s = optimal.get('time_seconds')
            variability = optimal.get('variability_percent')
            avg_vol = optimal.get('average_obtained_volume_mL', vol) * 1000  # Convert to μL
            replicates = optimal.get('precision_replicates', PRECISION_REPLICATES)
            
            # Format values with proper None handling
            deviation_str = f"{deviation:.1f}%" if deviation is not None else "N/A"
            time_str = f"{time_s:.0f}s" if time_s is not None else "N/A"
            variability_str = f"{variability:.1f}%" if variability is not None else "N/A"
            
            print(f"  {vol*1000:.0f}μL → {avg_vol:.1f}μL (deviation: {deviation_str}, variability: {variability_str}, time: {time_str}, n={replicates})")
    else:
        print(f"\n❌ No volumes successfully completed calibration and precision test")
    
    if optimal_conditions:
        print(f"\n📁 OPTIMAL CONDITIONS SAVED:")
        print(f"   File: {optimal_conditions_path}")
        print(f"   Contains: {len(optimal_conditions)} volume(s) with optimized parameters")
        print(f"   Use this file for future pipetting with these volumes!")
    
    # Show detailed results for each volume
    results_df = pd.DataFrame(all_results)
    print(f"\n📊 WHAT ACTUALLY HAPPENED - STEP BY STEP:")
    print(f"🔍 DEBUG: all_results contains {len(all_results)} trials")
    if len(all_results) > 0:
        volumes_in_results = [r.get('volume') for r in all_results]
        unique_volumes = list(set(volumes_in_results))
        print(f"🔍 DEBUG: Volumes in all_results: {unique_volumes}")
        print(f"🔍 DEBUG: Volume counts: {pd.Series(volumes_in_results).value_counts().to_dict()}")
    
    # Count precision measurements from raw data
    precision_measurements_count = len([m for m in raw_measurements if m.get('replicate', -1) in range(PRECISION_REPLICATES)])
    optimization_trials_count = len(results_df)
    
    for volume in VOLUMES:
        vol_results = results_df[results_df['volume'] == volume] if 'volume' in results_df.columns else []
        volume_completed = any(v[0] == volume for v in completed_volumes)
        
        print(f"\n🎯 Volume {volume*1000:.0f}μL:")
        
        if len(vol_results) > 0:
            # Count trials for this volume
            trials_count = len(vol_results)
            best = vol_results.loc[vol_results['deviation'].idxmin()]
            
            print(f"   📈 OPTIMIZATION PHASE:")
            print(f"      • Total optimization trials: {trials_count}")
            print(f"      • Best result: {best['deviation']:.1f}% deviation, {best['time']:.1f}s")
            
            if volume_completed:
                print(f"   ✅ PRECISION TEST PHASE:")
                print(f"      • Precision test: PASSED (all {PRECISION_REPLICATES} measurements within ±{tolerances['variation_ul']:.0f}μL)")
                print(f"      • Status: ✅ VOLUME COMPLETED & SAVED TO OPTIMAL CONDITIONS")
            else:
                print(f"   ❌ PRECISION TEST PHASE:")
                print(f"      • Precision test: FAILED (parameters blacklisted)")
                print(f"      • Status: ❌ VOLUME NOT COMPLETED")
        else:
            print(f"   ⚠️  NO TRIALS RUN (insufficient wells remaining)")
    
    print(f"\n📋 EXPERIMENT SUMMARY:")
    print(f"   • Total optimization trials: {optimization_trials_count}")
    print(f"   • Total precision measurements: {precision_measurements_count}")
    print(f"   • Volumes completed: {len(completed_volumes)}/{len(VOLUMES)}")
    print(f"   • Parameters blacklisted: {len(blacklisted_params)}")
    
    print(f"\n🔍 WHAT THESE TERMS MEAN:")
    print(f"   • 'Optimization trials': Parameter combinations tested to find acceptable conditions")
    print(f"   • 'Acceptable': Deviation ≤{criteria['max_deviation_ul']:.0f}μL AND time ≤{criteria['max_time']:.0f}s")
    print(f"   • 'Precision test': {PRECISION_REPLICATES} replicates using candidate parameters, all must be within tolerance of target")
    print(f"   • 'Blacklisted': Parameters that failed precision test and will never be tried again")
    print(f"   • 'Optimal conditions': Final successful parameters with precision test performance metrics")
    print(f"      - 'deviation': Average difference from target volume (% of target)")
    print(f"      - 'variability': Standard deviation of measurements (% of target)")  
    print(f"      - 'time': Average pipetting time from precision test replicates")
    
    print(f"\n📋 RAW DATA FILES:")
    print(f"   📊 OPTIMIZATION TRIALS TABLE ({optimization_trials_count} rows): {autosave_summary_path}")
    print(f"      • Contains all parameter combinations tested during optimization")
    print(f"      • Each row = 1 well used for finding good parameters")
    
    print(f"\n   🎯 PRECISION TEST MEASUREMENTS ({precision_measurements_count} measurements): {autosave_raw_path}")
    print(f"      • Contains individual replicate measurements from precision tests")
    print(f"      • Each measurement = 1 well used for validation")
    
    if raw_measurements and precision_measurements_count > 0:
        # Show sample precision measurements
        precision_raw = [m for m in raw_measurements if m.get('replicate', -1) in range(PRECISION_REPLICATES)]
        print(f"      Sample precision measurements (first 5):")
        for i, measurement in enumerate(precision_raw[:5]):
            volume_ul = measurement.get('mass', 0) / 1.26 * 1000  # Convert to μL
            replicate_num = measurement.get('replicate', 'unknown')
            target_vol = measurement.get('volume', 0) * 1000  # Convert to μL
            print(f"        Volume {target_vol:.0f}μL, Replicate {replicate_num}: {measurement.get('mass', 0):.4f}g = {volume_ul:.1f}μL")
    
    print(f"\n   🧮 WELL COUNT VERIFICATION:")
    print(f"      Optimization trials: {optimization_trials_count} wells")
    print(f"      Precision measurements: {precision_measurements_count} wells")  
    print(f"      Total: {optimization_trials_count + precision_measurements_count} wells (should equal {trial_count})")
    
    # Log results without Unicode characters to avoid encoding issues
    try:
        lash_e.logger.info(f"Experiment completed with {len(results_df)} total trials across {len(completed_volumes)} volumes")
        lash_e.logger.info(f"Results summary: {len(completed_volumes)}/{len(VOLUMES)} volumes completed successfully")
    except Exception as e:
        print(f"Warning: Could not log results due to encoding issue: {e}")
    
    # Save analysis results (both simulation and real data)
    # Only generate scatter plot and SHAP analysis by default
    save_analysis(
        results_df,
        pd.DataFrame(raw_measurements),
        autosave_dir,
        include_shap=False,
        include_scatter=True,
        include_boxplots=False,
        include_pairplot=False,
        include_learning_curves=True,
        include_improvement=False,
        include_top_trials=False,
        optimal_conditions=optimal_conditions,
        learning_curve_metrics=['deviation','time']
    )
    
    if not SIMULATE and SLACK_AVAILABLE:
        try:
            # Build detailed summary of completed volumes
            completed_list = [f"{int(v*1000)}µL" for v, _ in completed_volumes]
            remaining_vols = [v for v in VOLUMES if v not in [cv for cv, _ in completed_volumes]]
            remaining_list = [f"{int(v*1000)}µL" for v in remaining_vols]
            completed_str = ", ".join(completed_list) if completed_list else "None"
            remaining_str = ", ".join(remaining_list) if remaining_list else "None"

            # Extract quick performance snapshot for each completed volume (deviation/time if available)
            performance_lines = []
            if 'deviation' in results_df.columns and 'time' in results_df.columns and 'volume' in results_df.columns:
                # For each completed volume, get best (lowest absolute deviation) trial summary
                for vol, _ in completed_volumes:
                    sub = results_df[results_df['volume'] == vol]
                    if not sub.empty and 'deviation' in sub and 'time' in sub:
                        best_row = sub.iloc[(sub['deviation']).abs().argmin()]
                        performance_lines.append(f"{int(vol*1000)}µL: dev={best_row['deviation']:.1f}%, time={best_row['time']:.1f}s")
            perf_block = "; ".join(performance_lines) if performance_lines else "(no trial metrics captured)"

            slack_msg = (
                f"Modular calibration with {LIQUID} COMPLETE\n"
                f"Volumes completed: {len(completed_volumes)}/{len(VOLUMES)} -> {completed_str}\n"
                f"Remaining (not calibrated): {remaining_str}\n"
                f"Performance snapshot: {perf_block}"
            )
            slack_agent.send_slack_message(slack_msg)
        except Exception as e:
            print(f"Warning: Failed to send detailed Slack summary: {e}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

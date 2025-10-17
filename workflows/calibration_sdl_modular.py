# calibration_sdl_modular.py
import sys
import os
#asd
sys.path.append("../North-Cytation")

from calibration_sdl_base import *
from master_usdl_coordinator import Lash_E
import recommenders.pipeting_optimizer_v2 as recommender_v2
import recommenders.pipetting_optimizer_v3 as recommender_v3
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
PRECISION_REPLICATES = 4
# Volume generation configuration
# MIN_VOLUME_ML = 0.3      # Minimum volume in mL
# MAX_VOLUME_ML = 1.0      # Maximum volume in mL  
# NUM_VOLUMES = 3          # Number of volumes to test
VOLUMES = [0.05, 0.025, 0.01]  # Manually specified volume list (in mL)
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

# Criteria (For real life testing) - Volume-dependent relative percentage tolerances
# Based on pipetting accuracy standards: relative bias and CV thresholds
BASE_TIME_SECONDS = 20  # Base time in seconds for optimization acceptance (cutoff)... Should probably calculate from viscosity. Eg 20s for water, 60s for glycerol... Can this be automated?
TIME_SCALING_FACTOR = 2.5  # +2.5 seconds per 100 μL above baseline (used for first volume only)
TIME_BUFFER_FRACTION = 0.1  # Buffer fraction: optimal_time = base_time * (1 - buffer)
# ADAPTIVE_TIME_SCALING removed - time scaling only used for first volume, no adaptive updates needed
TIME_TRANSITION_MODE = "asymmetric"  # Options: "relu" (max(0,x)), "smooth" (log(1+exp(x))), "asymmetric" (gentle penalty for fast times)

# Relative percentage tolerances (applies to both optimization and precision test)
# Volume ranges defined as (min_volume_ul, max_volume_ul, tolerance_pct)
# Updated volume tolerance ranges with smoother transition around 100µL
# Uses gradual scaling instead of sharp cutoffs to avoid excessive failure at boundary volumes
VOLUME_TOLERANCE_RANGES = [
    {'min_ul': 200, 'max_ul': 1000, 'tolerance_pct': 1.0, 'name': 'large_volume'},   # ≥200µL: 1%
    {'min_ul': 50,  'max_ul': 200,  'tolerance_pct': 2.0, 'name': 'medium_large_volume'}, # 50-199µL: 2%
    {'min_ul': 10,  'max_ul': 50,   'tolerance_pct': 3.0, 'name': 'medium_volume'}, # 10-49µL: 3%  
    {'min_ul': 1,   'max_ul': 10,   'tolerance_pct': 5.0, 'name': 'small_volume'},  # 1-9µL: 5%
    {'min_ul': 0,   'max_ul': 1,    'tolerance_pct': 10.0, 'name': 'micro_volume'}, # <1µL: 10% (fallback)
]

# Selective parameter optimization config
# Max overaspirate: Base volume + percentage scaling
OVERASPIRATE_BASE_UL = 5.0        # Base overaspirate volume in microliters
OVERASPIRATE_SCALING_PERCENT = 5.0  # Additional percentage of total volume

# Auto-calibration of overvolume parameters
AUTO_CALIBRATE_OVERVOLUME = True  # Enable automatic overvolume calibration after SOBOL trials
OVERVOLUME_CALIBRATION_BUFFER_UL = 2.0  # Buffer to add above fitted line (μL)
OVERVOLUME_MAX_BASE_UL = 50.0     # Maximum allowed base overvolume (μL)
OVERVOLUME_MAX_PERCENT = 100.0    # Maximum allowed percentage scaling (%)

USE_SELECTIVE_OPTIMIZATION = True  # Enable selective parameter optimization
USE_HISTORICAL_DATA_FOR_OPTIMIZATION = False  # Load data from previous volumes into optimizer
VOLUME_DEPENDENT_PARAMS = ["blowout_vol", "overaspirate_vol"]  # Parameters to optimize for each volume
ALL_PARAMS = ["aspirate_speed", "dispense_speed", "aspirate_wait_time", "dispense_wait_time", 
              "retract_speed", "blowout_vol", "post_asp_air_vol", "overaspirate_vol"]

if SIMULATE:
    DEFAULT_LOCAL_AUTOSAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'calibration_runs'))
    os.makedirs(DEFAULT_LOCAL_AUTOSAVE_DIR, exist_ok=True)
    BASE_AUTOSAVE_DIR = os.environ.get('CALIBRATION_AUTOSAVE_DIR', DEFAULT_LOCAL_AUTOSAVE_DIR)
    print(f"[info] Using BASE_AUTOSAVE_DIR={BASE_AUTOSAVE_DIR}")
else:
    BASE_AUTOSAVE_DIR='C:\\Users\\Imaging Controller\\Desktop\\Calibration_SDL_Output\\New_Method'

# --- Helper Methods ---
def generate_volumes(min_vol_ml, max_vol_ml, num_volumes):
    """Generate volume sequence with GCD-based rounding and optimal sequencing.
    
    Args:
        min_vol_ml: Minimum volume in mL
        max_vol_ml: Maximum volume in mL  
        num_volumes: Number of volumes to generate
        
    Returns:
        list: Volumes in optimal testing sequence (middle → high → remaining)
    """
    import math
    
    if num_volumes < 2:
        return [min_vol_ml]
    if num_volumes == 2:
        return [max_vol_ml, min_vol_ml]  # Start with harder case
    
    # Convert to μL for easier GCD calculation
    min_vol_ul = int(min_vol_ml * 1000)
    max_vol_ul = int(max_vol_ml * 1000)
    
    # Calculate GCD for common denominator
    gcd = math.gcd(min_vol_ul, max_vol_ul)
    
    # Generate evenly spaced volumes
    range_ul = max_vol_ul - min_vol_ul
    spacing_ul = range_ul / (num_volumes - 1)
    
    volumes_ul = []
    for i in range(num_volumes):
        vol_ul = min_vol_ul + (i * spacing_ul)
        # Round down to nearest GCD multiple
        vol_ul_rounded = (int(vol_ul) // gcd) * gcd
        volumes_ul.append(vol_ul_rounded)
    
    # Ensure endpoints are exact
    volumes_ul[0] = min_vol_ul
    volumes_ul[-1] = max_vol_ul
    
    # Convert back to mL
    volumes_ml = [vol / 1000.0 for vol in volumes_ul]
    
    # Sort by optimal testing sequence: middle → high → low → remaining
    if len(volumes_ml) >= 3:
        middle_idx = len(volumes_ml) // 2
        sequence = [volumes_ml[middle_idx]]  # Start with middle
        sequence.append(volumes_ml[-1])      # Then highest  
        sequence.extend([vol for i, vol in enumerate(volumes_ml) if i != middle_idx and i != len(volumes_ml)-1])
    else:
        sequence = volumes_ml[::-1]  # Just reverse for 2 volumes
    
    return sequence

def get_max_overaspirate_ul(volume_ml):
    """Calculate maximum overaspirate volume based on base + percentage scaling."""
    volume_ul = volume_ml * 1000  # Convert to μL
    scaling_volume = volume_ul * (OVERASPIRATE_SCALING_PERCENT / 100.0)
    max_overaspirate = OVERASPIRATE_BASE_UL + scaling_volume
    
    # Safety check: ensure minimum range for optimization
    # Ax requires upper bound > lower bound, so we need at least 1μL range
    min_overaspirate = 1.0  # Minimum 1μL to ensure valid parameter range
    if max_overaspirate < min_overaspirate:
        print(f"   ⚠️  Calculated max overaspirate ({max_overaspirate:.1f}μL) too low, using minimum ({min_overaspirate:.1f}μL)")
        max_overaspirate = min_overaspirate
    
    return max_overaspirate

def get_volume_dependent_tolerances(volume_ml, is_first_volume=True):
    """Calculate volume-dependent tolerances based on scalable volume ranges.
    
    Args:
        volume_ml: Volume in milliliters
        is_first_volume: If True, include time constraints and optimization.
                        If False, exclude time criteria (subsequent volumes).
    """
    volume_ul = volume_ml * 1000  # Convert to μL
    
    # Find the appropriate tolerance range for this volume
    tolerance_pct = None
    range_name = 'unknown'
    
    for vol_range in VOLUME_TOLERANCE_RANGES:
        if vol_range['min_ul'] <= volume_ul < vol_range['max_ul']:
            tolerance_pct = vol_range['tolerance_pct']
            range_name = vol_range['name']
            break
    
    # Fallback if no range matched (shouldn't happen with properly defined ranges)
    if tolerance_pct is None:
        tolerance_pct = 10.0  # Default to 10% for safety
        range_name = 'fallback'
    
    # Convert percentage to absolute μL tolerance (same for both deviation and variation)
    tolerance_ul = volume_ul * (tolerance_pct / 100.0)
    
    # Time scaling only for first volume
    if is_first_volume:
        volume_excess_ul = max(0, volume_ul - 100)  # Only scale above 100μL baseline
        scaling_factor = volume_excess_ul / 100  # Per 100μL scaling
        time_seconds = BASE_TIME_SECONDS + (TIME_SCALING_FACTOR * scaling_factor)
        
        # Calculate optimal time as a fraction of the cutoff time (automatic buffer)
        time_optimal_target = time_seconds * (1 - TIME_BUFFER_FRACTION)
    else:
        # No time constraints for subsequent volumes
        time_seconds = None
        time_optimal_target = None
    
    # In simulation we can relax tolerances if needed
    if SIMULATE:
        try:
            # Allow environment override - use multipliers on the standard tolerances
            dev_multiplier = float(os.environ.get('SIM_DEV_MULTIPLIER', '2.0'))  # default 2x more lenient
            var_multiplier = float(os.environ.get('SIM_VAR_MULTIPLIER', '2.0'))  # default 2x more lenient  
            time_multiplier = float(os.environ.get('SIM_TIME_MULTIPLIER', '1.5'))  # default 1.5x more time
        except ValueError:
            dev_multiplier, var_multiplier, time_multiplier = 10.0, 10.0, 3.0
        
        deviation_ul = tolerance_ul * dev_multiplier
        variation_ul = tolerance_ul * var_multiplier
        if is_first_volume and time_seconds is not None:
            time_seconds = time_seconds * time_multiplier
            time_optimal_target = time_seconds * (1 - TIME_BUFFER_FRACTION)
    else:
        deviation_ul = tolerance_ul
        variation_ul = tolerance_ul

    tolerances = {
        'deviation_ul': deviation_ul,
        'variation_ul': variation_ul,
        'tolerance_percent': tolerance_pct,  # For reference/logging
        'range_name': range_name             # Which range was used
    }
    
    # Only include time criteria for first volume
    if is_first_volume:
        tolerances['time_optimal_target'] = time_optimal_target
        tolerances['time_seconds'] = time_seconds
    
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
        # Use parameters from the FIRST successful volume (most conservative/transferable)
        _, first_successful_params = completed_volumes[0]
        fixed_params = {k: v for k, v in first_successful_params.items() 
                      if k not in VOLUME_DEPENDENT_PARAMS}
        print(f"   Using parameters from FIRST successful volume {completed_volumes[0][0]*1000:.0f}μL (most transferable)")
        if len(completed_volumes) > 1:
            print(f"   📊 Strategy: Using conservative baseline instead of latest optimized parameters")
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
                if col == 'time' and col in result and result[col] is not None:
                    # Compute time_score for historical data using same method as current experiment
                    raw_time = float(result[col])
                    result_volume = result.get('volume', 0.1)  # Default to 100μL if not specified
                    volume_tolerances = get_volume_dependent_tolerances(result_volume, is_first_volume=True)  # Historical data includes time
                    scaled_optimal_target = volume_tolerances.get('time_optimal_target')
                    
                    if scaled_optimal_target is not None:
                        import numpy as np
                        if TIME_TRANSITION_MODE == "smooth":
                            time_score = np.log(1 + np.exp(raw_time - scaled_optimal_target))
                        elif TIME_TRANSITION_MODE == "asymmetric":
                            if raw_time < scaled_optimal_target:
                                low_time_penalty_factor = 0.1  # Match optimizer setting
                                time_score = (scaled_optimal_target - raw_time) * low_time_penalty_factor
                            else:
                                time_score = max(0, raw_time - scaled_optimal_target)
                        else:  # "relu"
                            time_score = max(0, raw_time - scaled_optimal_target)
                    else:
                        time_score = 0  # No time optimization for this data point
                    
                    raw_data[col] = (time_score, 0.0)  # (mean, sem)
                elif col in result and result[col] is not None:
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
            max_pct = 100.0  # Extremely lenient percent threshold for simulation
            try:
                max_pct = float(os.environ.get('SIM_MAX_DEV_PCT', '100'))
            except ValueError:
                pass
            
            # Check if there's a time constraint (only for first volume)
            if 'max_time' in criteria:
                relaxed_time = criteria['max_time'] * 3.0  # Much more lenient time
                meets_criteria = ((df['deviation'] <= max_pct) | (absolute_deviation_ul <= criteria['max_deviation_ul'])) & (df['time'] <= relaxed_time)
            else:
                # No time constraint for subsequent volumes
                meets_criteria = (df['deviation'] <= max_pct) | (absolute_deviation_ul <= criteria['max_deviation_ul'])
        else:
            # Real mode
            if 'max_time' in criteria:
                meets_criteria = (absolute_deviation_ul <= criteria['max_deviation_ul']) & (df['time'] <= criteria['max_time'])
            else:
                # No time constraint for subsequent volumes
                meets_criteria = (absolute_deviation_ul <= criteria['max_deviation_ul'])

        if not meets_criteria.any():
            # Debug first few rows
            sample = df.head(3)[['volume','deviation','time']].to_dict(orient='records')
            print(f"[criteria-debug] No trial meets criteria yet. Example rows: {sample} | criteria={criteria} | SIMULATE={SIMULATE}")
        return meets_criteria.any()
    return False

def calibrate_overvolume_parameters(screening_candidates, remaining_volumes, lash_e, state, autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set, criteria, autosave_dir=None):
    """
    Automatically calibrate overvolume base and scaling parameters using multi-volume testing.
    
    Args:
        screening_candidates: List of acceptable candidates from SOBOL trials
        remaining_volumes: List of volumes to test (excluding first volume)
        lash_e, state, etc.: Standard experiment parameters
        criteria: Criteria dict with time cutoff
        
    Returns:
        tuple: (new_base_ul, new_scaling_percent, calibration_data) or (None, None, None) if failed
    """
    if not screening_candidates or not remaining_volumes:
        print("⚠️  OVERVOLUME CALIBRATION: No candidates or volumes to test - skipping")
        return None, None, None
    
    # Step 1: Select best candidate (lowest deviation with time > cutoff)
    print(f"\n🔬 OVERVOLUME CALIBRATION: Selecting candidate from {len(screening_candidates)} options...")
    
    # Filter candidates that meet time criteria
    time_cutoff = criteria.get('max_time', float('inf'))
    valid_candidates = [c for c in screening_candidates if c.get('time', 0) >= time_cutoff]
    
    if not valid_candidates:
        print(f"   ⚠️  No candidates with time >= {time_cutoff:.1f}s - using all candidates")
        valid_candidates = screening_candidates
    
    # Select candidate with lowest deviation
    best_candidate = min(valid_candidates, key=lambda x: x.get('deviation', float('inf')))
    best_params = best_candidate['params']
    
    print(f"   ✅ Selected candidate: {best_candidate['deviation']:.1f}% deviation, {best_candidate['time']:.1f}s")
    print(f"   📋 Testing these parameters on {len(remaining_volumes)} additional volumes...")
    
    # Step 2: Start with the first volume's data from the selected candidate
    calibration_data = []
    
    # Add the first volume's measurement data (the volume that was tested in SOBOL)
    # The first volume is the one NOT in remaining_volumes
    first_volume = [v for v in VOLUMES if v not in remaining_volumes][0]  # Should be VOLUMES[0]
    first_volume_ul = first_volume * 1000  # Convert to μL
    deviation_pct = best_candidate['deviation']
    
    # Calculate measured volume from deviation
    # Deviation = (target - measured) / target * 100
    # So: measured = target * (1 - deviation/100)
    measured_volume_ul = first_volume_ul * (1 - deviation_pct / 100)
    
    calibration_data.append({
        'volume_set': first_volume_ul,
        'volume_measured': measured_volume_ul,
        'deviation_pct': deviation_pct,
        'existing_overaspirate_ul': best_params.get('overaspirate_vol', 0) * 1000  # Convert mL to μL
    })
    existing_overaspirate_first = best_params.get('overaspirate_vol', 0) * 1000
    print(f"   📊 Including first volume: {first_volume_ul:.0f}μL → {measured_volume_ul:.1f}μL ({deviation_pct:.1f}% dev, had {existing_overaspirate_first:.1f}μL overaspirate)")
    
    # Step 3: Test best parameters on remaining volumes  
    for volume in remaining_volumes:
        print(f"   🧪 Testing {volume*1000:.0f}μL...", end=" ")
        
        expected_mass = volume * LIQUIDS[liquid]["density"]
        expected_time = volume * 10.146 + 9.5813
        
        check_if_measurement_vial_full(lash_e, state)
        liquid_source = get_liquid_source(lash_e)
        
        # Single measurement (n=1 as specified)
        result = pipet_and_measure(lash_e, liquid_source, state["measurement_vial_name"], 
                                 volume, best_params, expected_mass, expected_time, 
                                 1, SIMULATE, autosave_raw_path, raw_measurements, 
                                 liquid, new_pipet_each_time_set)
        
        # Get actual measured volume from raw_measurements
        if raw_measurements:
            actual_mass = raw_measurements[-1]['mass']
            actual_volume = actual_mass / LIQUIDS[liquid]["density"]  # Convert back to mL
        else:
            actual_volume = volume  # Fallback
        
        calibration_data.append({
            'volume_set': volume * 1000,      # Convert to μL for easier math
            'volume_measured': actual_volume * 1000,  # Convert to μL
            'deviation_pct': result.get('deviation', 0),
            'existing_overaspirate_ul': best_params.get('overaspirate_vol', 0) * 1000  # Store existing overaspirate in μL
        })
        
        print(f"{actual_volume*1000:.1f}μL measured ({result.get('deviation', 0):.1f}% dev)")
    
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
            print(f"     {d['volume_set']:.0f}μL → {d['volume_measured']:.1f}μL (shortfall: {shortfall:.1f}μL, had {existing_over:.1f}μL overaspirate)")
        
        # Fit line to shortfalls: shortfall = slope * volume + intercept
        model = LinearRegression()
        model.fit(x_data, shortfalls)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        print(f"   📈 Shortfall fit: additional_shortfall = {slope:.4f} * volume + {intercept:.2f}")
        
        # CRITICAL: Account for existing overaspirate in total recommendation
        # Total overaspirate needed = existing_overaspirate + additional_shortfall + buffer
        
        # Calculate the average existing overaspirate to add to our formula
        avg_existing_overaspirate = np.mean(existing_overaspirates)
        print(f"   📊 Average existing overaspirate: {avg_existing_overaspirate:.1f}μL")
        
        # Calculate total overaspirate needed (existing + additional + buffer)
        min_overaspirate = 2.0  # Minimum 2μL to prevent crashes
        
        # Total formula: total_overaspirate = avg_existing + shortfall_from_line + buffer  
        # For base+percentage: overaspirate = base + (percentage/100) * volume
        new_base_ul = avg_existing_overaspirate + intercept + OVERVOLUME_CALIBRATION_BUFFER_UL
        new_scaling_percent = slope * 100  # Convert slope to percentage
        
        # Ensure minimum overaspirate for all volumes by adjusting base if needed
        # Test the smallest volume to see if it meets minimum
        smallest_volume = min([d['volume_set'] for d in calibration_data])
        min_overaspirate_at_smallest = new_base_ul + (new_scaling_percent/100) * smallest_volume
        
        if min_overaspirate_at_smallest < min_overaspirate:
            adjustment = min_overaspirate - min_overaspirate_at_smallest
            new_base_ul += adjustment
            print(f"   � Adjusted base by +{adjustment:.1f}μL to ensure minimum {min_overaspirate:.1f}μL overaspirate")
        
        print(f"   🎯 Calibrated formula: overaspirate = {new_base_ul:.1f}μL + {new_scaling_percent:.1f}% * volume")
        
        # Step 4: Apply safety bounds
        if new_base_ul > OVERVOLUME_MAX_BASE_UL:
            print(f"   ⚠️  Base {new_base_ul:.1f}μL exceeds limit {OVERVOLUME_MAX_BASE_UL:.1f}μL - capping")
            new_base_ul = OVERVOLUME_MAX_BASE_UL
            
        if new_scaling_percent > OVERVOLUME_MAX_PERCENT:
            print(f"   ⚠️  Scaling {new_scaling_percent:.1f}% exceeds limit {OVERVOLUME_MAX_PERCENT:.1f}% - capping")
            new_scaling_percent = OVERVOLUME_MAX_PERCENT
            
        if new_base_ul < 0:
            print(f"   ⚠️  Negative base {new_base_ul:.1f}μL - setting to 0")
            new_base_ul = 0
            
        if new_scaling_percent < 0:
            print(f"   ⚠️  Negative scaling {new_scaling_percent:.1f}% - setting to 0")
            new_scaling_percent = 0
        
        # Safety check: ensure at least some overaspirate capability
        # If both base and scaling are 0, we'd have no overaspirate volume at all
        if new_base_ul == 0 and new_scaling_percent == 0:
            print(f"   ⚠️  Both base and scaling are 0 - setting minimum base to 1μL to maintain optimization capability")
            new_base_ul = 1.0
        
        print(f"   ✅ Final calibrated values: base = {new_base_ul:.1f}μL, scaling = {new_scaling_percent:.1f}%")
        
        # Store raw shortfall coefficients for accurate reporting
        for point in calibration_data:
            point['slope'] = slope  # Store raw slope coefficient
            point['intercept'] = intercept  # Store raw intercept coefficient
        
        # Generate calibration plot
        try:
            # Create the plot
            plt.figure(figsize=(10, 8))
            
            # Plot data points
            x_plot = [d['volume_set'] for d in calibration_data]
            y_plot = [d['volume_measured'] for d in calibration_data]
            plt.scatter(x_plot, y_plot, color='blue', s=100, alpha=0.7, label='Measured Data', zorder=3)
            
            # Plot original fitted line
            x_range = np.linspace(min(x_plot), max(x_plot), 100)
            y_original = slope * x_range + intercept
            plt.plot(x_range, y_original, 'r--', alpha=0.6, linewidth=2, label=f'Original Fit: y = {slope:.4f}x + {intercept:.1f}')
            
            # Plot translated line (represents existing overaspirate adjustment)
            adjusted_intercept = intercept + avg_existing_overaspirate
            y_translated = slope * x_range + adjusted_intercept
            plt.plot(x_range, y_translated, 'orange', alpha=0.8, linewidth=2, label=f'With Existing OA: y = {slope:.4f}x + {adjusted_intercept:.1f}')
            
            # Plot final line (with buffer)
            final_intercept = adjusted_intercept + OVERVOLUME_CALIBRATION_BUFFER_UL
            y_final = slope * x_range + final_intercept
            plt.plot(x_range, y_final, 'green', linewidth=3, label=f'Final + Buffer: y = {slope:.4f}x + {final_intercept:.1f}')
            
            # Plot ideal 1:1 line for reference
            ideal_min = min(min(x_plot), min(y_plot))
            ideal_max = max(max(x_plot), max(y_plot))
            plt.plot([ideal_min, ideal_max], [ideal_min, ideal_max], 'k:', alpha=0.5, linewidth=1, label='Ideal 1:1 Line')
            
            # Add data point labels
            for i, (x, y) in enumerate(zip(x_plot, y_plot)):
                deviation = calibration_data[i]['deviation_pct']
                plt.annotate(f'{x:.0f}→{y:.1f}\n({deviation:.1f}%)', 
                           (x, y), xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, alpha=0.8)
            
            # Formatting
            plt.xlabel('Target Volume (μL)', fontsize=12)
            plt.ylabel('Measured Volume (μL)', fontsize=12)
            plt.title(f'Overvolume Calibration Results\nCalibrated: {new_base_ul:.1f}μL + {new_scaling_percent:.1f}%', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=10)
            
            # Equal aspect ratio and reasonable margins
            plt.axis('equal')
            margin = (max(max(x_plot), max(y_plot)) - min(min(x_plot), min(y_plot))) * 0.1
            plt.xlim(min(min(x_plot), min(y_plot)) - margin, max(max(x_plot), max(y_plot)) + margin)
            plt.ylim(min(min(x_plot), min(y_plot)) - margin, max(max(x_plot), max(y_plot)) + margin)
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"overvolume_calibration_{timestamp}.png"
            
            # Try to save in the autosave directory, fallback to output
            try:
                if autosave_dir and os.path.exists(autosave_dir):
                    plot_path = os.path.join(autosave_dir, plot_filename)
                else:
                    # Fallback to output directory
                    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
                    os.makedirs(output_dir, exist_ok=True)
                    plot_path = os.path.join(output_dir, plot_filename)
            except:
                # Final fallback - current directory
                plot_path = plot_filename
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()  # Close to free memory
            
            print(f"   📊 Calibration plot saved: {plot_path}")
            
        except Exception as e:
            print(f"   ⚠️  Could not generate calibration plot: {e}")
        
        return new_base_ul, new_scaling_percent, calibration_data
        
    except Exception as e:
        print(f"   ❌ Calibration failed: {e}")
        return None, None, None

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
        liquid_source = get_liquid_source(lash_e)
        result = pipet_and_measure(lash_e, liquid_source, state["measurement_vial_name"], volume, best_params, expected_mass, expected_time, 1, SIMULATE, autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set)
        
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

def get_liquid_source(lash_e, minimum_volume=2.0):
    """Check liquid_source volume and switch to liquid_source_2 if needed"""
    try:
        current_vol = lash_e.nr_robot.get_vial_info("liquid_source", "vial_volume")
        if current_vol <= minimum_volume:
            lash_e.logger.info(f"[info] liquid_source volume is {current_vol:.1f}mL, switching to liquid_source_2")
            return "liquid_source_2"
        else:
            return "liquid_source"
    except:
        # If liquid_source doesn't exist or error, default to liquid_source_2
        lash_e.logger.info("[info] Using liquid_source_2 as fallback")
        return "liquid_source_2"

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
        'min_volume_ml': min(VOLUMES) if VOLUMES else None,
        'max_volume_ml': max(VOLUMES) if VOLUMES else None,
        'num_volumes': len(VOLUMES),
        'volumes': VOLUMES,
        'max_wells': MAX_WELLS,
        'overaspirate_base_ul': OVERASPIRATE_BASE_UL,
        'overaspirate_scaling_percent': OVERASPIRATE_SCALING_PERCENT,
        'new_pipet_each_time_set': new_pipet_each_time_set,
        'base_time_seconds': BASE_TIME_SECONDS,
        'time_scaling_factor': TIME_SCALING_FACTOR,
        'time_buffer_fraction': TIME_BUFFER_FRACTION,
        'time_transition_mode': TIME_TRANSITION_MODE,
        'volume_tolerance_ranges': VOLUME_TOLERANCE_RANGES,
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

def generate_calibration_report(volume_report_data, volumes, completed_volumes):
    """Generate a comprehensive calibration report with diagnostics and recommendations."""
    
    report_lines = []
    report_lines.append("CALIBRATION REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Overaspirate calibration section (first, most important)
    if AUTO_CALIBRATE_OVERVOLUME:
        report_lines.append("OVERASPIRATE CALIBRATION:")
        report_lines.append("-" * 30)
        
        # Get calibration info from first volume data
        first_volume_data = volume_report_data.get(volumes[0], {}) if volumes else {}
        calibration_info = first_volume_data.get('overvolume_calibration', {})
        
        if calibration_info.get('enabled'):
            if calibration_info.get('failed'):
                report_lines.append("Status: FAILED - using original values")
                report_lines.append(f"Formula: {OVERASPIRATE_BASE_UL:.1f}uL + {OVERASPIRATE_SCALING_PERCENT:.1f}%")
            elif calibration_info.get('skipped'):
                reason = calibration_info.get('reason', 'unknown')
                report_lines.append(f"Status: SKIPPED ({reason})")
                report_lines.append(f"Formula: {OVERASPIRATE_BASE_UL:.1f}uL + {OVERASPIRATE_SCALING_PERCENT:.1f}%")
            else:
                # Successful calibration
                old_base = calibration_info.get('old_base_ul', 'N/A')
                old_scaling = calibration_info.get('old_scaling_percent', 'N/A')
                new_base = calibration_info.get('new_base_ul', OVERASPIRATE_BASE_UL)
                new_scaling = calibration_info.get('new_scaling_percent', OVERASPIRATE_SCALING_PERCENT)
                cal_data = calibration_info.get('calibration_data', [])
                
                report_lines.append("Status: SUCCESS")
                report_lines.append(f"Original formula: {old_base:.1f}uL + {old_scaling:.1f}%")
                
                # Convert to simpler μL + % format for reporting
                # The actual formula accounts for existing overaspirate + buffer, but report as equivalent μL + %
                report_lines.append(f"Calibrated formula: {new_base:.1f} μL + {new_scaling:.1f}%")
                
                if cal_data:
                    report_lines.append(f"Data points used: {len(cal_data)}")
                    # Get the raw coefficients from the first data point (they're all the same)
                    slope = cal_data[0].get('slope', new_scaling/100)  # Fallback to converted value
                    intercept = cal_data[0].get('intercept', new_base - 2.0)  # Fallback to adjusted value
                    
                    for point in cal_data:
                        vol_set = point.get('volume_set', 0)
                        vol_meas = point.get('volume_measured', 0)
                        deviation = point.get('deviation_pct', 0)
                        existing_over = point.get('existing_overaspirate_ul', 0)
                        shortfall = vol_set - vol_meas
                        
                        # Calculate total recommended overaspirate using the new formula
                        # This includes existing overaspirate + additional needed + buffer
                        total_recommended = new_base + (new_scaling/100) * vol_set
                        total_recommended = max(total_recommended, 2.0)  # Ensure minimum 2μL
                        
                        report_lines.append(f"  {vol_set:.0f}uL target -> {vol_meas:.1f}uL measured (shortfall: {shortfall:.1f}uL, had {existing_over:.1f}uL) -> recommend {total_recommended:.1f}uL total")
        else:
            report_lines.append("Status: DISABLED")
            report_lines.append(f"Using static formula: {OVERASPIRATE_BASE_UL:.1f}uL + {OVERASPIRATE_SCALING_PERCENT:.1f}%")
        
        report_lines.append("")
    
    # Volume-by-volume details
    for volume in volumes:
        data = volume_report_data.get(volume, {})
        volume_ul = int(volume * 1000)
        
        report_lines.append(f"Volume_{volume_ul}uL:")
        
        # SOBOL trials
        sobol_count = data.get('sobol_trials', 0)
        if sobol_count > 0:
            report_lines.append(f"   {sobol_count} SOBOL TRIALS")
        
        # Optimization trials with failure breakdown
        opt_count = data.get('optimization_trials', 0)
        time_failures = data.get('time_failures', 0)
        accuracy_failures = data.get('accuracy_failures', 0)
        
        if opt_count > 0:
            failure_text = ""
            if time_failures > 0 or accuracy_failures > 0:
                failure_parts = []
                if time_failures > 0:
                    failure_parts.append(f"Time Failures {time_failures}")
                if accuracy_failures > 0:
                    failure_parts.append(f"Accuracy Failures {accuracy_failures}")
                failure_text = f"; {'; '.join(failure_parts)}"
            
            report_lines.append(f"   {opt_count} OPTIMIZATION TRIALS{failure_text}")
        
        # Precision test results
        precision_failed = data.get('precision_trials_failed', 0)
        precision_passed_count = data.get('precision_trials_passed', 0)
        precision_overall_passed = data.get('precision_passed', False)
        
        # Show failed precision trials first if any
        if precision_failed > 0:
            report_lines.append(f"   {precision_failed} PRECISION TRIALS (FAILED)")
        
        # Show passed precision trials if any
        if precision_passed_count > 0:
            report_lines.append(f"   {precision_passed_count} PRECISION TRIALS (PASSED)")
        
        # Calculate and show total trials for this volume
        sobol_count = data.get('sobol_trials', 0)
        total_volume_trials = sobol_count + opt_count + precision_failed + precision_passed_count
        report_lines.append(f"   TOTAL: {total_volume_trials} trials")
        
        # Overall status
        completed = data.get('completed', False)
        if completed:
            report_lines.append("   STATUS: [COMPLETED]")
        else:
            candidate_found = data.get('candidate_found', False)
            if not candidate_found:
                report_lines.append("   STATUS: [FAILED] NO CANDIDATE FOUND")
            else:
                report_lines.append("   STATUS: [FAILED] PRECISION TEST FAILED")
        
        report_lines.append("")
    
    # Summary section
    report_lines.append("SUMMARY:")
    report_lines.append("-" * 20)
    
    # Calculate overall statistics
    total_volumes = len(volumes)
    completed_count = len(completed_volumes)
    
    report_lines.append(f"Volumes Completed: {completed_count}/{total_volumes}")
    report_lines.append("")
    
    # Optimization trial statistics
    report_lines.append("Optimization Trials:")
    for volume in volumes:
        data = volume_report_data.get(volume, {})
        volume_ul = int(volume * 1000)
        opt_count = data.get('optimization_trials', 0)
        time_failures = data.get('time_failures', 0)
        accuracy_failures = data.get('accuracy_failures', 0)
        
        if opt_count > 0:
            time_pass_rate = ((opt_count - time_failures) / opt_count) * 100
            accuracy_pass_rate = ((opt_count - accuracy_failures) / opt_count) * 100
            report_lines.append(f"  {volume_ul}uL: Time passing: {time_pass_rate:.1f}%, Accuracy passing: {accuracy_pass_rate:.1f}%")
    
    report_lines.append("")
    
    # Precision test statistics
    report_lines.append("Precision Tests:")
    for volume in volumes:
        data = volume_report_data.get(volume, {})
        volume_ul = int(volume * 1000)
        precision_failed = data.get('precision_trials_failed', 0)
        precision_passed_count = data.get('precision_trials_passed', 0)
        precision_overall_passed = data.get('precision_passed', False)
        
        if precision_failed > 0 or precision_passed_count > 0:
            if precision_overall_passed:
                report_lines.append(f"  {volume_ul}uL: Passed ({precision_passed_count} trials)")
            else:
                total_attempts = precision_failed + precision_passed_count
                if precision_failed > 0:
                    report_lines.append(f"  {volume_ul}uL: Failed ({total_attempts} trials total)")
                else:
                    report_lines.append(f"  {volume_ul}uL: No trials completed")
    
    report_lines.append("")
    
    # Calculate and show grand total trials
    grand_total = 0
    for volume in volumes:
        data = volume_report_data.get(volume, {})
        sobol_count = data.get('sobol_trials', 0)
        opt_count = data.get('optimization_trials', 0)
        precision_failed = data.get('precision_trials_failed', 0)
        precision_passed = data.get('precision_trials_passed', 0)
        volume_total = sobol_count + opt_count + precision_failed + precision_passed
        grand_total += volume_total
    
    report_lines.append(f"TOTAL TRIALS ACROSS ALL VOLUMES: {grand_total}")
    report_lines.append("")
    
    # Success check and diagnostics
    all_completed = completed_count == total_volumes
    if all_completed:
        report_lines.append("[SUCCESS] CALIBRATION SUCCESSFUL!")
    else:
        report_lines.append("[INCOMPLETE] CALIBRATION INCOMPLETE - DIAGNOSTICS:")
        report_lines.extend(generate_failure_diagnostics(volume_report_data, volumes))
    
    return "\n".join(report_lines)

def generate_failure_diagnostics(volume_report_data, volumes):
    """Generate diagnostic recommendations based on failure patterns."""
    
    diagnostics = []
    
    if not volumes:
        return diagnostics
    
    first_volume = volumes[0]
    first_data = volume_report_data.get(first_volume, {})
    
    # Check first volume completion
    first_completed = first_data.get('completed', False)
    first_candidate_found = first_data.get('candidate_found', False)
    
    if not first_completed:
        if not first_candidate_found:
            # Failed on optimization for first volume
            opt_count = first_data.get('optimization_trials', 0)
            time_failures = first_data.get('time_failures', 0)
            accuracy_failures = first_data.get('accuracy_failures', 0)
            
            if opt_count > 0:
                time_pass_rate = ((opt_count - time_failures) / opt_count) * 100
                accuracy_pass_rate = ((opt_count - accuracy_failures) / opt_count) * 100
                
                if time_pass_rate < 20:  # Less than 20% passing time
                    diagnostics.append("- Time restrictions appear too strict for the first volume")
                    diagnostics.append("  Recommendation: Increase BASE_TIME_SECONDS or TIME_SCALING_FACTOR")
                
                if accuracy_pass_rate < 20:  # Less than 20% passing accuracy
                    diagnostics.append("- Accuracy restrictions appear too strict for the first volume") 
                    diagnostics.append("  Recommendation: Increase BASE_DEVIATION_UL or DEVIATION_SCALING_FACTOR -or OVERASPIRATE_VOLUME_%")
            
            diagnostics.append(f"- First volume failed to find acceptable candidate after {opt_count} trials")
        
        else:
            # Found candidate but precision test failed
            diagnostics.append("- First volume precision test failed")
            diagnostics.append("  Recommendation: Increase BASE_VARIATION_UL (precision window too narrow)")
    
    else:
        # First volume completed, check progression to higher volumes
        failed_volumes = []
        for volume in volumes[1:]:  # Skip first volume
            data = volume_report_data.get(volume, {})
            if not data.get('completed', False):
                failed_volumes.append(volume)
        
        if failed_volumes:
            diagnostics.append("- First volume completed but later volumes failed")
            diagnostics.append("  Recommendation: The scaling factors may be too aggressive")
            diagnostics.append("  Consider increasing DEVIATION_SCALING_FACTOR, TIME_SCALING_FACTOR, or VARIATION_SCALING_FACTOR")
            
            # Check if it's mainly precision failures at higher volumes
            precision_failures = 0
            for vol in failed_volumes:
                data = volume_report_data.get(vol, {})
                if data.get('candidate_found', False) and not data.get('precision_passed', False):
                    precision_failures += 1
            
            if precision_failures > 0:
                diagnostics.append("  Focus on: VARIATION_SCALING_FACTOR (precision tests failing)")
    
    return diagnostics

def main():
    # Use manually specified volumes instead of auto-generation
    # Commented out: automatic volume generation
    # global VOLUMES  
    # VOLUMES = generate_volumes(MIN_VOLUME_ML, MAX_VOLUME_ML, NUM_VOLUMES)
    print(f"\n📋 USING MANUAL VOLUME SEQUENCE: {[f'{v*1000:.0f}μL' for v in VOLUMES]}")
    print(f"   Note: Volumes are manually specified in VOLUMES list")
    
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
    transition_names = {"relu": "ReLU (max)", "smooth": "Smooth (log)", "asymmetric": "Asymmetric (gentle fast penalty)"}
    print(f"   ⏱️  Time Transition: {transition_names.get(TIME_TRANSITION_MODE, TIME_TRANSITION_MODE)}")
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
    
    # Report data collection - track statistics for each volume
    volume_report_data = {}  # Dictionary keyed by volume with statistics
    
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
        
        # Initialize report data for this volume
        volume_report_data[volume] = {
            'sobol_trials': 0,
            'optimization_trials': 0,
            'time_failures': 0,
            'accuracy_failures': 0,
            'precision_trials_attempted': 0,
            'precision_trials_failed': 0,
            'precision_trials_passed': 0,
            'precision_passed': False,
            'completed': False,
            'candidate_found': False
        }
        
        # Calculate volume-dependent tolerances
        is_first_volume = (volume_index == 0)  # First volume gets time optimization and constraints
        tolerances = get_volume_dependent_tolerances(volume, is_first_volume=is_first_volume)
        
        # Build criteria - only include time for first volume
        criteria = {
            'max_deviation_ul': tolerances['deviation_ul'],
        }
        if is_first_volume:
            criteria['max_time'] = tolerances['time_seconds']
            print(f"📊 VOLUME {volume*1000:.0f}μL: First volume - time optimization ENABLED (max: {tolerances['time_seconds']:.1f}s)")
        else:
            print(f"📊 VOLUME {volume*1000:.0f}μL: Subsequent volume - time optimization DISABLED (accuracy only)")
            print(f"   Using time-optimized parameters from previous volumes")
        
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
        all_screening_results = []  # Initialize for all volumes
        
        # Always create ax_client with the correct selective optimization configuration
        # Use half of the time cap as the maximum wait time constraint (only for first volume)
        max_wait_time = criteria.get('max_time', 30.0) / 2.0  # Default to 15s if no time constraint
        max_overaspirate_for_volume = get_max_overaspirate_ul(volume)
        ax_client = get_recommender().create_model(SEED, INITIAL_SUGGESTIONS, bayesian_batch_size=BATCH_SIZE, 
                                           volume=volume, tip_volume=tip_volume, model_type=bayesian_model_type, 
                                           optimize_params=optimize_params, fixed_params=fixed_params, simulate=SIMULATE,
                                           max_overaspirate_ul=max_overaspirate_for_volume, max_wait_time=max_wait_time)
        
        # Step 1: Determine starting candidate
        if len(completed_volumes) > 0:
            # Use the FIRST successful volume's parameters as starting point (most conservative)
            first_volume, first_params = completed_volumes[0]
            print(f"🔄 TESTING CONSERVATIVE BASELINE: Using parameters from FIRST successful volume {first_volume*1000:.0f}μL")
            candidate_params = first_params
            
            # Find the original trial number for these parameters
            first_optimal = optimal_conditions[0] if optimal_conditions else {}
            candidate_trial_number = first_optimal.get('trial_number', 'unknown')
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
                    
                # Count this as a SOBOL trial (initial screening)
                volume_report_data[volume]['sobol_trials'] += 1
                
                check_if_measurement_vial_full(lash_e, state)
                liquid_source = get_liquid_source(lash_e)
                result = pipet_and_measure(lash_e, liquid_source, state["measurement_vial_name"], volume, params, 
                                         expected_mass, expected_time, REPLICATES, SIMULATE, autosave_raw_path, 
                                         raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET)
                
                # Get the most recent measurement for display
                recent_mass = raw_measurements[-1]['mass'] if raw_measurements else expected_mass
                recent_volume = raw_measurements[-1]['calculated_volume'] if raw_measurements else volume
                
                # Check if this trial meets criteria - convert % deviation to absolute μL
                deviation_pct = result.get('deviation', float('inf'))
                absolute_deviation_ul = (deviation_pct / 100) * (volume * 1000)  # Convert to μL
                
                # For non-first volumes, ignore time criteria
                if is_first_volume:
                    meets_criteria = (absolute_deviation_ul <= criteria['max_deviation_ul'] and 
                                    result.get('time', float('inf')) <= criteria['max_time'])
                else:
                    meets_criteria = (absolute_deviation_ul <= criteria['max_deviation_ul'])
                    
                status = "✅ CANDIDATE" if meets_criteria else "❌ reject"
                print(f"   Trial {i+1}/{INITIAL_SUGGESTIONS}: {recent_mass:.4f}g → {recent_volume*1000:.1f}μL, {result.get('deviation', 'N/A'):.1f}% dev, {result.get('time', 'N/A'):.0f}s - {status}")
                
                # Add result to optimizer - only use time scoring for first volume
                if is_first_volume:
                    volume_tolerances = get_volume_dependent_tolerances(volume, is_first_volume=True)
                    scaled_optimal_target = volume_tolerances.get('time_optimal_target', BASE_TIME_SECONDS)
                    get_recommender().add_result(ax_client, trial_index, result, BASE_TIME_SECONDS, scaled_optimal_target, TIME_TRANSITION_MODE)
                else:
                    # For subsequent volumes, pass None to disable time optimization in recommender
                    get_recommender().add_result(ax_client, trial_index, result, BASE_TIME_SECONDS, None, TIME_TRANSITION_MODE)
                result.update(params)
                result.update({"volume": volume, "trial_index": trial_index, "strategy": screening_method, "liquid": LIQUID, "time_reported": datetime.now().isoformat()})
                result = strip_tuples(result)
                all_results.append(result)
                if not SIMULATE:
                    pd.DataFrame([result]).to_csv(autosave_summary_path, mode='a', index=False, header=not os.path.exists(autosave_summary_path))
                
                trial_count += 1
                
                # Collect ALL screening results for calibration (regardless of criteria)
                screening_result_info = {
                    'params': params.copy(),
                    'deviation': result.get('deviation', float('inf')),
                    'time': result.get('time', float('inf')),
                    'score': result.get('deviation', float('inf')),
                    'trial_number': trial_count
                }
                all_screening_results.append(screening_result_info)
                
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
            
            # Choose the FIRST acceptable candidate (most conservative/transferable)
            if screening_candidates:
                # Sort by trial number (chronological order) to get first acceptable solution
                first_candidate = min(screening_candidates, key=lambda x: x['trial_number'])
                candidate_params = first_candidate['params']
                candidate_trial_number = first_candidate['trial_number']
                print(f"   ✅ Selected FIRST acceptable candidate from trial #{candidate_trial_number}: {first_candidate['deviation']:.1f}% deviation, {first_candidate['time']:.0f}s")
                print(f"   📊 Strategy: Using first successful solution for better transferability ({len(screening_candidates)} total candidates)")
                print(f"   📊 {len(screening_candidates)}/{INITIAL_SUGGESTIONS} {screening_method} trials met criteria")
            else:
                print(f"   ❌ No {screening_method} trials met criteria - will need optimization")
            
            print(f"   ✅ Completed {INITIAL_SUGGESTIONS} {screening_method} conditions")
        
        # Optional: Overvolume calibration phase (only for first volume)
        if is_first_volume and AUTO_CALIBRATE_OVERVOLUME and all_screening_results and len(VOLUMES) > 1:
            print(f"\n🔬 OVERVOLUME CALIBRATION: Starting automatic calibration...")
            
            # Get remaining volumes to test
            remaining_volumes = VOLUMES[1:]  # Skip first volume (already tested)
            
            # Run calibration using all screening results (not just candidates)
            new_base_ul, new_scaling_percent, calibration_data = calibrate_overvolume_parameters(
                all_screening_results, remaining_volumes, lash_e, state, 
                autosave_raw_path, raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET, criteria, autosave_dir
            )
            
            # Update global parameters if calibration succeeded
            if new_base_ul is not None and new_scaling_percent is not None:
                global OVERASPIRATE_BASE_UL, OVERASPIRATE_SCALING_PERCENT
                old_base = OVERASPIRATE_BASE_UL
                old_scaling = OVERASPIRATE_SCALING_PERCENT
                
                OVERASPIRATE_BASE_UL = new_base_ul
                OVERASPIRATE_SCALING_PERCENT = new_scaling_percent
                
                print(f"   ✅ OVERVOLUME PARAMETERS UPDATED:")
                print(f"      Old: {old_base:.1f}μL + {old_scaling:.1f}%")
                print(f"      New: {OVERASPIRATE_BASE_UL:.1f}μL + {OVERASPIRATE_SCALING_PERCENT:.1f}%")
                
                # Store calibration info for reporting
                volume_report_data[volume]['overvolume_calibration'] = {
                    'enabled': True,
                    'old_base_ul': old_base,
                    'old_scaling_percent': old_scaling,
                    'new_base_ul': OVERASPIRATE_BASE_UL,
                    'new_scaling_percent': OVERASPIRATE_SCALING_PERCENT,
                    'calibration_data': calibration_data
                }
            else:
                print(f"   ❌ OVERVOLUME CALIBRATION FAILED - keeping original values")
                print(f"      Using: {OVERASPIRATE_BASE_UL:.1f}μL + {OVERASPIRATE_SCALING_PERCENT:.1f}%")
                
                volume_report_data[volume]['overvolume_calibration'] = {
                    'enabled': True,
                    'failed': True,
                    'keeping_original': True
                }
        elif is_first_volume and AUTO_CALIBRATE_OVERVOLUME:
            if not all_screening_results:
                print(f"   ⚠️  OVERVOLUME CALIBRATION: Skipped (no screening results)")
                reason = 'no_screening_results'
            elif len(VOLUMES) <= 1:
                print(f"   ⚠️  OVERVOLUME CALIBRATION: Skipped (single volume)")
                reason = 'single_volume'
            else:
                print(f"   ⚠️  OVERVOLUME CALIBRATION: Skipped (unknown reason)")
                reason = 'unknown'
                
            volume_report_data[volume]['overvolume_calibration'] = {
                'enabled': True,
                'skipped': True,
                'reason': reason
            }
        
        # Step 2: Optimization loop until we have a successful precision test
        while not volume_completed and trial_count < MAX_WELLS - PRECISION_REPLICATES:
            
            # If we don't have a candidate yet, we need to optimize
            if candidate_params is None:
                if is_first_volume:
                    print(f"🔍 OPTIMIZATION: Finding acceptable parameters (target: ≤{criteria['max_deviation_ul']:.0f}μL deviation, ≤{criteria['max_time']:.0f}s)")
                else:
                    print(f"🔍 OPTIMIZATION: Finding acceptable parameters (target: ≤{criteria['max_deviation_ul']:.0f}μL deviation, no time limit)")
                
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
                        
                        # Count this as an optimization trial
                        volume_report_data[volume]['optimization_trials'] += 1
                        
                        check_if_measurement_vial_full(lash_e, state)
                        liquid_source = get_liquid_source(lash_e)
                        result = pipet_and_measure(lash_e, liquid_source, state["measurement_vial_name"], volume, params, 
                                                 expected_mass, expected_time, REPLICATES, SIMULATE, autosave_raw_path, 
                                                 raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET)
                        
                        # Get the most recent measurement for display
                        recent_mass = raw_measurements[-1]['mass'] if raw_measurements else expected_mass
                        recent_volume = raw_measurements[-1]['calculated_volume'] if raw_measurements else volume
                        
                        # Check if this trial meets criteria - convert % deviation to absolute μL
                        deviation_pct = result.get('deviation', float('inf'))
                        absolute_deviation_ul = (deviation_pct / 100) * (volume * 1000)  # Convert to μL
                        
                        # For non-first volumes, ignore time criteria
                        if is_first_volume:
                            time_fails = result.get('time', float('inf')) > criteria['max_time']
                            meets_criteria = not (time_fails or absolute_deviation_ul > criteria['max_deviation_ul'])
                            if time_fails:
                                volume_report_data[volume]['time_failures'] += 1
                        else:
                            meets_criteria = (absolute_deviation_ul <= criteria['max_deviation_ul'])
                            
                        if absolute_deviation_ul > criteria['max_deviation_ul']:
                            volume_report_data[volume]['accuracy_failures'] += 1
                        
                        status = "✅ CANDIDATE" if meets_criteria else "❌ reject"
                        print(f"   Optimization trial: {recent_mass:.4f}g → {recent_volume*1000:.1f}μL, {result.get('deviation', 'N/A'):.1f}% dev, {result.get('time', 'N/A'):.0f}s - {status}")
                        
                        # Add result to optimizer - only use time scoring for first volume
                        if is_first_volume:
                            volume_tolerances = get_volume_dependent_tolerances(volume, is_first_volume=True)
                            scaled_optimal_target = volume_tolerances.get('time_optimal_target', BASE_TIME_SECONDS)
                            get_recommender().add_result(ax_client, trial_index, result, BASE_TIME_SECONDS, scaled_optimal_target, TIME_TRANSITION_MODE)
                        else:
                            # For subsequent volumes, pass None to disable time optimization in recommender
                            get_recommender().add_result(ax_client, trial_index, result, BASE_TIME_SECONDS, None, TIME_TRANSITION_MODE)
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
                            volume_report_data[volume]['candidate_found'] = True
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
            
            # Track precision test results
            # First replicate (0) counts as final optimization trial, rest are precision trials
            if len(precision_measurements) > 0:
                # Count first replicate as optimization (candidate validation)
                volume_report_data[volume]['optimization_trials'] += 1
                
                # Count remaining replicates as precision trials
                precision_count = len(precision_measurements) - 1
                if precision_count > 0:
                    volume_report_data[volume]['precision_trials_attempted'] += precision_count
                    if passed:
                        volume_report_data[volume]['precision_trials_passed'] += precision_count
                        volume_report_data[volume]['precision_passed'] = True
                    else:
                        volume_report_data[volume]['precision_trials_failed'] += precision_count
            
            if passed:
                # SUCCESS! 
                completed_volumes.append((volume, candidate_params))
                volume_report_data[volume]['completed'] = True
                
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
                
                # Adaptive time scaling removed - time optimization only for first volume
                
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
    print(f"Total trials used: {len(all_results)}/{MAX_WELLS} (includes {len([r for r in all_results if r.get('strategy') == 'PRECISION_TEST'])} precision test measurements)")
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
        
        # Only show volumes that were actually tested (have results)
        if len(vol_results) > 0:
            print(f"\n🎯 Volume {volume*1000:.0f}μL:")
            
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
    
    print(f"\n📋 EXPERIMENT SUMMARY:")
    print(f"   • Total optimization trials: {optimization_trials_count}")
    print(f"   • Total precision measurements: {precision_measurements_count}")
    print(f"   • Volumes completed: {len(completed_volumes)}/{len(VOLUMES)}")

    # Save analysis results (both simulation and real data)
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
            completed_list = [f"{int(v*1000)}µL" for v, _ in completed_volumes]
            remaining_vols = [v for v in VOLUMES if v not in [cv for cv, _ in completed_volumes]]
            remaining_list = [f"{int(v*1000)}µL" for v in remaining_vols]
            completed_str = ", ".join(completed_list) if completed_list else "None"
            remaining_str = ", ".join(remaining_list) if remaining_list else "None"

            slack_msg = (
                f"Modular calibration with {LIQUID} COMPLETE\n"
                f"Volumes completed: {len(completed_volumes)}/{len(VOLUMES)} -> {completed_str}\n"
                f"Remaining (not calibrated): {remaining_str}"
            )
            slack_agent.send_slack_message(slack_msg)
        except Exception as e:
            print(f"Warning: Failed to send detailed Slack summary: {e}")

    # Generate and save calibration report
    try:
        report_content = generate_calibration_report(volume_report_data, VOLUMES, completed_volumes)
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"calibration_report_{timestamp}.txt"
        report_path = os.path.join(autosave_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n📊 CALIBRATION REPORT SAVED: {report_path}")
        
        # Simple completion status
        completed_count = len(completed_volumes)
        total_count = len(VOLUMES)
        if completed_count == total_count:
            print("[SUCCESS] CALIBRATION SUCCESSFUL!")
        else:
            print(f"[INCOMPLETE] CALIBRATION INCOMPLETE: {completed_count}/{total_count} volumes completed")
        
    except Exception as e:
        print(f"Warning: Failed to generate calibration report: {e}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

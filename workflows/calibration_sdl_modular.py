# calibration_sdl_modular.py
import sys
import os

from sympy import false
#asd
sys.path.append("../North-Cytation")

from calibration_sdl_base import (
    pipet_and_measure_simulated, pipet_and_measure, strip_tuples, save_analysis,
    LIQUIDS
)
import calibration_sdl_base as base_module  # Import module to preserve global state
from master_usdl_coordinator import Lash_E
import recommenders.pipeting_optimizer_v2 as recommender_v2
import recommenders.pipetting_optimizer_v3 as recommender_v3
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yaml

# Vial mode configuration helpers
def get_vial_config(mode):
    """
    Get vial configuration for a given mode
    
    Args:
        mode: One of 'legacy', 'maintain', 'swap'
    
    Returns:
        dict: Configuration including vial_file and mode parameters
    """
    config_path = "settings/vial_mode_config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if mode not in config['vial_modes']:
            raise ValueError(f"Unknown vial mode: {mode}. Available modes: {list(config['vial_modes'].keys())}")
        
        return config['vial_modes'][mode]
    except FileNotFoundError:
        # Fallback to hardcoded values if config file doesn't exist
        fallback_configs = {
            'legacy': {
                'vial_file': 'status/calibration_vials_short.csv',
                'mode': 'legacy',
                'description': 'Legacy mode - single vial used for all volumes'
            },
            'maintain': {
                'vial_file': 'status/calibration_vials_overnight.csv', 
                'mode': 'maintain',
                'description': 'Maintain mode - keep vials, use reservoir/waste'
            },
            'swap': {
                'vial_file': 'status/calibration_vials_overnight.csv',
                'mode': 'swap', 
                'description': 'Swap mode - new vials for each volume'
            },
            'single': {
                'vial_file': 'status/calibration_vials_overnight.csv',
                'mode': 'single', 
                'description': 'Single mode - one vial for infinite liquid recycling'
            }
        }
        if mode in fallback_configs:
            return fallback_configs[mode]
        else:
            raise ValueError(f"Unknown vial mode: {mode}")

def setup_vial_mode(mode, simulate=True):
    """
    Set up vial management based on mode
    
    Args:
        mode: One of 'legacy', 'maintain', 'swap'
        simulate: Whether running in simulation mode
    
    Returns:
        tuple: (vial_file_path, lash_e_instance)
    """
    vial_config = get_vial_config(mode)
    vial_file = vial_config['vial_file']
    
    print(f"🔧 Vial Mode: {mode}")
    print(f"   📁 File: {vial_file}")
    print(f"   📝 {vial_config['description']}")
    
    lash_e = Lash_E(vial_file, simulate=simulate, initialize_biotek=False)
    return vial_file, lash_e

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

# --- DEFAULT EXPERIMENT CONFIG (IMMUTABLE) ---
# These constants store the TRUE defaults and should NEVER be modified at runtime
DEFAULT_LIQUID = "glycerol"
DEFAULT_SIMULATE = False
DEFAULT_SEED = 7
DEFAULT_INITIAL_SUGGESTIONS = 5  # replaces SOBOL_CYCLES_PER_VOLUME
DEFAULT_BATCH_SIZE = 1
DEFAULT_REPLICATES = 1  # for optimization
DEFAULT_PRECISION_REPLICATES = 4
DEFAULT_VOLUMES = [0.05, 0.025, 0.01]  # Manually specified volume list (in mL)
DEFAULT_MAX_WELLS = 96
DEFAULT_INPUT_VIAL_STATUS_FILE = "status/calibration_vials_short.csv"
# Time-related parameters removed - now using raw time optimization only
DEFAULT_USE_LLM_FOR_SCREENING = False     # LLM vs SOBOL for initial exploration (first volume)
DEFAULT_USE_LLM_FOR_OPTIMIZATION = False  # LLM vs Bayesian for optimization loops
DEFAULT_BAYESIAN_MODEL_TYPE = 'qEI'  # Default Bayesian acquisition function
DEFAULT_CANDIDATE_SELECTION_PERCENTILE = 35  # Target percentile for candidate selection (avoid fastest, pick reasonably fast)

# Default scoring weights for candidate evaluation
DEFAULT_INITIAL_SELECTION_TIME_WEIGHT = 0.4      # Initial selection: time proximity weight
DEFAULT_INITIAL_SELECTION_ACCURACY_WEIGHT = 0.6  # Initial selection: accuracy weight  
DEFAULT_CASCADING_TIME_WEIGHT = 0.3              # Cascading ranking: time distance weight
DEFAULT_CASCADING_ACCURACY_WEIGHT = 0.7          # Cascading ranking: accuracy weight
DEFAULT_CASCADING_SWEET_SPOT_BONUS = -0.1        # Cascading ranking: sweet spot bonus

# --- RUNTIME EXPERIMENT CONFIG (MUTABLE) ---
# These variables can be modified during experiments but should be reset to defaults between experiments
LIQUID = DEFAULT_LIQUID
SIMULATE = DEFAULT_SIMULATE
SEED = DEFAULT_SEED
INITIAL_SUGGESTIONS = DEFAULT_INITIAL_SUGGESTIONS
BATCH_SIZE = DEFAULT_BATCH_SIZE
REPLICATES = DEFAULT_REPLICATES
PRECISION_REPLICATES = DEFAULT_PRECISION_REPLICATES
VOLUMES = DEFAULT_VOLUMES.copy()  # Make a copy to avoid modifying the default
MAX_WELLS = DEFAULT_MAX_WELLS
INPUT_VIAL_STATUS_FILE = DEFAULT_INPUT_VIAL_STATUS_FILE
EXPERIMENT_INDEX = 0  # Tracks index in multi-experiment runs; 0 = first
_CACHED_LASH_E = None  # Reused robot controller across experiments
RETAIN_PIPET_BETWEEN_EXPERIMENTS = False  # Default: remove pipet between experiments (set True to retain)

# Time-related runtime variables removed - using raw time optimization only

USE_LLM_FOR_SCREENING = DEFAULT_USE_LLM_FOR_SCREENING
USE_LLM_FOR_OPTIMIZATION = DEFAULT_USE_LLM_FOR_OPTIMIZATION

# --- Bayesian Model Configuration ---
# Controls which Bayesian acquisition function to use for optimization
# NOTE: This is only used when USE_LLM_FOR_OPTIMIZATION = False
# Options: 'qEI' (Expected Improvement), 'qLogEI' (Log Expected Improvement), 'qNEHVI' (Noisy Expected Hypervolume Improvement)
BAYESIAN_MODEL_TYPE = DEFAULT_BAYESIAN_MODEL_TYPE

# --- Candidate Selection Configuration ---
CANDIDATE_SELECTION_PERCENTILE = DEFAULT_CANDIDATE_SELECTION_PERCENTILE  # Can be modified at runtime

# --- Candidate Scoring Weights ---
# For initial candidate selection (select_best_candidate_from_accurate_trials)
INITIAL_SELECTION_TIME_WEIGHT = DEFAULT_INITIAL_SELECTION_TIME_WEIGHT
INITIAL_SELECTION_ACCURACY_WEIGHT = DEFAULT_INITIAL_SELECTION_ACCURACY_WEIGHT

# For cascading precision test ranking (get_ordered_candidates_from_results)  
CASCADING_TIME_WEIGHT = DEFAULT_CASCADING_TIME_WEIGHT
CASCADING_ACCURACY_WEIGHT = DEFAULT_CASCADING_ACCURACY_WEIGHT
CASCADING_SWEET_SPOT_BONUS = DEFAULT_CASCADING_SWEET_SPOT_BONUS

# Relative percentage tolerances (applies to both optimization and precision test)
# Volume ranges defined as (min_volume_ul, max_volume_ul, tolerance_pct)
# Updated volume tolerance ranges with smoother transition around 100�uL
# Uses gradual scaling instead of sharp cutoffs to avoid excessive failure at boundary volumes
VOLUME_TOLERANCE_RANGES = [
    {'min_ul': 200, 'max_ul': 1000, 'tolerance_pct': 1.0, 'name': 'large_volume'},   # ≥200�uL: 1%
    {'min_ul': 60,  'max_ul': 200,  'tolerance_pct': 2.0, 'name': 'medium_large_volume'}, # 50-199�uL: 2%
    {'min_ul': 20,  'max_ul': 60,   'tolerance_pct': 3.0, 'name': 'medium_volume'}, # 10-49�uL: 3%  
    {'min_ul': 1,   'max_ul': 20,   'tolerance_pct': 5.0, 'name': 'small_volume'},  # 1-9�uL: 5%
    {'min_ul': 0,   'max_ul': 1,    'tolerance_pct': 10.0, 'name': 'micro_volume'}, # <1�uL: 10% (fallback)
]

# --- DEFAULT OVERASPIRATE CONFIG (IMMUTABLE) ---
DEFAULT_OVERASPIRATE_BASE_UL = 5.0        # Base overaspirate volume in microliters
DEFAULT_OVERASPIRATE_SCALING_PERCENT = 5.0  # Additional percentage of total volume
DEFAULT_AUTO_CALIBRATE_OVERVOLUME = True  # Enable automatic overvolume calibration after SOBOL trials
DEFAULT_OVERVOLUME_CALIBRATION_BUFFER_UL = 2.0  # Buffer to add above fitted line (uL)
DEFAULT_OVERVOLUME_MAX_BASE_UL = 50.0     # Maximum allowed base overvolume (uL)
DEFAULT_OVERVOLUME_MAX_PERCENT = 100.0    # Maximum allowed percentage scaling (%)

# --- RUNTIME OVERASPIRATE CONFIG (MUTABLE) ---
# Selective parameter optimization config
# Max overaspirate: Base volume + percentage scaling
OVERASPIRATE_BASE_UL = DEFAULT_OVERASPIRATE_BASE_UL
OVERASPIRATE_SCALING_PERCENT = DEFAULT_OVERASPIRATE_SCALING_PERCENT

# Auto-calibration of overvolume parameters
AUTO_CALIBRATE_OVERVOLUME = DEFAULT_AUTO_CALIBRATE_OVERVOLUME
OVERVOLUME_CALIBRATION_BUFFER_UL = DEFAULT_OVERVOLUME_CALIBRATION_BUFFER_UL
OVERVOLUME_MAX_BASE_UL = DEFAULT_OVERVOLUME_MAX_BASE_UL
OVERVOLUME_MAX_PERCENT = DEFAULT_OVERVOLUME_MAX_PERCENT

USE_SELECTIVE_OPTIMIZATION = True  # Enable selective parameter optimization
USE_HISTORICAL_DATA_FOR_OPTIMIZATION = False  # Load data from previous volumes into optimizer
VOLUME_DEPENDENT_PARAMS = ["blowout_vol", "overaspirate_vol"]  # Parameters to optimize for each volume
ALL_PARAMS = ["aspirate_speed", "dispense_speed", "aspirate_wait_time", "dispense_wait_time", 
              "retract_speed", "blowout_vol", "post_asp_air_vol", "overaspirate_vol"]

# --- ADAPTIVE TOLERANCE TRACKING ---
# Track achieved tolerance multipliers for progressive tolerance learning
ACHIEVED_TOLERANCE_MULTIPLIERS = {}  # {volume: multiplier} - tracks what tolerance each volume actually needed
ADAPTIVE_TOLERANCE_MODE = True      # Enable adaptive tolerance learning from successful volumes

# --- CONFIG MANAGEMENT FUNCTIONS ---
def reset_config_to_defaults():
    """Reset all global configuration variables back to their original default values.
    
    This function MUST be called before each experiment to ensure a clean slate
    and prevent config persistence between experiments.
    """
    global LIQUID, SIMULATE, SEED, INITIAL_SUGGESTIONS, BATCH_SIZE, REPLICATES
    global PRECISION_REPLICATES, VOLUMES, MAX_WELLS, INPUT_VIAL_STATUS_FILE
    global USE_LLM_FOR_SCREENING, USE_LLM_FOR_OPTIMIZATION, BAYESIAN_MODEL_TYPE
    global CANDIDATE_SELECTION_PERCENTILE
    global INITIAL_SELECTION_TIME_WEIGHT, INITIAL_SELECTION_ACCURACY_WEIGHT
    global CASCADING_TIME_WEIGHT, CASCADING_ACCURACY_WEIGHT, CASCADING_SWEET_SPOT_BONUS
    global OVERASPIRATE_BASE_UL, OVERASPIRATE_SCALING_PERCENT, AUTO_CALIBRATE_OVERVOLUME
    global OVERVOLUME_CALIBRATION_BUFFER_UL, OVERVOLUME_MAX_BASE_UL, OVERVOLUME_MAX_PERCENT
    
    print("🔄 Resetting configuration to default values...")
    
    # Reset adaptive tolerance learning
    reset_tolerance_learning()
    
    LIQUID = DEFAULT_LIQUID
    SIMULATE = DEFAULT_SIMULATE
    SEED = DEFAULT_SEED
    INITIAL_SUGGESTIONS = DEFAULT_INITIAL_SUGGESTIONS
    BATCH_SIZE = DEFAULT_BATCH_SIZE
    REPLICATES = DEFAULT_REPLICATES
    PRECISION_REPLICATES = DEFAULT_PRECISION_REPLICATES
    VOLUMES = DEFAULT_VOLUMES.copy()  # Make a copy to avoid modifying the immutable default
    MAX_WELLS = DEFAULT_MAX_WELLS
    INPUT_VIAL_STATUS_FILE = DEFAULT_INPUT_VIAL_STATUS_FILE
    USE_LLM_FOR_SCREENING = DEFAULT_USE_LLM_FOR_SCREENING
    USE_LLM_FOR_OPTIMIZATION = DEFAULT_USE_LLM_FOR_OPTIMIZATION
    BAYESIAN_MODEL_TYPE = DEFAULT_BAYESIAN_MODEL_TYPE
    CANDIDATE_SELECTION_PERCENTILE = DEFAULT_CANDIDATE_SELECTION_PERCENTILE
    INITIAL_SELECTION_TIME_WEIGHT = DEFAULT_INITIAL_SELECTION_TIME_WEIGHT
    INITIAL_SELECTION_ACCURACY_WEIGHT = DEFAULT_INITIAL_SELECTION_ACCURACY_WEIGHT
    CASCADING_TIME_WEIGHT = DEFAULT_CASCADING_TIME_WEIGHT
    CASCADING_ACCURACY_WEIGHT = DEFAULT_CASCADING_ACCURACY_WEIGHT
    CASCADING_SWEET_SPOT_BONUS = DEFAULT_CASCADING_SWEET_SPOT_BONUS
    OVERASPIRATE_BASE_UL = DEFAULT_OVERASPIRATE_BASE_UL
    OVERASPIRATE_SCALING_PERCENT = DEFAULT_OVERASPIRATE_SCALING_PERCENT
    AUTO_CALIBRATE_OVERVOLUME = DEFAULT_AUTO_CALIBRATE_OVERVOLUME
    OVERVOLUME_CALIBRATION_BUFFER_UL = DEFAULT_OVERVOLUME_CALIBRATION_BUFFER_UL
    OVERVOLUME_MAX_BASE_UL = DEFAULT_OVERVOLUME_MAX_BASE_UL
    OVERVOLUME_MAX_PERCENT = DEFAULT_OVERVOLUME_MAX_PERCENT
    
    print("✅ Configuration reset complete")

def set_candidate_selection_percentile(percentile):
    """Set the percentile used for candidate selection.
    
    Args:
        percentile: Target percentile (0-100). Common values:
                   - 10: Aggressive (pick fast candidates) 
                   - 25: Conservative (avoid fastest, pick reasonably fast)
                   - 50: Median (middle of the pack)
    """
    global CANDIDATE_SELECTION_PERCENTILE
    
    if not (0 <= percentile <= 100):
        raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
    
    old_percentile = CANDIDATE_SELECTION_PERCENTILE
    CANDIDATE_SELECTION_PERCENTILE = percentile
    
    print(f"🎯 Candidate selection percentile: {old_percentile} → {percentile}")
    print(f"   Strategy: {'Aggressive (fast)' if percentile <= 15 else 'Conservative (avoid fastest)' if percentile <= 30 else 'Moderate (balanced)'}")

def get_current_config_summary():
    """Print a summary of current configuration settings for logging/debugging."""
    print("📋 CURRENT EXPERIMENT CONFIG:")
    print(f"   Liquid: {LIQUID}")
    print(f"   Simulate: {SIMULATE}")
    print(f"   Volumes: {[f'{v*1000:.0f}uL' for v in VOLUMES]}")
    print(f"   Candidate selection: P{CANDIDATE_SELECTION_PERCENTILE}")
    print(f"   Precision replicates: {PRECISION_REPLICATES}")
    print(f"   Initial suggestions: {INITIAL_SUGGESTIONS}")
    print(f"   LLM screening/optimization: {USE_LLM_FOR_SCREENING}/{USE_LLM_FOR_OPTIMIZATION}")
    print(f"   Bayesian model: {BAYESIAN_MODEL_TYPE}")
    print(f"   Overaspirate: {OVERASPIRATE_BASE_UL:.1f}uL + {OVERASPIRATE_SCALING_PERCENT:.1f}%")

# Configure autosave directory - respect environment variable first, then fall back to defaults
DEFAULT_LOCAL_AUTOSAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'calibration_runs'))
DEFAULT_HARDWARE_AUTOSAVE_DIR = 'C:\\Users\\Imaging Controller\\Desktop\\Calibration_SDL_Output\\New_Method'

# Always check environment variable first
BASE_AUTOSAVE_DIR = os.environ.get('CALIBRATION_AUTOSAVE_DIR')

if BASE_AUTOSAVE_DIR is None:
    # No environment variable set, use simulation-based defaults
    if SIMULATE:
        BASE_AUTOSAVE_DIR = DEFAULT_LOCAL_AUTOSAVE_DIR
        os.makedirs(BASE_AUTOSAVE_DIR, exist_ok=True)
        print(f"[info] Simulation mode: Using local autosave directory")
    else:
        # Check if hardware directory exists and is writable, otherwise fall back to local
        try:
            if os.path.exists(os.path.dirname(DEFAULT_HARDWARE_AUTOSAVE_DIR)):
                # Test write access
                test_dir = os.path.join(DEFAULT_HARDWARE_AUTOSAVE_DIR, 'test_write_access')
                os.makedirs(test_dir, exist_ok=True)
                os.rmdir(test_dir)  # Clean up test
                BASE_AUTOSAVE_DIR = DEFAULT_HARDWARE_AUTOSAVE_DIR
                print(f"[info] Hardware mode: Using hardware autosave directory")
            else:
                raise PermissionError("Hardware directory not accessible")
        except (PermissionError, OSError) as e:
            print(f"[warning] Cannot access hardware directory ({e}), falling back to local directory")
            BASE_AUTOSAVE_DIR = DEFAULT_LOCAL_AUTOSAVE_DIR
            os.makedirs(BASE_AUTOSAVE_DIR, exist_ok=True)
else:
    # Environment variable set, ensure it exists
    os.makedirs(BASE_AUTOSAVE_DIR, exist_ok=True)
    print(f"[info] Using environment-specified autosave directory")

print(f"[info] Final BASE_AUTOSAVE_DIR={BASE_AUTOSAVE_DIR}")

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
    
    # Convert to uL for easier GCD calculation
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
    volume_ul = volume_ml * 1000  # Convert to uL
    scaling_volume = volume_ul * (OVERASPIRATE_SCALING_PERCENT / 100.0)
    max_overaspirate = OVERASPIRATE_BASE_UL + scaling_volume
    
    # Safety check: ensure minimum range for optimization
    # Ax requires upper bound > lower bound, so we need at least 1uL range
    min_overaspirate = 1.0  # Minimum 1uL to ensure valid parameter range
    if max_overaspirate < min_overaspirate:
        print(f"   ⚠️  Calculated max overaspirate ({max_overaspirate:.1f}uL) too low, using minimum ({min_overaspirate:.1f}uL)")
        max_overaspirate = min_overaspirate
    
    return max_overaspirate

def calculate_dynamic_time_cutoff_with_dual_criteria(current_trial_data, volume_ml=None, min_trials=6):
    """
    Simple dynamic cutoff: decide when to stop optimization based on hit rate and time spread.
    
    Logic:
    - If hit rate < 20% and ≥6 accurate trials → STOP_NOW (hard liquid, save wells)  
    - If hit rate high + time spread low → STOP_NOW (confident, good data)
    - If hit rate high + time spread high → CONTINUE (need more data to find sweet spot)
    
    Args:
        current_trial_data: List of dicts with {'time': float, 'accurate': bool, 'volume': float}
        volume_ml: Target volume for filtering (if None, uses all data) 
        min_trials: Minimum accurate trials needed (default: 6)
    
    Returns:
        dict with 'decision' ('STOP_NOW', 'STOP_SOON', 'CONTINUE'), hit_rate_pct, time_spread_pct, etc.
        Returns None if insufficient data
    """
    if not current_trial_data:
        return None
    
    # Convert to DataFrame and filter by volume
    df = pd.DataFrame(current_trial_data) if isinstance(current_trial_data, list) else current_trial_data.copy()
    if volume_ml is not None and 'volume' in df.columns:
        df = df[abs(df['volume'] - volume_ml) < 0.001]
    
    # Calculate hit rate (percentage of accurate trials)
    if 'accurate' in df.columns:
        accurate_trials = df[df['accurate'] == True]
        hit_rate_pct = (len(accurate_trials) / len(df)) * 100 if len(df) > 0 else 0
    else:
        accurate_trials = df
        hit_rate_pct = 100.0
    
    # Need at least min_trials accurate trials
    if len(accurate_trials) < min_trials:
        return None
    
    # Calculate time spread (simple standard deviation approach)
    times = accurate_trials['time'].values
    time_spread_pct = (np.std(times) / np.mean(times)) * 100 if len(times) > 1 else 0
    
    # MULTI-FACTOR DECISION ALGORITHM
    hit_rate_norm = min(hit_rate_pct / 100.0, 1.0)
    spread_norm = min(time_spread_pct / 100.0, 1.0)
    
    # Decision factors
    confidence_factor = max(0, 1 - spread_norm)  # High when spread is low (0-1)
    difficulty_factor = hit_rate_norm            # High when hit rate is high (0-1) 
    sample_factor = min(len(accurate_trials) / 10.0, 1.0)  # Saturates at 10 trials
    
    # Resource-aware decision logic for laboratory constraints
    easy_liquid_score = difficulty_factor * confidence_factor  # 0-1 range
    
    # Hard liquid evidence: stop earlier when hit rate is very low
    # If hit rate < 20% after reasonable sample, it's likely a very hard liquid
    if hit_rate_norm < 0.2 and len(accurate_trials) >= 6:
        # Force early stop for very difficult liquids to preserve wells
        hard_liquid_evidence = 0.7  # High enough to trigger STOP_NOW
    else:
        hard_liquid_evidence = sample_factor * (1 - difficulty_factor)
    
    # Combined score: stop for either easy+confident OR hard+enough_samples
    decision_score = max(easy_liquid_score, hard_liquid_evidence)
    
    # Decision thresholds
    if decision_score >= 0.5:
        decision = "STOP_NOW"
    elif decision_score >= 0.3:
        decision = "STOP_SOON"
    else:
        decision = "CONTINUE"
    
    return {
        'decision': decision,
        'decision_score': decision_score,
        'hit_rate_pct': hit_rate_pct,
        'time_spread_pct': time_spread_pct,
        'trials_used': len(accurate_trials),
        'total_trials': len(df),
        'strategy': f"P{CANDIDATE_SELECTION_PERCENTILE}",
        'cutoff': np.percentile(times, CANDIDATE_SELECTION_PERCENTILE),
        'spread_method': "std_dev",
        # Multi-factor diagnostics (the smart part!)
        'easy_liquid_score': easy_liquid_score,
        'hard_liquid_evidence': hard_liquid_evidence,
        'confidence_factor': confidence_factor,
        'difficulty_factor': difficulty_factor,
        'sample_factor': sample_factor
    }

def isolate_pipetting_params(source_params, context="unknown"):
    """
    Extract only clean pipetting parameters, filtering out all metadata and contamination.
    
    Args:
        source_params: Dictionary that may contain contaminated or extra fields
        context: Description for debugging (e.g., "precision_test", "optimization")
    
    Returns:
        dict: Clean parameters containing only valid pipetting parameters
    """
    VALID_PIPETTING_PARAMS = {
        'aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 'dispense_wait_time', 
        'retract_speed', 'blowout_vol', 'post_asp_air_vol', 'overaspirate_vol'
    }
    
    # Extract only valid pipetting parameters
    clean_params = {k: v for k, v in source_params.items() if k in VALID_PIPETTING_PARAMS}
    
    # Log any contamination that was filtered out for debugging
    contaminated_keys = set(source_params.keys()) - VALID_PIPETTING_PARAMS
    if contaminated_keys:
        print(f"   🧹 [{context}] Filtered out contaminated keys: {contaminated_keys}")
    
    # DEBUG: Always show what we're returning
    print(f"   🔍 [{context}] Returning clean_params: {clean_params}")
    
    return clean_params

def record_achieved_tolerance(volume, tolerance_multiplier):
    """
    Record the tolerance multiplier that was actually achieved for a volume.
    
    Args:
        volume (float): Volume in mL
        tolerance_multiplier (float): Multiplier used (1.0 = original, 1.5 = 1.5x, 2.0 = 2x)
    """
    global ACHIEVED_TOLERANCE_MULTIPLIERS
    ACHIEVED_TOLERANCE_MULTIPLIERS[volume] = tolerance_multiplier
    
    if tolerance_multiplier == 1.0:
        print(f"   📊 TOLERANCE LEARNING: Volume {volume*1000:.0f}μL achieved original tolerance")
    else:
        print(f"   📊 TOLERANCE LEARNING: Volume {volume*1000:.0f}μL needed {tolerance_multiplier}x tolerance")

def get_adaptive_tolerance_multiplier(volume, original_tolerance_ul):
    """
    Get the adaptive tolerance multiplier for a new volume based on learned tolerances.
    
    Args:
        volume (float): Target volume in mL
        original_tolerance_ul (float): Original tolerance in μL
        
    Returns:
        tuple: (tolerance_multiplier, adaptive_tolerance_ul, explanation)
    """
    global ACHIEVED_TOLERANCE_MULTIPLIERS, ADAPTIVE_TOLERANCE_MODE
    
    if not ADAPTIVE_TOLERANCE_MODE or not ACHIEVED_TOLERANCE_MULTIPLIERS:
        return 1.0, original_tolerance_ul, "No adaptive learning (first volume or disabled)"
    
    # Find the most relevant learned tolerance
    # Strategy: Use the maximum tolerance multiplier from smaller or similar volumes
    # Rationale: If smaller volumes needed relaxed tolerance, larger volumes likely need at least the same
    
    relevant_multipliers = []
    for learned_volume, multiplier in ACHIEVED_TOLERANCE_MULTIPLIERS.items():
        if learned_volume <= volume:  # Only consider same or smaller volumes
            relevant_multipliers.append(multiplier)
    
    if not relevant_multipliers:
        # No smaller volumes completed yet - use original tolerance
        return 1.0, original_tolerance_ul, "No smaller volumes completed yet"
    
    # Use the maximum tolerance multiplier needed by smaller volumes
    # This provides a conservative approach - if any smaller volume needed relaxed tolerance,
    # assume this volume will need at least the same relaxation
    adaptive_multiplier = max(relevant_multipliers)
    adaptive_tolerance_ul = original_tolerance_ul * adaptive_multiplier
    
    if adaptive_multiplier == 1.0:
        explanation = f"All smaller volumes achieved original tolerance"
    else:
        explanation = f"Smaller volumes needed up to {adaptive_multiplier}x tolerance"
    
    return adaptive_multiplier, adaptive_tolerance_ul, explanation

def reset_tolerance_learning():
    """Reset tolerance learning for a new experiment."""
    global ACHIEVED_TOLERANCE_MULTIPLIERS
    ACHIEVED_TOLERANCE_MULTIPLIERS = {}
    print("🔄 Reset adaptive tolerance learning")

def select_best_candidate_from_accurate_trials(accurate_candidates):
    """
    Select the best candidate from accurate trials using configurable percentile selection.
    
    Strategy: Use sophisticated scoring that balances time proximity to target percentile 
    and accuracy (deviation). Scoring uses normalized metrics with configurable weights 
    and non-linear scaling to emphasize accuracy while still considering time performance.
    
    Scoring Formula:
        combined_score = (TIME_WEIGHT × scaled_distance) + (ACCURACY_WEIGHT × scaled_deviation)
        where scaled_distance = (norm_distance)^1.2 and scaled_deviation = (norm_deviation)^1.8
        
    Default weights: 40% time proximity, 60% accuracy (configurable via constants)
    
    Args:
        accurate_candidates: List of dicts with candidate data
                            Each dict should have: {'params': dict, 'deviation': float, 'time': float}
        time_spread_strategy: Ignored - kept for compatibility
    
    Returns:
        dict: Selected candidate with all original fields plus 'selection_reason'
        Returns None if no candidates provided
    """
    if not accurate_candidates or len(accurate_candidates) == 0:
        return None
    
    # If only one candidate, return it
    if len(accurate_candidates) == 1:
        candidate = accurate_candidates[0].copy()
        candidate['selection_reason'] = "only_candidate"
        return candidate
    
    import numpy as np
    
    # Extract times for analysis
    times = [c['time'] for c in accurate_candidates]
    times_array = np.array(times)
    
    print(f"   📊 CANDIDATE SELECTION: {len(accurate_candidates)} accurate candidates")
    print(f"      Time range: {np.min(times_array):.1f}s - {np.max(times_array):.1f}s")
    print(f"      Using P{CANDIDATE_SELECTION_PERCENTILE} selection (configured)")
    
    # Calculate target time using configured percentile
    target_time = np.percentile(times_array, CANDIDATE_SELECTION_PERCENTILE)
    
    # Find candidate closest to target time
    # Add small preference for lower deviation if times are very close
    best_candidate = None
    best_score = float('inf')
    
    # Use the same unified scoring system (40% time, 60% accuracy)
    distances = [abs(c['time'] - target_time) for c in accurate_candidates]
    deviations = [c.get('deviation', 0) for c in accurate_candidates]
    
    max_distance = max(distances) if max(distances) > 0 else 1.0
    max_deviation = max(deviations) if max(deviations) > 0 else 1.0
    
    # Same weights used everywhere
    TIME_WEIGHT = 0.4      # 40% weight on time proximity
    ACCURACY_WEIGHT = 0.6  # 60% weight on accuracy
    
    for candidate in accurate_candidates:
        # Calculate normalized scores
        time_distance = abs(candidate['time'] - target_time)
        norm_distance = time_distance / max_distance
        norm_deviation = candidate.get('deviation', 0) / max_deviation
        
        # Apply same non-linear scaling
        scaled_distance = norm_distance ** 1.2  # Mild penalty for time differences
        scaled_deviation = norm_deviation ** 1.8 # Strong penalty for accuracy problems
        
        # Same scoring formula
        combined_score = (TIME_WEIGHT * scaled_distance) + (ACCURACY_WEIGHT * scaled_deviation)
        
        if combined_score < best_score:
            best_score = combined_score
            best_candidate = candidate.copy()
            # Add scoring details for debugging
            best_candidate['selection_score_breakdown'] = {
                'combined_score': combined_score,
                'time_component': TIME_WEIGHT * scaled_distance,
                'accuracy_component': ACCURACY_WEIGHT * scaled_deviation,
                'time_distance': time_distance,
                'norm_distance': norm_distance,
                'norm_deviation': norm_deviation
            }
    
    # Add selection metadata to the best candidate
    if best_candidate:
        best_candidate['selection_reason'] = f"percentile_p{CANDIDATE_SELECTION_PERCENTILE}"
        best_candidate['target_time'] = target_time
        best_candidate['target_percentile'] = CANDIDATE_SELECTION_PERCENTILE
        best_candidate['configured_percentile'] = CANDIDATE_SELECTION_PERCENTILE
        
        # Show detailed scoring breakdown
        breakdown = best_candidate.get('selection_score_breakdown', {})
        total_score = breakdown.get('combined_score', 0)
        time_score = breakdown.get('time_component', 0)
        accuracy_score = breakdown.get('accuracy_component', 0)
        
        print(f"      ✅ Selected: {best_candidate['time']:.1f}s candidate (P{CANDIDATE_SELECTION_PERCENTILE})")
        print(f"      🎯 Target: {target_time:.1f}s, actual: {best_candidate['time']:.1f}s, dev: {best_candidate.get('deviation', 0):.1f}%")
        print(f"      📊 Selection score: {total_score:.3f} (time: {time_score:.3f}, accuracy: {accuracy_score:.3f})")
    
    return best_candidate

def get_volume_dependent_tolerances(volume_ml, is_first_volume=True, historical_data=None):
    """Calculate volume-dependent tolerances based on scalable volume ranges.
    
    Args:
        volume_ml: Volume in milliliters
        is_first_volume: Legacy parameter (kept for compatibility, no longer affects behavior)
    """
    volume_ul = volume_ml * 1000  # Convert to uL
    
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
    
    # Convert percentage to absolute uL tolerance (same for both deviation and variation)
    base_tolerance_ul = volume_ul * (tolerance_pct / 100.0)
    
    # Apply adaptive tolerance learning for subsequent volumes
    if not is_first_volume and ADAPTIVE_TOLERANCE_MODE:
        adaptive_multiplier, adaptive_tolerance_ul, explanation = get_adaptive_tolerance_multiplier(volume_ml, base_tolerance_ul)
        if adaptive_multiplier > 1.0:
            print(f"   🎯 ADAPTIVE TOLERANCE: Using {adaptive_multiplier}x tolerance for {volume_ml*1000:.0f}μL")
            print(f"      Reason: {explanation}")
            print(f"      Tolerance: ±{base_tolerance_ul:.1f}μL → ±{adaptive_tolerance_ul:.1f}μL")
            tolerance_ul = adaptive_tolerance_ul
        else:
            tolerance_ul = base_tolerance_ul
    else:
        tolerance_ul = base_tolerance_ul
    
    # NO TIME CONSTRAINTS FOR CALIBRATION - accuracy only
    time_seconds = None
    time_optimal_target = None
    
    if is_first_volume:
        print(f"   📊 First volume - NO TIME CONSTRAINTS (accuracy-only calibration)")
    else:
        print(f"   📊 Subsequent volume - NO TIME CONSTRAINTS (accuracy-only calibration)")
        if ADAPTIVE_TOLERANCE_MODE and ACHIEVED_TOLERANCE_MULTIPLIERS:
            print(f"   📚 Learned tolerances from previous volumes: {ACHIEVED_TOLERANCE_MULTIPLIERS}")
    
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
            time_optimal_target = None  # No time optimization target - use raw time
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
    global _CACHED_LASH_E
    DENSITY_LIQUID = LIQUIDS[LIQUID]["density"]
    NEW_PIPET_EACH_TIME_SET = LIQUIDS[LIQUID]["refill_pipets"]
    state = {"measurement_vial_index": 0, "measurement_vial_name": "measurement_vial_0"}

    # Reuse lash_e across experiments to avoid repeated hardware initialization
    if _CACHED_LASH_E is None:
        print("[DEBUG] Creating new Lash_E controller...")
        _CACHED_LASH_E = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False)
        print("[DEBUG] Lash_E created, testing logger...")
        _CACHED_LASH_E.logger.info("Creating new Lash_E controller")
        print("[DEBUG] Logger test successful")
        # Only perform the (potentially slow) input file integrity check on first creation
        try:
            _CACHED_LASH_E.nr_robot.check_input_file()
        except Exception as e:
            _CACHED_LASH_E.logger.warning(f"Initial check_input_file failed: {e}")
    else:
        print("[DEBUG] Reusing existing Lash_E controller...")
        _CACHED_LASH_E.logger.info(f"Reusing existing Lash_E controller for experiment index {EXPERIMENT_INDEX}")
        print("[DEBUG] Reuse logging successful")

    # Always move a fresh measurement vial into place for each experiment run
    try:
        _CACHED_LASH_E.nr_robot.move_vial_to_location("measurement_vial_0", "clamp", 0)
    except Exception as e:
        _CACHED_LASH_E.logger.warning(f"Could not move measurement vial: {e}")

    return _CACHED_LASH_E, DENSITY_LIQUID, NEW_PIPET_EACH_TIME_SET, state

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
        # Note: lash_e not available in this context, use print for now
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
        # Only fix actual pipetting parameters, not metadata fields
        fixed_params = {k: v for k, v in first_successful_params.items() 
                      if k in ALL_PARAMS and k not in VOLUME_DEPENDENT_PARAMS}
        print(f"   Using parameters from FIRST successful volume {completed_volumes[0][0]*1000:.0f}uL (most transferable)")
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
                    result_volume = result.get('volume', 0.1)  # Default to 100uL if not specified
                    volume_tolerances = get_volume_dependent_tolerances(result_volume, is_first_volume=True, historical_data=None)  # Historical data includes time
                    scaled_optimal_target = volume_tolerances.get('time_optimal_target')
                    
                    # Use raw time directly - no transformations
                    time_score = raw_time
                    
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

def get_llm_suggestions(ax_client, n, all_results, volume=None, liquid=None):
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
        # deviation is % already. Compute absolute uL deviation for original logic
        absolute_deviation_ul = (df['deviation'] / 100) * (df['volume'] * 1000)

        if SIMULATE:
            # In simulation: allow pass if percent deviation <= dynamic_pct OR absolute deviation <= specified uL
            max_pct = 100.0  # Extremely lenient percent threshold for simulation
            try:
                max_pct = float(os.environ.get('SIM_MAX_DEV_PCT', '100'))
            except ValueError:
                pass
            
            # ACCURACY ONLY - no time constraints for calibration
            meets_criteria = (df['deviation'] <= max_pct) | (absolute_deviation_ul <= criteria['max_deviation_ul'])
        else:
            # Real mode - ACCURACY ONLY - no time constraints for calibration
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
    
    # Step 1: Select best candidate (lowest deviation)
    print(f"\n🔬 OVERVOLUME CALIBRATION: Selecting best candidate from {len(screening_candidates)} options...")
    
    # Select candidate with lowest deviation (accuracy-focused calibration)
    best_candidate = min(screening_candidates, key=lambda x: x.get('deviation', float('inf')))
    best_params = best_candidate['params'].copy()  # CRITICAL: Always copy to prevent contamination
    
    print(f"   ✅ Selected candidate: {best_candidate['deviation']:.1f}% deviation, {best_candidate['time']:.1f}s")
    print(f"   📋 Testing these parameters on {len(remaining_volumes)} additional volumes...")
    
    # Step 2: Start with the first volume's data from the selected candidate
    calibration_data = []
    
    # Add the first volume's measurement data (the volume that was tested in SOBOL)
    # The first volume is the one NOT in remaining_volumes
    first_volume = [v for v in VOLUMES if v not in remaining_volumes][0]  # Should be VOLUMES[0]
    first_volume_ul = first_volume * 1000  # Convert to uL
    deviation_pct = best_candidate['deviation']
    
    # Calculate measured volume from deviation
    # Deviation = (target - measured) / target * 100
    # So: measured = target * (1 - deviation/100)
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
        
        # Single measurement (n=1 as specified)
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
        
        # CRITICAL: Account for existing overaspirate in total recommendation
        # Total overaspirate needed = existing_overaspirate + additional_shortfall + buffer
        
        # Calculate the average existing overaspirate to add to our formula
        avg_existing_overaspirate = np.mean(existing_overaspirates)
        print(f"   📊 Average existing overaspirate: {avg_existing_overaspirate:.1f}uL")
        
        # Calculate total overaspirate needed (existing + additional + buffer)
        min_overaspirate = 2.0  # Minimum 2uL to prevent crashes
        
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
            print(f"   � Adjusted base by +{adjustment:.1f}uL to ensure minimum {min_overaspirate:.1f}uL overaspirate")
        
        print(f"   🎯 Calibrated formula: overaspirate = {new_base_ul:.1f}uL + {new_scaling_percent:.1f}% * volume")
        
        # Step 4: Apply safety bounds
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
        # If both base and scaling are 0, we'd have no overaspirate volume at all
        if new_base_ul == 0 and new_scaling_percent == 0:
            print(f"   ⚠️  Both base and scaling are 0 - setting minimum base to 1uL to maintain optimization capability")
            new_base_ul = 1.0
        
        print(f"   ✅ Final calibrated values: base = {new_base_ul:.1f}uL, scaling = {new_scaling_percent:.1f}%")
        
        # Store raw shortfall coefficients for accurate reporting
        for point in calibration_data:
            point['slope'] = slope  # Store raw slope coefficient
            point['intercept'] = intercept  # Store raw intercept coefficient
        
        # Generate calibration plot (only for first experiment to avoid pauses)
        try:
            if EXPERIMENT_INDEX != 0:
                print(f"   📊 Skipping plot generation for experiment index {EXPERIMENT_INDEX}")
                return new_base_ul, new_scaling_percent, calibration_data
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
            plt.xlabel('Target Volume (uL)', fontsize=12)
            plt.ylabel('Measured Volume (uL)', fontsize=12)
            plt.title(f'Overvolume Calibration Results\nCalibrated: {new_base_ul:.1f}uL + {new_scaling_percent:.1f}%', fontsize=14, fontweight='bold')
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

def get_ordered_candidates_from_results(optimization_results, criteria):
    """
    Extract ordered candidates from optimization results using sweet spot analysis.
    
    Args:
        optimization_results: List of optimization trial results 
        criteria: Criteria dict with accuracy/time thresholds
        
    Returns:
        list: Ordered candidates (best first), each with 'params', 'deviation', 'time', etc.
        Returns empty list if no candidates meet criteria.
    """
    if not optimization_results:
        return []
    
    # Filter for accurate candidates that meet the basic criteria
    accurate_candidates = []
    for result in optimization_results:
        # Check if result meets accuracy criteria (using same logic as meets_criteria)
        df = pd.DataFrame([result])  # Convert single result to DataFrame
        if 'deviation' in df.columns and 'volume' in df.columns:
            absolute_deviation_ul = (df['deviation'] / 100) * (df['volume'] * 1000)
            
            if SIMULATE:
                # In simulation: allow pass if percent deviation <= 100% OR absolute deviation <= criteria
                max_pct = 100.0  # Extremely lenient percent threshold for simulation
                try:
                    max_pct = float(os.environ.get('SIM_MAX_DEV_PCT', '100'))
                except ValueError:
                    pass
                
                # ACCURACY ONLY - no time constraints for calibration
                meets_criteria = (df['deviation'].iloc[0] <= max_pct) | (absolute_deviation_ul.iloc[0] <= criteria['max_deviation_ul'])
            else:
                # Real mode - ACCURACY ONLY - no time constraints for calibration
                meets_criteria = (absolute_deviation_ul.iloc[0] <= criteria['max_deviation_ul'])
            
            if meets_criteria:
                # Format as candidate for sweet spot analysis
                candidate = {
                    'params': {k: v for k, v in result.items() if k not in ['deviation', 'time', 'volume', 'trial_index', 'strategy', 'liquid', 'time_reported']},
                    'deviation': result.get('deviation', 0),
                    'time': result.get('time', 0),
                    'score': result.get('deviation', 0),  # Use deviation as score
                    'trial_number': result.get('trial_index', 0)
                }
                accurate_candidates.append(candidate)
    
    if not accurate_candidates:
        return []
    
    # Use sweet spot analysis to identify the optimal region, then rank ALL candidates
    selected_candidate = select_best_candidate_from_accurate_trials(accurate_candidates)
    
    if not selected_candidate:
        return []
    
    # Get sweet spot candidates and target time from analysis
    sweet_spot_candidates = selected_candidate.get('all_sweet_spot_candidates', [])
    target_time = selected_candidate.get('target_time', selected_candidate.get('time', 0))
    
    # Create comprehensive ranking of ALL candidates by distance from sweet spot target
    all_candidates_with_distance = []
    
    for candidate in accurate_candidates:
        candidate_time = candidate.get('time', 0)
        distance_from_target = abs(candidate_time - target_time)
        
        # Check if this candidate is in the sweet spot
        is_in_sweet_spot = any(
            c.get('trial_number') == candidate.get('trial_number') 
            for c in sweet_spot_candidates
        )
        
        candidate_copy = candidate.copy()
        candidate_copy['distance_from_sweet_spot'] = distance_from_target
        candidate_copy['is_sweet_spot'] = is_in_sweet_spot
        
        all_candidates_with_distance.append(candidate_copy)
    
    # Unified scoring system: 40% time distance, 60% accuracy
    if len(all_candidates_with_distance) > 1:
        distances = [c['distance_from_sweet_spot'] for c in all_candidates_with_distance]
        deviations = [c['deviation'] for c in all_candidates_with_distance]
        
        max_distance = max(distances) if max(distances) > 0 else 1.0
        max_deviation = max(deviations) if max(deviations) > 0 else 1.0
        
        # Single set of weights used everywhere
        TIME_WEIGHT = 0.4      # 40% weight on time proximity  
        ACCURACY_WEIGHT = 0.6  # 60% weight on accuracy
        
        for candidate in all_candidates_with_distance:
            # Normalize both metrics to 0-1 scale
            norm_distance = candidate['distance_from_sweet_spot'] / max_distance
            norm_deviation = candidate['deviation'] / max_deviation
            
            # Apply non-linear scaling (your preferred approach)
            scaled_distance = norm_distance ** 1.2  # Mild penalty for time differences
            scaled_deviation = norm_deviation ** 1.8 # Strong penalty for accuracy problems
            
            # Calculate combined score (lower is better)
            candidate['combined_score'] = (TIME_WEIGHT * scaled_distance) + (ACCURACY_WEIGHT * scaled_deviation)
            
            # Debug info for transparency
            candidate['score_breakdown'] = {
                'time_component': TIME_WEIGHT * scaled_distance,
                'accuracy_component': ACCURACY_WEIGHT * scaled_deviation,
                'norm_distance': norm_distance,
                'norm_deviation': norm_deviation
            }
    else:
        # Single candidate - just assign score of 0
        all_candidates_with_distance[0]['combined_score'] = 0.0
        all_candidates_with_distance[0]['score_breakdown'] = {}
    
    # Sort by combined score (lower is better)
    ordered_candidates = sorted(all_candidates_with_distance, key=lambda c: c['combined_score'])
    
    return ordered_candidates

def run_precision_test(lash_e, state, best_params, volume, expected_mass, expected_time, autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set, max_variation_ul, all_results=None):
    print(f"🎯 PRECISION TEST: Testing candidate parameters with {PRECISION_REPLICATES} replicates...")
    
    # DEBUG: Show what we received before cleaning
    print(f"   🔍 [DEBUG] Raw best_params received: {best_params}")
    
    # CRITICAL: Isolate clean parameters to prevent contamination
    clean_params = isolate_pipetting_params(best_params, "precision_test")
    
    # DEBUG: Show what we got after cleaning
    print(f"   🔍 [DEBUG] Clean params after isolation: {clean_params}")
    
    # Calculate acceptable range around target volume using absolute tolerance
    target_volume = volume  # mL
    variation_range = max_variation_ul / 1000  # Convert uL to mL
    min_acceptable = target_volume - variation_range
    max_acceptable = target_volume + variation_range
    
    print(f"   Target: {target_volume*1000:.0f}uL, Range: {min_acceptable*1000:.0f}uL - {max_acceptable*1000:.0f}uL (±{max_variation_ul:.0f}uL)")
    
    measurements = []
    deviations = []
    times = []  # Capture timing data
    
    # Store the starting point in raw_measurements to track precision replicate numbers
    precision_start_idx = len(raw_measurements)
    
    for i in range(PRECISION_REPLICATES):
        print(f"   Replicate {i+1}/{PRECISION_REPLICATES}...", end=" ")
        check_if_measurement_vial_full(lash_e, state)
        
        # Get single measurement result
        liquid_source = get_liquid_source_with_vial_management(lash_e, state)
        
        # DEBUG: Verify clean_params at start of each iteration 
        print(f"🔍 [DEBUG] Iteration {i+1} - clean_params keys: {list(clean_params.keys())}")
        if 'variability' in clean_params:
            print(f"� [DEBUG] CONTAMINATION DETECTED! clean_params contains 'variability': {clean_params['variability']}")
        
        # CRITICAL: Make a fresh copy to prevent cross-iteration contamination
        iteration_params = clean_params.copy()
        
        result = pipet_and_measure(lash_e, liquid_source, state["measurement_vial_name"], volume, iteration_params, expected_mass, expected_time, 1, SIMULATE, autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set, "PRECISION")
        
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
            print(f"❌ FAILED ({current_volume*1000:.0f}uL outside range)")
            print(f"   Precision test FAILED after {len(measurements)} replicates")
            return False, measurements, times[:len(measurements)]
        else:
            print(f"✅ {current_volume*1000:.0f}uL")
        
        # Add precision test measurement to all_results for tracking
        if all_results is not None:
            precision_result = dict(clean_params)  # Copy clean parameters only
            precision_result.update({
                "volume": volume,
                "deviation": deviation,
                "time": result.get('time', 0),
                "trial_type": "PRECISION",
                "strategy": "PRECISION_TEST",
                "liquid": liquid,
                "trial_index": f"precision_{i+1}",
                "time_reported": datetime.now().isoformat(),
                "precision_replicate": i+1,
                "target_volume": target_volume,
                "acceptable_range": f"{min_acceptable*1000:.1f}-{max_acceptable*1000:.1f}uL"
            })
            all_results.append(precision_result)
    
    # If we reach here, all measurements were within the acceptable range
    mean_volume = np.mean(measurements)
    std_volume = np.std(measurements)
    cv_percent = (std_volume / mean_volume) * 100  # Coefficient of variation
    
    print(f"   ✅ PRECISION TEST PASSED: Mean {mean_volume*1000:.0f}uL ± {std_volume*1000:.1f}uL (CV: {cv_percent:.1f}%)")
    
    return True, measurements, times

def run_cascading_precision_tests(optimization_results, criteria, lash_e, state, volume, expected_mass, expected_time, autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set, tolerances, blacklisted_params, volume_index=None):
    """
    Try precision testing on multiple candidates with adaptive fallback.
    
    Strategy:
    1. Require at least 6 candidates from optimization
    2. Rank ALL candidates by distance from sweet spot (sweet spot first)
    3. Test up to 5 candidates; if first passes, success!
    4. If 5 candidates fail, complete remaining replicates for best candidate (adaptive fallback)
    
    Args:
        optimization_results: All optimization trial results
        criteria: Accuracy/time criteria 
        lash_e, state, etc.: Standard experiment parameters
        blacklisted_params: List of parameter sets to avoid
        
    Returns:
        tuple: (success, candidate_params, precision_measurements, precision_times, candidate_info, is_adaptive)
        is_adaptive=True means we used fallback due to tolerance issues
    """
    print(f"\n🎯 CASCADING PRECISION TESTS: Finding robust candidate...")
    
    # DYNAMIC CUTOFF ANALYSIS: Only run on first volume (volume_index == 0)
    # For subsequent volumes, we use transferred parameters and don't need intelligent stopping
    is_first_volume = (volume_index == 0) if volume_index is not None else ('max_time' in criteria)  # Direct check
    
    if is_first_volume:
        print(f"\n📊 DYNAMIC CUTOFF ANALYSIS: Evaluating optimization progress (first volume only)...")
        
        # Prepare data for dynamic analysis - convert optimization results to format expected by algorithm
        current_trial_data = []
        for result in optimization_results:
            if 'deviation' in result and 'time' in result and 'volume' in result:
                # Calculate if this trial meets accuracy criteria (same logic as check_optimization_criteria)
                absolute_deviation_ul = (result['deviation'] / 100) * (result['volume'] * 1000)
                
                if SIMULATE:
                    max_pct = 100.0
                    try:
                        max_pct = float(os.environ.get('SIM_MAX_DEV_PCT', '100'))
                    except ValueError:
                        pass
                    meets_accuracy = (result['deviation'] <= max_pct) or (absolute_deviation_ul <= criteria['max_deviation_ul'])
                else:
                    meets_accuracy = absolute_deviation_ul <= criteria['max_deviation_ul']
                
                current_trial_data.append({
                    'time': result['time'],
                    'accurate': meets_accuracy,
                    'volume': result['volume']
                })
        
        # Run dynamic cutoff analysis
        cutoff_result = calculate_dynamic_time_cutoff_with_dual_criteria(current_trial_data, volume_ml=volume, min_trials=6)
        
        if cutoff_result:
            print(f"   📈 ANALYSIS RESULTS:")
            print(f"      Hit rate: {cutoff_result['hit_rate_pct']:.1f}% ({cutoff_result['trials_used']}/{cutoff_result['total_trials']} accurate)")
            print(f"      Time spread: {cutoff_result['time_spread_pct']:.1f}% ({cutoff_result['spread_method']})")
            print(f"      Decision score: {cutoff_result['decision_score']:.3f}")
            print(f"      Algorithm decision: {cutoff_result['decision']}")
            print(f"      Recommended strategy: {cutoff_result['strategy']} (using P{CANDIDATE_SELECTION_PERCENTILE})")
            
            # Show the decision breakdown for transparency
            print(f"      └─ Easy liquid score: {cutoff_result['easy_liquid_score']:.3f}")
            print(f"      └─ Hard liquid evidence: {cutoff_result.get('hard_liquid_evidence', 0):.3f}")
            print(f"      └─ Confidence factor: {cutoff_result['confidence_factor']:.3f}")
            
            # Make decision based on algorithm output
            if cutoff_result['decision'] == 'CONTINUE':
                print(f"   🔄 DYNAMIC DECISION: Continue optimization (high hit rate + high spread = need more data)")
                print(f"   💡 Reasoning: Hit rate is good ({cutoff_result['hit_rate_pct']:.1f}%) but time spread is high ({cutoff_result['time_spread_pct']:.1f}%)")
                print(f"      More trials will help find the sweet spot and reduce variability")
                return False, None, [], [], None, False
            
            elif cutoff_result['decision'] == 'STOP_SOON':
                print(f"   ⚠️  DYNAMIC DECISION: Stop soon (moderate confidence)")
                # Continue to precision testing but with higher priority
            
            elif cutoff_result['decision'] == 'STOP_NOW':
                print(f"   ✅ DYNAMIC DECISION: Stop now and proceed with precision testing")
                if cutoff_result['hit_rate_pct'] < 20:
                    print(f"   🔥 Hard liquid detected: {cutoff_result['hit_rate_pct']:.1f}% hit rate < 20% threshold")
                    print(f"      Conserving wells for very challenging liquid")
                else:
                    print(f"   🎯 Confidence achieved: Low time spread ({cutoff_result['time_spread_pct']:.1f}%) + good hit rate")
        else:
            print(f"   📊 Insufficient data for dynamic analysis (need ≥6 accurate trials)")
            print(f"   🔄 Continuing with standard n≥6 candidate requirement...")
    
    else:
        print(f"\n📊 SUBSEQUENT VOLUME: Skipping dynamic cutoff analysis (using transferred parameters)")
        print(f"   🎯 Strategy: Using P{CANDIDATE_SELECTION_PERCENTILE} percentile selection from simplified algorithm")
    
    # Get ordered candidates using comprehensive ranking
    ordered_candidates = get_ordered_candidates_from_results(optimization_results, criteria)
    
    if len(ordered_candidates) < 6:
        print(f"   ⚠️  Only {len(ordered_candidates)} candidates found, need at least 6")
        print(f"   🔄 Will continue optimization to accumulate more candidates...")
        # Return None to indicate "insufficient data" rather than "test failed"
        return None, None, [], [], None, False
    
    # TODO: RESTORE CONFIDENCE ASSESSMENT (accidentally deleted in commit 3391578)
    # The system had:
    # - Custom formula combining hit rate (% trials meeting criteria) + time spread (CV of candidate times)
    # - Parameter called "difficult_liquid" that affected thresholds
    # - Would return None to force more optimization if confidence too low
    # - This prevented precision testing on inconsistent parameter sets (critical for water)
    # SEARCH FOR: difficult_liquid, confidence formula, hit_rate calculation
    
    # Filter out blacklisted candidates
    available_candidates = []
    for candidate in ordered_candidates:
        params = candidate['params']
        # Check if these parameters are blacklisted
        is_blacklisted = False
        for blacklisted in blacklisted_params:
            if all(abs(params.get(k, 0) - blacklisted.get(k, 0)) < 1e-6 for k in params.keys() if k in blacklisted):
                is_blacklisted = True
                break
        
        if not is_blacklisted:
            available_candidates.append(candidate)
    
    if len(available_candidates) < 6:
        print(f"   ⚠️  Only {len(available_candidates)} non-blacklisted candidates available, need at least 6")
        print(f"   🔄 Will continue optimization for more candidates...")
        return False, None, [], [], None, False
    
    # Count sweet spot vs non-sweet spot candidates
    sweet_spot_count = sum(1 for c in available_candidates if c.get('is_sweet_spot', False))
    print(f"   📋 Testing {len(available_candidates)} candidates: {sweet_spot_count} in sweet spot, {len(available_candidates) - sweet_spot_count} ranked by distance")
    
    # Track precision test results for adaptive fallback
    precision_results = []
    max_tests = 5
    
    # Try each candidate in order (up to 5 tests)
    for i, candidate in enumerate(available_candidates[:max_tests]):
        candidate_params = isolate_pipetting_params(candidate['params'], f"cascading_test_{i+1}")
        is_sweet_spot = candidate.get('is_sweet_spot', False)
        spot_type = "🎯 sweet spot" if is_sweet_spot else f"📍 distance {candidate.get('distance_from_sweet_spot', 0):.1f}s"
        
        # Show the unified scoring
        score = candidate.get('combined_score', 0)
        score_breakdown = candidate.get('score_breakdown', {})
        time_component = score_breakdown.get('time_component', 0)
        accuracy_component = score_breakdown.get('accuracy_component', 0)
        
        print(f"\n   🧪 CANDIDATE #{i+1}/{max_tests}: {candidate['time']:.1f}s, {candidate['deviation']:.1f}% dev ({spot_type})")
        print(f"      📊 Score: {score:.3f} (time: {time_component:.3f}, accuracy: {accuracy_component:.3f})")
        
        # Run precision test on this candidate
        passed, precision_measurements, precision_times = run_precision_test(
            lash_e, state, candidate_params, volume, expected_mass, expected_time, 
            autosave_raw_path, raw_measurements, liquid, new_pipet_each_time_set, 
            tolerances['variation_ul']
        )
        
        # Store result for potential adaptive fallback
        # Calculate precision score using CV with sample size weight (Option 2)
        if len(precision_measurements) >= 2:
            cv = np.std(precision_measurements) / np.mean(precision_measurements)
            sample_weight = min(len(precision_measurements) / PRECISION_REPLICATES, 1.0)  # Max weight at full replicates
            precision_score = cv / sample_weight  # Lower is better (more precise + more complete)
        else:
            precision_score = float('inf')  # Single measurement or no measurements = worst score
            
        precision_results.append({
            'candidate': candidate,
            'params': candidate_params.copy(),  # Fix: Make copy to avoid reference issues
            'passed': passed,
            'measurements': precision_measurements,
            'times': precision_times,
            'precision_score': precision_score,
            'replicates_completed': len(precision_measurements),
            'cv_percent': cv * 100 if len(precision_measurements) >= 2 else float('inf')
        })
        
        if passed:
            print(f"   ✅ PRECISION TEST PASSED on candidate #{i+1}")
            if len(available_candidates) > 1:
                print(f"   💪 Robust solution: Success on attempt {i+1} with {len(available_candidates) - 1} backup(s)")
            
            # Record that original tolerance was achieved
            record_achieved_tolerance(volume, 1.0)
            
            return True, candidate_params, precision_measurements, precision_times, candidate, False
        else:
            print(f"   ❌ PRECISION TEST FAILED on candidate #{i+1} (variability too high)")
            # Blacklist this candidate for future attempts
            blacklisted_params.append(candidate_params.copy())
    
    # If we reach here, all 5 candidates failed precision testing with original tolerance
    print(f"\n   ❌ ALL {max_tests} CANDIDATES FAILED precision testing with original tolerance (±{tolerances['variation_ul']:.0f}μL)")
    print(f"   🔄 PROGRESSIVE TOLERANCE FALLBACK: Testing relaxed tolerances...")
    
    # Sort candidates by precision score (best first)
    sorted_candidates = sorted(precision_results, key=lambda x: x['precision_score'])
    
    # First, ensure all candidates have full replicates completed (do this only once)
    print(f"\n   🧪 COMPLETING FULL REPLICATES for all candidates...")
    completed_candidates = []
    
    for i, precision_result in enumerate(sorted_candidates):
        candidate = precision_result['candidate']
        candidate_params = precision_result['params'].copy()
        measurements = precision_result['measurements'].copy()
        times = precision_result['times'].copy()
        replicates_completed = precision_result['replicates_completed']
        
        remaining_replicates = PRECISION_REPLICATES - replicates_completed
        if remaining_replicates > 0:
            print(f"   📊 Candidate #{i+1}: Completing {remaining_replicates} remaining replicates...")
            
            for j in range(remaining_replicates):
                replicate_num = replicates_completed + j + 1
                print(f"      Replicate {replicate_num}/{PRECISION_REPLICATES}...", end=" ")
                
                check_if_measurement_vial_full(lash_e, state)
                liquid_source = get_liquid_source_with_vial_management(lash_e, state)
                result = pipet_and_measure(lash_e, liquid_source, state["measurement_vial_name"], 
                                         volume, candidate_params, expected_mass, expected_time, 
                                         1, SIMULATE, autosave_raw_path, raw_measurements, 
                                         liquid, new_pipet_each_time_set, "PRECISION")
                
                # Extract measurement and add to results
                if raw_measurements:
                    actual_mass = raw_measurements[-1]['mass']
                    liquid_density = LIQUIDS[liquid]["density"]
                    actual_volume = actual_mass / liquid_density
                    measurements.append(actual_volume)
                
                # Capture timing data
                time_taken = result.get('time', 0)
                times.append(time_taken)
                
                current_volume = measurements[-1]
                print(f"📋 {current_volume*1000:.0f}μL")
        else:
            print(f"   📊 Candidate #{i+1}: Already has all {PRECISION_REPLICATES} replicates")
        
        # Store completed candidate data
        completed_candidates.append({
            'candidate': candidate,
            'params': candidate_params,
            'measurements': measurements,
            'times': times,
            'cv_percent': np.std(measurements) / np.mean(measurements) * 100 if len(measurements) >= 2 else float('inf')
        })
    
    # Now test progressive tolerance levels using completed data (NO additional trials)
    tolerance_multipliers = [1.5, 2.0]
    
    for multiplier in tolerance_multipliers:
        relaxed_tolerance = tolerances['variation_ul'] * multiplier
        print(f"\n   🎯 TESTING {multiplier}x TOLERANCE: ±{relaxed_tolerance:.1f}μL (using existing {PRECISION_REPLICATES} replicates)")
        
        # Test each completed candidate with this relaxed tolerance  
        for i, completed_result in enumerate(completed_candidates):
            candidate = completed_result['candidate']
            candidate_params = completed_result['params']
            measurements = completed_result['measurements']
            times = completed_result['times']
            
            print(f"   📊 Testing candidate #{i+1}: {candidate['time']:.1f}s, CV: {completed_result['cv_percent']:.1f}%")
            
            # Test if this candidate passes the relaxed tolerance with full replicate count
            if len(measurements) >= PRECISION_REPLICATES:
                # Check if ALL measurements are within relaxed tolerance
                target_volume = volume
                min_acceptable = target_volume - (relaxed_tolerance / 1000.0)  # Convert μL to mL
                max_acceptable = target_volume + (relaxed_tolerance / 1000.0)
                
                all_within_tolerance = all(min_acceptable <= m <= max_acceptable for m in measurements)
                
                if all_within_tolerance:
                    final_cv = np.std(measurements) / np.mean(measurements) * 100
                    print(f"      ✅ CANDIDATE PASSED with {multiplier}x tolerance!")
                    print(f"      📈 Final precision: CV = {final_cv:.1f}%, all {len(measurements)} replicates within ±{relaxed_tolerance:.1f}μL")
                    print(f"      ⚠️  NOTE: Using {multiplier}x relaxed tolerance (original: ±{tolerances['variation_ul']:.0f}μL)")
                    
                    # Record the achieved tolerance for adaptive learning
                    record_achieved_tolerance(volume, multiplier)
                    
                    return True, candidate_params, measurements, times, candidate, True  # is_adaptive=True
                else:
                    failed_count = sum(1 for m in measurements if not (min_acceptable <= m <= max_acceptable))
                    print(f"      ❌ FAILED: {failed_count}/{len(measurements)} measurements outside ±{relaxed_tolerance:.1f}μL range")
            else:
                print(f"      ⚠️  ERROR: Should have {PRECISION_REPLICATES} replicates but only has {len(measurements)}")
    
    # If no candidate passes even 2x tolerance, use emergency fallback (best available)
    print(f"\n   ⚠️  EMERGENCY FALLBACK: No candidate passed even 2x tolerance")
    print(f"   📊 Using best available candidate (already completed all replicates)")
    
    # Use the best completed candidate (first in list, already sorted by precision score)
    best_completed = completed_candidates[0]
    best_candidate = best_completed['candidate']
    best_params = best_completed['params']
    best_measurements = best_completed['measurements']
    best_times = best_completed['times']
    
    final_cv = best_completed['cv_percent']
    print(f"   ⚠️  CRITICAL: Unable to meet any tolerance requirement")
    print(f"   📈 Emergency precision: CV = {final_cv:.1f}% with {len(best_measurements)} replicates")
    print(f"   🔧 Consider adjusting experimental parameters or system calibration")
    
    return True, best_params, best_measurements, best_times, best_candidate, True  # is_adaptive=True

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
                # Extract parameter values and clean them
                raw_params = {k: result[k] for k in param_keys if k in result}
                params = isolate_pipetting_params(raw_params, "fallback_suggestion")
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
        try:
            import calibration_sdl_base as base_module
            # Check if vial management is active
            if base_module._VIAL_MANAGEMENT_MODE_OVERRIDE and base_module._VIAL_MANAGEMENT_MODE_OVERRIDE.lower() != "legacy":
                print(f"[vial-mgmt] Measurement vial {current_vial} full ({vol:.1f}mL) - vial management will handle emptying")
                # Let vial management handle this - don't switch vials
                return
        except Exception as e:
            print(f"[vial-mgmt] Could not check vial management status: {e}")
        
        # Legacy behavior: switch to next measurement vial
        print(f"[legacy] Measurement vial {current_vial} full ({vol:.1f}mL) - switching to next vial")
        # Only remove pipet when switching vials if we're not retaining between experiments
        if not RETAIN_PIPET_BETWEEN_EXPERIMENTS:
            lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.return_vial_home(current_vial)
        state["measurement_vial_index"] += 1
        new_vial_name = f"measurement_vial_{state['measurement_vial_index']}"
        state["measurement_vial_name"] = new_vial_name
        lash_e.logger.info(f"[info] Switching to new measurement vial: {new_vial_name}")
        lash_e.nr_robot.move_vial_to_location(new_vial_name, "clamp", 0)

def get_liquid_source(lash_e, minimum_volume=2.0):
    """Check liquid_source_0 volume and switch to liquid_source_2 if needed"""
    try:
        current_vol = lash_e.nr_robot.get_vial_info("liquid_source_0", "vial_volume")
        if current_vol <= minimum_volume:
            lash_e.logger.info(f"[info] liquid_source_0 volume is {current_vol:.1f}mL, switching to liquid_source_2")
            return "liquid_source_2"
        else:
            return "liquid_source_0"
    except Exception as e:
        # If liquid_source_0 doesn't exist or error, default to liquid_source_2
        lash_e.logger.info(f"[info] Could not check liquid_source_0 volume ({e}), using liquid_source_2 as fallback")
        return "liquid_source_2"

def get_liquid_source_with_vial_management(lash_e, state, minimum_volume=2.0):
    """Get liquid source after applying vial management (refill/swap if needed)"""
    try:
        import calibration_sdl_base as base_module
        # Apply vial management first (refill source if low, empty measurement if full)
        if base_module._VIAL_MANAGEMENT_MODE_OVERRIDE and base_module._VIAL_MANAGEMENT_MODE_OVERRIDE.lower() != "legacy":
            base_module.manage_vials(lash_e, state)
            
            # CRITICAL FIX: After vial management, get the current source vial from config
            if base_module._VIAL_MANAGEMENT_CONFIG_OVERRIDE:
                cfg = {**base_module.VIAL_MANAGEMENT_DEFAULTS, **base_module._VIAL_MANAGEMENT_CONFIG_OVERRIDE}
            else:
                cfg = base_module.VIAL_MANAGEMENT_DEFAULTS
            
            current_source = cfg.get('source_vial', 'liquid_source_0')
            lash_e.logger.info(f"[vial-mgmt] Using current source vial: {current_source}")
            return current_source
    except Exception as e:
        lash_e.logger.info(f"[vial-mgmt] pre-pipetting management skipped: {e}")
    
    # Fallback to default source vial for legacy mode
    return "liquid_source_0"

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
        'optimization_focus': 'accuracy_only',
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

def count_actual_trials_from_raw_data(raw_measurements, volumes):
    """Count actual trial types from raw measurement data."""
    trial_counts = {}
    
    if not raw_measurements:
        # Fallback to empty counts if no raw data
        for volume in volumes:
            volume_ul = int(volume * 1000)
            trial_counts[volume] = {
                'SCREENING': 0,
                'OPTIMIZATION': 0, 
                'PRECISION': 0,
                'OVERVOLUME_ASSAY': 0,
                'total': 0
            }
        return trial_counts
    
    # Count trials by volume and type
    for volume in volumes:
        volume_trials = [m for m in raw_measurements if abs(m.get('volume', 0) - volume) < 0.0001]
        
        trial_counts[volume] = {
            'SCREENING': len([m for m in volume_trials if m.get('trial_type') == 'SCREENING']),
            'OPTIMIZATION': len([m for m in volume_trials if m.get('trial_type') == 'OPTIMIZATION']),
            'PRECISION': len([m for m in volume_trials if m.get('trial_type') == 'PRECISION']),
            'OVERVOLUME_ASSAY': len([m for m in volume_trials if m.get('trial_type') == 'OVERVOLUME_ASSAY']),
            'total': len(volume_trials)
        }
    
    return trial_counts

def generate_calibration_report(volume_report_data, volumes, completed_volumes, raw_measurements=None):
    """Generate a comprehensive calibration report with diagnostics and recommendations."""
    
    # Get actual trial counts from raw data
    actual_trial_counts = count_actual_trials_from_raw_data(raw_measurements, volumes)
    
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
                
                # Convert to simpler uL + % format for reporting
                # The actual formula accounts for existing overaspirate + buffer, but report as equivalent uL + %
                report_lines.append(f"Calibrated formula: {new_base:.1f} uL + {new_scaling:.1f}%")
                
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
                        total_recommended = max(total_recommended, 2.0)  # Ensure minimum 2uL
                        
                        report_lines.append(f"  {vol_set:.0f}uL target -> {vol_meas:.1f}uL measured (shortfall: {shortfall:.1f}uL, had {existing_over:.1f}uL) -> recommend {total_recommended:.1f}uL total")
        else:
            report_lines.append("Status: DISABLED")
            report_lines.append(f"Using static formula: {OVERASPIRATE_BASE_UL:.1f}uL + {OVERASPIRATE_SCALING_PERCENT:.1f}%")
        
        report_lines.append("")
    
    # Volume-by-volume details
    for volume in volumes:
        data = volume_report_data.get(volume, {})
        actual_counts = actual_trial_counts.get(volume, {})
        volume_ul = int(volume * 1000)
        
        report_lines.append(f"Volume_{volume_ul}uL:")
        
        # Screening trials (SOBOL/LLM)
        screening_count = actual_counts.get('SCREENING', 0)
        if screening_count > 0:
            report_lines.append(f"   {screening_count} SCREENING TRIALS")
        
        # Optimization trials
        opt_count = actual_counts.get('OPTIMIZATION', 0)
        if opt_count > 0:
            # Get accuracy failure info from tracked data (if available)
            accuracy_failures = data.get('accuracy_failures', 0)
            failure_text = ""
            if accuracy_failures > 0:
                failure_text = f" (Accuracy Failures: {accuracy_failures})"
            
            report_lines.append(f"   {opt_count} OPTIMIZATION TRIALS{failure_text}")
        
        # Precision test results
        precision_count = actual_counts.get('PRECISION', 0)
        precision_overall_passed = data.get('precision_passed', False)
        
        if precision_count > 0:
            if precision_overall_passed:
                report_lines.append(f"   {precision_count} PRECISION TRIALS (PASSED)")
            else:
                report_lines.append(f"   {precision_count} PRECISION TRIALS (FAILED)")
        
        # Overvolume assay trials
        overvolume_count = actual_counts.get('OVERVOLUME_ASSAY', 0)
        if overvolume_count > 0:
            report_lines.append(f"   {overvolume_count} OVERVOLUME TRIALS")
        
        # Calculate and show total trials for this volume
        total_volume_trials = actual_counts.get('total', 0)
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
        accuracy_failures = data.get('accuracy_failures', 0)
        
        if opt_count > 0:
            accuracy_pass_rate = ((opt_count - accuracy_failures) / opt_count) * 100
            report_lines.append(f"  {volume_ul}uL: Accuracy passing: {accuracy_pass_rate:.1f}%")
    
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
    
    # Calculate and show grand total trials from actual data
    grand_total = sum(actual_counts.get('total', 0) for actual_counts in actual_trial_counts.values())
    
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
            accuracy_failures = first_data.get('accuracy_failures', 0)
            
            if opt_count > 0:
                accuracy_pass_rate = ((opt_count - accuracy_failures) / opt_count) * 100
                
                if accuracy_pass_rate < 20:  # Less than 20% passing accuracy
                    diagnostics.append("- Accuracy restrictions appear too strict for the first volume") 
                    diagnostics.append("  Recommendation: Increase tolerance or adjust overaspirate parameters")
            
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
            diagnostics.append("  Consider adjusting tolerance ranges or experimental parameters")
            
            # Check if it's mainly precision failures at higher volumes
            precision_failures = 0
            for vol in failed_volumes:
                data = volume_report_data.get(vol, {})
                if data.get('candidate_found', False) and not data.get('precision_passed', False):
                    precision_failures += 1
            
            if precision_failures > 0:
                diagnostics.append("  Focus on: VARIATION_SCALING_FACTOR (precision tests failing)")
    
    return diagnostics


def run_single_experiment(config_overrides=None):
    """Run a single experiment with optional configuration overrides
    
    Args:
        config_overrides (dict): Dictionary of configuration values to override defaults
                                 e.g., {'liquid': 'water', 'volumes': [0.1, 0.05], 'seed': 42}
    
    Returns:
        dict: Experiment results including paths, statistics, and completion status
    """
    # CRITICAL: Reset all globals to defaults before each experiment
    # This prevents config persistence between experiments
    reset_config_to_defaults()
    
    # Start with default configuration
    config = get_default_config()
    
    # Apply any overrides
    if config_overrides:
        config.update(config_overrides)
        print(f"\nCONFIGURATION OVERRIDES APPLIED:")
        # Show only keys that differ from defaults to avoid noise
        default_cfg = get_default_config()
        for key, value in config_overrides.items():
            if key not in default_cfg or default_cfg[key] != value:
                print(f"   {key}: {value}")
    
    # Update global variables with the configuration (for backward compatibility)
    global LIQUID, SIMULATE, SEED, VOLUMES, MAX_WELLS, PRECISION_REPLICATES
    global USE_LLM_FOR_SCREENING, USE_LLM_FOR_OPTIMIZATION, BAYESIAN_MODEL_TYPE
    global AUTO_CALIBRATE_OVERVOLUME, OVERASPIRATE_BASE_UL, OVERASPIRATE_SCALING_PERCENT
    global INPUT_VIAL_STATUS_FILE
    
    LIQUID = config['liquid']
    SIMULATE = config['simulate'] 
    SEED = config['seed']
    VOLUMES = config['volumes']
    MAX_WELLS = config['max_wells']
    PRECISION_REPLICATES = config['precision_replicates']
    USE_LLM_FOR_SCREENING = config['use_llm_for_screening']
    USE_LLM_FOR_OPTIMIZATION = config['use_llm_for_optimization']
    BAYESIAN_MODEL_TYPE = config['bayesian_model_type']
    AUTO_CALIBRATE_OVERVOLUME = config['auto_calibrate_overvolume']
    OVERASPIRATE_BASE_UL = config['overaspirate_base_ul']
    OVERASPIRATE_SCALING_PERCENT = config['overaspirate_scaling_percent']
    INPUT_VIAL_STATUS_FILE = config['input_vial_status_file']
    
    # Handle new simplified vial_mode configuration
    if config_overrides and 'vial_mode' in config_overrides:
        vial_config = get_vial_config(config_overrides['vial_mode'])
        INPUT_VIAL_STATUS_FILE = vial_config['vial_file']
        print(f"🔧 Using vial mode: {config_overrides['vial_mode']} -> {INPUT_VIAL_STATUS_FILE}")
        # CRITICAL: Also set the vial management mode from the config
        config_overrides['vial_management_mode'] = vial_config['mode']
    
    print(f"\n📋 EXPERIMENT CONFIGURATION:")
    print(f"   Liquid: {LIQUID}")
    print(f"   Simulate: {SIMULATE}")
    print(f"   Volumes: {[f'{v*1000:.0f}uL' for v in VOLUMES]}")
    print(f"   Seed: {SEED}")
    print(f"   Max Wells: {MAX_WELLS}")
    print(f"   Precision Replicates: {PRECISION_REPLICATES}")
    print(f"   Vial File: {INPUT_VIAL_STATUS_FILE}")
    
    # Optional per-experiment vial management overrides
    # Allow config keys: vial_management_mode, vial_management, min_source_ml, refill_target_ml, max_measurement_ml, source_vial, measurement_vial, waste_vial, reservoir_index
    try:
        import calibration_sdl_base as base_module
        vm_mode = None
        vm_cfg = None
        if config_overrides:
            # Auto-infer maintain mode if a vial status file was provided and mode not explicitly set
            if 'vial_management_mode' in config_overrides:
                vm_mode = config_overrides['vial_management_mode']
            elif 'input_vial_status_file' in config_overrides:
                vm_mode = 'maintain'
            vm_keys = ['source_vial','measurement_vial','waste_vial','reservoir_index','min_source_ml','refill_target_ml','max_measurement_ml']
            present = {k: config_overrides[k] for k in vm_keys if k in config_overrides}
            if present:
                if 'reservoir_index' in present:
                    present['reservoir_index'] = int(present['reservoir_index'])
                for f in ['min_source_ml','refill_target_ml','max_measurement_ml']:
                    if f in present:
                        present[f] = float(present[f])
                vm_cfg = present
        base_module.set_vial_management(vm_mode, vm_cfg)
        if vm_mode or vm_cfg:
            print(f"[vial-mgmt] active: mode={vm_mode or 'inherit'} cfg_keys={(list(vm_cfg.keys()) if vm_cfg else [])}")
        
        # Verify the vial management mode was set correctly
        print(f"[vial-mgmt] Verification: global mode = {base_module._VIAL_MANAGEMENT_MODE_OVERRIDE}")
        print(f"[vial-mgmt] Verification: global config = {base_module._VIAL_MANAGEMENT_CONFIG_OVERRIDE}")
        
    except Exception as e:
        print(f"[vial-mgmt] override setup failed (non-fatal): {e}")
        import traceback
        traceback.print_exc()

    # Run the actual experiment logic
    return run_experiment_logic()

def main(config_overrides=None):
    """Main experiment function - can be called with configuration overrides"""
    # Call run_single_experiment which handles the config setup and calls the actual experiment logic
    return run_single_experiment(config_overrides)

def run_experiment_logic():
    """Core experiment logic separated from configuration handling"""
    lash_e, DENSITY_LIQUID, NEW_PIPET_EACH_TIME_SET, state = initialize_experiment()
    
    # Show vial management status at start of experiment
    try:
        import calibration_sdl_base as base_module
        vial_mode = base_module._VIAL_MANAGEMENT_MODE_OVERRIDE
        vial_config = base_module._VIAL_MANAGEMENT_CONFIG_OVERRIDE
        print(f"\n🔧 VIAL MANAGEMENT STATUS:")
        print(f"   Mode: {vial_mode or 'legacy (default)'}")
        if vial_config:
            print(f"   Config: {vial_config}")
        else:
            print(f"   Config: Using defaults")
        lash_e.logger.info(f"Vial management mode: {vial_mode or 'legacy'}")
    except Exception as e:
        print(f"Could not check vial management status: {e}")
    
    # Log experiment start (with error handling)
    try:
        lash_e.logger.info(f"Starting calibration experiment - liquid: {LIQUID}, volumes: {VOLUMES}")
        lash_e.logger.info(f"Configuration: seed={SEED}, simulate={SIMULATE}")
    except Exception as e:
        print(f"Error in experiment logging: {e}")
        
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
    print(f"   🎯 Calibration Focus: ACCURACY ONLY (no time constraints)")
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
    
    # Load historical calibration data for dynamic time cutoffs
    historical_data = None
    try:
        # Look for historical data in common locations
        historical_paths = [
            "pipetting_data",
            "output", 
            "analysis"
        ]
        
        for path in historical_paths:
            if os.path.exists(path):
                # Look for CSV files with calibration data
                for filename in os.listdir(path):
                    if filename.endswith('.csv') and 'replicate' in filename.lower():
                        file_path = os.path.join(path, filename)
                        try:
                            df = pd.read_csv(file_path)
                            # Check if it has the required columns for dynamic cutoffs
                            time_col = 'time' if 'time' in df.columns else ('replicate_time' if 'replicate_time' in df.columns else None)
                            if time_col is not None and 'volume' in df.columns and len(df) > 10:
                                historical_data = df
                                print(f"📊 Loaded historical data: {filename} ({len(df)} trials)")
                                lash_e.logger.info(f"Dynamic cutoffs enabled: loaded {len(df)} historical trials from {filename}")
                                break
                        except Exception as e:
                            continue
                if historical_data is not None:
                    break
        
        if historical_data is None:
            print("📊 No historical calibration data found - using static time cutoffs")
            lash_e.logger.info("Dynamic cutoffs disabled: no historical data available")
            
    except Exception as e:
        print(f"📊 Error loading historical data: {e} - using static time cutoffs")
        lash_e.logger.warning(f"Failed to load historical data: {e}")
        historical_data = None
    
    for volume_index, volume in enumerate(VOLUMES):
        print(f"\n{'='*60}")
        print(f"🧪 VOLUME: {volume*1000:.0f}uL")
        print(f"{'='*60}")
        
        # Initialize report data for this volume
        volume_report_data[volume] = {
            'sobol_trials': 0,
            'optimization_trials': 0,
            'accuracy_failures': 0,
            'precision_trials_attempted': 0,
            'precision_trials_failed': 0,
            'precision_trials_passed': 0,
            'precision_passed': False,
            'completed': False,
            'candidate_found': False
        }
        
        # Calculate volume-dependent tolerances (with historical data for dynamic cutoffs)
        is_first_volume = (volume_index == 0)  # Track volume index for logging purposes
        tolerances = get_volume_dependent_tolerances(volume, is_first_volume=is_first_volume, historical_data=historical_data)
        
        # Build criteria - ACCURACY ONLY for calibration
        criteria = {
            'max_deviation_ul': tolerances['deviation_ul'],
        }
        # NO TIME CONSTRAINTS - calibration is about accuracy, not speed
        lash_e.logger.info(f"VOLUME {volume*1000:.0f}uL: Accuracy-only calibration (no time constraints)")
        
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
        max_overaspirate_for_volume = get_max_overaspirate_ul(volume)
        ax_client = get_recommender().create_model(SEED, INITIAL_SUGGESTIONS, bayesian_batch_size=BATCH_SIZE, 
                                           volume=volume, tip_volume=tip_volume, model_type=bayesian_model_type, 
                                           optimize_params=optimize_params, fixed_params=fixed_params, simulate=SIMULATE,
                                           max_overaspirate_ul=max_overaspirate_for_volume)
        
        # Step 1: Determine starting candidate
        if len(completed_volumes) > 0:
            # Use the FIRST successful volume's parameters as starting point (most conservative)
            first_volume, first_params = completed_volumes[0]
            print(f"🔄 TESTING CONSERVATIVE BASELINE: Using parameters from FIRST successful volume {first_volume*1000:.0f}uL")
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
                screening_suggestions = get_llm_suggestions(ax_client, INITIAL_SUGGESTIONS, all_results, 
                                                           volume=volume, liquid=LIQUID)
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
                liquid_source = get_liquid_source_with_vial_management(lash_e, state)
                
                # DEFENSIVE: Clean parameters and log for debugging
                clean_params = isolate_pipetting_params(params, f"screening_trial_{i+1}")
                
                result = pipet_and_measure(lash_e, liquid_source, state["measurement_vial_name"], volume, clean_params, 
                                         expected_mass, expected_time, REPLICATES, SIMULATE, autosave_raw_path, 
                                         raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET, "SCREENING")
                
                # Get the most recent measurement for display
                recent_mass = raw_measurements[-1]['mass'] if raw_measurements else expected_mass
                recent_volume = raw_measurements[-1]['calculated_volume'] if raw_measurements else volume
                
                # Check if this trial meets criteria - convert % deviation to absolute uL
                deviation_pct = result.get('deviation', float('inf'))
                absolute_deviation_ul = (deviation_pct / 100) * (volume * 1000)  # Convert to uL
                
                # ACCURACY ONLY - no time constraints for calibration
                meets_criteria = (absolute_deviation_ul <= criteria['max_deviation_ul'])
                    
                status = "✅ CANDIDATE" if meets_criteria else "❌ reject"
                print(f"   Trial {i+1}/{INITIAL_SUGGESTIONS}: {recent_mass:.4f}g → {recent_volume*1000:.1f}uL, {result.get('deviation', 'N/A'):.1f}% dev, {result.get('time', 'N/A'):.0f}s - {status}")
                
                # Add result to optimizer for future suggestions
                get_recommender().add_result(ax_client, trial_index, result)
                result.update(clean_params)  # Use clean_params to prevent contamination
                result.update({"volume": volume, "trial_index": trial_index, "strategy": screening_method, "trial_type": "SCREENING", "liquid": LIQUID, "time_reported": datetime.now().isoformat()})
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
            # NOTE: This modification is TEMPORARY and only affects the current experiment.
            # The reset_config_to_defaults() function ensures these values are reset before each new experiment.
            if new_base_ul is not None and new_scaling_percent is not None:
                global OVERASPIRATE_BASE_UL, OVERASPIRATE_SCALING_PERCENT
                old_base = OVERASPIRATE_BASE_UL
                old_scaling = OVERASPIRATE_SCALING_PERCENT
                
                OVERASPIRATE_BASE_UL = new_base_ul
                OVERASPIRATE_SCALING_PERCENT = new_scaling_percent
                
                print(f"   ✅ OVERVOLUME PARAMETERS UPDATED:")
                print(f"      Old: {old_base:.1f}uL + {old_scaling:.1f}%")
                print(f"      New: {OVERASPIRATE_BASE_UL:.1f}uL + {OVERASPIRATE_SCALING_PERCENT:.1f}%")
                
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
                print(f"      Using: {OVERASPIRATE_BASE_UL:.1f}uL + {OVERASPIRATE_SCALING_PERCENT:.1f}%")
                
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
                print(f"🔍 OPTIMIZATION: Finding acceptable parameters (target: ≤{criteria['max_deviation_ul']:.0f}uL deviation, accuracy only)")
                
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
                            suggestions = get_llm_suggestions(ax_client, BATCH_SIZE, all_results, 
                                                             volume=volume, liquid=LIQUID)
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
                        liquid_source = get_liquid_source_with_vial_management(lash_e, state)
                        
                        # DEFENSIVE: Clean parameters and log for debugging
                        clean_params = isolate_pipetting_params(params, f"optimization_trial_{trial_count}")
                        
                        result = pipet_and_measure(lash_e, liquid_source, state["measurement_vial_name"], volume, clean_params, 
                                                 expected_mass, expected_time, REPLICATES, SIMULATE, autosave_raw_path, 
                                                 raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET, "OPTIMIZATION")
                        
                        # Get the most recent measurement for display
                        recent_mass = raw_measurements[-1]['mass'] if raw_measurements else expected_mass
                        recent_volume = raw_measurements[-1]['calculated_volume'] if raw_measurements else volume
                        
                        # Check if this trial meets criteria - convert % deviation to absolute uL
                        deviation_pct = result.get('deviation', float('inf'))
                        absolute_deviation_ul = (deviation_pct / 100) * (volume * 1000)  # Convert to uL
                        
                        # ACCURACY ONLY - no time constraints for calibration
                        meets_criteria = (absolute_deviation_ul <= criteria['max_deviation_ul'])
                            
                        if absolute_deviation_ul > criteria['max_deviation_ul']:
                            volume_report_data[volume]['accuracy_failures'] += 1
                        
                        status = "✅ CANDIDATE" if meets_criteria else "❌ reject"
                        print(f"   Optimization trial: {recent_mass:.4f}g → {recent_volume*1000:.1f}uL, {result.get('deviation', 'N/A'):.1f}% dev, {result.get('time', 'N/A'):.0f}s - {status}")
                        
                        # Add result to optimizer - only use time scoring for first volume
                        # Track timing data for analysis purposes
                        get_recommender().add_result(ax_client, trial_index, result)
                        result.update(clean_params)  # Use clean_params to prevent contamination
                        optimization_strategy = "LLM" if use_llm_for_optimization else "Bayesian"
                        result.update({"volume": volume, "trial_index": trial_index, "strategy": optimization_strategy, "trial_type": "OPTIMIZATION", "liquid": LIQUID, "time_reported": datetime.now().isoformat()})
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
                    print(f"❌ Could not find acceptable parameters for {volume*1000:.0f}uL within well limit")
                    break  # Move to next volume
            
            # Step 3: Precision testing - approach differs based on whether we have successful volumes
            if trial_count + PRECISION_REPLICATES > MAX_WELLS:
                print(f"⚠️ Not enough wells remaining for precision test ({MAX_WELLS - trial_count} left)")
                break
            
            if completed_volumes and USE_SELECTIVE_OPTIMIZATION:
                # Volumes 2+: Use the EXACT parameters from the candidate we just found!
                # Don't reconstruct from contaminated all_results - use the fresh candidate_params directly
                print(f"🎯 SELECTIVE OPTIMIZATION: Testing newly found candidate parameters")
                
                # CRITICAL: Clean the candidate_params we just found to prevent contamination
                clean_candidate_params = isolate_pipetting_params(candidate_params, "selective_optimization_fresh_candidate")
                
                # Use simple precision test (not cascading)
                passed, precision_measurements, precision_times = run_precision_test(
                    lash_e, state, clean_candidate_params, volume, expected_mass, expected_time, 
                    autosave_raw_path, raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET, 
                    tolerances['variation_ul']
                )
                selected_candidate = None  # Not needed for fresh candidate tracking
                is_adaptive = False
                
            else:
                # Volume 1: Use cascading precision tests to find winner from multiple candidates
                passed, candidate_params, precision_measurements, precision_times, selected_candidate, is_adaptive = run_cascading_precision_tests(
                    all_results, criteria, lash_e, state, volume, expected_mass, expected_time, 
                    autosave_raw_path, raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET, 
                    tolerances, blacklisted_params, volume_index=volume_index
                )
            
            if precision_measurements:
                trial_count += len(precision_measurements)
                candidate_trial_number = selected_candidate.get('trial_number', trial_count) if selected_candidate else trial_count
            
            # Track precision test results
            # ALL precision measurements are precision trials, not optimization trials
            if len(precision_measurements) > 0:
                precision_count = len(precision_measurements)
                volume_report_data[volume]['precision_trials_attempted'] += precision_count
                if passed:
                    volume_report_data[volume]['precision_trials_passed'] += precision_count
                    volume_report_data[volume]['precision_passed'] = True
                else:
                    volume_report_data[volume]['precision_trials_failed'] += precision_count
            
            if passed is None:
                # Insufficient candidates - continue optimization without blacklisting
                print(f"\n⚠️  INSUFFICIENT CANDIDATES - continuing optimization to gather more data")
                candidate_params = None  # Force new optimization (but don't blacklist anything)
            elif passed:
                # SUCCESS! Store clean parameters that worked in the precision test
                # Make sure we isolate them properly to prevent future contamination
                clean_working_params = isolate_pipetting_params(candidate_params, f"volume_{volume}_winner")
                completed_volumes.append((volume, clean_working_params))
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
                    'adaptive_tolerance': is_adaptive,           # Flag if tolerance was relaxed
                    **candidate_params  # Add all parameter values
                }
                
                optimal_conditions.append(optimal_condition)
                save_optimal_conditions(optimal_conditions, optimal_conditions_path)
                
                # Adaptive time scaling removed - time optimization only for first volume
                
                if is_adaptive:
                    print(f"\n🎉 VOLUME {volume*1000:.0f}uL: ⚠️  COMPLETED WITH ADAPTED TOLERANCE")
                    print(f"   ⚠️  WARNING: Could not meet original precision requirement (±{tolerances['variation_ul']:.0f}μL)")
                    print(f"   📊 Used most precise available candidate (CV: {variability_percent:.1f}%)")
                    print(f"   🔧 Consider adjusting system or tolerance settings for this volume range")
                else:
                    print(f"\n🎉 VOLUME {volume*1000:.0f}uL: ✅ COMPLETED")
                    print(f"   Precision test PASSED - all {len(precision_measurements)} replicates within ±{tolerances['variation_ul']:.0f}uL range")
                volume_completed = True
                
            else:
                # ALL CANDIDATES FAILED! Need more optimization trials
                print(f"\n❌ ALL PRECISION TESTS FAILED - need more optimization trials")
                print(f"   📈 Blacklisted candidates will be avoided in next optimization round")
                candidate_params = None  # Force new optimization
        
        if not volume_completed:
            print(f"\n⚠️  VOLUME {volume*1000:.0f}uL: Could not complete within well limit ({MAX_WELLS - trial_count} wells remaining)")
        
        print(f"Wells used: {trial_count}/{MAX_WELLS}")
        
        if trial_count >= MAX_WELLS:
            print(f"Reached maximum wells ({MAX_WELLS}), stopping experiment.")
            break
    # Wrap up
    if not RETAIN_PIPET_BETWEEN_EXPERIMENTS:
        try:
            lash_e.nr_robot.remove_pipet()
        except Exception as e:
            print(f"Warning: could not remove pipet at experiment wrap-up: {e}")
    lash_e.nr_robot.return_vial_home(state["measurement_vial_name"])
    try:
        lash_e.nr_robot.move_home()
    except Exception:
        pass
    
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
            avg_vol = optimal.get('average_obtained_volume_mL', vol) * 1000  # Convert to uL
            replicates = optimal.get('precision_replicates', PRECISION_REPLICATES)
            
            # Format values with proper None handling
            deviation_str = f"{deviation:.1f}%" if deviation is not None else "N/A"
            time_str = f"{time_s:.0f}s" if time_s is not None else "N/A"
            variability_str = f"{variability:.1f}%" if variability is not None else "N/A"
            
            # Add tolerance information
            tolerance_multiplier = ACHIEVED_TOLERANCE_MULTIPLIERS.get(vol, 1.0)
            tolerance_str = "original" if tolerance_multiplier == 1.0 else f"{tolerance_multiplier}x"
            
            print(f"  {vol*1000:.0f}uL → {avg_vol:.1f}uL (deviation: {deviation_str}, variability: {variability_str}, time: {time_str}, tolerance: {tolerance_str}, n={replicates})")
        
        # Show adaptive tolerance learning summary
        if ACHIEVED_TOLERANCE_MULTIPLIERS and any(m > 1.0 for m in ACHIEVED_TOLERANCE_MULTIPLIERS.values()):
            print(f"\n🎯 ADAPTIVE TOLERANCE LEARNING:")
            print(f"   📚 Tolerance multipliers achieved: {ACHIEVED_TOLERANCE_MULTIPLIERS}")
            relaxed_volumes = [vol for vol, mult in ACHIEVED_TOLERANCE_MULTIPLIERS.items() if mult > 1.0]
            if relaxed_volumes:
                print(f"   ⚠️  Volumes requiring relaxed tolerance: {[f'{vol*1000:.0f}μL' for vol in relaxed_volumes]}")
                print(f"   💡 Future experiments with similar volumes should expect similar tolerance requirements")
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
            print(f"\n🎯 Volume {volume*1000:.0f}uL:")
            
            # Count trials for this volume
            trials_count = len(vol_results)
            best = vol_results.loc[vol_results['deviation'].idxmin()]
            
            print(f"   📈 OPTIMIZATION PHASE:")
            print(f"      • Total optimization trials: {trials_count}")
            print(f"      • Best result: {best['deviation']:.1f}% deviation, {best['time']:.1f}s")
            
            if volume_completed:
                print(f"   ✅ PRECISION TEST PHASE:")
                print(f"      • Precision test: PASSED (all {PRECISION_REPLICATES} measurements within ±{tolerances['variation_ul']:.0f}uL)")
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
            completed_list = [f"{int(v*1000)}uL" for v, _ in completed_volumes]
            remaining_vols = [v for v in VOLUMES if v not in [cv for cv, _ in completed_volumes]]
            remaining_list = [f"{int(v*1000)}uL" for v in remaining_vols]
            completed_str = ", ".join(completed_list) if completed_list else "None"
            remaining_str = ", ".join(remaining_list) if remaining_list else "None"

            # Calculate per-volume metrics and overall statistics using actual trial counts
            actual_trial_counts = count_actual_trials_from_raw_data(raw_measurements, VOLUMES)
            volume_summaries = []
            total_trials = 0
            
            for i, (volume, params) in enumerate(completed_volumes):
                # Get actual trial count for this volume from raw data
                actual_counts = actual_trial_counts.get(volume, {})
                vol_trials = actual_counts.get('total', 0)
                total_trials += vol_trials
                
                # Get accuracy metrics from optimal_conditions
                if i < len(optimal_conditions):
                    oc = optimal_conditions[i]
                    deviation = oc.get('deviation_percent', 0)
                    variability = oc.get('variability_percent', 0)
                    volume_summaries.append(f"{int(volume*1000)}uL({vol_trials}t,{deviation:.1f}%,{variability:.1f}%)")
                else:
                    volume_summaries.append(f"{int(volume*1000)}uL({vol_trials}t,N/A,N/A)")
            
            # Calculate overall averages
            if optimal_conditions:
                avg_deviation = sum(oc.get('deviation_percent', 0) for oc in optimal_conditions) / len(optimal_conditions)
                avg_time = sum(oc.get('time_seconds', 0) for oc in optimal_conditions if oc.get('time_seconds')) / len([oc for oc in optimal_conditions if oc.get('time_seconds')])
            else:
                avg_deviation = 0
                avg_time = 0

            volumes_line = " ".join(volume_summaries) if volume_summaries else "None"
            time_str = f"{avg_time:.0f}s" if avg_time else "N/A"

            slack_msg = (
                f"🎯 Modular calibration with {LIQUID} COMPLETE\n"
                f"✅ {volumes_line}\n"
                f"🎯 Total: {total_trials} trials, {avg_deviation:.1f}% avg accuracy\n"
                f"⏱️ Avg time: {time_str}"
            )
            slack_agent.send_slack_message(slack_msg)
        except Exception as e:
            print(f"Warning: Failed to send detailed Slack summary: {e}")

    # Generate and save calibration report
    try:
        report_content = generate_calibration_report(volume_report_data, VOLUMES, completed_volumes, raw_measurements)
        
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
        report_path = None
    
    print(f"{'='*60}")
    
    # Log experiment completion
    lash_e.logger.info(f"Completed experiment: {len(completed_volumes)}/{len(VOLUMES)} volumes successful")
    
    # Return experiment results
    return {
        'success': len(completed_volumes) == len(VOLUMES),
        'completed_volumes': len(completed_volumes),
        'total_volumes': len(VOLUMES),
        'volumes': VOLUMES,
        'liquid': LIQUID,
        'simulate': SIMULATE,
        'seed': SEED,
        'autosave_dir': autosave_dir,
        'report_path': report_path,
        'volume_report_data': volume_report_data,
        'completed_volume_list': completed_volumes,
        'optimal_conditions': optimal_conditions
    }

def run_multiple_experiments(experiment_configs):
    """Run multiple experiments with different configurations
    
    Args:
        experiment_configs (list): List of dictionaries, each containing config overrides for one experiment
                                  e.g., [
                                      {'liquid': 'water', 'seed': 1},
                                      {'liquid': 'glycerol', 'seed': 2}, 
                                      {'volumes': [0.1, 0.05], 'precision_replicates': 6}
                                  ]
    
    Returns:
        list: List of experiment results
    """
    results = []
    
    print(f"\n{'='*80}")
    print(f"🚀 MULTI-EXPERIMENT BATCH: {len(experiment_configs)} experiments planned")
    print(f"{'='*80}")
    
    for i, config_overrides in enumerate(experiment_configs, 1):
        print(f"\n{'🧪'*3} EXPERIMENT {i}/{len(experiment_configs)} {'🧪'*3}")
        # Set global experiment index (0-based)
        global EXPERIMENT_INDEX
        EXPERIMENT_INDEX = i - 1
        print(f"[info] Set EXPERIMENT_INDEX={EXPERIMENT_INDEX}")
        
        # Show what's being changed for this experiment
        if config_overrides:
            print(f"📝 Configuration changes for this experiment:")
            for key, value in config_overrides.items():
                print(f"   • {key}: {value}")
        else:
            print(f"📝 Using default configuration (no overrides)")
        
        try:
            result = run_single_experiment(config_overrides)
            result['experiment_number'] = i
            result['config_overrides'] = config_overrides
            results.append(result)
            # Optional removal (disabled by default). Use cached controller.
            try:
                if not RETAIN_PIPET_BETWEEN_EXPERIMENTS and _CACHED_LASH_E is not None:
                    _CACHED_LASH_E.nr_robot.remove_pipette()
            except Exception:
                pass
            
            print(f"✅ Experiment {i} completed successfully")
            
        except Exception as e:
            error_result = {
                'experiment_number': i,
                'config_overrides': config_overrides,
                'error': str(e),
                'success': False
            }
            results.append(error_result)
            print(f"❌ Experiment {i} failed: {e}")
            
            # For automated operation, continue with remaining experiments
            if i < len(experiment_configs):
                print(f"\n⚠️  Experiment {i} failed. Automatically continuing with remaining experiments...")
                # Note: In interactive mode, you could add a flag to enable user prompts
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 MULTI-EXPERIMENT BATCH SUMMARY")
    print(f"{'='*80}")
    successful = [r for r in results if r.get('success', True) != False]
    failed = [r for r in results if r.get('success', True) == False]
    
    print(f"✅ Successful experiments: {len(successful)}/{len(experiment_configs)}")
    print(f"❌ Failed experiments: {len(failed)}")
    
    if successful:
        print(f"\n🎉 Successful experiments:")
        for result in successful:
            overrides = result.get('config_overrides', {})
            summary = ", ".join([f"{k}={v}" for k, v in overrides.items()]) if overrides else "default config"
            print(f"   • Experiment {result['experiment_number']}: {summary}")
    
    if failed:
        print(f"\n💥 Failed experiments:")
        for result in failed:
            overrides = result.get('config_overrides', {})
            summary = ", ".join([f"{k}={v}" for k, v in overrides.items()]) if overrides else "default config"
            print(f"   • Experiment {result['experiment_number']}: {summary} - Error: {result.get('error', 'Unknown')}")
    
    return results

# Example usage functions
def example_volume_study():
    """Example: Test different volume sets"""
    experiments = [
        {'volumes': [0.1, 0.05, 0.01]},
        {'volumes': [0.2, 0.1, 0.05]},
        {'volumes': [0.05, 0.025, 0.01]}
    ]
    return run_multiple_experiments(experiments)

def run_glycerol():
    custom_experiments = [
        {'liquid': 'glycerol', 'volumes': [0.1, 0.05, 0.01], 'base_time_seconds': 60, 'simulate': False}
    ]
    run_multiple_experiments(custom_experiments)

def get_default_config():
    """Get default experiment configuration using IMMUTABLE default constants.
    
    This function returns the TRUE defaults, not the current runtime values.
    This ensures config consistency and prevents config persistence between experiments.
    """
    return {
        'liquid': DEFAULT_LIQUID,  # str: 'water', 'glycerol', 'ethanol'
        'simulate': DEFAULT_SIMULATE,  # bool: True/False
        'seed': DEFAULT_SEED,  # int: random seed for reproducibility
        'initial_suggestions': DEFAULT_INITIAL_SUGGESTIONS,  # int: number of SOBOL/LLM initial trials
        'batch_size': DEFAULT_BATCH_SIZE,  # int: batch size for optimization loops
        'replicates': DEFAULT_REPLICATES,  # int: replicates per optimization trial
        'precision_replicates': DEFAULT_PRECISION_REPLICATES,  # int: replicates for final precision test
        'volumes': DEFAULT_VOLUMES.copy(),  # list[float]: volumes in mL, e.g., [0.05, 0.025, 0.1] - copy to prevent mutation
        'max_wells': DEFAULT_MAX_WELLS,  # int: maximum wellplate wells to use
        'input_vial_status_file': DEFAULT_INPUT_VIAL_STATUS_FILE,  # str: path to vial CSV file
        'use_llm_for_screening': DEFAULT_USE_LLM_FOR_SCREENING,  # bool: LLM vs SOBOL for initial exploration
        'use_llm_for_optimization': DEFAULT_USE_LLM_FOR_OPTIMIZATION,  # bool: LLM vs Bayesian for optimization
        'bayesian_model_type': DEFAULT_BAYESIAN_MODEL_TYPE,  # str: 'qEI', 'qLogEI', 'qNEHVI'
        'volume_tolerance_ranges': VOLUME_TOLERANCE_RANGES,  # list[dict]: tolerance specs by volume
        'overaspirate_base_ul': DEFAULT_OVERASPIRATE_BASE_UL,  # float: base overaspirate volume in uL
        'overaspirate_scaling_percent': DEFAULT_OVERASPIRATE_SCALING_PERCENT,  # float: % scaling per volume
        'auto_calibrate_overvolume': DEFAULT_AUTO_CALIBRATE_OVERVOLUME,  # bool: enable auto overvolume calibration
        'overvolume_calibration_buffer_ul': DEFAULT_OVERVOLUME_CALIBRATION_BUFFER_UL,  # float: safety buffer in uL
        'overvolume_max_base_ul': DEFAULT_OVERVOLUME_MAX_BASE_UL,  # float: max allowed base overvolume uL
        'overvolume_max_percent': DEFAULT_OVERVOLUME_MAX_PERCENT,  # float: max allowed scaling %
        'use_selective_optimization': USE_SELECTIVE_OPTIMIZATION,  # bool: optimize subset of params after first volume
        'use_historical_data_for_optimization': USE_HISTORICAL_DATA_FOR_OPTIMIZATION,  # bool: load previous volume data
        'volume_dependent_params': VOLUME_DEPENDENT_PARAMS,  # list[str]: params to optimize per volume
        'all_params': ALL_PARAMS,  # list[str]: all parameter names
        'base_autosave_dir': BASE_AUTOSAVE_DIR,  # str: directory for saving results
    }

if __name__ == "__main__":
    # SIMPLE PER-EXPERIMENT OVERRIDES (specify only what changes)
    # 
    # VIAL MANAGEMENT MODES (be explicit!):
    # - 'vial_management_mode': 'legacy' = no vial management (default if not specified)
    # - 'vial_management_mode': 'maintain' = refill source, empty measurement when thresholds hit
    # - 'vial_management_mode': 'swap' = swap source<->measurement when volumes cross
    # 
    # Vial Management Modes:
    # - 'legacy': Single vial for all volumes (uses calibration_vials_short.csv)
    # - 'maintain': Keep vials, use reservoir/waste for volume changes (uses calibration_vials_overnight.csv)
    # - 'swap': New vials for each volume, no reservoir/waste (uses calibration_vials_overnight.csv)
    #
    # NEW: Just specify vial_mode - the vial file is automatically selected!
    #

    # Current experiment - uncomment to run single experiment
    EXPERIMENTS = [{'liquid': 'water', 'volumes': [0.1, 0.05, 0.025], 'simulate': True}]

    # OVERNIGHT EXPERIMENT SUITE - Volume Order & Multi-Factor Studiesok 
    # EXPERIMENTS = [
    #     # ============================================================================
    #     # STUDY 1: VOLUME ORDER EFFECTS (Small Volumes)
    #     # Compare [0.05, 0.025, 0.1] vs [0.1, 0.05, 0.025] - 3 replicates each
    #     # ============================================================================
        
    #     # Volume Order A: [0.05, 0.025, 0.1] - Small to Large
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1], 'vial_mode': 'swap', 'seed': 1, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1], 'vial_mode': 'swap', 'seed': 2, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1], 'vial_mode': 'swap', 'seed': 3, 'simulate': False},
        
    #     # Volume Order B: [0.1, 0.05, 0.025] - Large to Small
    #     {'liquid': 'water', 'volumes': [0.1, 0.05, 0.025], 'vial_mode': 'swap', 'seed': 1, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.1, 0.05, 0.025], 'vial_mode': 'swap', 'seed': 2, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.1, 0.05, 0.025], 'vial_mode': 'swap', 'seed': 3, 'simulate': False},
        
    #     # ============================================================================
    #     # STUDY 2: VOLUME ORDER EFFECTS (Wider Range)
    #     # Compare [0.5, 0.1, 0.05, 0.025] vs [0.05, 0.025, 0.1, 0.5] - 3 replicates each
    #     # ============================================================================
        
    #     # Volume Order C: [0.5, 0.1, 0.05, 0.025] - Large to Small
    #     {'liquid': 'water', 'volumes': [0.5, 0.1, 0.05, 0.025], 'vial_mode': 'swap', 'seed': 1, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.5, 0.1, 0.05, 0.025], 'vial_mode': 'swap', 'seed': 2, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.5, 0.1, 0.05, 0.025], 'vial_mode': 'swap', 'seed': 3, 'simulate': False},
        
    #     # Volume Order D: [0.05, 0.025, 0.1, 0.5] - Small to Large
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.5], 'vial_mode': 'swap', 'seed': 1, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.5], 'vial_mode': 'swap', 'seed': 2, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.5], 'vial_mode': 'swap', 'seed': 3, 'simulate': False},
        
    #     # ============================================================================
    #     # STUDY 3: MULTI-FACTOR EXPERIMENT
    #     # Volume Set: [0.05, 0.025, 0.1, 0.3, 0.8, 0.01]
    #     # Factors: 3 Model Types x 2 Screening Types x 3 Seeds = 18 experiments
    #     # ============================================================================
        
    #     # Factor 1: qEI (Default Bayesian Model)
    #     # SOBOL Screening
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 1, 'bayesian_model_type': 'qEI', 'use_llm_for_screening': False, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 2, 'bayesian_model_type': 'qEI', 'use_llm_for_screening': False, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 3, 'bayesian_model_type': 'qEI', 'use_llm_for_screening': False, 'simulate': False},
        
    #     # LLM Screening  
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 1, 'bayesian_model_type': 'qEI', 'use_llm_for_screening': True, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 2, 'bayesian_model_type': 'qEI', 'use_llm_for_screening': True, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 3, 'bayesian_model_type': 'qEI', 'use_llm_for_screening': True, 'simulate': False},
        
    #     # Factor 2: qLogEI (Log Expected Improvement)
    #     # SOBOL Screening
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 1, 'bayesian_model_type': 'qLogEI', 'use_llm_for_screening': False, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 2, 'bayesian_model_type': 'qLogEI', 'use_llm_for_screening': False, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 3, 'bayesian_model_type': 'qLogEI', 'use_llm_for_screening': False, 'simulate': False},
        
    #     # LLM Screening
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 1, 'bayesian_model_type': 'qLogEI', 'use_llm_for_screening': True, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 2, 'bayesian_model_type': 'qLogEI', 'use_llm_for_screening': True, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 3, 'bayesian_model_type': 'qLogEI', 'use_llm_for_screening': True, 'simulate': False},
        
    #     # Factor 3: qNEHVI (Noisy Expected Hypervolume Improvement)
    #     # SOBOL Screening
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 1, 'bayesian_model_type': 'qNEHVI', 'use_llm_for_screening': False, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 2, 'bayesian_model_type': 'qNEHVI', 'use_llm_for_screening': False, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 3, 'bayesian_model_type': 'qNEHVI', 'use_llm_for_screening': False, 'simulate': False},
        
    #     # LLM Screening
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 1, 'bayesian_model_type': 'qNEHVI', 'use_llm_for_screening': True, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 2, 'bayesian_model_type': 'qNEHVI', 'use_llm_for_screening': True, 'simulate': False},
    #     {'liquid': 'water', 'volumes': [0.05, 0.025, 0.1, 0.3, 0.8, 0.01], 'vial_mode': 'swap', 'seed': 3, 'bayesian_model_type': 'qNEHVI', 'use_llm_for_screening': True, 'simulate': False},
    # ]
    

    print("\nConfigured experiments:")
    for i, cfg in enumerate(EXPERIMENTS, 1):
        print(f"  {i}. {cfg}")
    run_multiple_experiments(EXPERIMENTS)

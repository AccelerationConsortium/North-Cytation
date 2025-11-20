# pipetting_optimizer_3objectives.py
"""
Multi-objective optimization for pipetting calibration with three objectives:
1. Accuracy (deviation) - minimize
2. Precision (variability) - minimize  
3. Time - minimize

Used for first volume optimization where we need to balance all three metrics.
Subsequent volumes can use the 2-objective version (v3) for simpler optimization.
"""

import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.core.observation import ObservationFeatures
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement

obj1_name = "deviation"
obj2_name = "variability"  # Precision metric (CV or relative range)
obj3_name = "time"         # Raw time in seconds

# Default parameter bounds for all parameters
DEFAULT_PARAMETER_BOUNDS = {
    "aspirate_speed": {"type": "range", "bounds": [8, 20]},
    "dispense_speed": {"type": "range", "bounds": [8, 20]},
    "aspirate_wait_time": {"type": "range", "bounds": [0.0, 5.0]},
    "dispense_wait_time": {"type": "range", "bounds": [0.0, 5.0]},
    "retract_speed": {"type": "range", "bounds": [1.0, 15.0]},
    "blowout_vol": {"type": "range", "bounds": [0.0, 0.2]},
    "post_asp_air_vol": {"type": "range", "bounds": [0.0, 0.1]},
    "overaspirate_vol": {"type": "range", "bounds": [0.0, None]},  # Will be set to fixed maximum in create_model()
    "volume": {"type": "range", "bounds": [0.001, 1.0]},  # Volume parameter for transfer learning (mL)
}

def create_model(seed, num_initial_recs, bayesian_batch_size, volume=None, tip_volume=1.0, model_type="qNEHVI", 
                 optimize_params=None, fixed_params=None, simulate=False, max_overaspirate_ul=10.0, 
                 min_overaspirate_ul=0.0, transfer_learning=False, volume_bounds=None, init_method="SOBOL"):
    """
    Create an Ax client for 3-objective parameter optimization with optional transfer learning.
    
    Args:
        seed: Random seed
        num_initial_recs: Number of initial SOBOL suggestions
        bayesian_batch_size: Batch size for Bayesian optimization
        volume: Target pipetting volume (for single-volume mode) or None (for transfer learning)
        tip_volume: Tip capacity
        model_type: Model type (SOBOL, qLogEI, etc.)
        optimize_params: List of parameter names to optimize. If None, optimize all parameters.
        fixed_params: Dict of parameter names and values to fix
        simulate: Whether running in simulation mode
        max_overaspirate_ul: Maximum overaspirate volume (μL)
        min_overaspirate_ul: Minimum overaspirate volume (μL) 
        transfer_learning: Whether to enable volume as optimization parameter for transfer learning
        volume_bounds: Optional custom volume bounds for transfer learning
        init_method: Initialization method - "SOBOL" (default), "UNIFORM", "FACTORIAL" (LATIN_HYPERCUBE not available in this Ax version)
        fixed_params: Dict of parameter names and values to keep fixed
        simulate: Whether in simulation mode
        max_overaspirate_ul: Maximum overaspirate volume in microliters (default 10.0 µL)
        transfer_learning: If True, include volume as a parameter for cross-volume learning
        volume_bounds: [min_vol, max_vol] in mL for transfer learning mode
    """
    
    # Default to optimizing all parameters except volume if not specified
    if optimize_params is None:
        optimize_params = [p for p in DEFAULT_PARAMETER_BOUNDS.keys() if p != "volume"]
    
    # Default to no fixed parameters if not specified
    if fixed_params is None:
        fixed_params = {}
    
    # Handle transfer learning setup
    if transfer_learning:
        print(f"Creating 3-objective TRANSFER LEARNING optimizer:")
        print(f"  Volume bounds: {volume_bounds[0]*1000:.0f}-{volume_bounds[1]*1000:.0f}μL")
        optimize_params.append("volume")  # Include volume as optimizable parameter
    else:
        print(f"Creating 3-objective optimizer for volume {volume*1000:.0f}μL:")
    
    print(f"  Optimizing parameters: {optimize_params}")
    print(f"  Fixed parameters: {list(fixed_params.keys())}")
    print(f"  Objectives: {obj1_name}, {obj2_name}, {obj3_name}")

    # Set up acquisition function - use NEHVI for multi-objective optimization
    if model_type == "qNEHVI":
        model_gen_kwargs = {
            "botorch_acqf_class": qNoisyExpectedHypervolumeImprovement,
            "deduplicate": True,
        }
    elif model_type == "qLogEI":
        # LogEI doesn't directly support 3+ objectives, fall back to NEHVI
        print("  Warning: qLogEI doesn't support 3+ objectives, using qNEHVI instead")
        model_gen_kwargs = {
            "botorch_acqf_class": qNoisyExpectedHypervolumeImprovement,
            "deduplicate": True,
        }
    else:
        # Default to NEHVI for multi-objective
        model_gen_kwargs = {
            "botorch_acqf_class": qNoisyExpectedHypervolumeImprovement,
            "deduplicate": True,
        }
    
    
    # Map initialization method string to Models enum
    init_model_map = {
        "SOBOL": Models.SOBOL,
        "UNIFORM": Models.UNIFORM,
        "FACTORIAL": Models.FACTORIAL
    }
    
    # LATIN_HYPERCUBE not available in this Ax version - fallback to SOBOL
    if init_method == "LATIN_HYPERCUBE":
        print(f"Warning: LATIN_HYPERCUBE not available in this Ax version, falling back to SOBOL")
        init_method = "SOBOL"
    
    if init_method not in init_model_map:
        print(f"Warning: Unknown init_method '{init_method}', falling back to SOBOL")
        init_method = "SOBOL"
    
    init_model = init_model_map[init_method]
    print(f"  Using {init_method} initialization with {num_initial_recs} initial points")
    
    # Create generation strategy
    if not simulate:
        if num_initial_recs > 0:
            gs = GenerationStrategy(
                steps=[
                    GenerationStep(
                        model=init_model,
                        num_trials=num_initial_recs,
                        min_trials_observed=num_initial_recs,
                        max_parallelism=num_initial_recs,
                        model_kwargs={"seed": seed},
                        model_gen_kwargs=model_gen_kwargs,
                    ),
                    GenerationStep(
                        model=Models.BOTORCH_MODULAR,
                        num_trials=-1,
                        max_parallelism=bayesian_batch_size,
                        model_gen_kwargs=model_gen_kwargs,
                    ),
                ]
            )
        else:
            gs = GenerationStrategy(
                steps=[
                    GenerationStep(
                        model=Models.BOTORCH_MODULAR,
                        num_trials=-1,
                        max_parallelism=bayesian_batch_size,
                        model_gen_kwargs=model_gen_kwargs,
                    ),
                ]
            )
    else:
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=init_model,
                    num_trials=-1,
                    max_parallelism=5,
                    model_kwargs={"seed": seed},
                    model_gen_kwargs=model_gen_kwargs,
                )])

    ax_client = AxClient(generation_strategy=gs, verbose_logging=False)

    # Build parameters list - only include parameters we're optimizing
    parameters = []
    for param_name in optimize_params:
        param_config = DEFAULT_PARAMETER_BOUNDS[param_name].copy()
        param_config["name"] = param_name
        
        # Special handling for volume-dependent bounds
        if param_name == "overaspirate_vol":
            # Convert overaspirate bounds (microliters) to mL for consistency with other volumes
            max_overaspirate_ml = max_overaspirate_ul / 1000.0
            min_overaspirate_ml = min_overaspirate_ul / 1000.0
            param_config["bounds"] = [min_overaspirate_ml, max_overaspirate_ml]
            print(f"🔧 OPTIMIZER DEBUG: Setting overaspirate_vol bounds to [{min_overaspirate_ml:.6f}, {max_overaspirate_ml:.6f}] mL ([{min_overaspirate_ul:.1f}μL, {max_overaspirate_ul:.1f}μL])")
        elif param_name == "volume" and transfer_learning:
            # Use provided volume bounds for transfer learning
            if volume_bounds:
                param_config["bounds"] = volume_bounds
            else:
                raise ValueError("volume_bounds must be provided for transfer learning mode")
        
        parameters.append(param_config)
    
    # Create parameter constraints
    constraints = []
    
    # Handle constraints differently for transfer learning vs single volume
    if transfer_learning:
        # For transfer learning, we need volume-dependent constraints
        # Note: Ax supports constraints with parameters, so we can use "volume" in the constraint
        if "post_asp_air_vol" in optimize_params and "overaspirate_vol" in optimize_params:
            constraints.append(f"post_asp_air_vol + overaspirate_vol <= {tip_volume} - volume")
        elif "post_asp_air_vol" in optimize_params and "overaspirate_vol" in fixed_params:
            fixed_overaspirate = fixed_params["overaspirate_vol"]
            constraints.append(f"post_asp_air_vol <= {tip_volume} - volume - {fixed_overaspirate}")
        elif "overaspirate_vol" in optimize_params and "post_asp_air_vol" in fixed_params:
            fixed_post_asp = fixed_params["post_asp_air_vol"]
            constraints.append(f"overaspirate_vol <= {tip_volume} - volume - {fixed_post_asp}")
    else:
        # Single volume mode - use fixed volume value
        if "post_asp_air_vol" in optimize_params and "overaspirate_vol" in optimize_params:
            constraint_value = tip_volume - volume
            constraint_str = f"post_asp_air_vol + overaspirate_vol <= {constraint_value}"
            constraints.append(constraint_str)
            print(f"🔧 OPTIMIZER DEBUG: Adding constraint: {constraint_str} (tip_volume={tip_volume}, volume={volume})")
            
            # DEBUG: Show constraint values for verification
            print(f"🔍 CONSTRAINT VALUES:")
            print(f"   Tip volume constraint: {constraint_value * 1000:.1f}μL available for post_asp + overaspirate")
            print(f"   Parameter bound for overaspirate_vol: {max_overaspirate_ul:.1f}μL")
            print(f"   This should allow overaspirate_vol up to {max_overaspirate_ul:.1f}μL")
        elif "post_asp_air_vol" in optimize_params and "overaspirate_vol" in fixed_params:
            fixed_overaspirate = fixed_params["overaspirate_vol"]
            constraints.append(f"post_asp_air_vol <= {tip_volume - volume - fixed_overaspirate}")
        elif "overaspirate_vol" in optimize_params and "post_asp_air_vol" in fixed_params:
            fixed_post_asp = fixed_params["post_asp_air_vol"]
            constraints.append(f"overaspirate_vol <= {tip_volume - volume - fixed_post_asp}")

    ax_client.create_experiment(
        parameters=parameters,
        objectives={
            obj1_name: ObjectiveProperties(minimize=True, threshold=50),      # Accuracy (deviation %)
            obj2_name: ObjectiveProperties(minimize=True, threshold=10),      # Precision (variability %)
            obj3_name: ObjectiveProperties(minimize=True, threshold=90),      # Time (seconds)
        },
        parameter_constraints=constraints,
    )

    # Store fixed parameters for later use
    ax_client._fixed_params = fixed_params
    ax_client._optimize_params = optimize_params

    return ax_client

def get_suggestions(ax_client, volume=None, n=1, fixed_features=None):
    """Get parameter suggestions, combining optimized and fixed parameters.
    
    Args:
        ax_client: The Ax client
        volume: Volume for single-volume mode (backwards compatibility)
        n: Number of suggestions
        fixed_features: Dict of features to fix (for transfer learning)
    """
    suggestions = []
    
    # Handle both old and new API
    if fixed_features is None:
        fixed_features = {}
    
    # Only add volume to fixed features if it's actually a parameter in the search space
    # (i.e., when transfer learning is enabled and volume is in optimize_params)
    if (volume is not None and 
        "volume" not in fixed_features and 
        hasattr(ax_client, '_optimize_params') and 
        "volume" in ax_client._optimize_params):
        fixed_features["volume"] = volume
    
    for _ in range(n):
        # Get suggestions for optimized parameters with fixed features
        if fixed_features:
            # Convert dict to ObservationFeatures for Ax API
            fixed_obs_features = ObservationFeatures(parameters=fixed_features)
            optimized_params, trial_index = ax_client.get_next_trial(fixed_features=fixed_obs_features)
        else:
            optimized_params, trial_index = ax_client.get_next_trial()
        
        # Combine with fixed parameters from model creation
        full_params = dict(ax_client._fixed_params)  # Start with model fixed parameters
        full_params.update(optimized_params)  # Add optimized parameters
        
        # Remove volume from full_params if it was just a fixed feature
        if "volume" in full_params and volume is not None:
            del full_params["volume"]  # Don't include volume in pipetting parameters
        
        suggestions.append((full_params, trial_index))
    
    return suggestions

def add_result(ax_client, trial_index, results):
    """Add results for deviation, variability, and raw time.
    
    Args:
        ax_client: The Ax client instance
        trial_index: The trial index
        results: Dictionary containing 'deviation', 'variability', and 'time' keys
                variability should be None/NaN for single-replicate trials
    """
    
    # Debug: Print the results to check for NaN values
    print(f"DEBUG: Trial {trial_index} results: {results}")
    
    # Check for NaN values in results
    for key, value in results.items():
        if pd.isna(value):
            print(f"WARNING: NaN found in {key}: {value}")
    
    # Use raw time directly - no transformations
    raw_time = results["time"]
    
    # Handle variability - could be None for single-replicate trials
    variability = results.get("variability", None)
    if variability is None or pd.isna(variability):
        # For trials without replicates, use a penalty value or skip variability
        # We'll use a high penalty value to indicate "unknown precision"
        variability_value = 100.0  # High penalty for unknown precision
        print(f"DEBUG: No variability data for trial {trial_index}, using penalty value {variability_value}")
    else:
        variability_value = float(variability)
        print(f"DEBUG: Using variability={variability_value:.2f}% from replicates")
    
    print(f"DEBUG: Using raw time={raw_time:.2f}s (no transformations)")
    
    # Use all three objectives
    data = {
        "deviation": (results["deviation"], 0.0),
        "variability": (variability_value, 0.0),
        "time": (raw_time, 0.0),
    }
    
    # Additional check for NaN in the data being passed to Ax
    for metric, (mean, sem) in data.items():
        if pd.isna(mean) or pd.isna(sem):
            raise ValueError(f"NaN detected in {metric}: mean={mean}, sem={sem}")
    
    print(f"DEBUG: Completing trial {trial_index} with data: {data}")
    ax_client.complete_trial(trial_index=trial_index, raw_data=data)

def load_previous_data_into_model(ax_client, all_results, transfer_learning=False):
    """Load previous experimental results into the model, filtering for optimized parameters only."""
    if not all_results:
        return
    
    # Get which parameters we're optimizing
    optimize_params = ax_client._optimize_params
    
    # Define outcome columns
    outcome_columns = ['deviation', 'variability', 'time']  # All three objectives
    
    print(f"Loading {len(all_results)} existing trials into new model...")
    
    # Add each result as a completed trial (only optimization trials, not precision tests)
    optimization_results = [r for r in all_results if r.get('strategy') != 'PRECISION_TEST']
    print(f"  Loading {len(optimization_results)} optimization trials")
    
    for result in optimization_results:
        try:
            # Extract only the parameters we're optimizing
            parameters = {}
            for param in optimize_params:
                if param == "volume" and transfer_learning:
                    # For transfer learning, get volume from result
                    if "volume" in result:
                        parameters[param] = float(result[param])
                    else:
                        print(f"  Skipping result missing volume parameter")
                        break
                elif param in result:
                    # Convert to appropriate type
                    if param in ['aspirate_speed', 'dispense_speed']:
                        parameters[param] = int(result[param])
                    else:
                        parameters[param] = float(result[param])
            
            # Skip if we don't have all required parameters
            if len(parameters) != len(optimize_params):
                print(f"  Skipping result missing parameters: {set(optimize_params) - set(parameters.keys())}")
                continue
            
            # Extract outcomes
            raw_data = {}
            for col in outcome_columns:
                if col in result and result[col] is not None:
                    raw_data[col] = (float(result[col]), 0.0)  # (mean, sem)
                elif col == 'variability':
                    # Use penalty value for missing variability data
                    raw_data[col] = (100.0, 0.0)  # High penalty for unknown precision
            
            # Skip if we don't have all required outcomes (except variability which gets penalty)
            required_outcomes = ['deviation', 'time']
            if not all(col in raw_data for col in required_outcomes):
                missing = set(required_outcomes) - set(raw_data.keys())
                print(f"  Skipping result missing critical outcomes: {missing}")
                continue
            
            # Ensure variability is present (with penalty if missing)
            if 'variability' not in raw_data:
                raw_data['variability'] = (100.0, 0.0)
            
            # Attach trial to get trial index
            parameterization, trial_index = ax_client.attach_trial(parameters)
            
            # Complete the trial with the outcomes
            ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
            
        except Exception as e:
            print(f"  Error loading result into model: {e}")
            continue
    
    print(f"Successfully loaded previous data into new model")
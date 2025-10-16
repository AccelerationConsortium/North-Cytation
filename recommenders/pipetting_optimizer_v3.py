# pipetting_optimizer_v3.py
"""
Selective parameter optimization for pipetting calibration.
Supports optimizing only specific parameters while fixing others from previous successful volumes.
Only optimizes for deviation and time (removes variability objective).
"""

import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement

obj1_name = "deviation"
obj2_name = "time"  # Note: This will actually receive time_score values computed from raw time

# Default parameter bounds for all parameters
DEFAULT_PARAMETER_BOUNDS = {
    "aspirate_speed": {"type": "range", "bounds": [10, 35]},
    "dispense_speed": {"type": "range", "bounds": [10, 35]},
    "aspirate_wait_time": {"type": "range", "bounds": [0.0, 30.0]},
    "dispense_wait_time": {"type": "range", "bounds": [0.0, 30.0]},
    "retract_speed": {"type": "range", "bounds": [1.0, 15.0]},
    "blowout_vol": {"type": "range", "bounds": [0.0, 0.2]},  # Changed from pre_asp_air_vol, increased range
    "post_asp_air_vol": {"type": "range", "bounds": [0.0, 0.1]},
    "overaspirate_vol": {"type": "range", "bounds": [0.0, None]},  # Will be set to fixed maximum in create_model()
}

def create_model(seed, num_initial_recs, bayesian_batch_size, volume, tip_volume, model_type, 
                 optimize_params=None, fixed_params=None, simulate=False, max_overaspirate_ul=10.0, 
                 max_wait_time=30.0):
    """
    Create an Ax client for selective parameter optimization.
    
    Args:
        seed: Random seed
        num_initial_recs: Number of initial SOBOL suggestions
        bayesian_batch_size: Batch size for Bayesian optimization
        volume: Target pipetting volume
        tip_volume: Tip capacity
        model_type: Model type (SOBOL, qLogEI, etc.)
        optimize_params: List of parameter names to optimize. If None, optimize all parameters.
        fixed_params: Dict of parameter names and values to keep fixed
        simulate: Whether in simulation mode
        max_overaspirate_ul: Maximum overaspirate volume in microliters (default 10.0 µL)
        max_wait_time: Maximum wait time for aspirate_wait_time and dispense_wait_time (default 30.0 s)
    """
    
    # Default to optimizing all parameters if not specified
    if optimize_params is None:
        optimize_params = list(DEFAULT_PARAMETER_BOUNDS.keys())
    
    # Default to no fixed parameters if not specified
    if fixed_params is None:
        fixed_params = {}
    
    print(f"Creating optimizer for volume {volume*1000:.0f}μL:")
    print(f"  Optimizing parameters: {optimize_params}")
    print(f"  Fixed parameters: {list(fixed_params.keys())}")

    # Set up acquisition function
    if model_type == "qLogEI":
        model_gen_kwargs = {
            "botorch_acqf_class": qLogExpectedImprovement,
            "acqf_kwargs": {"eta": 1e-3},
            "deduplicate": True,
        }
    elif model_type == "qNEHVI":
        model_gen_kwargs = {
            "botorch_acqf_class": qNoisyExpectedHypervolumeImprovement,
            "deduplicate": True,
        }
    elif model_type == "qEI":
        model_gen_kwargs = {"deduplicate": True}
    else:
        model_gen_kwargs = {"deduplicate": True}
    
    # Create generation strategy
    if not simulate:
        if num_initial_recs > 0:
            gs = GenerationStrategy(
                steps=[
                    GenerationStep(
                        model=Models.SOBOL,
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
                    model=Models.SOBOL,
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
            # Convert max_overaspirate_ul (microliters) to mL for consistency with other volumes
            max_overaspirate_ml = max_overaspirate_ul / 1000.0
            param_config["bounds"] = [0.0, max_overaspirate_ml]  # Fixed maximum overaspirate volume
        elif param_name in ["aspirate_wait_time", "dispense_wait_time"]:
            # Use dynamic max_wait_time instead of hardcoded 30.0
            param_config["bounds"] = [0.0, max_wait_time]
        
        parameters.append(param_config)
    
    # Create parameter constraints
    constraints = []
    # Only add constraint if both parameters are being optimized
    if "post_asp_air_vol" in optimize_params and "overaspirate_vol" in optimize_params:
        constraints.append(f"post_asp_air_vol + overaspirate_vol <= {tip_volume - volume}")
    elif "post_asp_air_vol" in optimize_params and "overaspirate_vol" in fixed_params:
        # Fixed overaspirate_vol, variable post_asp_air_vol
        fixed_overaspirate = fixed_params["overaspirate_vol"]
        constraints.append(f"post_asp_air_vol <= {tip_volume - volume - fixed_overaspirate}")
    elif "overaspirate_vol" in optimize_params and "post_asp_air_vol" in fixed_params:
        # Fixed post_asp_air_vol, variable overaspirate_vol
        fixed_post_asp = fixed_params["post_asp_air_vol"]
        constraints.append(f"overaspirate_vol <= {tip_volume - volume - fixed_post_asp}")

    ax_client.create_experiment(
        parameters=parameters,
        objectives={
            obj1_name: ObjectiveProperties(minimize=True, threshold=50),
            obj2_name: ObjectiveProperties(minimize=True, threshold=90),
        },
        parameter_constraints=constraints,
    )

    # Store fixed parameters for later use
    ax_client._fixed_params = fixed_params
    ax_client._optimize_params = optimize_params

    return ax_client

def get_suggestions(ax_client, volume, n=1):
    """Get parameter suggestions, combining optimized and fixed parameters."""
    suggestions = []
    for _ in range(n):
        # Get suggestions for optimized parameters
        optimized_params, trial_index = ax_client.get_next_trial()
        
        # Combine with fixed parameters
        full_params = dict(ax_client._fixed_params)  # Start with fixed parameters
        full_params.update(optimized_params)  # Add optimized parameters
        
        suggestions.append((full_params, trial_index))
    
    return suggestions

def add_result(ax_client, trial_index, results, base_time_seconds=20, time_optimal_target=17, time_transition_mode="relu"):
    """Add results for only deviation and time_score (no variability).
    
    Args:
        ax_client: The Ax client instance
        trial_index: The trial index
        results: Dictionary containing 'deviation' and 'time' keys
        base_time_seconds: Base time threshold (used for other criteria, kept for compatibility)
        time_optimal_target: Optimal time target in seconds - score approaches 0 at optimal
        time_transition_mode: "relu" (max(0,x)), "smooth" (log(1+exp(x))), or "asymmetric" (gentle penalty for fast times)
    """
    
    # Debug: Print the results to check for NaN values
    print(f"DEBUG: Trial {trial_index} results: {results}")
    
    # Check for NaN values in results
    for key, value in results.items():
        if pd.isna(value):
            print(f"WARNING: NaN found in {key}: {value}")
    
    # Compute time_score using selected transition method:
    import numpy as np
    raw_time = results["time"]
    
    if time_transition_mode == "smooth":
        # Smooth transition (soft ReLU): log(1 + exp(x))
        # - Below optimal: Very small score (smooth approach to 0)
        # - At optimal: Small score (~0.69)  
        # - Above optimal: Approximately linear increase
        # This avoids sharp discontinuity issues with Bayesian optimization
        time_score = np.log(1 + np.exp(raw_time - time_optimal_target))
    elif time_transition_mode == "asymmetric":
        # Asymmetric transition: gentle penalty for fast times, standard ReLU for slow times
        # - Below optimal: Small linear penalty to discourage unstable fast times
        # - At optimal: No penalty (0)
        # - Above optimal: Standard linear penalty
        # Addresses instability of very fast times while not heavily penalizing them
        if raw_time < time_optimal_target:
            # Gentle discouragement for fast times (configurable factor)
            low_time_penalty_factor = 0.1  # Can be made configurable later
            time_score = (time_optimal_target - raw_time) * low_time_penalty_factor
        else:
            # Standard ReLU for times at or above optimal
            time_score = max(0, raw_time - time_optimal_target)
    else:  # "relu" (default)
        # ReLU transition: max(0, x)
        # - Below optimal: No penalty (0)
        # - At optimal: No penalty (0)
        # - Above optimal: Linear increase
        # Sharp cutoff - the original method that worked well
        time_score = max(0, raw_time - time_optimal_target)
    
    print(f"DEBUG: Computed time_score={time_score:.2f} from raw_time={raw_time:.2f}, optimal_target={time_optimal_target}s")
    
    # Only use deviation and time_score (ignore variability)
    data = {
        "deviation": (results["deviation"], 0.0),
        "time": (time_score, 0.0),
    }
    
    # Additional check for NaN in the data being passed to Ax
    for metric, (mean, sem) in data.items():
        if pd.isna(mean) or pd.isna(sem):
            raise ValueError(f"NaN detected in {metric}: mean={mean}, sem={sem}")
    
    print(f"DEBUG: Completing trial {trial_index} with data: {data}")
    ax_client.complete_trial(trial_index=trial_index, raw_data=data)

def load_previous_data_into_model(ax_client, all_results):
    """Load previous experimental results into the model, filtering for optimized parameters only."""
    if not all_results:
        return
    
    # Get which parameters we're optimizing
    optimize_params = ax_client._optimize_params
    
    # Define outcome columns
    outcome_columns = ['deviation', 'time']  # Only these two objectives
    
    print(f"Loading {len(all_results)} existing trials into new model...")
    
    # Add each result as a completed trial (only optimization trials, not precision tests)
    optimization_results = [r for r in all_results if r.get('strategy') != 'PRECISION_TEST']
    print(f"  Loading {len(optimization_results)} optimization trials")
    
    for result in optimization_results:
        try:
            # Extract only the parameters we're optimizing
            parameters = {}
            for param in optimize_params:
                if param in result:
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

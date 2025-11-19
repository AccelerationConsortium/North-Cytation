# pipetting_optimizer_single_objective.py
"""
Single-objective optimization for pipetting calibration focused on deviation only.
Used for subsequent volumes where we only need to optimize accuracy (deviation).
"""

import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from botorch.acquisition.logei import qLogExpectedImprovement

obj1_name = "deviation"  # Only objective

# Default parameter bounds (same as 3-objective version)
DEFAULT_PARAMETER_BOUNDS = {
    "aspirate_speed": {"type": "range", "bounds": [10, 35]},
    "dispense_speed": {"type": "range", "bounds": [10, 35]},
    "aspirate_wait_time": {"type": "range", "bounds": [0.0, 30.0]},
    "dispense_wait_time": {"type": "range", "bounds": [0.0, 30.0]},
    "retract_speed": {"type": "range", "bounds": [1.0, 15.0]},
    "blowout_vol": {"type": "range", "bounds": [0.0, 0.2]},
    "post_asp_air_vol": {"type": "range", "bounds": [0.0, 0.1]},
    "overaspirate_vol": {"type": "range", "bounds": [0.0, None]},  # Will be set to fixed maximum in create_model()
}

def create_model(seed, num_initial_recs, bayesian_batch_size, volume, tip_volume, model_type, 
                 optimize_params=None, fixed_params=None, simulate=False, max_overaspirate_ul=10.0, 
                 min_overaspirate_ul=0.0, init_method="SOBOL"):
    """
    Create an Ax client for single-objective parameter optimization (deviation only).
    
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
        init_method: Initialization method - "SOBOL" (default), "UNIFORM", "FACTORIAL" (LATIN_HYPERCUBE not available in this Ax version)
    """
    
    # Default to optimizing all parameters if not specified
    if optimize_params is None:
        optimize_params = list(DEFAULT_PARAMETER_BOUNDS.keys())
    
    # Default to no fixed parameters if not specified
    if fixed_params is None:
        fixed_params = {}
    
    print(f"Creating single-objective (deviation only) optimizer for volume {volume*1000:.0f}μL:")
    print(f"  Optimizing parameters: {optimize_params}")
    print(f"  Fixed parameters: {list(fixed_params.keys())}")
    print(f"  Objective: {obj1_name} (minimize)")

    # Set up acquisition function - use LogEI for single-objective optimization
    if model_type == "qLogEI":
        model_gen_kwargs = {
            "botorch_acqf_class": qLogExpectedImprovement,
            "acqf_kwargs": {"eta": 1e-3},  # numerical stability
            "deduplicate": True,
        }
    else:
        # Default for single-objective
        model_gen_kwargs = {"deduplicate": True}
    
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
            obj1_name: ObjectiveProperties(minimize=True, threshold=50),      # Only deviation objective
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

def add_result(ax_client, trial_index, results):
    """Add results for deviation only.
    
    Args:
        ax_client: The Ax client instance
        trial_index: The trial index
        results: Dictionary containing 'deviation' key (other keys ignored)
    """
    
    # Debug: Print the results to check for NaN values
    deviation = results["deviation"]
    print(f"DEBUG: Single-objective trial {trial_index} deviation: {deviation:.2f}%")
    
    # Check for NaN values
    if pd.isna(deviation):
        raise ValueError(f"NaN found in deviation: {deviation}")
    
    # Use only deviation objective
    data = {
        "deviation": (float(deviation), 0.0),  # (mean, sem)
    }
    
    print(f"DEBUG: Completing single-objective trial {trial_index} with deviation={deviation:.2f}%")
    ax_client.complete_trial(trial_index=trial_index, raw_data=data)

def load_previous_data_into_model(ax_client, all_results):
    """Load previous experimental results into the model, filtering for optimized parameters only."""
    if not all_results:
        return
    
    # Get which parameters we're optimizing
    optimize_params = ax_client._optimize_params
    
    print(f"Loading {len(all_results)} existing trials into single-objective model...")
    
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
            
            # Extract deviation outcome only
            if 'deviation' not in result or result['deviation'] is None:
                print(f"  Skipping result missing deviation")
                continue
            
            raw_data = {
                'deviation': (float(result['deviation']), 0.0)  # (mean, sem)
            }
            
            # Attach trial to get trial index
            parameterization, trial_index = ax_client.attach_trial(parameters)
            
            # Complete the trial with the outcomes
            ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
            
        except Exception as e:
            print(f"  Error loading result into single-objective model: {e}")
            continue
    
    print(f"Successfully loaded previous data into single-objective model")
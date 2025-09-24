# pipeting_optimizer_honegumi.py

import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement

obj1_name = "deviation"
obj2_name = "variability"
obj3_name = "time"

def create_model(seed, num_initial_recs, bayesian_batch_size, volume, model_type, simulate=False):

    if model_type == "qLogEI":
            # For qLogEI
        model_gen_kwargs = {
            "botorch_acqf_class": qLogExpectedImprovement,
            "acqf_kwargs": {"eta": 1e-3},  # often required for numerical stability
            "deduplicate": True,
        }

    elif model_type == "qNEHVI":
        # For qNEHVI
        model_gen_kwargs = {
            "botorch_acqf_class": qNoisyExpectedHypervolumeImprovement,
            "deduplicate": True,
            
        }
    elif model_type == "qEI":
        #Default qEI
        model_gen_kwargs = {"deduplicate": True}
    else:
        # Default for any other model type (like LLM)
        model_gen_kwargs = {"deduplicate": True}
    
    # ...existing code...
    
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
            # Skip SOBOL entirely if num_initial_recs is 0
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
    # ...existing code...

    ax_client = AxClient(generation_strategy=gs, verbose_logging=False)

    ax_client.create_experiment(
        parameters=[
            {"name": "aspirate_speed", "type": "range", "bounds": [5, 35]},
            {"name": "dispense_speed", "type": "range", "bounds": [5, 35]},
            {"name": "aspirate_wait_time", "type": "range", "bounds": [0.0, 30.0]},
            {"name": "dispense_wait_time", "type": "range", "bounds": [0.0, 30.0]},
            {"name": "retract_speed", "type": "range", "bounds": [1.0, 15.0]},
            {"name": "pre_asp_air_vol", "type": "range", "bounds": [0.0, 0.1]},
            {"name": "post_asp_air_vol", "type": "range", "bounds": [0.0, 0.1]},
            {"name": "overaspirate_vol", "type": "range", "bounds": [0.0, max(volume)/2]},
        ],
        objectives={
            obj1_name: ObjectiveProperties(minimize=True, threshold=50),
            obj2_name: ObjectiveProperties(minimize=True, threshold=5),
            obj3_name: ObjectiveProperties(minimize=True, threshold=90),
        },
        parameter_constraints=[
            f"post_asp_air_vol + overaspirate_vol <= {0.25-max(volume)}"
        ],
    )

    return ax_client

def get_suggestions(ax_client, volume, n=1):
    suggestions = []
    for _ in range(n):
        params, trial_index = ax_client.get_next_trial(        )
        suggestions.append((params, trial_index))
    return suggestions

def add_result(ax_client, trial_index, results):
    # Debug: Print the results to check for NaN values
    print(f"DEBUG: Trial {trial_index} results: {results}")
    
    # Check for NaN values in results
    for key, value in results.items():
        if pd.isna(value):
            print(f"WARNING: NaN found in {key}: {value}")
    
    data = {
        "deviation": (results["deviation"], 0.0),  # Use 0.0 instead of None for SEM
        "variability": (results["variability"], 0.0),
        "time": (results["time"], 0.0),
    }
    
    # Additional check for NaN in the data being passed to Ax
    for metric, (mean, sem) in data.items():
        if pd.isna(mean) or pd.isna(sem):
            raise ValueError(f"NaN detected in {metric}: mean={mean}, sem={sem}")
    
    print(f"DEBUG: Completing trial {trial_index} with data: {data}")
    ax_client.complete_trial(trial_index=trial_index, raw_data=data)

def load_data(ax_client, file_name):
    """
    Load existing experimental data from CSV file and add to Ax experiment.
    
    Args:
        file_name (str): Path to CSV file with experimental results
    """
    # Load the CSV data
    df = pd.read_csv(file_name)
    
    # Define parameter columns (the 8 columns after the 3 outcome columns)
    parameter_columns = [
        'aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 
        'dispense_wait_time', 'retract_speed', 'pre_asp_air_vol', 
        'post_asp_air_vol', 'overaspirate_vol'
    ]
    
    # Define outcome columns
    outcome_columns = ['deviation', 'variability', 'time']
    
    print(f"Loading {len(df)} existing trials from {file_name}")
    
    # Add each row as a completed trial
    for idx, row in df.iterrows():
        # Extract parameters - convert to int for speed parameters, float for others
        parameters = {
            'aspirate_speed': int(row['aspirate_speed']),
            'dispense_speed': int(row['dispense_speed']),
            'aspirate_wait_time': float(row['aspirate_wait_time']),
            'dispense_wait_time': float(row['dispense_wait_time']),
            'retract_speed': float(row['retract_speed']),
            'pre_asp_air_vol': float(row['pre_asp_air_vol']),
            'post_asp_air_vol': float(row['post_asp_air_vol']),
            'overaspirate_vol': float(row['overaspirate_vol'])
        }
        
        # Extract outcomes
        raw_data = {col: (float(row[col]), 0.0) for col in outcome_columns}  # (mean, sem)
        
        # Get trial from Ax - attach_trial returns (parameterization, trial_index)
        parameterization, trial_index = ax_client.attach_trial(parameters)
        
        # Complete the trial with the outcomes
        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        
    print(f"Successfully loaded {len(df)} trials into Ax experiment")
    return len(df)
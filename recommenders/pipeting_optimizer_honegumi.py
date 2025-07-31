# pipeting_optimizer_honegumi.py

import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Specified_Task_ST_MTGP_trans
from ax.core.observation import ObservationFeatures
from botorch.acquisition import UpperConfidenceBound

obj1_name = "deviation"
obj2_name = "variability"
obj3_name = "time"

def create_model(seed, num_initial_recs, volumes, model_type, simulate=False):

    if model_type == "explore":
        model_kwargs = {
            "transforms": Specified_Task_ST_MTGP_trans,
        }
        model_gen_kwargs = {
            "botorch_acqf_class": UpperConfidenceBound,
            "acqf_kwargs": {"beta": 2.0},
            "deduplicate": True,
        }
    else:
        model_kwargs = {
            "transforms": Specified_Task_ST_MTGP_trans,
        }
        model_gen_kwargs = {"deduplicate": True}

    
    if not simulate:
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=num_initial_recs,
                    min_trials_observed=num_initial_recs,
                    max_parallelism=5,
                    model_kwargs={"seed": seed},
                    model_gen_kwargs=model_gen_kwargs,
                ),
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=-1,
                    max_parallelism=3,
                    model_kwargs=model_kwargs,
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

    ax_client = AxClient(generation_strategy=gs)

    ax_client.create_experiment(
        parameters=[
            {"name": "aspirate_speed", "type": "range", "bounds": [5, 30]},
            {"name": "dispense_speed", "type": "range", "bounds": [5, 30]},
            {"name": "aspirate_wait_time", "type": "range", "bounds": [0.0, 30.0]},
            {"name": "dispense_wait_time", "type": "range", "bounds": [0.0, 30.0]},
            {"name": "retract_speed", "type": "range", "bounds": [1.0, 15.0]},
            {"name": "pre_asp_air_vol", "type": "range", "bounds": [0.0, 0.1]},
            {"name": "post_asp_air_vol", "type": "range", "bounds": [0.0, 0.1]},
 #           {"name": "blowout_vol", "type": "range", "bounds": [0.0, 0.1]},
            {
                "name": "volume",
                "type": "choice",
                "values": volumes,
                "is_ordered": True,
                "is_task": True,
                "target_value":volumes[0],
                "value_type": "float",
                "sort_values": True,
            },
        ],
        objectives={
            obj1_name: ObjectiveProperties(minimize=True, threshold=50),
            obj2_name: ObjectiveProperties(minimize=True, threshold=5),
            obj3_name: ObjectiveProperties(minimize=True, threshold=90),
        },
        parameter_constraints=[
            f"pre_asp_air_vol + post_asp_air_vol <= {0.25-max(volumes)}"
        ],
    )

    return ax_client

def get_suggestions(ax_client, volume, n=1):
    suggestions = []
    for _ in range(n):
        params, trial_index = ax_client.get_next_trial(
            fixed_features=ObservationFeatures({"volume": volume})
        )
        suggestions.append((params, trial_index))
    return suggestions

def add_result(ax_client, trial_index, results):
    data = {
        "deviation": (results["deviation"], None),
        "variability": (results["variability"], None),
        "time": (results["time"], None),
    }
    ax_client.complete_trial(trial_index=trial_index, raw_data=data)

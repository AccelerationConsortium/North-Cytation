# Recommender Guide

Reference for building and integrating recommenders in SDL workflows.
See also: `workflow_construction.md`, `analysis_and_data.md`

---

## What a Recommender Does

A recommender takes the existing experimental data (input parameters + measured outputs)
and returns the next batch of points to test. It is the component that makes a workflow
"self-driving" — rather than exhaustively scanning a grid, the recommender allocates
measurement budget toward the most informative regions.

Recommenders live in `recommenders/`.

The suggested parameters can be anything — concentrations, temperatures, timing, choice of reagent, protocol variant. They will often end up as volumes once translated to robot instructions, but the recommender itself works in whatever parameter space you define.

**When to add active learning**: only after a basic workflow (fixed grid, no recommender)
has run successfully on hardware and is producing reliable, reproducible data. A recommender
trained on noisy or inconsistent data will make bad suggestions and waste experimental budget.
Get the chemistry and robotics working first.

---

## Ax Campaign (default)

Ax is the default choice. It wraps BoTorch GPs and handles the full loop: Sobol initialization, then Bayesian optimization. Use `AxClient` for straightforward single- or multi-objective optimization.

```python
# recommenders/my_recommender.py
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.factory import Models

def create_ax_client(num_sobol_trials=10):
    generation_strategy = GenerationStrategy(
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=num_sobol_trials,
                           min_trials_observed=num_sobol_trials),
            GenerationStep(model=Models.GPEI, num_trials=-1),  # Bayesian phase
        ]
    )

    ax_client = AxClient(generation_strategy=generation_strategy)
    ax_client.create_experiment(
        parameters=[
            {"name": "param_1", "type": "range", "bounds": [0.0, 10.0]},
            {"name": "param_2", "type": "range", "bounds": [0.1, 50.0]},
            # Categorical parameter (e.g. choice of solvent, protocol, reagent)
            {"name": "solvent", "type": "choice", "values": ["water", "ethanol", "dmso"]},
        ],
        objectives={
            "my_output": ObjectiveProperties(minimize=False),  # or minimize=True
        },
    )
    return ax_client

def get_next_suggestion(ax_client):
    params, trial_index = ax_client.get_next_trial()
    return params, trial_index

def report_result(ax_client, trial_index, result_value):
    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data={"my_output": (result_value, None)},  # (mean, SEM); None = SEM unknown
    )
```

**Integration in workflow:**
```python
ax_client = create_ax_client(num_sobol_trials=10)

for trial in range(MAX_TRIALS):
    params, trial_index = get_next_suggestion(ax_client)
    # params is a dict: {"param_1": 4.3, "param_2": 12.1, "solvent": "ethanol"}

    # ... run experiment with params, measure result ...

    report_result(ax_client, trial_index, result_value)
```

**Multi-objective** (minimize two outputs simultaneously):
```python
ax_client.create_experiment(
    parameters=[...],
    objectives={
        "output_a": ObjectiveProperties(minimize=True),
        "output_b": ObjectiveProperties(minimize=True),
    },
)
# Report both values
ax_client.complete_trial(
    trial_index=trial_index,
    raw_data={"output_a": (val_a, None), "output_b": (val_b, None)},
)
```

See `sdl_pipette_calibration/bayesian_recommender.py` for a full production example with generation strategy configuration, parameter constraints, and multi-objective setup.

**Persisting state across sessions:**
```python
import json
with open(os.path.join(output_folder, "ax_client.json"), "w") as f:
    json.dump(ax_client.to_json_snapshot(), f)
# Restore
with open("ax_client.json") as f:
    ax_client = AxClient.from_json_snapshot(json.load(f))
```

---

## BayBe Campaign (alternative)

BayBe is simpler to set up than Ax and is a reasonable choice for straightforward single-objective optimization (minimize, maximize, or match a target value). It accumulates measurements as a DataFrame rather than trial-by-trial.

```python
from baybe import Campaign
from baybe.targets import NumericalTarget, TargetMode
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, CategoricalParameter
from baybe.searchspace import SearchSpace

campaign = Campaign(
    searchspace=SearchSpace.from_product(parameters=[
        NumericalContinuousParameter(name="param_1", bounds=(0.0, 10.0)),
        CategoricalParameter(name="solvent", values=["water", "ethanol", "dmso"]),
    ]),
    objective=SingleTargetObjective(
        NumericalTarget(name="my_output", mode=TargetMode.MAX)
    ),
)

# Get first batch (random)
suggestions = campaign.recommend(batch_size=8)

# After running experiments, add results and get next batch
# results_df must have all parameter columns + "my_output"
campaign.add_measurements(results_df)
next_suggestions = campaign.recommend(batch_size=8)
```

`suggestions` is a `pd.DataFrame` with one row per point and columns matching parameter names.

---

## Custom Recommenders (BoTorch)

If Ax does not support what you need — custom acquisition functions, multi-output GP models, boundary/transition-region tracking — you need to build something custom using BoTorch directly.

This codebase has an existing family of custom GP recommenders in `recommenders/` built on `TransitionRecommenderBase` (`recommenders/_transition_base.py`). These are purpose-built for mapping phase transitions across a concentration space and are a good reference if you need something similar. Before building your own, check whether one of these already covers your use case.

For anything else, start with the [BoTorch tutorials](https://botorch.org/tutorials/). Custom recommender work is research-level — budget time accordingly.

# degradation_optimizer.py
"""
Multi-objective Bayesian optimizer for acid degradation experiments using qNEHVI.

Inputs:
    acid              - categorical (e.g. "HCl", "H2SO4", ...)
    solvent           - categorical (e.g. "MeOH", "EtOH", ...)
    acid_molar_excess - float range

Objectives (all tracked as outcomes):
    max_degradation_ratio    - maximize
    degradation_rate_constant - maximize
    acid_molar_excess         - minimize (same value as the input parameter)
"""

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement

OBJ_DEGRADATION_RATIO    = "max_degradation_ratio"
OBJ_RATE_CONSTANT        = "degradation_rate_constant"
OBJ_ACID_EXCESS          = "acid_molar_excess"


def create_model(
    seed,
    num_initial_recs,
    bayesian_batch_size,
    acid_choices,
    solvent_choices,
    acid_molar_excess_bounds=(1.0, 5.0),
):
    """
    Create an AxClient configured for qNEHVI multi-objective optimisation.

    Args:
        seed                   : Random seed for reproducibility.
        num_initial_recs       : Number of quasi-random SOBOL trials before
                                 switching to Bayesian optimisation.
        bayesian_batch_size    : Parallel batch size for Bayesian steps.
        acid_choices           : List of acid strings, e.g. ["HCl", "H2SO4"].
        solvent_choices        : List of solvent strings, e.g. ["MeOH", "EtOH"].
        acid_molar_excess_bounds: (min, max) float bounds for acid molar excess.

    Returns:
        ax_client: Configured AxClient ready for get_suggestions / add_result.
    """

    model_gen_kwargs = {
        "botorch_acqf_class": qNoisyExpectedHypervolumeImprovement,
        "deduplicate": True,
    }

    if num_initial_recs > 0:
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=num_initial_recs,
                    min_trials_observed=num_initial_recs,
                    max_parallelism=num_initial_recs,
                    model_kwargs={"seed": seed},
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

    ax_client = AxClient(generation_strategy=gs, verbose_logging=False)

    ax_client.create_experiment(
        parameters=[
            {
                "name": "acid",
                "type": "choice",
                "values": acid_choices,
                "is_ordered": False,
            },
            {
                "name": "solvent",
                "type": "choice",
                "values": solvent_choices,
                "is_ordered": False,
            },
            {
                "name": OBJ_ACID_EXCESS,
                "type": "range",
                "bounds": list(acid_molar_excess_bounds),
                "value_type": "float",
            },
        ],
        objectives={ #Owen's Note: threshold means "good enough to care"... for ratio, anything below this will be considered bad; I have guessed these values
            OBJ_DEGRADATION_RATIO: ObjectiveProperties(minimize=False, threshold=2.0),
            OBJ_RATE_CONSTANT:     ObjectiveProperties(minimize=False, threshold = -0.5), #Test this
            OBJ_ACID_EXCESS:       ObjectiveProperties(minimize=True, threshold = 500),
        },
    )

    return ax_client


def get_suggestions(ax_client, n=1):
    """
    Get up to n experimental suggestions from the model.

    Returns:
        List of (parameters_dict, trial_index) tuples.
    """
    trials = ax_client.get_next_trials(max_trials=n)

    # Ax returns a mapping like:
    # {trial_index: parameters_dict, ...}
    suggestions = [(params, trial_index) for trial_index, params in trials.items()]
    return suggestions


def add_result(ax_client, trial_index, results):
    """
    Report observed outcomes for a completed trial.

    Args:
        ax_client   : The AxClient instance.
        trial_index : Trial index returned by get_suggestions.
        results     : Dict with keys:
                        "max_degradation_ratio"    -> float
                        "degradation_rate_constant" -> float
                        "acid_molar_excess"         -> float  (echoed back from suggestion)

    All SEM values default to 0.0 (noiseless observations).
    """
    required = {OBJ_DEGRADATION_RATIO, OBJ_RATE_CONSTANT, OBJ_ACID_EXCESS}
    missing = required - set(results.keys())
    if missing:
        raise ValueError(f"Results dict is missing keys: {missing}")

    raw_data = {key: (float(results[key]), 0.0) for key in required}

    print(f"Trial {trial_index} results: {raw_data}")
    ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)


def load_previous_data(ax_client, all_results):
    """
    Warm-start the model by attaching previously observed results.

    Args:
        ax_client   : The AxClient instance (experiment must already be created).
        all_results : List of dicts, each containing:
                        "acid", "solvent", "acid_molar_excess",
                        "max_degradation_ratio", "degradation_rate_constant"

    Rows missing any required field are skipped with a warning.
    """
    if not all_results:
        return

    param_keys   = {"acid", "solvent", OBJ_ACID_EXCESS}
    outcome_keys = {OBJ_DEGRADATION_RATIO, OBJ_RATE_CONSTANT, OBJ_ACID_EXCESS}
    loaded = 0

    for i, row in enumerate(all_results):
        try:
            missing_params   = param_keys   - set(row.keys())
            missing_outcomes = outcome_keys - set(row.keys())
            if missing_params or missing_outcomes:
                print(f"  Row {i}: skipping - missing fields {missing_params | missing_outcomes}")
                continue

            parameters = {
                "acid":          str(row["acid"]),
                "solvent":       str(row["solvent"]),
                OBJ_ACID_EXCESS: float(row[OBJ_ACID_EXCESS]),
            }

            raw_data = {key: (float(row[key]), 0.0) for key in outcome_keys}

            _, trial_index = ax_client.attach_trial(parameters)
            ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
            loaded += 1

        except Exception as e:
            print(f"  Row {i}: error loading - {e}")
            continue

    print(f"Loaded {loaded}/{len(all_results)} previous results into model.")

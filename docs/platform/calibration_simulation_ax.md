# Fast calibration simulation with Ax

Investigated 2026-10-19. Recommendation only; no workflow implementation changed.

## Recommendation

Keep Ax's existing trial lifecycle, search space and constraints, but select a single unlimited Sobol generation step during fast simulation. Sobol supplies quasi-random suggestions without fitting a Bayesian surrogate. Ax documents both manually configured generation strategies and the built-in `no_bayesian_optimization=True` strategy-selection option. The explicit step fits this repository's existing custom strategy best. [Ax generation strategy documentation](https://ax.dev/docs/0.5.0/tutorials/generation_strategy/)

For the old API already used in `sdl_pipette_calibration/bayesian_recommender.py`:

```python
generation_strategy = GenerationStrategy(steps=[
    GenerationStep(model=Models.SOBOL, num_trials=-1,
                   model_kwargs={"seed": 0}),
])
```

`num_trials=-1` means unlimited generation in the final step; a seed makes suggestions reproducible. This is an inference from the documented API and current local imports, not a runtime-verified patch. Official 0.4.3 archive/source URLs were unavailable to the browser; the official 0.5.0 documentation confirms this old API shape. [Ax generation strategy documentation](https://ax.dev/docs/0.5.0/tutorials/generation_strategy/)

## Local integration findings

- `bayesian_recommender.py:404` constructs Sobol-then-Bayesian or pure Bayesian strategies. Branch before `_get_model_and_kwargs` in fast simulation so no Bayesian backend construction is needed.
- `suggest_parameters` still calls `AxClient.get_next_trial`, applies rounding/fixed parameters, and records the trial index. Feedback uses `complete_trial`, so those checks remain exercised.
- `suggest_parameters` requests model predictions unconditionally at line 581. Skip this in Sobol mode; random generators do not provide predictions (the existing exception handler already tolerates their absence). The official tutorial shows the corresponding unsupported-prediction warning. [Ax generation strategy documentation](https://ax.dev/docs/0.5.0/tutorials/generation_strategy/)
- Local `get_best_parameters` at line 724 selects the best observed trial directly, so it does not require Ax model predictions.
- Preserve objective definitions independently of suggestion mode, particularly multi-objective metric formatting.
- Sobol does not learn from outcomes. Simulation tests trial execution and workflow control flow, but cannot validate GP fitting, acquisition optimization, or calibration quality. Keep an independent switch to restore Bayesian simulation.

## Version caveat

The root `requirements.txt` pins `ax-platform==1.1.2`, while `sdl_pipette_calibration/requirements.txt` pins `ax-platform==0.4.3`. Calibration source imports `ax.modelbridge` and `Models`, the older API. Resolve the actual calibration runtime before implementation; the default Python 3.12 interpreter has no installed `ax-platform` metadata, so no timing or runtime compatibility claim was tested.

Ax 1.1.2 uses `from ax.adapter.registry import Generators`, `GenerationStep` from `ax.generation_strategy.generation_node`, and `GenerationStrategy` from `ax.generation_strategy.generation_strategy`. Its step argument is `generator=Generators.SOBOL`, not `model=Models.SOBOL`. It also provides a uniform-random generator, but Sobol is the closest fit to the existing calibration implementation. [Ax 1.1.2 GenerationStep source](https://raw.githubusercontent.com/facebook/Ax/1.1.2/ax/generation_strategy/generation_node.py), [Ax 1.1.2 registry source](https://raw.githubusercontent.com/facebook/Ax/1.1.2/ax/adapter/registry.py)

## Scope assessment

The suggestion strategy change is small. File suppression is broader: parent investigation found existing output flags miss incremental/emergency saves and enhanced outputs, while workflow preflight and validation consume final CSV data. Preserve those checks through a shared in-memory table representation before suppressing export. Recommended independent internal simulation-only controls: `_SAVE_FILES_ON_SIMULATE = False` and `_USE_BAYES_OPT_ON_SIMULATE = False`. Hardware behavior remains on the existing path.

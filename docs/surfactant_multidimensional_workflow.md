# Surfactant multidimensional workflow (N surfactants, N >= 2)

Run from the repository root:

```powershell
python -m workflows.surfactant_multidimensional_workflow
```

Generalizes
[`workflows/surfactant_grid_adaptive_concentrations.py`](../workflows/surfactant_grid_adaptive_concentrations.py)
from 2D (SDS + TTAB) to N-D (any 2-5 surfactants from `SURFACTANT_LIBRARY`),
reusing its substock prep, dispensing primitives, and measurement helpers.
Set `SURFACTANTS` to the list of surfactant names to screen (default
`["TTAB", "DTAB", "SDS"]`); the whole workflow is parameterized over this list.

Differs from the 2D workflow in three places only:
1. `SURFACTANTS` drives column generation (N concentration/volume/substock
   columns instead of a fixed `surf_A_*` / `surf_B_*` pair).
2. Initial sampling is a log-spaced N-D grid (cube approach) instead of a 2D
   meshgrid; the per-axis max is `stock_conc * (max_vol_per_surf / well_vol)`
   with `max_vol_per_surf = budget / N`, so every cube point is feasible by
   construction.
3. The recommender call uses the N-D-capable transition recommenders
   (`DelaunaySimplexTransitionRecommender`, `BayesianTransitionRecommender`,
   `GradientTransitionRecommender`, `LevelSetTransitionRecommender`, or
   Sobol/Random baselines) via the shared `TransitionRecommenderBase`
   interface. Select one with `RECOMMENDER_TYPE`
   (`'triangle' | 'bayesian' | 'gradient' | 'levelset' | 'sobol' | 'random'`).

Excluded versus the 2D workflow: kinetics, CMC controls, adaptive
baseline-rectangle re-bounding, 2D heatmaps, and contour plots. Post-experiment
N-D analysis is left to a separate script that reads the saved CSV.

Turbidity-based filtering controls what the recommender sees:
`FILTER_UNRELIABLE_RATIOS` nulls out `ratio` for high-turbidity wells above
`TURBIDITY_FILTER_THRESHOLD`; `FLAT_TURBIDITY_MAX` drops turbidity entirely
from the recommender's output columns if the whole dataset's max turbidity
stays below it. `OUTPUT_COLUMNS_OVERRIDE` bypasses this automatic logic when
set to an explicit column list. `TURBIDITY_PLOT_THRESHOLD` only affects the
final 3D turbidity visualization, not active-learning decisions.

Uses a guarded `execute(config=None, show_gui=True)` entrypoint and module
globals for startup configuration, same as the other retained workflows. The
shared adaptive-helper failure-reporting caveats in
[`workflows/notes/workflow_comments.md`](../workflows/notes/workflow_comments.md)
apply here too; no full hardware run has been performed for this workflow yet.

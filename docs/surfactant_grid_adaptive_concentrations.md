# Surfactant grid (adaptive concentrations)

Run from the repository root:

```powershell
python -m workflows.surfactant_grid_adaptive_concentrations
```

Screens a 2D grid of two surfactants (default SDS + TTAB) for turbidity and
pyrene fluorescence, with each surfactant's concentration range derived from
its stock concentration and the volume budget per well
(`max_conc = stock_conc * allocated_volume / well_volume`), spaced
logarithmically over a fixed number of levels (9 by default).

Set `VALIDATE_LIQUIDS=True` to run pipetting validation alongside the full
experiment, or `VALIDATION_ONLY=True` to run just the validation and skip the
experiment (useful for testing). Both modes save results under
`experiment_name/calibration_validation/`.

Raw Cytation reads are backed up immediately to
`output/cytation_raw_backups/`, and processed measurements are backed up to
`output/measurement_backups/` after each interval. If processing fails mid-run,
use `recover_raw_cytation_data()` and `recover_from_measurement_backups()` to
rebuild results from the backups instead of re-running the plate.

Failure results can be returned with `workflow_complete=False` without
raising, and some status messages use `print` instead of the experiment
logger — do not assume a failed run always produced an ERROR log record.

This file is also the shared library for two companion workflows:
`surfactant_grid_ailsa.py` (fixed SDS/BDDAC recipe replay with per-workflow
dye/solvent config) and `surfactant_multidimensional_workflow.py` (N-surfactant
generalization, N = 2-5). Public helper names listed in
[`workflows/notes/REFACTORING_surfactants.md`](../workflows/notes/REFACTORING_surfactants.md)
must stay stable, or those imports must be updated in the same change.
See that file for the current refactor plan — the module is intentionally not
yet split up (~5,200 lines, ~80 top-level definitions).

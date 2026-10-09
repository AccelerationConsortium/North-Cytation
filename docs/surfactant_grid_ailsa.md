# Surfactant grid - Ailsa (SDS + BDDAC replay)

Run from the repository root:

```powershell
python -m workflows.surfactant_grid_ailsa
```

Re-runs a previous surfactant grid experiment from its recipe/stock CSVs in
one continuous pass, reusing the dispensing, substock, and measurement helpers
from [`workflows/surfactant_grid_adaptive_concentrations.py`](../workflows/surfactant_grid_adaptive_concentrations.py)
(imported, not copied). It owns its own top-level control flow so study
metadata, replicates, randomization, and dye/solvent behavior can be added
here without changing the baseline replay workflow.

Uses a dedicated vial layout (`status/surfactant_grid_ailsa_vials.csv`) for
SDS + BDDAC — do not reuse the shared `adaptive_concentrations` vial file,
which other experiments rely on. Recipe/stock CSVs default to
`inputs/gradient_proposal_snapped_96well_recipe.csv` and
`inputs/experiment_plan_stock_solutions_SDS_BDDAC.csv`; override both in
`workflow_configs/surfactant_grid_ailsa.yaml`. Set `MAX_WELLS` to `0` to run
every well, or to a smaller number (e.g. 96 for one plate) to stop early.

Runtime behavior matches the replay workflow when `DYE="pyrene"`.
`coumarin-6` and `nile-red` dispensing is implemented for hardware runs, but
`simulate_dye_fluorescence()` only fabricates data for `DYE="pyrene"` —
`SIMULATE=True` with another dye still raises `NotImplementedError`.
`DISPENSE_ORDER` controls the per-well pipetting sequence (default
`["surfactant_B", "water", "surfactant_A", "dye"]`); `DYE_SOLVENT` is the
calibrated liquid used to dispense the dye.

Water vials are topped up before every `REFILL_CHECK_CHUNK_SIZE`-well chunk
using `WATER_REFILL_THRESHOLD_ML` (kept high so the check always triggers;
`fill_water_vial` skips internally if already near full). Substocks/stocks use
the separate `REFILL_THRESHOLD_ML = 4.0 mL`. `EXPERIMENT_TAG` is appended to
the generated output folder name.

Startup refreshes the confirmed config after GUI review and accepts a
complete supplied config with `show_gui=False`, same as the other retained
workflows — see
[`workflows/notes/workflow_comments.md`](../workflows/notes/workflow_comments.md)
for outstanding reconciliation notes on this launch path.

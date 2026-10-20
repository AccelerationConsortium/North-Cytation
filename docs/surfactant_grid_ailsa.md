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

`WELLPLATE_TYPE` selects the physical plate. The recipe rows are split across as
many plates as needed (96 rows use two 48-well plates or four 24-well plates).
`VOLUME_SCALE_BY_PLATE` scales all per-well recipe ingredient volumes and the
configured base `DYE_VOLUME_UL` without changing either source CSV. The defaults
are 1x for 96 wells, 2x for 48 wells, and 4x for 24 wells. The workflow checks
the scaled total against the configured plate capacity. A 24-well plate also
requires a robot wellplate definition and compatible reader protocols before use.

`MEASUREMENT_SCHEDULE_MIN` sets elapsed minutes from the end of each dispensing
block. The default `[0]` runs the original single measurement sequence. At
every timepoint, each `MEASUREMENT_REPLICATES` read repeats the same shake/wait,
turbidity, then fluorescence sequence. Results include the scheduled timepoint
and actual elapsed time. Mean tables group reads by scheduled timepoint, and
plots and scoring are generated separately for each timepoint when there are
multiple scheduled times.

`RANDOMIZED_MEASUREMENT_ORDER: true` independently shuffles the physical well
read sequence for each measurement replicate and timepoint. Set
`MEASUREMENT_RANDOMIZATION_SEED` for repeatable shuffles. Both turbidity and
fluorescence use the same order in a given replicate; each well is sent to the
reader separately so its normal grouping logic cannot sort the sequence. This
can make reads substantially slower. The default `false` keeps the existing
reader grouping and physical well order. Results record the 1-based
`measurement_order_position` for each read.

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

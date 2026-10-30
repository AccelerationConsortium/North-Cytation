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
calibrated liquid used to dispense the dye. `DYE_VIAL` selects the physical
primary source vial; `DYE` selects the fluorescence protocol and analysis metric.
Set both when switching dye solutions, and update any backup vial names too.

The dye can dispense directly from an ordered set of 8 mL vials. The local
coumarin-6 preset uses:

```yaml
DYE_VIAL: dye_coumarin_6
DYE_BACKUP_VIALS: [dye_coumarin_6_2]
DYE_SOURCE_RESERVE_ML: 2.0
```

`dye_coumarin_6_2` has its home at `main_8mL_rack[18]` and starts at **0 mL**
in the dedicated inventory. Load the same dye concentration and solvent as the
primary vial, then enter its actual loaded volume. Backups must have separate
main-rack home positions outside the dispensing staging slots. An empty backup
list retains a single dye source. Reload settings in already queued jobs.

The workflow uses the primary vial first, leaving a configurable 2 mL reserve
for small-tip aspiration. When it cannot supply another full well dose, it
returns that vial home, stages the next source at the clamp, and conditions a
fresh tip before continuing. A plate can use both sources; each well receives
its entire dye dose from one vial. `dye_source_vial` and `dye_vial` in the output
identify the actual source for each well. No dye is transferred between vials.

Startup checks recorded supply for every preparation replicate in the current
run, and each dispensing block rechecks the live volumes before staging a dye
vial. Insufficient combined supply raises an error before dye dispensing.
Optional liquid validation uses each dye source allocated to the run. Selection
uses inventory volumes rather than physical empty-vial detection. For two
loaded vials, allow 4 mL total retained reserve in addition to the dye volume
dispensed into wells; the previous 8.64 mL two-run estimate therefore needs at
least 12.64 mL loaded across the two vials, plus handling allowance.

Water vials are topped up before every `REFILL_CHECK_CHUNK_SIZE`-well chunk
using `WATER_REFILL_THRESHOLD_ML` (kept high so the check always triggers;
`fill_water_vial` skips internally if already near full). Substocks/stocks use
the separate `REFILL_THRESHOLD_ML = 4.0 mL`. `EXPERIMENT_TAG` is appended to
the generated output folder name.

Stock refills can draw from multiple vials in order. The local preset now uses:

```yaml
STOCK_REFILL_SOURCES:
  SDS_stock: [SDS_refill, SDS_refill_2]
  BDDAC_stock: [BDDAC_refill, BDDAC_refill_2]
STOCK_REFILL_RESERVE_ML: 0.1
```

The dedicated inventory assigns `SDS_refill_2` to `large_vial_rack[2]` and
`BDDAC_refill_2` to `large_vial_rack[3]`. These new entries start at **0 mL**.
Load each with the same solution and concentration as its corresponding parent
stock, then enter the actual loaded volume in the vial CSV before running.
The four-slot large-vial rack holds the two original sources and these backups.
Additional sources can be listed if they have suitable, physically available
inventory locations. Do not assign the same source to different stock solutions.
An empty mapping retains the original `SDS_refill` / `BDDAC_refill` convention.
Queued scheduler jobs retain their own settings: reload the updated preset or
edit their refill lists and inventory before launching them.

At startup, the workflow validates source lists, recorded volumes, and substock
dependencies. Each stock top-up targets 7.5 mL and is allocated completely before
its first transfer. It uses each source down to its configured reserve, then
continues with the next source, including splitting one top-up between vials.
Contributions below 200 uL are left behind; allocations leave at least 200 uL
for the final contribution. Top-ups smaller than 200 uL are rounded up to
200 uL, keeping the destination below its 8 mL capacity. Each transfer is logged.
If the pool cannot supply a complete top-up, the workflow raises an error
before that top-up transfers any liquid. This is a check of each refill, not a
prediction that the entire experiment has enough supply. Selection uses the
controller's recorded volumes, not physical empty-vial detection; accurate
loading and inventory records remain necessary.

Substock preparation now checks its parent source before every transfer and
can refill either a stock or an upstream dilution. Preparation failures stop
the Ailsa workflow. Other workflows retain the shared helper's existing behavior.
Before plate dispensing, each chunk is limited by both `REFILL_CHECK_CHUNK_SIZE`
and the working vial's usable volume. If scaled wells need more liquid than
the current vial holds, it refills before that chunk, even above the normal
4 mL threshold. Chunks retain 0.1 mL in the working vial. Water continues to
use reservoir 1; these backup lists apply to the parent surfactant stocks.

Simulations skip study CSVs, measurement backups, plots, and scoring by default.
Set the private `_SAVE_SIMULATION_OUTPUT` constant in the workflow to `True`
for a diagnostic run that needs those files. The scheduler still records its
simulation state and job result for subsequent jobs.

Startup refreshes the confirmed config after GUI review and accepts a
complete supplied config with `show_gui=False`, same as the other retained
workflows — see
[`workflows/notes/workflow_comments.md`](../workflows/notes/workflow_comments.md)
for outstanding reconciliation notes on this launch path.
# Embedded pipetting validation

Set `VALIDATE_LIQUIDS: true` in the workflow config to run gravimetric
checks before the study, or `VALIDATION_ONLY: true` to run only those checks.
The checks use the scaled well recipe volumes and run separate sessions for
water, dye solvent, and each named surfactant. Sessions are separated by the
tip resolved for each planned volume. Each tip group checks its minimum,
middle, and maximum planned volumes with three replicates. SDS and BDDAC
source vials are validated separately. Both surfactant solutions use
`liquid_type="SDS"` for pipetting parameters and gravimetric density; the vial
name identifies the physical source.


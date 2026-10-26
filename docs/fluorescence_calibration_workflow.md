# Fluorescence calibration

After startup confirmation, every run calls `home_robot_components()` before
planning, validation, or liquid handling. This includes each queued preparation
and measurement job. The existing controller method homes the carousel, robot
axes, pumps, and configured track axes; simulation logs the call without motion.

Run from the repository root:

```powershell
python -m workflows.fluorescence_calibration_workflow
```

Set `TWO_RUNS_PER_PLATE: true` to use exactly two runs per plate. Each run must
have one repetition and fit in half the plate (48 wells on a 96-well plate).
The first run uses the top half and leaves the plate at the pipetting area;
the second uses the bottom half and discards it. With 45 wells per run this is
A1–D9, then E1–H9. Each run keeps its own measurement schedule.
The separate `robot_state/fluorescence_plate.yaml` marker file survives
process restarts. Scheduler simulations carry it in their private state copies,
without changing live state. An interrupted dispensing/measurement run is marked
`in_progress` and cannot automatically reuse the plate: retire that plate first.
Do not manually swap a retained plate without updating track state. Other
workflows do not consume the second half; retire the plate before switching to
one of those workflows. The default is false; the local preset enables it.
Already queued scheduler rows have private settings: reload the updated preset
or set `TWO_RUNS_PER_PLATE: true` in both rows before running them.

The first run creates `workflow_configs/fluorescence_calibration_workflow.yaml`.
The default `SIMULATE: true` writes plans only, without importing hardware drivers
or fabricating fluorescence measurements. Config files follow the repository's
existing convention and are locally ignored by Git.

The default experiment uses 45 wells: three media (water, premixed surfactant,
native dye solvent), three dye levels plus a matched solvent-only blank, each
in triplicate. Each well receives 195 uL medium and 5 uL dye substock or solvent.
For acetone or ethyl acetate runs, set `VOLATILE_SOLVENT: true`. This skips
solvent-medium wells even when `TEST_SOLVENT` is enabled. `TOTAL_VOLUME_UL` then
means the aqueous volume intended to remain after the dye solvent evaporates:
with `TOTAL_VOLUME_UL: 250` and `DYE_VOLUME_UL: 20`, the robot dispenses
250 uL water or surfactant solution plus 20 uL dye substock (or pure-solvent
blank). The initial 270 uL must fit the plate. Reported dye concentration uses
20/250, and the planned final solvent fraction is zero. These calculations
assume the added solvent evaporates completely while the dye stays in the well;
the workflow does not measure or wait for evaporation. Lower the physical parent
stock concentration, set `STOCK_CONCENTRATION_UM` to its actual value, or adjust
`DILUTION_FACTORS` when increasing the dye addition. With
`VOLATILE_SOLVENT: false`, the original total-dispensed-volume behavior applies.

The local Nile Red / ethyl acetate preset preserves the former 5 uL dye
concentrations with 20 uL additions by using factors 0, 0.025, 0.0625, 0.125
and 0.25. All four nonzero levels now require substocks. The fourth destination,
`nilered_ethylacetate_s4`, is assigned to main-rack slot 35. The four 6 mL
recipes consume 2.775 mL parent stock and 21.225 mL pure ethyl acetate.
Each run uses 3.75 mL water, 3.75 mL surfactant solution, 0.12 mL of each
substock and 0.12 mL working solvent for blanks.

The bulk rack is occupied, so pure-solvent sources `ethyl_acetate` through
`ethyl_acetate_4` are assigned to main-rack slots 2, 3, 4 and 5. Load each
with 8 mL and enter the actual loaded volumes in the vial CSV; new entries
start at zero. This supplies every preparation addition while retaining the
configured 2 mL per-source reserve. Keep main-rack slots 0 and 1 empty for
closed-cap transfer staging. The existing working solvent is recorded at
7.77 mL, so it does not need an initial refill for this run.

Startup solvent, water, and surfactant refills occur before liquid validation.
Validated runs prefer filling the working solvent to its normal target before
checks, rather than validating from a nearly empty working vial. After validation,
the workflow re-reads controller inventory, checks supply again, and adds working
solvent if required to cover plate dispensing plus 2 mL. Validation-only runs
also perform the prerequisite refills.

Without validation, working solvent is refilled only when its recorded volume cannot cover the
upcoming plate demand plus 2 mL. A 6 mL fill is preferred, not mandatory:
the refill can stop below that target if it still covers the run, while
reserving solvent needed for pending substocks. Each solvent addition draws
from sources in configured order, using each down to its reserve before moving
to the next. Source allocations retain the configured
per-container reserve and use whole-uL amounts. Validation includes the
source-specific additions before resolving their executable pipetting chunks.
Each source contribution must be at least 200 uL. Smaller usable leftovers
remain in their source; splits leave at least 200 uL for the final contribution.
The total requested addition is unchanged, or allocation fails before transfers.

Before startup refills and preparation, the workflow checks the whole run's
parent-stock demand, prepared or reused substock supply, aqueous supply after
startup refills, and solvent allocation in actual transfer order. It retains
2 mL in the working solvent vial, 0.1 mL in other plate/stock sources, and the configured reserve in bulk solvent
sources. Plans requiring more aqueous liquid than the startup fill provides
are rejected; there are no automatic mid-run aqueous refills. These checks
depend on accurate inventory records and do not measure evaporation losses.

`TOTAL_VOLUME_UL` and `DYE_VOLUME_UL` are base 96-well values. Selecting a
different `WELLPLATE_TYPE` applies `VOLUME_SCALE_BY_PLATE` to both volumes:
1x for 96 wells, 2x for 48 wells, and 4x for 24 wells by default. The plan
spreads wells across as many physical plates as needed, preserving the dye
fraction and final concentrations. Substock preparation volumes remain fixed;
the workflow checks that the prepared amount covers the scaled dye demand.
The 24-well layout can be planned now but requires robot plate geometry and
additional within-plate liquid refills before hardware execution. The current
8 mL working-vial refill path does not cover every scaled multi-plate plan.
For 48-well runs, the workflow fills the working solvent vial to the amount
required by the plan plus its reserve.
The default dilution factors are 0, 0.1, 0.25, 0.5 and 1. Each diluted
substock is 6 mL; for example, 0.25x uses 1.5 mL parent stock and 4.5 mL solvent.
Substocks are diluted directly from the parent stock with pure solvent and
vortexed; the undiluted level uses the parent vial. All aqueous wells therefore
have 2.5% solvent. The solvent reference is entirely native solvent.

The built-in parent-stock concentrations are pyrene 48.6 uM, Nile Red 50 uM,
and Coumarin-6 3 uM. Override `STOCK_CONCENTRATION_UM` when needed. A dilution
factor is relative to the parent stock; the final well concentration also
includes the 5/200 addition dilution. Thus factor 1.0 is 1.215 uM pyrene in
the well, not 48.6 uM. Set surfactant concentration and CMC
in mM to check that the final diluted surfactant remains above the supplied CMC.
The workflow uses a pre-made surfactant solution; it does not prepare this stock.

Each workflow row specifies one pair with `DYE` and `DYE_SOLVENT`.
`VIAL_NAME_OVERRIDES` optionally maps generated stock, substock and working
solvent names to the names already used in the inventory. For example,
`{DMSO_working: DMSO_rf_1}` selects that physical vial for blanks and native
solvent wells. It is separate from `SOLVENT_SOURCE_VIALS`; the working vial
cannot also be a preparation source in the same config. Empty overrides
preserve the generated names. Paired preparation/measurement jobs must use
the same mappings and recipes.
`SOLVENT_SOURCE_VIALS` is an ordered list of pure-solvent inventory names
(default `[DMSO]`); set it explicitly for the selected solvent. For example,
`[DMSO, DMSO_2, DMSO_3]` uses DMSO first, then DMSO_2, then DMSO_3.
Additions can span containers, retaining each source's reserve. The six-pair
queue puts open bulk ethyl acetate first during preparation, but excludes both
open bulk sources from measurement configs: later jobs can only refill from
the capped ethyl-acetate supplies, not evaporation-sensitive open containers.
`DYE_SOLVENT` remains the chemical identity used for
pipetting parameters and stock/working-vial names. Every source must contain
that same pure solvent. `SOLVENT_SOURCE_RESERVE_ML` (default 2 mL) is retained
in each container; increase it if the physical container requires more dead
volume. Starting demand is checked against the combined usable volume.
Scheduler simulations carry each container's updated volume between jobs.
Existing queued rows are private copies: reload/recreate them with both new
settings before simulating. Substock reuse is unchanged.

For example, `pyrene` and `DMSO` generate the prefix `pyrene_DMSO`;
`coumarin-6` and `acetone` generate `coumarin6_acetone`. The prefix is the
dye and solvent with punctuation removed, joined by an underscore. Each pair
generates its parent stock name
`<pair>_stock` and dilution vial names `<pair>_s<level>`. With
`FRESH_SUBSTOCKS: true`, it includes the repetition number, for example
`pyrene_DMSO_r2_s1`. Bulk solvent vial names come from `SOLVENT_SOURCE_VIALS`;
they need not match `DYE_SOLVENT`. `water` and `surfactant` remain fixed vial names. The
solvent-only medium and blank wells remain available through `TEST_SOLVENT`.
The example vial CSV contains both pairs in one inventory. Queue separate rows
with different `DYE` and `DYE_SOLVENT` values and the same CSV to run both pairs in order.

Set `PREPARE_ONLY: true` to run the existing startup solution refills and prepare
all required substock batches, including every repetition when
`FRESH_SUBSTOCKS: true`. Vials are returned home using the normal cap-handling
path and inventory is saved in live mode. No plates are acquired or dispensed,
no measurement waits or reads occur, and no fluorescence analysis is generated.
Preparation does not require loaded plates or an available Cytation protocol;
the config, well plan and recipes are still saved.
Queue preparation-only runs for each pair first, then run the same configurations
with `PREPARE_ONLY: false` (the default), using the same vial inventory.
Substocks with recorded volume strictly greater than 1 mL are reused in either
mode; 1 mL or less triggers the full preparation recipe. This assumes destinations
are empty or already correctly prepared, not partially filled, and does not verify
concentration. Keep recipes and vial identities unchanged between passes.
Working-solvent, water and surfactant refill behavior is unchanged in both modes;
a later measurement run can still require bulk solvent.

`REPLICATES` counts independently dispensed wells. `REPETITIONS` repeats the
whole experiment on new plates. `FRESH_SUBSTOCKS` creates distinct dilution vials
for each repetition; otherwise all repetitions share substocks.
`MEASUREMENT_SCHEDULE_MIN` lists elapsed minutes (since the plate is ready) at
which to read it for kinetics, e.g. `[0, 10, 20]`; it must start at 0 and each
gap must be at least `MIN_SCHEDULE_GAP_MIN` minutes. `MEASUREMENT_REPLICATES`
repeats reads back-to-back at each scheduled timepoint. Set
`SHAKE_BEFORE_MEASUREMENT: true` to run `SHAKE_PROTOCOL_FILE` once before the
first read at every timepoint, in the same reader visit. It defaults to `false`.
Replicate reads do not repeat shaking. The schedule still starts when dispensing
finishes; shake/wait duration adds to the actual read time recorded in the results.
Choose schedule gaps that accommodate shaking and all reads. The parent
stock remains shared even when fresh dilutions are prepared. `RANDOMIZED_ORDER`
randomizes well assignment reproducibly with `RANDOMIZATION_SEED`. Dispensing is
grouped by source; this option does not randomize chronological pipetting order.

`DISPENSE_ORDER` can be `[medium, dye]` or `[dye, medium]`. The selected pair's
solvent is passed as the dye pipetting liquid name. The aqueous surfactant pipetting liquid defaults
to water and can be changed with `SURFACTANT_LIQUID`.

Before hardware execution, update the example vial status CSV to the actual
loaded rack positions and volumes, including empty destinations named in the
generated substock recipes. The workflow checks that all generated vial names
exist before dispensing. Extra levels and fresh batches require extra vials.
When `TEST_WATER` is enabled, startup tops the water vial up to 8 mL from
reservoir 1, matching the surfactant workflow. It skips water already within
0.1 mL of full. This also replenishes water between queued workflow runs.
Working solvent requirements include a 2 mL reserve to match the small-tip low-volume warning threshold; each bulk source retains
`SOLVENT_SOURCE_RESERVE_ML`.
When `TEST_SURFACTANT` is enabled, startup tops the `surfactant` vial up to
7.5 mL from `surfactant_refill`, skipping it when within 0.1 mL of that target.
Load the same premixed surfactant solution in this 20 mL source vial at
`large_vial_rack[2]`; the example starting volume is 19.9 mL. The workflow checks
the refill source has enough liquid before starting any transfers and uses
`SURFACTANT_LIQUID` for the refill's pipetting parameters.
The example inventory supplies both dye/solvent pairs with 19.9 mL in each bulk
solvent vial. Each solvent has an additional empty closed-cap 8 mL working vial:
`DMSO_working` at `main_8mL_rack[27]` and `acetone_working` at `[28]`.
The working vial name is the actual `DYE_SOLVENT` value plus `_working`.
The workflow fills the selected working vial to 6 mL from the configured bulk
sources before dispensing; substocks use the same ordered pool. All solvent
additions to the plate, including matched blanks and solvent-only medium,
use the selected working vial staged at the clamp. A resumed run skips filling a working
vial already holding enough for the planned demand plus 2 mL. Volume checks reject runs exceeding the
available bulk or working solvent, including that 2 mL working reserve. Plans needing
more than 6 mL of working solvent are rejected because demand plus reserve exceeds
the 8 mL vial capacity; no mid-run working-solvent refill is performed.
Replace example values with the actual loaded inventory.
Before plate dispensing, each 8 mL source vial is moved individually to
`clamp[0]` before each plate dispense. Keep the clamp empty and do not assign
it as a vial home. Tips are removed before staging;
serial dispensing removes the used tip and returns the source home before the
next source is staged. Large vials stay in place.
During substock preparation, vial-to-vial transfers involving at least one
closed cap stage movable source and destination vials at `main_8mL_rack[0]`
and `[1]`, respectively. Keep these staging slots free. Temporary home locations
keep clamp exchanges safe while a pipet is attached; original homes are restored
and staged vials returned after the transfer with the tip removed. Large sources
remain in place, and open/open transfers use the existing direct dispensing call.
North Safe continues to handle cap accessibility and collision checks.
Set total well volume for your plate. 48- and 96-well formats are supported;
24-well requires robot geometry and is rejected. Match the track configuration,
physical fluorescence plate, and Cytation protocols to the selected format.
Then set `SIMULATE: false`.

Pyrene uses the existing Ailsa `CMC_Fluorescence_96.prt` protocol and raw channels
`334_373` and `334_384`, preserving both absolute intensities. The proposed
355 nm excitation / 373 and 383 nm emission settings still need Cytation verification.
Coumarin-6 (485 +/-20 / 528 +/-20 nm) and Nile Red (550 / 648 nm) are placeholders
from the requested experiment. To enable either, supply `PROTOCOL_FILE` and
`RAW_CHANNELS` matching actual Cytation output. The workflow accepts additional
intensity channels for exported spectra. Set `SHAKE_PROTOCOL_FILE` to a matching
shake-only protocol; its default is `shake_5.prt` (described by the existing
kinetics workflow as a five-second shake). Verify its settings on the Cytation PC.
Each scheduled time starts a cycle: transfer, one shake, then measurement.
Targets stay relative to the end of dispensing, so a cycle at 0 minutes does
not push the 10-minute target later. If a cycle overruns the next target, the
next cycle starts immediately. `elapsed_actual_min` records completion after
the plate returns, not the precise acquisition time.

Outputs include the config, well map, dilution recipes, raw reader exports,
per-well intensities, and mean/SD/count summaries with medium-specific blank
correction. Repeat reads are averaged within wells before computing replicate
statistics. Calibration fits are separate for each medium, repetition and channel,
with slope, intercept, R-squared, RMSE and tested concentration bounds.
Review linearity and choose the useful concentration range before applying
`concentration = (blank_corrected_intensity - intercept) / slope` to unknowns
measured under matching conditions. Fits are provisional and never extrapolate
or back-calculate unknowns automatically. Encapsulated concentration and relative
micelle volume require an additional model; this workflow provides calibration
measurements, not those estimates.

Tests: `python -m unittest discover -s tests -p test_fluorescence_calibration.py`
# Embedded pipetting validation

Set `VALIDATE_LIQUIDS: true` in the workflow config to run gravimetric
checks before dispensing, or set `VALIDATION_ONLY: true` to stop after the
checks. Sessions are separate for each source liquid and resolved tip. The
checked volumes are the minimum, middle, and maximum values used by each tip.
`liquid_type` is the
exact liquid registry name (`water`, `DMSO`, or `SURFACTANT_LIQUID`), while the
source vial identifies the physical container. Validation uses three replicates
per volume and may consume liquid. Simulations do not save validation data.

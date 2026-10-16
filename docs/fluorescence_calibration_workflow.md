# Fluorescence calibration

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
For example, `pyrene` and `DMSO` generate the prefix `pyrene_DMSO`;
`coumarin-6` and `acetone` generate `coumarin6_acetone`. The prefix is the
dye and solvent with punctuation removed, joined by an underscore. Each pair
generates its parent stock name
`<pair>_stock` and dilution vial names `<pair>_s<level>`. With
`FRESH_SUBSTOCKS: true`, it includes the repetition number, for example
`pyrene_DMSO_r2_s1`. The solvent
vial is named by the row's actual `DYE_SOLVENT` value (`DMSO` or `acetone`), not a
literal `solvent` vial. `water` and `surfactant` remain fixed vial names. The
solvent-only medium and blank wells remain available through `TEST_SOLVENT`.
The example vial CSV contains both pairs in one inventory. Queue separate rows
with different `DYE` and `DYE_SOLVENT` values and the same CSV to run both pairs in order.

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
Source requirements include a 0.1 mL reserve; there is no automatic refill.
The example inventory supplies 8 mL parent stock and 20 mL solvent in a 20 mL
vial to cover these larger preparations plus plate dispensing. Replace these
example values with the actual loaded inventory.
Before plate dispensing, each 8 mL source vial is moved individually to
`clamp[0]` before each plate dispense. Keep the clamp empty and do not assign
it as a vial home. Tips are removed before staging;
serial dispensing removes the used tip and returns the source home before the
next source is staged. Large vials stay in place. Substock preparation retains
the existing vial-to-vial transfer behavior. The robot's cap accessibility check
still applies; use the example open-cap setup for dispensing at the staged position.
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

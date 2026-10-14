# Fluorescence calibration

Run from the repository root:

```powershell
python -m workflows.fluorescence_calibration_workflow
```

The first run creates `workflow_configs/fluorescence_calibration_workflow.yaml`.
The default `SIMULATE: true` writes plans only, without importing hardware drivers
or fabricating fluorescence measurements. Config files follow the repository's
existing convention and are locally ignored by Git.

The default experiment uses 36 wells: three media (water, premixed surfactant,
native dye solvent), three dye levels plus a matched solvent-only blank, each
in triplicate. Each well receives 195 uL medium and 5 uL dye substock or solvent.
Each diluted substock is 6 mL: 1.5 mL parent stock + 4.5 mL solvent for 0.25x,
or 3 mL parent stock + 3 mL solvent for 0.5x.
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

`REPLICATES` counts independently dispensed wells. `REPETITIONS` repeats the
whole experiment on new plates. `FRESH_SUBSTOCKS` creates distinct dilution vials
for each repetition; otherwise all repetitions share substocks.
`MEASUREMENT_SCHEDULE_MIN` lists elapsed minutes (since the plate is ready) at
which to read it for kinetics, e.g. `[0, 10, 20]`; it must start at 0 and each
gap must be at least `MIN_SCHEDULE_GAP_MIN` minutes. `MEASUREMENT_REPLICATES`
repeats reads back-to-back at each scheduled timepoint. There is no shake/wait
step between dispensing and the first read; add one back if needed. The parent
stock remains shared even when fresh dilutions are prepared. `RANDOMIZED_ORDER`
randomizes well assignment reproducibly with `RANDOMIZATION_SEED`. Dispensing is
grouped by source; this option does not randomize chronological pipetting order.

`DISPENSE_ORDER` can be `[medium, dye]` or `[dye, medium]`. Set `DYE_SOLVENT` to the
robot's calibrated liquid name. The aqueous surfactant pipetting liquid defaults
to water and can be changed with `SURFACTANT_LIQUID`.

Before hardware execution, update the example vial status CSV to the actual
loaded rack positions and volumes, including empty destinations named in the
generated substock recipes. Extra levels and fresh batches require extra vials.
Source requirements include a 0.1 mL reserve; there is no automatic refill.
The example inventory supplies 7.8 mL parent stock and 19.9 mL solvent in a 20 mL
vial. Load an additional empty closed-cap 8 mL vial named `solvent_working` at
`main_8mL_rack[27]`. The workflow fills it with 6 mL from the bulk `solvent`
source before dispensing; substocks still use the bulk source. All solvent
additions to the plate, including matched blanks and solvent-only medium,
use `solvent_working` staged at the clamp. A resumed run skips filling a working
vial already holding at least 5.9 mL. Volume checks reject runs exceeding the
available bulk or working solvent, including a 0.1 mL reserve.
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
mix/equilibration protocol; its default is Ailsa's `shake_5_wait_5.prt`.

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

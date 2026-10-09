# Pipette calibration workflow

Run from the repository root:

```powershell
python -m workflows.pipette_calibration_workflow
```

Single-vial pipette calibration followed by independent validation, without
automatically promoting the calibration result into shared use. Launch goes
through `execute(config=None, show_gui=True)`.

Set `LIQUID`/`TARGET_VIAL` for the liquid and source vial, and
`VOLUME_TARGETS_ML` for the calibration volumes (default `[0.02, 0.01, 0.005]`
mL). `CALIBRATION_CONFIG_FILE` points at the shared
`sdl_pipette_calibration/experiment_config.yaml`. Screening runs up to
`NUM_SCREENING_TRIALS` trials with up to `MAX_REPLICATES_PER_TRIAL` replicates
each, bounded by `MAX_TOTAL_MEASUREMENTS` and
`MAX_MEASUREMENTS_FIRST_VOLUME`; `MIN_GOOD_TRIALS` is the minimum accepted
trial count before two-point calibration
(`TWO_POINT_CALIBRATION_REPLICATES` replicates) runs. `QUALITY_STD_THRESHOLD_G`
is the mass-reading noise cutoff used to accept a trial.

After calibration, `VALIDATION_VOLUMES_ML` are each dispensed
`REPLICATES_PER_VOLUME` times to confirm accuracy; `ADJUST_VOLUME` allows the
validation step to correct target volumes based on calibration results, and
`CONTINUOUS_MONITORING` keeps checking accuracy across validation reps rather
than only at the end.

Each hardware parameter in `_HARDWARE_PARAMETERS` (aspirate/dispense speed,
aspirate wait time, pre-aspirate air volume, blowout volume, aspirate/dispense
cycles, post-aspirate air volume, post-retract wait time, retract speed,
dispense wait time) has `<NAME>_FIXED`, `<NAME>_FIXED_VALUE`, `<NAME>_MIN`, and
`<NAME>_MAX` config entries: set `_FIXED=True` to hold a parameter at its fixed
value, or `False` to let calibration search between `_MIN` and `_MAX`.
`OVERASPIRATE_VOL_MIN`/`MAX`/`MAX_FRACTION_OF_TARGET` bound how much extra
volume calibration may aspirate relative to the target. `RANDOM_SEED` makes
trial selection reproducible.

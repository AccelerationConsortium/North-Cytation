# Recording routine pipette dispenses

Routine recording observes how well the pipette dispenses during a workflow and
saves that evidence in the unified measurement CSV. It is disabled by default.

Enable it when constructing the coordinator:

```python
lash_e = Lash_E(..., dispense_recording="auto")
```

Direct `North_Robot` construction accepts the same trailing option. Supported values
are `"off"` and `"auto"`; invalid values fail before robot initialization.

## When a measurement is collected

A real pipette dispense qualifies when recording is enabled, `liquid=` explicitly
names a registered liquid with usable density, the nominal target is positive, and
the prepared destination is at the primary balance clamp (`clamp`, position `0`).

The normal accessibility helper runs first. Recording does not add vial movements,
uncapping, tip retraction, or a different destination. An accessible vial elsewhere
is dispensed into normally without a routine measurement. Unknown or omitted liquid,
other clamp positions, and simulation do not enable automatic weighing or recording.

Automatic weighing uses the existing continuous weighing path, including the usual
1 s settling, 2 s pre-baseline, 2 s post-baseline, and 0.2 s polling cadence. Its
existing stable-reading means and time-based fallback are preserved. When explicit
weighing is already requested, recording uses those results without another set of
balance reads or waits; traditional weighing remains traditional.

## What gets saved

Each physical pipette dispense, including each chunk of a larger transfer, adds one
row to `pipetting_data/master_pipetting_measurements.csv` through the existing writer.

Rows reuse the current schema and units: nominal target and measured volume in mL,
mass difference in g, density, signed percentage error, source/destination, actual
settings, and stability information. Aspiration settings are captured after vial
aspiration completes, including resolved speed defaults. They are cleared for a new
aspiration, raw pipette aspiration, tip disposal, or reset. Unknown aspiration
settings stay empty. The effective dispense speed is recorded too.

`validation_type="vial_to_vial"` identifies the method; unique `routine_` session IDs
identify routine provenance. Tip type is included in the existing `stability_info`
JSON. Calibration stage and selection fields remain empty. The records remain in
the shared dataset for analysis and reflection.

Measured volume is mass difference divided by registered density. Missing readings,
insufficient pre/post data, and nonfinite mass are saved as unavailable rather than
as zero. A genuinely measured zero remains zero. Existing stability counts and
variability remain available for interpretation; a finite result is not a declaration
that the dispense passed a calibration tolerance.

## Execution and failures

Recording does not update calibration, compensation, nominal vial-volume accounting,
or workflow decisions. Existing return shapes and values are preserved: adding
automatic weighing to an otherwise unweighed call does not change its return result.
Explicit weighing and transfer-total calculation retain their previous contracts.

The existing polling loop logs balance-read failures. Missing routine measurements
are recorded as unavailable. New metadata-copy and CSV-save failures log warnings;
they do not pause or repeat delivery. No broad exception handlers wrap delivery,
existing weighing calculations, or cleanup. Existing weighing failures and
interruptions retain their previous handling, including thread-start errors.
Filesystem writes use the existing master-file writer; no queue or database
migration was introduced.

Mixing and buffer returns opt out. Embedded pipette validation and standalone pipette
calibration opt out because they already own their measurement records. Other special
callers can pass `record_measurement=False` to `dispense_into_vial` or
`dispense_from_vial_into_vial` without disabling their explicit weighing.

Routine recording adds no simulation records or balance traces. Existing explicitly
requested weighing behavior remains under its existing flags.

## Reservoir scope

Routine reservoir recording is deferred. Reservoir execution and calibration remain
as they were; enabling this option affects pipette additions only. The deeper
per-stroke alignment is described in
[the reservoir plan](../plans/reservoir_calibration_alignment.md).

Software validation uses fake hardware and temporary CSVs. No physical validation
run is part of this implementation.

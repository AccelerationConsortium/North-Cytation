# Recording routine clamp dispenses

Enable recording when constructing the coordinator:

```python
lash_e = Lash_E(..., dispense_recording="auto")
```

The default is `"off"`. The setting is passed to `North_Robot` and applies to
pipette and reservoir additions. It only records a supplied, registered `liquid`
when the destination is physically at `clamp`, position `0`, at dispense time.
`liquid=None`, an unavailable density, `clamp_2`, and other dispensing locations
do not produce routine records. Recording does not move a vial to qualify it;
normal dispense methods retain their existing movement behavior.

Use the existing `liquid=` argument on transfers or direct dispensing calls.
Recording does not guess a liquid from a vial name, a previous call, or a pump.

Each physical dispense produces one row in
`pipetting_data/master_pipetting_measurements.csv`. The existing column names
and units are reused, with `validation_type` distinguishing `vial_to_vial` and
`reservoir_dispensing`. Reservoir records use the existing reservoir schema's
`reservoir_index` field. The historical reservoir files are not migrated.
Future reservoir validation measurements and summaries use the pipetting master
files too.

The stable baseline means and pre/post collection windows match the existing
continuous weighing procedure. The measured volume is the mass increase divided
by the supplied liquid's registered density. Each chunk of a split pipette or
reservoir transfer is measured separately. Its target and compensation values
describe that chunk, not the combined transfer.
Reservoir measurements surround strokes in the existing pump loop. Recording
does not repeat carousel positioning, capping, or returning the vial home.

Routine rows use unique `routine_` session IDs. Calibration-specific stage and
decision fields stay empty. Weighing failure leaves mass, measured volume and
accuracy empty, sets `measurement_acceptable=False`, and stores a reason inside
the existing `stability_info` field. A measured zero remains zero. Missing or
unstable baselines are not replaced by zero or an unstable average. Recording
and weighing failures never pause or retry an ordinary physical dispense.
Existing calibration callers explicitly opt out of automatic recording. Their
numeric mass return is preserved: unavailable mass returns zero with the failure
reason in `stability_info`, so their existing quality checks and arithmetic keep
working. This compatibility return does not replace missing values in routine
measurement records.

Mixing and buffer returns are excluded. For another special operation, pass
`record_measurement=False` to `dispense_into_vial`,
`dispense_from_vial_into_vial`, or `dispense_into_vial_from_reservoir`.
Calibration callers use that opt-out because their existing writer already
saves the measurements, preserving calibration stages without duplicate rows.

Simulation never writes routine measurement records or balance traces, even
with explicit measurement and trace-saving flags. Standalone calibration also
suppresses simulation artifacts even if `output.save_files` is true.
These records never update
calibration parameters, volume compensation, or nominal vial-volume accounting.

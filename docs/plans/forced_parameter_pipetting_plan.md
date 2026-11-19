# Forced-parameter pipetting workflow — plan

Date: 2026-11-16 · Status: **draft for discussion, nothing implemented.**
North_Safe is about to be updated (refill/dispense functions not yet synced), so
every North_Safe call named below must be re-checked against the synced version
before implementation (see "Re-check after the North_Safe sync").

## Goal

An idle-time data-gathering workflow. Given a list of rows, each naming a
**liquid**, a **volume** and a full **forced parameter set**, pipette that liquid with exactly those parameters (no
calibrated values, no corrections, no defaults filling gaps), weigh every dispense,
convert mass to volume with the liquid's density, and save each measurement the
moment it is taken. It can be stopped at any instant and resumed later without
losing or duplicating data.

It replaces the archived `workflows/archive/glycerol_dispense_baseline.py`
(Sobol campaign for glycerol), generalised to any registered liquid and its
handling properties.

Proposed file: `workflows/forced_parameter_pipetting.py` + config
`workflow_configs/forced_parameter_pipetting.yaml` (scheduler Setup convention:
`INPUT_VIAL_STATUS_FILE` pointing at an existing vial CSV).

## What exists today (checked 2026-11-16)

- **Archived glycerol workflow**: reads a Sobol CSV, one row = one tip/measurement;
  aspirates and dispenses back into the **same vial on the clamp/scale**, passes
  `liquid=None` so calibration can't override; saves an `incremental_results.csv`
  row after each measurement and resumes from the count of `status == ok` rows;
  copies mass traces; Slack every 12 tips; vial swap on `input()`; custom SIGINT
  handler. Hard-coded glycerol density 1.26 and bottle-opened date.
- **Parameter resolution** (`North_Safe._get_optimized_parameters`): defaults →
  liquid calibration (only if `liquid` given) → user overrides. A user
  `PipettingParameters` object overrides every attribute that is **not None**.
  `blowout_speed` defaults to `None`, so with `liquid` set it would silently keep
  the calibrated value. `resolve_pipetting_setup` also chooses the tip unless
  `specified_tip` is given.
- **Liquid properties**: `robot_state/liquids.yaml` via
  `pipetting_data/liquid_registry.get_liquid()` → `density`, `refill_pipets`.
- **Conditioning / tip switching** (calibration protocol,
  `sdl_pipette_calibration/protocols/calibration_protocol_northrobot.py:105-127`):
  if `refill_pipets` is false, a new tip is conditioned with 4 aspirate/dispense
  cycles of `get_tip_conditioning_volume(max target)` (1.2× target, capped at tip
  size) using fixed conditioning parameters, then waits for scale stability;
  if `refill_pipets` is true, the tip is removed after every measurement and no
  conditioning is done. The same logic is duplicated in
  `workflows/pipette_calibration_workflow.py:277-289` and `:368-381`.
- **Volume tracking correction** after a same-vial measurement, with the noise
  guards from the 2026-11-10 incident (clamp to ≤ pre-measurement volume and vial
  capacity): calibration protocol `:623-668`.
- **Unified measurement database**: `pipetting_data/master_pipetting_measurements.csv`
  (columns include `session_id, validation_type, liquid_type, density_used,
  target_volume_ml, measured_volume_ml, accuracy_pct`, all parameters,
  `mass_difference_g, stability_info, tip_type, temp_c, humidity_pct, pressure_pa`).
  `pipetting_data/dispense_recording.record_dispense` writes routine rows there.
  The master writer rewrites the whole file on every append.
- **Scheduler stop** (`scheduler/interrupt.py`): touching `interrupt.requested`
  raises `KeyboardInterrupt` in the workflow at any point (mid-aspirate included).
  Workflows catch it and call `workflows/helpers/interrupt_cleanup.cleanup_after_keyboard_interrupt`,
  which returns liquid in the tip to its source vial, drops the tip, returns a
  held vial/cap, homes.

## Inputs

| Name | Meaning |
|---|---|
| `SOURCE_VIALS` | Per liquid, an ordered list of vial names holding it; used in turn as each runs low. Every liquid in the file must have an entry. |
| `PARAMETER_FILE` | CSV, one row per measurement condition (see below). |
| `REPLICATES` | Measurements per parameter set (default 1, as in the Sobol campaign). |
| `CAMPAIGN_ID` | Names the output folder; same ID = resume, new ID = fresh campaign. |
| `VIAL_LOW_THRESHOLD_ML` | Switch source vial below this volume. |
| `MAX_ROWS_FOR_SIMULATION` | Rows processed in simulation. |

**Parameter file columns (all required, no blanks):** `liquid` (exact key in
`liquids.yaml`; gives density and `refill_pipets`), `volume_ul, aspirate_speed, dispense_speed, retract_speed, blowout_speed, aspirate_wait_time,
dispense_wait_time, post_retract_wait_time, overaspirate_vol_ul, pre_asp_air_vol_ul,
post_asp_air_vol_ul, blowout_vol_ul, asp_disp_cycles`. uL in the file, mL to the
robot (as the glycerol workflow did). Optional `param_set_id`; otherwise the row
number is the ID. Extra columns are copied into the output untouched.

Making every field required is what makes "override all parameters" true: there is
no field left for a default or a calibration to fill.

**Tip type is not an input** (decided 2026-11-16): it is determined from the row's
volume and forced parameters, using North_Safe's own rule in
`resolve_pipetting_setup` with `liquid=None` — smallest tip with
`volume ≥ min_suggested_volume` and load `volume + overaspirate + post_asp_air`
below capacity (preferring tips outside their capacity margin). The chosen tip is
then passed as `specified_tip` so it can't change between planning and execution.

## How a parameter set is forced (the key guarantee)

1. Build a `PipettingParameters` with **every** field set from the row.
2. Call `aspirate_from_vial(..., parameters=p, liquid=None, specified_tip=tip)`
   and `dispense_into_vial(..., parameters=p, liquid=None, measure_weight=True,
   continuous_mass_monitoring=True, record_measurement=False)`.
   `liquid=None` is the glycerol precedent: the calibration branch is never entered,
   so nothing (calibrated values, overaspirate corrections, compensation) can leak in.
   The liquid name is used only by this workflow for density, tip policy and labels.
3. **Echo check** before the first aspirate of each set: call
   `resolve_pipetting_setup(volume, None, p, tip)` and compare the returned
   parameters field by field with the row. Any difference → skip the set,
   log it as `rejected_override_mismatch`, carry on. This is a test that the
   override is really total, run against the live North_Safe.
4. Store the parameters **actually used** (from the echo) with each measurement.

## Validation before any hardware moves

Run on the whole file at start-up; bad rows are listed and skipped, not fatal:
- all columns present, numeric, finite, non-negative; speeds/cycles integers;
- some tip fits the row (tip rule above); a row no tip fits is rejected;
- every row's `liquid` in the registry and in `SOURCE_VIALS`; each source vial
  exists in the vial file, and its tracked volume is enough for at least one
  conditioning + one measurement;
- conditioning volume (1.2× largest volume in the file, capped per tip) fits.
Fail-fast only if **no** row is valid or the liquid/vials are unusable.

## Execution loop

**Tip grouping (minimise tips).** Pending measurements are sorted into *tip
groups* before running, and the order is recomputed from what's pending on every
start/resume:
- a group = one liquid + one tip type (the tip and the source vial on the clamp
  are shared by the whole group);
- `refill_pipets: true` liquids: every measurement is its own tip (no grouping
  possible, no conditioning) — calibration behaviour;
- `refill_pipets: false` liquids: one tip per group, conditioned once with the
  calibration's conditioning routine (sized to the group's largest volume), then
  reused across all rows in the group, whatever their volumes and parameters.
  No per-tip use limit for now (decided 2026-11-16);
- groups run liquid by liquid, so each source vial goes to the clamp once.
Tips used ≈ (refill-liquid measurements) + (number of non-refill groups)
+ one per resume after an interrupt.
The conditioning routine should be moved into one shared helper used by the
calibration protocol, the calibration workflow and this workflow, not copied a
fourth time.

```
setup: home, load campaign state, build tip groups from pending rows
for each tip group:
    move that liquid's source vial to clamp 0, uncap if needed (if not already there)
    for each pending (row, replicate) in the group:
        if stop requested -> KeyboardInterrupt path below
        ensure source vial has volume, else swap (see Vial handling)
        ensure tip (new tip -> condition if refill_pipets is false)
        aspirate + dispense into same vial on scale with forced params
        measured_ul = mass / row density
        save measurement to campaign file    <- unit of work ends here
        correct source volume from mass (with tracker guards)
        if refill_pipets or tip hit its max uses: remove tip
        fault check (below); Slack progress every N
    remove tip; recap and return vial home when the liquid changes
teardown: remove tip, recap, return vial home, home, final Slack
```

After an interrupt the held tip is dropped, so on resume the group restarts with a
fresh (conditioned) tip.

**Tip budget.** Read remaining tips per type from `robot_status.yaml` /
`pipet_racks.yaml` (as the glycerol workflow did). When a needed tip type runs out,
use the new North_Safe tip-refill flow if one exists after the sync; otherwise stop
cleanly with a Slack message (it's idle-time work; stopping is fine).

## Saving and resuming

Unit of work = one measurement. A row is written only after the dispense and
weighing finish; anything interrupted earlier is simply redone on resume.

1. **Campaign file** `output/forced_parameter_pipetting/<CAMPAIGN_ID>/measurements.csv`:
   append one line, flush, fsync. Columns: `param_set_id, replicate, timestamp,
   liquid, density, source_vial, tip_type, new_tip, conditioned`, every parameter
   actually used, `target_ul, mass_g, measured_ul, accuracy_pct, elapsed_s,
   stability metrics, temp_c, humidity_pct, pressure_pa, mass_trace_file, status`.
   This file is the resume source of truth: pending work = file rows × replicates
   minus `(param_set_id, replicate)` pairs already present with status `ok`.
2. **Not merged into the master database for now** (decided 2026-11-16). The
   campaign file is its own dataset. To keep a later merge easy, its columns use
   the master schema's names and units where they overlap (`liquid_type,
   density_used, target_volume_ml, measured_volume_ml, accuracy_pct,
   mass_difference_g`, parameter names, `tip_type`, `temp_c`, ...), with a planned
   `validation_type = forced_parameters` tag. Merging later is then a copy.
3. A `campaign.yaml` beside it snapshots the inputs (parameter file hash, source
   vials, start time). Resuming with a changed parameter file → warn and refuse to
   mix (decision 2).

Simulation: writes to `<CAMPAIGN_ID>_simulate/`.

## Interruption

- Rely on the scheduler's `KeyboardInterrupt`; no custom SIGINT handler (the
  glycerol one called `sys.exit` from a signal handler).
- On interrupt: `cleanup_after_keyboard_interrupt(lash_e)` (returns tip liquid to
  the source vial — correct here since source = destination), then recap the clamp
  vial and return it home if the helper didn't. Save nothing for the partial step.
- Per-measurement state is never kept only in memory; everything needed to resume
  is in the campaign file.

## Vial handling

- After each measurement, correct the source vial's tracked volume from the mass
  change, using the same guarded correction as the calibration protocol (reuse it;
  don't copy it). Never loosen North_Safe capacity checks.
- Below `VIAL_LOW_THRESHOLD_ML`: return vial home, take the next `SOURCE_VIALS`
  entry. If none left: use the new North_Safe refill function if it applies to
  this liquid (after sync), otherwise Slack "source empty" and stop cleanly.
  No `input()` waits.

## Faults (per the no-crash rule)

- Off-target volumes are the data — never warn on accuracy.
- Unstable baseline: record the measurement with its stability info and a
  `measurement_acceptable` flag; no automatic retry (decision 1).
- "Clearly broken" — e.g. 3 consecutive measurements < 10 % of target (empty vial,
  no tip, clogged) → stop the campaign cleanly with Slack. Everything saved stays.
- Balance read failure on one measurement → save as `status=measurement_failed`,
  continue; repeated failures → stop with Slack.
- Physical robot errors propagate as today.

## Environment data

Read the latest MQTT temperature/humidity/pressure per measurement (as the
glycerol workflow did) and store it. Stale or missing data → warning in the log
and Slack at start, never a block or `input()` prompt.

## Tests (mocked; named files only)

`software_tests/test_forced_parameter_pipetting.py`:
- every parameter column required; a missing value rejects the row;
- echo check: with a fake calibrated liquid present, resolved params equal the row
  exactly (including `blowout_speed`); a forced mismatch rejects the set;
- resume: kill at each step (before aspirate, between aspirate and dispense, after
  dispense before save, after save) → no duplicate and no lost measurement;
- tip grouping: mixed liquids/tip types → expected groups, tip count and order;
  refill_pipets true/false → tip removal and conditioning counts; max uses per tip
  splits a group; resume rebuilds groups from pending rows only;
- vial swap and empty-source stop; tip-out stop; "clearly broken" stop;
- the master measurement file is never written.

## Re-check after the North_Safe sync

- signatures/returns of `aspirate_from_vial`, `dispense_into_vial`
  (`measure_weight`, `record_measurement`, return tuple), `resolve_pipetting_setup`;
- whether `liquid=None` still bypasses every correction path;
- new refill / tip-refill / dispensing functions to use for vial and tip run-out;
- whether the new code adds its own routine recording that would double-write.

## Decided (2026-11-16)

- Every row specifies liquid, volume and all parameters; one file can mix liquids.
- Rows are sorted into tip groups (liquid + tip type) to minimise tip use; no
  per-tip use limit for now.
- Tip type is derived from volume and parameters, not given in the file.
- The workflow keeps its own dataset; no merge into the master database for now.

## Deferred

- Order and shuffling within a tip group — to discuss later. Until then: file order.

## Open decisions

1. **Unstable measurement**: record and move on (plan), or retry up to N times
   like the calibration protocol?
2. **Changed parameter file on resume**: refuse (plan), or continue matching by
   `param_set_id`?
3. Does "conditioning" need to be configurable per liquid (cycles, volume,
   parameters) in `liquids.yaml`, or keep the calibration's fixed routine?

## Implementation notes (current syntax, checked 2026-11-18)

`robot` = `lash_e.nr_robot`. Volumes are mL (convert from the file's uL). No flow fits this
workflow, so it uses unit operations directly.

| Step | Call |
|---|---|
| Parameters | `PipettingParameters(aspirate_speed=..., dispense_speed=..., retract_speed=..., blowout_speed=..., aspirate_wait_time=..., dispense_wait_time=..., post_retract_wait_time=..., overaspirate_vol=..., pre_asp_air_vol=..., post_asp_air_vol=..., blowout_vol=..., asp_disp_cycles=...)` (`pipetting_data/pipetting_parameters.py`). Set every field: `blowout_speed` defaults to `None` |
| Tip choice + echo check | `params_used, tip = robot.resolve_pipetting_setup(volume_mL, None, p)`; pass `tip` as `specified_tip` from then on; compare `params_used` with `p` |
| Vial to clamp | `robot.move_vial_to_location(vial, "clamp", 0)`, `robot.uncap_clamp_vial()`; at the end `robot.recap_clamp_vial()`, `robot.return_vial_home(vial)` |
| Tips | `robot.get_pipet(tip)`, `robot.remove_pipet()` |
| Conditioning | `robot.condition_tip(vial, volume_per_cycle_mL=..., cycles=4, specified_tip=tip)`. **This now exists in North_Safe**, so the "shared helper" step in this plan is done |
| Measure | `robot.aspirate_from_vial(vial, v, parameters=p, liquid=None, specified_tip=tip)` then `robot.dispense_into_vial(vial, v, parameters=p, liquid=None, measure_weight=True, continuous_mass_monitoring=True, save_mass_data=True, record_measurement=False)`; returns `(mass_g, stability_info)` |
| Liquid info | `liquid_registry.get_liquid(name)` gives `density` and `refill_pipets` (the flows use `condition_tip`) |
| Stop / crash | `cleanup_after_keyboard_interrupt(lash_e)` |
| Output folder | `lash_e.save_run_info(output_dir)` |

- `liquid=None` keeps calibration out, but sample tracking then has no ±. That's acceptable for this data-gathering run.
- Check whether `dispense_into_vial` still turns weighing on by itself when a density is known (`routine_only` near `North_Safe.py:2937`).

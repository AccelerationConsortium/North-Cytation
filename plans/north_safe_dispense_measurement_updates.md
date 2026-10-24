# Restore routine dispense measurements in North Safe

## Purpose

Reimplement the reverted North Safe integration for optional mass measurement and recording of routine pipette and reservoir dispenses. This document is a plan only; it does not authorize running hardware or restoring code automatically.

## Current state and scope

As inspected on 2026-10-24, `pipetting_data/dispense_recording.py`, `docs/dispense_recording.md`, and `tests/test_dispense_recording.py` remain in the workspace. `Lash_E` accepts `dispense_recording="off"`, but North Safe currently lacks the corresponding recording integration. Treat these remaining artifacts as implementation references and intended behavior, and reconcile them against actual callers before reuse.

Keep the current gripper tracing, release motions, timing, cap handling, staging, transfer planning, and vial inventory changes intact. The reported capped-vial incident is a separate investigation. Measurement integration must not change which vial is moved, when it is uncapped, or where a tip enters.

## Required behavior

- Default recording to `off`; support explicit `auto` through `Lash_E` and `North_Robot`.
- Automatically measure only real dispenses into the primary balance clamp (`location="clamp"`, index `0`) with an explicitly supplied, registered liquid and usable density.
- Evaluate eligibility after existing destination preparation. Recording must never move a vial just to make it eligible.
- Record one row per physical dispensing chunk, using the existing master measurement schema and units. Keep routine rows distinguishable from calibration rows with `routine_` session IDs and the existing validation type fields.
- Calculate measured volume as mass increase divided by liquid density. Store each chunk's nominal target and effective parameters, including its compensation contribution.
- Preserve nominal vial-volume accounting and existing calibration/compensation decisions; measured results are observational.
- Preserve existing public return shapes and numeric contracts for explicit calibration weighing. Missing routine measurements remain missing in storage; measured zero remains zero.
- Optional balance or recording failures must not pause, retry, or repeat liquid delivery. Physical motion or delivery failures must still propagate normally.
- Simulation must not poll the physical balance or save routine records or balance traces, even when measurement flags are explicitly enabled.

## Implementation sequence

### 1. Establish compatibility and failing tests

Inventory the signatures and return contracts of `aspirate_from_vial`, `dispense_into_vial`, `dispense_from_vial_into_vial`, and `dispense_into_vial_from_reservoir`. Check callers in embedded validation, standalone calibration protocols, mixing, and cleanup/buffer-return paths.

Adapt the remaining recording tests to the current controller. First capture the existing hardware command order with recording disabled, including source/destination clamp exchanges, cap ownership, tip removal, pump strokes, carousel movements, and return-home behavior. These traces become the comparison for recording enabled. Use the real controller methods with hardware and storage redirected to fakes and temporary files.

### 2. Restore configuration plumbing

Add the optional recording argument to `North_Robot` without changing existing positional argument meanings. Validate `off`/`auto` before hardware initialization and pass the coordinator's setting through when constructing the robot. Use `off` as the fallback for older or partially constructed controller objects.

Add `record_measurement=True` as an optional trailing argument on the relevant dispense methods. Forward it through vial-to-vial transfers to each destination dispense. This controls routine recording independently of existing explicit weighing flags.

### 3. Restore measurement helpers and lifecycle

Review and reuse `recording_density`, `analyze_baselines`, `read_steady_mass`, and `record_dispense` from `pipetting_data/dispense_recording.py`. Require a finite, positive density. Preserve the current collection windows and sampling cadence; require finite stable readings in both pre- and post-dispense baselines for a valid routine measurement. Never substitute an unstable average or zero for missing data.

Give each measurement an explicit lifecycle: pre-baseline, delivery, post-baseline, analysis, optional persistence. Stop polling in `finally`, including on `KeyboardInterrupt`, and use bounded thread shutdown. Ensure polling cannot continue into the next physical operation or overwrite the original exception. Keep optional measurement errors isolated from hardware delivery errors. Confirm SDK transport serialization before enabling background scale reads alongside motion commands.

Reuse a single set of readings when explicit weighing and automatic recording are both enabled. Avoid duplicate polling threads, baseline windows, and physical dispenses.

### 4. Integrate pipette dispensing and provenance

Wrap the existing delivery in `dispense_into_vial` with measurement after accessibility and tip positioning are established. Collect the post-baseline after the existing tip retraction and before recap or vial movement can alter balance load. Do not reorder the existing physical commands.

Capture source identity and effective aspiration parameters when aspiration actually succeeds. Make that context available through the matching dispense, including split transfers. Clear or replace it on tip disposal, reset, and unrelated aspiration so records cannot inherit stale source/settings. Unknown aspiration settings stay empty rather than being copied from unrelated dispense defaults.

Record the destination, liquid, density, target, mass difference, volume, signed accuracy, stability information, tip type, and actual resolved parameters. Preserve legacy explicit weighing return behavior. If any chunk lacks a valid mass, do not report the remaining chunks' partial sum as a valid total; return failure information through the compatible contract.

### 5. Integrate reservoir dispensing inside its existing loop

Keep the existing move-to-clamp, uncap, carousel positioning, pump-loop, carousel reset, recap, and return-home sequence. Surround each existing delivery stroke with its measurement window; do not recursively invoke the complete reservoir method per chunk.

Preserve the exact number of pump strokes and total commanded volume. For equal chunks, allocate nominal target and total overaspirate proportionally so their sums equal the original request and compensation. Record the existing `reservoir_index` field and actual reservoir parameters. Finish each post-baseline before subsequent movements affect the load.

### 6. Exclude duplicate and non-routine records

Pass `record_measurement=False` from calibration paths that already persist their own measurements, while retaining their explicit weighing. Exclude mixing, conditioning, and liquid-return operations. Review all call sites rather than relying only on liquid omission or a single helper test.

Reconcile the remaining reservoir master-file consolidation with existing validation writers and readers. Future reservoir validation rows and summaries should use the shared pipetting masters as described in the remaining documentation, preserving existing schema fields and calibration stages. Do not migrate historical files as part of this change.

### 7. Validate and document

Run the focused recording suite in the project's Python environment with its dependencies installed:

```powershell
python -m pytest tests/test_dispense_recording.py tests/test_vial_transfer_chunk_calibration.py tests/test_alternating_vial_transfer.py tests/test_validation_decisions.py tests/test_calibration_simulation_mode.py tests/test_pipette_calibration_workflow.py tests/test_fluorescence_calibration.py -q -p no:cacheprovider
```

The shell's default Python currently lacks `pytest`; select the project environment before using this command. Do not run hardware scripts as tests.

Cover eligible/ineligible destinations, unknown liquid/density, true zero, unstable/missing baselines, balance exceptions, disk-write failure, split pipette/reservoir transfers, parameter provenance, calibration opt-out, mixing exclusion, interruption cleanup, and simulation with explicit save flags. Include a polling test with a fake balance and deterministic synchronization.

Compare recording-off and recording-on hardware traces for successful and failed weighing. Only balance reads and baseline waiting may be added; motion commands, pump totals, cap transitions, and final robot state must match. Include closed-cap fluorescence transfers through the real accessibility helper so shallow mocks cannot hide clamp-exchange regressions.

Update `docs/dispense_recording.md` and `CHANGELOG.md` to describe the final implemented behavior. Keep recording disabled by default. Any later hardware validation should be a separately requested run with a known liquid and verified vial/cap state.

## Completion criteria

The focused suites pass, every eligible delivery produces exactly one correctly attributed row per physical chunk, unavailable mass is distinguishable from zero, calibration records are not duplicated, simulation produces no measurement artifacts, and recording does not alter the physical command sequence or mask delivery failures.

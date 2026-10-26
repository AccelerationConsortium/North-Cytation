# Restore routine dispense measurements in North Safe

## Purpose

Reimplement the reverted North Safe integration for optional mass measurement and recording of routine pipette and reservoir dispenses. This document is a plan only; it does not authorize running hardware or restoring code automatically.

## Agreed constraints and implementation review (2026-10-26)

Status: Pipette-only recording implemented and narrowed on 2026-10-26. The broad
weighing error wrappers and new cleanup restructuring were removed. Focused checks
pass in the existing `north_env` environment (81 passed). Reservoir changes remain
deferred to the separate alignment plan. No hardware run has been performed.

Current weighing control flow and error handling are preserved. New exception
handling is confined to recording metadata and persistence helpers; thread-start
or analysis errors retain their existing behavior. See the local
[reduced patch review](../output/recording_patch_review/REVIEW.md) for the exact
recording-only controller diff and verification details.

The implemented scope is the existing pipette weighing path plus optional recording
to the unified CSV, with existing mass calculations and return contracts preserved.
The broader reservoir proposals below are design history, not implemented behavior.
See [the recording documentation](../docs/dispense_recording.md) for current usage.

Reservoir follow-up: [Align reservoir transfers and calibration with pipetting](reservoir_calibration_alignment.md)
now owns the deeper changes to reservoir splitting, per-stroke calibration,
compensation, measurement timing, and aggregation. Reservoir proposals below remain
subject to that separate review; do not implement those execution changes as an
incidental part of adding recording.

User clarifications: measure only when convenient within the existing execution;
retain the working weighing positions, waits, balance polling, and mass calculation.
Do not redesign weighing or introduce stricter acceptance rules in this change.
Measurement and save errors must not interrupt routine workflows. Capture settings
actually used. Prefer existing CSV columns for identifying routine provenance.
Per-stroke reservoir measurements are preferred, but their implementation and any
additional baseline waits remain to be discussed. The controller currently weighs
the complete reservoir call; it does not sum independently measured strokes.

Persistence recommendation: initially reuse the existing master-file writer,
with save errors isolated from workflow execution. Measure save duration before
proposing a different writer, queue, or database format. This recommendation is
not a request to redesign persistence during the recording implementation.

Routine measurements belong in the unified measurement database,
`pipetting_data/master_pipetting_measurements.csv`. They are evidence for analysis
and reflection alongside calibration measurements. Do not create a separate routine
measurement database or automatically exclude routine data from analysis. Preserve
enough context to distinguish routine use from a controlled calibration experiment.

The user accepts the usual measurement waits already implemented. Preserve robot
movements, their order, pump strokes and volumes, speeds, tip selection, cap handling,
transfer splitting, nominal inventory accounting, calibration choices, and existing
return values. Add measurement collection and persistence only. Do not silently add
new per-stroke settling windows or refactor existing liquid handling.

Findings from the current code:

| Area | What exists or could change | Scope for this implementation |
| --- | --- | --- |
| Configuration | `Lash_E` accepts the recording setting, but does not pass it to `North_Robot`; the robot lacks the integration. | Add opt-in configuration without changing existing defaults or positional arguments. |
| Pipette timing | Existing continuous weighing adds 1 s settling, 2 s pre-baseline and 2 s post-baseline, plus thread shutdown and processing time. | Use the usual measurement windows; avoid duplicate waits when explicit weighing is already requested. |
| Tip position | `dispense_into_vial` collects its existing post-baseline after `pipet_dispense`; neither method provides the tip retraction assumed by the earlier plan. | Correct the plan only. Keep the working measurement position and settings. |
| Reservoir timing | Existing continuous weighing surrounds the complete pump loop. Separate baselines per stroke would add waits between strokes. | Per-stroke recording is preferred. Discuss how to obtain each stroke's mass without refactoring the physical loop or silently adding baseline waits. |
| Calibration results | Existing explicit weighing falls back to time-based baselines, and transfer totals treat missing increments as zero. | Preserve working mass calculation, acceptance behavior, and return contracts. Do not introduce a new baseline algorithm. |
| Balance polling | The user confirms that existing balance polling works. | Reuse it; no transport or concurrency redesign. New recording errors must not alter existing delivery recovery. |
| Persistence | `append_csv` reads and rewrites the full master file for every addition, under a process-local lock. | Reuse initially and measure save duration. Save failures must not repeat delivery or replace its result/exception. No historical data migration. |
| Analysis | `scheduler/liquid_review.py` matches measurements by method/settings; the OAV analysis scripts also read the unified master. These readers do not explicitly exclude `routine_` sessions. | Routine data can affect counts and comparisons, which is part of unified analysis. Review interpretation and provenance; do not change readers or live calibration selection without identifying a specific need. |
| Existing artifacts | The remaining recording documentation and tests describe integration that is absent from the current controller. Some tests mock the entire polling thread. | Treat them as references, not proof that the feature currently works or preserves execution. |

Keep unrelated reservoir validation-file consolidation and calibration weighing
cleanup outside this change. Implementation details below are subject to these
constraints. No controller code or hardware has been changed by this review.

## Current state and scope

As inspected on 2026-10-24, `pipetting_data/dispense_recording.py`, `docs/dispense_recording.md`, and `tests/test_dispense_recording.py` remain in the workspace. `Lash_E` accepts `dispense_recording="off"`, but North Safe currently lacks the corresponding recording integration. Treat these remaining artifacts as implementation references and intended behavior, and reconcile them against actual callers before reuse.

Keep the current gripper tracing, release motions, timing, cap handling, staging, transfer planning, and vial inventory changes intact. The reported capped-vial incident is a separate investigation. Measurement integration must not change which vial is moved, when it is uncapped, or where a tip enters.

## Required behavior

- Default recording to `off`; support explicit `auto` through `Lash_E` and `North_Robot`.
- Automatically measure only real dispenses into the primary balance clamp (`location="clamp"`, index `0`) with an explicitly supplied, registered liquid and usable density.
- Evaluate eligibility after existing destination preparation. Recording must never move a vial just to make it eligible.
- Save routine rows in the unified master measurement database, using the existing schema and units. Keep their provenance identifiable with `routine_` session IDs and the existing validation type fields.
- Calculate measured volume as mass increase divided by liquid density. Store the nominal target and effective parameters for the actual measurement window, including its compensation contribution. Do not claim per-stroke measurements when only a whole-operation mass difference is available.
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

Review the remaining helpers before reuse. Preserve the working weighing collection windows, sampling cadence, mass calculation, and stability assessment; do not replace them with `analyze_baselines` or a stricter stable-only rule. Require a finite, positive density for calculating volume. Keep unavailable measurements distinguishable from a genuine measured zero without changing existing caller returns.

Reuse the working measurement lifecycle: pre-baseline, delivery, post-baseline, analysis, optional persistence. Keep new recording errors isolated from existing execution and error handling. Do not redesign transport or polling in this change. If implementation exposes a concrete cleanup problem, explain it separately before broadening scope.

Reuse a single set of readings when explicit weighing and automatic recording are both enabled. Avoid duplicate polling threads, baseline windows, and physical dispenses.

### 4. Integrate pipette dispensing and provenance

Wrap the existing delivery in `dispense_into_vial` with measurement after accessibility and tip positioning are established. Collect the post-baseline at the existing position, before recap or vial movement can alter balance load. There is no existing tip retraction in this method to reuse. Do not add or reorder physical commands.

Capture source identity and effective aspiration parameters when aspiration actually succeeds. Make that context available through the matching dispense, including split transfers. Clear or replace it on tip disposal, reset, and unrelated aspiration so records cannot inherit stale source/settings. Unknown aspiration settings stay empty rather than being copied from unrelated dispense defaults.

Record the destination, liquid, density, target, mass difference, volume, signed accuracy, stability information, tip type, and actual resolved parameters. Preserve legacy explicit weighing return behavior and transfer-total calculation. Flag missing routine measurements in storage; do not repair existing return contracts as part of recording.

### 5. Integrate reservoir dispensing inside its existing loop

Keep the existing move-to-clamp, uncap, carousel positioning, pump-loop, carousel reset, recap, and return-home sequence. Per-stroke measurements are preferred; discuss their baseline placement and any additional waits before implementing. Do not recursively invoke the complete reservoir method per chunk.

Preserve the exact number of pump strokes and total commanded volume. Attribute nominal target and compensation to the actual measurement window. Record the existing `reservoir_index` field and actual reservoir parameters. Finish the post-baseline before subsequent movements affect the load.

### 6. Exclude duplicate and non-routine records

Pass `record_measurement=False` from calibration paths that already persist their own measurements, while retaining their explicit weighing. Exclude mixing, conditioning, and liquid-return operations. Review all call sites rather than relying only on liquid omission or a single helper test.

Save new routine reservoir measurements in the unified measurement master. Leave existing reservoir calibration writers, summaries, and historical files unchanged in this implementation; their consolidation is separate work.

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

The focused suites pass, every eligible measurement window produces exactly one correctly attributed row in the unified master, unavailable mass is distinguishable from zero, calibration records are not duplicated, simulation produces no routine measurement artifacts, and recording does not alter the physical command sequence, existing return contracts, or delivery failure handling. Measurement waits match the agreed existing windows. Reservoir measurement granularity is explicit and matches the readings actually collected.

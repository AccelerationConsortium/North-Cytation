# Align reservoir transfers and calibration with pipetting

Date: 2026-10-26
Status: End-to-end code examination complete on 2026-10-26. Ordered implementation
steps below are proposed; no reservoir execution changes have been made.

## Goal

Treat each physical reservoir dispense as an individual dispense, with its own
nominal target, resolved settings, compensation, and measurement. A larger addition
coordinates those dispenses and aggregates their results. Three dispenses are three
dispenses, even when requested through one method call.

Use the pipetting approach as the reference for planning and calibration. Keep
reservoir-specific valve and pump behavior. This document plans a deeper reservoir
revision; it is separate from the measurement-only integration in
[the routine recording plan](north_safe_dispense_measurement_updates.md).
Creating this plan does not implement changes or authorize hardware runs.

## Multiple liquids, pump positions, and capacities

User requirement: additional reservoir pumps will be installed at different
positions, with different liquids and syringe capacities. Water, pump 1, and
2.5 mL describe the current example only; they are not system-wide assumptions.

- Each configured reservoir/pump position has its own identity, capacity, and
  carousel/access coordinates. Planning and calibration constraints use the
  selected pump's configured capacity, including compensation. Do not silently
  substitute a 2.5 mL capacity when that pump's capacity is missing.
- Select calibration for the liquid and the particular pump/setup together.
  The same liquid may be used on multiple pumps with different capacities or
  delivery behavior. A liquid-only match must not silently select another pump's
  calibration. A changed syringe capacity or delivery setup requires confirming
  calibration compatibility rather than reusing the old calibration automatically.
- Carry the selected pump position and liquid through calibration, independent
  validation, everyday execution, and saved measurement provenance. Resolve
  positioning from that pump's configuration rather than hardcoding an angle or
  height. Preserve explicit liquid identification in workflow calls.
- Use the same planning and parameter-resolution approach for every pump, while
  allowing pump-specific calibration values, constraints, and capacities. No
  separate execution implementation should be needed for each installed position.
- Verify at least two pump positions with different capacities, multiple liquids,
  and the same liquid on two pumps. Check that settings, capacity limits,
  coordinates, and measurement identity follow the selected pump without mixing.

## What the current code does

These findings come from inspecting the working tree on 2026-10-26.

| Area | Pipetting reference | Current reservoir behavior |
| --- | --- | --- |
| Planning | `plan_vial_transfer` chooses executable equal chunks and resolves calibration for the chunk volume, considering tip capacity and calibrated range. | `dispense_into_vial_from_reservoir` resolves parameters for the full requested volume, adds compensation once, then divides the compensated volume into equal strokes. |
| Capacity | Planning checks the resolved chunk load, including relevant overhead, and increases the repeat count when needed. | Stroke count is `ceil((volume + overaspirate_vol) / max_volume)`, where pump capacity comes from configuration, with a 2.5 mL fallback. |
| Measurement | The transfer loop invokes the destination dispense method for each chunk; explicit weighing can measure each destination dispense. | Continuous weighing has a pre-baseline before the complete pump loop and a post-baseline afterward. Traditional weighing also returns a whole-call difference. Individual stroke masses are not calculated or summed. |
| Calibration lookup | `resolve_pipetting_setup` and `PipettingWizard` resolve settings for the executable chunk. | `ReservoirWizard` exists, but its integration in `_get_optimized_reservoir_parameters` is commented out. The controller currently applies defaults and explicit overrides. |
| Calibration experiments | Executable pipetting chunks are the reference for comparing settings and performance. | Embedded validation and the standalone reservoir protocol each call the reservoir method once per requested calibration volume and receive one whole-call result, even if that call splits into several strokes. |
| Storage | Individual measurements and summaries use the shared pipetting masters. | Embedded reservoir validation already aliases its measurement and session master paths to the shared pipetting masters. Do not plan a second consolidation that has already happened. |

Relevant sources:

- [Controller planning and execution](../North_Safe.py): `plan_vial_transfer`, `resolve_pipetting_setup`, `_get_optimized_reservoir_parameters`, and `dispense_into_vial_from_reservoir`.
- [Pipetting chunk regression tests](../tests/test_vial_transfer_chunk_calibration.py).
- [Reservoir parameter lookup](../pipetting_data/reservoir_wizard.py).
- [Parameter definitions](../pipetting_data/pipetting_parameters.py): `ReservoirParameters`.
- [Embedded calibration and validation](../pipetting_data/embedded_calibration_validation.py): `validate_reservoir_accuracy` and shared measurement writers.
- [Standalone reservoir calibration protocol](../sdl_pipette_calibration/protocols/calibration_protocol_reservoir.py).

## End-to-end examination

The intended lifecycle is:

`configure reservoir -> calibrate individual dispenses -> validate those settings
-> select the validated calibration -> execute planned dispenses -> record performance`

The existing pieces support much of this already. Unification should connect them,
using reservoir parameters, rather than introduce another optimization framework or
rewrite the working pipetting path.

### What can already be shared

`CalibrationExperiment`, `CleanCSVExporter`, and `ValidationRunner` already provide
the optimization, calibration-table export, and independent validation stages.
The newer [calibration reader](../sdl_pipette_calibration/pipetting_wizard.py) is
hardware agnostic despite being named `PipettingWizard`: it discovers
`hardware_parameters_*` columns dynamically. It is a different module from the
legacy everyday pipetting lookup in `pipetting_data/pipetting_wizard.py`.

A read-only, in-memory check in `north_env` confirmed that the shared exporter and
newer reader round-trip and interpolate all six `ReservoirParameters` fields:
aspirate speed, dispense speed, aspiration wait, dispense wait, valve-switch delay,
and overaspirate volume. The check used 0.5 and 1 mL calibration points and resolved
a 0.75 mL point, preserving the valve delay. No hardware or measurement files were
used. This establishes format compatibility, not physical calibration accuracy.

Use these existing modules. A reservoir parameter whitelist/configuration is enough
to avoid importing pipette tip, air-gap, or retraction settings into reservoir trials.

### Concrete gaps to address

1. **Protocol configuration:** the reservoir-specific protocol hardcodes pump 1,
   `simulate=False`, its vial file, quality threshold, and priming amount. It creates
   its own coordinator and has no equivalent of the pipette protocol's
   `workflow_session` for borrowing the workflow's existing controller. The generic
   `ProtocolWrapper._config_to_dict` currently forwards only liquid and seed; simply
   putting more fields in YAML will not make this protocol receive them. Generic
   simulated-protocol runs do not exercise reservoir-specific hardware logic.
2. **Capacity source:** hardware configuration defines pump 1 as a 2.5 mL water
   reservoir in [syringe_pumps.yaml](../robot_state/syringe_pumps.yaml). Reservoir
   protocol constraints separately hardcode 2.5 mL. Planning, trial constraints,
   validation, and execution should use the same configured capacity.
3. **Calibration-table handoff:** the modern exporter writes explicit
   `volume_target_ml`/`volume_target_ul`, `volume_measured_ml`,
   `calibration_overaspirate_vol`, and `hardware_parameters_*` columns. The old
   reservoir wizard expects unprefixed fields and `volume_target` in mL. Its lookup
   is also disconnected from the controller. Enabling that old code as-is will not
   establish the complete modern calibration-to-workflow handoff.
4. **Compensation consistency:** the old reservoir lookup caps adjustment at 50%
   of target; the newer generic reader uses a different correction policy. Embedded
   reservoir validation's optional adaptive branch only logs a need for manual pump
   correction, even commenting that reservoirs do not use overaspirate, while actual
   execution and optimization do support it. Choose one explicit compensation path
   and apply it once; do not combine these inconsistent policies silently.
5. **Physical unit:** the controller resolves the full addition and measures the
   whole stroke loop. Calibration/validation callers consequently receive one
   aggregate even when the method performs several physical dispenses.
6. **Everyday callers:** active reservoir refills in fluorescence and surfactant
   workflows omit `liquid=`. Even with lookup connected, those calls will not select
   liquid calibration or qualify for the agreed known-liquid routine recording.
   Update these known-water calls explicitly rather than guess from a vial name.
7. **Measurement ownership:** embedded reservoir validation already writes to the
   unified masters. Standalone calibration/validation have their own run artifacts;
   inspect and connect their physical-measurement persistence deliberately. Keep
   attempted calibration replicates, validation results, routine rows, and aggregates
   from being mistaken for duplicate or independent physical deliveries.

The reference for workflow orchestration is
[pipette_calibration_workflow.py](../workflows/pipette_calibration_workflow.py):
borrow one controller, run calibration, validate the exact resulting table (or its
in-memory equivalent in simulation), and report the result without silently promoting
the calibration to everyday use.

## Intended design

### 1. Plan nominal dispenses before executing them

Introduce a reservoir planning module, or a small controller helper following the
existing pipetting planner, that produces executable nominal chunk volumes and
their resolved settings without moving hardware.

Start with equal nominal chunks, matching pipetting, unless investigation identifies
a reservoir-specific reason otherwise. Resolve calibration at each chunk's nominal
volume. Check that the resulting commanded liquid volume, including that chunk's
compensation, fits the configured pump capacity. Replan if it does not.

The nominal chunk targets must sum to the requested addition. Compensated pump
commands may sum to a different volume, just as they do for calibrated pipetting.
Do not keep splitting one request-level compensation among strokes once calibration
is defined per stroke; apply the compensation resolved for each nominal stroke.

Example, with an illustrative 2.5 mL capacity:

- A 6 mL nominal addition becomes three 2 mL nominal dispenses.
- Resolve the settings for a 2 mL dispense, not a 6 mL dispense.
- If the selected compensation is +0.02 mL per dispense, command 2.02 mL three times.
- Save three targets of 2 mL, each with +0.02 mL compensation and its own measurement.

This deliberately changes reservoir calibration/compensation semantics. It must be
reviewed and tested as this reservoir revision, rather than introduced silently by
the measurement recorder. Existing request-level calibrations cannot automatically
be assumed to describe individual strokes.

### 2. Separate one-time vial preparation from individual dispenses

Keep destination preparation, clamp access, uncapping, carousel positioning, final
carousel reset, recap, and return-home in the coordinating operation. Perform them
once where appropriate, preserving existing cap ownership and physical sequencing.

Use an internal single-dispense interface for one reservoir aspiration/delivery
cycle with already resolved settings. It should execute one physical dispense;
it must not silently split again. Keep parameter resolution, capacity validation,
and measurement attribution consistent with the plan.

Do not implement this by recursively calling the complete public reservoir method
for every chunk: that would repeat preparation and cleanup motions.

### 3. Measure individual dispenses using the working weighing system

Reuse existing balance polling, mass calculation, stability assessment, and usual
measurement windows. Do not redesign those working elements as part of alignment.
Place measurement windows around each physical dispense before another delivery or
vial movement can contaminate its result.

Per-stroke windows may add measurement waits between strokes compared with today's
whole-loop weighing. Show the exact placement and timing before implementation.
Do not fabricate individual masses by dividing a whole-loop mass difference.

For complete measurements, the coordinator can sum individual masses and measured
volumes. If any stroke is unmeasured, retain the valid individual rows but mark the
aggregate as incomplete; decide how that maps to existing public return contracts.
Do not silently return a partial sum as a complete measured total.

Routine measurement and save failures must log a warning and allow delivery to
continue. They must not retry or repeat liquid delivery. Preserve the distinction
between optional recording failure and actual pump/motion failure.

### 4. Calibrate the same individual operation used in workflows

Have calibration and ordinary reservoir additions use the same planner and
single-dispense execution interface. A calibration point must identify the nominal
physical dispense volume whose settings it determines.

For calibration of an individual dispense, choose a target that fits one resolved
stroke. If a requested calibration experiment intentionally spans several strokes,
record the individual targets and outcomes and label any overall result as an
aggregate; do not fit a single-stroke calibration to that aggregate accidentally.

Review both embedded validation and the standalone protocol, including replicates,
quality retries, summary counts, and settings reported in results. Calibration
retries are additional physical deliveries and must retain their attribution.
Do not copy calibration retry behavior into routine dispensing.

Align parameter selection, explicit overrides, compensation controls, and selected
calibration provenance with pipetting. Review baseline/correction/verification stages
where applicable. Do not copy pipette-specific air gaps, tip rules, or numerical
correction bounds into syringe-pump calibration without justification.

### 5. Connect calibration lookup only after auditing its inputs

The existing wizard is a reference, not a ready-to-enable integration:

- It documents reservoir `volume_target` values as mL, whereas the pipetting lookup
  uses uL. The existing `optimal_conditions_reservoir.csv` has targets such as 50
  and 100; establish their actual units and origin before interpreting them.
- Its discovery patterns expect liquid-specific reservoir filenames; verify that
  intended calibration exports are actually discoverable and belong to the correct
  liquid and pump setup. The generic filename alone does not establish identity.
- Define calibration range behavior and pump/setup identity so unrelated or
  unsupported calibration is not selected silently.
- Define whether files contain raw or already corrected compensation. Avoid
  applying compensation twice when enabling the wizard.
- Preserve explicit override precedence, and make missing-calibration behavior
  deliberate and consistent with the agreed pipetting approach.

Keep historical files intact. Document any compatibility handling or new exports;
do not infer units or relabel historical measurements without evidence.

### 6. Record individual rows in the unified database

Use `pipetting_data/master_pipetting_measurements.csv` for physical measurements and
the existing session master for summaries. Routine and calibration measurements
remain available for analysis together. Avoid adding an aggregate measurement row
that would double-count the same physical deliveries.

Capture nominal target, mass, measured volume, density, actual resolved settings,
compensation, destination, liquid, reservoir/pump identity, stability, and available
calibration provenance. Do not re-query settings afterward to reconstruct execution.
Keep nominal vial-volume accounting distinct from measured volume.

Reuse `validation_type` for dispensing method and existing session identifiers for
routine/calibration provenance. Group the strokes belonging to one addition so
totals can be reconstructed. Decide whether existing session/replicate fields are
sufficient for that grouping before proposing additional columns.

Audit the real shared CSV header: the inspected header does not currently contain
`reservoir_index` or `valve_switch_delay`, although reservoir code can emit them.
Specify how those actual settings are represented without losing data or corrupting
older rows. Ensure exactly one writer owns each measurement; explicit calibration
and automatic routine recording must not save the same delivery twice.

Keep simulation out of real measurement files. Reuse the existing writer initially;
measure save cost before considering a separate persistence redesign.

## Work sequence

### Step 1: Establish the reservoir configuration and calibration-table contract

Use the six existing reservoir parameters. Take pump capacity and identity from the
existing hardware configuration; make liquid, pump, vial, simulation mode, priming,
and quality settings explicit in the reservoir calibration configuration. Retain
existing defaults where deliberate, but do not hardcode simulation off.

Make this configuration support several installed pumps with different capacities
from the outset. Bind calibration to both liquid and pump/setup identity. The
current water/pump-1 configuration is one instance of that contract.

Select the modern export format as the calibration/validation interchange. Record
liquid and pump/setup identity alongside the selected artifact. Audit old files and
their units separately; they must not become active implicitly.

Completion: a configuration and sample table can be validated without constructing
hardware. No `North_Safe.py` changes are needed for this step.

### Step 2: Reuse one parameter-resolution path

Use the existing hardware-agnostic table reader for explicit modern calibration
tables, with a small reservoir-specific interface to map its result into
`ReservoirParameters`. Use that same interface for post-calibration validation,
embedded validation, and everyday calibrated execution.

Define defaults, calibrated values, and explicit-override precedence; determine
range handling and whether residual-error compensation is enabled. Both validation
and everyday use must resolve identical effective settings from the same table.
Do not apply the reader's residual correction again to an already corrected value.

Completion: in-memory/exported table equivalence, units, valve delay, overrides,
range behavior, and compensation-once checks pass in `north_env`, without hardware.
Keep this outside `North_Safe.py` initially.
Include liquid/pump selection checks so a matching liquid on a different pump does
not provide calibration accidentally.

### Step 3: Plan executable individual dispenses

Implement a pure reservoir planner patterned on `plan_vial_transfer`: split nominal
targets, resolve settings at each physical dispense volume, and check each
compensated load against configured capacity. Calibration of one dispense must
reject a trial that needs splitting; validation of a larger addition may deliberately
plan several dispenses and report each result plus an aggregate.

Completion: fake lookup/capacity tests establish that targets sum correctly,
compensation is per stroke, and replanning fits every command. Still no physical
execution changes.

### Step 4: Make the smallest controller change for individual execution

Keep preparation and cleanup once per addition. Reuse the existing valve/pump
command sequence for each planned dispense. Place existing measurement windows
around individual strokes, and aggregate complete results for the public call.
Do not recursively repeat the full public reservoir method.

Review a narrow `North_Safe.py` diff before extending it. Show measurement-wait
placement and the expected added duration for a split addition. Preserve existing
weighing calculations; do not add generic error wrappers, transport changes, or
cleanup refactors. Define incomplete aggregate results and compatibility explicitly.

Completion: one-stroke command traces match the existing physical sequence, and
multi-stroke traces show only the planned calibration/measurement differences.
No physical run is required to review this step.

### Step 5: Connect calibration and independent validation end to end

Give the reservoir protocol a workflow binding analogous to pipetting so both
phases borrow the same confirmed controller/configuration. Reuse the existing
optimization engine with reservoir-only parameters and capacity constraints.
Keep trial settings explicit so optimization is not overridden by an older active
calibration. Priming is preparation, not a calibration sample.

Pass the exact calibration output into `ValidationRunner`, including its existing
in-memory simulation route. Validate without changing that candidate during the
validation phase. Record the outcome; select/promote an artifact for everyday use
as an explicit separate action. Do not create a second optimizer or rewrite the
working pipette workflow to achieve this.

Align embedded reservoir validation with the same resolver and physical dispense
definition. Initially retain its role of checking current settings; any requested
recalibration should use the shared calibration engine rather than grow another
adaptive correction algorithm. Existing pipetting correction behavior is untouched.

Completion: fake-controller calibration -> exact table -> independent validation
checks pass, including simulation mode, configuration propagation, and writer
ownership. Physical calibration/validation is a separately requested run.

### Step 6: Activate the same settings in everyday execution

Connect the resolver/planner to the existing reservoir entry point with a deliberately
selected calibration artifact. Update known-water refill callers to supply
`liquid="water"`. Preserve existing method defaults/positional meanings and explicit
overrides; review any newly introduced option separately from the plumbing.

Completion: an everyday 6 mL addition plans three 2 mL nominal dispenses (when capacity
and calibration support that), resolves the same settings as 2 mL validation, and
applies each dispense's compensation once. A selected table's identity and actual
settings are visible in logs/records. No automatic promotion from a calibration run.

### Step 7: Add observational routine recording and review the data

Extend the agreed opt-in recording behavior to convenient reservoir dispenses using
the same unified master and recording helpers. Save one row per physical dispense,
its actual settings, and enough grouping to reconstruct an addition's total.
Resolve missing reservoir identity/valve-delay fields explicitly. Calibration and
validation retain sole ownership of their physical rows; the routine writer opts
out for those calls. Do not also save a summed mass as another physical measurement.

Completion: three physical deliveries produce three attributable rows; totals are
reconstructible, routine failures never trigger a delivery retry, and simulation
produces no real-data artifacts. Use existing CSV persistence. Review software
checks and then, when requested, a small physical trial before broader use.

Each step should be a separately reviewable change. Steps 1-3 establish most of the
contract without editing the controller. Do not implement all seven as one rewrite.

## Verification and completion

Use fake hardware and temporary measurement files for software checks. Cover one
stroke, several strokes, capacity edges, positive/negative compensation, invalid
volumes, missing calibration, unit conversion, overrides, and simulation. Verify
that compensation is resolved per nominal stroke and that planning finishes before
hardware preparation begins.
Exercise different pump capacities and access positions, different liquids, and
one liquid on multiple pumps. Missing capacity or incompatible pump/setup identity
must be identified before physical execution rather than replaced by pump-1 defaults.

For three completed physical dispenses, require three attributable measurement
records when weighing is enabled, with nominal targets summing to the request and
recorded settings matching actual commands. Where readings are unavailable, retain
that fact without inventing a mass or repeating delivery. Verify grouping and sums,
no duplicate aggregate measurement row, and compatibility with existing readers.

Compare destination preparation, cap transitions, carousel operations, and cleanup
against current behavior. Document intended changes in compensated pump volumes
and measurement timing rather than claiming all execution remains identical.

Complete when calibration and routine execution share the same physical dispense
definition, select settings at that volume, measure and record each dispense, and
aggregate larger additions without losing individual results or counting them twice.

## Relationship to the measurement-only work

This plan owns changes to reservoir splitting, calibration lookup, compensation,
per-stroke measurement timing, and aggregation. The recording integration must not
make those execution changes incidentally. Pipette measurement recording can be
reviewed independently while this reservoir design is resolved.

# Mixed wellplate inventory and position tracking

Date: 2026-10-28
Status: Proposed; plan only. No implementation or hardware actions in this task.

## Objective and existing coverage

Allow different physical plates to coexist at different positions, with each
operation deriving the correct type from the plate it actually handles. Keep
ordinary workflows as simple as grab, measure, discard.

Mixed source and waste stacks are required in the first release. A design that
requires homogeneous stacks or manual waste clearing between different plate
types does not meet the objective.

Concurrent Cytation reading and pipetting on another plate are supported behavior.
Do not reserve the pipetting area for an absent plate or require a new physical
holder to enable this. The workflow controls when it clears the return position.

## Capability and restriction audit

Physical capacity rules confirmed by the user:

- The track gripper holds exactly one of: a detached lid, a lidless wellplate,
  or a wellplate with its attached lid. It may also be empty. It cannot hold
  a second independent object in addition to any of these.
- Each position outside source/waste initially has configured capacity for one
  wellplate. This is a current setting, not a hardcoded rule. The active area
  could hold multiple plates; retain that possibility in the model.
  Source and waste are ordered stacks and can hold multiple mixed-type plates.
  This is a per-position limit, not a limit of one plate across the system:
  Cytation and the pipetting area can each contain a plate simultaneously.
- An attached lid does not consume another wellplate slot. Removing a lid
  leaves its plate occupying that position.

Recommended additional command requirements for discussion, not new broad
workflow policies: pick up an object that is actually recorded at the source;
release the object actually held; respect the destination's capacity; do not
detach/attach a nonexistent or already attached lid; do not move the Cytation
carrier during a read; and require the numeric geometry needed for the movement.
Apply these locally to the requested command. Ordinary plate operations should
handle lid attachment and source occupancy without overwriting either object.

The inventory refactor should describe and enable operations, not impose new
workflow policies. The user explicitly rejected unnecessary blocks. An operation
not previously exercised is not thereby unsupported.

Remove these restrictions from the proposal:

- Homogeneous-only source or waste stacks, and manual clearing between types.
- Return-slot reservations lasting for the duration of a Cytation measurement.
- Blanket serialization of reading, pipetting and track work. Coordinate actual
  shared motion resources only when their commands would conflict.
- A requirement for an alternative parking holder before concurrent plate work.
- A mandatory approved/validated type-pair list before mixed stacking is allowed.
  Geometry parameters and physical verification are needed; certification flags
  or per-pair approvals are not an inventory requirement.
- Mandatory whole-workflow planning/preflight for every direct plate command.
- A blanket ban on old workflows, occupied positions at scheduler handoff, or
  further unrelated operations after any interruption/manual axis movement.
- A ban on operator corrections while another independent operation is running.

Retain only operation-specific requirements needed to execute the requested
command correctly: resolve the selected object and numeric motion inputs, do not
overwrite another object's placement, do not put two plates into a single-slot
holder, respect actual axis/clearance limits, and do not move the reader tray
during an active read. Report the exact conflicting operation or missing input.
Uncertainty about one object/resource does not lock the entire inventory.

For stacks, adding to an occupied stack is normal. Lid attachment is likewise
normal at an occupied plate position. Occupancy alone is not a generic rejection.
Unknown calibration must not silently become zero, but existing/user-supplied
geometry remains usable without a separate approval workflow. Any additional
restriction proposed during implementation must be explained and discussed,
not introduced as an assumed safety policy.

No existing plan for this change was found in `plans/` (including its archive),
`agent/research/`, or the repo's `.agents/` content. Existing workflow guidance
describes one global track plate type. Related documents cover selecting a
format for an experiment, not an inventory of different plates:

- [24-well readiness](../docs/24_well_readiness.md).
- [Workflow construction](../agent/research/workflow_construction.md).
- [Fluorescence calibration](../docs/fluorescence_calibration_workflow.md).
- [Surfactant grid](../docs/surfactant_grid_ailsa.md).

The user's successful physical 24-well round trip establishes a useful baseline
for that tested setup. It does not validate mixed stacking or multi-plate state.

## Diagnosis

`wellplates.yaml` is already the right kind of file: definitions of plate
types, including geometry, material and the Cytation name. Keep its existing
plate entries and keys. Consolidate the separate lid grip definitions into a
flat `lid_gripping_z_offset` on each compatible plate definition, as agreed
below. This is a targeted catalog change, not a redesign.

## Agreed lid behavior

Decisions confirmed during discussion:

- An attachment exists only while a physical lid is on a physical plate. It
  does not mean permanent ownership of that lid by that plate.
- Record attached lid identity, and derive lid presence from the attachment.
  Do not maintain a second mutable `has_lid` boolean.
- Keep lid grip calibration as a flat field on the plate definition:

  ```yaml
  96 WELL PLATE:
    gripping_z_offset: 0
    lid_gripping_z_offset: 3000
    thickness: 6400
    num_wells: 96
    # Other existing plate properties remain.
  ```

- Move a plate normally whether or not its lid is attached: use the plate's
  `gripping_z_offset`. Use `lid_gripping_z_offset` only when handling the lid
  separately. Do not introduce an offset for moving a lidded plate.
- Separate lid objects reference their compatible plate type. For example,
  `{kind: lid, plate_type: quartz}` resolves lid grip calibration through
  `wellplates.yaml[quartz].lid_gripping_z_offset`. It does not reference a
  standalone `quartz_lid` definition or duplicate calibration in runtime state.
- Removing a lid clears the attachment and places that lid in the gripper,
  then storage; the plate's position remains unchanged. Attaching does the
  reverse. Moving a plate carries its attached lid without detaching it.
- Missing/null lid grip calibration means detached-lid handling is unsupported;
  zero is a valid calibrated offset. Lidless plate handling remains possible.

Migration should move the existing `quartz_lid.gripping_z_offset` (7500) to
`quartz.lid_gripping_z_offset`, and `96_wellplate_lid.gripping_z_offset` (3000)
to `96 WELL PLATE.lid_gripping_z_offset`. Do not invent lid calibrations for
24- or 48-well plates. Update callers before removing legacy lid entries.
Reader `use_lid` still derives from attachment state. Pipetting requires the
lid to be removed. Stack pitch with lids is a separate calibration question;
normal plate movement does not establish stacking geometry.

`track_status.yaml` mixes three different concerns:

1. Global type selection (`wellplate_type`).
2. Inventory reduced to counts (`num_in_source`, `num_in_waste`).
3. A single plate's location (`active_wellplate_position`), alongside gripper coordinates.

Consequently, it cannot express a 24-well plate in Cytation while a different
96-well plate occupies the pipetting area. Even two plates of the same type
cannot be independently identified at those positions. Changing a global type
would reinterpret plates that have not moved.

The low-level motion code is reusable, but its bookkeeping must change with
its callers. Adding only a type argument or another global current-type field
would leave conflicting sources of truth.

## Proposed domain language

Capture these terms in this plan while the design is proposed. A later
implementation can add them to the project glossary.

| Term | Meaning |
| --- | --- |
| Plate type | A reusable definition in `wellplates.yaml`, such as `24 WELL PLATE` or `quartz`. |
| Plate | One physical plate with a stable local ID and a reference to a plate type. No barcode hardware is required. |
| Position | A named physical place such as `pipetting_area` or `cytation_tray`. |
| Stack | An ordered collection of physical plates, recorded bottom to top. Only the top plate is accessible. |
| Lid | A separately identifiable physical item, either parked or attached to a plate. |
| Plate inventory | The authoritative mapping of objects to positions/stacks/attachments. |
| Track pose | The gripper's mechanical position, independent of which object it holds. |
| Transfer | A checked change in physical occupancy, including pickup and release stages. |
| Expected type | A caller's assertion about the selected plate, never an instruction to relabel it. |

Use `cytation_tray` consistently. Some workflow guards currently compare against
`cytation`, although transfer methods record `cytation_tray`.

## State proposal

Keep `track_status.yaml` as the single persisted document initially, with a
versioned schema. Separate inventory and mechanical pose inside it. This avoids
a second file whose updates could disagree with the first.

Illustrative proposed schema (not a file to install on hardware):

```yaml
schema_version: 2
revision: 17
objects:
  plate-001: {kind: plate, type: '24 WELL PLATE'}
  plate-002: {kind: plate, type: '96 WELL PLATE'}
  plate-003: {kind: plate, type: '48 WELL PLATE'}
  lid-001: {kind: lid, plate_type: quartz}
positions:
  source_stack: [plate-003]       # bottom -> top
  waste_stack: []
  pipetting_area: [plate-002]
  cytation_tray: [plate-001]
  gripper: null
  lid_storage: lid-001
  lid_storage_96: null
attachments: {}                  # plate ID -> attached lid ID
track_pose:
  location: origin
  x: 0
  z: 0
pending_transfer: null           # persisted intent/progress for interruption recovery
```

Object definitions hold identity and type; placement exists only in the
positions/attachments structure. Do not also store a mutable location on each
object. Counts are derived from stack lengths. No global plate type and no
single active-plate field in the new authoritative model.

Represent plate-holding positions as ordered lists, including those currently
limited to one plate. This avoids changing the inventory schema if the active
area later supports multiple plates. The gripper remains a single object reference
with its explicit physical carrying rules. Capacity comes from the position's
configuration, not a hardcoded position-name test. Adding a software position
never implies a new physical holder exists.
Registered but unlocated objects must be explicitly marked for reconciliation,
not quietly treated as available. Used plates remain identified in waste.

Invariants:

- Every placed object has exactly one placement: a position, stack entry, or
  attachment. An attached lid moves with its plate and cannot also be parked.
- Plate `type` and lid `plate_type` references exist in the plate catalog;
  lids are distinguished from plates. The referenced plate type resolves lid
  geometry; it is not a new whitelist of allowed attachments. Lid compatibility
  beyond the currently described matching type remains a design question.
- The gripper can carry one plate assembly or one detached lid, not both.
- Pickup requires an accessible source. Release requires capacity for the
  requested placement: an empty single-plate holder, space atop a stack, or a
  plate receiving a lid. A plate cannot be removed from underneath another plate
  without first moving the plates above it.
- A lid can occupy the same station as its plate through attachment, not by
  replacing the plate's occupancy record.
- Catalog edits or operator corrections must not silently change the identity
  or type of an existing physical object.

## Per-position capacity in track_positions.yaml

Add `max_wellplates` to each plate-holding position in `track_positions.yaml`.
Retain the existing `limits.max_stack_height` as the height constraint. The user
confirmed that this already constrains total thickness through the calculated
Z position, using the same robot units as plate thickness. Do not introduce a
second `max_wellplate_thickness` setting for the same constraint.
Capacity belongs to the physical position definition; actual plate IDs and order
belong to `track_status.yaml`. Do not duplicate mutable limits in both files.
Set the active area and Cytation to 1 initially. Source and waste need their
actual holder count limits established; do not invent a number or interpret the
GUI's range of 0-100 as a physical capacity.

Illustrative addition, retaining all existing coordinates:

```yaml
positions:
  pipetting_area:
    max_wellplates: 1
    # Existing coordinates remain.
  cytation_tray:
    max_wellplates: 1
    # Existing coordinates remain.
  # source_stack and waste_stack each get their actual numeric limit for:
  # max_wellplates: <count>
```

Current code has no configured count limit. `calculate_wp_stack_height()` in
`North_Safe.py` instead checks the calculated Z against the global
`limits.max_stack_height: 20000` in `track_positions.yaml`, plus the selected
grab/release base coordinate. This is an absolute Z threshold, not 20000 units
of stack thickness or a plate count. `North_Track` loads this configuration into
`self.TRACK_POSITIONS` and accesses it through `get_limit()`. It also loads the
per-type `thickness` definitions from `wellplates.yaml`, but does not track total
stack thickness as a separate state value. The GUI count widgets allow 0-100.

Preserve the existing absolute Z constraint and value. Use the ordered mixed
plate contributions when calculating pickup/release Z, then check that result
against `max_stack_height` as today. Derive heights from contents rather than
persisting a second mutable total. Retain operation-specific base coordinates
and plate/lid contributions where applicable; no separate thickness-limit
configuration or mandatory new physical-height model is required by this change.

The new capacity checks use both count and actual ordered-stack geometry:
adding a plate must fit the configured count and the existing reachable
height/clearance constraints.
With mixed types/lids, a stack may reach its height limit before its count limit.
An attached lid does not add to the wellplate count, though its physical height
can affect stack geometry. Detached-lid storage capacity is a separate question;
`max_wellplates` must not silently define that too.

Use the configured capacity in inventory validation, destination selection and
GUI presentation; do not add separate hardcoded 1s in those callers. If the
active area is later configured above 1, use explicit stacking or indexed-slot
geometry matching the real arrangement. Raising a count alone supplies neither
coordinates nor access to a covered plate. A top-first stack can use the same
ordered geometry model; side-by-side positions need separate slot coordinates.

## Positions and gripper GUI plan

Replace the current `TrackStatusWidget` form in `vial_manager_gui.py` with a
position-based editor, under the existing Track Status tab. Suggested header:
"Plate positions and track gripper". Remove the global plate-type selector,
editable source/waste counts, and free-text active-plate-position field.

This is an editor of recorded physical state, just like the existing startup
review. Adding, removing, reordering or relocating an item here does not command
robot motion. Label the editor "Record the current physical setup". Actual
hardware controls remain separate; no surprise movement on Save or row actions.

### Layout

Use a compact positions summary on the left and the selected position's contents
on the right. Include a persistent, separately labeled **Track gripper** panel
so its contents cannot be confused with the North robot arm's vial/cap gripper.

```text
Plate positions and track gripper
Record the current physical setup

Position             Plates / max     Contents: Source stack
Source stack          3 / N           Bottom -> top
Waste stack           1 / N           1   plate-003   96 WELL PLATE   Lid on
Pipetting area        1 / 1           2   plate-004   48 WELL PLATE   Lid off
Cytation tray         1 / 1           3   plate-005   24 WELL PLATE   Lid off  <- TOP / next pickup
Lid storage           Lid present
96-well lid storage   Empty           [Add] [Remove] [Edit] [Move lower] [Move higher]

Track gripper: Empty                  [Set contents...]
```

`N` stands for the configured physical count limit, not an invented default.
Display every real holding position, including empty positions and lid storage.
Do not show routing waypoints such as `cytation_safe_area`, `transfer_stack` or
`max_height` as inventory holders merely because they appear in the positions
file. Define which entries are holders explicitly in the position definitions
or reuse a clear existing classification during implementation. Do not infer
holder capability from a position's name alone.

### Position contents and actions

- **Contents:** show plate ID, plate type, lid on/off and attached lid ID where
  present. Friendly types are primary; IDs are visible for distinguishing equal
  types but generated automatically. Display plate count/capacity from config.
- **Add:** select plate or detached lid, then its plate definition. Adding a
  plate offers "lid on" and creates/attaches the lid automatically if selected.
  At stacks, allow adding a quantity of one type as a contiguous group, with
  another Add for a different type. Default insertion is at the top; allow
  choosing another layer while recording an already assembled physical setup.
- **Remove:** remove the selected physical item from this position's recorded
  contents. Removing a plate with its lid removes that assembly from the deck
  inventory, not just the plate while leaving an orphan attachment. Keep prior
  run provenance; removing an object from the setup does not erase past results.
  No removal is implicitly a waste transfer or hardware discard command.
- **Edit:** correct the selected item's recorded type or lid state. Distinguish
  correcting a description (same ID) from replacing the physical plate (new ID).
  Preserve IDs/order for untouched objects and do not regenerate all IDs on Save.
- **Reorder:** move a selected plate higher/lower within a stack. Show bottom/top
  and the next pickup explicitly. This records physical order and is not a robot
  sorting feature. Use the same contents model for an active-area stack if its
  configured capacity later exceeds one.
- **Record relocation:** let an existing item be reassigned to another position
  or to the track gripper while preserving its ID. Update both positions together
  so correcting where a plate is does not require deleting and recreating it.
- **Lids:** allow recording attachment, detachment and placement of an existing
  lid, as well as adding a newly loaded lid. Changing a lid from attached to
  parked updates both records together. If a lid was physically removed from
  the setup, record that rather than inventing a storage destination. Keep the
  plate's position and identity intact throughout these edits.

At capacity-one positions, use the same editor with one row; do not implement a
different singleton state model. Contents can be empty. At lid storage, present
the detached-lid view rather than counting lids as wellplates. Do not turn
`max_wellplates` into a hidden detached-lid capacity policy.

### Track gripper panel

Present these explicit contents choices:

- Empty.
- Lid only: select/create a lid and its plate-type reference.
- Wellplate without lid: select/create a plate.
- Wellplate with lid: select/create a plate and attached lid.

Existing items can be selected from other positions to record a physical pickup;
reassign their placement atomically, including any attached lid. A new item can
also be recorded directly in the gripper. The gripper model cannot record two
independent held items. Record dropping/removing or relocating contents through
the same editing actions; do not trigger the existing automatic startup discard
behavior just because the operator records a held plate.

Keep track pose (named location and X/Z) separate from gripper contents. Preserve
pose on ordinary inventory edits. Expose pose correction in a secondary area if
needed, without making coordinate entry necessary for adding/removing plates.

### Loading, saving and feedback

- Load the actual selected live/private-simulation status path. Use the shared
  versioned inventory parser, geometry calculation and validation; no separate
  GUI interpretation of types, counts or attachments.
- Edit a draft. Integrate with the parent GUI's existing Save/Review/Continue and
  reload behavior, preserving `state_loaded`/`state_saved` integration. Do not
  silently save every click to the hardware's live state file.
- Save the complete positions, object references, attachments and gripper state
  atomically, with revision handling. Preserve pose and unrelated state fields;
  never write legacy global fields over the new inventory.
- Show local, actionable feedback for duplicate placement, configured capacity,
  missing motion inputs or a calculated height outside the existing limit.
  Permit editing/reconciling an initially inconsistent setup instead of refusing
  to display it. Do not introduce approvals for a new type combination or lock
  all edits merely because an unrelated measurement is running.
- Show derived counts and, where useful, calculated stack/pickup height. Capacity
  values are read from `track_positions.yaml`; this status tab edits contents,
  not the physical configuration limits.

### GUI acceptance checks

- Build a mixed source stack, change its order, save/reload, and preserve every
  ID, type, lid attachment and bottom/top ordering.
- Add/remove contents independently at source, waste, pipetting area, Cytation
  and lid storage without changing other positions or track pose.
- Populate each of the four gripper states, save/reload, and ensure a lidded
  plate is represented as one held assembly with an attachment.
- Reassign an existing plate/lid between a position and the gripper without
  duplicating it, replacing its ID, or commanding hardware.
- Remove a lidded plate or change lid placement without orphaning a lid or
  deleting the plate beneath it. Preserve historical result references.
- Capacity displays/actions use `max_wellplates`; a multi-plate active area is
  represented with the same ordered contents editor when configured.
- Opening and saving status, including nonempty gripper state, never commands
  motors. Existing parent-GUI cancellation and private simulation paths work.

## Where plate type affects behavior today

Line numbers are inspection anchors and may move during implementation.

| Concern | Current code/call sites | Proposed handling |
| --- | --- | --- |
| State load/save | `North_Track.__init__`, `get_track_status`, `save_track_status` in `North_Safe.py:147-246` | Load/validate versioned inventory; stop writing the legacy global fields. |
| Pickup/release | `grab_wellplate_from_location`, `release_wellplate_in_location` at `North_Safe.py:532,581` | Resolve source/held object; derive plate `gripping_z_offset` or detached-lid `lid_gripping_z_offset`; move the same ID through gripper to destination. A plate's offset is unchanged by an attached lid. |
| Stack height | `calculate_wp_stack_height`, `get_new_wellplate`, `discard_wellplate` at `North_Safe.py:644,671,699` | Use ordered stack contents and the moving object's geometry. No count multiplied by one global type. |
| Recovery and occupancy | `reset_after_initialization`, `set_wellplate_position`; disabled occupancy check in release | Reconcile a specific held object. Replace single-active-position updates and enable checked destination occupancy. |
| Coordinator transport | `Lash_E.move_wellplate_to_cytation`, `move_wellplate_back_from_cytation`, `grab_new_wellplate`, `discard_used_wellplate` | Infer identity at each source; preserve routing/arm parking; validate explicit type expectations. |
| Reader mapping | Coordinator transport/measurement and `Biotek_Wrapper.CarrierIn` | Derive `name_in_cytation` and lid status from the selected plate assembly. Do not confuse catalog key with reader name. |
| Measurement | `measure_wellplate`, `measure_wellplate_async`, `MeasurementHandle.complete` | Bind measurement to a plate ID, protocol and return destination; revalidate occupancy at return. Null measurements still work. |
| Reader addressing | `biotek_new.py:298,392` (`run_protocol`, `group_wells`) | Keep the reader adapter explicit about format; pass the selected plate's layout, validate well indices. Existing 24/48/96 support remains. |
| Pipetting geometry | `pipet_from_wellplate`, `mix_well_in_wellplate`, `select_wellplate_grid`, `dispense_into_wellplate`, serial/batched vial-to-plate helpers (`North_Safe.py:2506,2576,3101-3395`) | Resolve the plate in the accessible pipetting position, then its Locator grid and depth. Propagate one resolved identity through each batch. |
| Capacity and well count | `SQ_Peptoid_LNP_test.py`, `fluorescence_calibration_workflow.py`, `surfactant_grid_ailsa.py` | Validate each planned physical plate's indices and volumes against its definition. Retain experiment-level desired types as requirements. |
| Material and geometry metadata | `wellplates.yaml`: material, radius, capacity, well count, depth | Keep definitions. Material compatibility and experiment volume scales remain scientific decisions; changing plates must not automatically rescale recipes. Radius/material are not general motion selectors today. |
| Lid moves | `workflows/Degradation_serena.py`, `tests/test_96_lid.py`, `tests/test_quartz_lid.py` | Existing low-level lid moves currently use the same single-active-plate state setter and standalone lid type keys. Introduce explicit detach/park/attach semantics and flat plate-level lid calibration without losing the plate beneath the lid. |
| Operator setup | `vial_manager_gui.py:1397-1505` (`TrackStatusWidget`) | Edit objects by position and ordered stack, with type selection from the catalog; display derived counts. |
| Scheduler | `scheduler/live_runner.py`, `simulation_runner.py`, `simulation_state.py`, GUI summaries | Carry all occupancy and operation state through handoffs; copy the complete new state into simulations; compute usage from plate IDs/transfers. Occupied positions are valid state, not a reason to reject a handoff. |
| Reuse across runs | `workflows/two_run_plate.py` and fluorescence marker file | Link half-plate progress to plate ID rather than matching global type plus source/waste counts. Same-type replacement is a different plate. |
| Workflow guards | `SQ_Peptoid_LNP_test.py`, fluorescence, surfactant/adaptive workflows and tester | Replace `CURRENT_WP_TYPE` and `ACTIVE_WELLPLATE_POSITION` checks with inventory queries; normalize `cytation` references. |
| Results and provenance | Workflow DataFrames, measurement outputs, logs, reuse markers | Add physical `plate_id` and plate type to provenance; joins across plates use `(plate_id, well_index)`. Keep existing within-plate labels compatible. |
| Manual motion | `track_control_gui.py` raw axis commands | These deliberately do not update status today. Preserve manual controls and permit explicit pose/inventory correction; never infer plate transfer from coordinates alone or impose a global lockout after any axis move. |
| Regression/workflow tests | Scheduler state tests, lid/stack utilities, calibration tests, new 24-well round-trip tests | Replace mutable singleton fixtures with inventory fixtures; keep the working one-plate behavior as a regression. |

`wellplate_index` in coordinator methods currently supplies logging/handle data,
not physical inventory selection. Elsewhere, similarly named columns mean a well
index. Introduce `plate_id` explicitly rather than assigning a new hidden meaning
to that argument.

## Interface and ownership

Introduce one inventory module (proposed `robot_state/plate_inventory.py`) shared
by `North_Track`, `North_Robot`, and the coordinator. Inject the same instance
within a run. Independent cached copies in robot and track would recreate the
problem. Persist through one store with revision checks/locking for writers;
normal startup GUI edits occur before execution. Support explicit operator
correction without overwriting newer unrelated changes; coordinate changes to
an object involved in an actual in-flight operation rather than forbidding all
edits while any hardware is running.

The small external interface should provide object-at-position/plate-by-ID
queries and checked transfer/attachment operations. Keep YAML parsing,
validation, counts, type resolution, transfer progress and persistence behind
this seam. Hardware movement remains with `North_Track`; the inventory module
does not command motors.

Preserve the ordinary workflow shape:

```python
plate_id = lash.grab_new_wellplate()  # top source plate -> pipetting area
lash.measure_wellplate(None)         # resolves that physical plate's type
lash.discard_used_wellplate()        # pipetting plate -> waste
```

For deliberate multi-plate operations, allow a `plate_id` on measurement and a
checked transfer between configured locations, e.g.:

```python
lash.move_wellplate(source="source_stack", destination="pipetting_area")
lash.measure_wellplate(protocol, plate_id="plate-001")
```

These are proposed interfaces, not callable additions yet. ID-based measurement
requires the plate to be at a supported accessible source; it does not search
inside stacks or silently shuffle other plates. `grab_new_wellplate` selects the
top plate only; an optional expected type rejects a mismatch rather than finding
a buried matching plate. Discard remains a move to waste. The generic transfer
can support a return to source with correct stack bookkeeping.

Existing explicit `plate_type`/`well_plate_type` and `use_lid` parameters become
optional assertions during migration. Omitted values resolve from inventory;
explicit mismatches fail before action. Replace 96-well defaults with `None`
where identity is available. A supplied type never substitutes for missing
inventory. The standalone reader adapter can retain explicit format inputs.

Pipetting is still physically supported at the configured pipetting position.
Knowing where a plate is does not create a Locator grid at Cytation or at a
storage position. Reject pipetting a missing plate, a lid-covered plate, an
unconfigured grid, or an out-of-range well before aspirating/dispensing liquid.

## Transfer, interruption and concurrency rules

1. Resolve identity, type and attachments. Validate source, destination, route,
   required motion inputs and actual resource conflicts before motion.
2. Persist transfer intent before movement. After successful pickup, atomically
   remove the source placement and record the held object. After successful
   release, atomically record destination occupancy and clear the gripper.
3. Save each physical milestone atomically with a revision. A file update is
   not proof of physical success: a power loss between action and save leaves
   pending intent that requires operator reconciliation.
4. A failed or interrupted move must not erase or duplicate a plate, decrement
   inventory twice, or guess that it arrived. Identify which object/resource
   needs reconciliation; allow explicit corrections and unrelated operations.
   Do not auto-discard a held item during startup recovery.
5. Coordinate commands using the same moving hardware when they actually
   conflict. Do not serialize whole workflows or prevent pipetting during an
   independent Cytation read. Inventory persistence locks cover state writes,
   not the entire read duration.

Asynchronous measurement captures a specific plate ID and its result. While
plate A is read in Cytation, plate B may occupy the pipetting area, receive
liquids, have its lid handled, and move to a stack as the workflow requests.
No return-slot reservation or release/reassignment ceremony is required.

`complete()` waits for reader completion and checks the actual destination
before return motion. The workflow can clear that destination by moving plate
B first. If it is still occupied, report that specific conflict before picking
up A, leave A recorded at Cytation, and allow completion to be retried after
the destination is cleared. Do not reread A or repeat a completed transfer.
An alternative configured destination can be selected when useful; it is not
a prerequisite for concurrent reading and pipetting.

## Mixed source and waste stacks: required initial scope

Support mixed types both across positions and within source/waste stacks in the
first release. Model each stack as ordered physical IDs, bottom to top. Source
pickup resolves the actual top plate; waste placement uses the existing support
stack and incoming plate. Homogeneous stacks are a special case of this model.

Inspection of the actual method confirms that the existing formula is
`base - thickness * (num - 1)`. Generalize it directly to a discrete sum of
the configured thicknesses of plates beneath the target. No new pair-specific
geometry model or additional height-limit setting is needed for this refactor.

For a stack recorded bottom to top:

```python
# Pick up the existing top plate: exclude that plate's own thickness.
z_grab = position["z_grab"] - sum(thickness(p) for p in stack[:-1])

# Place a held plate on top: every existing destination plate is below it.
z_release = position["z_release"] - sum(thickness(p) for p in stack)

# Preserve the existing check, using the appropriate base for the operation.
valid = limits["max_stack_height"] <= z <= base
```

Do not sum all plates when picking up the top plate: the current formula uses
`num - 1`, not `num`. Existing discard passes `NUM_WASTE + 1`, so its multiplication
already counts exactly the plates currently underneath the incoming plate.
Empty destination sums to zero. An empty source has no top plate to pick up.
The selected object's `gripping_z_offset` is applied separately by grab/release,
as today; do not fold that offset into the thickness sum or apply it twice.

The limit is equivalent to `sum_beneath <= base - max_stack_height` for
nonnegative thicknesses. It therefore continues to constrain mixed-stack height
without modification. Preserve the different grab/release bases.

Numerical check with current definitions: picking up the top of a source stack
`[96 WELL PLATE, 48 WELL PLATE, 24 WELL PLATE]` gives
`85500 - (6400 + 9500) = 69600`, above the 20000 limit. Releasing a plate onto
that existing waste stack gives `83000 - (6400 + 9500 + 9500) = 57600`.

Verification during planning: extracted and executed the actual existing
`calculate_wp_stack_height` method without hardware initialization. Compared its
result with the discrete-sum equivalent for three types, two stacks, both
operations and counts 1-15: **180 comparisons passed**, including invalid-height
cases. Confirmed `z == 20000` passes and `z == 19999` fails. These are arithmetic
checks, not claims of new hardware calibration.

Use current per-type thickness semantics. Moving a lidded plate remains normal
plate handling; do not invent a lid-height addition from `lid_gripping_z_offset`,
which is only for handling a detached lid. This refactor does not require a new
lid-specific stack model. If an actual geometry issue is later observed, address
that concrete case separately instead of imposing speculative restrictions.
`quartz.thickness` is currently null and the old method treats it as zero;
report a missing numeric thickness only when it is needed as a supporting plate
in a sum. A lone quartz plate, or quartz as the top target over defined supports,
does not require its own thickness for that operation.

Validate the stack transition requested by each command, including lid state
and actual height/clearance limits. A workflow may optionally preflight a known
source/waste sequence, but direct commands do not require a complete future
workflow or a predicted waste order.

## Migration and implementation sequence

1. **Inventory model and fixtures.** Add versioned parsing, invariants, queries,
   derived counts, atomic persistence and transfer stages. Exercise multiple
   types at multiple positions without importing hardware drivers.
2. **Explicit legacy conversion.** Convert source/waste counts into generated
   IDs of the recorded type and materialize the one active plate at its recorded
   position. Preserve pose; normalize known location aliases. Require review
   because the old file cannot prove other positions are empty, whether a lid
   is present, or physical identity. Unknown/ambiguous positions block conversion.
   Back up the original; no automatic live-state rewrite merely on import.
3. **Operator review.** Replace the single type/count form with per-position and
   stack editing, ordering and lid attachments. Preserve IDs on ordinary edits;
   mark physical replacement explicitly. Refuse duplicate object placement and
   invalid type references; coordinate edits affecting an in-flight transfer
   without banning unrelated corrections. Preserve unknown
   fields/version handling rather than silently downgrading the document.
4. **Track motion integration.** Resolve objects in grab/release, source/waste
   helpers and recovery; implement occupancy checks, progress recording and
   mixed-stack geometry for ordered source and waste contents, including
   applicable lid states. Validate each requested transition; optional workflow
   preflight must not be required for direct commands. Add a checked
   position-to-position transfer.
   Eliminate single-plate setters from plate and lid moves together. Migrate
   existing lid offsets into the flat parent plate fields and update lid callers;
   use the normal plate offset for plates with attached lids.
5. **Coordinator and reader integration.** Preserve the three simple commands,
   infer types, validate legacy expectations, return selected IDs from grab,
   bind sync/async measurements to IDs, with no return-slot reservation. Ensure tray close/open
   commands do not accidentally interpret the wrong or absent plate type.
6. **Pipetting integration.** Give `North_Robot` the shared inventory, resolve
   the active position once per operation, and migrate aspiration, mixing,
   serial/batched dispense paths. Validate before preparing liquid transfers.
7. **Workflow and scheduler consumers.** Migrate named guards, half-plate markers,
   handoff checks, usage summaries, private simulation copies, output IDs, manual
   utilities and tests. Keep experiment configuration as desired plate requirements.
8. **Remove compatibility state.** Temporary read-only count views are acceptable.
   Do not expose a synthetic global `CURRENT_WP_TYPE` or active position in mixed
   mode; ambiguous legacy access must fail clearly. Remove mutable legacy writes.
   Preserve old calls where their meaning can be resolved from the selected
   position or plate. Report genuinely ambiguous legacy access at that call;
   do not blacklist whole workflows merely because they use an older interface.
9. **Document and commission.** Update workflow construction, system capabilities,
   readiness docs and test instructions. Validate the current one-plate round trip,
   then multiple separately positioned plates and mixed source/waste sequences
   with calibrated combinations. Mixed-stack validation is a release requirement.

Develop and validate the combined migration against private state copies before
any live cutover. Intermediate commits must not expose a half-migrated schema to
live workflows. Roll back code and reviewed inventory together; never turn a
mixed v2 state into a guessed single-type v1 file.

## Acceptance scenarios

- A 24-well plate at Cytation, 96-well at the pipetting area and 48-well at source
  retain independent IDs/types across save/reload and private simulations.
- Moving any plate preserves all unrelated placements and objects. Counts remain
  equal to stack lengths; only top plates can be selected.
- A null measurement uses the selected plate's gripping offset, reader mapping
  and lid status, returns that same ID, and performs no read.
- A return to an occupied single-slot active area performs no pickup from Cytation
  and does not overwrite either plate. The operator can deliberately free it or
  choose a supported alternative; retry does not duplicate movement.
- Explicit wrong plate type/lid expectations and missing plate identity fail
  before action. Omitted type resolves correctly after changing plates.
- Pipetting uses the 48-well grid when that plate occupies the active area,
  reports a missing 24-well grid only if none is configured, accepts one once
  configured without an additional approval flag, and rejects an invalid well index on
  a 24-well plate before liquid handling. No automatic recipe volume scaling.
- Detaching/parking a quartz lid leaves its plate recorded at the original
  position; reattachment records both identities and reader lid state correctly.
- Plate pickup/release uses exactly the same plate offset with and without a
  lid. Detached lid handling resolves the flat `lid_gripping_z_offset` from
  the referenced plate definition, accepts zero, and rejects absent calibration.
- During an async read of A, B can be grabbed, pipetted into, have its lid
  handled, and moved out of the active area without a reservation override.
  Completing A returns that same plate. If B remains in the return position,
  completion reports only that conflict and succeeds on retry after B is moved,
  without repeating the read. A second completion does not duplicate motion.
- Inject failures before pickup, after gripping and after release, including
  persistence failures. Each leaves either a known placement or an explicit
  pending reconciliation record, with no automatic cleanup motion.
- A mixed source stack (for example, 96/48/24 wells, bottom to top) is consumed
  in physical top-first order with the correct type and pickup height each time.
- Those plates can be processed and accumulated into the same mixed waste stack
  without manual clearing between types. Each placement uses the actual ordered
  contents and the incoming plate's geometry; IDs and counts remain correct.
- Changing stack contents recalculates heights from the actual plates beneath
  the target. Reordering the same supporting plates does not change their sum;
  changing which plate is top can change the pickup height because that plate
  is excluded. Plate transport uses its normal grip offset with or without a lid.
- Homogeneous calibrated stacks preserve present geometry. Missing required
  geometry and excessive heights produce specific failures before motion;
  there is no unvalidated-pair whitelist or mandatory workflow preflight. One
  source plate and empty waste remain usable without inter-plate pitch data.
- A same-type replacement plate cannot inherit another plate's half-run marker.
  Cross-plate result joins do not confuse well 0 on two different plates.
- Scheduler handoffs preserve every occupied position and pending operation.
  Retained plates do not require a new ownership declaration merely to exist;
  the next command resolves its own plate and actual prerequisites. Simulation
  never changes live state, including attachments and operation records.
- Legacy conversion is reproducible on copies, preserves inventory quantities,
  requires review of ambiguity, and never silently discards v2 information.

## Recommended scope and deferred choices

Implement identity and per-position occupancy end to end, including lids and
existing scheduler/reuse consumers. Keep the catalog and existing motion routes.
Include mixed source and waste stacks and concurrent Cytation reading/pipetting
in the first release. Do not add default return-slot reservations or generic
capability restrictions beyond the operation-specific requirements above.

## Design scrutiny before implementation

The lid decisions, mixed-stack requirement and concurrent reading/pipetting
capability without unnecessary blocking are confirmed. The
remaining inventory design is proposed. Work through these decisions one at a time, recording
answers here before implementation:

1. **Mixed stacks in the first release — resolved: required.** Support both
   mixed source and mixed waste stacks without manual clearing between types.
   Determine and validate the correct height model during implementation and
   commissioning; do not substitute a homogeneous-only release.
2. **Simultaneous reading and pipetting — resolved: supported.** No duration-long
   return reservation, override ceremony or additional holder is required.
   The return command checks occupancy when it actually needs the destination.
3. **Lid interchangeability?** Proposed compatibility is one lid variant per
   plate type, with lids interchangeable within that type. No permanent ownership
   is recorded. Confirm whether that fits actual handling/contamination needs.
4. **Operator setup effort?** Stable IDs should be generated automatically.
   The GUI should offer adding a count of one type to a stack while storing
   individual IDs internally. Confirm which positions operators need to edit
   and how replacing a physical plate should be distinguished from correcting
   a mistaken location. Avoid making routine setup require manual ID entry.
5. **Compatibility scope?** Decide which active workflows must be migrated at
   cutover and which archived/manual utilities need call-site updates. Do not
   force the new inventory into a synthetic global type for old callers.
6. **Persistence scope?** Revision checking and pending transfers serve physical
   recovery. Do not add a general reservation system to control plate availability.
   Keep only coordination needed for actual shared-resource conflicts. Avoid building
   a general scheduler as part of this inventory refactor.

Use existing code and setup facts to resolve factual questions; ask the operator
for intended behavior and scope. Do not start implementation until this design
discussion has reached a shared understanding.

Decisions to settle before implementation: intended lid interchangeability and
operator setup/correction details. Mixed-stack height uses the verified discrete
sum above. Existing motion capabilities remain available; an untested operation is
not automatically forbidden. None of these prevent
saving this plan or designing the inventory correctly now.

Do not add automatic plate sorting, barcode readers, general deck planning,
per-well liquid inventory, arbitrary-position pipetting, or recipe rescaling to
this refactor. Stable IDs and the proposed inventory leave room for those later.

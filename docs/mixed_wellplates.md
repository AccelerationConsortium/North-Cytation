# Mixed wellplates

The track now records physical plates and lids at each position. Source and
waste are ordered bottom-to-top lists and may contain different plate types.
Plate movements infer type from the object at the source; explicit type/lid
arguments check an expectation instead of changing the recorded plate.

`wellplates.yaml` remains the type catalog. Separate lid grip offsets are flat
`lid_gripping_z_offset` fields on the plate definitions. A lid attachment exists
only while the lid is on the plate. Plates move using their ordinary plate
offset whether or not they have lids.

## Operator setup

The Track Status tab shows a clickable diagram of the holders, plate stacks,
lid storage and **Track gripper**. Colors identify plate types; plates are drawn
in physical stack order, with the top plate marked as the next pickup. Click a
holder or plate to select it. Use Add for each plate type, entering a quantity
where useful. The details list runs bottom to top; its last row is the next
pickup. Compact buttons below the details list provide Add, Edit, Remove,
Move to, Move higher and Move lower. Relocation preserves its ID.
Holders with capacity above one or no count limit start twice as tall as
single-item holders. Each holder grows further as needed to display its stack.
Attached lids appear as child rows; relocating that child parks the lid without
moving or deleting its plate. The gripper can contain a lid, a lidless plate,
or a plate with an attached lid.

Drag a plate between holders to record a relocation, or within/between stacks
to set its layer. The green insertion line shows where it will land. A plate
carries its attached lid; drag the lid's narrow top edge to move just that lid.
Standalone lids can also be dragged from storage to a plate or the gripper.
Full or unsuitable destinations highlight red and leave the setup unchanged.
Cancelling a drag leaves the original placement intact. Dragging edits the
draft only; use Save to persist it. It never commands hardware.

These edits describe the physical setup and command no motors. Save through
the main GUI. Existing legacy state is converted in memory for review; the first
save writes schema version 2 and retains a `.legacy.bak` copy of the original.
Do not replace the current live state with an example unless the physical setup
matches it. IDs remain stable across saves and measurement returns.

`max_wellplates` belongs to each holding position in `track_positions.yaml`.
Active area and Cytation default to 1. Source/waste currently use `null` for an
unspecified count limit, preserving height-based limits until their physical
count limits are established. Set these to known counts when available.

Recovery controls appear only when an interrupted transfer needs attention.
The message asks you to correct the displayed positions to match the physical
setup, then choose **Positions are correct** and save. Track coordinates are
available in a collapsed section; they do not dominate the ordinary editor.

## Source-stack transport test

From the repository root, using the robot's Python environment:

```powershell
python tests/mixed_wellplate_round_trip.py
```

Record the plates actually at source, in bottom-to-top order. Any number or
combination is accepted; the script processes the initial source list top first.
Existing waste remains in place, and an empty source finishes without plate
movement. The normal per-command requirements still apply, such as space at
the active area and valid stack heights. There is no hardcoded two-plate guard.

For the original example, put **24 WELL PLATE on the bottom, 96 WELL PLATE on
top**. Lidless plates make the simplest test; no optical protocol is required.

The script defaults to simulation on first creation of its workflow config.
Normal startup review can select the intended mode; a saved config retains the
previously selected mode, so check SIMULATE before each intended live run.
Example inventory: [mixed_wellplate_track_status.yaml](../tests/fixtures/mixed_wellplate_track_status.yaml).

For each plate, the workflow executes:

```python
plate_id = lash.grab_new_wellplate()
lash.measure_wellplate(None)
lash.discard_used_wellplate()
```

In the example it processes 96 first, then 24. Each null measurement moves the plate into
Cytation and returns it without running an optical read. Final waste order is
96 bottom, 24 top, above any waste plates already present. The script checks identities and final positions and does
not issue automatic recovery moves on failure.

Pickup Z subtracts the sum of thicknesses beneath the top plate. Release Z
subtracts the sum of all existing destination plates. The existing
`max_stack_height` constraint is unchanged. With current geometry, the initial
96 pickup is `85500 - 9500 = 76000`; the final 24 waste placement is
`83000 - 6400 = 76600`, before the moving plate's grip offset.

## Concurrent plates and recovery

`measure_wellplate_async()` captures the specific plate ID. Another plate can
occupy the pipetting area while the reader runs. Clear the return position when
needed, then call `handle.complete()`. If it remains full, completion reports the
specific conflict before reader pickup and can be retried after clearing it;
the read is not repeated. Repeated successful completion commands no more motion.

Track state records transfer intent before motion and placement after gripper
action. An interrupted transfer requires recording the actual affected object's
position. Use the GUI's relocation/correction actions and the ?Positions are correct? button
when the displayed setup matches reality; unrelated inventory remains intact.
Startup no longer automatically discards a recorded held plate.

The robot and track share one inventory for pipetting type/grid resolution.
Pipetting requires an actual lidless plate with a configured grid. The 24-well
grid is still unconfigured; this transport test does not need it. Quartz still
needs a numeric thickness when it supports another plate; its own thickness is
not needed for a lone/top plate pickup. No combination approval registry or
whole-workflow preflight is required for direct transport commands.

## Motion regression and validation

The first implementation accidentally used the source pickup route for discard.
That omitted the `transfer_stack` approach waypoint and the initial gripper-close
step. Both are restored to match synced `origin/main` commit `9ba125e0`.

- Source pickup raises to `max_height` before horizontal travel.
- Discard from home closes the gripper, visits `transfer_stack` at its current
  safe height, then lowers to active-area transfer height and approaches it.
- Leaving the active area visits `transfer_stack` before rising to stack transfer
  height, matching the established route rather than reversing that sequence.
- Cytation load/return and both detached-lid routes retain their original motor
  and carrier command order.

`tests/test_plate_motion_sequences.py` compares full motor/pneumatic/carrier
traces with that fixed commit. Its mixed-stack case substitutes only the new
thickness sum into the original code and checks the entire two-plate cycle.
The focused checks include 13 differential motion comparisons, varied-source
workflow cases and offscreen GUI interaction tests.
These tests run with mocked hardware; matching commands does not establish
physical alignment. No live hardware was commanded by the agent during checks.

Run the focused checks in a Python environment with pytest and PySide6:

```powershell
python -m pytest tests/test_plate_motion_sequences.py tests/test_plate_inventory.py tests/test_plate_inventory_gui.py tests/test_wellplate_24_round_trip.py tests/test_scheduler_simulation_state.py tests/test_scheduler_simulation_runner.py tests/test_scheduler_live_runner.py tests/test_scheduler_simulation_gui.py tests/test_workflow_launch_contract.py tests/test_biotek_24_well_grouping.py -q -p no:cacheprovider
```

Broader fluorescence/SQ workflow tests also contain existing failures involving
liquid supply fixtures, validation-call arguments, expected vial positions,
cleanup order and output paths. Their failing test names were reproduced using
the pre-change HEAD sources in an isolated copy; they are not evidence of a new
plate path regression. The focused plate/motion/scheduler tests exercise this
change directly.

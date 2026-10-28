# 24-well plate readiness

Repository inspection, 2026-10-28. Software support is partial; physical
compatibility and calibration have not been established by this investigation.

| Area | Current support / missing work |
| --- | --- |
| Plate configuration | `robot_state/wellplates.yaml` defines `24 WELL PLATE`, 24 wells, and Cytation name `24 WELL PLATE`. Verify that exact definition in the installed Cytation/Gen5 setup for the actual plate. |
| Reader addressing | `biotek_new.py` supports four rows by six columns (A1-D6); a regression test covers row grouping. A real read still needs a compatible 24-well protocol. |
| Track transfers | Generic transfers accept the 24-well type. Grip offset is currently zero; verify pickup/release heights and seating at source, active area, Cytation and waste using the actual plate. Track positions are shared with other formats. |
| Stacking | Thickness is provisional (9500 robot units, copied from 48-well). Measure the actual stacking pitch before multiple plates. The test uses one source plate and empty waste, so stack spacing multiplies by zero. Base pickup/release heights still need checking. |
| Pipetting | `name_in_Locator` is null: no calibrated 24-well grid. Pipetting depth is provisional; radius and capacity are approximate. Calibrate and verify these before liquid handling. They are unused by the transport test. |
| Startup review | Added `24 WELL PLATE` to the Track Status selector. Grab/discard use the saved track plate type; measure uses its explicit `plate_type` argument. These must match. |

At inspection, `robot_state/track_status.yaml` described **96 WELL PLATE**, three
source plates, zero waste plates and no active plate. This is recorded state,
not confirmation of the physical setup. This investigation leaves it unchanged.

## Minimal transport test

From the repository root:

```powershell
python tests/wellplate_24_round_trip.py
```

The saved workflow config defaults to simulation. In the normal startup review,
select `24 WELL PLATE` under Track Status, set one source plate, zero waste,
and no active plate. Physically reconcile these values before a live run:
one lidless plate at source, clear active/pipetting area and Cytation tray,
empty track gripper, and empty waste stack. Verify grip/seating heights and the
reader plate definition before setting `SIMULATE: false` in the workflow config.

The automation is:

```python
lash.grab_new_wellplate()
lash.measure_wellplate(None, plate_type="24 WELL PLATE")
lash.discard_used_wellplate()
```

Route: source stack -> active pipetting area -> Cytation -> active pipetting
area -> waste stack. `measure_wellplate` includes the return from Cytation.
Its null protocol exercises tray loading/unloading in a live run, runs no
optical read, and correctly returns `None`. Discard means transfer to the waste
stack, not return to the source stack. There is no simple coordinator command
to return a used plate to source with source inventory bookkeeping.

The test checks plate type and initial inventory before its three commands,
then checks each command's resulting software state. Startup initialization
can itself move/home hardware before those checks. State checks and simulation
cannot establish physical alignment. On interruption or failure the workflow
does not issue recovery/discard moves; inspect and reconcile the saved state
before restarting.

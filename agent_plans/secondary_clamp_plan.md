# Secondary clamp integration plan

Date: 2026-10-21
Status: Investigation complete; implementation scope and hardware facts remain to be confirmed.

## Objective

Support the secondary clamp while preserving the main clamp as the default.
Avoid spreading optional location arguments throughout methods that can already
derive the correct station from a vial's recorded location.

Separate three concepts currently bundled together:

- Vial location.
- Clamp actuation.
- Balance access.

This investigation was read-only. No hardware was operated and no behavior was
validated on hardware.

## Current implementation

| Area | Current behavior | Implication |
| --- | --- | --- |
| Secondary location | `clamp_2` has gripper and pipetting coordinates in [Locator.py](../robot_state/Locator.py), plus a one-position definition in [vial_positions.yaml](../robot_state/vial_positions.yaml). | Reuse the existing location system. |
| Pneumatics | Main-clamp operations use the controller's built-in `open_clamp()` and `close_clamp()`. [test_pnematic_valve.py](../tests/test_pnematic_valve.py) switches output 6 alongside output 3. | Secondary actuation needs a robot-level interface; open/closed polarity is not established by this script. |
| Uncap/recap | `North_Robot.uncap_clamp_vial()` and `recap_clamp_vial()` in [North_Safe.py](../North_Safe.py) hardcode main-clamp location, actuator, and coordinates. | Generalize these central operations. No separate secondary recap coordinate currently exists. |
| Ordinary pipetting | `get_vial_location()` resolves coordinates from the vial's recorded location. | Already-accessible vials do not need a new location argument just to pipette at the secondary station. |
| Automatic preparation | `_ensure_vial_accessible_for_pipetting()` moves capped vials to the main clamp. | Automatic station selection is a separate behavior change. |
| Weighing | Aspiration and dispensing check location `"clamp"` to enable balance measurements. | Do not replace these checks with "either clamp." |
| Workflow staging | [surfactant_grid_adaptive_concentrations.py](../workflows/surfactant_grid_adaptive_concentrations.py) already includes `clamp_2` as a staging position. | Location support exists, but secondary capping and recovery are not wired through. |
| Manual controls | [robot_arm_control_gui.py](../tools/robot_arm_control_gui.py) controls the main clamp through the raw controller. | Review this surface if secondary manual controls are included. |

## Constraints and findings

### Cap ownership must determine automatic recapping

The following methods currently assume the main clamp:

- `return_vial_home()`.
- `grab_vial()`.
- `vortex_vial()`.
- `_perform_physical_cleanup()`.
- The final cleanup in `dispense_from_vial_into_vial()`.

Uncapping already records the cap's owning vial in `GRIPPER_VIAL_INDEX`, and its
DataFrame row supplies the vial location. Reuse that authoritative state rather
than adding a second mutable "active clamp" variable.

Recapping must verify that the selected station contains the vial whose cap is
held. Having a cap in the gripper is not sufficient authorization to recap an
arbitrary vial.

### Two clamps do not provide storage for two caps

The existing uncapping sequence holds the removed cap in the gripper and opens
the clamp after uncapping.

Two stations therefore do not automatically allow two closed-cap vials to remain
uncapped simultaneously. That would need cap parking or another handling
strategy, which is outside this plan unless explicitly requested.

Potential uses without cap parking include open-cap vial staging/pipetting and
sequential capping operations.

### Output 3 is shared

`North_Spin` in [North_Safe.py](../North_Safe.py) uses output 3 alongside outputs
10 and 11 for the spin-coater vacuum and lid. These operations independently
turn output 3 off.

Confirm before reusing that pattern for output 6:

- Which output 6 state opens versus closes the secondary clamp.
- Required output switching order and settling delay.
- What output 3 actually enables or selects.
- Whether switching output 3 affects another active pneumatic device.
- Whether pressure must remain applied to maintain either clamp state.
- Behavior on loss of pressure or controller restart.

Existing switching code is evidence of usage, not proof of independent operation.

### Tip clearance is a separate constraint

`is_location_safe_for_vial_movement_with_pipet()` currently permits the main
clamp, but not `clamp_2`.

Aspiration intentionally obtains a tip before uncapping to avoid cap/rack
interference. Secondary routing must account for this ordering; do not simply
add the station to the safe-location list without confirming physical clearance.

## Staged implementation

### Stage 1: Establish hardware facts and calibration

1. Confirm output 6/output 3 actuation behavior and interactions with the
   spin-coater lid and vacuum.
2. Calibrate secondary vial-grip, uncapping, and recapping poses.
3. Confirm approach/retraction paths and clearance, both with and without tips.
4. Confirm pipetting coordinates and minimum height.
5. Establish appropriate capping parameters for the secondary station; do not
   assume the main station's parameters are physically validated there.

Do not derive the secondary recap height from the main clamp's offset. The main
station already has distinct vial-grip and recap heights.

Exit condition: operator-confirmed actuation and coordinates. No automatic
routing enabled.

### Stage 2: Generalize central clamp operations

Keep a small interface on the existing `North_Robot` module:

- Named-clamp open/close operations, defaulting to `"clamp"`.
- A trailing keyword-only `clamp_location="clamp"` argument on existing
  `uncap_clamp_vial()` and `recap_clamp_vial()` methods, preserving existing
  positional argument behavior.
- Internal resolution of actuator and operation-specific coordinates.
- Explicit rejection of unsupported station names or missing required settings.

Use [robot_hardware.yaml](../robot_state/robot_hardware.yaml) for pneumatic
mappings and the existing location configuration for position references.

Preserve the main clamp's controller calls and existing cap settings. Do not
introduce a new class hierarchy or duplicate the complete cap/uncap sequences.
Resolve shared-output behavior before deciding whether a shared pneumatic helper
is necessary.

Exit condition: explicit secondary uncap/recap uses the correct actuator,
coordinates, and vial state while existing default calls remain unchanged.

### Stage 3: Make lifecycle methods location-aware

1. Update return-home, grabbing, vortexing, and recovery to recap using the held
   cap's owning vial and that vial's station.
2. Verify station occupancy and cap ownership before motion.
3. Review initialization: it currently opens only the main clamp before cleanup.
   Do not blindly open both clamps without considering interrupted state and
   confirmed pneumatic behavior.
4. Preserve DataFrame-based vial state and existing status persistence.
5. If manual secondary controls are included, route them through the same
   station-aware actuation behavior rather than duplicating raw output logic.

Exit condition: secondary uncap -> recap -> return-home and interrupted recovery
work without modifying an unrelated vial at the main station.

### Stage 4: Add automatic pipetting routing only if needed

Keep automatic staging at the main clamp by default.

If secondary routing is desired:

1. Add explicit selection through the shared preparation method and relevant
   public pipetting methods.
2. Define source and destination selection separately for vial-to-vial transfer.
3. Preserve location-derived pipetting for already-accessible vials.
4. Make transfer cleanup follow the vials involved, rather than "whatever is in
   the main clamp."
5. Define occupancy behavior explicitly; do not silently pick another station.
6. Retain tip-clearance restrictions until physically validated.

Do not generalize these operations merely because they contain a main-clamp
reference:

- Balance measurements.
- `dispense_into_vial_from_reservoir()`, which also relies on carousel geometry.
- `Lash_E.mass_dispense_into_vial()` in
  [master_usdl_coordinator.py](../master_usdl_coordinator.py), which uses the
  powder dispenser.

Requested weighing at an unsupported station must fail explicitly, not silently
return no measurement.

### Stage 5: Validate before workflow adoption

When implementation and validation work are authorized, cover:

- Existing calls without a station argument preserve main-clamp behavior.
- Explicit secondary operations select the correct outputs and coordinates.
- Empty station, wrong cap, invalid station, and missing calibration are rejected
  before actuation.
- Both stations occupied: only the intended vial and station change.
- Return-home, grabbing, vortexing, and interrupted cleanup respect cap ownership.
- Tip-clearance restrictions remain enforced.
- Balance, reservoir, and powder operations remain main-station-only.
- Shared output 3 interactions match the confirmed hardware behavior.
- Vial state and held-cap ownership are persisted consistently.

Use simulation to execute the real operation sequence with recorded actuator
calls. Do not skip automation calls in simulation, and do not treat unrestricted
mocks as proof that controller methods exist.

The existing [test_alternating_vial_transfer.py](../tests/test_alternating_vial_transfer.py)
provides relevant exchange/recovery coverage, but its cap operations are mocked
and are not proof of secondary hardware behavior.

Physical polarity, clearance, grip stability, and cap torque require supervised
hardware validation. Hardware utility scripts can actuate equipment at import;
do not import or collect them indiscriminately as simulation tests.

## Decisions deliberately left open

- Initial use case: explicit capping, open-cap staging/pipetting, automatic
  routing, or a combination.
- Shared pneumatic switching behavior and any required coordination.
- Whether secondary vial manipulation with a tip attached is physically safe.
- Whether secondary-specific cap parameters differ from the main station.
- Whether manual GUI controls belong in the first implementation.

## Recommended implementation seam

Generalize clamp actuation and cap/uncap operations centrally. Derive station
identity from existing vial state for lifecycle operations. Add explicit station
selection only where a caller is actually choosing a destination.

This keeps most existing callers unchanged and avoids a broad location-argument
refactor. Hardware facts and calibration are the first prerequisites, not the
number of pipetting methods to modify.

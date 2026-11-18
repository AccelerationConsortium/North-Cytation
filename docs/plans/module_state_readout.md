# Module state readout (plan)

Status: **built 2026-11-13** for the resources whose state is saved: `Lash_E.get_module_state()` in `master_usdl_coordinator.py`, settings under `module_state` in `robot_state/robot_hardware.yaml`, tests in `software_tests/test_module_state.py`. The other resources are `unknown` placeholders (see "State we don't save yet").

## Purpose

`lash.get_module_state()` reports the state of Lash-E's resources: anything that affects what the module can do next. It is the foundation for running resources at the same time (e.g. dispensing while the Cytation reads).

- **State** decides whether a resource can do its next action. **Inventory** (tips remaining, reservoir volumes, vial volumes, vial caps) is not state. An uncapped vial is a limit on that vial (`is_vial_movable`), not module state.
- State is reported **per resource**, not as one "ready" for all of Lash-E. A flow names the resources it uses and must leave those as it found them (the paper's "preserved domain").
- The readout only reads saved state. It never moves hardware.

## Resources

Every resource gets the same three common fields:

- `status`: `ready` / `not_ready` for movers and stations; `idle` / `busy` / `unknown` for instruments
- `in_use_by`: the flow or run ID that has claimed it, or empty. **The field is reserved now; claiming and releasing come with asynchronous operation.**
- `source`: `saved` (from state files) or `device` (read from the hardware). Some state can also be confirmed another way, e.g. vial positions by the deck camera check (`scheduler/deck_check.py`).

| Resource | Kind | State fields | Not ready / busy when | Source today |
|---|---|---|---|---|
| Robot arm | Mover | `gripper_holding` (whatever `GRIPPER_STATUS` says: Vial, Cap, and later tools such as a substrate tool or powder cartridge), `gripper_item_id` (vial name, or whose cap, or tool name), `tip_type`, `tip_liquid_mL`, `tip_source_vial`, `vials_not_home` | Holding anything, tip attached, or any vial not home | `robot_status.yaml`, vial CSV |
| Track | Mover | `gripper_plate`, `pending_transfer`, `location` | Holding a plate, a move that didn't finish, or not parked (`origin`) | `track_status.yaml` |
| Clamp, clamp_2 | Station | `vial` (or empty) | Occupied | vial CSV |
| Cytation | Instrument | `plate` (or empty); later busy reading | Plate in it (for now) | `track_status.yaml` |
| Heater | Instrument | `vials` (several positions), later temperature on/off | Unknown until it reports | vial CSV |
| Photoreactor | Instrument | `vials`, later running | Unknown | vial CSV |
| Spin coater (`lash.spinner`) | Instrument | later `substrate` loaded, lid open/closed, vacuum on/off, spinning | Unknown | not tracked yet |
| Carousel | Shared station | `position`: a name from `carousel_positions.yaml` (home, powder_dispense_mg) or a reservoir position from `syringe_pumps.yaml` | Not at home | not tracked yet (`move_carousel_to` doesn't save it) |
| Powder dispenser | Instrument | later running | Unknown | not tracked yet |
| Reservoir pumps | Instrument | later valve position, primed, dispensing | Unknown | not tracked yet (config in `syringe_pumps.yaml`) |
| Camera | Instrument | later capturing | Unknown | not tracked yet |

Pipetting-area, source-stack and waste-stack plates are reported but never make the track not ready: a plate on the deck between flows is normal.

## State we don't save yet

The readout can only report state that something saves. These are not saved anywhere today:

| Resource | State not saved | How it could be filled in |
|---|---|---|
| Carousel | Position | Save the commanded position in `move_carousel_to` (cheapest; likely first). The C9 may also report the axis position. |
| Heater | On/off, set and actual temperature | Save the commanded set point in `set_temp` / `turn_off_heating`; `get_temp` reads the device. |
| Photoreactor | Running, light and stir settings | Save the commanded settings; the Pico may report them. |
| Spin coater | Substrate loaded, lid, vacuum, spinning | Save commanded lid/vacuum/speed; substrate tracking comes with the substrate tool. |
| Powder dispenser | Running | Save when a dispense starts and ends. |
| Reservoir pumps | Valve position, primed, dispensing | Save the commanded valve position and priming. |
| Cytation | Reading | Comes with the parallel-read split of `measure_wellplate`. |
| Camera | Capturing | Save when a capture starts and ends. |

Two ways to fill these in: **save what was commanded** (cheap, but it is what the software asked for, not what happened), or **ask the device** (closer to the truth, needs hardware testing). Some state can also be confirmed by the deck camera.

## Open points

- The readout reports `gripper_holding` exactly as `GRIPPER_STATUS` says, so new tools need no readout change. A list of allowed gripper items (e.g. `gripper_items` in `robot_hardware.yaml`) can be added when the first new tool arrives. The track's parked location (`origin`) is in `robot_hardware.yaml` under `module_state`.
- The carousel is shared by the reservoirs and the powder dispenser, so it is its own resource. Its position needs to be saved when it moves before it can be reported.
- The spin coater holds substrates, not vials. Substrates aren't tracked yet; they arrive with the substrate tool.
- Errors per resource come with structured error records.

## Later steps it enables

- Flow tests: the state of the flow's resources before and after should match.
- Parallel Cytation: split `measure_wellplate` into "start read" and "collect read" (see `measure_wellplate_async` in the automation catalog to-do).
- Claiming resources (`in_use_by`) so two flows can't use the clamp or track at once.

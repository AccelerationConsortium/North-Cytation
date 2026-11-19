# Automation Catalog

This catalog sorts Lash-E's automation methods into **primitives**, **unit operations** and **flows**. The terms follow the SDL4 Integration Plan, which is based on Gottstein et al., *A foundational representation for an orchestrated lab* (ChemRxiv, 2026).

- **Primitive:** the smallest action that a single instrument or piece of software exposes. Primitives are specific to one device and do not, on their own, accomplish a scientific task.
- **Unit operation:** a combination of primitives that accomplishes one well-defined task, with clear inputs, outputs, and a clear success or failure result. It does not need to leave the module ready for its next task.
- **Flow:** a combination of unit operations (and primitives where needed) that completes a procedure and is state-preserving. When it finishes, the parts of the module it uses are back in a known, ready state, or, if something goes wrong, stopped safely.

**Writing a new workflow:** start from `workflows/workflow_template.py` and build it from the [flows](#flows) below; use unit operations only where no flow fits. See [Sample tracking](#sample-tracking-what-a-workflow-needs-to-do) for the three things every workflow should do.

## Sample tracking: what a workflow needs to do

In live runs every vial, well, reservoir load and powder cartridge load records what it holds at each transfer: materials, concentrations, a ± uncertainty (from the pipetting validation history, split by run) and the stocks it came from. The pipette tip carries it from source to destination. This is automatic; a workflow only needs to:

1. **Pass `liquid=`** (the name in `robot_state/liquids.yaml`) to every dispense. It selects the calibrated parameters and gives the ±; without it the ± is blank.
2. **Use vials that hold a stock** (`stock_id` in the vial file, set in the scheduler's Stocks tab and vial dialog, or by the agent). A vial with no stock and no record contributes volume but unknown contents.
3. **Call `lash_e.save_run_info(output_dir)`** once the run's output folder exists, so the config, run ID and sample-tracking files are saved next to its data.

Each live run saves `output/sample_tracking/<run_id>/` (start and end vial files and contents, and `well_contents.csv`), and a copy in the workflow's output folder when it was recorded. Live records: `status/container_contents.csv` (not in git); history: `status/transfers.csv`; stocks: `status/stocks.csv`; reservoir and cartridge loads: `status/sources.csv`. The code is in `experiment_tracking/sample_tracking.py` and `experiment_tracking/sample_contents.py`.

## Primitives

### North_Robot (`North_Safe.py`)

| Primitive | What it does |
|---|---|
| `move_home` | Park the arm and home the carousel |
| `home_robot_components` | Home all axes, pumps and carousel (startup) |
| `goto_location_if_not_there` | Move the arm to a named location |
| `move_rel_xyz` | Relative XYZ move |
| `pipet_aspirate` | Pump draws a volume (no motion) |
| `pipet_dispense` | Pump expels a volume (no motion) |
| `adjust_pump_speed` | Set the pump speed |
| `get_current_height` | Read the Z position |
| `open_vial_gripper` / `close_vial_gripper` | Open / close the vial gripper (output set in `robot_hardware.yaml`) |
| `open_clamp` / `close_clamp` | Open / close the vial clamp |

### Shared by all North classes (`North_Base`)

| Primitive | What it does |
|---|---|
| `read_scale` | Read the balance once; returns `(steady_status, mass_g)` |
| `read_steady_scale` | Wait for a steady reading; returns the mass (g) |
| `zero_scale` | Tare the balance |
| `move_carousel_to` | Move the carousel to a named position (`carousel_positions.yaml`) |

### North_Track

| Primitive | What it does |
|---|---|
| `open_gripper` / `close_gripper` | Open / close the track gripper |
| `origin` | Park the track |
| `move_through_path` | Move along named waypoints |
| `set_horizontal_speed` / `set_vertical_speed` | Set the track speeds (currently unused) |

### North_Temp

| Primitive | What it does |
|---|---|
| `set_temp` / `get_temp` | Set / read the heater temperature |
| `turn_off_heating` | Turn off the heater |
| `turn_on_stirring` / `turn_off_stirring` | Turn stirring on / off |
| `autotune` | Autotune the heater (currently unused) |

### North_Powder

| Primitive | What it does |
|---|---|
| `activate_powder_channel` | Select and home a powder channel |
| `shake` | Shake the powder cartridge |
| `set_opening` | Set the cartridge opening angle |

### North_Spin (spin coater; currently unused)

| Primitive | What it does |
|---|---|
| `set_speed` / `stop_spin` | Spin / stop |
| `open_lid` / `close_lid` | Lid pneumatic (output set in `robot_hardware.yaml`) |
| `turn_on_vacuum` / `turn_off_vacuum` | Vacuum pneumatic (output set in `robot_hardware.yaml`) |

### Biotek_Wrapper (Cytation 5, `biotek_new.py`)

| Primitive | What it does |
|---|---|
| `CarrierIn` / `CarrierOut` | Retract / extend the plate carrier |
| `run_protocol` | Run a Gen5 protocol |

### Photoreactor_Controller (`photoreactor_controller.py`)

| Primitive | What it does |
|---|---|
| `initialize_photoreactor` | Initialize the reactor board |
| `run_photoreactor` | Run a reactor at a set rpm, intensity and duration |
| `turn_on_reactor_led` / `turn_off_reactor_led` | LED on / off |
| `turn_on_reactor_fan` / `turn_off_reactor_fan` | Fan (stirring) on / off |
| `stir_reactor` / `turn_off_stirring` | Aliases for the fan on / off |

## Unit Operations

**Test:** compare the module's state at the start with its state at the end. A unit operation changes the state (for example, it leaves a tip on, a vial in the gripper, or a vial somewhere new). A flow ends in the same state it started in.

### North_Robot: tips

| Unit operation | What it does | State after |
|---|---|---|
| `get_pipet` | Pick up a tip | Tip attached |
| `remove_pipet` | Discard the held tip | No tip |

### North_Robot: vials

| Unit operation | What it does | State after |
|---|---|---|
| `grab_vial` | Pick up a vial | Vial in gripper |
| `drop_off_vial` | Put the held vial down at a location | Gripper empty; vial at the new location |
| `move_vial_to_location` | `grab_vial` followed by `drop_off_vial` | Vial at the new location |
| `return_vial_home` | Move a vial back to its home position | Vial home. This is the reset step that flows end with, not a flow itself. |
| `uncap_clamp_vial` | Remove the cap from the vial in the clamp | Cap held in gripper; vial open |
| `recap_clamp_vial` | Put the held cap back on the vial in the clamp | Vial capped; gripper empty |

### North_Robot: liquids

| Unit operation | What it does | State after |
|---|---|---|
| `dispense_into_wellplate` | Dispense into wells. Refuses bad inputs before moving: one volume (mL) per well, no missing or negative volumes, each well within `max_volume_per_well` (`wellplates.yaml`). Warns if the doses exceed the tracked tip volume; logs the plate type used | Tip empty (still attached) |
| `pipet_from_wellplate` | Draw liquid from a well | Tip full |
| `mix_well_in_wellplate` | Mix a well by pipetting up and down | Tip still attached |
| `prime_reservoir_line` | Prime a reservoir line | Line primed |
| `aspirate_from_open_vial` | Aspirate from a vial that is already open, with the right tip held. Refuses (before moving) a closed vial or the wrong tip; does not get a tip or move the vial. Can record the scale (`start_scale_recording`); returns the readings | Tip full |
| `aspirate_from_vial` | Combined unit operation: get the right tip, make the vial accessible (move to the clamp and uncap if needed), then `aspirate_from_open_vial`; with scale recording, analyze the mass (mass analyzer). Same arguments and behaviour as before the split | Tip full; vial may be left in the clamp, uncapped |

`dispense_into_vial`, `dispense_into_vial_from_reservoir` and `mix_vial` are not listed here; see [Neither unit operation nor flow](#neither-unit-operation-nor-flow-fix-later) below.

### North_Robot: tip conditioning and staging

| Unit operation | What it does | State after |
|---|---|---|
| `condition_tip` | Aspirate from and dispense back into a vial several times (moved here from `surfactant_grid_adaptive_concentrations.py`) | Wet tip attached |
| `stage_vials` | Move vials to staging positions before pipetting, with no tip held. Closed vials go to rack 0-5 (safe to move with a tip held, so they can go to the clamp and swap through it); open vials go to small-tip-safe positions (`small_tip_staging_rank` in `vial_positions.yaml`: rack 47, 46, 45, heater 2, 5, 8, rack 44, 43). Highest priority first; the caller decides the order (e.g. with `workflows/helpers/vial_risk_analyzer.py`). Remembers each staged spot: a staged vial leaving the clamp goes back there, not home. Both flows (`dispense_from_vial_into_vial`, `dispense_from_vials_into_wellplate`) call it | Vials staged (deliberate state change) |
| `return_staged_vials` | Return staged vials home; the reset for `stage_vials` | Vials home |

### North_Robot: measurement

| Unit operation | What it does | State after |
|---|---|---|
| `monitor_weight_over_time` | Record the mass over time | Unchanged (data recorded) |
| `start_scale_recording` | Record the balance in the background around one liquid step (settle, 2 s before, mark start/end, 2 s after; timings in `scale_recording`, `robot_hardware.yaml`). No readings and no waits in simulation. Used by `aspirate_from_vial`, `dispense_into_vial` and `dispense_into_vial_from_reservoir` | Unchanged (readings recorded) |

**Mass analyzer** (`pipetting_data/mass_analysis.py`, not robot code): turns scale readings recorded around an aspirate or dispense into a mass and stability summary, and saves the CSV and plot. Shared by `aspirate_from_vial`, `dispense_into_vial` and `dispense_into_vial_from_reservoir` (step 1 of splitting them, below).

### North_Track: wellplates

| Unit operation | What it does | State after |
|---|---|---|
| `grab_wellplate_from_location` | Pick up a wellplate | Plate in track gripper |
| `release_wellplate_in_location` | Put the held wellplate down | Plate at the new location |
| `get_new_wellplate` | Take a new plate from the source stack | Plate in pipetting position |
| `discard_wellplate` | Move the current plate to the waste stack | Plate in waste |

### Powder, Cytation and coordinator

| Unit operation | Class | What it does | State after |
|---|---|---|---|
| `dispense_powder_mg` | North_Powder | Dispense a target mass of powder (shake and weigh until reached; settings in `powder_settings.yaml`). Refuses a channel not listed in `powder_channels.yaml` before moving | Powder dispensed; returns the mass dispensed |
| `run_plate` | Biotek_Wrapper | Read a plate that is already loaded in the Cytation | Data produced; plate still inside |
| `move_wellplate_to_cytation` | Lash_E | Move a plate from the deck into the Cytation | Plate inside the Cytation |
| `move_wellplate_back_from_cytation` | Lash_E | Move a plate from the Cytation back to the deck | Plate on the deck |

### Neither unit operation nor flow (fix later)

These methods mix the task itself with flow-level sequencing and with data analysis, so they fit neither definition. A unit operation should hold the logic for doing one task safely (checks, heights, speeds, failure detection). The ordering of steps belongs in a flow, and analysis and saving belong in an analyzer. They are being split up one at a time (see below).

| Method | Why it is neither |
|---|---|
| `dispense_into_vial` | Can move and uncap the vial, and includes the same scale monitoring, analysis, saving and plotting. |
| `dispense_into_vial_from_reservoir` | Moves the vial, homes the robot and analyzes the mass within one method. |
| `mix_vial` | Built on `aspirate_from_vial` / `dispense_into_vial` and calls `dispense_from_vial_into_vial` (a flow), so it inherits the same problem. |

Done: the mass analysis and the scale recording are now shared (Mass analyzer and `start_scale_recording` above). `aspirate_from_vial` is now split (`aspirate_from_open_vial` above). Still to do: the same for `dispense_into_vial`. Original direction: a slim aspirate or dispense unit operation (task and checks only), a separate scale-monitoring unit operation (see `monitor_weight_over_time`) plus a mass analyzer, and flows for the sequencing. Keeping the current method names as flows built from those parts would keep existing workflows working.

## Flows

A flow ends in the same state it started in: no tip held, gripper empty, and vials back home. Inventory (volumes, tips used) changes as intended.

| Flow | Class | Steps | Ends with |
|---|---|---|---|
| `dispense_from_vial_into_vial` | North_Robot | (condition fresh tip if the liquid needs it) > aspirate > dispense > remove tip > recap > return vial | No tip, vials home |
| `dispense_from_vials_into_wellplate` | North_Robot | Input: a table (columns = source vials, rows = well indices, values = mL) and `liquid=` as one name or `{vial: liquid}`. Checked before moving: no missing or negative volumes, each well's total within the plate's `max_volume_per_well`, a liquid for every vial, a tip for every dose. Then: stage small-tip vials if needed > per vial and tip group: condition if needed, dispense > fresh tip per vial > return vials home | No tip, vials home |
| `vortex_vial` | North_Robot | grab > vortex > drop off at home (`return_location` can override). Refuses a time or speed outside `vortex_limits` (`robot_hardware.yaml`: 1-60 s, speed 1-99) before moving; logs the time and speed used | Vial home |
| `refill_vial` | North_Robot | Top a vial up to a target volume from a specified reservoir or source vial | No tip, vials home |
| `create_substock` | North_Robot | Add each component (reservoir additions first, then pipetted), then optionally vortex | No tip, vials home |
| `measure_wellplate` | Lash_E | robot home > plate into Cytation > read > plate back | Plate back on deck |
| `mass_dispense_into_vial` | Lash_E | vial to clamp (if not there) > take off a closed cap (powder goes through an open cap) > robot home > dispense powder > put the cap back > return vial. Needs `initialize_p2=True`; returns the mass dispensed | Vial home, capped as before |
| `run_photoreactor` | Lash_E | move vial > run reactor > return vial | Vial home |

Notes:
- `refill_vial` and `create_substock` take their sources as arguments. Each container's contents are now stored (see [Sample tracking](#sample-tracking-what-a-workflow-needs-to-do)), so picking a source automatically (for example, a vial with matching contents) is possible but not built yet; `refill_vial` raises `NotImplementedError` if no source is given.
- `create_substock` returns the vial home after each addition. Keeping the vial on the clamp to weigh every addition is a possible improvement.
- `vortex_vial` now returns the vial home by default. Previously it returned the vial to where it was picked up.
- `refill_vial`'s reservoir path depends on `dispense_into_vial_from_reservoir`, and the vial-to-vial flows depend on `aspirate_from_vial` and `dispense_into_vial` (see [Neither unit operation nor flow](#neither-unit-operation-nor-flow-fix-later)).

### Workflow-file functions now covered by North_Safe

Existing workflows have not been changed; new workflows should use the North_Safe methods.

| Workflow function (`surfactant_grid_adaptive_concentrations.py`) | Replaced by |
|---|---|
| `refill_surfactant_vial`, `fill_water_vial` | `refill_vial` |
| `create_substocks_from_recipes` | `create_substock` |
| `condition_tip` | `condition_tip` (unit operation) |
| `position_surfactant_vials_by_concentration` | `stage_vials` (unit operation) |
| `return_surfactant_vials_home`, `return_water_vial_home` | `return_staged_vials` / `return_vial_home` (unit operations) |

### To do

- **Staging for moving vials while a tip is held:** done 2026-11-18. `stage_vials` puts closed vials in rack 0-5 and both flows call it, so `create_substock` and `refill_vial` work with closed vials anywhere. Still to do: remove the old workarounds that temporarily rehome vials (`safe_pipet` in `Degradation_serena.py`, rack 4/5; `dispense_vial_to_vial` in `fluorescence_calibration_workflow.py`, rack 0/1), and move the hard-coded 0-5 limit (`is_location_safe_for_vial_movement_with_pipet`) into YAML.
- **Embedded validation.** `validate_pipetting_accuracy` and `validate_reservoir_accuracy` (`pipetting_data/embedded_calibration_validation.py`) are flows, but they also contain analysis, plotting, CSV saving and parameter updates. Separate the flow from the analysis and improve them.
- **`measure_wellplate_async`** (Lash_E) has not been used yet. Examine it later. It looks like a flow split into two halves: the start moves the plate in and begins the read, and `MeasurementHandle.complete()` returns the plate.

### Recovery scripts (not classified yet)

`tools/reset.py`, `tools/move_all_vials_home.py`, `tools/return_vial_home.py` and `tools/home_track.py` return the module to a known state after a problem. They relate to failure behaviour (a flow should stop safely if something goes wrong), which has not been reviewed yet. Decide later whether they become recovery flows or emergency-stop routines.

### Tip conditioning rule

Flows condition a fresh tip when the liquid has `condition_tip: true` in `robot_state/liquids.yaml`. The setting currently matches `refill_pipets` for every liquid; liquids without it are not conditioned. `dispense_from_vial_into_vial` conditions before its first aspiration with a fresh tip, so `refill_vial` and `create_substock` (which use it) follow the same rule. `dispense_from_vials_into_wellplate` conditions once per tip group, at the group's largest dose. Conditioning is not added inside `aspirate_from_vial`.

### dispense_from_vials_into_wellplate

This flow took over what `dispense_component_to_wellplate` (in `surfactant_grid_adaptive_concentrations.py`) did: staging, conditioning, and grouping doses by tip. The batched strategy and the `strategy=` argument were removed because batching is not compatible with the liquids in use.
- **Staging:** only vials that will use the small tip, are movable, and are not already in a small-tip-safe position. The vial that will be emptiest after dispensing gets the safest position; ties go to the lowest starting volume.
- **Tips:** within a vial, doses are grouped by resolved tip (smallest tip first, largest dose first) and the tip is locked per group. Each vial gets a fresh tip, because vials may hold different contents.
- **Staging:** `stage_vials` before the first tip: closed vials to rack 0-5 (they are uncapped and used in the clamp), open small-tip vials to small-tip-safe positions. The clamp is never a staging spot. Each vial goes home right after its own dispense.
- **Future option (now possible with the stored contents, not built; not needed yet, tip use is low):** for vials holding the same solute in the same solvent, optionally reuse one tip from most dilute to most concentrated, conditioning only once (as `dispense_surfactant_component` in `surfactant_grid_ailsa.py` does). Off by default. Reuse means a tip is held when the next closed vial moves to the clamp; staging already puts closed vials in rack 0-5, so that is safe.

`dispense_component_to_wellplate` and the workflows that call it are unchanged; new workflows should use the North_Safe flow and keep only the recipe-to-volumes preparation in the workflow.

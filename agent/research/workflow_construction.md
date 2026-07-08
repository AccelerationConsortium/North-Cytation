# Workflow Construction

Reference for building workflows on the North Robotics SDL platform.
See also: `system_capabilities.md`, `safety_chemical_compatibility.md`, `analysis_and_data.md`, `recommender_guide.md`

---

## Workflow Pattern

```
initialize Lash_E → validate vial CSV
→ prepare solutions / set conditions (substocks, set heater temp, powder dispense, etc.)
→ run reactions / process samples (heat, irradiate, mix, stir, etc.)
→ dispense wells → grab new wellplate
→ measure wellplate (track → Cytation → track back)
→ analyze results → recommender suggests next batch → repeat
```

Not all steps are present in every workflow. Simple screening workflows skip the reaction step entirely.

---

## Step-by-Step Reference

### 1. Initialize

```python
lash_e = Lash_E(
    vial_file="status/my_vials.csv",
    simulate=SIMULATE,
    workflow_globals=globals(),
    workflow_name="my_workflow"
)
```

- `Lash_E.__init__()` — `master_usdl_coordinator.py`
- Loads workflow config (YAML) if `workflow_globals` + `workflow_name` are provided
- `SIMULATE=True` for development — no hardware required
- Shows the vial manager GUI by default (`show_gui=True`); set `show_gui=False` for automated runs

### 2. Validate vial CSV

Validation is done interactively via the vial manager GUI, which opens automatically on `Lash_E` init (`show_gui=True`). Review vial positions and volumes there before proceeding.

### 3. Prepare solutions / set conditions

#### Liquid–liquid transfers (substock prep, serial dilution)
```python
lash_e.nr_robot.dispense_from_vial_into_vial(source, dest, volume_mL, parameters, liquid)
lash_e.nr_robot.dispense_into_vial(dest, volume_mL, parameters, liquid)
lash_e.nr_robot.mix_vial(vial_name, volume_mL, repeats=3)
```
- `North_Robot.dispense_from_vial_into_vial()` — `North_Safe.py`
- `North_Robot.dispense_into_vial()` — `North_Safe.py`
- `North_Robot.mix_vial()` — `North_Safe.py`

#### Set heater temperature (pre-conditioning / equilibration)
```python
lash_e.temp_controller.set_temp(target_temp, channel=0)
lash_e.temp_controller.turn_on_stirring(speed=10000)
```
- `North_Temp.set_temp()`, `get_temp()`, `turn_off_heating()` — `North_Safe.py`
- `North_Temp.turn_on_stirring()`, `turn_off_stirring()` — `North_Safe.py`
- Use when pre-warming the block before moving vials onto it; for reaction heating see step 4

#### Powder dispensing
```python
lash_e.mass_dispense_into_vial(vial_name, mass_mg, channel=0)
```
- `Lash_E.mass_dispense_into_vial()` — `master_usdl_coordinator.py` (wraps `North_Powder.dispense_powder_mg()`)

### 4. Run reactions / process samples

#### Photoreactor
```python
lash_e.run_photoreactor(vial_index, target_rpm, intensity, duration, reactor_num=0)
```
- `Lash_E.run_photoreactor()` — `master_usdl_coordinator.py`
- Moves vial to reactor, runs, returns vial home automatically
- Light colour is hardware-selected before the experiment; cannot change mid-run

#### Heating / stirring
```python
lash_e.nr_robot.move_vial_to_location(vial_name, 'heater', location_index)
lash_e.temp_controller.set_temp(target_temp, channel=0)
lash_e.temp_controller.turn_on_stirring(speed=10000)
# ... wait for reaction ...
lash_e.temp_controller.turn_off_heating()
lash_e.temp_controller.turn_off_stirring()
lash_e.nr_robot.return_vial_home(vial_name)
```
- Vial must be physically moved to `'heater'` location before `set_temp` / `turn_on_stirring` will have any effect
- Temperature feedback is block-based — allow equilibration time before assuming vial is at setpoint
- Call `turn_off_heating()` and `turn_off_stirring()` before returning the vial

#### Vortex
```python
lash_e.nr_robot.vortex_vial(vial_name, vortex_time, vortex_speed=70)
```
- `North_Robot.vortex_vial()` — `North_Safe.py`

#### Move vials (general)
```python
lash_e.nr_robot.move_vial_to_location(vial_name, location, location_index)
lash_e.nr_robot.return_vial_home(vial_name)
```
- `North_Robot.move_vial_to_location()` — `North_Safe.py`
- `North_Robot.return_vial_home()` — `North_Safe.py`

### 5. Dispense wells

#### Dispense from vials into wellplate

**Option 1 — composite (high-level):**
```python
lash_e.nr_robot.dispense_from_vials_into_wellplate(well_plate_df, vial_names, parameters, liquid)
```
- `North_Robot.dispense_from_vials_into_wellplate()` — `North_Safe.py`
- Concise; input is a structured DataFrame describing the full plate layout
- Low control — not used much in current workflows

**Option 2 — explicit (preferred):**
```python
lash_e.nr_robot.dispense_into_wellplate(dest_wp_num_array, amount_mL_array, parameters, liquid)
```
- `North_Robot.dispense_into_wellplate()` — `North_Safe.py`
- One liquid at a time; couples with `aspirate_from_vial` for multi-liquid layouts
- Allows specifying `liquid` type per dispense, which affects pipetting parameters — important when dispensing different solvents or viscosities into different wells

#### Mix a well
```python
lash_e.nr_robot.mix_well_in_wellplate(wp_index, volume_mL, repeats=3)
```
- `North_Robot.mix_well_in_wellplate()` — `North_Safe.py`
- **Caution**: pipet-tip mixing risks cross-contamination between wells if tip management is not careful
- Preferred alternative: include a plate shake step in the Cytation measurement protocol (no tips involved)

### 6. Wellplate management

```python
lash_e.grab_new_wellplate()
lash_e.discard_used_wellplate()
```
- `Lash_E.grab_new_wellplate()` — `master_usdl_coordinator.py`
- `Lash_E.discard_used_wellplate()` — `master_usdl_coordinator.py`
- Max 96 wells per plate; track count in workflow and rotate plates accordingly

### 7. Measure wellplate

```python
data = lash_e.measure_wellplate(
    protocol_file_path=["protocols/my_protocol.prt"],  # list of .prt files
    wells_to_measure=None,   # None = all wells
    plate_type="96 WELL PLATE",
    repeats=1,
    use_lid=False
)
```
- `Lash_E.measure_wellplate()` — `master_usdl_coordinator.py`
- Handles track movement to/from Cytation internally
- Returns `pd.DataFrame` with MultiIndex columns `(rep_protocol, wavelength)`, or `None` in simulate mode
- Multiple protocols can be passed as a list; each is run in sequence per replicate

### 8. Analyze + recommend

Analysis and recommender logic is workflow-specific. The general pattern:

```python
# Analyze: merge measurements into recipes, derive outputs
well_recipes_df = analyze(raw_data, well_recipes_df, output_folder, lash_e.logger)

# Save after analysis — before recommender call
well_recipes_df.to_csv(os.path.join(output_folder, f"results_iter_{iteration}.csv"), index=False)

# Recommend next batch
exp_df = well_recipes_df[well_recipes_df["well_type"] == "experiment"]
recs_df = recommender.get_recommendations(exp_df, n_points=BATCH_SIZE)
next_points = recs_df.to_dict("records")
```

- Analysis scripts live in `analysis/` — see `analysis_and_data.md` for data handling patterns
- Recommenders live in `recommenders/` — see `recommender_guide.md` for how to build and integrate them
- Filter out control wells (`well_type == 'control'`) before passing data to the recommender

---

## AI Agent Development Loop

Use this loop when building or debugging a workflow autonomously.

**CRITICAL: An agent must NEVER set `SIMULATE = False`. Hardware execution is a human decision only.**

### Step 1 — Generate the workflow
Write the workflow Python file in `workflows/`. Follow the patterns in this document and existing workflows. Set `SIMULATE = True` and `show_gui = False` at the top.

### Step 2 — Set up state files

**`robot_state/robot_status.yaml`** — reset tip usage to 0:
```yaml
gripper_status: null
gripper_vial_index: null
held_pipet_type: null
pipet_fluid_vial_index: null
pipet_fluid_volume: 0.0
pipets_used:
  large_tip_rack_1: 0
  large_tip_rack_2: 0
  small_tip_rack_1: 0
  small_tip_rack_2: 0
```

**`robot_state/track_status.yaml`** — set `num_in_source` to the number of wellplates the workflow needs:
```yaml
active_wellplate_position: null
num_in_source: <N>   # how many plates the workflow will use
num_in_waste: 0
wellplate_type: 96 WELL PLATE
```

**Vial CSV** — the specific file passed to `Lash_E(vial_file=...)` in the workflow. Set volumes to reflect a realistic starting state:
- Vials that should be empty: `vial_volume = 0`
- Vials that should have liquid: set to a reasonable volume below the vial's maximum capacity

### Step 3 — Run in simulation
```powershell
python workflows/my_workflow.py
```
The workflow must have `SIMULATE = True` and `show_gui = False`. The log file will be written to `logs/`.

### Step 4 — Read the log and iterate

After the run completes, find the most recent log:
```powershell
Get-ChildItem logs/ | Sort-Object LastWriteTime | Select-Object -Last 1
```

Scan for errors:
```powershell
Select-String -Path logs/<logfile> -Pattern "ERROR|Traceback|Exception"
```

Diagnose the error, fix the workflow, reset state files, and re-run. Repeat until the simulation completes without errors.

If after several iterations the root cause is unclear, send a message to Slack via `lash_e.slack_agent.send_message()` describing the issue and stop — do not attempt further autonomous changes.

### What simulation does NOT catch
Simulation validates the control flow and method calls, but cannot verify:
- Correct volumes for the chemistry
- Correct vial assignments
- Scientific validity of the experimental design

These require human review before a hardware run.

---

## Vial Volume Management

Vial volumes deplete as the workflow runs. Two strategies:

### Refill from reservoir
- Keep a large reservoir vial on the rack
- Dispense from it back into the working vial when volume drops below threshold
- Simplest approach; reservoir must be prepared and loaded at the start

### Remake substock
- When a vial runs low, re-make the solution (from stocks) into the same vial or a fresh one
- More complex but avoids keeping large reservoirs on deck
- Preferred when the solution is unstable or expensive to pre-prepare in bulk

**Note**: Track `vial_volume` in the CSV; update it after each dispense. The robot uses this to plan pipetting — stale volume values cause aspiration errors.

---

## Configuration

### Vial CSV (`status/*.csv`)
Defines all vials on the robot. Required columns:

| Column | Description |
|---|---|
| `vial_index` | Unique integer ID |
| `vial_name` | Name used in workflow code |
| `location` | Current rack location |
| `location_index` | Position within rack |
| `vial_volume` | Current volume (mL) |
| `capped` | True/False |
| `home_location` | Where the vial lives when not in use |
| `home_location_index` | Home position within rack |

Validate with the vial manager GUI (`vial_manager_gui.py`) before running.

### Workflow config (YAML)
User-editable parameters are auto-saved to `workflow_configs/<workflow_name>.yaml` on first run.
Pass `workflow_globals=globals()` and `workflow_name=__name__` to `Lash_E.__init__()` to enable.

---

## Key Conventions

- **`SIMULATE = True`** during development — all hardware calls are mocked
- **Never use `print()` in workflow files** — use `logger.info()` or `lash_e.logger.info()`
- **No Unicode in log messages** — use `uL` not `μL`, `->` not `→`
- **Save at every step** — call `df.to_csv(...)` after each measurement, after each analysis, and after each recommender call; never only at the end (see `analysis_and_data.md`)
- **`original_vials_data` is the ground truth** for vial state; update it, don't rely on widget state
- Errors surface via `lash_e.nr_robot.pause_after_error(msg)` — logs, sends Slack, pauses for human

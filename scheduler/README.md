# Scheduler Internals

The user-facing window remains `scheduler_gui.py` in the repository root.
This folder contains internal scheduler support, not another workflow entrypoint.

## Per-Row Configuration

Each selected workflow row owns a private configuration snapshot, copied once
from `workflow_configs/<workflow>.yaml`. Repeated workflows have independent
settings. The queue has five columns; all configuration actions are in Setup.
Private snapshots use stable job IDs under
`temp/scheduler_setup/<sessionUUID>/<jobUUID>.yaml`, not row numbers.
Missing or invalid presets are reported on the affected row without importing
the workflow. Setup retries an unavailable initial preset when opened again.

Setup edits a separate draft. **Save and Return** applies that draft to this row
and saves its vial CSV, but never writes robot or track YAML. It does not overwrite the workflow
preset. **Cancel** or window close discards the draft, with a discard prompt for
unsaved edits. Saving a draft through the File menu is not a row commit.

The Configuration tab has two explicit preset actions:
- **Load Preset** chooses a YAML and confirms replacing the current draft.
- **Save to Disk** collects current editor values, writes configuration only to
    the chosen preset (confirming overwrite), and updates the draft while leaving
    Setup open. Other queued rows do not change. A later Cancel discards the draft
    but does not undo an explicitly saved disk preset.

Moving a row preserves its job identity, snapshot, preset provenance and reports.
Changing its workflow initializes a fresh snapshot; loading or applying settings
clears live approval. External preset edits never automatically update existing
rows. Relative resource paths resolve from the repository root, not the preset
or snapshot directory. Shared vial CSVs are read once and remain evolving
physical inventory, not per-row resets. Robot/track tabs belong to the scheduler,
not the row-owned Setup window. Row preparation never creates hidden shared-state editors.

Occupancy, simulation and live preparation all read private row snapshots.
Fingerprints cover ordered job identities/workflows/snapshot paths and their
contents, plus referenced vial/base inputs and shared state. Preset provenance
alone is not fingerprinted. Job `input.json` contains the full effective config
and private `config_file`; runners do not reload the source preset. Private
snapshots are retained when removing rows, so execution records remain readable.

For programmatic preset loading, `window.set_row_config(selector, path)` copies
that preset into the row snapshot; `None` explicitly reloads the workflow default.
`window.row_config_path(selector)` returns the private path. The selector's
`config_source` property holds optional preset provenance; workflow identity
remains in `currentData()`. Configuration and queue edits are blocked while Setup
or a child process is open. Standalone vial-manager Save All behavior is unchanged.

## Small Tasks and Optional Vial Tracking

Each scheduler task still needs a matching workflow YAML and the common
`execute(config=None, show_gui=True)` entrypoint. Vial tracking is optional:
set `INPUT_VIAL_STATUS_FILE: null` explicitly for tasks that do not use vials.
A missing key, empty string or invalid path is still an error, not a request
to disable tracking. No empty/dummy CSV is created.

For a no-vial job, Setup shows config only;
the vial views are disabled and contribute no occupancy/conflicts. Simulation
still copies and carries shared robot/track state and saves end snapshots.
Robot state saves omit only the absent CSV. Live jobs keep the same explicit
null value and use actual shared status files. Vial-dependent operations require
a real vial CSV; opting out does not make those operations valid.

### Vortex Port Example

`tests/test_vortex.py` remains unchanged. Its scheduler counterpart is
`workflows/test_vortex_scheduler.py`, with
`workflow_configs/test_vortex_scheduler.yaml`. Both execute the same three
robot operations: vortex the selected vial, return it home, and move home.

The scheduler version adds module-level `SIMULATE`, `INPUT_VIAL_STATUS_FILE`,
`TARGET_VIAL` and `VORTEX_TIME`, a guarded script entrypoint, and the standard
reviewed/supplied config paths. Simulation defaults to True. Only the robot is
initialized; unnecessary track/reader initialization is disabled. GUI-reviewed
target/time values are read from `lash_e.workflow_config` after Lash_E returns. The scheduler calls
`execute(config=complete_config, show_gui=False)`. The original test uses
hardcoded call arguments and starts directly at import, with live mode by default.

Configuration selection belongs to the existing ConfigManager, called by
Lash_E. Workflow code passes its globals/name, optional config and show_gui;
there is no extra startup module or separate preparation/confirmation import.
Normal construction selects saved settings, reviews them and reloads before
controllers are created. Supplied config selects no-YAML automation and requires
show_gui=False. Plain legacy Lash_E calls without workflow globals/name continue
to use explicit vial_file and simulate arguments.

Vortex itself cannot be vial-less, because it must locate and manipulate a
physical vial. Use explicit null for tasks such as robot homing, not for vortex.
Config YAML is Git-ignored; the example script can generate its matching YAML
on normal launch on another computer, as other workflows do.

## Private Simulation State

`simulation_state.create_session(vial_files)` copies the current robot/track
status and selected vial CSVs into `scheduler/state/<session_id>/`. Generated
state is ignored by Git. Original inputs are never replaced or restored.

For each future scheduler child process:

```python
with workflow_state(session_root, job_id):
    config = simulated_config(session_root, saved_config)
    workflow.execute(config=config, show_gui=False)
```

The config copy forces simulation and points to the private vial CSV. The
session must be activated before the coordinator is constructed. Only this
opt-in path enables simulation state writes. Ordinary simulations still do not
save; ordinary live runs retain their existing state paths and save behavior.

The robot and track load and update the shared private status files. When the
workflow returns, its final controller values are saved and copied into
`end_states/<job_id>/`. The next job uses the UPDATED shared files, not a new
copy of the original starting state. Each workflow's private vial file remains
separate; identical source paths share one copy.

Exceptions retain their original traceback; a best-effort failed end-state
snapshot is diagnostic only. `result.json` reports whether the Python call
returned, not whether logged simulation errors occurred or the experiment is
physically ready. The runner must collect those errors separately and must not
launch dependent jobs from untrusted failure state. Do not rely on atexit for
the final export: the child runner must exit the context before it exits.

## Pending Integration

The GUI Simulate button launches one child at a time using the interpreter
running the GUI, saved config copies and `show_gui=False`. Simulation is forced
on in memory; saved configs are not changed. Child output is redirected to each
session's `jobs/<job_id>/console.log` and mirrored to the launching terminal every
100 ms for both simulation and live jobs. Output uses UTF-8; the final log tail
is mirrored before advancing the queue. The child still writes directly to its
log, so a closed terminal does not interrupt recording. There is no output-pipe accumulation and no
runtime, inactivity or estimate-based timeout. Ordinary per-experiment logs
and run tracking still operate.

## Live Run

Run is enabled only after every queued job completes simulation with zero
ERROR/CRITICAL records and a valid handoff, and the current vial layout has no
conflicts or input issues. Warnings alone are advisory. Input fingerprints cover
the queue order, saved configs, vial files, robot/track/hardware YAML, workflow
scripts and existing file-valued config inputs. Changes require re-simulation.
Approval is local to this window and is consumed when starting a live queue.

Run requires explicit real-hardware confirmation. The scheduler copies the
saved config into each live job, sets `SIMULATE=False` in that copy and invokes
`execute(config=..., show_gui=False)` using original vial/state paths, never
the simulation's final state. Original config YAML is not rewritten. Each live
child retains the workflow's normal logging and run tracking. Live job reports
and console output are in `scheduler/state/live_<id>/jobs/<job_id>/`.

The same queue/process controls and yellow/green/red rows apply. Any live error,
exception, failed/missing result or invalid handoff prevents launching the next
job, including a logged error followed by exit code zero. The active process is
not automatically killed. There is no runtime, inactivity or estimate deadline.
Stop After Current prevents later launches; closing while running is blocked.

The Windows file lock prevents overlapping scheduler child jobs, not separate
hardware scripts launched outside this scheduler. The operator must confirm
no other hardware workflow is running. Existing controller error pauses remain;
this window does not provide an operator-input terminal for answering those
pauses. It never automatically answers them. Tests use mocks/harmless children;
real hardware execution has not been tested on this development computer.

Notes show ERROR/CRITICAL and WARNING record counts separately. Double-click
the Notes cell for messages, tip/plate consumption and log/state paths.
Recoverable logged errors continue, including tip exhaustion using the existing
simulated refill behavior. Exhaustion remains an ERROR and the row finishes red.
Tip totals are marked unavailable after counter resets; no actual refill occurs.
Exceptions, unexpected input prompts, or dirty end state stop launching dependent jobs.
Stop After Current prevents the next launch without killing the active child;
the window cannot close while a child is running.

The scheduler includes Robot Status and Track Status tabs using the existing
vial-manager widgets. Each tab has explicit Save and Reload controls for
the shared starting inventory, including the source plate count and tips used.
These are global starting values copied once per queue simulation, not a reset
before each workflow. Pending shared edits immediately clear live approval and
disable both Simulate and Run until saved or discarded with Reload. Reload
confirms discarding pending edits. Saving or reloading requires a new simulation
before Run; launching never automatically saves shared state. Missing or unreadable
shared files block launches and cannot be overwritten through Save until Reload succeeds.
Shared controls and the queue are disabled during Setup and queue execution.
Live queue completion reloads shared editors only when there are no pending edits;
simulation leaves the saved lab state and editor baseline untouched.
Standalone vial-manager and non-row-owned preparation retain their original
robot/track tabs and Save All behavior.

Invalid files must be resolved before simulation. Current-position conflicts
offer an explicit simulation-only confirmation; accepting it records conflict
warnings and permits subsequent test jobs, without enabling live Run or ignoring
other state blockers. There is NO live conflict override. End-state checks reject held tips/liquid/vials/caps and
active plates. These are state checks, not a full
physical collision/trajectory simulation. Final state is not copied back into
the live Vial Layout view or real lab files.

Full scientific workflow behavior, instrument protocols, all shared helpers and
external dependencies still require review. A completed subprocess does not
certify a physically safe experiment, and logged simulation errors should be
reviewed even when a clean handoff allows the next simulation to proceed.

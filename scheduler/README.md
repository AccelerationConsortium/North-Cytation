# Scheduler Internals

The user-facing window remains `scheduler_gui.py` in the repository root.
This folder contains internal scheduler support, not another workflow entrypoint.

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
session's `jobs/<job_id>/console.log`, with no output-pipe accumulation and no
runtime, inactivity or estimate-based timeout. Ordinary per-experiment logs
and run tracking still operate. Live Run remains disabled.

Notes show ERROR/CRITICAL and WARNING record counts separately. Double-click
the Notes cell for messages, tip/plate consumption and log/state paths.
Recoverable logged errors continue. Exceptions, unexpected input prompts, tip
exhaustion, or dirty end state stop launching dependent jobs. Scheduler
simulation never resets tip counters as if an operator had refilled them.
Stop After Current prevents the next launch without killing the active child;
the window cannot close while a child is running.

Setup includes editable robot/track tabs. Save All / Save and Return persist
the shared starting inventory, including the source plate count and tips used.
These are global starting values copied once per queue simulation, not a reset
before each workflow. Shared-only edits participate in the unsaved-change prompt.

Invalid files must be resolved before simulation. Current-position conflicts
offer an explicit simulation-only confirmation; accepting it records conflict
warnings and permits subsequent test jobs, without enabling live Run or ignoring
other state blockers. End-state checks reject held tips/liquid/vials/caps and
active plates. These are state checks, not a full
physical collision/trajectory simulation. Final state is not copied back into
the live Vial Layout view or real lab files.

Full scientific workflow behavior, instrument protocols, all shared helpers and
external dependencies still require review. A completed subprocess does not
certify a physically safe experiment, and logged simulation errors should be
reviewed even when a clean handoff allows the next simulation to proceed.
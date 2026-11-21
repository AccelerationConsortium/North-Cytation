# North Robotics Laboratory Automation System

A self-driving lab: the North robot (pipetting, vial handling), North track (wellplate
transport), Cytation 5 plate reader and other instruments, coordinated by the `Lash_E` class
in `master_usdl_coordinator.py`. Robot control is in `North_Safe.py`, the reader in
`biotek_new.py`, hardware and state YAMLs in `robot_state/`, vial files in `status/`.

## Working with the user

- Be succinct: short answers, one point at a time; give more detail only when asked.
- Python: use the `north_async_test` conda env
  (`C:\Users\Imaging Controller\anaconda3\envs\north_async_test\python.exe`), run from the
  repository root. Other envs lack the North SDK (`No module named 'north'`). Don't create
  environments or install packages without asking.
- AI may run live hardware (reader, robot, `SIMULATE=False`), but only in a visible terminal
  window the user can watch, never hidden inside the AI's own tool. Never run all of
  `hardware_tests/` at once.
- Start from the real data (output files, logs), not assumptions. Say what is measured and what
  is assumed. Don't claim a fix works until it is verified.
- Prefer small, reversible changes; ask before large changes to working code.
- Don't create file clutter. Throwaway scripts and test files go in `ai_scratch/` (not in git).

## Writing a workflow

1. Copy `workflows/workflow_template.py`. Build the experiment from the flows in
   `docs/platform/automation_catalog.md`; avoid low-level pipetting calls in workflows.
2. Pass `liquid=` to every dispense, use stock vials with a `stock_id`, and call
   `lash_e.save_run_info(output_dir)` once the output folder exists (sample tracking).
3. Config: module-level UPPERCASE constants listed in `_CONFIG_KEYS`. Use only
   `lash_e.workflow_config` after `Lash_E` returns; don't preload or cache settings.
4. Don't hard-code positions, volumes or device settings; read them from `robot_state/`.

## Rules

- **Simulate first.** `SIMULATE=True` must run every automation step; the controllers stub the
  hardware. Never skip dispensing or moves in simulation. Skipping file output or substituting
  synthetic reader data is fine.
- **No silent defaults.** If a value feeds a calculation, report or decision, read it from its
  real source and fail loudly if missing. Never `x = a if a else 0.004` or `.get(key, guess)`.
- **Keep metadata in the DataFrame**, never in parallel arrays (filtering misaligns them).
- **Logging, not print**, in workflow code. ASCII only in log messages ("uL", "->", "+/-"):
  Unicode crashes the Windows console.
- **Slack:** workflow messages add information (parameters, progress, results); none in
  simulation; delivery failures are warnings, not errors. Use `_send_workflow_slack` from the
  template. Don't change the scheduler's started/finished/failed message format.
- **Errors:** use `pause_after_error(...)`; attempt hardware cleanup before reporting an interrupt.
- Don't make backup copies of files; git history is the backup. No files in the repository root. Update CHANGELOG.md only
  for significant user-visible changes.

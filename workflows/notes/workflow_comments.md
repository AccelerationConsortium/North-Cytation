# Workflow Comments

## Workflow Template Corrected

Updated on 2026-10-07. The earlier review found an out-of-date template; the
following corrections are now implemented in `workflows/workflow_template.py`:

1. Derive the workflow/config name from the script filename and refresh the
   local parameter dictionary after startup GUI review.
2. Preserve `execute()` with the normal `show_gui=True` startup. For explicit
  parameters, use `execute(config=..., show_gui=False)`. Pass config, workflow
  globals/name and show_gui directly to Lash_E, then read lash_e.workflow_config.
  ConfigManager owns config selection, required-key checking and snapshots;
  supplied config is independent and never merged with/read from saved YAML.
  The extra workflow-startup file and template initialization helper were
  removed on 2026-10-07. Workflow authors no longer preload config or pass a
  redundant simulate argument. Normal-run experiment validation follows review.
3. Remove deprecated `check_input_file()` calls and their terminal prompts.
   Validate vial-file existence, well count, simulation-mode agreement, and
  live protocol existence. Lash_E now chooses the startup vial path from config
  and uses the final reviewed path for controllers, rather than keeping a stale
  constructor argument captured before review.
4. Use `lash_e.logger` for workflow steps and failures.
5. Remove the nonexistent photoreactor `emergency_stop()` call. Independently
   attempt supported heater shutdown, stirrer shutdown and robot homing;
   report cleanup errors while re-raising the original error or KeyboardInterrupt.
6. Disable powder dispenser initialization in this temperature/plate example.
   If adding photoreactor steps, add supported per-reactor shutdown calls to
   cleanup; do not invent a general-purpose emergency-stop method.

Fourteen hardware-free regression tests in `tests/test_workflow_template.py`
cover GUI-edited values, filename/config identity, explicit-config behavior,
cancel, supported method names, full simulated body, cleanup failure, Ctrl-C,
caller-config isolation, missing keys and GUI correction before validation.
The template also sends live-only Slack start/completion/failure/interruption
updates using the existing safe sender. Simulation does not import Slack.
Notification failures are logged as warnings and cannot replace the experiment
result; failure/interruption notification is attempted after hardware cleanup.
Slack tests use mocks only and never contact the network.
Mocks restrict available methods rather than accepting arbitrary API names.
The template still requires a real vial CSV and experiment-specific protocol
and recipe edits before use. It is an example, not a ready experiment.

## Retained Workflow Verification

Reviewed on 2026-10-07 without executing hardware workflows. All five retained
workflow YAMLs contain valid `INPUT_VIAL_STATUS_FILE` paths. Static checks
resolved directly named controller methods and checked 189 explicit calls
against the current method signatures, with no name/argument mismatches.
Dynamic dispatch, helper-internal behavior and physical readiness are not
certified by these checks. Missing explicit `show_gui` is not an issue: the
coordinator defaults it to True. All five use the normal GUI startup path.

Remaining concerns and subsequent fixes:

- `surfactant_grid_ailsa.py`: startup now refreshes confirmed config after GUI
  review and accepts complete supplied config with `show_gui=False`. Launch
  changes were approved locally despite the upstairs fix; reconcile the
  `execute()` change when syncing with that version.
- `fluorescence_calibration_workflow.py`: corrected on 2026-10-07. Normal startup
  now completes GUI review, refreshes the config, and then builds the plan,
  recipes and protocol selection. Output is saved only after plan/protocol/
  inventory validation. Cancel returns without planning or dispensing.
  Config/controller simulation or vial-file mismatches raise before planning.
  Coordinator initialization itself still happens during `Lash_E` construction;
  this correction does not redesign that lifecycle. Its launch interface now
  accepts `show_gui=False` and bypasses YAML when config is supplied.
  Ten hardware-free tests now pass, including
  GUI-edited plan/recipes/protocols/channels and simulation mode, cancel,
  mismatches and invalid confirmed parameters. The stale `DEFAULTS` import was
  replaced with snapshots of current constants; tests use temporary files.
- `surfactant_grid_adaptive_concentrations.py`: failure results with
  `workflow_complete=False` can be printed and returned without raising.
  A zero process exit cannot reliably be treated as experiment success.
  Some messages use print instead of the experiment logger, so a future Notes
  report must not assume every failure is an ERROR logging record.
- `Degradation_serena.py`: launch moved into `execute(config=None, show_gui=True)`
  with a normal `__main__` guard. Importing no longer starts the experiment.
  Its missing external spectral analyzer is imported only during the existing
  post-experiment analysis step, not during startup. That analysis dependency
  still needs to be available on the computer running the real analysis.
- `surfactant_multidimensional_workflow.py`: uses a guarded entrypoint and
  module globals for startup configuration. Its shared adaptive helpers retain
  the failure-reporting concerns above. No full run was performed.

## Tracked Follow-Ups

- [ ] **Adaptive failure signalling - deferred by user.** An incomplete result
  (`workflow_complete=False`) can currently end with exit code zero. Before
  enabling queued execution, ensure the outer entrypoint logs/propagates this
  failure so the scheduler does not launch the next experiment. Leave current
  behavior unchanged for now; preserve simulation log-and-continue behavior.
- [x] **Launch compatibility.** All five retained workflows now expose
  `execute(config=None, show_gui=True)`. Normal script Run still opens the GUI.
  `execute(config=complete_config, show_gui=False)` uses supplied values without
  YAML reload/override. This is managed centrally by ConfigManager and Lash_E;
  legacy helpers receive supplied module constants too. Supplied config must
  contain all declared/detected config keys. The vortex example uses this same
  path without a separate startup import.
- [x] **Generic scheduler Setup.** Removed the temporary Ailsa-only restriction.
  The same matching-YAML/INPUT_VIAL_STATUS_FILE handler opens each workflow's
  editor without importing or executing it. Missing config/key/path is reported.
- [ ] **Ailsa upstairs merge reconciliation.** User approved local launch
  changes. Review the `execute()` section when merging the upstairs refresh fix;
  preserve its post-GUI refresh and the new config/show_gui launch interface.
- [ ] **Full execution validation.** Eighty-five focused hardware-free tests
  pass. Legacy launch tests execute the actual entrypoint in an isolated
  namespace and stop at a mocked first automation call; they do not certify
  full workflow imports, long-run scientific behavior, external dependencies,
  instrument protocols or physical readiness. Scheduler simulation and gated
  live Run are implemented; hardware validation is still required. Real vortex
  simulation is checked separately; no physical run was performed here.

## Archived Workflow Configs

Configs belonging to archived workflows are stored in `workflow_configs/archive/`.
Restore the corresponding config along with its workflow if bringing it back
into use. Vial CSVs were not moved. Test, calibration, and retained-workflow
configs remain in their original locations.

- `color_mixing.yaml`
- `enhanced_SP_arm_position_program_v1.1_xy_coordinate_system.yaml`
- `glycerol_dispense_baseline.yaml`
- `ilya_workflow_v2.yaml`
- `mof_synthesis_workflow.yaml`
- `peroxide_serena_v3.yaml` (belongs to the old peroxide workflow)
- `polymer_dye_kinetics_workflow.yaml`
- `surfactant_grid_replay.yaml`
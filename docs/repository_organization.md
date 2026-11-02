# Repository organization assessment and proposed plan

Date: 2026-11-02

## Root utility cleanup

On 2026-11-02 the operator explicitly approved:

- test_booking.py -> tools/book_robot.py. The utility retains its root secrets
  path; title/dates must be reviewed before running. No email was sent and no
  credential file was read. No caller/launcher reference surfaced.
- Retire the root experiment_run_logger.py duplicate. The coordinator imports
  experiment_tracking.experiment_run_logger; that newer implementation and its
  CSV were not changed. The retired root copy had no local edits or callers in
  the inspected source.
- Move both enhanced arm GUI log files into logs and update active plus archived
  GUI writers. Retain the contents rather than deleting operational log history.

The status rename is deferred at operator request. Powder settings, controllers,
main scheduler and vial GUIs remain in place; broader relocation is not bundled
into this low-risk cleanup.

## Local folder consolidation completed

The operator approved and implemented the remaining root-folder consolidation:

- LLM_prompts -> logs/LLM_prompts. Every prompt retained; both prompt writers
  and the calibration template now use this default. Explicit custom log paths
  remain authoritative.
- recovery -> pipetting_data/archives/recovery. Entire provenance tree retained
  and ignored/untracked. Rebased helper paths outside source_snapshots; dated
  manifests, CSV evidence and original source snapshots preserve their recorded
  historical paths rather than being rewritten.
- temp -> output/temp. Remaining artifacts retained. Scheduler setup scratch,
  arm GUI exports, an archived arm tool and the standalone GUI preview/helper
  scripts use the new path. Old scheduler history/config snapshots preserve their
  historical paths and are not rewritten or certified for replay.
- backups contained only compiled bytecode; removed that cache and empty folder.

All 2,824 retained files were SHA-256 verified immediately after relocation.
Subsequent helper changes are path-only; scientific evidence and prompt content
are unchanged. The 505 still-present tracked temp files were moved with git mv
and therefore appear as staged renames. Prior approved deletions remain separate;
this is not blanket untracking of temp. All other changes remain uncommitted.
The nested utoronto_demo environment fragment remains pending deletion approval.
Earlier assessments below retain old paths as historical context.

Verification: 34 scheduler-simulation/vial-CSV tests passed, the relocated
offscreen vial-fill assertions passed, and both real prompt-save methods were
checked in temporary directories for the new default and explicit custom paths
without network calls. Scheduler setup tests also expose an unrelated stale
num_source_spin assumption; the broad setup batch was stopped after hanging.
No hardware ran. Both staged and unstaged whitespace checks passed.

Status: organization proposal and retention decisions saved for review. Explicitly
approved obsolete programs have been deleted as recorded below. Test/tool
relocations described below were subsequently approved and implemented.
No scientific data, environment or experiment-entrypoint relocations are authorized.
Approved documentation consolidations, including the docs-root organization,
have been implemented as recorded below; experiment and controller locations
remain unchanged.

## Test and utility ownership (approved and implemented)

### Next folder-retention assessment (2026-11-02)

Initial source/filesystem assessment (subsequent approved move recorded below):

- inputs: ten tracked scientific recipe/parameter CSVs (4.3 MB). Active Ailsa
  configuration uses two; archived glycerol/calibration workflows use others.
  Keep tracked and retain paths rather than treating these as disposable output.
- other: 350 files (74.5 MB), all ignored/untracked. Primarily historical
  experiment results, calibration data and plots. Candidate for consolidation
  under output after approval; an archived reservoir workflow still names a path
  inside it, so relocation needs reference updates.
- nested utoronto_demo: only a local .conda environment, 982 files (38.1 MB),
  no tracked files. Not a second source checkout. Do not remove until external
  interpreter usage is checked and the operator explicitly approves.
- data_tracker: four ignored/untracked backup-integration files. Documentation
  describes Restic deployment; C:/restic/backup.ps1 exists. Scheduled-task lookup
  did not establish an active matching task. Keep pending backup ownership review;
  credential file contents were not inspected.
- backups: only one compiled Python cache file, no tracked source/data.
  Strong deletion candidate, pending explicit approval.
- recovery: 1,091 ignored/untracked provenance files (1,153.4 MB). Preserve
  pending the previously discussed consolidation; do not infer completion means
  all provenance is disposable.
- LLM_prompts: 852 ignored/untracked files (4.2 MB). Operator explicitly wants
  these retained.
- temp: 1,970 local files (42.4 MB), with 1,536 still tracked. Only its
  scheduler_setup subtree currently has a dedicated ignore rule. It mixes copied
  packages, backups, coordinate exports and a GUI assertion script: review those
  categories before blanket untracking or deletion.

Ignoring a folder does not remove existing tracked files from Git. No ignore or
index changes were made during this assessment.

Subsequent operator-approved consolidation: moved other into output/other,
using the existing singular output directory rather than creating outputs.
All 350 SHA-256 file hashes match before/after; files remain ignored/untracked.
Updated the archived reservoir-calibration export path. No results deleted.

Nested utoronto_demo investigation: only .conda remains, with a Python 3.10
environment history dated 2024-10-16. Its Lib/site-packages directory is absent.
It is not listed in the local Conda environments registry. The registered
repository-root .conda is a separate environment and must not be confused with it.
No running python/pythonw process was returned by the process query; desktop
launchers use north_async_test, and prior desktop shortcut inspection found no
nested-environment target. No tracked nested-environment path reference surfaced.
These checks suggest an abandoned environment fragment, not proof that no
external editor ever selects it. Retain until explicit deletion approval.

Temp investigation, still no deletion/untracking:

- Copied dependencies: calibration_ax_test (957 tracked files), pdf_review_tools
  (72 tracked, 132 local files). No references surfaced in the inspected source.
- Disposable-looking test artifacts: scheduler_setup (198 tracked, 573 local),
  simulation (57 tracked), calibration_pytest (one tracked), sr_35a0b4d6 (13
  tracked screenshots/config/hash artifacts), plus previews and bytecode.
- Historical backups: two Ailsa backfill folders (179 tracked files), merge
  resolution snapshots (21), output-reorganization source copies (12).
- Preserve/rehome candidates: coordinate JSON/text exports, extracted Gen5
  reference text, and check_vial_fill.py. The latter asserts real GUI rendering
  and capacity propagation and saves a screenshot; not merely a scratch print.

No temp dependency/reference surfaced in the inspected software/hardware tests,
tools, scheduler, robot_state, pipetting_data, workflows or analysis Python source.
Absence of those references does not establish that backups or coordinates lack
operator value. Ask separately before deleting or untracking these categories.

The operator subsequently explicitly approved deleting only calibration_ax_test
and pdf_review_tools. Verified no tracked local edits or filesystem links before
removing those two directories (1,029 tracked files; 1,089 local files including
ignored bytecode). Installed working environments were not touched. All other
temp categories and the nested environment fragment remain unchanged.

The following decisions supersede older provisional tests/tools/scripts proposals
and path listings in this dated assessment.

- **software_tests**: automated regression coverage, using mocks, temporary
  storage or simulation rather than operating equipment. Contains 57 test files,
  conftest and two fixtures, including regression tests of hardware diagnostics.
- **hardware_tests**: 31 explicitly operator-run equipment checks. Supporting
  simulation does not make an equipment diagnostic a software regression test.
  Several legacy programs move hardware or alter state on import. Do not collect
  this folder with automated runners.
- **tools**: reusable operator utilities and control GUIs. Moved camera capture,
  track homing, reset, all-vial homing, and emergency vial return here from tests;
  moved utilization analysis here from scripts and the track GUI here from root.
- **workflows**: experimental workflows, template and workflow-owned helpers.
  Existing scheduler diagnostic entrypoints remain in place pending the separate
  scheduler-discovery decision; this change does not redesign the scheduler.
- **scripts**: no remaining source files; use tools for reusable utilities.

Default pytest discovery and VS Code now select only software_tests. The root
pytest.ini is an operator-approved exception to avoiding new root files.
Explicit runner arguments can override discovery; never target hardware_tests.
Run operator programs individually from the repository root.

On 2026-11-02 relocated 98 files without further retirements, preserving the
untracked config-documentation regression test. Updated exact source/document
references, pathlib references, fixtures and cross-folder imports. The track GUI
and emergency vial-return utility resolve repository imports from their files.
Updated Desktop/lab_launchers/run_robot_gui.cmd to launch tools/track_control_gui.py.
Inspected desktop shortcuts and Start Lab.cmd: none directly targets a relocated
file, so those launchers do not require changes.

Pre-move hardware-free baseline: 98 passed and one existing plate-inventory test
failed because its constructed robot lacks logger. This issue is not repaired
as part of folder organization. No hardware diagnostic or GUI is launched.

Post-move verification: the same focused batch still has 98 passes and the same
one failure. Additional fixture/workflow checks have 64 passes and two vortex
mock failures; both also reproduce with the original tracked test and workflow
sources. Default collection finds 693 tests, all under software_tests.

## Documentation navigation

The docs root now contains only this overview. Use the existing subfolders:

- [workflows](workflows/): individual workflow and diagnostic guides, plus
  workflow-specific design notes.
- [platform](platform/): shared robot, calibration, scheduler and labware guides,
  including readiness assessments and instrument investigations.
- [plans](plans/): design and implementation plans. Completed plans remain in
  [plans/archive](plans/archive/); placement alone does not certify completion.
- [archive](archive/): historical workflow documentation.
- [papers](papers/): reference literature.

On 2026-11-02, with operator approval, moved the sixteen remaining subject-specific
root documents into those existing folders: eight workflow guides, seven platform
guides/investigations, and the calibration robustness plan. Updated relative links,
incoming document links and the vial-photo workflow's documentation pointer.
No new folders, further deletions or executable/configuration relocations were
needed. Existing document edits were retained; this was organization, not a
technical recertification of the guides.

## Goal and agreed constraints

Make the repository easier to navigate by giving each location a clear purpose.
Reduce competing destinations, not merely the number of visible files.
Thoughtful subfolders are acceptable, including a workflow-owned helpers folder.

- The top level of workflows contains actual runnable workflows and the template.
- Preserve all main workflows, GUIs, controllers, calibration and pipetting
  helpers, startup review, scheduler behavior, and scientific results.
- Audit the whole repository and save this plan before approving moves.
- Propose relocation of generated data, logs, backups, and environments, but
  obtain separate approval before relocating any of them. Do not delete them.
- Preserve existing uncommitted changes. Several workflows and the template
  already have edits; the planning-marker helper and its tests are untracked.
- No live hardware runs, credential access, package installation, or environment
  creation as part of organization work.
- No new generic misc, shared, or utils dumping ground.

### Retention before relocation

Latest approved manual-trial retirement (2026-11-02): removed
tests/surfactant_workflow_tester.py and its README_workflow_tester.md,
tests/Surfacant_Worflow_vial_movement_SP.py, and tests/SERENA_CONSTANTS.py.
The operator explicitly selected all four files for retirement after reviewing
their purposes, including the solvent observations in SERENA_CONSTANTS.
All four were tracked with no local edits. Reference searches found only
historical changelog entries and the companion README, not callers.
No hardware programs were run; regression coverage and scientific data remain.

The operator identified accumulated AI-generated, one-off programs that may
never be used again as a core source of clutter. Do not design permanent homes
for every existing script before deciding whether it deserves to remain.

Review programs as keep, investigate, or retire candidate before proposing
moves. Record purpose, known callers, standalone/operator use, external launch
references, unique functionality, and whether a one-off task is complete.
Absence of imports does not establish that a program is unused; references
between generated scripts do not establish operational value either.
AI authorship alone is not a retirement criterion.

The operator approves retirement decisions. No deletion, untracking, or
relocation follows automatically from a candidate classification. The proposed
tests/tools/scripts boundaries below remain provisional, not agreed folder names.

### First retention shortlist: maintenance programs

Source inspection only; none of these programs was executed. References were
searched in tracked files outside temp/logs, with an additional search of current
source, tests, and documentation. No external callers were found for these five
entrypoints in those searches; external scheduled tasks, shortcuts, and historical
references in excluded directories remain unchecked. Completion of their
historical tasks has not been established.

| Program | Actual purpose and side effects | Recommendation |
|---|---|---|
| scripts/_tmp_abort_scan.py | Scans completed hardware-run logs for operator abort messages; prints findings. Executes its scan on import. | Strong retire candidate: narrow investigation, not production functionality. Preserve if an unresolved investigation still needs it. |
| scripts/fix_simulate_mismatches.py | Repairs historical run-history simulation flags from logs and rewrites the run-history CSV when run. | Retire candidate after confirming the historical repair is complete. Not a routine workflow dependency. |
| scripts/backfill_ailsa_analysis.py | Recomputes analysis for exactly ten historical Ailsa studies. Default preflight is read-only; --apply backs up and replaces generated analysis. | Retire from everyday source navigation if the task is complete; preserve with study/recovery provenance rather than discard reproducibility evidence. |
| scripts/sync_glycerol_data.py | Copies campaign data to another checkout, switches branches, pulls, commits, and pushes. Uses machine-specific paths. | Investigate operator use; not safe to classify as disposable. It advertises repeated use after batches. |
| scripts/finalize_pipetting_recovery.py | Finalizes recovery ledgers, provenance snapshots, and review-only data candidates. Imports audit_pipetting_recovery; enrich_pipetting_recovery also imports that module. | Preserve as part of the recovery toolchain until provenance/retention is reviewed. Assess the chain together, not as independent junk files. |

Operator review: all five programs above are no longer needed. The operator
explicitly approved deleting the five scripts, and they were removed on
2026-11-02 after verifying that all five were tracked and had no local edits.
Historical versions remain available in Git. This decision covers these five
programs only, not the rest of the recovery toolchain or any data, logs,
snapshots, or results. The table records the assessment before deletion.

### Second retention shortlist: historical audits and backfills

Source inspection only; none executed. A tracked reference search outside
temp/logs found only the validation-backfill documentation reference noted below
(excluding this assessment). External/manual usage and task completion remain
unverified by source inspection. The table records the assessment before deletion.

| Program | Actual purpose and side effects | Recommendation |
|---|---|---|
| scripts/audit_completed_status.py | Explicitly labeled one-off audit of completed hardware runs for failure markers and suspicious endings. Prints findings; does not repair records. | Strong retire candidate if that investigation is complete. |
| scripts/backfill_experiment_runs.py | Explicitly labeled one-time reconstruction of historical run-history rows from old logs. Writes run history when executed. | Retire candidate if historical import is complete; normal runtime logging is separate. |
| scripts/backfill_environment_from_mqtt.py | Fills missing historical temperature/humidity in pipetting masters from sensor logs. Dry run by default; --apply backs up and rewrites both masters. | Retire candidate if historical repair is complete; confirm no recurring use first. |
| scripts/backfill_validation_decisions.py | Reconstructs accepted validation stages from preserved replicate order and corrections, verifies backup hashes, then rewrites measurements, sessions, and corrections. Referenced by docs/embedded_validation_decisions.md. | Retire candidate if migration is complete; update that documentation if deletion is approved without altering runtime validation logic. |
| scripts/enrich_pipetting_recovery.py | Enriches generated recovery audit records with logged-transfer and balance evidence. Writes only the audit directory; imports audit_pipetting_recovery. | Retire candidate if recovery work is complete, consistent with the finalizer's retirement. Its upstream audit script requires a separate retention decision. |

Operator review: deletion of all five programs in this second batch was explicitly
approved. Removed on 2026-11-02 after verifying they were tracked with no local
edits. Updated the validation-backfill documentation to describe the retired
script historically. Runtime libraries, audit results, and scientific data were
not changed; audit_pipetting_recovery remains pending separate review.

### Workflow boundary decisions confirmed during review

- Retain active workflows, the template, a helpers subfolder, and the existing
  archive within workflows.
- Keep the template on disk but hide it from runnable scheduler choices.
- Keep a way to run diagnostics through the scheduler after moving diagnostic
  scripts out of workflows.
- Proposed UI, not yet approved: an explicit Workflows / Diagnostics selector,
  defaulting to Workflows. The exact discovery and execution design still needs
  review; do not broaden discovery indiscriminately.

## Evidence and limits

The assessment used a shallow on-disk inventory, tracked-file counts,
representative documentation/source reads, and repository-wide searches for
references to the workflow helpers and diagnostics. Counts describe the working
copy on the assessment date, not permanent repository properties.

- There are 1,536 tracked files under temp and 159 tracked files under logs.
  Ignore rules do not remove files already tracked.
- The local logs directory has 6,745 immediate files; LLM_prompts has 284
  immediate subdirectories.
- Workflows has experiments, diagnostic entrypoints, helper modules, notes,
  archived workflows, and a temporary directory.
- Tests mixes automated tests and manually operated hardware scripts.
- The scheduler discovers every non-underscore top-level Python file in
  workflows. It therefore includes helper files and the template.
- Configuration and state locations are both code-relative and working-directory
  relative. Moving a script can change its calculated repository root.
- The root experiment logger and the experiment_tracking implementation both
  exist. Similar names are not evidence that either is unused.

This is a repository-wide folder-purpose assessment, not an exhaustive
file-by-file dependency or data-retention audit. No application functionality has
been validated by executing code in this documentation-only phase. Some output
subdirectories denied access during enumeration; their contents remain unaudited.

### Third retention shortlist: remaining maintenance programs

Source inspection only; none executed. Tracked references outside temp/logs and
this assessment show Cytation probe documentation and an importing regression
test. No other external filename references were found. No cross-script callers
were found among the remaining scripts. Manual/external usage remains unknown.

| Program | Actual purpose and side effects | Recommendation |
|---|---|---|
| scripts/audit_pipetting_recovery.py | Historical pipetting provenance audit; writes new review datasets and snapshots, not production sources. Contains historical output-relocation handling. | Retire candidate if the recovery investigation is finished, consistent with the previously retired enrichment/finalization stages. |
| scripts/build_experiment_config.py | Explicitly one-time generator of a calibration YAML with hardcoded experiment settings, including simulate=False. Runs generation at module scope. | Strong retire candidate; not the runtime configuration manager. |
| scripts/calibration_range_analyzer.py | Mines dispensing logs for historical volume ranges and produces plots/statistics/reports. | Investigate recurring analysis value; not required for normal pipetting/calibration execution. |
| scripts/cytation_xml_only.py | Connects to Gen5 and probes procedure XML without a plate read or carrier movement; saves diagnostic evidence. | Discuss before retirement: docs/cytation_labware_test.md provides active launch instructions and tests/test_cytation_xml_only.py imports the module. If retired, address those surfaces together, preserve historical result links, and leave the production reader untouched. |
| scripts/plot_partial_fluorescence.py | Explicitly one-off plotting command for incomplete fluorescence runs; calls workflow plotting functions and writes plots into a selected results folder. | Retire candidate if no longer used for interrupted/in-progress runs; retain the workflow plotting functions. |
| scripts/reset_vial_volumes.py | Unconditionally rewrites one specific multidimensional vial CSV at import/run, assigning 0 mL to dilution vials and 7.8 mL to everything else. | Strong retire candidate unless intentionally used as a physical refill/reset procedure. Removing the script must not change the vial CSV. |
| scripts/utilization_analyzer.py | Analyzes log-based equipment utilization and produces reports/plots; default analysis period is May 2026. | Investigate operator use; potentially recurring reporting, not a workflow/controller dependency. |

Operator review: deletion approved for six scripts in this batch and the
dedicated Cytation XML-probe regression test. All seven files were tracked with
no local edits and were removed on 2026-11-02. The XML-probe documentation now
describes historical evidence rather than advertising removed launch commands.
The operator did not select utilization_analyzer for deletion; it remains.
Workflow plotting functions, production configuration/reader/calibration code,
vial inventories, and all historical evidence remain unchanged.

### Fourth retention shortlist: manual debug scripts in tests

Source inspection only; no imports or execution. Existing docs/test_usage_audit.csv
was found and consulted as historical evidence, not treated as an authoritative
current inventory: it lists at least one missing input that now exists locally.
Older run records are backfilled/inferred and cannot prove present-day use.

| Program | Actual purpose and side effects | Recommendation |
|---|---|---|
| tests/debug_validation_scale.py | Reproduces one water/scale validation at 50 uL; creates output and initializes live hardware at module scope with simulate=False. | Retire candidate if the scale investigation is complete. Keep the production validation implementation. |
| tests/validate_debug_test_SP.py | Near-copy of the same investigation with a 10 uL target and different compensation option. Historical audit lists June usage. Initializes live hardware at module scope. | Retire candidate if that investigation is complete; do not treat as automated regression coverage. |
| tests/demo_move_rel_xyz.py | Homes and demonstrates a fixed sequence of relative XYZ moves. Initializes controllers at module scope without explicitly selecting simulation. Historical audit lists May usage. | Retire candidate if the demonstration is no longer needed. Removing it does not remove robot movement methods. |
| tests/test_slack.py | Sends one fixed Slack test message at module scope. Contains no test assertions. | Retire candidate if no longer used as a manual connectivity check; preserve production Slack support. |

No executable callers for these four were found in current code searches.
Historical run/audit references are retained. External/manual callers remain
unknown. The operator approved deleting all four scripts. Removed on 2026-11-02
after verifying that each was tracked with no local modifications. Shared
validation, robot motion, Slack functionality, historical data, and audit records
were not modified.

Two other inspected programs should not be swept into this batch:

- tests/test_biotek_24_well_grouping.py is an assertion-based regression test of
  current 24-well grouping, referenced in mixed-plate validation documentation.
  Keep pending any specific evidence that its behavior is obsolete.
- tests/Get_C110_Frame.py is a documented camera capture/detection utility using
  shared camera functions, not a software regression test. Its recurring operator
  value needs separate review; lack of run-history entries is not disuse evidence.

### Fifth retention shortlist: older cap and positioning trials

Source inspection only; no hardware scripts imported or executed. Reference
searches excluded historical logs, run register, usage audit, changelog, temp,
and this assessment to distinguish executable consumers from provenance.
No external executable callers were found for these six programs. The two
cap-script configurations contain only empty mappings. Historical references
and usage do not establish current operator requirements.

| Program | Actual behavior | Recommendation |
|---|---|---|
| tests/cap_decap.py | Repeated clamp uncap/recap of water, 25 times; runs at module scope with sample inventory. | Retire candidate: near-duplicate of capdecap. Preserve newer cap-cycle diagnostics and their regression tests. |
| tests/capdecap.py | Same basic clamp cycle, three repeats, and different controller initialization. | Retire candidate alongside cap_decap; do not claim exact equivalence with newer rack/single-vial diagnostics, which require different fixtures and routes. |
| tests/carousel_test.py | Fixed reservoir-1 to water-vial transfer of 3 mL; executes at module scope. | Retire candidate if this fixed dispensing trial is finished. Its CMC_double_input_5 inventory currently exists locally; an older audit's missing-file claim is stale. |
| tests/position_test_SP.py | Configurable vial movement through specific rack/heater safe positions, with optional homing and CLI simulation flag. | Review operator use; candidate for retirement after the positioning investigation, not a pure duplicate of pipette-depth testing. |
| tests/Pipette_testing_positions_SP.py | Temporarily changes tracked liquid volume to test tip reach at fixed positions, without pumping; restores volume on normal completion. Default is live mode. | Retire candidate if reach/depth trials are complete. Preserve production height calculation and positioning code. |
| tests/Heating positioning.py | Live-default pipetting reach/mixing trial at heater positions. Current map contains V3 at slot 11 despite the older introductory sequence. Runs at module scope. | Retire candidate if heater positioning trials are complete; not a temperature-controller test. |

Preserve rack_cap_cycle and single_vial_rack_cap_cycle, plus their dedicated
regression tests, pending separate retention decisions. Their GUI-reviewed
settings, fixtures, and sequences differ from the older clamp-only scripts.
The operator approved deleting all six trials and both empty cap presets.
Removed on 2026-11-02 after verifying that all eight were tracked with no local
modifications. No other inventory/configuration or data was changed. The newer
cap-cycle diagnostics and their regression tests remain.

### Sixth retention shortlist: analysis, manual trials, and temp

Source inspection only; no programs executed. Targeted tracked reference searches
found historical recovery-plan links to the two OAV analyses, but no current
executable callers for the nine candidates below. This does not establish
external/manual disuse or completion of any historical task.

| Programs | Purpose / evidence | Proposed review |
|---|---|---|
| analysis/check_correction_history.py and check_oscillation.py | Fixed water/SDS correction investigations; manually read the first ten CSV fields to accommodate a historical schema problem. Both run at module scope. | Retire if investigations are complete; leave runtime corrections and CSVs unchanged. |
| analysis/oav_calibration_analysis.py and oav_current_vs_best.py | Standalone water/SDS historical overaspirate analyses with fixed filtering/weighting assumptions. One creates plots; the other compares historical estimates with calibration CSV values. Linked from an archived recovery plan. | Retire if no longer used; annotate archived links if approved, preserve datasets and production estimators. |
| tests/SP_wellplate_moving.py | Fixed source-to-pipetting-to-reader-to-waste plate sequence; executes at module scope with SIMULATE=False. | Retire if this manual movement trial is no longer needed. |
| tests/Water_pipetting_test_SP.py | Fixed 400 uL water aspiration/dispense and scale trial, three repetitions, live initialization at module scope. | Retire if experiment is complete; preserve production pipetting/validation. |
| tests/test_photoreactor_SP.py | Fixed COM6 direct Pico fan-command investigation followed by controller fan trial, with real waits. Runs at module scope. | Retire if firmware-command investigation is complete; preserve firmware/controller and other diagnostics. |
| temp/inspect_gen5_known_wells.py | Live reader metadata inspection for three named protocols; saves XML evidence. | Retire if investigation is complete; leave saved evidence and reader code. |
| temp/gen5_protocol_metadata_probe.py | Single NewExperimentEx investigation using a timestamp-specific saved XML input. Imports Capture from the now-retired scripts/cytation_xml_only module. | Retirement candidate; currently cannot import its retired dependency. Do not repair/reintroduce that retired probe without a new requirement. |

The operator approved all four groups (nine scripts). Removed on 2026-11-02
after verifying that each was tracked with no local modifications. Updated the
archived recovery-plan links to describe the retired OAV analyses historically.
No scientific data, saved diagnostic evidence, production dependencies, or
temporary directories were removed.

Temporary material requires a separate retention decision:

- temp/calibration_ax_test: 957 tracked files; immediate children include copied
  Ax, BoTorch, GPyTorch and related package distributions. Check interpreter,
  launch and external path references before treating this as disposable.
- temp/pdf_review_tools: 72 tracked files, including pypdf and typing_extensions
  package distributions. No path reference was found outside temp in the targeted
  search; external use remains unknown. Do not delete as ordinary scratch data.
- temp/scheduler_setup: 198 tracked files in generated-ID directories; the current
  scheduler still creates setup sessions at this root. Need per-session active-use
  checks, not a blanket directory removal.
- temp/simulation: isolated calibration/simulation artifacts; inspect provenance
  and active sessions before retention decisions.
- temp/output_reorganization_source_backup and merge-resolution/backfill backups:
  historical source/recovery material. Existing data-retention approval gates apply.
- temp/check_vial_fill.py: actual GUI assertions plus a screenshot, not merely a
  throwaway script. Preserve pending comparison with maintained GUI coverage.
- Coordinate JSON/text files and previews: potentially operator-useful position
  exports. Their location alone does not justify deletion.

### Seventh retention shortlist: legacy spectral utility and reader/pump trials

Source inspection only; no imports or execution. No external executable callers
were found outside temp for the following five candidates. The measurement
preset is an empty mapping. Historical usage records do not establish current use.

| Program | Actual purpose | Recommendation |
|---|---|---|
| analysis/spectral_difference.py | Legacy file-based spectral-distance functions with optional plotting hooks. | Retire candidate if no longer used manually/externally. Keep spectral_analyzer, which archived color matching imports; similar function names do not prove interchangeability. |
| tests/measure_wellplate.py | Initializes controllers ten times and calls a 96-well measurement with protocol=None and three selected wells. Runs at module scope. | Retire candidate if this alignment/measurement trial is complete. |
| tests/test_biotek_measure_wellplate.py | Four controller sessions, each cycling four plates, with protocol explicitly set to None and live mode. Runs at module scope. | Retire candidate if no longer used; include its empty preset only with approval. |
| tests/test_biotek.py | Initializes controllers and performs one reader carrier-out/carrier-in cycle at module scope. | Retire candidate if manual check is no longer needed. |
| tests/test_c9_asp.py | Five direct syringe aspiration/dispense cycles at module scope. | Retire candidate if direct-pump trial is complete; preserve controller methods. |

The operator approved deleting spectral_difference, both measurement trials,
test_c9_asp, and the empty measurement preset. All five files were tracked with
no local edits and were removed on 2026-11-02. test_biotek was not selected and
remains. Production reader/pump code, spectral_analyzer, regression coverage,
and data remain unchanged.

Inspected but not proposed for independent deletion:

- analysis/cmc_exp.py and cmc_exp_new.py implement historical experiment planners.
  Multiple archived CMC workflows import them. Assess with those workflows as a
  historical dependency group, not as unused standalone analysis.
- analysis/cof_analyzer.py contains fixed May/June 2025 study processing, but an
  archived photooxidation workflow imports it. It also writes plots at module
  scope. Review that historical pair together; do not silently break the archive.

### Recovery consolidation decision

The operator agreed to keep the untracked recovery material but consolidate it
into another folder later, rather than keep a separate root-level destination.
No relocation is approved yet. Prefer assessing the existing
pipetting_data/archives location as its eventual home because these investigations
document historical pipetting-data provenance and repairs; the exact destination
still requires review.

Preserve investigation boundaries, reports, snapshots, row mappings, and unresolved
evidence. Before moving, check documentation and historical scripts containing
recovery paths, backup coverage, ignored/untracked status at the destination, and
whether any external process uses the existing location. Do not mix recovered
candidate datasets into active pipetting masters or duplicate the archive merely
to hide the original. This is a future consolidation, not deletion authorization.

### Eighth retention shortlist: manual component checks

Source inspection only; no imports or execution. No external executable callers
were found outside historical temp/logs for these candidates. Function-name
matches in GUI regression tests refer to different tests, not these scripts.

| Program | Actual behavior | Recommendation |
|---|---|---|
| tests/test_heating.py | Sets 40 C, starts stirring, reads temperature for ten seconds, then stops stirring. Heating-off call is commented out. Runs at module scope using a legacy vial path. | Retire candidate if this fixed temperature trial is obsolete; preserve temperature controller functionality. |
| tests/test_heater_stirring.py | Initializes controllers and only turns stirring off; the stirring trial is commented out. Runs at module scope. | Retire candidate if no longer used as an operator stop command. |
| tests/test_gripper.py | Toggles robot output 7 five times with half-second waits, without startup GUI review. Despite its names/comments, it does not vortex a vial. | Retire candidate if this fixed actuator trial is complete. |
| tests/test_gripper_arm.py | Initializes controllers without startup GUI review and calls lash_e.c9.open_gripper(). No assertions or automated isolation. | Retire candidate if no longer used; do not infer controller API compatibility without further investigation. |

The operator approved only test_heater_stirring and test_gripper_arm. Both were
tracked with no local modifications and removed on 2026-11-02. test_heating and
test_gripper were not selected and remain. Production temperature/gripper code
and GUI regression coverage were not changed.

Keep tests/test_pnematic_valve.py pending a separate operator-use decision. It
toggles outputs 6 and 3 for five seconds, is cited by plans/secondary_clamp_plan.md,
and the historical usage audit lists an October 21 run. That does not prove it is
still needed, but is stronger operational evidence than the other four. Do not
equate fixed output toggling with a validated actuator polarity or safe shutdown.

### Documentation consolidation completed

On 2026-11-02, with operator approval:

- Moved the three ongoing/partially implemented plans into docs/plans.
- Moved the implemented mixed-wellplate plan into docs/plans/archive and retained
  its software-only validation caveat. This is not a new hardware certification.
- Moved the already completed/archived pipetting recovery plan into that same
  archive. Its older pending statements remain historical, not reopened tasks.
- Moved both workflow notes into docs/workflows.
- Rebased relative document links and updated incoming links in current docs.
  Corrected three existing stale links in the moved surfactant refactoring guide.
- Preserved historical path references in the dated test-usage audit as evidence
  of the original audit, not a current navigation index.
- Preserved the recovery plan's local edits and all seven documents' content
  other than relocation links and the wellplate archive status.

The original plans and workflows/notes directories are removed only if empty.
No code, configs, data, GUIs, environments, or recovery material is moved.

### Agent material ownership consolidation completed

On 2026-11-02, with operator approval:

- Moved the six robot/platform guides from agent/research into docs/platform:
  system capabilities, workflow construction, analysis/data, recommenders,
  chemical compatibility, and the historical Ax simulation investigation.
- Moved new_experiment_agent_workflow into docs/operations in the sibling
  Desktop/lab_slack_agent checkout, updating its links and clarifying which
  checkout owns generated robot code versus agent proposals.
- Moved both literature JSONL files into that agent checkout's existing papers
  area, under literature_search/data. Verified both SHA-256 hashes after moving.
  The destination ignores papers content; these datasets are now local-only in
  that checkout unless separately backed up or explicitly tracked. No ignore
  policy was changed, and the files were not ingested into a database.
- Updated the incoming workflow-construction link and the repository overview.
- Removed only empty source research directories. agent/lab_slack_agent still
  contains its environment-example file, pending a separate ownership decision.

The reference guides retain their original technical content; relocation is
not verification or correction of their potentially outdated examples. No
application code, credentials, runtime configs, databases, or hardware changed.
The agent process guide is an untracked addition in the other repository;
cross-repository transfers are not an atomic Git rename or a commit.

### Documentation-root retention review

On 2026-11-02 the operator explicitly approved deleting five tracked, unchanged
historical artifacts: ilya_pipetting_log_handoff.json,
surfactant_project_implementation_checklist.md, test_usage_audit.csv,
overaspirate_review.md, and overaspirate_review.csv.

Removed those artifacts and updated the calibration plan to describe the retired
review snapshot historically, rather than instructing its restoration. Earlier
references to the test-usage audit in this assessment describe evidence consulted
before its retirement; it remains recoverable from Git history.
No scientific masters, corrections, runtime estimators, or operating guides
were removed. Other document consolidation remains subject to review.

## Protected operational core

Established by source-reference inspection on 2026-11-02. This is a conservative
protection boundary for cleanup, not certification that every listed program
works or proof that the operator regularly uses every entrypoint. No application
or hardware was executed to produce this map.

Protect these capabilities and their dependencies together. A file outside the
map is unclassified, not disposable. Retiring a protected component requires a
specific usage/dependency decision, not a bulk cleanup.

### Experiment entrypoints and construction

Retain all eight current top-level experimental/data-collection entrypoints:

- workflows/Degradation_serena.py
- workflows/fluorescence_calibration_workflow.py
- workflows/pipette_calibration_workflow.py
- workflows/SQ_Peptoid_LNP_test.py
- workflows/surfactant_grid_adaptive_concentrations.py
- workflows/surfactant_grid_ailsa.py
- workflows/surfactant_multidimensional_workflow.py
- workflows/vial_shuffle_capture_workflow.py

Keep workflows/workflow_template.py. Its Slack helper is imported by the peptoid
workflow; it is not merely an isolated copyable example.

Keep all five workflow support modules until their approved relocation:
interrupt_cleanup, two_run_plate, _planning_markers, _vial_shuffle, and
vial_risk_analyzer. The Ailsa and multidimensional workflows import the adaptive
grid workflow; Ailsa also imports fluorescence solvent planning. These workflows
cannot be independently retired or relocated based only on their launch history.

Workflow diagnostics remain pending a separate scheduler-access design; they
are not approved deletions just because they are excluded from this experiment list.

### Controllers, startup, and operator interfaces

Protect the coordinator, North_Safe, biotek_new, photoreactor_controller,
powder_settings, workflow_config_manager, vial_csv, and production Slack support.
The coordinator uses experiment_tracking/experiment_run_logger, not merely the
similarly named root file. Keep the root logger pending its own caller review.

Protect current main GUI entrypoints:

- scheduler_gui.py, including the scheduler liquid-review GUI and queue runners.
- vial_manager_gui.py, including plate-inventory widget/layout support.
- track_control_gui.py and robot_state/manual_track.py.
- tools/robot_arm_control_gui.py, including Locator and hardware definitions.
- sdl_pipette_calibration/tools/calibration_test_gui.py and the calibration
  subsystem's wizard/dashboard tools pending their individual operator-use review.

The calibration test GUI launches run_calibration and shap_analyzer by path.
The arm-control GUI uses its own parent directory to locate the repository,
hardware definitions, default vial input, temporary files, and logs. A move of
either GUI must account for these paths, not just Python imports.

Protect scheduler config-list handling, live/simulation/reset subprocess runners,
isolated simulation state, plate handoff checks, interruptions, console output,
liquid review, and environment analysis. They form one operational subsystem.

### Pipetting and calibration chain

Protect these runtime modules in pipetting_data:

- pipetting_parameters and pipetting_wizard: controller parameter lookup.
- dispense_recording and liquid_registry: recording and authoritative liquid data.
- embedded_calibration_validation and embedded_liquid_validations: validation
  invoked from current experiments/template.
- overaspirate_corrections, overaspirate_estimation, and validation_decisions:
  corrections, estimation, and accepted-stage/record persistence.

Preserve other pipetting modules, including reservoir_wizard and
import_calibration_run, until their direct and external usage is assessed.

Protect the calibration subsystem as a dependency chain, not as a folder full
of independently disposable scripts: experiment/config/data structures,
parameter constraints, optimization/constraint calibration, analysis, exporters,
visualization, external-data handling, wizard, validation runner, and protocol
loader. Optimizer and LLM paths can be selected conditionally. Protocols are
loaded dynamically by name from configuration; ordinary import searches cannot
prove that a protocol is unused. Protect protocol configuration and lookup
locations pending a dedicated audit.

### Scientific processing and recommendation

Current workflow references establish protection for:

- analysis/control_cmc_analysis.py, surfactant_grid_scoring.py, and
  complete_linear_contours.py.
- analysis/multidim_visualizer.py, plot_3d_interactive.py, and iteration_metrics.py.
- Delaunay triangle/simplex, Bayesian, gradient, and level-set recommenders,
  including their shared transition base and dependencies.
- tools/c110_camera.py and the configured external camera integration.

This list identifies observed entry dependencies; it is not an exhaustive
transitive import graph. Keep unreviewed analysis/recommender modules until their
callers and optional/config-selected use have been checked.

### Operational inputs and persistence

Keep paths and contents unchanged for hardware/state YAML, vial CSVs, saved
workflow settings, queue presets, plate-reuse markers, liquid definitions,
pipetting master/correction/calibration CSVs, run history, protocol resources,
camera fixtures, and external services. Preserve all existing logs/results,
recovery evidence, and environments, including data associated with retired
programs. Config-relative, working-directory-relative, and file-relative paths
currently coexist.

### Regression protection

Protect assertion-based coverage of current behavior: workflow launch/template
and confirmed configs; scientific planning/dispensing; calibration constraints,
simulation and validation; corrections, recording and transfer accounting;
vial parsing; plate inventory/motion/grouping; GUI contracts; scheduler execution,
handoff/reset/interruptions; and vial-capture planning/restoration.

Existing examples include test_workflow_launch_contract, test_workflow_template,
test_coordinator_workflow_config, test_fluorescence_calibration,
test_pipette_calibration_workflow, test_validation_decisions,
test_overaspirate_corrections, test_dispense_recording, test_plate_inventory,
test_plate_motion_sequences, test_biotek_24_well_grouping, test_track_control_gui,
test_scheduler_* and test_vial_shuffle_capture. Names alone do not prove safety
or quality; inspect each test before executing or proposing retirement.

### Next cleanup boundaries

1. Review remaining manual hardware experiments in small capability-based groups:
   older positioning trials, duplicated cap/decap checks, and historical
   wellplate/pipetting demonstrations. Do not mix them with regression-test review.
2. Before consolidating duplicates, compare the actual hardware sequence,
   fixtures, state effects, and operator purpose. Similar names are not enough.
3. Then complete the approved workflow folder boundary, preserving diagnostic
   scheduler access and validating nested-helper dependency fingerprints.
4. Assess unclassified analysis, documentation, archives, and temporary material
   separately. Do not use a protected-core inventory as a deletion allowlist.

## Folder-purpose map

Every destination below is proposed unless explicitly described as retained.
An unclear folder is an investigation item, not permission to remove it.

| Current location | Purpose / issue | Proposed disposition |
|---|---|---|
| Root Python files | Controllers, coordinator, GUIs, config management, logging, and a booking test | Keep production import names and launch paths initially. Assess the booking script and duplicate-looking loggers individually. |
| workflows | Runnable experimental procedures plus misplaced support material | Keep active workflows and template at top level; use helpers subfolder for genuinely shared workflow support. |
| workflows/archive | Historical procedures | Retain pending an archive-location decision; do not make archived code operational as part of cleanup. |
| workflows/notes (former) | Workflow documentation and design notes | Consolidated into docs/workflows on 2026-11-02. |
| workflows/temp | Unclassified temporary material | Inspect ownership and retention before proposing a move. |
| tests | Automated regression tests and manual hardware diagnostics | Keep automated tests and fixtures here; propose a diagnostics subfolder for manual scripts, after checking discovery and launch paths. |
| tools | Operator utilities and camera integration | Keep as the destination for reusable operator-facing utilities; do not mix in disposable maintenance scripts. |
| scripts | Audits, backfills, repair scripts, and some analysis/config utilities | Keep repeatable maintenance commands. Review one-offs and analysis tools individually before consolidation. |
| analysis | Scientific processing and reporting | Retain; distinguish reusable analysis from historical one-off investigations. |
| recommenders | Experimental-condition recommendation algorithms | Retain as a coherent scientific subsystem. |
| pipetting_data | Runtime pipetting libraries, authoritative CSVs, historical and quarantine data | Retain paths initially. Do not merge code/data or remove historical evidence before a dedicated dependency and provenance review. |
| sdl_pipette_calibration | Calibration engine, protocols, GUIs, configuration, and results | Retain subsystem and protocol layout initially. Assess internal output placement separately from code. |
| robot_state | Hardware definitions, live state, inventory logic, and GUI support | Retain all paths initially. State and code are mixed, but moving these first would create unnecessary risk. |
| status | Workflow vial inventories, older definitions, settings, and scripts | Retain inventory paths initially; classify scripts and legacy files before proposing destinations. |
| workflow_configs | Saved workflow settings, grouped presets, and recovered settings | Retain names and locations initially. Classify active, diagnostic, archived, and recovered presets without changing selected values. |
| inputs | Experiment recipes, grids, and input datasets | Retain; clarify boundary with vial inventories and calibration inputs. |
| experiment_tracking | Run-history implementation and authoritative run history | Retain paths initially; trace every reader/writer before any consolidation. |
| scheduler | Queue execution, simulation, review, reset, saved queues, and state | Retain. Saved queue and state paths are operational contracts. |
| agent | Remaining environment-example stub | Robot guides moved into docs/platform; agent process/literature moved to desktop agent checkout. Review the remaining stub separately. |
| .agents | Agent skills and their support files | Retain tool-discovery location. Not ordinary project documentation. |
| .github | Repository automation/instructions | Retain discovery location; update moved-file references when relevant. |
| .vscode | Local/editor configuration | Retain discovery location; inspect launch/interpreter references before changing locations elsewhere. |
| docs | User/developer documentation and reference material | Use as the single documentation home, with archive/plans subfolders only where useful. |
| plans (former) | Active design plans plus archived plans | Consolidated into docs/plans on 2026-11-02; completed wellplate/recovery plans are in its archive. |
| logs | Runtime logs and historical tracked logs | Keep writers unchanged initially. Propose retention/relocation separately; no deletion or untracking without approval. |
| output | Generated experiment and analysis results | Retain as the normal results destination. Audit existing _organization material before designing any new data migration. |
| LLM_prompts | Accumulated generated prompt histories | Treat as runtime/provenance material; propose archival placement only after checking producers and consumers. |
| other | Mixed historical datasets and investigations | Classify by provenance and active usage before proposing consolidation with output/archive. |
| temp | Scratch work, recovery copies, simulations, tool dependencies, and tracked history | Audit by subfolder; never assume everything here is disposable or unreferenced. |
| recovery | Recovery/audit evidence | Preserve; propose archival placement only with provenance and references intact. |
| backups | Local backup directory, currently showing only a cache subfolder | Verify contents before any cleanup proposal. |
| data_tracker | Local data-backup integration | Retain until its external scheduling/deployment references are understood. |
| photo-reactor | Device firmware and device-side test program | Preserve separately from host controllers; verify deployment instructions before renaming or moving. |
| .conda | Local Python environment | Do not move. Environments and external launchers may embed absolute paths. |
| utoronto_demo | Nested directory containing another .conda directory | Determine interpreter ownership and external use; do not treat as a redundant checkout. |
| __pycache__, .pytest_cache | Generated Python/test caches | Not source. Exclude from navigation where appropriate; no cleanup needed to reorganize source. |

## First proposed source changes: workflows

Keep the existing workflow filenames, execute contracts, template, and associated
config names. SQ_Peptoid_LNP_test is an experiment despite its name; retain it.
Vial shuffle capture is a real data-collection workflow; retain it.

### Diagnostic entrypoints

Propose moving these to tests/diagnostics:

- test_booking: no external reference found in the initial search. Also inspect
  the separate root booking script before deciding whether it duplicates this.
  Its secrets lookup is relative to its own file; a move must not relocate or
  expose credentials, or silently change where they are loaded.
- ailsa_clamp_test: named in the config-documentation test exemption list.
- cytation_labware_test: imported by regression tests and referenced by docs.
- test_vortex_scheduler: used by contract tests, scheduler documentation, and
  repository instructions. Preserve scheduler diagnostic execution when moving
  it; removing that capability is not approved.

Update import-based tests, direct launch commands, config references, and
documentation in the same change as each move. Retain the automated regression
tests at their existing locations initially; inspect recursive test discovery
before placing executable diagnostic scripts beneath tests.

### Support modules

Proposed destination: workflows/helpers, not additional root-level packages.
Keep existing module names within that subfolder unless a concrete reason
requires a rename. Do not add forwarding wrappers merely to hide incomplete moves.

| Module | Known callers | Proposal |
|---|---|---|
| interrupt_cleanup | Eight workflows, three hardware scripts, regression tests | Move shared implementation; update every caller together. |
| two_run_plate | Fluorescence workflow, live/simulation runners, regression tests | Move shared implementation; retain marker filename, ownership, and persistence behavior. |
| _planning_markers | Ailsa and fluorescence workflows, template guidance, config tests | Move shared implementation; retain exact log format and timing semantics. This is currently an untracked local addition. |
| vial_risk_analyzer | Multidimensional workflow | Move intact initially; avoid combining organization with a behavioral rewrite. |
| _vial_shuffle | Vial-capture workflow and regression tests | Prefer keeping substantial pure planning code separate in helpers rather than enlarging the workflow; final placement requires approval. |

The template stays but is excluded from executable scheduler choices. Preserve
separate access to diagnostics. Do not introduce recursive workflow discovery
just because helpers have moved into a subfolder.

## Path-preservation checklist for every proposed move

1. Record old/new module names, script launch commands, and external consumers.
2. Find Python imports, dynamic imports, runpy calls, subprocess commands,
   filename-based workflow identities, saved queue entries, and docs references.
3. Check __file__ parent assumptions, working directories, sys.path mutations,
   relative resource paths, and environment-specific absolute paths.
4. Check read and write locations, not just imports: vial CSVs, calibration
   masters, YAML, logs, run history, marker files, protocols, camera fixtures,
   temporary simulation state, and output directories.
5. Preserve authoritative config review and complete supplied-config behavior.
   Never compensate for a broken path with an invented default or fallback.
6. Preserve scheduler fingerprint coverage. Its current top-level workflow scan
   does not establish that moved nested helpers participate in invalidation;
   verify relevant dependency coverage before accepting helper moves.
7. Leave mutable state and scientific files byte-for-byte unchanged during
   simulated checks, except isolated, explicitly designated test outputs.
8. Verify the supported launch contexts. Do not promise arbitrary working-directory
   support if it was never supported; avoid regressing existing launch modes.
9. Search for old references after moving, distinguishing historical provenance
   from executable references. Check case and Windows paths with spaces.
10. Ask the operator about desktop shortcuts, scheduled tasks, other checkouts,
    firmware deployment, and scripts outside this repository. Repository search
    cannot prove these consumers do not exist.

## Small implementation stages, each requiring approval

These are suggested independently reviewable changes, not instructions to commit.

1. Record the active entrypoint/config matrix and an unchanged baseline using an
   existing working interpreter. Identify safe tests without importing manual
   hardware scripts.
2. Move the booking diagnostic only, resolving the root duplicate and credential
   path contract first.
3. Move the clamp diagnostic and its config-documentation references.
4. Move the Cytation diagnostic with import-based tests and launch documentation.
5. Agree on and implement separate scheduler diagnostic access, then move vortex,
   updating its contract tests and config linkage without losing scheduler access.
6. Move interrupt cleanup and all references; verify interruption behavior.
7. Move plate-reuse support and all references; verify both scheduler runners.
8. Move planning markers, including existing uncommitted callers/tests and template
   references; verify duration parsing and dependency fingerprints.
9. Move vial-risk support intact and verify the multidimensional workflow caller.
10. Move vial-shuffle support and verify capture, fixture paths, and restoration.
11. Consolidate workflow notes and plans into documentation after checking links.
12. Review tests, tools, and scripts for ownership and duplicates. Propose specific
    moves rather than wholesale directory restructuring.
13. Audit path-sensitive controller/GUI/calibration modules. Retain their current
    public locations unless a later approved plan demonstrates a clear benefit.
14. Separately propose a data/provenance layout and tracked-artifact policy.
    No data moves, deletions, environment changes, or untracking in source stages.

Stop after any failed baseline or newly failing check. Distinguish pre-existing
failures from move-related regressions; do not hide them or rewrite unrelated code.

## Validation and completion criteria

The implementation must use simulation only and the existing selected/working
environment. Confirm the interpreter on this computer; do not assume an
interpreter path from another machine is present.

Existing focused coverage includes:

- Scheduler setup, config lists, simulation GUI/runner/state, live-runner
  contracts, reset, interruptions, runtime estimates, and vial layout.
- Coordinator workflow config, workflow config docs, vortex contract, and
  Cytation diagnostic regression tests.
- Track-control GUI and plate-inventory GUI tests.
- Fluorescence and pipette-calibration workflows, workflow interrupt cleanup,
  plate inventory, and vial-shuffle capture.

Inspect these tests before selecting commands. A filename starting with test
does not prove a script is safe to run. Use offscreen GUI checks where supported.
Mocked contract tests alone do not prove complete workflow execution or that all
GUI launchers still work.

For each approved source stage, run the smallest relevant safe test selection,
then verify imports/entrypoints and supported launch modes without hardware.
At the end, verify every retained main workflow with confirmed representative
simulation settings, exercising automation calls rather than skipping them.
Smoke-check every GUI launcher and the pipetting/calibration helper paths,
including dialogs, protocol loading, configuration review, persistence targets,
and scheduler subprocess launches.

Some workflows require external protocols, camera services, representative input
data, or simulation support that may be absent. Report these as blocked checks
and obtain the missing inputs or an explicit scope decision; do not claim those
workflows are verified or fabricate data to make a check pass.

Completion means an approved layout is implemented, all intended entrypoints
remain usable, relevant path-sensitive behavior is checked, and data/config/state
has not been unintentionally modified. Until then this document is a proposal.

## Decisions still needed

- Confirm the tests/diagnostics destination and exact scheduler diagnostic UI and
  discovery contract. The workflow helpers layout is accepted; diagnostic
  scheduler access must remain, and the template must not be a runnable choice.
- Identify external launchers and operational workflows/GUI usage for the final
  validation matrix.
- Approve any later data/archive destination and relocation policy separately.

## Out of scope

Algorithm changes, scientific recalibration, controller rewrites, blanket package
conversion, mass file renaming, rewriting historical results, installing tools,
deleting backups/data/environments, and executing physical hardware.

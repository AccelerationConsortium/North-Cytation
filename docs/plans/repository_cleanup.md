# Repository cleanup: current state and future work

Updated: 2026-11-02

## Current state

The broad clutter cleanup is complete enough to stop reorganizing for its own
sake. Most remaining folders have distinct responsibilities.

- `workflows`: eight experiment/data-collection entrypoints and the template.
  Five shared modules are grouped under `helpers`; historical programs remain
  under `archive`.
- `software_tests`: automated regression coverage, isolated from hardware
  discovery.
- `hardware_tests`: 22 retained operator-run checks, including three moved
  scheduler-compatible diagnostics. Similar-looking checks were retained when
  they isolate different physical effects.
- `tools`: reusable operator utilities and arm/track GUIs.
- `scheduler`: GUI, queue execution and supporting modules. Moved diagnostics
  retain their original job names and saved config filenames.
- `status`: 19 inputs used by current workflows, tests, tools or calibration.
  Deleted 57 unused/historical inputs rather than creating another archive.
- `robot_state`: hardware configuration and robot/track state, including powder
  presets.
- `source/firmware/photoreactor`: Pico device source. Host controllers remain
  at root.
- `docs`: guides and plans grouped by purpose.
- `output`, `logs` and pipetting recovery history: generated/local material
  consolidated into appropriate existing locations.

All 505 formerly tracked scratch files were untracked. All 884 local scratch
files were retained with unchanged hashes. Current scratch writers use
`output/temp`; no local scratch deletion is planned.

Cleanup changes remain uncommitted. Existing scientific edits were preserved.
Focused hardware-free checks passed for the organization changes, but the
repository does not have a clean full-suite baseline: existing failures were
observed in scheduler mocks, fluorescence/template and peptoid coverage, among
others. No physical hardware certification was performed.

## Future priorities

### 1. Review and checkpoint the completed cleanup

- Inspect staged and unstaged changes together before making a cleanup commit.
  Do not fold unrelated scientific edits into it accidentally.
- Confirm ignored/local preset changes and the desktop launchers remain usable;
  they are not all represented by tracked repository changes.
- Keep fresh scheduler simulation/review mandatory after changed job/helper
  sources; old approval snapshots must not be reused.

### 2. Maintain software tests rather than deleting by age

- Review coverage by current behavior and actual callers.
- Investigate known failures against the intended workflow contract and current
  data before deciding whether a test or implementation is wrong.
- Retire tests only when their behavior is obsolete or genuinely duplicated.
  AI authorship is not a reason to delete coverage.
- Use focused batches; avoid blanket execution of operator-run hardware checks.

### 3. Finish the small status-folder leftovers

- Review the obsolete peroxide migration script and hardcoded vial-tray
  visualization for retirement. Neither is a current workflow input.
- Preserve the pipet-refill utility and its desktop launcher unless migrating
  both together.
- Review stale presets for retired programs separately from current presets.
  Archived workflows and old presets may now reference intentionally deleted
  inputs; do not make them appear runnable with fabricated substitutes.

### 4. Rename status only when explicitly resumed

**Deferred at the operator's request.** Suggested name: `vial_inputs`.

If resumed, treat this as one coordinated path migration:

- Update current workflows, hardware/software tests, tools and scheduler queues.
- Update ignored/local workflow presets, not just tracked source.
- Cover North robot calibration defaults, hardware YAML and heated/reservoir
  protocols in `sdl_pipette_calibration`.
- Update the desktop pipet-refill launcher and other verified external users.
- Review archived source references, but preserve historical logs, result
  snapshots and recorded provenance.
- Do not globally replace the word `status`: status fields and robot/track
  state filenames are unrelated.
- Verify retained input hashes, current config resolution, GUI startup paths and
  focused software coverage without running equipment.

### 5. Consider host-controller placement last

Moving host automation modules into `source` is the largest remaining root
cleanup, but is optional and higher-risk.

- The coordinator is not their only caller: diagnostics, software tests and
  archived workflows import controllers directly.
- Audit imports, dynamic loading, mocks, file-relative resources and external
  launchers before moving anything.
- Preserve hardware settings, scientific calculations and physical behavior.
- Establish focused baselines and migrate in small independently verifiable
  steps. Do not combine placement changes with controller refactoring.
- Keep operator-facing GUIs and Slack integration at their current locations
  unless a separate concrete benefit justifies moving them.

## Boundaries

- This roadmap does not authorize further deletions, renames or controller moves.
- Leave analysis, recommenders, calibration, scientific data and existing
  environments alone unless there is a specific demonstrated cleanup need.
- Do not add folders, wrappers or compatibility copies merely to make the tree
  look uniform.
- Preserve simulation-first execution and reviewed configuration behavior.
- Prefer deletion of confirmed obsolete code over creating new archive clutter;
  preserve uncertain scientific evidence until explicitly approved otherwise.

The [organization assessment](../repository_organization.md) contains the detailed
retention decisions, historical paths and validation results. This roadmap is
the short future-facing summary, not another copy of that audit.

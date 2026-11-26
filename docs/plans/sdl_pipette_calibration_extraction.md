# Plan: publish `sdl_pipette_calibration` as DispenseRITE

Goal: a separate public repository, `AccelerationConsortium/dispense-rite`, to go with the paper.
People can clone it to learn the method and calibrate their own pipetting hardware. It includes the
North Robot protocols as a worked example. Lash-E keeps working by installing the new package instead
of carrying its own copy. The readme already brands the package as DispenseRITE.

## Decisions (2026-11-26)

- **Repo:** `dispense-rite` under the Acceleration Consortium GitHub org.
- **Import name: `dispense_rite`.** The package installs as `pip install dispense-rite` and is imported
  as `import dispense_rite`. The rename happens when the new repo is created (step 3). Lash-E's imports
  are updated in step 4, in the same change that switches it to the installed package. That covers
  about 10 files: both calibration workflows, `scheduler/liquid_review.py`,
  `pipetting_data/import_calibration_run.py`, and the calibration tests. Do a repo-wide search for
  `sdl_pipette_calibration`, including docs and `workflow_configs/`, so no path is missed.
- **LLM recommender: removed.** It is small to take out:
  - delete `llm_recommender/`
  - remove the two LLM branches in `experiment.py` (about lines 970-1075 and 1274-1345)
  - remove six `config_manager.py` methods and the `llm_optimization` / `use_llm_suggestions` /
    `llm_config_path` yaml keys
  - remove the readme section

  This also removes the `input()` stall (item 6), the private IP and the `hardware_config_north_c9.json`
  leak. Lash-E has a catch: `workflows/pipette_calibration_workflow.py:326` and
  `workflows/reservoir_calibration_workflow.py:257` read `raw["optimization"]["llm_optimization"]` and
  `raw["screening"]["llm_config_path"]`. Both workflows need those two entries dropped in the same change.
- **Simulated runs save output.** `should_save_files` will read `output.save_files` from the config
  (default `true`) instead of always refusing in simulation. The run's metadata and summary should say
  `simulated: true`, so a simulated result can't pass for a real one.
  Lash-E keeps its "never persist simulated data" rule with no extra work: its workflows already write
  `output.save_files = not SIMULATE` (`workflows/pipette_calibration_workflow.py:335`). The package
  currently ignores that key.
- **Git history: clean start.** The new repo begins with a single initial commit and carries none of
  the Lash-E history.
- **Paper data: Zenodo plus one example.** The full `validation/`, `master_data/` and
  `optimized_parameters/` sets go to Zenodo with a DOI cited in the readme and `CITATION.cff`. The repo
  ships one small example run.

Status (2026-11-26): steps 1-3 done. `AccelerationConsortium/dispense-rite` is public and its CI passes.
Step 4 (switching Lash-E to the package) is an **open decision**; see "Open questions". Step 5 (Zenodo,
paper citation, v1.0.0 tag) is not started. File references are relative to `sdl_pipette_calibration/`
unless they start with another top-level folder.

## 1. What stays and what goes

The folder holds 364 tracked files, 17 MB. 288 of those files (12 MB) are `validation/` run outputs.

### Keep: the package

| Area | Files |
|---|---|
| Engine | `experiment.py`, `bayesian_recommender.py`, `config_manager.py`, `constraint_calibration.py`, `parameter_constraints.py`, `data_structures.py`, `optimization_structures.py`, `analysis.py`, `csv_export.py`, `protocol_loader.py`, `__init__.py` |
| Entry points | `run_calibration.py`, `run_validation.py`, `pipetting_wizard.py` |
| Hardware interface | `protocols/calibration_protocol_base.py`, `protocols/calibration_protocol_template.py`, `protocols/calibration_protocol_simulated.py` |
| Optional features (kept, decided 2026-11-26) | `external_data.py` + `input_data/external_calibration_data.csv`, `experiment_analysis.py`, `visualization.py`, `tools/post_optimization_dashboard.py` |
| Config and docs | `experiment_config.yaml`, `readme.md`, `LICENSE`, `requirements.txt`, `.gitignore` |

Not copied into the new repo: `tools/shap_analyzer.py` (duplicates the SHAP analysis in
`experiment_analysis.py` and needs `xgboost`) and `yaml_io.py` (used only by the Lash-E bench tools).

### North Robot example: `calibration_protocol_northrobot.py` only

- Copy `protocols/calibration_protocol_northrobot.py` and `protocols/north_robot_hardware.yaml` into
  `protocols/north_robot/` of the new repo.
- The protocol imports `master_usdl_coordinator`, `pipetting_data` and `slack_agent` by adding the
  Lash-E root to `sys.path`. It should instead state "requires Lash-E installed" and fail with a clear
  message. The Slack calls should be optional.
- The reservoir and heated protocols are not copied. They stay in Lash-E, because
  `workflows/reservoir_calibration_workflow.py` loads `calibration_protocol_reservoir` by name. So
  `protocol_loader.load_hardware_protocol` in the new package needs to load a protocol from a folder
  outside the package (Lash-E's own).

### Move in from Lash-E

- Tests that exercise only the package: `software_tests/test_calibration_simulation_mode.py`,
  `software_tests/test_two_point_constraints.py`, `software_tests/test_calibration_constraint_handoff.py`,
  and the engine-only parts of `software_tests/test_pipette_calibration_workflow.py`.
- Today the package has no tests of its own.

### Leave behind in Lash-E, or delete

- `batch_calibration_automation.py`: North-only (it creates `Lash_E` and edits `north_robot_hardware.yaml`).
- Leftover backups: `experiment_config_backup.yaml`, `north_robot_hardware_backup.yaml`, `calibration_vials_short_backup.csv`.
- `experiment_config_workflow.yaml`: the Lash-E workflows' config. Move it to `workflow_configs/`.
- `tools/calibration_test_gui.py` (PySide6, imports `experiment_tracking`) and
  `tools/two_point_series_calibration_demo.py` (North protocol). Both are Lash-E bench tools.
- `llm_recommender/test_llm_with_real_data.py`: a manual script, not a test.

### Decide: paper data

The data files are `validation/` (12 MB, 48 runs), `master_data/` (2.6 MB) and `optimized_parameters/`
(23 CSVs, ignored by `.gitignore` but tracked anyway).

Recommendation: put the paper's datasets on Zenodo with a DOI and cite it. The repo keeps one small
example: a single validation run, plus one `optimal_conditions_*.csv` that replaces the missing
`optimal_conditions_example.csv`. Some run folders have working-note names
(`..._relativelypoorresults`, `..._tipissue`), which are better curated than published as they are.

## 2. Is the core still hardware-independent?

Mostly yes. No core module imports Lash-E. Only the North protocols and
`batch_calibration_automation.py` do. Two things tie it to this repo, plus a few leaks.

**Tied to the Lash-E layout**

- The runners and tools add the folder above the package to `sys.path` (`run_calibration.py:32`,
  `run_validation.py:41`, `protocol_loader.py:33`). They work only because the package sits inside Lash-E.
  The fix is a `pyproject.toml` so people can `pip install -e .`, and the runners then become normal entry points.
- `protocols/calibration_protocol_*.py` import each other as top-level modules
  (`from calibration_protocol_base import ...`), which relies on that `sys.path` insert.

**Leaks of North-specific details into the core**

- `experiment.py:277`: the emergency-CSV header is a fixed list of North parameter names
  (`blowout_vol`, `post_asp_air_vol`, ...). On other hardware those columns would be empty, and their own
  parameters would be dropped. It should take the names from the parameters actually in use.
- `llm_recommender/llm_recommender.py:722`: hardcodes `hardware_config_north_c9.json`.
- `llm_recommender.py:353,372` and `calibration_screening_llm_template.json:42`: the default LLM URL is
  a private lab IP (`100.77.238.68`). Change it to `localhost` before publishing.
- `pipetting_wizard.py:243,297`: rounds North's speed and cycle parameter names to integers. Minor; better
  to use the parameter's declared type from the config.
- `experiment_config.yaml:7-8`: defaults to `simulate: false` with `calibration_protocol_northrobot`.
  A fresh clone should default to simulation.

The core treating `overaspirate_vol` as the one required calibration parameter is a deliberate design
choice (`data_structures.py:51`), not a leak. The readme should still say so plainly.

## 3. How did the earlier clean-up do?

It did well on structure: a clear base class plus template, config-driven parameters, an honest
`requirements.txt`, and a simulated protocol that needs no hardware. What's left falls into the groups
below. I confirmed the starred items in the code myself. The rest come from a second-opinion review
and are worth a look as each one is fixed.

### Fix before publishing

1. ★ **Broken import.** `protocols/calibration_protocol_base.py:163` imports `calibration_protocol_hardware`,
   which doesn't exist. Delete this second `create_protocol`; `protocol_loader.create_protocol` is the real one.
2. ★ **Simulated runs save nothing.** `config_manager.py:480` (`should_save_files` returns
   `not is_simulation()`). That is the right rule inside Lash-E. But the readme Quick Start tells
   newcomers to simulate and then open `output/<run>/`, which never appears. The public repo needs simulated
   output (perhaps written to a clearly marked `output/simulated/`), or the readme has to change.
3. ★ **Objective thresholds ignored.** `bayesian_recommender.py:496-504` fixes the Ax objective thresholds
   at 50 / 20 / 90. The `optimization.objective_thresholds` set in the yaml is read in but never used.
   Make sure the paper's methods match whatever the code really does.
4. ★ **Config snapshot can be wrong.** `experiment.py:1943` saves `experiment_config_used.yaml` by copying
   the default config file. If you ran with a different config, the snapshot shows the wrong settings.
5. **Constraints dropped silently.** Hardware constraints that fail to load or simplify are dropped
   with only a warning (`bayesian_recommender.py:201,285`). The block at `:207-229` calls a method that
   doesn't exist, hidden by `except AttributeError`.
6. **Unattended runs can stall.** The LLM fallback calls `input()` (`experiment.py:1061,1073`), which
   blocks an overnight run.
7. **Stray imports.** Bare (non-relative) imports in `data_structures.py:265` and
   `optimization_structures.py:368` break once the package is installed.
8. ★ **Emoji crash on Windows.** `protocols/calibration_protocol_simulated.py:261` prints an emoji. On
   Windows with output redirected, the first trial crashes with `UnicodeEncodeError` (cp1252). Emoji
   also appear in `bayesian_recommender.py`, `data_structures.py`, `optimization_structures.py` and the
   North protocols. Replace them with plain text.

### Standalone check (done 2026-11-26)

The folder was copied out of Lash-E with `git archive` into a fresh venv with only `requirements.txt`
(Python 3.12, Windows), then a 20 µL, 30-measurement simulated calibration was run.

- The install is fine.
- The first attempt crashed on item 8.
- With `PYTHONIOENCODING=utf-8` the run finished: 8 screening trials plus optimization, 29 measurements.
  It printed "Results saved to: None" and wrote no `output/` folder (item 2).
- So the engine already runs outside Lash-E. The blockers for newcomers are items 2 and 8 and the
  config defaults.

### Should fix

- **Readme vs code.** The `PipettingWizard` method name and output file name are wrong in the readme,
  one example constraint is nonlinear (only linear ones are supported), and "strict schema validation"
  overstates what the code checks.
- **Stated defaults vs code defaults.** The yaml comments state defaults the code doesn't use
  (`num_screening_trials`, `adaptive_measurement.enabled`, `deviation_threshold_pct`, ...), and
  `max_total_measurements` has no default at all.
- **Tolerance-range gaps.** Volumes between the yaml's ranges (for example 19.5 µL) silently get 5%.
- **Protocol interface gaps for outsiders.** The template promises `cfg['hardware_parameters']`, which
  is never passed. A `wrapup` returning `None` logs a false warning. `measurement_budget_consumed` is documented only in a comment.
- **Errors swallowed.** Bare `except:` and swallowed hardware errors end a volume early with no clear failure
  (`experiment.py:997,1185,1301,1351`).
- **Debug logging left at INFO level.** "BUG DETECTED", "[CONFIG DEBUG]", "CRITICAL FIX" comments.
- **Very long functions.** `experiment._run_optimization_phase` (about 270 lines) and `experiment.run` (about 180).
- **`requirements.txt` gaps.** It is missing `requests` (LLM) and `xgboost` (SHAP tool).
  `__init__.py` `__author__` is a placeholder.
- ★ **Two-point stub.** `constraint_calibration.run_two_point_calibration_measurements` is a
  `NotImplementedError` stub. Remove it.

### Nice to have

- **Unused code.** About 20 functions are never called (listed in the review notes), and some code is duplicated:
  two `create_protocol` functions, and identical `_flatten_recursive` in `csv_export.py` and `visualization.py`.
- **Magic numbers.** Two-point step sizes and "successful volume" cut-offs should become named constants or config.

## Step 2 progress (2026-11-26, not yet committed)

**Done in place**

- **LLM recommender removed:** the folder, the engine branches, the config methods, the yaml keys and
  the readme section. The two Lash-E calibration workflows no longer read the LLM keys.
- **Simulated runs save output.** `should_save_files`: real runs always save; simulated runs follow
  `output.save_files` (default `true`). The folders end in `_SIMULATED`. Lash-E workflows still pass
  `false` for simulated runs.
- **Items 1-5, 7 and 8 fixed.**
  - Deleted the broken `create_protocol` in the protocol base, and fixed the bare imports.
  - Ax objective thresholds now come from the config. The yaml is set to the 50 / 20 / 90 that Ax has
    really used since July 2026, so behavior is unchanged.
  - `experiment_config_used.yaml` is written from the settings in effect.
  - A constraint that fails to load or simplify now stops the run. The North constraints were checked
    and simplify fine.
  - Emoji replaced with plain text. Slack messages were left as they are.
- **Emergency-CSV header** comes from the configured and fixed parameter names, not a North list.
- **Debug logging** demoted to debug level; the "BUG DETECTED" block is removed.
- **The shipped `experiment_config.yaml` defaults to `simulate: true`.**
  `batch_calibration_automation.py` now sets `simulate: false` itself.

**Verified**

- The calibration tests pass: `software_tests/test_calibration_*`, `test_two_point_constraints`,
  `test_pipette_calibration_workflow` and `test_liquid_review`. Two `test_liquid_review` GUI tests
  could not run because the test environment lacks PySide6.
- A two-volume simulated run from a copy outside Lash-E, with only `requirements.txt`, finished and
  wrote `output/run_..._SIMULATED/`.

**Moved to step 3 (new repo):** `pyproject.toml`, removing the `sys.path` inserts,
`optimal_conditions_example.csv`, and the readme / config-default corrections in "Should fix".

## Step 3 progress (2026-11-26): `dispense-rite` built locally, not pushed

The repo is at `North Robotics/dispense-rite` (next to `utoronto_demo`). It has a `src/dispense_rite/`
layout, a `pyproject.toml` with the `dispense-rite-calibrate` and `dispense-rite-validate` commands,
`examples/`, `tests/` (34 tests, including an end-to-end simulated calibration plus validation), CI
for Ubuntu and Windows, and `CITATION.cff` (the paper and DOI are still TODO).

**What Lash-E must change when it switches to the package (step 4)**

- **Imports:** `sdl_pipette_calibration` becomes `dispense_rite`. Protocols import the base as
  `from dispense_rite.protocols.calibration_protocol_base import CalibrationProtocolBase`.
- **Reservoir and heated protocols stay in Lash-E**, for example in `calibration_protocols/`.
  `load_hardware_protocol(name, search_paths=[...])` takes the folder. Otherwise, set
  `experiment.protocol_search_paths` in `experiment_config_workflow.yaml`, or set the
  `DISPENSE_RITE_PROTOCOL_PATH` environment variable.
- **`PipettingWizard()` with no `search_directory`** now searches the current working directory
  instead of the package folder. Lash-E callers that relied on the default must pass their folder.
- **`calibration_protocol_northrobot` reads `volume_targets_ml` from the `cfg` the engine passes.**
  It used to look for a `protocols/experiment_config.yaml` that never existed, so it always used its
  fallback. Tip conditioning now follows the real targets. Check this on the robot.
- **Tolerance ranges** in the new example config share endpoints (20, 60, 200 µL), with no gaps.
  `experiment_config_workflow.yaml` still has the gaps; copy the fix if wanted.

## 4. Extraction steps

1. **Decide the open questions** (below).
2. **Clean up in place, in Lash-E**, so its tests keep guarding the change: fix items 1-7, remove the
   leaks in section 2, and add `pyproject.toml`.
   Success: a simulated run works from a fresh venv with only `requirements.txt`, from outside Lash-E.
3. **Create the new repo.** It contains `src/sdl_pipette_calibration/` (package),
   `protocols/north_robot/` (example), `tests/`, `examples/` (one config and one small dataset),
   `README.md`, `LICENSE`, `CITATION.cff` (paper and Zenodo DOI) and a GitHub Actions workflow that runs
   the tests and a short simulated calibration.
4. **Point Lash-E at it (open decision, not started).** `pip install` the new repo at a pinned tag. Then delete the folder from Lash-E,
   move `experiment_config_workflow.yaml` to `workflow_configs/`, and re-run Lash-E's calibration tests
   (`software_tests/test_pipette_calibration_workflow.py`, `software_tests/test_calibration_preflight.py`,
   `software_tests/test_liquid_review.py`).
   `scheduler/liquid_review.py` (`past_results`, around line 455) searches
   `sdl_pipette_calibration/output` for past calibration and validation results. Once the folder is
   gone, keep those old runs somewhere it still searches, or update the path.
5. **Tag v1.0.0** to match the paper and archive it on Zenodo.

## Open questions

### Should Lash-E switch to the `dispense-rite` package? (deferred 2026-11-26)

It would work technically:

- Install with `pip install git+https://github.com/AccelerationConsortium/dispense-rite@v1.0.0`, or for
  development `pip install -e ../dispense-rite` from a local clone.
- Ax matches: `dispense-rite` pins 0.4.3, the same as `environment.yml` and `requirements-lock.txt`.
- The North protocol inside the package imports `master_usdl_coordinator` and `pipetting_data`, which are
  importable when Lash-E workflows run.
- The package's `PipettingWizard` is used only by the two calibration workflows. `North_Safe.py` and
  `pipetting_data/` use Lash-E's own `pipetting_data/pipetting_wizard.py`, so the switch touches only the
  calibration workflows, their tests, `scheduler/liquid_review.py` and the reservoir and heated protocols.

The trade-off is where calibration changes get made:

- **Switch:** there is one copy. Calibration changes are made in `dispense-rite`, and Lash-E then moves
  its pinned version (or uses an editable local clone). Lab members need to know the code lives in a
  second repo.
- **Keep both:** Lash-E keeps `sdl_pipette_calibration/` and `dispense-rite` is the published version
  from the paper. Lash-E needs no work, but the two copies drift apart and fixes are copied by hand.

Leaning: switch once the calibration method is settled. Keep both while it is still being reworked.

### Unrelated cleanup found along the way

- `requirements.txt` says `ax-platform==1.1.2`, but `environment.yml` and `requirements-lock.txt`
  pin 0.4.3, and only `recommenders/old/` uses Ax outside the calibration package. Align
  `requirements.txt` with the lock file.

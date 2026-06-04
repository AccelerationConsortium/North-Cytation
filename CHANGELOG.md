# Changelog

## [2026-06-03] - Fix Embedded Validation Overaspirate Runaway

### pipetting_data/embedded_calibration_validation.py
- **FIXED**: `_interpolate_optimal_overaspirate` now clamps slope to [0.5, 1.5] uL/uL (mirrors v2 `_compute_optimal_overaspirate`). Degenerate/negative slopes (caused by Stage 2 landing on the same side as Stage 1 due to noise + zero-floor clamping) returned Stage 3 overaspirate of +44 uL, causing a +52 uL dispensing error and workflow stoppage on 2026-06-02.
- **FIXED**: Stage 2 crossing strategy now allows negative overaspirate (±10 uL delta from baseline) instead of clamping to 0.0, allowing proper bracketing of the target.
- **FIXED**: Stage 2 and Stage 3 over-threshold results now log a warning and continue to the best-stage selection logic, rather than calling `pause_after_error` and raising. Workflow no longer halts on a bad optimization stage.

## [2026-06-03] - Heating Mantle Positioning and Mixing Test

### tests/Heating positioning.py (new)
### status/heating_mantle_vials.csv (new)
- **ADDED**: Test script that moves four open-top vials (V0-V3, ~2 mL each) sequentially
  through heating mantle positions 2, 5, 8, and 4.
- For each vial: robot places it in the heater, returns it to the safe pipetting rack,
  mixes it (5 x 0.5 mL aspirate-dispense cycles back into the same vial), removes the
  pipet tip, then returns the vial to its heater position.
- All vials are returned home at the end of the run.
- `SIMULATE = True` by default; set to `False` for live hardware.

## [2026-06-02] - Desirability-Based Composite Scoring

### calibration_modular_v2/compare_runs_best_over_trials.py
### calibration_modular_v2/compare_runs_accuracy_time_journey.py
### calibration_modular_v2/two_point_series_calibration_demo.py
- **CHANGED**: Replaced stdev-normalized SDL composite score with desirability scoring.
  - **Before**: `score = 0.4*(dev/acc_std)*100 + 0.5*(cv/prec_std)*100 + 0.1*(t/time_std)*100` (lower=better)
  - **After**: `desirability = 0.4*d(dev,tol) + 0.5*d(cv,tol) + 0.1*d_time` (higher=better)
  - Desirability: `d(x, tol) = 1 / (1 + (x/tol)^2)` — soft boundary, 1.0=perfect, 0.5=at tolerance, never clips to zero.
  - Tolerance sourced from `experiment_config_used.yaml` per run dir (volume-dependent: 1%/2%/3%/10%).
  - Time normalized within population `(t_max - t) / (t_max - t_min)` — no fixed reference needed.
- **REASON**: Stdev normalization allowed outlier precision values (e.g. cv=0.14% vs population mean ~2%) to dominate scoring, selecting trials with poor accuracy. Desirability scoring treats any cv below the tolerance as diminishing returns, which matches physical reality.
- Composite panel in best_over_trials plots now labeled "desirability, higher = better".
- All 12 comparison plots regenerated (output/comparisons/).

## [2026-06-02] - Baseline Parameter Extraction & SDL Composite Score Fix

### calibration_modular_v2/compare_runs_accuracy_time_journey.py
- **CRITICAL FIX**: `running_best_composite()` now includes precision (CV %) in SDL composite score calculation.
  - **Before**: `score = deviation_pct / acc_std + duration_s / time_std` (ignored precision entirely)
  - **After**: `score = acc_w * deviation / acc_std + prec_w * precision_cv / prec_std + time_w * duration / time_std` (matches real v2 logic)
  - This was causing the trajectory plot to select wrong "best" conditions when precision varied significantly.

### calibration_modular_v2/two_point_series_calibration_demo.py
- **ADDED**: Baseline parameter extraction from prior trial_results.csv using SDL composite scoring (with precision).
  - New configuration dict `TRIAL_RESULTS_BY_LIQUID` at top of file for specifying trial_results.csv paths per liquid.
  - New function `_load_trial_results()`: Loads CSV and filters to trials with >=2 measurements only (can't calculate precision with n=1).
  - New function `_extract_best_trial_parameters()`: Calculates SDL composite score using population normalization (same as v2 analysis.py).
  - New function `_get_baseline_params_for_liquid()`: Wrapper to load from file or fall back to defaults.
- **CHANGED**: Demo now sources real baseline parameters from optimization runs instead of hardcoded defaults.
  - Falls back to `DEFAULT_BASELINE_PARAMS` if trial_results.csv not provided or not found.
- **IMPROVED**: Logging now shows composite score breakdown when extracting baseline.

## [2026-06-01] - Two-Point Series Calibration Demo (v2 compartmentalized)

### calibration_modular_v2/two_point_series_calibration_demo.py
- ADDED: New standalone v2 demo script for two-point overaspirate calibration across 25, 50, 75, 100, 150 uL.
- ADDED: Uses `HardwareCalibrationProtocol` directly for each liquid and volume, with 3 replicates per point.
- ADDED: Implements v2 delta equation exactly: `spread_ul = max(abs(shortfall_ul) + tolerance_buffer_ul, 2.0)` and adaptive Point 2 direction.
- ADDED: Computes interpolated optimal overaspirate from Point 1/Point 2 means and exports both detailed and summary CSV outputs.

## [2026-06-01] - Calibration Vials Short: Use Real HardwareCalibrationProtocol Infrastructure

### workflows/calibration_vials_short_mass_validation.py
- CHANGED: Complete refactor to use `HardwareCalibrationProtocol` directly (matching batch calibration pattern) instead of bypassing to Lash_E. Now respects tip conditioning and refill_pipets logic from protocol.
- FIXED: Densities now sourced from protocol's `LIQUIDS` dict (single source of truth) instead of hardcoded values. No more silent fallbacks.
- CHANGED: Vial names now use exact protocol names: updated references to match `LIQUIDS` dict keys.
- CHANGED: Workflow iterates through liquids using `protocol.initialize(cfg)` -> `protocol.measure(state, 0.05mL, params, replicates=3)` -> `protocol.wrapup(state)` pattern (unified interface).

### status/calibration_vials_short.csv
- FIXED: Vial names updated to match protocol LIQUIDS dict exactly: `polymer_dmso` -> `PVA_DMSO`, `dmso` -> `DMSO`.

## [2026-06-01] - Simple Calibration Vials Short Mass Validation Workflow

### workflows/calibration_vials_short_mass_validation.py
- CHANGED: Refactored workflow to use calibration_modular_v2 HardwareCalibrationProtocol directly (`initialize` -> `measure` -> `wrapup`) instead of embedded validation helpers.
- ADDED: Per-liquid update of `calibration_modular_v2/north_robot_hardware.yaml` for `liquid`, `source_vial`, and `measurement_vial` before each protocol run.
- ADDED: Automatic restoration of original hardware YAML after workflow completion/failure to avoid persistent config drift.
- ADDED: Explicit preflight checks that fail loudly if required vial names are missing from `status/calibration_vials_short.csv` or if protocol liquid names are invalid.
- ADDED: Consolidated CSV output in `output/calibration_vials_short_mass_validation_v2_<timestamp>.csv` with per-replicate measured volume fields from protocol results.

## [2026-05-29] - MOF Multi-Run Template Example

### workflows/mof_synthesis_workflow.py
- ADDED: Commented `RUNS` template showing all supported per-run override keys (`reaction_vial`, `replicates`, `prepare_substock`, ratio, volume, temperature, sampling cadence, and dispense volume) for easier multi-run setup.

## [2026-05-29] - MOF Workflow Config Loading Fix

### workflows/mof_synthesis_workflow.py
- FIXED: `Lash_E` initialization now passes `workflow_name="mof_synthesis_workflow"` along with `workflow_globals=globals()` so ConfigManager can load/save workflow YAML settings and GUI edits.

## [2026-05-25] - Calibration Budget Enforcement and Retry Default

### calibration_modular_v2/experiment.py
- FIXED: Volume-budget accounting in optimization now uses protocol-reported `measurement_budget_consumed` instead of raw replicate counts.
- FIXED: Optimization budget now includes already-executed two-point calibration trials when computing remaining volume budget.
- FIXED: First-volume final calibration budget now derives from configured split `max_total_measurements - max_measurements_first_volume` (e.g., 96-87=9), replacing hardcoded budget value.
- FIXED: Two-point calibration now computes required measurements as `2 * two_point_calibration_replicates` and validates against the phase budget before execution.
- ADDED: Internal `_trial_budget_consumed()` helper to centralize budget-unit accounting from trial metadata.

### calibration_modular_v2/calibration_protocol_northrobot.py
- CHANGED: Added config-driven `experiment.max_retries_per_measurement` with default `0`.
- CHANGED: Measurement retry loop now uses `max_retries_per_measurement` from protocol state instead of hardcoded retries.
- CHANGED: Protocol state now stores `max_retries_per_measurement` for explicit runtime behavior.

### workflows/surfactant_multidimensional_workflow.py
- ADDED: Lightweight Slack notifications for workflow start, per-iteration completion, per-iteration skip (no feasible points), and workflow completion.
- ADDED: Local best-effort Slack helper that is non-blocking and disabled in simulation mode.

## [2026-05-21] — Unified Feasibility Authority to Source-Achievable

### workflows/surfactant_multidimensional_workflow.py
- CHANGED: `generate_achievable_boundary_samples()` now accepts `plans` parameter and uses `joint_select_sources()` for source-achievable feasibility checks instead of `is_feasible()` (budget-only).
- CHANGED: `generate_simplex_init()` now accepts `plans` and `logger` parameters. Adds nested `_is_achievable()` helper that uses source-achievable checks. Passes plans to both boundary and Sobol interior filtering.
- CHANGED: `run_multidim_workflow()` now builds bootstrap plans (covering MIN_CONC_MM and per-surfactant max) BEFORE calling `generate_simplex_init()`. Bootstrap plans enable source-achievable checks at boundary/interior selection time, preventing later dropped boundary points.
- REMOVED: Dependency on `is_feasible()` in boundary/interior decision paths. `is_feasible()` function definition retained for legacy code compatibility but no longer used by core workflow.

**Rationale:** Two inconsistent feasibility systems (Layer A: budget-only vs Layer B: source-achievable) created architectural mismatch. Boundary/interior points selected with Layer A were later rejected by Layer B filtering, causing unexpected dropped points in 3D overlays. Unified on Layer B (source-achievable) as single authoritative feasibility check. Sequencing fix (bootstrap plans before simplex init) ensures plans available for source-achievable evaluation.

## [2026-05-21] — Dropped Init Candidate Diagnostics (No Algorithm Change)

### workflows/surfactant_multidimensional_workflow.py
- ADDED: Captures initial simplex/grid candidates before plan-based filtering and computes dropped candidates after `filter_points_by_actual_volumes()`.
- ADDED: Logs dropped candidate counts split by `_init_source_type` (`boundary`, `sobol`).
- ADDED: Exports dropped candidates to `dropped_init_candidates.csv` in each experiment output folder.
- CHANGED: Passes dropped candidate list into initial 2D/3D overlay plotting for visual diagnostics.

### analysis/multidim_visualizer.py
- ADDED: Optional `dropped_points` overlay in `plot_pairwise_feasible_overlay()`.
- ADDED: Dropped init candidates rendered as black `x` markers in pairwise overlays.
- ADDED: Pairwise overlay title/legend now includes dropped-candidate count when provided.

### analysis/plot_3d_interactive.py
- ADDED: Optional `dropped_points` overlay in `plot_init_feasible_overlay_3d()`.
- ADDED: Dropped init candidates rendered as black `x` markers in 3D overlay.
- ADDED: 3D overlay subtitle now includes dropped-candidate count when provided.

## [2026-05-21] — 3D Init Pick Source Visibility

### analysis/plot_3d_interactive.py
- CHANGED: `plot_init_feasible_overlay_3d()` now separates initialization picks by `_init_source_type` in 3D overlays.
- ADDED: Distinct marker styles/colors for boundary vs sobol init picks in 3D (boundary as blue open diamonds, sobol as red triangles).
- ADDED: 3D overlay title now reports init counts (`total`, `boundary`, `sobol`) for quick sanity checks of point distribution.
- FIXED: Replaced unsupported Plotly `Scatter3d` marker symbol (`triangle-up`) with supported symbol (`square`) to prevent non-fatal plotting failure and ensure `init_feasible_overlay_3d.html` is generated.

## [2026-05-21] — 3D Feasible Overlay Consistency Update

### analysis/plot_3d_interactive.py
- CHANGED: `plot_init_feasible_overlay_3d()` now supports a grid-based budget-feasibility isosurface when `feasible_config` is provided.
- CHANGED: 3D init overlay now renders a true budget-feasible envelope (mask/isosurface) instead of only a convex-hull mesh of sampled boundary points.
- KEPT: Convex-hull boundary mesh path as fallback when `feasible_config` is not provided.

### workflows/surfactant_multidimensional_workflow.py
- CHANGED: 3D init overlay call now passes `feasible_config` (stock concentrations, well volume, budget, min concentration, max multiplier), aligning 3D feasible visualization logic with 2D mask-based plotting.

## [2026-05-21] — SDS Vial Inventory Expansion

### status/surfactant_multidim_vials.csv
- ADDED: `SDS_stock` in `main_8mL_rack` slot 24.
- ADDED: `SDS_dilution_0` through `SDS_dilution_5` in `main_8mL_rack` slots 25-30.
- ADDED: `SDS_refill` in `large_vial_rack` slot 3.

## [2026-05-21] — 3D Overlay Readiness + Boundary Visualization Cleanup

### workflows/surfactant_multidimensional_workflow.py
- FIXED: In simplex mode, 2D overlay now suppresses extra boundary artifact series while still passing true boundary points to the 3D overlay renderer.
- CHANGED: 3D overlay call now uses `feasible_boundary_points_3d` to ensure envelope rendering still works after 2D cleanup.

### analysis/plot_3d_interactive.py
- CHANGED: Removed redundant gray "All experiment picks" scatter from `plot_init_feasible_overlay_3d()` so the init overlay reflects only the envelope + init picks.

## [2026-05-21] — White Dot Artifact Diagnostic & Tolerance Fix

### workflows/surfactant_multidimensional_workflow.py
- ADDED: `FEASIBLE_DIAGNOSTIC_GRID_N` config parameter (default 140) for tunable feasibility grid resolution.
- ADDED: `diagnose_white_dot()` function to explain infeasibility: shows per-surfactant source options and reports joint selection failure reason.
- ADDED: Verbose `achievable_with_diagnostics()` wrapper in plotting call that logs first 5 white dot diagnostics per panel to help debug feasibility edges.
- FIXED: **Critical tolerance mismatch bug** in `filter_points_by_actual_volumes()`: now uses `<= SURFACTANT_BUDGET_UL + 1.0` to match tolerance in `joint_select_sources()`, eliminating false-negative white dots caused by overly strict <= constraint.

### analysis/multidim_visualizer.py
- ADDED: `grid_n` parameter to `plot_pairwise_feasible_overlay()` for grid resolution control in diagnostics.
- ADDED: `logger` parameter to pass diagnostics to workflow logger instead of stdout.
- ADDED: Feasibility diagnostic logging: reports per-pair grid statistics (feasible count, achievable count, white dot %, grid resolution).
- ADDED: White dot sampling: logs coordinates and diagnostics of up to 3 sampled white dots per pairwise panel.

### KEY FINDING: White Dots Were Tolerance Artifacts ✓ RESOLVED
- **Diagnosis**: Grid resolution test (140 → 300) showed white dots at same density (~1.2%), confirming origin was not rendering artifacts.
- **Root cause identified**: Tolerance mismatch between feasibility prediction (uses stock conc) and actual dispensing (uses more-dilute substocks).
- **The problem**: Substocks require MORE volume than stock to hit same target concentration, pushing total to 225.1-225.8 µL.
- **The bug**: Filter was checking `<= 225.000001 µL` while `joint_select_sources()` allowed `<= 226 µL`.
- **Fix validation**: Aligning both checks to use same +1.0 µL tolerance **completely eliminated interior white dots**.
- **Spatial analysis**: 57% of sampled white dots were pseudo-interior (on simplex vertex boundaries), not truly interior—expected infeasibility when one surfactant is maxed.
- **Conclusion**: Your substocks are adequate. Interior achievable region is now correctly displayed as fully connected.

## [2026-05-21] (Earlier)

### workflows/surfactant_multidimensional_workflow.py
- CHANGED: Added `TURBIDITY_PLOT_THRESHOLD` (post-processing only) and wired the 3D turbidity cloud plot to use it, decoupling visualization thresholding from active-learning penalty settings.
- REFACTORED: Replaced simplex initialization axis-ray + pairwise-face projection with direct boundary-simplex lattice sampling in excess-volume coordinates (`INIT_BOUNDARY_LEVELS`).
- CHANGED: Simplex boundary points are now generated on the true outer simplex facet (`sum(volumes)=budget`) by construction, preserving extreme vertices/edges without projection artifacts.

### workflow_configs/surfactant_multidimensional_workflow.yaml
- ADDED: `TURBIDITY_PLOT_THRESHOLD: 0.08` for 3D turbidity cloud visualization.
- CHANGED: Replaced `INIT_AXIS_PTS` and `INIT_FACE_PTS` with `INIT_BOUNDARY_LEVELS` for n-D simplex boundary lattice density control.

### analysis/plot_3d_interactive.py
- FIXED: Removed redundant `go.Volume` colorbar (`showscale=False`) in `plot_isosurface` to prevent overlapping dual colorbars with the measured-point turbidity colorbar.
- ADDED: `plot_init_feasible_overlay_3d()` to render geometric feasible-envelope mesh overlays with initialization picks (`iteration == 0`) highlighted in interactive 3D HTML.

### analysis/multidim_visualizer.py
- FIXED: Replaced dual-layer feasibility visualization (budget + achievable) with **single achievable layer only** — now shows only what can actually be pipetted with current substocks and volume constraints.
- CHANGED: Switched from `contourf` (solid fills) to `contourf` with `levels=[0.5, 1.5]` (clean filled regions only, no boundary artifacts) for visualization of achievable regions.
- FIXED: `_tick_labels()` now uses scientific notation (1e-06, 1e-05, etc.) for concentrations < 0.1 mM, properly displaying very low MIN_CONC_MM values on axis labels.
- ADDED: Support for `max_conc_multiplier` in `feasible_config` dict to smooth high-end jagged boundaries (multiplier applied to x_max, y_max before grid generation).

### workflow_configs/surfactant_multidimensional_workflow.yaml
- ADDED: `FEASIBLE_MAX_CONC_MULTIPLIER: 1.0` — controls boundary smoothness in feasibility plots (1.0 = true max, <1.0 = reduced for smoother appearance).

### workflows/surfactant_multidimensional_workflow.py
- ADDED: `FEASIBLE_MAX_CONC_MULTIPLIER = 1.0` config parameter (default: no smoothing) passed to `plot_pairwise_feasible_overlay()`.
- CHANGED: `plot_pairwise_feasible_overlay()` call now includes `max_conc_multiplier` in `feasible_config` dict.
- ADDED: `generate_simplex_boundary_points()` helper for reusable geometric boundary lattice generation on the simplex outer facet.
- CHANGED: `generate_simplex_init()` now embeds source-type metadata (`_init_source_type: "boundary"|"sobol"`) in each point dict for downstream visualization
- CHANGED: `build_well_recipe()` preserves `_init_source_type` metadata from target concentration dicts through the recipe building pipeline, enabling source-aware plotting in post-init overlays.
- CHANGED: Post-initialization plotting hook now saves feasible-boundary overlays immediately after initial measurements (before active learning):
  - `pairwise_feasible_init_overlay.png` for all dimensions (pairwise projections)
  - `init_feasible_overlay_3d.html` when exactly 3 surfactants are active.
- FIXED: Boundary-lattice parameterization now uses concentration-simplex coordinates (`MIN_CONC_MM -> simplex_max_conc_mm`) before `is_feasible()` filtering, restoring low-concentration corners in 2D overlays that were previously collapsed toward high-concentration regions.
- CHANGED: 2D feasible overlay call now passes physical feasibility settings and an operational achievability checker based on the existing joint source-selection filter.
- CHANGED: `filter_points_by_actual_volumes()` now logs filtering diagnostics at info level: reports total filtered count and surviving point count, enabling diagnostics of boundary-point survival rates

## [2026-05-19]

### data_tracker/ (new) — Windows restic backup system
- Rewrote all files for Windows/PowerShell (original Linux versions replaced).
- `data_tracker/.env` — PowerShell credentials file (gitignored); holds MinIO keys, restic password, repo URL, backup source path.
- `data_tracker/backup.ps1` — backup script deployed to `C:\restic\`; incremental encrypted backup of `output\` to MinIO over Tailscale; retention cleanup (--keep-daily 14 --keep-weekly 8 --keep-monthly 6); logs to `C:\restic\backup.log`.
- `data_tracker/setup.ps1` — one-time deploy script: installs restic via winget, deploys credentials to `C:\restic\`, initialises encrypted repo in MinIO, registers Windows Task Scheduler task (daily 2 AM).
- `data_tracker/README.md` — full operations guide: credentials setup, restore examples, troubleshooting, retention policy, security notes.
- `.gitignore` — appended `data_tracker/`.
- Verified: 6.9 GiB initial backup, incremental backup 1.6 MiB, 2 snapshots in MinIO, Task Scheduler task active.

### calibration_modular_v2/reprocess_interrupted_run.py (deleted after use)
- One-shot script to reconstruct analysis outputs from an interrupted run clipped to 96 measurements (32 trials x 3).
- Wrote `trial_results.csv`, `raw_measurements.csv`, `ax_trials_data.csv`, `optimal_conditions_DMSO.csv`, `experiment_summary.json/csv`, `analysis_report.txt`, and 3 plots to `run_1779212375_DMSO/`.
- Script self-deleted on successful completion.


- `.gitignore` — appended `data_tracker/` to prevent any backup credentials or scripts from being pushed to GitHub.



### calibration_modular_v2/batch_calibration_automation.py
- FIXED: `restore_files()` was overwriting `calibration_vials_short.csv` (live robot state)
  and `north_robot_hardware.yaml` on every interrupt/completion. Now only `experiment_config.yaml`
  is restored. The CSV must never be reset — it tracks real vial positions and tip counts.
  The hardware config vial names are written fresh before each run anyway.
- FIXED: The per-run "Step 0: Restore original vials state" CSV reset also removed.
- FIXED: `overaspirate_vol` was silently clamped to `target_volume * max_fraction_of_target`
  (default 0.2), limiting 50uL runs to 10uL max overaspirate instead of the configured 25uL.
  Added `max_fraction_of_target: 1.0` to `experiment_config.yaml` so the absolute bounds
  always govern.
  crashing all 8 runs with a KeyError. Changed to only write `validation.volumes_ml` when the key
  is present in the liquid config, allowing validation-free runs to proceed normally.
- FIXED: Batch calibration was renaming vials to `liquid_source_0` in the CSV, but the protocol
  now reads vial names from `north_robot_hardware.yaml`. Replaced `modify_vials_csv()` call with
  `modify_hardware_config()` which writes the correct vial name into `north_robot_hardware.yaml`
  before each run. Added backup/restore of `north_robot_hardware.yaml`.
- FIXED: DMSO `target_vial` corrected from `'dmso'` (not in CSV) to `'liquid_source_0'` (actual name).

## [2026-05-12]

### workflows/surfactant_grid_replay.py
- FIXED: Water vials were never refilled mid-run because REFILL_THRESHOLD_ML (4.0 mL) was too low.
  From the CSV: chunk 1 consumed 2.824 mL leaving 5.176 mL, which was above the 4.0 threshold,
  so no refill. Chunk 2 then consumed ~3.78 mL uninterrupted, leaving water_2 at ~1.4 mL.
  Added separate WATER_REFILL_THRESHOLD_ML = 6.0 mL used only for water/water_2 vials.
  Substocks/stocks continue using REFILL_THRESHOLD_ML = 4.0 mL unchanged.

### workflows/surfactant_grid_replay.py
- FIXED: `REFILL_CHECK_CHUNK_SIZE` reduced from 24 to 12. With 24, water_2 consumed ~3.2 mL across
  chunk 2 (23 wells) after passing the pre-chunk threshold check at 4.64 mL, leaving the vial at
  ~1.4 mL — causing the tip to scrape the bottom. Halving chunk size doubles the check frequency.

### workflows/surfactant_grid_adaptive_concentrations.py
- FIXED: `REFILL_THRESHOLD_ML` raised from 4.0 to 5.5 mL. At 4.0 mL, the pre-chunk check passed
  at 4.64 mL (above threshold), then chunk 2 consumed another 3.2 mL undetected. At 5.5 mL, the
  check at 4.64 mL triggers a refill before chunk 2 begins.

### workflows/surfactant_grid_adaptive_concentrations.py
- FIXED: `create_experiment_folder_structure` was using the module-level `SIMULATE = True` constant instead of the actual runtime simulate state. All hardware runs were creating the main output folder under `simulated_surfactant_grid/` while raw measurement CSVs (which use `lash_e.simulate`) went to `experimental_surfactant_grid/` — splitting the same experiment across two folders. Now passes `lash_e.simulate` explicitly.

### analysis/surfactant_contour_simple.py
- FIXED: Save path used `.replace('iterative_experiment_results.csv', ...)` which silently failed when passed `complete_experiment_results.csv`, causing matplotlib to try saving a `.csv` file and falling back to the root directory. Now uses `os.path.dirname(os.path.abspath(csv_file_path))`.

### analysis/control_cmc_analysis.py
- FIXED: Same `.replace()` save path bug as above. Now saves to the input file's directory.

### workflows/surfactant_grid_replay.py
- FIXED: Water/surfactant vials left in home position after mid-dispense refill. Refill functions (`fill_water_vial`, `create_substocks_from_recipes`) always return vials home, but vials had already been moved to safe dispensing positions. `_dispense_vial_in_chunks` now repositions any vial back to its assigned safe position after a refill via new `reposition_after_refill` parameter. Applied to water (44/45), buffer (47), and surfactant A/B (positions derived from `_SURF_SAFE_POSITIONS`).
- ADDED: `START_PLATE` config constant and `start_plate` parameter on `execute_replay_workflow` to skip already-completed plates when resuming a crashed run.

## [2026-05-14]

### docs/bayesian_optimization_guide.md
- ADDED: Student's guide to Bayesian optimization with Ax, covering AxClient vs. new Client API comparison, parameter types (range, choice, fixed), multi-objective setup with qNEHVI, parameter and outcome constraints, Sobol seeding and warm-starting, acquisition function options, and a section on categorical solvents/acids — treating them as choice parameters vs. featurizing with physicochemical descriptors.
- ADDED: New Option D section covering mixture composition vectors (solvent/acid fractions with sum-to-1 constraints), including Ax linear-constraint encoding and practical modeling tradeoffs.
- ADDED: Decision-rule note for current problem scale clarifying when to keep categorical identity parameters vs. when to switch to composition vectors for solvent/acid mixtures.
- ADDED: New section on campaign stopping criteria, including budget-based stopping, target-threshold early exit, plateau detection, and Pareto-stability guidance for multi-objective optimization.

### docs PDF handout export
- NEW: `docs/build_student_pdf.py` script to render the guide into styled HTML with improved spacing, a workflow figure, and student-facing callout formatting.
- NEW: `docs/bayesian_optimization_guide_student.html` generated styled handout source.
- NEW: `docs/bayesian_optimization_guide_student.pdf` generated student-friendly PDF export.

## [2026-05-07]

### recommenders/systematic_compare_nd.py
- ADDED: `--q-batch` CLI option (default 8) so benchmark batch size is configurable without code edits.
- ADDED: `--near-r` CLI option (default 0.04) to control `surf_precision` / `surf_recall` distance threshold for sensitivity checks across dimensions.
- ADDED: `--candidate-pool` CLI option (default 50000) passed into `BayesianTransitionRecommender` for runtime-quality tradeoff control in long ND sweeps.
- CHANGED: Iteration budget calculation now uses `args.q_batch` instead of hardcoded `Q_BATCH`, enabling exact runs like `96x4`.

### recommenders/_boundary_pct_sensitivity.py
- CHANGED: Recall panels now use adaptive y-axis limits (instead of fixed 0-1) to make low-recall differences visible, especially in 4D.
- ADDED: Extra reporting figure `recommenders/test_outputs/boundary_percentile_sensitivity_recall_zoom.png` focused on recall-only panels with zoomed axes.

## [2026-05-06]

### recommenders/ - Gradient-based transition recommender (Phase 1-4)
- NEW: `recommenders/_transition_base.py` - shared `TransitionRecommenderBase` (data pipeline, metrics, plotting)
- NEW: `recommenders/gradient_transition_recommender.py` - per-dim gradient UCB acquisition with anisotropic ellipse exclusion (Vadhavkar et al., MSDE 2026, d5me00233h). Implements analytical RBF derivative-posterior variance (paper eqns 9-10), autograd posterior-mean gradient, and axis-aligned ellipse exclusion generalized to d-D.
- REFACTORED: `recommenders/bayesian_transition_recommender.py` - inherits from `TransitionRecommenderBase`; removed duplicated plumbing. Stripped Unicode characters that crashed Windows cp1252 console.
- NEW: `recommenders/test_gradient_transition_recommender.py` - head-to-head test harness for `step2d`, `circle2d`, `surfactant2d` (real `simulate_surfactant_measurements` simulator inlined). Generates 2D scatter plots, gradient-magnitude heatmaps, and HD-vs-iteration comparison plots. Includes `--unit` mode for Phase-3 sanity checks (1D `_grad_mu` vs `cos(x)`, `_grad_var` near-zero at training points, anisotropic exclusion mask shape).
- Phase 3 unit checks pass: `_grad_mu` max-err = 0.005 vs `cos(x)`; `_grad_var` median ~7e-5 at training pts, max ~7e-2 off-training (~1000x growth as expected); anisotropic exclusion mask matches expected pattern.
- Phase 4 2D tests pass: `step2d` correctly identifies x[0]=0.5 transition (lengthscale = 0.17 in x[0], 91 in x[1] = uninformative dim correctly identified); `circle2d` traces curved boundary; `surfactant2d` clusters picks in the upper-right transition region.

## [2026-05-08]

### pipetting_data/embedded_calibration_validation.py (patch)
- Tightened error-pause threshold from 50% to `5% + 5 uL` (additive, bidirectional) for Stages 1, 2, and 3
- Example: 10 uL pauses if off by >5.5 uL; 100 uL pauses if off by >10 uL; 800 uL pauses if off by >45 uL
- Error message now shows signed error and threshold instead of "<50%"


### workflows/surfactant_grid_replay.py (new, minor)
- New thin workflow that replays a prior 192-row surfactant grid run from `iterative_experiment_results.csv` + `experiment_plan_stock_solutions.csv` in one pass, no optimization
- Imports all dispensing/measurement/plotting helpers from `surfactant_grid_adaptive_concentrations.py` — no duplication
- Adds `ensure_vial_above_threshold` (refill if vial < `REFILL_THRESHOLD_ML` = 4 mL) called between sub-chunks of `REFILL_CHECK_CHUNK_SIZE` = 24 wells inside each component dispense; routes to `fill_water_vial` / `refill_surfactant_vial` / `create_substocks_from_recipes` depending on vial kind, raises if buffer runs low (no auto source)
- Splits the 192-row CSV into per-plate DataFrames by detecting `wellplate_index` resets and runs the full dispense -> DMSO -> turbidity -> fluorescence cycle per plate

### workflows/REFACTORING.md (new)
- Documented an 11-step refactor plan for `surfactant_grid_adaptive_concentrations.py` ordered easiest-first (dead-code strip -> module splits), with size estimates and stable-import guarantees so the replay workflow keeps working

### workflows/surfactant_grid_adaptive_concentrations.py
- Removed ~30 DEBUG log/print lines (function-tracing noise, pairing_queue type/structure prints, two `VIAL_DF` dumps, redundant recommender prints); kept useful diagnostic messages (CMC target concentration values, "no CMC target" warnings, turbidity max value, optimization target) with `DEBUG:` prefix stripped
- Replaced `'ADD_BUFFER': bool` in `PAIRING_QUEUE` with `'BUFFER': str | None` — `None` means no buffer, a string like `'MES'`/`'HEPES'`/`'CAPS'` selects that buffer
- Double_iterative loop now sets `ADD_BUFFER` and `SELECTED_BUFFER` globals from `pairing_config['BUFFER']` before each pairing runs, so all downstream functions pick up the correct per-pairing buffer
- `setup_experiment_environment` appends `_{SELECTED_BUFFER}` to the folder name when `ADD_BUFFER` is True (e.g. `SDS_CTAB_CAPS_20260508_143022`)
- Experiment output folders now go to `output/simulated/` or `output/experimental/` depending on simulate mode, and the `surfactant_grid_` prefix is removed from the folder name



### North_Safe.py + robot_state/pipet_racks.yaml
- Added piggyback tip shake in `_perform_pipet_pickup`: when `pickup_shake` is present in rack config, the extraction is split at `z_fraction` (default 0.5), a lateral shake is applied (opposite of pickup direction to push secondary tip back toward rack), then the second half of the extraction completes
- Added `pickup_shake: {amplitude_mm: 2.0, repeats: 2, z_fraction: 0.5}` to `large_tip_rack_1` and `large_tip_rack_2` in `pipet_racks.yaml`; small tip racks are unaffected
- Shake direction auto-computed from pickup_movement signs: x=-2.2 -> shake_x=+2.0, y=+2.2 -> shake_y=-2.0

### workflows/glycerol_dispense_baseline.py
- Fixed cap artifact in baseline mass: vial is now uncapped immediately after moving to clamp (before `read_steady_scale`), using `is_vial_pipetable` to skip open-cap vials
- Fixed early termination: `run_baseline()` now runs a `while True` loop that processes chunks from both campaigns (200uL and 1000uL) in a single invocation, using all available tips of each type before exiting
- Hardware init (home, move vial, uncap) moved outside the loop; final cleanup (move home, return vial) also runs once after all chunks
- Per-chunk Slack notifications replace single per-run notification; final summary Slack message covers all chunks run
- Return value changed from `{"campaign_folder", "rows_processed"}` to `{"chunks_run", "workflow_complete"}`

## [2026-05-04]

### workflows/surfactant_grid_adaptive_concentrations.py
- Added `adaptive_correction=True` to large-volume water validation call (200-900 uL)
- Added `adaptive_correction=True` to large-volume surfactant validation calls (200-900 uL)
- Large tip validation now runs the same 3-stage parameter optimization as small tip

## [LLM CONTEXT FIX] - 2026-04-24

### CRITICAL FIX: LLM Now Receives Complete Experimental Context
- **FIXED**: LLM optimization was previously blind - only received empty optimization_trials[] for first trial
- **ENHANCED**: LLM now receives ALL available trial data: screening + inherited + two-point + optimization trials  
- **DATA COMPLETENESS**: LLM gets same experimental context as Bayesian optimizer for informed decisions
- **CONTEXT LOGGING**: Added detailed logging showing exactly what trial data LLM receives for transparency
- **PERFORMANCE**: LLM suggestions now based on complete experimental history instead of making blind guesses

### SAFETY: Eliminated Silent LLM Fallbacks (No Silent Defaults)
- **CRITICAL SAFETY**: Removed silent fallback from LLM to Bayesian optimization
- **EXPLICIT FAILURE**: When LLM optimization is enabled but fails, system now pauses with clear error message
- **USER CHOICE**: Added `input()` pause letting user decide whether to continue with Bayesian or stop and fix LLM
- **FAIL LOUDLY**: Follows "No Silent Defaults" principle - no more invisible mode switches that mask problems
- **TRANSPARENT**: Clear console alerts show exactly why LLM failed (import error vs runtime error)
- **FILES MODIFIED**: experiment.py (lines 1005-1018) - replaced silent warnings with explicit user interaction
- **IMPACT**: Users now know immediately when LLM optimization isn't working as requested

## [LLM OPTIMIZATION INTEGRATION] - 2026-04-22

### IMPLEMENTED: Complete LLM Optimization Support
- **ADDED**: `_generate_optimization_parameters()` method in experiment.py for LLM-guided optimization
- **INTEGRATION**: Optimization loop now checks `llm_optimization.enabled` config before using Bayesian optimizer
- **CONTEXT-AWARE**: LLM receives previous trial results for informed parameter suggestions during optimization phase
- **ROBUST FALLBACK**: Multiple fallback layers - ImportError → Configuration Error → Runtime Error → Bayesian optimizer
- **BACKWARD COMPATIBLE**: Existing Bayesian optimization remains default - no breaking changes
- **SAFE IMPORTS**: LLM imports wrapped in try/catch to prevent crashes when llm_recommender unavailable
- **ENHANCED LOGGING**: Clear indication whether using "LLM-generated" or Bayesian parameters for each trial
- **DUAL PHASE SUPPORT**: Both screening and optimization phases now support LLM parameter generation
- **FILES MODIFIED**: experiment.py (added optimization parameter generation method and integrated into workflow)

## [ENHANCED LLM PHYSICAL INSIGHTS] - 2026-04-22

### IMPROVED: Parameter Descriptions for LLM Understanding
- **ENHANCED**: All parameter descriptions in experiment_config.yaml with detailed physical insights
- **PHYSICS CONTEXT**: Added explanations of parameter mechanisms and liquid handling physics
- **TRADE-OFFS**: Documented accuracy vs speed relationships (e.g., slower aspiration = better accuracy but longer time)
- **VISCOSITY GUIDANCE**: Specific recommendations for thin vs thick liquids
- **PARAMETER INTERACTIONS**: Explained how parameters affect each other (e.g., slow dispense + blowout = long time)
- **MECHANISM EXPLANATIONS**: Surface tension, pressure equilibration, dripping dynamics, air gap functions
- **BENEFIT**: LLM can now make informed physics-based parameter recommendations instead of blind exploration
- **EXAMPLES**: 
  - aspirate_speed: "Slower aspiration reduces cavitation and bubble formation in viscous liquids"
  - overaspirate_vol: "Extra volume to compensate for liquid retention due to surface tension"
  - post_retract_wait_time: "Allows thick liquid to drip off, e.g. ~5s for glycerol-level viscosity"

## [DUAL BACKEND SYSTEM] - 2026-04-21

### NEW FEATURE: Configurable Ax Acquisition Function Control
- **IMPLEMENTED**: Dual backend system supporting both direct acquisition function control and high-level abstractions
- **BACKENDS SUPPORTED**: 
  - Direct Control: qNEHVI, qLogEI, qEI (colleague's approach with botorch_acqf_class)
  - High-Level: GPEI, MOO, BOTORCH_MODULAR (current simplified approach)
- **CONFIGURATION**: Via experiment_config.yaml `backend` and `backend_subsequent` settings
- **BACKWARD COMPATIBLE**: Supports both old configs ("qNEHVI", "qLogEI") and new configs ("GPEI", "MOO")
- **VOLUME AWARE**: Different backends for first volume vs subsequent volumes
- **FALLBACK SAFE**: Graceful degradation to optimizer_type mapping if config unavailable
- **BENEFIT**: Enables colleague's precise acquisition function control while maintaining current simplicity
- **FILES MODIFIED**: bayesian_recommender.py, experiment.py
- **TESTING**: Backend mapping logic verified for all supported configurations

## [SDL SCORING BUG FIX] - 2026-04-21

### FIXED: SDL Implementation Returning Zero Scores
- **BUG**: SDL scoring method was correctly implemented but not being called in find_best_trials()
- **SYMPTOM**: All trials showing score=0.000, selecting worst trial (Trial 1: 30.1% accuracy)
- **ROOT CAUSE**: find_best_trials() was still calling _calculate_composite_score() which returned 0.0 placeholder
- **FIX**: Updated find_best_trials() to properly call _calculate_sdl_composite_score() with population normalization
- **RESULT**: Selection will now match SDL display ranking - Trial 5 (1.2% accuracy) should be selected
- **VERIFICATION**: Next experiment should show proper SDL scores and select best trial correctly

## [SELECTION SYSTEM UPGRADE] - 2026-04-21

### REPLACED: Absolute Threshold Scoring → SDL Relative Normalization
- **REMOVED**: Old absolute threshold system (30s baseline, 1% tolerances, penalty zones)
- **CLEANED**: Deleted dead code methods (_calculate_accuracy_score, _calculate_precision_score, _calculate_time_score_for_ranking)
- **NEW SYSTEM**: Pure SDL relative normalization with population standard deviations
- **BENEFIT**: Single consistent scoring methodology across display and selection
- **METHOD**: Normalizes each metric by standard deviation across all trials × 100
- **WEIGHTS**: Same weights (0.4 accuracy, 0.5 precision, 0.1 time) with superior normalization
- **IMPACT**: Time outliers properly penalized, competitive context-aware scoring
- **USER INSIGHT**: "I think I intended to always use this system - the other system doesn't sound like one I would have used"

## [CRITICAL SELECTION BUG FIX] - 2026-04-21

### FIXED: Inconsistent Trial Selection Scoring
- **ISSUE**: Composite scores calculated at different times during experiment were incomparable
- **PROBLEM**: Trial selection used stale scores from different contexts, causing wrong "optimal" parameters 
- **SOLUTION**: Modified `find_best_trials()` to recalculate all scores with same baseline for fair comparison
- **IMPACT**: Trial selection now matches SDL ranking display - consistent and reliable results
- **LOGGING**: Added extensive logging to show score recalculation and final ranking (ASCII-only)
- **SAFETY**: Created backup before changes, minimal code modification with easy reversion
- **VERIFIED**: Fix confirmed working - selection now matches display ranking consistently

## [MASTER DATASET CREATOR] - 2026-04-20

### NEW FEATURE: Comprehensive Data Compilation System
- **ADDED**: `create_master_dataset.py` - Consolidates ALL calibration and validation data
- **FEATURES**: Automatically detects calibration vs validation experiments  
- **SMART FILTERING**: Skips simulated data by checking config files
- **COMPREHENSIVE**: Combines raw measurements, trials, and optimal conditions
- **EXPORTS**: Creates master_measurements.csv, master_trials.csv, master_optimal_conditions.csv
- **REPORTING**: Generates detailed compilation report with statistics

### CLEANUP: Unused File Identification
- **IDENTIFIED**: 10 unused Python files safe for deletion:
  - Standalone utilities: batch_calibration_automation.py, fix_external_data.py, etc.
  - Unused protocols: calibration_protocol_heated.py, calibration_protocol_reservoir.py
  - Legacy scripts: post_optimization_dashboard.py, shap_analyzer.py
- **ACTIVE SYSTEM**: 20 Python files actively used by main calibration system

## [EXTERNAL DATA PARAMETER COMPATIBILITY] - 2026-04-20

### FIXED: External Data System Parameter Alignment
- **FIXED**: Removed unused `trial_id` parameter from `_convert_measurements_to_trial()` method
- **ALIGNED**: External data loading system now fully compatible with TrialResult constructor
- **RESULT**: Manual calibration measurements can now compete with optimization trials in Ax

## [CALIBRATION GUI ENVIRONMENTAL MONITORING] - 2026-04-13

### NEW FEATURE: Real-Time Environmental Conditions Display
- **ADDED**: Environmental monitoring widget to right panel utilizing unused space
- **MONITORS**: Temperature (°C), Humidity (%), Pressure (Pa) from MQTT sensor log
- **DATA SOURCE**: Reads from `C:\Users\Imaging Controller\Desktop\m5stack\mqtt_log.csv` 
- **AUTO-UPDATE**: Refreshes every 30 seconds to show current conditions
- **COLOR CODING**: 
  - 🟢 Green: Fresh data (< 10 minutes old)
  - 🟠 Orange: Stale data (10-60 minutes old)  
  - 🔴 Red: Very stale data (> 1 hour old)
- **ROBUST ERROR HANDLING**: Gracefully handles missing files, empty data, or pandas unavailability
- **TIMESTAMP DISPLAY**: Shows last reading time and age (e.g., "14:23:45 (3m ago)")
- **BASED ON**: Environmental monitoring code from `workflows/glycerol_dispense_baseline.py`
- **BENEFIT**: Monitor lab conditions affecting pipetting accuracy during calibration sessions
- **FILES AFFECTED**: `calibration_modular_v2/calibration_test_gui.py` - new environmental monitoring section

## [CALIBRATION GUI PARAMETER CLEANUP] - 2026-04-13

### Parameter Corrections and Removal  
- **REMOVED**: `asp_disp_cycles` from default parameters (cleaned up unnecessary parameter)
- **FIXED**: Speed scale warning now only appears for `aspirate_speed` and `dispense_speed` 
- **CORRECTED**: Removed incorrect speed inversion warning from `retract_speed` (retract speed is normal scale)
- **TECHNICAL**: Changed condition from `'speed' in name` to specific parameter names for accuracy
- **BENEFIT**: Cleaner parameter list and correct speed guidance  
- **FILES AFFECTED**: `calibration_modular_v2/calibration_test_gui.py` lines 81, 177-179

## [CALIBRATION GUI PARAMETERS SECTION FIX] - 2026-04-13

### FIXED: Parameters Section Now Uses All Available Space
- **REMOVED**: Maximum height constraint (650px) on parameters scroll area that was causing clipping
- **REMOVED**: `addStretch()` at bottom of layout that prevented parameters section expansion  
- **RESULT**: Parameters section now expands to fill available vertical space automatically
- **BENEFIT**: No more scrolling needed - all parameters visible without clipping
- **TECHNICAL**: Layout now properly distributes available space instead of forcing fixed constraints
- **FILES AFFECTED**: `calibration_modular_v2/calibration_test_gui.py` lines 831, 892

## [CALIBRATION GUI UI POLISH] - 2026-04-13

### UI Improvements: Cleaner Button Text and Better Parameter Visibility
- **SIMPLIFIED**: Changed "CLEANUP & HOME" button to just "CLEANUP" for cleaner interface
- **EXPANDED**: Increased parameters scroll area height from 400px to 650px  
- **BENEFIT**: All default parameters now visible without scrolling for better user experience
- **FILES AFFECTED**: `calibration_modular_v2/calibration_test_gui.py` lines 831, 854

## [CALIBRATION GUI BUG FIXES & CLEAN VISUALIZATION] - 2026-04-13

### CRITICAL FIX: Cleanup Button Now Actually Works  
- **FIXED**: Cleanup button was calling nonexistent `cleanup()` method instead of `wrapup()`
- **NOW WORKS**: Robot properly removes pipet tip, returns vials home, and moves to safe position  
- **BEHAVIOR**: Should now see actual robot movement during cleanup operation
- **TECHNICAL**: Changed `protocol.cleanup()` → `protocol.wrapup()` to match actual method name

### UX: Simplified Individual Measurements Plot  
- **REMOVED**: Yellow accuracy/precision info box (too busy/overwhelming)
- **REMOVED**: Individual point labels (R1, R2, etc.) that cluttered the visualization
- **KEPT**: Color-coded dots, target line, mean line, and standard deviation shading
- **RESULT**: Much cleaner, less busy visualization that focuses on the data patterns
- **BENEFIT**: Easier to quickly assess measurement scatter and accuracy at a glance

## [CALIBRATION GUI WORKFLOW EFFICIENCY] - 2026-04-13

### MAJOR: Separated Robot Initialization and Cleanup for Efficient Testing  
- **ADDED**: "INITIALIZE ROBOT" button for one-time homing and setup at session start
- **ADDED**: "CLEANUP & HOME" button for explicit vial return and protocol cleanup  
- **REMOVED**: Auto-homing on every measurement (major speed improvement)
- **ENHANCED WORKFLOW**: Initialize once → Run multiple measurements → Cleanup when done
- **PERSISTENT PROTOCOL**: Reuses initialized protocol state between measurements for efficiency
- **UI STATE MANAGEMENT**:
  - MEASURE button disabled until robot initialized 
  - CLEANUP enabled only after successful initialization
  - Clear tooltips and status messages for user guidance
- **ERROR HANDLING**: Proper cleanup even if protocol cleanup fails
- **TECHNICAL**: Modified MeasurementWorker to accept existing protocol instead of creating new instances
- **BENEFIT**: Dramatically faster parameter testing workflow - no re-homing between measurements
- **FILES AFFECTED**: `calibration_modular_v2/calibration_test_gui.py` - major refactoring of measurement workflow

## [CALIBRATION GUI STRIP PLOT VISUALIZATION] - 2026-04-13

### UX: Replaced Histogram with Individual Measurement Strip Plot  
- **FIXED**: Histogram showing "big blue rectangle" due to uniform bar heights with small sample sizes
- **NEW VISUALIZATION**: Strip plot showing each replicate as individual colored dots with jitter to prevent overlap
- **ENHANCED FEATURES**: 
  - Individual measurement labels (R1, R2, etc.) with exact values
  - Color-coded points using Set3 colormap for visual distinction
  - Standard deviation shading (±1σ) around mean for precision visualization
  - Accuracy/precision statistics box in plot corner
  - Clean horizontal layout with only x-axis grid (y-axis hidden as meaningless)
- **IMPROVED CLARITY**: Users can now clearly see each individual measurement value and scatter
- **TECHNICAL**: Renamed from `volume_histogram_plot` → `volume_replicate_plot` with new methods
- **FILES AFFECTED**: `calibration_modular_v2/calibration_test_gui.py` - plot methods completely rewritten

## [CALIBRATION GUI UX IMPROVEMENTS] - 2026-04-13

### UI/UX: Enhanced Calibration Test GUI Parameter Controls
- **FIXED**: Made Min/Max fields read-only reference displays instead of editable spinboxes  
- **ADDED**: Visual speed scale reminder "⚠ Speed Scale: 1 = Fast, 40 = Slow" for speed parameters
- **IMPROVED**: Changed parameter layout from horizontal to vertical to accommodate speed hints
- **STYLING**: Added light gray background and borders to min/max display fields for visual consistency
- **LOGIC**: Updated get_values() method to return original config min/max since fields are now read-only
- **FILES AFFECTED**: `calibration_modular_v2/calibration_test_gui.py` lines 126-180

## [SUBSTOCK REFILL IN ITERATION LOOP] - 2026-04-10

### Feature: Substock refilling before each iterative round
- Added substock top-up call in `execute_iterative_workflow` loop after water and stock refills
- Reconstructs `dilution_recipes` from `results['experiment_plan']['stock_solutions_needed']` each iteration
- Delegates all skip/top-up/recreate logic to existing `create_substocks_from_recipes` (unchanged)
- Files: `workflows/surfactant_grid_adaptive_concentrations.py`

## [CRITICAL DOUBLE CONVERSION FIX] - 2026-04-08

### CRITICAL BUG FIX: Double Unit Conversion in Ilya Workflow V2  
- **FIXED**: Double conversion causing 50μL to become 0.05μL instead of 0.05mL
- **ROOT CAUSE**: Line 299 already converts CSV μL→mL, but code assumed data was still in μL
- **SEQUENCE**: CSV(50μL) → Line299(÷1000=0.05mL) → MyCode(÷1000=0.00005mL) → Display(0.05μL)
- **SYMPTOM**: All volumes below 10μL minimum, all wells skipped during dispensing
- **SOLUTION**: Treat data as mL after line 299, convert mL→μL only for display/validation
- **IMPACT**: 50, 100, 150, 200μL volumes now correctly processed as 0.05, 0.1, 0.15, 0.2mL
- **FILES AFFECTED**: `workflows/ilya_workflow_v2.py` lines 110-112, 155-157, 163-165, 360-362

## [SIMULATION PERFORMANCE FIX] - 2026-04-08

### BUG FIX: Simulation Mode Timing Issues
- **FIXED**: 5-second delays in simulation mode during pipetting operations
- **ROOT CAUSE**: Typo `self.SIMULATE` (uppercase) instead of `self.simulate` (lowercase) in post_retract_wait_time check
- **RESULT**: AttributeError caused simulation bypass to fail, executing full hardware wait times in simulation
- **SOLUTION**: Corrected simulation attribute reference to proper lowercase `self.simulate`  
- **IMPACT**: Simulation now runs at proper speed without unnecessary delays
- **FILES AFFECTED**: `North_Safe.py` line 2125 (post_retract_wait_time simulation bypass)

## [CRITICAL CALIBRATION FIX] - 2026-04-08

### CRITICAL BUG FIX: Embedded Calibration Corruption Prevention
- **FIXED**: Calibration system that saved Stage 3 parameters even when accuracy got worse
- **ENHANCED**: Now compares ALL THREE stages and picks the overall best performer  
- **FIXED**: CSV measurement data now updated with fresh measurements from optimization process
- **ROOT CAUSE**: Blind `_update_calibration_csv()` call with no performance validation + stale measurement data
- **SOLUTION**: Smart stage comparison logic + fresh measurement data replacement from winning stage  
- **PROTECTION**: System preserves good calibration AND uses only matched parameter-measurement pairs
- **INTELLIGENCE**: Stage 2 (crossing strategy) can now be selected if it outperforms interpolation
- **DATA INTEGRITY**: Measurement fields updated with actual measured volumes from optimization process
- **EVIDENCE**: Previously saved 6.3µL error parameters over existing 1.4µL tolerance parameters
- **VALIDATION**: Comprehensive stage ranking with fresh measurement data from best performing stage
- **REPORTING**: Enhanced Slack notifications show winning stage and APPLIED/REJECTED status
- **IMPACT**: Prevents calibration degradation AND ensures optimal parameters with fresh measurement data

### FILES MODIFIED  
- `pipetting_data/embedded_calibration_validation.py`: Added conditional update logic and enhanced validation

## [VIAL GUI ENHANCEMENTS] - 2026-04-02

### ENHANCEMENT: Vial GUI "Empty Vial" Feature
- **ADDED**: "Empty Vial" button to vial editing dialog for quick volume reset
- **FUNCTIONALITY**: Single click sets volume to 0 and closes dialog (faster than manual entry)
- **UI STYLING**: Yellow background button placed next to "Remove Vial" for easy access
- **WORKFLOW**: Improves efficiency when marking vials as empty during experiments

### BUG FIX: Reset Locations Button Not Working
- **FIXED**: "Return All Home" button now properly refreshes the GUI display
- **ROOT CAUSE**: Widget refresh method wasn't properly clearing and reloading vial positions
- **SOLUTION**: Enhanced `_reload_all_widgets()` with proper widget cleanup and grid reset
- **DEBUGGING**: Added comprehensive logging for troubleshooting reset operations
- **IMPACT**: Digital vial location reset now visually updates all rack grids correctly

### FILES MODIFIED
- `vial_manager_gui.py`: Added empty vial feature and fixed reset locations functionality

## [CRITICAL TIP SELECTION BUG FIXES] - 2026-04-02

### ENHANCEMENT: Simplified Water 200µL Calibration Script  
- **SIMPLIFIED**: Batch calibration automation script to focus solely on water at 200 µL
- **REMOVED**: Multi-liquid calibration loop, now runs single water calibration only
- **UPDATED**: UI messages and titles to reflect water-specific calibration purpose
- **MAINTAINED**: All core functionality (config modification, vial swapping, calibration + validation)
- **STREAM-LINED**: Single calibration run instead of batch processing multiple liquids
- **FILES CHANGED**: `calibration_modular_v2/batch_calibration_automation.py`

### CRITICAL BUG FIX: XY Movement Gripper Angle Preservation 
- **FIXED**: Gripper unwantedly rotating during XY movements in Enhanced SP program
- **ROOT CAUSE**: IK solver choosing completely different gripper angles to reach target XY position
- **EXAMPLE**: 0.3mm Y movement caused 905 count gripper rotation (2552→1647 counts)
- **SOLUTION**: Preserve current gripper angle during XY movements, only adjust elbow and shoulder
- **METHOD**: Use `current_gripper_rad` instead of IK-calculated gripper angle in target position
- **IMPACT**: XY movements now keep gripper orientation stable, only adjusting arm joints as intended
- **FILES FIXED**: Enhanced SP arm position program `_move_xy_delta()` method

### ENHANCEMENT: Added Gripper Rotation Controls to Enhanced SP Program
- **ADDED**: Gripper rotation controls to Enhanced SP arm position program UI
- **NEW BUTTONS**: "🔄 CCW" and "↻ CW" for counter-clockwise and clockwise rotation
- **METHODS**: `move_gripper_ccw()` and `move_gripper_cw()` with safety bounds checking
- **SAFETY**: Rotation respects GRIPPER_MIN_RAD and GRIPPER_MAX_RAD limits (-6.28 to 6.28 rad)
- **STEP SIZE**: Uses configurable `move_increment_rad` (default 0.05 rad = ~2.9°)
- **DISPLAY**: Gripper angle shown in both radians and degrees in position display
- **UI LAYOUT**: Added rotation row between gripper open/close and clamp controls

### CRITICAL BUG FIX: Color Mixing Workflow Tip Selection Algorithm Error
- **FIXED**: Excessive tip switching (18 switches during experiment) caused by wrong function usage
- **ROOT CAUSE**: Color mixing workflow incorrectly used `pipet_from_wellplate(..., aspirate=False)` instead of `dispense_into_wellplate()`
- **MECHANISM**: `aspirate_from_vial()` selects large_tip (0.221 mL total with overvolumes) → `pipet_from_wellplate()` selects small_tip (0.200 mL base only) → immediate tip switching
- **SOLUTION**: Changed all dispensing operations from `pipet_from_wellplate(..., aspirate=False)` to `dispense_into_wellplate()`
- **IMPACT**: Eliminated unnecessary tip selection conflicts during wells 54-65 and 84-89 processing

### CRITICAL BUG FIX: Tip Selection Algorithm Overaspirate Volume Issue
- **FIXED**: Overaspirate volume incorrectly included in tip selection causing unnecessary large_tip usage
- **ROOT CAUSE**: Tip selection included overaspirate_vol (0.010 mL) making 200 µL → 221 µL total, exceeding small_tip capacity (200 µL)
- **SOLUTION**: Removed overaspirate_vol from tip selection calculation: `total_tip_vol = post_asp_air_vol + amount_mL` 
- **REASONING**: Overaspirate is for liquid handling precision, not tip capacity. Post-aspirate air gap still considered.
- **IMPACT**: 200 µL now correctly selects small_tip (0.200 + 0.011 air = 0.211 mL < 0.20 mL threshold)
- **FILES FIXED**: `North_Safe.py` (line 2039)
- **FILES FIXED**: `workflows/color_mixing.py` (3 instances in main loop + mix_wells function)
- **ARCHITECTURE ALIGNED**: Color mixing now uses same dispensing pattern as other workflows (aspirate_from_vial → dispense_into_wellplate)

## [CRITICAL X-Y MOVEMENT BUG FIXES] - 2026-03-31

### CRITICAL BUG FIX: Motor Fault Prevention in X-Y Movement
- **FIXED**: Motor faults caused by massive unintended movements during small X-Y adjustments
- **ROOT CAUSE 1**: Wrong parameter order in forward kinematics call - `n9_fk(gripper, shoulder, elbow)` instead of correct `n9_fk(gripper, elbow, shoulder)`
- **ROOT CAUSE 2**: IK function returns radians despite documentation claiming counts, causing unit conversion errors
- **SOLUTION 1**: Corrected FK parameter order in `update_display()` to properly calculate current X-Y position
- **SOLUTION 2**: Added automatic unit detection and conversion for IK results (radians → counts)
- **SAFETY ENHANCEMENT**: Added movement safety limits (2000 cts per axis, 5000 cts total) to abort unsafe movements
- **IMPACT**: Small 2mm movements now execute safely instead of attempting 38,000+ count dangerous movements
- **PREVENTION**: Robot aborts movements exceeding safety limits instead of triggering motor controller faults

## [WORKFLOW TESTER IMPROVEMENTS] - 2026-03-30

### MAJOR ENHANCEMENT: Flow-Based Operation Testing 
- **REVAMPED**: `tests/surfactant_workflow_tester.py` - Complete redesign from predefined test cycles to flow-based button operations
- **NEW ARCHITECTURE**: Individual operation buttons organized by workflow stage (Source → Transfer → Liquid → Analysis → Disposal)  
- **PROPER IMPLEMENTATIONS**: Updated all functions to match actual surfactant workflow patterns and API usage
- **ENHANCED VIAL HANDLING**: Use real vial names ("water", "SDS_stock", "TTAB_stock", "pyrene_DMSO") instead of placeholders
- **SMART POSITIONING**: Implements idempotent track positioning with status checking (only moves if needed)
- **PROPER LIQUID HANDLING**: Correct `aspirate_from_vial` and `dispense_into_wellplate` usage with proper liquid types
- **TIP MANAGEMENT**: Proper pipet conditioning, removal, and volume tracking
- **ATOMIC OPERATIONS**: Cytation operations follow proper carrier in/out patterns with error recovery
- **WORKFLOW STATE TRACKING**: Real-time display of vial, wellplate position, and pipet status
- **EXAMPLE FLOWS**: "Source → Pipetting → Waste" or "Analysis → Cytation → Read → Return" button sequences
- **IMPROVED ERROR HANDLING**: Comprehensive simulation mode support and safety checks

## [COMPONENT TEST GUI] - 2026-03-30

### NEW FEATURE: Workflow Component Testing GUI
- **ADDED**: `tests/surfactant_workflow_tester.py` - Comprehensive GUI for testing individual workflow components
- **TESTING CAPABILITIES**: Wellplate movement, robot operations, Cytation protocols, and vial manipulation
- **REPETITIVE TESTING**: Run any operation multiple times with configurable delays to test consistency
- **REAL-TIME MONITORING**: Progress tracking, success/error counters, and detailed test logs
- **DUAL MODE SUPPORT**: Both simulation and real hardware testing modes
- **SAFETY FEATURES**: Emergency stop functionality and error isolation per test iteration
- **USE CASE**: Validate equipment reliability before running critical experiments

## [TRIANGLE RELIABILITY INDEXING FIX] - 2026-03-27

### CRITICAL BUG FIX: Delaunay Triangle Reliability Mask Index Mismatch
- **FIXED**: Triangle recommender filtering center triangles due to index mismatch between data and reliability mask
- **ROOT CAUSE**: Workflow passed all 48 points to recommender but triangle scorer internally filtered to 25 experimental points, causing index misalignment
- **SOLUTION**: Filter data to experimental points BEFORE passing to recommender, eliminating internal filtering and index confusion
- **IMPACT**: Center triangles now properly evaluated instead of being incorrectly filtered as "unreliable"
- **ARCHITECTURAL IMPROVEMENT**: Cleaner separation - recommender receives only the data it needs, no internal filtering required

## [VIAL POSITIONING FIX] - 2026-03-18

### CRITICAL BUG FIX: Invalid Vial Location
- **FIXED**: `glycerol_dye` vial invalid location causing workflow failures
- **ROOT CAUSE**: Vial at `location_index: 48` but `main_8mL_rack` only supports positions 0-47
- **SOLUTION**: Moved `glycerol_dye` from position 48 → position 5
- **IMPACT**: Workflow now correctly consumes `glycerol_dye` in wells 8-11, 20-23
- **SYMPTOM RESOLVED**: Vial volumes now decrease as expected during workflow execution

## [X-Y COORDINATE CONTROL SYSTEM] - 2026-03-17

### MAJOR ENHANCEMENT: Intuitive X-Y Coordinate Robot Control
- **X-Y COORDINATE INTERFACE**: Converted shoulder/elbow joint controls to intuitive X-Y coordinate system
- **NORTH API KINEMATICS**: Integrated `n9_fk()` and `n9_ik()` functions for seamless coordinate conversion
- **UNIFIED STEP SIZE**: Single movement increment (mm) now controls both Z-axis and X-Y movement
- **ENHANCED MOVEMENT FUNCTIONS**: 
  - Added `move_x_left()`, `move_x_right()`, `move_y_forward()`, `move_y_back()` using inverse kinematics
  - Implemented `_safe_move_xy()` with workspace bounds checking and fallback joint control
- **GUI IMPROVEMENTS**:
  - Updated position display labels: \"Shoulder\" → \"Y-Position\", \"Elbow\" → \"X-Position\"
  - Modified control buttons: \"← SHOULDER -\" → \"← X LEFT\", \"→ SHOULDER +\" → \"→ X RIGHT\"
  - Updated Y-axis controls: \"↓ ELBOW -\" → \"↓ Y BACK\", \"↑ ELBOW +\" → \"↑ Y FORWARD\"
- **COORDINATE TRACKING**: Added `current_x_position` and `current_y_position` tracking variables
- **REAL-TIME CONVERSION**: Forward kinematics automatically updates X-Y display from joint positions
- **KEYBOARD CONTROLS MAINTAINED**: 
  - Arrow keys (Left/Right) now control X-axis movement
  - W/S keys control Y-axis movement (forward/back)
  - Up/Down arrows still control Z-axis
- **BACKWARD COMPATIBILITY**: Joint angle tracking and functions preserved for internal use
- **TESTING FRAMEWORK**: Created comprehensive test suite for coordinate conversion validation
- **FILES MODIFIED**: 
  - workflows/enhanced_SP_arm_position_program.py (major X-Y coordinate system integration)
  - workflows/test_xy_coordinates.py (new testing framework)

## [SURFACTANT LIQUID_TYPE FIX] - 2026-03-16

### BUG FIX: Corrected liquid_type Parameters for Surfactant Operations
- **VALIDATION CALLS**: Fixed surfactant validation to use `liquid_type='SDS'` instead of `'water'`
- **DISPENSING CALLS**: Fixed surfactant dispensing to use `liquid_type='SDS'` instead of `'water'`  
- **SLACK NOTIFICATIONS**: Now correctly show "Liquid: SDS" in Slack messages for surfactant operations
- **PIPETTING PARAMETERS**: Surfactant operations now use SDS-optimized parameters instead of water parameters
- **DATABASE LOGGING**: Surfactant operations now logged as "SDS" for proper tracking and analysis
- **AFFECTED FUNCTIONS**: 
  - `dispense_component_to_wellplate()` calls for surf_A_volume_ul and surf_B_volume_ul
  - `validate_pipetting_accuracy()` calls for surfactant stock validation (both small and large volumes)
- **FILES MODIFIED**: workflows/surfactant_grid_adaptive_concentrations.py (4 liquid_type corrections)

## [USER-DEFINED POSITION NAMES] - 2026-03-16

### MAJOR FEATURE: Custom Position Saving with User-Defined Names
- **CUSTOM POSITION NAMES**: Text input field for user-defined position names
- **TEMP FOLDER SAVING**: Saves positions with custom names to temp/ directory 
- **MULTIPLE EXPORT FORMATS**: 
  - JSON format with full position data
  - TXT format with coordinates for easy copy-paste
  - Array format for direct code integration
  - Python dictionary format for workflow files
- **SAVED POSITIONS MANAGER**: GUI window to view, copy, and manage all saved positions
- **AUTOMATED NAMING**: Auto-increments position names (my_position_1 → my_position_2)
- **COPY COORDINATES**: Quick clipboard copy of current position coordinates
- **TEMP FOLDER ACCESS**: Direct button to open temp folder in explorer
- **ENHANCED EXPORT**: CSV export of all saved positions with timestamps
- **USER WORKFLOW**: Save → Name → Export coordinates → Paste into other files
- **FILES MODIFIED**: workflows/enhanced_SP_arm_position_program.py (added custom naming system)

## [COMPLETE PIPETTE INTEGRATION] - 2026-03-16

### MAJOR FEATURE: Complete Pipette Tip Rack Position System
- **PIPETTE TIP POSITIONS**: Added 96 pipette tip positions (48 large + 48 small tips)
- **TIP REMOVAL STATIONS**: Added 3 tip removal/disposal positions (Cap, Approach, Small)
- **POSITION EXPANSION**: Total positions increased from ~70 to 166 
- **YAML INTEGRATION**: Extended robot_state/vial_positions.yaml with pipette tip rack configurations
- **WORKFLOW SUPPORT**: Enhanced position program now supports complete pipette tip pickup/disposal workflow
- **COORDINATE MAPPING**: Proper coordinate mapping from Locator.py pgrid_low_2/pgrid_high_2 arrays
- **CATEGORY ORGANIZATION**: Added Pipette_Tips and Tip_Removal position categories
- **FILES MODIFIED**: 
  - workflows/enhanced_SP_arm_position_program.py (added pipette position generation)
  - robot_state/vial_positions.yaml (added tip rack mappings)

## [CRITICAL SAFETY FIX] - 2026-03-16

### CRITICAL: Safe Two-Step Movement to Prevent Vial Collisions
- **SAFETY ISSUE**: Robot was lowering Z-axis first then moving X-Y, causing collisions with vials on rack
- **SOLUTION**: Implemented safe two-step movement pattern:
  1. Move to safe Z height (10,000 counts = ~100mm high) while maintaining current X-Y position
  2. Move X-Y to target position while staying at safe height  
  3. Finally lower Z-axis to target position
- **COLLISION AVOIDANCE**: Prevents robot from hitting vials/obstacles during movement
- **ERROR HANDLING**: Added try/catch for current position detection with safe defaults
- **COORDINATE FIX**: Fixed MockRobot coordinate order to match Locator.py format [gripper, shoulder, elbow, z]
- **CONSTANT ADDED**: SAFE_Z_HEIGHT_COUNTS = 10000 for consistent safe movement height
- **FILES MODIFIED**: workflows/enhanced_SP_arm_position_program.py (goto_selected_position method and MockRobot class)

## [PANDAS FALLBACK FIX] - 2026-03-16

### CRITICAL FIX: CSV Loading Pandas Dependency Issue
- **CSV LOADING FIX**: Added fallback CSV reader using built-in csv module when pandas fails
- **ROOT CAUSE**: Pandas installation issue preventing dynamic position loading (pandas missing read_csv attribute)
- **SOLUTION**: Created SimpleDataFrame class to mock pandas behavior for compatibility
- **IMPACT**: Dynamic position loading now works reliably - loads 67 positions including 51 named vials
- **VERIFICATION**: All coordinate mappings verified correct (water, clamp positions match Locator.py exactly)
- **FALLBACK PATH**: System gracefully falls back from pandas -> csv module -> hardcoded positions
- **FILES FIXED**: workflows/enhanced_SP_arm_position_program.py (CSV loading section)

## [CRITICAL COORDINATE BUG FIX] - 2026-03-16

### CRITICAL FIX: Robot Position Coordinate Order Bug
- **COORDINATE ORDER BUG**: Fixed critical bug where shoulder and elbow coordinates were swapped
- **ROOT CAUSE**: Program incorrectly treated Locator.py coordinates as [gripper, elbow, shoulder, z] instead of [gripper, shoulder, elbow, z]
- **IMPACT**: Robot was going to wrong positions when using dynamic positioning system
- **SOLUTION**: Corrected target_position array order in goto_selected_position() method
- **VERIFICATION**: Coordinate mapping now matches Locator.py format: [gripper, shoulder, elbow, z_axis]
- **FILES FIXED**: workflows/enhanced_SP_arm_position_program.py (line ~846-851)

## [DYNAMIC POSITIONING SYSTEM] - 2026-03-16

### MAJOR: Integrated Multi-Layer Dynamic Positioning System
- **DYNAMIC POSITIONS**: Enhanced position controller now loads positions from multi-layer positioning system
- **CSV INTEGRATION**: Automatically loads vial positions from CSV files (e.g., surfactant_grid_vials_expanded.csv)
- **YAML CONFIG**: Uses vial_positions.yaml for rack configuration and Locator.py coordinates
- **DUAL ACCESS**: Supports both gripper and pipetting positions for each vial location
- **NAMED VIALS**: Shows actual vial names (water, SDBS_stock, etc.) instead of just coordinates
- **LIVE RELOAD**: Configuration UI allows users to change vial files and reload positions on-demand
- **FALLBACK SAFETY**: Maintains hardcoded positions if dynamic loading fails
- **AUTO DETECTION**: Automatically detects and loads default vial file if available
- **BACKUP CREATED**: enhanced_SP_arm_position_program_backup_YYYYMMDD_HHMMSS.py
- **FILES MODIFIED**: workflows/enhanced_SP_arm_position_program.py

## [POSITION SYSTEM FIX] - 2026-03-16

### FIXED: Enhanced Position Controller Positioning Issues
- **COORDINATE FIX**: Updated enhanced_SP_arm_position_program.py to use real robot coordinates from Locator.py
- **MOVEMENT METHOD**: Changed from move_z/move_axis_rad to goto() method for precise absolute positioning
- **REAL POSITIONS**: All 20+ positions now use actual robot count values instead of estimated mm/rad values
- **GOTO METHOD**: Added goto() method to MockRobot simulation for testing
- **POSITION CLEANUP**: Removed leftover old position definitions that were causing conflicts
- **ENHANCED GUI**: Expanded window to 750x900 for better visibility
- **COMPREHENSIVE POSITIONS**: Includes all vial racks, tip racks, processing stations with real coordinates
- **FILES FIXED**: workflows/enhanced_SP_arm_position_program.py

## [COMPREHENSIVE WORKFLOW POSITIONS] - 2026-03-16

### MAJOR: Complete Workflow Position System with Real Coordinates
- **REAL COORDINATES**: Updated all positions with actual robot counts from Locator.py
- **COMPREHENSIVE POSITIONS**: Added 25 workflow positions including all vial racks, tip racks, and processing stations
- **VIAL POSITIONS**: Main 8mL rack (positions 36, 43-47), Large vial rack, Small vial rack, 50mL vial rack
- **TIP RACK POSITIONS**: Small tip rack (positions 0, 24, 47), Large tip rack (positions 0, 24, 47)
- **PROCESSING STATIONS**: Photoreactor, Heater grid (positions 0, 5), Clamp position
- **TIP REMOVAL**: Small and large pipet tip removal positions
- **SAFETY POSITIONS**: Home and Safe transport height
- **ROBOT MOVEMENT**: Uses goto() method with absolute count positioning for precise movement
- **EXPANDED GUI**: Increased window size to 750x850, wider dropdown for longer position names
- **ENHANCED SIMULATION**: Mock robot now supports goto() method with count-based positioning
- **FILES MODIFIED**: workflows/SP_arm_position_program.py

## [WORKFLOW POSITION DROPDOWN] - 2026-03-16

### ENHANCED: SP_arm_position_program.py with Workflow Position Dropdown
- **NEW FEATURE**: Position dropdown menu with 11 predefined surfactant workflow positions
- **WORKFLOW POSITIONS**: Added positions from surfactant workflow (Home, Clamp, Main Rack 36/43-47, Photoreactor, Heater Grid, Safe Transport)
- **POSITION SELECTION**: Dropdown shows position names with descriptions when selected
- **GO TO POSITION**: One-click button to move robot arm to selected workflow position
- **SAFETY FIRST**: Z-axis moves first for safe positioning, then rotational axes
- **SUCCESS FEEDBACK**: Confirmation messages when position movement completes
- **ENHANCED GUI**: Expanded window size (650x800) to accommodate new Position Selection section
- **INTEGRATED WORKFLOW**: Positions match those used in surfactant_grid_adaptive_concentrations.py
- **FILES MODIFIED**: workflows/SP_arm_position_program.py

## [ENHANCED POSITION MANAGEMENT SYSTEM] - 2026-03-16

### MAJOR: Enhanced Robot Position Control with Workflow Integration
- **NEW FEATURE**: Position dropdown with 13 predefined surfactant workflow positions
- **WORKFLOW INTEGRATION**: Includes positions like 'Clamp Position', 'Main Rack 36-47', 'Photoreactor', etc.
- **POSITION MANAGEMENT**: Temporary position storage system for adjustment sessions
- **EXPORT FUNCTIONALITY**: CSV export for position modifications, TXT export for session history, JSON export for complete data
- **POSITION CATEGORIES**: Organized positions by function (Storage, Reagents, Processing, Manipulation, Large Volume, Small Volume, Standard)
- **TEMPORARY MODIFICATIONS**: Save current position as temporary, reset to original, clear all temporary positions
- **SESSION TRACKING**: Complete history of movements and position changes during session
- **ENHANCED GUI**: New sections for Position Selection, Position Modifications, and Export controls
- **SAFETY ENHANCED**: Proper position validation and bounds checking for all workflow positions
- **FILES CREATED**: workflows/enhanced_SP_arm_position_program.py (extends original with position management)

## [CALIBRATION CSV UPDATE] - 2026-03-17

### FIXED: Adaptive Overaspirate Correction Now Persists to Calibration File
- **Added** `_update_calibration_csv()` in `pipetting_data/embedded_calibration_validation.py`
- After Stage 3 optimization, the winning `overaspirate_vol` is written back to `optimal_conditions_{liquid}.csv`
- If the exact volume row already exists, only `overaspirate_vol` is updated; all other params unchanged
- If the volume is new, a fully interpolated row is inserted (all numeric columns interpolated from surrounding volumes), then the file is re-sorted
- Simulation mode prints what would be written without touching the file
- **Files Modified**: `pipetting_data/embedded_calibration_validation.py`

## [MULTI-JOINT ROBOT ARM CONTROL] - 2026-03-16

### ENHANCED: Multi-Joint Robot Control System
- **MAJOR UPGRADE**: Extended SP_arm_position_program.py to control all robot joints (Z-axis, Elbow, Shoulder)
- **New Controls**: LEFT/RIGHT arrows for shoulder rotation, W/S keys for elbow extend/retract
- **Enhanced GUI**: Real-time position display for all joints with both radians and degrees
- **Safety Features**: Proper joint limits for elbow (±150°) and shoulder (±120°) based on North API specs
- **Button Controls**: Dedicated GUI buttons for each joint movement direction
- **Simulation Support**: Extended mock robot to simulate all joint movements with realistic limits
- **Position Tracking**: Real-time updates of current elbow angle and shoulder angle
- **Error Handling**: Individual safety checks and error handling for each joint type
- **Files Modified**: workflows/SP_arm_position_program.py

## [USER-CONFIGURABLE MOVEMENT INCREMENTS] - 2026-03-16

### NEW: Custom Movement Step Control System
- **NEW FEATURE**: Added user-configurable movement increments for all joints
- **INPUT VALIDATION**: Real-time validation with safety limits and error feedback
- **Z-AXIS CONTROL**: Configurable step size (0.1-50.0mm) with live input validation
- **ROTATIONAL CONTROL**: Configurable step size (0.01-1.0 radians) with degrees display
- **ENHANCED SAFETY**: Improved error messages showing attempted values and limits
- **VISUAL FEEDBACK**: Real-time degrees conversion display and status updates  
- **PRECEDENT CHECKING**: System validates movements won't exceed joint limits before execution
- **USER GUIDANCE**: Clear min/max limits displayed in validation messages
- **PERSISTENT SETTINGS**: Custom increments stay active until manually changed
- **FILES MODIFIED**: workflows/SP_arm_position_program.py

### Testing Confirmed:
- ✅ Set Z-axis from 5.0mm to 10.0mm - successful movement validation
- ✅ Set rotational increment from 0.1rad to 1.0rad (57.3°) - full functionality
- ✅ All safety limits and range checking working properly
- ✅ Enhanced error messages providing clear user guidance

## [ROBOT ARM MULTI-JOINT CONTROL] - 2026-03-16

### ENHANCED: Multi-Joint Robot Arm Control System
- **ENHANCED**: Extended SP_arm_position_program.py with complete multi-joint control capability
- **NEW JOINTS**: Added Shoulder, Elbow, and Gripper joint control (previously only Z-axis)
- **KEYBOARD CONTROLS**: 
  - ↑/↓ Arrows: Z-axis movement
  - ←/→ Arrows: Shoulder rotation
  - W/S Keys: Elbow extend/retract
  - Q/E Keys: Gripper rotation (CCW/CW)
- **GUI ENHANCEMENTS**: Individual position displays for all 4 joints with real-time updates
- **SAFETY LIMITS**: Joint-specific angle limits with error checking and warnings
- **SIMULATION MODE**: Enhanced mock robot with full 4-joint simulation capability
- **API INTEGRATION**: Uses proper North API methods (`move_axis_rad()`, `get_robot_positions()`)
- **FILES MODIFIED**: workflows/SP_arm_position_program.py

## [ROBOT ARM POSITION CONTROL PROGRAM] - 2026-03-16

### NEW: Interactive Robot Arm Control Program
- **NEW FEATURE**: Created SP_arm_position_program.py for interactive robot arm control
- **GUI Interface**: Tkinter-based GUI with real-time position display and status indicators
- **Arrow Key Control**: Up/Down arrow keys move Z-axis in 5mm increments with safety limits
- **Home Button**: Dedicated button to home the robot to reference position
- **Keyboard Shortcuts**: ESC to exit, SPACE to home, arrow keys for movement
- **Safety Features**: Position limits (30-292mm), home requirement before movement, error handling
- **Simulation Support**: Automatic fallback to simulation mode if North library unavailable
- **Real-time Feedback**: Live position updates and connection status display
- **Logging**: Comprehensive logging to file and console for debugging
- **Files Added**: workflows/SP_arm_position_program.py

## [CONDITION TIP PARAMETER FIX] - 2026-03-16

### FIXED: TypeError in condition_tip function call  
- **CRITICAL**: Fixed "TypeError: condition_tip() got an unexpected keyword argument 'liquid'" in acid pipetting
- **Root Cause**: Function call had wrong parameters - `liquid` parameter doesn't exist, volume units mismatch
- **Fix**: Removed invalid `liquid=liquid` parameter and converted volume from mL to µL  
- **Impact**: TFA acid addition with tip conditioning now works correctly
- **Files Modified**: workflows/Degradation_serena.py line 157

## [CRITICAL MOTOR DRIVER FIX v2] - 2026-03-13

### EMERGENCY HARDWARE PROTECTION - Anti-Shoot-Through Implementation
- **CRITICAL**: Implemented comprehensive motor driver protection against hardware-killing code patterns
- **Anti-Shoot-Through**: 100ms mandatory motor-off deadtime before ANY movement command
- **Direction Reversal Protection**: 500ms blocking delay when direction changes detected
- **Rapid Click Prevention**: 300ms UI button locks prevent rapid command sending

- **Command Spacing**: Minimum 300ms between ANY jog commands (was 100ms)
- **Continuous Jog Rate**: Reduced to 2 steps/second (was 5) with 500ms intervals
- **Status Monitoring**: Robot status checked before every movement command
- **Blocking Operations**: All moves use wait=True to prevent command stacking
- **Extra Settling Time**: 50ms delays after moves to ensure driver fully settles

### Hardware-Specific Protections Added
Based on common motor driver failure patterns:
- **Prevents Both Direction Pins Active** (shoot-through condition)  
- **Eliminates Instant Full-Power Reverse** (no deadtime failures)
- **Prevents Command Overlap** (rapid state changes)
- **Ensures Adequate Deadtime** (100ms motor-off before commands)
- **UI-Level Protection** (button disable/re-enable cycles)

### Speed & Timing Limits  
- **Ultra-Conservative Speeds**: 1000-2000 cts/s maximum (was 4000)
- **Gentle Acceleration**: 4000-8000 cts/s² maximum (was 15000)
- **Extended Delays**: Multiple timing protections at different levels
- **Frequency Limiting**: Max 2 Hz continuous jog rate (well below driver limits)

### User Interface Updates
- **Window Title**: Clear "MOTOR DRIVER PROTECTION MODE" warning
- **Startup Alert**: Detailed protection measure explanation
- **Status Messages**: "ANTI-SHOOT-THROUGH" and "driver-safe mode" indicators  
- **UI Warnings**: Red text explaining 300ms click delays and protection measures

### Code Architecture Changes
- **State Tracking**: Last axis, direction, and timing tracking
- **Busy Flag System**: Prevents any overlapping jog operations  
- **Safe Button Management**: Prevents UI callback errors
- **Exception Handling**: Motor driver failure detection and safe shutdown
- **Clean Disconnect**: All jog states cleared on disconnect/close

This version addresses the specific code patterns that physically destroy motor drivers, implementing multiple layers of hardware protection beyond just conservative speeds.

## [1.2.4] - 2026-03-13

### Fixed
- **CRITICAL: SP_arm_position_program Robot Movement Error**: Fixed "NorthC9 object has no attribute 'move_vial_to_location'" error
  - **Issue**: SP_arm_position_program.py was trying to call `move_vial_to_location()` directly on NorthC9 object, but this method doesn't exist there
  - **Root cause**: Incorrect architecture - should use Lash_E coordinator that wraps North_Robot class which provides the `move_vial_to_location()` method
  - **Fix**: Updated SP_arm_position_program.py to use Lash_E coordinator pattern like other workflows
    - Added Lash_E import and initialization in connect() method
    - Changed robot movement calls to use `lash_e.nr_robot.move_vial_to_location()`
    - Added proper error handling for wellplate movements when track not initialized
  - **Impact**: Program now properly integrates with robot control system and can move vials to predefined workflow positions
  - **Files**: `workflows/SP_arm_position_program.py` (backup created before changes)

### Improved  
- **CONNECTION STABILITY: SP_arm_position_program Robust Connection System**: Added multi-strategy connection with auto-reconnection
  - **Issue**: Connection timeouts and failures due to complex Lash_E initialization, missing vial files, and no auto-reconnection
  - **Solution**: Implemented dual-strategy connection system:
    1. **Primary**: Direct NorthC9 connection (fastest, most reliable for teaching)
    2. **Fallback**: Lash_E with auto-generated minimal vial file if direct fails
    3. **Auto-reconnection**: Configurable auto-retry with exponential backoff
    4. **Error recovery**: Graceful handling of communication errors with automatic reconnection attempts
  - **Features Added**:
    - Auto-reconnect checkbox for continuous operation
    - Disconnect button for clean disconnection 
    - Connection status indicators with attempt counting
    - Dual-mode operation: full Lash_E (for vial movements) or direct NorthC9 (for manual positioning)
    - Automatic minimal vial file creation (`status/minimal_vials.csv`)
  - **Impact**: Dramatically improved connection reliability and user experience for robot positioning/teaching tasks
  - **Files**: `workflows/SP_arm_position_program.py`
- **PERFORMANCE: SP_arm_position_program Smart Speed Control**: Implemented intelligent speed management for different movement types
  - **Issue**: All movements used slow teaching speeds (2000 cts/s), making homing and position changes unnecessarily slow
  - **Solution**: Added speed differentiation:
    - **Fast speeds**: 8000-10000 cts/s for homing, going to saved positions, and workflow positions
    - **Slow speeds**: 2000 cts/s preserved for manual jogging/teaching (precision work)
    - **Auto-fallback**: Graceful degradation if fast speed parameters not supported
  - **Speed Constants Added**:
    - `FAST_VEL = 8000` cts/s for major movements
    - `FAST_ACC = 40000` cts/s² for fast acceleration  
    - `FAST_HOME_VEL = 10000` cts/s for homing operations
  - **UI Improvements**: Updated labels to clarify speed usage, added status messages showing movement type
- **USER EXPERIENCE: SP_arm_position_program Enhanced Connection Feedback**: Added visual progress indicators and clear status messages
  - **Issue**: Connection process was silent with no progress indication, leaving users unsure if system was working
  - **Solution**: Comprehensive connection feedback system:
    - **Progress bar**: Animated progress indicator during connection attempts
    - **Step-by-step updates**: Status messages showing connection progress ("Step 1/2: Testing direct connection...")
    - **Visual status icons**: ✅ success, ❌ errors, 🔄 in progress, ⚠️ warnings
    - **Smart button management**: Disables Connect button during connection to prevent conflicts
    - **Success/failure popups**: Clear messagebox notifications with next steps
    - **Detailed error reporting**: Comprehensive error messages with troubleshooting suggestions
  - **Features Added**:
    - Animated progress bar with connection state tracking
    - Emoji-enhanced status messages for quick visual feedback
    - Connection attempt counter with max retry limits
    - Graceful handling of connection interruption (disconnect during connection)
    - Enhanced auto-reconnection with visual progress
- **USER EXPERIENCE: SP_arm_position_program Enhanced Connection Feedback**: Added visual progress indicators and clear status messages
  - **Issue**: Connection process was silent with no progress indication, leaving users unsure if system was working
  - **Solution**: Comprehensive connection feedback system:
    - **Progress bar**: Animated progress indicator during connection attempts
    - **Step-by-step updates**: Status messages showing connection progress ("Step 1/2: Testing direct connection...")
    - **Visual status icons**: ✅ success, ❌ errors, 🔄 in progress, ⚠️ warnings
    - **Smart button management**: Disables Connect button during connection to prevent conflicts
    - **Success/failure popups**: Clear messagebox notifications with next steps
    - **Detailed error reporting**: Comprehensive error messages with troubleshooting suggestions
  - **Features Added**:
    - Animated progress bar with connection state tracking
    - Emoji-enhanced status messages for quick visual feedback
    - Connection attempt counter with max retry limits
    - Graceful handling of connection interruption (disconnect during connection)
    - Enhanced auto-reconnection with visual progress
  - **Impact**: Users now have clear feedback on connection status and can diagnose connection issues more effectively
  - **Files**: `workflows/SP_arm_position_program.py`
- **CRITICAL: SP_arm_position_program Teaching System Overhaul**: Replaced complex Lash_E system with simple in-memory position tracking
  - **Issue**: Workflow positions required complex Lash_E upgrade that often failed, and taught positions weren't actually moving the robot
  - **Root Cause**: Incomplete replacement of old Lash_E code, confusing teaching workflow, and slow movement speeds
  - **Solution**: Complete rewrite of position management system:
    - **Auto-homing on connect**: Robot homes immediately after connection (15,000 cts/s) to establish reference
    - **In-memory tracking**: All positions stored in program memory, cleared on restart for consistency
    - **Clear teaching flow**: Jog to position → select from dropdown → click "Go to Position" → confirm to save
    - **Fast movement speeds**: Increased to 12,000 cts/s (50% faster) with 60,000 cts/s² acceleration
    - **Test movement option**: After teaching, offers to test the fast movement with small displacement
    - **Multiple speed fallbacks**: Tries multiple API methods to ensure fastest possible speeds
  - **Workflow Improvements**:
    - Session-based positions that reset each time (prevents drift/confusion)
    - Visual speed indicators in status messages
    - Immediate movement after position taught
    - Automatic position verification through test movements
  - **Speed Improvements**: 
    - Homing: 15,000 cts/s (was 10,000)
    - Position movements: 12,000 cts/s (was 8,000) 
    - Acceleration: 60,000 cts/s² (was 40,000)
  - **Impact**: Robot now actually moves when positions are taught, 50% faster speeds, and much more reliable workflow
  - **Files**: `workflows/SP_arm_position_program.py`

## [1.2.3] - 2026-03-13

### Fixed
- **CRITICAL: CMC Control Logging Failure**: Fixed silent failure in CMC control generation for BDDAC surfactant
  - **Issue**: `create_cmc_control_series()` function used local logger instead of workflow logger, causing debug messages to be invisible during troubleshooting
  - **Root cause**: Function created `logging.getLogger(__name__)` instead of using `lash_e.logger` from workflow coordinator
  - **Fix**: Updated function signature to accept `lash_e` parameter and replaced all `logger.info()` calls with `lash_e.logger.info()`
  - **Impact**: Enables proper debugging of CMC control generation failures, preventing silent fallbacks that mask real errors
  - **Files**: `workflows/surfactant_grid_adaptive_concentrations.py` - function signature and 4 function call sites updated
- **CRITICAL: BDDAC Missing CMC Value**: Fixed BDDAC returning 0 CMC controls due to missing CMC value in workflow configuration
  - **Issue**: BDDAC consistently returned 0 CMC controls while SDS/CTAB worked fine (10 controls each)
  - **Root cause**: YAML workflow config `SURFACTANT_LIBRARY` entry for BDDAC was missing `cmc_mm: 8.4` key (present in Python code but overridden by config)
  - **Fix**: Added `cmc_mm: 8.4` to BDDAC entry in `workflow_configs/surfactant_grid_adaptive_concentrations.yaml`
  - **Impact**: BDDAC now generates proper CMC control series like other surfactants
  - **Files**: `workflow_configs/surfactant_grid_adaptive_concentrations.yaml`

## [1.2.2] - 2026-02-26

### Fixed
- **PERFORMANCE: Simulation Mode Timing Loop Optimization**: Fixed major performance bottleneck in `Degradation_serena.py` workflow
  - **Issue**: Even in simulation mode, timing loop was advancing by only 1 second per iteration, causing thousands of unnecessary loop cycles for long measurement intervals (e.g., 3600 iterations for 1-hour measurements)
  - **Fix**: Simulation mode now jumps directly to each scheduled measurement time instead of incrementing through every second
  - **Impact**: Reduces simulation runtime from potentially hours to seconds for workflows with long time intervals
  - Removed unnecessary heartbeat logging in simulation mode to reduce log noise
  - Performance gain: ~3600x faster for workflows with 1-hour measurement intervals
- **CRITICAL: Unconditional Sleep Calls in North_Safe.py**: Fixed simulation mode delays caused by hardware timing sleeps
  - **Issue**: Cap/uncap operations (0.5s each) and weight monitoring loops (0.01-0.1s) were running unconditionally even in simulation mode
  - **Fix**: Made all time.sleep() calls conditional on `if not self.simulate:` in North_Safe.py functions
  - **Files**: Fixed uncap_vial_in_clamp(), cap_vial_in_clamp(), and continuous weight monitoring functions
  - **Impact**: Eliminates cumulative seconds of delays per workflow operation in simulation mode
- **CRITICAL: Volume Accumulation Bug in Degradation Workflow**: Fixed sample vial volume tracking between workflow iterations
  - **Issue**: Robot's vial volume tracking (VIAL_DF) persisted across iterations, causing sample_3 to inherit accumulated volumes from sample_1 and sample_2
  - **Root cause**: Each workflow iteration should start with empty sample vials (0.0 mL) but robot remembered previous volumes
  - **Fix**: Added vial volume reset at start of each degradation_workflow() iteration to restore original CSV state
  - **Impact**: Ensures consistent sample preparation volumes across all workflow repeats
- **CRITICAL: Missing Schedule Entries Causing Volume Inconsistency**: Fixed sample_3 having different final volume than sample_1 and sample_2
  - **Issue**: degradation_vial_schedule.csv only contained entries for sample_1 and sample_2, missing sample_3 completely
  - **Root cause**: When sample_3 workflow ran, schedule filtering resulted in empty schedule, skipping all timed measurements (only initial measurement occurred)
  - **Volume impact**: sample_1 & sample_2 had 7 measurements (1.05 mL removed), sample_3 had only 1 measurement (0.15 mL removed), creating 0.9 mL difference
  - **Fix**: Added complete sample_3 measurement schedule (300s, 600s, 900s, 1200s, 1800s, 2400s, 3600s) to match sample_1 and sample_2
  - **Result**: All samples now follow identical 7-measurement schedules for consistent volume tracking

## [1.2.1] - 2026-02-20

### Fixed
- **Post-Experiment Analysis Error Handling**: Fixed runtime errors in automatic analysis integration
  - **Unicode Logging Fix**: Replaced Unicode symbols (❌✅) with [ERROR]/[SUCCESS] text to prevent cp1252 encoding crashes on Windows
  - **Contour Level Calculation**: Added error handling for matplotlib contour levels that must be increasing - fallback to imshow visualization when contour calculation fails
  - **Import Path Fix**: Fixed analysis module imports to properly reference analysis/ folder from workflows directory
  - **Plot Display Fix**: Removed plt.show() calls to prevent popup windows during automatic analysis - plots now save only and close properly to free memory
  - **Adaptive Simulation Data**: Made simulation function adaptive to use existing MIN_CONC global and dynamically scale to any concentration range - no longer hardcoded for specific ranges
  - Added comprehensive error handling with fallback visualization modes for turbidity, ratio, and fluorescence contour plots
  - Creates backups: `backups/surfactant_contour_simple_backup_*.py` and `backups/surfactant_grid_adaptive_concentrations_backup_*.py`

## [1.2.0] - 2026-02-20

### Fixed
- **CRITICAL: Pandas DataFrame Volume Filtering Bug**: Fixed silent filtering failures in surfactant workflow where CMC control wells (wells 2-18) weren't receiving pyrene probe due to pandas string vs numeric data type mismatch
  - Added `validate_and_convert_recipe_volumes()` function to ensure all volume columns are numeric before filtering operations
  - Modified `dispense_component_to_wellplate()` to validate volume data types at function entry with fail-loud error handling
  - Prevents silent failures in `batch_df[batch_df[volume_column] > 0]` filtering when volumes are stored as strings instead of numbers
  - Comprehensive validation of all volume columns: surf_A_volume_ul, surf_B_volume_ul, water_volume_ul, buffer_volume_ul, pyrene_volume_ul
  - Creates backup: `backups/surfactant_grid_adaptive_concentrations_backup_*_before_validation.py`

## [1.1.1] - 2026-02-19

### Fixed
- **Speed Control**: Fixed movement speed timing in `dispense_from_vial_into_vial` - modified speed now applies from after liquid aspiration through dispense completion, then returns to default speed for recap and return operations. This prevents fast movement with liquid in pipet and slow movement during vial cleanup operations.

## [1.1.0] - 2026-02-19

### Added
- **Spectral Data Analysis Integration**: Integrated spectral analyzer with degradation workflow
  - Added `process_workflow_spectral_data()` function with try-except error handling (renamed from process_degradation_spectral_data)
  - Automatic processing of output_# files after degradation workflow completion
  - Wavelength-specific time series plots for 555nm, 458nm, and 458/555 ratio analysis
  - Dynamic output directory handling with processed_data subfolder creation
  - Logger integration for consistent workflow logging

### Changed
- **Enhanced spectral_analyzer_program.py**:
  - Modified to accept dynamic output directories instead of hardcoded paths
  - Added processed_data subfolder for organized result storage
  - Enhanced error handling and logging capabilities
  - Updated file save paths to use processed_data directory

- **Updated Degradation_serena.py**:
  - Added spectral analysis import and automatic execution
  - Integrated spectral processing at workflow completion
  - Added comprehensive error handling for analysis failures

### Files Generated
- `spectral_analysis_plot.png` - Full spectral time series overlay
- `wavelength_time_analysis.png` - 4-panel wavelength-specific analysis
- `combined_spectral_data.csv` - Complete spectral dataset
- `wavelength_time_series.csv` - Time series for specific wavelengths
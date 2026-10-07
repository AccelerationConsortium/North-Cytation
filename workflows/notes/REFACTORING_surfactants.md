# Refactoring Guide: `workflows/surfactant_grid_adaptive_concentrations.py`

This document is for whoever picks up the simplification work on
`surfactant_grid_adaptive_concentrations.py`. Nothing here has been executed —
this is a plan with concrete step-by-step recommendations and an ordered
suggestion of where to start (easiest, lowest-risk wins first).

The current file is **~5,200 lines** with **~80 top-level definitions** and
mixes ~7 concerns:

1. Substock / dilution math (planning)
2. Robot-level dispensing primitives
3. Cytation measurement adapters
4. Heatmap / contour plotting
5. Adaptive-bounds + recommender / gradient logic
6. Top-level workflow orchestrators (3 near-duplicates)
7. Kinetics-time-series workflows (a separate experiment family)

The companion replay workflow in
[`workflows/surfactant_grid_replay.py`](surfactant_grid_replay.py) was
written to import from this file rather than copy code, so any refactor must
keep the public function names below stable (or update the replay file's
imports in the same commit):

- `setup_experiment_environment`
- `create_substocks_from_recipes`
- `dispense_component_to_wellplate`
- `position_surfactant_vials_by_concentration`
- `return_surfactant_vials_home`
- `return_water_vial_home`
- `dispense_dmso`
- `measure_and_process_turbidity`
- `measure_and_process_fluorescence`
- `fill_water_vial`
- `refill_surfactant_vial`
- `get_pipette_usage_breakdown`
- `run_post_experiment_analysis`
- `generate_surfactant_grid_heatmaps`
- Constants: `MEASUREMENT_INTERVAL`, `WELL_VOLUME_UL`, `BUFFER_VOLUME_UL`,
  `PYRENE_VOLUME_UL`, `ADD_BUFFER`, `SELECTED_BUFFER`, `INPUT_VIAL_STATUS_FILE`,
  `FINAL_SUBSTOCK_VOLUME_ML`, `REFILL_THRESHOLD_ML`

---

## Recommended order (easiest wins first)

The early steps are **mechanical**, **local**, and **revertable**. Pay-off per
hour of work drops as the list goes on. Stop when the file feels manageable
again — you do not have to do all nine remaining steps.

### ~~Step 1 — Strip dead `DEBUG:` log/print calls~~ DONE (2026-05-08)

### ~~Step 2 — Strip inline re-imports~~ DONE (2026-05-08)

### Step 3 — Replace per-call-site CSV-save try/except with the existing helper (~45 min)

**Difficulty: easy. Risk: low.**
**Note: the inline blocks construct their own output path (`output/sim_folder/experiment_name/measurement_backups/`) which differs from how `save_measurement_csv` is called elsewhere. Consolidating these requires computing and passing the right folder explicitly — not a one-liner. May add lines before removing them. Tackle after Step 6 (extract plotting) so the output-folder logic is clearer.**

`save_measurement_csv` (around line 46) already exists and handles filename
construction, directory creation, error logging, and simulation labelling.
Yet at lines ~3210, ~3290, and ~3845 the code repeats inline:

```python
try:
    timestamp = datetime.now().strftime(...)
    filename = f"turbidity_plate{...}_wells{...}_{timestamp}.csv"
    sim_folder = "simulated_..." if lash_e.simulate else "experimental_..."
    path = os.path.join("output", sim_folder, ...)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    turbidity_data.to_csv(path, index=True)
    ...
except Exception as e:
    lash_e.logger.error(...)
```

Each of these blocks is 15–20 lines. Replace with one call to
`save_measurement_csv(...)`. Net deletion ~50 lines.

### Step 4 — Collapse the three near-duplicate top-level workflows into one core (~2 hours)

**Difficulty: medium. Risk: medium (these are the entry points users run).**

Three functions all implement essentially the same setup → plan → substocks
→ dispense → measure → save sequence:

- `execute_adaptive_surfactant_screening` (line ~3697) — single-pass.
- `execute_iterative_workflow` (line ~3920) — adds an iteration loop.
- `execute_2_stage_workflow` (line ~5019) — adds a two-stage handoff.

Recommended structure:

```python
def run_grid_workflow(lash_e, surfactant_a_name, surfactant_b_name, *,
                     existing_stock_solutions=None,
                     number_concentrations=NUMBER_CONCENTRATIONS,
                     num_substocks=None,
                     bounds=None,            # dict or None
                     experiment_output_folder=None,
                     skip_measurements=False,
                     simulate=True) -> dict:
    """Single-pass core. Builds plan, dispenses, measures, saves. Returns
    the standard results dict."""
    # body = current execute_adaptive_surfactant_screening but with bounds
    # passed as a small dict instead of four separate kwargs.
```

Then:
- `execute_adaptive_surfactant_screening` → 5-line wrapper that calls the core.
- `execute_iterative_workflow` → loop around the core, with refill calls
  between iterations. Most of its body is duplicated from the core today.
- `execute_2_stage_workflow` → two calls to the core.

How to do this safely:
1. Add the new `run_grid_workflow` next to the others without removing
   anything.
2. Delete the body of each old function and have it call the core. Run
   `simulate=True` end-to-end after each replacement.
3. Once green, the iterative loop can be cleaned: today it inlines refill,
   substock recreation, and gradient-suggestion code that should each be
   their own helpers (Steps 5/6).

Net deletion: ~600 lines.

### Step 5 — Extract `recommenders/` content out of the workflow file (~1 hour)

**Difficulty: easy–medium. Risk: low.**

These belong elsewhere (the repo already has a [`recommenders/`](../recommenders/)
folder):

- `find_high_gradient_areas` (~L4085)
- `get_suggested_concentrations` (~L4743)
- `_create_bayesian_visualization` (~L4952)

How:
1. Move them to `recommenders/surfactant_grid_recommenders.py`.
2. Replace the originals with `from recommenders.surfactant_grid_recommenders import ...`.
3. Run `execute_iterative_workflow` once in simulation to confirm.

Net deletion from the main file: ~400 lines.

### Step 6 — Extract plotting / bounds calculation to `analysis/` (~1 hour)

**Difficulty: easy. Risk: low.**

- `generate_surfactant_grid_heatmaps` (~L458)
- `calculate_adaptive_concentration_bounds` (~L655)
- `run_post_experiment_analysis` (~L317)

These are pure plotting / data-summary functions with no robot dependency.
Move them to `analysis/surfactant_grid_visualization.py`. The replay
workflow already imports two of them from the original file — update that
import in the same PR.

Net deletion: ~400 lines.

### Step 7 — Move kinetics workflows to a new file (~1 hour)

**Difficulty: easy. Risk: low (kinetics is a separate use case).**

- `execute_kinetics_workflow` (~L4197)
- `execute_all_kinetics_sequences` (~L4264)
- `execute_single_kinetics_sequence` (~L4357)
- `pause_kinetics_measurements` / `resume_kinetics_measurements` (~L2353)
- `sleep_with_progress` (~L2369)

Move to `workflows/surfactant_kinetics.py`. They share helpers with the
grid workflow — those helpers stay in the main module (or move with
Step 8/9), and the kinetics file imports them.

Net deletion: ~600 lines.

### Step 8 — Extract substock planning to `workflows/lib/substock_planning.py` (~2 hours)

**Difficulty: medium. Risk: medium (this is the largest self-contained
unit, but it's pure logic with no robot calls except `get_vial_info`).**

Move:
- `class SurfactantSubstockTracker` (~L1078)
- `calculate_systematic_dilution_series` (~L1249)
- `calculate_smart_dilution_plan` (~L1308)
- `calculate_dilution_recipes` (~L1353)
- `create_substocks_from_recipes` (~L1450) — keep here even though it does
  robot calls; it's tightly coupled to the planning data structures.
- `create_plan_from_existing_stocks` (~L1722)
- `get_achievable_concentrations` (~L1819)
- `calculate_grid_concentrations` (~L1629)
- `calculate_dual_surfactant_grids` (~L1671)
- `rank_options_with_conservation_external` (~L1693)

How:
1. Create `workflows/lib/__init__.py` and `workflows/lib/substock_planning.py`.
2. Move the symbols. Keep their signatures identical.
3. In the original file, replace with one `from workflows.lib.substock_planning import *`.
4. Run a full simulated screening + a replay to confirm.

Net deletion: ~700 lines.

### Step 9 — Extract dispensing primitives (~1 hour)

**Difficulty: medium. Risk: medium.**

Move to `workflows/lib/dispensing.py`:
- `dispense_component_to_wellplate` (~L2114)
- `position_surfactant_vials_by_concentration` (~L2240)
- `return_surfactant_vials_home` (~L2308)
- `return_water_vial_home` (~L2320)
- `condition_tip` (~L951)
- `validate_and_convert_recipe_volumes` (~L2065)
- `shake_wellplate` (~L2392)

The replay workflow imports several of these — update its import line in
the same PR.

Net deletion: ~400 lines.

### Step 10 — Extract measurement adapters (~1 hour)

**Difficulty: medium. Risk: medium.**

Move to `workflows/lib/measurements.py`:
- `measure_turbidity` / `measure_fluorescence` (~L2563/L2594)
- `measure_turbidity_protocol_only` / `measure_fluorescence_protocol_only`
  (~L2400/L2470)
- `measure_and_process_turbidity` / `measure_and_process_fluorescence`
  (~L3160/L3282)
- `save_measurement_csv` / `merge_measurement_data` /
  `save_and_merge_measurement` (~L46–250)
- `backup_measurement_data` (~L907)

Net deletion: ~700 lines.

### Step 11 — Extract validation / recovery (~30 min)

**Difficulty: easy. Risk: low.**

Move to `workflows/lib/validation.py`:
- `validate_pipetting_system` (~L3517)
- `recover_raw_cytation_data` / `recover_from_measurement_backups` (referenced
  in the file header but defined deeper — find via `grep_search`).

Net deletion: ~200 lines.

---

## After all steps

Estimated end state:

- `workflows/surfactant_grid_adaptive_concentrations.py` — ~1,000–1,200
  lines: top-level workflow orchestrators + the core
  `create_complete_experiment_plan` + the module-level config block.
- `workflows/surfactant_grid_replay.py` — unchanged (already lean).
- `workflows/surfactant_kinetics.py` — kinetics workflows.
- `workflows/lib/substock_planning.py` — substock tracker + dilution math.
- `workflows/lib/dispensing.py` — robot dispensing primitives.
- `workflows/lib/measurements.py` — Cytation adapters + CSV save helpers.
- `workflows/lib/validation.py` — pipetting validation + recovery utilities.
- `recommenders/surfactant_grid_recommenders.py` — gradient / Bayesian
  suggestion logic.
- `analysis/surfactant_grid_visualization.py` — heatmaps + bounds + post-run
  analysis.

That's roughly 5,200 → 1,200 lines in the main file with ~6 small focused
modules holding the rest.

---

## Testing strategy at every step

For each PR:

1. Run the workflow it touches in `simulate=True` end-to-end. The log should
   look identical to the previous run except for the lines you intentionally
   removed.
2. For Steps 4, 8, 9, 10: also run
   `python workflows/surfactant_grid_replay.py` in simulation against a
   known 192-row CSV to catch regressions in the imports the replay
   workflow depends on.
3. Avoid commits that mix unrelated steps. Each step above is a separate PR.

## Anti-patterns the refactor should NOT introduce

These exist in the current file and should be removed (or at least not
extended) when touching the affected code:

- **Silent fallback defaults.** Several places use
  `data.get("key", hardcoded_value)` for values that must come from an
  authoritative source. Replace with `data["key"]` and let the KeyError
  surface. See the [copilot-instructions](../.github/copilot-instructions.md#critical-no-silent-defaults)
  for the canonical example.
- **Parallel arrays for metadata.** Always embed flags into the same
  DataFrame (e.g. `df['is_reliable_ratio'] = df['ratio'] <= 1.0`) rather
  than building a separate boolean mask that depends on the original index.
- **Unicode in log messages.** Windows PowerShell crashes on `μ`, `→`, `±`.
  Use `uL`, `->`, `+/-`. The current file is already mostly clean here —
  don't introduce new ones.
- **`print(...)` for debugging.** Use `lash_e.logger.info(...)` so output
  is captured in log files.

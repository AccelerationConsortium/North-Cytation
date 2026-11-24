# Surfactant target-samples workflow — plan

Date: 2026-11-16 · Updated 2026-11-24 · Status: **built and simulated, not run on hardware.** See "Built (2026-11-24)".
North_Safe is about to be updated (new refill/dispensing functions not yet synced);
refill and dispense steps below name *what* is needed and must be mapped onto the
synced functions before implementation.

## Goal

Given a list of target samples (which surfactants, at what final concentration,
whether an additive is present, plus free metadata), the workflow:

1. **Checks feasibility** against the stocks and substocks we actually have
   (identity, concentration, volume on hand).
2. **Plans and makes any missing substocks** by dilution, never pipetting below a
   set minimum volume, inserting intermediate dilutions automatically.
3. **Makes the samples** in wellplates the way the current workflow does, then
   measures as today.

Planning (1–2) is pure code with no hardware, so it can run as a dry run from the
scheduler or a script, and be unit-tested.

**Scope (decided 2026-11-16):** this is its own, simpler workflow. The target list
is decided elsewhere and handed in. There is no recommender, no Bayesian
optimisation, no grid generation and no iteration loop: one pass of
plan → make substocks → make samples → measure. It is not an engine for multidim
and does not replace Ailsa.

Proposed files: `workflows/lib/substock_planning.py` (planner; the home already
proposed in `docs/workflows/REFACTORING_surfactants.md` Step 8),
`workflows/surfactant_target_samples.py` (workflow), config
`workflow_configs/surfactant_target_samples.yaml`.

## What exists today (checked 2026-11-16)

- **Current workflow**: `workflows/surfactant_multidimensional_workflow.py` (hardware
  runs 11-07 to 11-12). Targets are generated internally (grid/simplex/recommender);
  helpers come from `workflows/surfactant_grid_adaptive_concentrations.py`.
  `workflows/surfactant_grid_ailsa.py` is a replay workflow reading per-well recipes
  from CSV and has the most robust refill logic.
- **Stocks are identified by name only**: vial CSVs have no identity/concentration
  column. `<S>_stock` concentration comes from `SURFACTANT_LIBRARY[S]["stock_conc"]`;
  `<S>_dilution_<n>` concentrations are *recomputed each run* from a fixed log series
  (`calculate_systematic_dilution_series`). If settings change, a leftover
  `SDS_dilution_2` is topped up as if it held a different concentration.
- **Substock recipes** (`calculate_dilution_recipes`): picks the source giving the
  smallest transfer ≥ `MIN_SUBSTOCK_VOLUME_UL` (200 uL), final 6 mL. If none
  qualifies it only logs "NEEDS INTERMEDIATE DILUTION" — no automatic intermediate.
- **Source selection per well** (multidim `joint_select_sources`): each surfactant
  volume in `[MIN_WELL_PIPETTE_VOLUME_UL=10, SURFACTANT_BUDGET_UL=225]`, total ≤ 225;
  picks the combination maximising (min volume, total). Ignores volume on hand.
- **Making samples** (`execute_dispensing_nd`): per surfactant, position vials
  (safe positions), condition tip once, aspirate per well → `dispense_into_wellplate`,
  grouped by tip from `resolve_pipetting_setup`; then water (split over `water`,
  `water_2`), optional buffer, pyrene/DMSO 5 uL; Cytation turbidity + fluorescence.
  Liquid class `"SDS"` is hard-coded for every surfactant.
- **Refills**: multidim tops up only between iterations, so a 96-well chunk can run a
  vial dry. Ailsa refills in chunks (`_dispense_vial_in_chunks`,
  `ensure_vial_above_threshold`, `validate_stock_refill_sources`).
- **Hidden globals**: multidim helpers silently use the adaptive module's defaults
  (e.g. `WELL_VOLUME_UL=200` vs multidim's 250, `PYRENE_VOLUME_UL`).
- **Registry**: `robot_state/components.yaml` (draft, not loaded) holds identities and
  MW; vial-contents storage is still undecided (see vial-contents plan).

The new workflow must not rely on name-based concentration inference or the
adaptive module's globals.

## Inputs

**Target file** (CSV), one row per sample:

| Column | Meaning |
|---|---|
| `sample_id` | unique, carried to all outputs |
| `<S>_conc_mm` | one column per surfactant, final in-well mM; blank/0 = absent. Any number of surfactants. |
| `<additive>` | one boolean column per additive (e.g. `pyrene`, `buffer`); present = fixed configured volume added. Concentration-based additives are a possible later extension (they'd be planned like surfactants), not in this version. |
| `replicates` | optional, default 1 |
| anything else | metadata, passed through untouched |

Surfactant names must be keys in `components.yaml` with `kind: surfactant`.

**Config**: well volume and plate type; minimum volumes (below); max substock final
volume; dead volume per vial type; liquid class per surfactant (default `SDS`, but
explicit); additive volumes; measurement protocols.

**Minimum volumes (the risk rule)** — all configurable:
- `MIN_WELL_TRANSFER_UL` — smallest volume pipetted into a well (today 10).
- `MIN_SUBSTOCK_TRANSFER_UL` — smallest source volume when making a substock (today 200).
- `MIN_WATER_DISPENSE_UL` — water remainders below this are handled as today.
Hence max dilution per step = `final_volume / MIN_SUBSTOCK_TRANSFER_UL`
(6 mL / 200 uL = 30×).

## Step i — Inventory and feasibility

1. **Inventory**: every vial with known identity + concentration + tracked volume.
   The vial CSV is **not** changed. Contents (identity, concentration, how long a
   solution has been open) come from sample tracking (`ContentsStore`,
   `sample_tracking.load_stocks()`). The planner reads/writes contents only through
   that interface, so it can be unit-tested against a fake. Vials
   whose contents are unknown are not used as concentration sources.
2. **Per-sample source assignment** (generalised `joint_select_sources`): for each
   surfactant in the sample, a source of concentration c is usable if
   `v = target × V_well / c` is within `[MIN_WELL_TRANSFER_UL, per-component max]`;
   pick the combination with total + additives + water reserve ≤ V_well, maximising
   the smallest volume (risk) then preferring existing vials over new substocks.
3. **Volume on hand**: sum each source's planned use + conditioning + dead volume
   (+ margin); compare with tracked volume + what can be refilled (`<S>_refill`
   vials, new North_Safe refill functions). Shortfalls on stocks → refill plan;
   shortfalls on substocks → top-up or remake.
4. **Classify each sample**: `ready` (existing sources), `needs_substock` (feasible
   once planned substocks exist), `infeasible` with a reason:
   - above max achievable (needs > stock concentration × budget),
   - combination exceeds well budget,
   - surfactant not in inventory and no stock,
   - not enough material even with refills.

## Step ii — Plan and make substocks

**Use what exists, make on demand (decided 2026-11-16).** For each target, an
existing stock or substock is used if it is within *healthy bounds*; only targets
no existing vial can serve healthily get new substocks. Healthy bounds (decided
2026-11-16):
- well transfer `t × V_well / c` within `[MIN_WELL_TRANSFER_UL, per-component max]`;
- enough volume on hand for all planned uses + dead volume + conditioning, after
  any refill/top-up that is possible;
- contents known.
No age or minimum-remaining-volume limit. How long a solution has been open is
kept as metadata (by the metadata system), not used to reject vials.
An existing vial that's healthy but gives a worse (smaller) transfer than a new
substock would is still used: reuse wins over making more vials.

**Choosing new concentrations (minimum number of vials).**
A substock at c serves target t if `t × V_well / v_max ≤ c ≤ t × V_well / v_min`.
Each target left uncovered gives an interval; the minimum set of concentrations
covering all of them is the greedy interval cover (sort by right end, place c at
the right end, rounded *down* to a "nice" value inside the interval). Rounding down
keeps transfers above the minimum. New substocks are made for this run but kept
(with recorded contents), so later runs can reuse them.

**Making each new concentration.** From the most concentrated available source
(existing or planned) with `c_src / c ≤ max dilution`: if none, insert intermediates
geometrically (e.g. stock 50 → 0.05 mM = 1000× → two steps of ~31.6×, rounded to
nice values that keep each step ≤ 30×). Intermediates are ordinary substocks and can
serve samples too. Order of creation = dependency order (topological; cycles
impossible by construction but checked).

**Final volume**: enough for planned use + dead volume + conditioning + margin, as
long as each step keeps the source transfer ≥ `MIN_SUBSTOCK_TRANSFER_UL` and fits
the vial; cap at the vial's capacity (decision 3).

**Concentrations recorded are computed from the actual volumes**, not the rounded
target, and recorded through the contents interface with preparation date and
recipe parentage.

**Physical vials**: new substocks need empty vials at known positions. The pre-run
report lists them (`main_8mL_rack [12]` style); the deck check confirms they're there
(decision 2).

**Making**: source → new vial (`dispense_from_vial_into_vial`, surfactant's liquid
class), water by reservoir, vortex — as `create_substocks_from_recipes` does today,
but driven by the plan and writing contents. Map to the synced North_Safe dispense /
refill functions.

## Step iii — Make the samples

One pass over the target list (no iterations). Reuse today's dispensing sequence
(`execute_dispensing_nd` behaviour) with the new source assignment:
- per surfactant: position vials, condition tip, dispense per well grouped by tip,
  dilute → concentrated;
- water remainder, optional buffer/additive, pyrene/DMSO (or configured dye);
- refill in chunks Ailsa-style (check before each chunk, refill/top-up, re-stage,
  re-condition) so a plate never runs a vial dry;
- liquid class from config per surfactant, not hard-coded;
- plate handling, controls and Cytation measurements as today;
- every sample row saved with its target, assigned sources, volumes, *achieved*
  concentrations, and metadata.

All functions take their settings as arguments — no reliance on another module's
globals.

## Pre-run report (dry run)

Printed and saved before any hardware motion, and available without hardware:
- per sample: status, sources, volumes, achieved concentrations;
- substocks to make: concentration, source, transfer volumes, final volume, vial
  position, dependency order;
- refills needed and from where; empty vials to place;
- infeasible samples with reasons;
- total consumption per vial vs on hand; tip counts.

Infeasible samples are dropped and reported (decision 4) — never a crash mid-run.

## Outputs

`output/surfactant_target_samples/<run>/`: `targets_input.csv` (copy),
`feasibility.csv`, `substock_plan.csv`, `well_recipes.csv`, `results.csv` (with
measurements). Contents of new substocks are recorded in the sample metadata system,
not in the vial CSV.

## Tests (mocked; named files only)

`software_tests/test_substock_planning.py`:
- one surfactant, targets reachable from stock → no substocks;
- low targets → minimum substock set; every transfer ≥ minimums;
- 1000× dilution → intermediates inserted, each step ≤ max dilution;
- existing leftover substock reused; a substock with unknown contents is ignored;
- multi-surfactant sample exceeding the well budget → infeasible with reason;
- insufficient volume → refill or remake planned; infeasible when nothing can cover it;
- achieved concentration computed from rounded volumes;
- determinism (same input → same plan).
Plus a simulated run of the workflow on a small target file.

## Decided (2026-11-16)

- Additives are fixed volume; concentration-based additives may be added later.
- No vial CSV change. Contents come from the sample tracking system
  (`sample_contents.py` / `sample_tracking.py`; decided 2026-11-24, tested as we build).
- Substocks: use existing vials when within healthy bounds, make new ones on demand.
- Healthy bounds = transfer within limits, enough volume, contents known. No age
  limit; open time is kept as metadata.
- Standalone, simple workflow: targets come from elsewhere; no setup, BO or iterations.
- (2026-11-24) Volumes: 20-240 uL into a well; substock transfer >= 100 uL, final volume ~4 mL
  (adjustable). Substocks chosen by lowest error, not largest volume: every sample under a
  max error limit, fewest new substocks, concentrations chosen from the targets. Planner:
  `workflows/helpers/substock_planning.py` (tests: `software_tests/test_substock_planning.py`).

## Open decisions

1. ~~Minimum volumes~~ decided 2026-11-24 (see Decided).
2. ~~Empty vials~~ decided 2026-11-24: as the scheduler works today, the planner proposes the
   vial file (stocks, substock empties, positions) and the user fills it in. Provide a template.
3. ~~Substock final volume~~ decided 2026-11-24: fixed, config `SUBSTOCK_VOLUME_ML` = 4.0.
   Sizing to need stays a possible later option.
4. ~~Infeasible samples~~ decided 2026-11-24:
   - physically impossible (volumes: above the cap, components overflow the well) -> warn, skip;
   - over the error limit -> warn, still make it by the lowest-error route (limit is a goal);
   - not enough stock -> not the design's concern; the agent that proposes the vial file sizes
     stock volumes from the planner's per-source totals;
   - unknown stock (not in sample tracking) -> error before the run; resolve first.
5. ~~Plate/well~~ decided 2026-11-24:
   - Plates: 96-well default (`WELLPLATE_TYPE`). The workflow splits the samples into plates
     and reads, taking a new plate when one is full; the agent and user supply enough plates
     (resource planning is not the workflow's job).
   - Batches: make then measure `BATCH_SIZE` wells at a time (e.g. 24) so samples don't sit;
     a batch does not span plates.
   - Well volume: `WELL_VOLUME_UL` = 250 default (config).
   - Controls: rows in the target file (e.g. blank = all 0); no special rules in the workflow.
   - Layout: each batch fills the next block of consecutive wells, randomized within that
     block only (`RANDOMIZED_ORDER`, `RANDOMIZATION_SEED`); matters most on absorbance plates.
     Helper: `workflows/helpers/plate_blocks.py` (tests: `software_tests/test_plate_blocks.py`).

## Built (2026-11-24)

- Workflow `workflows/surfactant_target_samples.py`; config `workflow_configs/surfactant_target_samples.yaml`;
  example targets `inputs/surfactant_target_samples_example.csv`; template vial file
  `status/surfactant_target_samples_vials.csv` (SDS stock + refill, water, 8 empty `SDS_sub_<n>`).
- Planner `workflows/helpers/substock_planning.py`; layout `workflows/helpers/plate_blocks.py`.
- Dry run: `python workflows/surfactant_target_samples.py --plan` -> report + planned CSVs in
  `output/surfactant_target_samples/plans/<time>/`.
- Tests: `software_tests/test_substock_planning.py`, `test_plate_blocks.py`,
  `test_surfactant_target_samples.py` (29 pass). Stress: 150 random planner trials; simulations of the
  example (48 wells) and 135 wells over 2 plates (6 substocks, top-ups) ran clean.
- `VIAL_DEAD_ML` = 2.0: North_Safe warns on small-tip draws below 2 mL.
- Not yet verified: live run; reader data merge (`raw_to_columns`) with real reads; two surfactants in a
  simulation (unit-tested only); reuse of leftover substocks (needs live sample-tracking records).

## Implementation notes (current syntax, checked 2026-11-18)

`robot` = `lash_e.nr_robot`. Volumes are mL. Making substocks and samples is now mostly
flow calls. Most of the work is the planner (pure code).

| Step | Call |
|---|---|
| Inventory | `ContentsStore().get(container_id)` (`experiment_tracking/sample_contents.py`) for what a vial holds; `sample_tracking.load_stocks()` for stock concentrations. Sample tracking is the contents source (decided 2026-11-24). |
| Comparing contents | `sample_contents.same_contents(a, b)` |
| New stock records | `sample_tracking.add_stock(components)` / `next_stock_id(material, stocks)` |
| Refills | `robot.refill_vial(vial, target_volume_mL, reservoir=... or source_vial=..., liquid=...)` (you must give the source; there's no automatic choice yet) |
| Make substock | `robot.create_substock(dest, [{"volume_mL", "reservoir"/"source_vial", "liquid"}], vortex_time=...)`; contents are recorded automatically in live runs |
| Make samples | `robot.dispense_from_vials_into_wellplate(df, liquid={vial: liquid})` per component; replaces `execute_dispensing_nd` |
| Plate / reads | `lash_e.grab_new_wellplate()`, `lash_e.measure_wellplate(protocols, wells)`, `lash_e.discard_used_wellplate()` |
| Output folder | `lash_e.save_run_info(output_dir)` |

- Refilling in chunks (Ailsa style) isn't in a flow. Simplest: check volumes before each `dispense_from_vials_into_wellplate` call, then `refill_vial`. The flow already refuses a vial that doesn't hold enough.

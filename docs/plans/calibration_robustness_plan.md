# Calibration Robustness Plan

Historical report retention update (2026-11-02): the October 30 overaspirate
review snapshot and CSV were retired with operator approval and remain in Git
history. Do not copy them back as part of the historical instructions below.
This note does not verify the older merge instructions or implementation status.

## START HERE (updated Fri 2026-10-30, ~21:15)

**The plan is implemented and tested in the copy `Desktop\utoronto_demo_calibration`.**
The main folder (`Desktop\utoronto_demo`), where Friday's experiment was running, has
only Friday's earlier changes (steps 1, 2, tip column, row cleanup), all uncommitted.

**Only step left: bring the copy into the main folder once the run has finished.**
Tell Claude "merge the calibration copy" — it copies exactly these files, commits,
and runs a simulation. Files to copy (everything else in the copy came *from* main):
- `pipetting_data/`: `embedded_calibration_validation.py`, `embedded_liquid_validations.py`,
  `pipetting_wizard.py`, `validation_decisions.py`, `optimal_conditions_water.csv`,
  `optimal_conditions_SDS.csv`, new `overaspirate_corrections.py`, new `overaspirate_estimation.py`
- `scheduler/`: `liquid_review.py`, `liquid_review_gui.py`
- `tests/`: `test_embedded_liquid_validations.py`, `test_liquid_review.py`,
  `test_surfactant_dispensing_tips.py`, `test_tip_calibration_identity.py`,
  `test_validation_decisions.py`, new `test_overaspirate_corrections.py`
- `docs/`: this file (the accompanying historical review reports are retired)
- Do NOT copy back: `experiment_tracking/experiment_runs.csv` (touched by a simulation),
  `robot_state/*`, `status/*`, `inputs/*`, `workflow_configs/*` (copied from main).

**What changed (chunks 1–4, Friday evening)**
1. Estimator `overaspirate_estimation.py` plus a historical review table
   (retired on 2026-11-02; available in Git history).
2. 7 rows set to their estimates (water small 50/70/100/150, SDS small 50/100/150).
3. GUI: single-V_OA sessions estimated (1:1 response), tip-aware coverage incl. "near
   (extrapolated)", tracked values shown, Past Validation reports each session's first
   (as-dispensed) batch instead of the best stage, corrections file in the staleness hash.
4. New validation `protocol="check_confirm"` **switched on** (`VALIDATION_PROTOCOL` in
   `embedded_liquid_validations.py`): 4 replicates at the value in use; confirm batch only
   if surprising; Kalman-damped update into `pipetting_data/overaspirate_corrections.csv`
   (base CSVs untouched); every used replicate accepted; one Slack summary per session;
   updates logged to `parameter_corrections.csv`. Validation never adds calibration rows.
   Undo: set `VALIDATION_PROTOCOL = "stages"`, or delete the corrections file.
   Recalibration guard: each tracked row stores the base value it started from; if a
   deliberate recalibration changes that base row, the correction is ignored (warning)
   and tracking restarts from the new base.

**Final review (Fri ~22:00)**: merge-list files unchanged in main since the snapshot (safe
to copy); main gained commit 60a021ef meanwhile, which touches none of them; other
planner users (fluorescence calibration, workflow template) work with the new default;
172 tests pass.

**Tests**: 128 pass in `north_env`; GUI/scheduler 42 pass in `north_async_test` (has
PySide6). Simulated Ailsa validation-only run with check_confirm: 7 sessions, 16 row
updates, all verdict types exercised, exit 0.

**Decisions still open (chunk 5)**
- DECIDED: a gross miss (>5% + 5 uL) no longer pauses or crashes: Slack message +
  warning, replicate left out of corrections, workflow continues (`_notify`). The
  near-zero check (all measurements <10% of target) still pauses with Slack: that means
  a real hardware problem (no tip, empty vial).
- Legacy stage code kept for now: `surfactant_grid_adaptive_concentrations.py`'s
  `validate_pipetting_system` calls it directly, and `surfactant_multidimensional_workflow.py`
  imports that when its `VALIDATE_LIQUIDS` is on (default off). User says those surfactant
  validations are out of date: next step is to switch them to `run_liquid_validations`
  (or drop them) and then delete the stage code.
- Rows with no history start with cautious default uncertainty (2% of volume), so their
  first update can move about halfway; fine, but watch the first real Slack summaries.
- Default `workflow_configs/surfactant_grid_ailsa.yaml` lists backup dye vial
  `dye_coumarin_6_2`, which isn't in the current vial file.

---

Status: design + partial implementation (started 2026-10-30).

## Guiding principle

**Calibration issues must never crash a workflow.** Anything calibration-related
(slightly out of range, tip borderline, validation drift) should degrade with a
warning, not raise. Keep the existing system of custom per-tip calibration
parameters that switch between tips automatically.

## Trigger incident (2026-10-30, live run, Ailsa surfactant grid)

`ValueError: No tip fits water at 193.6 uL: small_tip: 202.4 uL reaches tip capacity`

1. `run_liquid_validations` planned water 193.6 uL onto `small_tip` using the
   calibration at session start (extrapolated overaspirate ~+5.7 uL -> load ~199.3 uL).
2. The 150 uL validation then rewrote the 150 uL row in `optimal_conditions_water.csv`
   from **+1.9 -> -4.9 uL** overaspirate (driven by a likely-outlier stage-1 reading of 159 uL).
3. At 193.6 uL the steeper 150->180 slope extrapolated overaspirate to ~+8.8 uL ->
   load 202.4 uL > 200 uL. Validation pins the tip, so no fallback -> crash.

The 150 uL row is still at -4.9 uL (user chose not to revert yet).

## Root problems

1. Each CSV row is a hard interpolation anchor; extrapolation uses the last two
   rows, so one noisy row dominates.
2. Validation updates are undamped (full replacement from one session).
3. Tip choice hinges on fragile numbers (no headroom; strict `load < capacity`).
4. Validation adds rows at arbitrary volumes -> row creep, jagged curves.
5. Within a validation session, later volumes see earlier volumes' CSV writes.

## Ideas and decisions

| Id | Idea | Decision |
|----|------|----------|
| A | Fixed calibrated volumes per tip; validation never adds rows; a measurement at v is split between the two neighbouring rows | **Agreed** |
| B | Damped updates (partial step, capped, require agreement) | **Agreed** — see update rule below |
| C | Base calibration files immutable; validation writes to a separate correction layer applied at lookup | **Agreed** |
| D | Safer extrapolation | **Agreed, modified**: near-range extrapolation must keep working (e.g. 182 uL vs 180 uL top row). Range limit is a *preference*, not a cutoff |
| E | Stable tip handover with headroom | Mostly exists in `resolve_pipetting_setup`; add **capacity margin as a preference** |
| F | Freeze resolved params for a session | Keep only as a cheap guard (resolve all volumes up front); mostly unnecessary once A+D+G exist |
| G | Validate only at existing calibrated volumes, not arbitrary workflow volumes | **Strongly agreed** |

### D detail — extrapolation

- Today (`pipetting_wizard.py` `interpolate_parameters`): two-point linear for all
  params, allowed up to `MAX_EXTRAPOLATION_FRACTION = 0.25`, beyond that returns None
  (defaults used).
- Proposal: overaspirate **scaled with volume** from the edge row (ratio
  overaspirate / volume held fixed); other params (speeds, waits, air) **held at edge**.
- Near-range (e.g. within ~10%) = preferred; further = still usable with a warning.
- Water small tip example (edge 180 uL, +4.5 uL):

  | Target | 2-pt (before) | 2-pt (after today) | Hold | Scale |
  |---|---|---|---|---|
  | 182 | 4.67 | 5.13 | 4.50 | 4.55 |
  | 189 | 5.27 | 7.32 | 4.50 | 4.73 |
  | 193.6 | 5.66 | 8.76 | 4.50 | 4.84 |

- Side effect: the same limit applies below a range; large tip would no longer reach
  down to ~150 uL for water (small tip covers it).

### E detail — capacity margin (preference only)

`tip load = volume + overaspirate + post_asp_air`. Per tip `capacity_margin` in
`robot_state/pipet_tips.yaml` (proposed small 5 uL, large 10 uL; 0 = today's behaviour).

- load <= capacity - margin -> good fit
- capacity - margin < load < capacity -> tight fit (used, with warning, only if no tip is a good fit)
- load >= capacity -> doesn't fit -> other tip / chunking / defaults

Tip preference order in `resolve_pipetting_setup`:
1. calibrated tip, volume inside range, good fit
2. calibrated tip, volume near range (extrapolated, warning)
3. other calibrated tip
4. further extrapolation / defaults with warning

Also applies to chunk sizing in `plan_vial_transfer`.

Handover then emerges from existing logic (no manual setting). Optionally add a
small-tip ~190 uL water row to keep 180–190 uL on the small tip.

### D + margin — worked examples (no new ERRORs)

Prototype (not repo code) run against the real CSVs on 2026-10-30, with overvolume
compensation on as in workflows. Rules: inside range = calibrated; <=10% outside =
near (warning); 10–25% = far (warning); >25% = defaults (warning, as today).
Margin 5 uL small / 10 uL large. Preference: inside+good > near+good > far+good >
tight fit > defaults; smaller tip breaks ties.

| Liquid | uL | Today | Proposed |
|---|---|---|---|
| water | 2.0 | small, calibrated | small, extrapolated 20% from 2.5 (warning) |
| water | 182 | small, load 188.5 | small, extrapolated 1% from 180, load 187.9 — **works** |
| water | 190 | small, load 199.0 (1 uL headroom) | **large**, extrapolated 2% from 193.6 — only tip-choice change |
| water | 193.6 | large | large (unchanged) |
| water | 950 | ERROR | same; no tip physically fits (see 900 uL row bug) — chunked in vial transfers |
| DMSO | 55 | small, calibrated | small, extrapolated 10% from 50 (warning) |
| DMSO | 100 | small, DEFAULTS | same (gap: small 2–50, large 200–800) |
| DMSO | 175 | large, calibrated | large, extrapolated 12% from 200 (warning) |
| 6M_HCl | 60 | small, calibrated | small, extrapolated 20% from 50 (warning) |
| 6M_HCl | 300 | large, DEFAULTS | same (no large-tip calibration) |
| 2MeTHF | 200 | large, DEFAULTS | same (gap: small 80–120, large 400–600) |
| water, validation pinned to small | 193.6 | **ERROR (the incident)** | no crash: tight fit 199.9 -> should skip with warning (see below) |

Takeaways:
- No case goes from working to ERROR; the incident case stops crashing.
- Only tip-choice change: borderline loads (e.g. water 190 uL) move to the tip with room.
- The only remaining ERROR is a physically impossible single aspiration (load >= largest
  tip), same as today; vial transfers already chunk these.
- In pinned validation, a tight fit (199.9 uL here) is too close to test safely; skip it.
  Better: plan validation tips with the same resolver as the workflow, so 193.6 uL
  would be validated on the large tip, which is what the workflow actually uses.

### G removes the incident risk from validation

With validation only at existing calibrated rows (e.g. water small tip 20/100/180,
large tip 200 instead of 26.7/150.17/193.6):
- No extrapolation in validation: exact-row lookups (180 uL row -> load ~186 uL).
- No cross-talk between validation volumes: an exact-row lookup doesn't depend on
  neighbouring rows, so the 150 uL update can't change the 180 uL test. Makes F mostly moot.
- Remaining risks: a calibrated row itself near capacity (pick calibrated volumes with
  headroom; margin check skips with warning) and corrections inflating a row's
  overaspirate (B damping + cap at capacity - margin).
- G protects validation only; workflows still dispense e.g. 182 / 193.6 uL, which D +
  margin protect. G and D are complementary.
- Selection rule (agreed 2026-10-30): per tip actually used by the workflow, round each
  requested volume to its nearest calibrated row, then test the lowest, middle and
  highest of those rows (max 3 per tip). Examples (water): requests 24/55/97.6 -> test
  20/50/100; Ailsa 26.7–190 small -> 20/100/180; 193.6/198 large -> 200; 182 -> 180
  (extrapolation edge); 2.0 -> 2.5.
- Trade-off: tests stored rows, not the exact dispensed volumes. Optional later:
  measure-only checks at real volumes that report error but never write calibration.

### Validation must not crash

If the pinned tip no longer fits a volume, skip that volume with a warning and finish
the rest. Planning should use the same `compensate_overvolume` setting as execution
(planner uses True, validation uses False today).

## Q3 — correction data (from `pipetting_data/parameter_corrections.csv`, 333 rows)

- Mean correction ~0 at every volume; large spread (water std 1.4 uL @20, 3.6 @100,
  5.3 @150, 8.5 @500).
- **66–69% of consecutive corrections at the same volume flip sign** -> chasing noise.
- Weak environmental link: water r~0.0 temp / 0.07 RH; SDS r=0.15 temp / **0.28 RH**.
- There is also a second undamped correction at lookup:
  `apply_overvolume_compensation` applies 100% of (target - measured) on top of
  what validation wrote.

User intent: corrections exist to track real changes (temperature, humidity), but
should be more cautious and use the statistics available. Keep the 3 stages (more data).

### Proposed update rule (for C)

1. Pool all stage measurements: fit `measured = target + offset + overaspirate`
   robustly (median/Huber) over all ~9 replicates instead of picking the best stage.
2. Per (liquid, tip, calibrated volume) store offset + uncertainty; update with
   gain `P / (P + R)` (simple Kalman): R = session uncertainty from replicate spread,
   P = stored uncertainty, grows slowly with time (drift allowance).
3. Dead band: no update unless |offset| > ~2x its standard error.
4. Correct once: lookup = base + correction layer; no second compensation.
5. Later: humidity/temperature as an input once data supports it (SDS RH candidate).
6. Tune offline by replaying the historical corrections.

## Step 4 design (A + B + C together) — in discussion

### User decisions (2026-10-30)

1. **Base values**: replace with the statistically optimal overaspirate per row, estimated
   from history (overaspirate most likely to give zero error). Water and SDS only (other
   liquids don't appear to have been updating this way).
2. **Lookup compensation** (`apply_overvolume_compensation`): intended for "best value a
   little off"; much of that is uncertainty. Fold into the new approach (likely remove).
3. **Updates**: combine new session data with historical confidence/spread; aim to
   separate true drift from random noise.
4. **3 stages**: not strictly needed; what matters is data that distinguishes drift from
   noise. Replicates at the same overaspirate are useful.
5. **Decay**: depends on the math in 1–4.
6. **Robustness**: must not become disruptive or error-prone; slow processing is fine
   (runs about once per workflow).

### Findings from `master_pipetting_measurements.csv` (8088 real replicates, Mar–Oct 2026)

No tip column; tip inferred from volume (<200 uL small). Water 3940, SDS 2169 rows.

- **Within a session, delivered volume moves ~1:1 with overaspirate** (within-session
  slope 0.96–1.18 for most rows). So every measurement implies the overaspirate that
  would have hit target: `oa* = oa_used - (measured - target)`, whatever stage it came
  from. (Across sessions the slope looks like 0.3–0.7, but that's biased by the
  algorithm lowering overaspirate when a session runs high.)
- **Both noise and real drift exist.** Session-to-session spread is 2–5x what replicate
  noise alone explains, and consecutive sessions are correlated (lag-1 autocorrelation
  ~0.4–0.5 for water 20–200 uL, SDS 100/150). Some rows show no persistence (SDS 20,
  50, 200: ~0).
- **Backtest (predict next session's oa* from earlier sessions), RMSE uL:**

  | Row | Current (jump to last session) | Damped gain 0.3 | Running median |
  |---|---|---|---|
  | water 20 | 0.66 | 0.59 | 0.67 |
  | water 100 | 2.52 | 2.17 | 2.64 |
  | water 150 | 3.40 | 2.93 | 3.52 |
  | water 200 | 5.09 | 4.21 | 4.59 |
  | SDS 20 | 3.16 | 2.43 | 2.24 |
  | SDS 150 | 1.62 | 1.32 | 1.50 |
  | SDS 200 | 7.78 | 6.10 | 5.73 |

  Current behaviour is worst or near-worst almost everywhere. Damping (~0.3) wins on
  rows with persistent drift; on noisy rows with no persistence, a long-run median wins.
  A per-row model that estimates its own noise vs drift would pick the right gain for
  each row automatically.
- Very noisy rows (water 500/800/900, SDS 50/200/900) have outliers / bad data (e.g.
  the removed water 900 row): estimation must be robust (medians, outlier guard).

### Follow-up decisions (2026-10-30)

- **Tip column**: added. `tip_type` is now written for every validation measurement and
  session record (`embedded_calibration_validation.py`) and every routine dispense
  (`dispense_recording.py`, also still in `stability_info`). History backfilled
  (uncommitted): 8088 measurements / 1533 sessions; rule = <200 uL small, else large,
  except water/SDS 150–200 uL rows with pre_asp_air_vol 0 (large-tip settings) ->
  large (9 measurement rows: water 193.6 session 1b971ff3). Verified the files are
  otherwise byte-identical apart from the new column.
- **Only current parameter sets**: analysis and updates use only measurements whose
  non-overaspirate settings (speeds, waits, air, blowout) match the row's current
  values. Coverage with that filter: most water/SDS rows keep 28–120 sessions; water
  large 500/800 and small 70 keep none (settings changed) -> keep current base;
  water small 5/80/180, SDS small 10/25/75 have no history -> keep current base.
- **GUI**: whatever changes must keep the user's liquid GUI able to read and display the
  tracked data; review the GUI as the next step after this design.
- **Stages**: OK to remove entirely if something better is proposed (below).

### Proposed replacement for the 3 stages: check, then confirm only if needed

1. **Check**: N replicates (e.g. 4) at the row's current best overaspirate. Each
   replicate gives an oa*; session estimate = median, spread from the replicates.
2. **Compare** with what the row's model expects (prediction + its uncertainty).
   - Consistent (within ~2 combined SD): small damped update; done. Normal case:
     4 dispenses instead of 3–9 today.
   - Inconsistent: possible drift or a bad batch -> go to 3.
3. **Confirm**: N more replicates at the updated estimate. Agrees with the check ->
   real shift; update using both batches (more data = bigger justified step).
   Disagrees -> first batch treated as transient; only a small robust update, flagged.
4. Throughout: drop unstable-balance replicates (existing) and per-session outliers
   (MAD guard). No "best stage" selection; every accepted replicate counts.

The confirm batch also tests that the new value actually works (what stage 2 did).
"Agrees" means the confirm batch's implied value lies nearer the check than the old
estimate (changed 2026-10-31: comparing against replicate scatter called a real 9 uL
drift on SDS 800 uL "transient" because the two batches differed by 1 uL). History also
drops dispenses below 10% of target (no-dispense failures, not calibration data).
GUI compatibility: keep `optimization_stage` (1 = check, 2 = confirm), `accepted`,
`decision_reason` populated — verify against the GUI.

### Liquid GUI review (2026-10-30) — `scheduler/liquid_review.py` + `liquid_review_gui.py`

| GUI part | Depends on | Effect of new design | Change needed |
|---|---|---|---|
| Overvolume Changes tab (`session_overaspirate_estimates`) | >=2 distinct V_OA per session (stages) | **Breaks**: check-only sessions test one V_OA -> excluded as "fewer than three stages" | Fixed ~1 uL/uL response when only one V_OA tested (data supports it; the file already notes ~1 uL/uL). Share one estimator with the robot so the tab shows exactly what's used. |
| Past Validation Results (`summarize_past_validation`) | `accepted` true/false | Works if `accepted` stays populated | Keep `accepted` (used replicate = True; outlier/unstable = False); relabel "Stage not accepted" |
| Environmental Analysis | session/volume/V_OA groups | Works with any number of V_OA | None |
| Coverage tab (`coverage`, `regime`) | volume <200 = small; `np.interp` without extrapolation | Already mismatched: 190 uL now on large tip -> "Tip differs"; 182 uL -> "Outside coverage" though calibrated extrapolation is used | Use recorded `tip_type` and the wizard's own range/extrapolation logic |
| Current parameters (`current_parameters`) | base CSV only | Would show base, not base + correction, once C exists | Read effective values via the same lookup the robot uses |
| Re-simulation guard (`fingerprint`) | hashes `optimal_conditions_*.csv` only | A correction change would **not** invalidate an approved simulation | Include the corrections file |
| `environmental_history` | `parameter_corrections.csv` | Not used by the GUI tabs | Keep logging each update there anyway |

Complements: the Overvolume tab's estimate is exactly decision #1's base estimate; the
drift plot can show the tracked estimate and its uncertainty band. Tests to update:
`software_tests/test_liquid_review.py` (e.g. `test_sessions_that_saved_one_overaspirate...`).

### Slack (2026-10-30)

Current: one "3-STAGE PARAMETER OPTIMIZATION" message per volume (stage errors, best
stage, old -> new overaspirate, env). Proposed: one message per validation session
with a row per volume: tip, replicates used, mean error, verdict (consistent /
drift confirmed / transient), old -> new overaspirate with uncertainty, environment.

Also noted (unchanged): `validate_pipetting_accuracy` pauses with Slack and then
**raises** if stage 1 misses by more than 5% + 5 uL (tip/vial problem check,
~line 878) — a remaining way validation can end a workflow after the pause.

### Proposed design (draft)

- **State per (liquid, tip, calibrated row)**: best overaspirate estimate + uncertainty,
  plus that row's noise and drift variances estimated from history.
- **Initial base (#1)**: robust estimate of oa* from recent history for water and SDS,
  written once as a deliberate recalibration of the base rows.
- **Update (#3)**: session estimate = median oa* across all its replicates (all stages
  count); move the row estimate by a Kalman gain from that row's drift vs noise
  (typically ~0.2–0.3; ~0.1 or less on noise-only rows); cap step size; outlier guard.
- **Lookup (#2)**: use the row estimate directly; remove `apply_overvolume_compensation`
  (oa* already absorbs measured error), so only one correction exists.
- **Measurement (#4)**: stages optional; a few replicates at the current best value are
  enough, since every replicate yields an oa*.
- **Decay (#5)**: falls out of the model — uncertainty grows with time between
  sessions, so after a long gap new data gets more weight. No separate decay rule.
- **Robustness (#6)**: runtime update is a few arithmetic lines; any problem (missing
  data, NaN, file error) -> skip the update with a warning, lookup keeps the last good
  value. Variance estimation runs offline/occasionally and stores per-row settings.
- **Storage (C)**: base CSV holds the recalibrated base; corrections file holds the
  tracked estimate per row (fixed size, resettable).

## Q1 — rows to clean up (added by validation, vs 10 commits back)

| File | Rows | Note |
|---|---|---|
| water | 193.6 (large), 260.87 (large), 400 (large) | 400 is round but validation-added |
| water | 150 (small) overwritten 2026-10-30 | +1.9 -> -4.9 uL |
| SDS | 97.97 (small), 201.6 (large), 329.21 (large) | 97.97 is 2 uL from 100 row -> steep local slope |
| water | **900 (large): volume_measured = 799.98** | Looks like a copy of the 800 row (commit f72a033e). Overvolume compensation adds ~+100 uL overaspirate at 900, and inflates everything interpolated between 800 and 900 uL. **Affects accuracy today; fix first.** |

Water 2.5 uL looks deliberate. Other liquids untouched. Under A, fold these into
neighbouring calibrated rows (their data is already in `parameter_corrections.csv`).

## Where changes live

| Change | File |
|---|---|
| Skip volume instead of crash when pinned tip doesn't fit | `pipetting_data/embedded_calibration_validation.py` (`validate_pipetting_accuracy`, ~line 775) |
| Validate at calibrated volumes (G); planning uses same compensation flag | `pipetting_data/embedded_liquid_validations.py` |
| Extrapolation method; report extrapolation distance (e.g. `_extrapolation_fraction`) | `pipetting_data/pipetting_wizard.py` (`interpolate_parameters`, `get_pipetting_parameters`) |
| Tip preference order + margin | `North_Safe.py` (`resolve_pipetting_setup`, `plan_vial_transfer`) |
| Margin values | `robot_state/pipet_tips.yaml` |
| Correction layer + update rule | `embedded_calibration_validation.py` (`_update_calibration_csv`) + wizard lookup |

## Suggested order

Revised 2026-10-30 (G moved first — it removes the incident at the source):

| Step | What | Fixes | Files |
|---|---|---|---|
| 0 | Fix water 900 uL row (`volume_measured` 799.98) | ~+100 uL overaspirate on water large tip 800–900 uL, today | `optimal_conditions_water.csv` (data; user to confirm value) |
| 1 | G: validate at calibrated volumes; plan validation tips with the workflow's resolver/compensation; skip-with-warning instead of crash; up-front resolution guard (F) | The incident; no new odd rows; no cross-talk | `embedded_liquid_validations.py`, `embedded_calibration_validation.py` |
| 2 | D + capacity margin | Workflow safety (182 uL still calibrated, borderline loads to roomier tip, no new errors) | `pipetting_wizard.py`, `North_Safe.py`, `pipet_tips.yaml` |
| 3 | Clean up validation-added rows (Q1 table) | Files hold only deliberate calibrated volumes | water / SDS CSVs (data) |
| 4 | A + B + C: fixed calibrated volumes, damped Kalman-style update, separate correction layer, single correction at lookup | Noise-chasing; clean, reversible base files | `embedded_calibration_validation.py`, `pipetting_wizard.py`, new corrections file |

**Steps 0 and 3 DONE 2026-10-30** (uncommitted): removed water large 193.6, 260.87,
400, 900 and SDS small 97.97, large 201.6, 329.21 (originals in git HEAD). Remaining:
water small 2.5–180 (10 rows), large 200/500/800; SDS small 10–180 (8), large
200/500/800/900. Effects: water 193.6–198 uL now extrapolates 1–3% from the large
200 row; water 850–900 uL overaspirate now ~-15 uL instead of being inflated by the bad
900 row; water large tip is sparse (3 rows) — add calibrated volumes (e.g. 300) via a
deliberate calibration run if needed. Water small 150 row is still -4.9 uL from the
2026-10-30 validation (previous value +1.94 uL, measured 149.6) — user to decide.

**Step 1 DONE 2026-10-30** (uncommitted):
- `embedded_liquid_validations.py`: new `calibrated_volumes_ul(liquid, tip)`; tips
  planned with the workflow's resolver (default compensation); each requested volume
  rounded to its nearest calibrated row for that tip, then lowest/middle/highest
  tested; unplannable volumes and tips with no calibrated rows skipped with a warning.
  Unknown liquid names / missing calibration files still fail before hardware (as before).
- `embedded_calibration_validation.py`: `validate_pipetting_accuracy` skips (warning)
  any volume whose setup fails up front or mid-session; all-skipped sessions return
  `_skipped_validation_result` (`success=False, skipped=True`) after returning the vial home.
- Tests: `software_tests/test_embedded_liquid_validations.py` updated + 4 new;
  `software_tests/test_validation_decisions.py` +3 skip tests. Run with the `north_env` conda env.
  17 failures in `test_fluorescence_calibration.py` / `test_workflow_template.py` are
  pre-existing (identical with step-1 files reverted).

**Step 2 DONE 2026-10-30** (uncommitted):
- `pipetting_wizard.py`: outside the range (tip-specific lookups), only the edge row is
  used — overaspirate scales with volume (`edge_oa * v / edge_v`), all other params keep
  the edge value. 25% outer limit unchanged. New `NEAR_EXTRAPOLATION_FRACTION = 0.10`
  and `extrapolation_fraction(liquid, volume_ml, tip_type)`.
- `North_Safe.py` `resolve_pipetting_setup`: tips tried by coverage (inside, near <=10%,
  further, none), then size; a load within `capacity_margin` is a tight fit, used (with
  warning) only if no calibrated tip fits with room; defaults fallback unchanged.
  `plan_vial_transfer` unchanged (already capped by calibrated max volume).
- `robot_state/pipet_tips.yaml`: `capacity_margin` 0.005 (small), 0.010 (large); 0 or
  missing = previous behaviour.
- Tests: new `software_tests/test_extrapolation_and_margin.py` (6); one expectation in
  `test_tip_calibration_identity.py` updated to edge-row scaling. Related suites
  unchanged vs baseline; 5 pre-existing failures (sq_peptoid x3, two tests expecting
  SDS ~195 uL on the small tip, which doesn't fit with current SDS data: 202.4 uL load).
- Real-code check: water 182 -> small; 190/193.6/198 -> large; pinned small 193.6 ->
  tight fit 198.4 uL, no crash; no example raises.

Steps 0 and 3 are data-only. Step 4 needs the most design (update rule, storage
format, offline replay of historical corrections).

## Open questions

- Near-range preference limit: 5% or 10%?
- Extrapolation method for overaspirate: scale-with-volume (recommended) vs hold vs robust fit?
- Margin values (5 uL small / 10 uL large?).
- Add a ~190 uL small-tip row for water?
- Should corrections apply within the same run, or from the next run?
- Correction layer storage format.

## Related changes already made (2026-10-30)

- `workflows/surfactant_grid_ailsa.py`: water vials filled before liquid validation
  (moved out of replicate setup).
- `North_Safe.py`: "Cannot aspirate more volume than in vial" now names vial and volumes.

## Out of scope / parked

- `scheduler_gui.py` `start_simulation` sets `execution_mode = "simulation"` before
  setup succeeds; after a live run a failed simulate start leaves it pointing at the
  `live_*` folder -> `session.json` not found on next simulate. User said ignore for now.

## Working setup while the robot runs (2026-10-30)

A live experiment started 17:51 in the main folder (old code in memory; it re-reads
calibration CSVs per dispense, so it uses the cleaned rows — small accuracy effect on
water 200–330 uL large tip, accepted by user; fix comes from chunk 2).

- **All new work happens in a separate copy**: `C:\Users\Imaging Controller\Desktop\utoronto_demo_calibration`
  (git worktree, detached at snapshot commit c1c53fdf = main folder's uncommitted state
  at ~20:30, plus the untracked plan doc and `software_tests/test_extrapolation_and_margin.py`).
  Nothing in the main folder is modified or committed while the run is going.
- After the run: bring the copy's changes into the main folder in one step, commit,
  simulate, then chunk 2 (new base values) and switch the new validation on.

### Chunks

| Chunk | Status | Where |
|---|---|---|
| 0. Save what's done | Snapshot taken; commit at merge | main folder |
| 1. Shared estimator + review table (current parameter sets only) | DONE | copy |
| 2. Apply approved base values (water, SDS) | DONE (in copy) | main folder |
| 3. GUI prep (single-V_OA sessions, recorded tip, honest accuracy, effective values) | DONE | copy |
| 4. New validation behind on/off switch (check -> confirm, damped update, correction file, Slack, re-simulation guard) | DONE, switched on | copy |
| 5. Clean-up (remove stage code, tests, pause-then-raise decision) | partly: tests updated; 2 decisions open | copy |

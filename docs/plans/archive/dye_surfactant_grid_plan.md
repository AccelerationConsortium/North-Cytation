# CMC / EE / K / solubilization workflow — plan

Updated 2026-11-21 · Status: first live run done 2026-11-20 (SDS + coumarin-6, reads to
480 min); four issues open before the next run (see **First live run**). Workflow
`surfactant_dye_cmc` built, simulated and queued
(`scheduler/config_lists/surfactant_dye_cmc_queue.yaml`).

## Goal

One 96-well plate, one surfactant and one dye per run. Measure, as accurately as
possible:

1. **CMC**
2. **K** (how strongly the dye binds micelles)
3. **Fraction bound** (share of a small dye dose inside micelles; called "EE%" in parts
   of this plan, which conflicts with Chanelle's usage, see First run)
4. **Capacity** (the most dye the micelles can hold; this is Chanelle's "EE"; called
   "solubilization" in parts of this plan)

and confirm the dye stock concentration. Must work for small surfactants and for
polymers (the project's main focus). Source request: Chanelle's "CMC and EE Studies"
draft (her factors become config keys; each of her studies is a preset changing 1-2 keys;
validation studies are repeats of a preset).

Three parts: **1. Analysis**, **2. Plate layout**, **3. Workflow**. Layout is driven by
what the analysis needs; the workflow barely depends on layout.

## 1. Analysis (decided)

**Model** (Piñeiro 2013, Soft Matter 9:10779; concentration model Al-Soufi 2012, JCIS
371:102): each well's fluorescence = F_free x (fraction free) + F_bound x (fraction
bound), with fraction bound = K[M] / (1 + K[M]) and micelle concentration from total
surfactant minus monomer. Replace the current Boltzmann sigmoid
(`analysis/control_cmc_analysis.py`): it gives no K, and its "CMC" (the midpoint) shifts
with the dye. Piñeiro: line-intersection CMCs for Triton X-100 ranged 0.2-0.4 mM across
dyes; the model fit gave 0.276 +/- 0.008 mM for all of them.

| Metric | How it is obtained |
|---|---|
| CMC, K, transition width | One model fit of the surfactant series (fixed low dye) |
| EE% per well | Where the well's signal sits between F_free (measured, 0-surfactant well) and F_bound (fitted ceiling, inferred from how the curve bends; no well is ever 100% bound) |
| Solubilization | Dye series at a few fixed surfactant levels above the CMC. Below the limit, more dye = brighter; above it, brightness levels off and 750 nm scatter rises. Kink = capacity at that level. Capacity vs micellised surfactant -> slope = moles dye per mole micellised surfactant |
| Error bars | Replicates + fit uncertainty |
| Change over time | Timepoint reads |

Rules:
- **Keep dye low for CMC/K/EE wells:** dye x K < 1 (under ~1 dye per micelle near the
  CMC), else dye-dye interactions distort the curve. Safe dye depends on the surfactant
  (pyrene: Triton X-100 < ~0.3 uM, SDS < ~5 uM).
- **Report K; derive EE% at any surfactant level from it.** For strong binders (most small
  surfactants) K is inherently imprecise; report "K >= X" there. EE% stays precise.
- **Solubilization deliberately exceeds the limit,** so it needs its own kink fit (the
  Piñeiro model assumes dye well below capacity). Risk: false levelling-off at high dye
  (inner filter, self-quenching); the absorbance read checks for it.
- Pyrene uses the I1/I3 ratio and needs the Piñeiro 2015 variant of the model.

## 2. Measurement (decided)

**Current run (coumarin-6), at every read time, one plate trip, four protocols:**

| Protocol | Reads | Setting | Saved in |
|---|---|---|---|
| `Coumarin_96.prt` | fluorescence 470 / 510 (micelle peak; 485 / 530 before 2026-11-23) | `FLUORESCENCE_PROTOCOL_FILE` (blank = this) | `results.csv` |
| `Coumarin_Abs_96.prt` | absorbance 460 (peak), 750 (scatter) | `ABSORBANCE_PROTOCOL_FILE` | `results.csv` |
| `Coumarin_FluorSpectrum_96.prt` | ex 450, em 490-700, 10 nm (20 nm band), manual gain 75 | `SPECTRUM_PROTOCOL_FILES` | `spectra.csv` |
| `Coumarin_AbsSpectrum_96.prt` | absorbance 400-750, 5 nm | `SPECTRUM_PROTOCOL_FILES` | `spectra.csv` |

- One read type per protocol: the reader wrapper classifies a protocol by its first read
  step; a spectrum comes back as wavelengths x wells. `spectra.csv` has one row per well,
  read time, protocol and wavelength, with the well's contents.
- `hardware_tests/cytation_protocol_preview.py` runs protocols without robot moves and
  saves the raw output and its shape: run it on new protocols first.
- Every column the reader returns is saved: raw file per read, and `results.csv` (one row
  per well per read, rewritten after every read, with `read_at` clock time).

Earlier general notes:

- **Every well, every timepoint:** one fluorescence pair + absorbance at the dye peak and
  at 750 nm (scatter / undissolved dye). Reads cost time, not wells.
- **Once per new dye-surfactant pair:** fluorescence and absorbance spectra on a few wells
  (low and high surfactant) to choose wavelengths. Not routine.
- **Wavelengths:**

| Dye | Fluorescence (current Gen5 protocol) | Absorbance |
|---|---|---|
| Nile Red | 550 / 648 (`NileRed_{plate}.prt`) | 579 nm in SDS micelles, 593 nm in water (Rus22); ethanol peak to confirm |
| Coumarin-6 | 485 / 530 (`Coumarin_{plate}.prt`); peaks are 465 / 505-508 in micelles (Rui15), so off-peak | ~465 nm |
| Pyrene | 334 / 373 + 384 (ratio) (`CMC_Fluorescence_{plate}.prt`) | skip (UV; PS plates absorb) |

  750 nm reference for all. Nile Red usually blue-shifts and brightens in micelles; pick
  the emission where bound is bright and free is dim.
- **Stock check:** make a concentrated dye stock, let it settle or filter it, then a robot
  dilution series in pure solvent read by absorbance: straight line = dissolved, no 750 nm
  signal = no particles, slope = true concentration (needs the molar absorptivity). Use the
  measured concentration in the analysis. Read immediately (ethanol evaporates).
- Watch: acetone attacks PS plates.

## 3. Plate layout

### Fixed vs adaptive: when each works

Simulation (Bayesian design study, 120 random samples per class drawn from the ranges
below; wells chosen by optimal design; 96 wells all for CMC/K/EE):

| Samples | Design | CMC error (median / 1-in-10 worst) | K error | EE% error | Success* |
|---|---|---|---|---|---|
| Small surfactants | fixed | 3% / 8% | 18% | 0.3 pts | 72% |
| Small surfactants | adaptive | 3% / 7-9% | 14-18% | 0.2-0.3 pts | 72% |
| Polymers | fixed | 20% / 52% | 12% | 2.7 pts | 18% |
| Polymers | 3 batches (32 x 3) | 5% / 30% | 7% | 1.4 pts | 61% |
| Class unknown | fixed | 8% / 33% | 13% | 0.9 pts | 53% |
| Class unknown | 2 batches (32 + 64) | 5% / 31% | 8% | 0.5 pts | 73% |

*CMC within 10%, EE% within 5 points, dye not too high.

- **Small surfactants, CMC roughly known: fixed is enough.** Their only failures were dye
  too high, so use a lower default dye.
- **Polymers: adaptive.** Batch 1 finds the rough CMC and a safe dye; batch 2 places the
  fine CMC/K/EE wells and the solubilization wells (both only need batch 1).
- **Both fail: polymers with CMC below ~1 uM.** The dye has to be so low the signal is
  lost. A limit of this dye method.
- Also learned: a scout needs ~24 wells (12 is too small); if the scout cannot pin K, keep
  a dye floor (an earlier version picked ~160x too little dye and lost the signal); per-
  well noise matters more than how wells are split.
- Simulation gaps: its CMC error bars are too narrow; no equilibration time between
  batches, no polymer concentration limit by mass, no solubilization wells.

### Current run: SDS + coumarin-6 (decided 2026-11-20)

Spec: three blocks on one 96-well black plate with clear bottom (`PLATE_KIND: fluorescence`),
96 wells, final well concentrations, random placement. Layout revised 2026-11-23 from the
weekend runs: 2D SDS levels near the CMC (saturation is only visible there), all taken from
the CMC series; one water blank (SDS does not change the blank).

| Block | Batch | Contents | Wells |
|---|---|---|---|
| Stock check | 1 | dye in ethanol 0, 1, 4, 12, 32 uM x 2; read once (ethanol evaporates) | 10 |
| CMC | 2 | SDS 0, 5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 11, 12, 14, 16, 32 mM x 3; dye 1 uM | 42 |
| Dye x SDS | 2 | SDS 0, 7.5, 8, 8.5, 9, 10, 12 x dye 4, 12, 32 uM x 2; blank (0 SDS, 0 dye) x 2; 1 uM from the CMC block | 44 |

- Reads: batch 2 at 0, 1, 2, 4, 8, 12, 16 h (Chanelle's Study 3 times, plus 12 h).
- Dye: coumarin-6 stock in ethanol, 918 uM (COUMARIN6-004) in the queue; 25 uL per well
  (dye substocks at 8x final). `DYE_SOLVENT_EVAPORATES: true` (ethanol is volatile over the
  18 h): water + SDS fill the full 200 uL, the dye solution goes on top. For a non-volatile
  solvent (DMSO) set it false: the dye solution then counts within the 200 uL.
- Queue (three plates): dye last; dye first with a 60 min pause; dye first, no pause.
  Order and pauses only change pipetting; volumes are the same.
- Substocks (7): dye 256, 96, 32, 8 uM; SDS 16.4, 6.2, 2.1 mM. Made in run 1, reused by
  later runs (record next to the vial file; reused up to 72 h).
- Vials: SDS refill in two closed 8 mL vials (an open vial loses water, SDS strengthens
  ~2.5-5% per run); ethanol refill open 20 mL, put in 20, recorded 17 (~2-3 mL evaporates
  over the queue). 1 mL left in stock and refill vials.
- Whole queue uses ~1.5 mL coumarin-6 stock, ~14.5 mL 50 mM SDS, ~19 mL ethanol.
- Analysis: runs automatically at the end of the workflow (output folder `analysis/`); a
  failure is a warning. By hand: `analysis/surfactant_dye_cmc_analysis.py <run folder>
  --fluorescence fluorescence_470_510 --absorbance absorbance_460 --epsilon 54000`: CMC and K
  per read, dye x SDS slopes, stock check, time course. Descriptive for loading: not a
  capacity or encapsulation measure.

### First live run (2026-11-20, reads to 480 min)

Run: `output/surfactant_dye_cmc/20261120_201530_235719_LIVE_wells-94`; preliminary plots
and scripts in its `prelim_analysis_t480/`.

Open issues (to check on the robot):

1. **Absorbance all invalid:** every well, both protocols (fixed 460/750 = -99999,
   spectrum = -636), blanks included. Ailsa's 600 nm reads on these plates were valid
   (~0.038 OD water blank). The preview only ran on an empty plate (0.0000), so it proved
   nothing. Next: run `CMC_Absorbance_96` and `Coumarin_Abs_96` on this plate; only
   Coumarin fails -> new protocol files; both fail -> reader or plate.
2. **Fluorescence spectrum saturates:** emission peak 490-590 nm overflows (~100k cap) in
   every well with dye >= 4 uM and SDS >= 10 mM, and the 12-32 uM ethanol wells. Next:
   check whether Gen5 allows Extended range for spectra (the fixed read uses it, which is
   why it reaches 8M without saturating); otherwise lower gain from 75 (start ~55, test on
   a 25 mM SDS / 32 uM well, aim for a 50-70k peak). Keep 20 nm bandwidth; check the
   excitation bandwidth ("450/0.0").
3. **`dye_sub_4` (8 uM) had no dye:** every well from it reads at blank (all 1 uM wells,
   so the whole CMC block, and the 4 uM ethanol well). Code, recipe math and log all
   correct (0.333 mL `dye_sub_2` -> `dye_sub_4`, vortexed; every dispense matches
   `planned_wells.csv`). Physical cause: check the vial (clear = transfer failed;
   yellow-green = wrong vial in rack 15).
4. **Two collapsed wells, B2 and B9 (16 mM SDS, 32 and 12 uM dye):** dosed (bright at
   0-60 min), then ~1000x drop by 120 min. Look for dye crashed out, film or bubble.

Learned:

- The fixed 485/530 read is fine: Extended range + auto gain, no saturation, stable over
  time (4 uM wells 2.1-2.5M at every read).
- Below the CMC the emission peaks near 620 nm; in micelles near 505 nm. With unsaturated
  spectra the ratio may be a dose-independent CMC readout (to test).
- 4 uM appears fully taken up from 10 mM SDS up (same brightness at 10, 16, 25 mM) and is
  stable; 12-32 uM lose signal over time, so excess dye seems to leave solution slowly
  (coumarin-6 forms microcrystals in water). Loading (dye per micelle) alone does not
  explain brightness.
- The ethanol series is sublinear from 12-32 uM; consistent with inner filter (estimated
  A ~0.7 at 32 uM), which absorbance should confirm. It is the optical control for the
  micelle wells.
- Replicates otherwise agree within ~1.3x at t0 and ~1.1x by 480 min.

**Suggestion (not decided):** run the CMC block at a second dye level (4 uM) beside 1 uM,
2 replicates each (56 wells instead of 42). The apparent CMC depends on dye binding
strength (Al-Soufi & Novo 2021, Molecules 26:5339); two dye levels through the same SDS
series would show whether the CMC shifts with dye and constrain K directly. Keep 12 and
32 uM in the dye matrix: that is where the time-dependent loss shows.

### Earlier proposal: SDS + Nile Red (superseded by the current run)

**Terminology problem (raised by Chanelle, 2026-11-19):** her doc defines EE as "the
amount of dye that can be encapsulated at and above the CMC", i.e. what this plan calls
solubilization / capacity. This plan also used "EE%" for the fraction of a small dye dose
inside micelles, which is not her meaning. Agree terms with Chanelle before going further.
Until then: "capacity" = most dye micelles hold (her EE); "fraction bound" = share of a
small dose in micelles.

- Why Nile Red: published values in SDS to compare with (Rus22: CMC 7.4-7.8 mM; capacity
  0.0044 mol per mol micellised SDS; absorptivity 3.4e4 M^-1 cm^-1 at 579 nm in SDS
  micelles, 1.25e4 in premicellar SDS; ~3 uM Nile Red clumped below the CMC over hours to
  days). Fluorescence protocol exists.
- Dye for the CMC block: one level, 1 uM (below the ~2 uM water solubility; Chanelle's doc
  also uses 1 uM). Two dye levels are only needed when K is unknown.
- CMC block SDS levels (current proposal, 60 wells): SDS 0 x3; 0.3x and 0.6x CMC x3 each;
  12 levels evenly 0.75-1.35x CMC x3; 1.5, 2, 2.5, 3.5, 5x CMC x3 each. From an optimal-
  design calculation (Fisher information, literature prior: CMC 8.6 mM +/-15%, K within
  x3, 3% noise): predicted CMC +/-2.6%, K +/-21%, fraction bound +/-1.1 pts. An
  optimizer-only design stacks wells at 5x and gains little (CMC 2.3-2.4%); spreading the
  high end checks the model across the curve.
- Stock check: dilution series of the dye stock in ethanol, absorbance (linearity, 750 nm,
  concentration). Nile Red absorptivity in ethanol not yet found.
- **Capacity (her EE) method: undecided.** Two designs were proposed here and both have
  problems: (1) dye series at fixed SDS, find the kink: settled / clumped dye stays in the
  light beam, clumps break Beer-Lambert (Rus22), and the kink is poorly sampled; (2)
  excess dye then transfer supernatant to read wells: impractical. Rus22 used excess solid
  (440 uM gross), ~2 days to equilibrium, kept 5 days; it does not say how solid was
  excluded from the optical path. Decide the method with Chanelle from her knowledge and
  the literature.
- Proposed weekend scope if the capacity method is not ready: CMC block + stock check
  only.

## 4. Workflow

Built from `workflows/workflow_template.py` on the North_Safe flows
(`docs/platform/automation_catalog.md`), reusing parts of `surfactant_grid_ailsa.py` and
`fluorescence_calibration_workflow.py`.

### Vials

- 1 dye stock, 1 surfactant stock, 1 dye-solvent vial, 1 water vial; `surfactant_refill`
  and bulk dye solvent; generated surfactant and dye substocks (`_PRODUCT_VIALS`).
- Stocks carry a `stock_id`; concentrations come from `status/stocks.csv`, not code tables
  (file and vial-file column not set up yet).

### Steps

1. **Plan** — pure function `build_plan(config, reused)`, testable without hardware. Plate
   contents come from `BLOCKS` (each a surfactant x dye grid). Concentrations are over
   `TOTAL_VOLUME_UL`; with `DYE_SOLVENT_EVAPORATES` the dye solution goes on top of it,
   otherwise it counts within it.
2. **Substocks** with `create_substock` (reservoir water first, then stock, vortex).
   - Dye: fixed dye volume per well, so the solvent fraction is the same in every well; one
     substock per level, sized to demand + reserve.
   - Surfactant: variable volume per well; fewest substocks so every volume fits (typically
     1-2).
   - `refill_vial` for water, surfactant (one or several refill vials, in order), solvent.
   - Substocks left from earlier runs are reused (`<vial file>_substocks.yaml`).
3. **Dispense** with `dispense_from_vials_into_wellplate`, one call per component in
   `DISPENSE_ORDER`; optional `DISPENSE_PAUSES` between components; seeded
   `RANDOMIZED_ORDER`; whole-run supply check before the first move.
4. **Measure** with `lash_e.measure_wellplate([fluorescence, absorbance])` (one trip).
5. **Timepoints:** repeat step 4 over `BATCH_SCHEDULES_MIN` (one list per batch)
   (`mark_clock_start` / `mark_wait_until`). No lid.

### Keep / drop from existing workflows

| From | Keep | Drop |
|---|---|---|
| Fluorescence calibration | Fixed dye volume + per-level substocks; `build_plan` without hardware; pre-move supply check; dye-in-solvent reference; timepoint schedule | Own staging workaround, multi-source solvent pooling, `TWO_RUNS_PER_PLATE` |
| Ailsa | `_DYE_PROTOCOLS`; seeded randomization; `DISPENSE_ORDER`; replicates | Turbidity, two-surfactant replay CSVs, per-well Cytation runs, chunked refill engine |
| Adaptive | Variable-volume surfactant source selection; cascade substock recipes; CMC-centred spacing | 90 uL surfactant cap; `SURFACTANT_LIBRARY` as stock source |

### Staging rules (implemented 2026-11-18)

- Open small-tip vials, safest first: rack 47, 46, 45, heater 2, 5, 8, rack 44, 43
  (`small_tip_staging_rank` in `vial_positions.yaml`). Closed vials: rack 0-5.
- One rule in `stage_vials` for both flows; the clamp is never a staging spot.
- Vial to wellplate: fresh tip per vial; each vial home right after its dispense.

## Before the first run

- Make the three new Gen5 protocols, run `hardware_tests/cytation_protocol_preview.py` live,
  and check its summary (data shape, processing OK, read times).
- Load: coumarin-6 stock (918 uM, 8 mL), SDS_stock + SDS_refill_1/2 (8 mL each, closed, rack
  6, 10, 11), ethanol (8 mL), ethanol_refill (20 mL), empty substock vials, 3 black plates.
- Confirm the North_Safe closed-cap vial-to-vial fix (stock -> substock moves vials while
  holding a tip; only rack 0-5 and the clamp are safe).
- Run `VALIDATE_LIQUIDS` once for small dye volumes. Dye substocks ~3 mL (small-tip vials
  keep a ~2 mL reserve).
- Simulation mode for Nile Red (Ailsa's simulator covers pyrene only).

## Implementation notes (current syntax, checked 2026-11-18)

`lash_e` = `Lash_E` (`master_usdl_coordinator.py`); `robot` = `lash_e.nr_robot` (`North_Safe.py`).
Volumes are mL.

| Step | Call |
|---|---|
| Output folder | `lash_e.save_run_info(output_dir)` |
| New plate | `lash_e.grab_new_wellplate()` |
| Top up water / surfactant | `robot.refill_vial(vial, target_volume_mL, reservoir=1)` or `source_vial="surfactant_refill", liquid="SDS"`; returns mL added |
| Substock | `robot.create_substock(dest, [{"volume_mL": v, "reservoir": 1}, {"volume_mL": v, "source_vial": "SDS_stock", "liquid": "SDS"}], vortex_time=10)` |
| Dispense (per component in `DISPENSE_ORDER`) | `robot.dispense_from_vials_into_wellplate(df, liquid={vial: liquid})`; `df`: columns = vial names, index = well index, values = mL. It checks the plan before moving and does staging, conditioning, tip grouping and returning vials home |
| Read | `lash_e.measure_wellplate([fluor_protocol, abs_protocol], wells, plate_type=..., use_lid=...)` |
| Timepoints | `mark_clock_start(lash_e.logger, clock_id)` / `mark_wait_until(lash_e.logger, clock_id, seconds)` from `workflows/helpers/_planning_markers.py`. These only log for planning; the workflow still sleeps (`fluorescence_calibration_workflow.py:826-832` pattern) |
| End | `lash_e.discard_used_wellplate()` |
| Stop / crash | `cleanup_after_keyboard_interrupt(lash_e)` (`workflows/helpers/interrupt_cleanup.py`) |

- Stock concentrations: `sample_tracking.load_stocks()` reads `status/stocks.csv`. It's looked up from the vial's `stock_id`.
- No direct calls to `stage_vials`, `condition_tip` or `dispense_into_wellplate`: the flows cover them.

## Not verified yet

Treat these as estimates, not facts: Nile Red K in SDS (~1.5e5, estimated from Rus22
capacity and water solubility); every "expected" number in `sds_nile_red_metrics.pdf`
(diagrams use simulated data); the polymer half-bound ranges (computed from paper values
with assumptions). Also unverified: Nile Red molar absorptivity and peak in ethanol; Nile Red shift size in our surfactants;
K for Coumarin-6 / Nile Red; bright/dim contrast for our dyes; Cytation read times; real
well-to-well noise for small dye volumes; equilibration time for polymers.

## Reference: realistic sample ranges

Chanelle's doc names no polymers (four empty rows), so these are typical stand-ins.

| Sample type | CMC | Dye half-bound at | Source |
|---|---|---|---|
| Small surfactants | ~0.17 mM (DDM) to ~25 mM (OG); lab data 0.9-15 mM | ~1.04-1.15 x CMC | Supplier tables; Piñeiro 2015 |
| Pluronics (37 C, PBS) | 1 uM (L121) to 5.3 mM (L35); F127 2.8 uM | ~3-8 x CMC | Kozlov 2000 |
| PS-PEO | 1-5 mg/L (upper bound) | ~5-13 x CMC (rough) | Wilhelm 1991 |
| PS-b-polyacrylate | 16 uM to 50 nM | n/a | Astafieva 1995 |
| Polyelectrolyte (frozen) micelles | effectively none | K ~1e5 | Štěpánek 1998 |

- Small surfactants bind dye just above the CMC; polymers rise slowly over a decade or
  more above it.
- Pluronic CMC is extremely temperature-sensitive (F127: 2.8 uM at 37 C vs 0.56 mM at
  25 C). Fix and record temperature.
- At nM-uM CMCs, dye can outnumber polymer chains (Wilhelm), so dye must go lower.
- Lab data: pyrene CMCs for 8 small surfactants moved 0.5-7% between 0 and 30 min reads;
  replicate CV ~2% (`Master Surfactants/CMC/all_CMC_data.csv`). The June 2025 P407 run
  (0.04-0.25 mM) showed no transition.

## Sources and scripts

- Papers in `01_Projects/Master Surfactants/`: Piñeiro 2013, Piñeiro 2015, Al-Soufi 2012,
  Tehrani-Bagha 2013, Kozlov 2000, Wilhelm 1991, Štěpánek 1998, Astafieva 1995. Also
  Rus22 (Nile Red in SDS, Molecules 2022) and Rui15 (Coumarin-6 in nonionic micelles).
- Simulation scripts are in the session scratchpad (`sim2/bd.py`, `sim2/run_study.py`);
  move into the repo if we keep them.

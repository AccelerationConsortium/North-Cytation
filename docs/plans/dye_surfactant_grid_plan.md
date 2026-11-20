# CMC / EE / K / solubilization workflow — plan

Updated 2026-11-19 · Status: analysis and measurement decided; plate layout for the first
(SDS) run being detailed; workflow not built yet. Target: SDS run over the weekend of
2026-11-21.

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

### First run: SDS + Nile Red, fixed plate (not final)

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

1. **Plan** — pure function `build_plan(config)`, testable without hardware. Outputs the
   well table and substock recipes. `TOTAL_VOLUME_UL` includes the dye; water = total -
   surfactant - dye; every well checked against `max_volume_per_well`. Surfactant
   concentration computed over the total volume.
2. **Substocks** with `create_substock` (reservoir water first, then stock, vortex).
   - Dye: fixed dye volume per well, so the solvent fraction is the same in every well; one
     substock per level, sized to demand + reserve.
   - Surfactant: variable volume per well; fewest substocks so every volume fits (typically
     1-2).
   - `refill_vial` for water, surfactant, solvent.
3. **Dispense** with `dispense_from_vials_into_wellplate`, one call per component in
   `DISPENSE_ORDER` (default water -> surfactant -> dye); seeded `RANDOMIZED_ORDER`;
   whole-run supply check before the first move.
4. **Measure** with `lash_e.measure_wellplate([fluorescence, absorbance])` (one trip).
5. **Timepoints:** repeat step 4 over `MEASUREMENT_SCHEDULE_MIN`
   (`mark_clock_start` / `mark_wait_until`). Consider `use_lid` for long schedules.

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

- Gen5 absorbance protocols for Nile Red (none exist yet); plates clear or clear-bottomed.
- Confirm the North_Safe closed-cap vial-to-vial fix (stock -> substock moves vials while
  holding a tip; only rack 0-5 and the clamp are safe).
- Run `VALIDATE_LIQUIDS` once for small dye volumes. Dye substocks ~3 mL (small-tip vials
  keep a ~2 mL reserve).
- Simulation mode for Nile Red (Ailsa's simulator covers pyrene only).

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

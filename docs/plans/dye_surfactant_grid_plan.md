# Dye x surfactant grid workflow — plan

Date: 2026-11-18 · Status: planning (design agreed in outline; details and open decisions below)

## Goal

One 96-well plate, one surfactant and one dye per run (both chosen in config). A 2D grid
crosses the surfactant's CMC on one axis and varies the dye concentration on the other,
with replicates. Read fluorescence for every dye, plus absorbance at the absorption
maximum for Nile Red and Coumarin-6 (to estimate how much dye is actually in solution).
Repeat the reads over a timepoint schedule. No turbidity read (only one surfactant).

Combines the useful parts of `workflows/surfactant_grid_ailsa.py` and
`workflows/fluorescence_calibration_workflow.py`, built on the North_Safe flows in
`docs/platform/automation_catalog.md`, starting from `workflows/workflow_template.py`.

## Vials

- 1 dye stock, 1 surfactant stock, 1 dye-solvent vial, 1 water vial
- `surfactant_refill` and (if needed) a dye stock refill; bulk dye solvent
- Generated: surfactant substocks and dye substocks (`_PRODUCT_VIALS`)
- These could move to reservoirs later; not set up yet.
- Stocks carry a `stock_id` (concentrations come from `status/stocks.csv`, not code tables).

## Steps

1. **Plan** (pure function `build_plan(config)`, testable without hardware)
   - Surfactant axis: 0 plus N log-spaced levels from CMC/k to CMC*k (optionally denser
     near the CMC, as in `create_cmc_control_series`).
   - Dye axis: 0 plus M levels as fractions of the dye stock.
   - Wells = surfactant x dye x replicates, plus optional dye-in-solvent reference wells.
   - Built-in controls: surfactant 0 = dye in water; dye 0 = surfactant-only blank at
     every surfactant level (blank for fluorescence and absorbance).
   - `TOTAL_VOLUME_UL` includes the dye; water = total - surfactant - dye. Every well
     checked against the plate's `max_volume_per_well`.
   - Outputs: well table + substock recipes, saved to the run folder.
2. **Substocks** with `create_substock` (reservoir water first, then pipetted stock, vortex).
   - Dye: fixed dye volume per well (fluorescence-workflow approach), so the solvent
     fraction is identical in every well. One substock per intermediate level; top level
     = stock, level 0 = pure solvent. Sized to plan demand + reserve.
   - Surfactant: variable volume per well (adaptive-workflow approach). Fewest substocks
     such that every level's volume is between the minimum well volume and the space
     left; one substock covers ~10-20x, so typically 1-2.
   - `refill_vial` for water (reservoir 1), surfactant (`surfactant_refill`), solvent.
3. **Dispense** with `dispense_from_vials_into_wellplate`, one call per component in
   `DISPENSE_ORDER` (default water -> surfactant -> dye). The flow stages small-tip
   vials, conditions, groups by tip and returns vials home. `RANDOMIZED_ORDER` (seeded)
   shuffles well positions. Whole-run supply check before the first move.
4. **Measure** with `lash_e.measure_wellplate([fluorescence_protocol, absorbance_protocol])`
   (one Cytation trip). Pyrene: fluorescence only (I1/I3 ratio, from ailsa's `_DYE_PROTOCOLS`).
5. **Kinetics**: repeat step 4 over `MEASUREMENT_SCHEDULE_MIN` with
   `mark_clock_start` / `mark_wait_until`.

## Worked example (SDS 50 mM stock, CMC 8.6 mM; Nile Red)

- Surfactant: 0 + 1.7, 3.0, 5.1, 8.7, 15, 26 mM. From stock: 6.9, 12, 20, 35, 60, 103 uL;
  6.9 uL is too small, so a 10 mM substock covers the two lowest (34, 59 uL).
- 7 surfactant x 4 dye x 3 replicates = 84 wells + 4 dye levels x 3 in pure solvent = 96.
- ~8 vials, ~10-15 tips per plate.

## Keep / drop from existing workflows

| From | Keep | Drop |
|---|---|---|
| Fluorescence calibration | Fixed dye volume + per-level substocks; `build_plan` without hardware; pre-move supply check; dye-in-solvent reference; timepoint schedule | Own staging workaround, multi-source solvent pooling, `TWO_RUNS_PER_PLATE` |
| Ailsa | `_DYE_PROTOCOLS` (per-plate protocol files, column mapping, pyrene ratio); seeded randomization; `DISPENSE_ORDER`; replicates | Turbidity, two-surfactant replay CSVs, per-well Cytation runs, chunked refill engine |
| Adaptive | Variable-volume surfactant source selection; cascade substock recipes; CMC-centred spacing | 90 uL surfactant cap; `SURFACTANT_LIBRARY` as stock-concentration source |

## Risks found (stress test)

1. **Absorbance signal too low at current stocks** (rough: eps ~4e4 Nile Red, ~5e4
   Coumarin-6 M^-1 cm^-1, ~0.6 cm path at 200 uL). Nile Red 50 uM stock x 5/200 -> A ~0.03;
   Coumarin-6 3 uM -> A ~0.002. A ~0.1 needs ~4 uM in the well (~150-200 uM top stock).
   Fluorescence is linear only below A ~0.05-0.1, so use a wide log dye axis: low end for
   fluorescence, high end for absorbance. Both dyes aggregate in water below the CMC.
2. **Absorption maximum shifts with environment** (Nile Red strongly solvatochromic).
   Options: spectral scan (slow), or 2-3 wavelengths + ~750 nm reference (also gives a
   scattering/turbidity correction).
3. **No absorbance protocols exist** for Nile Red / Coumarin-6 (`_96` copies needed in
   Gen5). Plates must be clear or clear-bottomed.
4. **Closed-cap vial-to-vial transfers** (`create_substock` -> `dispense_from_vial_into_vial`)
   move vials while a tip is held; only rack 0-5 and the clamp are safe, otherwise
   `pause_after_error` mid-transfer. Affects stock -> substock for both dye and surfactant.
   Fix in North_Safe (in progress; see below).
5. **Small dye substocks vs small-tip reserve** (~2 mL kept in small-tip source vials), so
   dye substocks ~3 mL. Run `VALIDATE_LIQUIDS` once for the 5 uL dye dose.
6. **Concentrations from `status/stocks.csv`** via `stock_id` (file and vial-file column
   not set up yet). CMC is a config value.
7. **Evaporation** over long schedules: consider `use_lid`.
8. **Simulation model** needed for all three dyes and both readouts (ailsa simulates
   pyrene only).
9. Surfactant concentration computed over the total volume (surfactant + water + dye).

## Open decisions

1. Dye stock concentrations: raise for absorbance (accept fluorescence saturation at the
   top), or keep fluorescence-range and treat absorbance as best-effort?
2. Absorbance read: spectral scan, fixed wavelengths + 750 nm reference, or single max?
3. Keep the 12 dye-in-solvent reference wells?
4. "Kinetic sequence" = timepoint schedule, or adaptive's `KINETICS_SEQUENCE` (when dye
   is added relative to reads)?
5. Closed caps: fix in North_Safe first (chosen 2026-11-18) vs vial-layout workaround.

## Staging rules (agreed 2026-11-18)

- Open small-tip vials, safest first: rack 47, 46, 45, heater 2, 5, 8, rack 44, 43
  (`small_tip_staging_rank` in `vial_positions.yaml`; clamp and clamp_2 no longer used).
- Closed vials: rack 0-5 (safe to move with a tip held).

- One rule, in `stage_vials`, used by both flows (vial to wellplate, vial to vial):
  closed -> rack 0-5, open small-tip -> small-tip safe positions. Clamp never a staging spot.
  A staged vial leaving the clamp goes back to its staged spot. Implemented 2026-11-18.
- Vial to wellplate: fresh tip per vial; each vial home right after its dispense.
- Future, not needed yet: tip reuse from dilute to concentrated.

## Next

- Fix the vial-to-vial / clamp issues in North_Safe.
- Then: `_CONFIG_KEYS` / `_CONFIG_DOCS` list and `build_plan` spec.

# Six fluorescence pairs: preparation and measurement

Import `scheduler/config_lists/fluorescence_six_pairs.yaml` into the Scheduler.
These twelve full configs live under the existing Git-ignored workflow config
directory. They are local presets; importing copies them into private queue rows.

| Jobs | Phase | Solvent | Dye order |
|---|---|---|---|
| 01-03 | Preparation only | Ethyl acetate | Pyrene, Coumarin-6, Nile Red |
| 04-06 | Preparation only | DMSO | Pyrene, Coumarin-6, Nile Red |
| 07-09 | Dispense and measure | Ethyl acetate | Pyrene, Coumarin-6, Nile Red |
| 10-12 | Dispense and measure | DMSO | Pyrene, Coumarin-6, Nile Red |

Preparation jobs acquire no plates. Measurement jobs use two halves per plate:
plate 1 is jobs 07/08, plate 2 is 09/10, plate 3 is 11/12. Each measurement job
uses 30 wells (water and surfactant, five concentrations, three replicates).
The retained plate may carry different dyes and protocols in its two halves.

All substocks are 4 mL. Ethyl acetate uses 20 uL dye additions, factors
0/0.025/0.0625/0.125/0.25, and volatile-solvent mode: 250 uL aqueous medium
plus dye solvent. DMSO uses 5 uL dye additions, factors 0/0.1/0.25/0.5/1,
and nonvolatile mode: 245 uL medium plus 5 uL dye solution. Both series
have the same final concentration relative to their own parent stock.

The parent concentration remains 48 uM in every config, inherited from the
existing preset. The measurement schedule remains
0/10/30/60/120/240/360/480 minutes, one read each: about 48 hours total for
six sequential measurement jobs, plus preparation and robot handling.
All-preparation-first does not keep the last volatile preparation fresh;
vial capping is handled by the existing robot routines. There is no added
evaporation wait, evaporation measurement, or dye-specific scheduling logic.

Working solvent is `ethyl_acetate_rf_1` or `DMSO_rf_1`, kept separate from
preparation sources. Ethyl acetate sources are `ethyl_acetate`,
`ethyl_acetate_2`, `ethyl_acetate_rf_2`, `ethyl_acetate_rf_3`. DMSO sources
are `DMSO`, `DMSO_rf_2`, `DMSO_rf_3`. All preparation sources retain 2 mL.
The second 20 mL ethyl acetate source was added at large-rack slot 3,
recorded at 20 mL per the requested supply. Verify its physical fill matches.

`VIAL_NAME_OVERRIDES` maps generated names to the existing inventory names,
including working-solvent names and the existing Pyrene/Coumarin ethyl
acetate stock/substock spellings. The original vial names were not changed.
Recipe volumes are reused between each preparation/measurement pair;
existing substocks over 1 mL are assumed correctly prepared and skipped.

Embedded pipetting validation is disabled in jobs 01-06 and enabled in jobs
07-12 (`VALIDATION_ONLY` remains false). Each measurement job validates before
startup refills and plate dispensing. Prepared substock recipes are excluded:
ethyl acetate is checked at 20 uL and water/surfactant at 250 uL; DMSO is
checked at 5 uL and water/surfactant at 245 uL. Surfactant uses water pipetting
settings, as in the existing preset. Checks use pure working solvent as a
proxy for dye solutions and use the actual selected pipette tip, three
replicates per chosen volume. Validation may apply adaptive corrections.
Reimport the list if the rows were imported before validation was enabled.

Validation on 2026-10-23, before enabling embedded pipetting checks: all twelve jobs completed the existing isolated
scheduler simulation, with successful handoffs, zero errors, and three plates
used. Session: `scheduler/state/a912a1b950dc4d1b90b6111630c51e6a`.
Simulation changed only private state copies, not live inventory or hardware.
Preparation demand was 42.45 mL ethyl acetate and 25.8 mL DMSO.
Final simulated solvent-source volumes were at least 2 mL. Working solvent
finished at 7.41 mL ethyl acetate and 7.71 mL DMSO. Surfactant refill finished
at 1.702 mL; aqueous reservoir top-ups total approximately 19.6 mL.

The simulation emitted warnings during ethyl acetate preparation: 100 uL
transfers use default pipetting parameters because the available calibrated
range is 10-50 uL; small-tip aspiration from staging slot 0 may cause an
automation issue. These warnings were not suppressed. Review the liquid
operations in the Scheduler before live execution. Current Cytation protocol
paths exist locally; optical settings still come from those protocol files.

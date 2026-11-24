# Pipetting follow-ups

Small issues to deal with later. Not urgent.

- **Validation conditions every liquid; calibration does not (flagged 2026-11-21).** Calibration
  conditions a tip only for `refill_pipets: false` liquids. The embedded validations
  (`embedded_validation_solvents.py`, `validate_all_calibrations.py`) pass
  `condition_tip_enabled=True` for every liquid, so `refill_pipets: true` liquids (glycerol,
  PVA_DMSO) are validated with conditioned tips but calibrated without. Not used at the moment.
  Fix: take the setting from `condition_tip` in `robot_state/liquids.yaml`, as the flows do.
- **Sample tracking: one unknown removes the ± for every material in a container (flagged
  2026-11-21).** The record's `known` flag is for the whole container, so e.g. a well loses its
  dye ± when only its SDS source is unknown. Make it per material in
  `experiment_tracking/sample_contents.py`.
- **Large tip, water: needs better calibration settings (flagged 2026-11-21).**
- **Small tip, water and SDS: add blowout to the calibration settings (flagged 2026-11-21).**
- **Large tip, ethanol: blowout is probably longer than needed (flagged 2026-11-21).**
- **Idea: LLM-driven improvement of pipetting parameters (flagged 2026-11-21).** Instead of
  the old-fashioned Bayesian optimization, use an LLM to improve pipetting parameters
  automatically: analyze existing pipetting data, take human feedback, and respond to its own
  results. Tried before and it didn't work, but it might now. Plan:
  `lab_slack_agent/docs/plans/2026-11-21-llm-pipetting-improvement.md` (agents repo).
- **Tip conditioning settings are hard-coded (flagged 2026-11-21).** `condition_tip` in
  `North_Safe.py` hard-codes cycles (3), aspirate_speed (10), dispense_speed (5) and
  blowout_vol (0.5 mL). Move them to a `tip_conditioning:` block in a
  `robot_state/` YAML and fail loudly if missing.
- **Well A4 reads low on the same layout two days running (flagged 2026-11-23).** Dye CMC
  runs `20261121_141525` and `20261122_101912` (identical layout): A4 (11 mM SDS, 1 uM dye)
  was -6.1% and -5.6% below its replicates (C8, C9). The other plateau wells did not repeat
  across days (r = 0.13), which fits random pipetting. A4 is an edge well, so suspect position
  (reader or plate edge), not pipetting. Check: does A4 read low in other runs, or on a uniform
  plate? Script: `ai_scratch/replicate_crossday.py`.
- **Ailsa's surfactant grid: coumarin-6 now reads 470/510 (flagged 2026-11-23).**
  `C:\Protocols\Coumarin_96.prt` was changed from ex 485 / em 530 to the micelle peak,
  470 / 510, and the channel in `_DYE_PROTOCOLS` renamed to `fluorescence_470_510`. Before
  the next `surfactant_grid_ailsa.py` coumarin run, check its analysis and note that earlier
  runs (485/530) are not directly comparable.

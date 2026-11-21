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

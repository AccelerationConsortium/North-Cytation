# Overaspirate review — water and SDS (chunk 1, 2026-10-30)

Estimated best overaspirate for each calibrated row, from validation history up to
Friday evening, using only measurements made with that row's current settings (speeds,
waits, air, blowout). Full numbers: `overaspirate_review.csv`. Code:
`pipetting_data/overaspirate_estimation.py`.

- **Effective now** = what the robot aspirates today (row value plus the old lookup
  compensation from the row's stored measured volume).
- **Estimate** = local-level (Kalman) estimate from all sessions; **± SD** is its
  uncertainty. Every replicate counts (not just the "best stage").
- **Backtest** = error predicting each next session: this model vs today's method
  (jump to the latest result). Lower is better.

## Recommended updates (7 rows)

| Liquid | Tip | Volume | Effective now | Estimate ± SD | Change | Sessions | Backtest model / today |
|---|---|---|---|---|---|---|---|
| water | small | 50 uL | 2.13 | 3.19 ± 0.48 | +1.07 | 113 | 0.94 / 1.04 |
| water | small | 70 uL | 5.77 | 4.17 ± 0.22 | -1.60 | 10 | 0.89 / 0.98 |
| water | small | 100 uL | 4.43 | 2.28 ± 1.11 | -2.15 | 120 | 2.05 / 2.42 |
| water | small | **150 uL** | **-3.68** | **1.97 ± 1.55** | **+5.65** | 120 | 2.99 / 3.40 |
| SDS | small | 50 uL | 4.51 | 5.66 ± 0.70 | +1.15 | 74 | 8.77 / 8.74 (noisy row) |
| SDS | small | 100 uL | 2.64 | 3.31 ± 0.55 | +0.67 | 74 | 1.07 / 1.21 |
| SDS | small | 150 uL | 5.16 | 3.56 ± 0.68 | -1.60 | 74 | 1.46 / 1.67 |

(All values uL of overaspirate.) Water 150 uL undoes Friday's bad validation jump.
SDS 50 uL is noisy (model no better than today); accepting it is optional.

Applying an update: `overaspirate_vol` = estimate and `volume_measured` = target, so the
old lookup compensation adds nothing and the robot uses exactly the estimate (works
with the old code still running and with the new code).

## Keep as is

- **Current value within uncertainty**: water 10, 20 (small), 200 (large); SDS 20 (small),
  200, 500, 800, 900 (large).
- **Too noisy to estimate**: water large 500 uL (session estimates swing -16 to +20 uL).
  Water 800 uL is similar (kept; within its wide uncertainty).
- **No history with current settings / too few sessions**: water small 2.5, 5, 80, 180;
  SDS small 10, 25, 75, 180.

## Noticed along the way

- Water 500 and 800 uL swing **together** within a session, so much of their spread is a
  shared per-session effect (vial level, tip batch or environment), not row error.
  Worth modelling later as a common session offset across volumes.
- The 200–500 uL water large-tip range interpolates between two different setting sets
  (200: aspirate 15 / dispense 10, no blowout; 500: 8 / 14 with blowout and air).

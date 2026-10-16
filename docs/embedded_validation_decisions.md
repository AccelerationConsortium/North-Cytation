# Embedded validation decisions

The final result is selected separately for each session, liquid, and target volume.
The master measurements retain every stage. The master session summary and saved
per-volume statistics describe the selected stage, not the highest stage attempted.

## Fields

| Field | Meaning |
| --- | --- |
| optimization_stage | 1 = baseline; 2 = crossing adjustment; 3 = interpolated adjustment |
| accepted | This stage supplies the final result. All its replicates share this value. Blank = unresolved historical decision. |
| adaptive_correction | Whether adjustment was enabled. Historical single-stage records leave this blank because the mode cannot be inferred. |
| within_tolerance | Whether the stage's absolute mean error meets the recorded tolerance; it is not a per-replicate test. |
| tolerance_ul | Accuracy tolerance used, in µL |
| decision_reason | baseline_passed, baseline_retained, best_stage_selected, or optimization_failed |
| decision_source | recorded, reconstructed_replicate_order, or reconstructed_correction_confirmed |

An accepted result can remain outside tolerance. Acceptance means selected as the
reported outcome, not that quality passed or that a calibration-file write succeeded.
An interrupted optimization is marked accepted=False; the baseline remains available
as the explicitly failed fallback summary.

Correction records describe the attempted correction. Their accepted flag indicates
whether that proposal was selected, and selected_stage identifies the final retained
stage. A rejected correction can therefore have accepted=False while the retained
baseline measurements have accepted=True. The session summary is authoritative for
the final outcome. No correction row is required when no adjustment was needed.

## Existing algorithm preserved

The selection criterion is absolute error of the stage mean, not precision or the
largest stage number. If adaptive correction is enabled and Stage 1 is outside
tolerance, the program tests Stages 2 and 3. It ranks stages by absolute mean error,
breaking ties in stage order. It selects the best stage when it meets tolerance or
improves on baseline; otherwise baseline remains selected.

The existing tolerance function is preserved exactly. No new acceptance threshold
or adjustment behavior was introduced. The independent writer test drives all three
stages and verifies that a winning Stage 2 supplies the summary even after Stage 3.

The fields are saved in master_pipetting_measurements.csv,
master_pipetting_sessions.csv, and new per-session raw_measurements.csv and
volume_statistics.csv files. Correction attempts also record explicit decision fields.
CSV appends align columns by name and extend old headers safely.

## Historical backfill, 2026-10-16

Backup: `pipetting_data/backups/acceptance_20261016_104525/`.
It contains the three original master/correction CSVs and original embedded validation
program, plus a SHA-256 manifest. `backfill_report.csv` records each session-volume result.

All historical groups contained either one replicate block or three equally sized
blocks with replicate numbers restarting from 1. The backfill inferred stages from
this ordering and reproduced the mean-error selection rule. When a stage-specific
correction record was available, both its selected stage and recorded overaspirate
were checked against the reconstruction. Conflicts remain blank instead of asserting
an unsupported historical decision. Reconstruction is identified in decision_source;
it does not constitute a newly recorded decision from the original experiment.

Results: 3,014 resolved session-volume groups; 210 unresolved groups. There are 3,010
matching session-summary rows (four resolved measurement groups lack a summary row).
All 21,024 original measurement rows and their original values were preserved.
508 session means changed to use the selected stage. All 284 correction records remain;
the historical three unnamed environmental columns now have proper headers.
The referenced per-session raw files were unavailable in this workspace, so no
historical raw files were modified. Future raw files receive the decision fields.

Independent verification checked original measurement equality, uniform CSV widths,
every resolved session's mean/count/stage, original backup hashes, and selection against
the original algorithm's arithmetic. Unresolved session outcomes retain their original
per-volume statistics; overall regression values are cleared for any session containing
an unresolved volume, to avoid presenting a mixture as a coherent selected-stage fit.

The backfill script requires a matching original backup and refuses a second run on
already changed input. It is `scripts/backfill_validation_decisions.py`.

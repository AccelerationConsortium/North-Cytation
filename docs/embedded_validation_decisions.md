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

## Recording guarantees, 2026-10-16

Vial-to-vial validation resolves the baseline parameters separately for each target
volume and passes that resolved object to the transfer. Stages 2 and 3 preserve
those settings except for their explicitly adjusted overaspirate. Each raw and
master measurement records its own stage's complete pipetting parameters:
aspirate/dispense/retract/blowout speeds, aspirate/dispense/post-retract waits,
overaspirate, pre/post-aspiration air volumes, blowout volume, and mixing cycles.
Missing attributes are errors, not substitutes from a hardcoded parameter snapshot.
The documented optional blowout-speed fallback is the aspirate speed.

Master session summaries and saved volume statistics carry the selected stage's
parameters, not baseline parameters or the last stage attempted. Raw records also
carry session ID and explicit simulation status. Master measurements preserve
measurement timestamp (`measured_at`), mass, retry count, stability information,
quality result, and output folder. Stage 2/3 stability information is retained when
the controller supplies it. A legacy controller returning mass only has no stability
information; no balance trace is fabricated.

With raw saving enabled, each completed measurement is checkpointed using an
atomic file replacement before the next transfer or calibration update. Final raw
output adds the stage-selection decisions. An interrupted checkpoint may therefore
lack final decision columns and must not be interpreted as a completed validation.
Raw/correction/master write failures propagate to the caller instead of returning
`success=True`. Writes to the different files are not a single transaction: a
failure can leave raw data or one master updated without the other. Do not blindly
rerun a failed append; reconcile using session ID and the preserved raw file.

Simulation still executes every automation step but writes no raw, statistics,
plots, corrections, or master records, and disables controller mass-trace saving.
Both master writers require the caller's explicit confirmed simulation flag;
they do not use the process-global simulation environment variable.
`save_raw_data=False` or `output_folder=None` explicitly disables local raw/plot
output for real runs, but does not disable real master recording.

Reservoir validation likewise records its resolved reservoir parameters per
volume in raw measurements and both reservoir masters, with explicit simulation
isolation and propagating write failures.

These guarantees apply to new recordings only. Existing CSV headers are extended
by name on the next successful append; historical values are not backfilled,
verified, or repaired by running a new validation.

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

## Real-only metadata repair, 2026-10-21

After simulated and unresolved-provenance rows were excluded, the retained snapshot
contained 7,512 measurements, 1,401 per-volume summaries, and 459 sessions.
A hash-verified copy of both cleaned masters was preserved in
`pipetting_data/archives/real_metadata_repair_20261021_175900_6251466/`.

The repair labels stages and selected-stage decisions for all 7,512 measurements
across 1,405 session/volume groups. It preserves the 101 originally recorded
decisions; the other 7,411 are explicitly identified as historical reconstructions.
Complete replicate blocks, compatible raw-stage identities, historical annotations,
and stage-specific correction decisions were reconciled. Identical numeric readings
in different stages remain separate observations.

Actual scalar overaspirates are supported for 7,509 measurements; 4,654 differ
from the original labels. Parameters were captured at the uniquely matched
aspiration operation, not from the last settings before the next transfer.
Millisecond-resolution log timestamps are compared with raw timestamps at that same
precision, with target, vials, mass and stage identity also required. Missing raw
files were supplemented only by unique complete ordered session/transfer matches.

Three Stage 1 SDS measurements in session `0b505162` (2026-05-06, 0.9 mL)
used split transfers. Their `overaspirate_vol` is intentionally blank; actual
component settings are retained in `overaspirate_parts_ml` and the evidence ledger.
Do not sum, average or copy a component into the scalar column. All three are
unselected, so none prevents a supported selected-stage summary.

`overaspirate_source` records this repair's evidence status. Existing future
recordings remain identified by their normal `decision_source=recorded` contract;
blank repair-specific provenance on a later append is not a historical audit verdict.
Detailed parameter evidence, original values and unresolved scalar reasons are in
`recovery/pipetting_real_metadata_20261021/staged/`.

All 1,401 existing summaries were reconciled to selected measurements, including
36 changed means, and complete session fits were recomputed. Four measurement
groups have no existing summary row; creating those missing rows belongs to the
separate recovery/import step. No measurements or summaries were added or removed.
Single-stage historical records still do not establish whether adaptive correction
was enabled; that optional field remains unknown unless independently recorded.

The repair stages replacements before activation, checks original hashes while
holding both master files exclusively, and restores the original pair if activation
fails. A persistent `activating` manifest after a process interruption requires
restoring both archived files before resuming writes; the two files are not a
filesystem-atomic transaction. Independent verification checks original measurement
conservation, recorded metadata preservation, stage selection, aspiration-boundary
parameters, summary arithmetic, and rollback on disposable files.

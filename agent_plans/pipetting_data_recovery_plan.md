# Pipetting data cleanup and recovery plan

Date: 2026-10-21

Status: Confirmed-simulation cleanup and the user-requested exclusion of the
remaining 50 unclear sessions are complete. The active real-only snapshot contains
7,512 measurements and 1,401 per-volume summaries. Stage/selected-stage metadata
repair is complete for those retained measurements; 7,509 have verified scalar
overaspirates, while three split transfers preserve component values instead.
All existing summaries are reconciled to the selected measurements. The original
measured values, identities and row counts remain unchanged. Analyzer/GUI review
and Step 4 (missing real data recovery) remain pending.

Pre-repair cleaned originals are preserved in
`pipetting_data/archives/real_metadata_repair_20261021_175900_6251466/`.
See the [repair report](../recovery/pipetting_real_metadata_20261021/REPORT.txt)
and [field-level ledger](../recovery/pipetting_real_metadata_20261021/staged/field_changes.csv).
The historical planning counts and investigation states below describe earlier
snapshots, not the current active dataset.

Among those 50 archived unclear sessions, 23 have strong but unconfirmed synthetic-fallback
evidence, 22 have protocol associations without enough independent execution
evidence, and five June 11 sessions lack contemporaneous local logs/raw sources.
See the [current investigation report](../recovery/pipetting_20261021_workflow_patterns/CURRENT_REPORT.txt)
and [individual remaining-session ledger](../recovery/pipetting_20261021_workflow_patterns/remaining_50_sessions.csv).

Execution evidence: [cleanup report](../recovery/pipetting_cleanup_20261021/REPORT.txt)
and [original archive manifest](../pipetting_data/archives/confirmed_simulation_cleanup_20261021_171031_367547/manifest.json).
The [unclear-session exclusion manifest](../pipetting_data/archives/historical_unresolved_exclusion_20261021_174651_828607/manifest.json)
records the subsequent removal of unresolved records without reclassifying them.
Historical audit row numbers refer to the archived originals. Use the archive's
`row_manifest.csv` and the metadata repair's backup-relative measurement ledger
to identify retained occurrences. The earlier
[row mapping](../recovery/pipetting_20261021_workflow_patterns/current_row_mapping.csv)
predates the final unclear-session exclusion. The counts in the original planning
tables below describe the source snapshot before the first cleanup.

## Objective

Make the pipetting analyzer useful with genuine experimental data, reduce the
amount of irrelevant data it processes, and recover real measurements and their
scientific context without losing historical evidence.

Execute the work in the user's requested order:

1. Archive the existing masters and remove confidently identified simulation.
2. Investigate the remaining unclear records.
3. Repair stage information and actual overaspirate values for confirmed real data.
4. Recover confirmed real measurements that never reached the masters.

The first cleanup must be independently usable. It must not wait for every
historical ambiguity, stage decision, or parameter discrepancy to be resolved.

## Current evidence and expected first cleanup

The latest classifications are in
[all_master_rows_updated_review.csv](../recovery/pipetting_20261021_unresolved_review/all_master_rows_updated_review.csv).
They supersede the larger unresolved counts in the first October 21 audit.

| Current master records | Measurements | Per-volume session summaries |
| --- | ---: | ---: |
| Confirmed real | 7,512 | 1,401 |
| Confirmed simulated | 12,758 | 1,517 |
| Unresolved | 1,920 | 452 |
| Total | 22,190 | 3,370 |
| Retained after removing only confirmed simulation | 9,432 | 1,853 |

These counts describe a particular source snapshot. Recheck hashes and reconcile
new recordings before using the classifications against current row positions.
The master hashes for this snapshot are recorded in the
[prior-work summary](../recovery/pipetting_prior_work_review_20261021/summary.json).

The analyzer's default scientific input should contain only confirmed real rows:
currently 7,512 measurements and 1,401 summaries. Keep unresolved rows in the
retained masters, explicitly labeled and excluded from the analyzer by default.
This preserves uncertain data while preventing it from silently becoming evidence
for a calibration estimate. Provide a separately labeled review mode for it.

Real provenance does not establish stage acceptance, measurement quality, or
correct parameter metadata. Those distinctions remain visible until Step 3.

## Rules for all four steps

- Preserve every original source occurrence in a verified archive. Removing a row
  from an active master does not delete its experimental evidence.
- Keep real, simulated, and unresolved provenance separate from accepted/rejected
  stages and passed/failed quality checks. A failed workflow can contain genuine
  measurements. A deliberately inaccurate optimization trial can also be real.
- Never classify from `_simulate` filenames, directory names, current YAML, old
  quarantine membership, usernames, or a duration cutoff. Timing supports a
  sufficiently specific identity match; simulations can take minutes.
- Use measurement-specific evidence where possible: raw measurement identity,
  unique ordered transfer sequences, actual mass acquisitions, and explicit
  controller branches. Preserve mode changes and contradictory evidence.
- Distinguish the saved live plan from the effective scheduler launch config.
  Scheduler dry runs deliberately override the live plan to simulation mode.
- Preserve unknown values. Do not replace an unknown historical setting with a
  current calibration value, runtime default, or the most common historical value.
- Keep large evidence JSON and balance traces in the audit archive. Production
  tables need compact identifiers, provenance status, and links to the evidence.
- Do not execute older repair/import scripts blindly: several rewrite production
  files, require missing backups, or target a different source snapshot.

## Step 1 Archive and remove confirmed simulation

### Prepare a reproducible migration

1. Identify the analyzer entry point in use and its actual data-loading path.
   Known consumers include [scheduler/liquid_review.py](../scheduler/liquid_review.py),
   [oav_calibration_analysis.py](../analysis/oav_calibration_analysis.py), and
   [oav_current_vs_best.py](../analysis/oav_current_vs_best.py). Inventory other
   readers and both embedded-validation writers before changing the schema.
2. Capture current master hashes, row counts, headers, and the audit versions used.
   Reconcile any rows added since the classified snapshot. Source-row numbers
   alone are not durable identifiers after filtering or rewriting a CSV.
3. Create a dated archive under `pipetting_data/archives/`. Copy both masters
   byte-for-byte and verify their hashes. Also preserve the relevant correction
   history, classification manifests, and code revision/reference hashes. Keep
   existing quarantine and recovery directories intact.
4. Give archived occurrences durable references based on original source hash and
   row, and carry existing measurement IDs where available. Maintain a mapping to
   retained entities so later investigations survive reordered or shortened CSVs.
5. Produce three review partitions for each master: confirmed real, confirmed
   simulated, and unresolved. List each proposed exclusion with its evidence and
   reason. Validate identity against the archived row, not a stale row index.

### Prepare the smaller active files and analyzer input

6. Stage replacements at the existing master filenames containing confirmed real
   plus unresolved rows. Remove only confirmed simulation. Preserve the retained
   scientific values and every real optimization stage during this step.
7. Add a compact, explicit provenance status through a compatible schema update
   or shared loading layer. Missing status must remain unresolved. Do not turn a
   blank legacy simulation flag into `False` merely to make it load.
8. Generate a compact confirmed-real analysis input, or use a shared loader that
   provides that same subset. Wire the user's analyzer to this default. Make the
   unresolved exclusion count visible and allow intentional review of uncertain
   data without mixing it into calibration estimates.
9. Normalize the existing `simulate` and `simulation_mode` naming at the reader
   boundary. Some consumers currently check `simulation_mode`, while the embedded
   masters use `simulate`. An absent column must not bypass provenance filtering.
10. Preserve append compatibility. New real recordings need the same provenance
    contract and durable identity; otherwise they could become invisible to the
    analyzer after cleanup. Confirm effective workflow, coordinator, and relevant
    controller modes agree. Simulation appends remain blocked.

### Validate and activate

11. Check row conservation separately for measurements and summaries:
    `original = retained real + retained unresolved + excluded simulated`.
    No retained value may change during the simulation-removal migration.
12. Confirm the October 19 real run retains all 101 measurements and 15 summaries.
    Confirm the October 14 simulated records are excluded. Check that archive
    copies reproduce both original master hashes.
13. Exercise the actual analyzer using the staged confirmed-real input. Measure
    load time and peak memory before and after; verify plotted/fitted inputs contain
    no simulated or unresolved rows. Read only required columns and avoid loading
    audit ledgers or all balance traces into the analyzer. Broader storage changes
    are justified only if a measured bottleneck remains after filtering.
14. Activate during a coordinated pause in master-file writes. Do not cancel a
    running robot experiment automatically. Recheck hashes under the writer lock;
    if inputs changed, regenerate the staged files from the new snapshot.
15. Replace both CSVs under that lock with a migration manifest and a tested
    recovery path. Two file replacements are not a single atomic transaction:
    interrupted activation must either finish the recorded generation or restore
    the archived pair before writes resume.
16. Verify that subsequent real appends appear in the analyzer, simulated appends
    create no rows, and a second migration run cannot duplicate data or overwrite
    the original archive.

Deliverables: verified originals, exclusion manifest, smaller active masters,
confirmed-real analyzer input, count/memory comparison, and rollback instructions.
Expected unchanged-snapshot retained counts are 9,432 measurements and 1,853
summaries; the analyzer default uses the confirmed-real subset within those files.

## Step 2 Resolve unclear records

Work from the
[remaining master review](../recovery/pipetting_20261021_unresolved_review/remaining_unresolved_master.csv)
and [session review](../recovery/pipetting_20261021_unresolved_review/session_review.csv).
The current remainder is 2,372 rows across 222 sessions.

| Remaining evidence | Rows | Next investigation |
| --- | ---: | --- |
| Simulated controller activity exists in candidate parent logs, but session identity is incomplete | 1,208 | Recover validation boundaries, session IDs, output paths, and protocol-specific identity links. |
| Simulation declared without enough controller or identity confirmation | 148 | Find actual controller branches or the originating computer's detailed logs. |
| No discriminating mode or measurement evidence | 822 | Inspect older log formats and archived raw/config artifacts; prioritize March 16 and April 2 groups. |
| Candidate logs contain real activity, but the session link is incomplete | 48 | Separate overlapping real and simulated workflows; nearby real activity cannot classify these rows. |
| No contemporaneous log found | 146 | Search for the five June 11 validation sessions' original logs/raw files, including external historical output locations. |

1. Review sessions as complete ordered measurement sequences. Allow extra logged
   transfers, such as retries, without inventing missing measurements or forcing
   an ambiguous match. Preserve the exact matching transfer lines and masses.
2. Use the verified folder-move manifest to repair old path associations. Check
   raw files, correction records, run-era source, saved execution results, and Git
   versions together. Record any machine-clock or timezone uncertainty.
3. Follow the other-computer leads separately: the earlier comparison found 22
   raw occurrences and 438 already-quarantined occurrences with time-only links
   to `owenm` runs. These are not additional unresolved current-master rows.
   The tracker records usernames, not hostnames; 105 detailed logs were unavailable
   locally in that snapshot. Importing the original logs could resolve more than
   further guesses from their timestamps.
4. For each decision, record the prior status, evidence, new status, and reviewer
   rationale. Promote confirmed real rows into analyzer input; move newly confirmed
   simulation out of active masters into the preserved exclusion set. Leave
   genuinely unresolved cases labeled and excluded from the analyzer default.

Deliverable: a shrinking unresolved-session ledger with explicit blockers and
versioned changes to the active-data partitions. Do not delay Step 1 for this work.

## Step 3 Repair stages and actual overaspirate for real data

Defer detailed parameter reconstruction until the first cleanup and unclear-data
review are complete enough to provide a trusted real subset.

The old code could record the initial/configured overaspirate rather than the
actual value used for a later optimization stage. A measured volume can therefore
be genuine while its attached parameter vector is wrong.

1. Restore historical stage/acceptance candidates only after matching current
   confirmed-real measurement identity. Historical revisions contain simulated
   data and omit later rows; never replace current masters wholesale from Git.
2. Start with the already matched real-only candidates: 5,363 measurements and
   1,146 summaries. Four measurement identities have conflicting historical stage
   annotations and require individual review. Preserve recorded versus reconstructed
   decision provenance. Reconstruct remaining real groups only where evidence permits.
3. Keep three parameter concepts distinct: original/configured overaspirate,
   actual overaspirate used for each measurement/stage, and the selected final
   stage's overaspirate. Correct the analyzer-facing actual value only with evidence;
   keep the old value and change rationale in the audit ledger.
4. Prefer explicit per-measurement raw settings and uniquely matched transfer logs.
   Use correction records only when they can be tied to the actual operation.
   A proposed correction or updated calibration CSV is not proof that it was used.
   Split transfers may require several parameter vectors; do not flatten them into
   a fictitious single setting.
5. Review the older audit's 4,202 distinct real master measurements with parameter
   discrepancies, including overaspirate and other settings. Do not replace missing
   values using the old exploratory modal-parameter filter.
6. Recompute each completed session/volume summary from its selected stage's
   replicates. Recompute cross-volume fits from those selected results when the
   constituent decisions are known. Keep incomplete or ambiguous outcomes explicit.
   Fourteen currently confirmed-real summaries already differ from the earlier
   reconstructed means and are a focused starting point for verification.
7. Preserve all real attempted stages. Accepted means selected as the reported
   outcome; it does not mean within tolerance or an independent successful run.
   Analyzer views must apply their own explicit stage and parameter-quality rules.

Deliverables: reviewed stage metadata, corrected actual settings where recoverable,
recomputed supported summaries, and field-level before/after evidence. Unrecoverable
parameters stay unknown and are excluded from analyses that require them.

## Step 4 Recover real measurements missing from the masters

1. Refresh the source inventory and confirm scope for each family: embedded
   validation, standalone validation, calibration search trials, and balance traces.
   Preserve all evidence; explicitly decide how genuinely real calibration trials
   belong in the master/analyzer model before importing them as validation results.
2. Start with existing recovery candidates rather than repeating the whole search:
   the October 16 proposal had 2,497 candidate measurements without an active-master
   association; the newer run audit had 2,447 supported-real raw occurrences without
   an association. These are different snapshots and units, not additive totals or
   counts of definitely missing measurements.
3. Reconcile each candidate against the cleaned masters and archive using stable
   measurement IDs, exact logged transfer identity, or sufficiently specific raw
   identity. Compare row content and numeric formatting. Never deduplicate different
   measurements just because their volume and measured value happen to match.
4. Account for copied raw files, interrupted checkpoints, completed measurements
   whose workflow later failed, and cases where one master append succeeded while
   the other failed. Track missing measurements separately from missing summaries.
5. Confirm actual execution provenance before import. Preserve the original
   measurement time, run/session link, liquid, density, source/destination, stage,
   available actual parameters, and quality outcome. Missing fields stay unknown.
6. Stage measurement and summary additions together. Derive summaries only where
   completion and selected-stage evidence support them; partial genuine measurements
   remain recoverable without pretending a session completed successfully.
7. Validate identity uniqueness, units, counts, raw-to-master numerical agreement,
   and analyzer inclusion. Re-running the same import must add zero duplicate rows.

Deliverable: an evidence-backed import manifest showing which real measurements
were added, which summaries were reconstructed, and which candidates remain blocked.

## Implementation checkpoints

| Checkpoint | Reviewable result |
| --- | --- |
| 1 Prepare cleanup | Verified archive, current identity reconciliation, and exact exclusion/retention manifests. |
| 2 Activate cleanup | Smaller masters, working confirmed-real analyzer input, append compatibility, and rollback verification. |
| 3 Resolve unclear sessions | Evidence-backed promotions/exclusions with remaining blockers. |
| 4 Repair real metadata | Stage and parameter corrections with field-level lineage and independently checked summaries. |
| 5 Import missing real data | Deduplicated, repeatable recovery import with complete conservation/accounting. |

Keep each checkpoint independently reviewable. The user authorized archiving the
two original masters and replacing them with versions excluding only confirmed
simulation, subsequently excluding the remaining unclear sessions and authorizing
the real metadata repair. Those steps are complete. Missing-record import and
GUI/analyzer changes remain pending.
During future Git merges, reconcile new recordings and historical metadata
deliberately: the October 20 merge retained an unannotated data version and lost
earlier stage reconstruction while the documentation still described that work.

## Evidence locations

- [Full run audit](../recovery/pipetting_20261021_run_audit/REPORT.txt).
- [Deeper unresolved review](../recovery/pipetting_20261021_unresolved_review/summary.json).
- [Earlier work and Git recovery inventory](../recovery/pipetting_prior_work_review_20261021/REPORT.txt).
- [Historical metadata candidates](../recovery/pipetting_prior_work_review_20261021/measurements_historical_metadata_candidates.csv).
- [Historical summary candidates](../recovery/pipetting_prior_work_review_20261021/sessions_historical_metadata_candidates.csv).
- [Prior parameter discrepancies](../recovery/pipetting_20261016_audit/parameter_discrepancies.csv).
- [Verified output relocations](../output/_organization/moves_2026-10-16.csv).

Original sources and the prior audits remain preserved throughout the work.

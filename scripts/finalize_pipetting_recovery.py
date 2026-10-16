"""Verify conservation and create a deduplicated, review-only real-data candidate."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil

from audit_pipetting_recovery import PARAMS, digest, dump, parse_time, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.audit.resolve()
    if not out.is_relative_to(root) or not (out / 'summary.json').exists():
        raise ValueError('Expected generated audit directory within workspace')
    rows = read(out / 'all_source_records.csv')
    by_id = {r['record_id']: r for r in rows}
    assert len(rows) == len(by_id), 'Duplicate source record IDs'
    # Revalidate provenance before publication, including preliminary audits
    # created before the stricter concurrent-test rule was applied.
    for r in rows:
        if r['evidence_status'].startswith('time_log_'):
            traces = json.loads(r.get('balance_trace_evidence_json', '[]'))
            physical = any(t.get('exists') and t.get('nonzero')
                           and t.get('phases', {}).get('baseline_pre', 0)
                           and t.get('phases', {}).get('baseline_post', 0) for t in traces)
            r['evidence_status'] = ('balance_trace_supported_real' if physical
                                    else 'unresolved')
            r['unresolved_reason'] = '' if physical else 'Time overlap alone cannot identify a run; unrelated tests can run concurrently'
    corroboration = collections.defaultdict(list)
    for r in rows:
        if r['origin'] == 'raw' and r['evidence_status'].endswith('_real'):
            ids = r['matching_master_record_ids'].split(';') if r['matching_master_record_ids'] else []
            if len(ids) == 1:
                corroboration[ids[0]].append(r['record_id'])
    for r in rows:
        if r['evidence_status'] == 'matching_raw_supported_real' and not corroboration[r['record_id']]:
            r['evidence_status'] = 'unresolved'
            r['unresolved_reason'] = 'Raw corroboration lacks run-specific or physical balance evidence'
    fields = list(rows[0])
    write(out / 'all_source_records.csv', rows)
    for kind in ('measurement', 'session_summary'):
        subset = [r for r in rows if r['record_type'] == kind]
        write(out / f'{kind}s_log_supported_real.csv', [r for r in subset if r['evidence_status'].endswith('_real')], fields)
        write(out / f'{kind}s_unresolved_preserve.csv', [r for r in subset if not r['evidence_status'].endswith(('_real', '_simulation'))], fields)
        write(out / f'{kind}s_simulation_evidence.csv', [r for r in subset if r['evidence_status'].endswith('_simulation')], fields)
    missing = [r for r in rows if r['origin'] == 'raw' and r['record_type'] == 'measurement'
               and not r['matching_master_record_ids'] and not r['evidence_status'].endswith('_simulation')]
    quarantine = [r for r in rows if r['origin'] == 'quarantine' and not r['evidence_status'].endswith('_simulation')]
    write(out / 'missing_from_master_recovery_candidates.csv', missing, fields)
    write(out / 'quarantine_reconsideration.csv', quarantine, fields)
    by_source = collections.defaultdict(dict)
    for r in rows:
        assert int(r['source_row']) not in by_source[r['source_file']]
        by_source[r['source_file']][int(r['source_row'])] = json.loads(r['original_record_json'])
    for path, recorded in by_source.items():
        original = read(root / path)
        assert len(original) == len(recorded), path
        for number, source_row in enumerate(original, 2):
            assert recorded[number] == source_row, (path, number)
    for manifest_row in read(out / 'source_manifest.csv'):
        source = root / manifest_row['path']
        assert digest(source) == manifest_row['sha256'], manifest_row['path']
        snapshot = out / 'source_snapshots/all_csv_sources' / manifest_row['path']
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            shutil.copy2(source, snapshot)
        assert digest(snapshot) == manifest_row['sha256'], str(snapshot)
    reference = []
    for name in ('embedded_calibration_validation.py', 'validation_decisions.py', 'parameter_corrections.csv'):
        source = root / 'pipetting_data' / name
        snapshot = out / 'source_snapshots/reference_code_and_corrections' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            shutil.copy2(source, snapshot)
        reference.append(dict(path=source.relative_to(root).as_posix(), sha256=digest(snapshot)))
    write(out / 'reference_snapshot_manifest.csv', reference)
    assert not any(r['evidence_status'].startswith('time_log_') for r in rows), 'Time-only provenance must remain unresolved'
    for kind in ('measurement', 'session_summary'):
        partition = []
        for suffix in ('log_supported_real', 'unresolved_preserve', 'simulation_evidence'):
            partition.extend(read(out / f'{kind}s_{suffix}.csv'))
        expected = {r['record_id'] for r in rows if r['record_type'] == kind}
        assert len(partition) == len(expected)
        assert {r['record_id'] for r in partition} == expected

    # Union only demonstrable identity links; ambiguous repeat readings remain
    # separate. The source occurrence ledger is never deduplicated.
    real = {r['record_id']: r for r in rows if r['record_type'] == 'measurement'
            and r['evidence_status'].endswith('_real')}
    parent = {key: key for key in real}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(keys):
        keys = [key for key in keys if key in real]
        if len(keys) > 1:
            first = find(keys[0])
            for key in keys[1:]:
                parent[find(key)] = first

    copies = collections.defaultdict(list)
    operations = collections.defaultdict(list)
    for r in real.values():
        if r['origin'] == 'raw':
            # Timestamped exact content copies are safe to identify. Untimed
            # identical readings could be independent measurements.
            if r['timestamp']:
                copies[r['original_record_json']].append(r['record_id'])
            evidence = json.loads(r['parameter_log_evidence_json'])
            if len(evidence) == 1:
                e = evidence[0]
                operations[(e['log'], e['transfer_start_line'])].append(r['record_id'])
            ids = r['matching_master_record_ids'].split(';') if r['matching_master_record_ids'] else []
            origins = collections.Counter(by_id[k]['origin'] for k in ids)
            if ids and all(v == 1 for v in origins.values()):
                union([r['record_id']] + ids)
    for keys in list(copies.values()) + list(operations.values()):
        union(keys)
    groups = collections.defaultdict(list)
    for key, r in real.items():
        groups[find(key)].append(r)

    recovered = []
    for group in groups.values():
        # Prefer timestamped raw source over a session-start master timestamp.
        primary = min(group, key=lambda r: (r['origin'] != 'raw', not bool(json.loads(r['recovered_log_parameters_json'])), r['record_id']))
        original = json.loads(primary['original_record_json'])
        ids = sorted(r['record_id'] for r in group)
        parsed = parse_time(primary['timestamp'])
        item = dict(recovery_id=hashlib.sha256(';'.join(ids).encode()).hexdigest()[:20],
                    session_id=primary['linked_session_id'], measured_at=parsed.isoformat() if parsed else primary['timestamp'],
                    timestamp_kind='measurement' if primary['origin'] == 'raw' else 'session_start',
                    liquid_type=primary['liquid_type'], target_volume_ml=primary['target_volume_ml'],
                    measured_volume_ml=primary['measured_volume_ml'], mass_difference_g=primary['mass_difference_g'],
                    density_used=original.get('density_used', ''), replicate_number=primary['replicate'],
                    source_vial=original.get('source_vial', ''), destination_vial=original.get('destination_vial', ''),
                    optimization_stage=original.get('optimization_stage', '') if primary['origin'] == 'raw' else '',
                    measurement_acceptable=original.get('measurement_acceptable', ''),
                    evidence_statuses=';'.join(sorted({r['evidence_status'] for r in group})),
                    source_occurrences=len(group), source_record_ids_json=dump(ids),
                    source_locations_json=dump([dict(file=r['source_file'], row=r['source_row']) for r in group]),
                    also_in_active_master=any(r['origin'] == 'active_master' for r in group),
                    logged_parameter_fields='', recorded_raw_parameter_fields='', parameter_conflicts_json='{}')
        conflicts, logged, raw_fields = {}, [], []
        for key in PARAMS:
            observed = []
            for r in group:
                p = json.loads(r['recovered_log_parameters_json'])
                if key in p and p[key] is not None:
                    observed.append((p[key], 'logged'))
            if not observed:
                for r in group:
                    if r['origin'] == 'raw':
                        p = json.loads(r['recorded_parameters_json'])
                        if key in p and p[key] is not None:
                            observed.append((p[key], 'raw'))
            # Historical master parameter vectors remain in the ledger but never
            # silently become trusted values in the proposed recovery database.
            normalized = {}
            for value, provenance in observed:
                try:
                    normalized[round(float(value), 12)] = (value, provenance)
                except (ValueError, TypeError):
                    normalized[str(value)] = (value, provenance)
            item[key] = ''
            if len(normalized) == 1:
                value, provenance = next(iter(normalized.values()))
                item[key] = value
                (logged if provenance == 'logged' else raw_fields).append(key)
            elif len(normalized) > 1:
                conflicts[key] = [v for v, _ in observed]
        item['logged_parameter_fields'] = ';'.join(logged)
        item['recorded_raw_parameter_fields'] = ';'.join(raw_fields)
        item['parameter_conflicts_json'] = dump(conflicts)
        recovered.append(item)
    recovered.sort(key=lambda r: (r['measured_at'], r['recovery_id']))
    write(out / 'proposed_real_measurements_REVIEW.csv', recovered)
    supported_ids = set(real)
    represented = {k for r in recovered for k in json.loads(r['source_record_ids_json'])}
    assert represented == supported_ids
    assert sum(r['source_occurrences'] for r in recovered) == len(real)

    overview = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    counts = collections.Counter((r['record_type'], r['origin'], r['evidence_status']) for r in rows)
    overview['classification_counts'] = [dict(record_type=a, origin=b, evidence_status=c, rows=n)
                                         for (a, b, c), n in sorted(counts.items())]
    write(out / 'classification_counts.csv', overview['classification_counts'])
    overview.update(missing_master_candidates=len(missing),
                    missing_master_supported_real=sum(r['evidence_status'].endswith('_real') for r in missing),
                    quarantine_reconsideration_occurrences=len(quarantine),
                    quarantine_supported_real=sum(r['evidence_status'].endswith('_real') for r in quarantine))
    overview['verification'] = dict(source_rows_preserved_exactly=len(rows), source_files_unchanged=True,
                                    classification_partitions_complete=True, time_only_classifications_rejected=True,
                                    real_candidate_occurrences_preserved=len(real))
    overview['proposed_real_measurements'] = len(recovered)
    overview['proposed_real_measurements_with_logged_parameters'] = sum(bool(r['logged_parameter_fields']) for r in recovered)
    overview['proposed_real_measurements_with_raw_parameters_only'] = sum(not r['logged_parameter_fields'] and bool(r['recorded_raw_parameter_fields']) for r in recovered)
    overview['proposed_real_measurements_without_supported_parameters'] = sum(not r['logged_parameter_fields'] and not r['recorded_raw_parameter_fields'] for r in recovered)
    overview['proposed_real_measurements_not_linked_to_active_master'] = sum(not r['also_in_active_master'] for r in recovered)
    overview['supported_real_raw_months'] = dict(collections.Counter(r['timestamp'][:7] for r in real.values() if r['origin'] == 'raw'))
    (out / 'summary.json').write_text(json.dumps(overview, indent=2), encoding='utf-8')
    lines = [
        'PIPETTING DATA RECOVERY AUDIT — 2026-10-16', '',
        'Original production code and data were not modified. No source row was discarded.',
        f'{len(rows):,} source occurrences were preserved exactly from {overview["source_files"]} CSV files.',
        f'{overview["logs"]:,} logs were indexed, including changes after launch confirmation.', '',
        'Files to review:',
        'proposed_real_measurements_REVIEW.csv — deduplicated measurements supported by run-specific log or balance evidence.',
        'measurements_unresolved_preserve.csv — retained measurements whose provenance is unresolved or conflicting.',
        'missing_from_master_recovery_candidates.csv — raw source occurrences without a master association; contains possible duplicates.',
        'quarantine_reconsideration.csv — previously quarantined occurrences without sufficient simulation proof.',
        'parameter_discrepancies.csv — direct comparisons of master settings with uniquely linked logged transfer settings.',
        'all_source_records.csv — complete occurrence ledger, original row JSON, file hashes and source row numbers.',
        'source_snapshots/ — byte-verified copies of all source CSVs, plus reference code and correction history.',
        'summary.json and classification_counts.csv — totals and validation results.', '',
        'Classification rules:',
        'A filename, balance precision, fast execution, old quarantine membership, or time overlap alone is insufficient.',
        'Path-supported classifications require explicit simulation state in a contemporaneous log.',
        'Balance-supported classifications require a unique matching transfer plus an existing nonzero pre/post balance trace.',
        'Time-only log matches stay unresolved, including possible concurrent test fixtures.',
        'Simulation-evidence files are for audit completeness and are excluded from the proposed real dataset.', '',
        'Parameter rules:',
        'Unknown values remain blank. Current calibration files and default values are never used to fill historical gaps.',
        'Logged parameters are restricted to a uniquely linked single-dispense transfer.',
        'Explicit raw parameters are distinguishable from log-corroborated parameters.',
        'Historical master parameters remain available in the ledger but are unverified and not promoted automatically.',
        'Master-only optimization stages and accepted flags are not inferred as historical facts.', '',
        'Identity and limits:',
        'The candidate merges only timestamped identical raw copies, the same logged transfer, or unique session/volume/replicate/result matches.',
        'Ambiguous repeated readings remain separate. Candidate counts can therefore include unresolved duplicate identities.',
        'Not linked to an active master does not necessarily mean absent: some historical folder/session links are missing.',
        'These are review candidates, not a replacement production database. Session completeness and final selected stages remain to be reconciled.',
        'Measurements from unresolved sources remain preserved outside the proposed real dataset.', '',
        'Reproduce with Python standard library:',
        'python -B scripts/audit_pipetting_recovery.py --output <new-audit-directory>',
        'python -B scripts/enrich_pipetting_recovery.py --audit <new-audit-directory>',
        'python -B scripts/finalize_pipetting_recovery.py --audit <new-audit-directory>', '',
        json.dumps(overview, indent=2),
    ]
    (out / 'REPORT.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: v for k, v in overview.items() if k.startswith('proposed_') or k == 'verification'}, indent=2))


if __name__ == '__main__':
    main()

"""Link recovery candidates to exact logged transfers and balance trace files.

Only writes the generated audit directory. Never edits source CSVs or logs.
Run after audit_pipetting_recovery.py. Unknown parameters are never defaulted.
"""
from __future__ import annotations

import argparse
import ast
import collections
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import re

from audit_pipetting_recovery import (PARAMS, STAMP, TRANSFER, PARAM, digest, dump,
                                     parse_time, read, write)

CALIBRATED = re.compile(r'Using calibrated (\w+)=(\S+) for ')
MASS = re.compile(r'Mass difference: ([-+\d.eE]+)g')
SAVED = re.compile(r'Saved (\d+) mass measurements to (output[\\/].+\.csv)')


def transfers(path):
    blocks = []
    current = None
    with path.open('rb') as handle:
        for number, raw in enumerate(handle, 1):
            if not STAMP.match(raw):
                continue
            if not any(token in raw for token in
                       (b'Dispensing ', b'Applied ', b'Using calibrated ', b'Mass difference:', b'mass measurements to')):
                continue
            line = raw.decode('utf-8', errors='replace').strip()
            at = parse_time(line[:23])
            match = TRANSFER.search(line)
            if match:
                if current:
                    current['end'] = at
                current = dict(start=at, end=None, line=number, volume=float(match.group(1)),
                               source=match.group(2), destination=match.group(3), masses=[],
                               traces=[], parameters={}, parameter_lines={}, mass_at=None)
                blocks.append(current)
            if current is None:
                continue
            match = PARAM.search(line) or CALIBRATED.search(line)
            if match and match.group(1) in PARAMS:
                key, value = match.groups()
                try:
                    value = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    continue
                current['parameters'][key] = value
                current['parameter_lines'][key] = number
            match = MASS.search(line)
            if match:
                current['masses'].append(float(match.group(1)))
                current['mass_at'] = at
            match = SAVED.search(line)
            if match:
                current['traces'].append(dict(path=match.group(2).replace('\\', '/'), line=number,
                                              recorded_readings=int(match.group(1))))
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.audit.resolve()
    if not out.is_relative_to(root) or not (out / 'summary.json').exists():
        raise ValueError('Expected an existing generated audit directory in the workspace')
    rows = read(out / 'all_source_records.csv')
    logs = {l['path']: l for l in json.loads((out / 'log_evidence_index.json').read_text(encoding='utf-8'))}
    raw_by_log = collections.defaultdict(list)
    for row in rows:
        row['balance_trace_evidence_json'] = '[]'
        row['parameter_difference_from_master_json'] = '{}'
        row['pre_enrichment_status'] = row['evidence_status']
        evidence = json.loads(row['log_evidence_json'])
        when = parse_time(row['timestamp'])
        # Test fixtures can be created while a real robot log is active. Time
        # overlap alone establishes neither real nor simulated provenance.
        if row['evidence_status'].startswith('time_log_'):
            row['evidence_status'] = 'unresolved'
            row['unresolved_reason'] = 'Time overlap alone cannot identify a run; unrelated tests can run concurrently'
        # For exact raw timestamps never use a mode change later in the same second.
        if row['origin'] == 'raw' and when:
            for item in evidence:
                applicable = [m for m in logs[item['log']]['modes'] if parse_time(m['at']) <= when]
                item['mode'] = applicable[-1] if applicable else None
            known = {item['mode']['simulate'] for item in evidence if item['mode']}
            if len(known) > 1:
                row['evidence_status'] = 'conflicting'
                row['unresolved_reason'] = 'Relevant logs disagree at exact raw timestamp'
            elif not evidence or any(item['mode'] is None for item in evidence):
                row['evidence_status'] = 'unresolved'
                row['unresolved_reason'] = 'No explicit mode evidence at exact raw timestamp'
            elif evidence and all(item['path_links'] for item in evidence):
                prefix = 'path_log' if any(item['path_links'] for item in evidence) else 'time_log'
                row['evidence_status'] = prefix + ('_simulation' if next(iter(known)) else '_real')
                row['unresolved_reason'] = ''
            row['log_evidence_json'] = dump(evidence)
        if row['origin'] == 'raw' and row['record_type'] == 'measurement' and row['mass_difference_g']:
            for item in evidence:
                raw_by_log[item['log']].append(row)

    matches = collections.defaultdict(list)
    print(f'Checking exact transfer and parameter evidence in {len(raw_by_log)} logs', flush=True)
    for index, (log_path, raw_rows) in enumerate(sorted(raw_by_log.items()), 1):
        log_file = root / log_path
        if digest(log_file) != logs[log_path]['sha256']:
            raise RuntimeError(f'Log changed since initial audit: {log_path}')
        blocks = transfers(log_file)
        for row in raw_rows:
            when = parse_time(row['timestamp'])
            if not when:
                continue
            original = json.loads(row['original_record_json'])
            for block in blocks:
                if not block['masses'] or not block['mass_at']:
                    continue
                if not (block['mass_at'] <= when <= (block['end'] or block['mass_at'] + timedelta(minutes=5))):
                    continue
                if abs(block['volume'] - float(row['target_volume_ml'])) > .00050001:
                    continue
                if original.get('source_vial') and original['source_vial'] != block['source']:
                    continue
                if original.get('destination_vial') and original['destination_vial'] != block['destination']:
                    continue
                if abs(sum(block['masses']) - float(row['mass_difference_g'])) > len(block['masses']) * .00000051:
                    continue
                matches[row['record_id']].append((log_path, block))
        if index % 100 == 0:
            print(f'Checked {index}/{len(raw_by_log)} logs', flush=True)

    trace_manifest = {}
    for row in rows:
        linked = matches.get(row['record_id'], [])
        if len(linked) != 1:
            continue
        log_path, block = linked[0]
        # Restrict recovered parameter values to a single-dispense transfer.
        # Split transfers need per-part parameter sequences, not one flat snapshot.
        if len(block['masses']) == 1:
            row['recovered_log_parameters_json'] = dump(block['parameters'])
            row['parameter_log_evidence_json'] = dump([dict(log=log_path, transfer_start_line=block['line'],
                                                          parameter_lines=block['parameter_lines'],
                                                          linkage='unique timestamp, target, source/destination and returned mass')])
            row['parameter_trust'] = 'observed_in_uniquely_linked_transfer; unlogged fields unknown'
        traces = []
        for evidence in block['traces']:
            path = root / evidence['path']
            if not path.exists():
                traces.append(dict(**evidence, exists=False))
                continue
            if evidence['path'] not in trace_manifest:
                readings = read(path)
                phases = collections.Counter(r.get('phase') for r in readings)
                numeric = [float(r['mass_g']) for r in readings if r.get('mass_g')]
                trace_manifest[evidence['path']] = dict(path=evidence['path'], sha256=digest(path),
                    readings=len(readings), phases=dict(phases),
                    nonzero=any(abs(m) > 0 for m in numeric))
            info = trace_manifest[evidence['path']]
            traces.append(dict(**evidence, exists=True, sha256=info['sha256'], readings=info['readings'],
                               nonzero=info['nonzero'], phases=info['phases']))
        row['balance_trace_evidence_json'] = dump(traces)
        physical = any(t.get('nonzero') and t.get('phases', {}).get('baseline_pre', 0) > 0
                       and t.get('phases', {}).get('baseline_post', 0) > 0 for t in traces)
        if physical:
            if row['evidence_status'].endswith('_simulation'):
                row['evidence_status'] = 'conflicting'
                row['unresolved_reason'] = 'Explicit simulation mode conflicts with linked nonzero dispense balance trace'
            elif row['evidence_status'] == 'unresolved':
                row['evidence_status'] = 'balance_trace_supported_real'
                row['unresolved_reason'] = ''

    by_id = {r['record_id']: r for r in rows}
    for row in rows:
        if row['origin'] != 'raw':
            continue
        ids = row['matching_master_record_ids'].split(';') if row['matching_master_record_ids'] else []
        if len(ids) != 1:
            continue
        master = by_id[ids[0]]
        recovered = json.loads(row['recovered_log_parameters_json'])
        old = json.loads(master['recorded_parameters_json'])
        differences = {}
        for key, value in recovered.items():
            if key not in old or value is None:
                continue
            try:
                differs = abs(float(value) - float(old[key])) > 1e-10
            except (TypeError, ValueError):
                differs = str(value) != str(old[key])
            if differs:
                differences[key] = dict(master=old[key], logged=value)
        row['parameter_difference_from_master_json'] = dump(differences)
        # A unique raw/master value match carries corroboration, never replacement.
        if row['evidence_status'].endswith('_real'):
            if master['evidence_status'].endswith('_simulation'):
                master['evidence_status'] = 'conflicting'
                master['unresolved_reason'] = 'Uniquely matching raw measurement is supported as real'
            elif master['evidence_status'] == 'unresolved':
                master['evidence_status'] = 'matching_raw_supported_real'
                master['unresolved_reason'] = ''
            master['balance_trace_evidence_json'] = row['balance_trace_evidence_json']
            # Do not assign a single parameter vector if several raw occurrences
            # disagree. The source observations remain inspectable in the ledger.
            previous = json.loads(master['recovered_log_parameters_json'])
            if previous and previous != recovered:
                master['parameter_trust'] = 'conflicting_raw_parameter_observations'
            elif recovered and master['parameter_trust'] != 'conflicting_raw_parameter_observations':
                master['recovered_log_parameters_json'] = dump(recovered)
                master['parameter_log_evidence_json'] = row['parameter_log_evidence_json']
                master['parameter_difference_from_master_json'] = dump(differences)
                master['parameter_trust'] = 'uniquely_matching_raw_and_logged_transfer; unlogged fields unknown'

    counts = collections.Counter((r['record_type'], r['origin'], r['evidence_status']) for r in rows)
    write(out / 'all_source_records.csv', rows)
    fields = list(rows[0])
    for kind in ('measurement', 'session_summary'):
        subset = [r for r in rows if r['record_type'] == kind]
        write(out / f'{kind}s_log_supported_real.csv', [r for r in subset if r['evidence_status'].endswith('_real')], fields)
        write(out / f'{kind}s_unresolved_preserve.csv', [r for r in subset if not r['evidence_status'].endswith(('_real', '_simulation'))], fields)
        write(out / f'{kind}s_simulation_evidence.csv', [r for r in subset if r['evidence_status'].endswith('_simulation')], fields)
    candidates = [r for r in rows if r['record_type'] == 'measurement' and r['origin'] == 'raw'
                  and not r['matching_master_record_ids'] and not r['evidence_status'].endswith('_simulation')]
    quarantined = [r for r in rows if r['origin'] == 'quarantine' and not r['evidence_status'].endswith('_simulation')]
    write(out / 'missing_from_master_recovery_candidates.csv', candidates, fields)
    write(out / 'quarantine_reconsideration.csv', quarantined, fields)
    differences = [r for r in rows if r['parameter_difference_from_master_json'] != '{}']
    write(out / 'parameter_discrepancies.csv', differences, fields)
    write(out / 'balance_trace_manifest.csv', [dict(path=p, sha256=i['sha256'], readings=i['readings'],
                                                   phases_json=dump(i['phases']), nonzero=i['nonzero'])
                                               for p, i in sorted(trace_manifest.items())])
    write(out / 'classification_counts.csv', [dict(record_type=a, origin=b, evidence_status=c, rows=n)
                                              for (a, b, c), n in sorted(counts.items())])
    manifest = read(out / 'source_manifest.csv')
    changed = [m['path'] for m in manifest if digest(root / m['path']) != m['sha256']]
    overview = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    overview.update(source_files_changed_during_audit=changed,
                    uniquely_linked_raw_transfer_rows=sum(len(matches.get(r['record_id'], [])) == 1 for r in rows),
                    preserved_balance_trace_files=len(trace_manifest),
                    parameter_discrepancy_occurrences=len(differences),
                    missing_master_candidates=len(candidates),
                    missing_master_supported_real=sum(r['evidence_status'].endswith('_real') for r in candidates),
                    quarantine_reconsideration_occurrences=len(quarantined),
                    quarantine_supported_real=sum(r['evidence_status'].endswith('_real') for r in quarantined),
                    classification_counts=[dict(record_type=a, origin=b, evidence_status=c, rows=n)
                                           for (a, b, c), n in sorted(counts.items())])
    (out / 'summary.json').write_text(json.dumps(overview, indent=2), encoding='utf-8')
    print(json.dumps(overview, indent=2), flush=True)
    if changed:
        raise RuntimeError('Original sources changed; review summary')


if __name__ == '__main__':
    main()

"""Preservation-first historical pipetting audit. Never changes source files.

The output is a review dataset, not a replacement production database. Simulation
classification requires explicit mode evidence in a contemporaneous log. Timing,
balance precision, filenames, and old quarantine decisions are not proof.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
from datetime import datetime, timedelta
import hashlib
import json
from functools import lru_cache
import os
from pathlib import Path
import re
import shutil


PARAMS = ('aspirate_speed dispense_speed retract_speed blowout_speed aspirate_wait_time '
          'dispense_wait_time post_retract_wait_time overaspirate_vol pre_asp_air_vol '
          'post_asp_air_vol blowout_vol asp_disp_cycles').split()
STAMP = re.compile(rb'^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}')
MODE = re.compile(r'(?:Confirmed )?SIMULATE mode:\s*(True|False)|'
                  r'SIMULATE flag updated:\s*(?:True|False)\s*->\s*(True|False)')
PATH = re.compile(r'output[\\/][^\s\"\'<>]+')
TRANSFER = re.compile(r'Dispensing ([\d.]+) mL from (.+) to (.+)$')
PARAM = re.compile(r'Applied (?:object|dict) param (\w+)=(.+)$')


def dump(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f'Malformed CSV: {path}')
        return rows


def write(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


@lru_cache(maxsize=200000)
def parse_time(value):
    if not value:
        return None
    for fmt in ('%Y%m%d_%H%M%S', '%Y-%m-%d %H:%M:%S,%f', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def norm(value):
    return value.replace('\\', '/').rstrip('/.,;:)')


def extract_params(row):
    values = {k: row[k] for k in PARAMS if row.get(k, '') != ''}
    sources = {k: 'recorded_column' for k in values}
    for column in ('parameters_used', 'protocol_result'):
        try:
            nested = ast.literal_eval(row.get(column, ''))
        except (ValueError, SyntaxError):
            continue
        if isinstance(nested, dict):
            for k in PARAMS:
                if k in nested and nested[k] is not None:
                    # Prefer the explicit parameters_used dictionary to derived columns.
                    if k not in values or column == 'parameters_used':
                        values[k] = nested[k]
                        sources[k] = column
    for k in PARAMS:
        for prefix in ('hardware_parameters_', 'metadata_'):
            if k not in values and row.get(prefix + k, '') != '':
                values[k] = row[prefix + k]
                sources[k] = prefix + k
    return values, sources


def scan_log(path):
    modes, paths = [], set()
    start = end = None
    sha = hashlib.sha256()
    # Only classification/path evidence is indexed globally. Parameter reconstruction
    # is restricted to exact transfer boundaries in a second, targeted pass.
    with path.open('rb') as handle:
        for line_number, raw in enumerate(handle, 1):
            sha.update(raw)
            if not STAMP.match(raw):
                continue
            stamp = raw[:23].decode('ascii')
            if start is None:
                start = stamp
            end = stamp
            if b'SIMULATE' in raw:
                line = raw.decode('utf-8', errors='replace').strip()
                match = MODE.search(line)
                if match:
                    modes.append(dict(at=stamp, simulate=(match.group(1) or match.group(2)) == 'True',
                                      line=line_number, text=line))
            if b'output' in raw and (b'validat' in raw.lower() or b'folder' in raw.lower()):
                paths.update(norm(p) for p in PATH.findall(raw.decode('utf-8', errors='replace')))
    return dict(path=path.as_posix(), start=start, end=end, modes=modes,
                output_paths=sorted(paths), sha256=sha.hexdigest(), size_bytes=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output.resolve()
    if not out.is_relative_to(root) or out == root:
        raise ValueError('Audit output must be a new directory within the workspace')
    out.mkdir(parents=True, exist_ok=False)
    snapshots = out / 'source_snapshots'
    snapshots.mkdir()
    manifest, errors = [], []
    originals = [root / 'pipetting_data' / n for n in
                 ('master_pipetting_measurements.csv', 'master_pipetting_sessions.csv')]
    originals += sorted((root / 'pipetting_data/simulated_quarantine').glob('*.csv'))
    for path in originals:
        sha = digest(path)
        shutil.copy2(path, snapshots / path.name)
        if digest(snapshots / path.name) != sha:
            raise ValueError(f'Snapshot verification failed: {path}')
        manifest.append(dict(path=path.relative_to(root).as_posix(), sha256=sha, size_bytes=path.stat().st_size))

    moves_file = root / 'output/_organization/moves_2026-10-16.csv'
    moves = read(moves_file) if moves_file.exists() else []
    moves.sort(key=lambda r: len(r['Source']), reverse=True)

    def relocated(value):
        value = norm(value)
        for move in moves:
            source = norm(move['Source'])
            if value == source or value.startswith(source + '/'):
                return norm(move['Destination']) + value[len(source):]
        return value

    all_logs = sorted((root / 'logs').glob('experiment_log*.log'))
    logs = []
    print(f'Indexing {len(all_logs)} logs, including mode changes after GUI confirmation.', flush=True)
    for i, path in enumerate(all_logs, 1):
        item = scan_log(path)
        item['path'] = path.relative_to(root).as_posix()
        item['output_paths'] = sorted(set(relocated(p) for p in item['output_paths']))
        logs.append(item)
        if i % 500 == 0:
            print(f'Indexed {i}/{len(all_logs)} logs', flush=True)
    (out / 'log_evidence_index.json').write_text(dump(logs), encoding='utf-8')
    by_day = collections.defaultdict(list)
    for i, log in enumerate(logs):
        if not log['start'] or not log['end']:
            continue
        start, end = parse_time(log['start']), parse_time(log['end'])
        for offset in range((end.date() - start.date()).days + 1):
            by_day[(start + timedelta(days=offset)).date()].append(i)

    session_rows = read(originals[1])
    quarantine_sessions = root / 'pipetting_data/simulated_quarantine/master_pipetting_sessions_simulated.csv'
    if quarantine_sessions.exists():
        session_rows += read(quarantine_sessions)
    session_folders = collections.defaultdict(set)
    for row in session_rows:
        if row.get('output_folder_path'):
            session_folders[row['session_id']].add(relocated(row['output_folder_path']))
    folder_sessions = collections.defaultdict(set)
    for sid, folders in session_folders.items():
        for folder in folders:
            folder_sessions[folder].add(sid)

    def classify(when, folders, explicit):
        if when is None:
            return 'unresolved', [], 'No usable event/session timestamp'
        candidates = []
        for i in by_day[when.date()]:
            log = logs[i]
            if not (parse_time(log['start']).replace(microsecond=0) <= when <= parse_time(log['end']) + timedelta(seconds=1)):
                continue
            applicable = [e for e in log['modes'] if parse_time(e['at']).replace(microsecond=0) <= when]
            mode = applicable[-1] if applicable else None
            links = []
            for folder in folders:
                for p in log['output_paths']:
                    # Generic output roots do not identify a run.
                    if len(p.split('/')) < 3 or p in ('output/calibration_validation', 'output/surfactant_grid/validation'):
                        continue
                    if folder == p or folder.startswith(p + '/') or p.startswith(folder + '/'):
                        links.append(p)
            candidates.append((i, mode, links))
        linked = [c for c in candidates if c[2]]
        selected = linked if linked else candidates
        evidence = [dict(log=logs[i]['path'], mode=mode, path_links=links) for i, mode, links in selected]
        if not selected:
            return 'unresolved', evidence, 'No contemporaneous log found'
        known = {mode['simulate'] for _, mode, _ in selected if mode is not None}
        if len(known) > 1:
            return 'conflicting', evidence, 'Contemporaneous logs disagree about simulation mode'
        if not linked:
            return 'unresolved', evidence, 'Time overlap alone cannot identify a run; unrelated tests can run concurrently'
        if any(mode is None for _, mode, _ in selected):
            return 'unresolved', evidence, 'Relevant log has no explicit mode evidence'
        # The master timestamp identifies session start. Raw timestamps identify
        # completed measurements. Both are recorded, but a time-only link stays
        # distinguishable from a path-supported association.
        mode = next(iter(known))
        if explicit in ('True', 'False') and (explicit == 'True') != mode:
            return 'conflicting', evidence, 'Recorded simulation flag contradicts log mode'
        prefix = 'path_log' if linked else 'time_log'
        return prefix + ('_simulation' if mode else '_real'), evidence, ''

    inputs = originals[:2] + [p for p in originals[2:] if p.name.startswith('master_')]
    for base in ('output', 'sdl_pipette_calibration/validation', 'temp/simulation'):
        inputs += sorted((root / base).rglob('raw_measurements.csv'))
        inputs += sorted((root / base).rglob('raw_validation_data.csv'))
    inputs = list(dict.fromkeys(inputs))
    records = []
    print(f'Reading {len(inputs)} measurement and summary files', flush=True)
    for path in inputs:
        relative = path.relative_to(root).as_posix()
        sha = digest(path)
        if not any(m['path'] == relative for m in manifest):
            manifest.append(dict(path=relative, sha256=sha, size_bytes=path.stat().st_size))
        try:
            rows = read(path)
        except Exception as exc:
            errors.append(dict(path=relative, error=str(exc)))
            continue
        master = path.name.startswith('master_')
        summary = 'sessions' in path.name
        for row_number, row in enumerate(rows, 2):
            sid = row.get('session_id', '')
            folders = session_folders.get(sid, set()) if master else {path.parent.relative_to(root).as_posix()}
            if not sid and not master:
                linked_ids = folder_sessions.get(path.parent.relative_to(root).as_posix(), set())
                if len(linked_ids) == 1:
                    sid = next(iter(linked_ids))
            when_text = row.get('measured_at') or row.get('timestamp', '')
            when = parse_time(when_text)
            classification, evidence, reason = classify(when, folders, row.get('simulate', ''))
            params, parameter_sources = extract_params(row)
            target = row.get('target_volume_ml', row.get('volume_target_ml', ''))
            measured = row.get('measured_volume_ml', row.get('volume_measured_ml', row.get('measured_mean_ml', '')))
            liquid = row.get('liquid_type', row.get('liquid', ''))
            if not liquid:
                # Context is exposed as a hint only, not silently inferred as a fact.
                liquid = ''
            origin = 'quarantine' if 'simulated_quarantine' in relative else 'active_master' if master else 'raw'
            record = dict(record_id=hashlib.sha256(f'{relative}:{row_number}'.encode()).hexdigest()[:20],
                          record_type='session_summary' if summary else 'measurement', origin=origin,
                          source_file=relative, source_row=row_number, source_sha256=sha,
                          original_session_id=row.get('session_id', ''), linked_session_id=sid,
                          timestamp=when_text, liquid_type=liquid, target_volume_ml=target,
                          measured_volume_ml=measured,
                          replicate=row.get('replicate_number', row.get('replicate', row.get('replicate_id', ''))),
                          mass_difference_g=row.get('mass_difference_g', ''),
                          evidence_status=classification, unresolved_reason=reason,
                          log_evidence_json=dump(evidence),
                          recorded_parameters_json=dump(params), parameter_sources_json=dump(parameter_sources),
                          parameter_trust='historical_master_unverified' if master else 'recorded_in_raw_unverified',
                          recovered_log_parameters_json='{}', parameter_log_evidence_json='[]',
                          original_record_json=dump(row))
            if not target or not measured:
                record['unresolved_reason'] += '; Nonstandard or missing volume fields'
            records.append(record)

    # Link raw and master rows only with a session association plus a unique
    # numeric/replicate match. Similar readings from different runs never merge.
    def match_key(r):
        try:
            return (r['linked_session_id'], round(float(r['target_volume_ml']), 9),
                    round(float(r['measured_volume_ml']), 10), str(r['replicate']))
        except (ValueError, TypeError):
            return None

    master_groups = collections.defaultdict(list)
    for r in records:
        if r['record_type'] == 'measurement' and r['origin'] != 'raw':
            master_groups[match_key(r)].append(r)
    for r in records:
        r['matching_master_record_ids'] = ''
        r['duplicate_review_group'] = ''
        if r['origin'] == 'raw' and r['linked_session_id']:
            matches = master_groups.get(match_key(r), [])
            r['matching_master_record_ids'] = ';'.join(x['record_id'] for x in matches)
            if len(matches) == 1:
                r['duplicate_review_group'] = matches[0]['record_id']
    # Exact raw content copies share a review group, but every source occurrence
    # remains in the ledger. No rows are dropped from candidate CSVs.
    raw_groups = collections.defaultdict(list)
    for r in records:
        if r['origin'] == 'raw':
            raw_groups[hashlib.sha256(r['original_record_json'].encode()).hexdigest()].append(r)
    for sha, group in raw_groups.items():
        if len(group) > 1:
            for r in group:
                if not r['duplicate_review_group']:
                    r['duplicate_review_group'] = 'identical_raw_' + sha[:20]

    # Cache the ledger before targeted parameter reconstruction, so lengthy
    # investigations are independently reviewable even if interrupted later.
    write(out / 'all_source_records.csv', records)
    write(out / 'source_manifest.csv', manifest)
    write(out / 'read_errors.csv', errors, ['path', 'error'])
    (out / 'audit_state.json').write_text(dump(dict(records=len(records), sources=len(inputs))), encoding='utf-8')

    counts = collections.Counter((r['record_type'], r['origin'], r['evidence_status']) for r in records)
    write(out / 'classification_counts.csv', [dict(record_type=a, origin=b, evidence_status=c, rows=n)
                                              for (a, b, c), n in sorted(counts.items())])
    for kind in ('measurement', 'session_summary'):
        subset = [r for r in records if r['record_type'] == kind]
        real = [r for r in subset if r['evidence_status'].endswith('_real')]
        simulation = [r for r in subset if r['evidence_status'].endswith('_simulation')]
        uncertain = [r for r in subset if not r['evidence_status'].endswith(('_real', '_simulation'))]
        write(out / f'{kind}s_log_supported_real.csv', real, list(records[0]))
        write(out / f'{kind}s_unresolved_preserve.csv', uncertain, list(records[0]))
        write(out / f'{kind}s_simulation_evidence.csv', simulation, list(records[0]))
    changed = []
    for item in manifest:
        if digest(root / item['path']) != item['sha256']:
            changed.append(item['path'])
    overview = dict(source_records=len(records), source_files=len(inputs), read_errors=errors,
                    logs=len(logs), logs_with_explicit_mode=sum(bool(l['modes']) for l in logs),
                    source_files_changed_during_audit=changed,
                    classification_counts=[dict(record_type=a, origin=b, evidence_status=c, rows=n)
                                           for (a, b, c), n in sorted(counts.items())])
    (out / 'summary.json').write_text(json.dumps(overview, indent=2), encoding='utf-8')
    print(json.dumps(overview, indent=2), flush=True)
    if changed:
        raise RuntimeError('Sources changed during audit; inspect summary before using output')


if __name__ == '__main__':
    main()

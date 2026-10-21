"""Snapshot-specific, reversible removal of confirmed simulation; no inference."""
import argparse
import collections
import contextlib
import csv
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'recovery/pipetting_20261021_unresolved_review/all_master_rows_updated_review.csv'
SPECS = {
    'master_pipetting_measurements.csv': ('measurement', '657f8844dd18d5972258844e6e95ae59e42348b4988797567d9338fcde4f6771', 22190, 12758),
    'master_pipetting_sessions.csv': ('session_summary', 'e2ca5ca30409bcfda30e168ee74a1084b62d99a182f189289e57312fcbebd231', 3370, 1517),
}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def parse(data):
    # Preserve byte-exact CSV records, including quoting/newlines and empty fields.
    lines = data.splitlines(keepends=True)
    reader = csv.reader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    header = next(reader)
    header_end = reader.line_num
    end = header_end
    records = []
    for values in reader:
        start, end = end, reader.line_num
        require(len(values) == len(header), 'Unexpected CSV width/blank record')
        records.append((dict(zip(header, values)), b''.join(lines[start:end])))
    require(end == len(lines), 'Unparsed trailing content')
    return b''.join(lines[:header_end]), records

def save_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def prepare():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    archive = ROOT / 'pipetting_data/archives' / ('confirmed_simulation_cleanup_' + stamp)
    archive.mkdir(parents=True, exist_ok=False)
    (archive / 'staged').mkdir()
    shutil.copy2(AUDIT, archive / 'classification_snapshot.csv')
    shutil.copy2(__file__, archive / 'cleanup.py')
    with AUDIT.open(newline='', encoding='utf-8-sig') as handle:
        audit = list(csv.DictReader(handle))
    require(len(audit) == 25560, 'Unexpected audit size')
    manifest = {'status': 'prepared', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'classification_sha256': sha(AUDIT.read_bytes()), 'files': {},
                'scope': 'Only confirmed_simulated removed; unresolved retained; schema and retained record bytes unchanged.'}
    fields = ['file', 'archived_source_sha256', 'original_source_row', 'retained_source_row',
              'record_id', 'classification', 'session_id', 'timestamp', 'reason', 'linked_logs']
    with (archive / 'row_manifest.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for name, (kind, expected_sha, expected_rows, removed_count) in SPECS.items():
            source = ROOT / 'pipetting_data' / name
            data = source.read_bytes()
            require(sha(data) == expected_sha, f'Source snapshot changed: {name}')
            (archive / name).write_bytes(data)
            require(sha((archive / name).read_bytes()) == expected_sha, 'Archive hash mismatch')
            header, records = parse(data)
            subset = [r for r in audit if r['record_type'] == kind]
            require(len(subset) == len(records) == expected_rows, 'Row count mismatch')
            by_row = {int(r['source_row']): r for r in subset}
            require(len(by_row) == expected_rows, 'Duplicate audit source rows')
            kept = []
            counts = collections.Counter()
            by_date = collections.Counter()
            for number, (row, raw) in enumerate(records, 2):
                evidence = by_row[number]
                require(evidence['source_sha256'] == expected_sha, 'Audit source hash mismatch')
                require(evidence['source_file'].replace('\\', '/') == 'pipetting_data/' + name,
                        'Wrong audit source file')
                require(json.loads(evidence['original_record_json']) == row,
                        f'Row identity mismatch: {name}:{number}')
                status = evidence['classification']
                require(status in ('confirmed_real', 'confirmed_simulated', 'unresolved'), 'Unknown classification')
                counts[status] += 1
                by_date[(row['timestamp'][:8], status)] += 1
                retained_row = ''
                if status != 'confirmed_simulated':
                    kept.append(raw)
                    retained_row = len(kept) + 1
                writer.writerow(dict(zip(fields, [name, expected_sha, number, retained_row,
                    evidence['record_id'], status, row['session_id'], row['timestamp'],
                    evidence['reason'], evidence['linked_logs']])))
            require(counts['confirmed_simulated'] == removed_count, 'Exclusion count mismatch')
            staged = header + b''.join(kept)
            (archive / 'staged' / name).write_bytes(staged)
            require(len(parse(staged)[1]) == len(kept), 'Staged parsing mismatch')
            require(len(kept) + removed_count == len(records), 'Row conservation failure')
            manifest['files'][name] = {'original_sha256': expected_sha, 'cleaned_sha256': sha(staged),
                'original_bytes': len(data), 'cleaned_bytes': len(staged),
                'original_rows': len(records), 'retained_rows': len(kept), 'classifications': dict(counts),
                'oct19_real_retained': by_date[('20261019', 'confirmed_real')],
                'oct14_simulated_removed': by_date[('20261014', 'confirmed_simulated')]}
    manifest['row_manifest_sha256'] = sha((archive / 'row_manifest.csv').read_bytes())
    save_json(archive / 'manifest.json', manifest)
    (archive / 'README.txt').write_text(
        'Originals are the two CSVs at this directory root. staged/ contains the cleaned pair.\n'
        'row_manifest.csv maps every original occurrence to its classification and retained row.\n'
        'classification_snapshot.csv preserves full evidence, including original record JSON.\n'
        'To roll back: pause master writers, verify original hashes in manifest.json, archive the\n'
        'then-current active pair first (new recordings may exist), and restore BOTH original CSVs.\n'
        'If manifest status is activation_in_progress, verify both active hashes before resuming.\n'
        'No stage/overaspirate edits or analyzer code changes are included. Unresolved rows remain.\n', encoding='utf-8')
    print(json.dumps({'archive': str(archive), **manifest}, indent=2))

@contextlib.contextmanager
def exclusive(path):
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                       ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    create.restype = ctypes.c_void_p
    handle = create(str(path), 0xC0000000, 0, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    fd = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
    with os.fdopen(fd, 'r+b') as stream:
        yield stream

def overwrite(stream, data):
    stream.seek(0)
    stream.write(data)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())
    stream.seek(0)
    require(stream.read() == data, 'Written bytes do not match staged bytes')

def activate(archive):
    archive = Path(archive).resolve()
    require(archive.parent == ROOT / 'pipetting_data/archives', 'Archive must be in expected directory')
    manifest = json.loads((archive / 'manifest.json').read_text(encoding='utf-8'))
    require(manifest['status'] == 'prepared', 'Archive already activated or requires manual recovery')
    # Fail closed if an experiment runner or an unfamiliar Python application exists.
    import psutil
    for process in psutil.process_iter(['name']):
        if process.pid == os.getpid() or (process.info['name'] or '').lower() not in ('python.exe', 'pythonw.exe'):
            continue
        command = process.cmdline()
        allowed = any(arg.replace('\\', '/').lower().endswith(suffix) for arg in command for suffix in
                      ('/lab_slack_agent/app.py', '/m5stack/mqtt_server_log.py'))
        require(allowed, f'Another Python process is running (PID {process.pid}); no activation performed')
    names = list(SPECS)
    originals, cleaned = {}, {}
    for name in names:
        originals[name] = (archive / name).read_bytes()
        cleaned[name] = (archive / 'staged' / name).read_bytes()
        require(sha(originals[name]) == manifest['files'][name]['original_sha256'], 'Archive changed')
        require(sha(cleaned[name]) == manifest['files'][name]['cleaned_sha256'], 'Staged file changed')
    # Exclusive Windows handles block ordinary opens, writes, and os.replace for both paths.
    # A hard process termination is recoverable from the persisted activation state + originals.
    with contextlib.ExitStack() as stack:
        streams = {name: stack.enter_context(exclusive(ROOT / 'pipetting_data' / name)) for name in names}
        for name, stream in streams.items():
            require(sha(stream.read()) == manifest['files'][name]['original_sha256'], 'Active master changed; rebuild required')
        manifest['status'] = 'activation_in_progress'
        save_json(archive / 'manifest.json', manifest)
        try:
            for name in names:
                overwrite(streams[name], cleaned[name])
            manifest['status'] = 'activated'
            manifest['activated_utc'] = datetime.now(timezone.utc).isoformat()
            save_json(archive / 'manifest.json', manifest)
        except BaseException:
            for name in names:
                overwrite(streams[name], originals[name])
            manifest['status'] = 'rolled_back'
            save_json(archive / 'manifest.json', manifest)
            raise
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activate', type=Path)
    args = parser.parse_args()
    activate(args.activate) if args.activate else prepare()

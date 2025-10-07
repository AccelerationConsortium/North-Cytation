"""Auto-resume labeling loop.

Repeatedly invokes llm_label.py with --resume until all records in the input file
are labeled or a maximum number of passes is reached.

Usage (PowerShell):
  python research/literature_search/scripts/auto_label_loop.py \
      --input research/literature_search/data/prompt_preview_top15.jsonl \
      --output research/literature_search/data/labels_top15.jsonl \
      --model gpt-4o-mini --rate-limit-per-min 40 --max-passes 20 \
      --sleep-between-passes 30 --graceful-interrupt --heartbeat-secs 10

Notes:
- Safe to interrupt; rerun continues where left off (delegated to --resume).
- Emits progress summary after each pass.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time, os, tempfile
from typing import Set

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'llm_label.py')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', default='gpt-4o-mini')
    ap.add_argument('--rate-limit-per-min', type=int, default=40)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--max-retries', type=int, default=3)
    ap.add_argument('--sleep-on-error', type=float, default=5.0)
    ap.add_argument('--sleep-between-passes', type=float, default=15.0, help='Seconds between consecutive passes.')
    ap.add_argument('--max-passes', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0, help='(Forwarded) Optional cap on records processed each individual pass before resuming.')
    ap.add_argument('--shuffle', action='store_true')
    ap.add_argument('--request-timeout', type=float, default=0.0)
    ap.add_argument('--heartbeat-secs', type=float, default=0.0)
    ap.add_argument('--graceful-interrupt', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    return ap.parse_args()


def collect_ids(path: str) -> Set[str]:
    ids: Set[str] = set()
    if not os.path.exists(path):
        return ids
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                obj=json.loads(line)
            except Exception:
                continue
            rid = obj.get('id') or obj.get('model_output',{}).get('id')
            if rid:
                ids.add(rid)
    return ids


def count_input(path: str) -> int:
    c=0
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip():
                c+=1
    return c


def run_pass(args, pass_index: int) -> int:
    cmd = [sys.executable, SCRIPT_PATH,
           '--input', args.input,
           '--output', args.output,
           '--model', args.model,
           '--rate-limit-per-min', str(args.rate_limit_per_min),
           '--max-retries', str(args.max_retries),
           '--sleep-on-error', str(args.sleep_on_error),
           '--resume']
    if args.limit: cmd += ['--limit', str(args.limit)]
    if args.shuffle: cmd.append('--shuffle')
    if args.request_timeout: cmd += ['--request-timeout', str(args.request_timeout)]
    if args.heartbeat_secs: cmd += ['--heartbeat-secs', str(args.heartbeat_secs)]
    if args.graceful_interrupt: cmd.append('--graceful-interrupt')
    if args.dry_run: cmd.append('--dry-run')
    if args.verbose: cmd.append('--verbose')

    print(f"[AUTO] Pass {pass_index}: launching llm_label.py")
    start=time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration=time.time()-start

    print(f"[AUTO] Pass {pass_index} completed in {duration:.1f}s, return code {proc.returncode}")
    if args.verbose:
        print('[AUTO][STDOUT]\n'+proc.stdout)
        print('[AUTO][STDERR]\n'+proc.stderr)
    else:
        # surface last line for quick glance
        last_stdout_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ''
        print(f"[AUTO] Tail: {last_stdout_line}")
    return proc.returncode


def main():
    args = parse_args()
    total = count_input(args.input)
    print(f"[AUTO] Input records: {total}")

    for p in range(1, args.max_passes+1):
        before_ids = collect_ids(args.output)
        before_count = len(before_ids)
        if before_count >= total:
            print(f"[AUTO] All {before_count}/{total} records already labeled. Done.")
            return
        rc = run_pass(args, p)
        after_ids = collect_ids(args.output)
        after_count = len(after_ids)
        gained = after_count - before_count
        print(f"[AUTO] Progress after pass {p}: {after_count}/{total} (+{gained})")
        if after_count >= total:
            print("[AUTO] Completed labeling.")
            return
        if rc != 0:
            print(f"[AUTO] Non-zero return code {rc}; sleeping then continuing (unless repeated).")
        time.sleep(args.sleep_between_passes)
    print(f"[AUTO] Reached max passes ({args.max_passes}) without completing. Partial progress saved: {collect_ids(args.output)} records.")

if __name__ == '__main__':
    main()

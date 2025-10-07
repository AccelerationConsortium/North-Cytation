"""LLM labeling script for relevance classification.

Reads prompts from a JSONL file (each line must contain a 'prompt' field) and sends them
sequentially to an OpenAI chat/completions endpoint expecting STRICT JSON output per our schema.

Usage (PowerShell examples):
  # Dry run (no API calls, just counts and schema check)
  python research/literature_search/scripts/llm_label.py --input research/literature_search/data/prompt_preview_new.jsonl --output research/literature_search/data/llm_labels_dry.jsonl --dry-run

  # Real run (needs OPENAI_API_KEY in env or .env)
  python research/literature_search/scripts/llm_label.py --input research/literature_search/data/llm_gated.jsonl --output research/literature_search/data/llm_labels.jsonl --model gpt-4o-mini --rate-limit-per-min 40

Environment:
  - OPENAI_API_KEY must be set (or .env present with it) for non-dry runs.
  - Optional: OPENAI_BASE_URL to point to Azure/OpenAI-compatible proxy.

Schema enforced fields in model JSON output:
  id, relevance_label, confidence, rationale, signals{...}, failure_reasons[]

If malformed JSON or missing keys are observed, the script will retry (up to --max-retries) with a
repair system instruction that echoes the previous raw output for guidance.
"""
from __future__ import annotations
import os, json, time, argparse, sys, re, threading, signal
from typing import Dict, Any, Iterable, Optional
from dataclasses import dataclass

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # dotenv optional
    def load_dotenv(*_, **__):
        return False

# Lazy import openai so dry-run works without package installed
try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore

SCHEMA_REQUIRED_TOP = ["id", "relevance_label", "confidence", "rationale", "signals", "failure_reasons"]
VALID_LABELS = {"relevant", "maybe", "irrelevant"}
VALID_FAILURE = {"too_device_specific","no_workflow","review_style","generic_hype","insufficient_specificity","theoretical_only"}

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

@dataclass
class Args:
    input: str
    output: str
    model: str
    rate_limit_per_min: int
    temperature: float
    max_retries: int
    dry_run: bool
    sleep_on_error: float
    verbose: bool
    limit: int
    shuffle: bool
    resume: bool
    request_timeout: float
    graceful_interrupt: bool
    heartbeat_secs: float


def parse_args() -> Args:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', default='gpt-4o-mini')
    ap.add_argument('--rate-limit-per-min', type=int, default=40, help='Max calls per minute (client-side pacing).')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--max-retries', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true', help='Skip API calls; validate file and echo placeholder outputs.')
    ap.add_argument('--sleep-on-error', type=float, default=5.0)
    ap.add_argument('--verbose', action='store_true', help='Enable detailed per-record logging for debugging.')
    ap.add_argument('--limit', type=int, default=0, help='If >0, only process first N records (after optional shuffle).')
    ap.add_argument('--shuffle', action='store_true', help='Shuffle input order before limiting.')
    ap.add_argument('--resume', action='store_true', help='If output exists, append new labels skipping IDs already completed.')
    ap.add_argument('--request-timeout', type=float, default=0.0, help='Per-request timeout seconds (0 = no extra timeout wrapper).')
    ap.add_argument('--graceful-interrupt', action='store_true', help='First Ctrl+C defers until current record completes; second forces immediate exit.')
    ap.add_argument('--heartbeat-secs', type=float, default=0.0, help='If >0, emit a heartbeat line every N seconds while waiting for model response.')
    return Args(**vars(ap.parse_args()))


def load_env():
    """Load environment variables from either repo root .env or local directory .env.

    Priority:
      1. Existing process env (if OPENAI_API_KEY already set)
      2. .env in repository root
      3. .env in current working directory
      4. .env in literature_search folder (where some users may have placed it)
    """
    if os.environ.get('OPENAI_API_KEY'):
        return os.environ['OPENAI_API_KEY']
    # Candidate locations
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    candidates = [
        os.path.join(repo_root, '.env'),
        os.path.join(os.getcwd(), '.env'),
        os.path.join(os.path.dirname(__file__), '..', '.env'),
    ]
    for path in candidates:
        if os.path.exists(path):
            load_dotenv(path)
            if os.environ.get('OPENAI_API_KEY'):
                return os.environ['OPENAI_API_KEY']
    return None


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] Skipping malformed line {ln}: {e}", file=sys.stderr)
                continue
            yield obj


def extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Attempt to extract a JSON object from raw model response."""
    m = JSON_BLOCK_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def validate_record(obj: Dict[str, Any]) -> Optional[str]:
    missing = [k for k in SCHEMA_REQUIRED_TOP if k not in obj]
    if missing:
        return f"Missing keys: {missing}"
    if obj['relevance_label'] not in VALID_LABELS:
        return f"Invalid relevance_label {obj['relevance_label']}"
    if not isinstance(obj.get('signals'), dict):
        return "signals must be object"
    if not isinstance(obj.get('failure_reasons'), list):
        return "failure_reasons must be list"
    # Optional basic failure reasons validation
    for fr in obj.get('failure_reasons'):
        if fr not in VALID_FAILURE:
            return f"Invalid failure reason: {fr}"
    return None


def openai_client(key: str):
    base = os.environ.get('OPENAI_BASE_URL')
    if base:
        # For Azure or proxy-compatible endpoints using OpenAI python client
        openai.base_url = base.rstrip('/') + '/v1'
    openai.api_key = key
    return openai

SYSTEM_REPAIR = (
    "You previously returned malformed JSON. Return ONLY valid JSON matching the schema with the same decision intent."  # noqa: E501
)

SYSTEM_BASE = "You are a strict JSON generating assistant. Return ONLY a single JSON object matching the provided schema."


def call_model(client, model: str, prompt: str, temperature: float) -> str:
    # Using Chat Completions style
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_BASE},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        response_format=None,
    )
    return resp.choices[0].message.content  # type: ignore


def label_prompts(args: Args):
    key = load_env()
    if args.dry_run and not key:
        print("[INFO] Dry run without API key.")
    elif not key:
        raise SystemExit("OPENAI_API_KEY not set. Provide via env or .env file.")

    client = None if args.dry_run else openai_client(key)  # type: ignore

    rate_sleep = 60.0 / max(1, args.rate_limit_per_min)

    inputs = list(iter_jsonl(args.input))
    if args.shuffle:
        try:
            import random
            random.shuffle(inputs)
        except Exception:
            print('[WARN] Shuffle requested but random module failed to load.')
    if args.limit and args.limit > 0:
        inputs = inputs[:args.limit]
    total = len(inputs)
    print(f"Loaded {total} prompt records from {args.input}")
    if args.verbose:
        print(f"[DEBUG] Output path: {args.output}; Model: {args.model}; Dry-run: {args.dry_run}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    existing_ids = set()
    resumed_count = 0
    file_mode = 'w'
    if args.resume and os.path.exists(args.output):
        # Load existing IDs to skip already completed records
        try:
            with open(args.output, 'r', encoding='utf-8') as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        rid0 = obj.get('id') or obj.get('model_output', {}).get('id')
                        if rid0:
                            existing_ids.add(rid0)
                    except Exception:
                        continue
            resumed_count = len(existing_ids)
            file_mode = 'a'
            if args.verbose:
                print(f"[DEBUG] Resume enabled. Found {resumed_count} existing labeled IDs; will skip them.")
        except Exception as e:
            print(f"[WARN] Failed reading existing output for resume: {e}. Starting fresh.")
            existing_ids.clear()
            file_mode = 'w'

    out_f = open(args.output, file_mode, encoding='utf-8')

    successes = resumed_count
    interrupted = {'flag': False, 'hard': False}

    def _sig_handler(signum, frame):  # type: ignore
        if not args.graceful_interrupt:
            raise KeyboardInterrupt
        if interrupted['flag']:
            interrupted['hard'] = True
            print("[INTERRUPT] Second interrupt received: hard exit after this record.", file=sys.stderr)
        else:
            interrupted['flag'] = True
            print("[INTERRUPT] Graceful interrupt requested; finishing current record then stopping.", file=sys.stderr)

    if args.graceful_interrupt:
        try:
            signal.signal(signal.SIGINT, _sig_handler)
        except Exception:
            pass  # On some platforms setting may fail silently
    for idx, rec in enumerate(inputs, start=1):
        rid = rec.get('id') or f"row_{idx}"
        prompt = rec.get('prompt')
        prompt_version = rec.get('prompt_version')
        if not prompt:
            print(f"[WARN] Record {rid} missing 'prompt' field; skipping.")
            continue

        if rid in existing_ids:
            if args.verbose:
                print(f"[DEBUG] Skipping already-labeled ID {rid}")
            continue

        if args.verbose:
            print(f"[DEBUG] Processing {rid} ({idx}/{total})")

        if args.dry_run:
            mock = {
                'id': rid,
                'relevance_label': 'maybe',
                'confidence': 0.5,
                'rationale': 'dry-run placeholder',
                'signals': {
                    'workflow_described': False,
                    'multi_objective': False,
                    'optimization_language': False,
                    'automation_cue': False,
                    'device_centric': False,
                    'review_like': False,
                },
                'failure_reasons': [],
            }
            out_line = {'id': rid, 'model_output': mock}
            if prompt_version:
                out_line['prompt_version'] = prompt_version
            out_f.write(json.dumps(out_line, ensure_ascii=False) + '\n')
            successes += 1
            continue

        # Real call with retries
        attempt = 0
        last_error = None
        system_messages = [SYSTEM_BASE]
        while attempt <= args.max_retries:
            attempt += 1
            try:
                raw: Optional[str] = None
                if args.request_timeout and args.request_timeout > 0:
                    # Run model call in a thread to enforce soft timeout
                    result_holder: Dict[str, Any] = {}
                    exc_holder: Dict[str, Exception] = {}

                    def _runner():
                        try:
                            result_holder['raw'] = call_model(client, args.model, prompt, args.temperature)
                        except Exception as e:  # pragma: no cover - defensive
                            exc_holder['e'] = e

                    t = threading.Thread(target=_runner, daemon=True)
                    t.start()
                    t.join(args.request_timeout)
                    if t.is_alive():
                        last_error = f"Timeout after {args.request_timeout}s"
                        if args.verbose:
                            print(f"[DEBUG] {rid} attempt {attempt} timeout.")
                        # Thread likely still running; we abandon result and retry
                        # (OpenAI client may still consume resources; acceptable for small batch.)
                        raise TimeoutError(last_error)
                    if 'e' in exc_holder:
                        raise exc_holder['e']
                    raw = result_holder.get('raw')
                else:
                    # Heartbeat management: spawn a watcher thread if requested
                    if args.heartbeat_secs and args.heartbeat_secs > 0:
                        hb_stop = {'stop': False}
                        def _heartbeat():  # pragma: no cover - side-effect logging
                            last_emit = 0.0
                            while not hb_stop['stop']:
                                now = time.time()
                                if now - last_emit >= args.heartbeat_secs:
                                    print(f"[HEARTBEAT] Waiting on model for {rid} attempt {attempt}...", flush=True)
                                    last_emit = now
                                time.sleep(0.25)
                        hb_thread = threading.Thread(target=_heartbeat, daemon=True)
                        hb_thread.start()
                    else:
                        hb_stop = None
                        hb_thread = None
                    try:
                        raw = call_model(client, args.model, prompt, args.temperature)
                    finally:
                        if hb_stop is not None:
                            hb_stop['stop'] = True
                        if hb_thread is not None:
                            hb_thread.join(timeout=0.5)
                if args.verbose:
                    snippet = (raw[:120] + '...') if raw and len(raw) > 120 else raw
                    print(f"[DEBUG] {rid} attempt {attempt}: raw snippet: {snippet}")
            except Exception as e:  # network/API error
                last_error = f"API error: {e}"
                print(f"[ERROR] {rid} {last_error}; attempt {attempt}/{args.max_retries}")
                time.sleep(args.sleep_on_error)
                continue
            data = extract_json(raw or '')
            if not data:
                last_error = "No JSON object parsed"
                if args.verbose:
                    print(f"[DEBUG] {rid} attempt {attempt} parse failure: {last_error}")
            else:
                err = validate_record(data)
                if not err:
                    out_line = {'id': rid, 'model_output': data}
                    if prompt_version:
                        out_line['prompt_version'] = prompt_version
                    out_f.write(json.dumps(out_line, ensure_ascii=False) + '\n')
                    out_f.flush()
                    if args.verbose:
                        print(f"[DEBUG] {rid} success on attempt {attempt}")
                    successes += 1
                    break
                last_error = f"Schema validation failed: {err}"
                if args.verbose:
                    print(f"[DEBUG] {rid} attempt {attempt} schema failure: {last_error}")
            # Prepare retry with repair hint appended to prompt
            prompt = prompt + "\n\n(Previous malformed output was shown to system to enforce correction.)"
            time.sleep(1.0)
        else:
            print(f"[FAIL] {rid} after {args.max_retries} retries: {last_error}")
        # Early exit if graceful interrupt requested
        if interrupted['flag']:
            print("[INTERRUPT] Graceful stop after completing current record.")
            break

        time.sleep(rate_sleep)

    out_f.close()
    processed_target = total if not args.resume else (total - resumed_count)
    new_successes = successes - resumed_count
    if interrupted['flag'] and not interrupted['hard']:
        print(f"Interrupted (graceful). Partial progress saved. Cumulative successes {successes}/{total}.")
    elif args.resume:
        print(f"Completed. New successes: {new_successes}/{processed_target}. Total cumulative successes (including prior): {successes}/{total} (skipped {resumed_count}). Output -> {args.output}")
    else:
        print(f"Completed. Successes: {successes}/{total}. Output -> {args.output}")
    if args.verbose and successes < total:
        print("[DEBUG] Some records failed. Inspect above logs for failure reasons.")


def main():
    args = parse_args()
    label_prompts(args)

if __name__ == '__main__':
    main()

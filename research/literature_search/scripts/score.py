"""Generic scoring stage.

Axes are declared in `scoring.axis_definitions` within the YAML config.
Each axis definition supplies a `type` plus any parameters required by the
handler (e.g. cap, source_field, kind_key, penalties, flags_mapping).

Adding a new axis now requires only editing the config and (if needed)
adding a small handler in AXIS_TYPE_HANDLERS below.
"""

from __future__ import annotations

import csv
import os
import json
import hashlib
from datetime import datetime, UTC
from typing import Dict, List, Callable

try:
    from .utils import load_yaml, iter_jsonl, overwrite_jsonl, summarize_stats
except ImportError:  # direct execution fallback
    import sys, os as _os
    _here = os.path.dirname(__file__)
    _parent = os.path.abspath(os.path.join(_here, '..', 'scripts'))
    if _parent not in sys.path:
        sys.path.append(_parent)
    from utils import load_yaml, iter_jsonl, overwrite_jsonl, summarize_stats  # type: ignore


def load_config() -> dict:
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'literature_pipeline_config.yaml'))
    return load_yaml(cfg_path)


def _safe_cap(v, default: int = 1) -> float:
    try:
        v = float(v) if v is not None else float(default)
        return v if v > 0 else float(default)
    except Exception:
        return float(default)


def axis_capability_fraction(record: dict, spec: dict, stats: dict, cfg: dict) -> float:
    toks = record.get("capability_tokens", []) or []
    if not toks:
        return 0.0
    cap = _safe_cap(spec.get("cap", 6))
    return min(1.0, len(toks) / cap)


def axis_unique_kind_fraction(record: dict, spec: dict, stats: dict, cfg: dict) -> float:
    source = spec.get("source_field")
    kind_key = spec.get("kind_key")
    if not source or not kind_key:
        return 0.0
    items = record.get(source, []) or []
    if not items:
        return 0.0
    kinds = {it.get(kind_key) for it in items if isinstance(it, dict) and it.get(kind_key)}
    if not kinds:
        return 0.0
    cap = _safe_cap(spec.get("cap", 6))
    return min(1.0, len(kinds) / cap)


def axis_unique_count_fraction(record: dict, spec: dict, stats: dict, cfg: dict) -> float:
    source = spec.get("source_field")
    if not source:
        return 0.0
    items = record.get(source, []) or []
    if not items:
        return 0.0
    cap = _safe_cap(spec.get("cap", len(items)))
    return min(1.0, len(items) / cap)


def axis_citation_log_minmax(record: dict, spec: dict, stats: dict, cfg: dict) -> float:
    c = record.get("cited_by_count", 0) or 0
    cmin, cmax = stats.get("cit_min", 0), stats.get("cit_max", 0)
    if cmax <= cmin:
        return 0.0
    import math
    scaled = (math.log1p(c) - math.log1p(cmin)) / (math.log1p(cmax) - math.log1p(cmin))
    return max(0.0, min(1.0, scaled))


def axis_pattern_penalty(record: dict, spec: dict, stats: dict, cfg: dict) -> float:
    """Generic pattern penalty axis.

    Config (example):
      novelty:
        type: pattern_penalty
        mode: sum            # sum | max
        cap: 1.0
        patterns:
          - any: ["self-driving lab", "self driving lab"]
            value: 0.6
          - all: ["autonomous", "lab"]
            value: 0.6
          - any: ["closed-loop", "closed loop"]
            value: 0.6
          - all: ["robotic", "automation"]
            value: 0.25
          - any: ["automation"]
            value: 0.10
    """
    abstract = (record.get("abstract") or "").lower()
    mode = (spec.get("mode") or "sum").lower()
    cap = float(spec.get("cap", 1.0))
    patterns = spec.get("patterns", []) or []
    total = 0.0
    best = 0.0
    for entry in patterns:
        if not isinstance(entry, dict):
            continue
        value = float(entry.get("value", 0.0))
        if value <= 0:
            continue
        matched = False
        if "any" in entry:
            opts = [o.lower() for o in (entry.get("any") or [])]
            matched = any(o and o in abstract for o in opts)
        elif "all" in entry:
            req = [o.lower() for o in (entry.get("all") or [])]
            matched = all(o and o in abstract for o in req) if req else False
        elif "regex" in entry:
            import re
            try:
                rgx = re.compile(entry.get("regex"), flags=re.IGNORECASE)
                matched = bool(rgx.search(abstract))
            except Exception:
                matched = False
        elif "pair" in entry:  # syntactic sugar for len==2 all
            pair = [o.lower() for o in (entry.get("pair") or [])]
            matched = len(pair) == 2 and all(p and p in abstract for p in pair)
        if matched:
            if mode == "max":
                if value > best:
                    best = value
            else:  # sum
                total += value
                if total >= cap:
                    total = cap
                    break
    score = best if mode == "max" else total
    return min(cap, score)


def axis_constraint_penalty_sum(record: dict, spec: dict, stats: dict, cfg: dict) -> float:
    flags = record.get("flags", []) or []
    mapping = spec.get("flags_mapping", {}) or {}
    total = 0.0
    for fl, val in mapping.items():
        if fl in flags:
            total += val
    return min(1.0, total)


AXIS_TYPE_HANDLERS: Dict[str, Callable[[dict, dict, dict, dict], float]] = {
    "capability_fraction": axis_capability_fraction,
    "unique_kind_fraction": axis_unique_kind_fraction,
    "unique_count_fraction": axis_unique_count_fraction,
    "citation_log_minmax": axis_citation_log_minmax,
    "pattern_penalty": axis_pattern_penalty,
    "constraint_penalty_sum": axis_constraint_penalty_sum,
}


def main():
    cfg = load_config()
    fetch_cfg = cfg["fetch"]
    scoring_cfg = cfg.get("scoring", {})
    parsed_path = fetch_cfg["parsed_jsonl"]
    scored_csv = fetch_cfg["scored_csv"]
    run_summary_path = os.path.join(os.path.dirname(scored_csv), "run_summary.json")

    records = list(iter_jsonl(parsed_path))
    if not records:
        print("No parsed records found; run extract stage first.")
        return

    stats = summarize_stats(records)
    weights = scoring_cfg.get("weights", {})
    axis_definitions: Dict[str, dict] = scoring_cfg.get("axis_definitions", {})

    scored_rows: List[dict] = []
    for r in records:
        axes: Dict[str, float] = {}
        for axis_name, spec in axis_definitions.items():
            handler = AXIS_TYPE_HANDLERS.get(spec.get("type"))
            if not handler:
                continue
            try:
                axes[axis_name] = float(handler(r, spec, stats, cfg))
            except Exception as e:
                print(f"WARN axis {axis_name} failed: {e}")
                axes[axis_name] = 0.0
        composite = 0.0
        for name, val in axes.items():
            composite += weights.get(name, 0) * val
        out_row = {
            "id": r.get("id"),
            "doi": r.get("doi"),
            "title": r.get("title"),
            "year": r.get("publication_year"),
            "cited_by_count": r.get("cited_by_count", 0),
            **axes,
            "score_total": round(composite, 4),
        }
        r["scores"] = out_row
        scored_rows.append(out_row)

    overwrite_jsonl(parsed_path, records)

    os.makedirs(os.path.dirname(scored_csv), exist_ok=True)
    fieldnames = list(scored_rows[0].keys())
    with open(scored_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_rows)
    print(f"Scored {len(scored_rows)} records -> {scored_csv}")

    cfg_bytes = json.dumps(cfg, sort_keys=True).encode("utf-8")
    cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()[:12]
    summary = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config_hash": cfg_hash,
        "version": cfg.get("version"),
        "counts": {"parsed_records": len(records), "scored_records": len(scored_rows)},
        "weights": scoring_cfg.get("weights", {}),
        "axes": list(axis_definitions.keys()),
        "score_fields": list(scored_rows[0].keys()),
    }
    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Run summary -> {run_summary_path} (config_hash={cfg_hash})")


if __name__ == "__main__":
    main()

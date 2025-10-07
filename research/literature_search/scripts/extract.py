"""Extraction stage.

Reads raw JSONL produced by fetch stage and emits a parsed JSONL with
generic signals (numeric factors, objective terms, capability tokens, flags).

Design choices:
 - Avoid embedding experiment-specific semantics; only surface raw matches.
 - Keep regex patterns fully in config (literature_pipeline_config.yaml).
 - Provide simple solvent mention capture (future: load whitelist if needed).
"""

from __future__ import annotations

import os
import re
from typing import List, Dict, Set

try:  # relative when executed as module
    from .utils import load_yaml, iter_jsonl, overwrite_jsonl, find_all
except ImportError:  # fallback for direct script execution
    import sys, os as _os
    _here = _os.path.dirname(__file__)
    _parent = _os.path.abspath(_os.path.join(_here, '..', 'scripts'))
    if _parent not in sys.path:
        sys.path.append(_parent)
    from utils import load_yaml, iter_jsonl, overwrite_jsonl, find_all  # type: ignore


def load_config() -> dict:
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "literature_pipeline_config.yaml"))
    return load_yaml(cfg_path)


def load_solvent_whitelist() -> Set[str]:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "capabilities", "solvent_whitelist.yaml"))
    if not os.path.exists(path):
        return set()
    data = load_yaml(path) or {}
    wl = set()
    for _group, items in (data.get("solvent_whitelist", {}) or {}).items():
        for item in items:
            wl.add(item.lower())
    return wl


def tokenize_objectives(abstract: str, verbs: List[str]) -> List[str]:
    if not abstract:
        return []
    lower = abstract.lower()
    found = []
    for v in verbs:
        if v.lower() in lower:
            found.append(v.lower())
    return list(sorted(set(found)))


def detect_capability_tokens(abstract: str, verbs: List[str]) -> List[str]:
    if not abstract:
        return []
    lower = abstract.lower()
    hits = []
    for v in verbs:
        if v.lower() in lower:
            hits.append(v.lower())
    return list(sorted(set(hits)))


def compile_solvent_pattern(cfg: dict):
    patterns = cfg.get("extraction", {}).get("solvent_patterns")
    if not patterns:
        # fallback to legacy default list if not provided
        patterns = [r"\b(acetonitrile|dmso|dichloromethane|chloroform|toluene|hexane|ethanol|methanol|acetone|water)\b"]
    joined = "|".join(f"({p})" for p in patterns) if len(patterns) > 1 else patterns[0]
    try:
        return re.compile(joined, re.IGNORECASE)
    except Exception:
        return re.compile(r"^$")  # matches nothing on failure


def detect_solvents(abstract: str, solvent_re: re.Pattern) -> List[str]:
    if not abstract:
        return []
    return list(sorted(set(m.group(0).lower() for m in solvent_re.finditer(abstract))))


def build_numeric_factors(abstract: str, cfg_ext: dict) -> List[Dict]:
    factors: List[Dict] = []
    # Use range, ratio, concentration, equivalents patterns
    range_hits = find_all(cfg_ext.get("range_patterns", []), abstract)
    ratio_hits = find_all(cfg_ext.get("ratio_patterns", []), abstract)
    conc_hits = find_all(cfg_ext.get("conc_patterns", []), abstract)
    equiv_hits = find_all(cfg_ext.get("equiv_patterns", []), abstract)

    def push(hit: str, kind: str):
        factors.append({"text": hit, "kind": kind})

    for h in range_hits:
        push(h, "range")
    for h in ratio_hits:
        push(h, "ratio")
    for h in conc_hits:
        push(h, "concentration")
    for h in equiv_hits:
        push(h, "equivalents")
    return factors


def derive_flags(record: dict, whitelist: Set[str]) -> List[str]:
    flags: List[str] = []
    mentions = record.get("solvent_mentions", [])
    for m in mentions:
        if whitelist and m.lower() not in whitelist:
            flags.append("non_whitelist_solvent")
            break
    return flags


def load_heuristic_list(cfg: dict, list_name: str) -> List[str]:
    heur_path = cfg.get("extraction", {}).get("heuristics_file")
    items: List[str] = []
    if heur_path and os.path.exists(heur_path):
        try:
            import yaml
            with open(heur_path, "r", encoding="utf-8") as f:
                hx = yaml.safe_load(f) or {}
            items = [t.lower() for t in (hx.get(list_name) or [])]
        except Exception as e:
            print(f"WARN: failed loading heuristics {list_name}: {e}")
    return items


def process_record(raw: dict, cfg: dict, whitelist: Set[str], solvent_re: re.Pattern) -> dict:
    ext_cfg = cfg.get("extraction", {})
    abstract = raw.get("abstract") or ""

    numeric_factors = build_numeric_factors(abstract, ext_cfg)
    objective_terms = tokenize_objectives(abstract, ext_cfg.get("objective_verbs", []))
    capability_tokens = detect_capability_tokens(abstract, ext_cfg.get("capability_verbs", []))
    solvents = detect_solvents(abstract, solvent_re)

    lower_abs = abstract.lower()
    # Generic heuristic term sets defined in heuristics file.
    term_sets = cfg.get("extraction", {}).get("heuristic_term_sets", {
        "polymer_terms": "polymer_term",
        "workflow_terms": "workflow_term",
        "device_terms": "device_term",
    })
    hits_map: Dict[str, List[str]] = {}
    for list_name, prefix in term_sets.items():
        terms = load_heuristic_list(cfg, list_name)
        matched: List[str] = []
        if terms:
            for t in terms:
                if t and t in lower_abs:
                    matched.append(t)
        if matched:
            unique_sorted = sorted(list(set(matched)))
        else:
            unique_sorted = []
        hits_map[f"{prefix}_term_hits"] = matched
        hits_map[f"{prefix}_term_unique"] = unique_sorted

    parsed = dict(raw)  # shallow copy
    parsed.update({
        "numeric_factors": numeric_factors,
        "objective_terms": objective_terms,
        "capability_tokens": capability_tokens,
        "solvent_mentions": solvents,
        "flags": derive_flags({"solvent_mentions": solvents}, whitelist),
        **hits_map,
    })
    return parsed


def main():
    cfg = load_config()
    fetch_cfg = cfg["fetch"]
    raw_path = fetch_cfg["raw_jsonl"]
    parsed_path = fetch_cfg["parsed_jsonl"]

    records = list(iter_jsonl(raw_path))
    whitelist = load_solvent_whitelist()
    solvent_re = compile_solvent_pattern(cfg)
    parsed_records = [process_record(r, cfg, whitelist, solvent_re) for r in records]
    overwrite_jsonl(parsed_path, parsed_records)
    print(f"Parsed {len(parsed_records)} records -> {parsed_path}")


if __name__ == "__main__":
    main()

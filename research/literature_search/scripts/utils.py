"""Shared utilities for literature search pipeline.

Design goals:
 - Keep pipeline stages (fetch/extract/score) decoupled from concrete
   experimental archetypes and keyword specifics.
 - Provide generic helpers for JSONL IO, config loading, normalization, and
   safe evaluation of scoring expressions.
 - Allow future plug-in of new axis strategy functions without changing
   core loop logic.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Callable, Tuple

import yaml


# ---------------------------------------------------------------------------
# Config & IO
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_jsonl(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        return iter(())  # empty iterator
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_jsonl(path: str, records: Iterable[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def overwrite_jsonl(path: str, records: Iterable[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Regex utilities (compiled caching)
# ---------------------------------------------------------------------------

_REGEX_CACHE: Dict[str, re.Pattern] = {}


def compile_pattern(p: str) -> re.Pattern:
    if p not in _REGEX_CACHE:
        _REGEX_CACHE[p] = re.compile(p, flags=re.IGNORECASE)
    return _REGEX_CACHE[p]


def find_all(patterns: List[str], text: str) -> List[str]:
    hits: List[str] = []
    if not text:
        return hits
    for p in patterns:
        rp = compile_pattern(p)
        for m in rp.finditer(text):
            span_txt = m.group(0)
            hits.append(span_txt)
    return hits


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def min_max_scale(value: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.0
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


def log1p_scale(value: float, vmin: float, vmax: float) -> float:
    if value < 0:
        value = 0
    if vmax <= vmin:
        return 0.0
    lv = math.log1p(value)
    lvmin = math.log1p(max(vmin, 0))
    lvmax = math.log1p(max(vmax, 0))
    return min_max_scale(lv, lvmin, lvmax)


# ---------------------------------------------------------------------------
# Scoring axis strategy registry (skeleton)
# ---------------------------------------------------------------------------

AxisFunc = Callable[[dict, dict, dict], float]


def axis_keyword_presence_combo(record: dict, axis_cfg: dict, ctx: dict) -> float:
    # axis_cfg:
    #   include_sets: [set_name,...]
    #   combine: 'weighted_sum' (future)
    abstract = record.get("abstract", "")
    abstract_lower = abstract.lower()
    keyword_sets = ctx.get("keyword_sets", {})
    total = 0
    for set_name in axis_cfg.get("include_sets", []):
        words = keyword_sets.get(set_name, [])
        present = any(w.lower() in abstract_lower for w in words)
        total += 1 if present else 0
    if not axis_cfg.get("include_sets"):
        return 0.0
    return total / len(axis_cfg.get("include_sets"))


def axis_numeric_factor_diversity(record: dict, axis_cfg: dict, ctx: dict) -> float:
    factors = record.get("numeric_factors", [])
    # naive diversity: number of unique unit labels
    kinds = set()
    for f in factors:
        unit = f.get("unit") or f.get("units") or ""
        if unit:
            kinds.add(unit)
    if not kinds:
        return 0.0
    cap = axis_cfg.get("diversity_cap", 6)
    return min(1.0, len(kinds) / cap)


def axis_objective_term_count(record: dict, axis_cfg: dict, ctx: dict) -> float:
    terms = set(record.get("objective_terms", []))
    if not terms:
        return 0.0
    max_terms = axis_cfg.get("max_terms", 4)
    return min(1.0, len(terms) / max_terms)


def axis_citation_minmax(record: dict, axis_cfg: dict, ctx: dict) -> float:
    c = record.get("cited_by_count", 0) or 0
    stats = ctx.get("stats", {})
    cmin, cmax = stats.get("cit_min", 0), stats.get("cit_max", 0)
    mode = axis_cfg.get("mode", "log1p")
    if mode == "log1p":
        return log1p_scale(c, cmin, cmax)
    return min_max_scale(c, cmin, cmax)


def axis_penalty_keyword(record: dict, axis_cfg: dict, ctx: dict) -> float:
    abstract = (record.get("abstract") or "").lower()
    keyword_sets = ctx.get("keyword_sets", {})
    set_name = axis_cfg.get("penalty_set")
    phrases = keyword_sets.get(set_name, [])
    penalty_total = 0.0
    per_hit = axis_cfg.get("per_hit", 0.1)
    cap = axis_cfg.get("cap", 1.0)
    for ph in phrases:
        if ph.lower() in abstract:
            penalty_total += per_hit
            if penalty_total >= cap:
                break
    return penalty_total


def axis_penalty_flags(record: dict, axis_cfg: dict, ctx: dict) -> float:
    flags = set(record.get("flags", []))
    watch = axis_cfg.get("flags", [])
    per_flag = axis_cfg.get("per_flag", 0.05)
    cap = axis_cfg.get("cap", 1.0)
    count = sum(1 for f in watch if f in flags)
    total = count * per_flag
    return min(cap, total)


AXIS_STRATEGIES: Dict[str, AxisFunc] = {
    "keyword_presence_combo": axis_keyword_presence_combo,
    "numeric_factor_diversity": axis_numeric_factor_diversity,
    "objective_term_count": axis_objective_term_count,
    "citation_minmax": axis_citation_minmax,
    "penalty_keyword": axis_penalty_keyword,
    "penalty_flags": axis_penalty_flags,
}


# ---------------------------------------------------------------------------
# Safe expression evaluation (simple arithmetic only)
# ---------------------------------------------------------------------------

ALLOWED_NAMES = {"min": min, "max": max, "abs": abs}


def eval_expression(expr: str, variables: Dict[str, float]) -> float:
    """Evaluate arithmetic expression with provided variables.

    Security: restrict builtins & names; no attribute or indexing.
    """
    code = compile(expr, "<expr>", "eval")
    for name in code.co_names:
        if name not in variables and name not in ALLOWED_NAMES:
            raise ValueError(f"Illegal name in expression: {name}")
    return float(eval(code, {"__builtins__": {}}, {**ALLOWED_NAMES, **variables}))


# ---------------------------------------------------------------------------
# Utility dataclasses for future typed expansion (optional use)
# ---------------------------------------------------------------------------

@dataclass
class AxisResult:
    name: str
    value: float


def summarize_stats(records: List[dict]) -> Dict[str, Any]:
    cit_values = [r.get("cited_by_count", 0) or 0 for r in records]
    if cit_values:
        return {"cit_min": min(cit_values), "cit_max": max(cit_values)}
    return {"cit_min": 0, "cit_max": 0}


__all__ = [
    "load_yaml",
    "iter_jsonl",
    "write_jsonl",
    "overwrite_jsonl",
    "find_all",
    "compile_pattern",
    "min_max_scale",
    "log1p_scale",
    "AXIS_STRATEGIES",
    "eval_expression",
    "summarize_stats",
]

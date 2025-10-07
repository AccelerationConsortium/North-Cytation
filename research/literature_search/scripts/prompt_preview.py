"""Render baseline LLM labeling prompts for a tri-segment sample (top/middle/bottom).

Outputs a JSONL file where each line contains:
  {
    "id": ..., "rank": int, "segment": "top|middle|bottom",
    "score_total": float,
    "axes": {...},
    "prompt": "<full baseline prompt text>"
  }

Usage:
  python research/literature_search/scripts/prompt_preview.py \
      --top 10 --middle 10 --bottom 10 \
      --out research/literature_search/data/prompt_preview.jsonl

The baseline prompt matches our high-fidelity design with axis glossary.

No API calls are performed; this is offline prompt generation for inspection.
"""
from __future__ import annotations
import os, csv, json, argparse, textwrap
from typing import List, Dict

try:
    from .utils import load_yaml
except ImportError:  # direct execution fallback
    import sys, os as _os
    _here = os.path.dirname(__file__)
    _parent = os.path.abspath(os.path.join(_here, '..', 'scripts'))
    if _parent not in sys.path:
        sys.path.append(_parent)
    from utils import load_yaml  # type: ignore

CFG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'literature_pipeline_config.yaml'))

AXIS_DESCRIPTIONS = {
    # Core positive/relevance axes
    "capability_fit": "action verbs coverage",
    "polymer_specificity": "domain vocab density",
    "parameter_space": "numeric/factor diversity",
    "multi_objective": ">=2 distinct optimization goals",
    "workflow_boost": "workflow operation verbs",
    # Neutral/auxiliary
    "impact": "citation influence (ignore for label)",
    # Penalties (higher = worse)
    "device_penalty": "device focus (higher=worse)",
    "constraint_penalty": "operational constraints",
    "novelty": "hype/cliché patterns",
}

BASE_DOMAIN_BRIEF = (
    "Identify abstracts describing actionable, automatable polymer (or analog domain) workflows that enable "
    "synthesis, post-functionalization, property tuning, or multi-objective optimization suitable for "
    "high-throughput or iterative experimentation."
)

CRITERIA_BLOCK = textwrap.dedent(
    """RELEVANCE CRITERIA:\n1. Actionable workflow or method: concrete experimental steps, reactions, transformations, optimization loops.\n2. Optimization or improvement context: explicit goals (conversion, stability, property balance, throughput).\n3. Domain specificity: polymer (or specified domain) modification/usage rather than generalized device performance.\n4. Exclude if primarily: device benchmarking, broad review, purely theoretical without experimental leverage, or generic hype lacking operational detail."""
)

JSON_SCHEMA_BLOCK = textwrap.dedent(
    """OUTPUT JSON SCHEMA:\n{\n  \"id\": \"string\",\n  \"relevance_label\": \"relevant|maybe|irrelevant\",\n  \"confidence\": 0.0,\n  \"rationale\": \"string\",\n  \"signals\": {\n    \"workflow_described\": false,\n    \"multi_objective\": false,\n    \"optimization_language\": false,\n    \"automation_cue\": false,\n    \"device_centric\": false,\n    \"review_like\": false\n  },\n  \"failure_reasons\": []\n}\n"""
)

INSTRUCTION_BLOCK = textwrap.dedent(
    """INSTRUCTIONS:\nLABEL RULES:\n- relevant: clear actionable workflow + domain-specific + NOT primarily device performance.\n- maybe: partial specificity OR weak workflow detail OR ambiguous focus.\n- irrelevant: lacks actionable workflow OR is review/device-only/theoretical hype.\nSIGNAL DEFINITIONS:\n- workflow_described: explicit procedural / synthesis / iterative steps.\n- multi_objective: two DISTINCT targeted properties with trade or simultaneous optimization (mere metric list without trade context = false).\n- optimization_language: optimize / maximize / improve / enhance / balance / minimize / reduce appear in an optimization context.\n- automation_cue: automated / iterative / high-throughput / platform / closed-loop hints.\n- device_centric: primary emphasis on device performance metrics over chemistry/process.\n- review_like: survey/review style enumeration or meta-summary tone.\nFAILURE REASONS (if label != relevant): [\"too_device_specific\",\"no_workflow\",\"review_style\",\"generic_hype\",\"insufficient_specificity\",\"theoretical_only\"]\nRATIONALE & CONFIDENCE:\n- rationale ≤ 40 tokens; cite 1–3 short quoted phrases evidencing decision.\n- confidence ∈ [0,1]; start baseline 0.5 then: add ~0.2 if multiple strong signals; subtract ~0.2 if conflicts / sparse.\nOUTPUT CONSTRAINT:\nReturn ONLY the JSON object conforming to schema — no extra text."""
)

# Extended confidence guidance appended below for stronger probability calibration
INSTRUCTION_BLOCK += """\nCONFIDENCE SCALE GUIDANCE:\n0.85–1.00: Clear multi-step workflow + polymer-centric + optimization/iteration cues.\n0.65–0.84: Strong workflow OR strong polymer specificity with minor ambiguity.\n0.45–0.64: Uncertain / partial signals (default region for maybe).\n0.25–0.44: Mostly missing workflow or strongly device-biased but a faint hint remains.\n0.00–0.24: Clearly out-of-scope (review/device-only/theoretical/hype with no actionable steps).\nIf labeling 'maybe', explicitly name 1 missing element preventing 'relevant'."""


def load_scored(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        dr = csv.DictReader(f)
        for r in dr:
            try:
                r['score_total'] = float(r.get('score_total', 0) or 0)
            except Exception:
                r['score_total'] = 0.0
            rows.append(r)
    rows.sort(key=lambda x: x['score_total'], reverse=True)
    return rows


def axis_line(row: Dict, axis_order: List[str]) -> str:
    """Render axis scores compactly, omitting zero-only auxiliary penalties when both zero.

    We keep core axes order, then penalties grouped as PENALTIES:[...].
    """
    core_axes = []
    penalty_axes = []
    for ax in axis_order:
        val = row.get(ax)
        try:
            fval = float(val)
        except Exception:
            fval = 0.0
        if ax in ("device_penalty", "constraint_penalty", "novelty"):
            penalty_axes.append((ax, fval))
        else:
            core_axes.append((ax, fval))
    core_str = " ".join(f"{a}={v:.2f}" for a,v in core_axes)
    # Filter penalty axes: drop novelty/constraint if 0.00; always show device_penalty.
    shown_pen = []
    for a,v in penalty_axes:
        if a == "device_penalty" or v > 0:
            shown_pen.append(f"{a}={v:.2f}")
    pen_str = f" PENALTIES:[{' '.join(shown_pen)}]" if shown_pen else ""
    return core_str + pen_str


def axis_glossary(axis_order: List[str]) -> str:
    """Compressed glossary; group penalties separately."""
    core_terms = []
    penalties = []
    for ax in axis_order:
        desc = AXIS_DESCRIPTIONS.get(ax, "axis")
        if ax in ("device_penalty", "constraint_penalty", "novelty"):
            penalties.append(f"{ax}={desc}")
        else:
            core_terms.append(f"{ax}={desc}")
    return "CORE: " + ", ".join(core_terms) + ("\nPENALTIES: " + ", ".join(penalties) if penalties else "")


def build_prompt(record: Dict, axis_order: List[str]) -> str:
    axes_line = axis_line(record, axis_order)
    glossary = axis_glossary(axis_order)
    abs_text = record.get('abstract') or ''
    title = record.get('title') or ''
    year = record.get('year') or record.get('publication_year') or ''
    cited = record.get('cited_by_count') or record.get('cited_by') or ''
    rid = record.get('id')
    prompt = (
        "System:\nYou are a precise scientific triage assistant. Output ONLY valid JSON.\n\n"
        "User:\nDOMAIN BRIEF:\n"
        f"{BASE_DOMAIN_BRIEF}\n\n{CRITERIA_BLOCK}\n\nAXIS SCORES (0–1):\n{axes_line}\n\n"
        f"AXIS LEGEND (compressed):\n{glossary}\n\nABSTRACT METADATA:\nID: {rid}\nTitle: {title}\nYear: {year}  Citations: {cited}\nAbstract:\n{abs_text}\n\n"
        f"{JSON_SCHEMA_BLOCK}{INSTRUCTION_BLOCK}\n"
    )
    return prompt


def pick_segments(rows: List[Dict], top_n: int, middle_n: int, bottom_n: int):
    n = len(rows)
    top = rows[:min(top_n, n)]
    mid_start = max(0, (n // 2) - (middle_n // 2))
    middle = rows[mid_start: mid_start + middle_n]
    bottom = rows[-bottom_n:] if bottom_n <= n else rows
    return top, middle, bottom


def attach_abstracts(rows: List[Dict], parsed_path: str):
    if not os.path.exists(parsed_path):
        return
    idx = {r['id']: r for r in rows if r.get('id')}
    with open(parsed_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rid = obj.get('id')
            if rid in idx:
                idx[rid]['abstract'] = obj.get('abstract') or ''
                idx[rid]['year'] = obj.get('publication_year')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--middle', type=int, default=10)
    ap.add_argument('--bottom', type=int, default=10)
    ap.add_argument('--out', default='research/literature_search/data/prompt_preview.jsonl')
    args = ap.parse_args()

    cfg = load_yaml(CFG_PATH)
    fetch_cfg = cfg.get('fetch', {})
    scored_csv = fetch_cfg.get('scored_csv')
    parsed_jsonl = fetch_cfg.get('parsed_jsonl')
    axis_defs = cfg.get('scoring', {}).get('axis_definitions', {})
    axis_order = list(axis_defs.keys())

    rows = load_scored(scored_csv)
    attach_abstracts(rows, parsed_jsonl)

    top, middle, bottom = pick_segments(rows, args.top, args.middle, args.bottom)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        def emit(segment_name: str, seg_rows: List[Dict]):
            for idx, r in enumerate(seg_rows, start=1):
                prompt = build_prompt(r, axis_order)
                axes_dict = {ax: float(r.get(ax, 0) or 0) for ax in axis_order}
                rec = {
                    'id': r.get('id'),
                    'rank': rows.index(r)+1,
                    'segment': segment_name,
                    'score_total': r.get('score_total'),
                    'axes': axes_dict,
                    'prompt': prompt,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        emit('top', top)
        emit('middle', middle)
        emit('bottom', bottom)

    print(f"Wrote prompt preview -> {args.out} (top={len(top)} middle={len(middle)} bottom={len(bottom)})")

if __name__ == '__main__':
    main()

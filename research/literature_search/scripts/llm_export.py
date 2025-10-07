"""LLM export preparation.

Generates JSONL files for:
  1. Gated (recommended) set based on existing gating output CSV.
  2. Exploration band between two percentiles (inclusive) over score_total.

Each output record includes:
  id, rank, score_total, axes (dict of axis scores), title, abstract, year,
  band: one of {gated, explore}

Config additions (expected existing keys):
  fetch.scored_csv
  gating.output_csv  (produced by recommend_stage1.py)
  fetch.parsed_jsonl

CLI options:
  --explore-low  0.70   lower percentile bound (0-1)
  --explore-high 0.85   upper percentile bound (0-1)
  --limit-explore N     optional cap on exploration records
  --out-dir path        base directory for exports (default: research/literature_search/data)

Usage:
  python research/literature_search/scripts/llm_export.py --explore-low 0.70 --explore-high 0.85

"""
from __future__ import annotations
import csv, json, os, argparse
from typing import List, Dict

try:
    from .utils import load_yaml, iter_jsonl
except ImportError:  # script direct
    import sys, os as _os
    _here = os.path.dirname(__file__)
    _parent = os.path.abspath(os.path.join(_here, '..', 'scripts'))
    if _parent not in sys.path:
        sys.path.append(_parent)
    from utils import load_yaml, iter_jsonl  # type: ignore

CFG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'literature_pipeline_config.yaml'))


def read_scored(scored_csv: str) -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.exists(scored_csv):
        return rows
    with open(scored_csv, 'r', encoding='utf-8') as f:
        dr = csv.DictReader(f)
        for r in dr:
            try:
                r['score_total'] = float(r.get('score_total', 0) or 0)
            except Exception:
                r['score_total'] = 0.0
            rows.append(r)
    rows.sort(key=lambda x: x['score_total'], reverse=True)
    return rows


def attach_abstracts(rows: List[Dict], parsed_jsonl: str):
    if not rows or not os.path.exists(parsed_jsonl):
        return
    idx = {r['id']: r for r in rows if r.get('id')}
    for obj in iter_jsonl(parsed_jsonl):
        rid = obj.get('id')
        if rid in idx:
            idx[rid]['abstract'] = obj.get('abstract') or ''
            idx[rid]['year'] = obj.get('publication_year')


def load_gated_ids(gated_csv: str) -> List[str]:
    ids: List[str] = []
    if not gated_csv or not os.path.exists(gated_csv):
        return ids
    with open(gated_csv, 'r', encoding='utf-8') as f:
        dr = csv.DictReader(f)
        for r in dr:
            rid = r.get('id')
            if rid:
                ids.append(rid)
    return ids


def percentile_bounds(rows: List[Dict], low: float, high: float) -> List[Dict]:
    if not rows:
        return []
    n = len(rows)
    lo_idx = int(max(0, min(n-1, round(low * (n-1)))))
    hi_idx = int(max(0, min(n-1, round(high * (n-1)))))
    if lo_idx > hi_idx:
        lo_idx, hi_idx = hi_idx, lo_idx
    return rows[lo_idx: hi_idx+1]


def extract_axis_scores(row: Dict) -> Dict[str, float]:
    axes: Dict[str, float] = {}
    for k, v in row.items():
        if k in {'id','doi','title','year','cited_by_count','score_total'}:
            continue
        # attempt float parse
        try:
            fv = float(v)
        except Exception:
            continue
        axes[k] = fv
    return axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--explore-low', type=float, default=0.70, help='Lower percentile (0-1) for exploration band')
    ap.add_argument('--explore-high', type=float, default=0.85, help='Upper percentile (0-1) for exploration band')
    ap.add_argument('--limit-explore', type=int, default=100, help='Optional cap on exploration records')
    ap.add_argument('--out-dir', default='research/literature_search/data', help='Output directory base')
    args = ap.parse_args()

    cfg = load_yaml(CFG_PATH)
    fetch_cfg = cfg.get('fetch', {})
    gating_cfg = cfg.get('gating', {})
    scored_csv = fetch_cfg.get('scored_csv')
    parsed_jsonl = fetch_cfg.get('parsed_jsonl')
    gated_csv = gating_cfg.get('output_csv')

    rows = read_scored(scored_csv)
    attach_abstracts(rows, parsed_jsonl)
    gated_ids = set(load_gated_ids(gated_csv))

    # Prepare gated export
    gated_records = [r for r in rows if r.get('id') in gated_ids]

    # Exploration band (exclude already gated)
    explore_candidates = percentile_bounds(rows, args.explore_low, args.explore_high)
    explore_filtered = [r for r in explore_candidates if r.get('id') not in gated_ids]
    if args.limit_explore and len(explore_filtered) > args.limit_explore:
        explore_filtered = explore_filtered[:args.limit_explore]

    os.makedirs(args.out_dir, exist_ok=True)
    gated_out = os.path.join(args.out_dir, 'llm_gated.jsonl')
    explore_out = os.path.join(args.out_dir, 'llm_explore.jsonl')

    def emit(path: str, recs: List[Dict], band: str):
        with open(path, 'w', encoding='utf-8') as f:
            for rank, r in enumerate(recs, start=1):
                obj = {
                    'id': r.get('id'),
                    'rank': rank,
                    'band': band,
                    'title': r.get('title'),
                    'abstract': r.get('abstract'),
                    'year': r.get('year'),
                    'score_total': r.get('score_total'),
                    'axes': extract_axis_scores(r),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    emit(gated_out, gated_records, 'gated')
    emit(explore_out, explore_filtered, 'explore')

    print(f"LLM export complete: gated={len(gated_records)} -> {gated_out}; explore={len(explore_filtered)} -> {explore_out}")

if __name__ == '__main__':
    main()

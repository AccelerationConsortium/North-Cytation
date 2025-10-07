"""Diagnostics for LLM labels vs heuristic scores.

Loads:
  - scores CSV (heuristic axis scores + score_total)
  - labels JSONL (produced by llm_label.py)
Outputs:
  - diagnostics JSON summarizing:
      * label counts
      * failure reason counts
      * mean axis values per relevance_label
      * disagreement buckets (high_score+irrelevant, low_score+relevant/maybe)
  - optional CSV of merged rows
Usage:
  python research/literature_search/scripts/label_diagnostics.py \
      --scores research/literature_search/data/scored_candidates.csv \
      --labels research/literature_search/data/llm_labels.jsonl \
      --out-prefix research/literature_search/data/diag
"""
from __future__ import annotations
import csv, json, argparse, os
from typing import Dict, List


def load_scores(path: str) -> List[Dict]:
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
    for rank, r in enumerate(rows, start=1):
        r['rank'] = rank
    return rows


def load_labels(path: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    if not os.path.exists(path):
        return out
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rid = obj.get('id') or obj.get('model_output', {}).get('id')
            mo = obj.get('model_output') or {}
            if rid:
                out[rid] = mo
    return out


def aggregate(scores: List[Dict], labels: Dict[str, Dict], high_pct: float = 0.15, low_pct: float = 0.50):
    labeled_rows = []
    for r in scores:
        rid = r.get('id')
        if rid in labels:
            lr = labels[rid]
            merged = {**r, **{f"llm_{k}": v for k, v in lr.items()}}
            labeled_rows.append(merged)

    n = len(scores)
    high_cut_rank = max(1, int(n * high_pct))
    low_cut_rank = max(1, int(n * low_pct))

    label_counts = {}
    failure_counts = {}
    axis_means: Dict[str, Dict[str, float]] = {}

    # detect axis columns
    axis_cols = [c for c in scores[0].keys() if c not in {'id','doi','title','year','cited_by_count','rank','score_total'}]

    for row in labeled_rows:
        lab = row.get('llm_relevance_label') or 'UNKNOWN'
        label_counts[lab] = label_counts.get(lab, 0) + 1
        for fr in row.get('llm_failure_reasons', []):
            failure_counts[fr] = failure_counts.get(fr, 0) + 1
        # axis means
        for ax in axis_cols + ['score_total']:
            try:
                val = float(row.get(ax, 0) or 0)
            except Exception:
                val = 0.0
            axis_means.setdefault(ax, {}).setdefault(lab, 0.0)
            axis_means[ax][lab] += val

    # normalize axis means
    for ax, mp in axis_means.items():
        for lab in list(mp.keys()):
            cnt = label_counts.get(lab, 1)
            axis_means[ax][lab] = mp[lab] / max(1, cnt)

    # Disagreement buckets
    high_irrelevant = []
    low_positive = []  # positive = relevant or maybe
    for row in labeled_rows:
        rank = row.get('rank', 10**9)
        lab = row.get('llm_relevance_label')
        if rank <= high_cut_rank and lab == 'irrelevant':
            high_irrelevant.append(row['id'])
        if rank > low_cut_rank and lab in {'relevant','maybe'}:
            low_positive.append(row['id'])

    return {
        'total_scored': len(scores),
        'total_labeled': len(labeled_rows),
        'label_counts': label_counts,
        'failure_reason_counts': failure_counts,
        'axis_means_by_label': axis_means,
        'high_cut_rank': high_cut_rank,
        'low_cut_rank': low_cut_rank,
        'high_irrelevant_ids': high_irrelevant,
        'low_positive_ids': low_positive,
    }, labeled_rows


def write_outputs(diag: Dict, merged_rows: List[Dict], out_prefix: str):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    with open(out_prefix + '_diagnostics.json', 'w', encoding='utf-8') as f:
        json.dump(diag, f, indent=2)
    if merged_rows:
        # ensure consistent field order
        fieldnames = list(merged_rows[0].keys())
        with open(out_prefix + '_merged.csv', 'w', encoding='utf-8', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            wr.writerows(merged_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--high-pct', type=float, default=0.15)
    ap.add_argument('--low-pct', type=float, default=0.50)
    args = ap.parse_args()

    scores = load_scores(args.scores)
    labels = load_labels(args.labels)
    diag, merged = aggregate(scores, labels, args.high_pct, args.low_pct)
    write_outputs(diag, merged, args.out_prefix)
    print(f"Diagnostics saved -> {args.out_prefix}_diagnostics.json (labeled={diag['total_labeled']})")

if __name__ == '__main__':
    main()

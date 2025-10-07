"""Analyze LLM labels vs heuristic scores.

Inputs:
  --scores  scored_candidates.csv (or any CSV containing id, score_total + axis cols)
  --labels  JSONL produced by llm_label.py (each line: {"id":..., "model_output": {...}})

Outputs:
  - metrics JSON summary (precision/recall at multiple score percentiles)
  - optional matplotlib plot of score vs. relevance probability approximations
  - aggregated CSV merging heuristic + label fields for downstream calibration

Usage:
  python research/literature_search/scripts/label_analysis.py \
      --scores research/literature_search/data/scored_candidates.csv \
      --labels research/literature_search/data/llm_labels.jsonl \
      --out-prefix research/literature_search/data/analysis \
      --plot

Simplified metrics:
  * Treat label mapping: relevant=1, maybe=0.5, irrelevant=0
  * Compute mean label value per decile & specified percentiles (80,85,90,95)
  * Precision/Recall: define positive = relevant; maybe ignored for precision (optional weighting)
"""
from __future__ import annotations
import csv, json, argparse, os, math
from typing import Dict, List, Any

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:
    plt = None  # type: ignore

LABEL_NUM = {"relevant": 1.0, "maybe": 0.5, "irrelevant": 0.0}


def load_scores(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        dr = csv.DictReader(f)
        for r in dr:
            try:
                r['score_total'] = float(r.get('score_total') or 0)
            except Exception:
                r['score_total'] = 0.0
            rows.append(r)
    # Sort descending by heuristic score
    rows.sort(key=lambda x: x['score_total'], reverse=True)
    return rows


def load_labels(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rid = obj.get('id')
            mo = obj.get('model_output') or {}
            if rid:
                out[rid] = mo
    return out


def compute_metrics(merged: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Build arrays for label numeric
    label_vals = [LABEL_NUM.get(m.get('relevance_label', 'irrelevant'), 0.0) for m in merged]
    scores = [m['score_total'] for m in merged]
    total = len(merged)
    if total == 0:
        return {}

    # Decile averages
    decile_stats = []
    for d in range(1, 11):
        cut = math.ceil(total * d / 10)
        subset = label_vals[:cut]
        decile_stats.append({
            'decile': d,
            'count': cut,
            'mean_label_value': sum(subset)/len(subset)
        })

    # Percentile thresholds of interest
    percentiles = [0.80, 0.85, 0.90, 0.95]
    pct_stats = []
    relevant_positions = [i for i,m in enumerate(merged) if m.get('relevance_label') == 'relevant']
    total_relevant = len(relevant_positions)
    for p in percentiles:
        k = int(total * p)
        subset = merged[:k]
        rel_in_top = sum(1 for m in subset if m.get('relevance_label') == 'relevant')
        maybe_in_top = sum(1 for m in subset if m.get('relevance_label') == 'maybe')
        precision = rel_in_top / max(1, len(subset))
        recall = rel_in_top / max(1, total_relevant) if total_relevant else 0.0
        pct_stats.append({
            'percentile': p,
            'k': k,
            'precision_relevant': precision,
            'recall_relevant': recall,
            'maybe_fraction_in_top': maybe_in_top / max(1, len(subset)),
        })

    return {
        'total_labeled': total,
        'total_relevant': total_relevant,
        'deciles': decile_stats,
        'percentiles': pct_stats,
    }


def plot_curve(merged: List[Dict[str, Any]], out_prefix: str):
    if plt is None:
        print("[WARN] matplotlib not installed; skipping plot.")
        return
    xs = list(range(1, len(merged)+1))
    ys = [LABEL_NUM.get(m.get('relevance_label', 'irrelevant'), 0.0) for m in merged]
    plt.figure(figsize=(7,3))
    plt.plot(xs, ys, '.', markersize=3, alpha=0.6)
    plt.xlabel('Rank (heuristic score descending)')
    plt.ylabel('Label numeric (relevant=1 maybe=0.5 irrelevant=0)')
    plt.title('LLM Label vs Heuristic Rank')
    plt.tight_layout()
    out_path = f"{out_prefix}_label_vs_rank.png"
    plt.savefig(out_path, dpi=140)
    print(f"Plot saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    scores = load_scores(args.scores)
    labels = load_labels(args.labels)

    # Merge on id; retain only those with labels
    merged = []
    for row in scores:
        rid = row.get('id') or row.get('openalex_id')
        if rid in labels:
            mo = labels[rid]
            merged.append({
                'id': rid,
                'score_total': row['score_total'],
                'relevance_label': mo.get('relevance_label'),
                'confidence': mo.get('confidence'),
                'rationale': mo.get('rationale'),
                **{k: row.get(k) for k in row.keys() if k not in ('abstract','title')},
            })

    metrics = compute_metrics(merged)
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    with open(f"{args.out_prefix}_merged.csv", 'w', encoding='utf-8', newline='') as f:
        if merged:
            headers = list(merged[0].keys())
            cw = csv.DictWriter(f, fieldnames=headers)
            cw.writeheader()
            cw.writerows(merged)
    with open(f"{args.out_prefix}_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {args.out_prefix}_metrics.json")

    if args.plot:
        plot_curve(merged, args.out_prefix)

if __name__ == '__main__':
    main()

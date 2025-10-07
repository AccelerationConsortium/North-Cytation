"""Quantile-based heuristic analysis to recommend an LLM triage cutoff.

Purpose:
  Avoid choosing an arbitrary top-X% purely on intuition by measuring how the
  heuristic-defined "workflow_candidate" density changes with score percentile.

Heuristic (same as segment_sampling):
  device_like: device/materials terms (perovskite, led, sensor, etc.)
  strong_polymer: polymer_specificity >= 0.5
  workflow_candidate: strong_polymer AND NOT device_like

Outputs:
  - A CSV with rows per evaluated percentile threshold
  - Console summary including a recommended knee (max F1 vs heuristic positives)
  - Optional basic cost model (cost_per_llm) and budget scenario

NOTE:
  These are proxy metrics (weak labels). Before finalizing a cutoff, take a
  small stratified random sample above and below the proposed threshold for a
  quick manual sanity check to ensure no systematic false negatives.
"""
from __future__ import annotations
import argparse, csv, os, math
from typing import List, Dict

CFG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'literature_pipeline_config.yaml'))

DEVICE_TERMS = [
    'perovskite',' led','led ','light-emitting diode','sensor','mos2','tio2',' cds','cds ',
    'graphene oxide','quantum dot','quantum dots'
]

def load_yaml(path: str):
    import yaml
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)

def read_scored(path: str) -> List[Dict]:
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        dr=csv.DictReader(f)
        for r in dr:
            try: r['score_total']=float(r.get('score_total',0) or 0)
            except: r['score_total']=0.0
            try: r['polymer_specificity']=float(r.get('polymer_specificity',0) or 0)
            except: r['polymer_specificity']=0.0
            rows.append(r)
    rows.sort(key=lambda x:x['score_total'], reverse=True)
    return rows

def attach_abstracts(rows, parsed_jsonl: str):
    import json
    if not os.path.exists(parsed_jsonl):
        return rows
    idx={r.get('id'): r for r in rows if r.get('id')}
    with open(parsed_jsonl,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: obj=json.loads(line)
            except: continue
            rid=obj.get('id')
            if rid in idx:
                idx[rid]['abstract']=obj.get('abstract') or ''
                idx[rid]['title']=obj.get('title') or idx[rid].get('title')
    return rows

def is_device_like(text_l: str) -> bool:
    return any(t in text_l for t in DEVICE_TERMS)

def classify(row) -> bool:
    text_l = ((row.get('title') or '') + ' ' + (row.get('abstract') or '')).lower()
    strong_polymer = float(row.get('polymer_specificity',0)) >= 0.5
    return strong_polymer and not is_device_like(text_l)

def wilson_interval(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = p + z**2/(2*n)
    margin = z*math.sqrt((p*(1-p) + z**2/(4*n))/n)
    low = (center - margin)/denom
    high = (center + margin)/denom
    return max(low,0.0), min(high,1.0)

def evaluate(rows, percentiles: List[float]):
    total_candidates = 0
    # Precompute classification for entire dataset
    for r in rows:
        r['_workflow_candidate'] = classify(r)
        total_candidates += int(r['_workflow_candidate'])
    results=[]
    n_total=len(rows)
    for pct in percentiles:
        k = max(1, int(n_total * pct))
        subset = rows[:k]
        hits = sum(int(r['_workflow_candidate']) for r in subset)
        precision = hits / k if k else 0.0
        recall = hits / total_candidates if total_candidates else 0.0
        low, high = wilson_interval(hits, k)
        f1 = (2*precision*recall/(precision+recall)) if (precision+recall)>0 else 0.0
        results.append({
            'percentile': pct,
            'k_selected': k,
            'heuristic_hits': hits,
            'heuristic_total': total_candidates,
            'precision': round(precision,4),
            'precision_low95': round(low,4),
            'precision_high95': round(high,4),
            'recall': round(recall,4),
            'f1': round(f1,4)
        })
    return results, total_candidates

def recommend_cutoff(rows, evaluated):
    # Strategy: find percentile with (a) f1 within 95% of max f1, (b) precision > global mean precision, (c) k not exceeding 2 * sqrt(total_candidates * len(rows)) heuristic (diminishing returns heuristic)
    if not evaluated:
        return None
    max_f1 = max(r['f1'] for r in evaluated)
    global_precision = sum(int(r['_workflow_candidate']) for r in rows)/len(rows)
    total_candidates = sum(int(r['_workflow_candidate']) for r in rows)
    size_heuristic_limit = int(2 * math.sqrt(max(1,total_candidates) * len(rows)))
    # Filter candidates
    viable=[r for r in evaluated if r['f1'] >= 0.95*max_f1 and r['precision'] > global_precision and r['k_selected'] <= size_heuristic_limit]
    if not viable:
        # fallback: highest f1
        return max(evaluated, key=lambda x:x['f1'])
    # choose smallest k for efficiency among viable
    viable.sort(key=lambda x:(x['k_selected'], -x['precision']))
    return viable[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--percentiles', default='0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.12,0.15,0.18,0.20,0.25,0.30', help='Comma list of cumulative percentiles to evaluate (0-1).')
    ap.add_argument('--out', default='research/literature_search/data/quantile_analysis.csv')
    args = ap.parse_args()

    cfg = load_yaml(CFG_PATH)
    fetch_cfg = cfg.get('fetch',{})
    scored_csv = fetch_cfg.get('scored_csv')
    parsed_jsonl = fetch_cfg.get('parsed_jsonl')
    if not (scored_csv and os.path.exists(scored_csv)):
        print('Missing scored CSV; run score.py first.')
        return
    rows = read_scored(scored_csv)
    attach_abstracts(rows, parsed_jsonl)

    percentiles=[float(x.strip()) for x in args.percentiles.split(',') if x.strip()]
    evaluated, total_candidates = evaluate(rows, percentiles)
    rec = recommend_cutoff(rows, evaluated)

    # Write CSV
    fieldnames = list(evaluated[0].keys()) if evaluated else []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,'w',newline='',encoding='utf-8') as f:
        if fieldnames:
            w=csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in evaluated:
                w.writerow(r)
    print('=== Quantile Heuristic Analysis ===')
    print(f'Total records: {len(rows)} | Heuristic workflow candidates: {total_candidates}')
    for r in evaluated:
        print(f"pct={int(r['percentile']*100):>3}% k={r['k_selected']:>4} hits={r['heuristic_hits']:>3} precision={r['precision']:.3f} (95%CI {r['precision_low95']:.3f}-{r['precision_high95']:.3f}) recall={r['recall']:.3f} f1={r['f1']:.3f}")
    if rec:
        print('\nRecommended cutoff:')
        print(f"  ~{int(rec['percentile']*100)}% (k={rec['k_selected']}) precision≈{rec['precision']:.3f} recall≈{rec['recall']:.3f} f1≈{rec['f1']:.3f}")
    print(f'CSV -> {args.out}')

if __name__ == '__main__':
    main()

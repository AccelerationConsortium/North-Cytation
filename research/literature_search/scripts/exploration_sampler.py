"""Sample an exploration band between two score percentiles.

Purpose:
  After setting a primary gating cutoff (e.g., top 15%), draw a smaller
  random (or stratified) sample from just below (e.g., 15–22% or 10–18%)
  to probe what the heuristics might be missing.

Outputs:
  CSV and JSONL (optional) for LLM evaluation.

Usage examples:
  python exploration_sampler.py --lower 0.82 --upper 0.90 --sample-size 25 \
      --out-csv research/literature_search/data/exploration_band.csv

Notes:
  Percentiles use cumulative fraction (0–1). A row at percentile 0.90 means
  it sits at the 90th percentile threshold (top 10%). We select rows whose
  rank-derived cumulative fraction is in (lower, upper].
"""
from __future__ import annotations
import csv, argparse, os, random, json

CFG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'literature_pipeline_config.yaml'))

def load_yaml(path: str):
    import yaml
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)

def read_scored(path: str):
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        dr=csv.DictReader(f)
        for r in dr:
            try: r['score_total']=float(r.get('score_total',0) or 0)
            except: r['score_total']=0.0
            rows.append(r)
    rows.sort(key=lambda x:x['score_total'], reverse=True)
    return rows

def attach_abstracts(rows, parsed_jsonl: str):
    if not parsed_jsonl or not os.path.exists(parsed_jsonl):
        return rows
    import json
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lower', type=float, default=0.90, help='Lower cumulative percentile bound (exclusive)')
    ap.add_argument('--upper', type=float, default=0.95, help='Upper cumulative percentile bound (inclusive)')
    ap.add_argument('--sample-size', type=int, default=25, help='Number of rows to sample from band')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-csv', default='research/literature_search/data/exploration_band.csv')
    ap.add_argument('--out-jsonl', default='research/literature_search/data/exploration_band.jsonl')
    args = ap.parse_args()

    if not (0.0 <= args.lower < args.upper <= 1.0):
        raise SystemExit('Require 0 <= lower < upper <= 1')

    cfg = load_yaml(CFG_PATH)
    fetch_cfg = cfg.get('fetch',{})
    scored_csv = fetch_cfg.get('scored_csv')
    parsed_jsonl = fetch_cfg.get('parsed_jsonl')
    if not (scored_csv and os.path.exists(scored_csv)):
        print('Missing scored CSV; run score.py first.')
        return
    rows = read_scored(scored_csv)
    attach_abstracts(rows, parsed_jsonl)

    n_total=len(rows)
    band=[]
    for rank, r in enumerate(rows, start=1):
        # cumulative fraction: rank/n_total (1-based). A record with rank=1 has frac=1/n_total.
        # We want top-slice complement; simpler: compute percentile_from_top.
        # We'll treat fraction_from_top = rank / n_total.
        frac = rank / n_total  # 0..1 increasing with lower score
        # Convert to cumulative from top: top_frac = 1 - (rank-1)/n_total ≈ (n_total - rank + 1)/n_total
        # But simpler to define percentile threshold by rank position from top: pct_from_top = (rank-1)/n_total.
        pct_from_top = (rank-1)/n_total  # 0 for top record, approaches 1 near bottom
        # We want rows where pct_from_top is within (lower, upper], meaning they sit just below the main gate at 'lower'.
        if args.lower < pct_from_top <= args.upper:
            band.append(r)

    if not band:
        print('No rows in specified band; adjust bounds.')
        return

    random.seed(args.seed)
    sample = band if len(band) <= args.sample_size else random.sample(band, args.sample_size)

    # Write CSV
    fieldnames = ['rank','id','score_total','title','abstract']
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sample:
            rank = rows.index(r)+1
            w.writerow({'rank': rank,'id': r.get('id'),'score_total': r.get('score_total'),'title': r.get('title'),'abstract': r.get('abstract')})

    # Write JSONL
    with open(args.out_jsonl,'w',encoding='utf-8') as f:
        for r in sample:
            out = {
                'id': r.get('id'),
                'rank': rows.index(r)+1,
                'score_total': r.get('score_total'),
                'title': r.get('title'),
                'abstract': r.get('abstract')
            }
            f.write(json.dumps(out, ensure_ascii=False) + '\n')

    print(f'Sampled {len(sample)} rows from band ({args.lower*100:.1f}%, {args.upper*100:.1f}%).')
    print(f'CSV -> {args.out_csv}')
    print(f'JSONL -> {args.out_jsonl}')

if __name__ == '__main__':
    main()

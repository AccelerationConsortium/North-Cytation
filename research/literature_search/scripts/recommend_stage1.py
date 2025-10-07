"""Stage-1 recommendation gating.

Applies modular gating rules (percentile / top-K / threshold / hybrid) to the
scored abstract set and emits a recommended subset CSV for the next stage
(LLM full-text retrieval / evaluation).

Config section (literature_pipeline_config.yaml):
gating:
  method: percentile|top_k|threshold|hybrid
  percentile: 0.95
  top_k: 50
  min_score: 0.0
  axis_requirements: { capability_fit: 0.1 }
  max_recommend: 40
  output_csv: path/to/recommended_stage1.csv
  explain: true

Outputs a CSV sorted by total score desc.
"""
from __future__ import annotations
import csv, json, os, math, statistics

CFG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'literature_pipeline_config.yaml'))
SCORED_CSV = None  # resolved after cfg load

def load_yaml(path: str):
    import yaml
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)

def read_scored(path: str):
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        dr=csv.DictReader(f)
        for r in dr:
            try:
                r['score_total']=float(r['score_total'])
            except Exception:
                r['score_total']=0.0
            for ax in ('capability_fit','multi_objective','impact','parameter_space','novelty','constraint_penalty'):
                if ax in r:
                    try: r[ax]=float(r[ax])
                    except: r[ax]=0.0
            rows.append(r)
    rows.sort(key=lambda r:r['score_total'], reverse=True)
    return rows

def compute_percentile_cut(rows, p: float):
    if not rows:
        return 0.0
    p = max(0.0, min(1.0, p))
    idx = int((len(rows)-1)*p)
    return rows[idx]['score_total']

def apply_axis_requirements(rows, reqs: dict):
    if not reqs:
        return rows
    out=[]
    for r in rows:
        ok=True
        for axis,thr in reqs.items():
            val = float(r.get(axis,0.0))
            if val < float(thr):
                ok=False
                break
        if ok:
            out.append(r)
    return out

def gate(rows, gating_cfg: dict):
    method = gating_cfg.get('method','percentile')
    percentile = gating_cfg.get('percentile',0.95)
    top_k = int(gating_cfg.get('top_k',0))
    min_score = float(gating_cfg.get('min_score',0.0))
    axis_reqs = gating_cfg.get('axis_requirements',{}) or {}
    max_rec = int(gating_cfg.get('max_recommend',0) or 0)

    selected=[]
    if method=='percentile':
        thr = compute_percentile_cut(rows, percentile)
        selected = [r for r in rows if r['score_total'] >= thr and r['score_total'] >= min_score]
    elif method=='top_k':
        k = top_k if top_k>0 else 50
        selected = [r for r in rows if r['score_total'] >= min_score][:k]
    elif method=='threshold':
        selected = [r for r in rows if r['score_total'] >= min_score]
    elif method=='hybrid':
        # percentile first then ensure at least top_k
        thr = compute_percentile_cut(rows, percentile) if percentile else 0.0
        pool = [r for r in rows if r['score_total'] >= thr and r['score_total'] >= min_score]
        if top_k and len(pool) < top_k:
            pool = rows[:top_k]
        selected = pool
    else:
        raise ValueError(f"Unknown gating method: {method}")

    # Axis requirements filter
    selected = apply_axis_requirements(selected, axis_reqs)

    # Cap
    if max_rec and len(selected) > max_rec:
        selected = selected[:max_rec]
    return selected

def write_output(path: str, rows, explain: bool):
    if not rows:
        print("No recommendations produced.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base_fields = ['id','doi','title','year','cited_by_count','score_total']
    axis_fields = ['capability_fit','multi_objective','impact','parameter_space','novelty','constraint_penalty'] if explain else []
    fieldnames = base_fields + axis_fields
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out={k:r.get(k,'') for k in fieldnames}
            w.writerow(out)
    print(f"Wrote {len(rows)} recommendations -> {path}")

def summarize(rows, recommended):
    if not rows:
        print('No scored rows to summarize.')
        return
    vals=[r['score_total'] for r in rows]
    def q(p):
        if not vals: return 0
        idx=int((len(vals)-1)*p)
        return vals[idx]
    print("Score distribution:")
    for p in (0.5,0.75,0.85,0.90,0.95,0.98,0.99):
        print(f"  q{int(p*100):02d}: {q(p):.4f}")
    if recommended:
        print("Top 5 recommended previews:")
        for r in recommended[:5]:
            title=r['title'] or ''
            print(f"  {r['score_total']:.4f} | {title[:80]}")

def main():
    cfg = load_yaml(CFG_PATH)
    fetch_cfg = cfg.get('fetch',{})
    gating_cfg = cfg.get('gating',{})
    global SCORED_CSV
    SCORED_CSV = fetch_cfg.get('scored_csv')
    if not SCORED_CSV or not os.path.exists(SCORED_CSV):
        print('Scored CSV missing, run score.py first.')
        return
    rows = read_scored(SCORED_CSV)
    recs = gate(rows, gating_cfg)
    summarize(rows, recs)
    out_path = gating_cfg.get('output_csv','recommended_stage1.csv')
    write_output(out_path, recs, gating_cfg.get('explain', True))

if __name__ == '__main__':
    main()

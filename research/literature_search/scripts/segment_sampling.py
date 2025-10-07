"""Segment the ranked scored set into top/middle/bottom bands for manual QA.

Outputs a CSV + pretty text summary with 3 * n samples (default n=15):
  - Top n by score_total
  - Middle n around the median index
  - Bottom n (lowest scores)

Adds lightweight heuristic flags:
  - device_like (perovskite, led, sensor, mos2, tio2, cds, graphene oxide, quantum dot)
  - bio_complex (protein, cell, tissue, scaffold)
  - strong_polymer (polymer_specificity >= 0.5)
  - workflow_candidate (strong_polymer AND NOT device_like)

These are NOT gating decisions, just a quick precision snapshot before LLM triage.
"""
from __future__ import annotations
import csv, os, argparse, json

CFG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'literature_pipeline_config.yaml'))

def load_yaml(path: str):
    import yaml
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)

DEVICE_TERMS = [
    'perovskite',' led','led ','light-emitting diode','sensor','mos2','tio2',' cds','cds ', 'graphene oxide','quantum dot','quantum dots'
]
BIO_TERMS = ['protein','membrane','cell','cells','tissue','scaffold']

def read_scored(path: str):
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        dr=csv.DictReader(f)
        for r in dr:
            try: r['score_total']=float(r.get('score_total',0) or 0)
            except: r['score_total']=0.0
            for ax in ['polymer_specificity','capability_fit','impact','multi_objective','parameter_space']:
                if ax in r:
                    try: r[ax]=float(r[ax])
                    except: r[ax]=0.0
            rows.append(r)
    rows.sort(key=lambda x:x['score_total'], reverse=True)
    return rows

def attach_abstracts(rows, parsed_jsonl: str):
    # Build id index first for speed
    idx={r['id']: r for r in rows if r.get('id')}
    if not os.path.exists(parsed_jsonl):
        return rows
    with open(parsed_jsonl,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj=json.loads(line)
            except: continue
            rid=obj.get('id')
            if rid in idx:
                idx[rid]['abstract']=obj.get('abstract') or ''
    return rows

def flags_for(r):
    abs_l = (r.get('abstract') or '').lower()
    title_l = (r.get('title') or '').lower()
    text = title_l + ' ' + abs_l
    device_like = any(t in text for t in DEVICE_TERMS)
    bio_complex = any(t in text for t in BIO_TERMS)
    strong_polymer = float(r.get('polymer_specificity',0)) >= 0.5
    workflow_candidate = strong_polymer and not device_like
    return device_like, bio_complex, strong_polymer, workflow_candidate

def sample_segments(rows, n: int):
    if not rows:
        return [],[],[]
    top = rows[:min(n, len(rows))]
    mid_start = max(0, (len(rows)//2) - n//2)
    middle = rows[mid_start: mid_start + n]
    bottom = rows[-n:] if len(rows) >= n else rows
    return top, middle, bottom

def summarize_band(name: str, band, n_total):
    if not band:
        return {"band": name, "count": 0}
    device_like = 0
    bio_complex = 0
    strong_polymer = 0
    workflow_candidate = 0
    for r in band:
        d,b,s,w = flags_for(r)
        device_like += int(d)
        bio_complex += int(b)
        strong_polymer += int(s)
        workflow_candidate += int(w)
    return {
        "band": name,
        "count": len(band),
        "device_like": device_like,
        "bio_complex": bio_complex,
        "strong_polymer": strong_polymer,
        "workflow_candidate": workflow_candidate,
        "workflow_candidate_pct": round(100*workflow_candidate/len(band),1) if band else 0.0,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n','--count', type=int, default=15, help='Samples per band')
    ap.add_argument('--out', default='research/literature_search/data/segment_sample.csv')
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
    top, middle, bottom = sample_segments(rows, args.count)

    # Write combined CSV
    fieldnames = ['segment','rank','id','score_total','polymer_specificity','capability_fit','impact','multi_objective','parameter_space','title','device_like','bio_complex','strong_polymer','workflow_candidate']
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        def write_segment(seg_name, seg_list):
            for r in seg_list:
                d,b,s,wc = flags_for(r)
                w.writerow({
                    'segment': seg_name,
                    'rank': rows.index(r)+1,
                    'id': r.get('id'),
                    'score_total': r.get('score_total'),
                    'polymer_specificity': r.get('polymer_specificity'),
                    'capability_fit': r.get('capability_fit'),
                    'impact': r.get('impact'),
                    'multi_objective': r.get('multi_objective'),
                    'parameter_space': r.get('parameter_space'),
                    'title': r.get('title'),
                    'device_like': int(d),
                    'bio_complex': int(b),
                    'strong_polymer': int(s),
                    'workflow_candidate': int(wc),
                })
        write_segment('top', top)
        write_segment('middle', middle)
        write_segment('bottom', bottom)

    # Console summary
    print('=== Segment Summary ===')
    for name, seg in [('Top', top), ('Middle', middle), ('Bottom', bottom)]:
        stats = summarize_band(name, seg, len(rows))
        print(f"{stats['band']}: n={stats['count']} workflow_candidates={stats['workflow_candidate']} ({stats['workflow_candidate_pct']}%) strong_polymer={stats['strong_polymer']} device_like={stats['device_like']} bio_complex={stats['bio_complex']}")
    print(f"Detailed rows -> {args.out}")

if __name__ == '__main__':
    main()

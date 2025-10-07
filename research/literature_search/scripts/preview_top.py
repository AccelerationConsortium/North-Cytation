"""Preview top scored literature records.

Reads scored CSV + parsed JSONL and prints top N entries with key axes
including the newly added polymer_specificity axis. This bypasses gating
so you can inspect candidates even if gating thresholds return zero.
"""
from __future__ import annotations
import csv, json, os, argparse

CFG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'literature_pipeline_config.yaml'))

def load_yaml(path: str):
    import yaml
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)

def iter_jsonl(path: str):
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def load_scores(scored_csv: str):
    rows=[]
    with open(scored_csv,'r',encoding='utf-8') as f:
        dr = csv.DictReader(f)
        for r in dr:
            try: r['score_total']=float(r.get('score_total',0) or 0)
            except: r['score_total']=0.0
            for ax in ['polymer_specificity','capability_fit','impact','multi_objective','parameter_space']:
                if ax in r:
                    try: r[ax]=float(r[ax])
                    except: r[ax]=0.0
            rows.append(r)
    return rows

def build_index(rows):
    return {r['id']: r for r in rows if r.get('id')}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n','--top', type=int, default=15, help='Number of top records to show')
    ap.add_argument('--abstract-chars', type=int, default=220, help='Abstract preview char limit')
    args = ap.parse_args()

    cfg = load_yaml(CFG_PATH)
    fetch_cfg = cfg.get('fetch',{})
    parsed_path = fetch_cfg.get('parsed_jsonl')
    scored_csv = fetch_cfg.get('scored_csv')
    if not (parsed_path and os.path.exists(parsed_path) and scored_csv and os.path.exists(scored_csv)):
        print('Missing parsed or scored data. Run extract.py and score.py first.')
        return

    scored_rows = load_scores(scored_csv)
    scored_rows.sort(key=lambda r: r['score_total'], reverse=True)
    score_idx = build_index(scored_rows)

    # Attach abstracts
    enriched=[]
    for rec in iter_jsonl(parsed_path):
        rid = rec.get('id')
        if not rid or rid not in score_idx: continue
        row = dict(score_idx[rid])
        row['abstract'] = rec.get('abstract') or ''
        enriched.append(row)
    enriched.sort(key=lambda r: r['score_total'], reverse=True)

    top = enriched[:args.top]
    if not top:
        print('No records available.')
        return

    header = f"# Top {len(top)} by score_total (showing polymer_specificity, capability_fit, impact, multi_objective, parameter_space)"
    print(header)
    for i, r in enumerate(top, 1):
        abs_txt = (r.get('abstract','') or '').replace('\n',' ').strip()
        if len(abs_txt) > args.abstract_chars:
            abs_preview = abs_txt[:args.abstract_chars].rstrip() + '…'
        else:
            abs_preview = abs_txt
        print(f"\n[{i}] score={r['score_total']:.4f} poly_spec={r.get('polymer_specificity',0):.3f} cap_fit={r.get('capability_fit',0):.3f} impact={r.get('impact',0):.3f} multi={r.get('multi_objective',0):.3f} param={r.get('parameter_space',0):.3f}")
        title = (r.get('title') or '').strip()
        print(f"Title: {title}")
        print(f"DOI: {r.get('doi') or 'n/a'} | Year: {r.get('year')}")
        print(f"Abstract: {abs_preview}")
    print('\nDone.')

if __name__ == '__main__':
    main()

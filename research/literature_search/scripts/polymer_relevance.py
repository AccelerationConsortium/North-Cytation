import json, pathlib, re, collections

DATA = pathlib.Path(__file__).resolve().parent.parent / 'data' / 'papers_parsed.jsonl'
records = [json.loads(l) for l in DATA.open(encoding='utf-8')]

poly_tokens = [
    'polymer','polymeric','polymerization','photopolymer','photopolymerization',
    'raft','pet-raft','radical polymerization','living polymerization','crosslink','cross-link'
]
# domain hints that often indicate non-polymer-centric focus (diagnostic only)
non_poly_hints = {
    'communication':'comm_net',
    'cancer therapy':'med_pharma',
    'chemotherapy':'med_pharma',
    'phototropin':'plant_bio',
    'nitrogen fixation':'inorganic_cat',
    'hydrogen production':'energy_cat',
    'borylation':'org_syn',
    'phase separation':'perovskite',
    'photoactivated nanomedicines':'med_pharma',
}

def has_any(text, toks):
    tl = text.lower()
    return any(t in tl for t in toks)

out = []
for r in records:
    text = (r.get('title','') + ' ' + (r.get('abstract') or '')).lower()
    poly = has_any(text, poly_tokens)
    tags = []
    if poly:
        tags.append('poly_hit')
    for hint,label in non_poly_hints.items():
        if hint in text:
            tags.append(label)
    out.append((r['id'], r.get('title','')[:70], poly, tags))

poly_count = sum(1 for _,_,p,_ in out if p)

print('Polymer-positive heuristic:', poly_count, '/', len(out), f'({poly_count/len(out):.1%})')
print('\nDetails:')
for rid,title,flag,tags in out:
    print(('P' if flag else '-'), '|', title, '|', ','.join(tags) if tags else 'none')

# crude differentiation suggestion metrics
lengths = [len((r.get('abstract') or '').split()) for r in records]
print('\nAbstract length (words) median:', sorted(lengths)[len(lengths)//2])

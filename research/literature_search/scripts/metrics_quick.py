import json, statistics, pathlib, collections
p = pathlib.Path(__file__).resolve().parent.parent / 'data' / 'papers_parsed.jsonl'
recs = [json.loads(l) for l in p.open(encoding='utf-8')]
N = len(recs)
num = [r for r in recs if r.get('numeric_factors')]
cap = [r for r in recs if r.get('capability_tokens')]
scored = [r.get('scores', {}) for r in recs]

def fetch_axis(name):
    return [s.get(name, 0) for s in scored if name in s]

impact = fetch_axis('impact')
param = fetch_axis('parameter_space')
multi = fetch_axis('multi_objective')
nov = fetch_axis('novelty')
capfit = fetch_axis('capability_fit')

f = lambda xs: {
    'min': min(xs) if xs else 0,
    'mean': (sum(xs)/len(xs)) if xs else 0,
    'median': statistics.median(xs) if xs else 0,
    'max': max(xs) if xs else 0,
}

verbs = collections.Counter(v for r in recs for v in r.get('capability_tokens', []))

report = {
    'counts': {
        'total_records': N,
        'with_numeric_factors': len(num),
        'with_capability_tokens': len(cap),
    },
    'coverage': {
        'numeric_factors_pct': round(len(num)/N*100, 1) if N else 0,
        'capability_tokens_pct': round(len(cap)/N*100, 1) if N else 0,
    },
    'axes': {
        'impact': f(impact),
        'parameter_space': f(param),
        'multi_objective': f(multi),
        'novelty': f(nov),
        'capability_fit': f(capfit),
    },
    'top_capability_tokens': verbs.most_common(15),
}

print(json.dumps(report, indent=2))

"""Stage-1 heuristics diagnostics.

Reads the current parsed + scored abstracts and produces a diagnostic report
to evaluate whether each heuristic axis is providing useful signal at the
abstract-only stage.

Focus Areas:
 1. Axis coverage & distribution (fraction non-zero, mean, median)
 2. Capability / objective token utilization (hits vs configured lists)
 3. Numeric factor extraction pattern mix (range/ratio/concentration/equivalents)
 4. Penalty incidence (novelty, constraint_penalty)
 5. Simple retain/adjust/defer recommendations per axis

Outputs:
  - JSON report: data/diagnostics_stage1.json
  - Human-readable console summary

Usage:
  Run after `extract.py` then `score.py`.
"""
from __future__ import annotations
import json, os, statistics, collections, math

HERE = os.path.abspath(os.path.dirname(__file__))
CFG_PATH = os.path.abspath(os.path.join(HERE, '..', 'literature_pipeline_config.yaml'))
DATA_DIR = os.path.abspath(os.path.join(HERE, '..', 'data'))
PARSED = os.path.join(DATA_DIR, 'papers_parsed.jsonl')
OUTPUT_JSON = os.path.join(DATA_DIR, 'diagnostics_stage1.json')


def load_yaml(path: str):
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def iter_jsonl(path: str):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def basic_stats(values):
    if not values:
        return {"count":0,"non_zero":0,"frac_non_zero":0.0,"mean":0.0,"median":0.0,"max":0.0}
    nz = [v for v in values if v>0]
    return {
        "count": len(values),
        "non_zero": len(nz),
        "frac_non_zero": round(len(nz)/len(values),3) if values else 0.0,
        "mean": round(sum(values)/len(values),4),
        "median": round(statistics.median(values),4),
        "max": round(max(values),4),
    }


def recommendation(axis_name: str, stats: dict) -> str:
    f = stats.get('frac_non_zero',0)
    if axis_name in {"parameter_space"} and f < 0.1:
        return "defer_or_expand_patterns"  # not enough abstract signal yet
    if axis_name in {"novelty","constraint_penalty"} and f < 0.05:
        return "expand_terms_or_temporarily_downweight"
    if f == 0:
        return "inactive_remove_or_fix"
    if f > 0.7 and stats.get('median',0) > 0.5:
        return "high_coverage_check_discriminative_power"
    return "retain"


def main():
    cfg = load_yaml(CFG_PATH)
    recs = list(iter_jsonl(PARSED))
    if not recs:
        print("No parsed records found; run extract+score first.")
        return

    # Axis stats
    axis_names = [
        "capability_fit","parameter_space","multi_objective",
        "impact","novelty","constraint_penalty"
    ]
    axis_stats = {}
    for a in axis_names:
        vals = []
        for r in recs:
            s = r.get('scores') or {}
            v = s.get(a)
            if v is not None:
                vals.append(float(v))
        axis_stats[a] = basic_stats(vals)
        axis_stats[a]['recommendation'] = recommendation(a, axis_stats[a])

    # Token utilization
    ext_cfg = cfg.get('extraction',{})
    configured_capability = set(v.lower() for v in ext_cfg.get('capability_verbs',[]))
    configured_objective = set(v.lower() for v in ext_cfg.get('objective_verbs',[]))
    observed_capability = collections.Counter(
        t for r in recs for t in (r.get('capability_tokens') or [])
    )
    observed_objective = collections.Counter(
        t for r in recs for t in (r.get('objective_terms') or [])
    )

    unused_capability = sorted(configured_capability - set(observed_capability))
    unused_objective = sorted(configured_objective - set(observed_objective))

    # Numeric factor patterns
    pattern_counts = collections.Counter()
    any_numeric = 0
    for r in recs:
        facs = r.get('numeric_factors') or []
        if facs:
            any_numeric += 1
        for fct in facs:
            kind = fct.get('kind','unknown')
            pattern_counts[kind]+=1

    numeric_summary = {
        "records_with_numeric": any_numeric,
        "total_records": len(recs),
        "coverage_pct": round(any_numeric/len(recs)*100,1) if recs else 0.0,
        "pattern_distribution": pattern_counts.most_common(),
    }

    # Penalty incidence detail
    novelty_hits = sum(1 for r in recs if (r.get('scores') or {}).get('novelty',0) > 0)
    constraint_hits = sum(1 for r in recs if (r.get('scores') or {}).get('constraint_penalty',0) > 0)
    penalty_incidence = {
        "novelty_records": novelty_hits,
        "constraint_records": constraint_hits,
        "novelty_pct": round(novelty_hits/len(recs)*100,1),
        "constraint_pct": round(constraint_hits/len(recs)*100,1),
    }

    report = {
        "counts":{"records":len(recs)},
        "axis_stats": axis_stats,
        "tokens": {
            "capability_observed_top10": observed_capability.most_common(10),
            "objective_observed_top10": observed_objective.most_common(10),
            "capability_unused": unused_capability,
            "objective_unused": unused_objective,
        },
        "numeric_factors": numeric_summary,
        "penalties": penalty_incidence,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_JSON,'w',encoding='utf-8') as f:
        json.dump(report,f,indent=2)

    # Console summary
    print("=== Stage-1 Heuristics Diagnostics ===")
    print(f"Records: {len(recs)}")
    print("-- Axis Coverage --")
    for a,st in axis_stats.items():
        print(f" {a:18s} nonzero={st['non_zero']:2d} ({st['frac_non_zero']*100:4.1f}%) median={st['median']:.3f} rec={st['recommendation']}")
    print("-- Numeric Factors --")
    print(f" Coverage: {numeric_summary['coverage_pct']}%  distribution={numeric_summary['pattern_distribution']}")
    print("-- Capability Tokens (top 10) --")
    for tok,cnt in observed_capability.most_common(10):
        print(f"  {tok}: {cnt}")
    print(" Unused capability tokens:", ", ".join(unused_capability) if unused_capability else "<none>")
    print("-- Objective Terms (top 10) --")
    for tok,cnt in observed_objective.most_common(10):
        print(f"  {tok}: {cnt}")
    print(" Unused objective verbs:", ", ".join(unused_objective) if unused_objective else "<none>")
    print("-- Penalties --")
    print(f" Novelty incidence: {penalty_incidence['novelty_pct']}% | Constraint incidence: {penalty_incidence['constraint_pct']}%")
    print(f"Report written: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

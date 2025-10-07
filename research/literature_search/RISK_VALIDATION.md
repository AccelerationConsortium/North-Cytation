# Risk & Validation Plan

Objective: Provide a lightweight, repeatable framework to assess whether literature ranking outputs are trustworthy and to surface systemic failure modes early.

## 1. Key User-Facing Promises
1. High-ranked abstracts are relevant to a targeted experimental archetype.
2. Extracted numeric factors reflect potential parameter ranges worth exploring.
3. Automation/novelty penalties down-rank already well-covered autonomous efforts.
4. Re-running with unchanged config yields stable ranks (idempotence barring new data).

## 2. Primary Failure Modes
| ID | Failure | Description | Likely Cause | Impact | Mitigation |
|----|---------|-------------|--------------|--------|------------|
| F1 | Irrelevant high-rank | Top-K includes off-domain papers | Over-broad keywords | Wasted manual review | Tighten archetype tokens; add exclusion filter |
| F2 | Missed numeric ranges | Few/zero factor extractions | Regex too narrow | Narrow experiment design space | Expand patterns & test set |
| F3 | Over-penalized novelty | Legit new space penalized | Aggressive automation keywords | Lost opportunity | Add whitelist phrases; scale penalty by density |
| F4 | Duplicate crowding | Near-duplicates cluster at top | No de-dup logic | Low diversity of ideas | Title similarity dedup pass |
| F5 | Citation dominance | Older/highly cited overshadow new | Impact weight too strong | Bias to mature topics | Add recency decay axis |
| F6 | Solvent misflag | Benign solvent marked non-whitelist | Incomplete whitelist | False constraints | Normalize solvent names, update list |
| F7 | Config drift unnoticed | Silent changes alter ranking | Lack of provenance | Untraceable regressions | Config hash + run summary diff |
| F8 | Scaling instability | Axis distributions collapse (0/1 only) | Poor normalization | Reduced discriminative power | Dynamic quantile scaling option |
| F9 | Automation claim miss | Fails to penalize obvious self-driving claims | Phrase variation | Inflated novelty score | Add embedding/semantic match |

## 3. Validation Metrics (MVP)
| Metric | Definition | Collection | Target (Initial) |
|--------|------------|-----------|------------------|
| Precision@20 | Relevant among top 20 | Manual label sheet | ≥ 0.60 |
| Factor Coverage | % abstracts with ≥1 numeric_factors | Automated count | ≥ 0.30 |
| Objective Diversity | Mean distinct objective_terms in top 20 | Automated | ≥ 1.5 |
| Rank Stability | Jaccard overlap top 20 across identical config rerun | Two successive runs | ≥ 0.90 |
| Novelty Penalty Incidence | % top 50 with any novelty hit | Automated | ≤ 0.25 |
| Solvent Flag Rate | % total with non_whitelist_solvent flag | Automated | < 0.15 (else whitelist review) |

## 4. Manual Labeling Protocol
1. Export top 30 rows from `scored_candidates.csv`.
2. For each: label Relevance (Y/N), Numeric Signal (Y/N), Automation Claim (Y/N), Comments.
3. Compute Precision@K (5,10,20) and note failure examples.
4. Feed misclassifications back into keyword or regex adjustments.

Template CSV columns (manual): `id,doi,title,relevant,has_numeric,automation_claim,notes`.

## 5. Evaluation Cadence
| Phase | Frequency | Activities |
|-------|-----------|------------|
| MVP Initial | Once after first real fetch | Full metric set |
| Iteration Cycle | After each config weight/pattern change | Precision@10, Factor Coverage |
| Pre-Release 0.5.x | Before merging Phase 1 features | Full metric set + stability |

## 6. Threshold Adjustment Logic
If Precision@20 < 0.60: tighten tokens_any (remove ambiguous terms) OR add mandatory_all tokens.
If Factor Coverage < 0.30: broaden range/conc/equiv regex patterns (add unit variants). Test on cached abstracts.
If Novelty Penalty Incidence > target: narrow automation phrase list or reduce penalty weight.
If Rank Stability < 0.90 with unchanged config hash: investigate nondeterministic code path (random ordering) and enforce sort tie-breaker (e.g., id).

## 7. Reproducibility & Drift Detection
Component: `run_summary.json` (config_hash, counts, weights). Compare new hash to previous; if unchanged but score distributions shift (KS test / simple mean diff > 0.1), flag for investigation.

## 8. Exit Criteria for MVP Validation
All target thresholds met OR written rationale for any miss + mitigation plan scheduled.

## 9. Future Metric Extensions (Phase 1+)
- Recall proxy via broadened sample labeling (random 50 mid-ranked abstracts)
- Diversity score (pairwise similarity if embeddings introduced)
- Time-normalized impact (citation velocity)

## 10. Risk Register Updates
Update this document whenever a new systematic error is observed twice; assign ID F10+ and mitigation owner.

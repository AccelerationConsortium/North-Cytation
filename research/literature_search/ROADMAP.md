# Literature Pipeline Roadmap

Purpose: Maintain a clear separation between the minimal viable pipeline (usable now) and staged enhancements so we can iterate without rewriting core code.

## Version Mapping (Current)
- 0.3.x: Mock fetch + extraction + scoring + run summary (no external API calls)
- Next (0.4.x): Real OpenAlex fetch (limited sample), validation loop

---
## MVP (Release 0.4.x Target)
Scope required before using rankings to guide experiment ideation.

Included:
1. Real OpenAlex fetch (rate-limited, pagination, year filter, exclude reviews)
2. Inverted index abstract resolution to plain text
3. Existing extraction heuristics (numeric factors, objectives, capability tokens, solvents)
4. Current axis scoring (capability_fit, parameter_space, multi_objective, impact, novelty, constraint_penalty)
5. Run summary with config hash & counts
6. Manual validation protocol (top 20 inspection) documented
7. CHANGELOG + semantic version bump

Explicitly Deferred:
- Embeddings / semantic similarity
- Active learning feedback loop
- Full-text PDF ingestion
- Duplicate clustering / near-duplicate detection

Exit Criteria:
- Precision@20 (manual) judged "acceptable" (>=60% clearly relevant)
- >=30% of abstracts produce at least one numeric_factors hit
- No silent failures (non-empty run summary, scored CSV exists)

---
## Phase 1 Enhancements (0.5.x – 0.6.x)
Objective: Improve relevance and reduce manual curation friction.

Planned:
1. Config-driven axis registry (axes declared in YAML w/ strategy types)
2. Citation recency decay (age-adjusted impact)
3. Solvent whitelist penalty integration into scoring (currently just flag)
4. Lightweight semantic expansion: synonym list injection (no embeddings yet)
5. Basic duplicate detection via title + normalized Levenshtein or hashing
6. Batch diff tool to compare two scored CSVs (rank shifts) using config hashes

Stretch:
- Optional CLI orchestrator `pipeline.py --stages fetch,extract,score`
- Export of top-K per archetype CSV and JSON summary

Exit Criteria:
- Config only edits required for weight or axis addition (no code changes)
- Demonstrated improvement in manual Precision@20 vs MVP baseline

---
## Phase 2 Enhancements (0.7.x – 0.9.x)
Objective: Introduce semantic intelligence & adaptive ranking.

Planned:
1. Embedding-based capability_fit (sentence-transformers or similar) with caching
2. Active learning loop: label interface + incremental model for relevance/novelty
3. Clustering of high-similarity abstracts to diversify top-K suggestions
4. Learning-to-rank experiment (pairwise preference fine-tuning if labels available)
5. Constraint inference model (classify likely inert atmosphere, temperature sensitivity)

Stretch:
- PDF full-text ingestion & section-aware extraction (experimental)
- Graph enrichment (citation network centrality features)

Exit Criteria:
- Measurable uplift (≥10% relative improvement) in labeled relevance over Phase 1
- Top-K list diversity (≤30% pairwise cosine similarity average within top 20)

---
## Phase 3 (1.0.0 Considerations)
Objective: Production stability and broader automation integration.

Planned:
1. Stable API / Python package interface for consuming ranked results
2. Continuous validation job (nightly small sample fetch to detect drift)
3. Metric dashboard (coverage, precision, trend of axis distributions)
4. Automated config diff & regression alert (score distribution Z-test)

---
## Risk Mitigations Summary
- Overfitting to polymer vocabulary: keep keyword sets modular & versioned
- Config drift: config hash + run summary gating merges
- Extraction false positives: maintain precision-focused regex; expand only after review
- Citation skew: log scaling + future recency normalization

---
## Immediate Next Steps
1. Implement real fetch (OpenAlex) with abstraction boundary
2. Add manual validation template (feed into Risk & Validation doc)
3. Integrate whitelist penalty into scoring (config gate)

---
## Change Log Tie-In
Each roadmap milestone triggers a minor version bump; breaking config schema changes increment minor or (pre-1.0) patch with clear migration notes.

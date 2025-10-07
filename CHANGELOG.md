# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-10-06
### Added
- Introduced `capabilities/` directory with:
  - `README.md` explaining literature-focused capability framing
  - `capabilities_catalog.yaml` listing core experimental capabilities
  - `constraints.yaml` defining cross-cutting operational limits
  - `solvent_whitelist.yaml` enumerating approved open-air solvents

### Notes
- Documentation-only addition; no runtime logic modified.

## [0.1.1] - 2025-10-06
### Added
- `capabilities/experiment_priority.md` defining:
  - Experiment archetype catalog
  - Inclusion / exclusion criteria for literature mining
  - Scoring axes and weighting formula
  - Query token mapping and extraction targets
### Notes
- Extends documentation to enable next step: automated literature ranking pipeline.

## [0.2.0] - 2025-10-06
### Changed
- Restructured repository: moved capability documentation into `research/capabilities/`.
- Introduced new `research/literature_search/` directory for literature mining pipeline scaffolding.

### Added
- `research/literature_search/README.md` describing pipeline data flow.
- `research/literature_search/literature_pipeline_config.yaml` centralizing fetch, extraction, scoring parameters.
- Script stubs: `fetch.py`, `extract.py`, `score.py`, `pipeline.py` under `research/literature_search/scripts/`.
- Sample schema record: `research/literature_search/data/sample_schema.jsonl`.
- `.gitignore` for raw data & logs in literature search directory.

### Notes
- Version bump minor due to directory restructuring and new (non-executable) pipeline scaffolding; no core experimental runtime logic altered.

## [0.3.0] - 2025-10-06
### Added
- Implemented functional literature pipeline stages (test-mode):
  - `fetch.py` now generates mock OpenAlex-like records (with DOI) honoring test limits.
  - `extract.py` performs regex-based numeric factor, objective term, capability verb, and solvent mention extraction.
  - `score.py` computes axis scores (capability_fit, parameter_space, multi_objective, impact, novelty, constraint_penalty) and outputs `scored_candidates.csv`.
  - `utils.py` centralizes config loading, JSONL IO, regex helpers, scaling, and a pluggable axis strategy skeleton.
  - Added solvent whitelist integration; extraction flags `non_whitelist_solvent` (placeholder logic for future penalty usage).
  - Added `__init__.py` and import fallbacks to allow direct script execution without installing package.

### Changed
- Extraction & scoring scripts now overwrite parsed JSONL with embedded `scores` object for provenance.

### Notes
- Pipeline currently operates in mock/test mode only (no real HTTP requests). Real fetch logic can replace `_mock_openalex_query` later with minimal changes.
- Axis computation intentionally lightweight to encourage future config-driven refactors (e.g., declarative axis registry in YAML).

## [0.4.0] - 2025-10-06
### Added
- Real OpenAlex fetch integration in `research/literature_search/scripts/fetch.py` (replacing mock-only mode when `test_mode: false`).
- Fallback simplified query and iterative per-token union retrieval strategy to mitigate zero-result broad queries.
- Embedding of `query_token` in fetched records for basic provenance of which token produced each hit.

### Changed
- Pipeline now capable of producing real dataset (20-record initial batch) instead of mock-only output.

### Notes
- Minor risk: broader queries may introduce noise; mitigated by later scoring refinement.

## [0.4.1] - 2025-10-06
### Changed
- Collapsed multiple archetypes into a single broad `polymer_photochemistry_broad` archetype to reduce early selection bias.
- Expanded extraction regex patterns (wavelength, power density, time units, concentration variants) and capability verb list.
- Cleaned previous mixed mock/real data prior to new run to avoid dataset contamination.

### Added
- Quick coverage metrics script `research/literature_search/scripts/metrics_quick.py` to report extraction & axis distribution statistics.

### Metrics (first broad real run)
- Records: 20
- Numeric factor coverage: 5% (1/20)
- Capability token coverage: 75% (15/20)
- Impact score range: 0.0–1.0 (median ~0.49)
- Parameter space scores low (median 0) due to sparse numeric factor extraction.

### Next Suggested Improvements (Not yet implemented)
- Add single-value numeric pattern capture (e.g., standalone concentrations/temperatures without explicit ranges or units trailing immediately).
- Consider synonym normalization (activate/activation) to reduce token fragmentation.
- Introduce citation age weighting (decay) for impact axis optional refinement.
- Prepare lightweight manual relevance labeling template to calibrate weights.

## [0.4.2] - 2025-10-06
### Added
- `diagnostics_stage1.py` script (Stage‑1 heuristics health check) summarizing:
  - Axis coverage: non-zero counts, mean/median values
  - Token utilization: capability & objective verbs hit vs configured
  - Numeric factor pattern distribution (range/ratio/concentration/equivalents)
  - Penalty axis incidence (novelty, constraint_penalty)
  - Automatic recommendations (retain / adjust / defer) per axis
- Outputs machine-readable report at `research/literature_search/data/diagnostics_stage1.json` plus console summary.

### Guidance
- If `parameter_space` non-zero coverage <10%: recommended temporary weight reduction to near 0 until single-value numeric extraction added.
- If `novelty` and `constraint_penalty` both <5% incidence: consider deferring penalties or expanding phrase/flag sources.

### Notes
- Purely analytical addition (no change to scoring logic). Safe to run after any extract+score pass.

## [0.4.3] - 2025-10-06
### Changed
- Increased `max_per_archetype` fetch limit from 50 to 100 for broader abstract sample.
- Reweighted axes: increased `capability_fit` (0.35) & `impact` (0.35); set `parameter_space` & `novelty` weights to 0 (insufficient abstract signal); modest bump to `multi_objective` (0.20); retained small `constraint_penalty` (0.10) for future flag expansion.

### Rationale
- Diagnostics (50-record set) showed parameter_space (2% coverage) & novelty (0%) non-informative at abstract stage.
- Emphasis shifted to axes with meaningful gradient (capability_fit, impact, multi_objective) while preserving placeholder penalty axis.

### Next
- After expanding to ~100 records: re-run diagnostics to confirm stability & check for any drift in capability token distributions.

## [0.4.4] - 2025-10-06
### Added
- Pagination enhancement for iterative union OpenAlex fetch (multi-page per token) and increased `max_per_archetype` to 300 (fetched 332 unique abstracts).
- `gating` config section enabling modular recommendation strategies (percentile / top_k / threshold / hybrid) with axis requirements and caps.
- `recommend_stage1.py` script producing `recommended_stage1.csv` (Stage-1 candidates for full-text / LLM evaluation).

### Defaults
- Current gating: percentile 0.95 (~top 5%), capability_fit ≥0.10, max_recommend=40.

### Rationale
- Larger corpus established stable axis distributions; gating layer separates heuristic ranking from LLM cost decisions while keeping emphasis reconfigurable via YAML weights + gating params.

### Next
- Consider adding polymer_specificity axis and unit-level numeric extraction before reintroducing parameter_space weight.

## [0.4.5] - 2025-10-06
### Added
- `polymer_specificity` axis: term-density / uniqueness metric using expanded `polymer_terms` in `heuristics.yaml`; integrated into `extract.py` (fields `polymer_term_hits`, `polymer_term_unique`) and `score.py` with normalization cap (`polymer_specificity_cap: 8`).
- `union_tokens` fallback list in archetype config to broaden recall without over-expanding strict AND core tokens; applied during iterative per-token union retrieval in `fetch.py`.
- `segment_sampling.py` script: top/middle/bottom sampling with heuristic flags (`device_like`, `bio_complex`, `strong_polymer`, `workflow_candidate`) for quick precision diagnostics.
- `quantile_analysis.py` script: evaluates heuristic workflow_candidate precision/recall/F1 vs cumulative percentiles; recommends gating percentile based on efficiency plateau.
- `exploration_sampler.py` script: draws a random sample from a percentile band below the main gate for Phase‑2 LLM probing (missed candidates analysis).

### Changed
- Scoring weights reallocated: added weight to `polymer_specificity` (0.25), reduced `impact` (0.15), reintroduced small `parameter_space` weight (0.05) after broader corpus increased coverage (>10%).
- Gating percentile adjusted from 0.95 (top 5%) to 0.85 (top 15%) following quantile analysis (F1 and recall improvement with acceptable precision trade‑off). Axis requirements (capability_fit ≥ 0.10, polymer_specificity ≥ 0.20) retained.
- Heuristics YAML regex quoting fixed (single-quoted pattern) to restore numeric single-value extraction and enable proper polymer term loading (previous backslash escape issue suppressed pattern parsing and flatlined axis values at 0).

### Metrics (post-change, 508-record corpus)
- Heuristic workflow candidates (strong polymer & not device-like): 30 (~5.9%).
- Quantile snapshot: top 10% precision ≈0.22 recall ≈0.37; top 15% precision ≈0.17 recall ≈0.43; top 18% precision ≈0.22 recall ≈0.67 (chosen 15% as balanced initial gate with exploration option below threshold).

### Rationale
- Polymer specificity axis elevated genuinely polymer-centric modification / usage workflows into early ranking positions, displacing high-citation device abstracts.
- Percentile gating (vs fixed top_k) future-proofs scaling as corpus expands; exploration sampler enables evidence-driven adjustment (e.g., potential future tightening to 10–12% after adding a workflow_signal axis or penalty refinement).

### Next (Not in this version)
- Optional `workflow_signal` positive axis (protocol / transformation verbs: dialyze, casting, swelling, gelation, fractionation) to concentrate workflow precision and unlock lower percentile gating without recall loss.
- Post-LLM calibration loop: fit logistic model over axes using LLM relevance labels to recalibrate score_total to probability.
- Device penalty axis deferred pending empirical need (user preference to avoid over-specific negative heuristics prematurely).

## [0.4.6] - 2025-10-06
### Added
- `workflow_terms` and `device_terms` lists to `heuristics.yaml`.
- Extraction of `workflow_term_hits` / `device_term_hits` (+ unique variants) in `extract.py`.
- New axes in `score.py`: `workflow_boost` (positive), `device_penalty` (penalty-style, applied via negative weight).

### Changed
- Scoring weights updated: added `workflow_boost: 0.10`, `device_penalty: -0.10`; retained previous axis weights.
- Config normalization caps introduced: `workflow_boost_cap: 4`, `device_penalty_cap: 5` for diminishing returns.

### Rationale
- `workflow_boost` rewards abstracts describing platform-supported, automatable polymer operations (crosslinking, curing, grafting, thiol-ene, post-functionalization, degradation monitoring, quench) without inflating score for unsupported lab steps (e.g., dialysis, lyophilization intentionally excluded).
- `device_penalty` gently demotes hardware-centric device performance papers (solar cell, photodetector, transistor, sensor) that previously leaked into top ranks due to polymer term density + impact.

### Notes
- Further tuning deferred until LLM-derived relevance labels collected—these will inform SHAP / logistic calibration to adjust or prune axes.
- If device-focused false positives persist, consider contextual window filtering rather than expanding term list aggressively.

## [0.4.7] - 2025-10-06
### Added
- Generic axis framework: introduced `scoring.axis_definitions` enabling declarative axis specification (type + parameters) without modifying Python code.
- Axis handlers implemented: capability_fraction, unique_kind_fraction, unique_count_fraction, citation_log_minmax, novelty_penalty, constraint_penalty_sum.

### Changed
- Rewrote `score.py` to iterate over axis_definitions; removed hard-coded axis function block and legacy duplicate code.
- Moved per-axis caps and penalty mappings into each axis stanza in config.

### Rationale
- Decouples domain-specific heuristics (polymer-centric today) from scoring engine; future users can swap vocab & axis set purely via YAML edits.
- Simplifies experimentation (add/remove axes, zero weights) while keeping provenance (run_summary lists active axes from config order).

### Next
- Optional `expression` axis type for composite meta-features after LLM calibration.
- Potential auto-discovery of list fields (introspection) to suggest new axes in diagnostics.

## [0.4.8] - 2025-10-06
### Changed
- Replaced hard-coded `novelty_penalty` handler with fully declarative `pattern_penalty` axis type (config-driven pattern specs: any/all/pair/regex, sum or max aggregation, cap enforcement).
- Removed all legacy duplicate axis_* functions and second `main()` block from `score.py` (eliminates domain-bound code paths and future drift risk).
- Externalized solvent term detection: solvent patterns now configurable via `extraction.solvent_patterns`; legacy inline regex retained only as fallback if none provided.
- Unified heuristic term extraction in `extract.py` via generic `term_sets` mapping (configurable as `extraction.heuristic_term_sets`) producing `<prefix>_term_hits` / `<prefix>_term_unique` for polymer, workflow, device (extensible without code edits).

### Added
- Generic pattern penalty handler (`axis_pattern_penalty`) supporting `any`, `all`, `pair`, and `regex` match modes with per-pattern weights.
- Config hook for extending heuristic term sets (`extraction.heuristic_term_sets`).

### Removed
- Deprecated polymer-specific loading helper (`load_polymer_terms`) superseded by generic `load_heuristic_list` logic.

### Rationale
- Completes generalization pass: all remaining domain semantics (vocabulary, hype phrases, solvent lists) now live in YAML/heuristics files; core Python modules are domain-agnostic.
- Reduces friction for pivoting to non-polymer domains (only adjust heuristics + axis_definitions + weights).

### Migration Notes
- Update config: change axis definition key `novelty` type from `novelty_penalty` to `pattern_penalty` and supply `patterns` list (old `penalties` map no longer used).
- (Optional) Supply `extraction.solvent_patterns` to override fallback list.

### Next
- Implement LLM export script to package gated + exploration abstracts with axis vectors for labeling.
- Post-label calibration: derive logistic / isotonic mapping from axis vector -> relevance probability; optionally introduce composite `expression` axis referencing calibrated weights.

## [0.4.9] - 2025-10-06
### Changed
- Refined LLM prompt preview generation (`prompt_preview.py`):
  - Compressed axis glossary (CORE vs PENALTIES grouping) to reduce token footprint.
  - Added explicit rationale length (≤40 tokens) and confidence calibration guidance.
  - Grouped penalty axes under `PENALTIES[...]` in axis line; suppresses zero-value `novelty` / `constraint_penalty` when not informative.
  - Clarified `device_penalty` meaning (higher = worse) and flagged `impact` as ignorable for labeling to mitigate halo bias.
  - Added stricter output constraint reminder (JSON only) and refined multi-objective definition to require explicit trade / simultaneous optimization.

### Notes
- No scoring logic or axis values changed—only presentation in offline prompt artifacts.
- Downstream calibration scripts remain unaffected; axis vectors in JSONL records unchanged.

### Next
- Optional: rename `device_penalty` to `device_focus` in config & invert sign after collecting initial LLM labels to align all axes directionally (higher = better) pre-calibration.

## [0.5.0] - 2025-10-06
### Added
- `research/literature_search/scripts/llm_label.py` LLM labeling runner (OpenAI-compatible) with:
  - Dry-run mode (schema validation without API calls).
  - Retry logic & basic malformed JSON repair attempt.
  - Client-side rate limiting and schema enforcement.
- README section documenting labeling workflow, .env usage, and run commands.
- Dependencies `openai` and `python-dotenv` appended to `requirements.txt`.

### Changed
- Root `.gitignore` already covered `*.env` (explicitly validated for secrets safety).

### Notes
- Scoring pipeline untouched; only tooling for post-heuristic label acquisition added.
- Next calibration step (not in this release): fit logistic/isotonic model mapping axis vector -> relevance probability.

### Next
- Implement calibration script (`calibrate_labels.py`) after collecting sufficient labeled set.
- Optional axis direction normalization before probability model.

## [0.5.1] - 2025-10-06
### Added
- `label_analysis.py` script to merge heuristic scores & LLM labels, compute decile/percentile metrics, and optionally plot label vs. rank.

### Changed
- Enhanced `.env` discovery in `llm_label.py` (searches repo root, CWD, and literature_search folder) so placing `.env` inside `research/literature_search/` works without moving it.
- Added `matplotlib` dependency for plotting (optional; warnings if not installed).

### Notes
- No scoring or axis logic changed; purely tooling & analysis improvements.
- Metrics JSON + merged CSV enable downstream calibration modeling.

## [0.5.2] - 2025-10-07
### Added
- Verbose debugging option (`--verbose`) to `llm_label.py` providing per-record attempt logs, raw output snippets, schema failure reasons, and success confirmations.

### Rationale
- Facilitates diagnosis of empty / partial output scenarios (e.g., silent JSON parse failures or missing API key issues) before large-batch labeling runs.

### Notes
- No behavior change when flag omitted; performance impact minimal (string slicing only when enabled).


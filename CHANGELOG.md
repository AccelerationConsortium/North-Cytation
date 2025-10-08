# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-10-06
### Added
  - `README.md` explaining literature-focused capability framing
  - `capabilities_catalog.yaml` listing core experimental capabilities

### Notes
- Documentation-only addition; no runtime logic modified.
### Changed
- `calibration_protocol_example.py`: Updated default `vial_file` path to `status/calibration_vials_short.csv` to align with existing modular calibration workflows.

### Rationale
- Ensures consistency with other calibration scripts referencing the short vial status file; reduces confusion when switching between workflows.

### Migration Notes
- If you previously relied on the older `res/vials_calibration.csv`, edit the protocol file or restore that path locally.

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

## [0.5.0] - 2025-10-08
### Changed
- `calibration_protocol_example.py` measurement pipeline now uses real mass-based measurement:
  - Sets `measure_weight=True` when dispensing into the clamped measurement vial.
  - Converts returned mass (g) to volume (mL) using density (from optional `liquids.yaml`).
  - Fallback to target volume if hardware returns `None` (e.g., vial not clamped or simulation path).
- Added lightweight loader for `liquids.yaml` (density + optional `refill_pipets` flag) with environment override `CALIBRATION_LIQUID` (defaults to `water`).
- Implemented `new_pipet_each_time` (legacy `refill_pipets`) behavior: removes pipet between replicates when flag true.

### Added
- Extra replicate metadata keys in results: `mass_g`, `density_g_per_ml`, `new_pipet_each_time` for downstream analyzers.

### Rationale
- Replaces placeholder volume echo with physically meaningful measurement aligned with legacy modular calibration workflows.

### Migration Notes
- If prior downstream code assumed `volume` equaled target request, adjust to treat it as measured value.
- Provide a `liquids.yaml` (or set `CALIBRATION_LIQUID`) to ensure accurate density; otherwise default 1.0 g/mL is used.

## [0.6.0] - 2025-10-08
### Changed
- Orchestrator (`next_gen_calibration/run.py`) now attempts `initialize(cfg)` first; falls back to `initialize()` for backward compatibility.
- `calibration_protocol_example.initialize` signature updated to accept optional `cfg` dict and liquid selection now derives from config (`cfg['liquid']` or `cfg['protocol']['liquid']`) instead of environment variables.
- Template & simulated protocol files updated to accept optional `cfg` parameter for consistent developer ergonomics.

### Removed
- Environment-based liquid selection in example protocol (previous `CALIBRATION_LIQUID` usage); configuration is now explicit and source-controlled via `params.yaml`.

### Rationale
- Aligns protocol initialization with reproducible, versioned configuration while preserving zero-argument compatibility for existing scripts.

### Migration Notes
- To specify a liquid: add top-level `liquid: glycerol` (example) or `protocol: { liquid: glycerol }` in `params.yaml`.
- Existing third-party protocols without `initialize(cfg)` continue to function unchanged.

## [0.6.1] - 2025-10-08
### Changed
- `calibration_protocol_example` now ENFORCES explicit liquid + density specification via config. Removed silent defaults (`water`, density 1.0 g/mL).

### Added
- Validation errors: missing config, missing liquid key, missing liquids.yaml, unknown liquid, or missing `density_g_per_ml` now raise clear exceptions instead of proceeding with unsafe assumptions.

### Rationale
- Prevents accidental calibration against incorrect physical assumptions and forces reproducible, explicit experimental specification.

### Migration
- Ensure `params.yaml` contains either:
  - `liquid: glycerol` (top-level), or
  - `protocol: { liquid: glycerol }`
- Provide a `liquids.yaml` entry: `glycerol: { density_g_per_ml: 1.261, refill_pipets: true }` (example).

## [0.6.2] - 2025-10-08
### Changed
- Simplified `calibration_protocol_example` liquids file resolution: removed multi-path search. Now uses single canonical path `next_gen_calibration/liquids.yaml` unless explicitly overridden via `liquids_file` in config.

### Rationale
- Avoids speculative path assumptions; enforces explicitness and reduces silent divergence between environments.

### Migration
- If you previously relied on root-level `liquids.yaml`, move it to `next_gen_calibration/liquids.yaml` or set `liquids_file` in `params.yaml`.

## [0.6.3] - 2025-10-08
### Changed
- `calibration_protocol_simulated.initialize` now seeds `random` from `cfg['random_seed']` (if present) before falling back to legacy `CAL_SIM_SEED` env variable.

### Rationale
- Ensures reproducibility is fully governed by versioned config rather than ephemeral environment state.

### Migration
- Remove any reliance on `CAL_SIM_SEED`; specify `random_seed:` in `params.yaml` instead.

## [0.6.4] - 2025-10-08
### Added
- Progress logging in `next_gen_calibration/run.py` for volume start, screening phase, optimization loop, precision test entry, and pass event.

### Changed
- `Analyzer.write_history_csv` now unions all record keys (dynamic headers) and ignores extras to prevent ValueError on new fields (e.g., added `iteration`).

### Rationale
- Enhances transparency during long optimization runs and removes brittleness when record schema evolves.

### Migration
- No action required; downstream CSV consumers should handle potential additional columns.

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

## [0.4.9] - 2025-10-07
### Added
- Introduced `next_gen_calibration/liquids.yaml` external registry for liquid properties (density) replacing hard-coded mapping in `robot_adapter.py`.
- Added lazy-loading `liquids.py` helper (get_density, list_liquids, register_liquid) with safe fallback to defaults.
### Changed
- `robot_adapter.py` now sources density via liquids module (supports dmso, ethanol by default).
### Notes
- Future extensions: viscosity, temperature correction factors, surface tension.
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

## [0.5.3] - 2025-10-07
### Added
- `--limit` and `--shuffle` flags to `llm_label.py` to enable stratified / randomized small-batch labeling.
- Confidence scale guidance block appended in `prompt_preview.py` to encourage dispersion of confidence values across 0–1.
- New diagnostics script `label_diagnostics.py` producing failure reason counts, axis means per label, and disagreement buckets (high-rank irrelevants, low-rank positives).

### Rationale
- Facilitates iterative evaluation of LLM vs heuristic gating without incurring full-batch labeling cost; early disagreement surfacing accelerates calibration planning.

### Notes
- No change to scoring logic; all additions are tooling and prompt instruction refinements.

## [0.5.4] - 2025-10-07
### Added
- Deterministic `prompt_version` hash embedded in each prompt preview line and propagated into LLM label outputs (`llm_label.py`).

### Rationale
- Enables mixing / filtering of label sets across prompt template changes without confusion or data leakage during calibration.

### Notes
- Hash derived from core instruction blocks only; axis numerical values and abstracts not included (ensures stability across corpus expansion).

## [0.5.6] - 2025-10-07
### Changed
- Enhanced final Slack notification in `workflows/calibration_sdl_modular.py` to include:
  - Explicit list of completed volumes (µL)
  - Remaining (not-calibrated) volumes
  - Per-volume performance snapshot (best deviation %, time s) when available

### Rationale
- Provides at-a-glance operational clarity without opening log or data files; accelerates remote monitoring decisions.

### Notes
- Falls back gracefully if metrics unavailable (e.g., empty results_df subset) and logs a warning if Slack send fails.

## [0.5.5] - 2025-10-07
## [0.5.7] - 2025-10-07
### Changed
- `workflows/calibration_sdl_modular.py`: Redirected `BASE_AUTOSAVE_DIR` to a repo-local `output/calibration_runs` directory by default for local/simulation runs.

### Added
- Environment variable override `CALIBRATION_AUTOSAVE_DIR` to point at production/lab storage without modifying source.

### Rationale
- Prevents accidental writes to lab workstation path during local development while keeping a single code path; simple env switch restores production destination.

### Usage
- Set `CALIBRATION_AUTOSAVE_DIR` before running to override: e.g. `set CALIBRATION_AUTOSAVE_DIR=C:\\Users\\Imaging Controller\\Desktop\\Calibration_SDL_Output\\New_Method` (PowerShell: `$env:CALIBRATION_AUTOSAVE_DIR='C:\\Users\\Imaging Controller\\Desktop\\Calibration_SDL_Output\\New_Method'`).

### Simulation
- Relaxed simulation tolerances (relative minimum deviation/variation windows) so simulated runs pass precision tests more often for easier visualization testing.
  - Further loosened (default min relative deviation 12%, variation 15%) with env overrides `SIM_MIN_REL_DEV` / `SIM_MIN_REL_VAR`.
  - Added simulation-specific optimization acceptance: pass if deviation <= 20% (override via `SIM_MAX_DEV_PCT`) OR absolute deviation within μL tolerance, with 1.25× relaxed time limit.
  - Refined time model: simulation time now centers on per-trial expected_time with bounded penalties (0.7×–1.6× baseline) instead of broad random offsets causing spurious time failures.
### Fixed
- `analysis/calibration_analyzer.py`: `plot_measured_volume_over_time` now prefers `calculated_volume` (mL) → µL instead of assuming mass density=1.0, eliminating inflated plotted volumes for high-density liquids (e.g. glycerol). Falls back gracefully.
 - Scatter plot: replaced unsupported marker "★" with portable "*"; added option to plot absolute deviation (µL) vs time; corrected y-axis label (percent vs µL) and ensured highlight points render.
 - Removed precision winner star overlay from scatter plot (visual simplification – color encodes volume only).

### Added
- `--resume` flag in `llm_label.py` allowing interrupted labeling runs to be safely continued without re-querying already labeled abstracts (skips IDs present in existing output file and appends new results).
- `--request-timeout` soft per-request timeout wrapper (thread-based) to convert stalled network calls into retries (prevents indefinite blocking leading to manual interrupts).

### Changed
- Labeling completion summary now distinguishes new successes vs cumulative when `--resume` is used.

### Rationale
- Repeated mid-run interruptions previously forced restarts and risk of partial / empty label sets; resume + timeout hardening ensures efficient incremental accumulation of labels for calibration.

### Notes
- Timeout wrapper is cooperative (does not forcibly cancel underlying HTTP if library call hangs internally) but sufficient for short batches; consider future replacement with client-native timeout if migrating SDK.

## [0.6.0] - 2025-10-07
### Added
- `next_gen_calibration/` prototype implementation:
  - `params.yaml`: YAML-driven parameter space, per-volume criteria scaling, simulation & output settings.
  - `robot_adapter.py`: Robot abstraction + seedable simulation model (density-aware, param-influenced deviation/time generation).
  - `optimizer.py`: Lightweight placeholder Bayesian-like optimizer (history-based localized random refinement) decoupled from Ax for fast iteration.
  - `analyzer.py`: Minimal analyzer producing `history.csv` and aggregate `summary.json` (mean deviation, time, variation).
  - `run.py`: Orchestrator loading YAML, performing multi-volume optimization loop (initial suggestions + refinement), applying scaled criteria, persisting outputs under timestamped run directory.

### Rationale
- Establishes configuration-first, robot-agnostic calibration core to progressively replace existing `calibration_sdl_*` scripts without entangling with legacy environment assumptions.

### Notes / Next Steps (Not yet implemented)
- Integrate real optimizer backend (Ax or scikit-optimize) behind strategy interface.
## [0.6.1] - 2025-10-08
### Added
- Phase configuration (`phases` block in `next_gen_calibration/params.yaml`): screening, optimization, accuracy_gate, precision, objective.
- Pluggable recommender interface (`recommender.py`) with `RandomLocalRecommender` and optional `AxRecommender` (auto-fallback if Ax not installed).
- Phased run loop in `run.py` executing screening -> optimization -> precision_test per volume with accuracy gating (deviation + time thresholds) prior to precision verification.
- Precision test logic: replicate run with best params, max deviation band check.

### Changed
- Simplified scalar objective (deviation + time_weight*time) ignoring std deviation; weights configurable via `objective.time_weight`.
- History records now include phase tags; precision test result appended when gate achieved.

### Notes
- Percent-based precision band omitted (absolute band only per updated design clarification).
- Ax example uses single-objective scalarization; multi-metric gating handled externally for phase transitions.
- Future: resume capability, best_per_volume artifact, replicate raw export, hardware adapter.
- Add robot hardware subclass implementing `_hardware_pipetting`.
- Port tolerance logic & percent-fallback acceptance from legacy modular workflow into configurable policy block.
- Introduce richer analyzer visuals (scatter & volume-over-time) gated by optional plotting dependency.
- Add resume capability (persisted optimizer state) and selective volume re-run support.

### Changed (post-0.6.1 config cleanup)
- Reworked simulation configuration semantics:
  - `simulation_thresholds` now mirrors the main `thresholds` schema (base_* + scaling_* keys).
  - When `simulate: true` and this block is present, its values fully override the corresponding base thresholds (no relative floors, no extra rules).
  - Removed prior relative floor keys (`min_relative_accuracy`, `min_relative_precision`) to eliminate divergent logic between real and simulated runs.

  ### Added (post-0.6.1)
  - Introduced protocol-based hardware lifecycle hooks replacing static `hardware.yaml`:
    - `calibration_protocol_example.py` providing reference `initialize / measure / wrapup` implementation using existing `lash_e` robot methods.
    - `calibration_protocol_template.py` empty scaffold with detailed inline guidance for custom lab adaptations.

  ### Deprecated / Removed
  - Removed `next_gen_calibration/hardware.yaml` (previous minimal handler schema). Rationale: procedural initialization & measurement sequences are better expressed as Python hooks than YAML DSL; keeps configuration YAML focused on static tunables (thresholds, parameter bounds) while delegating ordered hardware actions to code.

  ### Rationale
  - Narrower, explicit contract (`initialize`, `measure`, `wrapup`) clarifies robot I/O surface, simplifies customization per lab, and avoids schema drift / speculative fields in YAML.
  - Facilitates simulation vs real hardware branching by swapping protocol module without touching orchestrator or parameter configuration.

  ### Migration Notes
  - If you previously relied on `hardware.yaml` handler import, migrate logic into a protocol module and point future orchestrator code to dynamically import it (planned minor refactor in upcoming version to wire selection through main run path).
  - Goal: identical acceptance logic shape; only numbers differ if explicitly overridden for simulation.
- Clarified `selective_optimization` semantics in `params.yaml` and changed `use_historical` default to `false`:
  - enable: once a volume passes precision, fix all non volume-dependent params for later volumes.
  - use_historical: controls whether optimizer history carries across volumes; when false we now re-instantiate the recommender each volume (while still applying fixed params).
  
### Renamed (config clarity)
- `selective_optimization` -> `volume_dependant_optimization`
- `seed` -> `random_seed`
- `max_wells` -> `max_measurements` (note: currently unused in logic; placeholder for future measurement budgeting)
### Restructured
- Moved `phases.optimization.max_trials_per_volume` to `phases.max_trials_per_volume` (now clearly applies to all trial types per volume) and set default example to 96 while increasing `optimization.max_iters` to 30.
### Documentation / Config Clarification
- Inlined dynamic overaspirate cap: removed `constraints.overvolume_percent`; added `overaspirate_vol.max_fraction_of_target_volume`.
- Expanded inline commentary across `next_gen_calibration/params.yaml` describing each top-level block and parameter semantics.

## [0.6.3] - 2025-10-08
### Added
- `protocol` section in `next_gen_calibration/params.yaml` (`module`, optional `class_name`, `kwargs`) plus env override `CALIBRATION_PROTOCOL_MODULE` for zero-code protocol swapping.
- Dynamic protocol loader in `next_gen_calibration/run.py` invoking `initialize/measure/wrapup` if present; falls back to legacy `RobotAdapter` when absent or invalid.

### Changed
- Removed legacy backward-compat extraction helper and any support for `calculated_volume` or `mass` result keys in next‑gen flow (volume-only contract enforced in protocol path).
- `calibration_protocol_template.py` simplified: no simulation branch, no density references; pure volume return scaffold.
- `calibration_protocol_example.py` cleaned to volume-only logic (internal conversion comment retained for hardware-specific adaptation).
- `calibration_protocol_simulated.py` rewritten as self-contained lightweight simulator (no dependency on legacy `calibration_sdl_base` or `LIQUIDS`).

### Removed
- Reliance on legacy simulation imports (`calibration_sdl_base`, `LIQUIDS`) in simulated protocol.
- Backward compatibility shims for mass/calculated_volume in orchestrator.

### Rationale
- Starts clean separation: core orchestrator agnostic to mass or density; protocol implementer bears responsibility for interpreting hardware measurements into volumes.
- Reduces drift risk by eliminating transitional dual-schema handling and external legacy dependencies.

### Migration Notes
- Custom protocols must implement volume-centric result dicts (`volume`, `elapsed_s`, `replicate`).
- Set protocol via YAML: `protocol.module: calibration_protocol_example` (or custom). Override at runtime with `CALIBRATION_PROTOCOL_MODULE` env var for quick switching.
- If no protocol is specified or import fails, system automatically reverts to existing `RobotAdapter` simulation/hardware abstraction.

### Next (Planned)
- Best-per-volume artifact (`best_per_volume.json`) and flagging (`is_best_for_volume`).
- Replicate deviation persistence and percent accuracy convenience fields.
- Optional explicit source/dest vial configuration in params.

## [0.6.4] - 2025-10-08
### Changed
- Simplified protocol interface: generic protocols now use signatures without `lash_e` or vial arguments:
    initialize(context) -> state
    measure(state, volume_mL, params, replicates, context) -> list[dict]
    wrapup(state, context) -> None
- `calibration_protocol_template.py` rewritten to reflect new minimal interface.
- `calibration_protocol_simulated.py` decoupled from any hardware object (pure function simulation).
- `calibration_protocol_example.py` explicitly marked hardware-specific and adds runtime guard enforcing presence of `lash_e.nr_robot`.
- `params.yaml` protocol block made optional; auto-selection now based on `simulate` flag when module omitted.

### Added
- Runtime guard in example protocol (`_require_lash_e`) for early, clear error messaging when required hardware context is missing.
- Comments in `params.yaml` documenting auto-selection logic and dual interface (hardware vs generic).

### Removed
- Implicit passing of `lash_e` to simulation/template protocols (they no longer accept it).

### Rationale
- Eliminates leaking hardware abstraction (`lash_e`) into protocols that do not require it, reducing user confusion and making simulation/custom backends trivial to implement.
- Keeps hardware-specific complexity isolated to a single example module while maintaining backward compatibility for that path.

### Migration Notes
- Custom protocols created from earlier template versions should drop the `lash_e`, `source_vial`, and `dest_vial` parameters and adapt to the simplified interface.
- Existing hardware protocol authors can keep the extended signature if desired, but only `calibration_protocol_example` is included by default.

### Next
- Potential config exposure of source/dest vial naming for hardware protocol; best-per-volume artifact enhancements.

## [0.6.5] - 2025-10-08
### Added
- `per_volume_summary.json` artifact with aggregated stats per dispense volume (trial counts, best accuracy/time/precision, precision pass flag, gate achievement).

### Changed
- Analyzer now uses canonical record keys: `avg_accuracy_ul` (was `avg_dev_ul`) and `std_precision_ul` (was `std_dev_ul`).
- Summary aggregates renamed to `mean_accuracy_dev_ul`, `mean_time_s`, `mean_precision_std_ul` (legacy `dev_mean`, `time_mean`, `var_mean` retained for backward compatibility).
- Plot labels updated to reflect new accuracy/precision naming.

### Rationale
- Align terminology with orchestrator record schema (`avg_accuracy_ul` / `std_precision_ul`) eliminating NaN aggregates when legacy keys absent.
- Provide quick per-volume inspection without parsing full `history.csv`.

### Migration Notes
- Downstream consumers relying on `dev_mean` / `var_mean` can continue unchanged; migrate to new names when convenient.
- If custom tooling still emits legacy keys, analyzer preserves functionality via backward-compatible lookups.

### Next
- Potential addition of `best_params_per_volume.json` capturing param sets for reproducibility.

## [0.6.6] - 2025-10-08
### Fixed
- Plot generation crash when legacy key `avg_dev_ul` absent; analyzer now falls back to `avg_accuracy_ul` and wraps each plot in try/except.

### Added
- Per-plot error resilience: failures are logged and do not abort remaining plot generation.

### Rationale
- Ensures forward schema changes (renamed metric keys) do not silently remove all plots.

## [0.6.7] - 2025-10-08
### Fixed
- Analyzer plots: filtered out precision-only records (lacking avg_time_s / accuracy metrics) to prevent missing-key skips.
- Resolved ndarray axes handling in volume small-multiples (previous attribute mismatch causing failure).

### Changed
- Deviation/Time iteration plots now use only records that contain the relevant metrics (screening + optimization phases), excluding precision test rows.

### Rationale
- Improves robustness and correctness of visualization when schema includes phase-specific record subsets.

## [0.6.8] - 2025-10-08
### Added
- `time_by_volume.png`: small-multiples time plots mirroring deviation-by-volume.
- `replicate_volume_scatter.png`: consolidated replicate-level measured volume scatter with dashed target lines.
- `replicate_time_scatter.png`: analogous replicate-level timing scatter.

### Removed
- Less informative iteration plots (`deviation_over_iterations`, `time_over_iterations`) from default generation list.

### Changed
- Trial results now include `replicate_vols_ul` and `replicate_times_s` enabling replicate-level visualization.

### Rationale
- Focuses on actionable cross-volume comparisons and replicate distributions matching user feedback; drops low-signal iteration trend charts.

## [0.6.9] - 2025-10-08
### Fixed
- Plot generation failure when a plot function returned None (filtered out before writing `plots.json`).

### Rationale
- Ensures optional plots that decide to skip (due to missing replicate data) don't abort artifact writing.

## [0.6.10] - 2025-10-08
### Fixed
- Missing replicate-level scatter plots (volume/time) due to replicate arrays not being persisted in phase records.

### Changed
- `run.py` now injects `replicate_vols_ul` and `replicate_times_s` into screening, optimization, and precision_test records.

### Rationale
- Ensures analyzer has full replicate context for new plots without additional parsing of raw protocol outputs.

## [0.7.0] - 2025-10-08
### Changed (BREAKING)
- Unified calibration protocol interface to minimal, argument-light form:
    initialize() -> state(dict)
    measure(state, volume_mL, params, replicates) -> list[dict]
    wrapup(state) -> None
- Removed `context` parameter from all protocol hooks; protocols now self-manage any external configuration (env vars, file reads, hard-coded defaults) without relying on orchestrator-passed dicts.
- Eliminated orchestrator injection of `lash_e`, `source_vial`, `dest_vial`; hardware initialization is fully encapsulated within protocol `initialize()` (example hardware protocol now constructs and stores `lash_e` inside returned state).
- Deleted adaptive signature trimming logic (`_adapt_call`) from `next_gen_calibration/run.py` along with backward hardware signature support.
- Simplified internal trial helpers to call `measure(state, vol, params, reps)` directly.
- Refactored `calibration_protocol_example.py` to internally:
  - Initialize `Lash_E` (using optional env overrides `CAL_VIAL_FILE`, `CAL_SIMULATE`)
  - Manage measurement vial rotation and source/destination vial selection locally
  - Expose only the unified hook signatures.
- Updated template & simulated protocols to match the new signature and removed unused `context` logic and random seed handling via config (now can use `CAL_SIM_SEED`).

### Removed
- Any reliance on `_context` entries in protocol state; state is now purely protocol-defined.
- Orchestrator lash_e detection/import path and adaptive positional argument trimming.

### Rationale
- Reduces cognitive overhead for users authoring custom protocols by presenting a single, stable, minimal contract.
- Avoids leaking lab-specific abstractions (`lash_e`, vial naming) into generic protocol space while preserving full flexibility inside hardware example.
- Encourages explicit, user-owned initialization patterns (construction logic lives where it is maintained: in the protocol module itself).

### Migration Notes
- Existing custom protocols must drop extra parameters (`lash_e`, `context`, `source_vial`, `dest_vial`).
- Replace earlier `initialize(context)` usage with `initialize()` and move any required config ingestion inside the protocol (read env vars or local files as needed).
- If prior code depended on orchestrator-provided `run_dir` or `simulate`, replicate that by reading `os.environ` or adding your own detection inside the protocol.

### Future
- Potential optional helper wrapper for users wanting a typed config object (will remain additive, not replacing minimal core contract).

## [0.7.1] - 2025-10-08
### Changed
- `calibration_protocol_example.py`: Removed all environment variable lookups (`CAL_SIMULATE`, `CAL_VIAL_FILE`, `CAL_SOURCE_VIAL`). Replaced with explicit hard-coded values (`simulate = False`, fixed `vial_file`, `source_vial`).

### Rationale
- Keeps the hardware example fully explicit and easier to understand for new users—no hidden dependency on process environment.
- Aligns with goal: protocols should not assume any external configuration schema or ambient flags.

### Migration Notes
- If you previously relied on environment variables to toggle simulation or change vial/source names, edit the example file directly now.
- To reintroduce dynamic behavior, you can manually add environment checks or CLI argument parsing within your forked protocol module without impacting the core orchestrator.


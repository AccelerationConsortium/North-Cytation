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

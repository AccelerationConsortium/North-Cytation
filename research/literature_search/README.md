# Literature Search Pipeline

Purpose: Discover and rank candidate experiments from the literature that align with platform capabilities and experiment priorities.

## Data Flow
1. Query assembly (archetype tokens + constraints) -> OpenAlex API
2. Raw response storage (verbatim JSON) in `data/raw_openalex/` (created lazily)
3. Parsing & extraction (factors, objectives, solvents, capability verbs)
4. Scoring (capability fit, parameter richness, multi-objective, impact, novelty, constraint penalty)
5. Ranked outputs (`scored_candidates.csv`, detailed JSONL records)

## Inputs
- `../capabilities/capabilities_catalog.yaml`
- `../capabilities/constraints.yaml`
- `../capabilities/solvent_whitelist.yaml`
- `../capabilities/experiment_priority.md`
- `literature_pipeline_config.yaml` (this directory)

## Outputs
- `data/papers_raw.jsonl` (one raw OpenAlex work per line)
- `data/papers_parsed.jsonl` (enriched fields)
- `data/scored_candidates.csv` (ranked summary)
- `logs/literature_pipeline.log` (optional future)

## Execution Stages (MVP)
- fetch.py: run archetype queries & store raw
- extract.py: transform raw -> normalized records
- score.py: compute metrics & final score
- pipeline.py: orchestration wrapper (future background mode)

## Background Mode (Planned)
Periodic run (cron, task scheduler) re-queries recent additions (using updated_since) and appends new papers only.

## Config-Driven Design
All tunables in `literature_pipeline_config.yaml`: years, caps, weights, penalties, regex patterns, file paths.

## Next Steps
Implement fetch and extraction stubs, then integrate scoring.

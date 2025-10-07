# Assistant Context Memo

_Last updated: 2025-10-07_

## Project One-Liner
Pipeline to fetch, extract features from, and score scientific abstracts (OpenAlex) with heuristic axes, then obtain LLM relevance labels on a stratified sample to calibrate / refine gating and future probabilistic mapping from axis vector -> relevance probability.

## Full Workflow (Stage-by-Stage)
1. Fetch (`fetch.py`): Acquire abstracts per query tokens (iterative per-token union) -> parsed JSONL.
2. Extract (`extract.py`): Regex + vocabulary extraction producing counts / unique hits (polymer, workflow, device, etc.) and numeric factor signals.
3. Score (`score.py`): Declarative `axis_definitions` (capability_fit, polymer_specificity, parameter_space, multi_objective, impact, workflow_boost, device_penalty, pattern_penalty, constraint_penalty) -> `score_total` via weights in config.
4. Gating: Percentile-based selection (currently 85th percentile chosen earlier) to define candidates.
5. Prompt Generation (`prompt_preview.py`): Build JSONL with prompt text, axis line, glossary, and deterministic `prompt_version` hash.
6. Labeling (`llm_label.py`): Sequential LLM calls producing JSON labels: relevance_label ∈ {relevant, maybe, irrelevant}, confidence, rationale (≤40 tokens), signals, failure_reasons.
7. Diagnostics (planned / partial): `label_diagnostics.py` to analyze axis separation, false positives, false negatives, failure reason distributions.
8. Calibration (future): Fit logistic / isotonic model over axes using labeled set; possibly derive composite axis or reweight gating.

## Current State (Top Sample)
- Prompt file: `research/literature_search/data/prompt_preview_top15.jsonl`
- Labels file: `research/literature_search/data/labels_top15.jsonl`
- Completed labels: 6 / 15
  - 2 irrelevant
  - 4 maybe (no clear "relevant" yet)
- Snapshot: `research/literature_search/data/labeling_state_top15.json`
  - Remaining: 9
  - `prompt_version`: `0e63d3afde`

## Recent Enhancements
- Resume support (`--resume`) skipping already labeled IDs.
- Request timeout wrapper (optional) and later graceful interrupt + heartbeat.
- Segment selection bug fixed (no accidental bottom overrun when bottom=0).
- Prompt provenance hashing (`prompt_version`).

## Why Progress Paused / Perceived Stuck
- Repeated unexpected `KeyboardInterrupt` during labeling after 1–2 records.
- Each interrupt halts loop mid-batch but partial progress is preserved (append-after-each-record design).
- Root cause not internal logic error (no schema or network failure logs); likely external signal (terminal / environment) but still under observation.

## Mitigations Implemented
- Resume: safe incremental continuation.
- Immediate flush after each write.
- Graceful interrupt mode to differentiate intentional vs unexplained signals.
- Heartbeat option for visibility during long waits.
- Snapshot file ensuring restart context is explicit.

## Next Concrete Steps (Short Path)
1. Finish top 15 labeling (resume until 15 lines in `labels_top15.jsonl`).
2. Generate middle and bottom prompt files (e.g., middle10, bottom10).
3. Label those sets with `--resume`.
4. Merge label files + run `label_diagnostics.py`.
5. Summarize axis means per label; evaluate heuristic precision and adjust weights/gating.
6. (Optional) Fit preliminary logistic calibration if distribution shows discernible separation.

## Optional Nice-to-Haves (Not Blocking)
- Auto-resume loop script for unattended completion.
- Label manifest (aggregate of all label sets with counts & prompt hashes).
- Integrity checker (duplicate IDs, mixed prompt_version detection).
- Calibration script skeleton.

## Open Questions / Watch Items
- Source of spurious KeyboardInterrupts (collect evidence with graceful handler messages).
- Will bottom / middle show at least a few clear "relevant" labels to enable separation? If not, may need to lower gate or broaden sample.
- Potential need for a dedicated positive workflow axis if signals remain ambiguous.

## Minimal Resume Command
```
python research/literature_search/scripts/llm_label.py \
  --input research/literature_search/data/prompt_preview_top15.jsonl \
  --output research/literature_search/data/labels_top15.jsonl \
  --model gpt-4o-mini --resume --graceful-interrupt --heartbeat-secs 5 --request-timeout 0
```
(Adjust heartbeat / timeout flags as desired.)

## Definition of Done for Current Milestone
- 35 labeled abstracts (15 top, 10 middle, 10 bottom) with a stable single `prompt_version`.
- Diagnostics JSON summarizing axis separation + false positive list.
- Recommendation for weight/gating adjustment or confirmation of current gate.

## Escalation If Interrupts Persist
- Run a simple endless sleep script to see if interrupts unrelated to labeling.
- Wrap LLM call with native client timeout if library version permits.
- Fallback to batch small subsets (e.g., 3 at a time via `--limit`) to isolate any problematic prompts.

---
Maintainer / assistant note: Everything necessary to continue is self-contained. If re-entering later, start by checking line count of `labels_top15.jsonl` and the snapshot, then proceed with remaining steps above.

"""
analysis_prompt.py — Prompts for the experiment analysis LLM node.

The LLM's role is strictly to *interpret* and *write*.
All numeric values must come from the pre-computed metrics in state —
the LLM must not invent numbers.
"""

from typing import Any, Dict, List


ANALYSIS_SYSTEM_PROMPT = """\
You are a laboratory data analysis assistant integrated into a Slack-based
self-driving lab system.

Your job is to write clear, structured experiment analysis reports based
on pre-computed metrics provided to you.  You must:

1. Only reference data that is explicitly provided in the user message.
2. Never invent numbers, thresholds, or results.
3. If data was unavailable or synthetic (placeholder), state that clearly
   at the top of the report.
4. Write in plain, professional language suitable for a scientist reading
   Slack on a mobile device.
5. Use Slack mrkdwn formatting:
   - *bold* for section headers and key values
   - `code` for run IDs, file paths, and metric names
   - :warning: emoji for anomalies
   - :white_check_mark: emoji for passing checks
6. Keep the report concise — aim for 300–500 words.

Report sections (in order):
  1. What experiment finished
  2. Data status (loaded successfully / synthetic placeholder)
  3. Key metrics
  4. Anomalies / warnings (or "none detected")
  5. Figures generated
  6. Interpretation
  7. Recommended next step
  8. Limitations
"""


def build_analysis_user_prompt(
    run_id: str,
    schema_summary: Dict[str, Any],
    qc_metrics: Dict[str, Any],
    warnings: List[str],
    comparison: Dict[str, Any],
    generated_plot_paths: List[str],
) -> str:
    """
    Build the user-turn prompt for the analysis LLM call.

    All values are injected from pre-computed tool outputs.
    """
    warnings_text = (
        "\n".join(f"  - {w}" for w in warnings)
        if warnings
        else "  None detected."
    )

    plots_text = (
        "\n".join(f"  - {p}" for p in generated_plot_paths)
        if generated_plot_paths
        else "  No plots generated."
    )

    trend_notes = comparison.get("trend_notes", [])
    trend_text = (
        "\n".join(f"  - {n}" for n in trend_notes)
        if trend_notes
        else "  No historical comparison available."
    )

    return f"""\
Write an analysis report for experiment run `{run_id}`.

--- DATA SCHEMA ---
Rows: {schema_summary.get('rows', 'unknown')}
Columns: {schema_summary.get('columns', [])}
Missing values: {schema_summary.get('missing_values', {})}

--- QC METRICS ---
{qc_metrics}

--- ANOMALIES / WARNINGS ---
{warnings_text}

--- TREND COMPARISON (vs last {len(comparison.get('previous_runs', []))} runs) ---
Previous runs: {comparison.get('previous_runs', [])}
Metric deltas: {comparison.get('metric_deltas', {})}
Notes:
{trend_text}

--- FIGURES GENERATED ---
{plots_text}

Write the full analysis report now, following the 8-section structure.
"""

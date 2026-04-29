"""
analysis_agent.py — LLM node for writing experiment analysis reports.

The LLM's job here is strictly *interpretation* and *writing*.
It receives pre-computed metrics and anomaly flags from the Python tools,
then writes a structured human-readable report.

The LLM must not invent numbers.  Every claim in the report must reference
data that was passed in via state.
"""

import logging
from typing import Any, Dict, List

from config import get_llm
from prompts.analysis_prompt import ANALYSIS_SYSTEM_PROMPT, build_analysis_user_prompt
from graph.state import AgentState

logger = logging.getLogger(__name__)


def write_report_with_llm(state: AgentState) -> str:
    """
    Call the LLM to write a structured analysis report.

    Inputs are read from state; the LLM receives a prompt containing all
    pre-computed values so it cannot fabricate data.

    Returns the report as a Slack-formatted string.
    """
    run_id          = state.get("run_id") or "unknown"
    analysis_results = state.get("analysis_results") or {}
    warnings        = state.get("warnings") or []
    generated_plots = state.get("generated_plot_paths") or []

    # Extract serialisable metrics (drop DataFrame object)
    qc_metrics     = analysis_results.get("qc_metrics") or {}
    schema_summary = analysis_results.get("schema_summary") or {}
    comparison     = analysis_results.get("comparison") or {}

    user_prompt = build_analysis_user_prompt(
        run_id=run_id,
        schema_summary=schema_summary,
        qc_metrics=qc_metrics,
        warnings=warnings,
        comparison=comparison,
        generated_plot_paths=generated_plots,
    )

    try:
        llm = get_llm(temperature=0.2)
        messages = [
            {"role": "system",  "content": ANALYSIS_SYSTEM_PROMPT},
            {"role": "user",    "content": user_prompt},
        ]
        response = llm.invoke(messages)
        report = response.content if hasattr(response, "content") else str(response)
        return report

    except Exception as exc:
        logger.error(f"LLM report generation failed: {exc}")
        # Return a minimal fallback report that clearly states what happened
        return (
            f"*Analysis Report — Run {run_id}*\n\n"
            f":warning: Report generation failed: `{exc}`\n\n"
            f"*QC Metrics (raw):*\n```\n{qc_metrics}\n```\n"
            f"*Warnings:*\n" + "\n".join(f"- {w}" for w in warnings)
            if warnings else "None"
        )

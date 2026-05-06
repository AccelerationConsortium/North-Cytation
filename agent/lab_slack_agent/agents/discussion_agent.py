"""
discussion_agent.py — Handlers for follow-up discussion and plot requests.

handle_discussion:
    LangGraph node for follow-up conversation.
    Retrieves thread context from memory and lets the LLM respond naturally.

handle_plot_request:
    LangGraph node for "show me the residual plot" style requests.
    Identifies the run, checks for existing figures, generates if missing,
    and sets recommended_plot_paths for the event handler to upload.
"""

import logging
import re
from pathlib import Path
from langgraph.types import RunnableConfig

from config import get_llm, PLOTS_DIR
from graph.state import AgentState
from memory.thread_memory import ThreadMemory
from prompts.discussion_prompt import (
    DISCUSSION_SYSTEM_PROMPT,
    build_discussion_user_prompt,
)
from tools.plot_tools import get_existing_plot, generate_requested_plot

logger = logging.getLogger(__name__)

# Lazy-initialised memory instance (shared across calls)
_memory: ThreadMemory | None = None


def _get_memory() -> ThreadMemory:
    global _memory
    if _memory is None:
        _memory = ThreadMemory()
    return _memory


# ─────────────────────────────────────────────────────────────────────────────
# Keyword → plot type mapping for plot requests
# ─────────────────────────────────────────────────────────────────────────────

_PLOT_KEYWORDS: dict[str, str] = {
    "residual":    "residual",
    "pareto":      "pareto",
    "calibration": "calibration",
    "anomaly":     "anomaly",
    "failed well": "failed_wells",
    "scatter":     "scatter",
    "objective":   "pareto",
}


def _extract_plot_type(message_text: str) -> str:
    """Return a normalised plot type from the message, or 'unknown'."""
    text_lower = message_text.lower()
    for keyword, plot_type in _PLOT_KEYWORDS.items():
        if keyword in text_lower:
            return plot_type
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# handle_discussion — LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def handle_discussion(state: AgentState) -> AgentState:
    """
    LangGraph node: respond to follow-up discussion messages.

    1. Retrieve recent thread history from memory.
    2. Build a prompt with that context.
    3. Let the LLM produce a natural reply.
    """
    channel_id   = state.get("channel_id", "")
    thread_ts    = state.get("thread_ts", "")
    message_text = state.get("message_text", "")

    # Load thread history from persistent memory
    mem = _get_memory()
    context = mem.get_thread_context(channel_id=channel_id, thread_ts=thread_ts)

    # Retrieve any stored analysis report for this thread
    stored_report = mem.get_latest_report(channel_id=channel_id, thread_ts=thread_ts)

    prompt = build_discussion_user_prompt(
        message=message_text,
        thread_context=context,
        analysis_report=stored_report,
    )

    try:
        llm = get_llm(temperature=0.5)
        response = llm.invoke([
            {"role": "system", "content": DISCUSSION_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ])
        reply = response.content.strip() if hasattr(response, "content") else ""
    except Exception as exc:
        logger.error(f"Discussion LLM call failed: {exc}")
        reply = f":warning: I encountered an error generating a response: `{exc}`"

    # Persist the user's message and bot reply
    mem.save_message(channel_id=channel_id, thread_ts=thread_ts,
                     user_id=state.get("user_id", ""), text=message_text)

    return {**state, "final_response": reply}


# ─────────────────────────────────────────────────────────────────────────────
# handle_plot_request — LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def handle_plot_request(state: AgentState, config: RunnableConfig = None) -> AgentState:
    """
    LangGraph node: fulfil a user request to see a specific figure.

    Steps:
    1. Identify the run_id associated with this thread (from memory).
    2. Extract the requested plot type from the message.
    3. Check if the figure already exists on disk.
    4. If not, generate it.
    5. Set recommended_plot_paths so the event handler uploads it.
    6. Add an explanation of the figure to final_response.
    """
    channel_id   = state.get("channel_id", "")
    thread_ts    = state.get("thread_ts", "")
    message_text = state.get("message_text", "")

    # ── 1. Find the run_id for this thread ───────────────────────────────────
    run_id = state.get("run_id")
    if not run_id:
        mem = _get_memory()
        run_id = mem.get_run_id_for_thread(channel_id=channel_id, thread_ts=thread_ts)

    if not run_id:
        return {
            **state,
            "final_response": (
                ":warning: I could not determine which experiment run to use. "
                "Please reference a run ID or run an analysis first."
            ),
        }

    # ── 2. Extract requested plot type ───────────────────────────────────────
    plot_type = _extract_plot_type(message_text)
    if plot_type == "unknown":
        return {
            **state,
            "final_response": (
                "I'm not sure which plot you'd like. "
                "Try: _residual_, _pareto_, _calibration_, _anomaly_, or _failed wells_."
            ),
        }

    # ── 3. Check for existing plot ───────────────────────────────────────────
    existing_path = get_existing_plot(run_id=run_id, plot_type=plot_type)

    if existing_path:
        plot_path = existing_path
        origin_note = "(retrieved from previous analysis)"
    else:
        # ── 4. Generate the plot on demand ───────────────────────────────────
        logger.info(f"Generating {plot_type} plot on demand for run {run_id}")
        try:
            plot_path = generate_requested_plot(run_id=run_id, plot_type=plot_type)
        except Exception as exc:
            logger.error(f"On-demand plot generation failed: {exc}")
            return {
                **state,
                "final_response": (
                    f":warning: Could not generate a *{plot_type}* plot for run `{run_id}`: `{exc}`"
                ),
            }
        origin_note = "(generated on demand)"

    # ── 5. Set paths for upload ──────────────────────────────────────────────
    explanation = _plot_explanation(plot_type)

    return {
        **state,
        "plot_file_path": plot_path,
        "requested_plot_type": plot_type,
        "recommended_plot_paths": [plot_path],
        "final_response": (
            f"*{plot_type.replace('_', ' ').title()} Plot* {origin_note} for run `{run_id}`\n\n"
            f"{explanation}"
        ),
    }


def _plot_explanation(plot_type: str) -> str:
    """Return a short explanation of what the requested plot shows."""
    explanations = {
        "residual":     "Shows prediction residuals (observed minus predicted). "
                        "Points far from zero indicate systematic model error.",
        "pareto":       "Pareto front of objective trade-offs. "
                        "Points on the frontier represent non-dominated solutions.",
        "calibration":  "Calibration curve mapping instrument signal to known concentrations. "
                        "Check linearity and R^2.",
        "anomaly":      "Highlights wells or time points flagged as anomalous by outlier detection.",
        "failed_wells": "Map of wells that failed QC filters (e.g., low signal, high CV).",
        "scatter":      "Scatter plot of two selected objectives or measurements.",
    }
    return explanations.get(plot_type, "")

"""
router.py — Message classification and routing.

classify_message:
    Node that sets state["intent"] based on source_type and message content.

route_message:
    Conditional edge function that returns the workflow key LangGraph uses
    to select the next node.
"""

import logging
import re
from graph.state import AgentState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Robot message patterns
# Matched against messages from ROBOT_BOT_USER_ID
# ─────────────────────────────────────────────────────────────────────────────

_ROBOT_EXPERIMENT_PATTERNS = [
    re.compile(r"experiment\s+complete", re.IGNORECASE),
    re.compile(r"run\s+finished", re.IGNORECASE),
    re.compile(r"calibration\s+experiment\s+completed", re.IGNORECASE),
    re.compile(r"run_id\s*=", re.IGNORECASE),
    re.compile(r"path\s*=\s*/", re.IGNORECASE),
]

# ─────────────────────────────────────────────────────────────────────────────
# Human message keyword buckets
# Each key maps to the intent string returned to LangGraph
# ─────────────────────────────────────────────────────────────────────────────

INTENT_KEYWORDS: dict[str, list[str]] = {
    "data_analysis": [
        "analyze", "analysis", "results", "data", "metrics",
        "performance", "evaluate", "summarize", "summary",
    ],
    "literature_research": [
        "research", "paper", "papers", "literature", "study", "studies",
        "find", "search", "reference", "cite", "citation", "publication",
        "journal", "review", "arxiv",
    ],
    "plot_request": [
        "plot", "show", "graph", "chart", "figure", "upload", "image",
        "residual", "pareto", "calibration curve", "anomaly", "failed well",
        "scatter", "histogram",
    ],
    "follow_up_discussion": [
        "why", "what does", "explain", "tell me more", "clarify",
        "what happened", "what's", "follow", "previous", "last run",
        "re-analyze", "reanalyze",
    ],
}


def _classify_human_intent(message_text: str) -> str:
    """
    Rule-based intent classifier for human messages.

    Checks each intent bucket and returns the first match.
    Falls back to "general_question" if no keywords match.
    """
    text_lower = message_text.lower()

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return intent

    return "general_question"


def classify_message(state: AgentState) -> AgentState:
    """
    LangGraph node: classify incoming message and set state["intent"].

    Robot messages containing experiment-completion signals → "experiment_complete".
    Human messages   → classified by keyword matching.
    Unknown source   → "general_question" as safe fallback.
    """
    source_type = state.get("source_type", "unknown")
    message_text = state.get("message_text", "")

    if source_type == "robot":
        # Check if this is an experiment-completion signal
        is_experiment_complete = any(
            pattern.search(message_text)
            for pattern in _ROBOT_EXPERIMENT_PATTERNS
        )
        intent = "experiment_complete" if is_experiment_complete else "general_question"

    elif source_type == "human":
        intent = _classify_human_intent(message_text)

    else:
        intent = "general_question"

    logger.info(f"Classified [{source_type}] message as intent='{intent}'")
    return {**state, "intent": intent}


def route_message(state: AgentState) -> str:
    """
    LangGraph conditional edge function.

    Returns the workflow key that LangGraph uses to select the next node.
    Mapping is defined in main_graph.py's add_conditional_edges call.
    """
    intent = state.get("intent", "general_question")

    route_map = {
        "experiment_complete": "analysis",
        "data_analysis":       "analysis",
        "literature_research": "literature",
        "follow_up_discussion":"discussion",
        "plot_request":        "plot_request",
        "general_question":    "general",
    }

    route = route_map.get(intent, "general")
    logger.info(f"Routing intent='{intent}' -> workflow='{route}'")
    return route

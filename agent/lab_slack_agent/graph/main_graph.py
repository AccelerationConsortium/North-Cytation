"""
main_graph.py — Top-level LangGraph graph.

Topology:
    classify_message
        │
        ▼ (conditional route based on intent)
    ┌───────────────────────────────────────────────────────────┐
    │ analysis_workflow  │ literature_workflow │                │
    │ discussion_workflow│ plot_request_workflow│ (general)     │
    └───────────────────────────────────────────────────────────┘
        │
        ▼
    format_final_response
        │
        ▼
       END

Sub-graphs (analysis, literature) are compiled separately and added as
nodes.  Discussion and plot-request are lightweight inline nodes defined
in agents/discussion_agent.py.
"""

import logging
from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.router import classify_message, route_message
from graph.analysis_graph import build_analysis_graph
from graph.literature_graph import build_literature_graph
from agents.discussion_agent import handle_discussion, handle_plot_request

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared utility node
# ─────────────────────────────────────────────────────────────────────────────

def format_final_response(state: AgentState) -> AgentState:
    """
    Ensure final_response is always populated before the graph ends.
    Sub-graphs should set it themselves; this is a safety fallback.
    """
    if not state.get("final_response"):
        intent = state.get("intent", "unknown")
        state = {
            **state,
            "final_response": (
                f"I processed your request (intent: {intent}) "
                "but no response was generated. Please check the logs."
            ),
        }
    return state


def handle_general_question(state: AgentState) -> AgentState:
    """Minimal handler for general questions not matched by other workflows."""
    # TODO: replace with a simple LLM call for open-ended questions
    return {
        **state,
        "final_response": (
            "I'm not sure how to classify your message. "
            "Try mentioning a specific action like 'analyze', 'research', or 'show plot'."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_main_graph():
    """Build and compile the top-level LangGraph graph."""
    g = StateGraph(AgentState)

    # ── Classifier node ──────────────────────────────────────────────────────
    g.add_node("classify_message", classify_message)

    # ── Sub-graph workflow nodes ─────────────────────────────────────────────
    g.add_node("analysis_workflow",      build_analysis_graph())
    g.add_node("literature_workflow",    build_literature_graph())
    g.add_node("discussion_workflow",    handle_discussion)
    g.add_node("plot_request_workflow",  handle_plot_request)
    g.add_node("general_workflow",       handle_general_question)

    # ── Final formatting node ────────────────────────────────────────────────
    g.add_node("format_final_response",  format_final_response)

    # ── Entry point ──────────────────────────────────────────────────────────
    g.set_entry_point("classify_message")

    # ── Conditional routing ──────────────────────────────────────────────────
    g.add_conditional_edges(
        "classify_message",
        route_message,
        {
            "analysis":    "analysis_workflow",
            "literature":  "literature_workflow",
            "discussion":  "discussion_workflow",
            "plot_request":"plot_request_workflow",
            "general":     "general_workflow",
        },
    )

    # ── All workflows converge to format_final_response ──────────────────────
    for workflow_node in [
        "analysis_workflow",
        "literature_workflow",
        "discussion_workflow",
        "plot_request_workflow",
        "general_workflow",
    ]:
        g.add_edge(workflow_node, "format_final_response")

    g.add_edge("format_final_response", END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Singleton compiled graph + public entry point
# ─────────────────────────────────────────────────────────────────────────────

_compiled_graph = None


def run_graph(initial_state: dict, slack_client=None) -> AgentState:
    """
    Public entry point called by event_handler.py.

    Lazily compiles the graph on first call and caches it.
    Pass slack_client so upload nodes can post files to Slack directly.
    """
    global _compiled_graph
    if _compiled_graph is None:
        logger.info("Compiling LangGraph main graph...")
        _compiled_graph = build_main_graph()

    config: dict = {}
    if slack_client is not None:
        config = {"configurable": {"slack_client": slack_client}}

    result = _compiled_graph.invoke(initial_state, config=config)
    return result

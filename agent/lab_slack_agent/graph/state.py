"""
state.py — Shared LangGraph state definition.

AgentState is a TypedDict that flows through every node in every graph.
All fields are optional (total=False) so each node only needs to set
the fields it is responsible for.
"""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # ── Input context ────────────────────────────────────────────────────────
    message_text: str           # Raw Slack message text
    user_id: str                # Slack user ID of the sender
    channel_id: str             # Slack channel where the message was posted
    thread_ts: str              # Thread timestamp — used for in-thread replies
    source_type: str            # "robot" | "human" | "unknown"

    # ── Classification ───────────────────────────────────────────────────────
    intent: str
    # Valid intents:
    #   "experiment_complete"    — robot says an experiment finished
    #   "data_analysis"          — human asks for data analysis
    #   "literature_research"    — human asks for literature search
    #   "follow_up_discussion"   — human follow-up in an existing thread
    #   "plot_request"           — human asks to see a specific figure
    #   "general_question"       — anything else

    # ── Experiment / run context ─────────────────────────────────────────────
    run_id: Optional[str]       # Experiment run ID parsed from the message
    data_path: Optional[str]    # File-system path to experiment data

    # ── Analysis results ─────────────────────────────────────────────────────
    analysis_results: Optional[Dict[str, Any]]      # Key metrics dict
    analysis_summary: Optional[str]                  # LLM-written report text
    warnings: Optional[List[str]]                    # Anomaly / QC warnings
    generated_plot_paths: Optional[List[str]]        # All plots created
    recommended_plot_paths: Optional[List[str]]      # Top 2-4 plots to upload
    optional_data_file_paths: Optional[List[str]]    # Extra data files

    # ── Slack file tracking ──────────────────────────────────────────────────
    uploaded_file_ids: Optional[List[str]]           # IDs returned by Slack

    # ── Plot requests ────────────────────────────────────────────────────────
    requested_plot_type: Optional[str]   # e.g. "residual", "pareto", "calibration"
    plot_file_path: Optional[str]        # Resolved path to the requested plot

    # ── Literature ───────────────────────────────────────────────────────────
    research_query: Optional[str]                   # Extracted search query
    papers: Optional[List[Dict[str, Any]]]          # Paper metadata list
    literature_summary: Optional[str]               # Synthesized answer text

    # ── Conversation memory ──────────────────────────────────────────────────
    conversation_context: Optional[List[Dict[str, str]]]  # Recent thread messages

    # ── Final output ─────────────────────────────────────────────────────────
    final_response: Optional[str]   # Text that will be posted back to Slack

    # ── Error handling ───────────────────────────────────────────────────────
    error: Optional[str]            # Set if a node encounters a recoverable error

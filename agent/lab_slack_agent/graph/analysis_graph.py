"""
analysis_graph.py — LangGraph sub-graph for experiment analysis.

Workflow nodes (executed in order):
  1.  extract_run_info           — parse run_id and data_path from message
  2.  load_experiment_data       — load data from disk (or placeholder)
  3.  inspect_data_schema        — examine columns and shape
  4.  run_standard_analysis      — compute key metrics
  5.  generate_plots             — produce standard figure PNGs
  6.  detect_anomalies           — flag outliers / failed wells
  7.  compare_to_previous_runs   — trend comparison (placeholder)
  8.  choose_key_figures         — select 2-4 most important plots
  9.  write_analysis_report      — LLM writes structured report
  10. upload_figures_to_slack    — upload recommended figures to Slack thread

All tool functions are imported from tools/ and agents/ so this file
stays as a pure workflow definition.
"""

import logging
from langgraph.graph import StateGraph, END
from langgraph.types import RunnableConfig

from graph.state import AgentState
from tools.data_tools import extract_run_id, load_run_data, summarize_dataframe, run_quality_checks
from tools.stats_tools import detect_outliers, compare_to_previous_runs
from tools.plot_tools import generate_standard_plots, select_best_figures
from agents.analysis_agent import write_report_with_llm
from slack.file_upload import upload_multiple_files

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Node functions
# ─────────────────────────────────────────────────────────────────────────────

def extract_run_info(state: AgentState) -> AgentState:
    """Node 1 — Extract run_id and data_path from the incoming message."""
    message_text = state.get("message_text", "")
    run_id, data_path = extract_run_id(message_text)
    logger.info(f"Extracted run_id={run_id!r}, data_path={data_path!r}")
    return {**state, "run_id": run_id, "data_path": data_path}


def load_experiment_data(state: AgentState) -> AgentState:
    """Node 2 — Load experiment data from disk using run_id or data_path."""
    run_id = state.get("run_id")
    data_path = state.get("data_path")

    df, load_error = load_run_data(run_id=run_id, data_path=data_path)

    updates: dict = {}
    if load_error:
        logger.warning(f"Data load warning: {load_error}")
        updates["warnings"] = (state.get("warnings") or []) + [load_error]

    # Store the DataFrame in analysis_results for downstream nodes
    updates["analysis_results"] = {
        **(state.get("analysis_results") or {}),
        "dataframe": df,
    }
    return {**state, **updates}


def inspect_data_schema(state: AgentState) -> AgentState:
    """Node 3 — Log and store basic schema information about the loaded data."""
    df = (state.get("analysis_results") or {}).get("dataframe")
    if df is None:
        return state

    schema_summary = summarize_dataframe(df)
    logger.info(f"Data schema: {schema_summary}")
    return {
        **state,
        "analysis_results": {
            **(state.get("analysis_results") or {}),
            "schema_summary": schema_summary,
        },
    }


def run_standard_analysis(state: AgentState) -> AgentState:
    """Node 4 — Compute standard quality metrics from the loaded data."""
    df = (state.get("analysis_results") or {}).get("dataframe")
    if df is None:
        return state

    qc_metrics = run_quality_checks(df)
    logger.info(f"QC metrics computed: {list(qc_metrics.keys())}")
    return {
        **state,
        "analysis_results": {
            **(state.get("analysis_results") or {}),
            "qc_metrics": qc_metrics,
        },
    }


def generate_plots(state: AgentState) -> AgentState:
    """Node 5 — Generate standard diagnostic plots and save as PNG files."""
    from config import PLOTS_DIR

    df = (state.get("analysis_results") or {}).get("dataframe")
    run_id = state.get("run_id") or "unknown_run"

    # Save plots under outputs/plots/{run_id}/
    output_dir = PLOTS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = generate_standard_plots(df, output_dir=str(output_dir), run_id=run_id)
    logger.info(f"Generated {len(plot_paths)} plot(s): {plot_paths}")
    return {**state, "generated_plot_paths": plot_paths}


def detect_anomalies(state: AgentState) -> AgentState:
    """Node 6 — Detect outliers and failed wells; append to warnings."""
    df = (state.get("analysis_results") or {}).get("dataframe")
    if df is None:
        return state

    anomaly_warnings = detect_outliers(df)
    all_warnings = (state.get("warnings") or []) + anomaly_warnings
    logger.info(f"Anomaly detection found {len(anomaly_warnings)} issue(s)")
    return {**state, "warnings": all_warnings}


def compare_to_previous_runs_node(state: AgentState) -> AgentState:
    """Node 7 — Compare current run metrics against historical runs."""
    run_id = state.get("run_id")
    comparison = compare_to_previous_runs(run_id)
    return {
        **state,
        "analysis_results": {
            **(state.get("analysis_results") or {}),
            "comparison": comparison,
        },
    }


def choose_key_figures(state: AgentState) -> AgentState:
    """Node 8 — Select 2-4 most informative figures to upload to Slack."""
    all_plots = state.get("generated_plot_paths") or []
    recommended = select_best_figures(all_plots, max_figures=4)
    logger.info(f"Selected {len(recommended)} key figures for upload")
    return {**state, "recommended_plot_paths": recommended}


def write_analysis_report(state: AgentState) -> AgentState:
    """Node 9 — Use the LLM to write a structured analysis report."""
    report = write_report_with_llm(state)
    logger.info("Analysis report written.")
    return {**state, "analysis_summary": report, "final_response": report}


def upload_figures_to_slack(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Node 10 — Upload the recommended figures to the Slack thread.

    The Slack client must be passed via LangGraph's configurable:
        graph.invoke(state, config={"configurable": {"slack_client": client}})

    If no client is available (e.g. during testing), this node is a no-op.
    """
    slack_client = (config.get("configurable") or {}).get("slack_client")
    plot_paths = state.get("recommended_plot_paths") or []

    if not slack_client:
        logger.warning("upload_figures_to_slack: no Slack client in config — skipping upload.")
        return state

    if not plot_paths:
        logger.info("upload_figures_to_slack: no plots to upload.")
        return state

    channel_id = state.get("channel_id", "")
    thread_ts = state.get("thread_ts", "")

    try:
        responses = upload_multiple_files(
            client=slack_client,
            channel_id=channel_id,
            thread_ts=thread_ts,
            file_paths=plot_paths,
        )
        uploaded_ids = [
            r.get("file", {}).get("id", "") for r in responses if r
        ]
        logger.info(f"Uploaded {len(uploaded_ids)} figure(s) to Slack")
        return {**state, "uploaded_file_ids": uploaded_ids}
    except Exception as exc:
        logger.error(f"Figure upload failed: {exc}")
        return {**state, "warnings": (state.get("warnings") or []) + [f"Figure upload failed: {exc}"]}


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_analysis_graph():
    """
    Compile and return the analysis sub-graph.

    Returns a CompiledGraph that can be invoked directly or added as a
    sub-graph node inside main_graph.
    """
    g = StateGraph(AgentState)

    g.add_node("extract_run_info", extract_run_info)
    g.add_node("load_experiment_data", load_experiment_data)
    g.add_node("inspect_data_schema", inspect_data_schema)
    g.add_node("run_standard_analysis", run_standard_analysis)
    g.add_node("generate_plots", generate_plots)
    g.add_node("detect_anomalies", detect_anomalies)
    g.add_node("compare_to_previous_runs", compare_to_previous_runs_node)
    g.add_node("choose_key_figures", choose_key_figures)
    g.add_node("write_analysis_report", write_analysis_report)
    g.add_node("upload_figures_to_slack", upload_figures_to_slack)

    # Linear pipeline
    g.set_entry_point("extract_run_info")
    g.add_edge("extract_run_info",          "load_experiment_data")
    g.add_edge("load_experiment_data",      "inspect_data_schema")
    g.add_edge("inspect_data_schema",       "run_standard_analysis")
    g.add_edge("run_standard_analysis",     "generate_plots")
    g.add_edge("generate_plots",            "detect_anomalies")
    g.add_edge("detect_anomalies",          "compare_to_previous_runs")
    g.add_edge("compare_to_previous_runs",  "choose_key_figures")
    g.add_edge("choose_key_figures",        "write_analysis_report")
    g.add_edge("write_analysis_report",     "upload_figures_to_slack")
    g.add_edge("upload_figures_to_slack",   END)

    return g.compile()

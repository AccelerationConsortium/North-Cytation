"""
literature_graph.py — LangGraph sub-graph for literature research.

Workflow nodes (executed in order):
  1. extract_research_query   — parse a clean search query from the message
  2. search_literature        — query Semantic Scholar, CrossRef, PubMed
  3. rank_papers              — rank results by relevance
  4. summarize_top_papers     — LLM summarizes each top paper
  5. synthesize_answer        — LLM synthesizes a cohesive answer
  6. cite_sources             — format citations and store in state
"""

import logging
from langgraph.graph import StateGraph, END

from graph.state import AgentState
from tools.literature_tools import (
    search_semantic_scholar,
    search_crossref,
    search_pubmed,
    fetch_paper_metadata,
    rank_papers_by_relevance,
)
from agents.literature_agent import (
    extract_query_with_llm,
    summarize_paper_with_llm,
    synthesize_literature_answer,
)

logger = logging.getLogger(__name__)

# Maximum number of papers to fetch per source
_MAX_RESULTS_PER_SOURCE = 5
# Top N papers to fully summarize
_TOP_N_TO_SUMMARIZE = 3


# ─────────────────────────────────────────────────────────────────────────────
# Node functions
# ─────────────────────────────────────────────────────────────────────────────

def extract_research_query(state: AgentState) -> AgentState:
    """Node 1 — Extract a clean research query from the Slack message."""
    message_text = state.get("message_text", "")
    query = extract_query_with_llm(message_text)
    logger.info(f"Extracted research query: {query!r}")
    return {**state, "research_query": query}


def search_literature(state: AgentState) -> AgentState:
    """Node 2 — Search multiple academic sources and combine results."""
    query = state.get("research_query", state.get("message_text", ""))

    all_papers: list = []

    # Fetch from each source and collect; log but don't crash on partial failures
    for source_fn, source_name in [
        (search_semantic_scholar, "Semantic Scholar"),
        (search_crossref,         "CrossRef"),
        (search_pubmed,           "PubMed"),
    ]:
        try:
            results = source_fn(query, max_results=_MAX_RESULTS_PER_SOURCE)
            logger.info(f"{source_name}: {len(results)} result(s)")
            all_papers.extend(results)
        except Exception as exc:
            logger.warning(f"{source_name} search failed: {exc}")

    logger.info(f"Total papers retrieved: {len(all_papers)}")
    return {**state, "papers": all_papers}


def rank_papers(state: AgentState) -> AgentState:
    """Node 3 — Rank retrieved papers by relevance to the query."""
    query = state.get("research_query", "")
    papers = state.get("papers") or []

    if not papers:
        logger.warning("rank_papers: no papers to rank.")
        return state

    ranked = rank_papers_by_relevance(papers, query)
    logger.info(f"Papers ranked, top paper: {ranked[0].get('title', '?')!r}")
    return {**state, "papers": ranked}


def summarize_top_papers(state: AgentState) -> AgentState:
    """Node 4 — Use the LLM to summarize the top N papers."""
    papers = state.get("papers") or []
    query = state.get("research_query", "")

    top_papers = papers[:_TOP_N_TO_SUMMARIZE]
    for i, paper in enumerate(top_papers):
        try:
            summary = summarize_paper_with_llm(paper, query=query)
            paper["llm_summary"] = summary
        except Exception as exc:
            logger.warning(f"Could not summarize paper {i}: {exc}")
            paper["llm_summary"] = "(Summary unavailable)"

    return {**state, "papers": papers}


def synthesize_answer(state: AgentState) -> AgentState:
    """Node 5 — Synthesize a concise answer from the summaries."""
    papers = state.get("papers") or []
    query = state.get("research_query", "")

    answer = synthesize_literature_answer(papers=papers[:_TOP_N_TO_SUMMARIZE], query=query)
    logger.info("Literature synthesis complete.")
    return {**state, "literature_summary": answer}


def cite_sources(state: AgentState) -> AgentState:
    """Node 6 — Format citations and build the final Slack response."""
    papers = state.get("papers") or []
    summary = state.get("literature_summary") or ""

    # Build a simple numbered citation block
    citations = []
    for i, paper in enumerate(papers[:_TOP_N_TO_SUMMARIZE], start=1):
        title   = paper.get("title", "Unknown title")
        authors = paper.get("authors", ["Unknown"])
        year    = paper.get("year", "?")
        url     = paper.get("url", "")
        line = f"{i}. *{title}* — {', '.join(authors[:2])} ({year})"
        if url:
            line += f" <{url}|[link]>"
        citations.append(line)

    citation_block = "\n".join(citations) if citations else "_No papers found._"

    final_response = (
        f"*Literature Search Results*\n\n"
        f"{summary}\n\n"
        f"*References:*\n{citation_block}"
    )

    return {**state, "final_response": final_response}


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_literature_graph():
    """
    Compile and return the literature research sub-graph.

    Returns a CompiledGraph that can be invoked directly or added as a
    sub-graph node inside main_graph.
    """
    g = StateGraph(AgentState)

    g.add_node("extract_research_query", extract_research_query)
    g.add_node("search_literature",      search_literature)
    g.add_node("rank_papers",            rank_papers)
    g.add_node("summarize_top_papers",   summarize_top_papers)
    g.add_node("synthesize_answer",      synthesize_answer)
    g.add_node("cite_sources",           cite_sources)

    g.set_entry_point("extract_research_query")
    g.add_edge("extract_research_query", "search_literature")
    g.add_edge("search_literature",      "rank_papers")
    g.add_edge("rank_papers",            "summarize_top_papers")
    g.add_edge("summarize_top_papers",   "synthesize_answer")
    g.add_edge("synthesize_answer",      "cite_sources")
    g.add_edge("cite_sources",           END)

    return g.compile()

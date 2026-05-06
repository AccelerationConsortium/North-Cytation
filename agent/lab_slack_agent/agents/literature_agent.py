"""
literature_agent.py — LLM nodes for literature research workflow.

Functions:
  extract_query_with_llm   — distil a clean search query from a Slack message
  summarize_paper_with_llm — write a short relevance-focused paper summary
  synthesize_literature_answer — combine summaries into a cohesive answer
"""

import logging
from typing import Any, Dict, List

from config import get_llm
from prompts.literature_prompt import (
    LITERATURE_SYSTEM_PROMPT,
    build_query_extraction_prompt,
    build_paper_summary_prompt,
    build_synthesis_prompt,
)

logger = logging.getLogger(__name__)


def extract_query_with_llm(message_text: str) -> str:
    """
    Use the LLM to extract a clean, concise academic search query
    from a potentially verbose Slack message.

    Returns the raw query string.
    """
    prompt = build_query_extraction_prompt(message_text)
    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke([
            {"role": "system", "content": LITERATURE_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ])
        query = response.content.strip() if hasattr(response, "content") else message_text
        logger.info(f"LLM extracted query: {query!r}")
        return query
    except Exception as exc:
        logger.warning(f"Query extraction failed ({exc}), using raw message.")
        # Fall back to the raw message rather than silently using a bad default
        return message_text


def summarize_paper_with_llm(paper: Dict[str, Any], query: str = "") -> str:
    """
    Generate a short relevance-focused summary of a single paper.

    Args:
        paper: Dict with keys title, authors, abstract, year, url, etc.
        query: The original research question for context.

    Returns:
        2-3 sentence summary string.
    """
    prompt = build_paper_summary_prompt(paper=paper, query=query)
    try:
        llm = get_llm(temperature=0.2)
        response = llm.invoke([
            {"role": "system", "content": LITERATURE_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ])
        return response.content.strip() if hasattr(response, "content") else ""
    except Exception as exc:
        logger.warning(f"Paper summary failed: {exc}")
        return "(Summary could not be generated)"


def synthesize_literature_answer(
    papers: List[Dict[str, Any]],
    query: str = "",
) -> str:
    """
    Synthesize a cohesive answer to the research question from multiple
    paper summaries.

    The LLM must only use information present in the paper summaries.
    It must not invent studies or data.

    Returns a formatted answer string suitable for posting to Slack.
    """
    if not papers:
        return "_No papers were found for this query._"

    prompt = build_synthesis_prompt(papers=papers, query=query)
    try:
        llm = get_llm(temperature=0.3)
        response = llm.invoke([
            {"role": "system", "content": LITERATURE_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ])
        return response.content.strip() if hasattr(response, "content") else ""
    except Exception as exc:
        logger.error(f"Literature synthesis failed: {exc}")
        return f"_Synthesis failed: {exc}_"

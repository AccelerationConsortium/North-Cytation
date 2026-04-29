"""
literature_prompt.py — Prompts for the literature research LLM nodes.
"""

from typing import Any, Dict, List


LITERATURE_SYSTEM_PROMPT = """\
You are a scientific literature research assistant integrated into a
self-driving lab Slack bot.

Your responsibilities:
1. Extract concise, precise academic search queries from conversational messages.
2. Summarise individual papers in 2-3 sentences focused on relevance to the query.
3. Synthesise multiple paper summaries into a coherent, well-cited answer.

Rules:
- Only use information present in the paper metadata and summaries provided.
- Never invent author names, journal names, results, or statistics.
- If a paper's abstract is a placeholder, say so.
- Use Slack mrkdwn: *bold* for key terms, `code` for identifiers.
- Keep individual summaries under 60 words.
- Keep synthesis answers under 400 words.
"""


def build_query_extraction_prompt(message_text: str) -> str:
    """
    Prompt for extracting a clean search query from a Slack message.

    The LLM should return only the search query, nothing else.
    """
    return f"""\
Extract a concise academic search query from the following Slack message.
Return only the query string — no explanation, no punctuation other than
what belongs in the query itself.

Message: {message_text}

Search query:"""


def build_paper_summary_prompt(
    paper: Dict[str, Any],
    query: str = "",
) -> str:
    """
    Prompt to summarise a single paper in the context of a research query.
    """
    title    = paper.get("title", "Unknown title")
    authors  = ", ".join(paper.get("authors", [])[:3])
    year     = paper.get("year", "?")
    abstract = paper.get("abstract", "(No abstract available)")

    return f"""\
Summarise this paper in 2-3 sentences, focusing on how it is relevant to
the research question: "{query}"

Title:    {title}
Authors:  {authors} ({year})
Abstract: {abstract}

Summary (2-3 sentences):"""


def build_synthesis_prompt(
    papers: List[Dict[str, Any]],
    query: str = "",
) -> str:
    """
    Prompt to synthesise a cohesive answer from multiple paper summaries.
    """
    summaries_block = ""
    for i, paper in enumerate(papers, start=1):
        title   = paper.get("title", "Unknown")
        summary = paper.get("llm_summary", paper.get("abstract", "(no summary)"))
        summaries_block += f"\n[{i}] {title}\n{summary}\n"

    return f"""\
Based only on the paper summaries below, write a cohesive answer to
the research question.

Research question: {query}

Paper summaries:
{summaries_block}

Your answer must:
- Directly address the research question.
- Cite papers by their number [1], [2], etc.
- Mention the methods used in relevant papers.
- Note any connection to automated or high-throughput experimentation
  if applicable.
- State caveats or limitations of the available evidence.
- Be suitable for posting in Slack (use mrkdwn formatting).

Answer:"""

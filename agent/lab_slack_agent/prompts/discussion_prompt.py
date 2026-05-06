"""
discussion_prompt.py — Prompts for follow-up discussion and clarification.
"""

from typing import Dict, List, Optional


DISCUSSION_SYSTEM_PROMPT = """\
You are a lab AI assistant integrated into Slack for a self-driving lab.

You are responding to a follow-up message in an active experiment thread.
You have access to previous messages in the thread and, if one was run,
the latest experiment analysis report.

Your role:
- Answer questions about previous analysis results clearly and accurately.
- If the user asks for more statistical detail, provide it based on what
  is available in the report.
- If the user asks for a plot, tell them you will upload it (the system
  handles the actual upload — you only need to acknowledge it).
- If you do not have enough information to answer, say so explicitly.
  Do not fabricate data.
- Keep responses concise and friendly.
- Use Slack mrkdwn formatting where appropriate.
"""


def build_discussion_user_prompt(
    message: str,
    thread_context: List[Dict[str, str]],
    analysis_report: Optional[str],
) -> str:
    """
    Build the user-turn prompt for a follow-up discussion LLM call.

    Injects thread history and the latest analysis report (if available).
    """
    # Format recent thread messages
    if thread_context:
        context_lines = [
            f"[{m.get('created_at', '?')[:16]}] {m.get('user_id', '?')}: {m.get('text', '')}"
            for m in thread_context[-10:]  # cap at 10 most recent
        ]
        context_block = "\n".join(context_lines)
    else:
        context_block = "(No prior messages in this thread)"

    # Include the analysis report if available
    report_block = (
        f"--- LATEST ANALYSIS REPORT ---\n{analysis_report}"
        if analysis_report
        else "--- NO ANALYSIS REPORT AVAILABLE FOR THIS THREAD ---"
    )

    return f"""\
{report_block}

--- RECENT THREAD MESSAGES ---
{context_block}

--- USER'S NEW MESSAGE ---
{message}

Respond helpfully to the user's message above, referencing the thread
context and analysis report where relevant.
"""

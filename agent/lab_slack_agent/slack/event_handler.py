"""
event_handler.py — Slack event listeners.

Receives incoming Slack messages, decides whether to process them, and
invokes the LangGraph workflow.  After the graph returns, posts text
responses and uploads any generated figures back into the Slack thread.
"""

import logging
from slack_bolt import App

from config import ROBOT_BOT_USER_ID, HUMAN_USER_ID
from graph.main_graph import run_graph
from slack.file_upload import post_text, upload_multiple_files

logger = logging.getLogger(__name__)

# Cache the bot's own user ID to avoid infinite self-reply loops
_BOT_USER_ID_CACHE: str = ""


def _get_bot_user_id(client) -> str:
    """Retrieve and cache the bot's own Slack user ID."""
    global _BOT_USER_ID_CACHE
    if not _BOT_USER_ID_CACHE:
        try:
            result = client.auth_test()
            _BOT_USER_ID_CACHE = result.get("user_id", "")
        except Exception as exc:
            logger.warning(f"Could not retrieve bot user ID: {exc}")
    return _BOT_USER_ID_CACHE


def _bot_is_mentioned(message_text: str, bot_user_id: str) -> bool:
    """Return True if the message contains a Slack mention of the bot."""
    return f"<@{bot_user_id}>" in message_text


def register_handlers(app: App) -> None:
    """Register all Slack event listeners on the Bolt app."""

    @app.event("message")
    def handle_message(event: dict, client) -> None:
        """
        Main message listener.

        Processing rules:
        - Ignore messages sent by the bot itself (loop prevention).
        - Ignore non-user message subtypes (joins, edits, etc.).
        - Robot bot (ROBOT_BOT_USER_ID): any message is forwarded to the graph.
        - Human user (HUMAN_USER_ID): only forwarded if the bot is @mentioned.
        - All other senders: ignored.
        """
        # ── Guard: ignore bot's own messages ────────────────────────────────
        bot_id = _get_bot_user_id(client)
        sender = event.get("user", "")

        if sender == bot_id:
            return
        if event.get("subtype") in ("bot_message", "message_changed", "message_deleted"):
            return

        channel_id: str = event.get("channel", "")
        # thread_ts defaults to the message's own ts so replies stay threaded
        thread_ts: str = event.get("thread_ts") or event.get("ts", "")
        message_text: str = event.get("text", "")

        # ── Determine source type ────────────────────────────────────────────
        if sender == ROBOT_BOT_USER_ID:
            source_type = "robot"
            logger.info(f"Robot message received in {channel_id}: {message_text[:100]!r}")

        elif sender == HUMAN_USER_ID:
            # Human messages only processed when the bot is directly mentioned
            if not _bot_is_mentioned(message_text, bot_id):
                logger.debug("Human message without bot mention — ignoring.")
                return
            source_type = "human"
            logger.info(f"Human message received in {channel_id}: {message_text[:100]!r}")

        else:
            logger.debug(f"Message from unknown user {sender!r} — ignoring.")
            return

        # ── Build initial graph state ────────────────────────────────────────
        initial_state = {
            "message_text": message_text,
            "user_id": sender,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "source_type": source_type,
        }

        # ── Run LangGraph workflow ───────────────────────────────────────────
        try:
            final_state = run_graph(initial_state, slack_client=client)
        except Exception as exc:
            logger.exception(f"LangGraph execution failed: {exc}")
            try:
                post_text(
                    client,
                    channel_id,
                    thread_ts,
                    f":warning: An error occurred while processing your message:\n```{exc}```",
                )
            except Exception:
                pass
            return

        # ── Post text response ───────────────────────────────────────────────
        response_text = final_state.get("final_response")
        if response_text:
            post_text(client, channel_id, thread_ts, response_text)

        # ── Upload figures ───────────────────────────────────────────────────
        plot_paths = final_state.get("recommended_plot_paths") or []
        if plot_paths:
            upload_multiple_files(
                client=client,
                channel_id=channel_id,
                thread_ts=thread_ts,
                file_paths=plot_paths,
            )
            logger.info(f"Uploaded {len(plot_paths)} figure(s) to thread {thread_ts}")

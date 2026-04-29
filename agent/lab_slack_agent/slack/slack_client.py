"""
slack_client.py — Thin wrapper around the Slack WebClient.

Provides a singleton web client so other modules can import get_web_client()
without needing to re-initialise each time.
"""

import logging
from slack_sdk import WebClient
from config import SLACK_BOT_TOKEN

logger = logging.getLogger(__name__)

_web_client: WebClient | None = None


def get_web_client() -> WebClient:
    """Return (or create) the shared Slack WebClient instance."""
    global _web_client
    if _web_client is None:
        _web_client = WebClient(token=SLACK_BOT_TOKEN)
        logger.debug("Slack WebClient initialised.")
    return _web_client

"""
app.py — Main entry point for the Lab Slack Agent.

Run this file to start the bot:
    python app.py

The bot connects to Slack via Socket Mode (no public URL required).
"""

import logging
import sys

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Config is imported first so .env is loaded before anything else
from config import SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SLACK_SIGNING_SECRET
from slack.event_handler import register_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def create_app() -> App:
    """Initialise and configure the Slack Bolt application."""
    app = App(
        token=SLACK_BOT_TOKEN,
        signing_secret=SLACK_SIGNING_SECRET,
    )
    register_handlers(app)
    logger.info("Slack event handlers registered.")
    return app


if __name__ == "__main__":
    logger.info("Starting Lab Slack Agent (Socket Mode)...")
    app = create_app()
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()

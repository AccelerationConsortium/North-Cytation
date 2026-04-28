"""
config.py — Central configuration loader.

All secrets are read from the .env file located in this directory.
Never hardcode API keys — fill in agent/lab_slack_agent/.env (copied
from .env.example) before running.
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from the same directory as this file so the app can be run
# from any working directory without breaking relative paths.
_ENV_PATH = Path(__file__).parent / ".env"

if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
    logger.info(f"Loaded environment from {_ENV_PATH}")
else:
    logger.warning(
        f".env file not found at {_ENV_PATH}. "
        "Copy .env.example to .env and fill in your credentials before starting."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require(key: str) -> str:
    """Return a required env var.  Raises clearly if missing."""
    val = os.environ.get(key, "").strip()
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Check your agent/lab_slack_agent/.env file."
        )
    return val


def _optional(key: str, default: str = "") -> str:
    """Return an optional env var with a fallback default."""
    return os.environ.get(key, default).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Slack
# ─────────────────────────────────────────────────────────────────────────────

SLACK_BOT_TOKEN: str = _require("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN: str = _require("SLACK_APP_TOKEN")
SLACK_SIGNING_SECRET: str = _require("SLACK_SIGNING_SECRET")

# ─────────────────────────────────────────────────────────────────────────────
# User / bot IDs
# ─────────────────────────────────────────────────────────────────────────────

ROBOT_BOT_USER_ID: str = _require("ROBOT_BOT_USER_ID")
HUMAN_USER_ID: str = _require("HUMAN_USER_ID")

# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────

LLM_PROVIDER: str = _optional("LLM_PROVIDER", "openai").lower()
LLM_MODEL: str = _optional("LLM_MODEL", "gpt-4o")
OPENAI_API_KEY: str = _optional("OPENAI_API_KEY")
ANTHROPIC_API_KEY: str = _optional("ANTHROPIC_API_KEY")

# ─────────────────────────────────────────────────────────────────────────────
# Data / output paths (always inside the agent folder — never in workspace root)
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT: str = _optional("DATA_ROOT", str(Path(__file__).parent / "outputs"))

OUTPUTS_DIR: Path = Path(__file__).parent / "outputs"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

# Create output directories on import so tools can always write to them
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.3):
    """
    Return a LangChain chat model based on LLM_PROVIDER.

    Supports "openai" (default) and "anthropic".
    Set LLM_PROVIDER and the corresponding API key in .env before calling.
    """
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=temperature,
            api_key=OPENAI_API_KEY,
        )

    elif LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        from langchain_anthropic import ChatAnthropic  # noqa: PLC0415
        return ChatAnthropic(
            model=LLM_MODEL,
            temperature=temperature,
            api_key=ANTHROPIC_API_KEY,
        )

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{LLM_PROVIDER}'. "
            "Set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic in .env."
        )

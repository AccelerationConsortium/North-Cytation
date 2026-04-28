"""
thread_memory.py — SQLite persistence for Slack thread context.

Stores per-thread data so the discussion and plot-request workflows can
recall what experiment was analysed and what was previously discussed.

Database file location: memory/thread_memory.db
(gitignored — never committed)

Tables:
  threads            — channel + thread_ts + run_id mapping
  messages           — recent message history per thread
  reports            — analysis reports per thread
  literature_results — paper search results per thread
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# DB lives alongside this source file so it is always inside the agent folder
_DEFAULT_DB_PATH = Path(__file__).parent / "thread_memory.db"

# How many recent messages to return in context
_MAX_CONTEXT_MESSAGES = 20


class ThreadMemory:
    """
    SQLite-backed store for Slack thread state.

    Thread-safe for single-process use.  For multi-process deployments,
    switch the backend to PostgreSQL or Redis.
    """

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    # ── Schema initialisation ────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS threads (
                    channel    TEXT    NOT NULL,
                    thread_ts  TEXT    NOT NULL,
                    run_id     TEXT,
                    created_at TEXT    NOT NULL,
                    PRIMARY KEY (channel, thread_ts)
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel    TEXT    NOT NULL,
                    thread_ts  TEXT    NOT NULL,
                    user_id    TEXT    NOT NULL,
                    text       TEXT    NOT NULL,
                    created_at TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reports (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel            TEXT    NOT NULL,
                    thread_ts          TEXT    NOT NULL,
                    run_id             TEXT,
                    report_text        TEXT,
                    plot_paths         TEXT,   -- JSON array
                    uploaded_file_ids  TEXT,   -- JSON array
                    created_at         TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS literature_results (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel    TEXT    NOT NULL,
                    thread_ts  TEXT    NOT NULL,
                    query      TEXT,
                    papers     TEXT,   -- JSON array
                    created_at TEXT    NOT NULL
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        """Return a context-managed SQLite connection with row_factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Thread / run_id mapping ──────────────────────────────────────────────

    def register_thread(
        self,
        channel: str,
        thread_ts: str,
        run_id: Optional[str] = None,
    ) -> None:
        """Create or update the thread record with an optional run_id."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO threads (channel, thread_ts, run_id, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (channel, thread_ts)
                DO UPDATE SET run_id = COALESCE(excluded.run_id, run_id)
                """,
                (channel, thread_ts, run_id, self._now()),
            )

    def get_run_id_for_thread(
        self,
        channel: str,
        thread_ts: str,
    ) -> Optional[str]:
        """Return the run_id associated with a thread, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_id FROM threads WHERE channel=? AND thread_ts=?",
                (channel, thread_ts),
            ).fetchone()
        return row["run_id"] if row else None

    # ── Message history ──────────────────────────────────────────────────────

    def save_message(
        self,
        channel: str,
        thread_ts: str,
        user_id: str,
        text: str,
    ) -> None:
        """Persist a single Slack message to the messages table."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (channel, thread_ts, user_id, text, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (channel, thread_ts, user_id, text, self._now()),
            )

    def get_thread_context(
        self,
        channel: str,
        thread_ts: str,
        limit: int = _MAX_CONTEXT_MESSAGES,
    ) -> List[Dict[str, str]]:
        """
        Return recent messages in a thread as a list of dicts.

        Each dict has keys: user_id, text, created_at.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_id, text, created_at
                FROM messages
                WHERE channel=? AND thread_ts=?
                ORDER BY id DESC
                LIMIT ?
                """,
                (channel, thread_ts, limit),
            ).fetchall()
        # Return chronological order
        return [dict(r) for r in reversed(rows)]

    # ── Analysis reports ─────────────────────────────────────────────────────

    def save_report(
        self,
        channel: str,
        thread_ts: str,
        run_id: Optional[str],
        report_text: str,
        plot_paths: Optional[List[str]] = None,
        uploaded_file_ids: Optional[List[str]] = None,
    ) -> None:
        """Persist an analysis report for a thread."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reports
                    (channel, thread_ts, run_id, report_text,
                     plot_paths, uploaded_file_ids, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    channel,
                    thread_ts,
                    run_id,
                    report_text,
                    json.dumps(plot_paths or []),
                    json.dumps(uploaded_file_ids or []),
                    self._now(),
                ),
            )
        # Also update the thread's run_id if provided
        if run_id:
            self.register_thread(channel=channel, thread_ts=thread_ts, run_id=run_id)

    def get_latest_report(
        self,
        channel: str,
        thread_ts: str,
    ) -> Optional[str]:
        """Return the most recent analysis report text for a thread."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT report_text
                FROM reports
                WHERE channel=? AND thread_ts=?
                ORDER BY id DESC
                LIMIT 1
                """,
                (channel, thread_ts),
            ).fetchone()
        return row["report_text"] if row else None

    # ── Literature results ───────────────────────────────────────────────────

    def save_literature_results(
        self,
        channel: str,
        thread_ts: str,
        query: str,
        papers: List[Dict[str, Any]],
    ) -> None:
        """Persist a literature search result set."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO literature_results
                    (channel, thread_ts, query, papers, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (channel, thread_ts, query, json.dumps(papers), self._now()),
            )

    def get_latest_literature_results(
        self,
        channel: str,
        thread_ts: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Return the most recent paper list for a thread."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT papers
                FROM literature_results
                WHERE channel=? AND thread_ts=?
                ORDER BY id DESC
                LIMIT 1
                """,
                (channel, thread_ts),
            ).fetchone()
        if row:
            try:
                return json.loads(row["papers"])
            except json.JSONDecodeError:
                return None
        return None

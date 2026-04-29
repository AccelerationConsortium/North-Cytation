"""
test_router.py — Unit tests for message classification and routing.

Run with:
    cd agent/lab_slack_agent
    pytest tests/test_router.py -v

These tests do NOT require Slack credentials or LLM API keys.
They verify the deterministic rule-based classification logic only.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so imports resolve without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch

# Patch config so it doesn't raise on missing .env values during testing
_FAKE_ENV = {
    "SLACK_BOT_TOKEN":     "xoxb-fake",
    "SLACK_APP_TOKEN":     "xapp-fake",
    "SLACK_SIGNING_SECRET":"fake-secret",
    "ROBOT_BOT_USER_ID":   "U_ROBOT",
    "HUMAN_USER_ID":       "U_HUMAN",
}
with patch.dict("os.environ", _FAKE_ENV):
    from graph.router import classify_message, route_message, _classify_human_intent
    from tools.data_tools import extract_run_id


# ─────────────────────────────────────────────────────────────────────────────
# classify_message tests
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyMessage:
    """Tests for the classify_message LangGraph node."""

    def _state(self, source_type: str, message: str) -> dict:
        return {"source_type": source_type, "message_text": message}

    # --- Robot messages ---

    def test_robot_experiment_complete_keyword(self):
        state = self._state("robot", "Experiment complete: run_id=12345")
        result = classify_message(state)
        assert result["intent"] == "experiment_complete"

    def test_robot_run_finished(self):
        state = self._state("robot", "Run finished: path=/data/runs/2026-04-28/run_001")
        result = classify_message(state)
        assert result["intent"] == "experiment_complete"

    def test_robot_calibration_complete(self):
        state = self._state("robot", "Calibration experiment completed successfully")
        result = classify_message(state)
        assert result["intent"] == "experiment_complete"

    def test_robot_run_id_equals_pattern(self):
        state = self._state("robot", "Job done run_id=XYZ-099")
        result = classify_message(state)
        assert result["intent"] == "experiment_complete"

    def test_robot_unknown_message(self):
        state = self._state("robot", "Heartbeat ping")
        result = classify_message(state)
        assert result["intent"] == "general_question"

    # --- Human messages ---

    def test_human_data_analysis(self):
        state = self._state("human", "<@UBOT> please analyze the results")
        result = classify_message(state)
        assert result["intent"] == "data_analysis"

    def test_human_literature_research(self):
        state = self._state("human", "<@UBOT> research papers on CMC determination")
        result = classify_message(state)
        assert result["intent"] == "literature_research"

    def test_human_plot_request(self):
        state = self._state("human", "<@UBOT> show me the residual plot")
        result = classify_message(state)
        assert result["intent"] == "plot_request"

    def test_human_pareto_upload(self):
        state = self._state("human", "<@UBOT> upload the pareto front")
        result = classify_message(state)
        assert result["intent"] == "plot_request"

    def test_human_follow_up(self):
        state = self._state("human", "<@UBOT> why did those wells fail?")
        result = classify_message(state)
        assert result["intent"] == "follow_up_discussion"

    def test_human_general_question(self):
        state = self._state("human", "<@UBOT> hello")
        result = classify_message(state)
        assert result["intent"] == "general_question"

    # --- Unknown source ---

    def test_unknown_source_defaults_to_general(self):
        state = self._state("unknown", "some message")
        result = classify_message(state)
        assert result["intent"] == "general_question"

    # --- State passthrough ---

    def test_state_is_preserved(self):
        state = self._state("human", "<@UBOT> analyze data")
        state["channel_id"] = "C123"
        state["thread_ts"] = "1234.5678"
        result = classify_message(state)
        assert result["channel_id"] == "C123"
        assert result["thread_ts"] == "1234.5678"


# ─────────────────────────────────────────────────────────────────────────────
# route_message tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteMessage:
    """Tests for the route_message conditional edge function."""

    def test_route_experiment_complete(self):
        assert route_message({"intent": "experiment_complete"}) == "analysis"

    def test_route_data_analysis(self):
        assert route_message({"intent": "data_analysis"}) == "analysis"

    def test_route_literature(self):
        assert route_message({"intent": "literature_research"}) == "literature"

    def test_route_discussion(self):
        assert route_message({"intent": "follow_up_discussion"}) == "discussion"

    def test_route_plot_request(self):
        assert route_message({"intent": "plot_request"}) == "plot_request"

    def test_route_general(self):
        assert route_message({"intent": "general_question"}) == "general"

    def test_route_unknown_intent_defaults_to_general(self):
        assert route_message({"intent": "something_weird"}) == "general"

    def test_route_missing_intent_defaults_to_general(self):
        assert route_message({}) == "general"


# ─────────────────────────────────────────────────────────────────────────────
# extract_run_id tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractRunId:
    """Tests for run ID parsing from robot messages."""

    def test_explicit_run_id_equals(self):
        run_id, _ = extract_run_id("Experiment complete: run_id=12345")
        assert run_id == "12345"

    def test_explicit_run_id_colon(self):
        run_id, _ = extract_run_id("run_id: abc-007")
        assert run_id == "abc-007"

    def test_path_extraction(self):
        _, data_path = extract_run_id("Run finished: path=/data/runs/2026-04-28/run_001")
        assert data_path == "/data/runs/2026-04-28/run_001"

    def test_run_id_derived_from_path(self):
        run_id, _ = extract_run_id("Run finished: path=/data/runs/2026-04-28/run_001")
        assert run_id == "run_001"

    def test_fallback_hash_is_stable(self):
        msg = "Calibration experiment completed"
        run_id_1, _ = extract_run_id(msg)
        run_id_2, _ = extract_run_id(msg)
        assert run_id_1 == run_id_2

    def test_both_none_gives_fallback(self):
        run_id, data_path = extract_run_id("No identifiers here")
        assert run_id is not None   # fallback hash
        assert data_path is None

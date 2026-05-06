"""
data_tools.py — Tools for loading and inspecting experiment data.

All functions return plain Python objects (no LLM calls).
The LLM in analysis_agent.py interprets these results — it does not
calculate them.

Placeholder implementations generate synthetic data so the workflow
runs end-to-end without real files on disk.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import DATA_ROOT

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Run ID / path extraction
# ─────────────────────────────────────────────────────────────────────────────

# Patterns for parsing robot completion messages
_RUN_ID_PATTERN    = re.compile(r"run_?id\s*[=:]\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)
_RUN_PATH_PATTERN  = re.compile(r"path\s*[=:]\s*([\S]+)", re.IGNORECASE)
_RUN_NAME_PATTERN  = re.compile(r"run[_\s]+([A-Za-z0-9_\-]+)", re.IGNORECASE)


def extract_run_id(message_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse run_id and data_path from a robot completion message.

    Examples handled:
        "Experiment complete: run_id=12345"
        "Run finished: path=/data/runs/2026-04-28/run_001"
        "Calibration experiment completed run_042"

    Returns:
        (run_id, data_path) — either may be None if not found.
    """
    run_id    = None
    data_path = None

    # Try explicit run_id=... pattern
    m = _RUN_ID_PATTERN.search(message_text)
    if m:
        run_id = m.group(1)

    # Try path=... pattern
    m = _RUN_PATH_PATTERN.search(message_text)
    if m:
        data_path = m.group(1)
        # Derive a run_id from the path if not already found
        if not run_id:
            run_id = Path(data_path).name

    # Fallback: look for generic "run_XYZ" in the message
    if not run_id:
        m = _RUN_NAME_PATTERN.search(message_text)
        if m:
            run_id = m.group(1)

    # Final fallback: use a hash of the message text
    if not run_id:
        run_id = f"run_{abs(hash(message_text)) % 100000:05d}"

    logger.debug(f"extract_run_id: run_id={run_id!r}, data_path={data_path!r}")
    return run_id, data_path


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_run_data(
    run_id: Optional[str] = None,
    data_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load experiment data for a given run_id or explicit data_path.

    Tries, in order:
      1. data_path (if provided)
      2. DATA_ROOT / run_id / *.csv
      3. Placeholder synthetic DataFrame (so the workflow always runs)

    Returns:
        (DataFrame, error_message)
        error_message is None on success, or a warning string if data
        could not be loaded and synthetic data was used.
    """
    # ── 1. Explicit path ─────────────────────────────────────────────────────
    if data_path:
        p = Path(data_path)
        if p.exists() and p.suffix == ".csv":
            try:
                df = pd.read_csv(p)
                logger.info(f"Loaded data from {p} ({len(df)} rows)")
                return df, None
            except Exception as exc:
                logger.warning(f"Failed to read {p}: {exc}")

    # ── 2. DATA_ROOT / run_id ────────────────────────────────────────────────
    if run_id:
        candidates = list(Path(DATA_ROOT).glob(f"**/{run_id}*.csv"))
        if candidates:
            p = candidates[0]
            try:
                df = pd.read_csv(p)
                logger.info(f"Loaded data from {p} ({len(df)} rows)")
                return df, None
            except Exception as exc:
                logger.warning(f"Failed to read {p}: {exc}")

    # ── 3. Placeholder synthetic data ────────────────────────────────────────
    logger.warning(
        f"No real data found for run_id={run_id!r}, data_path={data_path!r}. "
        "Using synthetic placeholder data."
    )
    df = _generate_placeholder_dataframe(run_id=run_id or "unknown")
    warning = (
        f"Could not locate real data for run `{run_id or 'unknown'}`. "
        "Results below are based on *synthetic placeholder data*."
    )
    return df, warning


def _generate_placeholder_dataframe(run_id: str) -> pd.DataFrame:
    """Generate a synthetic experiment DataFrame for testing."""
    rng = np.random.default_rng(seed=abs(hash(run_id)) % 2**31)
    n = 96  # typical 96-well plate

    df = pd.DataFrame({
        "well":          [f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)],
        "concentration": rng.uniform(0.0, 1.0, n).round(4),
        "absorbance_450": rng.normal(0.8, 0.15, n).clip(0).round(4),
        "absorbance_600": rng.normal(0.05, 0.02, n).clip(0).round(4),
        "cv_percent":    rng.uniform(0.5, 15.0, n).round(2),
        "passed_qc":     rng.choice([True, False], n, p=[0.88, 0.12]),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Schema inspection
# ─────────────────────────────────────────────────────────────────────────────

def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return a compact, JSON-serialisable summary of a DataFrame.
    Used by inspect_data_schema to describe the data to the LLM.
    """
    if df is None or df.empty:
        return {"error": "DataFrame is empty or None"}

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    summary: Dict[str, Any] = {
        "rows":          len(df),
        "columns":       df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes":         df.dtypes.astype(str).to_dict(),
    }

    # Basic stats for numeric columns
    if numeric_cols:
        desc = df[numeric_cols].describe().round(4)
        summary["stats"] = desc.to_dict()

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Quality checks
# ─────────────────────────────────────────────────────────────────────────────

def run_quality_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute standard quality-control metrics from the experiment DataFrame.

    Returns a dict of metric_name -> value (all JSON-serialisable).
    Extend this function with domain-specific checks as needed.

    TODO: add assay-specific thresholds from settings/ YAML files.
    """
    if df is None or df.empty:
        return {"error": "No data to check"}

    metrics: Dict[str, Any] = {
        "total_wells": len(df),
    }

    # Passed QC wells
    if "passed_qc" in df.columns:
        passed = int(df["passed_qc"].sum())
        metrics["passed_qc"]       = passed
        metrics["failed_qc"]       = len(df) - passed
        metrics["pass_rate_pct"]   = round(100 * passed / len(df), 1)

    # CV stats (coefficient of variation)
    if "cv_percent" in df.columns:
        metrics["mean_cv_pct"]   = round(float(df["cv_percent"].mean()), 2)
        metrics["max_cv_pct"]    = round(float(df["cv_percent"].max()), 2)
        metrics["high_cv_wells"] = int((df["cv_percent"] > 10).sum())

    # Signal range for absorbance columns
    for col in [c for c in df.columns if "absorbance" in c]:
        metrics[f"{col}_mean"] = round(float(df[col].mean()), 4)
        metrics[f"{col}_std"]  = round(float(df[col].std()), 4)

    return metrics

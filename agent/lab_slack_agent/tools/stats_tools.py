"""
stats_tools.py — Statistical analysis tools.

All functions operate on pandas DataFrames and return plain Python objects.
The LLM interprets these outputs — it never computes them directly.

Placeholder implementations are provided for:
  - detect_outliers
  - compare_to_previous_runs

TODO: Replace with domain-specific statistical methods as needed.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Outlier / anomaly detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_outliers(
    df: Optional[pd.DataFrame],
    z_threshold: float = 3.0,
    cv_threshold: float = 20.0,
) -> List[str]:
    """
    Flag statistical outliers and high-CV wells in the DataFrame.

    Uses Z-score on numeric columns (|z| > z_threshold) and
    optionally a CV column threshold.

    Args:
        df:             Experiment DataFrame.
        z_threshold:    Z-score magnitude above which a value is an outlier.
        cv_threshold:   CV % above which a well is flagged.

    Returns:
        List of human-readable warning strings (empty if no issues found).

    TODO: add domain-specific rules (e.g. blank well subtraction, saturation).
    """
    if df is None or df.empty:
        return ["No data available for outlier detection."]

    warnings: List[str] = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Z-score outlier detection per numeric column
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 3:
            continue
        z = (series - series.mean()) / series.std(ddof=1)
        outlier_indices = series.index[np.abs(z) > z_threshold].tolist()
        if outlier_indices:
            n = len(outlier_indices)
            warnings.append(
                f"Column '{col}': {n} outlier(s) detected "
                f"(|z| > {z_threshold}) at indices {outlier_indices[:5]}"
                + (" ..." if n > 5 else "")
            )

    # CV threshold check
    if "cv_percent" in df.columns:
        high_cv = df[df["cv_percent"] > cv_threshold]
        if not high_cv.empty:
            wells_str = (
                ", ".join(high_cv["well"].astype(str).tolist()[:10])
                if "well" in high_cv.columns
                else str(len(high_cv))
            )
            warnings.append(
                f"{len(high_cv)} well(s) with CV > {cv_threshold}%: {wells_str}"
            )

    # Failed QC check
    if "passed_qc" in df.columns:
        failed = int((~df["passed_qc"]).sum())
        if failed > 0:
            warnings.append(f"{failed} well(s) failed QC filters.")

    if not warnings:
        logger.info("detect_outliers: no anomalies found.")

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Cross-run comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_to_previous_runs(
    run_id: Optional[str],
    n_previous: int = 3,
) -> Dict[str, Any]:
    """
    Compare key metrics of the current run against recent historical runs.

    Currently a placeholder that returns synthetic comparison data.

    TODO:
      - Load historical metrics from a database or CSV archive.
      - Compute deltas against rolling mean.
      - Flag regressions (e.g. pass rate drops > 5%).

    Args:
        run_id:       Current run identifier.
        n_previous:   Number of prior runs to compare against.

    Returns:
        Dict with keys:
          - previous_runs:  list of run IDs compared
          - metric_deltas:  dict of metric_name -> delta from mean
          - trend_notes:    list of narrative observations
    """
    # ── Placeholder implementation ───────────────────────────────────────────
    logger.info(f"compare_to_previous_runs: placeholder for run_id={run_id!r}")

    # Simulate historical metrics
    rng = np.random.default_rng(seed=abs(hash(str(run_id))) % 2**31)
    prev_pass_rates = rng.uniform(84, 95, n_previous).round(1).tolist()
    prev_mean_cv    = rng.uniform(3.5, 7.5, n_previous).round(2).tolist()

    current_pass_rate = rng.uniform(85, 95)
    current_mean_cv   = rng.uniform(3.5, 7.5)

    mean_hist_pass = float(np.mean(prev_pass_rates))
    mean_hist_cv   = float(np.mean(prev_mean_cv))

    return {
        "previous_runs": [f"run_sim_{i:03d}" for i in range(1, n_previous + 1)],
        "metric_deltas": {
            "pass_rate_pct": round(current_pass_rate - mean_hist_pass, 1),
            "mean_cv_pct":   round(current_mean_cv   - mean_hist_cv,   2),
        },
        "trend_notes": [
            f"Pass rate {'+' if current_pass_rate >= mean_hist_pass else ''}"
            f"{current_pass_rate - mean_hist_pass:.1f}% vs {n_previous}-run average.",
            f"Mean CV {'+' if current_mean_cv >= mean_hist_cv else ''}"
            f"{current_mean_cv - mean_hist_cv:.2f}% vs {n_previous}-run average.",
            "(NOTE: Comparison is based on placeholder data — connect to real run archive)",
        ],
    }

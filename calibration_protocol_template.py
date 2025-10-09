"""Calibration Protocol Template (minimal, volume-centric)

Unified interface (0.7.0+):
    initialize() -> state(dict)
    measure(state, volume_mL, params, replicates) -> list[dict]
    wrapup(state) -> None

Per replicate dict must include:
    replicate (int), volume (mL), elapsed_s (float)
Optional: start_time, end_time, echoed params.

If your hardware needs a controller object, capture/construct it inside
initialize and store it in state (e.g., state['hw']). Do NOT rely on an
external lash_e argument.
"""
from __future__ import annotations
from typing import Dict, Any, List
from datetime import datetime
import time


def initialize(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    # Optional: construct hardware or simulation resources, store in state.
    return {}


def measure(state: Dict[str, Any], volume_mL: float, params: Dict[str, Any], replicates: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for r in range(replicates):
        start_ts = datetime.now().isoformat()
        t0 = time.time()
        # Insert hardware or simulation steps here using objects in `state`.
        measured_volume = volume_mL  # Replace with actual measured/interpreted volume
        elapsed = time.time() - t0
        end_ts = datetime.now().isoformat()
        results.append({
            'replicate': r,
            'volume': measured_volume,
            'elapsed_s': elapsed,
            'start_time': start_ts,
            'end_time': end_ts,
            **params
        })
    return results


def wrapup(state: Dict[str, Any]) -> None:
    # Clean up hardware resources if present in state.
    return

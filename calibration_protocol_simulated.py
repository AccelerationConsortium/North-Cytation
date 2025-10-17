"""Simulated calibration protocol (self-contained, volume-centric).

Unified minimal interface (0.7.0+):
    initialize() -> state (dict)
    measure(state, volume_mL, params, replicates) -> list[dict]
    wrapup(state) -> None

Per-replicate result keys:
    replicate, volume (mL), elapsed_s
Optional: start_time, end_time, echoed params.
"""
from __future__ import annotations
import os, random
from datetime import datetime
from typing import Dict, Any, List, Tuple


def initialize(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:  # noqa: D401
    # Deterministic seeding precedence: cfg.random_seed > env CAL_SIM_SEED > no seed
    seeded = False
    if cfg and isinstance(cfg, dict) and 'random_seed' in cfg:
        try:
            random.seed(int(cfg['random_seed']))
            seeded = True
        except Exception:
            raise ValueError("random_seed in config must be an int convertible value")
    if not seeded:
        seed_env = os.environ.get("CAL_SIM_SEED")
        if seed_env is not None:
            try:
                random.seed(int(seed_env))
            except ValueError:
                pass
    return {"_sim": True, "_seeded": seeded}


def _simulate_once(target_vol: float, params: Dict[str, Any]) -> Tuple[float, float]:
    asp = params.get("aspirate_speed", 15)
    dsp = params.get("dispense_speed", 15)
    asp_wait = params.get("aspirate_wait_time", 5)
    dsp_wait = params.get("dispense_wait_time", 5)
    over = params.get("overaspirate_vol", 0.0)
    # Base pipetting bias: typically slightly under-delivers due to surface tension, viscosity
    base_bias = -0.005 * target_vol + 0.002 * (asp - 15)/15 + 0.002 * (dsp - 15)/15  # Negative bias = underdelivery
    
    # Overaspirate compensation: partially compensates for underdelivery but not perfectly  
    over_comp = min(over, target_vol * 0.25) * 0.7  # 70% effectiveness (was 80%)
    
    # Measured volume: target + bias - shortfall + compensation + noise
    measured = max(target_vol + base_bias + over_comp + random.gauss(0, target_vol*0.008), 0)
    speed_factor = (30 / max(asp,1) + 30 / max(dsp,1)) * 0.2
    wait_factor = (asp_wait + dsp_wait) * 0.05
    base_time = 1.2 + speed_factor + wait_factor + (target_vol * 4)
    elapsed = max(random.gauss(base_time, base_time*0.05), 0.1)
    return measured, elapsed


def measure(state: Dict[str, Any], volume_mL: float, params: Dict[str, Any], replicates: int) -> List[Dict[str, Any]]:
    debug_sim = os.environ.get("CAL_SIM_DEBUG", "0") == "1"
    results: List[Dict[str, Any]] = []
    for r in range(replicates):
        start_ts = datetime.now().isoformat()
        measured_volume, elapsed = _simulate_once(volume_mL, params)
        end_ts = start_ts
        results.append({
            "replicate": r,
            "elapsed_s": elapsed,
            "start_time": start_ts,
            "end_time": end_ts,
            "volume": measured_volume,
            **params
        })
        if debug_sim and r == 0:
            dev_ul = abs(measured_volume - volume_mL)*1000
            print(f"[sim-protocol] target_ul={volume_mL*1000:.1f} meas_ul={measured_volume*1000:.1f} dev_ul={dev_ul:.2f} time={elapsed:.2f}s")
    return results


def wrapup(state: Dict[str, Any]) -> None:  # noqa: D401
    return

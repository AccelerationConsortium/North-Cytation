"""Simulated calibration protocol for modular system.

Unified minimal interface:
    initialize(cfg) -> state (dict)
    measure(state, volume_mL, params, replicates) -> list[dict]
    wrapup(state) -> None

Per-replicate result keys:
    replicate, volume (mL), elapsed_s
Optional: start_time, end_time, echoed params.
"""
import os
import random
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple


def initialize(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Initialize simulation protocol."""
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
    """
    Simulate a single pipetting measurement using realistic physics from calibration_sdl_base.
    
    This simulation is calibrated against real robot data for realistic optimization behavior.
    """
    import numpy as np
    
    # Normalize parameters (handle different naming conventions)
    asp_speed = params.get("aspirate_speed", 15)
    dsp_speed = params.get("dispense_speed", 15)
    asp_wait = params.get("aspirate_wait_time", 5)
    dsp_wait = params.get("dispense_wait_time", 5)
    overasp_vol = params.get("overaspirate_vol", 0.0)
    blowout_vol = params.get("blowout_vol", 0.1)
    retract_speed = params.get("retract_speed", 8)
    post_asp_air_vol = params.get("post_asp_air_vol", 0.05)
    
    # Start with sophisticated parameter-dependent error modeling
    mass_error_factor = 0.0
    
    # Speed parameters: Faster speeds (lower numbers) less accurate but faster
    # Higher speeds (higher numbers) more accurate but slower - speed/accuracy tradeoff
    min_speed_penalty = 0.002  # Penalty for very fast speeds (accuracy issues)
    fast_speed_penalty_aspirate = max(0, (15 - asp_speed)) * min_speed_penalty
    fast_speed_penalty_dispense = max(0, (15 - dsp_speed)) * min_speed_penalty
    mass_error_factor += fast_speed_penalty_aspirate + fast_speed_penalty_dispense
    
    # Wait time parameters: Longer waits improve accuracy (settling time)
    wait_accuracy_benefit = -0.001  # Slight accuracy improvement with longer waits
    mass_error_factor += max(0, (5 - asp_wait)) * 0.002  # Penalty for very short waits
    mass_error_factor += max(0, (5 - dsp_wait)) * 0.002  # Penalty for very short waits
    mass_error_factor += asp_wait * wait_accuracy_benefit  # Small benefit from longer waits
    mass_error_factor += dsp_wait * wait_accuracy_benefit  # Small benefit from longer waits
    
    # VOLUME-DEPENDENT PARAMETERS - Critical for forcing selective optimization!
    # Different optimal values for different volumes to force re-optimization
    
    # blowout_vol: VERY volume dependent with strict optimal values
    if target_vol <= 0.03:  # Small volumes (0.025 mL)
        optimal_blowout = 0.03  # Very specific optimal value
    elif target_vol <= 0.06:  # Medium-small volumes (0.05 mL)  
        optimal_blowout = 0.07  # Different optimal value
    elif target_vol <= 0.15:  # Medium volumes (0.1 mL)
        optimal_blowout = 0.11  # Yet another optimal value
    else:  # Large volumes (0.5 mL)
        optimal_blowout = 0.15  # High optimal value
    
    # STRICT penalty - if you're more than 0.03 away from optimal, big penalty
    blowout_error = np.abs(blowout_vol - optimal_blowout)
    if blowout_error > 0.03:  # Sharp penalty threshold
        mass_error_factor += blowout_error * 0.4  # Very high penalty
    else:
        mass_error_factor += blowout_error * 0.1  # Small penalty if close
    
    # overaspirate_vol: Also VERY volume-dependent
    # Different optimal fractions for different volumes
    if target_vol <= 0.03:  # Small volumes need higher fraction
        optimal_overasp = target_vol * 0.08  # 8% of volume
    elif target_vol <= 0.06:  # Medium-small volumes
        optimal_overasp = target_vol * 0.04  # 4% of volume  
    elif target_vol <= 0.15:  # Medium volumes
        optimal_overasp = target_vol * 0.02  # 2% of volume
    else:  # Large volumes need very small fraction
        optimal_overasp = target_vol * 0.01  # 1% of volume
    
    # STRICT penalty for overaspirate_vol too
    overasp_error = np.abs(overasp_vol - optimal_overasp)
    relative_error = overasp_error / target_vol if target_vol > 0 else 0  # Error as fraction of volume
    if relative_error > 0.03:  # More than 3% of volume off
        mass_error_factor += relative_error * 0.3  # High penalty
    else:
        mass_error_factor += relative_error * 0.05  # Small penalty if close
    
    # Other parameters with their optimal values
    mass_error_factor += np.abs(retract_speed - 8) * 0.002  # optimal ~8
    mass_error_factor += np.abs(post_asp_air_vol - 0.05) * 0.04  # optimal ~0.05
    
    # Add base random noise
    mass_error_factor += np.random.normal(0, 0.01)
    
    # SIMPLIFIED REALISTIC SIMULATION
    # 1. Systematic under-delivery bias (5%)
    underdelivery_bias = -0.05  # 5% systematic underdelivery
    
    # 2. Overaspirate compensation: linear effect with higher effectiveness for testing
    # If you overaspirate by X μL, you gain 0.8*X μL in delivered volume (increased from 0.5)
    overasp_absolute_compensation = overasp_vol * 0.8  # 80% effectiveness in absolute terms
    overasp_relative_compensation = overasp_absolute_compensation / target_vol if target_vol > 0 else 0
    
    # 3. Parameter effects (keep existing parameter modeling but reduce magnitude)
    parameter_effect = mass_error_factor * 0.1  # Reduced from 0.3 to 0.1
    
    # 4. Total error: bias + overaspirate compensation + parameter effects
    total_error = underdelivery_bias + overasp_relative_compensation + parameter_effect
    
    # 5. Soft saturation: use tanh to prevent extreme values but preserve differences
    # Allow wider range (-50% to +50%) but compress extreme values
    if total_error > 0:
        final_error = 0.5 * np.tanh(total_error / 0.3)  # Positive errors compressed more gently
    else:
        final_error = 0.5 * np.tanh(total_error / 0.3)  # Negative errors same treatment
    
    # Generate simulated volume with parameter-dependent error
    measured_volume = target_vol * (1 + final_error)
    measured_volume = max(measured_volume, 0)  # Can't have negative volume
    
    # Realistic time simulation
    baseline = 12.0  # Base pipetting time in seconds
    
    # Wait times ALWAYS add to the time (no "optimal" value)
    wait_time_penalty = asp_wait + dsp_wait
    
    # Speed penalties: Higher numbers (like 35) are SLOWER, lower numbers (like 10) are FASTER
    # Convert speed to time penalty: higher speed = more time
    aspirate_time_penalty = (asp_speed - 10) * 0.3  # 0.3s per speed unit above 10
    dispense_time_penalty = (dsp_speed - 10) * 0.3  # 0.3s per speed unit above 10
    
    # Calculate total time with some randomness
    total_time = baseline + wait_time_penalty + aspirate_time_penalty + dispense_time_penalty
    total_time += np.random.normal(0, 0.8)  # Add time variability
    elapsed = max(total_time, 2.0)  # Minimum 2 seconds
    
    return measured_volume, elapsed


def measure(state: Dict[str, Any], volume_mL: float, params: Dict[str, Any], replicates: int) -> List[Dict[str, Any]]:
    """Simulate pipetting measurements."""
    debug_sim = os.environ.get("CAL_SIM_DEBUG", "0") == "1"
    results: List[Dict[str, Any]] = []
    
    for r in range(replicates):
        start_ts = datetime.now().isoformat()
        measured_volume, elapsed = _simulate_once(volume_mL, params)
        end_ts = datetime.now().isoformat()
        
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


def wrapup(state: Dict[str, Any]) -> None:
    """Clean up simulation protocol."""
    return  # No cleanup needed for simulation
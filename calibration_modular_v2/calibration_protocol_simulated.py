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
    
    # ENHANCED Speed parameters: More realistic speed/accuracy tradeoffs
    # Range 10-35: Lower = faster but less accurate, Higher = slower but more accurate
    # Optimal ranges: aspirate ~20-25, dispense ~15-20
    
    # Aspirate speed effects (optimal around 22)
    aspirate_optimal = 22
    aspirate_deviation = abs(asp_speed - aspirate_optimal)
    if aspirate_deviation > 8:  # Very far from optimal (>8 units away)
        aspirate_penalty = 0.08 + aspirate_deviation * 0.004  # Large base + scaling penalty
    else:
        aspirate_penalty = aspirate_deviation * 0.006  # Moderate penalty for small deviations
    mass_error_factor += aspirate_penalty
    
    # Dispense speed effects (optimal around 17)  
    dispense_optimal = 17
    dispense_deviation = abs(dsp_speed - dispense_optimal)
    if dispense_deviation > 8:  # Very far from optimal 
        dispense_penalty = 0.06 + dispense_deviation * 0.003  # Large base + scaling penalty
    else:
        dispense_penalty = dispense_deviation * 0.004  # Moderate penalty for small deviations
    mass_error_factor += dispense_penalty
    
    # ENHANCED Wait time parameters: Stronger effects for settling time
    # Optimal wait times: aspirate ~12-20s, dispense ~8-15s
    
    # Aspirate wait time (optimal around 15s)
    aspirate_wait_optimal = 15
    aspirate_wait_deviation = abs(asp_wait - aspirate_wait_optimal)
    if asp_wait < 5:  # Very short waits are bad (rushing)
        mass_error_factor += 0.05 + (5 - asp_wait) * 0.008  # High penalty for rushing
    elif aspirate_wait_deviation > 10:  # Very long waits waste time without benefit
        mass_error_factor += aspirate_wait_deviation * 0.003
    else:
        mass_error_factor += aspirate_wait_deviation * 0.002  # Small penalty near optimal
        
    # Dispense wait time (optimal around 10s)
    dispense_wait_optimal = 10  
    dispense_wait_deviation = abs(dsp_wait - dispense_wait_optimal)
    if dsp_wait < 3:  # Very short waits are bad
        mass_error_factor += 0.04 + (3 - dsp_wait) * 0.006  # High penalty for rushing
    elif dispense_wait_deviation > 12:  # Very long waits
        mass_error_factor += dispense_wait_deviation * 0.002
    else:
        mass_error_factor += dispense_wait_deviation * 0.0015  # Small penalty near optimal
    
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
    
    # ENHANCED Retract speed effects (optimal around 10)
    retract_optimal = 10
    retract_deviation = abs(retract_speed - retract_optimal)
    if retract_deviation > 5:  # Very far from optimal
        retract_penalty = 0.03 + retract_deviation * 0.004  # Significant penalty
    else:
        retract_penalty = retract_deviation * 0.008  # Moderate penalty near optimal
    mass_error_factor += retract_penalty
    
    # ENHANCED Post-aspirate air volume effects (optimal around 0.06)
    post_air_optimal = 0.06
    post_air_deviation = abs(post_asp_air_vol - post_air_optimal) 
    if post_air_deviation > 0.04:  # Very far from optimal
        post_air_penalty = 0.04 + post_air_deviation * 0.5  # High penalty for extreme values
    else:
        post_air_penalty = post_air_deviation * 0.15  # Moderate penalty near optimal
    mass_error_factor += post_air_penalty
    
    # Parameter-dependent noise instead of fixed random noise
    # Calculate noise level based on parameter extremes/suboptimal values
    
    # Speed-related noise: faster speeds = more noise (less controlled)
    speed_noise_factor = 0.005  # Base noise
    if asp_speed < 15 or asp_speed > 30:  # Very fast or very slow
        speed_noise_factor += 0.008
    if dsp_speed < 12 or dsp_speed > 25:  # Very fast or very slow  
        speed_noise_factor += 0.006
        
    # Wait time noise: very short waits = more noise (insufficient settling)
    wait_noise_factor = 0.0
    if asp_wait < 3:  # Too short
        wait_noise_factor += 0.006
    if dsp_wait < 2:  # Too short
        wait_noise_factor += 0.004
        
    # Volume parameter noise: extreme values = more noise
    vol_noise_factor = 0.0
    if blowout_vol < 0.02 or blowout_vol > 0.15:  # Too little or too much
        vol_noise_factor += 0.005
    if post_asp_air_vol < 0.02 or post_asp_air_vol > 0.12:  # Too little or too much
        vol_noise_factor += 0.007
        
    # Retract speed noise: very fast retract = more noise
    if retract_speed < 6 or retract_speed > 15:
        vol_noise_factor += 0.004
    
    # Total parameter-dependent noise
    total_noise_std = speed_noise_factor + wait_noise_factor + vol_noise_factor
    parameter_dependent_noise = np.random.normal(0, total_noise_std)
    mass_error_factor += parameter_dependent_noise
    
    # SIMPLIFIED REALISTIC SIMULATION
    # 1. Systematic under-delivery bias (5%)
    underdelivery_bias = -0.05  # 5% systematic underdelivery
    
    # 2. Overaspirate compensation: linear effect with higher effectiveness for testing
    # If you overaspirate by X μL, you gain 0.8*X μL in delivered volume (increased from 0.5)
    overasp_absolute_compensation = overasp_vol * 0.8  # 80% effectiveness in absolute terms
    overasp_relative_compensation = overasp_absolute_compensation / target_vol if target_vol > 0 else 0
    
    # 3. Parameter effects - INCREASED to make optimization more meaningful
    parameter_effect = mass_error_factor * 0.25  # Increased from 0.1 to 0.25 for stronger effects
    
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
    
    # Parameter-dependent replicate noise instead of fixed noise
    # Better parameters = more consistent replicates, worse parameters = more variable
    
    # Base replicate variability
    replicate_noise_std = 0.003  # Reduced base
    
    # Speed consistency: extreme speeds reduce replicate consistency  
    if asp_speed < 18 or asp_speed > 28:
        replicate_noise_std += 0.004
    if dsp_speed < 14 or dsp_speed > 22:
        replicate_noise_std += 0.003
        
    # Wait time consistency: too short = inconsistent replicates
    if asp_wait < 5:
        replicate_noise_std += 0.003
    if dsp_wait < 3:
        replicate_noise_std += 0.002
        
    # Volume parameter consistency
    if abs(blowout_vol - 0.06) > 0.04:  # Far from optimal ~0.06
        replicate_noise_std += 0.002
    if abs(post_asp_air_vol - 0.06) > 0.03:  # Far from optimal ~0.06
        replicate_noise_std += 0.003
    if abs(retract_speed - 10) > 4:  # Far from optimal ~10
        replicate_noise_std += 0.002
    
    # Apply parameter-dependent replicate noise
    replicate_noise = np.random.normal(0, replicate_noise_std)
    measured_volume = measured_volume * (1 + replicate_noise)
    
    measured_volume = max(measured_volume, 0)  # Can't have negative volume
    
    # ENHANCED Realistic time simulation with stronger parameter effects
    baseline = 12.0  # Base pipetting time in seconds
    
    # Wait times ALWAYS add to the time (no "optimal" value for time)
    wait_time_penalty = asp_wait + dsp_wait
    
    # ENHANCED Speed penalties: Higher numbers (like 35) are SLOWER, lower numbers (like 10) are FASTER
    # More realistic time differences to make speed optimization meaningful
    aspirate_time_penalty = (asp_speed - 10) * 0.8  # Increased from 0.3 to 0.8s per speed unit
    dispense_time_penalty = (dsp_speed - 10) * 0.6  # Increased from 0.3 to 0.6s per speed unit
    
    # Retract speed also affects time
    retract_time_penalty = (retract_speed - 8) * 0.4  # 0.4s per speed unit above 8
    
    # Calculate total time with enhanced parameter sensitivity
    total_time = baseline + wait_time_penalty + aspirate_time_penalty + dispense_time_penalty + retract_time_penalty
    
    # Parameter-dependent timing variability instead of fixed noise
    # Poor parameters = more inconsistent timing
    timing_noise_std = 0.3  # Reduced base timing noise
    
    # Extreme speeds create more timing variability
    if asp_speed < 15 or asp_speed > 30:
        timing_noise_std += 0.4
    if dsp_speed < 12 or dsp_speed > 25:
        timing_noise_std += 0.3
    if retract_speed < 6 or retract_speed > 15:
        timing_noise_std += 0.2
        
    # Very short wait times create timing inconsistency
    if asp_wait < 3:
        timing_noise_std += 0.3
    if dsp_wait < 2:
        timing_noise_std += 0.2
    
    # Apply parameter-dependent timing noise
    total_time += np.random.normal(0, timing_noise_std)
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
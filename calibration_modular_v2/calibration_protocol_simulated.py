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
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from calibration_protocol_base import CalibrationProtocolBase


class SimulatedCalibrationProtocol(CalibrationProtocolBase):
    """Simulated calibration protocol implementing the abstract interface."""
    
    def initialize(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
                    seeded = True
                except Exception:
                    pass
        
        # Get liquid from experiment config - FAIL if missing
        if not cfg or 'experiment' not in cfg:
            raise ValueError("Missing required 'experiment' configuration section")
        
        if 'liquid' not in cfg['experiment']:
            raise ValueError("Missing required 'liquid' in experiment configuration")
        liquid = cfg['experiment']['liquid']
        
        print(f"✅ Simulation initialized (liquid: {liquid}, seeded: {seeded})")
        
        state = {
            'initialized_at': datetime.now(),
            'liquid': liquid,
            'seeded': seeded,
            'measurement_count': 0
        }
        
        return state

    def measure(self, state: Dict[str, Any], volume_mL: float, params: Dict[str, Any], replicates: int = 1) -> List[Dict[str, Any]]:
        """Simulate pipetting measurements."""
        results = []
        
        for rep in range(replicates):
            rep_start = time.perf_counter()
            start_time = datetime.now()
            
            # Simulate measurement
            measured_volume, elapsed_s = self._simulate_pipetting(volume_mL, params, state)
            
            rep_end = time.perf_counter()
            end_time = datetime.now()
            actual_elapsed = rep_end - rep_start
            
            # Increment measurement counter
            state['measurement_count'] += 1
            
            result = {
                'replicate': rep + 1,
                'volume': measured_volume,
                'elapsed_s': elapsed_s,  # Simulated time
                'actual_elapsed_s': actual_elapsed,  # Real time
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'target_volume_mL': volume_mL,
                **params  # Echo back all parameters
            }
            
            results.append(result)
            print(f"  Rep {rep+1}: {measured_volume*1000:.1f}uL (target: {volume_mL*1000:.1f}uL) in {elapsed_s:.2f}s")
        
        return results

    def wrapup(self, state: Dict[str, Any]) -> None:
        """Clean up simulation resources."""
        print(f"✅ Simulation cleanup completed. Total measurements: {state.get('measurement_count', 0)}")

    def _simulate_pipetting(self, volume_mL: float, params: Dict[str, Any], state: Dict[str, Any]) -> Tuple[float, float]:
        """Simulate a single pipetting operation."""
        # Get liquid properties
        liquid = state.get('liquid', 'water')
        
        # Base accuracy depends on volume and liquid
        if liquid == 'water':
            base_accuracy = 0.98  # 98% accuracy for water
            base_precision_cv = 0.02  # 2% CV
        elif 'glycerol' in liquid.lower():
            base_accuracy = 0.95  # Lower accuracy for viscous liquids
            base_precision_cv = 0.04  # Higher variability
        else:
            base_accuracy = 0.97  # Default
            base_precision_cv = 0.03
        
        # Volume-dependent effects
        volume_ul = volume_mL * 1000
        
        # Small volume penalty
        if volume_ul < 10:
            base_accuracy *= 0.9
            base_precision_cv *= 1.5
        elif volume_ul < 50:
            base_accuracy *= 0.95
            base_precision_cv *= 1.2
        
        # Parameter effects on accuracy
        overaspirate_vol = params.get('overaspirate_vol', 0.005)
        
        # Optimal overaspirate is around 1-3% of volume
        optimal_overaspirate = volume_mL * 0.02
        overaspirate_error = abs(overaspirate_vol - optimal_overaspirate) / optimal_overaspirate
        accuracy_factor = 1.0 - (overaspirate_error * 0.1)  # 10% penalty for poor overaspirate
        
        # Speed effects (higher speed values = slower operation in North Robot)
        aspirate_speed = params.get('aspirate_speed', 20)
        dispense_speed = params.get('dispense_speed', 15)
        
        # Optimal speeds are around 15-25 (slower is better for accuracy)
        speed_penalty = 0
        if aspirate_speed < 10:  # Too fast
            speed_penalty += 0.05
        elif aspirate_speed > 30:  # Too slow (no benefit)
            speed_penalty += 0.02
            
        if dispense_speed < 8:  # Too fast
            speed_penalty += 0.05
        elif dispense_speed > 25:  # Too slow
            speed_penalty += 0.02
        
        # Wait time effects
        aspirate_wait = params.get('aspirate_wait_time', 2.0)
        dispense_wait = params.get('dispense_wait_time', 1.0)
        
        # Optimal wait times: aspirate 1-3s, dispense 0.5-2s
        wait_penalty = 0
        if aspirate_wait < 1.0:
            wait_penalty += 0.03
        if dispense_wait < 0.5:
            wait_penalty += 0.02
        
        # Final accuracy
        final_accuracy = base_accuracy * accuracy_factor * (1.0 - speed_penalty - wait_penalty)
        final_accuracy = max(0.7, min(1.05, final_accuracy))  # Clamp to reasonable range
        
        # Calculate measured volume
        accuracy_error = np.random.normal(0, base_precision_cv)
        measured_volume = volume_mL * (final_accuracy + accuracy_error)
        measured_volume = max(0, measured_volume)  # No negative volumes
        
        # Calculate timing
        base_time = 8.0  # Base operation time
        
        # Speed effects on timing (higher speed = slower operation)
        speed_time = (aspirate_speed / 20.0) * 2.0 + (dispense_speed / 15.0) * 2.0
        wait_time = aspirate_wait + dispense_wait
        
        # Volume effects
        volume_time = volume_ul / 100.0  # Larger volumes take longer
        
        # Parameter-dependent timing noise
        timing_noise_std = 0.5
        
        # Apply parameter-dependent timing noise
        total_time = base_time + speed_time + wait_time + volume_time
        total_time += np.random.normal(0, timing_noise_std)
        elapsed = max(total_time, 2.0)  # Minimum 2 seconds
        
        return measured_volume, elapsed


# Backward compatibility: maintain function-based interface
_protocol_instance = SimulatedCalibrationProtocol()

def initialize(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Backward compatibility function."""
    return _protocol_instance.initialize(cfg)

def measure(state: Dict[str, Any], volume_mL: float, params: Dict[str, Any], replicates: int = 1) -> List[Dict[str, Any]]:
    """Backward compatibility function."""
    return _protocol_instance.measure(state, volume_mL, params, replicates)

def wrapup(state: Dict[str, Any]) -> None:
    """Backward compatibility function."""
    return _protocol_instance.wrapup(state)
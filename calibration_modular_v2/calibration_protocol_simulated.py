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
    
    def _calculate_generic_parameter_penalty(self, param_name: str, param_value: float, all_params: Dict[str, float]) -> float:
        """Calculate generic penalty for any hardware parameter.
        
        Applies subtle effects based on parameter value relative to typical ranges.
        This works with any parameter names - hardware agnostic.
        """
        if not isinstance(param_value, (int, float)):
            return 0.0
            
        # Skip overaspirate_vol - it's handled separately in main simulation logic
        if param_name == 'overaspirate_vol':
            return 0.0
            
        # For parameters that look like speeds (higher usually = slower/more precise)
        if any(keyword in param_name.lower() for keyword in ['speed', 'rate', 'velocity']):
            # Assume reasonable speed range is 10-30, with optimal around 20
            optimal = 20.0
            if param_value < 10:
                return 0.02  # Too fast - small precision penalty
            elif param_value > 30:
                return 0.01  # Too slow - tiny penalty
            else:
                return 0.0  # In good range
                
        # For parameters that look like wait times (more usually = better but diminishing returns)
        elif any(keyword in param_name.lower() for keyword in ['wait', 'time', 'delay', 'pause']):
            # Assume reasonable wait range is 0.5-5s
            if param_value < 0.5:
                return 0.015  # Too fast - small penalty
            elif param_value > 5.0:
                return 0.005  # Too slow - tiny penalty  
            else:
                return 0.0  # In good range
                
        # For volume parameters (usually small effects, excluding overaspirate_vol)
        elif any(keyword in param_name.lower() for keyword in ['vol', 'volume', 'air']):
            # Small effect for extreme values
            if param_value < 0 or param_value > 0.01:  # Outside 0-10uL range
                return 0.005  # Very small penalty
            else:
                return 0.0
                
        # Generic parameters - minimal effect
        else:
            return 0.001  # Tiny generic penalty for any parameter
    
    def _calculate_generic_timing_effect(self, param_name: str, param_value: float) -> float:
        """Calculate generic timing effect for any hardware parameter."""
        if not isinstance(param_value, (int, float)):
            return 0.0
            
        # Speed parameters affect timing more
        if any(keyword in param_name.lower() for keyword in ['speed', 'rate', 'velocity']):
            # Higher speed values = slower operation (North Robot style)
            return (param_value / 20.0) * 0.5  # Small timing effect
            
        # Wait parameters add directly to timing
        elif any(keyword in param_name.lower() for keyword in ['wait', 'time', 'delay', 'pause']):
            return param_value * 0.3  # Partial contribution
            
        # Volume parameters have minimal timing effect
        elif any(keyword in param_name.lower() for keyword in ['vol', 'volume', 'air']):
            return param_value * 10.0  # Convert mL to minimal timing effect
            
        # Generic parameters
        else:
            return param_value * 0.01  # Minimal timing contribution
    
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

    def get_parameter_constraints(self, target_volume_ml: float) -> List[str]:
        """Get hardware-specific parameter constraints for North Robot simulation."""
        constraints = []
        
        # North Robot tip volume constraint (same logic as hardware)
        # Use 0.2 mL tips for volumes <= 150 µL, otherwise 1.0 mL tips
        if target_volume_ml <= 0.15:  # 150 µL or less
            tip_volume_ml = 0.2
        else:
            tip_volume_ml = 1.0
            
        # Calculate available volume for air and overaspiration
        available_volume_ml = tip_volume_ml - target_volume_ml
        
        # Add tip volume constraint if relevant parameters exist
        constraint = f"overaspirate_vol <= {available_volume_ml:.6f}"
        constraints.append(constraint)
        
        print(f"📏 Simulated constraint: {constraint} (tip: {tip_volume_ml*1000:.0f}µL, target: {target_volume_ml*1000:.0f}µL)")
        
        return constraints

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
        
        # Overaspirate volume directly affects measured volume
        # In real pipetting: measured_volume ≈ target_volume + (overaspirate_vol * efficiency)
        # Efficiency is typically 0.7-0.9 (some overaspirate is lost due to surface tension, etc.)
        overaspirate_efficiency = 0.8  # 80% of overaspirate volume is retained
        
        # Volume-dependent efficiency (smaller volumes have lower efficiency)
        if volume_ul < 10:
            overaspirate_efficiency *= 0.6  # Lower efficiency for small volumes
        elif volume_ul < 25:
            overaspirate_efficiency *= 0.7
        
        # Calculate base measured volume (before overaspirate)
        base_measured_volume = volume_mL * base_accuracy
        
        # Add overaspirate contribution
        overaspirate_contribution = overaspirate_vol * overaspirate_efficiency
        
        # Total measured volume (target + overaspirate contribution)
        measured_volume_base = base_measured_volume + overaspirate_contribution
        
        # Speed effects (higher speed values = slower operation in North Robot)
        # Generic hardware parameter effects (works with any parameter names)
        hardware_penalty = 0.0
        hardware_timing_effect = 0.0
        
        for param_name, param_value in params.items():
            if param_name != 'overaspirate_vol':  # Skip overaspirate_vol (handled separately)
                penalty = self._calculate_generic_parameter_penalty(param_name, param_value, params)
                hardware_penalty += penalty
                
                timing_effect = self._calculate_generic_timing_effect(param_name, param_value)
                hardware_timing_effect += timing_effect
        
        # Apply parameter penalties to precision (not accuracy)
        total_penalty = hardware_penalty  # Use generic hardware penalty
        final_precision_cv = base_precision_cv * (1.0 + total_penalty)
        
        # Cap precision CV to prevent unrealistic variability
        final_precision_cv = min(final_precision_cv, 0.1)  # Max 10% CV
        
        # Add random variation to the measured volume
        precision_error = np.random.normal(0, final_precision_cv)
        # Cap precision error to prevent extreme swings
        precision_error = np.clip(precision_error, -0.15, 0.15)  # Max ±15% variation
        
        measured_volume = measured_volume_base * (1.0 + precision_error)
        measured_volume = max(0, measured_volume)  # No negative volumes
        
        # Calculate timing
        base_time = 8.0  # Base operation time
        
        # Generic hardware timing effects (replaces hardcoded speed/wait calculations)
        volume_time = volume_ul / 100.0  # Larger volumes take longer
        
        # Parameter-dependent timing noise
        timing_noise_std = 0.5
        
        # Apply generic hardware timing and noise
        total_time = base_time + hardware_timing_effect + volume_time
        total_time += np.random.normal(0, timing_noise_std)
        elapsed = max(total_time, 2.0)  # Minimum 2 seconds
        
        return measured_volume, elapsed


# Export the protocol instance for clean importing
protocol_instance = SimulatedCalibrationProtocol()
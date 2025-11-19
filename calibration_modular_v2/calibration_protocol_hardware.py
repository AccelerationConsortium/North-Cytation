"""Hardware calibration protocol for North Robot system.

Unified minimal interface:
    initialize(cfg) -> state (dict)
    measure(state, volume_mL, params, replicates) -> list[dict]
    wrapup(state) -> None

Per-replicate result keys:
    replicate, volume (mL), elapsed_s
Optional: start_time, end_time, echoed params.
"""
import time
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from calibration_protocol_base import CalibrationProtocolBase

# Add parent directory to path for North Robot imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from master_usdl_coordinator import Lash_E


class HardwareCalibrationProtocol(CalibrationProtocolBase):
    """Hardware calibration protocol implementing the abstract interface."""
    
    def initialize(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize hardware protocol with North Robot's internal simulation."""
        
        # Extract hardware configuration - FAIL if missing required values
        if not cfg or 'hardware' not in cfg:
            raise ValueError("Missing required 'hardware' configuration section")
        
        if 'device_serial' not in cfg['hardware']:
            raise ValueError("Missing required 'device_serial' in hardware configuration")
        device_serial = cfg['hardware']['device_serial']
        
        # Get liquid from experiment config - FAIL if missing
        if not cfg or 'experiment' not in cfg:
            raise ValueError("Missing required 'experiment' configuration section")
        
        if 'liquid' not in cfg['experiment']:
            raise ValueError("Missing required 'liquid' in experiment configuration")
        liquid = cfg['experiment']['liquid']
        
        print(f"Initializing North Robot hardware protocol for {liquid}")
        print(f"Device serial: {device_serial}")

        # Use hardware simulation mode (different from our simulation protocol)
        simulate = True  # This enables North Robot's internal simulation
        
        # Initialize hardware
        try:
            # Initialize vial file path
            vial_file = "status/experiment_vials.csv"
            
            # Initialize Lash_E coordinator
            lash_e = Lash_E(vial_file, simulate=simulate)
            
            # Validate hardware files
            lash_e.nr_robot.check_input_file()
            lash_e.nr_track.check_input_file()
            
            # Move to working position
            lash_e.nr_robot.move_vial_to_location("target_vial", "clamp", 0)
            
            # Get fresh wellplate for measurements
            lash_e.nr_track.get_new_wellplate()
            
            print("✅ Hardware initialized successfully")
            
            return {
                'initialized_at': datetime.now(),
                'hardware_type': 'north_robot_c9',
                'device_serial': device_serial,
                'liquid': liquid,
                'lash_e': lash_e,
                'measurement_count': 0
            }
            
        except Exception as e:
            raise RuntimeError(f"Hardware initialization failed: {e}")

    def measure(self, state: Dict[str, Any], volume_mL: float, params: Dict[str, Any], replicates: int = 1) -> List[Dict[str, Any]]:
        """Perform hardware measurement with given parameters."""
        
        results = []
        lash_e = state['lash_e']
        
        for rep in range(replicates):
            rep_start = time.perf_counter()
            start_time = datetime.now()
            
            # Convert volume to microliters
            volume_uL = volume_mL * 1000
            
            # Extract pipetting parameters
            pipet_params = {
                'aspirate_speed': params.get('aspirate_speed', 20),
                'dispense_speed': params.get('dispense_speed', 15),
                'aspirate_wait_time': params.get('aspirate_wait_time', 2.0),
                'dispense_wait_time': params.get('dispense_wait_time', 1.0),
                'retract_speed': params.get('retract_speed', 5.0),
                'blowout_vol': params.get('blowout_vol', 0.02),
                'post_asp_air_vol': params.get('post_asp_air_vol', 0.01),
                'overaspirate_vol': params.get('overaspirate_vol', 0.004)
            }
            
            print(f"  Rep {rep+1}/{replicates}: {volume_uL:.1f}uL with params {pipet_params}")
            
            # Perform pipetting operation
            # Note: This will crash if pipet_and_measure doesn't exist - INTENTIONAL
            measured_volume = lash_e.pipet_and_measure(
                volume_target_uL=volume_uL,
                **pipet_params
            )
            
            # Convert back to mL
            measured_volume_mL = measured_volume / 1000
            
            rep_end = time.perf_counter()
            end_time = datetime.now()
            elapsed_s = rep_end - rep_start
            
            # Increment measurement counter
            state['measurement_count'] += 1
            
            result = {
                'replicate': rep + 1,
                'volume': measured_volume_mL,
                'elapsed_s': elapsed_s,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'target_volume_mL': volume_mL,
                'measured_volume_uL': measured_volume,
                'target_volume_uL': volume_uL,
                **pipet_params  # Echo back parameters
            }
            
            results.append(result)
            print(f"    Measured: {measured_volume:.1f}uL (target: {volume_uL:.1f}uL) in {elapsed_s:.2f}s")
        
        return results

    def wrapup(self, state: Dict[str, Any]) -> None:
        """Clean up hardware resources."""
        
        lash_e = state.get('lash_e')
        if lash_e:
            # Move robot to safe position - will crash if methods don't exist
            lash_e.nr_robot.move_to_safe_position()
            
            # Return wellplate to track - will crash if methods don't exist  
            lash_e.nr_track.return_wellplate()
            
            print(f"✅ Hardware cleanup completed. Total measurements: {state.get('measurement_count', 0)}")


# Backward compatibility: maintain function-based interface
_protocol_instance = HardwareCalibrationProtocol()

def initialize(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Backward compatibility function."""
    return _protocol_instance.initialize(cfg)

def measure(state: Dict[str, Any], volume_mL: float, params: Dict[str, Any], replicates: int = 1) -> List[Dict[str, Any]]:
    """Backward compatibility function."""
    return _protocol_instance.measure(state, volume_mL, params, replicates)

def wrapup(state: Dict[str, Any]) -> None:
    """Backward compatibility function."""
    return _protocol_instance.wrapup(state)
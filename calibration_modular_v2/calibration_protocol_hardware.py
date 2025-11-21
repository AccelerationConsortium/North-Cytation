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
        
        # Get liquid from experiment config - FAIL if missing
        if not cfg or 'experiment' not in cfg:
            raise ValueError("Missing required 'experiment' configuration section")
        
        if 'liquid' not in cfg['experiment']:
            raise ValueError("Missing required 'liquid' in experiment configuration")
        liquid = cfg['experiment']['liquid']
        
        print(f"Initializing North Robot hardware protocol for {liquid}")

        # Use hardware simulation mode (different from our simulation protocol)
        simulate = False  # This enables North Robot's internal simulation
        
        # Vial management mode - swap roles when measurement vial gets too full
        SWAP = True  # If True, enables vial swapping when needed
        
        # Initialize hardware
        try:
            # Initialize vial file path
            vial_file = "status/calibration_vials_overnight.csv"
            
            # Initialize Lash_E coordinator
            lash_e = Lash_E(vial_file, simulate=simulate)
            
            # Validate hardware files
            lash_e.nr_robot.check_input_file()
            lash_e.nr_track.check_input_file()
            
            # Simple vial management: Set up source and measurement vials
            source_vial = "liquid_source_0"
            measurement_vial = "measurement_vial_0" 
            
            # Move measurement vial to clamp position for pipetting operations
            lash_e.nr_robot.move_vial_to_location(measurement_vial, "clamp", 0)
            
            print("✅ Hardware initialized successfully")
            
            return {
                'initialized_at': datetime.now(),
                'liquid': liquid,
                'lash_e': lash_e,
                'source_vial': source_vial,
                'measurement_vial': measurement_vial,
                'swap_enabled': SWAP,
                'measurement_count': 0
            }
            
        except Exception as e:
            raise RuntimeError(f"Hardware initialization failed: {e}") from e

    def _check_and_swap_vials(self, state: Dict[str, Any], swap_enabled: bool = True) -> None:
        """Check measurement vial volume and swap if needed."""
        if not swap_enabled:
            return
            
        try:
            lash_e = state['lash_e']
            measurement_vial = state['measurement_vial']
            source_vial = state['source_vial']
            
            # Check current volume in measurement vial
            measurement_volume = lash_e.nr_robot.get_vial_info(measurement_vial, 'vial_volume')
            max_measurement_volume = 7.0  # mL - threshold for swapping
            
            if measurement_volume is not None and measurement_volume >= max_measurement_volume:
                print(f"🔄 SWAP: Measurement vial ({measurement_vial}) volume {measurement_volume:.2f}mL >= {max_measurement_volume}mL")
                
                # First: Return the old measurement vial (at clamp) to its home position
                lash_e.nr_robot.remove_pipet()
                lash_e.nr_robot.return_vial_home(measurement_vial)
                
                # Swap the vial roles in stateok 
                state['source_vial'] = measurement_vial  # Old measurement becomes new source
                state['measurement_vial'] = source_vial   # Old source becomes new measurement
                
                # Finally: Move the new measurement vial to clamp position
                lash_e.nr_robot.move_vial_to_location(state['measurement_vial'], "clamp", 0)
                
                print(f"✅ SWAP complete: source={state['source_vial']}, measurement={state['measurement_vial']}")
                
        except Exception as e:
            print(f"⚠️ Vial swap check failed: {e}")

    def measure(self, state: Dict[str, Any], volume_mL: float, params: Dict[str, Any], replicates: int = 1) -> List[Dict[str, Any]]:
        """Perform hardware measurement with given parameters."""
        
        # Check if we need to swap vials before pipetting
        self._check_and_swap_vials(state, swap_enabled=state.get('swap_enabled', False))
        
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
            
            # Perform pipetting operation: aspirate from source, dispense into measurement vial
            source_vial = state['source_vial']
            measurement_vial = state['measurement_vial']
            
            # Check if we're in simulation mode
            simulate = True  # This should match the simulate flag from initialize
            
            if not simulate:
                # Real hardware measurements
                # Aspirate from source vial
                lash_e.nr_robot.aspirate_from_vial(source_vial, volume_mL, parameters=pipet_params)
                
                # Dispense into measurement vial and measure weight
                measurement = lash_e.nr_robot.dispense_into_vial(
                    measurement_vial, volume_mL, parameters=pipet_params, measure_weight=True
                )
                measured_volume_mL = measurement  # Real measurement
                
            else:
                # Simple simulation: target volume - 20% + overaspirate + noise
                import random
                
                # Still call the robot methods for vial tracking, but ignore measurement result
                lash_e.nr_robot.aspirate_from_vial(source_vial, volume_mL, parameters=pipet_params)
                lash_e.nr_robot.dispense_into_vial(
                    measurement_vial, volume_mL, parameters=pipet_params, measure_weight=True
                )
                
                # Basic simulation logic
                base_efficiency = 0.8  # Start at 80% efficiency (target - 20%)
                overaspirate_effect = pipet_params.get('overaspirate_vol', 0.004) * 1000  # Convert to uL
                noise = random.uniform(-0.02, 0.02) * volume_mL  # ±2% noise
                
                # Simple formula: base efficiency + overaspirate helps + noise
                simulated_volume_mL = (volume_mL * base_efficiency) + (overaspirate_effect / 1000) + noise
                simulated_volume_mL = max(simulated_volume_mL, 0)  # Can't be negative
                
                measured_volume_mL = simulated_volume_mL
            
            # Convert to microliters for display
            measured_volume_uL = measured_volume_mL * 1000
            
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
                'measured_volume_uL': measured_volume_uL,
                'target_volume_uL': volume_uL,
                **pipet_params  # Echo back parameters
            }
            
            results.append(result)
            print(f"    Measured: {measured_volume_uL:.1f}uL (target: {volume_uL:.1f}uL) in {elapsed_s:.2f}s")
        
        return results

    def wrapup(self, state: Dict[str, Any]) -> None:
        """Clean up hardware resources."""
        
        lash_e = state.get('lash_e')
        if lash_e:
            try:
                # Move robot to safe position
                lash_e.nr_robot.move_home()
                
                print(f"✅ Hardware cleanup completed. Total measurements: {state.get('measurement_count', 0)}")
                
            except Exception as e:
                print(f"⚠️ Hardware cleanup warning: {e}")

    def get_parameter_constraints(self, target_volume_ml: float) -> List[str]:
        """Get hardware-specific parameter constraints for North Robot system."""
        constraints = []
        
        # North Robot tip volume constraint
        # Use 0.2 mL tips for volumes <= 150 µL, otherwise 1.0 mL tips
        if target_volume_ml <= 0.15:  # 150 µL or less
            tip_volume_ml = 0.2
        else:
            tip_volume_ml = 1.0
            
        # Calculate available volume for air and overaspiration
        available_volume_ml = tip_volume_ml - target_volume_ml
        
        # Add tip volume constraint if relevant parameters exist
        constraint = f"post_asp_air_vol + overaspirate_vol <= {available_volume_ml:.6f}"
        constraints.append(constraint)
        
        print(f"📏 North Robot constraint: {constraint} (tip: {tip_volume_ml*1000:.0f}µL, target: {target_volume_ml*1000:.0f}µL)")
        
        return constraints


# Export the protocol instance for clean importing
protocol_instance = HardwareCalibrationProtocol()
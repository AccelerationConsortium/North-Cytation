"""Template for creating custom calibration protocols.

Copy this file and replace the TODO sections with your hardware-specific code.
"""
from datetime import datetime
from typing import Dict, Any, List, Optional
from calibration_protocol_base import CalibrationProtocolBase


class TemplateCalibrationProtocol(CalibrationProtocolBase):
    """Template calibration protocol - replace with your hardware implementation."""
    
    def initialize(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize your hardware."""
        
        # Get liquid type from config
        liquid = cfg['experiment']['liquid']
        
        # TODO: Initialize your hardware here
        # my_robot = MyRobot()
        # my_robot.connect()
        # my_robot.home()
        
        return {
            'initialized_at': datetime.now(),
            'liquid': liquid,
            'measurement_count': 0
            # TODO: Add your hardware objects/state here
        }
    
    def measure(self, state: Dict[str, Any], volume_mL: float, params: Dict[str, Any], replicates: int = 1) -> List[Dict[str, Any]]:
        """Perform measurements with given parameters."""
        
        results = []
        
        for rep in range(replicates):
            # Extract parameters - use only what you need for your hardware
            overaspirate_vol = params.get('overaspirate_vol', 0.004) # You must use this parameter
            
            # TODO: Extract any hardware-specific parameters you need:
            # my_speed_param = params.get('my_speed_param', default_value)
            # my_timing_param = params.get('my_timing_param', default_value)
            
            # TODO: Replace this simulation with your hardware calls
            import random
            import time
            
            start_time = time.perf_counter()
            
            # Simulate pipetting - replace with real hardware
            measured_volume_mL = volume_mL + random.uniform(-0.01, 0.01) * volume_mL + overaspirate_vol
            
            # TODO: Your actual hardware measurement here:
            # my_robot.aspirate(volume_mL+overaspirate_vol, **other_params)
            # measured_volume_mL = my_robot.dispense_and_measure()
            # NOTE: Ensure overaspirate_vol is used to increase your pipetting volume!
            
            elapsed_s = time.perf_counter() - start_time
            
            # Track measurement count
            state['measurement_count'] += 1
            
            # Return required result format
            result = {
                'replicate': rep + 1,
                'volume': measured_volume_mL,  # Measured volume in mL
                'elapsed_s': elapsed_s,        # Time taken in seconds
                'target_volume_mL': volume_mL, # Target volume
                **params  # Echo back all parameters
            }
            
            results.append(result)
        
        return results
    
    def wrapup(self, state: Dict[str, Any]) -> None:
        """Clean up hardware resources."""
        
        # TODO: Clean up your hardware here
        # my_robot.home()
        # my_robot.disconnect()
        
        print(f"Cleanup completed. Total measurements: {state.get('measurement_count', 0)}")

    def get_parameter_constraints(self, target_volume_ml: float) -> List[str]:
        """Get hardware-specific parameter constraints for your system.
        
        This method should return a list of constraint strings that will be
        passed to the Bayesian optimizer. Constraints should be in the format
        that Ax understands, e.g., "param1 + param2 <= 100".
        
        Args:
            target_volume_ml: The target volume for this optimization run
            
        Returns:
            List of constraint strings for the optimizer
        """
        constraints = []
        
        # TODO: Add your hardware-specific constraints here
        # Example: Tip volume constraint
        # tip_volume_ml = 1.0  # Your tip volume
        # available_volume = tip_volume_ml - target_volume_ml
        # constraints.append(f"my_air_param + overaspirate_vol <= {available_volume}")
        
        # Example: Speed/timing constraints
        # constraints.append("my_speed_param1 * my_speed_param2 <= 1000")
        
        return constraints


# Export the protocol instance for the system to use
protocol_instance = TemplateCalibrationProtocol()
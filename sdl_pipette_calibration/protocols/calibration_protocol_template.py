"""Template for creating custom calibration protocols.

Copy this file and replace the TODO sections with your hardware-specific code.

State Dictionary
----------------
The state dict is a plain Python dict returned by initialize() and passed to
every subsequent measure() and wrapup() call. It is your protocol's persistent
memory for the duration of a calibration run — put anything in it that your
hardware needs to keep track of between calls.

Suggested keys (add or remove freely):
    initialized_at    datetime   When the protocol was initialized
    liquid            str        Liquid type label (from config)
    measurement_count int        Running count of measurements taken (increment in measure())

    # Hardware objects — examples:
    robot             object     Your robot/instrument handle
    balance           object     Your measurement device handle

    # Workflow state — examples from calibration_protocol_northrobot.py:
    source_vial       object     Vial to aspirate from
    measurement_vial  object     Vial used for mass measurement
    swap_enabled      bool       Whether vial swapping is active

There are no enforced required keys beyond what your own measure() and
wrapup() implementations expect. The framework passes state through opaquely
and never inspects its contents.
"""
from datetime import datetime
from typing import Dict, Any, List, Optional
from calibration_protocol_base import CalibrationProtocolBase


class TemplateCalibrationProtocol(CalibrationProtocolBase):
    """Template calibration protocol - replace with your hardware implementation."""
    
    def initialize(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize your hardware.
        
        This method is called once at the start of calibration to set up hardware
        and create a state dictionary that will be passed to all subsequent calls.
        
        Args:
            cfg: Configuration dictionary containing experiment settings.
                 Access your settings via cfg['experiment'] and cfg['hardware_parameters'].
        
        Returns:
            state: Dictionary containing hardware objects and settings for later use.
                   At minimum, should contain: initialized_at, liquid, measurement_count.
                   Add your hardware objects here (robot, balance, etc.).
        
        Raises:
            RuntimeError: If hardware initialization fails (connection, calibration, etc.)
            ValueError: If configuration is invalid or missing required settings.
        
        Note:
            This state dict will be passed to measure() and wrapup(), so store anything
            you need throughout the calibration run.
        """
        
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
        """Perform pipetting measurements with given parameters.
        
        Called by the optimizer to test parameter combinations. Must execute the pipetting
        operation and return measured volumes for each replicate.
        
        Args:
            state: Protocol state dictionary from initialize().
            volume_mL: Target volume to pipette (in milliliters).
            params: Dictionary of hardware parameters from the optimizer.
                    Includes 'overaspirate_vol' (required) and any other parameters
                    defined in hardware_parameters section of the config.
            replicates: Number of times to repeat the measurement (default: 1).
        
        Returns:
            List of measurement dictionaries, one per replicate.
            Each dict MUST contain:
                - replicate: int (1-indexed replicate number)
                - volume: float (measured volume in mL)
                - elapsed_s: float (time taken in seconds)
                - target_volume_mL: float (the target volume)
            Each dict SHOULD ALSO echo back all input params for analysis.
        
        Raises:
            RuntimeError: If hardware fails during measurement (tip error, scale timeout, etc.)
            ValueError: If parameters are out of hardware limits.
        
        Important:
            - Always extract and use overaspirate_vol from params
            - Update state['measurement_count'] after each measurement
            - Return results in the exact format specified above
        """
        
        results = []
        
        for rep in range(replicates):
            # Extract parameters - use only what you need for your hardware.
            # Parameter names here must match the names defined in hardware_parameters in the config.
            overaspirate_vol = params.get('overaspirate_vol', 0.004) # You must use this parameter
            
            # TODO: Extract your hardware-specific parameters the same way, e.g.:
            # aspirate_speed = params.get('aspirate_speed', 10)   # matches hardware_parameters.aspirate_speed in config
            # aspirate_wait_time = params.get('aspirate_wait_time', 1.0)  # matches hardware_parameters.aspirate_wait_time
            # Then pass them to your hardware calls below.
            
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
                'measurement_budget_consumed': 1,  # Budget units consumed for this measurement (adjust if needed)
                **params  # Echo back all parameters
            }
            
            results.append(result)
        
        return results
    
    def wrapup(self, state: Dict[str, Any]) -> None:
        """Clean up hardware resources and return to safe state.
        
        Called once at the end of calibration to safely shut down hardware,
        close connections, and prepare for the next run.
        
        Args:
            state: Protocol state dictionary from initialize().
        
        Returns:
            None
        
        Note:
            Should not raise exceptions. Log any cleanup issues as warnings instead.
            Move hardware to safe positions (home position, tip ejection, etc.).
        """
        
        # TODO: Clean up your hardware here
        # my_robot.home()
        # my_robot.disconnect()
        
        print(f"Cleanup completed. Total measurements: {state.get('measurement_count', 0)}")

    def get_parameter_constraints(self, target_volume_ml: float) -> List[str]:
        """Get hardware-specific parameter constraints for optimization.
        
        Return a list of constraint strings that limit how parameter combinations
        can vary. Constraints are passed to the Ax optimizer in algebraic format.
        
        Args:
            target_volume_ml: The target volume for this optimization (in mL).
                              Use to calculate volume-dependent constraints.
        
        Returns:
            List of constraint strings in Ax format. Examples:
                - "my_air_vol + overaspirate_vol <= 0.15"  (volume limit)
                - "aspirate_speed * dispense_speed <= 1000"  (speed interaction)
                - []  (no constraints)
        
        Note:
            Constraints help the optimizer respect hardware limits without wasting
            trials on impossible parameter combinations.
            Return empty list if your hardware has no special constraints.
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
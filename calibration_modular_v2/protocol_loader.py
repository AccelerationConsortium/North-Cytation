"""
Simple Hardware Protocol Loader
===============================

Loads single-file hardware protocols that follow the 3-function interface:
- initialize(config) -> state
- measure(state, volume_mL, params, replicates) -> results  
- wrapup(state) -> None

Each hardware type gets exactly one file:
- calibration_protocol_example.py (Hardware interface)
- calibration_protocol_simulated.py (Simulation hardware)  
- calibration_protocol_template.py (Template for new hardware)

No complex classes, inheritance, or abstractions - just simple function imports.
"""

import importlib
import os
from typing import Dict, Any, List


def load_hardware_protocol(protocol_name: str):
    """
    Load a hardware protocol module by name.
    
    Args:
        protocol_name: Name of protocol file (without .py extension)
                      e.g., 'calibration_protocol_example' or 'calibration_protocol_simulated'
    
    Returns:
        Module with initialize(), measure(), and wrapup() functions
        
    Raises:
        ImportError: If protocol file doesn't exist or is missing required functions
    """
    try:
        # Import the protocol module
        protocol_module = importlib.import_module(protocol_name)
        
        # Verify it has the required 3 functions
        required_functions = ['initialize', 'measure', 'wrapup']
        missing_functions = [func for func in required_functions if not hasattr(protocol_module, func)]
        
        if missing_functions:
            raise ImportError(f"Protocol '{protocol_name}' missing required functions: {missing_functions}")
        
        return protocol_module
        
    except ImportError as e:
        available_protocols = _get_available_protocols()
        raise ImportError(f"Cannot load protocol '{protocol_name}': {e}\n"
                         f"Available protocols: {available_protocols}")


def _get_available_protocols() -> List[str]:
    """Get list of available protocol files in current directory."""
    protocols = []
    current_dir = os.path.dirname(__file__)
    
    for filename in os.listdir(current_dir):
        if filename.startswith('calibration_protocol_') and filename.endswith('.py'):
            protocol_name = filename[:-3]  # Remove .py extension
            protocols.append(protocol_name)
    
    return protocols


def get_available_protocols() -> List[str]:
    """Public function to list available hardware protocols."""
    return _get_available_protocols()


# Simple factory function for backward compatibility with experiment.py
def create_protocol(config, simulate: bool = True):
    """
    Create a protocol wrapper for backward compatibility.
    
    Args:
        config: ExperimentConfig object with protocol settings
        simulate: Whether to use simulation (ignored, determined by config)
        
    Returns:
        ProtocolWrapper object that provides the same interface as old protocol classes
    """
    # Determine which protocol to use based on config
    if hasattr(config, 'is_simulation') and config.is_simulation():
        protocol_name = 'calibration_protocol_simulated'
    else:
        protocol_name = 'calibration_protocol_example'  # Default hardware protocol
    
    # Load the protocol module
    protocol_module = load_hardware_protocol(protocol_name)
    
    # Return a wrapper that provides the old interface
    return ProtocolWrapper(protocol_module, config)


class ProtocolWrapper:
    """
    Wrapper to make the simple 3-function protocols work with existing experiment.py code.
    
    This provides the same interface as the old protocol classes while using the clean
    3-function hardware protocols underneath.
    """
    
    def __init__(self, protocol_module, config):
        self.protocol_module = protocol_module
        self.config = config
        self.state = None
    
    def initialize(self) -> bool:
        """Initialize the hardware protocol."""
        try:
            # Convert config to dict format expected by protocols
            config_dict = self._config_to_dict()
            self.state = self.protocol_module.initialize(config_dict)
            return True
        except Exception as e:
            print(f"Protocol initialization failed: {e}")
            return False
    
    def measure(self, parameters, target_volume_ml: float):
        """
        Execute measurement and return results.
        
        Args:
            parameters: PipettingParameters object
            target_volume_ml: Target volume in mL
            
        Returns:
            RawMeasurement object (converted from protocol results)
        """
        if self.state is None:
            raise RuntimeError("Protocol not initialized - call initialize() first")
        
        # Convert parameters to dict format
        params_dict = self._parameters_to_dict(parameters)
        
        # Call protocol measure function (single replicate)
        results = self.protocol_module.measure(self.state, target_volume_ml, params_dict, replicates=1)
        
        # Convert back to expected format
        if results and len(results) > 0:
            result = results[0]
            return self._create_raw_measurement(result, parameters, target_volume_ml)
        else:
            raise RuntimeError("Protocol measure() returned no results")
    
    def wrapup(self) -> bool:
        """Clean up the hardware protocol."""
        try:
            if self.state is not None:
                self.protocol_module.wrapup(self.state)
            return True
        except Exception as e:
            print(f"Protocol cleanup failed: {e}")
            return False
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert ExperimentConfig to dict format expected by protocols."""
        # Basic config conversion - add more fields as needed
        return {
            'liquid': self.config.get_liquid_name() if hasattr(self.config, 'get_liquid_name') else 'water',
            'random_seed': self.config.get_random_seed() if hasattr(self.config, 'get_random_seed') else None
        }
    
    def _parameters_to_dict(self, parameters) -> Dict[str, Any]:
        """Convert PipettingParameters to dict format."""
        # Convert dataclass to dict, handling different parameter formats
        if hasattr(parameters, '__dict__'):
            return {k: v for k, v in parameters.__dict__.items()}
        else:
            # Fallback for other parameter formats
            return dict(parameters)
    
    def _create_raw_measurement(self, protocol_result: Dict[str, Any], parameters, target_volume_ml: float):
        """Convert protocol result back to RawMeasurement format."""
        # Import here to avoid circular imports
        from data_structures import RawMeasurement
        import time
        
        return RawMeasurement(
            measurement_id=f"measurement_{int(time.time() * 1000)}",
            parameters=parameters,
            target_volume_ml=target_volume_ml,
            actual_volume_ml=protocol_result['volume'],
            duration_s=protocol_result['elapsed_s'],
            timestamp=time.time(),
            metadata=protocol_result  # Include all protocol data as metadata
        )
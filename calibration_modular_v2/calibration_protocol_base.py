"""Abstract base class for calibration protocols.

Defines the unified interface that all calibration protocols must implement.
This ensures consistency between simulation and hardware protocols.

Interface Contract:
    initialize(cfg) -> state (dict)
    measure(state, volume_mL, params, replicates) -> list[dict]
    wrapup(state) -> None

Per-replicate result keys (required):
    - replicate: int (1-indexed)
    - volume: float (measured volume in mL)
    - elapsed_s: float (measurement duration)

Per-replicate result keys (optional):
    - start_time: str (ISO format)
    - end_time: str (ISO format)
    - error: str (if measurement failed)
    - Any echoed parameters from input params dict
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class CalibrationProtocolBase(ABC):
    """Abstract base class for calibration protocols."""
    
    @abstractmethod
    def initialize(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize the protocol.
        
        Args:
            cfg: Optional configuration dictionary
            
        Returns:
            state: Protocol state dictionary for subsequent operations
            
        Raises:
            RuntimeError: If initialization fails
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def measure(self, state: Dict[str, Any], volume_mL: float, 
               params: Dict[str, Any], replicates: int = 1) -> List[Dict[str, Any]]:
        """Perform measurements with given parameters.
        
        Args:
            state: Protocol state from initialize()
            volume_mL: Target volume in milliliters
            params: Pipetting parameters dictionary
            replicates: Number of replicate measurements
            
        Returns:
            List of measurement results, one per replicate
            Each result must contain: replicate, volume, elapsed_s
            
        Raises:
            RuntimeError: If measurement hardware fails
            ValueError: If parameters are invalid
        """
        pass
    
    @abstractmethod
    def wrapup(self, state: Dict[str, Any]) -> None:
        """Clean up protocol resources.
        
        Args:
            state: Protocol state from initialize()
            
        Note:
            Should not raise exceptions - log warnings instead
        """
        pass
    
    def validate_measurement_result(self, result: Dict[str, Any], replicate_num: int) -> bool:
        """Validate that a measurement result meets interface requirements.
        
        Args:
            result: Single measurement result dictionary
            replicate_num: Expected replicate number (1-indexed)
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['replicate', 'volume', 'elapsed_s']
        
        # Check required keys exist
        for key in required_keys:
            if key not in result:
                return False
                
        # Check types and values
        try:
            if not isinstance(result['replicate'], int) or result['replicate'] != replicate_num:
                return False
            if not isinstance(result['volume'], (int, float)) or result['volume'] < 0:
                return False
            if not isinstance(result['elapsed_s'], (int, float)) or result['elapsed_s'] < 0:
                return False
        except (TypeError, KeyError):
            return False
            
        return True
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate that a state dictionary is properly formatted.
        
        Args:
            state: Protocol state dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(state, dict):
            return False
            
        # State should at minimum contain initialization timestamp
        required_keys = ['initialized_at']
        for key in required_keys:
            if key not in state:
                return False
                
        return True


# Factory function for protocol loading
def create_protocol(protocol_name: str) -> CalibrationProtocolBase:
    """Create a protocol instance by name.
    
    Args:
        protocol_name: Name of protocol ('simulated', 'hardware', etc.)
        
    Returns:
        Protocol instance implementing CalibrationProtocolBase
        
    Raises:
        ValueError: If protocol name is unknown
        ImportError: If protocol module cannot be imported
    """
    if protocol_name == 'simulated':
        from calibration_protocol_simulated import SimulatedCalibrationProtocol
        return SimulatedCalibrationProtocol()
    elif protocol_name == 'hardware':
        from calibration_protocol_hardware import HardwareCalibrationProtocol
        return HardwareCalibrationProtocol()
    else:
        raise ValueError(f"Unknown protocol: {protocol_name}. Available: 'simulated', 'hardware'")

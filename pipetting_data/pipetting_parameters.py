"""
Pipetting Parameters Configuration

This module defines the standard parameters for all pipetting operations,
ensuring consistency across the robot control methods.
"""

from dataclasses import dataclass
from typing import Optional
import copy


@dataclass
class PipettingParameters:
    """
    Standard parameters for pipetting liquid handling operations.
    
    These parameters control the core liquid handling behavior during aspirate 
    and dispense operations. Robot movement and measurement parameters are 
    handled separately at the method level.
    """
    
    # === SPEED PARAMETERS ===
    aspirate_speed: Optional[int] = None     # Speed for aspiration (uses tip default if None)
    dispense_speed: Optional[int] = None     # Speed for dispensing (uses tip default if None)
    retract_speed: Optional[int] = None      # Speed for retracting from liquid (uses 'retract' default if None)
    blowout_speed: Optional[int] = None      # Speed for blowout (uses aspirate speed if None)
    
    # === TIMING PARAMETERS ===
    aspirate_wait_time: float = 1.0         # Wait time after aspiration (seconds)
    dispense_wait_time: float = 1.0         # Wait time after dispensing (seconds)
    
    # === AIR GAP PARAMETERS ===
    pre_asp_air_vol: float = 0.0            # Air volume to aspirate before liquid (mL)
    post_asp_air_vol: float = 0.02           # Air volume to aspirate after liquid (mL)
    blowout_vol: float = 0.05                # Volume for blowout after dispensing (mL)
    air_vol: float = 0.0                    # Extra air volume for dispensing operations (mL)
    
    # === LIQUID HANDLING TECHNIQUES ===
    asp_disp_cycles: int = 0                # Number of aspirate/dispense mixing cycles




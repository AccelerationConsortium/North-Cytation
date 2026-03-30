#!/usr/bin/env python3
"""
Enhanced North Robotics Arm Position Control with X-Y Coordinate Interface
==========================================================================

This program provides a comprehensive GUI interface to control the North robot arm
with position presets, adjustment capabilities, and export functionality.

NEW FEATURES:
- Position dropdown with predefined locations from surfactant workflow
- Temporary position storage during adjustment sessions
- CSV/TXT export of position modifications
- Position validation and safety checks
- Integration with workflow position data
- X-Y coordinate control system (more intuitive than joint angles)
- Unified step size setting for both Z-axis and X-Y movement

Features:
- Full robot arm control (Z-axis, X-Y coordinates, gripper)
- Predefined position templates
- Position adjustment and temporary storage
- Export position data for workflow integration
- Real-time position display in Cartesian coordinates
- Safety features and error handling
- Support for both simulation and real hardware

Controls:
- Arrow keys for X-Y movement (Left/Right = X-axis, W/S keys = Y-axis)
- Up/Down arrow keys for Z-axis movement
- Unified step size setting (affects both Z and X-Y movement)
- Dropdown position selection
- Save/Load position presets
- Export position modifications

Coordinate System:
- X-axis: Left/Right movement (negative X = left, positive X = right)
- Y-axis: Forward/Back movement (negative Y = back, positive Y = forward) 
- Z-axis: Up/Down movement (higher mm = higher position)
- Step size setting controls movement increment for all axes

Author: Enhanced for North Robotics X-Y Position Management
Date: March 17, 2026
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
import time
import sys
import os
import csv
import json
import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path
from math import cos, sin

# Add the project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# Also add utoronto_demo path for positioning system
sys.path.append("../utoronto_demo")

try:
    from north import NorthC9
    NORTH_AVAILABLE = True
except ImportError:
    NORTH_AVAILABLE = False
    print("Warning: North robotics library not available. Running in simulation mode.")

# Configuration
SIMULATE = not NORTH_AVAILABLE  # Automatically simulate if North library not available
NETWORK_SERIAL = "AU06CNCF"  # Default network serial - update as needed
CONTROLLER_ADDR = "A"  # Controller address

# Movement parameters
DEFAULT_MOVE_INCREMENT_MM = 2.0      # Default mm per key press for Z-axis (reduced for precision)
DEFAULT_MOVE_INCREMENT_RAD = 0.05    # Default radians per key press (reduced for precision)
MAX_MOVE_INCREMENT_MM = 50.0         # Maximum allowed Z increment (mm)
MAX_MOVE_INCREMENT_RAD = 1.0         # Maximum allowed rotational increment (radians)
MIN_MOVE_INCREMENT_MM = 0.1          # Minimum allowed Z increment (mm)
MIN_MOVE_INCREMENT_RAD = 0.01        # Minimum allowed rotational increment (radians)
Z_AXIS_MIN_MM = 30.0                 # Minimum Z position (mm)
Z_AXIS_MAX_MM = 292.0                # Maximum Z position (mm)
SAFE_Z_HEIGHT_COUNTS = 10000         # Safe Z height in counts for collision avoidance (100mm high)
ELBOW_MIN_RAD = -(5/6) * 3.14159    # Minimum elbow angle (radians)
ELBOW_MAX_RAD = (5/6) * 3.14159     # Maximum elbow angle (radians)
SHOULDER_MIN_RAD = -(2/3) * 3.14159 # Minimum shoulder angle (radians)
SHOULDER_MAX_RAD = (2/3) * 3.14159  # Maximum shoulder angle (radians)
GRIPPER_MIN_RAD = -6.28              # Minimum gripper angle (radians)
GRIPPER_MAX_RAD = 6.28               # Maximum gripper angle (radians)

def load_positions_from_system(vial_file_path=None, include_pipetting=False):
    """
    Load positions dynamically from the multi-layer positioning system.
    
    Args:
        vial_file_path: Path to CSV file with vial definitions 
        include_pipetting: Include pipetting positions in addition to gripper positions
    
    Returns:
        Dictionary of positions in WORKFLOW_POSITIONS format
    """
    positions = {}
    
    # Add standard positions
    positions["Home"] = {
        "description": "Robot home position",
        "gripper_cts": 0,
        "shoulder_cts": 0, 
        "elbow_cts": 0,
        "z_cts": 0,
        "category": "Standard"
    }
    
    try:
        # Import positioning system
        import robot_state.Locator as Locator
        
        # Load vial positions YAML
        yaml_path = os.path.join("robot_state", "vial_positions.yaml")
        if not os.path.exists(yaml_path):
            yaml_path = os.path.join("..", "robot_state", "vial_positions.yaml")
            if not os.path.exists(yaml_path):
                # Try from project root
                yaml_path = os.path.join("..", "utoronto_demo", "robot_state", "vial_positions.yaml")
        with open(yaml_path, 'r') as f:
            vial_config = yaml.safe_load(f)
        
        # Load vial definitions from CSV if provided
        if vial_file_path and os.path.exists(vial_file_path):
            # Try pandas first, fallback to csv module if pandas fails
            try:
                vial_df = pd.read_csv(vial_file_path)
            except (AttributeError, ImportError):
                # Fallback CSV reader using built-in csv module
                import csv
                vial_data = []
                with open(vial_file_path, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        vial_data.append(row)
                print(f"Loaded CSV using fallback reader: {len(vial_data)} vials")
                
                # Mock pandas DataFrame behavior for compatibility
                class SimpleDataFrame:
                    def __init__(self, data):
                        self.data = data
                    def iterrows(self):
                        for i, row in enumerate(self.data):
                            yield i, row
                
                vial_df = SimpleDataFrame(vial_data)
            
            # Add positions for each defined vial
            for _, vial in vial_df.iterrows():
                vial_name = vial['vial_name']
                location = vial['location']
                location_index = int(vial['location_index'])
                
                if location in vial_config:
                    config = vial_config[location]
                    
                    # Add gripper position
                    if config.get('vial_positions_in_Locator'):
                        locator_var = config['vial_positions_in_Locator']
                        if hasattr(Locator, locator_var):
                            coords_array = getattr(Locator, locator_var)
                            if location_index < len(coords_array):
                                coords = coords_array[location_index]
                                positions[f"{vial_name} (Gripper)"] = {
                                    "description": f"{vial_name} at {location}[{location_index}] - gripper access",
                                    "gripper_cts": coords[0],
                                    "shoulder_cts": coords[1], 
                                    "elbow_cts": coords[2],
                                    "z_cts": coords[3],
                                    "category": "Vial_Gripper"
                                }
                    
                    # Add pipetting position if requested
                    if include_pipetting and config.get('pipetting_positions_in_Locator'):
                        pip_var = config['pipetting_positions_in_Locator']
                        if hasattr(Locator, pip_var):
                            coords_array = getattr(Locator, pip_var)
                            if location_index < len(coords_array):
                                coords = coords_array[location_index]
                                positions[f"{vial_name} (Pipetting)"] = {
                                    "description": f"{vial_name} at {location}[{location_index}] - pipetting access",
                                    "gripper_cts": coords[0],
                                    "shoulder_cts": coords[1],
                                    "elbow_cts": coords[2], 
                                    "z_cts": coords[3],
                                    "category": "Vial_Pipetting"
                                }
        
        # Add special positions
        if hasattr(Locator, 'vial_clamp'):
            coords = Locator.vial_clamp
            positions["Clamp Position"] = {
                "description": "Vial clamping station for manipulation", 
                "gripper_cts": coords[0],
                "shoulder_cts": coords[1],
                "elbow_cts": coords[2],
                "z_cts": coords[3],
                "category": "Manipulation"
            }
            
        # Add main rack positions for reference
        if hasattr(Locator, 'rack'):
            for i in range(min(10, len(Locator.rack))):  # Limit to first 10 for UI
                coords = Locator.rack[i]
                positions[f"Main Rack - Position {i}"] = {
                    "description": f"8mL vial rack position {i}",
                    "gripper_cts": coords[0],
                    "shoulder_cts": coords[1],
                    "elbow_cts": coords[2], 
                    "z_cts": coords[3],
                    "category": "Storage"
                }
        
        # Add large vial positions
        if hasattr(Locator, 'large_vial_pip'):
            for i, coords in enumerate(Locator.large_vial_pip):
                positions[f"Large Vial Rack - Position {i}"] = {
                    "description": f"20mL vial rack position {i}",
                    "gripper_cts": coords[0],
                    "shoulder_cts": coords[1],
                    "elbow_cts": coords[2],
                    "z_cts": coords[3],
                    "category": "Large Volume"
                }
        
        # Add pipette tip rack positions (MISSING from original implementation)
        if hasattr(Locator, 'pgrid_low_2'):
            for i, coords in enumerate(Locator.pgrid_low_2):
                positions[f"Large Pipette Tip - Position {i}"] = {
                    "description": f"Large pipette tip rack position {i} (for picking up tips)",
                    "gripper_cts": coords[0],
                    "shoulder_cts": coords[1],
                    "elbow_cts": coords[2],
                    "z_cts": coords[3],
                    "category": "Pipette_Tips"
                }
        
        if hasattr(Locator, 'pgrid_high_2'):
            for i, coords in enumerate(Locator.pgrid_high_2):
                positions[f"Small Pipette Tip - Position {i}"] = {
                    "description": f"Small pipette tip rack position {i} (for picking up tips)",
                    "gripper_cts": coords[0],
                    "shoulder_cts": coords[1],
                    "elbow_cts": coords[2],
                    "z_cts": coords[3],
                    "category": "Pipette_Tips"
                }
        
        # Add tip removal positions (CRITICAL for tip disposal)
        if hasattr(Locator, 'p_remove_cap'):
            coords = Locator.p_remove_cap
            positions["Tip Removal - Cap"] = {
                "description": "Tip removal station for cap tips",
                "gripper_cts": coords[0],
                "shoulder_cts": coords[1],
                "elbow_cts": coords[2],
                "z_cts": coords[3],
                "category": "Tip_Removal"
            }
        
        if hasattr(Locator, 'p_remove_approach'):
            coords = Locator.p_remove_approach
            positions["Tip Removal - Approach"] = {
                "description": "Safe approach position before tip removal",
                "gripper_cts": coords[0],
                "shoulder_cts": coords[1],
                "elbow_cts": coords[2],
                "z_cts": coords[3],
                "category": "Tip_Removal"
            }
        
        if hasattr(Locator, 'p_remove_small'):
            coords = Locator.p_remove_small
            positions["Tip Removal - Small"] = {
                "description": "Tip removal station for small tips",
                "gripper_cts": coords[0],
                "shoulder_cts": coords[1],
                "elbow_cts": coords[2],
                "z_cts": coords[3],
                "category": "Tip_Removal"
            }
                
    except Exception as e:
        print(f"Warning: Could not load dynamic positions: {e}")
        print("Using fallback hardcoded positions")
        # Add fallback positions
        positions.update({
            "Clamp Position": {
                "description": "Vial clamping station for manipulation",
                "gripper_cts": 2476,
                "shoulder_cts": 7635,
                "elbow_cts": 44106,
                "z_cts": 17350,
                "category": "Manipulation"
            },
            "Main Rack - Position 3 (Water)": {
                "description": "8mL vial rack position 3 (water)",
                "gripper_cts": 1922,
                "shoulder_cts": 12072,
                "elbow_cts": 38811,
                "z_cts": 22200,
                "category": "Reagents"
            }})
        
    return positions

# Initialize with default positions - will be updated by controller
WORKFLOW_POSITIONS = load_positions_from_system()

# Legacy fallback positions for reference
FALLBACK_POSITIONS = {
    # Standard positions
    "Home": {
        "description": "Robot home position",
        "gripper_cts": 0,
        "shoulder_cts": 0,
        "elbow_cts": 0,
        "z_cts": 0,
        "category": "Standard"
    },
    
    # Clamp position (from Locator.py vial_clamp)
    "Clamp Position": {
        "description": "Vial clamping station for manipulation",
        "gripper_cts": 2476,
        "shoulder_cts": 7635,
        "elbow_cts": 44106,
        "z_cts": 17350,
        "category": "Manipulation"
    },
    
    # Main 8mL vial rack positions (key positions used in surfactant workflow)
    "Main Rack - Position 36": {
        "description": "8mL vial rack position 36",
        "gripper_cts": 1708,
        "shoulder_cts": 9251,
        "elbow_cts": 27829,
        "z_cts": 22200,
        "category": "Storage"
    },
    "Main Rack - Position 43": {
        "description": "8mL vial rack position 43",
        "gripper_cts": 2021,
        "shoulder_cts": 1163,
        "elbow_cts": 19726,
        "z_cts": 22200,
        "category": "Storage"
    },
    "Main Rack - Position 44 (Water)": {
        "description": "8mL vial rack position 44 (water)",
        "gripper_cts": 1928,
        "shoulder_cts": 2422,
        "elbow_cts": 19859,
        "z_cts": 22200,
        "category": "Reagents"
    },
    "Main Rack - Position 45 (Water_2)": {
        "description": "8mL vial rack position 45 (water_2)",
        "gripper_cts": 1845,
        "shoulder_cts": 3750,
        "elbow_cts": 20404,
        "z_cts": 22200,
        "category": "Reagents"
    },
    "Main Rack - Position 46": {
        "description": "8mL vial rack position 46",
        "gripper_cts": 1768,
        "shoulder_cts": 5155,
        "elbow_cts": 21233,
        "z_cts": 22200,
        "category": "Storage"
    },
    "Main Rack - Position 47": {
        "description": "8mL vial rack position 47",
        "gripper_cts": 1692,
        "shoulder_cts": 6655,
        "elbow_cts": 22291,
        "z_cts": 22200,
        "category": "Storage"
    },
    
    # Large vial positions
    "Large Vial Rack - Position 0": {
        "description": "20mL vial rack position 0",
        "gripper_cts": 721,
        "shoulder_cts": 9854,
        "elbow_cts": 26732,
        "z_cts": 13921,
        "category": "Large Volume"
    },
    "Large Vial Rack - Position 1": {
        "description": "20mL vial rack position 1", 
        "gripper_cts": 405,
        "shoulder_cts": 17735,
        "elbow_cts": 34375,
        "z_cts": 13921,
        "category": "Large Volume"
    },
    
    # Small vial positions
    "Small Vial Rack - Position 0": {
        "description": "1mL vial rack position 0",
        "gripper_cts": 848,
        "shoulder_cts": 15038,
        "elbow_cts": 44585,
        "z_cts": 16166,
        "category": "Small Volume"
    },
    "Small Vial Rack - Position 1": {
        "description": "1mL vial rack position 1",
        "gripper_cts": 788,
        "shoulder_cts": 16021,
        "elbow_cts": 45013,
        "z_cts": 16166,
        "category": "Small Volume"
    },
    
    # 50mL vial positions
    "50mL Vial Rack - Position 0": {
        "description": "50mL vial rack position 0",
        "gripper_cts": -1378,
        "shoulder_cts": 14164,
        "elbow_cts": 35944,
        "z_cts": 11100,
        "category": "Large Volume"
    },
    "50mL Vial Rack - Position 3": {
        "description": "50mL vial rack position 3",
        "gripper_cts": -1518,
        "shoulder_cts": 15521,
        "elbow_cts": 35093,
        "z_cts": 11100,
        "category": "Large Volume"
    },
    
    # Photoreactor position
    "Photoreactor Position": {
        "description": "Photoreactor array position",
        "gripper_cts": 69,
        "shoulder_cts": 33380,
        "elbow_cts": 24059,
        "z_cts": 20577,
        "category": "Processing"
    },
    
    # Heater grid positions
    "Heater Grid - Position 0": {
        "description": "Heater grid position 0",
        "gripper_cts": -1232,
        "shoulder_cts": 39645,
        "elbow_cts": 13715,
        "z_cts": 15241,
        "category": "Processing"
    },
    "Heater Grid - Position 5": {
        "description": "Heater grid position 5 (center area)",
        "gripper_cts": -949,
        "shoulder_cts": 39840,
        "elbow_cts": 21249,
        "z_cts": 15241,
        "category": "Processing"
    },
    
    # Tip rack positions - Small tips
    "Small Tip Rack - Position 0": {
        "description": "Small pipet tip rack position 0",
        "gripper_cts": 2668,
        "shoulder_cts": 37357,
        "elbow_cts": 8247,
        "z_cts": 9150,
        "category": "Tip Racks"
    },
    "Small Tip Rack - Position 24": {
        "description": "Small pipet tip rack position 24 (middle)",
        "gripper_cts": 2893,
        "shoulder_cts": 35374,
        "elbow_cts": 9991,
        "z_cts": 9150,
        "category": "Tip Racks"
    },
    
    # Tip rack positions - Large tips
    "Large Tip Rack - Position 0": {
        "description": "Large pipet tip rack position 0",
        "gripper_cts": 2677,
        "shoulder_cts": 39055,
        "elbow_cts": 11930,
        "z_cts": 12562,
        "category": "Tip Racks"
    },
    "Large Tip Rack - Position 24": {
        "description": "Large pipet tip rack position 24 (middle)",
        "gripper_cts": 2912,
        "shoulder_cts": 36904,
        "elbow_cts": 13611,
        "z_cts": 12562,
        "category": "Tip Racks"
    },
    
    # Safe positions for transport
    "Safe Transport Height": {
        "description": "Safe height for moving between positions",
        "gripper_cts": 0,
        "shoulder_cts": 0,
        "elbow_cts": 0,
        "z_cts": 5000,  # High safe position
        "category": "Standard"
    }
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_arm_position_control.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedRobotArmController:
    """Enhanced controller class with position presets and export functionality."""
    
    def __init__(self, vial_file_path=None, include_pipetting=False):
        self.robot = None
        self.root = None
        self.is_connected = False
        self.current_z_position = 0.0
        self.current_x_position = 0.0
        self.current_y_position = 0.0
        self.current_gripper_angle = 0.0
        self.gripper_is_open = False
        self.is_homed = False
        
        # Position system configuration
        self.vial_file_path = vial_file_path
        self.include_pipetting = include_pipetting
        self.workflow_positions = load_positions_from_system(vial_file_path, include_pipetting)
        
        # Position management
        self.selected_position = None
        self.original_position = None  # Store original position before modifications
        self.temporary_modifications = {}  # Store temporary position changes
        self.session_history = []  # Track all position changes in session
        
        # Custom position naming
        self.custom_position_name = None  # Will be set in GUI initialization
        
        # Create temp directory for saved positions
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Movement increments
        self.move_increment_mm = DEFAULT_MOVE_INCREMENT_MM
        self.move_increment_rad = DEFAULT_MOVE_INCREMENT_RAD
        
        # GUI components
        self.status_label = None
        self.position_dropdown = None
        self.position_description_label = None
        self.z_position_label = None
        self.x_position_label = None
        self.y_position_label = None
        self.gripper_position_label = None
        self.gripper_status_label = None
        self.home_button = None
        self.z_increment_entry = None
        self.rad_increment_entry = None
        self.increment_status_label = None
        self.modifications_display = None
        
        # Configuration components
        self.vial_file_var = None
        self.include_pipetting_var = None
        self.config_frame = None
        
    def connect_robot(self):
        """Initialize connection to the North robot."""
        try:
            if SIMULATE:
                logger.info("Starting in simulation mode")
                self.robot = self._create_mock_robot()
                self.is_connected = True
                self.current_z_position = 100.0
                return True
            else:
                logger.info(f"Connecting to robot at address {CONTROLLER_ADDR} with serial {NETWORK_SERIAL}")
                self.robot = NorthC9(
                    CONTROLLER_ADDR,
                    network_serial=NETWORK_SERIAL,
                    verbose=True
                )
                
                self.robot.get_info()
                self.is_connected = True
                logger.info("Successfully connected to robot")
                
                # Get current positions for all axes
                positions = self.robot.get_robot_positions()
                self.current_gripper_angle = self.robot.counts_to_rad(self.robot.GRIPPER, positions[0])
                self.current_elbow_angle = self.robot.counts_to_rad(self.robot.ELBOW, positions[1])
                self.current_shoulder_angle = self.robot.counts_to_rad(self.robot.SHOULDER, positions[2])
                self.current_z_position = self.robot.counts_to_mm(self.robot.Z_AXIS, positions[3])
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to robot: {str(e)}")
            messagebox.showerror("Connection Error", f"Failed to connect to robot:\n{str(e)}")
            return False
            
    def _create_mock_robot(self):
        """Create a mock robot object for simulation."""
        class MockRobot:
            Z_AXIS = 3
            ELBOW = 1
            SHOULDER = 2
            GRIPPER = 0
            
            def __init__(self):
                self.z_position_mm = 100.0
                self.x_position_mm = 0.0
                self.y_position_mm = 0.0
                self.gripper_angle_rad = 0.0
                self.shoulder_angle_rad = 0.0
                self.elbow_angle_rad = 0.0
                self.gripper_is_open = False
                self.is_homed = False
                
                # Count representations for compatibility
                self.gripper_cts = 0
                self.shoulder_cts = 0
                self.elbow_cts = 0
                self.z_cts = 10000  # 100mm * 100 counts/mm
                
            def n9_fk(self, gripper_cts, elbow_cts, shoulder_cts):
                """Simple forward kinematics for mock robot."""
                # Convert counts to radians
                gripper_rad = gripper_cts / 1000.0
                elbow_rad = elbow_cts / 1000.0
                shoulder_rad = shoulder_cts / 1000.0
                
                # Simplified forward kinematics approximation
                # This is just for display purposes in simulation
                r1 = 150.0  # Approximate upper arm length in mm
                r2 = 150.0  # Approximate forearm length in mm
                
                x = r1 * cos(shoulder_rad) + r2 * cos(shoulder_rad + elbow_rad)
                y = r1 * sin(shoulder_rad) + r2 * sin(shoulder_rad + elbow_rad)
                
                return (x, y, shoulder_rad + elbow_rad)  # x, y, tool angle
                
            def home_robot(self, wait=True):
                logger.info("SIMULATION: Homing robot")
                time.sleep(0.5)
                self.z_position_mm = Z_AXIS_MAX_MM
                self.gripper_angle_rad = 0.0
                self.shoulder_angle_rad = 0.0
                self.elbow_angle_rad = 0.0
                self.gripper_is_open = False
                self.is_homed = True
                
                # Update count representations
                self.gripper_cts = int(self.gripper_angle_rad * 1000)
                self.shoulder_cts = int(self.shoulder_angle_rad * 1000)
                self.elbow_cts = int(self.elbow_angle_rad * 1000)
                self.z_cts = int(self.z_position_mm * 100)
                
                # Calculate X-Y from joint angles using forward kinematics
                x, y, _ = self.n9_fk(self.gripper_cts, self.elbow_cts, self.shoulder_cts)
                self.x_position_mm = x
                self.y_position_mm = y
                
                return True
                
            def move_z(self, mm, wait=True):
                logger.info(f"SIMULATION: Moving Z-axis to {mm:.1f}mm")
                if mm < Z_AXIS_MIN_MM or mm > Z_AXIS_MAX_MM:
                    raise ValueError(f"Z position {mm:.1f}mm is out of range [{Z_AXIS_MIN_MM}-{Z_AXIS_MAX_MM}]")
                time.sleep(0.2)
                self.z_position_mm = mm
            
            def move_xy(self, x, y, wait=True, **kwargs):
                """Move to X-Y coordinates."""
                logger.info(f"SIMULATION: Moving to X={x:.1f}mm, Y={y:.1f}mm")
                # Basic workspace bounds checking
                if abs(x) > 300 or abs(y) > 300:
                    raise ValueError(f"X-Y position ({x:.1f}, {y:.1f}) is out of workspace")
                time.sleep(0.3)
                self.x_position_mm = x
                self.y_position_mm = y
                
            def move_axis_rad(self, axis, rad, wait=True):
                logger.info(f"SIMULATION: Moving axis {axis} to {rad:.2f} radians")
                time.sleep(0.2)
                if axis == self.SHOULDER:
                    if rad < SHOULDER_MIN_RAD or rad > SHOULDER_MAX_RAD:
                        raise ValueError(f"Shoulder position {rad:.2f}rad is out of range")
                    self.shoulder_angle_rad = rad
                    self.shoulder_cts = int(rad * 1000)
                elif axis == self.ELBOW:
                    if rad < ELBOW_MIN_RAD or rad > ELBOW_MAX_RAD:
                        raise ValueError(f"Elbow position {rad:.2f}rad is out of range")
                    self.elbow_angle_rad = rad
                    self.elbow_cts = int(rad * 1000)
                elif axis == self.GRIPPER:
                    if rad < GRIPPER_MIN_RAD or rad > GRIPPER_MAX_RAD:
                        raise ValueError(f"Gripper position {rad:.2f}rad is out of range")
                    self.gripper_angle_rad = rad
                    self.gripper_cts = int(rad * 1000)
                
                # Recalculate X-Y coordinates after joint movement
                x, y, _ = self.n9_fk(self.gripper_cts, self.elbow_cts, self.shoulder_cts)
                self.x_position_mm = x
                self.y_position_mm = y
                
            def open_gripper(self):
                logger.info("SIMULATION: Opening gripper")
                self.gripper_is_open = True
                
            def close_gripper(self):
                logger.info("SIMULATION: Closing gripper")
                self.gripper_is_open = False
                
            def goto(self, position_list, wait=True):
                """Move to absolute position using counts [gripper, shoulder, elbow, z] (Locator.py format)."""
                logger.info(f"SIMULATION: Moving to position {position_list}")
                time.sleep(0.5)  # Simulate movement time
                
                # Convert counts to internal values (simplified conversion)
                # Format: [gripper, shoulder, elbow, z_axis] as per Locator.py
                self.gripper_angle_rad = position_list[0] / 1000.0   # Mock conversion
                self.shoulder_angle_rad = position_list[1] / 1000.0  # shoulder is index 1
                self.elbow_angle_rad = position_list[2] / 1000.0     # elbow is index 2  
                self.z_position_mm = position_list[3] / 100.0        # Mock conversion (100 counts = 1mm)
                
                # Apply limits
                self.gripper_angle_rad = max(GRIPPER_MIN_RAD, min(GRIPPER_MAX_RAD, self.gripper_angle_rad))
                self.elbow_angle_rad = max(ELBOW_MIN_RAD, min(ELBOW_MAX_RAD, self.elbow_angle_rad))
                self.shoulder_angle_rad = max(SHOULDER_MIN_RAD, min(SHOULDER_MAX_RAD, self.shoulder_angle_rad))
                self.z_position_mm = max(Z_AXIS_MIN_MM, min(Z_AXIS_MAX_MM, self.z_position_mm))
                
                # Update count representations
                self.gripper_cts = int(self.gripper_angle_rad * 1000)
                self.shoulder_cts = int(self.shoulder_angle_rad * 1000)
                self.elbow_cts = int(self.elbow_angle_rad * 1000)
                self.z_cts = int(self.z_position_mm * 100)
                
                # Calculate X-Y from joint angles using forward kinematics
                x, y, _ = self.n9_fk(self.gripper_cts, self.elbow_cts, self.shoulder_cts)
                self.x_position_mm = x
                self.y_position_mm = y
                
            def get_axis_position(self, axis):
                if axis == self.Z_AXIS:
                    return int(self.z_position_mm * 100)
                elif axis == self.ELBOW:
                    return int(self.elbow_angle_rad * 1000)
                elif axis == self.SHOULDER:
                    return int(self.shoulder_angle_rad * 1000)
                elif axis == self.GRIPPER:
                    return int(self.gripper_angle_rad * 1000)
                return 0
                
            def get_robot_positions(self):
                """Return current robot positions in Locator.py format [gripper, shoulder, elbow, z]."""
                return [
                    int(self.gripper_angle_rad * 1000),   # gripper (index 0)
                    int(self.shoulder_angle_rad * 1000),  # shoulder (index 1)  
                    int(self.elbow_angle_rad * 1000),     # elbow (index 2)
                    int(self.z_position_mm * 100)         # z_axis (index 3)
                ]
                
            def counts_to_mm(self, axis, counts):
                return counts / 100.0
                
            def counts_to_rad(self, axis, counts):
                return counts / 1000.0
            

                
            def get_info(self):
                logger.info("SIMULATION: Robot info - Mock North C9 Controller")
        
        return MockRobot()
    
    def create_gui(self):
        """Create the enhanced GUI window with position management."""
        self.root = tk.Tk()
        self.root.title("Enhanced North Robot Arm Position Control")
        self.root.geometry("750x900")
        self.root.resizable(True, False)
        
        # Set up the GUI layout
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Enhanced North Robot Arm Control", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Connection Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Disconnected", foreground="red")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Position Selection section
        position_select_frame = ttk.LabelFrame(main_frame, text="Position Selection", padding="10")
        position_select_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(position_select_frame, text="Select Workflow Position:").grid(row=0, column=0, sticky=tk.W, pady=2)
        
        # Position dropdown
        position_names = list(self.workflow_positions.keys())
        self.position_dropdown = ttk.Combobox(position_select_frame, values=position_names, 
                                            state="readonly", width=30)
        self.position_dropdown.grid(row=0, column=1, padx=(10, 5), pady=2)
        self.position_dropdown.bind('<<ComboboxSelected>>', self.on_position_selected)
        
        # Go to position button
        goto_button = ttk.Button(position_select_frame, text="Go To Position", 
                                command=self.goto_selected_position)
        goto_button.grid(row=0, column=2, padx=(5, 0), pady=2)
        
        # Position description
        self.position_description_label = ttk.Label(position_select_frame, text="Select a position to see description",
                                                   font=("Arial", 9), foreground="gray")
        self.position_description_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Current Position section
        position_frame = ttk.LabelFrame(main_frame, text="Current Joint Positions", padding="10")
        position_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.z_position_label = ttk.Label(position_frame, text="Z-Axis: Unknown", 
                                         font=("Arial", 10))
        self.z_position_label.grid(row=0, column=0, sticky=tk.W)
        
        self.shoulder_position_label = ttk.Label(position_frame, text="Shoulder: Unknown", 
                                                font=("Arial", 10))
        self.shoulder_position_label.grid(row=1, column=0, sticky=tk.W)
        
        self.elbow_position_label = ttk.Label(position_frame, text="Elbow: Unknown", 
                                             font=("Arial", 10))
        self.elbow_position_label.grid(row=2, column=0, sticky=tk.W)
        
        self.gripper_position_label = ttk.Label(position_frame, text="Gripper: Unknown", 
                                               font=("Arial", 10))
        self.gripper_position_label.grid(row=3, column=0, sticky=tk.W)
        
        self.gripper_status_label = ttk.Label(position_frame, text="Gripper: Closed", 
                                             font=("Arial", 10, "bold"))
        self.gripper_status_label.grid(row=4, column=0, sticky=tk.W)
        
        # Position Modifications section
        modifications_frame = ttk.LabelFrame(main_frame, text="Position Modifications", padding="10")
        modifications_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Text widget to show modifications
        self.modifications_display = tk.Text(modifications_frame, height=4, width=70, 
                                           font=("Consolas", 8))
        self.modifications_display.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        self.modifications_display.insert(tk.END, "No modifications yet. Select a position and make adjustments.")
        self.modifications_display.config(state=tk.DISABLED)
        
        # Custom position name input
        name_frame = ttk.Frame(modifications_frame)
        name_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 5))
        ttk.Label(name_frame, text="Position Name:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.custom_position_name = ttk.Entry(name_frame, width=25, font=("Arial", 9))
        self.custom_position_name.grid(row=0, column=1, padx=(0, 10), sticky=(tk.W, tk.E))
        self.custom_position_name.insert(0, "my_position_1")
        
        # Modification buttons
        button_frame = ttk.Frame(modifications_frame)
        button_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
        save_custom_button = ttk.Button(button_frame, text="💾 Save with Custom Name", 
                                       command=self.save_custom_position)
        save_custom_button.grid(row=0, column=0, padx=(0, 5), pady=2)
        
        save_temp_button = ttk.Button(button_frame, text="Save as Temp", 
                                     command=self.save_temporary_position)
        save_temp_button.grid(row=0, column=1, padx=(0, 5), pady=2)
        
        reset_button = ttk.Button(button_frame, text="Reset to Original", 
                                 command=self.reset_to_original_position)
        reset_button.grid(row=0, column=2, padx=(0, 5), pady=2)
        
        view_saved_button = ttk.Button(button_frame, text="📋 View Saved", 
                                      command=self.show_saved_positions_window)
        view_saved_button.grid(row=1, column=0, padx=(0, 5), pady=2)
        
        clear_temp_button = ttk.Button(button_frame, text="Clear All Temp", 
                                      command=self.clear_temporary_positions)
        clear_temp_button.grid(row=1, column=1, pady=2)
        
        # Configure grid weights for name frame
        name_frame.columnconfigure(1, weight=1)
        
        # Movement Settings section
        increment_frame = ttk.LabelFrame(main_frame, text="Movement Settings", padding="10")
        increment_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Z-axis increment
        z_inc_frame = ttk.Frame(increment_frame)
        z_inc_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(z_inc_frame, text="Movement Step (mm):").grid(row=0, column=0, sticky=tk.W)
        self.z_increment_entry = ttk.Entry(z_inc_frame, width=8)
        self.z_increment_entry.grid(row=0, column=1, padx=(10, 5))
        self.z_increment_entry.insert(0, str(DEFAULT_MOVE_INCREMENT_MM))
        ttk.Button(z_inc_frame, text="Set", command=self.update_z_increment).grid(row=0, column=2, padx=2)
        
        # Rotational increment
        rad_inc_frame = ttk.Frame(increment_frame)
        rad_inc_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(rad_inc_frame, text="Rotation Step (rad):").grid(row=0, column=0, sticky=tk.W)
        self.rad_increment_entry = ttk.Entry(rad_inc_frame, width=8)
        self.rad_increment_entry.grid(row=0, column=1, padx=(10, 5))
        self.rad_increment_entry.insert(0, str(DEFAULT_MOVE_INCREMENT_RAD))
        ttk.Button(rad_inc_frame, text="Set", command=self.update_rad_increment).grid(row=0, column=2, padx=2)
        
        # Status for increments
        self.increment_status_label = ttk.Label(increment_frame, text="Ready", foreground="green")
        self.increment_status_label.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        
        # Controls section
        control_frame = ttk.LabelFrame(main_frame, text="Robot Controls", padding="10")
        control_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Home button
        self.home_button = ttk.Button(control_frame, text="🏠 HOME ROBOT", 
                                     command=self.home_robot, style="Accent.TButton")
        self.home_button.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        
        # Movement buttons (condensed layout)
        ttk.Label(control_frame, text="Z-Axis:").grid(row=1, column=0, sticky=tk.W)
        ttk.Button(control_frame, text="▲ UP", command=self.move_up).grid(row=1, column=1, padx=2)
        ttk.Button(control_frame, text="▼ DOWN", command=self.move_down).grid(row=1, column=2, padx=2)
        
        ttk.Label(control_frame, text="X-Axis:").grid(row=2, column=0, sticky=tk.W)
        ttk.Button(control_frame, text="← LEFT (-X)", command=self.move_x_left).grid(row=2, column=1, padx=2)
        ttk.Button(control_frame, text="→ RIGHT (+X)", command=self.move_x_right).grid(row=2, column=2, padx=2)
        
        ttk.Label(control_frame, text="Y-Axis:").grid(row=3, column=0, sticky=tk.W)
        ttk.Button(control_frame, text="↓ BACK (-Y)", command=self.move_y_back).grid(row=3, column=1, padx=2)
        ttk.Button(control_frame, text="↑ FORWARD (+Y)", command=self.move_y_forward).grid(row=3, column=2, padx=2)
        
        ttk.Label(control_frame, text="Gripper:").grid(row=4, column=0, sticky=tk.W)
        ttk.Button(control_frame, text="✋ OPEN", command=self.open_gripper, 
                  style="Success.TButton").grid(row=4, column=1, padx=2)
        ttk.Button(control_frame, text="👊 CLOSE", command=self.close_gripper, 
                  style="Warning.TButton").grid(row=4, column=2, padx=2)
        
        # Configuration section
        if hasattr(self, 'create_config_frame'):
            self.create_config_frame(main_frame)
        
        # Export section
        export_frame = ttk.LabelFrame(main_frame, text="Export Position Data", padding="10")
        export_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(export_frame, text="📊 Export Saved Positions (CSV)", 
                  command=self.export_saved_positions_csv).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(export_frame, text="📋 Copy Position Coordinates", 
                  command=self.copy_current_position).grid(row=0, column=1, padx=(0, 5))  
        ttk.Button(export_frame, text="📄 Export Session History", 
                  command=self.export_session_history).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(export_frame, text="🗂️ Open Temp Folder", 
                  command=self.open_temp_folder).grid(row=1, column=0, padx=(0, 5), pady=(5, 0))
        ttk.Button(export_frame, text="📤 Export All Positions (JSON)", 
                  command=self.export_all_positions_json).grid(row=1, column=1, padx=(0, 5), pady=(5, 0))
        
        # Configure styles
        style = ttk.Style()
        style.configure("Accent.TButton", foreground="blue")
        style.configure("Success.TButton", foreground="green")
        style.configure("Warning.TButton", foreground="orange")
        
        # Bind keyboard events
        self.root.bind('<Key>', self.on_key_press)
        self.root.focus_set()
        
        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)
        
    def on_position_selected(self, event):
        """Handle position dropdown selection."""
        position_name = self.position_dropdown.get()
        if position_name in self.workflow_positions:
            position_data = self.workflow_positions[position_name]
            description = f"{position_data['description']} (Category: {position_data['category']})"
            self.position_description_label.config(text=description)
            self.selected_position = position_name
            
    def goto_selected_position(self):
        """Move robot to the selected workflow position."""
        if not self.is_connected:
            messagebox.showerror("Error", "Robot not connected")
            return
            
        if not self.selected_position:
            messagebox.showwarning("Warning", "Please select a position first")
            return
            
        try:
            position_data = self.workflow_positions[self.selected_position]
            
            # Store original position before movement
            self.store_original_position()
            
            # Move to position
            logger.info(f"Moving to position: {self.selected_position}")
            
            # For home position, use the home command
            if self.selected_position == "Home":
                self.robot.home_robot(wait=True)
                logger.info(f"Successfully homed robot to {self.selected_position}")
            else:
                # SAFE TWO-STEP MOVEMENT to avoid collision with vials:
                # Step 1: Move to safe Z height first if not already high enough
                try:
                    current_pos = self.robot.get_robot_positions()  # [gripper, shoulder, elbow, z]
                    current_z = current_pos[3]
                except (AttributeError, IndexError):
                    # If we can't get position, assume we need to go to safe height
                    current_z = SAFE_Z_HEIGHT_COUNTS + 1000
                    current_pos = [0, 0, 0, current_z]  # Default safe position
                
                if current_z > SAFE_Z_HEIGHT_COUNTS:  # Lower count = higher position
                    logger.info(f"Moving to safe Z height ({SAFE_Z_HEIGHT_COUNTS} counts) first")
                    safe_position = [
                        current_pos[0],  # Keep current gripper
                        current_pos[1],  # Keep current shoulder  
                        current_pos[2],  # Keep current elbow
                        SAFE_Z_HEIGHT_COUNTS  # Move to safe height
                    ]
                    self.robot.goto(safe_position, wait=True)
                
                # Step 2: Move X-Y to target position while staying at safe height
                logger.info(f"Moving to target X-Y position: {self.selected_position}")
                safe_xy_position = [
                    position_data["gripper_cts"],
                    position_data["shoulder_cts"],  # shoulder comes second in Locator format
                    position_data["elbow_cts"],     # elbow comes third in Locator format
                    SAFE_Z_HEIGHT_COUNTS  # Stay at safe height
                ]
                self.robot.goto(safe_xy_position, wait=True)
                
                # Step 3: Finally lower to target Z position
                logger.info(f"Lowering to target Z position ({position_data['z_cts']} counts)")
                target_position = [
                    position_data["gripper_cts"],
                    position_data["shoulder_cts"],  
                    position_data["elbow_cts"],     
                    position_data["z_cts"]  # Final target Z
                ]
                self.robot.goto(target_position, wait=True)
                logger.info(f"Successfully moved to {self.selected_position}")
            
            # Record the movement in session history
            self.session_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "action": "goto_position",
                "position_name": self.selected_position,
                "target_position": position_data.copy()
            })
            
            logger.info(f"Successfully moved to {self.selected_position}")
            self.update_modifications_display()
            
        except Exception as e:
            logger.error(f"Error moving to position: {str(e)}")
            messagebox.showerror("Error", f"Failed to move to position:\n{str(e)}")
        finally:
            self.update_display()
    
    def store_original_position(self):
        """Store the current position as the original reference."""
        if self.is_connected:
            self.original_position = {
                "z_mm": self.current_z_position,
                "shoulder_rad": self.current_shoulder_angle, 
                "elbow_rad": self.current_elbow_angle,
                "gripper_rad": self.current_gripper_angle,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            
    def save_temporary_position(self):
        """Save current position as a temporary modification."""
        if not self.is_connected or not self.selected_position:
            messagebox.showwarning("Warning", "Connect robot and select a position first")
            return
            
        temp_position = {
            "z_mm": self.current_z_position,
            "shoulder_rad": self.current_shoulder_angle,
            "elbow_rad": self.current_elbow_angle,
            "gripper_rad": self.current_gripper_angle,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "original_position": self.selected_position
        }
        
        # Store with a unique name
        temp_name = f"{self.selected_position}_modified_{len(self.temporary_modifications)+1}"
        self.temporary_modifications[temp_name] = temp_position
        
        logger.info(f"Saved temporary position: {temp_name}")
        messagebox.showinfo("Success", f"Saved current position as: {temp_name}")
        self.update_modifications_display()
        
    def reset_to_original_position(self):
        """Reset robot to the original position before modifications."""
        if not self.original_position:
            messagebox.showwarning("Warning", "No original position stored")
            return
            
        try:
            # Move to original position
            self.robot.move_z(self.original_position["z_mm"], wait=True)
            self.robot.move_axis_rad(self.robot.SHOULDER, self.original_position["shoulder_rad"], wait=True)
            self.robot.move_axis_rad(self.robot.ELBOW, self.original_position["elbow_rad"], wait=True)
            self.robot.move_axis_rad(self.robot.GRIPPER, self.original_position["gripper_rad"], wait=True)
            
            logger.info("Reset to original position")
            messagebox.showinfo("Success", "Reset to original position")
            
        except Exception as e:
            logger.error(f"Error resetting position: {str(e)}")
            messagebox.showerror("Error", f"Failed to reset position:\n{str(e)}")
        finally:
            self.update_display()
            
    def clear_temporary_positions(self):
        """Clear all temporary position modifications."""
        self.temporary_modifications.clear()
        logger.info("Cleared all temporary positions")
        messagebox.showinfo("Success", "Cleared all temporary positions")
        self.update_modifications_display()
        
    def update_modifications_display(self):
        """Update the modifications display text widget."""
        self.modifications_display.config(state=tk.NORMAL)
        self.modifications_display.delete(1.0, tk.END)
        
        if not self.temporary_modifications:
            self.modifications_display.insert(tk.END, "No modifications yet. Select a position and make adjustments.")
        else:
            self.modifications_display.insert(tk.END, f"Temporary positions ({len(self.temporary_modifications)}):\n")
            for name, data in self.temporary_modifications.items():
                self.modifications_display.insert(tk.END, 
                    f"• {name} [{data['timestamp']}]: Z={data['z_mm']:.1f}mm, "
                    f"S={data['shoulder_rad']:.2f}rad, E={data['elbow_rad']:.2f}rad\n")
                
        self.modifications_display.config(state=tk.DISABLED)
        
    def export_temporary_positions_csv(self):
        """Export temporary positions to CSV file."""
        if not self.temporary_modifications:
            messagebox.showwarning("Warning", "No temporary positions to export")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Temporary Positions"
        )
        
        if filename:
            try:
                with open(filename, 'w', newline='') as csvfile:
                    fieldnames = ['position_name', 'original_position', 'z_mm', 'shoulder_rad', 
                                'elbow_rad', 'gripper_rad', 'timestamp']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for name, data in self.temporary_modifications.items():
                        writer.writerow({
                            'position_name': name,
                            'original_position': data['original_position'],
                            'z_mm': data['z_mm'],
                            'shoulder_rad': data['shoulder_rad'],
                            'elbow_rad': data['elbow_rad'],
                            'gripper_rad': data['gripper_rad'],
                            'timestamp': data['timestamp']
                        })
                
                logger.info(f"Exported temporary positions to {filename}")
                messagebox.showinfo("Success", f"Exported {len(self.temporary_modifications)} positions to {filename}")
                
            except Exception as e:
                logger.error(f"Error exporting CSV: {str(e)}")
                messagebox.showerror("Error", f"Failed to export CSV:\n{str(e)}")
                
    def export_session_history(self):
        """Export session history to text file."""
        if not self.session_history:
            messagebox.showwarning("Warning", "No session history to export")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Session History"
        )
        
        if filename:
            try:
                with open(filename, 'w') as txtfile:
                    txtfile.write(f"Robot Arm Position Control Session History\n")
                    txtfile.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    txtfile.write("="*60 + "\n\n")
                    
                    for entry in self.session_history:
                        txtfile.write(f"[{entry['timestamp']}] {entry['action']}\n")
                        if 'position_name' in entry:
                            txtfile.write(f"  Position: {entry['position_name']}\n")
                        if 'target_position' in entry:
                            pos = entry['target_position']
                            txtfile.write(f"  Target: Z={pos['z_mm']:.1f}mm, "
                                        f"Shoulder={pos['shoulder_rad']:.2f}rad, "
                                        f"Elbow={pos['elbow_rad']:.2f}rad, "
                                        f"Gripper={pos['gripper_rad']:.2f}rad\n")
                        txtfile.write("\n")
                
                logger.info(f"Exported session history to {filename}")
                messagebox.showinfo("Success", f"Exported session history to {filename}")
                
            except Exception as e:
                logger.error(f"Error exporting history: {str(e)}")
                messagebox.showerror("Error", f"Failed to export history:\n{str(e)}")
                
    def export_all_positions_json(self):
        """Export all position data (workflow + temporary) to JSON file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export All Position Data"
        )
        
        if filename:
            try:
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "workflow_positions": self.workflow_positions,
                    "temporary_modifications": self.temporary_modifications,
                    "session_history": self.session_history,
                    "robot_settings": {
                        "move_increment_mm": self.move_increment_mm,
                        "move_increment_rad": self.move_increment_rad,
                        "simulate_mode": SIMULATE
                    }
                }
                
                with open(filename, 'w') as jsonfile:
                    json.dump(export_data, jsonfile, indent=2, default=str)
                
                logger.info(f"Exported all position data to {filename}")
                messagebox.showinfo("Success", f"Exported complete position data to {filename}")
                
            except Exception as e:
                logger.error(f"Error exporting JSON: {str(e)}")
                messagebox.showerror("Error", f"Failed to export JSON:\n{str(e)}")
                
    def save_custom_position(self):
        """Save current position with user-defined name to temp folder."""
        try:
            # Get custom name from entry field
            custom_name = self.custom_position_name.get().strip()
            if not custom_name:
                messagebox.showwarning("Warning", "Please enter a position name")
                return
            
            # Sanitize the filename
            safe_filename = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            if not safe_filename:
                messagebox.showerror("Error", "Position name must contain alphanumeric characters")
                return
            
            # Create temp directory if it doesn't exist
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            # Get current position
            current_pos = self.get_current_positions()
            
            # Create position data
            position_data = {
                "name": custom_name,
                "timestamp": datetime.now().isoformat(),
                "gripper_cts": current_pos[0],
                "shoulder_cts": current_pos[1], 
                "elbow_cts": current_pos[2],
                "z_cts": current_pos[3],
                "description": f"User-defined position: {custom_name}",
                "category": "User_Defined"
            }
            
            # Save as JSON file
            filename = temp_dir / f"{safe_filename.replace(' ', '_')}.json"
            with open(filename, 'w') as f:
                json.dump(position_data, f, indent=2)
            
            # Also save coordinates in easy copy-paste format
            coords_filename = temp_dir / f"{safe_filename.replace(' ', '_')}_coordinates.txt"
            with open(coords_filename, 'w') as f:
                f.write(f"Position Name: {custom_name}\\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
                f.write("Robot Coordinates:\\n")
                f.write(f"gripper_cts: {current_pos[0]}\\n")
                f.write(f"shoulder_cts: {current_pos[1]}\\n") 
                f.write(f"elbow_cts: {current_pos[2]}\\n")
                f.write(f"z_cts: {current_pos[3]}\\n\\n")
                f.write("Array Format (for copy-paste):\\n")
                f.write(f"[{current_pos[0]}, {current_pos[1]}, {current_pos[2]}, {current_pos[3]}]\\n\\n")
                f.write("Python Dictionary Format:\\n")
                f.write(f"'{custom_name}': {{\\n")
                f.write(f"    'gripper_cts': {current_pos[0]},\\n")
                f.write(f"    'shoulder_cts': {current_pos[1]},\\n")
                f.write(f"    'elbow_cts': {current_pos[2]},\\n")
                f.write(f"    'z_cts': {current_pos[3]},\\n")
                f.write(f"    'description': '{custom_name}',\\n")
                f.write(f"    'category': 'User_Defined'\\n")
                f.write(f"}}\\n")
            
            logger.info(f"Saved custom position '{custom_name}' to temp folder")
            messagebox.showinfo("Success", f"Saved position '{custom_name}' to temp folder\\n\\nFiles created:\\n• {filename.name}\\n• {coords_filename.name}")
            
            # Update the custom name for next save
            base_name = custom_name.rstrip('0123456789_')
            if base_name != custom_name:
                # Extract number and increment
                num_part = custom_name[len(base_name):].lstrip('_')
                try:
                    next_num = int(num_part) + 1
                    self.custom_position_name.delete(0, tk.END)
                    self.custom_position_name.insert(0, f"{base_name}_{next_num}")
                except:
                    pass
            else:
                self.custom_position_name.delete(0, tk.END)
                self.custom_position_name.insert(0, f"{custom_name}_2")
                
        except Exception as e:
            logger.error(f"Error saving custom position: {str(e)}")
            messagebox.showerror("Error", f"Failed to save custom position:\\n{str(e)}")
    
    def show_saved_positions_window(self):
        """Show a window with all saved positions from temp folder."""
        try:
            temp_dir = Path("temp")
            if not temp_dir.exists():
                messagebox.showinfo("Info", "No saved positions found. Save some positions first!")
                return
            
            # Find all JSON position files
            json_files = list(temp_dir.glob("*.json"))
            if not json_files:
                messagebox.showinfo("Info", "No saved positions found in temp folder")
                return
            
            # Create new window
            saved_window = tk.Toplevel(self.root)
            saved_window.title("Saved Positions")
            saved_window.geometry("800x500")
            saved_window.transient(self.root)
            
            # Main frame
            main_frame = ttk.Frame(saved_window, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Title
            ttk.Label(main_frame, text="Saved Positions Manager", 
                     font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=3, pady=(0, 10))
            
            # Treeview for positions
            columns = ("Name", "Timestamp", "Coordinates")
            tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=15)
            tree.heading("Name", text="Position Name")
            tree.heading("Timestamp", text="Saved At")  
            tree.heading("Coordinates", text="Coordinates [G,S,E,Z]")
            
            tree.column("Name", width=200)
            tree.column("Timestamp", width=150)
            tree.column("Coordinates", width=300)
            
            tree.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
            
            # Scrollbar for treeview
            scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.grid(row=1, column=3, sticky=(tk.N, tk.S))
            
            # Load and display positions
            for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    name = data.get('name', json_file.stem)
                    timestamp = data.get('timestamp', 'Unknown')
                    if 'T' in timestamp:
                        timestamp = timestamp.split('T')[0] + ' ' + timestamp.split('T')[1].split('.')[0]
                    
                    coords = f"[{data.get('gripper_cts', 0)}, {data.get('shoulder_cts', 0)}, {data.get('elbow_cts', 0)}, {data.get('z_cts', 0)}]"
                    
                    tree.insert("", tk.END, values=(name, timestamp, coords))
                except Exception as e:
                    logger.warning(f"Could not load position file {json_file}: {e}")
            
            # Buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
            
            def copy_selected():
                selection = tree.selection()
                if selection:
                    item = tree.item(selection[0])
                    coords_text = item['values'][2]  # Coordinates column
                    saved_window.clipboard_clear()
                    saved_window.clipboard_append(coords_text)
                    messagebox.showinfo("Copied", f"Coordinates copied to clipboard:\\n{coords_text}")
                else:
                    messagebox.showwarning("Warning", "Please select a position first")
            
            def open_temp_folder():
                try:
                    import subprocess
                    subprocess.run(['explorer', str(temp_dir.resolve())])
                except Exception as e:
                    messagebox.showerror("Error", f"Could not open folder:\\n{e}")
            
            def refresh_list():
                # Clear and reload
                tree.delete(*tree.get_children())
                json_files = list(temp_dir.glob("*.json"))
                for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        name = data.get('name', json_file.stem)
                        timestamp = data.get('timestamp', 'Unknown')
                        if 'T' in timestamp:
                            timestamp = timestamp.split('T')[0] + ' ' + timestamp.split('T')[1].split('.')[0]
                        coords = f"[{data.get('gripper_cts', 0)}, {data.get('shoulder_cts', 0)}, {data.get('elbow_cts', 0)}, {data.get('z_cts', 0)}]"
                        tree.insert("", tk.END, values=(name, timestamp, coords))
                    except:
                        pass
            
            ttk.Button(button_frame, text="📋 Copy Coordinates", command=copy_selected).grid(row=0, column=0, padx=(0, 5))
            ttk.Button(button_frame, text="📂 Open Temp Folder", command=open_temp_folder).grid(row=0, column=1, padx=(0, 5))
            ttk.Button(button_frame, text="🔄 Refresh", command=refresh_list).grid(row=0, column=2, padx=(0, 5))
            ttk.Button(button_frame, text="Close", command=saved_window.destroy).grid(row=0, column=3)
            
            # Configure grid weights
            saved_window.columnconfigure(0, weight=1)
            saved_window.rowconfigure(0, weight=1)
            main_frame.columnconfigure(0, weight=1)
            main_frame.rowconfigure(1, weight=1)
            
        except Exception as e:
            logger.error(f"Error showing saved positions: {str(e)}")
            messagebox.showerror("Error", f"Failed to show saved positions:\\n{str(e)}")
    
    def export_saved_positions_csv(self):
        """Export all saved positions to a CSV file."""
        try:
            temp_dir = Path("temp")
            if not temp_dir.exists() or not list(temp_dir.glob("*.json")):
                messagebox.showinfo("Info", "No saved positions found to export")
                return
            
            filename = filedialog.asksaveasfilename(
                title="Export Saved Positions as CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialname=f"saved_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            if filename:
                with open(filename, 'w', newline='') as csvfile:
                    fieldnames = ['name', 'timestamp', 'gripper_cts', 'shoulder_cts', 'elbow_cts', 'z_cts', 'description', 'category']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    json_files = list(temp_dir.glob("*.json"))
                    for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                            writer.writerow(data)
                        except Exception as e:
                            logger.warning(f"Could not export position file {json_file}: {e}")
                
                logger.info(f"Exported saved positions to {filename}")
                messagebox.showinfo("Success", f"Exported saved positions to\\n{filename}")
                
        except Exception as e:
            logger.error(f"Error exporting saved positions CSV: {str(e)}")
            messagebox.showerror("Error", f"Failed to export CSV:\\n{str(e)}")
    
    def copy_current_position(self):
        """Copy current position coordinates to clipboard."""
        try:
            current_pos = self.get_current_positions() 
            coords_text = f"[{current_pos[0]}, {current_pos[1]}, {current_pos[2]}, {current_pos[3]}]"
            
            self.root.clipboard_clear()
            self.root.clipboard_append(coords_text)
            
            messagebox.showinfo("Copied", f"Current position coordinates copied to clipboard:\\n{coords_text}")
            logger.info(f"Copied coordinates to clipboard: {coords_text}")
            
        except Exception as e:
            logger.error(f"Error copying position: {str(e)}")
            messagebox.showerror("Error", f"Failed to copy position:\\n{str(e)}")
    
    def open_temp_folder(self):
        """Open the temp folder in file explorer."""
        try:
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)  # Create if doesn't exist
            
            import subprocess
            subprocess.run(['explorer', str(temp_dir.resolve())])
            
        except Exception as e:
            logger.error(f"Error opening temp folder: {str(e)}")
            messagebox.showerror("Error", f"Failed to open temp folder:\\n{str(e)}")
                
    # Movement methods (simplified from original)
    def move_up(self):
        """Move Z-axis up."""
        self._safe_move_z(self.current_z_position + self.move_increment_mm)
        
    def move_down(self):
        """Move Z-axis down."""
        self._safe_move_z(self.current_z_position - self.move_increment_mm)
        
    def move_x_left(self):
        """Move X-axis left by moving shoulder joint independently."""
        # Convert mm step size to shoulder angle increment
        angle_increment = self._convert_mm_to_joint_increment()
        target_angle = self.current_shoulder_angle - angle_increment
        self._safe_move_shoulder(target_angle)
        
    def move_x_right(self):
        """Move X-axis right by moving shoulder joint independently."""
        # Convert mm step size to shoulder angle increment  
        angle_increment = self._convert_mm_to_joint_increment()
        target_angle = self.current_shoulder_angle + angle_increment
        self._safe_move_shoulder(target_angle)
        
    def move_y_back(self):
        """Move Y-axis back by moving elbow joint independently."""
        # Convert mm step size to elbow angle increment
        angle_increment = self._convert_mm_to_joint_increment()
        target_angle = self.current_elbow_angle - angle_increment
        self._safe_move_elbow(target_angle)
        
    def move_y_forward(self):
        """Move Y-axis forward by moving elbow joint independently."""
        # Convert mm step size to elbow angle increment
        angle_increment = self._convert_mm_to_joint_increment()
        target_angle = self.current_elbow_angle + angle_increment
        self._safe_move_elbow(target_angle)
    
    def _convert_mm_to_joint_increment(self):
        """Convert Z-axis mm step size to equivalent joint angle increment."""
        # Approximate conversion: 1mm ≈ 0.01 radians for intuitive feel
        # User can adjust this by changing the Z-axis step size
        return self.move_increment_mm * 0.01
        
    def _safe_move_z(self, target_mm):
        """Safely move Z-axis with bounds checking."""
        if not self.is_connected:
            return
            
        target_mm = max(Z_AXIS_MIN_MM, min(Z_AXIS_MAX_MM, target_mm))
        try:
            self.robot.move_z(target_mm, wait=True)
            self.update_display()
        except Exception as e:
            logger.error(f"Error moving Z-axis: {str(e)}")
            
    def _safe_move_shoulder(self, target_angle):
        """Safely move shoulder joint and update X-Y coordinates."""
        if not self.is_connected:
            return
            
        # Apply joint limits for shoulder
        SHOULDER_MIN, SHOULDER_MAX = -(2/3) * 3.14159, (2/3) * 3.14159
        target_angle = max(SHOULDER_MIN, min(SHOULDER_MAX, target_angle))
        
        try:
            logger.info(f"Moving shoulder to {target_angle:.3f} rad (X-axis control)")
            self.robot.move_axis_rad(self.robot.SHOULDER, target_angle, wait=True)
            self.current_shoulder_angle = target_angle
            self._update_xy_from_joints()  # Calculate new X-Y position
            self.update_display()
        except Exception as e:
            logger.error(f"Error moving shoulder: {str(e)}")
            
    def _safe_move_elbow(self, target_angle):
        """Safely move elbow joint and update X-Y coordinates."""
        if not self.is_connected:
            return
            
        # Apply joint limits for elbow
        ELBOW_MIN, ELBOW_MAX = -(5/6) * 3.14159, (5/6) * 3.14159
        target_angle = max(ELBOW_MIN, min(ELBOW_MAX, target_angle))
        
        try:
            logger.info(f"Moving elbow to {target_angle:.3f} rad (Y-axis control)")
            self.robot.move_axis_rad(self.robot.ELBOW, target_angle, wait=True)
            self.current_elbow_angle = target_angle
            self._update_xy_from_joints()  # Calculate new X-Y position
            self.update_display()
        except Exception as e:
            logger.error(f"Error moving elbow: {str(e)}")
            
    def _update_xy_from_joints(self):
        """Update X-Y coordinates from current joint angles using forward kinematics."""
        try:
            if hasattr(self.robot, 'n9_fk'):  # Real robot with kinematics
                # Convert current angles to counts for forward kinematics
                gripper_cts = self.robot.rad_to_counts(self.robot.GRIPPER, self.current_gripper_angle)
                elbow_cts = self.robot.rad_to_counts(self.robot.ELBOW, self.current_elbow_angle) 
                shoulder_cts = self.robot.rad_to_counts(self.robot.SHOULDER, self.current_shoulder_angle)
                
                # Calculate X-Y coordinates
                x, y, theta = self.robot.n9_fk(gripper_cts, elbow_cts, shoulder_cts)
                self.current_x_position = x
                self.current_y_position = y
                logger.info(f"Joint movement resulted in X-Y: ({x:.1f}, {y:.1f}) mm")
            else:  # Mock robot - approximate X-Y from joint angles
                # Simple approximation for display purposes
                self.current_x_position = 150 + (self.current_shoulder_angle * 50)  # Approximate
                self.current_y_position = 100 + (self.current_elbow_angle * 50)     # Approximate
        except Exception as e:
            logger.warning(f"Could not update X-Y coordinates: {e}")
            
    def open_gripper(self):
        """Open gripper."""
        if self.is_connected:
            try:
                self.robot.open_gripper()
                self.gripper_is_open = True
                self.update_display()
            except Exception as e:
                logger.error(f"Error opening gripper: {str(e)}")
                
    def close_gripper(self):
        """Close gripper."""
        if self.is_connected:
            try:
                self.robot.close_gripper()
                self.gripper_is_open = False
                self.update_display()
            except Exception as e:
                logger.error(f"Error closing gripper: {str(e)}")
                
    def home_robot(self):
        """Home the robot."""
        if not self.is_connected:
            messagebox.showerror("Error", "Robot not connected")
            return
            
        try:
            logger.info("Homing robot...")
            self.home_button.config(text="Homing...", state="disabled")
            self.root.update()
            
            self.robot.home_robot(wait=True)
            self.is_homed = True
            self.gripper_is_open = False
            
            # Record homing in session history
            self.session_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "action": "home_robot",
                "target_position": {
                    "z_mm": Z_AXIS_MAX_MM,
                    "shoulder_rad": 0.0,
                    "elbow_rad": 0.0, 
                    "gripper_rad": 0.0
                }
            })
            
            logger.info("Robot homed successfully")
            messagebox.showinfo("Success", "Robot homed successfully!")
            
        except Exception as e:
            logger.error(f"Error homing robot: {str(e)}")
            messagebox.showerror("Error", f"Failed to home robot:\n{str(e)}")
        finally:
            self.home_button.config(text="🏠 HOME ROBOT", state="normal")
            self.update_display()
            
    def update_z_increment(self):
        """Update movement increment for both Z-axis and X-Y movement."""
        try:
            value = float(self.z_increment_entry.get())
            if value < MIN_MOVE_INCREMENT_MM or value > MAX_MOVE_INCREMENT_MM:
                self.increment_status_label.config(
                    text=f"Movement increment out of range [{MIN_MOVE_INCREMENT_MM}-{MAX_MOVE_INCREMENT_MM}]mm", 
                    foreground="red")
                return
            
            self.move_increment_mm = value
            self.increment_status_label.config(text=f"Movement step set to {value:.1f}mm (Z-axis & X-Y)", foreground="green")
            logger.info(f"Movement increment updated to {value:.1f}mm for Z-axis and X-Y movement")
            
        except ValueError:
            self.increment_status_label.config(text="Invalid movement increment value", foreground="red")
            
    def update_rad_increment(self):
        """Update rotational movement increment."""
        try:
            value = float(self.rad_increment_entry.get())
            if value < MIN_MOVE_INCREMENT_RAD or value > MAX_MOVE_INCREMENT_RAD:
                self.increment_status_label.config(
                    text=f"Rotation increment out of range [{MIN_MOVE_INCREMENT_RAD}-{MAX_MOVE_INCREMENT_RAD}]rad", 
                    foreground="red")
                return
            
            self.move_increment_rad = value
            self.increment_status_label.config(text=f"Rotation step set to {value:.3f}rad", foreground="green")
            logger.info(f"Rotational movement increment updated to {value:.3f}rad")
            
        except ValueError:
            self.increment_status_label.config(text="Invalid rotation increment value", foreground="red")
            
    def update_display(self):
        """Update position and status display."""
        if self.is_connected:
            try:
                # Get current positions
                if hasattr(self.robot, 'z_position_mm'):  # Mock robot
                    self.current_z_position = self.robot.z_position_mm
                    self.current_gripper_angle = self.robot.gripper_angle_rad
                    self.gripper_is_open = self.robot.gripper_is_open
                    self.is_homed = self.robot.is_homed
                    
                    # Use forward kinematics for consistent coordinate calculation
                    x, y, _ = self.robot.n9_fk(self.robot.gripper_cts, self.robot.elbow_cts, self.robot.shoulder_cts)
                    self.current_x_position = x
                    self.current_y_position = y
                    
                else:  # Real robot
                    positions = self.robot.get_robot_positions()
                    self.current_gripper_angle = self.robot.counts_to_rad(self.robot.GRIPPER, positions[0])
                    self.current_z_position = self.robot.counts_to_mm(self.robot.Z_AXIS, positions[3])
                    # Convert joint positions to x-y coordinates using forward kinematics
                    try:
                        cartesian_pos = self.robot.n9_fk(positions[0], positions[2], positions[1])  # gripper, elbow, shoulder order
                        self.current_x_position = cartesian_pos[0]
                        self.current_y_position = cartesian_pos[1]
                    except Exception as e:
                        logger.warning(f"Could not get x-y position from kinematics during display update: {e}")
                        # Keep current values if FK fails
                
                # Update position labels
                self.z_position_label.config(text=f"Z-Axis: {self.current_z_position:.1f} mm")
                self.x_position_label.config(text=f"X-Position: {self.current_x_position:.1f} mm")
                self.y_position_label.config(text=f"Y-Position: {self.current_y_position:.1f} mm")
                self.gripper_position_label.config(text=f"Gripper: {self.current_gripper_angle:.2f} rad ({self.current_gripper_angle*180/3.14159:.1f}°)")
                
                # Update gripper status
                if self.gripper_is_open:
                    self.gripper_status_label.config(text="Gripper: OPEN ✋", foreground="green")
                else:
                    self.gripper_status_label.config(text="Gripper: CLOSED 👊", foreground="red")
                
                # Update connection status
                if self.is_homed:
                    self.status_label.config(text="Connected & Homed", foreground="green")
                else:
                    self.status_label.config(text="Connected - Not Homed", foreground="orange")
                    
            except Exception as e:
                logger.error(f"Error updating display: {str(e)}")
                self.status_label.config(text="Error reading position", foreground="red")
        else:
            # Disconnected state
            self.z_position_label.config(text="Z-Axis: Unknown")
            self.x_position_label.config(text="X-Position: Unknown")
            self.y_position_label.config(text="Y-Position: Unknown")
            self.gripper_position_label.config(text="Gripper: Unknown")
            self.gripper_status_label.config(text="Gripper: Unknown", foreground="gray")
            self.status_label.config(text="Disconnected", foreground="red")
    
    def get_current_positions(self):
        """Get current robot positions in counts format [gripper, shoulder, elbow, z]."""
        try:
            if not self.is_connected:
                # Return default/last known positions if not connected
                if hasattr(self, 'robot') and hasattr(self.robot, 'gripper_cts'):  # Mock robot
                    return [
                        self.robot.gripper_cts,
                        self.robot.shoulder_cts, 
                        self.robot.elbow_cts,
                        self.robot.z_cts
                    ]
                else:
                    # Default positions if nothing available
                    return [0, 0, 0, 0]
            
            if hasattr(self.robot, 'z_position_mm'):  # Mock robot
                return [
                    self.robot.gripper_cts,
                    self.robot.shoulder_cts,
                    self.robot.elbow_cts, 
                    self.robot.z_cts
                ]
            else:  # Real robot
                positions = self.robot.get_robot_positions()
                return [
                    positions[0],  # gripper_cts
                    positions[2],  # shoulder_cts (note: real robot returns [gripper, elbow, shoulder, z])
                    positions[1],  # elbow_cts
                    positions[3]   # z_cts
                ]
                
        except Exception as e:
            logger.error(f"Error getting current positions: {str(e)}")
            # Return default positions on error
            return [0, 0, 0, 0]
    
    def on_key_press(self, event):
        """Handle keyboard input for robot control."""
        if not self.is_connected:
            return
            
        key = event.keysym
        
        # Z-axis movement
        if key == 'Up':
            self.move_up()
        elif key == 'Down':
            self.move_down()
        # X-Y coordinate movement
        elif key == 'Left':
            self.move_x_left()
        elif key == 'Right':
            self.move_x_right()
        # Y-axis movement using W/S keys
        elif key.lower() == 'w':
            self.move_y_forward()
        elif key.lower() == 's':
            self.move_y_back()
        # Gripper control
        elif key.lower() == 'o':
            self.open_gripper()
        elif key.lower() == 'c':
            self.close_gripper()
        # Home robot
        elif key == 'space':
            self.home_robot()
        # Exit
        elif key == 'Escape':
            self.on_closing()
    
    def on_closing(self):
        """Handle window close event."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            logger.info("Application closing")
            self.root.destroy()
    
    def run(self):
        """Main application loop."""
        # Connect to robot
        if not self.connect_robot():
            logger.error("Failed to connect to robot. Exiting.")
            return
        
        # Create and run GUI
        self.create_gui()
        self.update_display()
        
        logger.info("Enhanced Robot Arm Controller started")
        self.root.mainloop()


def main():
    """Main entry point with configuration options."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_robot_arm_controller.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize with optional vial file for full integration
    default_vial_file = "status/surfactant_grid_vials_expanded.csv"
    if not os.path.exists(default_vial_file):
        default_vial_file = "../status/surfactant_grid_vials_expanded.csv"
        if not os.path.exists(default_vial_file):
            default_vial_file = "../utoronto_demo/status/surfactant_grid_vials_expanded.csv"
    vial_file = default_vial_file if os.path.exists(default_vial_file) else None
    
    try:
        controller = EnhancedRobotArmController(
            vial_file_path=vial_file,
            include_pipetting=True  # Include both gripper and pipetting positions by default
        )
        controller.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        messagebox.showerror("Application Error", f"An error occurred:\n{str(e)}")


if __name__ == "__main__":
    main()
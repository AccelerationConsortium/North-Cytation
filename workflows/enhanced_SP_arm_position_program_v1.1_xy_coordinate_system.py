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

try:
    from master_usdl_coordinator import Lash_E
    LASHE_AVAILABLE = True
except ImportError:
    LASHE_AVAILABLE = False

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

def load_positions_from_system():
    """Load all positions directly from robot_state/Locator.py.
    
    Single positions (flat list of 4) are keyed by variable name.
    Array positions (list of lists) are keyed as name[index].
    category = variable name for arrays, 'single' for standalone positions.
    """
    import inspect
    positions = {}
    
    try:
        import robot_state.Locator as Locator

        for name, value in inspect.getmembers(Locator):
            if name.startswith('_') or not isinstance(value, list) or len(value) == 0:
                continue

            # Single position: flat list of 4 numbers
            if len(value) == 4 and not isinstance(value[0], list):
                positions[name] = {
                    "description": name,
                    "gripper_cts": int(value[0]),
                    "elbow_cts":   int(value[1]),  # API index 1 = elbow
                    "shoulder_cts": int(value[2]), # API index 2 = shoulder
                    "z_cts": int(value[3]),
                    "category": "single",
                    "locator_var": name,
                    "locator_index": None,
                }

            # Array of positions: list of lists
            elif isinstance(value[0], list):
                for i, coords in enumerate(value):
                    if isinstance(coords, list) and len(coords) == 4:
                        positions[f"{name}[{i}]"] = {
                            "description": f"{name} index {i}",
                            "gripper_cts":  int(coords[0]),
                            "elbow_cts":    int(coords[1]),  # API index 1 = elbow
                            "shoulder_cts": int(coords[2]),  # API index 2 = shoulder
                            "z_cts": int(coords[3]),
                            "category": name,
                            "locator_var": name,
                            "locator_index": i,
                        }

    except Exception as e:
        logger.warning(f"Could not load positions from Locator: {e}")
        positions["home"] = {
            "description": "Home position",
            "gripper_cts": 0, "shoulder_cts": 0, "elbow_cts": 0, "z_cts": 0,
            "category": "single", "locator_var": "home", "locator_index": None,
        }

    return positions

# Initialize with default positions - will be updated by controller
WORKFLOW_POSITIONS = load_positions_from_system()

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
    
    def __init__(self, vial_file_path=None):
        self.lash_e = None      # Lash_E coordinator (owns all hardware)
        self.nr_robot = None    # lash_e.nr_robot  - high-level robot ops (get_pipet, move_vial, etc.)
        self.robot = None       # lash_e.nr_robot.c9 - raw NorthC9 (goto, n9_fk/ik, move_z, etc.)
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
        self.workflow_positions = load_positions_from_system()
        
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
        
        # Grid array generator state
        self.grid_origin = None  # {x, y, z_cts, gripper_cts, shoulder_cts}
        self.grid_configs = self._load_grid_configs()  # from vial_positions.yaml
        
    def _load_grid_configs(self):
        """Load grid_params from vial_positions.yaml for all objects that have them."""
        import yaml
        configs = {}
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'robot_state', 'vial_positions.yaml')
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            for obj_name, obj_data in data.items():
                gp = obj_data.get('grid_params', {})
                if gp and gp.get('num_cols') is not None:
                    configs[obj_name] = {
                        'num_cols':   gp.get('num_cols'),
                        'num_rows':   gp.get('num_rows'),
                        'x_delta_mm': gp.get('x_delta_mm'),
                        'y_delta_mm': gp.get('y_delta_mm'),
                    }
        except Exception as e:
            logger.warning(f"Could not load grid configs from vial_positions.yaml: {e}")
        return configs

    def connect_robot(self):
        """Initialize Lash_E coordinator and wire up robot references.
        
        self.lash_e   - full Lash_E coordinator (high-level ops, track, cytation, etc.)
        self.nr_robot - lash_e.nr_robot (get_pipet, remove_pipet, move_vial_to_location, aspirate, etc.)
        self.robot    - lash_e.nr_robot.c9 (raw NorthC9: goto, n9_fk/ik, move_z, counts, etc.)
        """
        try:
            if SIMULATE:
                logger.info("Starting in simulation mode via Lash_E")
                if LASHE_AVAILABLE:
                    self.lash_e = Lash_E(self.vial_file_path, simulate=True)
                    self.nr_robot = self.lash_e.nr_robot
                else:
                    logger.warning("Lash_E not available - nr_robot will be None in simulation")
                    self.lash_e = None
                    self.nr_robot = None
                self.robot = self._create_mock_robot()
                self.is_connected = True
                self.current_z_position = 100.0
                return True
            else:
                logger.info(f"Connecting via Lash_E (serial {NETWORK_SERIAL})")
                self.lash_e = Lash_E(
                    self.vial_file_path,
                    simulate=False
                )
                self.nr_robot = self.lash_e.nr_robot
                self.robot = self.lash_e.nr_robot.c9  # raw NorthC9 alias

                self.robot.get_info()
                self.is_connected = True
                logger.info("Successfully connected via Lash_E")

                # Get current positions for all axes
                positions = self.robot.get_robot_positions()
                self.current_gripper_angle  = self.robot.counts_to_rad(self.robot.GRIPPER,   positions[0])
                self.current_elbow_angle    = self.robot.counts_to_rad(self.robot.ELBOW,     positions[1])  # index 1 = elbow
                self.current_shoulder_angle = self.robot.counts_to_rad(self.robot.SHOULDER,  positions[2])  # index 2 = shoulder
                self.current_z_position     = self.robot.counts_to_mm(self.robot.Z_AXIS,     positions[3])

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
                
            def goto_safe(self, position_list, wait=True):
                """Move to position safely (same as goto in simulation)."""
                logger.info(f"SIMULATION: Moving to position safely {position_list}")
                self.goto(position_list, wait)  # Just use regular goto
                
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
                
            def open_clamp(self):
                logger.info("SIMULATION: Opening clamp")
                
            def close_clamp(self):
                logger.info("SIMULATION: Closing clamp")
                
            def goto(self, position_list, wait=True):
                """Move to absolute position [gripper, elbow, shoulder, z] per API."""
                logger.info(f"SIMULATION: Moving to position {position_list}")
                time.sleep(0.5)
                self.gripper_angle_rad = position_list[0] / 1000.0  # index 0 = gripper
                self.elbow_angle_rad   = position_list[1] / 1000.0  # index 1 = elbow
                self.shoulder_angle_rad = position_list[2] / 1000.0 # index 2 = shoulder
                self.z_position_mm     = position_list[3] / 100.0   # index 3 = z
                
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
                """Return [gripper, elbow, shoulder, z] per API (same order as goto)."""
                return [
                    int(self.gripper_angle_rad * 1000),   # index 0 = gripper
                    int(self.elbow_angle_rad * 1000),     # index 1 = elbow
                    int(self.shoulder_angle_rad * 1000),  # index 2 = shoulder
                    int(self.z_position_mm * 100)         # index 3 = z
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
        self.root.title("North Robot Arm Position Control")
        self.root.geometry("1100x740")
        self.root.resizable(True, True)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # ── Title (full width) ──────────────────────────────────────────────
        ttk.Label(main_frame, text="North Robot Arm Position Control",
                  font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 8))

        # ── Left column ────────────────────────────────────────────────────
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky=(tk.N, tk.W, tk.E), padx=(0, 8))
        left_frame.columnconfigure(0, weight=1)

        # Right column
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.N, tk.W, tk.E))
        right_frame.columnconfigure(0, weight=1)

        # ── LEFT: Connection Status ─────────────────────────────────────────
        status_frame = ttk.LabelFrame(left_frame, text="Connection Status", padding="8")
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        self.status_label = ttk.Label(status_frame, text="Disconnected", foreground="red")
        self.status_label.pack(anchor=tk.W)

        # ── LEFT: Position Selection (2 dropdowns) ─────────────────────────
        pos_sel_frame = ttk.LabelFrame(left_frame, text="Position Selection", padding="8")
        pos_sel_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        # Build position groups: locator_var -> [full_key, ...]
        self.position_groups = {}
        for key, data in self.workflow_positions.items():
            var = data['locator_var']
            self.position_groups.setdefault(var, []).append(key)
        obj_names = sorted(self.position_groups.keys())

        obj_row = ttk.Frame(pos_sel_frame)
        obj_row.pack(fill=tk.X, pady=2)
        ttk.Label(obj_row, text="Object:", width=7).pack(side=tk.LEFT)
        self.position_object_dropdown = ttk.Combobox(obj_row, values=obj_names,
                                                      state="readonly", width=22)
        self.position_object_dropdown.pack(side=tk.LEFT, padx=(4, 0))
        self.position_object_dropdown.bind('<<ComboboxSelected>>',
                                           self.on_position_object_selected)

        idx_row = ttk.Frame(pos_sel_frame)
        idx_row.pack(fill=tk.X, pady=2)
        ttk.Label(idx_row, text="Index:", width=7).pack(side=tk.LEFT)
        self.position_dropdown = ttk.Combobox(idx_row, values=[], state="readonly", width=22)
        self.position_dropdown.pack(side=tk.LEFT, padx=(4, 6))
        self.position_dropdown.bind('<<ComboboxSelected>>', self.on_position_selected)
        ttk.Button(idx_row, text="Go To",
                   command=self.goto_selected_position).pack(side=tk.LEFT)

        self.position_description_label = ttk.Label(pos_sel_frame,
            text="Select an object then an index",
            font=("Arial", 9), foreground="gray")
        self.position_description_label.pack(anchor=tk.W, pady=(4, 0))

        # ── LEFT: Current Position ──────────────────────────────────────────
        pos_frame = ttk.LabelFrame(left_frame, text="Current Position", padding="8")
        pos_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        self.z_position_label = ttk.Label(pos_frame, text="Z: Unknown")
        self.z_position_label.grid(row=0, column=0, sticky=tk.W)
        self.shoulder_position_label = ttk.Label(pos_frame, text="Shoulder: Unknown")
        self.shoulder_position_label.grid(row=0, column=1, sticky=tk.W, padx=(12, 0))
        self.elbow_position_label = ttk.Label(pos_frame, text="Elbow: Unknown")
        self.elbow_position_label.grid(row=1, column=0, sticky=tk.W)
        self.gripper_position_label = ttk.Label(pos_frame, text="Gripper: Unknown")
        self.gripper_position_label.grid(row=1, column=1, sticky=tk.W, padx=(12, 0))
        self.gripper_status_label = ttk.Label(pos_frame, text="Gripper: Closed",
                                              font=("Arial", 9, "bold"))
        self.gripper_status_label.grid(row=2, column=0, columnspan=2, sticky=tk.W)

        # ── LEFT: Robot Controls ────────────────────────────────────────────
        control_frame = ttk.LabelFrame(left_frame, text="Robot Controls", padding="8")
        control_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        self.home_button = ttk.Button(control_frame, text="HOME ROBOT",
                                      command=self.home_robot, style="Accent.TButton")
        self.home_button.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 6))
        
        ttk.Button(control_frame, text="MOVE HOME",
                   command=self.move_home, style="Warning.TButton").grid(row=0, column=2, sticky=(tk.W, tk.E), pady=(0, 6), padx=(4, 0))

        ttk.Label(control_frame, text="Z:").grid(row=1, column=0, sticky=tk.W)
        ttk.Button(control_frame, text="▲ UP",
                   command=self.move_up).grid(row=1, column=1, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(control_frame, text="▼ DOWN",
                   command=self.move_down).grid(row=1, column=2, padx=2, sticky=(tk.W, tk.E))

        ttk.Label(control_frame, text="X:").grid(row=2, column=0, sticky=tk.W)
        ttk.Button(control_frame, text="← -X",
                   command=self.move_x_left).grid(row=2, column=1, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(control_frame, text="→ +X",
                   command=self.move_x_right).grid(row=2, column=2, padx=2, sticky=(tk.W, tk.E))

        ttk.Label(control_frame, text="Y:").grid(row=3, column=0, sticky=tk.W)
        ttk.Button(control_frame, text="↓ -Y",
                   command=self.move_y_back).grid(row=3, column=1, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(control_frame, text="↑ +Y",
                   command=self.move_y_forward).grid(row=3, column=2, padx=2, sticky=(tk.W, tk.E))

        ttk.Label(control_frame, text="Grip:").grid(row=4, column=0, sticky=tk.W)
        ttk.Button(control_frame, text="✋ OPEN", command=self.open_gripper,
                   style="Success.TButton").grid(row=4, column=1, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(control_frame, text="👊 CLOSE", command=self.close_gripper,
                   style="Warning.TButton").grid(row=4, column=2, padx=2, sticky=(tk.W, tk.E))
        
        ttk.Label(control_frame, text="Clamp:").grid(row=5, column=0, sticky=tk.W)
        ttk.Button(control_frame, text="🔓 OPEN", command=self.open_clamp,
                   style="Success.TButton").grid(row=5, column=1, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(control_frame, text="🔒 CLOSE", command=self.close_clamp,
                   style="Warning.TButton").grid(row=5, column=2, padx=2, sticky=(tk.W, tk.E))

        ttk.Label(control_frame, text="Pipet:").grid(row=6, column=0, sticky=tk.W)
        ttk.Button(control_frame, text="GET LARGE",
                   command=self.get_pipet_large).grid(row=6, column=1, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(control_frame, text="GET SMALL",
                   command=self.get_pipet_small).grid(row=6, column=2, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(control_frame, text="REMOVE PIPET",
                   command=self.remove_pipet_action).grid(
            row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(2, 0))

        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)

        # ── RIGHT: Position Modifications ──────────────────────────────────
        modifications_frame = ttk.LabelFrame(right_frame, text="Position Modifications", padding="8")
        modifications_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        self.modifications_display = tk.Text(modifications_frame, height=4, width=46,
                                             font=("Consolas", 8))
        self.modifications_display.pack(fill=tk.X, pady=(0, 5))
        self.modifications_display.insert(tk.END,
            "No modifications yet. Select a position and make adjustments.")
        self.modifications_display.config(state=tk.DISABLED)

        name_row = ttk.Frame(modifications_frame)
        name_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(name_row, text="Name:").pack(side=tk.LEFT)
        self.custom_position_name = ttk.Entry(name_row, width=24)
        self.custom_position_name.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)
        self.custom_position_name.insert(0, "my_position_1")

        btn_row1 = ttk.Frame(modifications_frame)
        btn_row1.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row1, text="Save w/ Name",
                   command=self.save_custom_position).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row1, text="Save Temp",
                   command=self.save_temporary_position).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row1, text="Reset",
                   command=self.reset_to_original_position).pack(side=tk.LEFT)

        btn_row2 = ttk.Frame(modifications_frame)
        btn_row2.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row2, text="View Saved",
                   command=self.show_saved_positions_window).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row2, text="Clear Temp",
                   command=self.clear_temporary_positions).pack(side=tk.LEFT)

        # ── RIGHT: Movement Settings ────────────────────────────────────────
        increment_frame = ttk.LabelFrame(right_frame, text="Movement Settings", padding="8")
        increment_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        step_row = ttk.Frame(increment_frame)
        step_row.pack(fill=tk.X, pady=2)
        ttk.Label(step_row, text="Step (mm):").pack(side=tk.LEFT)
        self.z_increment_entry = ttk.Entry(step_row, width=7)
        self.z_increment_entry.pack(side=tk.LEFT, padx=(6, 4))
        self.z_increment_entry.insert(0, str(DEFAULT_MOVE_INCREMENT_MM))
        ttk.Button(step_row, text="Set", command=self.update_z_increment).pack(side=tk.LEFT)

        rot_row = ttk.Frame(increment_frame)
        rot_row.pack(fill=tk.X, pady=2)
        ttk.Label(rot_row, text="Rot (rad):").pack(side=tk.LEFT)
        self.rad_increment_entry = ttk.Entry(rot_row, width=7)
        self.rad_increment_entry.pack(side=tk.LEFT, padx=(6, 4))
        self.rad_increment_entry.insert(0, str(DEFAULT_MOVE_INCREMENT_RAD))
        ttk.Button(rot_row, text="Set", command=self.update_rad_increment).pack(side=tk.LEFT)

        self.increment_status_label = ttk.Label(increment_frame, text="Ready",
                                                foreground="green")
        self.increment_status_label.pack(anchor=tk.W, pady=(4, 0))

        # ── RIGHT: Export ───────────────────────────────────────────────────
        export_frame = ttk.LabelFrame(right_frame, text="Export Position Data", padding="8")
        export_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        ttk.Button(export_frame, text="Export CSV",
                   command=self.export_saved_positions_csv).grid(row=0, column=0, padx=(0, 4), pady=2)
        ttk.Button(export_frame, text="Copy Coords",
                   command=self.copy_current_position).grid(row=0, column=1, padx=(0, 4), pady=2)
        ttk.Button(export_frame, text="Export History",
                   command=self.export_session_history).grid(row=0, column=2, pady=2)
        ttk.Button(export_frame, text="Open Temp Folder",
                   command=self.open_temp_folder).grid(row=1, column=0, padx=(0, 4), pady=2)
        ttk.Button(export_frame, text="Export JSON",
                   command=self.export_all_positions_json).grid(row=1, column=1, pady=2)

        # ── BOTTOM: Grid Array Generator (full width) ───────────────────────
        grid_gen_frame = ttk.LabelFrame(main_frame, text="Grid Array Generator", padding="10")
        grid_gen_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(8, 0))

        self.grid_origin_label = ttk.Label(grid_gen_frame, text="Origin: not set",
                                           foreground="gray")
        self.grid_origin_label.grid(row=0, column=0, columnspan=7, sticky=tk.W, pady=(0, 5))

        ttk.Label(grid_gen_frame, text="Load config:").grid(row=1, column=0, sticky=tk.W)
        obj_names_grid = [n for n, c in self.grid_configs.items() if c['num_cols'] is not None]
        self.grid_object_dropdown = ttk.Combobox(grid_gen_frame, values=obj_names_grid,
                                                  state="readonly", width=20)
        self.grid_object_dropdown.grid(row=1, column=1, padx=(4, 16))
        self.grid_object_dropdown.bind('<<ComboboxSelected>>', self.on_grid_object_selected)

        ttk.Button(grid_gen_frame, text="Set Origin (Current Position)",
                   command=self.set_grid_origin).grid(row=1, column=2, columnspan=5,
                   sticky=(tk.W, tk.E), padx=(0, 0))

        ttk.Label(grid_gen_frame, text="X offset (mm):").grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
        self.grid_x_offset_entry = ttk.Entry(grid_gen_frame, width=7)
        self.grid_x_offset_entry.grid(row=2, column=1, padx=(4, 16), pady=(6, 0))
        self.grid_x_offset_entry.insert(0, "9.0")

        ttk.Label(grid_gen_frame, text="Y offset (mm):").grid(row=2, column=2, sticky=tk.W, pady=(6, 0))
        self.grid_y_offset_entry = ttk.Entry(grid_gen_frame, width=7)
        self.grid_y_offset_entry.grid(row=2, column=3, padx=(4, 16), pady=(6, 0))
        self.grid_y_offset_entry.insert(0, "9.0")

        ttk.Label(grid_gen_frame, text="Cols:").grid(row=2, column=4, sticky=tk.W, pady=(6, 0))
        self.grid_cols_entry = ttk.Entry(grid_gen_frame, width=5)
        self.grid_cols_entry.grid(row=2, column=5, padx=(4, 12), pady=(6, 0))
        self.grid_cols_entry.insert(0, "8")

        ttk.Label(grid_gen_frame, text="Rows:").grid(row=2, column=6, sticky=tk.W, pady=(6, 0))
        self.grid_rows_entry = ttk.Entry(grid_gen_frame, width=5)
        self.grid_rows_entry.grid(row=2, column=7, padx=(4, 0), pady=(6, 0))
        self.grid_rows_entry.insert(0, "6")

        ttk.Button(grid_gen_frame, text="Generate Grid Array",
                   command=self.generate_grid_array).grid(
            row=3, column=0, columnspan=8, sticky=(tk.W, tk.E), pady=(6, 4))

        self.grid_output_text = tk.Text(grid_gen_frame, height=6, font=("Consolas", 8),
                                        state=tk.DISABLED)
        self.grid_output_text.grid(row=4, column=0, columnspan=7, sticky=(tk.W, tk.E))
        grid_scroll = ttk.Scrollbar(grid_gen_frame, orient=tk.VERTICAL,
                                    command=self.grid_output_text.yview)
        grid_scroll.grid(row=4, column=7, sticky=(tk.N, tk.S))
        self.grid_output_text.config(yscrollcommand=grid_scroll.set)

        ttk.Button(grid_gen_frame, text="Copy Output to Clipboard",
                   command=self.copy_grid_output).grid(
            row=5, column=0, columnspan=8, sticky=(tk.W, tk.E), pady=(4, 0))

        grid_gen_frame.columnconfigure(1, weight=1)
        grid_gen_frame.columnconfigure(3, weight=1)

        # ── Styles, bindings ────────────────────────────────────────────────
        style = ttk.Style()
        style.configure("Accent.TButton", foreground="blue")
        style.configure("Success.TButton", foreground="green")
        style.configure("Warning.TButton", foreground="orange")

        self.root.bind('<Key>', self.on_key_press)
        self.root.focus_set()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_position_object_selected(self, event):
        """Populate the index dropdown when an object is chosen."""
        obj = self.position_object_dropdown.get()
        keys = self.position_groups.get(obj, [])
        # Sort numerically by locator_index so [0],[1]... appear in order
        keys_sorted = sorted(
            keys,
            key=lambda k: (self.workflow_positions[k].get('locator_index') or 0)
        )
        self.position_dropdown['values'] = keys_sorted
        self.position_dropdown.set('')
        self.position_description_label.config(
            text=f"{len(keys_sorted)} position(s) in '{obj}'", foreground="gray")
        self.selected_position = None

    def on_position_selected(self, event):
        """Handle index dropdown selection."""
        position_name = self.position_dropdown.get()
        if position_name in self.workflow_positions:
            position_data = self.workflow_positions[position_name]
            description = f"{position_data['description']} (Category: {position_data['category']})"
            self.position_description_label.config(text=description, foreground="black")
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
                # Direct movement to target position using goto_safe for collision avoidance
                target_position = [
                    position_data["gripper_cts"],
                    position_data["elbow_cts"],
                    position_data["shoulder_cts"],
                    position_data["z_cts"]
                ]
                logger.info(f"Moving directly to target position: {target_position}")
                
                # Use goto_safe if available for collision-aware direct movement
                if hasattr(self.robot, 'goto_safe'):
                    self.robot.goto_safe(target_position)
                    logger.info("Moved to position using goto_safe")
                else:
                    self.robot.goto(target_position, wait=True)  
                    logger.info("Moved to position using goto")
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
                "elbow_cts": current_pos[1],      # FIXED: elbow is now index 1
                "shoulder_cts": current_pos[2],   # FIXED: shoulder is now index 2
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
                f.write(f"elbow_cts: {current_pos[1]}\\n")      # FIXED: elbow is now index 1
                f.write(f"shoulder_cts: {current_pos[2]}\\n")   # FIXED: shoulder is now index 2
                f.write(f"z_cts: {current_pos[3]}\\n\\n")
                f.write("Array Format (for copy-paste):\\n")
                f.write(f"[{current_pos[0]}, {current_pos[1]}, {current_pos[2]}, {current_pos[3]}]\\n\\n")
                f.write("Python Dictionary Format:\\n")
                f.write(f"'{custom_name}': {{\\n")
                f.write(f"    'gripper_cts': {current_pos[0]},\\n")
                f.write(f"    'elbow_cts': {current_pos[1]},\\n")      # FIXED: elbow is now index 1
                f.write(f"    'shoulder_cts': {current_pos[2]},\\n")   # FIXED: shoulder is now index 2
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
                    
                    coords = f"[{data.get('gripper_cts', 0)}, {data.get('elbow_cts', 0)}, {data.get('shoulder_cts', 0)}, {data.get('z_cts', 0)}]"
                    
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
                        coords = f"[{data.get('gripper_cts', 0)}, {data.get('elbow_cts', 0)}, {data.get('shoulder_cts', 0)}, {data.get('z_cts', 0)}]"
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
        if self._check_robot_health():
            self._move_xy_delta(-self.move_increment_mm, 0)

    def move_x_right(self):
        if self._check_robot_health():
            self._move_xy_delta(self.move_increment_mm, 0)

    def move_y_back(self):
        if self._check_robot_health():
            self._move_xy_delta(0, -self.move_increment_mm)

    def move_y_forward(self):
        if self._check_robot_health():
            self._move_xy_delta(0, self.move_increment_mm)

    def _check_robot_health(self):
        """Check if robot is in a safe state to move."""
        if not self.is_connected:
            logger.error("Robot not connected - cannot move")
            return False
        
        try:
            # Check robot status
            status = self.robot.get_robot_status()
            logger.info(f"Robot status check: {status}")
            
            # Check for any error states
            if hasattr(self.robot, 'get_robot_errors'):
                errors = self.robot.get_robot_errors()
                if errors:
                    logger.error(f"Robot has errors: {errors}")
                    return False
            
            # For real robot, check individual axis status
            if not hasattr(self.robot, 'z_position_mm'):  # Real robot
                for axis in [0, 1, 2, 3]:  # gripper, elbow, shoulder, z
                    axis_status = self.robot.get_axis_status(axis)
                    logger.info(f"Axis {axis} status: {axis_status}")
                    if 'ERR' in str(axis_status) or 'FAULT' in str(axis_status):
                        logger.error(f"Axis {axis} has fault: {axis_status}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking robot health: {e}")
            return False

    def diagnose_motor_fault(self):
        """Diagnose motor fault issues and provide recovery suggestions."""
        logger.info("=== MOTOR FAULT DIAGNOSIS ===")
        
        if not self.is_connected:
            logger.error("Cannot diagnose: Robot not connected")
            return
        
        try:
            # Get overall robot status
            try:
                overall_status = self.robot.get_robot_status()
                logger.info(f"Overall robot status: {overall_status}")
            except Exception as e:
                logger.error(f"Could not get overall status: {e}")
            
            # Check each axis individually
            axis_names = ["GRIPPER", "ELBOW", "SHOULDER", "Z_AXIS"]
            for i, name in enumerate(axis_names):
                try:
                    axis_status = self.robot.get_axis_status(i)
                    axis_pos = self.robot.get_axis_position(i)
                    logger.info(f"Axis {i} ({name}): status={axis_status}, position={axis_pos}")
                    
                    # Check if axis is served/enabled
                    try:
                        # Try to read servo status if available
                        if hasattr(self.robot, 'get_axis_servo_status'):
                            servo_status = self.robot.get_axis_servo_status(i)
                            logger.info(f"Axis {i} servo status: {servo_status}")
                    except Exception as servo_e:
                        logger.warning(f"Could not get servo status for axis {i}: {servo_e}")
                        
                except Exception as e:
                    logger.error(f"Error checking axis {i} ({name}): {e}")
            
            # Try to read any error messages from the controller
            try:
                # Some robots have error message queues
                if hasattr(self.robot, 'read_error_queue'):
                    errors = self.robot.read_error_queue()
                    if errors:
                        logger.error(f"Error queue: {errors}")
                    else:
                        logger.info("Error queue is clear")
            except Exception as e:
                logger.warning(f"Could not read error queue: {e}")
            
            # Check temperature and power status if available
            try:
                if hasattr(self.robot, 'get_temp'):
                    for ch in range(4):  # Common number of temp channels
                        try:
                            temp = self.robot.get_temp(ch)
                            logger.info(f"Temperature channel {ch}: {temp}")
                        except:
                            break  # Stop when we run out of valid channels
            except Exception as e:
                logger.warning(f"Could not read temperature: {e}")
            
            logger.info("=== SUGGESTED RECOVERY ACTIONS ===")
            logger.info("1. Check if robot is properly homed (robot.home_robot())")
            logger.info("2. Check all cables and connections")
            logger.info("3. Verify power supply is adequate")
            logger.info("4. Try restarting the controller")
            logger.info("5. Check for mechanical obstructions")
            logger.info("6. Verify axis limits and positions are within range")
            
        except Exception as e:
            logger.error(f"Error during motor fault diagnosis: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

    def _move_xy_delta(self, dx, dy):
        """Move end-effector by (dx, dy) mm in Cartesian space.
        On real hardware uses IK (n9_fk + n9_ik) for true Cartesian steps.
        In simulation falls back to single-joint nudge since mock has no n9_ik.
        """
        if not self.is_connected:
            logger.error("Cannot move XY: Robot not connected")
            return
        try:
            logger.info(f"=== Starting XY movement: dx={dx:.1f}mm, dy={dy:.1f}mm ===")
            
            # Check robot status before movement
            try:
                robot_status = self.robot.get_robot_status()
                logger.info(f"Robot status before movement: {robot_status}")
            except Exception as e:
                logger.warning(f"Could not get robot status: {e}")

            # Current joint counts: get_robot_positions() -> [gripper, elbow, shoulder, z]
            logger.info("Getting current robot positions...")
            positions = self.robot.get_robot_positions()
            logger.info(f"Current positions (counts): gripper={positions[0]}, elbow={positions[1]}, shoulder={positions[2]}, z={positions[3]}")
            
            gripper_cts, elbow_cts, shoulder_cts, z_cts = (
                positions[0], positions[1], positions[2], positions[3]
            )

            # Current XY via FK: n9_fk(gripper, elbow, shoulder) per API
            logger.info("Computing current X-Y position via forward kinematics...")
            x, y, tool_angle = self.robot.n9_fk(gripper_cts, elbow_cts, shoulder_cts)
            logger.info(f"Current X-Y position: ({x:.1f}, {y:.1f})mm, tool_angle={tool_angle:.2f}rad")
            
            target_x = x + dx
            target_y = y + dy
            logger.info(f"Target X-Y position: ({target_x:.1f}, {target_y:.1f})mm")

            if hasattr(self.robot, 'n9_ik'):
                logger.info("Real robot detected - using inverse kinematics")
                
                # Real robot: solve IK for both shoulder configurations,
                # pick whichever keeps the shoulder closest to its current count
                # so the arm stays in whatever pose it was already in.
                logger.info("Computing IK solutions...")
                
                sol_c = self.robot.n9_ik(target_x, target_y,
                                          shoulder_preference=self.robot.SHOULDER_CENTER)
                logger.info(f"Center shoulder solution: gripper={sol_c[0]}, elbow={sol_c[1]}, shoulder={sol_c[2]}")
                
                sol_o = self.robot.n9_ik(target_x, target_y,
                                          shoulder_preference=self.robot.SHOULDER_OUT)
                logger.info(f"Out shoulder solution: gripper={sol_o[0]:.3f}, elbow={sol_o[1]:.3f}, shoulder={sol_o[2]:.3f}")
                
                # Convert current position to radians for proper comparison
                current_gripper_rad = self.robot.counts_to_rad(self.robot.GRIPPER, gripper_cts)
                current_elbow_rad = self.robot.counts_to_rad(self.robot.ELBOW, elbow_cts)
                current_shoulder_rad = self.robot.counts_to_rad(self.robot.SHOULDER, shoulder_cts)
                
                # Pick solution that keeps shoulder closest to current position (compare radians to radians)
                center_diff = abs(sol_c[2] - current_shoulder_rad)
                out_diff = abs(sol_o[2] - current_shoulder_rad)
                chosen = sol_c if center_diff <= out_diff else sol_o
                logger.info(f"Current shoulder: {current_shoulder_rad:.3f}rad")
                logger.info(f"Choosing solution with smaller shoulder change: center_diff={center_diff:.3f}rad, out_diff={out_diff:.3f}rad")
                logger.info(f"Selected: {'CENTER' if chosen == sol_c else 'OUT'}")
                
                # Convert IK solution from radians to counts
                gripper_cts_target = self.robot.rad_to_counts(self.robot.GRIPPER, chosen[0])
                elbow_cts_target = self.robot.rad_to_counts(self.robot.ELBOW, chosen[1])
                shoulder_cts_target = self.robot.rad_to_counts(self.robot.SHOULDER, chosen[2])
                target_pos = [gripper_cts_target, elbow_cts_target, shoulder_cts_target, z_cts]
                
                logger.info(f"IK solution (radians): gripper={chosen[0]:.3f}, elbow={chosen[1]:.3f}, shoulder={chosen[2]:.3f}")
                logger.info(f"Converted to counts: gripper={gripper_cts_target}, elbow={elbow_cts_target}, shoulder={shoulder_cts_target}")
                
                # Check movement safety with converted values
                gripper_diff = abs(gripper_cts_target - gripper_cts)
                elbow_diff = abs(elbow_cts_target - elbow_cts)
                shoulder_diff = abs(shoulder_cts_target - shoulder_cts)
                
                logger.info(f"Movement deltas: gripper={gripper_diff:.0f}cts, elbow={elbow_diff:.0f}cts, shoulder={shoulder_diff:.0f}cts")
                logger.info(f"Moving to target position: {target_pos}")
                
                # Safety checks - prevent motor faults from excessive movement
                MAX_SAFE_DELTA = 5000  # Maximum safe movement per axis (counts)
                MAX_TOTAL_MOVEMENT = 5000  # Maximum total movement magnitude (counts)
                
                max_single_delta = max(gripper_diff, elbow_diff, shoulder_diff)
                total_movement = gripper_diff + elbow_diff + shoulder_diff
                
                if max_single_delta > MAX_SAFE_DELTA:
                    logger.error(f"UNSAFE: Single axis movement too large! Max delta = {max_single_delta:.0f}cts (limit: {MAX_SAFE_DELTA}cts) - ABORTING")
                    return
                    
                if total_movement > MAX_TOTAL_MOVEMENT:
                    logger.error(f"UNSAFE: Total movement too large! Total = {total_movement:.0f}cts (limit: {MAX_TOTAL_MOVEMENT}cts) - ABORTING")
                    return
                
                logger.info(f"Movement safety check PASSED: max_delta={max_single_delta:.0f}cts, total={total_movement:.0f}cts")
                
                self.robot.goto(target_pos, wait=True)
                logger.info(f"Successfully completed XY move ({dx:+.1f}, {dy:+.1f})mm -> ({target_x:.1f}, {target_y:.1f})mm")
            else:
                logger.info("Simulation robot detected - using joint nudging")
                # Simulation: nudge shoulder for X, elbow for Y
                if dx != 0:
                    new_s = max(SHOULDER_MIN_RAD, min(SHOULDER_MAX_RAD,
                                self.current_shoulder_angle + dx * 0.01))
                    logger.info(f"Moving shoulder from {self.current_shoulder_angle:.3f} to {new_s:.3f} rad")
                    self.robot.move_axis_rad(self.robot.SHOULDER, new_s, wait=True)
                    self.current_shoulder_angle = new_s
                if dy != 0:
                    new_e = max(ELBOW_MIN_RAD, min(ELBOW_MAX_RAD,
                                self.current_elbow_angle + dy * 0.01))
                    logger.info(f"Moving elbow from {self.current_elbow_angle:.3f} to {new_e:.3f} rad")
                    self.robot.move_axis_rad(self.robot.ELBOW, new_e, wait=True)
                    self.current_elbow_angle = new_e
                logger.info(f"SIM XY nudge ({dx:+.1f}, {dy:+.1f})mm completed")

            logger.info("=== XY movement completed successfully ===")
            self.update_display()
            
        except Exception as e:
            logger.error(f"=== ERROR in XY movement: {str(e)} ===")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Try to read robot status after error
            try:
                robot_status = self.robot.get_robot_status()
                logger.error(f"Robot status after error: {robot_status}")
            except Exception as status_e:
                logger.error(f"Could not get robot status after error: {status_e}")

    def _safe_move_z(self, target_mm):
        """Safely move Z-axis with bounds checking."""
        if not self.is_connected:
            logger.error("Cannot move Z: Robot not connected")
            return
            
        # Check robot health first
        if not self._check_robot_health():
            logger.error("Cannot move Z: Robot health check failed")
            return
            
        target_mm = max(Z_AXIS_MIN_MM, min(Z_AXIS_MAX_MM, target_mm))
        logger.info(f"Moving Z-axis to {target_mm:.1f}mm (current: {self.current_z_position:.1f}mm)")
        
        try:
            self.robot.move_z(target_mm, wait=True)
            logger.info(f"Successfully moved Z-axis to {target_mm:.1f}mm")
            self.update_display()
        except Exception as e:
            logger.error(f"Error moving Z-axis to {target_mm:.1f}mm: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Try to get more diagnostic info
            try:
                positions = self.robot.get_robot_positions()
                current_z_cts = positions[3]
                current_z_mm = self.robot.counts_to_mm(self.robot.Z_AXIS, current_z_cts)
                logger.error(f"Current Z position after error: {current_z_mm:.1f}mm ({current_z_cts} cts)")
            except Exception as diag_e:
                logger.error(f"Could not get diagnostic position info: {diag_e}")
            
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

    def open_clamp(self):
        """Open the robot clamp (clamp 0)."""
        if not self.is_connected:
            logger.warning("Cannot open clamp: Robot not connected")
            return
        try:
            if hasattr(self.robot, 'open_clamp'):
                self.robot.open_clamp()  # Remove wait parameter
                logger.info("Clamp opened")
            else:
                logger.warning("Clamp control not available on this robot")
        except Exception as e:
            logger.error(f"Error opening clamp: {str(e)}")
    
    def close_clamp(self):
        """Close the robot clamp (clamp 0)."""
        if not self.is_connected:
            logger.warning("Cannot close clamp: Robot not connected")
            return
        try:
            if hasattr(self.robot, 'close_clamp'):
                self.robot.close_clamp()  # Remove wait parameter
                logger.info("Clamp closed")
            else:
                logger.warning("Clamp control not available on this robot")
        except Exception as e:
            logger.error(f"Error closing clamp: {str(e)}")

    def get_pipet_large(self):
        """Get a large tip using nr_robot."""
        if self.nr_robot is not None:
            try:
                self.nr_robot.get_pipet("large_tip")
                self.update_display()
            except Exception as e:
                logger.error(f"Error getting large tip: {str(e)}")
        else:
            logger.warning("nr_robot not available - connect robot first")

    def get_pipet_small(self):
        """Get a small tip using nr_robot."""
        if self.nr_robot is not None:
            try:
                self.nr_robot.get_pipet("small_tip")
                self.update_display()
            except Exception as e:
                logger.error(f"Error getting small tip: {str(e)}")
        else:
            logger.warning("nr_robot not available - connect robot first")

    def remove_pipet_action(self):
        """Remove pipet tip using nr_robot."""
        if self.nr_robot is not None:
            try:
                self.nr_robot.remove_pipet()
                self.update_display()
            except Exception as e:
                logger.error(f"Error removing pipet: {str(e)}")
        else:
            logger.warning("nr_robot not available - connect robot first")

    def on_grid_object_selected(self, event):
        """Populate grid entry fields from the selected object's config in vial_positions.yaml."""
        obj_name = self.grid_object_dropdown.get()
        cfg = self.grid_configs.get(obj_name)
        if cfg is None:
            return
        for entry, key in [
            (self.grid_cols_entry,     'num_cols'),
            (self.grid_rows_entry,     'num_rows'),
            (self.grid_x_offset_entry, 'x_delta_mm'),
            (self.grid_y_offset_entry, 'y_delta_mm'),
        ]:
            entry.delete(0, tk.END)
            val = cfg.get(key)
            entry.insert(0, str(val) if val is not None else "")
        logger.info(f"Grid config loaded from '{obj_name}': {cfg}")

    def set_grid_origin(self):
        """Freeze current robot position as the grid origin for array generation."""
        if not self.is_connected:
            messagebox.showerror("Error", "Robot not connected")
            return
        try:
            positions = self.robot.get_robot_positions()
            gripper_cts, elbow_cts, shoulder_cts, z_cts = (
                positions[0], positions[1], positions[2], positions[3]
            )
            x, y, _ = self.robot.n9_fk(gripper_cts, elbow_cts, shoulder_cts)
            self.grid_origin = {
                'x': x, 'y': y,
                'z_cts': z_cts,
                'gripper_cts': gripper_cts,
                'shoulder_cts': shoulder_cts,
            }
            self.grid_origin_label.config(
                text=f"Origin: X={x:.1f}mm, Y={y:.1f}mm  |  Z={z_cts}cts  G={gripper_cts}cts",
                foreground="green"
            )
            logger.info(f"Grid origin set: X={x:.1f}, Y={y:.1f}, Z_cts={z_cts}, G_cts={gripper_cts}")
        except Exception as e:
            logger.error(f"Error setting grid origin: {str(e)}")

    def generate_grid_array(self):
        """Generate Locator.py-format position array from origin + grid parameters.
        
        Ordering: column-major (all rows in col 0, then all rows in col 1, ...)
        so index = col * n_rows + row.  Different rows have different Y,
        different columns have different X.
        Output format: [gripper, elbow, shoulder, z]  (matches standard API convention)
        """
        if self.grid_origin is None:
            messagebox.showerror("Error", "Set origin first (jog to position, then click Set Origin)")
            return
        if not hasattr(self.robot, 'n9_ik'):
            messagebox.showerror("Error",
                "Grid generation requires real hardware.\n"
                "n9_ik is not available in simulation mode.")
            return
        try:
            x_offset = float(self.grid_x_offset_entry.get())
            y_offset = float(self.grid_y_offset_entry.get())
            n_cols = int(self.grid_cols_entry.get())
            n_rows = int(self.grid_rows_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values - check x_offset, y_offset, cols, rows")
            return

        x0 = self.grid_origin['x']
        y0 = self.grid_origin['y']
        z_cts = self.grid_origin['z_cts']
        shoulder_cts = self.grid_origin['shoulder_cts']

        entries = []
        ik_errors = []
        last_shoulder_rad = self.robot.counts_to_rad(self.robot.SHOULDER, shoulder_cts)  # Track shoulder progression
        
        for col in range(n_cols):          # COLUMN-MAJOR: Column loop (outer)
            for row in range(n_rows):      # Row loop (inner) - go down Y first
                target_x = x0 + col * x_offset
                target_y = y0 + row * y_offset
                try:
                    sol_c = self.robot.n9_ik(target_x, target_y,
                                             shoulder_preference=self.robot.SHOULDER_CENTER)
                    sol_o = self.robot.n9_ik(target_x, target_y,
                                             shoulder_preference=self.robot.SHOULDER_OUT)
                    # Pick config closest to PREVIOUS position shoulder for smooth progression
                    chosen = sol_c if abs(sol_c[2] - last_shoulder_rad) <= abs(sol_o[2] - last_shoulder_rad) else sol_o
                    
                    # Update shoulder reference for next position
                    last_shoulder_rad = chosen[2]
                    
                    # CRITICAL: n9_ik returns radians, must convert to counts before creating entry
                    gripper_cts = int(self.robot.rad_to_counts(self.robot.GRIPPER, chosen[0]))
                    elbow_cts = int(self.robot.rad_to_counts(self.robot.ELBOW, chosen[1]))
                    shoulder_cts_target = int(self.robot.rad_to_counts(self.robot.SHOULDER, chosen[2]))
                    
                    entry = [gripper_cts, elbow_cts, shoulder_cts_target, int(z_cts)]
                    entries.append((col, row, entry))
                except Exception as e:
                    ik_errors.append(f"col={col}, row={row}: {e}")
                    entries.append((col, row, None))

        # Build output string
        valid_positions = [pos for _, _, pos in entries if pos is not None]
        output = f"my_grid = {valid_positions}"
        self.grid_output_text.config(state=tk.NORMAL)
        self.grid_output_text.delete("1.0", tk.END)
        self.grid_output_text.insert(tk.END, output)
        self.grid_output_text.config(state=tk.DISABLED)

        n_ok = sum(1 for _, _, p in entries if p is not None)
        if ik_errors:
            logger.warning(f"Grid generation: {n_ok}/{len(entries)} succeeded. Failures: {ik_errors}")
        else:
            logger.info(f"Grid generated: {n_cols}x{n_rows}={len(entries)} positions")

    def copy_grid_output(self):
        """Copy the grid output text to clipboard."""
        content = self.grid_output_text.get("1.0", tk.END).strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            logger.info("Grid array copied to clipboard")

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
            
    def move_home(self):
        """Move robot to home position (faster than homing - just moves to [0,0,0,0])."""
        if not self.is_connected:
            messagebox.showerror("Error", "Robot not connected")
            return
            
        try:
            logger.info("Moving robot to home position...")
            
            # Home position from Locator.py: [0, 0, 0, 0]
            home_position = [0, 0, 0, 0]
            
            if hasattr(self.robot, 'goto_safe'):
                # Use goto_safe if available (real robot via nr_robot)
                self.robot.goto_safe(home_position, wait=True)
                logger.info("Moved to home position using goto_safe")
            else:
                # Fallback to regular goto for direct NorthC9 access
                self.robot.goto(home_position, wait=True)
                logger.info("Moved to home position using goto")
            
            # Record move in session history
            self.session_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "action": "move_home",
                "target_position": {
                    "gripper_cts": 0,
                    "elbow_cts": 0,
                    "shoulder_cts": 0,
                    "z_cts": 0
                }
            })
            
            logger.info("Successfully moved to home position")
            # Don't show popup for move home since it's a quick operation
            
        except Exception as e:
            logger.error(f"Error moving to home: {str(e)}")
            messagebox.showerror("Error", f"Failed to move to home:\n{str(e)}")
        finally:
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
                        cartesian_pos = self.robot.n9_fk(positions[0], positions[1], positions[2])  # gripper, elbow, shoulder (CORRECT ORDER)
                        self.current_x_position = cartesian_pos[0]
                        self.current_y_position = cartesian_pos[1]
                        logger.debug(f"FK: positions[gripper={positions[0]}, elbow={positions[1]}, shoulder={positions[2]}] -> XY=({cartesian_pos[0]:.1f}, {cartesian_pos[1]:.1f})")
                    except Exception as e:
                        logger.warning(f"Could not get x-y position from kinematics during display update: {e}")
                        # Keep current values if FK fails
                
                # Update position labels
                if self.z_position_label is not None:
                    self.z_position_label.config(text=f"Z-Axis: {self.current_z_position:.1f} mm")
                if self.x_position_label is not None:
                    self.x_position_label.config(text=f"X-Position: {self.current_x_position:.1f} mm")
                if self.y_position_label is not None:
                    self.y_position_label.config(text=f"Y-Position: {self.current_y_position:.1f} mm")
                if self.gripper_position_label is not None:
                    self.gripper_position_label.config(text=f"Gripper: {self.current_gripper_angle:.2f} rad ({self.current_gripper_angle*180/3.14159:.1f}°)")
                
                # Update gripper status
                if self.gripper_status_label is not None:
                    if self.gripper_is_open:
                        self.gripper_status_label.config(text="Gripper: OPEN ✋", foreground="green")
                    else:
                        self.gripper_status_label.config(text="Gripper: CLOSED 👊", foreground="red")
                
                # Update connection status
                if self.status_label is not None:
                    if self.is_homed:
                        self.status_label.config(text="Connected & Homed", foreground="green")
                    else:
                        self.status_label.config(text="Connected - Not Homed", foreground="orange")
                        
            except Exception as e:
                logger.error(f"Error updating display: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                if self.status_label is not None:
                    self.status_label.config(text="Error reading position", foreground="red")
        else:
            # Disconnected state
            if self.z_position_label is not None:
                self.z_position_label.config(text="Z-Axis: Unknown")
            if self.x_position_label is not None:
                self.x_position_label.config(text="X-Position: Unknown")
            if self.y_position_label is not None:
                self.y_position_label.config(text="Y-Position: Unknown")
            if self.gripper_position_label is not None:
                self.gripper_position_label.config(text="Gripper: Unknown")
            if self.gripper_status_label is not None:
                self.gripper_status_label.config(text="Gripper: Unknown", foreground="gray")
            if self.status_label is not None:
                self.status_label.config(text="Disconnected", foreground="red")
    
    def get_current_positions(self):
        """Get current robot positions in counts format [gripper, elbow, shoulder, z]."""
        try:
            if not self.is_connected:
                # Return default/last known positions if not connected
                if hasattr(self, 'robot') and hasattr(self.robot, 'gripper_cts'):  # Mock robot
                    return [
                        self.robot.gripper_cts,
                        self.robot.elbow_cts,      # Fixed order: elbow before shoulder
                        self.robot.shoulder_cts, 
                        self.robot.z_cts
                    ]
                else:
                    # Default positions if nothing available
                    return [0, 0, 0, 0]
            
            if hasattr(self.robot, 'z_position_mm'):  # Mock robot
                return [
                    self.robot.gripper_cts,
                    self.robot.elbow_cts,        # Fixed order: elbow before shoulder
                    self.robot.shoulder_cts,
                    self.robot.z_cts
                ]
            else:  # Real robot
                positions = self.robot.get_robot_positions()  # Returns [gripper, elbow, shoulder, z]
                return [
                    positions[0],  # gripper_cts
                    positions[1],  # elbow_cts (FIXED: was positions[2])  
                    positions[2],  # shoulder_cts (FIXED: was positions[1])
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
            vial_file_path=vial_file
        )
        controller.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        messagebox.showerror("Application Error", f"An error occurred:\n{str(e)}")


if __name__ == "__main__":
    main()
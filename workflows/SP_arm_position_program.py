#!/usr/bin/env python3
"""
North Robotics Arm Position Control Program
==========================================

This program provides a simple GUI interface to control the North robot arm
using keyboard arrow keys and includes a home button.

Features:
- Up/Down arrow keys to move the Z-axis
- Home button to home the robot
- Real-time position display
- Safety features and error handling
- Support for both simulation and real hardware

Controls:
- UP Arrow: Move arm up by 5mm
- DOWN Arrow: Move arm down by 5mm
- HOME Button: Home the robot
- ESC: Exit program

Author: Generated for North Robotics Control
Date: March 16, 2026
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
import time
import sys
import os
from pathlib import Path

# Add the project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
DEFAULT_MOVE_INCREMENT_MM = 5.0      # Default mm per arrow key press for Z-axis
DEFAULT_MOVE_INCREMENT_RAD = 0.1     # Default radians per key press for rotational joints
MAX_MOVE_INCREMENT_MM = 50.0         # Maximum allowed Z increment (mm)
MAX_MOVE_INCREMENT_RAD = 1.0         # Maximum allowed rotational increment (radians)
MIN_MOVE_INCREMENT_MM = 0.1          # Minimum allowed Z increment (mm)
MIN_MOVE_INCREMENT_RAD = 0.01        # Minimum allowed rotational increment (radians)
Z_AXIS_MIN_MM = 30.0                 # Minimum Z position (mm)
Z_AXIS_MAX_MM = 292.0                # Maximum Z position (mm)
ELBOW_MIN_RAD = -(5/6) * 3.14159    # Minimum elbow angle (radians)
ELBOW_MAX_RAD = (5/6) * 3.14159     # Maximum elbow angle (radians)
SHOULDER_MIN_RAD = -(2/3) * 3.14159 # Minimum shoulder angle (radians)
SHOULDER_MAX_RAD = (2/3) * 3.14159  # Maximum shoulder angle (radians)
GRIPPER_MIN_RAD = -6.28              # Minimum gripper angle (radians)
GRIPPER_MAX_RAD = 6.28               # Maximum gripper angle (radians)

# Predefined positions from surfactant workflow
WORKFLOW_POSITIONS = {
    # Standard positions
    "Home": {
        "description": "Robot home position",
        "gripper_cts": 0,
        "shoulder_cts": 0,
        "elbow_cts": 0,
        "z_cts": 0
    },
    
    # Clamp position (from Locator.py vial_clamp)
    "Clamp Position": {
        "description": "Vial clamping station for manipulation",
        "gripper_cts": 2476,
        "shoulder_cts": 7635,
        "elbow_cts": 44106,
        "z_cts": 17350
    },
    
    # Main 8mL vial rack positions (key positions used in surfactant workflow)
    "Main Rack - Position 36": {
        "description": "8mL vial rack position 36",
        "gripper_cts": 1708,
        "shoulder_cts": 9251,
        "elbow_cts": 27829,
        "z_cts": 22200
    },
    "Main Rack - Position 43": {
        "description": "8mL vial rack position 43",
        "gripper_cts": 2021,
        "shoulder_cts": 1163,
        "elbow_cts": 19726,
        "z_cts": 22200
    },
    "Main Rack - Position 44 (Water)": {
        "description": "8mL vial rack position 44 (water)",
        "gripper_cts": 1928,
        "shoulder_cts": 2422,
        "elbow_cts": 19859,
        "z_cts": 22200
    },
    "Main Rack - Position 45 (Water_2)": {
        "description": "8mL vial rack position 45 (water_2)",
        "gripper_cts": 1845,
        "shoulder_cts": 3750,
        "elbow_cts": 20404,
        "z_cts": 22200
    },
    "Main Rack - Position 46": {
        "description": "8mL vial rack position 46",
        "gripper_cts": 1768,
        "shoulder_cts": 5155,
        "elbow_cts": 21233,
        "z_cts": 22200
    },
    "Main Rack - Position 47": {
        "description": "8mL vial rack position 47",
        "gripper_cts": 1692,
        "shoulder_cts": 6655,
        "elbow_cts": 22291,
        "z_cts": 22200
    },
    
    # Large vial positions
    "Large Vial Rack - Position 0": {
        "description": "20mL vial rack position 0",
        "gripper_cts": 721,
        "shoulder_cts": 9854,
        "elbow_cts": 26732,
        "z_cts": 13921
    },
    "Large Vial Rack - Position 1": {
        "description": "20mL vial rack position 1", 
        "gripper_cts": 405,
        "shoulder_cts": 17735,
        "elbow_cts": 34375,
        "z_cts": 13921
    },
    
    # Small vial positions
    "Small Vial Rack - Position 0": {
        "description": "1mL vial rack position 0",
        "gripper_cts": 848,
        "shoulder_cts": 15038,
        "elbow_cts": 44585,
        "z_cts": 16166
    },
    "Small Vial Rack - Position 1": {
        "description": "1mL vial rack position 1",
        "gripper_cts": 788,
        "shoulder_cts": 16021,
        "elbow_cts": 45013,
        "z_cts": 16166
    },
    
    # 50mL vial positions
    "50mL Vial Rack - Position 0": {
        "description": "50mL vial rack position 0",
        "gripper_cts": -1378,
        "shoulder_cts": 14164,
        "elbow_cts": 35944,
        "z_cts": 11100
    },
    "50mL Vial Rack - Position 3": {
        "description": "50mL vial rack position 3",
        "gripper_cts": -1518,
        "shoulder_cts": 15521,
        "elbow_cts": 35093,
        "z_cts": 11100
    },
    
    # Photoreactor position
    "Photoreactor Position": {
        "description": "Photoreactor array position",
        "gripper_cts": 69,
        "shoulder_cts": 33380,
        "elbow_cts": 24059,
        "z_cts": 20577
    },
    
    # Heater grid positions
    "Heater Grid - Position 0": {
        "description": "Heater grid position 0",
        "gripper_cts": -1232,
        "shoulder_cts": 39645,
        "elbow_cts": 13715,
        "z_cts": 15241
    },
    "Heater Grid - Position 5": {
        "description": "Heater grid position 5 (center area)",
        "gripper_cts": -949,
        "shoulder_cts": 39840,
        "elbow_cts": 21249,
        "z_cts": 15241
    },
    
    # Tip rack positions - Small tips (high)
    "Small Tip Rack - Position 0": {
        "description": "Small pipet tip rack position 0",
        "gripper_cts": 2668,
        "shoulder_cts": 37357,
        "elbow_cts": 8247,
        "z_cts": 9150
    },
    "Small Tip Rack - Position 24": {
        "description": "Small pipet tip rack position 24 (middle)",
        "gripper_cts": 2893,
        "shoulder_cts": 35374,
        "elbow_cts": 9991,
        "z_cts": 9150
    },
    "Small Tip Rack - Position 47": {
        "description": "Small pipet tip rack position 47 (end)",
        "gripper_cts": 3080,
        "shoulder_cts": 32757,
        "elbow_cts": 9540,
        "z_cts": 9150
    },
    
    # Tip rack positions - Large tips
    "Large Tip Rack - Position 0": {
        "description": "Large pipet tip rack position 0",
        "gripper_cts": 2677,
        "shoulder_cts": 39055,
        "elbow_cts": 11930,
        "z_cts": 12562
    },
    "Large Tip Rack - Position 24": {
        "description": "Large pipet tip rack position 24 (middle)",
        "gripper_cts": 2912,
        "shoulder_cts": 36904,
        "elbow_cts": 13611,
        "z_cts": 12562
    },
    "Large Tip Rack - Position 47": {
        "description": "Large pipet tip rack position 47 (end)",
        "gripper_cts": 3095,
        "shoulder_cts": 34238,
        "elbow_cts": 12950,
        "z_cts": 12562
    },
    
    # Tip removal positions
    "Tip Removal - Small": {
        "description": "Small pipet tip removal position",
        "gripper_cts": -270,
        "shoulder_cts": 27025,
        "elbow_cts": 6845,
        "z_cts": 7679
    },
    "Tip Removal - Large": {
        "description": "Large pipet tip removal position",
        "gripper_cts": -97,
        "shoulder_cts": 27004,
        "elbow_cts": 6864,
        "z_cts": 9160
    },
    
    # Safe positions for transport
    "Safe Transport Height": {
        "description": "Safe height for moving between positions",
        "gripper_cts": 0,
        "shoulder_cts": 0,
        "elbow_cts": 0,
        "z_cts": 5000  # High safe position
    }
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arm_position_control.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RobotArmController:
    """Main controller class for the robot arm GUI application."""
    
    def __init__(self):
        self.robot = None
        self.root = None
        self.is_connected = False
        self.current_z_position = 0.0
        self.current_elbow_angle = 0.0
        self.current_shoulder_angle = 0.0
        self.current_gripper_angle = 0.0
        self.gripper_is_open = False  # Track gripper open/close state
        self.is_homed = False
        
        # Movement increments (user configurable)
        self.move_increment_mm = DEFAULT_MOVE_INCREMENT_MM
        self.move_increment_rad = DEFAULT_MOVE_INCREMENT_RAD
        
        # GUI components will be set in create_gui()
        self.z_position_label = None
        self.elbow_position_label = None
        self.shoulder_position_label = None
        self.gripper_position_label = None
        self.status_label = None
        self.home_button = None
        self.z_increment_entry = None
        self.rad_increment_entry = None
        self.increment_status_label = None
        self.gripper_status_label = None  # Display gripper open/close status
        
        # Position dropdown components
        self.position_dropdown = None
        self.position_description_label = None
        self.selected_position = None
        
    def connect_robot(self):
        """Initialize connection to the North robot."""
        try:
            if SIMULATE:
                logger.info("Starting in simulation mode")
                # Create a mock robot for simulation
                self.robot = self._create_mock_robot()
                self.is_connected = True
                self.current_z_position = 100.0  # Start at middle position
                return True
            else:
                logger.info(f"Connecting to robot at address {CONTROLLER_ADDR} with serial {NETWORK_SERIAL}")
                self.robot = NorthC9(
                    CONTROLLER_ADDR,
                    network_serial=NETWORK_SERIAL,
                    verbose=True
                )
                
                # Test connection
                self.robot.get_info()
                self.is_connected = True
                logger.info("Successfully connected to robot")
                
                # Get current positions for all axes
                positions = self.robot.get_robot_positions()  # [gripper, elbow, shoulder, z-axis]
                self.current_gripper_position = self.robot.counts_to_rad(self.robot.GRIPPER, positions[0])
                self.current_elbow_position = self.robot.counts_to_rad(self.robot.ELBOW, positions[1])
                self.current_shoulder_position = self.robot.counts_to_rad(self.robot.SHOULDER, positions[2])
                self.current_z_position = self.robot.counts_to_mm(self.robot.Z_AXIS, positions[3])
                
                elbow_counts = self.robot.get_axis_position(self.robot.ELBOW)
                self.current_elbow_angle = self.robot.counts_to_rad(self.robot.ELBOW, elbow_counts)
                
                shoulder_counts = self.robot.get_axis_position(self.robot.SHOULDER)
                self.current_shoulder_angle = self.robot.counts_to_rad(self.robot.SHOULDER, shoulder_counts)
                
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
                self.elbow_angle_rad = 0.0
                self.shoulder_angle_rad = 0.0
                self.gripper_angle_rad = 0.0
                self.gripper_is_open = False  # Gripper starts closed
                self.is_homed = False
                
            def home_robot(self, wait=True):
                logger.info("SIMULATION: Homing robot")
                time.sleep(0.5)  # Simulate homing time
                self.z_position_mm = Z_AXIS_MAX_MM  # Home position is at top
                self.elbow_angle_rad = 0.0  # Home elbow to center
                self.shoulder_angle_rad = 0.0  # Home shoulder to center
                self.gripper_angle_rad = 0.0  # Home gripper to center
                self.gripper_is_open = False  # Gripper closes on home
                self.is_homed = True
                return True
                
            def move_z(self, mm, wait=True):
                logger.info(f"SIMULATION: Moving Z-axis to {mm:.1f}mm")
                if mm < Z_AXIS_MIN_MM or mm > Z_AXIS_MAX_MM:
                    raise ValueError(f"Z position {mm:.1f}mm is out of range [{Z_AXIS_MIN_MM}-{Z_AXIS_MAX_MM}]")
                time.sleep(0.2)  # Simulate movement time
                self.z_position_mm = mm
                
            def move_axis_rad(self, axis, rad, wait=True):
                logger.info(f"SIMULATION: Moving axis {axis} to {rad:.2f} radians")
                time.sleep(0.2)  # Simulate movement time
                if axis == self.SHOULDER:
                    if rad < SHOULDER_MIN_RAD or rad > SHOULDER_MAX_RAD:
                        raise ValueError(f"Shoulder position {rad:.2f}rad is out of range [{SHOULDER_MIN_RAD:.2f}-{SHOULDER_MAX_RAD:.2f}]")
                    self.shoulder_angle_rad = rad
                elif axis == self.ELBOW:
                    if rad < ELBOW_MIN_RAD or rad > ELBOW_MAX_RAD:
                        raise ValueError(f"Elbow position {rad:.2f}rad is out of range [{ELBOW_MIN_RAD:.2f}-{ELBOW_MAX_RAD:.2f}]")
                    self.elbow_angle_rad = rad
                elif axis == self.GRIPPER:
                    if rad < GRIPPER_MIN_RAD or rad > GRIPPER_MAX_RAD:
                        raise ValueError(f"Gripper position {rad:.2f}rad is out of range [{GRIPPER_MIN_RAD:.2f}-{GRIPPER_MAX_RAD:.2f}]")
                    self.gripper_angle_rad = rad
                    
            def goto(self, position_list, wait=True):
                """Move to absolute position using counts [gripper, elbow, shoulder, z]."""
                logger.info(f"SIMULATION: Moving to position {position_list}")
                time.sleep(0.5)  # Simulate movement time
                
                # Convert counts to internal values (simplified conversion)
                self.gripper_angle_rad = position_list[0] / 1000.0  # Mock conversion
                self.elbow_angle_rad = position_list[1] / 1000.0   # Mock conversion  
                self.shoulder_angle_rad = position_list[2] / 1000.0 # Mock conversion
                self.z_position_mm = position_list[3] / 100.0      # Mock conversion (100 counts = 1mm)
                
                # Apply limits
                self.gripper_angle_rad = max(GRIPPER_MIN_RAD, min(GRIPPER_MAX_RAD, self.gripper_angle_rad))
                self.elbow_angle_rad = max(ELBOW_MIN_RAD, min(ELBOW_MAX_RAD, self.elbow_angle_rad))
                self.shoulder_angle_rad = max(SHOULDER_MIN_RAD, min(SHOULDER_MAX_RAD, self.shoulder_angle_rad))
                self.z_position_mm = max(Z_AXIS_MIN_MM, min(Z_AXIS_MAX_MM, self.z_position_mm))
                
            def open_gripper(self):
                logger.info("SIMULATION: Opening gripper")
                self.gripper_is_open = True
                
            def close_gripper(self):
                logger.info("SIMULATION: Closing gripper")
                self.gripper_is_open = False
                
            def get_axis_position(self, axis):
                if axis == self.Z_AXIS:
                    return int(self.z_position_mm * 100)  # Mock conversion to counts
                elif axis == self.ELBOW:
                    return int(self.elbow_angle_rad * 1000)  # Mock conversion to counts
                elif axis == self.SHOULDER:
                    return int(self.shoulder_angle_rad * 1000)  # Mock conversion to counts
                elif axis == self.GRIPPER:
                    return int(self.gripper_angle_rad * 1000)  # Mock conversion to counts
                return 0
                
            def get_robot_positions(self):
                # Return [gripper, elbow, shoulder, z-axis] positions in counts
                return [
                    int(self.gripper_angle_rad * 1000),  # Mock conversion
                    int(self.elbow_angle_rad * 1000),
                    int(self.shoulder_angle_rad * 1000), 
                    int(self.z_position_mm * 100)
                ]
                
            def counts_to_mm(self, axis, counts):
                return counts / 100.0  # Mock conversion from counts
                
            def counts_to_rad(self, axis, counts):
                return counts / 1000.0  # Mock conversion from counts to radians
                
            def get_info(self):
                logger.info("SIMULATION: Robot info - Mock North C9 Controller")
        
        return MockRobot()
    
    def create_gui(self):
        """Create the main GUI window."""
        self.root = tk.Tk()
        self.root.title("North Robot Arm Position Control")
        self.root.geometry("750x850")
        self.root.resizable(False, False)
        
        # Set up the GUI layout
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="North Robot Arm Control", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.status_label = ttk.Label(status_frame, text="Disconnected", foreground="red")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Position Selection section
        position_select_frame = ttk.LabelFrame(main_frame, text="Workflow Positions", padding="10")
        position_select_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        ttk.Label(position_select_frame, text="Select Position:").grid(row=0, column=0, sticky=tk.W, pady=2)
        
        # Position dropdown
        position_names = list(WORKFLOW_POSITIONS.keys())
        self.position_dropdown = ttk.Combobox(position_select_frame, values=position_names, 
                                            state="readonly", width=40)
        self.position_dropdown.grid(row=0, column=1, padx=(10, 10), pady=2)
        self.position_dropdown.bind('<<ComboboxSelected>>', self.on_position_selected)
        
        # Go to position button
        goto_button = ttk.Button(position_select_frame, text="Go To Position", 
                                command=self.goto_selected_position)
        goto_button.grid(row=0, column=2, padx=(0, 0), pady=2)
        
        # Position description
        self.position_description_label = ttk.Label(position_select_frame, text="Select a position to see description",
                                                   font=("Arial", 9), foreground="gray")
        self.position_description_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Position section
        position_frame = ttk.LabelFrame(main_frame, text="Joint Positions", padding="10")
        position_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
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
        
        # Gripper status (open/closed)
        self.gripper_status_label = ttk.Label(position_frame, text="Gripper: Closed", 
                                             font=("Arial", 10, "bold"))
        self.gripper_status_label.grid(row=4, column=0, sticky=tk.W)
        
        # Movement increment settings
        increment_frame = ttk.LabelFrame(main_frame, text="Movement Settings", padding="10")
        increment_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Z-axis increment setting
        z_inc_frame = ttk.Frame(increment_frame)
        z_inc_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(z_inc_frame, text="Z-Axis Step (mm):").grid(row=0, column=0, sticky=tk.W)
        self.z_increment_entry = ttk.Entry(z_inc_frame, width=8)
        self.z_increment_entry.grid(row=0, column=1, padx=(10, 5))
        self.z_increment_entry.insert(0, str(DEFAULT_MOVE_INCREMENT_MM))
        
        z_update_button = ttk.Button(z_inc_frame, text="Set", command=self.update_z_increment)
        z_update_button.grid(row=0, column=2, padx=2)
        
        # Rotational increment setting
        rad_inc_frame = ttk.Frame(increment_frame)
        rad_inc_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(rad_inc_frame, text="Rotation Step (rad):").grid(row=0, column=0, sticky=tk.W)
        self.rad_increment_entry = ttk.Entry(rad_inc_frame, width=8)
        self.rad_increment_entry.grid(row=0, column=1, padx=(10, 5))
        self.rad_increment_entry.insert(0, str(DEFAULT_MOVE_INCREMENT_RAD))
        
        rad_update_button = ttk.Button(rad_inc_frame, text="Set", command=self.update_rad_increment)
        rad_update_button.grid(row=0, column=2, padx=2)
        
        # Show degrees equivalent
        self.rad_degrees_label = ttk.Label(increment_frame, text=f"({DEFAULT_MOVE_INCREMENT_RAD * 180/3.14159:.1f}°)", 
                                          font=("Arial", 8), foreground="gray")
        self.rad_degrees_label.grid(row=1, column=1, sticky=tk.W, padx=(120, 0))
        
        # Status for increment changes
        self.increment_status_label = ttk.Label(increment_frame, text="Ready", foreground="green")
        self.increment_status_label.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        
        # Control buttons
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.home_button = ttk.Button(control_frame, text="🏠 HOME ROBOT", 
                                     command=self.home_robot, style="Accent.TButton")
        self.home_button.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        # Z-Axis movement buttons
        z_frame = ttk.Frame(control_frame)
        z_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(z_frame, text="Z-Axis:").grid(row=0, column=0, pady=2)
        
        up_button = ttk.Button(z_frame, text="▲ UP (5mm)", 
                              command=self.move_up)
        up_button.grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        
        down_button = ttk.Button(z_frame, text="▼ DOWN (5mm)", 
                                command=self.move_down)
        down_button.grid(row=0, column=2, padx=2, sticky=(tk.W, tk.E))
        
        # Elbow movement buttons
        elbow_frame = ttk.Frame(control_frame)
        elbow_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(elbow_frame, text="Elbow:").grid(row=0, column=0, pady=2)
        
        elbow_up_button = ttk.Button(elbow_frame, text="↗ EXTEND", 
                                    command=self.move_elbow_extend)
        elbow_up_button.grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        
        elbow_down_button = ttk.Button(elbow_frame, text="↙ RETRACT", 
                                      command=self.move_elbow_retract)
        elbow_down_button.grid(row=0, column=2, padx=2, sticky=(tk.W, tk.E))
        
        # Shoulder movement buttons
        shoulder_frame = ttk.Frame(control_frame)
        shoulder_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(shoulder_frame, text="Shoulder:").grid(row=0, column=0, pady=2)
        
        shoulder_left_button = ttk.Button(shoulder_frame, text="← LEFT", 
                                         command=self.move_shoulder_left)
        shoulder_left_button.grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        
        shoulder_right_button = ttk.Button(shoulder_frame, text="→ RIGHT", 
                                          command=self.move_shoulder_right)
        shoulder_right_button.grid(row=0, column=2, padx=2, sticky=(tk.W, tk.E))
        
        # Gripper movement buttons
        gripper_frame = ttk.Frame(control_frame)
        gripper_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(gripper_frame, text="Gripper:").grid(row=0, column=0, pady=2)
        
        gripper_ccw_button = ttk.Button(gripper_frame, text="↺ CCW", 
                                       command=self.move_gripper_ccw)
        gripper_ccw_button.grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        
        gripper_cw_button = ttk.Button(gripper_frame, text="↻ CW", 
                                      command=self.move_gripper_cw)
        gripper_cw_button.grid(row=0, column=2, padx=2, sticky=(tk.W, tk.E))
        
        # Gripper Open/Close buttons
        gripper_action_frame = ttk.Frame(control_frame)
        gripper_action_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(gripper_action_frame, text="Actions:").grid(row=0, column=0, pady=2)
        
        gripper_open_button = ttk.Button(gripper_action_frame, text="✋ OPEN", 
                                        command=self.open_gripper,
                                        style="Success.TButton")
        gripper_open_button.grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        
        gripper_close_button = ttk.Button(gripper_action_frame, text="👊 CLOSE", 
                                         command=self.close_gripper,
                                         style="Warning.TButton")
        gripper_close_button.grid(row=0, column=2, padx=2, sticky=(tk.W, tk.E))
        
        # Configure grid weights for button frames
        z_frame.columnconfigure(1, weight=1)
        z_frame.columnconfigure(2, weight=1)
        elbow_frame.columnconfigure(1, weight=1)
        elbow_frame.columnconfigure(2, weight=1)
        shoulder_frame.columnconfigure(1, weight=1)
        shoulder_frame.columnconfigure(2, weight=1)
        gripper_frame.columnconfigure(1, weight=1)
        gripper_frame.columnconfigure(2, weight=1)
        gripper_action_frame.columnconfigure(1, weight=1)
        gripper_action_frame.columnconfigure(2, weight=1)
        
        # Instructions
        instruction_frame = ttk.LabelFrame(main_frame, text="Keyboard Controls", padding="10")
        instruction_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        instructions = [
            "↑/↓ Arrows: Move Z-axis up/down (current step shown above)",
            "←/→ Arrows: Rotate shoulder left/right (current step shown above)",
            "W/S Keys: Extend/Retract elbow (current step shown above)",
            "Q/E Keys: Rotate gripper CCW/CW (current step shown above)",
            "O/C Keys: Open/Close gripper teeth",
            "SPACE: Home robot",
            "ESC: Exit program"
        ]
        
        for i, instruction in enumerate(instructions):
            ttk.Label(instruction_frame, text=instruction).grid(row=i, column=0, sticky=tk.W, pady=2)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        
        # Bind keyboard events
        self.root.bind('<Key>', self.on_key_press)
        self.root.focus_set()
        
        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure button styles
        style = ttk.Style()
        style.configure("Accent.TButton", foreground="blue")
        style.configure("Success.TButton", foreground="green")
        style.configure("Warning.TButton", foreground="orange")
        
    def update_display(self):
        """Update the position and status display."""
        if self.is_connected:
            try:
                # Get current positions
                if hasattr(self.robot, 'z_position_mm'):  # Mock robot
                    self.current_z_position = self.robot.z_position_mm
                    self.current_elbow_angle = self.robot.elbow_angle_rad
                    self.current_shoulder_angle = self.robot.shoulder_angle_rad
                    self.current_gripper_angle = self.robot.gripper_angle_rad
                    self.gripper_is_open = self.robot.gripper_is_open
                    self.is_homed = self.robot.is_homed
                else:  # Real robot
                    positions = self.robot.get_robot_positions()  # [gripper, elbow, shoulder, z-axis]
                    self.current_gripper_angle = self.robot.counts_to_rad(self.robot.GRIPPER, positions[0])
                    self.current_elbow_angle = self.robot.counts_to_rad(self.robot.ELBOW, positions[1])
                    self.current_shoulder_angle = self.robot.counts_to_rad(self.robot.SHOULDER, positions[2])
                    self.current_z_position = self.robot.counts_to_mm(self.robot.Z_AXIS, positions[3])
                    
                    # For real robot, we need to track gripper state separately
                    # since the API doesn't provide a way to query open/closed state
                
                # Update all position labels
                self.z_position_label.config(text=f"Z-Axis: {self.current_z_position:.1f} mm")
                self.elbow_position_label.config(text=f"Elbow: {self.current_elbow_angle:.2f} rad ({self.current_elbow_angle*180/3.14159:.1f}°)")
                self.shoulder_position_label.config(text=f"Shoulder: {self.current_shoulder_angle:.2f} rad ({self.current_shoulder_angle*180/3.14159:.1f}°)")
                self.gripper_position_label.config(text=f"Gripper: {self.current_gripper_angle:.2f} rad ({self.current_gripper_angle*180/3.14159:.1f}°)")
                
                # Update gripper status
                if self.gripper_is_open:
                    self.gripper_status_label.config(text="Gripper: OPEN ✋", foreground="green")
                else:
                    self.gripper_status_label.config(text="Gripper: CLOSED 👊", foreground="red")
                
                if self.is_homed:
                    self.status_label.config(text="Connected & Homed", foreground="green")
                else:
                    self.status_label.config(text="Connected - Not Homed", foreground="orange")
                    
            except Exception as e:
                logger.error(f"Error updating display: {str(e)}")
                self.status_label.config(text="Error reading position", foreground="red")
        else:
            self.z_position_label.config(text="Z-Axis: Unknown")
            self.elbow_position_label.config(text="Elbow: Unknown")
            self.shoulder_position_label.config(text="Shoulder: Unknown")
            self.gripper_position_label.config(text="Gripper: Unknown")
            self.gripper_status_label.config(text="Gripper: Unknown", foreground="gray")
            self.status_label.config(text="Disconnected", foreground="red")
    
    def home_robot(self):
        """Home the robot to its reference position."""
        if not self.is_connected:
            messagebox.showerror("Error", "Robot not connected")
            return
            
        try:
            logger.info("Homing robot...")
            self.home_button.config(text="Homing...", state="disabled")
            self.root.update()
            
            self.robot.home_robot(wait=True)
            self.is_homed = True
            # Home operation typically closes the gripper
            self.gripper_is_open = False
            
            logger.info("Robot homed successfully")
            messagebox.showinfo("Success", "Robot homed successfully!")
            
        except Exception as e:
            logger.error(f"Error homing robot: {str(e)}")
            messagebox.showerror("Error", f"Failed to home robot:\n{str(e)}")
        finally:
            self.home_button.config(text="🏠 HOME ROBOT", state="normal")
            self.update_display()
    
    def update_z_increment(self):
        """Update the Z-axis movement increment from user input."""
        try:
            value = float(self.z_increment_entry.get())
            if value < MIN_MOVE_INCREMENT_MM:
                self.increment_status_label.config(text=f"Z increment too small (min: {MIN_MOVE_INCREMENT_MM}mm)", foreground="red")
                return
            if value > MAX_MOVE_INCREMENT_MM:
                self.increment_status_label.config(text=f"Z increment too large (max: {MAX_MOVE_INCREMENT_MM}mm)", foreground="red")
                return
            
            self.move_increment_mm = value
            self.increment_status_label.config(text=f"Z-axis step set to {value:.1f}mm", foreground="green")
            logger.info(f"Z-axis movement increment updated to {value:.1f}mm")
            
        except ValueError:
            self.increment_status_label.config(text="Invalid Z increment value", foreground="red")
    
    def update_rad_increment(self):
        """Update the rotational movement increment from user input."""
        try:
            value = float(self.rad_increment_entry.get())
            if value < MIN_MOVE_INCREMENT_RAD:
                self.increment_status_label.config(text=f"Rotation increment too small (min: {MIN_MOVE_INCREMENT_RAD:.3f}rad)", foreground="red")
                return
            if value > MAX_MOVE_INCREMENT_RAD:
                self.increment_status_label.config(text=f"Rotation increment too large (max: {MAX_MOVE_INCREMENT_RAD:.1f}rad)", foreground="red")
                return
            
            self.move_increment_rad = value
            degrees = value * 180 / 3.14159
            self.rad_degrees_label.config(text=f"({degrees:.1f}°)")
            self.increment_status_label.config(text=f"Rotation step set to {value:.3f}rad ({degrees:.1f}°)", foreground="green")
            logger.info(f"Rotational movement increment updated to {value:.3f}rad ({degrees:.1f}°)")
            
        except ValueError:
            self.increment_status_label.config(text="Invalid rotation increment value", foreground="red")
    
    def move_up(self):
        """Move the arm up by the configured increment."""
        self._move_z_relative(self.move_increment_mm)
    
    def move_down(self):
        """Move the arm down by the configured increment."""
        self._move_z_relative(-self.move_increment_mm)
    
    def move_shoulder_left(self):
        """Move the shoulder joint left (counter-clockwise)."""
        self._move_joint_relative("shoulder", MOVE_INCREMENT_RAD)
    
    def move_shoulder_right(self):
        """Move the shoulder joint right (clockwise)."""
        self._move_joint_relative("shoulder", -MOVE_INCREMENT_RAD)
    
    def move_elbow_in(self):
        """Move the elbow joint inward."""
        self._move_joint_relative("elbow", MOVE_INCREMENT_RAD)
    
    def move_elbow_out(self):
        """Move the elbow joint outward."""
        self._move_joint_relative("elbow", -MOVE_INCREMENT_RAD)
    
    def move_gripper_ccw(self):
        """Rotate the gripper counter-clockwise."""
        self._move_joint_relative("gripper", MOVE_INCREMENT_RAD)
    
    def move_gripper_cw(self):
        """Rotate the gripper clockwise."""
        self._move_joint_relative("gripper", -MOVE_INCREMENT_RAD)
    
    def move_elbow_extend(self):
        """Extend the elbow joint."""
        self._move_elbow_relative(MOVE_INCREMENT_RAD)
    
    def move_elbow_retract(self):
        """Retract the elbow joint."""
        self._move_elbow_relative(-MOVE_INCREMENT_RAD)
    
    def move_shoulder_left(self):
        """Rotate shoulder left."""
        self._move_shoulder_relative(-MOVE_INCREMENT_RAD)
    
    def move_shoulder_right(self):
        """Rotate shoulder right."""
        self._move_shoulder_relative(MOVE_INCREMENT_RAD)
    
    def _move_z_relative(self, delta_mm):
        """Move the Z-axis by a relative amount."""
        if not self.is_connected:
            messagebox.showerror("Error", "Robot not connected")
            return
            
        if not self.is_homed:
            messagebox.showwarning("Warning", "Robot must be homed before moving")
            return
            
        try:
            new_position = self.current_z_position + delta_mm
            
            # Safety check
            if new_position < Z_AXIS_MIN_MM:
                messagebox.showwarning("Movement Not Allowed", 
                    f"Cannot move Z-axis to {new_position:.1f}mm\n" +
                    f"Minimum limit: {Z_AXIS_MIN_MM}mm")
                return
            if new_position > Z_AXIS_MAX_MM:
                messagebox.showwarning("Movement Not Allowed", 
                    f"Cannot move Z-axis to {new_position:.1f}mm\n" +
                    f"Maximum limit: {Z_AXIS_MAX_MM}mm")
                return
            
            logger.info(f"Moving Z-axis to {new_position:.1f}mm")
            self.robot.move_z(new_position, wait=True)
            
            self.update_display()
            
        except Exception as e:
            logger.error(f"Error moving robot: {str(e)}")
            messagebox.showerror("Error", f"Failed to move robot:\n{str(e)}")
    
    def _move_joint_relative(self, joint_name, delta_rad):
        """Move a rotational joint by a relative amount."""
        if not self.is_connected:
            messagebox.showerror("Error", "Robot not connected")
            return
            
        if not self.is_homed:
            messagebox.showwarning("Warning", "Robot must be homed before moving")
            return
            
        try:
            # Get current position and calculate new position
            if joint_name == "shoulder":
                current_pos = self.current_shoulder_angle
                new_position = current_pos + delta_rad
                min_pos, max_pos = SHOULDER_MIN_RAD, SHOULDER_MAX_RAD
                axis = self.robot.SHOULDER
            elif joint_name == "elbow":
                current_pos = self.current_elbow_angle
                new_position = current_pos + delta_rad
                min_pos, max_pos = ELBOW_MIN_RAD, ELBOW_MAX_RAD
                axis = self.robot.ELBOW
            elif joint_name == "gripper":
                current_pos = self.current_gripper_angle
                new_position = current_pos + delta_rad
                min_pos, max_pos = GRIPPER_MIN_RAD, GRIPPER_MAX_RAD
                axis = self.robot.GRIPPER
            else:
                logger.error(f"Unknown joint name: {joint_name}")
                return
            
            # Safety check
            if new_position < min_pos:
                degrees_min = min_pos * 180 / 3.14159
                degrees_new = new_position * 180 / 3.14159
                messagebox.showwarning("Movement Not Allowed", 
                    f"Cannot move {joint_name} to {new_position:.2f}rad ({degrees_new:.1f}°)\n" +
                    f"Minimum limit: {min_pos:.2f}rad ({degrees_min:.1f}°)")
                return
            if new_position > max_pos:
                degrees_max = max_pos * 180 / 3.14159
                degrees_new = new_position * 180 / 3.14159
                messagebox.showwarning("Movement Not Allowed", 
                    f"Cannot move {joint_name} to {new_position:.2f}rad ({degrees_new:.1f}°)\n" +
                    f"Maximum limit: {max_pos:.2f}rad ({degrees_max:.1f}°)")
                return
            
            logger.info(f"Moving {joint_name} to {new_position:.2f} radians")
            self.robot.move_axis_rad(axis, new_position, wait=True)
            
            self.update_display()
            
        except Exception as e:
            logger.error(f"Error moving {joint_name}: {str(e)}")
            messagebox.showerror("Error", f"Failed to move {joint_name}:\n{str(e)}")
    
    def _move_elbow_relative(self, delta_rad):
        """Move the elbow by a relative amount."""
        if not self.is_connected:
            messagebox.showerror("Error", "Robot not connected")
            return
            
        if not self.is_homed:
            messagebox.showwarning("Warning", "Robot must be homed before moving")
            return
            
        try:
            new_angle = self.current_elbow_angle + delta_rad
            
            # Safety check
            if new_angle < ELBOW_MIN_RAD:
                messagebox.showwarning("Warning", f"Cannot move elbow below {ELBOW_MIN_RAD:.2f} radians")
                return
            if new_angle > ELBOW_MAX_RAD:
                messagebox.showwarning("Warning", f"Cannot move elbow above {ELBOW_MAX_RAD:.2f} radians")
                return
            
            logger.info(f"Moving elbow to {new_angle:.2f} radians")
            self.robot.move_axis_rad(self.robot.ELBOW, new_angle, wait=True)
            
            self.update_display()
            
        except Exception as e:
            logger.error(f"Error moving elbow: {str(e)}")
            messagebox.showerror("Error", f"Failed to move elbow:\n{str(e)}")
    
    def _move_shoulder_relative(self, delta_rad):
        """Move the shoulder by a relative amount."""
        if not self.is_connected:
            messagebox.showerror("Error", "Robot not connected")
            return
            
        if not self.is_homed:
            messagebox.showwarning("Warning", "Robot must be homed before moving")
            return
            
        try:
            new_angle = self.current_shoulder_angle + delta_rad
            
            # Safety check
            if new_angle < SHOULDER_MIN_RAD:
                messagebox.showwarning("Warning", f"Cannot move shoulder below {SHOULDER_MIN_RAD:.2f} radians")
                return
            if new_angle > SHOULDER_MAX_RAD:
                messagebox.showwarning("Warning", f"Cannot move shoulder above {SHOULDER_MAX_RAD:.2f} radians")
                return
            
            logger.info(f"Moving shoulder to {new_angle:.2f} radians")
            self.robot.move_axis_rad(self.robot.SHOULDER, new_angle, wait=True)
            
            self.update_display()
            
        except Exception as e:
            logger.error(f"Error moving shoulder: {str(e)}")
            messagebox.showerror("Error", f"Failed to move shoulder:\n{str(e)}")
    
    def move_shoulder_left(self):
        """Move the shoulder joint left (counter-clockwise)."""
        self._move_shoulder_relative(self.move_increment_rad)
    
    def move_shoulder_right(self):
        """Move the shoulder joint right (clockwise)."""
        self._move_shoulder_relative(-self.move_increment_rad)
    
    def move_elbow_extend(self):
        """Extend the elbow joint outward."""
        self._move_elbow_relative(self.move_increment_rad)
    
    def move_elbow_retract(self):
        """Retract the elbow joint inward."""
        self._move_elbow_relative(-self.move_increment_rad)
    
    def move_gripper_ccw(self):
        """Rotate gripper counter-clockwise."""
        self._move_joint_relative("gripper", self.move_increment_rad)
    
    def move_gripper_cw(self):
        """Rotate gripper clockwise."""
        self._move_joint_relative("gripper", -self.move_increment_rad)
    
    def open_gripper(self):
        """Open the gripper teeth."""
        if not self.is_connected:
            messagebox.showerror("Error", "Robot not connected")
            return
            
        if not self.is_homed:
            messagebox.showwarning("Warning", "Robot must be homed before opening gripper")
            return
            
        if self.gripper_is_open:
            messagebox.showinfo("Info", "Gripper is already open")
            return
            
        try:
            logger.info("Opening gripper...")
            self.robot.open_gripper()
            self.gripper_is_open = True
            logger.info("Gripper opened successfully")
            self.update_display()
            
        except Exception as e:
            logger.error(f"Error opening gripper: {str(e)}")
            messagebox.showerror("Error", f"Failed to open gripper:\n{str(e)}")
    
    def close_gripper(self):
        """Close the gripper teeth."""
        if not self.is_connected:
            messagebox.showerror("Error", "Robot not connected")
            return
            
        if not self.is_homed:
            messagebox.showwarning("Warning", "Robot must be homed before closing gripper")
            return
            
        if not self.gripper_is_open:
            messagebox.showinfo("Info", "Gripper is already closed")
            return
            
        try:
            logger.info("Closing gripper...")
            self.robot.close_gripper()
            self.gripper_is_open = False
            logger.info("Gripper closed successfully")
            self.update_display()
            
        except Exception as e:
            logger.error(f"Error closing gripper: {str(e)}")
            messagebox.showerror("Error", f"Failed to close gripper:\n{str(e)}")
    
    def on_position_selected(self, event):
        """Handle position dropdown selection."""
        position_name = self.position_dropdown.get()
        if position_name in WORKFLOW_POSITIONS:
            position_data = WORKFLOW_POSITIONS[position_name]
            self.position_description_label.config(text=position_data['description'])
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
            position_data = WORKFLOW_POSITIONS[self.selected_position]
            
            logger.info(f"Moving to position: {self.selected_position}")
            
            # For home position, use the home command
            if self.selected_position == "Home":
                self.robot.home_robot(wait=True)
                logger.info(f"Successfully homed robot to {self.selected_position}")
                messagebox.showinfo("Success", f"Homed robot")
            else:
                # Move to specific position using counts (absolute positioning)
                target_position = [
                    position_data["gripper_cts"],
                    position_data["elbow_cts"], 
                    position_data["shoulder_cts"],
                    position_data["z_cts"]
                ]
                
                self.robot.goto(target_position, wait=True)
                logger.info(f"Successfully moved to {self.selected_position}")
                messagebox.showinfo("Success", f"Moved to {self.selected_position}")
            
        except Exception as e:
            logger.error(f"Error moving to position: {str(e)}")
            messagebox.showerror("Error", f"Failed to move to position:\n{str(e)}")
        finally:
            self.update_display()
    
    def on_key_press(self, event):
        """Handle keyboard input for robot control."""
        key = event.keysym
        
        if key == 'Up':
            self.move_up()
        elif key == 'Down':
            self.move_down()
        elif key == 'Left':
            self.move_shoulder_left()
        elif key == 'Right':
            self.move_shoulder_right()
        elif key == 'w' or key == 'W':
            self.move_elbow_extend()
        elif key == 's' or key == 'S':
            self.move_elbow_retract()
        elif key == 'q' or key == 'Q':
            self.move_gripper_ccw()
        elif key == 'e' or key == 'E':
            self.move_gripper_cw()
        elif key == 'o' or key == 'O':
            self.open_gripper()
        elif key == 'c' or key == 'C':
            self.close_gripper()
        elif key == 'Escape':
            self.on_closing()
        elif key == 'space':
            self.home_robot()
    
    def on_closing(self):
        """Handle application closing."""
        try:
            logger.info("Closing application...")
            if self.is_connected and hasattr(self.robot, 'network'):
                try:
                    self.robot.network.disconnect()
                except:
                    pass
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        finally:
            self.root.destroy()
    
    def run(self):
        """Main application entry point."""
        logger.info("Starting North Robot Arm Control Program")
        
        # Create GUI
        self.create_gui()
        
        # Connect to robot
        if self.connect_robot():
            logger.info("Robot connection successful")
        else:
            logger.error("Failed to connect to robot")
            # Still show GUI even if connection failed
        
        # Update initial display
        self.update_display()
        
        # Start GUI event loop
        try:
            logger.info("Starting GUI...")
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Program interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
        finally:
            logger.info("Program ended")


def main():
    """Program entry point."""
    try:
        app = RobotArmController()
        app.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

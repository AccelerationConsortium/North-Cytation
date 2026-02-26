from robot_state.Locator import * #Let's try to eliminate this in the future
import robot_state.Locator as Locator
import numpy as np
import time
import math
import pandas as pd
import yaml
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from pipetting_data.pipetting_parameters import PipettingParameters, ReservoirParameters
from pipetting_data.pipetting_wizard import PipettingWizard
import matplotlib.patches as patches
import threading

class North_Base:
    """Base class for North robot components with shared functionality"""
    
    def pause_after_error(self, err_message, send_slack=True):
        """Pause execution after an error with logging and optional Slack notification"""
        if not self.simulate:
            import slack_agent
        
        self.logger.error(err_message)
        if send_slack and not self.simulate:
            try:
                slack_agent.send_slack_message(err_message)
            except Exception as e:
                self.logger.error(f"Failed to send Slack message: {e}")
        
        if not self.simulate:
            input(f"Error: {err_message}. Press Enter to continue...")
        else:
            self.logger.warning(f"SIMULATION MODE: Would pause for error: {err_message}")
    
    def _load_yaml_file(self, file_path, description, required=True, convert_none=False):
        """
        Unified YAML file loader with consistent error handling
        
        Args:
            file_path (str): Path to the YAML file
            description (str): Human-readable description for error messages
            required (bool): Whether the file is required (default: True)
            convert_none (bool): Whether to convert "None"/"null" strings to None (default: False)
            
        Returns:
            dict: Loaded YAML data, or None if file not found and not required
        """
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
                
            if convert_none:
                data = self._convert_none_values(data)
                
            self.logger.debug(f"Loaded {description} from {file_path}")
            return data
            
        except FileNotFoundError:
            if required:
                self.pause_after_error(f"{description} file not found: {file_path}")
            else:
                self.logger.warning(f"{description} file not found: {file_path}")
            return None
        except yaml.YAMLError as e:
            self.pause_after_error(f"Error parsing {description} YAML file: {e}")
            return None
        except IOError as e:
            self.pause_after_error(f"Error reading {description} file: {e}")
            return None

    def _convert_none_values(self, value):
        """Convert "None" or "null" strings to actual None values recursively"""
        if isinstance(value, dict):
            return {k: self._convert_none_values(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._convert_none_values(v) for v in value]
        elif value in ["None", "null"]:
            return None
        return value
    
    def _load_wellplate_config(self, file_path="robot_state/wellplates.yaml"):
        """Load wellplate configuration from YAML file using unified method"""
        self.WELLPLATES = self._load_yaml_file(
            file_path, 
            "wellplate properties configuration", 
            required=False, 
            convert_none=True
        )
    
    def get_config_parameter(self, config_name, key, parameter, default=None, error_on_missing=True):
        """
        Unified method to safely get parameters from any configuration dictionary
        
        Args:
            config_name (str): Name of the configuration (for error messages)
            key (str): First-level key in the config (e.g., tip_type, rack_name, location_name)
            parameter (str): Parameter name to retrieve
            default: Default value if parameter not found
            error_on_missing (bool): Whether to pause on missing config (default: True)
            
        Returns:
            Parameter value or default
        """
        config_map = {
            'vial_positions': 'VIAL_POSITIONS',
            'pipet_tips': 'PIPET_TIPS', 
            'pipet_racks': 'PIPET_RACKS',
            'wellplates': 'WELLPLATES',
            'pumps': 'PUMP_CONFIG',
            'robot_hardware': 'ROBOT_HARDWARE'
        }
        
        if config_name not in config_map:
            self.pause_after_error(f"Unknown configuration name: {config_name}")
            return default
            
        config_attr = config_map[config_name]
        config_dict = getattr(self, config_attr, None)
        
        if config_dict is None:
            if error_on_missing:
                self.pause_after_error(f"Cannot access {config_name} parameter '{parameter}' - {config_name} configuration not loaded")
            else:
                self.logger.warning(f"Cannot access {config_name} parameter '{parameter}' - {config_name} configuration not loaded, using default: {default}")
            return default
        
        # Try both integer and string keys since YAML may parse numeric keys as integers
        item_config = config_dict.get(key, {})
        if not item_config:
            item_config = config_dict.get(str(key), {})
        
        return item_config.get(parameter, default)

class North_Track(North_Base):

    # === 1. INITIALIZATION & CONFIGURATION ===
    def __init__(self, c9, simulate = False, logger=None):
        self.c9 = c9
        self.logger = logger
        self.logger.info("Initializing North Track...")

        self.NUM_SOURCE = 0
        self.NUM_WASTE = 0
        self.CURRENT_WP_TYPE = "96 WELL PLATE"
        self.ACTIVE_WELLPLATE_POSITION = None  # null, 'gripper', 'pipetting_area', 'cytation_tray', etc.
        self.CURRENT_GRIPPER_LOCATION = None  # Track where the gripper physically is: 'pipetting_area', 'cytation_tray', etc.
        self.CURRENT_GRIPPER_POSITION = {'x': None, 'z': None}  # Track physical coordinates
        self.simulate = simulate

        #Load yaml data
        self.logger.debug("Loading track status from file: %s", "robot_state/track_status.yaml")
        self.TRACK_STATUS_FILE = "robot_state/track_status.yaml"
        self.get_track_status() #set NUM_SOURCE, NUM_WASTE, CURRENT_WP_TYPE and NR_OCCUPIED from yaml file
        
        # Load track positions configuration
        self.TRACK_POSITIONS_FILE = "robot_state/track_positions.yaml"
        self._load_track_positions()
        
        # Load robot hardware configuration for axis mappings
        self.ROBOT_HARDWARE_FILE = "robot_state/robot_hardware.yaml"
        self._load_robot_hardware()
        
        # Load wellplate configuration using base class method
        self._load_wellplate_config()

        #Bias    
        self.reset_after_initialization()
    
    def _load_track_positions(self):
        """Load track positioning configuration from YAML file"""
        self.logger.debug("Loading track positions from: %s", self.TRACK_POSITIONS_FILE)
        self.TRACK_POSITIONS = self._load_yaml_file(
            self.TRACK_POSITIONS_FILE, 
            "track positions configuration", 
            required=True, 
            convert_none=False
        )
    
    def _load_robot_hardware(self):
        """Load robot hardware configuration from YAML file"""
        self.logger.debug("Loading robot hardware configuration from: %s", self.ROBOT_HARDWARE_FILE)
        self.ROBOT_HARDWARE = self._load_yaml_file(
            self.ROBOT_HARDWARE_FILE, 
            "robot hardware configuration", 
            required=True, 
            convert_none=False
        )

    def get_track_status(self):
        self.logger.debug("Getting track status from file: %s", self.TRACK_STATUS_FILE)
        """Get the track status from the yaml file."""
        
        # Load track status using unified method from North_Base
        track_status = self._load_yaml_file(
            self.TRACK_STATUS_FILE, 
            "track status", 
            required=True, 
            convert_none=True
        )
        
        if track_status is None:
            return

        try:
            self.NUM_SOURCE = track_status['num_in_source']
            self.NUM_WASTE = track_status['num_in_waste']
            self.CURRENT_WP_TYPE = track_status['wellplate_type']
            self.ACTIVE_WELLPLATE_POSITION = track_status.get('active_wellplate_position')  # New unified field
            self.CURRENT_GRIPPER_LOCATION = track_status.get('current_gripper_location')  # Track gripper location name
            self.CURRENT_GRIPPER_POSITION = track_status.get('current_gripper_position', {'x': None, 'z': None})  # Track gripper coordinates
        except KeyError as e:
            self.pause_after_error(f"Missing required field in track status: {e}", False)
        except Exception as e:
            self.pause_after_error(f"Issue processing track status: {e}", False)

    def save_track_status(self):
        #self.logger.debug("Saving track status to file: %s", self.TRACK_STATUS_FILE)
        track_status = {
            "num_in_source": self.NUM_SOURCE,
            "num_in_waste": self.NUM_WASTE,
            "wellplate_type": self.CURRENT_WP_TYPE,
            "active_wellplate_position": self.ACTIVE_WELLPLATE_POSITION,
            "current_gripper_location": self.CURRENT_GRIPPER_LOCATION,
            "current_gripper_position": self.CURRENT_GRIPPER_POSITION
        }

        if not self.simulate: #not simulating
            # Writing to a file
            with open(self.TRACK_STATUS_FILE, "w") as file:
                yaml.dump(track_status, file, default_flow_style=False)

    def check_input_file(self, pause_after_check=True, visualize=True):
        print("\n" + "="*60)
        print("DEPRECATION WARNING: check_input_file()")
        print("="*60)
        print("This method is deprecated. Please use the GUI launched during")
        print("Lash_E initialization to check and modify robot states.")
        print("Remove the .check_input_file() calls from your workflow.")
        print("Press Enter to continue...")
        print("="*60)
        input()
        
        return
        # self.logger.info(f"--Wellplate status-- \n Wellplate type: {self.CURRENT_WP_TYPE} \n Number in source: {self.NUM_SOURCE} \n Number in waste: {self.NUM_WASTE} \n Active wellplate position: {self.ACTIVE_WELLPLATE_POSITION}")

        # if visualize:
        #     self.logger.info("Visualizing wellplate status...")
            
        #     # Debug matplotlib backend
        #     try:
        #         import matplotlib
        #         self.logger.debug(f"Matplotlib backend: {matplotlib.get_backend()}")
                
        #         # Try to set an interactive backend for Windows
        #         if matplotlib.get_backend() == 'Agg':
        #             try:
        #                 matplotlib.use('TkAgg')
        #                 self.logger.info("Switched matplotlib backend to TkAgg")
        #             except:
        #                 try:
        #                     matplotlib.use('Qt5Agg')
        #                     self.logger.info("Switched matplotlib backend to Qt5Agg") 
        #                 except:
        #                     self.logger.warning("Could not set interactive backend, using default")
                
        #     except Exception as e:
        #         self.logger.warning(f"Matplotlib backend check failed: {e}")
            
        #     fig, ax = plt.subplots(figsize=(10, 6))

        #     plate_width = 2.5
        #     plate_height = 0.4
        #     spacing = 0.1

        #     for i in range(self.NUM_SOURCE):
        #         rect = plt.Rectangle((1, i * (plate_height + spacing)), plate_width, plate_height,
        #                             edgecolor='black', facecolor='lightblue')
        #         ax.add_patch(rect)
        #         ax.text(1 + plate_width / 2, i * (plate_height + spacing) + plate_height / 2,
        #                 self.CURRENT_WP_TYPE, ha='center', va='center', fontsize=8)

        #     for i in range(self.NUM_WASTE):
        #         rect = plt.Rectangle((5, i * (plate_height + spacing)), plate_width, plate_height,
        #                             edgecolor='black', facecolor='lightcoral')
        #         ax.add_patch(rect)
        #         ax.text(5 + plate_width / 2, i * (plate_height + spacing) + plate_height / 2,
        #                 self.CURRENT_WP_TYPE, ha='center', va='center', fontsize=8)

        #     if self.ACTIVE_WELLPLATE_POSITION == 'pipetting_area':
        #         rect = plt.Rectangle((9, 0), plate_width, plate_height,
        #                             edgecolor='black', facecolor='khaki')
        #         ax.add_patch(rect)
        #         ax.text(9 + plate_width / 2, plate_height / 2,
        #                 "Occupied", ha='center', va='center', fontsize=8)
        #         ax.text(9 + plate_width / 2, plate_height + 0.2, "NR Pipette Area",
        #                 ha='center', va='bottom', fontsize=10, weight='bold')

        #     ax.text(1 + plate_width / 2, self.NUM_SOURCE * (plate_height + spacing) + 0.2, "Source Stack",
        #             ha='center', va='bottom', fontsize=10, weight='bold')
        #     ax.text(5 + plate_width / 2, self.NUM_WASTE * (plate_height + spacing) + 0.2, "Waste Stack",
        #             ha='center', va='bottom', fontsize=10, weight='bold')

        #     ax.set_xlim(0, 12)
        #     ax.set_ylim(0, max(self.NUM_SOURCE, self.NUM_WASTE) * (plate_height + spacing) + 1)
        #     ax.axis('off')
        #     ax.set_title("-- Please Confirm Wellplate Status --", fontsize=14, weight='bold')
        #     plt.tight_layout()
            
        #     try:
        #         plt.show(block=True)
        #         self.logger.debug("Wellplate visualization displayed successfully")
        #     except Exception as e:
        #         self.logger.error(f"Failed to display wellplate visualization: {e}")
        #         self.logger.info("Continuing without visualization...")

        # if pause_after_check and not self.simulate:
        #     input("Only hit enter if the status of the well plates is correct, otherwise hit ctrl-c")

    def reset_after_initialization(self):
        """Reset robot to known state after initialization"""
        if self.ACTIVE_WELLPLATE_POSITION == 'gripper':
            # Discard the wellplate currently in gripper
            response = False
            release = False
            while not response:
                text = input("The System reports having a wellplate in the gripper... Please confirm y/n:")
                if text.lower() == 'y':
                    release = True
                    response = True
                elif text.lower() == 'n':
                    release = False
                    response = True
            if release:
                self.release_wellplate_in_location('pipetting_area', waypoint_locations=['cytation_safe_area'])
        self.origin()
     
    # === 2. CONFIGURATION ACCESS METHODS ===  
    def get_position(self, position_name):
        """Get position coordinates by name"""
        positions = self.TRACK_POSITIONS.get('positions', {})
        position = positions.get(position_name)
        if not position:
            self.logger.error(f"Position '{position_name}' not found in track configuration")
            return None
        return position
    
    # === WELLPLATE POSITION TRACKING ===
    def set_wellplate_position(self, position):
        """Set the current wellplate position (null, 'gripper', 'pipetting_area', 'cytation_tray', etc.)"""
        self.ACTIVE_WELLPLATE_POSITION = position
        self.save_track_status()
    
    def update_gripper_location(self, location_name, x=None, z=None):
        """Update tracked gripper location when moving to a named position from track_positions.yaml"""
        # Update coordinates first
        if x is not None:
            self.CURRENT_GRIPPER_POSITION['x'] = x
        if z is not None:
            self.CURRENT_GRIPPER_POSITION['z'] = z
        
        # Check for origin position (0,0) and override location name
        if self.CURRENT_GRIPPER_POSITION.get('x') == 0 and self.CURRENT_GRIPPER_POSITION.get('z') == 0:
            self.CURRENT_GRIPPER_LOCATION = "origin"
            self.logger.debug(f"Detected origin position (0,0) - setting location to 'origin'")
        else:
            self.CURRENT_GRIPPER_LOCATION = location_name
        
        self.save_track_status()
        self.logger.debug(f"Updated gripper location to: {self.CURRENT_GRIPPER_LOCATION} at position {self.CURRENT_GRIPPER_POSITION}")
    
    def get_speed(self, speed_name):
        """Get movement speed by name"""
        speeds = self.TRACK_POSITIONS.get('speeds', {})
        speed = speeds.get(speed_name)
        if speed is None:
            self.logger.warning(f"Speed '{speed_name}' not found, using default")
            return 50  # Default fallback speed
        return speed
    
    def get_limit(self, limit_name):
        """Get movement limit by name"""
        limits = self.TRACK_POSITIONS.get('limits', {})
        limit = limits.get(limit_name)
        if limit is None:
            self.logger.warning(f"Limit '{limit_name}' not found in track configuration")
            return 0  # Default fallback
        return limit
    
    def get_timing(self, timing_name):
        """Get timing parameter by name"""
        timing = self.TRACK_POSITIONS.get('timing', {})
        value = timing.get(timing_name)
        if value is None:
            self.logger.warning(f"Timing '{timing_name}' not found in track configuration")
            return 1  # Default fallback
        return value
    
    def get_axis(self, axis_name):
        """Get hardware axis number by name from robot hardware configuration"""
        if not hasattr(self, 'ROBOT_HARDWARE') or not self.ROBOT_HARDWARE:
            self.logger.error("Robot hardware configuration not loaded")
            # Fallback to legacy hardcoded values for safety
            fallbacks = {
                'x_axis': 7,
                'z_axis': 6, 
                'gripper_open': 4,
                'gripper_close': 5
            }
            return fallbacks.get(axis_name, 0)
        
        # Check track_axes for track-specific axes
        track_axes = self.ROBOT_HARDWARE.get('track_axes', {})
        axis = track_axes.get(axis_name)
        
        # Check track_pneumatics for pneumatic outputs
        if axis is None:
            track_pneumatics = self.ROBOT_HARDWARE.get('track_pneumatics', {})
            axis = track_pneumatics.get(axis_name)
        
        if axis is None:
            self.logger.error(f"Hardware axis '{axis_name}' not found in robot hardware configuration")
            # Fallback to legacy hardcoded values for safety
            fallbacks = {
                'x_axis': 7,
                'z_axis': 6, 
                'gripper_open': 4,
                'gripper_close': 5
            }
            return fallbacks.get(axis_name, 0)
        return axis

    # === 3. BASIC MOVEMENT & GRIPPER PRIMITIVES ===
    def set_horizontal_speed(self,vel):
        self.logger.debug("Setting horizontal speed to %d", vel)
        self.c9.DEFAULT_X_SPEED = vel

    def set_vertical_speed(self,vel):
        self.logger.debug("Setting vertical speed to %d", vel)
        self.c9.DEFAULT_Y_SPEED = vel    

    def open_gripper(self):
        self.logger.debug("Opening gripper")
        self.c9.set_output(self.get_axis('gripper_open'), True)  
        self.c9.set_output(self.get_axis('gripper_close'), False)
        self.c9.delay(self.get_timing('gripper_delay'))
    
    def close_gripper(self):
        self.logger.debug("Closing gripper")
        self.c9.set_output(self.get_axis('gripper_close'), True)  #gripper close
        self.c9.set_output(self.get_axis('gripper_open'), False)
        self.c9.delay(self.get_timing('gripper_delay'))

    def origin(self):
        self.c9.move_axis(self.get_axis('z_axis'), self.get_limit('max_safe_height'), vel=self.get_speed('default_z')) #max_height
        self.c9.move_axis(self.get_axis('x_axis'), 0, vel=self.get_speed('default_x'))
        self.logger.debug("Moving North Track to home position")
        
        # Update gripper location tracking - now at home position (0, max_safe_height)
        max_height = self.get_limit('max_safe_height')
        self.update_gripper_location("home", x=0, z=max_height)

    def move_through_path(self, waypoint_locations):
        """
        Move through a series of waypoint locations for safe routing.
        
        Args:
            waypoint_locations (list): List of location names to visit as waypoints
        """
        if not waypoint_locations:
            self.logger.debug("No waypoint locations provided, skipping path movement")
            return
        
        for waypoint_name in waypoint_locations:
            waypoint = self.get_position(waypoint_name)
            if not waypoint:
                self.logger.error(f"Waypoint '{waypoint_name}' not found in configuration")
                continue
            
            # Move to waypoint X and Z positions
            if 'x' in waypoint:
                self.c9.move_axis(self.get_axis('x_axis'), waypoint['x'], vel=self.get_speed('default_x'))
            if 'z' in waypoint:
                self.c9.move_axis(self.get_axis('z_axis'), waypoint['z'], vel=self.get_speed('default_z'))
            
            # Update gripper location tracking after successful movement
            self.update_gripper_location(waypoint_name, x=waypoint.get('x'), z=waypoint.get('z'))

    # === 4. WELLPLATE MOVEMENT SEQUENCES ===
    def transfer_wellplate_via_path(self, destination_x, destination_z, waypoint_locations=None):
        """
        Transfer wellplate using configurable safe routing via specified waypoint locations.
        
        Args:
            destination_x (float): Final X position for wellplate placement
            destination_z (float): Final Z position for wellplate placement
            waypoint_locations (list): List of location names to visit as waypoints (e.g., ['transfer_stack', 'cytation_safe_area'])
        """
        
        self.move_through_path(waypoint_locations)  # Move through waypoints first
        
        # Move to final destination
        self.c9.move_axis(self.get_axis('z_axis'), destination_z, vel=self.get_speed('default_z'))
        self.c9.move_axis(self.get_axis('x_axis'), destination_x, vel=self.get_speed('default_x'))
        
        # Update gripper location tracking with final coordinates 
        # Note: location_name not available in this function, but coordinates are tracked
        self.CURRENT_GRIPPER_POSITION = {"x": destination_x, "z": destination_z}
        
        # Check for origin position (0,0) and set location accordingly
        if destination_x == 0 and destination_z == 0:
            self.CURRENT_GRIPPER_LOCATION = "origin"
            self.logger.debug(f"Moved to origin position (0,0) - setting location to 'origin'")
            self.save_track_status()

    def grab_wellplate_from_location(self, location_name, wellplate_type="96 WELL PLATE", waypoint_locations=None, z_override=None):
        """
        Unified method to grab a wellplate from any configured location.
        
        Args:
            location_name (str): Name of location in track_positions.yaml (e.g., 'pipetting_area', 'cytation_tray')
            wellplate_type (str): Type of wellplate for height adjustment
            waypoint_locations (list): Optional list of waypoint location names for safe routing after grabbing
            z_override (float): Optional Z position override for dynamic stack heights
        """
        self.logger.info(f"Grabbing wellplate from location: {location_name}")
        
        # Get location position from YAML
        location_pos = self.get_position(location_name)
        if not location_pos:
            self.logger.error(f"Location '{location_name}' not found in configuration")
            return
        
        # Extract position coordinates (handle different location formats)
        x_pos = location_pos['x']
        z_transfer = location_pos['z_transfer']
        
        if z_override is not None: # Use provided Z override for dynamic positioning (e.g., stack heights)
            z_grab = z_override
        elif 'z_grab' in location_pos:  #Defined z_grab
            z_grab = location_pos['z_grab']
        else:
            self.logger.error(f"No Z coordinate found for location '{location_name}' and no z_override provided")
            return
        
        # Calculate vertical offset based on wellplate type
        move_up = 0
        move_up += self.get_config_parameter('wellplates', wellplate_type, "gripping_z_offset", error_on_missing=False) or 0
        z_grab_final = z_grab - move_up
        
        self.transfer_wellplate_via_path(x_pos, z_transfer, waypoint_locations)  # Move to transfer height first

        # Execute movement sequence
        self.open_gripper()
        self.c9.move_axis(self.get_axis('z_axis'), z_grab_final, vel=self.get_speed('slow_movement')) #Move to grab height
        self.close_gripper()
        self.set_wellplate_position('gripper')
        
        #Move to transfer height
        self.c9.move_axis(self.get_axis('z_axis'), z_transfer, vel=self.get_speed('default_z'))
        
        # Update gripper location tracking - now at this location with wellplate in gripper
        self.update_gripper_location(location_name, x=x_pos, z=z_transfer)
       
    def release_wellplate_in_location(self, location_name, wellplate_type="96 WELL PLATE", waypoint_locations=None, z_override=None):
        """
        Unified method to release a wellplate at any configured location with optional safe routing.
        
        Args:
            location_name (str): Name of location in track_positions.yaml
            wellplate_type (str): Type of wellplate for height adjustment  
            waypoint_locations (list): Optional list of waypoint location names for safe routing (e.g., ['cytation_safe_area'])
            z_override (float): Optional Z position override for dynamic stack heights
        """
        self.logger.info(f"Releasing wellplate at location: {location_name}")
        
        # COLLISION PREVENTION: Check if destination location is already occupied
        self.get_track_status()  # Refresh current wellplate position
        if self.ACTIVE_WELLPLATE_POSITION == location_name:
            self.pause_after_error(
                f"COLLISION PREVENTION: Cannot release wellplate at '{location_name}' - "
                f"location already occupied. Current wellplate position: {self.ACTIVE_WELLPLATE_POSITION}",
                send_slack=True
            )
        
        # Get location position from YAML
        location_pos = self.get_position(location_name)
        if not location_pos:
            self.logger.error(f"Location '{location_name}' not found in configuration")
            return
        
        # Extract position coordinates (handle different location formats)
        x_pos = location_pos['x']
        z_transfer = location_pos['z_transfer']
        
        if z_override is not None:
            # Use provided Z override for dynamic positioning (e.g., stack heights)
            z_release = z_override
        elif 'z_release' in location_pos:  # Cytation-style location
            z_release = location_pos['z_release']
        else:
            self.logger.error(f"No Z coordinate found for location '{location_name}' and no z_override provided")
            return
        
        # Calculate vertical offset based on wellplate type
        move_up = 0
        move_up += self.get_config_parameter('wellplates', wellplate_type, "gripping_z_offset", error_on_missing=False) or 0
        
        # Calculate final Z position
        final_z_release = z_release - move_up

        # Execute movement sequence to get there
        self.transfer_wellplate_via_path(x_pos, z_transfer, waypoint_locations)  # Move to transfer height first
        
        #Move down to release height & Release
        self.c9.move_axis(self.get_axis('z_axis'), final_z_release, vel=self.get_speed('slow_movement'))
        self.open_gripper()
        self.set_wellplate_position(location_name)  # Wellplate released at this location
        
        # Return to safe height
        self.c9.move_axis(self.get_axis('z_axis'), z_transfer, vel=self.get_speed('default_z'))
        
        # Update gripper location tracking - now at this location (without wellplate)
        self.update_gripper_location(location_name, x=x_pos, z=z_transfer)


    # === 6. STACK MANAGEMENT ===
    def calculate_wp_stack_height(self, num, wp_type, stack_name, operation='grab'):
        self.logger.debug("Calculating well plate stack height for num: %d, wp_type: %s, stack: %s, operation: %s", num, wp_type, stack_name, operation)
        thickness = self.get_config_parameter('wellplates', wp_type, "thickness", error_on_missing=False) or 0
        
        # Get base height from the specific stack configuration based on operation
        stack_pos = self.get_position(stack_name)
        if not stack_pos:
            self.logger.error(f"Stack '{stack_name}' not found in configuration")
            return -1
            
        # Use operation-specific Z coordinate (like other unified methods)
        if operation == 'release' and 'z_release' in stack_pos:
            base_height = stack_pos['z_release']
        elif operation == 'grab' and 'z_grab' in stack_pos:
            base_height = stack_pos['z_grab']
        else:
            self.logger.error(f"No appropriate Z coordinate found for stack '{stack_name}' operation '{operation}'")
            return -1
            
        height = base_height - thickness * (num - 1)
        #print(f"HEIGHT = {height}")

        if height >= self.get_limit('max_stack_height') and height <= base_height: #beneath max height and higher than the base
            return height
        else:
            return -1 #invalid input
        
    def get_new_wellplate(self, move_home_afterwards = True): #double WP stack 
        self.logger.info("Getting new wellplate from source stack")
        """Get a new wellplate from the source stack (in the double stack holder) and move to north robot pipetting area"""
        DOUBLE_SOURCE_Y = self.calculate_wp_stack_height(self.NUM_SOURCE, self.CURRENT_WP_TYPE, 'source_stack', 'grab')
        

        if self.NUM_SOURCE > 0 and DOUBLE_SOURCE_Y != -1 and self.ACTIVE_WELLPLATE_POSITION != 'pipetting_area': #still have well plates in stack & pipetting area is empty
            self.logger.debug(f"Getting {self.get_ordinal(self.NUM_SOURCE)} wellplate from source at {DOUBLE_SOURCE_Y}")
                     
            # Use unified method with dynamic Z override for wellplate pickup
            self.grab_wellplate_from_location('source_stack', self.CURRENT_WP_TYPE, z_override=DOUBLE_SOURCE_Y, waypoint_locations=['max_height'])
            
            self.NUM_SOURCE -= 1
            self.save_track_status() #update yaml

            self.release_wellplate_in_location('pipetting_area', self.CURRENT_WP_TYPE, waypoint_locations=['transfer_stack'])

            if move_home_afterwards:
                self.origin()
        
        else:
            if self.NUM_SOURCE <= 0:
                self.pause_after_error("Wellplate stack is empty")
            elif DOUBLE_SOURCE_Y == -1:
                self.pause_after_error("Invalid height calculated (too many wellplates in stack)")
            if self.ACTIVE_WELLPLATE_POSITION == 'pipetting_area':
                self.pause_after_error("Cannot get wellplate, NR pipetting area is occupied")
    
    def discard_wellplate(self, move_home_afterwards = True, pickup_location='pipetting_area', discard_location='waste_stack'): #double WP stack
        """Grabs a wellplate (from desginated wellplate stack) and discards it into the waste stack (in the double stack holder). ENSURE North Robot is homed!!!
        
        Args:
            `wellplate_num`(int): The number identifying which wellplate stand to discard a wellplate from (0 for the pipetting one) 
        """
        DOUBLE_WASTE_Y = self.calculate_wp_stack_height(self.NUM_WASTE+1, self.CURRENT_WP_TYPE, 'waste_stack', 'release')
        
        if DOUBLE_WASTE_Y != -1: #can still hold an additional wellplate & pipetting area is occupied
            self.logger.info(f"Discarding wellplate as the {self.get_ordinal(self.NUM_WASTE+1)} WP in waste stack at height: {DOUBLE_WASTE_Y}")
            
            # Move to max height, then grab wellplate from NR
            self.close_gripper()
            if self.ACTIVE_WELLPLATE_POSITION == pickup_location:
                self.grab_wellplate_from_location(pickup_location, self.CURRENT_WP_TYPE, waypoint_locations=['transfer_stack'])
            self.release_wellplate_in_location(discard_location, self.CURRENT_WP_TYPE, z_override=DOUBLE_WASTE_Y, waypoint_locations=['transfer_stack'])

            # Use unified transfer method to drop off at waste stack
            self.set_wellplate_position(None)  # Wellplate discarded to waste, no longer tracked

            self.NUM_WASTE += 1
            self.save_track_status()

            if move_home_afterwards: #home after dropping off wellplate into waste stack
                self.origin()
        else:
            if DOUBLE_WASTE_Y == -1:
                self.pause_after_error("Wellplate stack is too full for discarding another well plate.")
            if self.ACTIVE_WELLPLATE_POSITION != 'pipetting_area':
                self.pause_after_error("Cannot discard wellplate, designated wellplate stand is empty.")

    # === 7. UTILITIES ===        
    def get_ordinal(self,n): #convert n into an ordinal number (ex. 1st, 2nd, 4th)  -- from chatgpt
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

class North_Powder:

    p2 = None
    c9 = None

    def __init__(self, c9,simulate = False, logger=None):
        
        self.c9 = c9
        self.simulate = simulate
        self.logger = logger
        self.logger.debug("Initializing North Powder Dispenser...")
        if not simulate:
            from north import NorthC9
            self.p2 = NorthC9('C', network=c9.network)

    def activate_powder_channel(self, channel=0):
        self.logger.debug(f"Activating powder channel: {channel}")
        self.p2.activate_powder_channel(channel)
        self.p2.home_OL_stepper(0, 300)

    # shake the cartridge for t ms, with frequency f [40, 180] hz, amplitude [60, 100] %
    def shake(self,t, f=80, a=80, wait=True):
        self.logger.debug("Shaking dispenser cartridge...")
        return self.p2.amc_pwm(int(f), int(t), int(a), wait=wait)
    
    def set_opening(self, channel, deg):
        self.logger.debug(f"Setting opening for channel {channel} to {deg} degrees")
        self.p2.move_axis(channel, deg*(1000/360.0))

    def dispense_powder_time(self, channel, time):
        self.logger.info(f"Dispensing powder for {time} ms on channel {channel}")
        self.activate_powder_channel(channel)
        self.c9.move_carousel(70,70)
        
        initial_mass = self.c9.read_steady_scale()
        self.set_opening(channel,45)
        self.shake(time)
        self.set_opening(channel,0)
        final_mass = self.c9.read_steady_scale()
        dispensed_mass = final_mass - initial_mass
        self.logger.info("Mass dispensed: ", dispensed_mass)
        
        self.c9.move_carousel(0,0)

    def cl_pow_dispense(self,mg_target, channel, protocol=None, zero_scale=False, max_tries=20):
        import settings.powder_settings as settings
        start_t = time.perf_counter()
        mg_togo = mg_target

        if protocol is None:
            protocol = settings.default_ps
        ps = protocol.fast_settings
        if mg_togo < protocol.slow_settings.thresh:
            ps = protocol.slow_settings
        elif mg_togo < protocol.med_settings.thresh:
            ps = protocol.med_settings

        # intialize
        self.set_opening(channel,0)  # make sure everything starts closed
        prev_mass = 0
        delta_mass = 0
        shake_t = ps.min_shake_t

        if zero_scale:
            self.c9.zero_scale()
            self.c9.delay(protocol.scale_delay)
        tare = self.c9.read_steady_scale() * 1000

        meas_mass = 0
        count = 0
        while mg_togo > protocol.tol and count < max_tries:
            count += 1

            self.set_opening(channel,ps.opening_deg)
            self.shake(shake_t, ps.freq, ps.amplitude)
            if ps.shut_valve:
                self.set_opening(channel,0)
            self.c9.delay(0.5)
            self.c9.read_steady_scale()  # dummy read to wait for steady
            self.c9.delay(protocol.scale_delay)  # delay after steady to allow for more settling time
            meas_mass = self.c9.read_steady_scale() * 1000 - tare

            mg_togo = mg_target - meas_mass
            delta_mass = meas_mass - prev_mass
            prev_mass = meas_mass

            if mg_togo < protocol.slow_settings.thresh:
                ps = protocol.slow_settings
            elif mg_togo < protocol.med_settings.thresh:
                ps = protocol.med_settings

            iter_target = (ps.percent_target * mg_togo)
            max_new_t = ps.max_growth * shake_t
            if delta_mass <= 0:
                shake_t = max_new_t
            else:
                shake_t *= (iter_target / delta_mass)
            shake_t = min(max_new_t, shake_t)  # no larger than max growth allows
            shake_t = max(ps.min_shake_t, shake_t)  # no shorter than min time
            shake_t = min(ps.max_shake_t, shake_t)  # no longer than max time

            self.logger.debug(f'Iteration {count}:')
            self.logger.debug(f'\tJust dispensed:  {delta_mass:.1f} mg')
            self.logger.debug(f'\tRemaining:       {mg_togo:.1f} mg')
            self.logger.debug(f'\tNext target:     {iter_target:.1f} mg')
            self.logger.debug(f'\tNext time:       {int(shake_t)} ms')
            self.logger.debug('')

            self.set_opening(channel, 0)

            self.logger.debug(f'Result:')
            self.logger.debug(f'\tLast iter:  {delta_mass:.1f} mg')
            self.logger.debug(f'\tDispensed: {meas_mass:.1f} mg')
            self.logger.debug(f'\tRemaining: {mg_togo:.1f} mg')
            self.logger.debug(f'\tTime:      {int(time.perf_counter() - start_t)} s')
            self.logger.debug('')

        return meas_mass

    def dispense_powder_mg(self, mass_mg, channel=0):
        self.logger.info(f"Dispensing {mass_mg} mg on channel {channel}")
        
        self.activate_powder_channel(channel)
        self.c9.move_carousel(66.5,70)
        #Dispense protocol
        mass_dispensed = self.cl_pow_dispense(mass_mg, channel)
        self.logger.info(f"Mass dispensed: {mass_dispensed} mg")

        self.c9.move_carousel(0,0)
        return mass_dispensed

class North_Temp:

    t8 = None
    c8 = None

    def __init__(self,c9,c8, simulate = False, logger=None):
        self.simulate = simulate
        self.c9 = c9
        self.c8 = c8
        self.logger = logger

        self.logger.debug("Initializing temperature controller...")

        if not self.simulate:
            from north import NorthC9
            self.t8 = NorthC9('B', network=c9.network)  
        else:
            self.t8 = MagicMock()
    
    def autotune(self,target_temp,channel=0):
        self.logger.debug(f"Autotuning channel {channel} to target temperature {target_temp}C")
        self.t8.disable_channel(channel)
        self.t8.set_temp(channel, target_temp)
        self.t8.enable_channel(channel)
        self.t8.temp_autotune(channel, True)

    def set_temp(self,target_temp,channel=0):
        self.logger.debug(f"Setting channel {channel} to target temperature {target_temp}C")
        self.t8.set_temp(channel, target_temp)
        self.t8.enable_channel(channel)
    
    def get_temp(self,channel=0):
        return self.t8.get_temp(channel)

    def turn_off_heating(self,channel=0):
        self.logger.debug(f"Turning off heating for channel {channel}")
        self.t8.disable_channel(channel)

    def turn_on_stirring(self,speed=10000):
        self.logger.debug(f"Turning on stirring at speed {speed}")
        self.c8.spin_axis(1, speed)

    def turn_off_stirring(self):
        self.logger.debug("Turning off stirring")
        self.c8.spin_axis(1,0)

class North_Spin:
    c8 = None
    c9 = None
    def __init__(self,c9,c8, simulate = False, logger=None):
        self.simulate = simulate
        self.c9 = c9
        self.c8 = c8
        self.logger = logger
        self.logger.debug("Initializing centrifuge controller...")
    
    def set_speed(self,speed):
        self.logger.debug(f"Setting centrifuge speed to {speed} RPM")
        self.c8.spin_axis(2, speed)

    def stop_spin(self):
        self.logger.debug("Stopping centrifuge spin")
        self.c8.spin_axis(2,0)

    def open_lid(self):
        self.logger.debug("Opening centrifuge lid")
        self.c9.set_output(3, True)  # Open lid pneumatic

    def close_lid(self):
        self.logger.debug("Closing centrifuge lid")
        self.c9.set_output(3, False)  # Close lid pneumatic

    def turn_on_vacuum(self):
        self.logger.debug("Turning on centrifuge vacuum")
        self.c9.set_output(2, True)  # Turn on vacuum pneumatic

    def turn_off_vacuum(self):
        self.logger.debug("Turning off centrifuge vacuum")
        self.c9.set_output(2, False)  # Turn off vacuum pneumatic

class North_Robot(North_Base):
    """
    North Robot Main Class - Coordinates liquid handling, vial management, and wellplate operations
    
    Organization:
    1. Class Setup & Initialization
    2. Configuration & Parameter Access  
    3. Core Robot Operations
    4. Pipetting & Tip Management
    5. Liquid Handling Operations
    6. Wellplate Operations  
    7. Vial & Container Management
    8. Reservoir System
    9. State & Status Management
    10. Vial Info & Location Utilities
    11. Validation & Safety Methods
    12. Utility Methods
    """
    
    # ====================================================================
    # CLASS CONSTANTS & FILE PATHS
    # ====================================================================
    
    ROBOT_STATUS_FILE = "robot_state/robot_status.yaml" #Store the state of the robot. Update this after every method that alters the state. 
    VIAL_POSITIONS_FILE = "robot_state/vial_positions.yaml" #File that contains the vial positions.
    WELLPLATE_POSITIONS_FILE = "robot_state/wellplates.yaml" #File that contains the wellplate positions.
    PIPET_TIP_DEFINITTIONS_FILE = "robot_state/pipet_tips.yaml" #File that contains the pipet tip definitions.
    PIPET_RACKS_FILE = "robot_state/pipet_racks.yaml" #File that contains the pipet rack configurations.
    PUMP_CONFIG_FILE = "robot_state/syringe_pumps.yaml" #File that contains the pump configurations.
    ROBOT_HARDWARE_FILE = "robot_state/robot_hardware.yaml" #File that contains robot hardware axis and pneumatic mappings.
    
    # ====================================================================
    # 1. CLASS SETUP & INITIALIZATION
    # ====================================================================
    
    #Initialize the status of the robot. 
    def __init__(self,c9,c8,vial_file=None,simulate=False, logger=None):

        self.c9 = c9
        self.c8 = c8
        self.logger = logger
        self.VIAL_FILE = vial_file #File that we save the vial data in 
        self.simulate = simulate

        self.logger.info("Initializing North Robot...")

        # Load all configuration files (static config)
        self._load_configuration_files()
        
        # Load all state files (dynamic state)  
        self._load_state_files()
        # Note: load_pumps() is now called after homing in reset_after_initialization()
        self.reset_after_initialization() #Reset everything that may not be as desired, eg return to "Home"
        
    def _load_configuration_files(self):
        """Load all robot configuration YAML files (static config, not state)"""
        self.logger.debug("Loading robot configuration files...")
        
        # Configuration files (required for robot operation)
        config_files = {
            'VIAL_POSITIONS': (self.VIAL_POSITIONS_FILE, "vial positions configuration"),
            'PIPET_TIPS': (self.PIPET_TIP_DEFINITTIONS_FILE, "pipet tips configuration"),
            'PIPET_RACKS': (self.PIPET_RACKS_FILE, "pipet racks configuration"),
            'PUMP_CONFIG': (self.PUMP_CONFIG_FILE, "pump configuration"),
            'ROBOT_HARDWARE': (self.ROBOT_HARDWARE_FILE, "robot hardware configuration"),
            'WELLPLATES': ("robot_state/wellplates.yaml", "wellplate properties configuration"),
        }
        
        for attr_name, (file_path, description) in config_files.items():
            # Use convert_none=True for all files, wellplates are optional
            setattr(self, attr_name, self._load_yaml_file(file_path, description, required=True, convert_none=True))
        
        # Initialize pipet usage tracking from loaded rack configuration
        self.PIPETS_USED = {rack_name: 0 for rack_name in self.PIPET_RACKS.keys()} if self.PIPET_RACKS else {}

    def _load_state_files(self):
        """Load all robot state YAML files (dynamic state that changes during operation)"""
        self.logger.debug("Loading robot state files...")
        
        # Load vial DataFrame (CSV file) - only if vial file is provided
        if self.VIAL_FILE is not None:
            try:
                self.VIAL_DF = pd.read_csv(self.VIAL_FILE, sep=",")
                self.VIAL_DF.index = self.VIAL_DF['vial_index'].values
                self.logger.debug(f"Loaded vial data from {self.VIAL_FILE}")
            except Exception as e:
                self.logger.error(f"Issue reading vial status file {self.VIAL_FILE}: {e}")
                self.pause_after_error("Issue reading vial status", False)
        else:
            # Create empty vial DataFrame if no vial file provided
            self.VIAL_DF = pd.DataFrame()
            self.logger.info("No vial file provided - operating without vial tracking")
        
        # State files (robot status is required, wellplate positions are optional)
        state_files = {
            'robot_status': (self.ROBOT_STATUS_FILE, "robot status", True, True),
            'wellplate_positions': (self.WELLPLATE_POSITIONS_FILE, "wellplate positions", False, True),
        }
        
        loaded_data = {}
        for key, (file_path, description, required, convert_none) in state_files.items():
            loaded_data[key] = self._load_yaml_file(file_path, description, required, convert_none)
        
        # Initialize critical robot status attributes with safe defaults first
        # This ensures they always exist even if loading fails
        self.GRIPPER_STATUS = None
        self.GRIPPER_VIAL_INDEX = None
        self.HELD_PIPET_TYPE = None
        self.PIPET_FLUID_VOLUME = 0.0
        self.PIPET_FLUID_VIAL_INDEX = None
        
        # Handle robot status data
        if loaded_data['robot_status']:
            robot_status = loaded_data['robot_status']
            try:
                # Override defaults with loaded values
                self.GRIPPER_STATUS = robot_status['gripper_status']
                self.GRIPPER_VIAL_INDEX = robot_status['gripper_vial_index']
                self.HELD_PIPET_TYPE = robot_status.get('held_pipet_type')
                
                # Load pipets used from robot status
                pipets_used_data = robot_status.get('pipets_used', {})
                if isinstance(pipets_used_data, dict):
                    # Load directly from the current format
                    for rack_name in self.PIPET_RACKS.keys() if self.PIPET_RACKS else []:
                        self.PIPETS_USED[rack_name] = pipets_used_data.get(rack_name, 0)
                else:
                    # Initialize all racks to 0 if invalid format
                    for rack_name in self.PIPET_RACKS.keys() if self.PIPET_RACKS else []:
                        self.PIPETS_USED[rack_name] = 0
                    
                self.PIPET_FLUID_VOLUME = robot_status.get('pipet_fluid_volume', 0.0) or 0.0
                self.PIPET_FLUID_VIAL_INDEX = robot_status.get('pipet_fluid_vial_index')
                
                self.logger.debug("Robot status loaded successfully")
            except KeyError as e:
                self.logger.warning(f"Missing field in robot status, using defaults: {e}")
            except Exception as e:
                self.logger.warning(f"Error processing robot status, using defaults: {e}")
        
        # Handle wellplate positions data (dynamic state, not static config)
        # Note: This is different from WELLPLATES which contains static wellplate type definitions
        self.WELLPLATE_POSITIONS = loaded_data['wellplate_positions']

    #Load the pumps and set volumes from YAML configuration
    def load_pumps(self):
        self.logger.debug("Loading pumps from YAML configuration...")
        
        if not self.PUMP_CONFIG:
            self.pause_after_error("Pump configuration not loaded or invalid")
            return
        
        # Initialize current pump speeds dictionary
        self.CURRENT_PUMP_SPEEDS = {}
        
        # Load each pump from configuration (flattened structure)
        for pump_index_str, pump_config in self.PUMP_CONFIG.items():
            pump_index = int(pump_index_str)
            
            # Set pump volume
            volume = pump_config.get('volume', 1.0)
            self.c9.pumps[pump_index]['volume'] = volume
            
            # Set pump speed
            speed = pump_config.get('default_speed', 11)
            self.c9.set_pump_speed(pump_index, speed)
            self.CURRENT_PUMP_SPEEDS[pump_index] = speed
            
            self.logger.debug(f"Loaded pump {pump_index}: volume={volume}mL, speed={speed}, liquid={pump_config.get('liquid', 'none')}")

    # ====================================================================
    # 2. CONFIGURATION & PARAMETER ACCESS
    # ====================================================================
    
    def get_speed(self, speed_name):
        """Get movement speed from robot hardware configuration"""
        return self.get_config_parameter('robot_hardware', 'movement_speeds', speed_name, error_on_missing=True)
    
    def get_safe_height(self):
        """Get safe height from robot hardware configuration"""
        return self.get_config_parameter('robot_hardware', 'physical_constants', 'safe_height_z', error_on_missing=True)

    def get_current_height(self):
        """Get the current Z-axis height in millimeters"""
        z_axis = self.get_config_parameter('robot_hardware', 'robot_axes', 'z_axis', error_on_missing=False) or 3
        return self.c9.counts_to_mm(z_axis, self.c9.get_axis_position(z_axis))
    
    def get_height_at_location(self, location):
        """Get the Z-axis height in millimeters for a given location coordinate"""
        z_axis = self.get_config_parameter('robot_hardware', 'robot_axes', 'z_axis', error_on_missing=False) or 3
        return self.c9.counts_to_mm(z_axis, location[3])

    def visualize_racks(self, vial_status, fig_size_x=16, fig_size_y=10, xlim=12, ylim=6):
        self.logger.info("Visualizing vial racks...")
        
        # Debug matplotlib backend
        try:
            import matplotlib
            self.logger.debug(f"Matplotlib backend: {matplotlib.get_backend()}")
            
            # Try to set an interactive backend for Windows
            if matplotlib.get_backend() == 'Agg':
                try:
                    matplotlib.use('TkAgg')
                    self.logger.info("Switched matplotlib backend to TkAgg")
                except:
                    try:
                        matplotlib.use('Qt5Agg')
                        self.logger.info("Switched matplotlib backend to Qt5Agg") 
                    except:
                        self.logger.warning("Could not set interactive backend, using default")
            
        except Exception as e:
            self.logger.warning(f"Matplotlib backend check failed: {e}")
        
        fig, ax = plt.subplots(figsize=(fig_size_x, fig_size_y))
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1, xlim)
        ax.set_ylim(-1, ylim)
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.grid(True)

        # Example Racks
        self.visualize_rack(
            ax=ax,
            rack_data=vial_status,
            rack_name='Main Rack',
            location_filter='main_8mL_rack',
            num_cols=8,
            num_rows=6,
            filled_color='lightgreen',
            offset_x=4.0,
            offset_y=0,
            title='Main Rack'
        )

        self.visualize_rack(
            ax=ax,
            rack_data=vial_status,
            rack_name='Second Rack',
            location_filter='large_vial_rack',
            num_cols=2,
            num_rows=2,
            filled_color='lightblue',
            offset_x=0.5,
            offset_y=0,
            title='20 mL Rack'
        )

        plt.tight_layout()
        
        try:
            plt.show(block=True)
            self.logger.debug("Vial visualization displayed successfully")
        except Exception as e:
            self.logger.error(f"Failed to display vial visualization: {e}")
            self.logger.info("Continuing without visualization...")

    def visualize_rack(self,ax,rack_data,rack_name,location_filter,num_cols,num_rows,filled_color='lightgreen',offset_x=0,offset_y=0,title=None):
        self.logger.debug(f"Visualizing rack: {rack_name} with filter: {location_filter}")
        rack_data = rack_data[rack_data['location'] == location_filter]

        all_indices = set(range(num_cols * num_rows))
        present_indices = set(rack_data['location_index'])
        missing_indices = all_indices - present_indices

        # Draw filled/empty vials
        for _, row in rack_data.iterrows():
            index = row['location_index']
            col = index // num_rows
            row_pos = index % num_rows

            # Adjust for offset
            x = col + offset_x
            y = row_pos + offset_y

            fill_color = filled_color if row['vial_volume'] > 0 else 'white'
            circle = patches.Circle((x, y), 0.45, edgecolor='black', facecolor=fill_color)
            ax.add_patch(circle)

            if not row['capped']:
                cap_status = "uncapped"
            elif row['cap_type'] == 'open':
                cap_status = "open cap"
            elif row['cap_type'] == 'closed':
                cap_status = "closed cap"
            else:
                cap_status = "unknown cap"

            text = f"{row['vial_name']}\n{row['vial_volume']:.2f} mL\n{cap_status}"
            ax.text(x, y, text, ha='center', va='center', fontsize=8)

        # Draw dotted circles for missing vials
        for index in missing_indices:
            col = index // num_rows
            row_pos = index % num_rows

            x = col + offset_x
            y = row_pos + offset_y

            circle = patches.Circle((x, y), 0.45, edgecolor='black', facecolor='none', linestyle='dotted')
            ax.add_patch(circle)

        # Add bounding box and title for this rack
        rect = patches.Rectangle(
            (offset_x - 0.5, offset_y - 0.5),
            num_cols,
            num_rows,
            linewidth=2,
            edgecolor='gray',
            facecolor='none'
        )
        ax.add_patch(rect)

        if title:
            ax.text(
                offset_x + num_cols / 2 - 0.5,
                offset_y + num_rows - 0.3,
                title,
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

    #Check the status of the input vial file
    def check_input_file(self,pause_after_check=True, visualize=True):
        """
        Prints the vial status dataframe for user to confirm the initial state of your vials.
        """
        print("\n" + "="*60)
        print("DEPRECATION WARNING: check_input_file()")
        print("="*60)
        print("This method is deprecated. Please use the GUI launched during")
        print("Lash_E initialization to check and modify robot states.")
        print("Remove the .check_input_file() calls from your workflow.")
        print("Press Enter to continue...")
        print("="*60)
        input()
        
        if self.VIAL_FILE is None:
            self.logger.info("No vial file loaded - skipping vial status check")
            return
            
        vial_status = pd.read_csv(self.VIAL_FILE, sep=",")
        self.logger.info(vial_status)

        # if visualize:
        #     self.visualize_racks(vial_status)

        # if pause_after_check and not self.simulate:
        #     input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    # ====================================================================
    # 3. CORE ROBOT OPERATIONS
    # ====================================================================

    def home_robot_components(self):
        """
        Comprehensive homing of all robot components.
        This method should be called when the robot needs to be homed due to initialization
        or error recovery. It homes all components systematically.
        """
        if self.simulate:
            self.logger.debug("Simulating robot component homing...")
            return
            
        self.logger.info("Homing all robot components...")
        
        try:
            # Home carousel first
            self.logger.debug("Homing carousel...")
            self.c9.home_carousel()
            
            # Home main robot axes  
            self.logger.debug("Homing robot axes...")
            self.c9.home_robot()
            
            # Home all pumps systematically
            pump_configs = self.PUMP_CONFIG or {}
            for pump_index in pump_configs.keys():
                self.logger.debug(f"Homing pump {pump_index}...")
                self.c9.home_pump(int(pump_index))
            
            # Home track axes from configuration
            self.logger.debug("Homing track axes...")
            track_axes = self.ROBOT_HARDWARE.get('track_axes', {}) if hasattr(self, 'ROBOT_HARDWARE') and self.ROBOT_HARDWARE else {}
            if track_axes:
                # Sort track axes by axis number to ensure consistent homing order (lower numbers first)
                sorted_track_axes = sorted(track_axes.items(), key=lambda x: x[1])
                for axis_name, axis_number in sorted_track_axes:
                    self.logger.debug(f"Homing track axis {axis_name} (axis {axis_number})...")
                    self.c9.home_axis(axis_number)
            else:
                self.logger.warning("No track axes configuration found, skipping track axis homing")
                
            # Home photoreactor stepper if available
            if hasattr(self, 'p2') and self.p2 is not None:
                self.logger.debug("Homing photoreactor stepper...")
                self.p2.home_OL_stepper(0, 300)
                
            self.logger.info("All robot components homed successfully")
            
            # Update gripper location tracking after homing - robot is now at home position
            # After homing, track gripper should be at origin position (0, max_safe_height)
            if hasattr(self, 'CURRENT_GRIPPER_LOCATION'):
                max_height = self.get_limit('max_safe_height')
                self.update_gripper_location("home", x=0, z=max_height)
                self.logger.debug(f"Updated gripper location to home position after homing: (0, {max_height})")
            
        except Exception as e:
            self.logger.error(f"Failed to home robot components: {e}")
            raise

    def reset_after_initialization(self, max_retries=2):
        """
        Physical initialization and cleanup of North Robot.
        Handles homing, pump loading, liquid disposal, tip removal, and gripper cleanup.
        
        Args:
            max_retries (int): Maximum number of retry attempts for homing-related errors (default: 2)
        """
        self.logger.info("Physical initialization of North Robot...")
        self.c9.default_vel = self.get_speed('default_robot')  # Set the default speed of the robot
        self.c9.open_clamp()
        self.c9.zero_scale()
        self.c9.delay(3)
       
        
        
        
        # Attempt physical cleanup with retry logic for remaining errors
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.load_pumps()
                self._perform_physical_cleanup()
                break  # Success, exit retry loop
                
            except Exception as e:
                # For any remaining errors, just log and retry
                retry_count += 1
                self.logger.warning(f"Physical cleanup failed (attempt {retry_count}/{max_retries}): {e}")
                
                 # Home all components first before any pump operations
                self.logger.info("Homing robot components before pump initialization...")
                try:
                    self.home_robot_components()
                    # Load pumps after homing

                except Exception as e:
                    self.logger.error(f"Failed to home robot components during initialization: {e}")
                    raise

                if retry_count >= max_retries:
                    self.logger.error(f"Failed to complete physical cleanup after {max_retries} attempts: {e}")
                    raise
        
        # Final setup
        # Now load pumps after homing
        
        self.c9.open_gripper()
 
    def _perform_physical_cleanup(self):
        """
        Internal method to perform the actual physical cleanup tasks.
        Separated for cleaner retry logic.
        """
        # Handle leftover liquid in pipet tip
        if self.PIPET_FLUID_VIAL_INDEX is not None and self.PIPET_FLUID_VOLUME > 0:
            self.logger.warning(f"The robot reports having {self.PIPET_FLUID_VOLUME:.3f} mL liquid in its tip... Returning that liquid...")
            vial_index = self.PIPET_FLUID_VIAL_INDEX 
            volume_existing = self.get_vial_info(vial_index, 'vial_volume')
            self.dispense_into_vial(vial_index, 1.0) #Dispense 1.0 mL to fully empty the pipet
            self.VIAL_DF.loc[self.VIAL_DF['vial_index'] == vial_index, 'vial_volume'] = volume_existing + self.PIPET_FLUID_VOLUME
        elif self.PIPET_FLUID_VIAL_INDEX is not None:
            self.logger.info(f"Robot reports having a vial index for liquid but only {self.PIPET_FLUID_VOLUME:.3f} mL volume - skipping dispense")

        # Remove pipet tip if present
        if self.HELD_PIPET_TYPE is not None:
            self.logger.warning("The robot reports having a tip, removing the tip")
            self.remove_pipet()
            self.c9.move_pump(0, 0) 
        
        # Handle gripper contents
        if self.GRIPPER_STATUS is not None:
            vial_index = self.GRIPPER_VIAL_INDEX
            if self.GRIPPER_STATUS == "Cap":
                self.logger.warning("The robot reports having a cap in its gripper... Recapping the clamp vial...")
                self.recap_clamp_vial()
                self.return_vial_home(vial_index)
            elif self.GRIPPER_STATUS == "Vial":
                self.logger.warning("The robot reports having a vial in the gripper... Returning that vial home...")
                location = self.get_vial_info(vial_index, 'home_location')
                location_index = self.get_vial_info(vial_index, 'home_location_index')
                self.drop_off_vial(vial_index, location=location, location_index=location_index)
        
        # Move to home position
        self.move_home()

    # ====================================================================
    # STATE & STATUS MANAGEMENT
    # ====================================================================

    #Save the status of the robot to memory
    def save_robot_status(self):
        #self.logger.debug("Saving robot status to file: %s", self.ROBOT_STATUS_FILE)
        # Robot status data - ensure all values are Python native types to avoid YAML serialization issues
        robot_status = {
            "gripper_status": self.GRIPPER_STATUS,
            "gripper_vial_index": int(self.GRIPPER_VIAL_INDEX) if self.GRIPPER_VIAL_INDEX is not None else None,
            "held_pipet_type": self.HELD_PIPET_TYPE,
            "pipets_used": self.PIPETS_USED,  # Save all rack usage directly
            "pipet_fluid_vial_index": int(self.PIPET_FLUID_VIAL_INDEX) if self.PIPET_FLUID_VIAL_INDEX is not None else None,
            "pipet_fluid_volume": float(self.PIPET_FLUID_VOLUME) if self.PIPET_FLUID_VOLUME is not None else 0.0
        }

        if not self.simulate: 
            # Writing to a file
            self.VIAL_DF.to_csv(self.VIAL_FILE, index=False,sep=',') #Save the status of the vial dataframe
            with open(self.ROBOT_STATUS_FILE, "w") as file:
                yaml.dump(robot_status, file, default_flow_style=False)

    #Update the status of the robot from memory
    def get_robot_status(self):
        self.logger.debug("Getting robot status from file: %s", self.ROBOT_STATUS_FILE)
        
        # Get the vial dataframe
        try:
            self.VIAL_DF = pd.read_csv(self.VIAL_FILE, sep=",")
            self.VIAL_DF.index = self.VIAL_DF['vial_index'].values
        except:
            self.pause_after_error("Issue reading vial status", False)

        # Load robot state files using unified method
        self._load_state_files()

    # ====================================================================
    # 4. PIPETTING & TIP MANAGEMENT
    # ====================================================================

    #Remove the pipet tip
    def remove_pipet(self):
        self.logger.info("Removing pipet")
        self.c9.goto_safe(p_remove_approach, vel=self.get_speed('fast_approach'))
        #Get removal location from YAML config based on tip type
        if self.HELD_PIPET_TYPE is not None:
            removal_location_name = self.get_config_parameter('pipet_tips', self.HELD_PIPET_TYPE, 'removal_location', error_on_missing=True)
            if removal_location_name:
                removal_location = getattr(Locator, removal_location_name)
                self.c9.goto(removal_location, vel=self.get_speed('precise_movement'))
        remove_pipet_height = self.get_safe_height() #Constant height to remove the pipet (doesn't change with the pipet type, just moving up)
        self.c9.move_z(remove_pipet_height, vel=self.get_speed('default_robot'))
        self.HELD_PIPET_TYPE = None
        self.PIPET_FLUID_VIAL_INDEX = None
        self.PIPET_FLUID_VOLUME = 0
        self.save_robot_status() #Update in memory

    #Take a pipet tip from the active rack with the active pipet tip dimensions 
    def get_pipet(self, tip_type):
        """Get a pipet tip from available racks, prioritizing rack 1 over rack 2 for same tip type"""
        
        # Check if already holding a pipet tip
        if self.HELD_PIPET_TYPE is not None:
            self.logger.error(f"DEBUG: get_pipet called with tip_type='{tip_type}' but HELD_PIPET_TYPE='{self.HELD_PIPET_TYPE}'")
            self.pause_after_error("Can't get pipet, already have pipet tip")
            return  # Early return to prevent getting another pipet
        
        # Find available racks for this tip type, prioritizing rack 1 over rack 2
        available_racks = []
        for rack_name, rack_config in self.PIPET_RACKS.items():
            if rack_config.get('tip_type') == tip_type:
                available_racks.append(rack_name)
        
        # Sort racks to prioritize rack 1 over rack 2 for same tip type
        available_racks.sort()
        
        if not available_racks:
            self.pause_after_error(f"No racks found for tip type: {tip_type}")
            return
        
        # Find first rack with available tips
        selected_rack = None
        for rack_name in available_racks:
            tips_used = self.PIPETS_USED.get(rack_name, 0)
            max_tips = self.get_config_parameter('pipet_racks', rack_name, 'num_tips', error_on_missing=True)
            
            if max_tips is None:
                self.pause_after_error(f"No num_tips configured for rack {rack_name}")
                return
            
            if tips_used < max_tips:
                selected_rack = rack_name
                break
        
        if selected_rack is None:
            self.pause_after_error(f"All {tip_type} racks are empty! Please refill tips then hit enter on the terminal!")
            # Reset all racks to 0 tips used
            self.logger.info("Resetting all pipet rack counters to 0 after refill")
            self.PIPETS_USED = {rack_name: 0 for rack_name in self.PIPET_RACKS.keys()}
            self.save_robot_status()
            # Try again with first available rack
            selected_rack = available_racks[0]
        
        # Get rack configuration and current tip count
        active_pipet_num = self.PIPETS_USED.get(selected_rack, 0)
        
        self.logger.info(f"Getting {tip_type} (pipet number {active_pipet_num}) from rack: {selected_rack}")

        # Calculate tip position using YAML-configured ordering
        tip_position = self._calculate_tip_position(selected_rack, active_pipet_num)
        
        # Get capture location and move to pipet tip
        self._move_to_pipet_tip(selected_rack, tip_position)
        
        # Perform approach movement based on rack configuration
        self._perform_pipet_pickup(selected_rack, tip_position)

        # Update robot state
        self.HELD_PIPET_TYPE = tip_type
        self.PIPETS_USED[selected_rack] += 1
        self.save_robot_status()

    def refill_pipets(self, tip_type=None):
        """
        Reset pipet counters for manual refilling
        
        Args:
            tip_type (str, optional): Reset only racks of this tip type ('large_tip' or 'small_tip')
                                    If None, reset all racks
        """
        if tip_type:
            # Reset only racks of specified tip type
            for rack_name, rack_config in self.PIPET_RACKS.items():
                if rack_config.get('tip_type') == tip_type:
                    self.PIPETS_USED[rack_name] = 0
                    self.logger.info(f"Reset {rack_name} ({tip_type}) pipet counter to 0")
        else:
            # Reset all racks
            self.PIPETS_USED = {rack_name: 0 for rack_name in self.PIPET_RACKS.keys()}
            self.logger.info("Reset all pipet counters to 0")
        
        self.save_robot_status()
        self.logger.info("Pipet refill complete - robot status saved")

    def _calculate_tip_position(self, rack_name, active_pipet_num):
        """Calculate the tip position based on rack ordering configuration"""
        tip_ordering = self.get_config_parameter('pipet_racks', rack_name, 'tip_ordering', error_on_missing=True)
        
        if tip_ordering is None:
            self.pause_after_error(f"No tip_ordering configured for rack {rack_name}")
            return 0
        
        if tip_ordering == '3x16_standard':
            # Standard layout: 16 rows, 3 columns
            return (active_pipet_num % 16) * 3 + math.floor(active_pipet_num / 16)
        elif tip_ordering == '3x16_reverse':
            # Reverse layout: 3 columns, 16 rows, reverse ordering (0-47 range)
            return (16 * (math.floor(active_pipet_num / 16) + 1) - 1) - active_pipet_num % 16
        else:
            # Default: simple sequential ordering
            self.logger.warning(f"Unknown tip_ordering '{tip_ordering}' for rack {rack_name}, using sequential")
            return active_pipet_num

    def _move_to_pipet_tip(self, rack_name, tip_position):
        """Move to the pipet tip location based on rack configuration"""
        capture_location_name = self.get_config_parameter('pipet_racks', rack_name, 'capture_location', error_on_missing=True)
        if not capture_location_name:
            self.pause_after_error(f"No capture_location defined for rack {rack_name}")
            return
            
        capture_location = getattr(Locator, capture_location_name, None)
        if capture_location is None:
            self.pause_after_error(f"Capture location '{capture_location_name}' not found for rack {rack_name}")
            return
            
        # Move to the specific tip position
        location = capture_location[tip_position]
        self.c9.goto_xy_safe(location)
        
        # Move to the base height
        base_height = self.get_height_at_location(location)
        self.c9.move_z(base_height)

    def _perform_pipet_pickup(self, rack_name, tip_position):
        """Perform the pickup movement to capture the pipet tip"""
        pickup_location_name = self.get_config_parameter('pipet_racks', rack_name, 'pickup_location', error_on_missing=True)
        
        if pickup_location_name:
            # Use specific pickup location
            pickup_location = getattr(Locator, pickup_location_name, None)
            if pickup_location:
                location = pickup_location[tip_position]
                self.c9.goto(location, vel=self.get_speed('precise_movement'))
                return
        
        # Check for relative movement configuration
        pickup_movement = self.get_config_parameter('pipet_racks', rack_name, 'pickup_movement', error_on_missing=True)
        if pickup_movement:
            self.move_rel_xyz(
                x_distance=pickup_movement.get('x', 0),
                y_distance=pickup_movement.get('y', 0), 
                z_distance=pickup_movement.get('z', 0),
                vel=self.get_speed('precise_movement')
            )
            return
        
        # Check for safe height movement
        if self.get_config_parameter('pipet_racks', rack_name, 'safe_height_movement', error_on_missing=False):
            self.c9.move_z(self.get_safe_height(), vel=self.get_speed('precise_movement'))
            return
        
        # Default: no additional movement
        pass
 
    def adjust_pump_speed(self, pump, pump_speed):
        self.logger.debug(f"Adjusting pump {pump} speed to {pump_speed}")
        
        # Ensure pump_speed is a native Python int and round if needed
        if not isinstance(pump_speed, int):
            original_speed = pump_speed
            pump_speed = int(round(float(pump_speed)))
            self.logger.warning(f"Converting pump speed {original_speed} to integer {pump_speed}")
        
        # Validate pump speed is within hardware boundaries
        if not (1 <= pump_speed <= 40):
            error_msg = f"Pump speed {pump_speed} outside valid range (1-40)."
            self.pause_after_error(error_msg)
            return
            
        
        # Get current speed for this pump (default to 0 if not tracked)
        current_speed = self.CURRENT_PUMP_SPEEDS.get(pump, 0)
        
        if pump_speed != current_speed:
            self.c9.set_pump_speed(pump, pump_speed)
            self.CURRENT_PUMP_SPEEDS[pump] = pump_speed

#New aspirate function
    def pipet_aspirate(self, amount, wait_time=1.0):
        """
        Internal method: Aspirate liquid with a pipet.
        
        Args:
            amount (float): Volume to aspirate in mL
            wait_time (float): Time to wait after aspiration in seconds
        """
        self.logger.debug(f"Aspirating {amount:.3f} mL then waiting {wait_time} s")

        if amount <= 1:
            try:
                self.c9.aspirate_ml(0, amount)
            except:
                self.logger.warning("Aspirate exceeded limit: Aspirating to maximum")
                max_pump_pos = self.get_config_parameter('pumps', 0, 'max_pump_position', error_on_missing=False) or 3000
                self.c9.move_pump(0, max_pump_pos)
                self.logger.warning(f"Aspirate was exceeded for {amount} mL. Aspirating to maximum volume of 1 mL.")
        else:
            self.pause_after_error("Cannot aspirate more than 1 mL", True)

        if not self.simulate:
            time.sleep(wait_time)

#New dispense function
    def pipet_dispense(self, amount, wait_time=0.0, blowout_vol=0.0):
        """
        Internal method: Dispense liquid with a pipet.
        
        Args:
            amount (float): Volume to dispense in mL
            wait_time (float): Time to wait after dispensing in seconds
            blowout_vol (float): Volume for blowout after dispensing in mL
        """
        self.logger.debug(f"Dispensing {amount:.3f} mL then waiting {wait_time} s")
      
        try:
            self.c9.dispense_ml(0, amount)
        except: # If there's not enough to dispense
            self.logger.warning("Dispense exceeded limit: Dispensing all liquid")
            self.c9.move_pump(0, 0)

       

        if blowout_vol > 0: #Adjust this later into pipetting parameter
            #blow_speed = 5
            self.logger.debug(f"Blowing out {blowout_vol:.3f} mL")
            #self.adjust_pump_speed(0, blow_speed)
            self.c9.set_pump_valve(0, self.c9.PUMP_VALVE_LEFT)
            self.c9.aspirate_ml(0, blowout_vol)
            self.c9.set_pump_valve(0, self.c9.PUMP_VALVE_RIGHT)
            self.c9.dispense_ml(0, blowout_vol)

        if not self.simulate:
            time.sleep(wait_time)

    #Select appropriate tip type based on volume or use specified tip type
    def select_pipet_tip(self, volume, specified_tip_type=None):
        """
        Select the appropriate tip type based on volume or use the specified tip type
        
        Args:
            volume (float): Volume in mL to determine tip type
            specified_tip_type (str, optional): Force specific tip type ("small_tip" or "large_tip")
            
        Returns:
            str: Tip type ("small_tip" or "large_tip")
        """
        if specified_tip_type is not None:
            # Use the specified tip type directly (should be "small_tip" or "large_tip")
            return specified_tip_type
        
        # Get all available tip types from configuration
        available_tips = []
        for tip_name, tip_config in self.PIPET_TIPS.items():
            max_volume = tip_config.get('volume', 0)
            min_volume = tip_config.get('min_suggested_volume', 0)
            
            # Check if volume fits within this tip's range
            if min_volume <= volume <= max_volume:
                available_tips.append((tip_name, min_volume, max_volume))
        
        if not available_tips:
            self.pause_after_error(f"No suitable pipet tip found for volume {volume} mL")
            return None
        
        # Sort by min_suggested_volume (ascending) to prefer smaller tips for smaller volumes
        available_tips.sort(key=lambda x: x[1])
        
        # Return the most appropriate tip (smallest suitable tip)
        selected_tip = available_tips[0][0]
        self.logger.debug(f"Selected {selected_tip} for volume {volume:.3f} mL")
        return selected_tip
    
    def _get_optimized_parameters(self, volume, liquid=None, user_parameters=None, compensate_overvolume=True, smooth_overvolume=False):
        """
        Get optimized pipetting parameters using intelligent hierarchy:
        defaults → liquid-calibrated → user overrides
        
        Args:
            volume (float): Target pipetting volume in mL
            liquid (str, optional): Liquid type for calibrated parameters
            user_parameters (PipettingParameters, optional): User-provided overrides
            
        Returns:
            PipettingParameters: Optimized parameters with intelligent defaults
        """
        # Start with system defaults
        optimized = PipettingParameters()
        
        # Apply liquid-specific calibration if available
        if liquid is not None:
            try:
                wizard = PipettingWizard()
                calibrated_params = wizard.get_pipetting_parameters(liquid, volume, compensate_overvolume, smooth_overvolume)
                
                if calibrated_params is not None:
                    # Update only the parameters that were successfully calibrated
                    for param_name, param_value in calibrated_params.items():
                        if hasattr(optimized, param_name) and param_value is not None:
                            setattr(optimized, param_name, param_value)
                            self.logger.debug(f"Using calibrated {param_name}={param_value} for {liquid} at {volume:.3f}mL")
                            
            except Exception as e:
                # Graceful fallback - calibration is optional enhancement
                self.logger.debug(f"Could not load calibrated parameters for {liquid}: {e}")
        
        # Apply user overrides (highest priority)
        if user_parameters is not None:
            # Handle both dictionary and object formats
            if isinstance(user_parameters, dict):
                # Dictionary format: {'param_name': value}
                for param_name, user_value in user_parameters.items():
                    if hasattr(optimized, param_name) and user_value is not None:
                        setattr(optimized, param_name, user_value)
                        self.logger.debug(f"Applied dict param {param_name}={user_value}")
            else:
                # Object format with attributes
                for param_name in dir(user_parameters):
                    if not param_name.startswith('_'):  # Skip private attributes
                        user_value = getattr(user_parameters, param_name)
                        if user_value is not None:  # Only override if user explicitly set a value
                            setattr(optimized, param_name, user_value)
                            self.logger.debug(f"Applied object param {param_name}={user_value}")
                        
        return optimized
    
    def _get_optimized_reservoir_parameters(self, volume, liquid=None, reservoir_params=None):
        """
        Get optimized reservoir parameters using intelligent hierarchy:
        defaults → liquid-calibrated → user overrides
        
        Args:
            volume (float): Target dispensing volume in mL
            liquid (str, optional): Liquid type for calibrated parameters
            reservoir_params (ReservoirParameters, optional): User-provided overrides
            
        Returns:
            ReservoirParameters: Optimized parameters with intelligent defaults
        """
        # Start with system defaults
        optimized = ReservoirParameters()
        
        # Apply liquid-specific calibration if available
        if liquid is not None:
            try:
                # TODO: Implement ReservoirWizard when ready
                # wizard = ReservoirWizard()
                # calibrated_params = wizard.get_reservoir_parameters(liquid, volume)
                # 
                # if calibrated_params is not None:
                #     # Update only the parameters that were successfully calibrated
                #     for param_name, param_value in calibrated_params.items():
                #         if hasattr(optimized, param_name) and param_value is not None:
                #             setattr(optimized, param_name, param_value)
                #             self.logger.debug(f"Using calibrated reservoir {param_name}={param_value} for {liquid} at {volume:.3f}mL")
                
                self.logger.debug(f"Reservoir calibration not yet implemented for {liquid} - using defaults")
                            
            except Exception as e:
                # Graceful fallback - calibration is optional enhancement
                self.logger.debug(f"Could not load calibrated reservoir parameters for {liquid}: {e}")
        
        # Apply user overrides (highest priority)
        if reservoir_params is not None:
            # Handle both dictionary and object formats
            if isinstance(reservoir_params, dict):
                # Dictionary format: {'param_name': value}
                for param_name, user_value in reservoir_params.items():
                    if hasattr(optimized, param_name) and user_value is not None:
                        setattr(optimized, param_name, user_value)
                        self.logger.debug(f"Applied reservoir dict param {param_name}={user_value}")
            else:
                # Object format with attributes
                for param_name in dir(reservoir_params):
                    if not param_name.startswith('_'):  # Skip private attributes
                        user_value = getattr(reservoir_params, param_name)
                        if user_value is not None:  # Only override if user explicitly set a value
                            setattr(optimized, param_name, user_value)
                            self.logger.debug(f"Applied reservoir user param {param_name}={user_value}")
        
        return optimized

    #Check if the aspiration volume is within limits... Now extensible and configuration-driven
    def check_if_aspiration_volume_unacceptable(self, amount_mL):
        if self.HELD_PIPET_TYPE is None:
            return False  # No tip held, no volume restrictions
            
        # Get volume limits from YAML configuration
        tip_config = self.PIPET_TIPS.get(self.HELD_PIPET_TYPE, {})
        max_volume = tip_config.get('volume', 0)
        min_volume = tip_config.get('min_suggested_volume', 0)
        
        error_check_list = []
        
        # Check maximum volume limit
        if max_volume > 0:
            error_check_list.append([
                amount_mL > max_volume, 
                False, 
                f"Can't pipet more than {max_volume} mL from {self.HELD_PIPET_TYPE}"
            ])
        
        # Check minimum volume limit  
        if min_volume > 0:
            error_check_list.append([
                amount_mL < min_volume, 
                False, 
                f"Can't pipet less than {min_volume} mL from {self.HELD_PIPET_TYPE}"
            ])
        
        return self.check_for_errors(error_check_list, True)  # Return True if issue

    def normalize_vial_index(self, vial):
        """Accepts either a vial index (int) or vial name (str) and returns the vial index (int)."""
        if isinstance(vial, str):
            vial_idx = self.get_vial_index_from_name(vial)
            if vial_idx is None:
                raise ValueError(f"Vial name '{vial}' not found in VIAL_DF.")
            return vial_idx
        return vial

    #Get adjust the aspiration height based on how much is there
    def get_aspirate_height(self, source_vial_num, amount_mL, track_height=True, buffer=1.0):

        #Get required information
        base_height = self.get_min_pipetting_height(source_vial_num)
        vial_location = self.get_vial_info(source_vial_num, 'location')
        source_vial_volume = self.get_vial_info(source_vial_num,'vial_volume')

        # Get height_volume_constant from vial position configuration
        height_volume_constant = self.get_config_parameter('vial_positions', vial_location, 'height_volume_constant', error_on_missing=False) or 0

        volume_adjusted_height = base_height + (height_volume_constant*(source_vial_volume - amount_mL - buffer)) #How low should we go?

        if track_height and volume_adjusted_height > base_height: #If we are adjusting for height and have a height above the base (can't go lower)
            return volume_adjusted_height
        else:
            return base_height #Default is the lowest position

    def get_tip_dependent_aspirate_speed(self):
        """
        Get the appropriate aspiration speed based on the currently held pipet tip type.
        Pauses with error if no tip is held since aspiration requires a tip.
        
        Returns:
            int: Aspiration speed for the current tip type
        """
        if self.HELD_PIPET_TYPE is None:
            self.pause_after_error("Cannot get aspirate speed - no pipet tip is held")
            return 11  # Fallback to prevent crashes
            
        tip_speed = self.get_config_parameter('pipet_tips', self.HELD_PIPET_TYPE, 'default_aspirate_speed', error_on_missing=True)
        return tip_speed

    def _ensure_vial_accessible_for_pipetting(self, vial_name, use_safe_location=False, move_speed=None):
        """
        Helper method to ensure a vial is accessible for pipetting by moving it to clamp if needed.
        
        Args:
            vial_name (str or int): Name or index of the vial
            use_safe_location (bool): Force movement to safe location even if vial is pipetable
            
        Returns:
            bool: True if vial is now accessible, False if operation should be aborted
        """
        vial_num = self.normalize_vial_index(vial_name)
        
        vial_move_required = use_safe_location or not self.is_vial_pipetable(vial_num)
        if vial_move_required:
            self.logger.debug("Vial move required for pipetting")
            if self.is_vial_movable(vial_num):
                self.logger.info(f"Moving vial {vial_name} to safe pipetting location")

                # Check if clamp is occupied
                clamp_vial_index = self.get_vial_in_location('clamp', 0)
                vial_index = self.get_vial_index_from_name(vial_name)
                if clamp_vial_index is not None and clamp_vial_index != vial_index:
                    self.logger.debug(f"Clamp occupied by vial {clamp_vial_index}, returning it home first")
                    self.return_vial_home(clamp_vial_index, move_speed=move_speed)
                
                # Move vial to clamp and uncap only if it has a closed cap
                self.move_vial_to_location(vial_num, location='clamp', location_index=0, move_speed=move_speed)
                
                # Only uncap if vial has a closed cap (not open caps)
                vial_cap_type = self.get_vial_info(vial_num, 'cap_type')
                if vial_cap_type != 'open':
                    self.uncap_clamp_vial(move_speed=move_speed)
                else:
                    self.logger.debug(f"Vial {vial_name} has open cap, skipping uncap operation")
                
            else:
                self.pause_after_error(f"Vial {vial_name} cannot be moved to a safe pipetting location")
                return False
                
        return True


    def get_tip_if_needed(self, required_tip_type):
        """
        Ensure the robot is holding an appropriate pipet tip for the specified volume.
        If no tip is held or the held tip is inappropriate, it will get the correct tip.
        
        Args:
            volume (float): Volume in mL to determine appropriate tip"""
        if self.HELD_PIPET_TYPE is None:
            # No pipet held, get the required one
            self.get_pipet(required_tip_type)
        elif self.HELD_PIPET_TYPE != required_tip_type:
            # Wrong pipet type held, need to switch
            self.logger.info(f"Switching from {self.HELD_PIPET_TYPE} to {required_tip_type}")
            self.remove_pipet()
            self.get_pipet(required_tip_type)

    # ====================================================================
    # 5. LIQUID HANDLING OPERATIONS
    # ====================================================================

    #Aspirate from a vial using the pipet tool
    def aspirate_from_vial(self, source_vial_name, amount_mL, parameters=None, liquid=None,
                          move_to_aspirate=True, track_height=True, move_up=True, specified_tip=None, use_safe_location=False):
        """
        Aspirate amount_ml from a source vial.
        
        Args:
            source_vial_name (str): Name of the source vial to aspirate from
            amount_mL (float): Amount to aspirate in mL
            parameters (PipettingParameters, optional): Liquid handling parameters (uses defaults if None)
            liquid (str, optional): Liquid type for calibrated parameter optimization
            move_to_aspirate (bool): Whether to move to aspiration location (default: True)
            track_height (bool): Whether to track liquid height during aspiration (default: True)
            move_up (bool): Whether to retract after aspiration (default: True)
            specified_tip (str, optional): Force specific tip type ("small_tip" or "large_tip")
        """
        # Use intelligent parameter resolution: defaults → liquid-calibrated → user overrides
        parameters = self._get_optimized_parameters(amount_mL, liquid, parameters)
        
        # Extract liquid handling values from parameters
        asp_disp_cycles = parameters.asp_disp_cycles
        aspirate_wait_time = parameters.aspirate_wait_time
        pre_asp_air_vol = parameters.pre_asp_air_vol
        post_asp_air_vol = parameters.post_asp_air_vol
        overaspirate_vol = parameters.overaspirate_vol
        post_retract_wait_time = parameters.post_retract_wait_time
        total_tip_vol = post_asp_air_vol + amount_mL + overaspirate_vol

        
        source_vial_num = self.normalize_vial_index(source_vial_name) #Convert to int if needed     

        # Ensure vial is accessible for pipetting
        if not self._ensure_vial_accessible_for_pipetting(source_vial_name, use_safe_location):
            return

        #Check if has pipet, get one if needed based on volume being aspirated (or if tip is specified)
        required_tip_type = self.select_pipet_tip(total_tip_vol, specified_tip)

        #Get a tip if we need one
        self.get_tip_if_needed(required_tip_type)
        
        #Check for an issue with the pipet and the specified amount, pause and send slack message if so
        self.check_if_aspiration_volume_unacceptable(total_tip_vol) 
        
        # Now that we have a pipet tip, we can safely determine speeds
        aspirate_speed = parameters.aspirate_speed or self.get_tip_dependent_aspirate_speed()
        retract_speed = parameters.retract_speed or self.get_speed('retract')

        #Get current volume and vial location
        location = self.get_vial_location(source_vial_num,True)
        source_vial_volume = self.get_vial_info(source_vial_num,'vial_volume')

        #Reject aspiration if the volume is not high enough
        if source_vial_volume < amount_mL:
            self.pause_after_error("Cannot aspirate more volume than in vial")

        self.logger.info(f"Pipetting from vial {self.get_vial_info(source_vial_num, 'vial_name')}, amount: {amount_mL:.3f} mL")
        self.logger.info(f"Additional volume from calibration, amount: {overaspirate_vol:.3f} mL")

        #Adjust the height based on the volume, then pipet type
        if move_to_aspirate:
            asp_height = self.get_aspirate_height(source_vial_name, amount_mL, track_height)
            asp_height = self.adjust_height_based_on_pipet_held(asp_height)

            #The dispense height is right above the location
            disp_height = self.get_height_at_location(location)
            disp_height = self.adjust_height_based_on_pipet_held(disp_height)
        else:
            asp_height = disp_height = self.get_current_height() #IF we aren't moving, everything stays the same

        self.adjust_pump_speed(0,10)

        #Move to the correct location in xy
        if move_to_aspirate:
            self.c9.goto_xy_safe(location, vel=self.get_speed('standard_xy'))

        
        #Step 1: Move to above the site and aspirate air if needed
        if pre_asp_air_vol > 0:
            self.c9.move_z(disp_height)
            self.pipet_aspirate(pre_asp_air_vol, wait_time=0)
        
        self.adjust_pump_speed(0,aspirate_speed) #Adjust the pump speed if needed

        #Step 2: Move to inside the site and aspirate liquid
        self.c9.move_z(asp_height)
        for i in range (0, asp_disp_cycles):
            self.pipet_aspirate(amount_mL+overaspirate_vol, wait_time=0)
            self.pipet_dispense(amount_mL+overaspirate_vol, wait_time=0)
        self.pipet_aspirate(amount_mL+overaspirate_vol, wait_time=aspirate_wait_time) #Main aspiration of liquid plus wait
        
        #Step 3: Retract and aspirate air if needed
        if move_up:
            self.c9.move_z(disp_height, vel=retract_speed) #Retract with a specific speed
            time.sleep(post_retract_wait_time) #Wait after retracting
        if post_asp_air_vol > 0:
            self.pipet_aspirate(post_asp_air_vol, wait_time=0) 

        #Record the volume change
        self.VIAL_DF.at[source_vial_num,'vial_volume']=(source_vial_volume-amount_mL)

        #Update the new volume in memory
        self.PIPET_FLUID_VIAL_INDEX = int(source_vial_num)
        self.PIPET_FLUID_VOLUME += amount_mL
        self.save_robot_status()
    
    #Adjust height based on the currently held pipet tip - now extensible to all tip types
    def adjust_height_based_on_pipet_held(self, height):
        height_shift_pipet = 0
        
        if self.HELD_PIPET_TYPE is not None:
            # Get delta_z_to_tip_bottom for the currently held pipet type
            height_shift_pipet = self.get_config_parameter('pipet_tips', self.HELD_PIPET_TYPE, 'delta_z_to_tip_bottom', error_on_missing=False) or 0
            
        height += height_shift_pipet
        return height

    #This method dispenses from a vial into another vial, using buffer transfer to improve accuracy if needed.
    #TODO: Maybe get rid of the buffer option here and replace with the other new parameters and potentially blowout
    def dispense_from_vial_into_vial(self, source_vial_name, dest_vial_name, volume, parameters=None, liquid=None, specified_tip=None, 
                                     remove_tip=True, use_safe_location=False, return_vial_home=True, move_speed=None, 
                                     compensate_overvolume=True, smooth_overvolume=False, measure_weight=False, 
                                     continuous_mass_monitoring=False, save_mass_data=False):
        """
        Transfer liquid from source vial to destination vial.

        Args:
            source_vial_name (str): Name of the source vial to aspirate from
            dest_vial_name (str): Name of the destination vial to dispense into
            volume (float): Volume (in mL) to transfer
            parameters (PipettingParameters, optional): Standardized parameters
            liquid (str, optional): Liquid type for calibrated parameter optimization
            specified_tip (str, optional): Force specific tip type ("small_tip" or "large_tip")
            compensate_overvolume (bool): Apply overvolume compensation based on measured accuracy (default: True)
            smooth_overvolume (bool): Apply local smoothing to remove overvolume outliers (default: False)
        """
        # Use intelligent parameter resolution: defaults → liquid-calibrated → user overrides
        parameters = self._get_optimized_parameters(volume, liquid, parameters, compensate_overvolume, smooth_overvolume)

        self.logger.info(f"Dispensing {volume:.3f} mL from {source_vial_name} to {dest_vial_name}")

        source_vial_index = self.normalize_vial_index(source_vial_name)
        dest_vial_index = self.normalize_vial_index(dest_vial_name)

        if volume <= 0:
            self.logger.warning("Cannot dispense <=0 mL")
            return

        # Calculate total tip volume requirement including optimized parameters
        total_tip_volume_required = (volume + 
                                    parameters.overaspirate_vol + 
                                    parameters.pre_asp_air_vol + 
                                    parameters.post_asp_air_vol)

        # Handle large volumes by splitting based on TOTAL tip volume requirement
        max_system_volume = max((tip_config.get('volume', 0) for tip_config in self.PIPET_TIPS.values()), default=1.0)
        repeats = 1
        
        if total_tip_volume_required > max_system_volume:
            # Calculate how many splits we need based on total tip volume
            repeats = math.ceil(total_tip_volume_required / max_system_volume)
            volume = volume / repeats  # Split the BASE volume, not the total tip volume
            self.logger.info(f"Total tip volume ({total_tip_volume_required:.3f} mL) exceeds capacity, splitting into {repeats} transfers of {round(volume,3)} mL each")
            
            # Recalculate optimized parameters for the new smaller volume
            parameters = self._get_optimized_parameters(volume, liquid, parameters)

        total_mass = 0
        for i in range(repeats):
            last_run = (i == repeats - 1)

            # Aspirate from source (at normal speed)
            self.aspirate_from_vial(source_vial_index, round(volume, 3), parameters=parameters, liquid=liquid, specified_tip=specified_tip, use_safe_location=use_safe_location)

            # Set custom movement speed if specified - after aspiration when carrying liquid
            original_vel = None
            if move_speed is not None:
                original_vel = self.c9.default_vel
                self.c9.default_vel = move_speed

            # Dispense into destination (at modified speed)
            dispense_result = self.dispense_into_vial(dest_vial_index, volume, parameters=parameters, liquid=liquid, move_speed=move_speed, measure_weight=measure_weight,
                                                      continuous_mass_monitoring=continuous_mass_monitoring, save_mass_data=save_mass_data)
            
            # Handle both old and new return formats for backwards compatibility
            if isinstance(dispense_result, tuple):
                mass_increment, stability_info = dispense_result
            else:
                mass_increment = dispense_result
                stability_info = None
                
            total_mass += mass_increment if mass_increment is not None else 0

            # Restore original movement speed AFTER dispense but BEFORE recap/return operations
            if move_speed is not None and original_vel is not None:
                self.c9.default_vel = original_vel


            # Return vials and remove pipet on last run
            if last_run:
                if remove_tip:
                    self.remove_pipet()
                # Return whatever vial is currently in the clamp (works for all cases)
                clamp_vial_index = self.get_vial_in_location('clamp', 0)
                if clamp_vial_index is not None and return_vial_home:
                    # Only recap if vial is actually uncapped and robot has the cap
                    vial_is_capped = self.get_vial_info(clamp_vial_index, 'capped')
                    robot_has_cap = (self.GRIPPER_STATUS == "Cap")
                    
                    if not vial_is_capped and robot_has_cap:
                        self.recap_clamp_vial()  # Use default speed for recapping
                    elif not vial_is_capped and not robot_has_cap:
                        self.logger.warning(f"Vial {clamp_vial_index} is uncapped but robot doesn't have cap - leaving uncapped")
                    # If vial is already capped, skip recapping
                    
                    self.return_vial_home(clamp_vial_index)  # Use default speed for returning vial


        return total_mass, stability_info

    #TODO add error checks and safeguards
    def pipet_from_wellplate(self, wp_index, volume, parameters=None, liquid=None, aspirate=True, specified_tip=None, move_to_aspirate=True, well_plate_type="96 WELL PLATE"):
        """
        Aspirate or dispense from/to a wellplate.
        
        Args:
            wp_index: Well plate index
            volume: Volume in mL
            parameters (PipettingParameters, optional): Liquid handling parameters (uses defaults if None)
            liquid (str, optional): Liquid type for calibrated parameter optimization
            aspirate (bool): True for aspirate, False for dispense (default: True)
            move_to_aspirate (bool): Whether to move to location (default: True)
            well_plate_type (str): Type of well plate (default: "96 WELL PLATE")
        """
        # Use intelligent parameter resolution: defaults → liquid-calibrated → user overrides
        parameters = self._get_optimized_parameters(volume, liquid, parameters)

        # Calculate volumes based on operation
        if aspirate:
            # For aspiration, need to consider overaspirate volume for tip selection
            total_tip_vol = volume + parameters.overaspirate_vol
            actual_aspirate_vol = volume + parameters.overaspirate_vol
        else:
            # For dispensing, calculate overdispense volume
            overdispense_vol = (volume + 
                               parameters.overaspirate_vol + 
                               parameters.pre_asp_air_vol + 
                               parameters.post_asp_air_vol)
            total_tip_vol = volume  # Use base volume for tip selection in dispense
        
        #Check if has pipet, get one if needed based on volume being aspirated (or if tip is specified)
        required_tip_type = self.select_pipet_tip(total_tip_vol, specified_tip)

        #Get a tip if we need one
        self.get_tip_if_needed(required_tip_type)
        

        # Get appropriate speed and wait time based on operation
        if aspirate:
            speed = parameters.aspirate_speed or self.get_tip_dependent_aspirate_speed()
            wait_time = parameters.aspirate_wait_time
            blowout_vol = 0.0  # No blowout for aspiration
        else:
            speed = parameters.dispense_speed or self.get_tip_dependent_aspirate_speed()
            wait_time = parameters.dispense_wait_time
            blowout_vol = parameters.blowout_vol
            
        self.logger.debug(f"Pipetting {'aspirating' if aspirate else 'dispensing'} {volume:.3f} mL from well plate index {wp_index} of type {well_plate_type}")
        
        location = self.select_wellplate_grid(well_plate_type)[wp_index]
        
        height = self.get_height_at_location(location)
        height = self.adjust_height_based_on_pipet_held(height) 

        # Get the height adjustment for the well plate type (distance to bottom of well)
        height_adjust = self.WELLPLATES[well_plate_type]["distance_to_bottom_for_pipet"] if well_plate_type in self.WELLPLATES else 0

        if aspirate:
            height = height - height_adjust #Go to the bottom of the well

        if move_to_aspirate:
            self.c9.goto_xy_safe(location, vel=self.get_speed('standard_xy'))
            self.c9.move_z(height)

        self.adjust_pump_speed(0, speed)
        if aspirate:
            self.pipet_aspirate(actual_aspirate_vol, wait_time=wait_time)
        else:
            self.pipet_dispense(overdispense_vol, wait_time=wait_time, blowout_vol=blowout_vol)

    #Mix the well
    def mix_well_in_wellplate(self, wp_index, volume, repeats=3, well_plate_type="96 WELL PLATE", parameters=None, liquid=None):
        """
        Mix a well in a wellplate by aspirating and dispensing repeatedly.
        
        Args:
            wp_index: Well plate index to mix
            volume: Volume to use for mixing in mL
            repeats: Number of mixing cycles (default: 3)
            well_plate_type: Type of well plate (default: "96 WELL PLATE")
            parameters (PipettingParameters, optional): Liquid handling parameters (uses defaults if None)
            liquid (str, optional): Liquid type for calibrated parameter optimization
        """
        self.logger.info(f"Mixing well {wp_index} in well plate of type {well_plate_type} with volume {volume:.3f} mL for {repeats} repeats")
        self.pipet_from_wellplate(wp_index, volume, parameters=parameters, liquid=liquid, well_plate_type=well_plate_type)
        self.pipet_from_wellplate(wp_index, volume, parameters=parameters, liquid=liquid, aspirate=False, move_to_aspirate=False, well_plate_type=well_plate_type)
        for i in range(1, repeats):
            self.pipet_from_wellplate(wp_index, volume, parameters=parameters, liquid=liquid, move_to_aspirate=False, well_plate_type=well_plate_type)
            self.pipet_from_wellplate(wp_index, volume, parameters=parameters, liquid=liquid, aspirate=False, move_to_aspirate=False, well_plate_type=well_plate_type)

    #Mix in a vial
    def mix_vial(self, vial_name, volume, repeats=3, parameters=None, liquid=None):
        """
        Mix a vial by aspirating and dispensing repeatedly.
        
        Args:
            vial_name: Name or index of the vial to mix
            volume: Volume to use for mixing in mL
            repeats: Number of mixing cycles (default: 3)
            parameters (PipettingParameters, optional): Liquid handling parameters (uses defaults if None)
            liquid (str, optional): Liquid type for calibrated parameter optimization
        """
        self.logger.info(f"Mixing vial {vial_name} with volume {volume:.3f} mL for {repeats} repeats")
        vial_index = self.normalize_vial_index(vial_name)
        self.aspirate_from_vial(vial_index, volume, parameters=parameters, liquid=liquid, move_up=False, track_height=False)
        self.dispense_into_vial(vial_index, volume, parameters=parameters, liquid=liquid, initial_move=False)
        for i in range(1, repeats):
            self.dispense_from_vial_into_vial(vial_index, vial_index, volume, parameters=parameters, liquid=liquid)

    #Dispense an amount into a vial
    def dispense_into_vial(self, dest_vial_name, amount_mL, parameters=None, liquid=None,
                          initial_move=True, measure_weight=False, continuous_mass_monitoring=False, 
                          save_mass_data=False, move_speed=None):
        """
        Dispense liquid into a vial.
        
        Args:
            dest_vial_name: Name of destination vial
            amount_mL: Amount to dispense in mL
            parameters (PipettingParameters, optional): Liquid handling parameters (uses defaults if None)
            liquid (str, optional): Liquid type for calibrated parameter optimization
            initial_move (bool): Whether to perform initial movement (default: True)
            measure_weight (bool): Whether to measure mass before/after dispensing (default: False)
            continuous_mass_monitoring (bool): Whether to continuously monitor mass during dispensing (default: False)
            save_mass_data (bool): Whether to save continuous mass data to file (default: False)
        """
        # Use intelligent parameter resolution: defaults → liquid-calibrated → user overrides
        parameters = self._get_optimized_parameters(amount_mL, liquid, parameters)
        
        # Extract liquid handling values from parameters
        wait_time = parameters.dispense_wait_time
        blowout_vol = parameters.blowout_vol
        
        # Calculate overdispense volume from constituent parts
        overdispense_vol = (amount_mL + 
                           parameters.overaspirate_vol + 
                           parameters.pre_asp_air_vol + 
                           parameters.post_asp_air_vol)
        
        dest_vial_num = self.normalize_vial_index(dest_vial_name) #Convert to int if needed

        # Ensure vial is accessible for pipetting (no use_safe_location for dispense)
        if not self._ensure_vial_accessible_for_pipetting(dest_vial_name, use_safe_location=False, move_speed=move_speed):
            return

        measured_mass = None
        continuous_mass_data = []
        stability_info = {
            'pre_stable_count': 0,
            'pre_total_count': 0, 
            'post_stable_count': 0,
            'post_total_count': 0,
            'pre_baseline_std': 0.0,
            'post_baseline_std': 0.0
        }

        self.logger.info(f"Pipetting into vial {self.get_vial_info(dest_vial_num, 'vial_name')}, amount: {amount_mL:.3f} mL")
        
        dest_vial_clamped = self.get_vial_info(dest_vial_num,'location')=='clamp' #Is the destination vial clamped?
        dest_vial_volume = self.get_vial_info(dest_vial_num,'vial_volume') #What is the current vial volume?

        #If the destination vial is at the clamp and you want weight measurement using TRADITIONAL method
        if measure_weight and not continuous_mass_monitoring and dest_vial_clamped:
            if not self.simulate:
                # Zero the scale before measurement to prevent baseline drift
                initial_mass = self.c9.read_steady_scale()
                self.logger.info(f"Initial mass reading (traditional): {initial_mass:.6f} g")
            else:
                initial_mass = 0
                self.logger.info("Simulation mode - initial mass set to 0")
        else:
            initial_mass = None  # Will be handled by continuous monitoring if needed

        # Determine dispense speed (can be done safely here since no pipet acquisition needed for dispense)
        dispense_speed = parameters.dispense_speed or self.get_tip_dependent_aspirate_speed()
        
        self.adjust_pump_speed(0,dispense_speed) #Adjust pump speed if needed

        #Where is the vial?
        location= self.get_vial_location(dest_vial_num,True)
        
        #What height do we need to go to?
        height = self.get_height_at_location(location) #baseline z-height
        height = self.adjust_height_based_on_pipet_held(height)

        #Move to the location if told to (Could change this to an auto-check)
        if initial_move:               
            self.c9.goto_xy_safe(location, vel=move_speed if move_speed is not None else self.get_speed('standard_xy'))
        
        # Setup continuous mass monitoring if requested
        monitoring_active = threading.Event()
        
        def read_scale_continuous():
            """Continuously read scale at maximum rate (5 Hz) during dispensing"""
            while monitoring_active.is_set():
                if not self.simulate and dest_vial_clamped:
                    try:
                        current_time = time.time()
                        steady_status, weight = self.c9.read_scale()  # Get BOTH stability and weight
                        
                        # Determine phase based on timing
                        time_relative = current_time - start_time
                        if time_relative < 1.0:
                            phase = 'baseline_pre'
                        elif dispense_start_time is None or current_time < dispense_start_time:
                            phase = 'baseline_pre'  # Still in pre-baseline
                        elif dispense_end_time is None or current_time < dispense_end_time:
                            phase = 'dispensing'
                        else:
                            phase = 'baseline_post'
                        
                        continuous_mass_data.append({
                            'timestamp': current_time,
                            'time_relative': time_relative,
                            'mass_g': weight,
                            'phase': phase,
                            'steady_status': steady_status  # NEW: Track if scale says reading is stable
                        })
                        self.logger.debug(f'Mass: {weight:.4f}g at {time_relative:.3f}s ({phase}) steady={steady_status}')
                    except Exception as e:
                        self.logger.warning(f"Scale reading failed: {e}")
                if not self.simulate:
                    time.sleep(0.2)  # 5 Hz maximum rate (0.2 second intervals)
        
        #Pipet into the vial
        if initial_move:
            self.c9.move_z(height, vel=move_speed if move_speed is not None else None)
            
        # Start continuous monitoring if requested (includes pre-dispense baseline)
        if measure_weight and continuous_mass_monitoring and dest_vial_clamped:
            # Conditional settling based on dispense wait time
            self.logger.info(f"Short wait time ({1.0:.1f}s), allowing scale to settle after robot positioning...")
            if not self.simulate:
                time.sleep(1.0)  # 2 second settling delay for short wait times
            
            start_time = time.time()
            dispense_start_time = None  # Will be set when dispensing starts
            dispense_end_time = None    # Will be set when dispensing ends
            monitoring_active.set()
            monitor_thread = threading.Thread(target=read_scale_continuous, daemon=True)
            monitor_thread.start()
            self.logger.info("Started continuous mass monitoring with pre-dispense baseline")
            
            # Wait for 1.0s baseline before dispensing for better stability
            if not self.simulate:
                time.sleep(2.0)
            
        # Record when actual dispensing starts
        if measure_weight and continuous_mass_monitoring and dest_vial_clamped:
            dispense_start_time = time.time()
        
        # Perform the actual dispensing
        self.pipet_dispense(overdispense_vol, wait_time=wait_time, blowout_vol=blowout_vol)
        
        # Record when dispensing ends and wait for post-dispense baseline
        if measure_weight and continuous_mass_monitoring and dest_vial_clamped:
            dispense_end_time = time.time()
            dispense_duration = dispense_end_time - dispense_start_time
            self.logger.info(f"Dispensing took {dispense_duration:.3f}s, collecting post-dispense baseline")
            
            # Let monitoring thread continue for post-baseline period
            if not self.simulate:
                time.sleep(2.0)  # 1.0s post-dispense baseline for better data
            
            # Stop continuous monitoring after collecting post-baseline data
            monitoring_active.clear()
            monitor_thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
            self.logger.info(f"Stopped continuous mass monitoring. Collected {len(continuous_mass_data)} data points")
        
         #Track the added volume in the dataframe
        self.VIAL_DF.at[dest_vial_num,'vial_volume']=self.VIAL_DF.at[dest_vial_num,'vial_volume']+amount_mL
        self.PIPET_FLUID_VOLUME -= amount_mL
        self.save_robot_status()

        # Analyze continuous mass data after monitoring is stopped
        if measure_weight and continuous_mass_monitoring and dest_vial_clamped and continuous_mass_data:
            if len(continuous_mass_data) >= 4:
                mass_df = pd.DataFrame(continuous_mass_data)
                
                # Calculate baselines using STABILITY instead of time phases
                
                # OLD WAY: Use arbitrary time phases (might include unstable readings)
                # pre_baseline = mass_df[mass_df['phase'] == 'baseline_pre']['mass_g'].mean()
                # post_baseline = mass_df[mass_df['phase'] == 'baseline_post']['mass_g'].mean()
                
                # NEW WAY: Use actual stability status from scale hardware
                pre_stable_readings = mass_df[(mass_df['phase'] == 'baseline_pre') & 
                                             (mass_df['steady_status'] == True)]
                post_stable_readings = mass_df[(mass_df['phase'] == 'baseline_post') & 
                                              (mass_df['steady_status'] == True)]
                
                # Calculate mass difference using INDEPENDENT baseline analysis
                # Each baseline uses stable readings if available, otherwise falls back to time-based
                
                # Pre-baseline calculation
                if len(pre_stable_readings) > 0:
                    pre_baseline = pre_stable_readings['mass_g'].mean()
                    pre_method = "STABLE"
                    self.logger.info(f"Pre-baseline: {len(pre_stable_readings)} stable readings, avg={pre_baseline:.6f}g")
                else:
                    pre_baseline = mass_df[mass_df['phase'] == 'baseline_pre']['mass_g'].mean()
                    pre_method = "TIME-BASED"
                    self.logger.info(f"Pre-baseline: No stable readings, using time-based avg={pre_baseline:.6f}g")
                
                # Post-baseline calculation  
                if len(post_stable_readings) > 0:
                    post_baseline = post_stable_readings['mass_g'].mean()
                    post_method = "STABLE"
                    self.logger.info(f"Post-baseline: {len(post_stable_readings)} stable readings, avg={post_baseline:.6f}g")
                else:
                    post_baseline = mass_df[mass_df['phase'] == 'baseline_post']['mass_g'].mean()
                    post_method = "TIME-BASED"
                    self.logger.info(f"Post-baseline: No stable readings, using time-based avg={post_baseline:.6f}g")
                
                # Calculate final result
                measured_mass = post_baseline - pre_baseline
                self.logger.info(f"INDEPENDENT BASELINE calculation:")
                self.logger.info(f"Pre: {pre_method}, Post: {post_method}, Mass difference: {measured_mass:.6f}g")
                
                # Log comparison info
                pre_count = len(mass_df[mass_df['phase'] == 'baseline_pre'])
                post_count = len(mass_df[mass_df['phase'] == 'baseline_post'])
                pre_stable_count = len(pre_stable_readings) if len(pre_stable_readings) > 0 else 0
                post_stable_count = len(post_stable_readings) if len(post_stable_readings) > 0 else 0
                
                # Update stability info
                pre_baseline_readings = mass_df[mass_df['phase'] == 'baseline_pre']['mass_g']
                post_baseline_readings = mass_df[mass_df['phase'] == 'baseline_post']['mass_g']
                
                stability_info.update({
                    'pre_stable_count': pre_stable_count,
                    'pre_total_count': pre_count,
                    'post_stable_count': post_stable_count, 
                    'post_total_count': post_count,
                    'pre_baseline_std': pre_baseline_readings.std() if len(pre_baseline_readings) > 1 else 0.0,
                    'post_baseline_std': post_baseline_readings.std() if len(post_baseline_readings) > 1 else 0.0
                })
                
                self.logger.info(f"Reading summary: {pre_count} pre-readings ({pre_stable_count} stable), "
                               f"{post_count} post-readings ({post_stable_count} stable)")
                self.logger.info(f"Baseline variability: pre_std={stability_info['pre_baseline_std']:.6f}g, "
                               f"post_std={stability_info['post_baseline_std']:.6f}g")
                self.logger.info(f"Target: {amount_mL:.3f} mL = ~{amount_mL*1.0:.6f} g for water")
            else:
                self.logger.warning("Insufficient continuous data for baseline calculation")
                measured_mass = 0.0
            
            # Save mass data if requested
            if save_mass_data and continuous_mass_data:
                import os
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                
                # Use log filename for folder organization instead of date
                if hasattr(self, 'log_filename') and self.log_filename:
                    # Remove .log extension and use as folder name
                    log_folder = self.log_filename.replace('.log', '')
                else:
                    # Fallback to date if log filename not available
                    log_folder = time.strftime("%Y-%m-%d")
                
                filename = f"mass_data_{dest_vial_name}_{timestamp_str}.csv"
                filepath = os.path.join("output", "mass_measurements", log_folder, filename)
                
                # Ensure log-specific directory exists
                os.makedirs(os.path.join("output", "mass_measurements", log_folder), exist_ok=True)
                
                # Convert to DataFrame and save
                mass_df = pd.DataFrame(continuous_mass_data)
                mass_df.to_csv(filepath, index=False)
                self.logger.info(f"Saved {len(continuous_mass_data)} mass measurements to {filepath}")
                
                # Create and save matplotlib graph with continuous line and phase backgrounds
                plt.figure(figsize=(12, 6))
                
                # Plot one continuous line
                plt.plot(mass_df['time_relative'], mass_df['mass_g'], 'k-', linewidth=2, label='Mass readings', zorder=3)
                
                # Add scatter overlay to show stable vs unstable readings
                stable_data = mass_df[mass_df['steady_status'] == True]
                unstable_data = mass_df[mass_df['steady_status'] == False]
                
                if len(stable_data) > 0:
                    plt.scatter(stable_data['time_relative'], stable_data['mass_g'], 
                              c='green', s=20, alpha=0.7, label='Stable readings', zorder=4)
                if len(unstable_data) > 0:
                    plt.scatter(unstable_data['time_relative'], unstable_data['mass_g'], 
                              c='red', s=20, alpha=0.7, label='Unstable readings', zorder=4)
                
                # Add phase background shading
                if len(mass_df) > 0:
                    # Get phase transition points
                    pre_data = mass_df[mass_df['phase'] == 'baseline_pre']
                    dispense_data = mass_df[mass_df['phase'] == 'dispensing'] 
                    post_data = mass_df[mass_df['phase'] == 'baseline_post']
                    
                    y_min, y_max = plt.ylim()
                    
                    # Shade background regions
                    if len(pre_data) > 0:
                        plt.axvspan(pre_data['time_relative'].min(), pre_data['time_relative'].max(), 
                                   alpha=0.2, color='green', label='Pre-baseline', zorder=1)
                    if len(dispense_data) > 0:
                        plt.axvspan(dispense_data['time_relative'].min(), dispense_data['time_relative'].max(), 
                                   alpha=0.2, color='blue', label='Dispensing', zorder=1)
                    if len(post_data) > 0:
                        plt.axvspan(post_data['time_relative'].min(), post_data['time_relative'].max(), 
                                   alpha=0.2, color='red', label='Post-baseline', zorder=1)
                
                # Add horizontal lines for averaged baselines
                plt.axhline(y=pre_baseline, color='g', linestyle='--', alpha=0.7, label=f'Pre-avg: {pre_baseline:.3f}g')
                plt.axhline(y=post_baseline, color='r', linestyle='--', alpha=0.7, label=f'Post-avg: {post_baseline:.3f}g')
                
                plt.xlabel('Time (seconds)')
                plt.ylabel('Mass (g)')
                plt.title(f'Mass vs Time During Dispense\nVial: {dest_vial_name}, Target: {amount_mL:.3f} mL, Measured: {measured_mass:.4f} g')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                plot_filename = f"mass_plot_{dest_vial_name}_{timestamp_str}.png"
                plot_filepath = os.path.join("output", "mass_measurements", log_folder, plot_filename)
                plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure to free memory
                self.logger.info(f"Saved mass vs time plot to {plot_filepath}")  

        # Traditional before/after measurement (only when continuous monitoring is NOT enabled)
        elif measure_weight and not continuous_mass_monitoring and dest_vial_clamped:
            if not self.simulate:
                final_mass = self.c9.read_steady_scale()
                self.logger.info(f"Final mass reading: {final_mass:.6f} g")
            else:
                final_mass = 0
                self.logger.info("Simulation mode - final mass set to 0")
            measured_mass = final_mass - initial_mass  
            self.logger.info(f"Mass difference: {measured_mass:.6f} g (target: {amount_mL:.3f} mL = ~{amount_mL*1.0:.6f} g for water)")

        return measured_mass, stability_info

    def monitor_weight_over_time(self, time_s):
        """
        Monitor scale weight continuously for a specified duration at maximum rate.
        
        Useful for studying evaporation, scale drift, environmental effects, 
        or general weight stability characterization.
        
        Args:
            time_s (float): Duration to monitor in seconds
        
        Returns:
            list[dict]: Weight data containing:
                - timestamp: absolute time 
                - time_relative: seconds from start
                - mass_g: weight reading
                - steady_status: True if scale considers reading stable
        """
        if self.simulate:
            self.logger.info("Simulation mode - generating dummy weight data")
            import random
            dummy_data = []
            start_time = time.time()
            
            # Simulate at ~5 Hz for realistic timing
            for i in range(int(time_s * 5)):
                current_time = start_time + (i * 0.2)
                # Simulate slight evaporation (gradual decrease) + noise
                base_mass = 10.0 - (i * 0.0001)  # Slow evaporation
                noise = random.uniform(-0.0005, 0.0005)  # Small noise
                dummy_data.append({
                    'timestamp': current_time,
                    'time_relative': i * 0.2,
                    'mass_g': base_mass + noise,
                    'steady_status': True  # Assume stable in simulation
                })
            
            self.logger.info(f"Completed simulated weight monitoring: {len(dummy_data)} readings over {time_s}s")
            return dummy_data
        
        else:
            self.logger.info(f"Starting weight monitoring: {time_s}s duration at maximum rate")
            
            weight_data = []
            start_time = time.time()
            
            while (time.time() - start_time) < time_s:
                try:
                    current_time = time.time()
                    steady_status, weight = self.c9.read_scale()
                    time_relative = current_time - start_time
                    
                    weight_data.append({
                        'timestamp': current_time,
                        'time_relative': time_relative,
                        'mass_g': weight,
                        'steady_status': steady_status
                    })
                    
                    self.logger.debug(f"Weight: {weight:.6f}g at {time_relative:.3f}s (steady={steady_status})")
                    
                    # Minimal sleep to prevent excessive CPU usage
                    if not self.simulate:
                        time.sleep(0.01)
                    
                except Exception as e:
                    self.logger.warning(f"Scale reading failed: {e}")
                    if not self.simulate:
                        time.sleep(0.1)
            
            total_duration = time.time() - start_time
            actual_rate = len(weight_data) / total_duration if total_duration > 0 else 0
            self.logger.info(f"Completed weight monitoring: {len(weight_data)} readings over {total_duration:.3f}s ({actual_rate:.1f} Hz)")
            return weight_data

    # ====================================================================
    # 6. WELLPLATE OPERATIONS
    # ====================================================================

    def select_wellplate_grid(self, well_plate_type):
        grid_name = self.WELLPLATES[well_plate_type]["name_in_Locator"]  # Gets "well_plate_new_grid"
        actual_grid = getattr(Locator, grid_name)  # Gets the actual coordinate array
        return actual_grid

    #Dispense into a series of wells (dest_wp_num_array) a specific set of amounts (amount_mL_array)
    def dispense_into_wellplate(self, dest_wp_num_array, amount_mL_array, parameters=None, liquid=None, well_plate_type=None):
        """
        Dispenses specified amounts into a series of wells in a well plate.
        Args:
            dest_wp_num_array (list or range): Array of well indices to dispense into (e.g., [0, 1, 2])
            amount_mL_array (list[float]): Array of amounts (in mL) to dispense into each well (e.g., [0.1, 0.2, 0.3])
            parameters (PipettingParameters, optional): Manual parameter overrides (uses defaults if None)
            liquid (str, optional): Liquid type for automatic parameter optimization
            well_plate_type (str, optional): Type of well plate (defaults to "96 WELL PLATE")
        """
        # Default well plate type if not specified
        if well_plate_type is None:
            well_plate_type = "96 WELL PLATE"

        first_dispense = True
        for i in range(0, len(dest_wp_num_array)):    
            try:
                location = self.select_wellplate_grid(well_plate_type)[dest_wp_num_array[i]]
            except:
                #location = well_plate_new_grid[self.convert_well_into_index(dest_wp_num_array[i])]
                self.pause_after_error("Can't parse wellplate wells in non-indexed form for now")

            amount_mL = amount_mL_array[i] #What amount for this well

            if amount_mL == 0: #Skip empty dispenses
                continue

            # Use intelligent parameter resolution for each volume: defaults → liquid-calibrated → user overrides
            optimized_parameters = self._get_optimized_parameters(amount_mL, liquid, parameters)
            
            # Extract values from optimized parameters
            wait_time = optimized_parameters.dispense_wait_time
            blowout_vol = optimized_parameters.blowout_vol
            
            # Determine dispense speed from optimized parameters
            dispense_speed = optimized_parameters.dispense_speed or self.get_tip_dependent_aspirate_speed()
            self.adjust_pump_speed(0, dispense_speed) #Adjust the pump speed if needed

            height = self.get_height_at_location(location)
            height = self.adjust_height_based_on_pipet_held(height) 

            if first_dispense:
                self.c9.goto_xy_safe(location, vel=self.get_speed('standard_xy'))
                self.c9.move_z(height)
                first_dispense = False
                
            else:
                self.c9.goto_xy_safe(location, vel=self.get_speed('precise_movement'), accel=1, safe_height=height) #Use safe_height here!

            self.logger.info(f"Transferring {amount_mL:.3f} mL into well #{dest_wp_num_array[i]} of {well_plate_type}")

            # Calculate overdispense volume using optimized parameters for this specific amount
            overdispense_vol = (amount_mL + 
                               optimized_parameters.overaspirate_vol + 
                               optimized_parameters.pre_asp_air_vol + 
                               optimized_parameters.post_asp_air_vol)

            #Dispense and then wait
            self.pipet_dispense(overdispense_vol, wait_time=wait_time, blowout_vol=blowout_vol)     

        self.PIPET_FLUID_VOLUME -= np.sum(amount_mL_array)  # <-- Add this line back
        self.save_robot_status()    
        return True

    def dispense_from_vials_into_wellplate(self, well_plate_df, vial_names=None, parameters=None, liquid=None, strategy="serial", 
                                          low_volume_cutoff=0.05, buffer_vol=0.02, well_plate_type="96 WELL PLATE"):
        """
        Dispense from multiple vials into wellplate wells using strategy pattern.
        
        Args:
            well_plate_df (DataFrame): DataFrame where columns are vial names and rows are well volumes
            vial_names (list, optional): DEPRECATED - for backwards compatibility only
            parameters (PipettingParameters, optional): Manual parameter overrides
            liquid (str, optional): Liquid type for automatic parameter optimization
            strategy (str): "serial", "batched", or "auto" (default: auto-select based on parameters)
            low_volume_cutoff (float): Volume threshold for switching between tip types (mL)
            buffer_vol (float): Extra volume to aspirate for accuracy, returned to source vial (mL)
            well_plate_type (str): Type of wellplate for positioning
            
        Example:
            well_plate_df = pd.DataFrame({
                "vial_A": [0.1, 0.2, 0.0, 0.1],  # volumes for wells 0-3
                "vial_B": [0.0, 0.1, 0.3, 0.0], 
                "vial_C": [0.2, 0.0, 0.1, 0.2]
            })
        """
        # Handle backwards compatibility for vial_names
        if vial_names is not None:
            self.logger.warning("vial_names parameter is deprecated. Use DataFrame columns for vial names.")
        
        # Extract vial names from DataFrame columns
        vial_names = well_plate_df.columns.tolist()
        
        self.logger.info(f"Dispensing from vials {vial_names} into wellplate using {strategy} strategy")
        
        # Dispatch to appropriate strategy
        if strategy == "serial":
            return self._dispense_wellplate_serial(well_plate_df, parameters, liquid, well_plate_type)
        elif strategy == "batched":
            return self._dispense_wellplate_batched(well_plate_df, parameters, liquid, low_volume_cutoff, buffer_vol, well_plate_type)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'serial' or 'batched'")

    def _dispense_wellplate_serial(self, well_plate_df, parameters, liquid, well_plate_type):
        """
        Serial strategy: One aspirate -> one dispense per well, full parameter support.
        Slower but supports all PipettingParameters features.
        """
        self.logger.debug("Using serial dispensing strategy")
        
        # Check if there's enough liquid in each vial
        vols_required = well_plate_df.sum(axis=0)
        
        for vial_name, volume_needed in vols_required.items():
            if volume_needed > 1e-6:  # Only check vials that are actually used
                vial_index = self.normalize_vial_index(vial_name)
                vial_volume = self.get_vial_info(vial_index, 'vial_volume')
                if vial_volume < volume_needed - 1e-6:
                    self.pause_after_error(f"Not enough solution in vial {vial_name}: need {volume_needed:.3f} mL, have {vial_volume:.3f} mL")
        
        # Process each vial with proper tip size grouping
        for vial_name in well_plate_df.columns:
            # Collect all volumes for this vial
            vial_volumes = well_plate_df[vial_name]
            
            # Skip vials with no volumes
            if vial_volumes.sum() < 1e-6:
                continue
            
            self.logger.debug(f"Processing vial {vial_name}")
            
            # Sort volumes by size to minimize tip changes (small to large)
            volume_list = [(well_idx, volume) for well_idx, volume in vial_volumes.items() if volume > 1e-6]
            volume_list.sort(key=lambda x: x[1])  # Sort by volume (ascending)
            
            self.logger.debug(f"Processing {len(volume_list)} volumes for vial {vial_name} in sorted order")
            
            # Process volumes in sorted order - this naturally groups similar volumes
            for well_idx, volume in volume_list:
                self.aspirate_from_vial(vial_name, volume, parameters=parameters, liquid=liquid)  # Automatic tip selection
                self.dispense_into_wellplate([well_idx], [volume], parameters=parameters, liquid=liquid,
                                           well_plate_type=well_plate_type)
            
            # Remove tip after processing all volumes for this vial
            if self.HELD_PIPET_TYPE is not None:
                self.remove_pipet()
                self.return_vial_home(vial_name)
              
        self.logger.info("Serial wellplate dispensing completed")
        return True

    def _dispense_wellplate_batched(self, well_plate_df, parameters, liquid, low_volume_cutoff, buffer_vol, well_plate_type):
        """
        Batched strategy: Multiple dispenses per aspirate for speed optimization.
        Faster but limited parameter support (no air volumes, minimal mixing).
        """
        self.logger.debug("Using batched dispensing strategy")
        
        # Extract basic parameters for batched mode (ignore complex parameters)
        # Note: aspirate_speed will be determined after tip is acquired
        base_wait_time = parameters.aspirate_wait_time or 1.0
        
        vial_names = well_plate_df.columns.tolist()
        vial_indices = [self.normalize_vial_index(v) for v in vial_names]
        
        # Check if there's enough liquid in each vial
        vols_required = well_plate_df.sum(axis=0)
        for i, vial_index in enumerate(vial_indices):
            volume_needed = vols_required.iloc[i]
            if volume_needed > 1e-6:
                vial_volume = self.get_vial_info(vial_index, 'vial_volume')
                if vial_volume < volume_needed - 1e-6:
                    self.pause_after_error(f"Not enough solution in vial {vial_index}: need {volume_needed:.3f} mL, have {vial_volume:.3f} mL")
        
        # Split dispenses based on tip type
        well_plate_df_low = well_plate_df.where(well_plate_df < low_volume_cutoff).fillna(0)
        well_plate_df_high = well_plate_df.mask(well_plate_df < low_volume_cutoff, 0)
        
        # Get tip capacities from configuration
        small_tip_volume = self.get_config_parameter('pipet_tips', 'small_tip', 'volume', error_on_missing=True)
        large_tip_volume = self.get_config_parameter('pipet_tips', 'large_tip', 'volume', error_on_missing=True)
        
        tip_configs = [
            (well_plate_df_low, small_tip_volume, "small_tip"),
            (well_plate_df_high, large_tip_volume, "large_tip")
        ]
        
        # Process each tip configuration
        for df_subset, max_volume, tip_type in tip_configs:
            if df_subset.empty or df_subset.sum().sum() < 1e-6:
                continue
                
            # Process each vial
            for vial_idx, vial_name in enumerate(vial_names):
                vial_index = self.normalize_vial_index(vial_name)
                vial_volumes = df_subset.iloc[:, vial_idx]
                total_needed = vial_volumes.sum()
                
                if total_needed < 1e-6:
                    continue
                
                self.logger.debug(f"Batched dispensing from {vial_name}: {total_needed:.3f} mL total")
                
                # Process batches for this vial
                dispensed = 0.0
                well_idx = 0
                
                while well_idx < len(vial_volumes):
                    # Collect next batch that fits within tip capacity
                    batch_wells, batch_volumes, next_idx = self._collect_next_batch(vial_volumes, well_idx, max_volume)
                    
                    if not batch_wells:  # No more batches
                        break
                    
                    # Execute the batch
                    batch_total = self._execute_batch(
                        vial_name, batch_wells, batch_volumes, buffer_vol, max_volume,
                        parameters, liquid, tip_type, well_plate_type
                    )
                    
                    dispensed += batch_total
                    well_idx = next_idx
                    self.logger.debug(f"Batched {len(batch_wells)} wells, total: {batch_total:.3f} mL")
                
                # Remove pipet after finishing this vial (if any dispensing occurred)
                if dispensed > 1e-6:
                    self.remove_pipet()
                    self.return_vial_home(vial_index)
                    self.logger.debug(f"Completed batched dispensing from vial {vial_name} ({tip_type})")
        
        self.logger.info("Batched wellplate dispensing completed")
        return True

    def _collect_next_batch(self, vial_volumes, start_idx, max_volume):
        """
        Collect volumes that fit in one batch.
        
        Args:
            vial_volumes: Pandas Series of volumes for wells
            start_idx: Index to start collecting from
            max_volume: Maximum volume that fits in tip
            
        Returns:
            tuple: (batch_wells, batch_volumes, next_idx)
        """
        batch_wells = []
        batch_volumes = []
        batch_total = 0.0
        current_idx = start_idx
        
        for i in range(start_idx, len(vial_volumes)):
            volume = vial_volumes.iloc[i]
            if volume > 1e-6 and batch_total + volume <= max_volume:
                batch_wells.append(vial_volumes.index[i])  # Well index
                batch_volumes.append(volume)
                batch_total += volume
                current_idx = i + 1
            elif volume > 1e-6:
                break  # Can't fit this volume, end batch
            else:
                current_idx = i + 1  # Skip zero volume, continue
        
        return batch_wells, batch_volumes, current_idx

    def _execute_batch(self, vial_name, batch_wells, batch_volumes, buffer_vol, max_volume, 
                      parameters, liquid, tip_type, well_plate_type):
        """
        Execute one complete batch: aspirate with buffer -> dispense -> return buffer.
        
        Returns:
            float: Total volume dispensed (excluding buffer)
        """
        batch_total = sum(batch_volumes)
        
        # Calculate buffer volume for accuracy
        extra_aspirate_vol = min(buffer_vol, max_volume - batch_total)
        total_aspirate = batch_total + extra_aspirate_vol
        
        # Determine speeds after tip is acquired (get tip first via aspirate)
        # Note: aspirate_from_vial will handle tip acquisition and speed setting
        
        # Aspirate with buffer
        self.aspirate_from_vial(vial_name, total_aspirate, parameters=parameters, liquid=liquid, specified_tip=tip_type)
        
        # Dispense to wells
        self.dispense_into_wellplate(batch_wells, batch_volumes, 
                                   parameters=parameters, liquid=liquid, well_plate_type=well_plate_type)
        
        # Return buffer volume to source vial if any
        if extra_aspirate_vol > 1e-6:
            self.logger.debug(f"Returning buffer volume: {extra_aspirate_vol:.3f} mL to vial {vial_name}")
            self.dispense_into_vial(vial_name, extra_aspirate_vol, parameters=parameters, liquid=liquid)
        
        return batch_total

    # ====================================================================
    # RESERVOIR SYSTEM
    # ====================================================================

    #Prime the line from the reservoir to the vial. In theory this could happen automatically. Probably good to do it if you are using a reservoir. 
    def prime_reservoir_line(self, reservoir_index, overflow_vial, volume=0.5):
        overflow_vial = self.normalize_vial_index(overflow_vial) #Convert to int if needed
        self.logger.info(f"Priming reservoir {reservoir_index} line into vial {overflow_vial}: {volume:.3f} mL")
        self.dispense_into_vial_from_reservoir(reservoir_index,overflow_vial,volume)

    def dispense_into_vial_from_reservoir(self, reservoir_index, vial_index, volume, reservoir_params=None, liquid=None, measure_weight=False, return_home=True, continuous_mass_monitoring=False, save_mass_data=False):
        """
        Dispense liquid from reservoir into a vial.
        
        Args:
            reservoir_index: Index of the reservoir to dispense from
            vial_index: Index/name of destination vial
            volume: Volume to dispense in mL
            reservoir_params (ReservoirParameters, optional): Liquid handling parameters (uses defaults if None)
            liquid (str, optional): Liquid type for calibrated parameter optimization (future feature)
            measure_weight (bool): Whether to measure mass before/after dispensing (default: False)
            continuous_mass_monitoring (bool): Whether to continuously monitor mass during dispensing (default: False)
            save_mass_data (bool): Whether to save continuous mass data to file (default: False)
        """
        vial_index = self.normalize_vial_index(vial_index) #Convert to int if needed
        
        # Use intelligent parameter resolution: defaults → liquid-calibrated → user overrides
        reservoir_params = self._get_optimized_reservoir_parameters(volume, liquid, reservoir_params)
        
        
        self.logger.info(f"Dispensing into vial {vial_index} from reservoir {reservoir_index}: {volume:.3f} mL")
        self.logger.debug(f"Reservoir parameters: aspirate_speed={reservoir_params.aspirate_speed}, dispense_speed={reservoir_params.dispense_speed}, overaspirate={reservoir_params.overaspirate_vol:.3f} mL")
        measured_mass = None
        stability_info = None

        #Step 1: move the vial to the clamp
        if not self.get_vial_info(vial_index,'location')=='clamp':
            # Safety is now handled in move_vial_to_location method
            self.move_vial_to_location(vial_index,'clamp',0)
        if not self.is_vial_pipetable(vial_index):
            self.uncap_clamp_vial()
        self.move_home()

        # Initialize for continuous monitoring if requested
        if continuous_mass_monitoring or measure_weight:
            initial_mass = None
            continuous_mass_data = []
            stability_info = {
                'pre_stable_count': 0,
                'pre_total_count': 0, 
                'post_stable_count': 0,
                'post_total_count': 0,
                'pre_baseline_std': 0.0,
                'post_baseline_std': 0.0
            }
            
            if not self.simulate:
                initial_mass = self.c9.read_steady_scale()
            else:
                initial_mass = 0.0

        #Step 2: move the carousel to reservoir position
        carousel_angle = self.get_config_parameter('pumps', reservoir_index, 'carousel_angle', error_on_missing=False)
        carousel_height = self.get_config_parameter('pumps', reservoir_index, 'carousel_height', error_on_missing=False)
        
        if carousel_angle is not None and carousel_height is not None:
            self.c9.move_carousel(carousel_angle, carousel_height)
        else:
            self.logger.warning(f"No carousel configuration found for pump {reservoir_index}")
            
        #Step 3: aspirate and dispense from the reservoir
        max_volume = self.get_config_parameter('pumps', reservoir_index, 'volume', error_on_missing=False) or 2.5

        # Set up continuous monitoring if requested
        if continuous_mass_monitoring:
            import threading
            monitoring_active = threading.Event()
            dispense_start_time = None
            dispense_end_time = None
            
            def read_scale_continuous():
                """Inner function to continuously read scale weight"""
                start_time = time.time()
                while monitoring_active.is_set():
                    try:
                        if not self.simulate:
                            steady_status, weight = self.c9.read_scale()  # Actually returns (steady_status, weight)
                        else:
                            weight, steady_status = 0.0, True
                            
                        current_time = time.time()
                        time_relative = current_time - start_time
                        
                        # Determine phase based on dispense timing
                        if dispense_start_time is None:
                            phase = 'baseline_pre'
                        elif dispense_end_time is None or current_time < dispense_end_time:
                            phase = 'dispensing'
                        else:
                            phase = 'baseline_post'
                        
                        continuous_mass_data.append({
                            'timestamp': current_time,
                            'time_relative': time_relative,
                            'mass_g': weight,
                            'phase': phase,
                            'steady_status': steady_status
                        })
                        self.logger.debug(f'Mass: {weight:.4f}g at {time_relative:.3f}s ({phase}) steady={steady_status}')
                    except Exception as e:
                        self.logger.warning(f"Scale reading failed: {e}")
                    if not self.simulate:
                        time.sleep(0.2)  # 5 Hz maximum rate
            
            # Start monitoring with pre-dispense baseline
            self.logger.info("Starting continuous mass monitoring for reservoir dispensing")
            start_time = time.time()
            monitoring_active.set()
            monitor_thread = threading.Thread(target=read_scale_continuous, daemon=True)
            monitor_thread.start()
            
            # Wait for pre-dispense baseline
            if not self.simulate:
                time.sleep(2.0)
        
        # Record when dispensing starts
        if continuous_mass_monitoring:
            dispense_start_time = time.time()
        
        # Perform actual dispensing with parameterized control
        total_volume_to_aspirate = volume + reservoir_params.overaspirate_vol
        num_dispenses = math.ceil(total_volume_to_aspirate/max_volume)
        dispense_vol = total_volume_to_aspirate/num_dispenses
        actual_dispense_vol = dispense_vol  # Dispense everything we aspirate each cycle
        
        self.logger.info(f"DISPENSING DEBUG: simulate={self.simulate}, volume={volume:.3f}mL, overaspirate={reservoir_params.overaspirate_vol:.3f}mL")
        self.logger.info(f"DISPENSING DEBUG: total_vol={total_volume_to_aspirate:.3f}mL, num_dispenses={num_dispenses}, dispense_vol={dispense_vol:.3f}mL, actual_dispense={actual_dispense_vol:.3f}mL")
        self.logger.debug(f"Dispensing {actual_dispense_vol:.3f} mL (with {reservoir_params.overaspirate_vol:.3f} mL overaspirate) {num_dispenses} times")
        
        for i in range(0, num_dispenses):        
            self.logger.info(f"DISPENSING LOOP {i+1}/{num_dispenses}: simulate={self.simulate}")
            
            # Aspirate from reservoir with speed control
            self.c9.set_pump_valve(reservoir_index, self.c9.PUMP_VALVE_LEFT)
            if reservoir_params.valve_switch_delay > 0:
                if not self.simulate:
                    time.sleep(reservoir_params.valve_switch_delay)
                
            if reservoir_params.aspirate_speed is not None:
                self.c9.set_pump_speed(reservoir_index, reservoir_params.aspirate_speed)
            self.logger.info(f"ASPIRATING: {dispense_vol:.3f}mL from reservoir {reservoir_index}")
            self.c9.aspirate_ml(reservoir_index, dispense_vol)
            
            if reservoir_params.aspirate_wait_time > 0:
                if not self.simulate:
                    time.sleep(reservoir_params.aspirate_wait_time)
                
            # Dispense into vial with speed control  
            self.c9.set_pump_valve(reservoir_index, self.c9.PUMP_VALVE_RIGHT)
            if reservoir_params.valve_switch_delay > 0:
                if not self.simulate:
                    time.sleep(reservoir_params.valve_switch_delay)
                
            if reservoir_params.dispense_speed is not None:
                self.c9.set_pump_speed(reservoir_index, reservoir_params.dispense_speed)
            self.logger.info(f"DISPENSING: {actual_dispense_vol:.3f}mL into vial {vial_index}")
            self.c9.dispense_ml(reservoir_index, actual_dispense_vol)
            
            if reservoir_params.dispense_wait_time > 0:
                if not self.simulate:
                    time.sleep(reservoir_params.dispense_wait_time)
        
        if not self.simulate:
            time.sleep(1)
        
        # Record when dispensing ends 
        if continuous_mass_monitoring:
            dispense_end_time = time.time()
            dispense_duration = dispense_end_time - dispense_start_time
            self.logger.info(f"Reservoir dispensing took {dispense_duration:.3f}s, collecting post-dispense baseline")
            
            # Collect post-dispense baseline
            if not self.simulate:
                time.sleep(2.0)
            
            # Stop monitoring
            monitoring_active.clear()
            monitor_thread.join(timeout=2.0)
            self.logger.info(f"Stopped continuous mass monitoring. Collected {len(continuous_mass_data)} data points")
        
        vial_volume = self.get_vial_info(vial_index,'vial_volume')
        self.VIAL_DF.at[vial_index,'vial_volume']=(vial_volume+volume)
        self.save_robot_status()

        #Step 4: Return the vial back to home
        self.c9.move_carousel(0,0)

        # Analyze continuous monitoring data or do traditional measurement
        if continuous_mass_monitoring and continuous_mass_data:
            # Process continuous mass data (similar to dispense_into_vial)
            if len(continuous_mass_data) >= 4:
                mass_df = pd.DataFrame(continuous_mass_data)
                
                # Calculate baselines using stability status
                pre_stable_readings = mass_df[(mass_df['phase'] == 'baseline_pre') & 
                                             (mass_df['steady_status'] == True)]
                post_stable_readings = mass_df[(mass_df['phase'] == 'baseline_post') & 
                                              (mass_df['steady_status'] == True)]
                
                # Pre-baseline calculation
                if len(pre_stable_readings) > 0:
                    pre_baseline = pre_stable_readings['mass_g'].mean()
                    pre_method = "STABLE"
                else:
                    pre_baseline = mass_df[mass_df['phase'] == 'baseline_pre']['mass_g'].mean()
                    pre_method = "TIME-BASED"
                
                # Post-baseline calculation  
                if len(post_stable_readings) > 0:
                    post_baseline = post_stable_readings['mass_g'].mean()
                    post_method = "STABLE"
                else:
                    post_baseline = mass_df[mass_df['phase'] == 'baseline_post']['mass_g'].mean()
                    post_method = "TIME-BASED"
                
                measured_mass = post_baseline - pre_baseline
                self.logger.info(f"Reservoir dispensing - Pre: {pre_method}, Post: {post_method}, Mass difference: {measured_mass:.6f}g")
                
                # Update stability info
                pre_baseline_readings = mass_df[mass_df['phase'] == 'baseline_pre']['mass_g']
                post_baseline_readings = mass_df[mass_df['phase'] == 'baseline_post']['mass_g']
                
                pre_count = len(mass_df[mass_df['phase'] == 'baseline_pre'])
                post_count = len(mass_df[mass_df['phase'] == 'baseline_post'])
                pre_stable_count = len(pre_stable_readings)
                post_stable_count = len(post_stable_readings)
                
                stability_info.update({
                    'pre_stable_count': pre_stable_count,
                    'pre_total_count': pre_count,
                    'post_stable_count': post_stable_count, 
                    'post_total_count': post_count,
                    'pre_baseline_std': pre_baseline_readings.std() if len(pre_baseline_readings) > 1 else 0.0,
                    'post_baseline_std': post_baseline_readings.std() if len(post_baseline_readings) > 1 else 0.0
                })
            else:
                self.logger.warning("Insufficient continuous data for baseline calculation")
                measured_mass = 0.0
                
            # Save mass data if requested
            if save_mass_data and continuous_mass_data:
                import os
                import matplotlib.pyplot as plt
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                
                # Use log filename for folder organization instead of date
                if hasattr(self, 'log_filename') and self.log_filename:
                    # Remove .log extension and use as folder name
                    log_folder = self.log_filename.replace('.log', '')
                else:
                    # Fallback to date if log filename not available
                    log_folder = time.strftime("%Y-%m-%d")
                
                filename = f"reservoir_mass_data_{vial_index}_res{reservoir_index}_{timestamp_str}.csv"
                filepath = os.path.join("output", "mass_measurements", log_folder, filename)
                
                os.makedirs(os.path.join("output", "mass_measurements", log_folder), exist_ok=True)
                mass_df = pd.DataFrame(continuous_mass_data)
                mass_df.to_csv(filepath, index=False)
                self.logger.info(f"Saved {len(continuous_mass_data)} reservoir mass measurements to {filepath}")
                
                # Create and save mass vs time plot (same as regular dispense method)
                if len(continuous_mass_data) > 4:
                    plt.figure(figsize=(10, 6))
                    
                    # Plot all data points
                    plt.plot(mass_df['time_relative'], mass_df['mass_g'], 'b.-', alpha=0.7, markersize=2, linewidth=1)
                    
                    # Add phase coloring
                    pre_data = mass_df[mass_df['phase'] == 'baseline_pre']
                    dispense_data = mass_df[mass_df['phase'] == 'dispensing'] 
                    post_data = mass_df[mass_df['phase'] == 'baseline_post']
                    
                    if len(pre_data) > 0:
                        plt.axvspan(pre_data['time_relative'].min(), pre_data['time_relative'].max(), 
                                   alpha=0.2, color='green', label='Pre-baseline', zorder=1)
                    if len(dispense_data) > 0:
                        plt.axvspan(dispense_data['time_relative'].min(), dispense_data['time_relative'].max(), 
                                   alpha=0.2, color='blue', label='Dispensing', zorder=1)
                    if len(post_data) > 0:
                        plt.axvspan(post_data['time_relative'].min(), post_data['time_relative'].max(), 
                                   alpha=0.2, color='red', label='Post-baseline', zorder=1)
                
                    # Add horizontal lines for averaged baselines
                    plt.axhline(y=pre_baseline, color='g', linestyle='--', alpha=0.7, label=f'Pre-avg: {pre_baseline:.3f}g')
                    plt.axhline(y=post_baseline, color='r', linestyle='--', alpha=0.7, label=f'Post-avg: {post_baseline:.3f}g')
                    
                    plt.xlabel('Time (seconds)')
                    plt.ylabel('Mass (g)')
                    plt.title(f'Reservoir Mass vs Time During Dispense\\nVial: {vial_index}, Target: {volume:.3f} mL, Measured: {measured_mass:.4f} g')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Save the plot
                    plot_filename = f"reservoir_mass_plot_{vial_index}_res{reservoir_index}_{timestamp_str}.png"
                    plot_filepath = os.path.join("output", "mass_measurements", log_folder, plot_filename)
                    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                    plt.close()  # Close the figure to free memory
                    self.logger.info(f"Saved reservoir mass vs time plot to {plot_filepath}")
        
        elif measure_weight: # Traditional before/after measurement
            if not self.simulate:
                final_mass = self.c9.read_steady_scale()
                measured_mass = final_mass - initial_mass
            else:
                measured_mass = volume * 1.0  # Simulate perfect transfer (assuming water density)

        if not self.get_vial_info(vial_index,'capped'):
            self.recap_clamp_vial()
        if return_home:
            self.return_vial_home(vial_index)
        
        # Return result in same format as dispense_into_vial
        if continuous_mass_monitoring:
            return measured_mass, stability_info
        else:
            return measured_mass

    # ====================================================================
    # 7. VIAL & CONTAINER MANAGEMENT
    # ====================================================================

    #Check the original status of the vial in order to send it to its home location
    def return_vial_home(self,vial_name, move_speed=None):
        """
        Return the specified vial to its home location.
        Args:
            `vial_name` (str): Name of the vial to return home.
            `move_speed` (float, optional): Speed override for vial movements
        """
        

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed
        home_location = self.get_vial_info(vial_index,'home_location')
        home_location_index = self.get_vial_info(vial_index,'home_location_index')
        self.logger.info(f"Returning vial {self.get_vial_info(vial_index, 'vial_name')} to home location: {home_location} index {home_location_index}")
        
        vial_location = self.get_vial_info(vial_index,'location')
        if vial_location == 'clamp' and self.GRIPPER_STATUS == "Cap":
            self.recap_clamp_vial(move_speed=move_speed)
        self.move_vial_to_location(vial_index,home_location,home_location_index, move_speed=move_speed)
        self.save_robot_status()

    #Drop off a vial at a location that you already have
    def drop_off_vial(self, vial_name, location, location_index, move_speed=None):

        self.logger.debug(f"Dropping off vial {vial_name} at location {location} with index {location_index}")

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        destination = self.get_location(False,location,location_index)
        occupying_vial = self.get_vial_in_location(location,location_index)
        # Allow drop-off if the location is empty or occupied by the same vial
        destination_empty = (occupying_vial is None) or (occupying_vial == vial_index)

        self.check_for_errors([[destination_empty, True, "Cannot move vial to destination, destination full"]],True)

        self.c9.goto_safe(destination, vel=move_speed if move_speed is not None else self.get_speed('default_robot')) #move vial to destination
        self.c9.open_gripper() #release vial
        
        self.VIAL_DF.at[vial_index, 'location']=location
        self.VIAL_DF.at[vial_index, 'location_index']=location_index
        self.GRIPPER_STATUS = None #We no longer have the vial
        self.GRIPPER_VIAL_INDEX = None
        self.save_robot_status() #Update in memory

    def grab_vial(self,vial_name, move_speed=None):
        
        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed
        
        self.logger.debug(f"Grabbing vial {vial_name} with index {vial_index}")
        initial_location = self.get_vial_location(vial_index, False)
        loc = self.get_vial_info(vial_index,'location')

        if loc == 'clamp' and self.GRIPPER_STATUS == "Cap":
            self.recap_clamp_vial(move_speed=move_speed)

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS is None, True, "Cannot move vial to destination, gripper full"])
        #error_check_list.append([self.HELD_PIPET_INDEX is None, True, "Cannot move vial to destination, robot holding pipet"])
        error_check_list.append([self.is_vial_movable(vial_index), True, "Can't move vial, vial is uncapped."])  

        self.check_for_errors(error_check_list,True) #Check for an error, and pause if there's an issue

        #self.open_gripper()
        self.goto_location_if_not_there(initial_location, move_speed=move_speed) #move to vial
        self.c9.close_gripper() #grip vial
        
        self.GRIPPER_STATUS = "Vial" #Update the status of the robot
        self.GRIPPER_VIAL_INDEX = vial_index
        self.save_robot_status() #Save the status of the robot

    #Send the vial to a specified location
    def move_vial_to_location(self,vial_name:str,location:str,location_index:int, move_speed=None):
        """
        Moves vial to specified location

        Args:
            vial_name (str): Name of the vial to move
            location (str): Description of the location to move to (e.g., 'clamp', 'photoreactor_array', 'main_8mL_rack', 'heater')
            location_index (int): Index of the location to move to (e.g., 0 for the first position, 1 for the second position, etc.)
         """
        
        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        # Safety check: If holding a pipet, ensure the movement path is safe
        if self.HELD_PIPET_TYPE is not None:
            if not self.is_vial_movement_safe_with_pipet(vial_index, location, location_index):
                self.pause_after_error(f"Cannot move vial {vial_name} while holding pipet - only main_8mL_rack positions 0-5 and clamp position 0 are safe")

        vial_location = self.get_vial_info(vial_index,'location')
        vial_location_index = self.get_vial_info(vial_index,'location_index')

        if vial_location == location and vial_location_index == location_index:
            self.logger.info(f"Vial {vial_name} already at {location} index {location_index}, no move needed")
            return

        self.logger.info(f"Moving vial {self.get_vial_info(vial_index, 'vial_name')} to {location}: {location_index}")
        self.grab_vial(vial_index, move_speed=move_speed) #Grab the vial
        self.drop_off_vial(vial_index,location,location_index, move_speed=move_speed) #Drop off the vial

    def get_vial_in_location(self, location_name, location_index):
        # Filter rows where both conditions match
        mask = (self.VIAL_DF['location'] == location_name) & (self.VIAL_DF['location_index'] == location_index)
        
        # Get the matching values
        matching_vials = self.VIAL_DF.loc[mask, 'vial_index'].values

        # Return the first match or None if no match is found
        return int(matching_vials[0]) if len(matching_vials) > 0 else None

    #Uncap the vial in the clamp
    def uncap_clamp_vial(self, revs=3, move_speed=None):
        self.logger.debug("Removing cap from clamped vial")

        clamp_vial_index = self.get_vial_in_location('clamp',0)

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS is None, True, "Cannot uncap, gripper full"])
        error_check_list.append([clamp_vial_index is None, False, "Cannot uncap, no vial in clamp"])
        error_check_list.append([self.is_vial_movable(clamp_vial_index), True, "Can't uncap, vial is uncapped already"])

        self.check_for_errors(error_check_list,True) #Check for an error and pause if there is one
        
        
        self.c9.close_clamp() #clamp vial
        self.goto_location_if_not_there(vial_clamp, move_speed=move_speed) #Maybe check if it is already there or not  
        if not self.simulate:
            time.sleep(0.5) 
        self.c9.close_gripper()
        self.c9.uncap(revs=revs)
        self.GRIPPER_STATUS = "Cap"
        self.c9.open_clamp()

        self.VIAL_DF.at[clamp_vial_index, 'capped']=False
        self.GRIPPER_VIAL_INDEX = clamp_vial_index
        self.save_robot_status()

    #Recap the vial in the clamp
    def recap_clamp_vial(self, revs=2.0, torque_thresh = 600, move_speed=None):
        self.logger.debug("Recapping clamped vial")
        
        clamp_vial_index = self.get_vial_in_location('clamp',0)

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS, "Cap", "Cannot recap, no cap in gripper"])
        error_check_list.append([clamp_vial_index is None, False, "Cannot recap, no vial in clamp"])
        error_check_list.append([self.is_vial_movable(clamp_vial_index), False, "Can't recap, vial is capped already"])
        
        self.check_for_errors(error_check_list,True) #Let's pause if there is an error

        
        self.c9.close_clamp() #Make sure vial is clamped
        self.goto_location_if_not_there(vial_clamp_cap, move_speed=move_speed)
        if not self.simulate:
            time.sleep(0.5)
        self.c9.cap(revs=revs, torque_thresh = torque_thresh) #Cap the vial #Cap the vial
        self.c9.open_gripper() #Open the gripper to release the cap
        self.GRIPPER_STATUS = None
        self.c9.open_clamp()

        self.VIAL_DF.at[clamp_vial_index, 'capped']=True #Update the vial status
        self.GRIPPER_VIAL_INDEX = None
        self.save_robot_status()

    #Checks first that you aren't already there... This mostly applies for cap/decap
    def goto_location_if_not_there(self, location, move_speed=None):
        difference_threshold = 550
        if self.get_location_distance(location, self.c9.get_robot_positions()) > difference_threshold:
            self.c9.goto_safe(location, vel=move_speed if move_speed is not None else self.get_speed('fast_approach'))

    #Measurement for how far two points are
    def get_location_distance(self, loc_1, loc_2):
        if self.simulate:
            return 0
        difference = np.sum(np.absolute(np.array(loc_2)[1:4] - np.array(loc_1)[1:4]))
        return difference
    
    #Removes the target vial, vortexes it, then puts it back
    def vortex_vial(self, vial_name, vortex_time, vortex_speed=70):
        """
        Moves target vial up, vortexes for a specified time and speed and returns it to its original location.
        Args:
            `vial_name` (str): Name of the vial to vortex
            `vortex_time` (float): Time (in seconds) to vortex
            `vortex_speed` (float): Speed of vortexing (default is 70)
        """
        
        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        self.logger.info(f"Vortexing Vial: {self.get_vial_info(vial_index,'vial_name')}")
        
        #Check to see if the vial is capped
        if self.GRIPPER_VIAL_INDEX == vial_index  and self.GRIPPER_STATUS == "Cap":
            self.recap_clamp_vial()
        self.grab_vial(vial_index)
    
        #Vortex
        self.c9.move_z(self.get_safe_height()) #Move to a higher height
        gripper_axis = self.get_config_parameter('robot_hardware', 'robot_axes', 'gripper_axis', error_on_missing=False) or 0
        self.c9.move_axis(gripper_axis, 1000*vortex_time*vortex_speed, vel=vortex_speed,accel=10000)
        self.c9.reduce_axis_position(axis=gripper_axis)

        location = self.get_vial_info(vial_index,'location')
        location_index = self.get_vial_info(vial_index,'location_index')

        #Move the vial back to its original location
        self.drop_off_vial(vial_index, location, location_index)
        
    #This is just to formalize the process of error checking so its more extensible
    #This may end up deprecated
    #Error check list is as follows error_check[0] is the value, error_check[1] is the target value, error_check[2] is the error message
    def check_for_errors(self, error_check_list, stop_after_error=False):
        error_occured = False
        for error_check in error_check_list:
            if error_check[0] != error_check[1]:
                error_occured = True
                self.logger.warning(error_check[2])
                if stop_after_error:
                    self.pause_after_error(error_check[2])
        return error_occured

    # ====================================================================
    # 8. VALIDATION & SAFETY METHODS
    # ====================================================================

   #Check to see if we can move the vial         
    def is_vial_movable(self, vial_name):

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if neede

        movable = False
        movable = self.get_vial_info(vial_index,'capped') == True and self.get_vial_info(vial_index,'vial_type') == "8_mL"
        return movable
    
    #Check to see if the pipet can have liquids added/removed
    def is_vial_pipetable(self, vial_name):

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        pipetable = False
        pipetable = self.get_vial_info(vial_index,'capped') == False or self.get_vial_info(vial_index,'cap_type') == "open"
        return pipetable

    #Check if a location is safe for vial movement while holding a pipet tip
    def is_location_safe_for_vial_movement_with_pipet(self, location, location_index):
        """
        Check if a specific location is safe for vial movement while holding a pipet tip.
        Due to height constraints, only certain positions are safe to access with a pipet.
        
        Args:
            location: Location name (e.g., 'main_8mL_rack', 'clamp')
            location_index: Location index number
            
        Returns:
            bool: True if location is safe with pipet, False otherwise
        """
        # Safe positions when holding a pipet:
        # - main_8mL_rack positions 0-5
        # - clamp position 0
        if location == 'main_8mL_rack' and location_index is not None:
            return location_index <= 5
        elif location == 'clamp' and location_index is not None:
            return location_index == 0
        
        # All other locations are unsafe when holding a pipet
        return False

    #Check if a vial movement path is safe while holding a pipet tip
    def is_vial_movement_safe_with_pipet(self, vial_name, dest_location, dest_location_index):
        """
        Check if moving a vial from its current position to a destination is safe while holding a pipet.
        Checks both the starting position and destination position for safety.
        
        Args:
            vial_name: Vial name or index to check
            dest_location: Destination location name (e.g., 'clamp', 'main_8mL_rack')
            dest_location_index: Destination location index
            
        Returns:
            bool: True if both start and end positions are safe with pipet, False otherwise
        """
        vial_index = self.normalize_vial_index(vial_name)
        
        # Get current vial location information
        current_location = self.get_vial_info(vial_index, 'location')
        current_location_index = self.get_vial_info(vial_index, 'location_index')
        
        # Check if current position is safe
        current_position_safe = self.is_location_safe_for_vial_movement_with_pipet(current_location, current_location_index)
        
        # Check if destination position is safe
        dest_position_safe = self.is_location_safe_for_vial_movement_with_pipet(dest_location, dest_location_index)
        
        return current_position_safe and dest_position_safe
 
    def move_rel_xyz(self, x_distance=0, y_distance=0, z_distance=0, vel=None):
        if vel is None:
            vel = self.get_speed('standard_xy')
        self.logger.debug(f"Moving robot relative to current position by x: {x_distance}, y: {y_distance}, z: {z_distance} mm, vel: {vel}")
        
        # Get axis numbers from configuration with fallbacks
        gripper_axis = self.get_config_parameter('robot_hardware', 'robot_axes', 'gripper_axis', error_on_missing=False) or 0
        elbow_axis = self.get_config_parameter('robot_hardware', 'robot_axes', 'elbow_axis', error_on_missing=False) or 1
        shoulder_axis = self.get_config_parameter('robot_hardware', 'robot_axes', 'shoulder_axis', error_on_missing=False) or 2
        z_axis = self.get_config_parameter('robot_hardware', 'robot_axes', 'z_axis', error_on_missing=False) or 3
        
        current_loc_mm = self.c9.n9_fk(self.c9.get_axis_position(gripper_axis), self.c9.get_axis_position(elbow_axis), self.c9.get_axis_position(shoulder_axis))
        target_x =  current_loc_mm[0] + x_distance
        target_y =  current_loc_mm[1] + y_distance
        target_z =  self.c9.counts_to_mm(z_axis, self.c9.get_axis_position(z_axis)) + z_distance

        self.c9.move_xyz(target_x, target_y, target_z, vel=vel)

    #Move the robot to the home position    
    def move_home(self):
        self.logger.info("Moving robot to home position")
        self.c9.goto_safe(home)
        self.c9.move_carousel(0,0)

    # ====================================================================
    # 9. VIAL INFO & LOCATION UTILITIES
    # ====================================================================

    def _vials_available(self):
        """Check if vial operations are available (i.e., if vial file was loaded)"""
        return not self.VIAL_DF.empty

    #Get some piece of information about a vial
    #vial_index,vial_name,location,location_index,vial_volume,capped,cap_type,vial_type
    def get_vial_info(self,vial_name,column_name):
        
        # Check if vial data is available
        if not self._vials_available():
            self.logger.warning(f"Cannot get vial info for '{vial_name}' - no vial data loaded")
            return None

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        values = self.VIAL_DF.loc[self.VIAL_DF['vial_index'] == vial_index, column_name].values
        if len(values) > 0:
            return values[0]  # Return the first match
        else:
            return None  # Handle case where no match is found    

    def get_vial_index_from_name(self,vial_name):
        # Check if vial data is available
        if not self._vials_available():
            self.logger.warning(f"Cannot get vial index for '{vial_name}' - no vial data loaded")
            return None
            
        values = self.VIAL_DF.loc[self.VIAL_DF['vial_name'] == vial_name, 'vial_index'].values
        if len(values) > 0:
            return values[0]  # Return the first match
        else:
            return None  # Handle case where no match is found  

    #Physical method that get's hard-coded minimum heights for pipetting    
    def get_min_pipetting_height(self,vial_name):

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed
        #The height at which the pipet touches the ground for the 1 mL pipet
        min_height = None   
        location_name = self.get_vial_info(vial_index,'location')
        # Get minimum pipetting height from YAML configuration
        min_height = self.get_config_parameter('vial_positions', location_name, 'min_pipetting_height', error_on_missing=True)
        if min_height is None:
            self.pause_after_error(f"No minimum pipetting height defined for location: {location_name}")

        return min_height

    #Get the position of a vial right now
    def get_vial_location(self,vial_name,use_pipet):
        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed
        location_name = self.get_vial_info(vial_index,'location')
        location_index = self.get_vial_info(vial_index,'location_index')
        return self.get_location(use_pipet,location_name,location_index)

    #Translate the location names and indexes to hard-coded locations for pipets or gripping vials
    def get_location(self, use_pipet, location_name, location_index):
        # Get the appropriate position source from YAML configuration
        param_key = 'pipetting_positions_in_Locator' if use_pipet else 'vial_positions_in_Locator'
        position_source = self.get_config_parameter('vial_positions', location_name, param_key, error_on_missing=True)
        
        if position_source is None:
            self.pause_after_error(f"No {param_key} defined for location: {location_name}")
            return None
        
        try:
            location_data = getattr(Locator, position_source)
            
            # Check if it's a single position (list of 4 coordinates) or array of positions
            if len(location_data) == 4 and all(isinstance(coord, (int, float)) for coord in location_data):
                # Single position - only index 0 is valid
                if location_index != 0:
                    self.pause_after_error(f"Location {location_name} only supports index 0 (single position)")
                    return None
                return location_data
            else:
                # Array of positions - ensure index is an int (avoid numpy.float64 issues)
                try:
                    if not isinstance(location_index, (int,)):
                        # Convert safely (e.g., numpy.float64 -> int)
                        location_index = int(location_index)
                except (TypeError, ValueError):
                    self.pause_after_error(f"Non-integer location index for {location_name}: {location_index}")
                    return None
                return location_data[location_index]
                
        except (AttributeError, IndexError) as e:
            self.pause_after_error(f"Invalid location {location_name}[{location_index}]: {e}")
            return None
    

from robot_state.Locator import * #Let's try to eliminate this in the future
import robot_state.Locator as Locator
import numpy as np
import time
import math
import pandas as pd
import slack_agent
import yaml
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from pipetting_data.pipetting_parameters import PipettingParameters
import matplotlib.patches as patches

class North_Base:
    """Base class for North robot components with shared functionality"""
    
    def pause_after_error(self, err_message, send_slack=True):
        """Pause execution after an error with logging and optional Slack notification"""
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
    
    def _load_wellplate_config(self, file_path="../utoronto_demo/robot_state/wellplates.yaml"):
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
        self.simulate = simulate

        #Load yaml data
        self.logger.debug("Loading track status from file: %s", "../utoronto_demo/robot_state/track_status.yaml")
        self.TRACK_STATUS_FILE = "../utoronto_demo/robot_state/track_status.yaml"
        self.get_track_status() #set NUM_SOURCE, NUM_WASTE, CURRENT_WP_TYPE and NR_OCCUPIED from yaml file
        
        # Load track positions configuration
        self.TRACK_POSITIONS_FILE = "../utoronto_demo/robot_state/track_positions.yaml"
        self._load_track_positions()
        
        # Load robot hardware configuration for axis mappings
        self.ROBOT_HARDWARE_FILE = "../utoronto_demo/robot_state/robot_hardware.yaml"
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
            "active_wellplate_position": self.ACTIVE_WELLPLATE_POSITION
        }

        if not self.simulate: #not simulating
            # Writing to a file
            with open(self.TRACK_STATUS_FILE, "w") as file:
                yaml.dump(track_status, file, default_flow_style=False)

    def check_input_file(self, pause_after_check=True, visualize=True):
        self.logger.info(f"--Wellplate status-- \n Wellplate type: {self.CURRENT_WP_TYPE} \n Number in source: {self.NUM_SOURCE} \n Number in waste: {self.NUM_WASTE} \n Active wellplate position: {self.ACTIVE_WELLPLATE_POSITION}")

        if visualize:
            self.logger.info("Visualizing wellplate status...")
            fig, ax = plt.subplots(figsize=(10, 6))

            plate_width = 2.5
            plate_height = 0.4
            spacing = 0.1

            for i in range(self.NUM_SOURCE):
                rect = plt.Rectangle((1, i * (plate_height + spacing)), plate_width, plate_height,
                                    edgecolor='black', facecolor='lightblue')
                ax.add_patch(rect)
                ax.text(1 + plate_width / 2, i * (plate_height + spacing) + plate_height / 2,
                        self.CURRENT_WP_TYPE, ha='center', va='center', fontsize=8)

            for i in range(self.NUM_WASTE):
                rect = plt.Rectangle((5, i * (plate_height + spacing)), plate_width, plate_height,
                                    edgecolor='black', facecolor='lightcoral')
                ax.add_patch(rect)
                ax.text(5 + plate_width / 2, i * (plate_height + spacing) + plate_height / 2,
                        self.CURRENT_WP_TYPE, ha='center', va='center', fontsize=8)

            if self.ACTIVE_WELLPLATE_POSITION == 'pipetting_area':
                rect = plt.Rectangle((9, 0), plate_width, plate_height,
                                    edgecolor='black', facecolor='khaki')
                ax.add_patch(rect)
                ax.text(9 + plate_width / 2, plate_height / 2,
                        "Occupied", ha='center', va='center', fontsize=8)
                ax.text(9 + plate_width / 2, plate_height + 0.2, "NR Pipette Area",
                        ha='center', va='bottom', fontsize=10, weight='bold')

            ax.text(1 + plate_width / 2, self.NUM_SOURCE * (plate_height + spacing) + 0.2, "Source Stack",
                    ha='center', va='bottom', fontsize=10, weight='bold')
            ax.text(5 + plate_width / 2, self.NUM_WASTE * (plate_height + spacing) + 0.2, "Waste Stack",
                    ha='center', va='bottom', fontsize=10, weight='bold')

            ax.set_xlim(0, 12)
            ax.set_ylim(0, max(self.NUM_SOURCE, self.NUM_WASTE) * (plate_height + spacing) + 1)
            ax.axis('off')
            ax.set_title("-- Please Confirm Wellplate Status --", fontsize=14, weight='bold')
            plt.tight_layout()
            plt.show()

        if pause_after_check and not self.simulate:
            input("Only hit enter if the status of the well plates is correct, otherwise hit ctrl-c")

    def reset_after_initialization(self):
        """Reset robot to known state after initialization"""
        if self.ACTIVE_WELLPLATE_POSITION == 'gripper':
            # Discard the wellplate currently in gripper
            self.discard_wellplate(move_home_afterwards=True)
     
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

    def __init__(self,c9,simulate = False, logger=None):
        self.simulate = simulate
        self.c9 = c9
        self.logger = logger

        self.logger.debug("Initializing temperature controller...")

        if not self.simulate:
            from north import NorthC9
            self.t8 = NorthC9('B', network=c9.network)
            self.c8 = NorthC9('D', network=c9.network)
        else:
            self.t8 = MagicMock()
            self.c8 = MagicMock()
    
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
    
    ROBOT_STATUS_FILE = "../utoronto_demo/robot_state/robot_status.yaml" #Store the state of the robot. Update this after every method that alters the state. 
    VIAL_POSITIONS_FILE = "../utoronto_demo/robot_state/vial_positions.yaml" #File that contains the vial positions.
    WELLPLATE_POSITIONS_FILE = "../utoronto_demo/robot_state/wellplates.yaml" #File that contains the wellplate positions.
    PIPET_TIP_DEFINITTIONS_FILE = "../utoronto_demo/robot_state/pipet_tips.yaml" #File that contains the pipet tip definitions.
    PIPET_RACKS_FILE = "../utoronto_demo/robot_state/pipet_racks.yaml" #File that contains the pipet rack configurations.
    PUMP_CONFIG_FILE = "../utoronto_demo/robot_state/syringe_pumps.yaml" #File that contains the pump configurations.
    ROBOT_HARDWARE_FILE = "../utoronto_demo/robot_state/robot_hardware.yaml" #File that contains robot hardware axis and pneumatic mappings.
    
    # ====================================================================
    # 1. CLASS SETUP & INITIALIZATION
    # ====================================================================
    
    #Initialize the status of the robot. 
    def __init__(self,c9,vial_file=None,simulate=False, logger=None):

        self.c9 = c9
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
            'WELLPLATES': ("../utoronto_demo/robot_state/wellplates.yaml", "wellplate properties configuration"),
        }
        
        for attr_name, (file_path, description) in config_files.items():
            # Use convert_none=True for all files, wellplates are optional
            setattr(self, attr_name, self._load_yaml_file(file_path, description, required=True, convert_none=True))
        
        # Initialize pipet usage tracking from loaded rack configuration
        self.PIPETS_USED = {rack_name: 0 for rack_name in self.PIPET_RACKS.keys()} if self.PIPET_RACKS else {}

    def _load_state_files(self):
        """Load all robot state YAML files (dynamic state that changes during operation)"""
        self.logger.debug("Loading robot state files...")
        
        # Load vial DataFrame (CSV file)
        try:
            self.VIAL_DF = pd.read_csv(self.VIAL_FILE, sep=",")
            self.VIAL_DF.index = self.VIAL_DF['vial_index'].values
            self.logger.debug(f"Loaded vial data from {self.VIAL_FILE}")
        except Exception as e:
            self.logger.error(f"Issue reading vial status file {self.VIAL_FILE}: {e}")
            self.pause_after_error("Issue reading vial status", False)
        
        # State files (robot status is required, wellplate positions are optional)
        state_files = {
            'robot_status': (self.ROBOT_STATUS_FILE, "robot status", True, True),
            'wellplate_positions': (self.WELLPLATE_POSITIONS_FILE, "wellplate positions", False, True),
        }
        
        loaded_data = {}
        for key, (file_path, description, required, convert_none) in state_files.items():
            loaded_data[key] = self._load_yaml_file(file_path, description, required, convert_none)
        
        # Handle robot status data
        if loaded_data['robot_status']:
            robot_status = loaded_data['robot_status']
            try:
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
                    
                self.PIPET_FLUID_VOLUME = robot_status['pipet_fluid_volume']
                self.PIPET_FLUID_VIAL_INDEX = robot_status['pipet_fluid_vial_index']
            except KeyError as e:
                self.pause_after_error(f"Missing required field in robot status: {e}")
            except Exception as e:
                self.pause_after_error(f"Error processing robot status: {e}")
        
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
        plt.show()

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
        vial_status = pd.read_csv(self.VIAL_FILE, sep=",")
        self.logger.info(vial_status)

        if visualize:
            self.visualize_racks(vial_status)

        if pause_after_check and not self.simulate:
            input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

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
        self.logger.debug("Physical initialization of North Robot...")
        self.c9.default_vel = self.get_speed('default_robot')  # Set the default speed of the robot
        self.c9.open_clamp()
        
        # Home all components first before any pump operations
        self.logger.info("Homing robot components before pump initialization...")
        try:
            self.home_robot_components()
        except Exception as e:
            self.logger.error(f"Failed to home robot components during initialization: {e}")
            raise
        
        # Now load pumps after homing
        self.load_pumps()
        
        # Attempt physical cleanup with retry logic for remaining errors
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self._perform_physical_cleanup()
                break  # Success, exit retry loop
                
            except Exception as e:
                # For any remaining errors, just log and retry
                retry_count += 1
                self.logger.warning(f"Physical cleanup failed (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count >= max_retries:
                    self.logger.error(f"Failed to complete physical cleanup after {max_retries} attempts: {e}")
                    raise
        
        # Final setup
        self.c9.open_gripper()
 
    def _perform_physical_cleanup(self):
        """
        Internal method to perform the actual physical cleanup tasks.
        Separated for cleaner retry logic.
        """
        # Handle leftover liquid in pipet tip
        if self.PIPET_FLUID_VIAL_INDEX is not None and self.PIPET_FLUID_VOLUME > 0:
            self.logger.warning("The robot reports having liquid in its tip... Returning that liquid...")
            vial_index = self.PIPET_FLUID_VIAL_INDEX 
            volume = self.PIPET_FLUID_VOLUME
            self.dispense_into_vial(vial_index, volume)
        
        # Remove pipet tip if present
        if self.HELD_PIPET_TYPE is not None:
            self.logger.warning("The robot reports having a tip, removing the tip")
            self.remove_pipet()
        
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
            "pipet_fluid_volume": float(self.PIPET_FLUID_VOLUME)
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
                slack_agent.send_slack_message(f"Aspirate was exceeded for {amount} mL. Aspirating to maximum volume of 1 mL.")
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

        if not self.simulate:
            time.sleep(wait_time)

        if blowout_vol > 0:
            blow_speed = self.get_tip_dependent_aspirate_speed()
            self.logger.debug(f"Blowing out {blowout_vol:.3f} mL at speed {blow_speed}")
            self.adjust_pump_speed(0, blow_speed)
            self.c9.set_pump_valve(0, self.c9.PUMP_VALVE_LEFT)
            self.c9.aspirate_ml(0, blowout_vol)
            self.c9.set_pump_valve(0, self.c9.PUMP_VALVE_RIGHT)
            self.c9.dispense_ml(0, blowout_vol)

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

    def _ensure_vial_accessible_for_pipetting(self, vial_name, use_safe_location=False):
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
                if clamp_vial_index is not None:
                    self.logger.debug(f"Clamp occupied by vial {clamp_vial_index}, returning it home first")
                    self.return_vial_home(clamp_vial_index)
                
                # Move vial to clamp and uncap
                self.move_vial_to_location(vial_num, location='clamp', location_index=0)
                self.uncap_clamp_vial()
                
            else:
                self.pause_after_error(f"Vial {vial_name} cannot be moved to a safe pipetting location")
                return False
                
        return True

    # ====================================================================
    # 5. LIQUID HANDLING OPERATIONS
    # ====================================================================

    #Aspirate from a vial using the pipet tool
    def aspirate_from_vial(self, source_vial_name, amount_mL, parameters=None, 
                          move_to_aspirate=True, track_height=True, move_up=True, specified_tip=None, use_safe_location=False):
        """
        Aspirate amount_ml from a source vial.
        
        Args:
            source_vial_name (str): Name of the source vial to aspirate from
            amount_mL (float): Amount to aspirate in mL
            parameters (PipettingParameters, optional): Liquid handling parameters (uses defaults if None)
            move_to_aspirate (bool): Whether to move to aspiration location (default: True)
            track_height (bool): Whether to track liquid height during aspiration (default: True)
            move_up (bool): Whether to retract after aspiration (default: True)
            specified_tip (str, optional): Force specific tip type ("small_tip" or "large_tip")
        """
        if parameters is None:
            parameters = PipettingParameters()  # Use all defaults
        
        # Extract liquid handling values from parameters
        asp_disp_cycles = parameters.asp_disp_cycles
        aspirate_wait_time = parameters.aspirate_wait_time
        pre_asp_air_vol = parameters.pre_asp_air_vol
        post_asp_air_vol = parameters.post_asp_air_vol
        
        source_vial_num = self.normalize_vial_index(source_vial_name) #Convert to int if needed     

        # Ensure vial is accessible for pipetting
        if not self._ensure_vial_accessible_for_pipetting(source_vial_name, use_safe_location):
            return

        #Check if has pipet, get one if needed based on volume being aspirated (or if tip is specified)
        required_tip_type = self.select_pipet_tip(amount_mL, specified_tip)
        
        if self.HELD_PIPET_TYPE is None:
            # No pipet held, get the required one
            self.get_pipet(required_tip_type)
        elif self.HELD_PIPET_TYPE != required_tip_type:
            # Wrong pipet type held, need to switch
            self.logger.info(f"Switching from {self.HELD_PIPET_TYPE} to {required_tip_type}")
            self.remove_pipet()
            self.get_pipet(required_tip_type)
        
        #Check for an issue with the pipet and the specified amount, pause and send slack message if so
        self.check_if_aspiration_volume_unacceptable(amount_mL) 
        
        # Now that we have a pipet tip, we can safely determine speeds
        aspirate_speed = parameters.aspirate_speed or self.get_tip_dependent_aspirate_speed()
        retract_speed = parameters.retract_speed or self.get_speed('retract')

        #Get current volume and vial location
        location = self.get_vial_location(source_vial_num,True)
        source_vial_volume = self.get_vial_info(source_vial_num,'vial_volume')

        #Reject aspiration if the volume is not high enough
        if source_vial_volume < amount_mL:
            self.pause_after_error("Cannot aspirate more volume than in vial")

        self.logger.info(f"Pipetting from vial {self.get_vial_info(source_vial_num, 'vial_name')}, amount: {round(amount_mL, 3)} mL")

        #Adjust the height based on the volume, then pipet type
        if move_to_aspirate:
            asp_height = self.get_aspirate_height(source_vial_name, amount_mL, track_height)
            asp_height = self.adjust_height_based_on_pipet_held(asp_height)

            #The dispense height is right above the location
            disp_height = self.get_height_at_location(location)
            disp_height = self.adjust_height_based_on_pipet_held(disp_height)
        else:
            asp_height = disp_height = self.get_current_height() #IF we aren't moving, everything stays the same

        self.adjust_pump_speed(0,aspirate_speed) #Adjust the pump speed if needed

        #Move to the correct location in xy
        if move_to_aspirate:
            self.c9.goto_xy_safe(location, vel=self.get_speed('standard_xy'))
        
        #Step 1: Move to above the site and aspirate air if needed
        if pre_asp_air_vol > 0:
            self.c9.move_z(disp_height)
            self.pipet_aspirate(pre_asp_air_vol, wait_time=0)
        
        #Step 2: Move to inside the site and aspirate liquid
        self.c9.move_z(asp_height)
        for i in range (0, asp_disp_cycles):
            self.pipet_aspirate(amount_mL, wait_time=0)
            self.pipet_dispense(amount_mL, wait_time=0)
        self.pipet_aspirate(amount_mL, wait_time=aspirate_wait_time) #Main aspiration of liquid plus wait
        
        #Step 3: Retract and aspirate air if needed
        if move_up:
            self.c9.move_z(disp_height, vel=retract_speed) #Retract with a specific speed
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
    def dispense_from_vial_into_vial(self, source_vial_name, dest_vial_name, volume, parameters=None, specified_tip=None, remove_tip=True):
        """
        Transfer liquid from source vial to destination vial.

        Args:
            source_vial_name (str): Name of the source vial to aspirate from
            dest_vial_name (str): Name of the destination vial to dispense into
            volume (float): Volume (in mL) to transfer
            parameters (PipettingParameters, optional): Standardized parameters
            specified_tip (str, optional): Force specific tip type ("small_tip" or "large_tip")
        """
        if parameters is None:
            parameters = PipettingParameters()

        self.logger.info(f"Dispensing {volume:.3f} mL from {source_vial_name} to {dest_vial_name}")

        source_vial_index = self.normalize_vial_index(source_vial_name)
        dest_vial_index = self.normalize_vial_index(dest_vial_name)

        if volume <= 0:
            self.logger.warning("Cannot dispense <=0 mL")
            return

        # Handle large volumes by splitting into multiple transfers
        max_system_volume = max((tip_config.get('volume', 0) for tip_config in self.PIPET_TIPS.values()), default=1.0)
        repeats = 1
        if volume > max_system_volume:
            repeats = math.ceil(volume / max_system_volume)
            volume = volume / repeats
            self.logger.info(f"Volume too high for single transfer, splitting into {repeats} transfers of {round(volume,3)} mL each")

        total_mass = 0
        for i in range(repeats):
            last_run = (i == repeats - 1)

            # Aspirate from source
            self.aspirate_from_vial(source_vial_index, round(volume, 3), parameters=parameters, specified_tip=specified_tip)

            # Dispense into destination
            mass_increment = self.dispense_into_vial(dest_vial_index, volume, parameters=parameters)
            total_mass += mass_increment if mass_increment is not None else 0

            # Return vials and remove pipet on last run
            if last_run:
                if remove_tip:
                    self.remove_pipet()
                # Return whatever vial is currently in the clamp (works for all cases)
                clamp_vial_index = self.get_vial_in_location('clamp', 0)
                if clamp_vial_index is not None:
                    self.recap_clamp_vial()
                    self.return_vial_home(clamp_vial_index)

        return total_mass

    #TODO add error checks and safeguards
    def pipet_from_wellplate(self, wp_index, volume, parameters=None, aspirate=True, move_to_aspirate=True, well_plate_type="96 WELL PLATE"):
        """
        Aspirate or dispense from/to a wellplate.
        
        Args:
            wp_index: Well plate index
            volume: Volume in mL
            parameters (PipettingParameters, optional): Liquid handling parameters (uses defaults if None)
            aspirate (bool): True for aspirate, False for dispense (default: True)
            move_to_aspirate (bool): Whether to move to location (default: True)
            well_plate_type (str): Type of well plate (default: "96 WELL PLATE")
        """
        if parameters is None:
            parameters = PipettingParameters()
            
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
            self.pipet_aspirate(volume, wait_time=wait_time)
        else:
            self.pipet_dispense(volume, wait_time=wait_time, blowout_vol=blowout_vol)

    #Mix the well
    def mix_well_in_wellplate(self,wp_index,volume,repeats=3,well_plate_type="96 WELL PLATE"):
        self.logger.info(f"Mixing well {wp_index} in well plate of type {well_plate_type} with volume {volume:.3f} mL for {repeats} repeats")
        self.pipet_from_wellplate(wp_index,volume,well_plate_type=well_plate_type)
        self.pipet_from_wellplate(wp_index,volume,aspirate=False,move_to_aspirate=False,well_plate_type=well_plate_type)
        for i in range (1, repeats):
            self.pipet_from_wellplate(wp_index,volume,move_to_aspirate=False,well_plate_type=well_plate_type)
            self.pipet_from_wellplate(wp_index,volume,aspirate=False,move_to_aspirate=False,well_plate_type=well_plate_type)

    #Mix in a vial
    def mix_vial(self,vial_name,volume,repeats=3):
        self.logger.info(f"Mixing vial {vial_name} with volume {volume:.3f} mL for {repeats} repeats")
        vial_index= self.normalize_vial_index(vial_name)
        self.aspirate_from_vial(vial_index, volume, move_up=False, track_height=False)
        self.dispense_into_vial(vial_index, volume, initial_move=False)
        for i in range (1,repeats):
            self.dispense_from_vial_into_vial(vial_index, vial_index, volume)

    #Dispense an amount into a vial
    def dispense_into_vial(self, dest_vial_name, amount_mL, parameters=None, 
                          initial_move=True, measure_weight=False):
        """
        Dispense liquid into a vial.
        
        Args:
            dest_vial_name: Name of destination vial
            amount_mL: Amount to dispense in mL
            parameters (PipettingParameters, optional): Liquid handling parameters (uses defaults if None)
            initial_move (bool): Whether to perform initial movement (default: True)
            measure_weight (bool): Whether to measure mass during dispensing (default: False)
        """
        if parameters is None:
            parameters = PipettingParameters()
        
        # Extract liquid handling values from parameters
        wait_time = parameters.dispense_wait_time
        blowout_vol = parameters.blowout_vol
        air_vol = parameters.air_vol     
        
        dest_vial_num = self.normalize_vial_index(dest_vial_name) #Convert to int if needed

        # Ensure vial is accessible for pipetting (no use_safe_location for dispense)
        if not self._ensure_vial_accessible_for_pipetting(dest_vial_name, use_safe_location=False):
            return

        measured_mass = None

        self.logger.info(f"Pipetting into vial {self.get_vial_info(dest_vial_num, 'vial_name')}, amount: {round(amount_mL, 3)} mL")
        
        dest_vial_clamped = self.get_vial_info(dest_vial_num,'location')=='clamp' #Is the destination vial clamped?
        dest_vial_volume = self.get_vial_info(dest_vial_num,'vial_volume') #What is the current vial volume?

        #If the destination vial is at the clamp and you want the weight, measure prior to pipetting
        if measure_weight and dest_vial_clamped:
            if not self.simulate:
                initial_mass = self.c9.read_steady_scale()
            else:
                initial_mass = 0

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
            self.c9.goto_xy_safe(location, vel=self.get_speed('standard_xy'))   
        
        #Pipet into the vial
        #self.pipet_from_location(amount_mL, dispense_speed, height, aspirate = False, initial_move=initial_move)
        if initial_move:
            self.c9.move_z(height)
        self.pipet_dispense(amount_mL + air_vol, wait_time=wait_time, blowout_vol=blowout_vol)

        #Track the added volume in the dataframe
        self.VIAL_DF.at[dest_vial_num,'vial_volume']=self.VIAL_DF.at[dest_vial_num,'vial_volume']+amount_mL
        self.PIPET_FLUID_VOLUME -= amount_mL
        self.save_robot_status()

        #If the destination vial is at the clamp and you want the weight, measure after pipetting
        if measure_weight and dest_vial_clamped:
            if not self.simulate:
                final_mass = self.c9.read_steady_scale()
            else:
                final_mass = 0
            measured_mass = final_mass - initial_mass  

        return measured_mass

    # ====================================================================
    # 6. WELLPLATE OPERATIONS
    # ====================================================================

    def select_wellplate_grid(self, well_plate_type):
        grid_name = self.WELLPLATES[well_plate_type]["name_in_Locator"]  # Gets "well_plate_new_grid"
        actual_grid = getattr(Locator, grid_name)  # Gets the actual coordinate array
        return actual_grid

    #Dispense into a series of wells (dest_wp_num_array) a specific set of amounts (amount_mL_array)
    def dispense_into_wellplate(self, dest_wp_num_array, amount_mL_array, parameters=None, well_plate_type=None):
        """
        Dispenses specified amounts into a series of wells in a well plate.
        Args:
            dest_wp_num_array (list or range): Array of well indices to dispense into (e.g., [0, 1, 2])
            amount_mL_array (list[float]): Array of amounts (in mL) to dispense into each well (e.g., [0.1, 0.2, 0.3])
            parameters (PipettingParameters, optional): Standardized parameters (uses defaults if None)
            well_plate_type (str, optional): Type of well plate (defaults to "96 WELL PLATE")
        """
        if parameters is None:
            parameters = PipettingParameters()
        
        # Extract values directly from parameters
        wait_time = parameters.dispense_wait_time
        blowout_vol = parameters.blowout_vol
        air_vol = parameters.air_vol
        
        # Default well plate type if not specified
        if well_plate_type is None:
            well_plate_type = "96 WELL PLATE"
        
        # Determine dispense speed (can be done safely here)
        dispense_speed = parameters.dispense_speed or self.get_tip_dependent_aspirate_speed()
            
        self.adjust_pump_speed(0, dispense_speed) #Adjust the pump speed if needed
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

            height = self.get_height_at_location(location)
            height = self.adjust_height_based_on_pipet_held(height) 

            if first_dispense:
                self.c9.goto_xy_safe(location, vel=self.get_speed('standard_xy'))
                self.c9.move_z(height)
                first_dispense = False
                
            else:
                self.c9.goto_xy_safe(location, vel=self.get_speed('precise_movement'), accel=1, safe_height=height) #Use safe_height here!

            self.logger.info(f"Transferring {amount_mL:.3f} mL into well #{dest_wp_num_array[i]} of {well_plate_type}")

            #Dispense and then wait
            self.pipet_dispense(amount_mL + air_vol, wait_time=wait_time, blowout_vol=blowout_vol)     

        self.PIPET_FLUID_VOLUME -= np.sum(amount_mL_array)  # <-- Add this line back
        self.save_robot_status()    
        return True

    def dispense_from_vials_into_wellplate(self, well_plate_df, vial_names=None, parameters=None, strategy="auto", 
                                          low_volume_cutoff=0.05, buffer_vol=0.02, well_plate_type="96 WELL PLATE"):
        """
        Dispense from multiple vials into wellplate wells using strategy pattern.
        
        Args:
            well_plate_df (DataFrame): DataFrame where columns are vial names and rows are well volumes
            vial_names (list, optional): DEPRECATED - for backwards compatibility only
            parameters (PipettingParameters, optional): Liquid handling parameters 
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
        
        if parameters is None:
            parameters = PipettingParameters()
        
        # Extract vial names from DataFrame columns
        vial_names = well_plate_df.columns.tolist()
        
        self.logger.info(f"Dispensing from vials {vial_names} into wellplate using {strategy} strategy")
        
        # Auto-select strategy based on parameter complexity
        if strategy == "auto":
            needs_precision = (parameters.pre_asp_air_vol > 0 or 
                              parameters.post_asp_air_vol > 0 or 
                              parameters.asp_disp_cycles > 0 or
                              parameters.blowout_vol > 0)
            strategy = "serial" if needs_precision else "batched"
            self.logger.debug(f"Auto-selected {strategy} strategy (precision needed: {needs_precision})")
        
        # Dispatch to appropriate strategy
        if strategy == "serial":
            return self._dispense_wellplate_serial(well_plate_df, parameters, low_volume_cutoff, well_plate_type)
        elif strategy == "batched":
            return self._dispense_wellplate_batched(well_plate_df, parameters, low_volume_cutoff, buffer_vol, well_plate_type)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'serial', 'batched', or 'auto'")

    def _dispense_wellplate_serial(self, well_plate_df, parameters, well_plate_type):
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
                self.aspirate_from_vial(vial_name, volume, parameters=parameters)  # Automatic tip selection
                self.dispense_into_wellplate([well_idx], [volume], parameters=parameters, 
                                           well_plate_type=well_plate_type)
            
            # Remove tip after processing all volumes for this vial
            if self.HELD_PIPET_TYPE is not None:
                self.remove_pipet()
        
        # Clean up any vials left in clamp after all dispensing
        clamp_vial_index = self.get_vial_in_location('clamp', 0)
        if clamp_vial_index is not None:
            self.recap_clamp_vial()
            self.return_vial_home(clamp_vial_index)
        
        self.logger.info("Serial wellplate dispensing completed")
        return True

    def _dispense_wellplate_batched(self, well_plate_df, parameters, low_volume_cutoff, buffer_vol, well_plate_type):
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
                        parameters, base_wait_time, tip_type, well_plate_type
                    )
                    
                    dispensed += batch_total
                    well_idx = next_idx
                    self.logger.debug(f"Batched {len(batch_wells)} wells, total: {batch_total:.3f} mL")
                
                # Remove pipet after finishing this vial (if any dispensing occurred)
                if dispensed > 1e-6:
                    self.remove_pipet()
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
                      parameters, wait_time, tip_type, well_plate_type):
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
        self.aspirate_from_vial(vial_name, total_aspirate, parameters=parameters, specified_tip=tip_type)
        
        # Dispense to wells
        self.dispense_into_wellplate(batch_wells, batch_volumes, 
                                   parameters=parameters, well_plate_type=well_plate_type)
        
        # Return buffer volume to source vial if any
        if extra_aspirate_vol > 1e-6:
            self.logger.debug(f"Returning buffer volume: {extra_aspirate_vol:.3f} mL to vial {vial_name}")
            self.dispense_into_vial(vial_name, extra_aspirate_vol, parameters=parameters, initial_move=False)
        
        return batch_total

    # ====================================================================
    # RESERVOIR SYSTEM
    # ====================================================================

    #Prime the line from the reservoir to the vial. In theory this could happen automatically. Probably good to do it if you are using a reservoir. 
    def prime_reservoir_line(self, reservoir_index, overflow_vial, volume=0.5):
        overflow_vial = self.normalize_vial_index(overflow_vial) #Convert to int if needed
        self.logger.info(f"Priming reservoir {reservoir_index} line into vial {overflow_vial}: {volume:.3f} mL")
        self.dispense_into_vial_from_reservoir(reservoir_index,overflow_vial,volume)

    def dispense_into_vial_from_reservoir(self,reservoir_index,vial_index,volume, measure_weight = False, return_home=True):
        
        vial_index = self.normalize_vial_index(vial_index) #Convert to int if needed
        self.logger.info(f"Dispensing into vial {vial_index} from reservoir {reservoir_index}: {volume:.3f} mL")
        measured_mass = None

        #Step 1: move the vial to the clamp
        if not self.get_vial_info(vial_index,'location')=='clamp':
            # Safety is now handled in move_vial_to_location method
            self.move_vial_to_location(vial_index,'clamp',0)
        if not self.is_vial_pipetable(vial_index):
            self.uncap_clamp_vial()
        self.move_home()

        if measure_weight: #weigh before dispense (if measure_weight = True)
            initial_mass = self.c9.read_steady_scale()

        #Step 2: move the carousel to reservoir position
        carousel_angle = self.get_config_parameter('pumps', reservoir_index, 'carousel_angle', error_on_missing=False)
        carousel_height = self.get_config_parameter('pumps', reservoir_index, 'carousel_height', error_on_missing=False)
        
        if carousel_angle is not None and carousel_height is not None:
            self.c9.move_carousel(carousel_angle, carousel_height)
        else:
            self.logger.warning(f"No carousel configuration found for pump {reservoir_index}")
            
        #Step 3: aspirate and dispense from the reservoir

        max_volume = self.get_config_parameter('pumps', reservoir_index, 'volume', error_on_missing=False) or 2.5

        
        num_dispenses = math.ceil(volume/max_volume)
        dispense_vol = volume/num_dispenses
        self.logger.debug(f"Dispensing {dispense_vol:.3f} mL {num_dispenses} times")
        for i in range (0, num_dispenses):        
             self.c9.set_pump_valve(reservoir_index,self.c9.PUMP_VALVE_LEFT)
             self.c9.aspirate_ml(reservoir_index,dispense_vol)
             self.c9.set_pump_valve(reservoir_index,self.c9.PUMP_VALVE_RIGHT)
             self.c9.dispense_ml(reservoir_index,dispense_vol)
        if not self.simulate:
            time.sleep(1)
        vial_volume = self.get_vial_info(vial_index,'vial_volume')
        self.VIAL_DF.at[vial_index,'vial_volume']=(vial_volume+volume)
        self.save_robot_status()

        #Step 4: Return the vial back to home
        self.c9.move_carousel(0,0)

        if measure_weight: #weigh after dispense (if measure_weight = True)
            final_mass = self.c9.read_steady_scale()
            measured_mass = final_mass - initial_mass

        if not self.get_vial_info(vial_index,'capped'):
            self.recap_clamp_vial()
        if return_home:
            self.return_vial_home(vial_index)
        return measured_mass

    # ====================================================================
    # 7. VIAL & CONTAINER MANAGEMENT
    # ====================================================================

    #Check the original status of the vial in order to send it to its home location
    def return_vial_home(self,vial_name):
        """
        Return the specified vial to its home location.
        Args:
            `vial_name` (str): Name of the vial to return home.
        """
        

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed
        self.logger.info(f"Returning vial {self.get_vial_info(vial_index, 'vial_name')} to home location")
        
        home_location = self.get_vial_info(vial_index,'home_location')
        home_location_index = self.get_vial_info(vial_index,'home_location_index')
        
        vial_location = self.get_vial_info(vial_index,'location')
        if vial_location == 'clamp' and self.GRIPPER_STATUS == "Cap":
            self.recap_clamp_vial()
        self.move_vial_to_location(vial_index,home_location,home_location_index)
        self.save_robot_status()

    #Drop off a vial at a location that you already have
    def drop_off_vial(self, vial_name, location, location_index):

        self.logger.debug(f"Dropping off vial {vial_name} at location {location} with index {location_index}")

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        destination = self.get_location(False,location,location_index)
        occupying_vial = self.get_vial_in_location(location,location_index)
        # Allow drop-off if the location is empty or occupied by the same vial
        destination_empty = (occupying_vial is None) or (occupying_vial == vial_index)

        self.check_for_errors([[destination_empty, True, "Cannot move vial to destination, destination full"]],True)

        self.c9.goto_safe(destination) #move vial to destination
        self.c9.open_gripper() #release vial
        
        self.VIAL_DF.at[vial_index, 'location']=location
        self.VIAL_DF.at[vial_index, 'location_index']=location_index
        self.GRIPPER_STATUS = None #We no longer have the vial
        self.GRIPPER_VIAL_INDEX = None
        self.save_robot_status() #Update in memory

    def grab_vial(self,vial_name):
        
        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed
        
        self.logger.debug(f"Grabbing vial {vial_name} with index {vial_index}")
        initial_location = self.get_vial_location(vial_index, False)
        loc = self.get_vial_info(vial_index,'location')

        if loc == 'clamp' and self.GRIPPER_STATUS == "Cap":
            self.recap_clamp_vial()

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS is None, True, "Cannot move vial to destination, gripper full"])
        #error_check_list.append([self.HELD_PIPET_INDEX is None, True, "Cannot move vial to destination, robot holding pipet"])
        error_check_list.append([self.is_vial_movable(vial_index), True, "Can't move vial, vial is uncapped."])  

        self.check_for_errors(error_check_list,True) #Check for an error, and pause if there's an issue

        #self.open_gripper()
        self.goto_location_if_not_there(initial_location) #move to vial
        self.c9.close_gripper() #grip vial
        
        self.GRIPPER_STATUS = "Vial" #Update the status of the robot
        self.GRIPPER_VIAL_INDEX = vial_index
        self.save_robot_status() #Save the status of the robot

    #Send the vial to a specified location
    def move_vial_to_location(self,vial_name:str,location:str,location_index:int):
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
        self.grab_vial(vial_index) #Grab the vial
        self.drop_off_vial(vial_index,location,location_index) #Drop off the vial

    def get_vial_in_location(self, location_name, location_index):
        # Filter rows where both conditions match
        mask = (self.VIAL_DF['location'] == location_name) & (self.VIAL_DF['location_index'] == location_index)
        
        # Get the matching values
        matching_vials = self.VIAL_DF.loc[mask, 'vial_index'].values

        # Return the first match or None if no match is found
        return int(matching_vials[0]) if len(matching_vials) > 0 else None

    #Uncap the vial in the clamp
    def uncap_clamp_vial(self, revs=3):
        self.logger.debug("Removing cap from clamped vial")

        clamp_vial_index = self.get_vial_in_location('clamp',0)

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS is None, True, "Cannot uncap, gripper full"])
        error_check_list.append([clamp_vial_index is None, False, "Cannot uncap, no vial in clamp"])
        error_check_list.append([self.is_vial_movable(clamp_vial_index), True, "Can't uncap, vial is uncapped already"])

        self.check_for_errors(error_check_list,True) #Check for an error and pause if there is one
        
        self.goto_location_if_not_there(vial_clamp) #Maybe check if it is already there or not   
        self.c9.close_clamp() #clamp vial
        self.c9.close_gripper()
        self.c9.uncap(revs=revs)
        self.GRIPPER_STATUS = "Cap"
        self.c9.open_clamp()

        self.VIAL_DF.at[clamp_vial_index, 'capped']=False
        self.GRIPPER_VIAL_INDEX = clamp_vial_index
        self.save_robot_status()

    #Recap the vial in the clamp
    def recap_clamp_vial(self, revs=2.0, torque_thresh = 600):
        self.logger.debug("Recapping clamped vial")
        
        clamp_vial_index = self.get_vial_in_location('clamp',0)

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS, "Cap", "Cannot recap, no cap in gripper"])
        error_check_list.append([clamp_vial_index is None, False, "Cannot recap, no vial in clamp"])
        error_check_list.append([self.is_vial_movable(clamp_vial_index), False, "Can't recap, vial is capped already"])
        
        self.check_for_errors(error_check_list,True) #Let's pause if there is an error

        self.goto_location_if_not_there(vial_clamp)
        self.c9.close_clamp() #Make sure vial is clamped
        self.c9.cap(revs=revs, torque_thresh = torque_thresh) #Cap the vial #Cap the vial
        self.c9.open_gripper() #Open the gripper to release the cap
        self.GRIPPER_STATUS = None
        self.c9.open_clamp()

        self.VIAL_DF.at[clamp_vial_index, 'capped']=True #Update the vial status
        self.GRIPPER_VIAL_INDEX = None
        self.save_robot_status()

    #Checks first that you aren't already there... This mostly applies for cap/decap
    def goto_location_if_not_there(self, location):
        difference_threshold = 550
        if self.get_location_distance(location, self.c9.get_robot_positions()) > difference_threshold:
            self.c9.goto_safe(location, vel=self.get_speed('fast_approach'))

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

    #Get some piece of information about a vial
    #vial_index,vial_name,location,location_index,vial_volume,capped,cap_type,vial_type
    def get_vial_info(self,vial_name,column_name):

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        values = self.VIAL_DF.loc[self.VIAL_DF['vial_index'] == vial_index, column_name].values
        if len(values) > 0:
            return values[0]  # Return the first match
        else:
            return None  # Handle case where no match is found    

    def get_vial_index_from_name(self,vial_name):
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
                # Array of positions - return the specified index
                return location_data[location_index]
                
        except (AttributeError, IndexError) as e:
            self.pause_after_error(f"Invalid location {location_name}[{location_index}]: {e}")
            return None
    

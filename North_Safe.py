from robot_state.Locator import * #Let's try to eliminate this in the future
import robot_state.Locator as Locator
import numpy as np
import os
import time
import math
from numbers import Integral
import pandas as pd
import yaml
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from pipetting_data.pipetting_parameters import PipettingParameters, ReservoirParameters
from pipetting_data.pipetting_wizard import PipettingWizard, NEAR_EXTRAPOLATION_FRACTION
from pipetting_data.dispense_recording import recording_density, record_dispense, capture_aspiration
from pipetting_data.liquid_registry import liquid_conditions_tip
from pipetting_data.mass_analysis import analyze_mass_readings, empty_stability, save_mass_readings
import matplotlib.patches as patches
import threading
import json
from robot_state.plate_inventory import PlateInventory, LID_ALIASES, number
from experiment_tracking import experiment_run_logger
from experiment_tracking import sample_tracking
from experiment_tracking import sample_contents

_QUIET_YAML_CACHE = {}

def _read_yaml_quietly(file_path):
    """Read a YAML file once and cache it; return {} if it is missing or invalid."""
    if file_path not in _QUIET_YAML_CACHE:
        try:
            with open(file_path, 'r') as file:
                _QUIET_YAML_CACHE[file_path] = yaml.safe_load(file) or {}
        except (OSError, yaml.YAMLError):
            _QUIET_YAML_CACHE[file_path] = {}
    return _QUIET_YAML_CACHE[file_path]


ROBOT_HARDWARE_PATH = "robot_state/robot_hardware.yaml"


def _hardware_limits(owner, section):
    """The `section` limits from robot_hardware.yaml (KeyError if missing)."""
    hardware = getattr(owner, 'ROBOT_HARDWARE', None) or _read_yaml_quietly(ROBOT_HARDWARE_PATH)
    return hardware[section]


def _check_optional_speed(owner, action, name, speed):
    """A move speed given by the caller (None: the default) must be a percent of max velocity within
    speed_limits (robot_hardware.yaml); the C9 reads 0-100 as percent and larger values as counts/s."""
    if speed is not None:
        limits = _hardware_limits(owner, 'speed_limits')
        _check_in_range(action, name, speed, limits['min_percent'], limits['max_percent'], '%',
                        'speed_limits')


def _check_parameter_limits(owner, action, parameters):
    """Each resolved PipettingParameters / ReservoirParameters value (None: the default) must be within
    parameter_limits (robot_hardware.yaml); ValueError before moving otherwise."""
    for name, limit in _hardware_limits(owner, 'parameter_limits').items():
        value = getattr(parameters, name, None)
        if value is not None:
            _check_in_range(action, name, value, limit['min'], limit['max'], f" {limit.get('unit', '')}".rstrip(),
                            'parameter_limits', whole=limit.get('whole', False))


def _check_speed_percent(owner, action, speed_name, speed, source):
    """A configured move speed must be a percent of max velocity within speed_limits (robot_hardware.yaml);
    the C9 reads 0-100 as percent and larger values as counts/s."""
    limits = _hardware_limits(owner, 'speed_limits')
    try:
        _check_in_range(action, f"speed '{speed_name}'", speed, limits['min_percent'], limits['max_percent'], '%',
                        'speed_limits')
    except ValueError as error:
        raise ValueError(f"{error} [set in {source}]") from None


def _check_in_range(action, name, value, low, high, unit, section, whole=False):
    """ValueError "<action> refused before moving" unless value is a number within [low, high]
    (and a whole number if whole). Returns nothing: callers pass the original value on."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{action} refused before moving: {name} {value!r} is not a number") from None
    if not low <= number <= high:  # also refuses NaN
        raise ValueError(f"{action} refused before moving: {name} {value!r} is outside "
                         f"{low}-{high}{unit} ({section} in robot_hardware.yaml)")
    if whole and number != int(number):
        raise ValueError(f"{action} refused before moving: {name} {value!r} is not a whole number")


class ScaleRecording:
    """Scale readings taken in the background around one liquid step (see North_Base.start_scale_recording).

    Phases: 'baseline_pre' until mark_start(), then the operation phase until mark_end(), then
    'baseline_post'. Inactive (not requested, or simulation): takes no readings and never waits."""

    def __init__(self, robot, active, phase, settle):
        self.robot, self.phase, self.readings = robot, phase, []
        self.active = bool(active) and not robot.simulate
        self._start = self._end = None
        self._thread = None
        if not self.active:
            return
        self.settings = _hardware_limits(robot, 'scale_recording')
        if settle:
            robot.logger.info(f"Waiting {self.settings['settle_s']:.1f}s for the scale to settle after positioning...")
            time.sleep(self.settings['settle_s'])
        self._running = threading.Event()
        self._running.set()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()
        robot.logger.info(f"Started continuous mass monitoring with pre-{phase} baseline")
        time.sleep(self.settings['pre_baseline_s'])

    def _record(self):
        while self._running.is_set():
            try:
                now = time.time()
                steady_status, weight = self.robot.read_scale()
                if self._start is None or now < self._start:
                    phase = 'baseline_pre'
                elif self._end is None or now < self._end:
                    phase = self.phase
                else:
                    phase = 'baseline_post'
                self.readings.append({'timestamp': now, 'time_relative': now - self._t0, 'mass_g': weight,
                                      'phase': phase, 'steady_status': steady_status})
                self.robot.logger.debug(f'Mass: {weight:.4f}g at {now - self._t0:.3f}s ({phase}) steady={steady_status}')
            except Exception as e:
                self.robot.logger.warning(f"Scale reading failed: {e}")
            time.sleep(self.settings['read_interval_s'])

    def mark_start(self):
        """The liquid step starts now."""
        if self.active:
            self._start = time.time()

    def mark_end(self):
        """The liquid step ended now."""
        if self.active:
            self._end = time.time()
            self.robot.logger.info(f"{self.phase.capitalize()} took {self._end - self._start:.3f}s, "
                                   f"collecting post-{self.phase} baseline")

    def stop(self):
        """Record the post-baseline, then stop. Returns the readings."""
        if self.active and self._thread is not None:
            time.sleep(self.settings['post_baseline_s'])
            self._running.clear()
            self._thread.join(timeout=self.settings['stop_timeout_s'])
            self._thread = None
            self.robot.logger.info(f"Stopped continuous mass monitoring. Collected {len(self.readings)} data points")
        return self.readings


class North_Base:
    """Base class for North robot components with shared functionality"""

    ROBOT_HARDWARE_PATH = ROBOT_HARDWARE_PATH
    CAROUSEL_POSITIONS_PATH = "robot_state/carousel_positions.yaml"

    def _hardware_setting(self, section, key, default):
        """Look up a value in robot_hardware.yaml, falling back to `default` if it is not set."""
        hardware = getattr(self, 'ROBOT_HARDWARE', None) or _read_yaml_quietly(self.ROBOT_HARDWARE_PATH)
        value = (hardware.get(section) or {}).get(key)
        return default if value is None else value

    # ---- Primitives: scale (shared C9 controller) ----
    def read_scale(self):
        """Primitive: read the balance once. Returns (steady_status, mass_g)."""
        return self.c9.read_scale()

    def read_steady_scale(self):
        """Primitive: wait for a steady balance reading and return the mass (g)."""
        return self.c9.read_steady_scale()

    def start_scale_recording(self, active, phase, settle=True):
        """Unit operation: record the scale in the background around one liquid step.

        Waits settle_s (if settle), starts recording, waits pre_baseline_s; call mark_start() and
        mark_end() around the step, then stop() (waits post_baseline_s, returns the readings).
        Timings: scale_recording in robot_hardware.yaml. Does nothing (no readings, no waits) if not
        active or in simulation. Analyze the readings with pipetting_data/mass_analysis.py."""
        return ScaleRecording(self, active, phase, settle)

    def zero_scale(self):
        """Primitive: tare the balance."""
        return self.c9.zero_scale()

    # ---- Primitives: carousel ----
    _DEFAULT_CAROUSEL_POSITIONS = {
        'home': {'angle': 0, 'height': 0},
        'powder_dispense_mg': {'angle': 66.5, 'height': 70},
    }

    def move_carousel_to(self, position_name):
        """Primitive: move the carousel to a named position from carousel_positions.yaml."""
        position = _read_yaml_quietly(self.CAROUSEL_POSITIONS_PATH).get(position_name)             or self._DEFAULT_CAROUSEL_POSITIONS[position_name]
        self.c9.move_carousel(position['angle'], position['height'])

    
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
            self.logger.warning("Paused, waiting for operator")
            started = time.monotonic()
            experiment_run_logger.begin_pause(self.logger)
            interrupted = False
            resumed = False
            try:
                input(f"Error: {err_message}. Press Enter to continue...")
                resumed = True
            except KeyboardInterrupt:
                interrupted = True
                raise
            finally:
                paused_sec = time.monotonic() - started
                experiment_run_logger.end_pause(self.logger, paused_sec, interrupted)
                if interrupted:
                    self.logger.warning(f"Stopped by operator after {paused_sec / 60:.2f} min")
                elif resumed:
                    self.logger.info(f"Resumed after {paused_sec / 60:.2f} min")
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

        if config_name == 'vial_positions':
            rack = config_dict[key]
            if type(rack['rack_present']) is not bool:
                raise ValueError(f"rack_present must be true or false: {key}")
            if not rack['rack_present']:
                raise ValueError(f"Rack is not present on the deck: {key}")
        
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

        # Inventory is loaded once definitions and positions are available.
        self._inventory = None
        self.CURRENT_GRIPPER_LOCATION = None  # Track where the gripper physically is: 'pipetting_area', 'cytation_tray', etc.
        self.CURRENT_GRIPPER_POSITION = {'x': None, 'z': None}  # Track physical coordinates
        self.simulate = simulate

        #Load yaml data
        self.logger.debug("Loading track status from file: %s", "robot_state/track_status.yaml")
        self.TRACK_STATUS_FILE = "robot_state/track_status.yaml"
        if os.environ.get("NORTH_SCHEDULER_SIMULATION_STATE"):
            from scheduler.simulation_state import configure_controller
            configure_controller(self, "track")
        # Read state after loading its definitions.
        
        # Load track positions configuration
        self.TRACK_POSITIONS_FILE = "robot_state/track_positions.yaml"
        self._load_track_positions()
        
        # Load robot hardware configuration for axis mappings
        self.ROBOT_HARDWARE_FILE = "robot_state/robot_hardware.yaml"
        self._load_robot_hardware()
        
        # Load wellplate configuration using base class method
        self._load_wellplate_config()

        self.get_track_status()
        # Preserve held objects at startup; never automatically discard them.
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
        """Reload the plate inventory and track pose from track_status.yaml."""
        loaded = PlateInventory.load(self.TRACK_STATUS_FILE, self.WELLPLATES, self.TRACK_POSITIONS)
        if getattr(self, "_inventory", None) is None:
            self._inventory = loaded
        else:
            # Preserve the object shared with North_Robot on explicit reload.
            self._inventory.data = loaded.data
            self._inventory.legacy = loaded.legacy
        pose = self._inventory.data.get("track_pose", {})
        self.CURRENT_GRIPPER_LOCATION = pose.get("location")
        self.CURRENT_GRIPPER_POSITION = {"x": pose.get("x"), "z": pose.get("z")}
        if self._inventory.legacy:
            self.logger.warning("Legacy track inventory loaded in memory; review plate/lid positions in Track Status before live use.")

    @property
    def inventory(self):
        """The PlateInventory (shared with North_Robot)."""
        return self._inventory

    @property
    def NUM_SOURCE(self):
        """Number of plates in the source stack."""
        return len(self.inventory.contents("source_stack"))

    @property
    def NUM_WASTE(self):
        """Number of plates in the waste stack."""
        return len(self.inventory.contents("waste_stack"))

    @property
    def CURRENT_WP_TYPE(self):
        """The one plate type on the deck; ValueError if there are several."""
        types = {obj["type"] for obj in self.inventory.objects.values() if obj["kind"] == "plate"}
        if len(types) != 1:
            raise ValueError("No single global plate type; select a plate by position or ID")
        return next(iter(types))

    @property
    def ACTIVE_WELLPLATE_POSITION(self):
        """The one position (not a stack) holding a plate, or None; ValueError if several."""
        locations = [location for location in self.inventory.positions
                     if location not in ("source_stack", "waste_stack")
                     and any(self.inventory.objects[item]["kind"] == "plate"
                             for item in self.inventory.contents(location))]
        if len(locations) > 1:
            raise ValueError("Several plates are active; query the required position")
        return locations[0] if locations else None

    def save_track_status(self):
        """Save the inventory and track pose to track_status.yaml (not in simulation, unless the scheduler allows it)."""
        self.inventory.data["track_pose"] = {
            "location": self.CURRENT_GRIPPER_LOCATION, **self.CURRENT_GRIPPER_POSITION,
        }
        save_state = not self.simulate
        if getattr(self, "_scheduler_state_root", None) is not None:
            from scheduler.simulation_state import can_save
            save_state = can_save(self, self.TRACK_STATUS_FILE)
        if save_state:
            self.inventory.save(self.TRACK_STATUS_FILE)

    def check_input_file(self, pause_after_check=True, visualize=True):
        """Deprecated: prints a notice and waits for Enter. Use the Lash_E start-up GUI."""
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
        """Move the track to its origin, unless a held plate or unfinished transfer is recorded."""
        if self.inventory.positions["gripper"] or self.inventory.data.get("pending_transfer"):
            self.logger.info("Preserving recorded gripper/transfer state; no automatic recovery motion.")
            return
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
        """Removed: always raises ValueError. Use inventory transfers."""
        raise ValueError("Use inventory transfers; a single active-plate position cannot describe this state")
    
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
        """Track speed by name from track_positions.yaml (percent of max velocity, 1-100;
        speed_limits in robot_hardware.yaml). ValueError before moving if outside that range."""
        speeds = self.TRACK_POSITIONS.get('speeds', {})
        speed = speeds.get(speed_name)
        if speed is None:
            fallback = self.TRACK_POSITIONS['missing_speed_fallback']
            self.logger.warning(f"Speed '{speed_name}' not found, using default ({fallback})")
            speed_name, speed = 'missing_speed_fallback', fallback
        _check_speed_percent(self, 'Track move', speed_name, speed, 'track_positions.yaml')
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
        """No effect (vel is ignored): nothing reads c9.DEFAULT_X_SPEED. Track moves use the speeds in track_positions.yaml."""
        self.logger.debug("Setting horizontal speed to %d", vel)
        self.c9.DEFAULT_X_SPEED = vel

    def set_vertical_speed(self,vel):
        """No effect (vel is ignored): nothing reads c9.DEFAULT_Y_SPEED. Track moves use the speeds in track_positions.yaml."""
        self.logger.debug("Setting vertical speed to %d", vel)
        self.c9.DEFAULT_Y_SPEED = vel    

    def open_gripper(self):
        """Open the track gripper (outputs in robot_hardware.yaml; waits gripper_delay s)."""
        self.logger.debug("Opening gripper")
        self.c9.set_output(self.get_axis('gripper_open'), True)  
        self.c9.set_output(self.get_axis('gripper_close'), False)
        self.c9.delay(self.get_timing('gripper_delay'))
    
    def close_gripper(self):
        """Close the track gripper (outputs in robot_hardware.yaml; waits gripper_delay s)."""
        self.logger.debug("Closing gripper")
        self.c9.set_output(self.get_axis('gripper_close'), True)  #gripper close
        self.c9.set_output(self.get_axis('gripper_open'), False)
        self.c9.delay(self.get_timing('gripper_delay'))

    def origin(self):
        """Raise the track to max_safe_height, then move to x = 0 (counts; track_positions.yaml)."""
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

    def grab_wellplate_from_location(self, location_name, wellplate_type=None, waypoint_locations=None, z_override=None, object_kind=None):
        """Pick up the top plate or lid at location_name (track_positions.yaml).

        wellplate_type: expected type (wellplates.yaml), or a lid alias; ValueError if it differs
        waypoint_locations: named waypoints for the approach
        z_override: grab height in track counts instead of the stack height
        object_kind: 'plate' or 'lid' to require one
        Refused before moving (ValueError) if the gripper is full or the object does not match.
        Returns the object id."""
        if location_name == "cytation_tray" and hasattr(self, "check_reader_motion"):
            self.check_reader_motion()
        inv = self.inventory
        if inv.positions["gripper"] is not None:
            raise ValueError("Track gripper is occupied")
        lid_type = LID_ALIASES.get(wellplate_type)
        kind = "lid" if lid_type else object_kind
        source_id = inv.top(location_name)
        if kind == "lid" and inv.objects[source_id]["kind"] == "plate":
            plate_id = source_id
            if plate_id not in inv.attachments:
                raise ValueError(f"Plate {plate_id} has no attached lid")
            source_id = inv.attachments[plate_id]
        obj = inv.objects[source_id]
        expected = lid_type or wellplate_type
        if expected is not None and inv.plate_type(source_id) != expected:
            raise ValueError(f"Object {source_id} is not expected type {expected}")
        if kind is not None and obj["kind"] != kind:
            raise ValueError(f"Expected {kind} at {location_name}")
        cfg = self.get_position(location_name)
        z = inv.stack_z(location_name, "grab", obj["kind"]) if z_override is None else number(z_override, "Z override")
        offset = number(self.WELLPLATES[inv.plate_type(source_id)].get(
            "lid_gripping_z_offset" if obj["kind"] == "lid" else "gripping_z_offset"), "gripping offset")
        self.logger.info("Pickup %s (%s) from %s: stack Z=%s, grip offset=%s, commanded Z=%s, path=%s",
                         source_id, inv.plate_type(source_id), location_name, z, offset, z-offset, waypoint_locations)
        inv.begin(source_id, location_name, "gripper")
        self.save_track_status()
        self.transfer_wellplate_via_path(cfg['x'], cfg['z_transfer'], waypoint_locations)
        self.open_gripper()
        self.c9.move_axis(self.get_axis('z_axis'), z - offset, vel=self.get_speed('slow_movement'))
        self.close_gripper()
        inv.commit_placement(source_id, "gripper")
        self.save_track_status()
        self.c9.move_axis(self.get_axis('z_axis'), cfg['z_transfer'], vel=self.get_speed('default_z'))
        self.update_gripper_location(location_name, x=cfg['x'], z=cfg['z_transfer'])
        inv.finish()
        self.save_track_status()
        return source_id

    def grab_lid_from_location(self, location_name, **kwargs):
        """Pick up the lid at location_name (grab_wellplate_from_location with object_kind='lid')."""
        return self.grab_wellplate_from_location(location_name, object_kind="lid", **kwargs)

    def release_wellplate_in_location(self, location_name, wellplate_type=None, waypoint_locations=None, z_override=None):
        """Put the held plate or lid down at location_name (track_positions.yaml).

        wellplate_type: expected type; ValueError if the held object differs
        waypoint_locations: named waypoints for the approach
        z_override: release height in track counts instead of the stack height
        Refused before moving (ValueError) if the destination cannot take it. Returns the object id."""
        if location_name == "cytation_tray" and hasattr(self, "check_reader_motion"):
            self.check_reader_motion()
        inv = self.inventory
        object_id = inv.top("gripper")
        expected = LID_ALIASES.get(wellplate_type, wellplate_type)
        if expected is not None and inv.plate_type(object_id) != expected:
            raise ValueError(f"Held object {object_id} is not expected type {expected}")
        inv.check_destination(object_id, location_name)
        obj = inv.objects[object_id]
        cfg = self.get_position(location_name)
        z = inv.stack_z(location_name, "release", obj["kind"]) if z_override is None else number(z_override, "Z override")
        offset = number(self.WELLPLATES[inv.plate_type(object_id)].get(
            "lid_gripping_z_offset" if obj["kind"] == "lid" else "gripping_z_offset"), "gripping offset")
        self.logger.info("Place %s (%s) at %s: stack Z=%s, grip offset=%s, commanded Z=%s, path=%s",
                         object_id, inv.plate_type(object_id), location_name, z, offset, z-offset, waypoint_locations)
        inv.begin(object_id, "gripper", location_name)
        self.save_track_status()
        self.transfer_wellplate_via_path(cfg['x'], cfg['z_transfer'], waypoint_locations)
        self.c9.move_axis(self.get_axis('z_axis'), z - offset, vel=self.get_speed('slow_movement'))
        self.open_gripper()
        inv.commit_placement(object_id, location_name)
        self.save_track_status()
        self.c9.move_axis(self.get_axis('z_axis'), cfg['z_transfer'], vel=self.get_speed('default_z'))
        self.update_gripper_location(location_name, x=cfg['x'], z=cfg['z_transfer'])
        inv.finish()
        self.save_track_status()
        return object_id


    # === 6. STACK MANAGEMENT ===
    def calculate_wp_stack_height(self, num=None, wp_type=None, stack_name="source_stack", operation='grab'):
        """Compatibility name: geometry comes from actual ordered contents, not num/type."""
        return self.inventory.stack_z(stack_name, operation)

    def move_wellplate(self, source, destination, expected_type=None, move_home_afterwards=True):
        """Move the plate at source to destination (track position names), checking both ends
        before picking up. expected_type: plate type (wellplates.yaml). Returns the plate id."""
        inv = self.inventory
        plate_id = inv.plate_at(source, expected_type)
        inv.check_destination(plate_id, destination)
        # Validate both ends before picking up; no whole-workflow preflight required.
        inv.stack_z(source, "grab")
        inv.stack_z(destination, "release")
        # Preserve the established stack routes. A source-stack pickup first
        # rises to max_height. Approaching the active area from home for discard
        # must visit transfer_stack BEFORE lowering to its transfer Z.
        if source in ("source_stack", "waste_stack"):
            pickup_path = ['max_height']
        else:
            self.close_gripper()
            pickup_path = ['transfer_stack']
        self.grab_wellplate_from_location(source, expected_type, waypoint_locations=pickup_path)
        self.release_wellplate_in_location(destination, expected_type, waypoint_locations=['transfer_stack'])
        if move_home_afterwards:
            self.origin()
        return plate_id

    def get_new_wellplate(self, move_home_afterwards=True):
        """Move a new plate from the source stack to the pipetting area. Returns the plate id."""
        return self.move_wellplate("source_stack", "pipetting_area", move_home_afterwards=move_home_afterwards)

    def discard_wellplate(self, move_home_afterwards=True, pickup_location='pipetting_area', discard_location='waste_stack'):
        """Move the plate at pickup_location to discard_location (default: waste stack). Returns the plate id."""
        return self.move_wellplate(pickup_location, discard_location, move_home_afterwards=move_home_afterwards)

    # === 7. UTILITIES ===        
    def get_ordinal(self,n): #convert n into an ordinal number (ex. 1st, 2nd, 4th)  -- from chatgpt
        """n as an ordinal string (1 -> '1st')."""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

class North_Powder(North_Base):

    p2 = None
    c9 = None
    POWDER_CHANNELS_PATH = "robot_state/powder_channels.yaml"

    def __init__(self, c9,simulate = False, logger=None):

        self.c9 = c9
        self.simulate = simulate
        self.logger = logger
        self.logger.debug("Initializing North Powder Dispenser...")
        if not simulate:
            from north import NorthC9
            self.p2 = NorthC9('C', network=c9.network)
        else:
            self.p2 = MagicMock()  # simulated dispenser board: commands are accepted and do nothing

    def powder_channels(self):
        """The cartridge slots listed in powder_channels.yaml, as {channel: settings}."""
        return {int(k): v for k, v in (_read_yaml_quietly(self.POWDER_CHANNELS_PATH) or {}).items()}

    def close_all_valves(self):
        """Safe stop: close the valve of every listed cartridge slot."""
        for channel in self.powder_channels():
            self.set_opening(channel, 0)

    def activate_powder_channel(self, channel=0):
        """Select powder channel (listed in powder_channels.yaml) and home its stepper."""
        self.logger.debug(f"Activating powder channel: {channel}")
        self.p2.activate_powder_channel(channel)
        self.p2.home_OL_stepper(0, 300)

    # shake the cartridge for t ms, with frequency f [40, 180] hz, amplitude [60, 100] %
    def shake(self,t, f=80, a=80, wait=True):
        """Shake the cartridge for t ms at frequency f (Hz, default 80) and amplitude a (%, default 80)."""
        self.logger.debug("Shaking dispenser cartridge...")
        return self.p2.amc_pwm(int(f), int(t), int(a), wait=wait)
    
    def set_opening(self, channel, deg):
        """Set the valve opening of channel to deg degrees (0 closed; protocols use up to 90)."""
        self.logger.debug(f"Setting opening for channel {channel} to {deg} degrees")
        self.p2.move_axis(channel, deg*(1000/360.0))

    def cl_pow_dispense(self,mg_target, channel, protocol=None, zero_scale=True, max_tries=20):
        """Closed-loop powder dispense: shake and weigh until mg_target (mg) is within the protocol's
        tol (mg) or max_tries shakes are used. protocol: a powder_settings.yaml protocol (None: default).
        zero_scale: zero the balance first. Returns the mass dispensed (mg). Use dispense_powder_mg,
        which checks the mass and channel first."""
        from robot_state import powder_settings as settings
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
            self.zero_scale()
            self.c9.delay(protocol.scale_delay)
        tare = self.read_steady_scale() * 1000

        meas_mass = 0
        count = 0
        while mg_togo > protocol.tol and count < max_tries:
            count += 1

            self.set_opening(channel,ps.opening_deg)
            self.shake(shake_t, ps.freq, ps.amplitude)
            if ps.shut_valve:
                self.set_opening(channel,0)
            self.c9.delay(0.5)
            self.read_steady_scale()  # dummy read to wait for steady
            self.c9.delay(protocol.scale_delay)  # delay after steady to allow for more settling time
            meas_mass = self.read_steady_scale() * 1000 - tare

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

            self.logger.info(f'Iteration {count}:')
            self.logger.info(f'\tJust dispensed:  {delta_mass:.1f} mg')
            self.logger.info(f'\tRemaining:       {mg_togo:.1f} mg')
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

    def check_powder_dispense(self, mass_mg, channel=0):
        """ValueError before moving unless mass_mg is > 0 and at most max_mg (powder_limits in
        robot_hardware.yaml) and the channel is listed in powder_channels.yaml."""
        try:
            positive = float(mass_mg) > 0  # also refuses NaN
        except (TypeError, ValueError):
            positive = False
        if not positive:
            raise ValueError(f"Powder dispense refused before moving: mass_mg must be > 0, got {mass_mg!r}")
        max_mg = _hardware_limits(self, 'powder_limits')['max_mg']
        if float(mass_mg) > max_mg:
            raise ValueError(f"Powder dispense refused before moving: mass_mg {mass_mg!r} is above {max_mg} mg "
                             f"(powder_limits in robot_hardware.yaml)")
        channels = self.powder_channels()
        if channel not in channels:
            raise ValueError(f"Powder channel {channel} isn't listed in {self.POWDER_CHANNELS_PATH} "
                             f"(listed: {sorted(channels) or 'none'}); add the slot there first")

    def dispense_powder_mg(self, mass_mg, channel=0):
        """Dispense mass_mg of powder (mg, > 0 and at most 1000: powder_limits in robot_hardware.yaml) from
        channel (listed in powder_channels.yaml) at the carousel. Refused before moving (ValueError)
        otherwise. Returns the mass dispensed (mg; the target in simulation)."""
        self.check_powder_dispense(mass_mg, channel)
        self.logger.info(f"Dispensing {mass_mg} mg on channel {channel}")

        self.activate_powder_channel(channel)
        self.move_carousel_to('powder_dispense_mg')
        #Dispense protocol
        if self.simulate:
            mass_dispensed = mass_mg  # no scale in simulation: assume the target was reached
        else:
            mass_dispensed = self.cl_pow_dispense(mass_mg, channel)
        self.logger.info(f"Mass dispensed: {mass_dispensed} mg")

        self.move_carousel_to('home')
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
    
    def _check_heater_inputs(self, action, target_temp, channel):
        """Refuse a temperature or channel outside heater_limits (robot_hardware.yaml); warn when hot."""
        limits = _hardware_limits(self, 'heater_limits')
        first, last = limits['channels']
        _check_in_range(action, 'channel', channel, first, last, '', 'heater_limits', whole=True)
        _check_in_range(action, 'target_temp', target_temp, limits['min_c'], limits['max_c'], ' C', 'heater_limits')
        if float(target_temp) > limits['warn_above_c']:
            self.logger.warning(f"Heater channel {channel} set to {target_temp} C, above {limits['warn_above_c']} C "
                                f"(heater_limits in robot_hardware.yaml)")

    def autotune(self,target_temp,channel=0):
        """Autotune heater channel (0-7) at target_temp (C, 0-250; heater_limits in robot_hardware.yaml).
        Refused before sending (ValueError) outside those limits; warns above warn_above_c."""
        self._check_heater_inputs('Autotune', target_temp, channel)
        self.logger.debug(f"Autotuning channel {channel} to target temperature {target_temp}C")
        self.t8.disable_channel(channel)
        self.t8.set_temp(channel, target_temp)
        self.t8.enable_channel(channel)
        self.t8.temp_autotune(channel, True)

    def set_temp(self,target_temp,channel=0):
        """Heat channel (0-7) to target_temp (C, 0-250; heater_limits in robot_hardware.yaml).
        Refused before sending (ValueError) outside those limits; warns above warn_above_c."""
        self._check_heater_inputs('Heater', target_temp, channel)
        self.logger.debug(f"Setting channel {channel} to target temperature {target_temp}C")
        self.t8.set_temp(channel, target_temp)
        self.t8.enable_channel(channel)
    
    def get_temp(self,channel=0):
        """Temperature of heater channel (0-7), in C."""
        return self.t8.get_temp(channel)

    def turn_off_heating(self,channel=0):
        """Turn off heater channel (0-7)."""
        self.logger.debug(f"Turning off heating for channel {channel}")
        self.t8.disable_channel(channel)

    def turn_on_stirring(self,speed=10000):
        """Stir at speed (motor counts/s, 1-100000; stirrer_limits in robot_hardware.yaml; default 10000).
        Refused before sending (ValueError) outside those limits; stop with turn_off_stirring."""
        limits = _hardware_limits(self, 'stirrer_limits')
        _check_in_range('Stirring', 'speed', speed, limits['min_speed'], limits['max_speed'], ' cts/s',
                        'stirrer_limits')
        self.logger.debug(f"Turning on stirring at speed {speed}")
        self.c8.spin_axis(1, speed)

    def turn_off_stirring(self):
        """Stop the stirrer (speed 0)."""
        self.logger.debug("Turning off stirring")
        self.c8.spin_axis(1,0)

class North_Spin(North_Base):
    c8 = None
    c9 = None
    def __init__(self,c9,c8, simulate = False, logger=None):
        self.simulate = simulate
        self.c9 = c9
        self.c8 = c8
        self.logger = logger
        self.logger.debug("Initializing centrifuge controller...")
    
    def set_speed(self,speed):
        """Spin the centrifuge at speed (motor counts/s, 1-100000; spinner_limits in robot_hardware.yaml).
        Refused before sending (ValueError) outside those limits; stop with stop_spin."""
        limits = _hardware_limits(self, 'spinner_limits')
        _check_in_range('Spin', 'speed', speed, limits['min_speed'], limits['max_speed'], ' cts/s',
                        'spinner_limits')
        self.logger.debug(f"Setting centrifuge speed to {speed} cts/s")
        self.c8.spin_axis(2, speed)

    def stop_spin(self):
        """Stop the centrifuge (speed 0)."""
        self.logger.debug("Stopping centrifuge spin")
        self.c8.spin_axis(2,0)

    #NOTE: Port 3 is the extender port. 

    def open_lid(self):
        """Open the centrifuge lid (output centrifuge_pneumatics.lid in robot_hardware.yaml)."""
        self.logger.debug("Opening centrifuge lid")
        self.c9.set_output(self._hardware_setting('centrifuge_pneumatics', 'lid', 11), True)  # Open lid pneumatic

    def close_lid(self):
        """Close the centrifuge lid (output centrifuge_pneumatics.lid in robot_hardware.yaml)."""
        self.logger.debug("Closing centrifuge lid")
        self.c9.set_output(self._hardware_setting('centrifuge_pneumatics', 'lid', 11), False)  # Close lid pneumatic

    def turn_on_vacuum(self):
        """Turn on the centrifuge vacuum (output centrifuge_pneumatics.vacuum in robot_hardware.yaml)."""
        self.logger.debug("Turning on centrifuge vacuum")
        self.c9.set_output(self._hardware_setting('centrifuge_pneumatics', 'vacuum', 10), True)  # Turn on vacuum pneumatic

    def turn_off_vacuum(self):
        """Turn off the centrifuge vacuum (output centrifuge_pneumatics.vacuum in robot_hardware.yaml)."""
        self.logger.debug("Turning off centrifuge vacuum")
        self.c9.set_output(self._hardware_setting('centrifuge_pneumatics', 'vacuum', 10), False)  # Turn off vacuum pneumatic

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
    def __init__(self,c9,c8,vial_file=None,simulate=False, logger=None, dispense_recording="off"):

        if dispense_recording not in ("off", "auto"):
            raise ValueError("dispense_recording must be 'off' or 'auto'")
        self.dispense_recording = dispense_recording

        self.c9 = c9
        self.c8 = c8
        self.logger = logger
        self.VIAL_FILE = vial_file #File that we save the vial data in 
        self.simulate = simulate

        self.logger.info("Initializing North Robot...")
        if os.environ.get("NORTH_SCHEDULER_SIMULATION_STATE"):
            from scheduler.simulation_state import configure_controller
            configure_controller(self, "robot")

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

        for rack_name, rack in self.VIAL_POSITIONS.items():
            if type(rack['rack_present']) is not bool:
                raise ValueError(f"rack_present must be true or false: {rack_name}")
        
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
                getattr(self, "_restore_tip_state", lambda saved: None)(robot_status.get('pipet_fluid_contents'))

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
        """Set each pump's volume (mL) and default speed code (0-40) from syringe_pumps.yaml."""
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
        """Arm speed by name from movement_speeds in robot_hardware.yaml (percent of max velocity, 1-100;
        speed_limits in robot_hardware.yaml). ValueError before moving if outside that range."""
        speed = self.get_config_parameter('robot_hardware', 'movement_speeds', speed_name, error_on_missing=True)
        _check_speed_percent(self, 'Move', speed_name, speed, 'movement_speeds in robot_hardware.yaml')
        return speed
    
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
        self._dispense_aspiration_settings = None
        self._dispense_aspiration_source = None
        self.logger.info("Physical initialization of North Robot...")
        self.c9.default_vel = self.get_speed('default_robot')  # Set the default speed of the robot
        try:
            self.open_clamp()  # First real command to the controller after connecting
        except Exception as error:
            # A read timeout here has always meant the controller is switched off. Same error is
            # re-raised; this only adds a clear message and a code for the run's error record.
            if type(error).__name__ == "SerialReadTimeoutException":
                error.error_code = "C9_NOT_RESPONDING"
                self.logger.error("C9 controller did not answer the first command: is it powered on?")
            raise
        self.zero_scale()
        self.open_vial_gripper()  # Ensure gripper is initially open
        self.c9.delay(3)
       
        
        
        
        # Attempt physical cleanup with retry logic for remaining errors
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.load_pumps()
                self._perform_physical_cleanup()
                break  # Success, exit retry loop
                
            except Exception as e:
                if getattr(e, "error_code", None) == "HELD_VIAL_UNKNOWN":
                    raise  # a data problem: retrying and homing can't fix it
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
        
        self._return_stray_vials_at_startup()

        # Final setup
        # Now load pumps after homing
        
        #self.c9.open_gripper()
        self.open_vial_gripper()  

    def _return_stray_vials_at_startup(self):
        """Return vials left away from home (e.g. after a crash or a closed window). Never raises:
        a vial that can't be returned is logged as a warning and left in place."""
        try:
            if not self._vials_available():
                return
            returned, left = self.return_all_vials_home()
            if returned:
                self.logger.warning("Startup cleanup returned vials that were not home: "
                                    + ", ".join(map(str, returned)))
                self.move_home()
            for name, why in left:
                self.logger.warning(f"Startup cleanup left vial {name} away from home ({why})")
        except Exception as error:
            self.logger.warning(f"Startup cleanup could not check vials are home: {type(error).__name__}: {error}")
 
    def _perform_physical_cleanup(self):
        """
        Internal method to perform the actual physical cleanup tasks.
        Separated for cleaner retry logic.
        """
        # Handle leftover liquid in pipet tip
        if self.PIPET_FLUID_VIAL_INDEX is not None and self.PIPET_FLUID_VOLUME > 0:
            self.logger.warning(f"The robot reports having {self.PIPET_FLUID_VOLUME:.3f} mL liquid in its tip... Returning that liquid...")
            # Credit only the recorded liquid volume to the vial. The air blowout
            # clears residual liquid without counting an extra 1 mL as liquid.
            self.dispense_into_vial(
                self.PIPET_FLUID_VIAL_INDEX,
                self.PIPET_FLUID_VOLUME,
                parameters={"blowout_vol": 1.0},
            )
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
                if vial_index is not None:
                    self.return_vial_home(vial_index)
                else:
                    # Recap uses the vial in the clamp; with no saved number it is returned with the other vials
                    self.logger.warning("No vial number saved for the held cap; the recapped clamp vial is returned with the other vials")
            elif self.GRIPPER_STATUS == "Vial":
                self.logger.warning("The robot reports having a vial in the gripper... Returning that vial home...")
                location = self.get_vial_info(vial_index, 'home_location') if vial_index is not None else None
                location_index = self.get_vial_info(vial_index, 'home_location_index') if vial_index is not None else None
                if location is None or location_index is None:
                    error = RuntimeError(
                        f"Gripper status says it holds a vial, but "
                        + ("no vial number is saved" if vial_index is None else f"vial {vial_index} has no home location in the vial file")
                        + ". Check the gripper by hand, then correct robot_status.yaml or the vial file.")
                    error.error_code = "HELD_VIAL_UNKNOWN"
                    raise error
                self.drop_off_vial(vial_index, location=location, location_index=location_index)
        
        # Move to home position
        self.move_home()

    # ====================================================================
    # STATE & STATUS MANAGEMENT
    # ====================================================================

    #Save the status of the robot to memory
    def save_robot_status(self):
        """Save the vial table and robot status (tip, gripper, tip contents) to file (not in simulation, unless the scheduler allows it)."""
        #self.logger.debug("Saving robot status to file: %s", self.ROBOT_STATUS_FILE)
        # Robot status data - ensure all values are Python native types to avoid YAML serialization issues
        robot_status = {
            "gripper_status": self.GRIPPER_STATUS,
            "gripper_vial_index": int(self.GRIPPER_VIAL_INDEX) if self.GRIPPER_VIAL_INDEX is not None else None,
            "held_pipet_type": self.HELD_PIPET_TYPE,
            "pipets_used": self.PIPETS_USED,  # Save all rack usage directly
            "pipet_fluid_vial_index": int(self.PIPET_FLUID_VIAL_INDEX) if self.PIPET_FLUID_VIAL_INDEX is not None else None,
            "pipet_fluid_volume": float(self.PIPET_FLUID_VOLUME) if self.PIPET_FLUID_VOLUME is not None else 0.0,
            # Sample tracking: what the liquid in the tip is and where it came from, so it survives a restart
            "pipet_fluid_contents": getattr(self, "_tip_state_for_saving", lambda: None)(),
        }

        save_state = not self.simulate
        if getattr(self, "_scheduler_state_root", None) is not None:
            from scheduler.simulation_state import can_save
            save_state = can_save(self, self.VIAL_FILE, self.ROBOT_STATUS_FILE)
        if save_state:
            # Writing to a file
            if self.VIAL_FILE is not None:
                self.VIAL_DF.to_csv(self.VIAL_FILE, index=False,sep=',') #Save the status of the vial dataframe
            with open(self.ROBOT_STATUS_FILE, "w") as file:
                yaml.dump(robot_status, file, default_flow_style=False)

    #Update the status of the robot from memory
    def get_robot_status(self):
        """Load the vial table and robot state files."""
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

    # --- Sample tracking (records only; never changes or stops what the robot does) ---------

    def _tip_contents(self):
        tip = getattr(self, "_tip_contents_state", None)
        if tip is None:
            tip = self._tip_contents_state = sample_tracking.TipContents()
        return tip

    def _vial_container_id(self, vial_index):
        """The vial's container_id from the vial file, or None if it has none yet."""
        try:
            if "container_id" not in self.VIAL_DF.columns:
                return None
            value = self.VIAL_DF.at[self.normalize_vial_index(vial_index), "container_id"]
            return value.strip() if isinstance(value, str) and value.strip() else None
        except Exception:
            return None

    def _well_container_id(self, well_index):
        plate_id = getattr(self, "last_plate_id", None)
        return sample_tracking.well_id(plate_id, well_index) if plate_id else None

    # Stored contents travel with the sample (experiment_tracking/sample_contents.py): the tip
    # picks up the source's record on aspirate; a dispense mixes it into the destination's record.

    def _tip_record(self):
        record = getattr(self, "_tip_record_state", None)
        if record is None:
            record = self._tip_record_state = sample_contents.empty_record()
        return record

    def _tip_state_for_saving(self):
        try:
            tip = self._tip_contents()
            if tip.total() <= 0:
                return None
            return {"parts": [[source, amount] for source, amount in tip.parts],
                    "record": sample_contents.to_json(self._tip_record())}
        except Exception:
            return None

    def _restore_tip_state(self, saved):
        try:
            self._tip_contents().clear()
            self._tip_record_state = sample_contents.empty_record()
            if saved:
                self._tip_contents().parts = [[source, float(amount)] for source, amount in saved.get("parts", [])]
                if saved.get("record"):
                    self._tip_record_state = sample_contents.from_json(saved["record"])
        except Exception as error:
            self.logger.warning(f"Sample tracking: could not restore the tip's contents: {error}")

    def _clear_tip_tracking(self):
        self._tip_contents().clear()
        self._tip_record_state = sample_contents.empty_record()

    def _contents_store(self):
        """The stored-contents file for this robot, or None in a plain simulation."""
        path = sample_contents.contents_path(self)
        if path is None:
            return None
        store = getattr(self, "_contents_store_state", None)
        if store is None or store.path != path:
            store = self._contents_store_state = sample_contents.ContentsStore(path)
        return store

    def _vial_stock_id(self, vial_index):
        if "stock_id" not in self.VIAL_DF.columns:
            return None
        value = self.VIAL_DF.at[self.normalize_vial_index(vial_index), "stock_id"]
        return value.strip() if isinstance(value, str) and value.strip() else None

    def _container_record(self, container_id, vial_index=None, volume_mL=None):
        """(record, stored) for a container: its stored record, else its stock, else unknown.
        For a vial, the record is set to the vial file's volume (a hand correction changes only
        the amount, not what's in it)."""
        store = self._contents_store()
        record = store.get(container_id) if (store is not None and container_id) else None
        if record is not None:
            if volume_mL is not None:
                sample_contents.rescale(record, volume_mL)
            return record, True
        stock_id = self._vial_stock_id(vial_index) if vial_index is not None else \
            sample_tracking.source_stocks().get(container_id)
        volume = volume_mL if volume_mL is not None else 0.0
        if stock_id:
            return sample_contents.stock_record(stock_id, sample_tracking.load_stocks(), volume), False
        return sample_contents.unknown_record(volume), False

    def _dispense_error_mL(self, liquid, amount_mL):
        """(random, shared) error of this dispense in mL from the validation history, or None."""
        estimator = getattr(self, "_dispense_errors", None)
        if estimator is None:
            from pipetting_data.dispense_uncertainty import DispenseUncertainty
            estimator = self._dispense_errors = DispenseUncertainty(logger=self.logger)
        found = estimator.lookup(liquid, self.HELD_PIPET_TYPE, float(amount_mL) * 1000)
        return None if found is None else (found[0] / 1000, found[1] / 1000)

    def _track_aspirate(self, source, amount_mL, vial_index=None):
        try:
            self._tip_contents().aspirate(source, amount_mL)
            if vial_index is not None:  # the vial file already has the volume after aspirating
                before = float(self.VIAL_DF.at[self.normalize_vial_index(vial_index), 'vial_volume']) + amount_mL
                record, stored = self._container_record(source, vial_index, before)
            else:  # a well: only its stored record knows its volume
                record, stored = self._container_record(source)
                if not stored:
                    record = sample_contents.unknown_record(amount_mL)
            portion, left = sample_contents.take(record, amount_mL)
            self._tip_record_state = sample_contents.add(self._tip_record(), portion, amount_mL, (0.0, 0.0),
                                                         getattr(self, "run_id", None))
            store = self._contents_store()
            if store is not None and source and stored:
                store.put(source, left)
        except Exception as error:
            self.logger.warning(f"Sample tracking: could not note aspiration from {source}: {error}")

    def _track_dispense(self, destination, amount_mL, liquid=None, vial_index=None):
        try:
            for source, share in self._tip_contents().dispense(amount_mL):
                sample_tracking.record_transfer(self, source, destination, share, "mL", self.HELD_PIPET_TYPE, liquid)
            portion, self._tip_record_state = sample_contents.take(self._tip_record(), amount_mL)
            store = self._contents_store()
            if store is None or not destination:
                return
            if vial_index is not None:  # the vial file already has the volume after dispensing
                before = float(self.VIAL_DF.at[self.normalize_vial_index(vial_index), 'vial_volume']) - amount_mL
                record, stored = self._container_record(destination, vial_index, max(before, 0.0))
            else:
                record, stored = self._container_record(destination)
                if not stored:
                    record = sample_contents.empty_record()  # a well the robot hasn't filled yet
            updated = sample_contents.add(record, portion, amount_mL, self._dispense_error_mL(liquid, amount_mL),
                                          getattr(self, "run_id", None))
            if (not stored and vial_index is not None and record["stocks"]
                    and sample_contents.same_contents(record, updated) and updated["stocks"] == record["stocks"]):
                return  # its own stock went back into a stock vial: it still just holds its stock
            store.put(destination, updated)
        except Exception as error:
            self.logger.warning(f"Sample tracking: could not record dispense into {destination}: {error}")

    # ASSUMPTION (user decision 2026-11-21): reservoir dispenses are treated as exact (no
    # volume error) in the contents records until the reservoir is validated; earlier tests
    # found it very accurate. Set to None to record reservoir error as unknown instead.
    RESERVOIR_ERROR_ML = (0.0, 0.0)

    def _track_reservoir_dispense(self, reservoir_index, vial_index, volume_mL, liquid=None):
        """Reservoir into a vial: logged, and the vial's record updated, with the reservoir's
        error taken as RESERVOIR_ERROR_ML (assumed exact until the reservoir is validated)."""
        try:
            source = sample_tracking.reservoir_id(reservoir_index)
            destination = self._vial_container_id(vial_index)
            sample_tracking.record_transfer(self, source, destination, volume_mL, "mL", "reservoir", liquid)
            store = self._contents_store()
            if store is None or not destination:
                return
            load_record, _ = self._container_record(source, volume_mL=volume_mL)
            portion, _ = sample_contents.take(load_record, volume_mL)
            before = float(self.VIAL_DF.at[self.normalize_vial_index(vial_index), 'vial_volume']) - volume_mL
            record, stored = self._container_record(destination, vial_index, max(before, 0.0))
            updated = sample_contents.add(record, portion, volume_mL, self.RESERVOIR_ERROR_ML,
                                          getattr(self, "run_id", None))
            if not (not stored and record["stocks"] and sample_contents.same_contents(record, updated)
                    and updated["stocks"] == record["stocks"]):
                store.put(destination, updated)
        except Exception as error:
            self.logger.warning(f"Sample tracking: could not record the reservoir dispense: {error}")

    def track_powder_dispense(self, channel, vial, mass_mg):
        """Powder into a vial: logged, and the vial's record updated with mass_mg (mg) of that powder."""
        try:
            source = sample_tracking.source_container(sample_tracking.cartridge_slot(channel))
            destination = self._vial_container_id(vial)
            sample_tracking.record_transfer(self, source, destination, mass_mg, "mg", "powder")
            store = self._contents_store()
            if store is None or not destination:
                return
            stock_id = sample_tracking.source_stocks().get(source)
            materials = [r["material"] for r in sample_tracking.load_stocks() if r["stock_id"] == stock_id]
            volume = float(self.VIAL_DF.at[self.normalize_vial_index(vial), 'vial_volume'])
            record, _ = self._container_record(destination, vial, volume)
            store.put(destination, sample_contents.add_solid(record, materials, float(mass_mg), stock_id))
        except Exception as error:
            self.logger.warning(f"Sample tracking: could not record the powder dispense: {error}")

    #Remove the pipet tip
    def remove_pipet(self):
        """Drop the held tip at its removal location (pipet_tips.yaml); does nothing if no tip is held."""
        self._dispense_aspiration_settings = None
        self._dispense_aspiration_source = None
        if self.HELD_PIPET_TYPE is None:
            self.logger.debug("Skipping pipet removal: no tip held")
            return
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
        self._clear_tip_tracking()  # liquid left in a dropped tip is lost, not transferred
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
            x_total = pickup_movement.get('x', 0)
            y_total = pickup_movement.get('y', 0)
            z_total = pickup_movement.get('z', 0)
            shake_config = self.get_config_parameter('pipet_racks', rack_name, 'pickup_shake', error_on_missing=False)
            if shake_config:
                z_fraction = shake_config.get('z_fraction', 0.5)
                amplitude = shake_config.get('amplitude_mm', 2.0)
                repeats = shake_config.get('repeats', 2)
                vel = self.get_speed('fast_approach')
                # First half of extraction
                self.move_rel_xyz(x_distance=x_total * z_fraction, y_distance=y_total * z_fraction, z_distance=z_total * z_fraction, vel=vel)
                # Shake: push back toward rack (opposite of pickup direction), then return
                shake_x = amplitude if x_total < 0 else (-amplitude if x_total > 0 else 0)
                shake_y = amplitude if y_total < 0 else (-amplitude if y_total > 0 else 0)
                self.logger.info(f"Piggyback shake: {repeats} cycles, amplitude={amplitude}mm, direction=({shake_x:.1f}, {shake_y:.1f})")
                for _ in range(repeats):
                    self.move_rel_xyz(x_distance=shake_x, y_distance=shake_y, vel=vel)
                    self.move_rel_xyz(x_distance=-shake_x, y_distance=-shake_y, vel=vel)
                # Second half of extraction
                self.move_rel_xyz(x_distance=x_total * (1 - z_fraction), y_distance=y_total * (1 - z_fraction), z_distance=z_total * (1 - z_fraction), vel=vel)
            else:
                self.move_rel_xyz(x_distance=x_total, y_distance=y_total, z_distance=z_total, vel=self.get_speed('precise_movement'))
            return
        
        # Check for safe height movement
        if self.get_config_parameter('pipet_racks', rack_name, 'safe_height_movement', error_on_missing=False):
            self.c9.move_z(self.get_safe_height(), vel=self.get_speed('precise_movement'))
            return
        
        # Default: no additional movement
        pass
 
    def adjust_pump_speed(self, pump, pump_speed):
        """Set pump's speed code (1-40; lower is faster). A non-integer is rounded with a warning;
        outside 1-40 pauses with an error (logged as an error in simulation)."""
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
        self._dispense_aspiration_settings = None
        self._dispense_aspiration_source = None
        self.logger.debug(f"Aspirating {amount:.3f} mL then waiting {wait_time} s")

        if amount <= 1:
            try:
                self.c9.aspirate_ml(0, amount)
            except BaseException as error:  # same as the old bare except; now logs the real error
                self.logger.warning(f"Aspirate exceeded limit: Aspirating to maximum ({type(error).__name__}: {error})")
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

        if self.simulate:
            self.c9.dispense_ml(0, amount)
        else:
            pump = self.c9.pumps[0]
            max_pump_position = self.get_config_parameter('pumps', 0, 'max_pump_position')
            if max_pump_position is None:
                raise ValueError("Missing max_pump_position for pipetting pump 0")

            target_position = int(
                pump['pos'] - amount * (max_pump_position / pump['volume'])
            )
            if target_position < 0:
                if target_position < -1:
                    self.logger.warning("Dispense exceeded limit: Dispensing all liquid")
                self.c9.move_pump(0, 0)
            else:
                self.c9.dispense_ml(0, amount)


        if blowout_vol > 0: #Adjust this later into pipetting parameter
            self.logger.debug(f"Blowing out {blowout_vol:.3f} mL")
            
            # Get current speed - should be known from pump initialization
            if 0 not in self.CURRENT_PUMP_SPEEDS:
                self.pause_after_error("Pump 0 speed not initialized - load_pumps() may have failed")
                return
            current_speed = self.CURRENT_PUMP_SPEEDS[0]
            
            # Use faster speed for left valve aspirate (air intake)
            fast_aspirate_speed = 5  # Much faster for air aspirate
            self.adjust_pump_speed(0, fast_aspirate_speed)
            
            self.c9.set_pump_valve(0, self.c9.PUMP_VALVE_LEFT)
            self.c9.aspirate_ml(0, blowout_vol)
            
            # Restore original speed for dispense
            self.adjust_pump_speed(0, current_speed)
            
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
        largest_capacity = max((config.get('volume', 0) for config in self.PIPET_TIPS.values()), default=0)
        for tip_name, tip_config in self.PIPET_TIPS.items():
            max_volume = tip_config.get('volume', 0)
            min_volume = tip_config.get('min_suggested_volume', 0)
            
            # Upper selection boundaries are exclusive, except the largest
            # available tip also accepts its exact maximum capacity.
            if min_volume <= volume and (volume < max_volume or volume == max_volume == largest_capacity):
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
    
    def _get_optimized_parameters(self, volume, liquid=None, user_parameters=None, compensate_overvolume=True, tip_type=None,
                                  require_calibration=False, report=True):
        """
        Get optimized pipetting parameters using intelligent hierarchy:
        defaults â†’ liquid-calibrated â†’ user overrides
        
        Args:
            volume (float): Target pipetting volume in mL
            liquid (str, optional): Liquid type for calibrated parameters
            user_parameters (PipettingParameters, optional): User-provided overrides
            
        Returns:
            PipettingParameters: Optimized parameters with intelligent defaults,
            or None if require_calibration is true and calibration is unavailable.
        """
        # Start with system defaults
        optimized = PipettingParameters()
        
        # Apply liquid-specific calibration if available
        if liquid is not None:
            try:
                wizard = PipettingWizard(logger=self.logger)
                calibrated_params = wizard.get_pipetting_parameters(
                    liquid, volume, compensate_overvolume, tip_type=tip_type, report=report)
                if calibrated_params is None and require_calibration:
                    return None
                
                if calibrated_params is not None:
                    # Update only the parameters that were successfully calibrated
                    for param_name, param_value in calibrated_params.items():
                        if hasattr(optimized, param_name) and param_value is not None:
                            setattr(optimized, param_name, param_value)
                            self.logger.debug(f"Using calibrated {param_name}={param_value} for {liquid} at {volume:.3f}mL")
                            
            except Exception as e:
                if tip_type is not None:
                    raise
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
                        
        self.logger.debug("Liquid review context: " + json.dumps({
            "liquid": liquid, "target_volume_ml": volume,
        }))
        _check_parameter_limits(self, 'Pipetting', optimized)
        return optimized

    def resolve_pipetting_setup(self, volume, liquid=None, user_parameters=None,
                                specified_tip=None, compensate_overvolume=True):
        """Resolve a tip and parameters together before aspirating from a vial.

        Calibrated tips are tried in order of how well their calibrated range covers the
        volume (inside, near, further out), smaller tip first. A tip whose load comes
        within its capacity_margin is a tight fit, used only when no calibrated tip fits
        with room. Range and margin are preferences: they choose between tips, never
        reject a volume that today's capacity check would accept.
        """
        wizard = PipettingWizard(logger=self.logger)
        tip_calibration = liquid is not None and bool(wizard.find_calibration_files(liquid))
        candidates = ([specified_tip] if specified_tip else
                      sorted(self.PIPET_TIPS, key=lambda name:
                             self.PIPET_TIPS[name].get('volume', 0)))
        failures = []
        fallback_tips = []
        tight_fits = []
        ranked = []
        for size_order, tip in enumerate(candidates):
            if tip not in self.PIPET_TIPS:
                raise ValueError(f"Unknown pipet tip: {tip}")
            config = self.PIPET_TIPS[tip]
            if volume < config.get('min_suggested_volume', 0):
                failures.append(f"{tip}: below minimum suggested volume")
                continue
            if volume >= config['volume']:
                failures.append(f"{tip}: target reaches tip capacity")
                continue
            fraction = wizard.extrapolation_fraction(liquid, volume, tip) if tip_calibration else 0.0
            coverage = (3 if fraction is None else 0 if fraction == 0
                        else 1 if fraction <= NEAR_EXTRAPOLATION_FRACTION else 2)
            ranked.append((coverage, size_order, tip))
        for _, _, tip in sorted(ranked):
            config = self.PIPET_TIPS[tip]
            try:
                params = self._get_optimized_parameters(
                    volume, liquid, user_parameters, compensate_overvolume,
                    tip_type=tip if tip_calibration else None,
                    require_calibration=tip_calibration, report=False)
            except ValueError as error:
                failures.append(f"{tip}: {error}")
                continue
            if params is None:
                fallback_tips.append(tip)
                failures.append(f"{tip}: calibration unavailable")
                continue
            tip_load = volume + max(0, params.overaspirate_vol) + max(0, params.post_asp_air_vol)
            if tip_load >= config['volume']:
                failures.append(f"{tip}: {tip_load * 1000:.1f} uL reaches tip capacity")
                continue
            if tip_load <= config['volume'] - config.get('capacity_margin', 0):
                if tip_calibration:
                    wizard.report_extrapolation(liquid, volume, tip)
                return params, tip
            self.logger.info("%s load %.1f uL is within its %.0f uL capacity margin; "
                             "trying other tips", tip, tip_load * 1000,
                             config.get('capacity_margin', 0) * 1000)
            tight_fits.append((params, tip, tip_load))
        if tight_fits:
            params, tip, tip_load = tight_fits[0]
            self.logger.warning(
                "No tip fits %s at %g uL with room; using %s with load %.1f uL of %.0f uL",
                liquid, volume * 1000, tip, tip_load * 1000, self.PIPET_TIPS[tip]['volume'] * 1000,
            )
            if tip_calibration:
                wizard.report_extrapolation(liquid, volume, tip)
            return params, tip
        # Prefer any calibrated tip that fits before falling back to defaults.
        # Only unavailable calibration permits fallback; capacity failures do not.
        for tip in sorted(fallback_tips, key=candidates.index):
            params = self._get_optimized_parameters(
                volume, None, user_parameters, compensate_overvolume)
            tip_load = volume + max(0, params.overaspirate_vol) + max(0, params.post_asp_air_vol)
            if tip_load >= self.PIPET_TIPS[tip]['volume']:
                failures.append(f"{tip}: default tip load {tip_load * 1000:.1f} uL reaches tip capacity")
                continue
            self.logger.warning(
                "No calibrated tip fits %s at %g uL; using default pipetting parameters "
                "with %s (user overrides retained). Dispensing accuracy is unvalidated.",
                liquid, volume * 1000, tip,
            )
            return params, tip
        raise ValueError(f"No tip fits {liquid} at {volume * 1000:g} uL: " +
                         '; '.join(failures))
    
    def _get_optimized_reservoir_parameters(self, volume, liquid=None, reservoir_params=None):
        """
        Get optimized reservoir parameters using intelligent hierarchy:
        defaults â†’ liquid-calibrated â†’ user overrides
        
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

        _check_parameter_limits(self, 'Reservoir dispense', optimized)
        return optimized

    #Check if the aspiration volume is within limits... Now extensible and configuration-driven
    def check_if_aspiration_volume_unacceptable(self, amount_mL):
        """Pause with an error if amount_mL (mL) is above the held tip's volume or below its
        min_suggested_volume (pipet_tips.yaml). Returns True if there was an issue."""
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
        """Aspirate height (mm) for amount_mL (mL) from the vial: the location's min_pipetting_height,
        raised by height_volume_constant (mm per mL, vial_positions.yaml) for the liquid left minus buffer (mL)
        when track_height. Never below the minimum."""

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
                    self.logger.debug(f"Clamp occupied by vial {clamp_vial_index}, moving it out first")
                    self._return_from_clamp(clamp_vial_index, move_speed=move_speed)
                
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
                          move_to_aspirate=True, track_height=True, move_up=True, specified_tip=None, use_safe_location=False, post_asp_shake=False, 
                          continuous_mass_monitoring=False, save_mass_data=False):
        """
        Aspirate amount_mL from a source vial, getting ready first (combined unit operation).

        Gets the right tip and makes the vial accessible (moves it to the clamp and uncaps it if
        needed), then runs the liquid step (aspirate_from_open_vial) and, with scale recording,
        analyzes the mass (pipetting_data/mass_analysis.py).
        
        Args:
            source_vial_name (str): Name of the source vial to aspirate from
            amount_mL (float): Amount to aspirate in mL
            parameters (PipettingParameters, optional): Liquid handling parameters (uses defaults if None)
            liquid (str, optional): Liquid type for calibrated parameter optimization
            move_to_aspirate (bool): Whether to move to aspiration location (default: True)
            track_height (bool): Whether to track liquid height during aspiration (default: True)
            move_up (bool): Whether to retract after aspiration (default: True)
            specified_tip (str, optional): Force specific tip type ("small_tip" or "large_tip")
            use_safe_location (bool): Use safe location for vial access (default: False)  
            post_asp_shake (bool): Apply post-aspiration shaking (default: False)
            continuous_mass_monitoring (bool): Whether to continuously monitor mass during aspiration (default: False)
            save_mass_data (bool): Whether to save continuous mass data to file (default: False)
        
        Returns:
            tuple: (measured_mass, stability_info) if continuous_mass_monitoring=True, otherwise None
        """
        self._dispense_aspiration_settings = None
        self._dispense_aspiration_source = None
        # Use intelligent parameter resolution: defaults â†’ liquid-calibrated â†’ user overrides
        parameters, resolved_tip = self.resolve_pipetting_setup(
            amount_mL, liquid, parameters, specified_tip)

        source_vial_num = self.normalize_vial_index(source_vial_name) #Convert to int if needed     

        # Check if source vial has a closed cap that will require uncapping
        source_vial_needs_uncapping = not self.is_vial_pipetable(source_vial_num)

        required_tip_type = resolved_tip

        if source_vial_needs_uncapping:
            # Get pipet tip FIRST to avoid cap-rack interference when uncapping
            
            self.get_tip_if_needed(required_tip_type)
            
            # Then ensure accessibility (will uncap and hold cap in gripper)
            if not self._ensure_vial_accessible_for_pipetting(source_vial_name, use_safe_location):
                return
        else:
            # Normal sequence: accessibility first, then pipet tip (for uncapped or open caps)
            if not self._ensure_vial_accessible_for_pipetting(source_vial_name, use_safe_location):
                return
                
            #Get a tip if we need one
            self.get_tip_if_needed(required_tip_type)

        # Small tip position/volume safety warnings
        if required_tip_type == 'small_tip':
            _vial_loc = self.get_vial_info(source_vial_num, 'location')
            _vial_loc_idx = int(self.get_vial_info(source_vial_num, 'location_index'))
            # Safe positions for small-tip aspiration: small_tip_safe_indices in vial_positions.yaml
            if _vial_loc_idx not in self.get_small_tip_safe_indices(_vial_loc):
                self.logger.warning(f"WARNING: Aspirating from {_vial_loc}[{_vial_loc_idx}] with a small tip may cause an automation issue")
            else:
                _remaining_vol = self.get_vial_info(source_vial_num, 'vial_volume')
                if _remaining_vol is not None and float(_remaining_vol) < 2.0:
                    self.logger.warning(f"WARNING: Small tip aspiration of {amount_mL:.3f} mL from {_vial_loc}[{_vial_loc_idx}] - remaining volume low ({float(_remaining_vol)*1000:.0f}uL)")

        # The liquid step (same as aspirate_from_open_vial, with the parameters already resolved)
        continuous_mass_data, source_vial_clamped = self._aspirate_open_vial(
            source_vial_name, source_vial_num, amount_mL, parameters, move_to_aspirate, track_height,
            move_up, post_asp_shake, continuous_mass_monitoring)

        measured_mass = None
        stability_info = empty_stability()
        # Analyze continuous mass data after monitoring is stopped (pipetting_data/mass_analysis.py)
        if continuous_mass_monitoring and source_vial_clamped and continuous_mass_data:
            source_name = self.get_vial_info(source_vial_num, 'vial_name')
            measured_mass = self._analyze_mass_readings(
                continuous_mass_data, 'aspiration', source_name, amount_mL, stability_info,
                save=save_mass_data, file_prefix="aspiration_mass_", file_stem=source_name)

        # Return mass monitoring results if enabled, otherwise return None for backwards compatibility
        if continuous_mass_monitoring and source_vial_clamped:
            return measured_mass, stability_info
        else:
            return None
    
    def aspirate_from_open_vial(self, vial_name, amount_mL, parameters=None, liquid=None, specified_tip=None,
                                move_to_aspirate=True, track_height=True, move_up=True, record_scale=False):
        """
        Unit operation: aspirate amount_mL from a vial that is already open and reachable, with the
        right tip already held. Does not get a tip or move the vial (aspirate_from_vial does that).

        Args:
            vial_name (str or int): vial to aspirate from; must be pipetable (open, or uncapped in the clamp)
            amount_mL (float): mL to aspirate, at most what the vial holds
            parameters (PipettingParameters, optional): overrides (None: calibrated for liquid, else defaults)
            liquid (str, optional): liquids.yaml name for calibrated parameters
            specified_tip (str, optional): tip type to use instead of the one chosen for the volume
            move_to_aspirate, track_height, move_up: as in aspirate_from_vial
            record_scale (bool): record the balance around the aspirate (vial in the clamp only)

        Refused before moving (ValueError): vial not pipetable, or the held tip is not the tip this
        volume needs. Returns the scale readings (empty unless recorded); analyze them with
        pipetting_data/mass_analysis.py.
        """
        self._dispense_aspiration_settings = None
        self._dispense_aspiration_source = None
        parameters, resolved_tip = self.resolve_pipetting_setup(amount_mL, liquid, parameters, specified_tip)
        vial_num = self.normalize_vial_index(vial_name)
        if not self.is_vial_pipetable(vial_num):
            raise ValueError(f"Aspirate refused before moving: vial {vial_name} is not open for pipetting "
                             f"(use aspirate_from_vial to uncap it first)")
        if self.HELD_PIPET_TYPE != resolved_tip:
            raise ValueError(f"Aspirate refused before moving: {amount_mL:.3f} mL needs {resolved_tip}, "
                             f"holding {self.HELD_PIPET_TYPE} (use aspirate_from_vial to get it)")
        readings, _ = self._aspirate_open_vial(vial_name, vial_num, amount_mL, parameters, move_to_aspirate,
                                               track_height, move_up, False, record_scale)
        return readings

    def _aspirate_open_vial(self, source_vial_name, source_vial_num, amount_mL, parameters, move_to_aspirate,
                            track_height, move_up, post_asp_shake, continuous_mass_monitoring):
        """The liquid step of an aspirate, with parameters already resolved and the tip and vial ready.
        Returns (scale readings, whether the vial was in the clamp)."""
        asp_disp_cycles = parameters.asp_disp_cycles
        aspirate_wait_time = parameters.aspirate_wait_time
        pre_asp_air_vol = parameters.pre_asp_air_vol
        post_asp_air_vol = parameters.post_asp_air_vol
        overaspirate_vol = parameters.overaspirate_vol
        post_retract_wait_time = parameters.post_retract_wait_time
        # Pre-aspiration air stays upstream; conservatively include liquid,
        # positive overaspirate, and post-aspiration air in the tip check.
        total_tip_vol = amount_mL + max(0, overaspirate_vol) + max(0, post_asp_air_vol)

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
            self.pause_after_error(
                f"Cannot aspirate more volume than in vial: {amount_mL:.3f} mL requested from "
                f"'{source_vial_name}' ({location}), which has {source_vial_volume:.3f} mL")

        source_vial_clamped = self.get_vial_info(source_vial_num,'location')=='clamp' #Is the source vial clamped?

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
            
        # Scale recording: only with the vial in the clamp; nothing in simulation
        scale = self.start_scale_recording(continuous_mass_monitoring and source_vial_clamped, 'aspiration')

        #Step 1: Move to above the site and aspirate air if needed
        if pre_asp_air_vol > 0:
            self.c9.move_z(disp_height)
            self.pipet_aspirate(pre_asp_air_vol, wait_time=0)
        
        self.adjust_pump_speed(0,aspirate_speed) #Adjust the pump speed if needed

        scale.mark_start()

        #Step 2: Move to inside the site and aspirate liquid
        self.c9.move_z(asp_height)
        for i in range (0, asp_disp_cycles):
            self.pipet_aspirate(amount_mL+overaspirate_vol, wait_time=0)
            self.pipet_dispense(amount_mL+overaspirate_vol, wait_time=0)
        self.pipet_aspirate(amount_mL+overaspirate_vol, wait_time=aspirate_wait_time) #Main aspiration of liquid plus wait
        
        scale.mark_end()

        #Step 3: Retract and aspirate air if needed
        if post_asp_shake:
            None
            # self.move_rel_xyz(z_distance=50)
            # self.move_rel_xyz(x_distance=2)
            # self.move_rel_xyz(x_distance=-2)
            # self.move_rel_xyz(y_distance=2)
            # self.move_rel_xyz(y_distance=-2)
        if move_up:
            self.c9.move_z(disp_height, vel=retract_speed) #Retract with a specific speed
            if not self.simulate:
                time.sleep(post_retract_wait_time) #Wait after retracting
        if post_asp_air_vol > 0:
            self.pipet_aspirate(post_asp_air_vol, wait_time=0) 

        readings = scale.stop()

        #Record the volume change
        self.VIAL_DF.at[source_vial_num,'vial_volume']=(source_vial_volume-amount_mL)

        #Update the new volume in memory
        self.PIPET_FLUID_VIAL_INDEX = int(source_vial_num)
        self.PIPET_FLUID_VOLUME += amount_mL
        self._track_aspirate(self._vial_container_id(source_vial_num), amount_mL, vial_index=source_vial_num)
        self.save_robot_status()

        # Observation only: retain the settings used, rather than looking them up again.
        if getattr(self, 'dispense_recording', 'off') == 'auto':
            capture_aspiration(self, parameters, source_vial_num, aspirate_speed, retract_speed)

        return readings, source_vial_clamped

    #Adjust height based on the currently held pipet tip - now extensible to all tip types
    def adjust_height_based_on_pipet_held(self, height):
        """height (mm) plus the held tip's delta_z_to_tip_bottom (mm, pipet_tips.yaml)."""
        height_shift_pipet = 0
        
        if self.HELD_PIPET_TYPE is not None:
            # Get delta_z_to_tip_bottom for the currently held pipet type
            height_shift_pipet = self.get_config_parameter('pipet_tips', self.HELD_PIPET_TYPE, 'delta_z_to_tip_bottom', error_on_missing=False) or 0
            
        height += height_shift_pipet
        return height

    #This method dispenses from a vial into another vial, using buffer transfer to improve accuracy if needed.
    #TODO: Maybe get rid of the buffer option here and replace with the other new parameters and potentially blowout
    def plan_vial_transfer(self, volume, liquid=None, parameters=None, specified_tip=None,
                           compensate_overvolume=True):
        """Resolve a vial transfer into equal executable chunks without moving hardware.

        Return chunk volume (mL), repeat count, parameters and tip identity.
        Validation and dispensing share this capacity/calibration calculation.
        """
        if not math.isfinite(volume) or volume <= 0:
            raise ValueError("Transfer volume must be finite and positive")
        max_system_volume = (self.PIPET_TIPS[specified_tip]['volume'] if specified_tip
                             else max(tip['volume'] for tip in self.PIPET_TIPS.values()))
        if max_system_volume <= 0:
            raise ValueError("Pipet tip capacity must be positive")
        transfer_limit = max_system_volume
        if liquid is not None:
            calibrated_max = PipettingWizard(logger=self.logger).get_max_calibrated_volume(liquid)
            if calibrated_max is not None:
                transfer_limit = min(transfer_limit, calibrated_max)
        repeats = max(1, math.ceil(volume / transfer_limit))
        for _ in range(10):
            sub_volume = volume / repeats
            try:
                resolved_parameters, resolved_tip = self.resolve_pipetting_setup(
                    sub_volume, liquid, parameters, specified_tip, compensate_overvolume)
            except ValueError:
                repeats += 1
                continue
            tip_load = (max(sub_volume, round(sub_volume, 3)) +
                        resolved_parameters.overaspirate_vol + resolved_parameters.post_asp_air_vol)
            if tip_load < self.PIPET_TIPS[resolved_tip]['volume']:
                return sub_volume, repeats, resolved_parameters, resolved_tip
            repeats += 1
        raise ValueError(f"Cannot split {volume:.3f} mL into transfers within pipet tip capacity "
                         f"({max_system_volume:.3f} mL) with the resolved pipetting parameters")

    def dispense_from_vial_into_vial(self, source_vial_name, dest_vial_name, volume, parameters=None, liquid=None, specified_tip=None, 
                                     remove_tip=True, use_safe_location=False, return_vial_home=True, move_speed=None, 
                                      compensate_overvolume=True, measure_weight=False,
                                     continuous_mass_monitoring=False, save_mass_data=False, record_measurement=True):
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
            move_speed (float, optional): speed of the vial moves and dispense approach, percent of max velocity, 1-100 (None: the default speed); refused before moving outside that range
        """
        _check_optional_speed(self, 'dispense_from_vial_into_vial', 'move_speed', move_speed)
        self.logger.info(f"Dispensing {volume:.3f} mL from {source_vial_name} to {dest_vial_name}")

        source_vial_index = self.normalize_vial_index(source_vial_name)
        dest_vial_index = self.normalize_vial_index(dest_vial_name)

        if volume <= 0:
            self.logger.warning("Cannot dispense <=0 mL")
            return

        sub_volume, repeats, parameters, resolved_tip = self.plan_vial_transfer(
            volume, liquid, parameters, specified_tip, compensate_overvolume)

        if repeats > 1:
            self.logger.info(f"Splitting {volume:.3f} mL into {repeats} transfers of {sub_volume:.3f} mL each to fit transfer limits")
        volume = sub_volume

        # Stage before a tip is picked up: closed vials to rack 0-5, so they can swap through the clamp
        # with a tip held (a staged vial leaving the clamp goes back to its staged spot); an open
        # small-tip source to a small-tip-safe position.
        to_stage = [vial for vial, uses_small_tip in ((source_vial_name, resolved_tip == 'small_tip'),
                                                      (dest_vial_name, False))
                    if self._needs_staging(vial, uses_small_tip)]
        staged = {}
        if to_stage:
            if self.HELD_PIPET_TYPE is not None:
                self.remove_pipet()
            staged = self.stage_vials(to_stage)

        # Condition a fresh tip first if this liquid needs it (condition_tip in liquids.yaml)
        if liquid_conditions_tip(liquid) and self.HELD_PIPET_TYPE != resolved_tip:
            self.condition_tip(source_vial_index, volume_per_cycle_mL=volume, specified_tip=resolved_tip)

        total_mass = 0
        for i in range(repeats):
            last_run = (i == repeats - 1)

            # Aspirate from source (at normal speed)
            # Debit and credit the same volume, including fractional microlitres.
            self.aspirate_from_vial(source_vial_index, volume, parameters=parameters,
                                   liquid=liquid, specified_tip=resolved_tip, use_safe_location=use_safe_location)

            # Set custom movement speed if specified - after aspiration when carrying liquid
            original_vel = None
            if move_speed is not None:
                original_vel = self.c9.default_vel
                self.c9.default_vel = move_speed

            # Dispense into destination (at modified speed)
            dispense_result = self.dispense_into_vial(dest_vial_index, volume, parameters=parameters, liquid=liquid, move_speed=move_speed, measure_weight=measure_weight,
                                                      continuous_mass_monitoring=continuous_mass_monitoring, save_mass_data=save_mass_data,
                                                      record_measurement=record_measurement)
            
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
                if staged and return_vial_home:
                    self.return_staged_vials(staged.keys())


        return total_mass, stability_info

    #TODO add error checks and safeguards
    def pipet_from_wellplate(self, wp_index, volume, parameters=None, liquid=None, aspirate=True, specified_tip=None, move_to_aspirate=True, well_plate_type=None):
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

        volume must be a number > 0 mL (ValueError before moving).
        """
        try:
            positive = float(volume) > 0  # also refuses NaN
        except (TypeError, ValueError):
            positive = False
        if not positive:
            raise ValueError(f"Wellplate {'aspirate' if aspirate else 'dispense'} refused before moving: "
                             f"volume must be > 0 mL, got {volume!r}")
        well_plate_type = self._resolve_pipetting_plate(well_plate_type, [wp_index])
        # Use intelligent parameter resolution: defaults â†’ liquid-calibrated â†’ user overrides
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
            self._track_aspirate(self._well_container_id(wp_index), volume)
        else:
            self.pipet_dispense(overdispense_vol, wait_time=wait_time, blowout_vol=blowout_vol)
            self._track_dispense(self._well_container_id(wp_index), volume, liquid)
        self.save_robot_status()  # keeps the tip's tracked contents across a restart

    #Mix the well
    def _check_mix_inputs(self, volume, repeats):
        """ValueError before moving unless volume is > 0 mL and repeats a whole number >= 1; returns int(repeats)."""
        try:
            ok = float(volume) > 0 and float(repeats) >= 1 and float(repeats) == int(repeats)
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:  # also refuses NaN
            raise ValueError(f"Mix refused before moving: need volume > 0 mL and a whole number of repeats >= 1, "
                             f"got volume={volume!r}, repeats={repeats!r}")
        return int(repeats)

    def mix_well_in_wellplate(self, wp_index, volume, repeats=3, well_plate_type=None, parameters=None, liquid=None):
        """
        Mix a well in a wellplate by aspirating and dispensing repeatedly.
        
        Args:
            wp_index: Well plate index to mix
            volume: Volume to use for mixing in mL
            repeats: Number of mixing cycles (default: 3)
            well_plate_type: Type of well plate (default: "96 WELL PLATE")
            parameters (PipettingParameters, optional): Liquid handling parameters (uses defaults if None)
            liquid (str, optional): Liquid type for calibrated parameter optimization

        volume must be > 0 mL and repeats a whole number >= 1 (ValueError before moving).
        """
        repeats = self._check_mix_inputs(volume, repeats)
        self.logger.info(f"Mixing well {wp_index} in well plate of type {well_plate_type} with volume {volume:.3f} mL for {repeats} repeats")
        self.pipet_from_wellplate(wp_index, volume, parameters=parameters, liquid=liquid, well_plate_type=well_plate_type)
        self.pipet_from_wellplate(wp_index, volume, parameters=parameters, liquid=liquid, aspirate=False, move_to_aspirate=False, well_plate_type=well_plate_type)
        for i in range(1, repeats):
            self.pipet_from_wellplate(wp_index, volume, parameters=parameters, liquid=liquid, move_to_aspirate=False, well_plate_type=well_plate_type)
            self.pipet_from_wellplate(wp_index, volume, parameters=parameters, liquid=liquid, aspirate=False, move_to_aspirate=False, well_plate_type=well_plate_type)

    #Mix in a vial
    def mix_vial(self, vial_name, volume, repeats=3, parameters=None, liquid=None):
        """
        Combined unit operation: mix a vial by pipetting up and down with one tip, the tip kept low
        in the liquid (like mix_well_in_wellplate).

        Gets the right tip and makes the vial accessible (aspirate_from_vial: moves it to the clamp
        and uncaps it if needed), then repeats aspirate_from_open_vial / dispense_into_open_vial in
        place. No dispense-recording rows.

        Args:
            vial_name: Name or index of the vial to mix
            volume (float): mL drawn up each cycle, > 0 (must fit the tip and be in the vial)
            repeats (int): number of up-and-down cycles, >= 1 (default: 3)
            parameters (PipettingParameters, optional): Liquid handling parameters (uses defaults if None)
            liquid (str, optional): Liquid type for calibrated parameter optimization

        Refused before moving (ValueError): volume not > 0, or repeats not a whole number >= 1.
        State after: tip still attached (empty); vial open where it was mixed (may be the clamp).
        """
        repeats = self._check_mix_inputs(volume, repeats)
        self.logger.info(f"Mixing vial {vial_name} with volume {volume:.3f} mL for {repeats} repeats")
        vial_index = self.normalize_vial_index(vial_name)
        self.aspirate_from_vial(vial_index, volume, parameters=parameters, liquid=liquid, move_up=False, track_height=False)
        self.dispense_into_open_vial(vial_index, volume, parameters=parameters, liquid=liquid, initial_move=False)
        for i in range(1, repeats):
            self.aspirate_from_open_vial(vial_index, volume, parameters=parameters, liquid=liquid,
                                         move_to_aspirate=False, track_height=False, move_up=False)
            self.dispense_into_open_vial(vial_index, volume, parameters=parameters, liquid=liquid, initial_move=False)

    #Dispense an amount into a vial
    def dispense_into_vial(self, dest_vial_name, amount_mL, parameters=None, liquid=None,
                          initial_move=True, measure_weight=False, continuous_mass_monitoring=False,
                          save_mass_data=False, move_speed=None, record_measurement=True):
        """
        Dispense liquid into a vial, getting ready first (combined unit operation).

        Makes the vial accessible (moves it to the clamp and uncaps it if needed), checks it has
        room, then runs the liquid step (dispense_into_open_vial). Around it: optional before/after
        weighing, scale recording with mass analysis (pipetting_data/mass_analysis.py), and the
        dispense-recording row when dispense_recording is 'auto'.

        Args:
            dest_vial_name: Name of destination vial
            amount_mL: Amount to dispense in mL
            parameters (PipettingParameters, optional): Liquid handling parameters (uses defaults if None)
            liquid (str, optional): Liquid type for calibrated parameter optimization
            initial_move (bool): Whether to perform initial movement (default: True)
            measure_weight (bool): Whether to measure mass before/after dispensing (default: False)
            continuous_mass_monitoring (bool): Whether to continuously monitor mass during dispensing (default: False)
            save_mass_data (bool): Whether to save continuous mass data to file (default: False)
            move_speed (float, optional): speed of the move to the vial, percent of max velocity, 1-100 (None: the default speed); refused before moving outside that range
        """
        _check_optional_speed(self, 'dispense_into_vial', 'move_speed', move_speed)
        # Use intelligent parameter resolution: defaults â†’ liquid-calibrated â†’ user overrides
        parameters = self._get_optimized_parameters(amount_mL, liquid, parameters)

        dest_vial_num = self.normalize_vial_index(dest_vial_name) #Convert to int if needed

        # Ensure vial is accessible for pipetting (no use_safe_location for dispense)
        if not self._ensure_vial_accessible_for_pipetting(dest_vial_name, use_safe_location=False, move_speed=move_speed):
            return

        measured_mass = None
        continuous_mass_data = []
        stability_info = empty_stability()

        self.logger.info(f"Pipetting into vial {self.get_vial_info(dest_vial_num, 'vial_name')}, amount: {amount_mL:.3f} mL")

        dest_vial_clamped = self.get_vial_info(dest_vial_num,'location')=='clamp' #Is the destination vial clamped?

        # Check if dispensing would exceed vial capacity
        too_full = self._vial_overfill_message(dest_vial_num, amount_mL)
        if too_full:
            self.pause_after_error(too_full)
            return

        # Recording opts into the existing weighing path; it never prepares a vial.
        original_result = (measured_mass, stability_info.copy())
        density = recording_density(self, liquid, enabled=record_measurement and amount_mL > 0)
        if density is not None and (not dest_vial_clamped
                                   or self.get_vial_info(dest_vial_num, 'location_index') != 0):
            density = None
        routine_only = density is not None and not measure_weight
        if routine_only:
            measure_weight = True
            continuous_mass_monitoring = True

        #If the destination vial is at the clamp and you want weight measurement using TRADITIONAL method
        if measure_weight and not continuous_mass_monitoring and dest_vial_clamped:
            if not self.simulate:
                # Zero the scale before measurement to prevent baseline drift
                initial_mass = self.read_steady_scale()
                self.logger.info(f"Initial mass reading (traditional): {initial_mass:.6f} g")
            else:
                initial_mass = 0
                self.logger.info("Simulation mode - initial mass set to 0")
        else:
            initial_mass = None  # Will be handled by continuous monitoring if needed

        # The liquid step (same as dispense_into_open_vial, with the parameters already resolved)
        continuous_mass_data, dispense_speed = self._dispense_open_vial(
            dest_vial_num, amount_mL, parameters, liquid, initial_move, move_speed,
            measure_weight and continuous_mass_monitoring and dest_vial_clamped)

        # Analyze continuous mass data after monitoring is stopped (pipetting_data/mass_analysis.py)
        if measure_weight and continuous_mass_monitoring and dest_vial_clamped and continuous_mass_data:
            measured_mass = self._analyze_mass_readings(
                continuous_mass_data, 'dispense', dest_vial_name, amount_mL, stability_info,
                save=save_mass_data, file_prefix="mass_", file_stem=dest_vial_name)

        # Traditional before/after measurement (only when continuous monitoring is NOT enabled)
        elif measure_weight and not continuous_mass_monitoring and dest_vial_clamped:
            if not self.simulate:
                final_mass = self.read_steady_scale()
                self.logger.info(f"Final mass reading: {final_mass:.6f} g")
            else:
                final_mass = 0
                self.logger.info("Simulation mode - final mass set to 0")
            measured_mass = final_mass - initial_mass  
            self.logger.info(f"Mass difference: {measured_mass:.6f} g (target: {amount_mL:.3f} mL = ~{amount_mL*1.0:.6f} g for water)")
        if density is not None:
            # Existing weighing uses zero for insufficient data. Preserve that
            # caller contract, but do not store it as an observed zero mass.
            recorded_mass = measured_mass
            if continuous_mass_monitoring and (len(continuous_mass_data) < 4
                    or not stability_info['pre_total_count']
                    or not stability_info['post_total_count']):
                recorded_mass = None
            if 'failure_reason' in stability_info:
                recorded_mass = None
            record_dispense(self, liquid=liquid, density=density, target=amount_mL,
                mass=recorded_mass, stability=stability_info, parameters=parameters,
                destination=dest_vial_num, dispense_speed=dispense_speed)

        if routine_only:
            return original_result
        return measured_mass, stability_info

    def dispense_into_open_vial(self, vial_name, amount_mL, parameters=None, liquid=None,
                                initial_move=True, move_speed=None, record_scale=False):
        """
        Unit operation: dispense amount_mL from the held tip into a vial that is already open and
        reachable. Does not move or uncap the vial (dispense_into_vial does that) and does not write
        a dispense-recording row.

        Args:
            vial_name (str or int): vial to dispense into; must be pipetable (open, or uncapped in the clamp)
            amount_mL (float): mL to dispense; the vial must have room (vial_volume in vial_positions.yaml)
            parameters (PipettingParameters, optional): overrides (None: calibrated for liquid, else defaults)
            liquid (str, optional): liquids.yaml name for calibrated parameters and the +/- uncertainty
            initial_move (bool): move to the vial first (default True)
            move_speed (float, optional): speed for that move (None: standard speeds)
            record_scale (bool): record the balance around the dispense (vial in the clamp only)

        Refused before moving (ValueError): vial not pipetable, no tip held, or not enough room.
        Returns the scale readings (empty unless recorded); analyze them with
        pipetting_data/mass_analysis.py.
        """
        _check_optional_speed(self, 'dispense_into_open_vial', 'move_speed', move_speed)
        parameters = self._get_optimized_parameters(amount_mL, liquid, parameters)
        vial_num = self.normalize_vial_index(vial_name)
        if not self.is_vial_pipetable(vial_num):
            raise ValueError(f"Dispense refused before moving: vial {vial_name} is not open for pipetting "
                             f"(use dispense_into_vial to uncap it first)")
        if self.HELD_PIPET_TYPE is None:
            raise ValueError("Dispense refused before moving: no tip held")
        too_full = self._vial_overfill_message(vial_num, amount_mL)
        if too_full:
            raise ValueError(f"Dispense refused before moving: {too_full}")
        clamped = self.get_vial_info(vial_num, 'location') == 'clamp'
        readings, _ = self._dispense_open_vial(vial_num, amount_mL, parameters, liquid, initial_move, move_speed,
                                               record_scale and clamped)
        return readings

    def _vial_overfill_message(self, dest_vial_num, amount_mL):
        """Why amount_mL would overfill the vial (its location's vial_volume), or None if it fits."""
        dest_vial_volume = self.get_vial_info(dest_vial_num,'vial_volume') #What is the current vial volume?
        dest_vial_location = self.get_vial_info(dest_vial_num, 'location')
        max_vial_volume = self.get_config_parameter('vial_positions', dest_vial_location, 'vial_volume', error_on_missing=False)
        if max_vial_volume is not None:
            post_dispense_volume = dest_vial_volume + amount_mL
            if (post_dispense_volume > max_vial_volume
                    and not math.isclose(post_dispense_volume, max_vial_volume,
                                         rel_tol=0.0, abs_tol=1e-9)):
                return (f"Cannot dispense {amount_mL:.3f} mL into vial {self.get_vial_info(dest_vial_num, 'vial_name')}: "
                        f"would exceed capacity ({post_dispense_volume:.6f} mL > {max_vial_volume:.6f} mL max)")
        return None

    def _dispense_open_vial(self, dest_vial_num, amount_mL, parameters, liquid, initial_move, move_speed, record_scale):
        """The liquid step of a dispense into a vial, with parameters resolved and the vial ready.
        Returns (scale readings, dispense speed used)."""
        wait_time = parameters.dispense_wait_time
        blowout_vol = parameters.blowout_vol
        # Calculate overdispense volume from constituent parts
        overdispense_vol = (amount_mL +
                           parameters.overaspirate_vol +
                           parameters.pre_asp_air_vol +
                           parameters.post_asp_air_vol)

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

        #Pipet into the vial
        if initial_move:
            self.c9.move_z(height, vel=move_speed if move_speed is not None else None)

        # Scale recording: only with the vial in the clamp; nothing in simulation
        scale = self.start_scale_recording(record_scale, 'dispensing')

        scale.mark_start()
        
        # Perform the actual dispensing
        self.pipet_dispense(overdispense_vol, wait_time=wait_time, blowout_vol=blowout_vol)
        
        scale.mark_end()
        readings = scale.stop()

         #Track the added volume in the dataframe
        self.VIAL_DF.at[dest_vial_num,'vial_volume']=self.VIAL_DF.at[dest_vial_num,'vial_volume']+amount_mL
        self.PIPET_FLUID_VOLUME -= amount_mL
        self._track_dispense(self._vial_container_id(dest_vial_num), amount_mL, liquid, vial_index=dest_vial_num)
        self.save_robot_status()

        return readings, dispense_speed

    def _analyze_mass_readings(self, readings, operation, vial_label, target_mL, stability_info,
                               save, file_prefix, file_stem):
        """Mass from recorded scale readings (pipetting_data/mass_analysis.py): logs the result,
        updates stability_info in place, and saves the CSV and plot if save is set.
        Files go to output/mass_measurements/<log name or date>/<prefix>data_<stem>_<time>.csv
        (and <prefix>plot_...png). Returns the mass in g (0.0 with too few readings)."""
        result = analyze_mass_readings(readings, operation)
        stability_info.update(result['stability'])
        stats = result['stability']
        if result['pre_baseline_g'] is None:
            self.logger.warning("Insufficient continuous data for baseline calculation")
        else:
            self.logger.info(f"{operation} mass - pre-baseline: {result['pre_method']} avg={result['pre_baseline_g']:.6f}g, "
                             f"post-baseline: {result['post_method']} avg={result['post_baseline_g']:.6f}g, "
                             f"mass difference: {result['mass_g']:.6f}g")
            self.logger.info(f"Reading summary: {stats['pre_total_count']} pre-readings ({stats['pre_stable_count']} stable), "
                             f"{stats['post_total_count']} post-readings ({stats['post_stable_count']} stable)")
            self.logger.info(f"Baseline variability: pre_std={stats['pre_baseline_std']:.6f}g, "
                             f"post_std={stats['post_baseline_std']:.6f}g")
            self.logger.info(f"Target: {target_mL:.3f} mL = ~{target_mL*1.0:.6f} g for water")
        if save and readings:
            if hasattr(self, 'log_filename') and self.log_filename:
                log_folder = self.log_filename.replace('.log', '')
            else:
                log_folder = time.strftime("%Y-%m-%d")
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            save_mass_readings(readings, result, operation, vial_label, target_mL,
                               os.path.join("output", "mass_measurements", log_folder),
                               f"{file_prefix}data_{file_stem}_{timestamp_str}.csv",
                               f"{file_prefix}plot_{file_stem}_{timestamp_str}.png", self.logger)
        return result['mass_g']

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
                    steady_status, weight = self.read_scale()
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

    def _resolve_pipetting_plate(self, expected_type=None, wells=None):
        inv = getattr(self, "plate_inventory", None)
        if inv is None:
            positions = self._load_yaml_file("robot_state/track_positions.yaml", "track positions")
            path = "robot_state/track_status.yaml"
            if getattr(self, "_scheduler_state_root", None):
                path = os.path.join(self._scheduler_state_root, "track_status.yaml")
            inv = self.plate_inventory = PlateInventory.load(path, self.WELLPLATES, positions)
        plate_id = inv.plate_at("pipetting_area", expected_type)
        if plate_id in inv.attachments:
            raise ValueError(f"Remove the lid from plate {plate_id} before pipetting")
        plate_type = inv.plate_type(plate_id)
        definition = self.WELLPLATES[plate_type]
        if not definition.get("name_in_Locator"):
            raise ValueError(f"No Locator grid configured for {plate_type}")
        if wells is not None:
            for well in wells:
                if not isinstance(well, (int, np.integer)) or isinstance(well, bool) or not 0 <= well < definition["num_wells"]:
                    raise ValueError(f"Well index {well} is invalid for plate {plate_id} ({plate_type})")
        self.last_plate_id = plate_id
        return plate_type

    def select_wellplate_grid(self, well_plate_type=None):
        """The Locator grid for well_plate_type (wellplates.yaml; None: the plate on the deck)."""
        well_plate_type = self._resolve_pipetting_plate(well_plate_type)
        return getattr(Locator, self.WELLPLATES[well_plate_type]["name_in_Locator"])

    #Dispense into a series of wells (dest_wp_num_array) a specific set of amounts (amount_mL_array)
    def dispense_into_wellplate(self, dest_wp_num_array, amount_mL_array, parameters=None, liquid=None, well_plate_type=None):
        """
        Unit operation: dispense from the held tip into one or more wells of the plate in the pipetting area.
        Usually one well: dispense_into_wellplate([well], [volume_mL]).

        Args:
            dest_wp_num_array (list or range): well indices, 0 to the plate's num_wells - 1 (wellplates.yaml)
            amount_mL_array (list[float]): mL into each well, same length as dest_wp_num_array.
                0 skips that well. Each well's total (this dose plus what is already recorded in it)
                must be at most the plate's max_volume_per_well (wellplates.yaml).
            parameters (PipettingParameters, optional): Manual parameter overrides (None: calibrated/default)
            liquid (str, optional): liquids.yaml name for calibrated parameters and the +/- uncertainty
            well_plate_type (str, optional): expected plate type. None: the type of the plate in the
                pipetting area (plate inventory). The type used is logged.

        Checked before anything moves (ValueError): list lengths match, well indices fit the plate,
        volumes are numbers in mL (not missing, not negative) and each well stays within its maximum.
        Warns, without stopping, if the doses add up to more than the tracked liquid in the tip.
        """
        requested_type = well_plate_type
        well_plate_type = self._resolve_pipetting_plate(well_plate_type, dest_wp_num_array)
        self.logger.info(f"Pipetting plate: {well_plate_type} "
                         f"({'as given' if requested_type else 'from the plate inventory'})")
        self._check_wellplate_dispense_inputs(dest_wp_num_array, amount_mL_array, well_plate_type)

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

            # Use intelligent parameter resolution for each volume: defaults â†’ liquid-calibrated â†’ user overrides
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
            self._track_dispense(self._well_container_id(dest_wp_num_array[i]), amount_mL, liquid)

        self.PIPET_FLUID_VOLUME -= np.sum(amount_mL_array)  # <-- Add this line back
        self.save_robot_status()
        return True

    def _check_wellplate_dispense_inputs(self, wells, amounts_mL, well_plate_type):
        """Inputs of dispense_into_wellplate, checked before anything moves (see its docstring)."""
        wells, amounts_mL = list(wells), list(amounts_mL)
        if len(wells) != len(amounts_mL):
            raise ValueError(f"Wellplate dispense refused before moving: {len(wells)} wells but "
                             f"{len(amounts_mL)} volumes; give one volume (mL) per well")
        # Same check as the wellplate flow: one column per dose, so a well listed twice adds up
        plan = pd.DataFrame(0.0, index=list(dict.fromkeys(wells)),
                            columns=[f"dose {n + 1}" for n in range(len(wells))], dtype=object)
        for n, (well, amount) in enumerate(zip(wells, amounts_mL)):
            plan.at[well, f"dose {n + 1}"] = amount
        checked = self._checked_wellplate_plan(plan, well_plate_type)
        total_mL = float(checked.to_numpy().sum())
        in_tip_mL = getattr(self, "PIPET_FLUID_VOLUME", None)
        if in_tip_mL is not None and total_mL > float(in_tip_mL) + 1e-6:
            self.logger.warning(f"Dispensing {total_mL:.3f} mL into the wellplate but the tip is tracked "
                                f"as holding {float(in_tip_mL):.3f} mL")

    def dispense_from_vials_into_wellplate(self, well_plate_df, parameters=None, liquid=None, well_plate_type=None):
        """
        Flow: dispense from one or more vials into wellplate wells. Ends with no tip held and the vials home.

        Steps:
            1. Stage vials if needed, with no tip held (stage_vials): closed vials go to a position
               safe to move with a tip (rack 0-5), since they are uncapped and used in the clamp;
               open vials that will use the small tip go to a small-tip-safe position. The vial that
               will be emptiest after dispensing goes first; ties go to the lowest starting volume.
            2. Condition the tip if the liquid has condition_tip: true in liquids.yaml, once per tip
               group, at that group's largest dose.
            3. Dispense with as few tip switches as possible: per vial, wells are grouped by resolved
               tip (smallest tip first, largest dose first within a group) and the tip is locked within
               a group. Each vial gets a fresh tip, because vials may hold different contents.
            4. Remove the tip after each vial and return that vial home.

        Future option (now possible with the stored sample contents, not built; not needed yet): when
        several vials hold the same solute in the same solvent, optionally reuse one tip across them,
        dispensing from the most dilute to the most concentrated and conditioning only once. Off by default.

        Checked before anything moves (ValueError naming the vial and well):
            - every volume is a number in mL, not missing and not negative (0 means skip that well);
            - every well index fits the plate in the pipetting area;
            - each well's total (this table plus what is already recorded in it) fits the plate's
              max_volume_per_well (wellplates.yaml);
            - every vial has a liquid name when liquid is given per vial;
            - each dose fits a tip (no tip can take it -> error), and each vial holds enough.

        Args:
            well_plate_df (DataFrame): Columns are vial names, index is well index, values are volumes in mL
            parameters (PipettingParameters, optional): Manual parameter overrides
            liquid (str or dict, optional): liquids.yaml name for calibrated parameters, the condition_tip
                setting and the ± uncertainty. One name for all vials, or {vial name: liquid name}.
            well_plate_type (str): Type of wellplate for positioning

        Example:
            well_plate_df = pd.DataFrame({
                "sds_stock": [0.1, 0.2, 0.0, 0.1],  # mL into wells 0-3
                "water": [0.0, 0.1, 0.3, 0.0],
            })
            robot.dispense_from_vials_into_wellplate(well_plate_df, liquid={"sds_stock": "SDS", "water": "water"})
        """
        well_plate_type = self._resolve_pipetting_plate(well_plate_type, well_plate_df.index)
        vial_names = well_plate_df.columns.tolist()
        liquids = self._wellplate_plan_liquids(vial_names, liquid)
        well_plate_df = self._checked_wellplate_plan(well_plate_df, well_plate_type)
        self.logger.info(f"Dispensing from vials {vial_names} into wellplate")

        # Check there is enough liquid in each vial
        vols_required = well_plate_df.sum(axis=0)
        for vial_name, volume_needed in vols_required.items():
            if volume_needed > 1e-6:
                vial_volume = self.get_vial_info(self.normalize_vial_index(vial_name), 'vial_volume')
                if vial_volume < volume_needed - 1e-6:
                    self.pause_after_error(f"Not enough solution in vial {vial_name}: need {volume_needed:.3f} mL, have {vial_volume:.3f} mL")

        # Resolve the tip and parameters for every dose before moving anything
        plans = {}
        for vial_name in vial_names:
            groups = {}
            for well_idx, volume in well_plate_df[vial_name].items():
                if volume > 1e-6:
                    resolved_params, resolved_tip = self.resolve_pipetting_setup(volume, liquids[vial_name], parameters)
                    groups.setdefault(resolved_tip, []).append((well_idx, volume, resolved_params))
            if groups:
                for doses in groups.values():
                    doses.sort(key=lambda dose: (-dose[1], dose[0]))  # Largest dose first, then well order
                plans[vial_name] = [(tip, groups[tip]) for tip in
                                    sorted(groups, key=lambda tip: self.PIPET_TIPS[tip]['volume'])]

        # 1. Stage (no tip held): closed vials and open small-tip vials that are not already in place.
        #    The vial that will be emptiest after dispensing goes first (safest position).
        to_stage = []
        for vial_name, tip_groups in plans.items():
            uses_small_tip = any(tip == 'small_tip' for tip, _ in tip_groups)
            if self._needs_staging(vial_name, uses_small_tip):
                current = float(self.get_vial_info(self.normalize_vial_index(vial_name), 'vial_volume'))
                to_stage.append((current - vols_required[vial_name], current, vial_name))
        staged = {}
        if to_stage:
            if self.HELD_PIPET_TYPE is not None:
                self.remove_pipet()
            staged = self.stage_vials([name for _, _, name in sorted(to_stage)])

        for vial_name, tip_groups in plans.items():
            vial_liquid = liquids[vial_name]
            conditioning = liquid_conditions_tip(vial_liquid)
            for tip, doses in tip_groups:
                # 2. Condition once per tip group, at the group's largest dose
                if conditioning:
                    self.condition_tip(vial_name, volume_per_cycle_mL=doses[0][1], specified_tip=tip)
                # 3. Dispense with the tip locked for this group
                for well_idx, volume, resolved_params in doses:
                    self.aspirate_from_vial(vial_name, volume, parameters=resolved_params,
                                            liquid=vial_liquid, specified_tip=tip)
                    self.dispense_into_wellplate([well_idx], [volume], parameters=resolved_params, liquid=vial_liquid,
                                                 well_plate_type=well_plate_type)
            # 4. Fresh tip for the next vial; return the vial home (frees the clamp and its staging spot)
            if self.HELD_PIPET_TYPE is not None:
                self.remove_pipet()
            if vial_name in staged:
                self.return_staged_vials([vial_name])
            else:
                self.return_vial_home(vial_name)

        self.return_staged_vials(staged.keys())
        self.logger.info("Wellplate dispensing completed")
        return True

    @staticmethod
    def _wellplate_plan_liquids(vial_names, liquid):
        """{vial: liquid name or None} from one name (or None) or a {vial: liquid} table."""
        if isinstance(liquid, dict):
            missing = [name for name in vial_names if name not in liquid]
            if missing:
                raise ValueError(f"No liquid given for vial(s) {missing}; liquid= needs an entry for every vial")
            return {name: liquid[name] for name in vial_names}
        return {name: liquid for name in vial_names}

    def _checked_wellplate_plan(self, well_plate_df, well_plate_type):
        """Reject a wellplate plan before anything moves: volumes must be numbers in mL, not missing
        or negative, and each well's total (plus what is already recorded in it) must fit the plate."""
        problems = []
        clean = {}
        for vial_name in well_plate_df.columns:
            column = {}
            for well_idx, value in well_plate_df[vial_name].items():
                try:
                    volume = float(value)
                except (TypeError, ValueError):
                    problems.append(f"{vial_name} → well {well_idx}: {value!r} is not a volume in mL")
                    continue
                if not math.isfinite(volume):
                    problems.append(f"{vial_name} → well {well_idx}: volume missing")
                elif volume < 0:
                    problems.append(f"{vial_name} → well {well_idx}: negative volume {volume:g} mL")
                column[well_idx] = volume
            clean[vial_name] = column
        if problems:
            raise ValueError("Wellplate plan refused before moving: " + "; ".join(problems))
        checked = pd.DataFrame(clean, index=well_plate_df.index).fillna(0.0)
        limit = ((getattr(self, "WELLPLATES", None) or {}).get(well_plate_type) or {}).get("max_volume_per_well")
        if limit is not None:
            store = self._contents_store()
            for well_idx, planned in checked.sum(axis=1).items():
                existing = 0.0
                if store is not None:
                    record = store.get(self._well_container_id(well_idx))
                    existing = record["volume_mL"] if record else 0.0
                if planned + existing > float(limit) + 1e-9:
                    problems.append(f"well {well_idx}: {planned:g} mL planned + {existing:g} mL already in it "
                                    f"> {float(limit):g} mL per well ({well_plate_type})")
            if problems:
                raise ValueError("Wellplate plan refused before moving: " + "; ".join(problems))
        return checked

    # ====================================================================
    # RESERVOIR SYSTEM
    # ====================================================================

    #Prime the line from the reservoir to the vial. In theory this could happen automatically. Probably good to do it if you are using a reservoir. 
    def prime_reservoir_line(self, reservoir_index, overflow_vial, volume=0.5):
        """Prime the reservoir line by dispensing volume (mL, > 0; default 0.5) into overflow_vial
        (dispense_into_vial_from_reservoir). Refused before moving like that method."""
        self._check_reservoir_inputs(reservoir_index, volume)
        overflow_vial = self.normalize_vial_index(overflow_vial) #Convert to int if needed
        self.logger.info(f"Priming reservoir {reservoir_index} line into vial {overflow_vial}: {volume:.3f} mL")
        self.dispense_into_vial_from_reservoir(reservoir_index,overflow_vial,volume)

    def _reservoir_carousel_position(self, reservoir_index):
        """(carousel_angle, carousel_height) for the reservoir from syringe_pumps.yaml; None for a missing one."""
        return (self.get_config_parameter('pumps', reservoir_index, 'carousel_angle', error_on_missing=False),
                self.get_config_parameter('pumps', reservoir_index, 'carousel_height', error_on_missing=False))

    def _check_reservoir_inputs(self, reservoir_index, volume):
        """ValueError before moving unless volume is a number > 0 mL and the reservoir has carousel settings."""
        try:
            positive = float(volume) > 0  # also refuses NaN
        except (TypeError, ValueError):
            positive = False
        if not positive:
            raise ValueError(f"Reservoir dispense refused before moving: volume must be > 0 mL, got {volume!r}")
        if None in self._reservoir_carousel_position(reservoir_index):
            raise ValueError(f"Reservoir dispense refused before moving: reservoir {reservoir_index!r} has no "
                             f"carousel_angle and carousel_height in syringe_pumps.yaml (pump 0 is the pipetting pump)")

    def dispense_from_reservoir_into_clamp_vial(self, reservoir_index, vial_name, volume, reservoir_params=None,
                                                liquid=None, record_scale=False):
        """
        Unit operation: dispense volume (mL) from a reservoir into the open vial already in the clamp.
        Moves the carousel to the reservoir, runs the pump strokes, and moves the carousel home.
        Does not move, uncap or recap the vial (dispense_into_vial_from_reservoir does that).

        Args:
            reservoir_index (int): pump in syringe_pumps.yaml
            vial_name (str or int): the vial in the clamp; must be open
            volume (float): mL to dispense; the vial must have room (vial_volume in vial_positions.yaml)
            reservoir_params (ReservoirParameters or dict, optional): overrides of the defaults
            liquid (str, optional): liquids.yaml name (recorded with the transfer)
            record_scale (bool): record the balance around the pump strokes

        Refused before moving (ValueError): volume not > 0 mL, a reservoir without carousel settings in
        syringe_pumps.yaml, or a vial not in the clamp, not open, or without room.
        Returns the scale readings (empty unless recorded); analyze them with
        pipetting_data/mass_analysis.py.
        """
        self._check_reservoir_inputs(reservoir_index, volume)
        vial_index = self.normalize_vial_index(vial_name)
        reservoir_params = self._get_optimized_reservoir_parameters(volume, liquid, reservoir_params)
        if self.get_vial_info(vial_index, 'location') != 'clamp':
            raise ValueError(f"Reservoir dispense refused before moving: vial {vial_name} is not in the clamp "
                             f"(use dispense_into_vial_from_reservoir to move it there)")
        if not self.is_vial_pipetable(vial_index):
            raise ValueError(f"Reservoir dispense refused before moving: vial {vial_name} is not open "
                             f"(use dispense_into_vial_from_reservoir to uncap it)")
        max_vial_volume = self.get_config_parameter('vial_positions', 'clamp', 'vial_volume', error_on_missing=False)
        if max_vial_volume is not None and self.get_vial_info(vial_index, 'vial_volume') + volume > max_vial_volume:
            raise ValueError(f"Reservoir dispense refused before moving: {volume:.3f} mL would overfill vial "
                             f"{vial_name} ({max_vial_volume:.3f} mL max)")
        return self._dispense_from_reservoir_into_clamp_vial(reservoir_index, vial_index, volume, reservoir_params,
                                                             liquid, record_scale)

    def _dispense_from_reservoir_into_clamp_vial(self, reservoir_index, vial_index, volume, reservoir_params,
                                                 liquid, record_scale):
        """The liquid step of a reservoir dispense, with parameters resolved and the vial open in the clamp.
        Returns the scale readings."""
        #Step 2: move the carousel to reservoir position
        carousel_angle, carousel_height = self._reservoir_carousel_position(reservoir_index)  # checked by callers
        self.c9.move_carousel(carousel_angle, carousel_height)

        #Step 3: aspirate and dispense from the reservoir
        max_volume = (self.get_config_parameter('pumps', reservoir_index, 'volume', error_on_missing=False)
                      or self.get_config_parameter('robot_hardware', 'reservoir_defaults', 'pump_volume_mL',
                                                   error_on_missing=True))

        # Scale recording (the vial is in the clamp); nothing in simulation
        scale = self.start_scale_recording(record_scale, 'dispensing', settle=False)

        scale.mark_start()
        
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
        
        scale.mark_end()
        readings = scale.stop()

        vial_volume = self.get_vial_info(vial_index,'vial_volume')
        self.VIAL_DF.at[vial_index,'vial_volume']=(vial_volume+volume)
        self.save_robot_status()
        self._track_reservoir_dispense(reservoir_index, vial_index, volume, liquid)

        #Step 4: Return the vial back to home
        self.move_carousel_to('home')

        return readings

    def dispense_into_vial_from_reservoir(self, reservoir_index, vial_index, volume, reservoir_params=None, liquid=None, measure_weight=False, return_home=True, continuous_mass_monitoring=False, save_mass_data=False):
        """
        Flow: dispense liquid from a reservoir into a vial, getting the vial ready and putting it back.

        Checks the vial has room, moves it to the clamp and uncaps it if needed, homes the arm,
        optionally weighs it, runs the liquid step (dispense_from_reservoir_into_clamp_vial),
        analyzes the mass if recorded, recaps the vial and (return_home=True) returns it home.
        With return_home=False the vial is left capped in the clamp.
        
        Args:
            reservoir_index: Index of the reservoir to dispense from
            vial_index: Index/name of destination vial
            volume: Volume to dispense in mL
            reservoir_params (ReservoirParameters, optional): Liquid handling parameters (uses defaults if None)
            liquid (str, optional): Liquid type for calibrated parameter optimization (future feature)
            measure_weight (bool): Whether to measure mass before/after dispensing (default: False)
            continuous_mass_monitoring (bool): Whether to continuously monitor mass during dispensing (default: False)
            save_mass_data (bool): Whether to save continuous mass data to file (default: False)

        Refused before moving (ValueError): volume not > 0 mL, or a reservoir without carousel_angle and
        carousel_height in syringe_pumps.yaml (e.g. pump 0, the pipetting pump).
        """
        self._check_reservoir_inputs(reservoir_index, volume)
        vial_index = self.normalize_vial_index(vial_index) #Convert to int if needed
        
        # Use intelligent parameter resolution: defaults â†’ liquid-calibrated â†’ user overrides
        reservoir_params = self._get_optimized_reservoir_parameters(volume, liquid, reservoir_params)
        
        
        self.logger.info(f"Dispensing into vial {vial_index} from reservoir {reservoir_index}: {volume:.3f} mL")
        self.logger.debug(f"Reservoir parameters: aspirate_speed={reservoir_params.aspirate_speed}, dispense_speed={reservoir_params.dispense_speed}, overaspirate={reservoir_params.overaspirate_vol:.3f} mL")
        measured_mass = None
        stability_info = None


        dest_vial_clamped = self.get_vial_info(vial_index,'location')=='clamp' #Is the destination vial clamped?
        dest_vial_volume = self.get_vial_info(vial_index,'vial_volume') #What is the current vial volume?

        # Check if dispensing would exceed vial capacity
        dest_vial_location = self.get_vial_info(vial_index, 'location')
        max_vial_volume = self.get_config_parameter('vial_positions', dest_vial_location, 'vial_volume', error_on_missing=False)
        if max_vial_volume is not None:
            post_dispense_volume = dest_vial_volume + volume
            if post_dispense_volume > max_vial_volume:
                self.pause_after_error(f"Cannot dispense {volume:.3f} mL into vial {self.get_vial_info(vial_index, 'vial_name')}: from reservoir {reservoir_index} - current volume {dest_vial_volume:.3f} mL + dispense volume {volume:.3f} mL = {post_dispense_volume:.3f} mL, which "
                                     f"would exceed capacity ({post_dispense_volume:.3f} mL > {max_vial_volume:.3f} mL max)")
                return

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
            stability_info = empty_stability()
            
            if not self.simulate:
                initial_mass = self.read_steady_scale()
            else:
                initial_mass = 0.0

        # The liquid step (same as dispense_from_reservoir_into_clamp_vial, parameters already resolved)
        continuous_mass_data = self._dispense_from_reservoir_into_clamp_vial(
            reservoir_index, vial_index, volume, reservoir_params, liquid, continuous_mass_monitoring)


        # Analyze continuous monitoring data or do traditional measurement
        if continuous_mass_monitoring and continuous_mass_data:  # pipetting_data/mass_analysis.py
            measured_mass = self._analyze_mass_readings(
                continuous_mass_data, 'reservoir', vial_index, volume, stability_info,
                save=save_mass_data, file_prefix="reservoir_mass_", file_stem=f"{vial_index}_res{reservoir_index}")

        elif continuous_mass_monitoring and self.simulate:
            measured_mass = 0.0  # No scale readings in simulation
        elif measure_weight: # Traditional before/after measurement
            if not self.simulate:
                final_mass = self.read_steady_scale()
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
        

        _check_optional_speed(self, 'return_vial_home', 'move_speed', move_speed)
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
        """Put the held vial down at location / location_index (vial_positions.yaml).
        Pauses with an error if the spot holds another vial. move_speed: percent of max velocity, 1-100 (None: the default speed)."""
        _check_optional_speed(self, 'drop_off_vial', 'move_speed', move_speed)
        self.logger.debug(f"Dropping off vial {vial_name} at location {location} with index {location_index}")

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        destination = list(self.get_location(False,location,location_index))
        release_offset_mm = self.get_config_parameter('vial_positions', location, 'release_height_offset_mm', error_on_missing=True)
        z_axis = self.get_config_parameter('robot_hardware', 'robot_axes', 'z_axis', error_on_missing=True)
        if not self.simulate:
            from north.n9_kinematics import counts_to_mm, mm_to_counts
            release_height_mm = counts_to_mm(z_axis, destination[z_axis]) + release_offset_mm
            destination[z_axis] = mm_to_counts(z_axis, release_height_mm)
        occupying_vial = self.get_vial_in_location(location,location_index)
        # Allow drop-off if the location is empty or occupied by the same vial
        destination_empty = (occupying_vial is None) or (occupying_vial == vial_index)

        self.check_for_errors([[destination_empty, True, "Cannot move vial to destination, destination full"]],True)

        release_settle_time_s = self.get_config_parameter('robot_hardware', 'gripper_timing', 'release_settle_time_s', error_on_missing=True)

        release_speed = self.get_speed('precise_movement')
        self.c9.goto_safe(destination, vel=move_speed if move_speed is not None else self.get_speed('default_robot')) #move vial to destination
        #self.c9.open_gripper()
        self.open_vial_gripper()  
        if not self.simulate:
            time.sleep(release_settle_time_s)

        if location == 'clamp':
            #self.close_clamp()
            self.move_rel_xyz(z_distance=10, vel=release_speed)
            #self.open_clamp()

        self.VIAL_DF.at[vial_index, 'location']=location
        self.VIAL_DF.at[vial_index, 'location_index']=location_index
        self.GRIPPER_STATUS = None #We no longer have the vial
        self.GRIPPER_VIAL_INDEX = None
        self.save_robot_status() #Update in memory

    def grab_vial(self,vial_name, move_speed=None):
        """Pick up the vial (recapping it first if its cap is held). Pauses with an error if the gripper
        is full or the vial cannot be moved (not capped or not 8 mL). move_speed: percent of max velocity, 1-100 (None: the default speed)."""
        
        _check_optional_speed(self, 'grab_vial', 'move_speed', move_speed)
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

        grab_settle_time_s = self.get_config_parameter(
            'robot_hardware', 'gripper_timing', 'grab_settle_time_s', error_on_missing=True
        )
        #self.open_gripper()
        self.goto_location_if_not_there(initial_location, move_speed=move_speed) #move to vial
        #self.c9.close_gripper() #grip vial
        self.close_vial_gripper()  

        if not self.simulate:
            time.sleep(grab_settle_time_s)
        
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
            move_speed (float, optional): percent of max velocity, 1-100 (None: the default speed); refused before moving outside that range
         """
        
        _check_optional_speed(self, 'move_vial_to_location', 'move_speed', move_speed)
        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        # Safety check: If holding a pipet, ensure the movement path is safe
        if self.HELD_PIPET_TYPE is not None:
            if not self.is_vial_movement_safe_with_pipet(vial_index, location, location_index):
                self.pause_after_error(f"Cannot move vial {vial_name} while holding pipet - only tip_held_move_indices in vial_positions.yaml are safe (rack 0-5, clamp 0)")

        vial_location = self.get_vial_info(vial_index,'location')
        vial_location_index = self.get_vial_info(vial_index,'location_index')

        if vial_location == location and vial_location_index == location_index:
            self.logger.info(f"Vial {vial_name} already at {location} index {location_index}, no move needed")
            return

        self.logger.info(f"Moving vial {self.get_vial_info(vial_index, 'vial_name')} to {location}: {location_index}")
        self.grab_vial(vial_index, move_speed=move_speed) #Grab the vial
        self.drop_off_vial(vial_index,location,location_index, move_speed=move_speed) #Drop off the vial

    def get_vial_in_location(self, location_name, location_index):
        """Index of the vial at location_name / location_index, or None."""
        # Filter rows where both conditions match
        mask = (self.VIAL_DF['location'] == location_name) & (self.VIAL_DF['location_index'] == location_index)
        
        # Get the matching values
        matching_vials = self.VIAL_DF.loc[mask, 'vial_index'].values

        # Return the first match or None if no match is found
        return int(matching_vials[0]) if len(matching_vials) > 0 else None

    #Uncap the vial in the clamp
    def uncap_clamp_vial(self, revs=None, move_speed=None):
        """Take the cap off the vial in the clamp (revs: unscrewing turns; None: capping.uncap_revs in
        robot_hardware.yaml, 3.0) and hold it.
        Pauses with an error if the gripper is full, the clamp is empty or the vial is open. move_speed: percent of max velocity, 1-100 (None: the default speed)."""
        _check_optional_speed(self, 'uncap_clamp_vial', 'move_speed', move_speed)
        if revs is None:
            revs = self.get_config_parameter('robot_hardware', 'capping', 'uncap_revs', error_on_missing=True)
        self.logger.debug("Removing cap from clamped vial")

        clamp_vial_index = self.get_vial_in_location('clamp',0)

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS is None, True, "Cannot uncap, gripper full"])
        error_check_list.append([clamp_vial_index is None, False, "Cannot uncap, no vial in clamp"])
        error_check_list.append([self.is_vial_movable(clamp_vial_index), True, "Can't uncap, vial is uncapped already"])

        self.check_for_errors(error_check_list,True) #Check for an error and pause if there is one
        
        position_settle_time_s = self.get_config_parameter(
            'robot_hardware', 'gripper_timing', 'uncap_position_settle_time_s', error_on_missing=True
        )
        grip_settle_time_s = self.get_config_parameter(
            'robot_hardware', 'gripper_timing', 'uncap_grip_settle_time_s', error_on_missing=True
        )
        self.close_clamp() #clamp vial
        self.goto_location_if_not_there(vial_clamp, move_speed=move_speed) #Maybe check if it is already there or not  
        if not self.simulate:
            time.sleep(position_settle_time_s)
        
        #self.c9.close_gripper()
        self.close_vial_gripper() 

        if not self.simulate:
            time.sleep(grip_settle_time_s)
        self.c9.uncap(revs=revs)
        # Record the cap, the vial it belongs to and the uncapped vial together, and save, before
        # moving again: a fault in the next moves must not leave "Cap" with no vial number
        self.GRIPPER_STATUS = "Cap"
        self.GRIPPER_VIAL_INDEX = clamp_vial_index
        self.VIAL_DF.at[clamp_vial_index, 'capped']=False
        self.save_robot_status()
        self.move_rel_xyz(z_distance=5)
        self.open_clamp()
        self.save_robot_status()

    #Recap the vial in the clamp
    #OAM: Note, April 14, 2026  changed revs=2.2 to 1.8 and torque_thresh from 600 to 550
    #SP: Note: I made this change on this date
    def recap_clamp_vial(self, revs=None, torque_thresh=None, move_speed=None):
        """Screw the held cap onto the vial in the clamp (revs: turns; torque_thresh: C9 torque limit;
        None: capping.recap_revs / recap_torque_thresh in robot_hardware.yaml, 1.8 and 550). Pauses with an error if no cap is held, the clamp is empty or the vial
        is capped. move_speed: percent of max velocity, 1-100 (None: the default speed)."""
        _check_optional_speed(self, 'recap_clamp_vial', 'move_speed', move_speed)
        if revs is None:
            revs = self.get_config_parameter('robot_hardware', 'capping', 'recap_revs', error_on_missing=True)
        if torque_thresh is None:
            torque_thresh = self.get_config_parameter('robot_hardware', 'capping', 'recap_torque_thresh',
                                                      error_on_missing=True)
        self.logger.debug("Recapping clamped vial")
        
        clamp_vial_index = self.get_vial_in_location('clamp',0)

        error_check_list = [] #List of specific errors for this method
        error_check_list.append([self.GRIPPER_STATUS, "Cap", "Cannot recap, no cap in gripper"])
        error_check_list.append([clamp_vial_index is None, False, "Cannot recap, no vial in clamp"])
        error_check_list.append([self.is_vial_movable(clamp_vial_index), False, "Can't recap, vial is capped already"])
        
        self.check_for_errors(error_check_list,True) #Let's pause if there is an error

        position_settle_time_s = self.get_config_parameter(
            'robot_hardware', 'gripper_timing', 'recap_position_settle_time_s', error_on_missing=True
        )
        self.close_clamp() #Make sure vial is clamped
        self.goto_location_if_not_there(vial_clamp_cap, move_speed=move_speed)
        if not self.simulate:
            time.sleep(position_settle_time_s)

        self.c9.cap(revs=revs, torque_thresh = torque_thresh) #Cap the vial #Cap the vial
        #self.c9.open_gripper() #Open the gripper to release the cap
        self.open_vial_gripper()  
        
        self.GRIPPER_STATUS = None
        self.open_clamp()

        self.VIAL_DF.at[clamp_vial_index, 'capped']=True #Update the vial status
        self.GRIPPER_VIAL_INDEX = None
        self.save_robot_status()

    #Checks first that you aren't already there... This mostly applies for cap/decap
    def goto_location_if_not_there(self, location, move_speed=None):
        """Move to location (Locator position) unless already within 50 counts. move_speed: percent of max velocity, 1-100 (None: the default speed)."""
        _check_optional_speed(self, 'goto_location_if_not_there', 'move_speed', move_speed)
        difference_threshold = self.get_config_parameter('robot_hardware', 'physical_constants',
                                                         'already_there_counts', error_on_missing=True)
        if self.get_location_distance(location, self.c9.get_robot_positions()) > difference_threshold:
            self.c9.goto_safe(location, vel=move_speed if move_speed is not None else self.get_speed('fast_approach'))

    #Measurement for how far two points are
    def get_location_distance(self, loc_1, loc_2):
        """Sum of the axis differences (counts) between two Locator positions; 0 in simulation."""
        if self.simulate:
            return 0
        difference = np.sum(np.absolute(np.array(loc_2)[1:4] - np.array(loc_1)[1:4]))
        return difference
    
    #Removes the target vial, vortexes it, then drops it off
    def vortex_vial(self, vial_name, vortex_time, vortex_speed=70,
                    return_location: tuple[str, int] | None = None):
        """
        Lift and vortex a vial, then safely transport and release it.
        Args:
            `vial_name` (str): Name of the vial to vortex
            `vortex_time` (float): seconds, within vortex_limits min_time_s-max_time_s (robot_hardware.yaml)
            `vortex_speed` (float): normalized 0-100 speed, within vortex_limits min_speed-max_speed.
                Default 70. The speed used is logged.
            `return_location` (tuple[str, int] | None): Destination name and
                index, e.g. ("main_8mL_rack", 25). None returns the vial to its
                home location, so the call ends state-preserving. An explicit
                destination avoids intermediate release and regripping; it does
                not change the safe transport approach.

        A time or speed that is missing, not a number or out of range is refused (ValueError)
        before the vial is picked up.
        """
        self._check_vortex_inputs(vortex_time, vortex_speed)  # values passed on unchanged

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed

        if return_location is not None:
            if (not isinstance(return_location, tuple) or len(return_location) != 2
                    or not isinstance(return_location[0], str) or not return_location[0].strip()
                    or not isinstance(return_location[1], Integral)
                    or isinstance(return_location[1], bool) or return_location[1] < 0):
                message = "return_location must be a (location_name, nonnegative integer index) tuple."
                self.logger.error(message)
                raise ValueError(message)
            destination_name, destination_index = return_location
            destination_index = int(destination_index)
            return_location = (destination_name, destination_index)
            if self.get_location(False, destination_name, destination_index) is None:
                raise ValueError(f"Invalid vortex return location: {return_location!r}")
            occupying_vial = self.get_vial_in_location(destination_name, destination_index)
            if occupying_vial is not None and occupying_vial != vial_index:
                message = f"Cannot vortex to occupied return location: {return_location!r}"
                self.logger.error(message)
                raise ValueError(message)

        self.logger.info(f"Vortexing Vial: {self.get_vial_info(vial_index,'vial_name')} "
                         f"for {vortex_time:g} s at speed {vortex_speed:g}")
        
        #Check to see if the vial is capped
        if self.GRIPPER_VIAL_INDEX == vial_index  and self.GRIPPER_STATUS == "Cap":
            self.recap_clamp_vial()
        self.grab_vial(vial_index)
    
        #Vortex
        self.c9.move_z(self.get_safe_height()) #Move to a higher height
        gripper_axis = self.get_config_parameter('robot_hardware', 'robot_axes', 'gripper_axis', error_on_missing=False) or 0
        self.c9.move_axis(gripper_axis, 1000*vortex_time*vortex_speed, vel=vortex_speed,accel=10000)
        self.c9.reduce_axis_position(axis=gripper_axis)

        if return_location is None:
            location = self.get_vial_info(vial_index,'home_location')
            location_index = self.get_vial_info(vial_index,'home_location_index')
        else:
            location, location_index = return_location

        self.drop_off_vial(vial_index, location, location_index)

    def _check_vortex_inputs(self, vortex_time, vortex_speed):
        """ValueError if the time or speed is not a number or is outside vortex_limits (robot_hardware.yaml)."""
        limits = _hardware_limits(self, 'vortex_limits')
        _check_in_range('Vortex', 'vortex_time', vortex_time, limits['min_time_s'], limits['max_time_s'], ' s',
                        'vortex_limits')
        _check_in_range('Vortex', 'vortex_speed', vortex_speed, limits['min_speed'], limits['max_speed'], '',
                        'vortex_limits')

    # ---- Unit operations: tip conditioning and small-tip staging ----
    def _present_vial_locations(self):
        """Locations from vial_positions.yaml whose rack is present on the deck."""
        positions = getattr(self, 'VIAL_POSITIONS', None) or {}
        return {name: rack for name, rack in positions.items() if isinstance(rack, dict) and rack.get('rack_present')}

    def get_small_tip_safe_indices(self, location):
        """Indices at a location where the small tip can aspirate safely (small_tip_safe_indices in vial_positions.yaml)."""
        rack = self._present_vial_locations().get(location) or {}
        return [int(i) for i in (rack.get('small_tip_safe_indices') or [])]

    def get_small_tip_staging_positions(self):
        """(location, index) staging targets, safest first (small_tip_staging_rank in vial_positions.yaml)."""
        ranked = []
        for name, rack in self._present_vial_locations().items():
            for index, rank in (rack.get('small_tip_staging_rank') or {}).items():
                ranked.append((rank, name, int(index)))
        return [(name, index) for _, name, index in sorted(ranked)]

    def condition_tip(self, vial_name, volume_per_cycle_mL=0.1, cycles=5, specified_tip=None):
        """
        Unit operation: condition the pipet tip by aspirating from and dispensing back
        into the same vial several times. Leaves the (wet) tip attached.

        Args:
            vial_name (str): Vial to condition the tip with
            volume_per_cycle_mL (float): Volume aspirated and dispensed per cycle (default 0.1 mL)
            cycles (int): Number of aspirate/dispense cycles (default 5)
            specified_tip (str, optional): Tip type to use and keep ("small_tip" or "large_tip")
        """
        # Fast, simple parameters: conditioning does not need calibrated accuracy.
        conditioning_params = PipettingParameters(
            aspirate_speed=15,
            dispense_speed=5,
            dispense_wait_time=0.0,
            blowout_vol=0.5,
        )
        self.logger.info(f"Conditioning tip with {vial_name}: {cycles} cycles of {volume_per_cycle_mL*1000:.1f} uL")
        for _ in range(cycles):
            self.aspirate_from_vial(vial_name, volume_per_cycle_mL, parameters=conditioning_params,
                                    specified_tip=specified_tip)
            self.dispense_into_vial(vial_name, volume_per_cycle_mL, parameters=conditioning_params)

    def get_pipet_safe_move_positions(self):
        """Rack positions a vial can be moved from or to while a tip is held (rack 0-5),
        safest first. Same rule as is_location_safe_for_vial_movement_with_pipet."""
        rack = self._present_vial_locations().get('main_8mL_rack') or {}
        return [('main_8mL_rack', index) for index in range(int(rack.get('rack_size') or 0))
                if self.is_location_safe_for_vial_movement_with_pipet('main_8mL_rack', index)]

    def _is_closed_vial(self, vial_index):
        """A movable vial that must be uncapped in the clamp before pipetting."""
        return self.is_vial_movable(vial_index) and not self.is_vial_pipetable(vial_index)

    def _needs_staging(self, vial_name, uses_small_tip):
        """True if stage_vials should move this vial before pipetting: a closed vial that is not
        where it can be moved with a tip held, or an open small-tip vial not in a small-tip-safe position."""
        if not self.is_vial_movable(vial_name):
            return False
        vial_index = self.normalize_vial_index(vial_name)
        location = self.get_vial_info(vial_index, 'location')
        location_index = int(self.get_vial_info(vial_index, 'location_index'))
        if self._is_closed_vial(vial_name):
            return not self.is_location_safe_for_vial_movement_with_pipet(location, location_index)
        return uses_small_tip and location_index not in self.get_small_tip_safe_indices(location)

    def stage_vials(self, vial_names, safe_positions=None):
        """
        Unit operation: move vials to staging positions before pipetting. Call it with no tip held.
        This is a deliberate state change; undo it with return_staged_vials().

        Where a vial goes depends on its cap:
            - closed: a rack position safe to move from with a tip held (rack 0-5). A closed vial is
              uncapped and used in the clamp, and the tip is picked up before it moves there, so it
              must start somewhere it can leave with a tip on.
            - open: a small-tip-safe position (small_tip_staging_rank in vial_positions.yaml,
              safest first), where it is pipetted in place.
        Vials already in such a position stay put. Within each group, vials are placed in the order
        given: the first gets the safest free position. Choosing the order (e.g. highest pull-out
        risk first) is left to the caller, for example with workflows/helpers/vial_risk_analyzer.py.

        The staged position is remembered: a staged vial leaving the clamp goes back there, not home.

        Args:
            vial_names (list): Vial names in priority order (highest priority first)
            safe_positions (list, optional): (location, index) pairs, safest first, used for every
                vial instead of the cap-based positions.

        Returns:
            dict: {vial_name: (location, index)} for each vial that was staged.
                Vials that could not be staged (no free position) stay where they are.
        """
        if safe_positions is not None:
            shared = list(safe_positions)
            pools = {True: shared, False: shared}
        else:
            pools = {True: self.get_pipet_safe_move_positions(), False: self.get_small_tip_staging_positions()}
        remembered = getattr(self, 'STAGED_POSITIONS', None)
        if remembered is None:
            remembered = self.STAGED_POSITIONS = {}
        staged = {}
        for vial_name in vial_names:
            vial_index = self.normalize_vial_index(vial_name)
            closed = self._is_closed_vial(vial_name)
            positions = pools[closed]
            current = (self.get_vial_info(vial_index, 'location'), self.get_vial_info(vial_index, 'location_index'))
            already_safe = current in positions or (
                closed and safe_positions is None
                and self.is_location_safe_for_vial_movement_with_pipet(current[0], current[1]))
            if already_safe:
                if current in positions:
                    positions.remove(current)
                staged[vial_name] = current
                remembered[vial_index] = current
                continue
            target = None
            while positions:
                candidate = positions.pop(0)
                occupant = self.get_vial_in_location(candidate[0], candidate[1])
                if occupant is None or occupant == vial_index:
                    target = candidate
                    break
            if target is None:
                kind = "position safe to move with a tip" if closed else "small-tip-safe position"
                self.logger.warning(f"No free {kind} for {vial_name}; leaving it in place")
                continue
            self.logger.info(f"Staging {vial_name} -> {target[0]}[{target[1]}]")
            self.move_vial_to_location(vial_index, target[0], target[1])
            staged[vial_name] = target
            remembered[vial_index] = target
        return staged

    def return_staged_vials(self, vial_names):
        """
        Unit operation: return staged vials to their home positions (the reset for stage_vials).
        Vials that are already home are not moved.

        Args:
            vial_names (iterable): Vial names to return (e.g. the keys returned by stage_vials)
        """
        remembered = getattr(self, 'STAGED_POSITIONS', None) or {}
        for vial_name in vial_names:
            vial_index = self.normalize_vial_index(vial_name)
            remembered.pop(vial_index, None)
            at_home = (self.get_vial_info(vial_index, 'location') == self.get_vial_info(vial_index, 'home_location')
                       and self.get_vial_info(vial_index, 'location_index') == self.get_vial_info(vial_index, 'home_location_index'))
            if not at_home:
                self.return_vial_home(vial_index)

    def _return_from_clamp(self, vial_index, move_speed=None):
        """Move a vial out of the clamp: back to its staged position if stage_vials placed it
        (so it can be moved again with a tip held), otherwise home."""
        staged_at = (getattr(self, 'STAGED_POSITIONS', None) or {}).get(vial_index)
        if staged_at is None:
            self.return_vial_home(vial_index, move_speed=move_speed)
            return
        if self.GRIPPER_STATUS == "Cap":
            self.recap_clamp_vial(move_speed=move_speed)
        self.move_vial_to_location(vial_index, staged_at[0], staged_at[1], move_speed=move_speed)

    def return_all_vials_home(self):
        """
        Cleanup: return every vial that is not at home, one by one. Used by Lash_E.safe_stop.
        Vials that can't be moved (is_vial_movable is False, e.g. uncapped) are left in place and
        reported. A failure on one vial is logged and the rest are still tried.

        Returns:
            (returned, left): lists of vial names; left is a list of (vial_name, reason)
        """
        returned, left = [], []
        vial_df = getattr(self, 'VIAL_DF', None)
        if vial_df is None:
            return returned, left
        for vial_index in list(vial_df.index):
            vial_name = self.get_vial_info(vial_index, 'vial_name')
            at_home = (self.get_vial_info(vial_index, 'location') == self.get_vial_info(vial_index, 'home_location')
                       and self.get_vial_info(vial_index, 'location_index') == self.get_vial_info(vial_index, 'home_location_index'))
            if at_home:
                continue
            if not self.is_vial_movable(vial_index):
                self.logger.warning(f"Vial {vial_name} is not home but can't be moved; leaving it in place")
                left.append((vial_name, "not movable"))
                continue
            try:
                self.return_vial_home(vial_index)
                returned.append(vial_name)
            except Exception as e:
                self.logger.exception(f"Could not return vial {vial_name} home")
                left.append((vial_name, f"return failed: {e}"))
        return returned, left

    # ---- Flows: adding liquid to vials ----
    # Sources are specified by the caller for now. Later these may choose sources from
    # sample-contents metadata (e.g. find a vial with matching contents), which is not implemented yet.

    def _add_liquid_to_vial(self, dest_vial_name, volume_mL, reservoir=None, source_vial=None,
                            liquid=None, measure_weight=False):
        """Add liquid to a vial from a reservoir or, by pipet transfer, from a source vial. Returns the volume added (mL)."""
        if (reservoir is None) == (source_vial is None):
            raise ValueError("Specify exactly one of reservoir or source_vial")
        if reservoir is not None:
            self.dispense_into_vial_from_reservoir(reservoir, dest_vial_name, volume_mL, liquid=liquid,
                                                   measure_weight=measure_weight)
        else:
            self.dispense_from_vial_into_vial(source_vial, dest_vial_name, volume_mL, liquid=liquid,
                                              measure_weight=measure_weight)
        return volume_mL

    def refill_vial(self, vial_name, target_volume_mL, reservoir=None, source_vial=None, liquid=None,
                    min_addition_mL=0.0, measure_weight=False):
        """
        Flow: top a vial up to a target volume from a reservoir or a source vial.
        Ends with no tip held and the vials home.

        Exactly one source must be given. Choosing the source automatically from
        sample-contents metadata is planned but not implemented yet.

        Args:
            vial_name (str): Vial to refill
            target_volume_mL (float): Volume the vial should hold afterwards
            reservoir (int, optional): Reservoir (pump) index to dispense from
            source_vial (str, optional): Vial to pipet from
            liquid (str, optional): Liquid type for calibrated parameters
            min_addition_mL (float): Skip the refill if less than this would be added (default 0)
            measure_weight (bool): Weigh the addition (default False)

        Returns:
            float: Volume added (mL); 0 if no refill was needed or possible
        """
        if reservoir is None and source_vial is None:
            raise NotImplementedError("Automatic source selection needs sample-contents metadata; "
                                      "specify reservoir or source_vial")
        current_mL = self.get_vial_info(vial_name, 'vial_volume')
        volume_mL = target_volume_mL - current_mL
        if source_vial is not None:
            volume_mL = min(volume_mL, self.get_vial_info(source_vial, 'vial_volume'))
        if volume_mL <= 0 or volume_mL < min_addition_mL:
            self.logger.info(f"No refill needed for {vial_name} ({current_mL:.2f} mL, target {target_volume_mL:.2f} mL)")
            return 0.0
        self.logger.info(f"Refilling {vial_name}: {current_mL:.2f} mL -> {current_mL + volume_mL:.2f} mL")
        return self._add_liquid_to_vial(vial_name, volume_mL, reservoir=reservoir, source_vial=source_vial,
                                        liquid=liquid, measure_weight=measure_weight)

    def create_substock(self, dest_vial_name, components, vortex_time=None, measure_weight=False):
        """
        Flow: prepare a substock by adding several components to one vial, then optionally vortex.
        Ends with no tip held and the vials home.

        Reservoir additions run first, then pipetted additions in the order given.
        Each addition returns the vial home; keeping the vial on the clamp to weigh every
        addition is a possible future improvement. Sources are specified for now; choosing
        them from sample-contents metadata is planned but not implemented.

        Args:
            dest_vial_name (str): Vial to prepare the substock in
            components (list of dict): Each has 'volume_mL' and either 'reservoir' (int) or
                'source_vial' (str), plus an optional 'liquid'
            vortex_time (float, optional): Vortex for this many seconds after the last addition
            measure_weight (bool): Weigh each addition (default False)

        Returns:
            float: Total volume added (mL)
        """
        for component in components:
            if (component.get('reservoir') is None) == (component.get('source_vial') is None):
                raise ValueError(f"Each component needs exactly one of 'reservoir' or 'source_vial': {component}")
        ordered = ([c for c in components if c.get('reservoir') is not None]
                   + [c for c in components if c.get('reservoir') is None])
        total_mL = 0.0
        for component in ordered:
            if component['volume_mL'] <= 0:
                continue
            total_mL += self._add_liquid_to_vial(dest_vial_name, component['volume_mL'],
                                                 reservoir=component.get('reservoir'),
                                                 source_vial=component.get('source_vial'),
                                                 liquid=component.get('liquid'),
                                                 measure_weight=measure_weight)
        if vortex_time:
            self.vortex_vial(dest_vial_name, vortex_time)
        return total_mL
        
    #This is just to formalize the process of error checking so its more extensible
    #This may end up deprecated
    #Error check list is as follows error_check[0] is the value, error_check[1] is the target value, error_check[2] is the error message
    def check_for_errors(self, error_check_list, stop_after_error=False):
        """For each [value, expected, message], log a warning if value != expected, and pause with
        the message if stop_after_error. Returns True if any check failed."""
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
        """True if the vial is capped and an 8 mL vial (the gripper can move it)."""

        vial_index = self.normalize_vial_index(vial_name) #Convert to int if neede

        movable = False
        movable = self.get_vial_info(vial_index,'capped') == True and self.get_vial_info(vial_index,'vial_type') == "8_mL"
        return movable
    
    #Check to see if the pipet can have liquids added/removed
    def is_vial_pipetable(self, vial_name):
        """True if the vial is uncapped or has an open cap."""

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
        # Safe positions: tip_held_move_indices in vial_positions.yaml (rack 0-5 and the clamp);
        # every other location is unsafe when holding a pipet
        positions = getattr(self, 'VIAL_POSITIONS', None) or _read_yaml_quietly(self.VIAL_POSITIONS_FILE)
        safe = (positions.get(location) or {}).get('tip_held_move_indices')
        return bool(safe) and location_index is not None and location_index in safe

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
 
    # ---- Primitives: vial gripper and clamp ----
    def open_vial_gripper(self):
        """Primitive: open the robot arm's vial gripper (releases a vial or cap)."""
        self.c9.set_output(self._hardware_setting('robot_pneumatics', 'vial_gripper', 7), False)

    def close_vial_gripper(self):
        """Primitive: close the robot arm's vial gripper (grips a vial or cap)."""
        self.c9.set_output(self._hardware_setting('robot_pneumatics', 'vial_gripper', 7), True)

    def open_clamp(self):
        """Primitive: open the vial clamp."""
        self.c9.open_clamp()

    def close_clamp(self):
        """Primitive: close the vial clamp."""
        self.c9.close_clamp()

    def move_rel_xyz(self, x_distance=0, y_distance=0, z_distance=0, vel=None):
        """Move the arm by x, y, z distances (mm) from where it is. vel: percent of max velocity, 1-100
        (None: standard_xy). Refused before moving (ValueError) outside that range."""
        _check_optional_speed(self, 'move_rel_xyz', 'vel', vel)
        if vel is None:
            vel = self.get_speed('standard_xy')
        self.logger.debug(f"Moving robot relative to current position by x: {x_distance}, y: {y_distance}, z: {z_distance} mm, vel: {vel}")
        
        # Get axis numbers from configuration with fallbacks
        gripper_axis = self.get_config_parameter('robot_hardware', 'robot_axes', 'gripper_axis', error_on_missing=False) or 0
        elbow_axis = self.get_config_parameter('robot_hardware', 'robot_axes', 'elbow_axis', error_on_missing=False) or 1
        shoulder_axis = self.get_config_parameter('robot_hardware', 'robot_axes', 'shoulder_axis', error_on_missing=False) or 2
        z_axis = self.get_config_parameter('robot_hardware', 'robot_axes', 'z_axis', error_on_missing=False) or 3
        
        gripper_angle = self.c9.get_axis_position(gripper_axis)
        current_loc_mm = self.c9.n9_fk(gripper_angle, self.c9.get_axis_position(elbow_axis), self.c9.get_axis_position(shoulder_axis))
        
        target_x =  current_loc_mm[0] + x_distance
        target_y =  current_loc_mm[1] + y_distance
        target_theta = math.degrees(current_loc_mm[2])
        target_z =  self.c9.counts_to_mm(z_axis, self.c9.get_axis_position(z_axis)) + z_distance

        self.c9.move_xyz(target_x, target_y, target_z, vel=vel, tool_orientation=target_theta)

    #Move the robot to the home position    
    def move_home(self):
        """Move the arm and the carousel home."""
        self.logger.info("Moving robot to home position")
        self.c9.goto_safe(home)
        self.move_carousel_to('home')

    # ====================================================================
    # 9. VIAL INFO & LOCATION UTILITIES
    # ====================================================================

    def _vials_available(self):
        """Check if vial operations are available (i.e., if vial file was loaded)"""
        return not self.VIAL_DF.empty

    #Get some piece of information about a vial
    #vial_index,vial_name,location,location_index,vial_volume,capped,cap_type,vial_type
    def get_vial_info(self,vial_name,column_name):
        """The value of column_name (vial table) for the vial, or None."""
        
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
        """Index of the vial called vial_name, or None."""
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
        """Lowest pipetting height (mm) at the vial's location (min_pipetting_height, vial_positions.yaml)."""

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
        """Locator position of the vial (pipetting position if use_pipet, else gripping position)."""
        vial_index = self.normalize_vial_index(vial_name) #Convert to int if needed
        location_name = self.get_vial_info(vial_index,'location')
        location_index = self.get_vial_info(vial_index,'location_index')
        return self.get_location(use_pipet,location_name,location_index)

    #Translate the location names and indexes to hard-coded locations for pipets or gripping vials
    def get_location(self, use_pipet, location_name, location_index):
        """Locator position for location_name / location_index (vial_positions.yaml): the pipetting
        position if use_pipet, else the gripping position. Pauses with an error for an unknown location or index."""
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
    

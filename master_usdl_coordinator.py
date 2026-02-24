#The purpose of this file is to combine multiple pieces of equipment into intuitive tools
import sys
sys.path.append("../utoronto_demo")
from North_Safe import North_Robot
from North_Safe import North_Track
from North_Safe import North_Temp
from North_Safe import North_Powder
from North_Safe import North_Spin
import pandas as pd
import os
from datetime import datetime
import logging

# Import ConfigManager for workflow config handling
try:
    from workflow_config_manager import ConfigManager
except ImportError:
    ConfigManager = None

# ================================================================================
# CYTATION DATA PROCESSING UTILITIES
# ================================================================================

def flatten_cytation_data(data, measurement_type="unknown"):
    """
    Reliably flatten Cytation MultiIndex DataFrame to simple columns.
    
    This utility handles the common Cytation data structure issues:
    - MultiIndex columns like [('rep1_CMC_Absorbance_96', '600')] -> ['600']
    - Well positions as index -> 'well_position' column
    - Proper column naming for different measurement types
    
    Args:
        data: DataFrame from lash_e.measure_wellplate()
        measurement_type: 'turbidity' or 'fluorescence' for column naming
        
    Returns:
        DataFrame with simple columns: ['well_position', 'turbidity_600'] or ['well_position', '334_373', '334_384']
        
    Usage:
        turbidity_data = flatten_cytation_data(turbidity_data, 'turbidity')
        fluorescence_data = flatten_cytation_data(fluorescence_data, 'fluorescence')
        
    Note:
        Ratio calculation for pyrene fluorescence: 334_373 / 334_384 (I₃/I₁ ratio)
        This is the standard pyrene ratio for CMC determination.
    """
    if data is None:
        return None
        
    # Handle MultiIndex columns - flatten to single level
    if isinstance(data.columns, pd.MultiIndex):
        # Take the second level (wavelength/measurement) as column names
        data.columns = [col[1] if isinstance(col, tuple) else col for col in data.columns]
    
    # Rename columns to standard names based on measurement type
    if measurement_type == 'turbidity':
        column_mapping = {}
        for col in data.columns:
            if '600' in str(col) or 'Absorbance' in str(col):
                column_mapping[col] = 'turbidity_600'
        if column_mapping:
            data = data.rename(columns=column_mapping)
    
    # Reset index to convert well positions (A1, A2, etc.) to a column
    data = data.reset_index()
    if 'index' in data.columns:
        data = data.rename(columns={'index': 'well_position'})
    elif data.columns[0] == 0 or str(data.columns[0]).startswith('Unnamed'):
        # Handle cases where reset_index creates unnamed column
        data.columns = ['well_position'] + list(data.columns[1:])
    
    return data

# ================================================================================

class Lash_E:
    nr_robot = None
    nr_track = None
    cytation = None
    spinner = None
    photoreactor = None
    temp_controller = None
    powder_dispenser = None
    simulate = None

    def __init__(self, vial_file=None, initialize_robot=True,initialize_track=True,initialize_biotek=True,initialize_t8=False,initialize_p2=False,simulate=False,logging_folder="../utoronto_demo/logs", workflow_globals=None, workflow_name=None):
        
        # Handle workflow config loading before any other initialization
        if workflow_globals is not None and workflow_name is not None and ConfigManager is not None:
            # Setup config file if it doesn't exist (preserves user edits if it does exist)
            ConfigManager.setup_config_if_missing(workflow_name, workflow_globals)
            
            # Load config from file (with user edits) and update globals
            updated_config = ConfigManager.load_and_update_globals(workflow_name, workflow_globals, None)  # Logger not yet available
            
            # Use updated SIMULATE value from config (overrides the passed simulate parameter)
            simulate = workflow_globals.get('SIMULATE', simulate)
            
            # Store config info for later reference
            self.workflow_name = workflow_name
            self.config_loaded = True
            self.config_keys_loaded = list(updated_config.keys()) if updated_config else []
        else:
            self.workflow_name = None
            self.config_loaded = False
            self.config_keys_loaded = []
        
        self.simulate = simulate
        self.vial_file = vial_file  # Store vial file for GUI access
        
        # Flag to control workflow continuation
        self._workflow_should_continue = True

        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.DEBUG)

        # Clear any old handlers to avoid duplication
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        suffix = "_simulate" if simulate else ""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"experiment_log{timestamp}{suffix}.log"
        log_path = os.path.join(logging_folder, log_filename)
        os.makedirs(logging_folder, exist_ok=True)
        
        # Store log filename for use by North_Robot for organizing mass measurement files
        self.log_filename = log_filename

        # File handler (DEBUG and up)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)

        # Console handler (INFO and up) → bind to stdout explicitly
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Make sure messages don't propagate to root logger (which might have its own handlers)
        self.logger.propagate = False

        # Log config loading results (now that logger is available)
        if self.config_loaded:
            self.logger.info(f"Workflow config loaded: {self.workflow_name}")
            self.logger.info(f"Config keys updated: {len(self.config_keys_loaded)} ({', '.join(self.config_keys_loaded)})")
            self.logger.info(f"SIMULATE mode: {self.simulate}")
        else:
            if workflow_globals is not None or workflow_name is not None:
                self.logger.warning("Incomplete config info provided - config loading skipped")
            else:
                self.logger.debug("No workflow config provided - using default initialization")

        # Check input status before hardware initialization
        self.check_input_status()
        
        # Exit early if workflow was aborted
        if not self._workflow_should_continue:
            self.logger.info("Workflow aborted by user - skipping hardware initialization")
            return
            
        # Reload config from file after GUI may have updated YAML values
        if workflow_globals is not None and workflow_name is not None and ConfigManager is not None:
            self.logger.debug("Reloading config after status check to get updated values")
            updated_config = ConfigManager.load_and_update_globals(workflow_name, workflow_globals, self.logger)
            
            # Update simulate flag with fresh value from file
            updated_simulate = workflow_globals.get('SIMULATE', self.simulate)
            if updated_simulate != self.simulate:
                self.update_simulate_flag(updated_simulate)

        # Create hardware objects AFTER config reload to use correct simulate flag
        if not self.simulate:
            from north import NorthC9
            c9 = NorthC9("A", network_serial="AU06CNCF")
            c8 = NorthC9('D', network=c9.network)
        else:
            from unittest.mock import MagicMock
            c9 = MagicMock()
            c8 = MagicMock()

        if initialize_robot:
            self.nr_robot = North_Robot(c9, c8, vial_file,simulate=self.simulate, logger=self.logger)
            # Pass log filename to robot for organized mass measurement file storage
            self.nr_robot.log_filename = self.log_filename
            self.spinner = North_Spin(c9, c8, simulate=self.simulate, logger=self.logger)

        if initialize_biotek:
            from biotek_new import Biotek_Wrapper
            self.cytation = Biotek_Wrapper(simulate=self.simulate, logger=self.logger)
        
        if initialize_track:
            self.nr_track = North_Track(c9, simulate=self.simulate, logger=self.logger)
        

        # Photoreactor initialization
        if not self.simulate:
            from photoreactor_controller import Photoreactor_Controller
            self.photoreactor = Photoreactor_Controller() #ADd logger

            if initialize_p2:
                self.powder_dispenser = North_Powder(c9, simulate=self.simulate, logger=self.logger)
            if initialize_t8:
                self.temp_controller = North_Temp(c9, c8, simulate=self.simulate, logger=self.logger)

        else:
            from unittest.mock import MagicMock
            self.photoreactor = MagicMock()
            self.powder_dispenser = MagicMock()
            self.temp_controller = MagicMock()
            #self.nr_track = MagicMock(c9)

    def update_simulate_flag(self, new_simulate_value):
        """
        Update the simulation flag after initialization (e.g., after ConfigManager loads updated values).
        
        This fixes the issue where Lash_E is initialized with one simulate value, but ConfigManager
        later updates the global SIMULATE variable. This method synchronizes the instance with
        the updated global value.
        
        Args:
            new_simulate_value: Updated simulation flag value (True/False)
        """
        if self.simulate == new_simulate_value:
            return  # No change needed
            
        old_simulate = self.simulate
        self.simulate = new_simulate_value
        
        self.logger.info(f"SIMULATE flag updated: {old_simulate} -> {new_simulate_value}")
        
        # Update simulation flag in all initialized components
        components_updated = []
        
        if hasattr(self, 'nr_robot') and self.nr_robot:
            self.nr_robot.simulate = new_simulate_value
            components_updated.append('nr_robot')
            
        if hasattr(self, 'nr_track') and self.nr_track:
            self.nr_track.simulate = new_simulate_value
            components_updated.append('nr_track')
            
        if hasattr(self, 'cytation') and self.cytation:
            self.cytation.simulate = new_simulate_value
            components_updated.append('cytation')
            
        if hasattr(self, 'spinner') and self.spinner:
            self.spinner.simulate = new_simulate_value
            components_updated.append('spinner')
            
        if hasattr(self, 'powder_dispenser') and self.powder_dispenser:
            self.powder_dispenser.simulate = new_simulate_value
            components_updated.append('powder_dispenser')
            
        if hasattr(self, 'temp_controller') and self.temp_controller:
            self.temp_controller.simulate = new_simulate_value
            components_updated.append('temp_controller')
        
        if components_updated:
            self.logger.debug(f"Updated simulate flag in components: {', '.join(components_updated)}")
        else:
            self.logger.debug("No components needed simulate flag update")

    def check_input_status(self):
        """Launch GUI for reviewing and finalizing robot states before hardware initialization."""

        self.logger.info("Launching status review GUI...")
        
        try:
            # Import and launch the GUI
            from vial_manager_gui import VialManagerMainWindow
            from PySide6.QtWidgets import QApplication
            import sys
            
            # Create Qt application if it doesn't exist
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            # Create and configure the GUI window
            gui = VialManagerMainWindow()
            gui.setWindowTitle("Robot Status Review - Workflow Initialization")
            
            # Detect workflow name from main module
            workflow_name = None
            try:
                import __main__
                if hasattr(__main__, '__file__') and __main__.__file__:
                    script_name = os.path.basename(__main__.__file__)
                    workflow_name = os.path.splitext(script_name)[0]  # Remove .py extension
                    self.logger.info(f"Detected workflow name: {workflow_name}")
            except Exception as e:
                self.logger.warning(f"Could not detect workflow name: {e}")
            
            # Configure GUI for workflow mode
            gui._setup_workflow_mode(self.vial_file, self, workflow_name)
            
            # Show GUI and wait for user decision
            gui.show()
            app.exec()  # Block until GUI closes
            
            # Check if workflow should continue
            self._workflow_should_continue = getattr(gui, '_workflow_continue', True)
            
            if self._workflow_should_continue:
                self.logger.info("Status review completed - proceeding with hardware initialization")
            else:
                self.logger.info("Workflow aborted by user")
                
        except Exception as e:
            self.logger.error(f"Error launching status GUI: {e}")
            # Continue with workflow if GUI fails
            self._workflow_should_continue = True

    def mass_dispense_into_vial(self,vial,mass_mg,channel=0, return_home=True):
        self.logger.info(f"Dispensing into vial: {vial} with mass: {mass_mg:.3f} mg") 
        self.nr_robot.move_vial_to_location(vial,'clamp',0)
        self.nr_robot.move_home()
        self.powder_dispenser.dispense_powder_mg(mass_mg=mass_mg,channel=channel) #Dispense 50 mg of solid into source_vial_a  
        if return_home:
            self.nr_robot.return_vial_home(vial) 

    def move_wellplate_to_cytation(self,wellplate_index=0,plate_type="96 WELL PLATE", use_lid=False, safe_movement=True):
        self.logger.info(f"Moving wellplate {wellplate_index} to Cytation")
        
        # Get Cytation software's wellplate name from robot's wellplate configuration
        cytation_plate_type = self.nr_robot.get_config_parameter('wellplates', plate_type, 'name_in_cytation', error_on_missing=True)
        if not cytation_plate_type:
            cytation_plate_type = plate_type  # Fallback to original if not found
            self.logger.warning(f"No 'name_in_cytation' found for plate type '{plate_type}', using '{cytation_plate_type}' for CarrierIn")
        else:
            self.logger.debug(f"Using robot plate type '{plate_type}' -> Cytation plate type '{cytation_plate_type}' for CarrierIn")
        
        if safe_movement:
            self.nr_track.move_through_path(['cytation_safe_area'])
        # Use robot's plate_type for robot movements
        self.nr_track.grab_wellplate_from_location('pipetting_area', plate_type)
        self.nr_track.move_through_path(['cytation_safe_area'])
        if not self.simulate:
            self.cytation.CarrierOut()
        # Use robot's plate_type for robot movements
        #input()
        self.nr_track.release_wellplate_in_location('cytation_tray', plate_type)
        #input()
        if not self.simulate:
            # Use cytation_plate_type for Cytation software commands
            self.cytation.CarrierIn(plate_type=cytation_plate_type, use_lid=use_lid)

    def move_wellplate_back_from_cytation(self,wellplate_index=0,plate_type="96 WELL PLATE", use_lid=False):
        self.logger.info("Moving wellplate %d back from Cytation", wellplate_index)
        
        # Get Cytation software's wellplate name from robot's wellplate configuration
        cytation_plate_type = self.nr_robot.get_config_parameter('wellplates', plate_type, 'name_in_cytation', error_on_missing=True)
        if not cytation_plate_type:
            cytation_plate_type = plate_type  # Fallback to original if not found
            self.logger.warning(f"No 'name_in_cytation' found for plate type '{plate_type}', using '{cytation_plate_type}' for CarrierIn")
        else:
            self.logger.debug(f"Using robot plate type '{plate_type}' -> Cytation plate type '{cytation_plate_type}' for CarrierIn")
        
        if not self.simulate:
            self.cytation.CarrierOut()
            None
        # Use robot's plate_type for robot movements
        self.nr_track.grab_wellplate_from_location('cytation_tray', plate_type)
        #input()
        self.nr_track.move_through_path(['cytation_safe_area'])
        if not self.simulate:
            # Use cytation_plate_type for Cytation software commands
            self.cytation.CarrierIn(plate_type=cytation_plate_type, use_lid=use_lid)
        # Use robot's plate_type for robot movements
        self.nr_track.release_wellplate_in_location('pipetting_area', plate_type)

    #Note from OAM: The data formatting from this can be annoying. Need to think about how to handle it. 
    def measure_wellplate(self, protocol_file_path=None, wells_to_measure=None, wellplate_index=0, plate_type="96 WELL PLATE", repeats=1, use_lid=False, safe_movement=True):
        """
        Measure a wellplate on the Cytation reader. Supports multiple protocols and replicate measurements.
        Each replicate includes all protocols, e.g.:
            rep1: fluorescence, absorbance
            rep2: fluorescence, absorbance
            ...
        
        Args:
            plate_type (str): Robot's wellplate type (e.g., "96 WELL PLATE", "quartz") - used for robot movements
            use_lid (bool): Whether the wellplate has a lid (default: False)
        """
        self.logger.info("Measuring wellplate %d with protocols: %s", wellplate_index, protocol_file_path)
        
        # Get Cytation software's wellplate name from robot's wellplate configuration
        cytation_plate_type = self.nr_robot.get_config_parameter('wellplates', plate_type, 'name_in_cytation', error_on_missing=True)
        if not cytation_plate_type:
            cytation_plate_type = plate_type  # Fallback to original if not found
            self.logger.warning(f"No 'name_in_cytation' found for plate type '{plate_type}', using '{cytation_plate_type}'")
        else:
            self.logger.debug(f"Using robot plate type '{plate_type}' -> Cytation plate type '{cytation_plate_type}'")
        
        self.nr_robot.move_home()
        self.move_wellplate_to_cytation(wellplate_index, plate_type=plate_type, use_lid=use_lid, safe_movement=safe_movement)

        all_data = []

        if not self.simulate and protocol_file_path is not None:  #Note from OAM: The data formatting from this can be annoying. Need to think about how to handle it. 
            # Ensure it's a list for consistent handling
            protocol_paths = protocol_file_path if isinstance(protocol_file_path, list) else [protocol_file_path]

            for i in range(repeats):
                for protocol_path in protocol_paths:
                    print(f"Running protocol {protocol_path} (rep {i+1})")
                    try:
                        # Use cytation_plate_type for the biotek system, not the robot's plate_type
                        data = self.cytation.run_protocol(protocol_path, wells_to_measure, plate_type=cytation_plate_type)
                        if data is not None:
                            # Add a MultiIndex: (replicate + protocol name, wavelength/column)
                            label = f"rep{i+1}_{os.path.splitext(os.path.basename(protocol_path))[0]}"
                            data.columns = pd.MultiIndex.from_tuples([(label, col) for col in data.columns])
                            all_data.append(data)
                    except RuntimeError as e:
                        self.logger.error(f"Plate read failed: {e}")
                        self.nr_robot.pause_after_error(f"Plate read failed: {e}")

            combined_data = pd.concat(all_data, axis=1) if all_data else None
        else:
            combined_data = None

        self.move_wellplate_back_from_cytation(wellplate_index, plate_type=plate_type, use_lid=use_lid)
        self.nr_track.origin()
        return combined_data

    def run_photoreactor(self,vial_index,target_rpm,intensity,duration,reactor_num):
        self.logger.info("Running photoreactor for vial %d with target RPM: %d, intensity: %d, duration: %d, reactor number: %d", vial_index, target_rpm, intensity, duration, reactor_num)
        self.nr_robot.move_vial_to_location(vial_index,'photoreactor_array',reactor_num)
        self.photoreactor.run_photoreactor(target_rpm,duration,intensity,reactor_num)
        self.nr_robot.return_vial_home(vial_index)

    def grab_new_wellplate(self):
        self.logger.info("Grabbing new wellplate")
        self.nr_robot.move_home()
        self.nr_track.get_new_wellplate(move_home_afterwards=True)
    
    def discard_used_wellplate(self):
        self.logger.info("Discarding used wellplate")
        self.nr_robot.move_home()
        self.nr_track.discard_wellplate(move_home_afterwards=True)
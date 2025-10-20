#The purpose of this file is to combine multiple pieces of equipment into intuitive tools
import sys
sys.path.append("../utoronto_demo")
from North_Safe import North_Robot
from North_Safe import North_Track
from North_Safe import North_Temp
from North_Safe import North_Powder
import pandas as pd
import os
from datetime import datetime
import logging

class Lash_E:
    nr_robot = None
    nr_track = None
    cytation = None
    photoreactor = None
    temp_controller = None
    powder_dispenser = None
    simulate = None

    def __init__(self, vial_file, initialize_robot=True,initialize_track=True,initialize_biotek=True,initialize_t8=False,initialize_p2=False,simulate=False,logging_folder="../utoronto_demo/logs"):
        
        self.simulate = simulate

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

        if not simulate:
            from north import NorthC9
            c9 = NorthC9("A", network_serial="AU06CNCF")
        else:
            from unittest.mock import MagicMock
            c9 = MagicMock()

        if initialize_robot:
            self.nr_robot = North_Robot(c9, vial_file,simulate=simulate, logger=self.logger)

        if initialize_biotek:
            from biotek_new import Biotek_Wrapper
            self.cytation = Biotek_Wrapper(simulate=simulate, logger=self.logger)
        
        if initialize_track:
            self.nr_track = North_Track(c9, simulate=simulate, logger=self.logger)
        

        # Photoreactor initialization
        if not self.simulate:
            from photoreactor_controller import Photoreactor_Controller
            self.photoreactor = Photoreactor_Controller() #ADd logger

            if initialize_p2:
                self.powder_dispenser = North_Powder(c9, simulate=simulate, logger=self.logger)
            if initialize_t8:
                self.temp_controller = North_Temp(c9, simulate=simulate, logger=self.logger)

        else:
            from unittest.mock import MagicMock
            self.photoreactor = MagicMock()
            self.powder_dispenser = MagicMock()
            self.temp_controller = MagicMock()
            #self.nr_track = MagicMock(c9)

    def mass_dispense_into_vial(self,vial,mass_mg,channel=0, return_home=True):
        self.logger.info(f"Dispensing into vial: {vial} with mass: {mass_mg:.3f} mg") 
        self.nr_robot.move_vial_to_location(vial,'clamp',0)
        self.nr_robot.move_home()
        self.powder_dispenser.dispense_powder_mg(mass_mg=mass_mg,channel=channel) #Dispense 50 mg of solid into source_vial_a  
        if return_home:
            self.nr_robot.return_vial_home(vial) 

    def move_wellplate_to_cytation(self,wellplate_index=0,plate_type="96 WELL PLATE"):
        self.logger.info(f"Moving wellplate {wellplate_index} to Cytation")
        
        # Get Cytation software's wellplate name from robot's wellplate configuration
        cytation_plate_type = self.nr_robot.get_config_parameter('wellplates', plate_type, 'name_in_cytation', error_on_missing=True)
        if not cytation_plate_type:
            cytation_plate_type = plate_type  # Fallback to original if not found
            self.logger.warning(f"No 'name_in_cytation' found for plate type '{plate_type}', using '{cytation_plate_type}' for CarrierIn")
        else:
            self.logger.debug(f"Using robot plate type '{plate_type}' -> Cytation plate type '{cytation_plate_type}' for CarrierIn")
        
        self.nr_track.move_through_path(['cytation_safe_area'])
        # Use robot's plate_type for robot movements
        self.nr_track.grab_wellplate_from_location('pipetting_area', plate_type)
        self.nr_track.move_through_path(['cytation_safe_area'])
        if not self.simulate:
            self.cytation.CarrierOut()
        # Use robot's plate_type for robot movements
        self.nr_track.release_wellplate_in_location('cytation_tray', plate_type)
        if not self.simulate:
            # Use cytation_plate_type for Cytation software commands
            self.cytation.CarrierIn(plate_type=cytation_plate_type)

    def move_wellplate_back_from_cytation(self,wellplate_index=0,plate_type="96 WELL PLATE"):
        self.logger.info("Moving wellplate %d back from Cytation", wellplate_index)
        
        # Get Cytation software's wellplate name from robot's wellplate configuration
        cytation_plate_type = self.nr_robot.get_config_parameter('wellplates', plate_type, 'name_in_cytation', error_on_missing=True)
        if not cytation_plate_type:
            cytation_plate_type = plate_type  # Fallback to original if not found
            self.logger.warning(f"No 'name_in_cytation' found for plate type '{plate_type}', using '{cytation_plate_type}' for CarrierIn")
        else:
            self.logger.debug(f"Using robot plate type '{plate_type}' -> Cytation plate type '{cytation_plate_type}' for CarrierIn")
        
        if not self.simulate:
            #self.cytation.CarrierOut()
            None
        # Use robot's plate_type for robot movements
        self.nr_track.grab_wellplate_from_location('cytation_tray', plate_type)
        self.nr_track.move_through_path(['cytation_safe_area'])
        if not self.simulate:
            # Use cytation_plate_type for Cytation software commands
            self.cytation.CarrierIn(plate_type=cytation_plate_type)
        # Use robot's plate_type for robot movements
        self.nr_track.release_wellplate_in_location('pipetting_area', plate_type)

    #Note from OAM: The data formatting from this can be annoying. Need to think about how to handle it. 
    def measure_wellplate(self, protocol_file_path=None, wells_to_measure=None, wellplate_index=0, plate_type="96 WELL PLATE", repeats=1):
        """
        Measure a wellplate on the Cytation reader. Supports multiple protocols and replicate measurements.
        Each replicate includes all protocols, e.g.:
            rep1: fluorescence, absorbance
            rep2: fluorescence, absorbance
            ...
        
        Args:
            plate_type (str): Robot's wellplate type (e.g., "96 WELL PLATE", "quartz") - used for robot movements
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
        self.move_wellplate_to_cytation(wellplate_index, plate_type=plate_type)

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

        self.move_wellplate_back_from_cytation(wellplate_index, plate_type=plate_type)
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
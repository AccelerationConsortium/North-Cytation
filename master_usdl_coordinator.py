#The purpose of this file is to combine multiple pieces of equipment into intuitive tools
import sys
sys.path.append("../utoronto_demo")
from North_Safe import North_Robot
from North_Safe import North_Track
from North_Safe import North_T8

class Lash_E:
    nr_robot = None
    nr_track = None
    cytation = None
    photoreactor = None
    temp_controller = None
    simulate = None

    def __init__(self, vial_file, initialize_robot=True,initialize_track=True,initialize_biotek=True,initialize_photoreactor=True,initialize_t8=False,simulate=False):
        if not simulate:
            from north import NorthC9
            c9 = NorthC9("A", network_serial="AU06CNCF")
        else:
            from unittest.mock import MagicMock
            c9 = MagicMock()

        self.simulate = simulate

        if initialize_robot:
            self.nr_robot = North_Robot(c9, vial_file,simulate=simulate)
        if initialize_track:
            self.nr_track = North_Track(c9)
        if initialize_biotek:
            from biotek_new import Biotek_Wrapper
            self.cytation = Biotek_Wrapper(simulate=simulate)
        else:
            self.cytation = None  # Or a mock object if you want to simulate calls

        # Photoreactor initialization
        if not self.simulate:
            from photoreactor_controller import Photoreactor_Controller
            self.photoreactor = Photoreactor_Controller()
        else:
            from unittest.mock import MagicMock
            self.photoreactor = MagicMock()

        if initialize_t8:
            self.temp_controller = North_T8(c9)

    def move_wellplate_to_cytation(self,wellplate_index=0,quartz=False,plate_type="96 WELL PLATE"):
        self.nr_track.grab_well_plate_from_nr(wellplate_index,quartz_wp=quartz)
        self.nr_track.move_gripper_to_cytation()
        if not self.simulate:
            self.cytation.CarrierOut()
        self.nr_track.release_well_plate_in_cytation(quartz_wp=quartz)
        if not self.simulate:
            self.cytation.CarrierIn(plate_type=plate_type)

    def move_wellplate_back_from_cytation(self,wellplate_index=0,quartz=False,plate_type="96 WELL PLATE"):
        if not self.simulate:
            self.cytation.CarrierOut()
        self.nr_track.grab_well_plate_from_cytation(quartz_wp=quartz)
        if not self.simulate:
            self.cytation.CarrierIn(plate_type=plate_type)
        self.nr_track.return_well_plate_to_nr(wellplate_index,quartz_wp=quartz)  

    def measure_wellplate(self,protocol_file_path=None,wells_to_measure=None,wellplate_index=0,quartz=False,plate_type="96 WELL PLATE"):
        """
        Move wellplate to the Cytation for plate reader measurements.
        
        Args:
            `protocol_file_path` (str): Path to the measurement protocol file.
            `wells_to_measure` (list or range): Indices of the wells to measure.
            `wellplate_index` (int): Index of where the wellplate is stored.
            `quartz` (bool): Whether the wellplate is a quartz plate.
            `plate_type` (str): Type of the plate (e.g., "96 WELL PLATE").
        """
        self.nr_robot.move_home()
        self.move_wellplate_to_cytation(wellplate_index,quartz=quartz,plate_type=plate_type)
        if not self.simulate and protocol_file_path is not None:
            data = self.cytation.run_protocol(protocol_file_path,wells_to_measure,plate_type = plate_type)
        else:
            data = None
        self.move_wellplate_back_from_cytation(wellplate_index,quartz=quartz,plate_type=plate_type)
        self.nr_track.origin()
        return data

    def run_photoreactor(self,vial_index,target_rpm,intensity,duration,reactor_num):
        self.nr_robot.move_vial_to_location(vial_index,'photoreactor_array',reactor_num)
        self.photoreactor.run_photoreactor(target_rpm,duration,intensity,reactor_num)
        self.nr_robot.return_vial_home(vial_index)

    def grab_new_wellplate(self,dest_wp_position=0):
        self.nr_track.get_next_WP_from_source()
        self.nr_track.return_well_plate_to_nr(dest_wp_position)
        self.nr_track.origin()
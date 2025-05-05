#The purpose of this file is to combine multiple pieces of equipment into intuitive tools
import sys
sys.path.append("../utoronto_demo")
from North_Safe import North_Robot
from North_Safe import North_Track
from North_Safe import North_T8
from photoreactor_controller import Photoreactor_Controller

class Lash_E:
    nr_robot = None
    nr_track = None
    cytation = None
    photoreactor = None
    temp_controller = None

    def __init__(self, vial_file, initialize_robot=True,initialize_track=True,initialize_biotek=True,initialize_photoreactor=True,initialize_t8=False,simulate=False):
        if not simulate:
            from north import NorthC9
            c9 = NorthC9("A", network_serial="AU06CNCF")
        else:
            from unittest.mock import MagicMock
            c9 = MagicMock()
        from biotek_new import Biotek_Wrapper

        print(simulate)
        if initialize_robot:
            self.nr_robot = North_Robot(c9, vial_file)
        if initialize_track:
            self.nr_track = North_Track(c9)
        if initialize_biotek:
            self.cytation = Biotek_Wrapper(simulate=simulate)
        if initialize_photoreactor:
            self.photoreactor = Photoreactor_Controller()
        if initialize_t8:
            self.temp_controller = North_T8(c9)

    def move_wellplate_to_cytation(self,wellplate_index=0,quartz=False,plate_type="96 WELL PLATE"):
        self.nr_track.grab_well_plate_from_nr(wellplate_index,quartz_wp=quartz)
        self.nr_track.move_gripper_to_cytation()
        self.cytation.CarrierOut()
        self.nr_track.release_well_plate_in_cytation(quartz_wp=quartz)
        self.cytation.CarrierIn(plate_type=plate_type)

    def move_wellplate_back_from_cytation(self,wellplate_index=0,quartz=False,plate_type="96 WELL PLATE"):
        self.cytation.CarrierOut()
        self.nr_track.grab_well_plate_from_cytation(quartz_wp=quartz)
        self.cytation.CarrierIn(plate_type=plate_type)
        self.nr_track.return_well_plate_to_nr(wellplate_index,quartz_wp=quartz)  

    def measure_wellplate(self,protocol_file_path,wells_to_measure=None,wellplate_index=0,quartz=False,plate_type="96 WELL PLATE"):
        self.nr_robot.move_home()
        self.move_wellplate_to_cytation(wellplate_index,quartz=quartz,plate_type=plate_type)
        data = self.cytation.run_protocol(protocol_file_path,wells_to_measure,plate_type = plate_type)
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
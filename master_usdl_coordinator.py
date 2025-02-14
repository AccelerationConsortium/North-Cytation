#The purpose of this file is to combine multiple pieces of equipment into intuitive tools
import sys
sys.path.append("../utoronto_demo")
from North_Safe import North_Robot
from North_Safe import North_Track
from north import NorthC9
import pandas as pd
from biotek import Biotek
from photoreactor_controller import Photoreactor_Controller
import time

class Lash_E:

    nr_robot = None
    nr_track = None
    cytation = None
    photoreactor = None

    def __init__(self, vial_file, initialize_robot=True,initialize_track=True,initialize_biotek=True,initialize_photoreactor=True):
        c9 = NorthC9("A", network_serial="AU06CNCF")
        if initialize_robot:
            self.nr_robot = North_Robot(c9, vial_file)
            self.nr_robot.reset_after_initialization()
        if initialize_track:
            self.nr_track = North_Track(c9)
        if initialize_biotek:
            self.cytation = Biotek()
        if initialize_photoreactor:
            self.photoreactor = Photoreactor_Controller()

    def move_wellplate_to_cytation(self,wellplate_index=0,quartz=False):
        self.nr_track.grab_well_plate_from_nr(wellplate_index,quartz_wp=quartz)
        self.nr_track.move_gripper_to_cytation()
        self.cytation.CarrierOut()
        self.nr_track.release_well_plate_in_cytation(quartz_wp=quartz)
        self.cytation.CarrierIn()

    def move_wellplate_back_from_cytation(self,wellplate_index=0,quartz=False):
        self.cytation.CarrierOut()
        self.nr_track.grab_well_plate_from_cytation(quartz_wp=quartz)
        self.cytation.CarrierIn()
        self.nr_track.return_well_plate_to_nr(wellplate_index,quartz_wp=quartz)  

    def run_cytation_program(self,protocol_file_path):
        plate = self.cytation.load_protocol(protocol_file_path)
        run = self.cytation.run_protocol(plate)
        while self.cytation.protocol_in_progress(run):
            print("Read in progress...")
            time.sleep(10)

    def measure_wellplate(self,protocol_file_path,wellplate_index=0,quartz=False):
        self.nr_robot.move_home()
        self.move_wellplate_to_cytation(wellplate_index,quartz=quartz)
        self.run_cytation_program(protocol_file_path)
        self.move_wellplate_back_from_cytation(wellplate_index,quartz=quartz)
        self.nr_track.origin()

    def run_photoreactor(self,vial_index,target_rpm,intensity,duration,reactor_num):
        self.nr_robot.move_vial_to_location(vial_index,'photoreactor_array',reactor_num)
        self.photoreactor.run_photoreactor(target_rpm,duration,intensity,reactor_num)
        self.nr_robot.return_vial_home(vial_index)

    def grab_new_wellplate(self,dest_wp_position=0):
        self.nr_track.get_next_WP_from_source()
        self.nr_track.return_well_plate_to_nr(dest_wp_position)
        self.nr_track.origin()
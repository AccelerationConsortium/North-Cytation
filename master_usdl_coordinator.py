#The purpose of this file is to combine multiple pieces of equipment into intuitive tools
import sys
sys.path.append("../utoronto_demo")
from North_Safe import North_Robot
from North_Safe import North_Track
from north import NorthC9
import pandas as pd
from biotek import Biotek
import time

nr_robot = None
nr_track = None
cytation = None
photo_reactor = None

#Need to also initialize the photoreactor
def initialize(vial_df):
    c9 = NorthC9("A", network_serial="AU06CNCF")
    nr_robot = North_Robot(c9, vial_df)
    nr_track = North_Track(c9)
    cytation = Biotek()
    nr_robot.reset_after_initialization()

def move_wellplate_to_cytation(wellplate_index=0,quartz=False):
    nr_track.grab_well_plate_from_nr(wellplate_index,quartz_wp=quartz)
    nr_track.move_gripper_to_cytation()
    cytation.CarrierOut()
    nr_track.release_well_plate_in_cytation(quartz_wp=quartz)
    cytation.CarrierIn()

def move_wellplate_back_from_cytation(wellplate_index=0,quartz=False):
    cytation.CarrierOut()
    nr_track.grab_well_plate_from_cytation(quartz_wp=quartz)
    cytation.CarrierIn()
    nr_track.return_well_plate_to_nr(wellplate_index,quartz_wp=quartz)  

def run_cytation_program(protocol_file_path):
    plate = cytation.load_protocol(protocol_file_path)
    run = cytation.run_protocol(plate)
    while cytation.protocol_in_progress(run):
        print("Read in progress...")
        time.sleep(10)

def measure_wellplate(protocol_file_path,wellplate_index=0,quartz=False):
    nr_robot.move_home()
    move_wellplate_to_cytation(wellplate_index,quartz=quartz)
    run_cytation_program(protocol_file_path)
    move_wellplate_back_from_cytation(wellplate_index,quartz=quartz)

#TBD. Need methods and imports
def run_photoreactor(dest_vial_position,target_rpm,intensity,duration,reactor_num):
    None

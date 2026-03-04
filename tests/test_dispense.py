#Test to move the vials to different locations with the gripper

import sys

from colorama import init
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

def move_vials(input_vial_status_file, repeats=3):
 
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file,initialize_biotek=False)

    lash_e.nr_robot.move_vial_to_location("SDS_stock", "clamp", 0)
    lash_e.nr_robot.dispense_from_vial_into_vial("SDS_stock", "SDS_stock", volume=0.05, liquid='water', return_vial_home=False)
    lash_e.nr_robot.dispense_from_vial_into_vial("SDS_stock", "SDS_stock", volume=0.85, liquid='water', return_vial_home=False)
    lash_e.nr_robot.return_vial_home("SDS_stock")

    

move_vials("../utoronto_demo/status/calibration_vials_short.csv", repeats=3)
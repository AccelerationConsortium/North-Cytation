#Test to move the vials to different locations with the gripper

import sys

from colorama import init
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

def move_vials(input_vial_status_file, repeats=3):
 
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file,initialize_biotek=False)

    lash_e.nr_robot.check_input_file()

    lash_e.nr_robot.move_vial_to_location("source_vial_b", "photoreactor_array", 0)
    lash_e.nr_robot.dispense_from_vial_into_vial("source_vial_b", "target_vial_2", volume=0.05, liquid='water')

    

move_vials("../utoronto_demo/status/sample_input_vials.csv", repeats=3)
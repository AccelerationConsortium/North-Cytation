#Test to move the vials to different locations with the gripper

import sys

from colorama import init
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

def move_vials(vial_file_path): 
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(vial_file=vial_file_path, initialize_biotek=False)

    # destination = lash_e.nr_robot.get_location(False,'clamp',0)
    # lash_e.nr_robot.c9.goto_safe(destination)
    lash_e.nr_robot.move_vial_to_location(vial_name="vial_1", location="clamp", location_index=0)
    

move_vials("../utoronto_demo/status/sample_input_vials.csv")
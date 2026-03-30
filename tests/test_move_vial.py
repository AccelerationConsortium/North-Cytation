#Test to move the vials to different locations with the gripper

import sys

from colorama import init
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

def move_vials(): 
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(None,initialize_biotek=False)

    destination = lash_e.nr_robot.get_location(False,'clamp',0)
    lash_e.nr_robot.c9.goto_safe(destination)

move_vials("../utoronto_demo/status/sample_input_vials.csv", repeats=3)
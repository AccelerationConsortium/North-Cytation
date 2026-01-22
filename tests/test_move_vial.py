#Test to move the vials to different locations with the gripper

import sys

from colorama import init
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

def move_vials(input_vial_status_file, repeats=3):
 
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file,initialize_biotek=False)

    for i in range (0, repeats):
        lash_e.nr_robot.move_vial_to_location(2,'clamp', 0)
        lash_e.nr_robot.uncap_clamp_vial()
        lash_e.nr_robot.return_vial_home(2)
        lash_e.nr_robot.move_home()

move_vials("../utoronto_demo/status/sample_input_vials.csv", repeats=1)
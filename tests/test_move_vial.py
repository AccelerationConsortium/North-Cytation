#Test to move the vials to different locations with the gripper

import sys

from colorama import init
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

def move_vials(input_vial_status_file, repeats=3):
 
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file,initialize_biotek=False)

    #lash_e.nr_robot.check_input_file()

    for i in range (0, 1):
        lash_e.nr_robot.move_vial_to_location("target_vial",'clamp', 0)

        input("Press Enter to continue...")

        for j in range(0, repeats):
            lash_e.nr_robot.dispense_from_vial_into_vial("target_vial","target_vial",0.100, return_vial_home=False, remove_tip=False)
        lash_e.nr_robot.remove_pipet()

        lash_e.nr_robot.return_vial_home("target_vial")
        lash_e.nr_robot.move_home()

move_vials("../utoronto_demo/status/sample_input_vials.csv", repeats=3)
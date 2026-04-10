#Test to move the vials to different locations with the gripper

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

def move_vials(input_vial_status_file):
 
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file,initialize_biotek=False)

    


    for vial in ("yellow5", None):
        lash_e.nr_robot.move_vial_to_location(vial, "main_8mL_rack", 0)
        lash_e.nr_robot.return_vial_home(vial)
        for i in range (0, 5):
            lash_e.nr_robot.aspirate_from_vial(vial, amount_mL=0.05, liquid='water', post_asp_shake=True)
            lash_e.nr_robot.dispense_into_vial("blue", amount_mL=0.05, liquid='water')
        lash_e.nr_robot.remove_pipet()
        #lash_e.nr_robot.dispense_from_vial_into_vial("water", "red", volume=0.10, liquid='water')

    

move_vials("../utoronto_demo/status/color_mixing_vials.csv")
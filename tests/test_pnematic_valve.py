#Test to move the vials to different locations with the gripper

import sys
import time

from colorama import init
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

def test_pneumatics(valve_num): 
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(vial_file=None, initialize_biotek=False, show_gui=False)

    lash_e.nr_robot.c9.set_output(valve_num, True)
    lash_e.nr_robot.c9.set_output(3, True)
    
    time.sleep(5)
   
    lash_e.nr_robot.c9.set_output(3, False)
    lash_e.nr_robot.c9.set_output(valve_num, False)

#6 to 11
test_pneumatics(valve_num=6)
#Vortex the vial
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import time


VIAL_FILE = "../utoronto_demo/status/sample_input_vials.csv"  # Vials used

#Define your workflow! Make sure that it has parameters that can be changed!
def test_vortex(repeats):
  
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(initialize_biotek=False, show_gui=False)

    #lash_e.nr_robot.move_robot_home()

    for i in range (0, repeats):
        lash_e.nr_robot.c9.open_gripper()
        time.sleep(0.5)
        lash_e.nr_robot.c9.close_gripper()
        time.sleep(0.5)
        

  

test_vortex(100)
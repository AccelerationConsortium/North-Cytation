#Vortex the vial
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import time


VIAL_FILE = "../utoronto_demo/status/sample_input_vials.csv"  # Vials used

#Define your workflow! Make sure that it has parameters that can be changed!
def test_vortex(input_vial_status_file, target_vial_num,vortex_time):
  
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file, initialize_biotek=False)

    for i in range (0, 1):
        lash_e.nr_track.open_gripper()
        time.sleep(0.5)
        lash_e.nr_track.close_gripper()
        time.sleep(0.5)
        

  

test_vortex(VIAL_FILE,2, 3)
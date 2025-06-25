#Vortex the vial
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E


VIAL_FILE = "../utoronto_demo/status/calibration_vials.csv"  # Vials used

#Define your workflow! Make sure that it has parameters that can be changed!
def test_vortex(input_vial_status_file, target_vial_num,vortex_time):
  
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    for i in range (0, 5):
        lash_e.nr_robot.dispense_into_vial('waste_vial', 0, blowout_vol=1.0)

  

test_vortex(VIAL_FILE,2, 3)
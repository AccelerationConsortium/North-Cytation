import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import time

#Define your workflow! 
#In this case we have two parameters: 
def sample_workflow():
  
    INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/sample_capped_vial.txt"

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, initialize_biotek=False, initialize_t8=True)

    #lash_e.spinner.turn_on_vacuum()

    #lash_e.spinner.turn_off_vacuum()

    #lash_e.spinner.open_lid()
    # time.sleep(2)
    lash_e.spinner.close_lid()

    # # lash_e.temp_controller.turn_on_stirring()
    # # time.sleep(3)
    # # lash_e.temp_controller.turn_off_stirring()

    # lash_e.spinner.set_speed(10000)
    # time.sleep(3)
    # lash_e.spinner.stop_spin()

#Execute the sample workflow.
#Specify that we are going to aspirate 0.6 from our two sample vials. We could also set the number of replicates to some other number than 3
#e.g. sample_workflow(aspiration_volume=0.6,replicates=5)
sample_workflow()

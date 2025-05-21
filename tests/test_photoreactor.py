#Turn on and off the photoreactor LED and fan

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import time

VIAL_FILE = "../utoronto_demo/status/sample_capped_vial.txt"  # Vials used

#Define your workflow! Make sure that it has parameters that can be changed!
def sample_workflow(input_vial_status_file, target_vial):
  
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file,initialize_robot=False,initialize_biotek=False,initialize_track=False)

    lash_e.photoreactor.turn_on_reactor_led(reactor_num=1,intensity=100)
    lash_e.photoreactor.turn_on_reactor_fan(reactor_num=1,rpm=600)

    time.sleep(5)

    lash_e.photoreactor.turn_off_reactor_led(1)
    lash_e.photoreactor.turn_off_reactor_led(1)

sample_workflow(VIAL_FILE,0)
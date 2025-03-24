import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd

#Define your workflow! Make sure that it has parameters that can be changed!
def sample_workflow(input_vial_status_file):
  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_vial_status_file, sep=",")
    print(vial_status)

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file,initialize_biotek=False)
    lash_e.photoreactor.turn_off_reactor_led(reactor_num=1)
    lash_e.photoreactor.turn_off_reactor_led(reactor_num=0)

    lash_e.nr_track.origin()
    lash_e.nr_robot.c9.move_z(292)

sample_workflow("../utoronto_demo/status/sample_input_vials.txt")
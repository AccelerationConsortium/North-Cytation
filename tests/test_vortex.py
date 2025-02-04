import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np

VIAL_FILE = "../utoronto_demo/status/sample_capped_vial.txt"  # Vials used

#Define your workflow! Make sure that it has parameters that can be changed!
def sample_workflow(input_vial_status_file, target_vial):
  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_vial_status_file, sep=r"\t", engine="python")
    print(vial_status)
    print( vial_status.at[0, 'cap_status'] == "open_cap")
    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    lash_e.nr_robot.vortex_vial(0,5)

sample_workflow(VIAL_FILE,0)
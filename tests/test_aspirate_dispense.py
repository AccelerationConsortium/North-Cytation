import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np

#Define your workflow! Make sure that it has parameters that can be changed!
def sample_workflow(input_vial_status_file):
  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_vial_status_file, sep=r"\t", engine="python")
    print(vial_status)
    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    lash_e.nr_robot.dispense_from_vial_into_vial(4,0,0.4)
    lash_e.nr_robot.remove_pipet()


sample_workflow("../utoronto_demo/status/color_matching_vials.txt")

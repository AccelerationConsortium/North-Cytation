import re
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np

VIAL_FILE = "../utoronto_demo/status/sample_capped_vial.txt"  # Vials used

#Define your workflow! Make sure that it has parameters that can be changed!
def sample_workflow(input_vial_status_file, target_vial):
  
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    lash_e.nr_robot.dispense_into_vial_from_reservoir(1,target_vial,2.5)

sample_workflow(VIAL_FILE,0)
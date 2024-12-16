import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd

#Define your workflow! Make sure that it has parameters that can be changed!
def sample_workflow(input_vial_status_file):
  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_vial_status_file, sep=r"\t", engine="python")
    print(vial_status)

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    lash_e.nr_robot.reset_robot()
    lash_e.nr_robot.c9.move_z(292)

sample_workflow("../utoronto_demo/status/sample_input_vials.txt")
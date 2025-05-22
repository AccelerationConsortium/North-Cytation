#Test dispensing into a vial from a reservoir on the carousel 

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

VIAL_FILE = "../utoronto_demo/status/sample_capped_vial.csv"  # Vials used

#Define your workflow! Make sure that it has parameters that can be changed!
def sample_workflow(input_vial_status_file, target_vial,reservoir_index,volume):
  
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file,initialize_biotek=False)

    lash_e.nr_robot.dispense_into_vial_from_reservoir(reservoir_index,target_vial,volume)
    lash_e.nr_robot.move_home()

sample_workflow(VIAL_FILE,'Sample',1,0.5)
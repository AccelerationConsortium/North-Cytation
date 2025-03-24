import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd

#Define your workflow! Make sure that it has parameters that can be changed!
def sample_workflow(input_vial_status_file):
 

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    lash_e.measure_wellplate(r"C:\Protocols\Quick_Measurement.prt")
    


sample_workflow("../utoronto_demo/status/sample_input_vials.txt")
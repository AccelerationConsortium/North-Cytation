import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd

#Define your workflow! Make sure that it has parameters that can be changed!
def sample_workflow(input_vial_status_file):
  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_vial_status_file, sep=",")

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)
    data = lash_e.measure_wellplate(r"C:\Protocols\CMC_Fluorescence.prt",[0,1,2],plate_type="48 WELL PLATE")
    #data = lash_e.cytation.run_protocol(r"C:\Protocols\CMC_Fluorescence.prt",[0,1,2,3])
    print (data)
    #lash_e.nr_track.origin()
sample_workflow("../utoronto_demo/status/sample_input_vials.txt")
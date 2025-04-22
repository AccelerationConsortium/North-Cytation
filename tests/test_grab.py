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
    lash_e.measure_wellplate(r"C:\Protocols\CMC_Fluorescence.prt", [0,1,2])
    #data = lash_e.cytation.run_protocol(r"C:\Protocols\CMC_Fluorescence.prt",[12,13,14,15,16,17,18,19,20,21,22,23],plate_type="48 WELL PLATE")
    #data = lash_e.cytation.run_protocol(r"C:\Protocols\test_read_abs.prt",[0,1,2,3,4,5,6,7,8,9,10],plate_type="48 WELL PLATE")
    #data['ratio']=data['1']/data['2']
    #print (data)
    #data.to_csv(f'C:/Users/Imaging Controller/Desktop/CMC/wellplate_data_T20_later.csv', index=False)
    #lash_e.nr_track.origin()
sample_workflow("../utoronto_demo/status/sample_input_vials.csv")
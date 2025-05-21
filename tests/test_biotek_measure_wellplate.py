import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd

def test_measure_wellplate(input_vial_status_file, wells=range(0,3)): #tests track movement of wellplate to and from cytation + obtaining biotek measurements (default: wells 0,1,2)
 
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    lash_e.measure_wellplate(r"C:\Protocols\Quick_Measurement.prt", wells_to_measure=wells) 
    


test_measure_wellplate("../utoronto_demo/status/sample_input_vials.csv")
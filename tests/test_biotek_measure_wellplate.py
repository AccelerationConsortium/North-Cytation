import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd

def test_measure_wellplate(input_vial_status_file): #tests track movement of wellplate to and from cytation + obtaining biotek measurements (default: wells 0,1,2)
 
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    protocols = [r"C:\Protocols\CMC_Absorbance.prt",r"C:\Protocols\CMC_Fluorescence.prt"]

    data = lash_e.measure_wellplate(protocols, [0,1,2,30,31,32], plate_type="48 WELL PLATE")

    #data = lash_e.cytation.run_protocol(protocols, wells=range(0,3), plate_type="48 WELL PLATE")

    print (data)
 
test_measure_wellplate("../utoronto_demo/status/sample_input_vials.csv")
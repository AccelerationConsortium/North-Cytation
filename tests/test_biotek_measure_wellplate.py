import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd

def test_measure_wellplate(input_vial_status_file): #tests track movement of wellplate to and from cytation + obtaining biotek measurements (default: wells 0,1,2)
 
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    for i in range (0, 5):
        print("Run: ", i)
        lash_e.measure_wellplate() 
 
test_measure_wellplate("../utoronto_demo/status/sample_input_vials.csv")
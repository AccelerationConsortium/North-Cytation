import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd

def strip_tuples(d):
    """Convert any (x, None) → x in a flat dict."""
    return {k: (v if not (isinstance(v, tuple) and v[1] is None) else v[0]) for k, v in d.items()}

def test_measure_wellplate(input_vial_status_file): #tests track movement of wellplate to and from cytation + obtaining biotek measurements (default: wells 0,1,2)
 
    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(input_vial_status_file)

    protocol = r"C:\Protocols\Ilya_Measurement.prt"
    #protocol = None

    data = lash_e.measure_wellplate(protocol, [0,1,2])

    #data = lash_e.cytation.run_protocol(protocols, wells=range(0,3), plate_type="48 WELL PLATE")

    print (data)

    # print(strip_tuples(data))

    # data.to_csv("../utoronto_demo/output/test.csv", mode='a', index=False)
 

#asd
test_measure_wellplate("../utoronto_demo/status/sample_input_vials.csv")
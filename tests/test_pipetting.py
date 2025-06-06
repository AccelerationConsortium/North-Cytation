#Test the capping and decapping of vials

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

#Define your workflow! Make sure that it has parameters that can be changed!
def cap_decap():
    lash_e = Lash_E("../utoronto_demo/status/sample_input_vials.csv")    
    lash_e.nr_robot.dispense_from_vial_into_vial(0,2,0.8)

cap_decap()
#Test the capping and decapping of vials

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

#Define your workflow! Make sure that it has parameters that can be changed!
def cap_decap(vial, repeats):
    lash_e = Lash_E("../utoronto_demo/status/sample_input_vials.csv")    
    for i in range(0, repeats):
        lash_e.nr_robot.move_vial_to_location(vial,'clamp',0)
        lash_e.nr_robot.uncap_clamp_vial()
        lash_e.nr_robot.move_home()
        lash_e.nr_robot.recap_clamp_vial()
        lash_e.nr_robot.move_home()
    lash_e.nr_robot.return_vial_home(vial)
    lash_e.nr_robot.move_home()

cap_decap('water',3)
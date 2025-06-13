#Test the capping and decapping of vials

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

def pipet_into_wp():
    lash_e = Lash_E("../utoronto_demo/status/sample_input_vials.csv", simulate = False)

    lash_e.nr_robot.check_input_file()
    lash_e.nr_track.check_input_file()

    lash_e.grab_new_wellplate()
    lash_e.nr_robot.dispense_from_vial_into_vial(0,2,0.8)
    lash_e.nr_robot.aspirate_from_vial(0,0.3)
    lash_e.nr_robot.dispense_into_wellplate([0,1,2], [0.1,0.1,0.1])
    lash_e.nr_robot.remove_pipet()
    lash_e.discard_used_wellplate()

pipet_into_wp()
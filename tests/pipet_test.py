import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

lash_e = Lash_E("../utoronto_demo/status/two_vials.txt")

# lash_e.nr_robot.aspirate_from_vial(0,0.6,track_height=False)
# lash_e.nr_robot.dispense_into_vial(1,0.6)

lash_e.nr_robot.c9.home_robot()
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

lash_e = Lash_E("../utoronto_demo/status/CMC_workflow_input.csv")

lash_e.nr_robot.dispense_from_vial_into_vial(7,2,0.04)

#lash_e.nr_robot.c9.home_robot()
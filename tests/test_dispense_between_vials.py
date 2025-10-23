from os import remove
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

#NOte update to new syntax

input_vial_status_file="../utoronto_demo/status/calibration_vials_short.csv"

lash_e = Lash_E(input_vial_status_file, initialize_biotek=False, simulate=False)

lash_e.nr_robot.check_input_file()

lash_e.nr_robot.aspirate_from_vial('liquid_source_0', 1.0)
input("Enter to continue ")
lash_e.nr_robot.dispense_into_vial('liquid_source_0', 1.0)

# lash_e.nr_robot.dispense_from_vial_into_vial('liquid_source','liquid_source_2', 0.5, remove_tip = False)
# input("Enter to continue ")

# lash_e.nr_robot.dispense_from_vial_into_vial('liquid_source_2','liquid_source', 0.5, remove_tip = True)

lash_e.nr_robot.move_home()
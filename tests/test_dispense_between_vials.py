import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

#NOte update to new syntax

input_vial_status_file="../utoronto_demo/status/sample_capped_vials.csv"

lash_e = Lash_E(input_vial_status_file, initialize_biotek=False, simulate=False)

lash_e.nr_robot.check_input_file()

lash_e.nr_robot.dispense_from_vial_into_vial('Sample_B','Sample_A', 1.2, use_safe_location=True)

lash_e.nr_robot.remove_pipet()

lash_e.nr_robot.move_home()
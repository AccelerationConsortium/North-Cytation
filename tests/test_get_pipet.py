import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

#NOte update to new syntax

input_vial_status_file="../utoronto_demo/status/color_matching_vials.txt"

lashe = Lash_E(input_vial_status_file, initialize_biotek=False)

lashe.nr_robot.check_input_file()

def test_get_pipet(index_list):
    for i in index_list:
        lashe.nr_robot.get_pipet(i)
        lashe.nr_robot.move_home()
        lashe.nr_robot.remove_pipet()
        # nr.move_home()
        # input("Waiting to move tip...")
        # nr.remove_pipet()

# test_get_pipet([lashe.nr_robot.HIGHER_PIPET_ARRAY_INDEX]*48)

test_get_pipet([1])
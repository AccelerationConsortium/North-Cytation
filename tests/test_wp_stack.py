import sys


sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo")
from master_usdl_coordinator import Lash_E

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/sample_input_vials.csv"
SIMULATE = False

lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, initialize_biotek=False, simulate=SIMULATE)

def test_wp_transfer_and_pipetting():
    try:
        #DEMO
        lash_e.nr_robot.check_input_file()
        lash_e.nr_track.check_input_file()
        wp_type = lash_e.nr_track.CURRENT_WP_TYPE

        lash_e.grab_new_wellplate()

        lash_e.nr_robot.aspirate_from_vial(source_vial_name=0, amount_mL=0.3) #gets pipet tip
        lash_e.nr_robot.dispense_into_wellplate(dest_wp_num_array=[0,1,2], amount_mL_array=[0.1,0.1,0.1], well_plate_type=wp_type)
        lash_e.nr_robot.remove_pipet()

        lash_e.discard_used_wellplate()
        

    except KeyboardInterrupt:
        lash_e.nr_track.c9.move_axis(6, 0, vel=5)

def test_wp_stack(num_wp): #num_wp = number of wellplates to move
    lash_e.nr_track.check_input_file()
    for i in range(num_wp):
        lash_e.grab_new_wellplate()
        lash_e.discard_used_wellplate()

test_wp_stack(1) #working!
#test_wp_transfer_and_pipetting() #working!
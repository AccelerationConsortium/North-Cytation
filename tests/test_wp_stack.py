import sys


sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo")
from master_usdl_coordinator import Lash_E

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/sample_capped_vials.csv"
SIMULATE = False

lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, initialize_biotek=True, simulate=SIMULATE)

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


def test_wp_movement(well_plate_type):
    #lash_e.nr_track.check_input_file()
    lash_e.nr_track.grab_wellplate_from_location('pipetting_area', wellplate_type=well_plate_type, waypoint_locations=['max_height', 'cytation_safe_area'])
    lash_e.nr_track.move_through_path(['cytation_safe_area'])
    lash_e.cytation.CarrierOut()
    lash_e.nr_track.release_wellplate_in_location('cytation_tray', wellplate_type=well_plate_type)


    #lash_e.nr_track.grab_wellplate_from_location('pipetting_area', wellplate_type=well_plate_type, waypoint_locations=['max_height', 'cytation_safe_area'])
    #lash_e.nr_track.release_wellplate_in_location('pipetting_area', wellplate_type=well_plate_type)
    #lash_e.move_wellplate_to_cytation(plate_type=well_plate_type)
    #lash_e.move_wellplate_back_from_cytation(plate_type=well_plate_type)

def move_back(well_plate_type):
    lash_e.move_wellplate_back_from_cytation(plate_type=well_plate_type)

test_wp_movement('48 WELL PLATE') #workcing!
input()
move_back('48 WELL PLATE')

#lash_e.nr_track.move_through_path(['max_height'])
#test_wp_stack(3) #working!
#test_wp_transfer_and_pipetting() #working!
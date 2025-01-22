import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np

#Define your workflow! Make sure that it has parameters that can be changed!
def create_initial_colors(measurement_file):
    lash_e.grab_new_wellplate()
    
    well_volume = 0.25

    lash_e.nr_robot.aspirate_from_vial(0,well_volume)
    lash_e.nr_robot.dispense_into_wellplate(['A1'], [well_volume])
    lash_e.nr_robot.finish_pipetting()

    initial_wells = np.linspace(1,6,5)
    initial_volumes = get_inputs_from_optimizer() #TBD
    active_vials = np.linspace(1,5,4)

    lash_e.nr_robot.dispense_from_vials_into_wellplate(well_plate_df,active_vials)
    lash_e.nr_robot.finish_pipetting()

    lash_e.measure_wellplate(measurement_file)

def analyze_data():
    return None

def find_closer_color_match(start_index,measurement_file):

    wells = np.linspace(start_index,start_index+6,6)
    volumes = get_inputs_from_optimizer() #TBD
    active_vials = np.linspace(1,5,4)

    lash_e.nr_robot.dispense_from_vials_into_wellplate(well_plate_df,active_vials)
    lash_e.nr_robot.finish_pipetting()

    lash_e.measure_wellplate(measurement_file)

input_vial_status_file="../utoronto_demo/status/color_matching_vial.txt"
vial_status = pd.read_csv(input_vial_status_file, sep=r"\t", engine="python")
print(vial_status)
input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

#Initialize the workstation, which includes the robot, track, cytation and photoreactors
lash_e = Lash_E(input_vial_status_file)

create_initial_colors()
results = analyze_data(initial=True)
status_complete = False
while status_complete == False:
    find_closer_color_match()
    results = analyze_data(initial=False)
import sys

from torch import ge

import pandas as pd

sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo")
from master_usdl_coordinator import Lash_E

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/sample_capped_vials.csv"
SIMULATE = False
protocol = r"C:\Protocols\300_900_sweep.prt"

lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, initialize_biotek=True, simulate=SIMULATE)

def move_lid_to_wellplate():
    lash_e.nr_track.grab_wellplate_from_location('lid_storage', wellplate_type='quartz_lid', waypoint_locations=['cytation_safe_area'])
    lash_e.nr_track.release_wellplate_in_location('pipetting_area', wellplate_type='quartz_lid')

def remove_lid_from_wellplate():
    lash_e.nr_track.grab_wellplate_from_location('pipetting_area', wellplate_type='quartz_lid', waypoint_locations=['cytation_safe_area'])
    lash_e.nr_track.release_wellplate_in_location('lid_storage', wellplate_type='quartz_lid')

for i in range (0, 3):
    move_lid_to_wellplate()

    # lash_e.nr_track.grab_wellplate_from_location('pipetting_area', wellplate_type='quartz')
    # lash_e.nr_track.release_wellplate_in_location('pipetting_area', wellplate_type='quartz')
    data = lash_e.measure_wellplate(protocol, [0], plate_type='quartz')

    print (data)
    
    # Save data as CSV
    # if data is not None:
    #     df = pd.DataFrame(data)
    #     df.to_csv(f"C:\\Users\\Imaging Controller\\Desktop\\SQ\\quartz_wellplate_test\\run_{i}.csv", index=False)

    remove_lid_from_wellplate()

    # lash_e.nr_track.grab_wellplate_from_location('pipetting_area', wellplate_type='quartz_lid')
    # lash_e.nr_track.release_wellplate_in_location('lid_storage', wellplate_type='quartz_lid')

# for i in range (0, 1):
#     lash_e.measure_wellplate(None, plate_type='quartz')
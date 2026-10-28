import sys

from torch import ge

import pandas as pd

sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo")
from master_usdl_coordinator import Lash_E

INPUT_VIAL_STATUS_FILE = None
SIMULATE = False
protocol = None

lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, initialize_biotek=False, simulate=SIMULATE, show_gui=False)

def move_lid_to_wellplate():
    lash_e.nr_track.grab_lid_from_location('lid_storage_96', wellplate_type='96 WELL PLATE', waypoint_locations=['cytation_safe_area'])
    lash_e.nr_track.release_wellplate_in_location('pipetting_area', wellplate_type='96 WELL PLATE')

def remove_lid_from_wellplate():
    lash_e.nr_track.grab_lid_from_location('pipetting_area', wellplate_type='96 WELL PLATE', waypoint_locations=['cytation_safe_area'])
    lash_e.nr_track.release_wellplate_in_location('lid_storage_96', wellplate_type='96 WELL PLATE')

for i in range (0, 3):
    move_lid_to_wellplate()

    #data = lash_e.measure_wellplate(protocol, [0])
    
    remove_lid_from_wellplate()

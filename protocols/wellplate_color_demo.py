
import numpy as np
import pandas as pd
import math
import sys
import os
sys.path.append("../utoronto_demo")

from north import NorthC9
from North_Safe import North_Robot
from North_Safe import North_Track
from Locator import *

#needed files: 1. vial_status 2.wellplate_recipe
VIAL_FILE = "../utoronto_demo/status/vials_color.txt" #txt

BLUE_DIMS = [20,85] #Not accurate

vial_df = pd.read_csv(VIAL_FILE, delimiter='\t')

c9 = NorthC9('A', network_serial='AU06CNCF')

nr = North_Robot(c9,vial_df)
                
nr.set_pipet_tip_type(BLUE_DIMS,0)
nr.PIPETS_USED = [0,0]

MIN_DISP = 20
MAX_DISP = 200

data_colors = np.random.random((96,5))

data_colors_uL = np.divide(data_colors, np.sum(data_colors, 1).reshape((96,1))) * 100

sum_colors = np.sum(data_colors_uL,0)/1000

print(sum_colors)

try:
    c9.move_z(300)
    nr.reset_after_initialization()

    for i in range (0,5):
        nr.move_vial_to_clamp(i)
        nr.uncap_clamp_vial()
        vol_needed = sum_colors[i]
        vol_dispensed = 0
        array_index = 0
        while vol_dispensed <= vol_needed:
            dispense_vol = min(1, vol_needed - vol_dispensed)
            nr.aspirate_from_vial(i,dispense_vol)

            well_plate_array = np.arange(0,96,1)
            dispense_array = data_colors_uL[:, i]

            nr.dispense_into_wellplate(well_plate_array,dispense_array,1,dispense_type="touch")

            vol_dispensed += dispense_vol
        nr.remove_pipet()
        nr.recap_clamp_vial()
        nr.return_vial_from_clamp(3)

    c9.move_z(300)
    
except KeyboardInterrupt:
    c9 = None
    os._exit(0) 

#for i in range (0, 5):


c9 = None
os._exit(0)
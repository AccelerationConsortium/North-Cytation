import North_Safe
import numpy as np
import pandas as pd

#Input data
POLYMER_VOLUME = 0.1 #how much polymer solution in mL?
ANTISOLVENT_VOLUME = 4 #How much antisolvent solution in mL?
pipet_length = [0.5, 0.2] #Need to measure these distances

vial_df = pd.read_csv("vial_status_precip.txt", delimiter='\t', index_col='vial index') #Edit this

nr = North_Safe.North_Robot(vial_df, pipet_length)

nr.reset_after_initialization()
nr.move_vial_to_clamp(0)
nr.uncap_clamp_vial()

nr.pipet_from_vial_into_vial(1, 0, POLYMER_VOLUME, wait_over_vial=True, track_height=True, dispense_speed=30)

nr.remove_pipet()
nr.recap_clamp_vial()

nr.vortex_vial(0, 100000)

nr.return_vial_from_clamp(0)
nr.c9.move_z(292)
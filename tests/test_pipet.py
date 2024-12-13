import sys
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo")
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo\\status")
# print(sys.path)
import North_Safe
from Locator import *
from north import NorthC9
import pandas as pd
import time

VIAL_FILE = "../utoronto_demo/status/vials_color.txt"  # Vials used
PIPET_FILE = "../utoronto_demo/status/pipets.txt"

DEFAULT_DIMS = [25,85]

c9 = NorthC9('A', network_serial='AU06CNCF')
nr = North_Safe.North_Robot(c9,VIAL_FILE,PIPET_FILE)
nr_track = North_Safe.North_Track(c9)
nr.set_robot_speed(20)
nr.set_pipet_tip_type(DEFAULT_DIMS, 0)
nr.reset_after_initialization() ##turn back on the home carousel & zerosscale

#input()

try:
    #nr.c9.goto_safe(vial_clamp_pip)
     vial_indices = [0,1] #This should be determined in step 1
     well_plate_df = pd.DataFrame()
     well_plate_df['well_index']= ['C7','C8']
     well_plate_df['Vial_0'] = [0,0.1]
     well_plate_df['Vial_1'] = [0.1,0.1]

     print(well_plate_df)

     #nr.get_pipet()
    
    
    # nr.remove_pipet()

     nr.dispense_from_vials_into_wellplate(well_plate_df, vial_indices)

     nr.c9.move_z(292)
except KeyboardInterrupt as e:
    print(e)
    nr.reset_robot()


#testing well-plate position -- test with normal pipette tip
# nr.get_pipet()
# nr.c9.goto_safe(well_plate_grid[0])
# time.sleep(3)
# nr.remove_pipet()


#**Testing vial to vial transfer

# nr.move_vial_to_clamp(0)
# nr.uncap_clamp_vial()
# 
# nr.get_pipet()
# 
# nr.c9.set_pump_speed(0,25)
# nr.aspirate_from_vial(1, 0.2)
# nr.dispense_into_vial(0, 0.2)
# nr.remove_pipet()
# 
# nr.recap_clamp_vial()
# nr.return_vial_from_clamp(0)
# nr.c9.move_z(292)

#print("**WEIGHTS:", weights)



#**TEST VIAL TO WP TRANSFER
# nr.move_vial_to_clamp(0)
# nr.uncap_clamp_vial()
# nr.c9.close_clamp()
# 
# nr.get_pipet()
# 
# nr.aspirate_from_vial(0, 0.2)
# nr.dispense_into_wellplate([12,0],0.1, 2)
# nr.remove_pipet()
# 
# nr.recap_clamp_vial()
# nr.return_vial_from_clamp(0)
# nr.c9.move_z(292)


#**Testing drips:
# nr.reset_after_initialization()
# test_vials = [0] #0 = Water, 1 = EA [0,1,2]
# amounts = [0.005, 0.007, 0.01]
# for t in test_vials:
#     print("moving vial ", t)
#     nr.move_vial_to_clamp(t)
#     nr.uncap_clamp_vial()
#     nr.c9.close_clamp()
# 
#     nr.get_pipet()
#     nr.c9.set_pump_speed(0, 20) #slow speed for drops (hopefully)
#     for amount in amounts:
#         nr.aspirate_from_vial(t, amount)
#         nr.dispense_into_vial(t, amount)
# 
#     nr.remove_pipet()
#     nr.recap_clamp_vial()
#     nr.return_vial_from_clamp(t)





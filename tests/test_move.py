import North_Safe
from Locator import *
import pandas as pd
import time

VIAL_FILE = "vial_status_wellplate.txt"

#will try to work
BLUE_DIMS = [20,77]
DEFAULT_DIMS = [25,85]
FILTER_DIMS = [24,98]

vial_df = pd.read_csv(VIAL_FILE, delimiter='\t', index_col='vial index')
nr = North_Safe.North_Robot(vial_df)
nr.set_robot_speed(8)

nr.set_pipet_tip_type(DEFAULT_DIMS, 1) #only works with default dims (because of going to location -- not height asdjusted) & bottom row pipettes cleared!!

nr.reset_after_initialization() ##turn back on the home carousel & zerosscale


##**TEST CAPPING**
# nr.move_vial_to_clamp(0) #open clamp at the end
# nr.uncap_clamp_vial() #opens clamp at the end 
# 
# nr.recap_clamp_vial()
# nr.c9.open_clamp()
# #nr.return_vial_from_clamp(0)
# nr.c9.move_z(292)

##**TEST vortexing before pipette
# nr.move_vial_to_clamp(5) #open clamp at the end
# nr.vortex_vial(5, 150000)
# nr.uncap_clamp_vial() #opens clamp at the end 
# nr.recap_clamp_vial()
# nr.c9.close_clamp()
# nr.return_vial_from_clamp(5)


##**TEST Quartz Wellplate
nr.get_pipet()
nr.c9.goto_safe(well_plate_grid[13])
time.sleep(2)

nr.move_rel_z(-5)
nr.move_rel_x(1.75)
time.sleep(2)

nr.move_rel_x(-3.5)
time.sleep(2)

nr.remove_pipet()



#**TESTING TOUCH Pipette location
# nr.get_pipet()
# for i in range(2):
#     nr.c9.goto_safe(well_plate_grid[i+40])
#     nr.move_rel_z(-5)
#     nr.move_rel_x(1.75)
#     #nr.move_rel_x(1)
#     time.sleep(3)
#     
# nr.remove_pipet()


#**TESTING move_rel_x, y and z
# nr.c9.move_z(292)
# 
# nr.c9.move_xy(100,200)
# print(nr.c9.n9_fk(nr.c9.get_axis_position(0), nr.c9.get_axis_position(1), nr.c9.get_axis_position(2)))
# 
# nr.move_rel_x(100)
# print(nr.c9.n9_fk(nr.c9.get_axis_position(0), nr.c9.get_axis_position(1), nr.c9.get_axis_position(2)))
# 
# nr.move_rel_y(-20)
# print(nr.c9.n9_fk(nr.c9.get_axis_position(0), nr.c9.get_axis_position(1), nr.c9.get_axis_position(2)))
# 
# nr.move_rel_y(40)
# print(nr.c9.n9_fk(nr.c9.get_axis_position(0), nr.c9.get_axis_position(1), nr.c9.get_axis_position(2)))
# 
# nr.move_rel_z(-30)
# nr.move_rel_z(10)






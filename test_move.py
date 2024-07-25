import North_Safe
from Locator import *
import pandas as pd
import time

VIAL_FILE = "vial_status_wellplate - test.txt"

#will try to work
BLUE_DIMS = [20,77]
DEFAULT_DIMS = [25,85]
FILTER_DIMS = [24,98]

vial_df = pd.read_csv(VIAL_FILE, delimiter='\t', index_col='vial index')
nr = North_Safe.North_Robot(vial_df)
nr.set_robot_speed(8)

nr.set_pipet_tip_type(DEFAULT_DIMS, 0)

nr.reset_after_initialization() ##turn back on the home carousel & zerosscale


#**TESTING TOUCH Pipette location
nr.get_pipet()
for i in range(2):
    nr.c9.goto_safe(well_plate_grid[i])
    nr.move_rel_z(-5)
    nr.move_rel_x(-4)
    time.sleep(5)
    
nr.remove_pipet()


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






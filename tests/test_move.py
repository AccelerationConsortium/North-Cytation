import sys
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo")
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo\\status")
import os

from north import NorthC9
import North_Safe
from Locator import *
import pandas as pd
import time

VIAL_FILE = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\status\sample_input_vials.txt"
#will try to work
BLUE_DIMS = [20,77]
DEFAULT_DIMS = [25,85]
FILTER_DIMS = [24,98]

c9 = NorthC9('A', network_serial='AU06CNCF')

#vial_df = pd.read_csv(VIAL_FILE, delimiter='\t', index_col='vial index')
#vial_df = pd.read_csv(VIAL_FILE, sep=r"\t", engine="python")
nr = North_Safe.North_Robot(c9,VIAL_FILE)
nr.set_robot_speed(20)

c9.move_z(300)
nr.reset_after_initialization() ##turn back on the home carousel & zerosscale

# nr.set_pipet_tip_type(BLUE_DIMS, 0) #only works with default dims (because of going to location -- not height asdjusted) & bottom row pipettes cleared!!

# nr.get_pipet()

# for i in range(3):
#     nr.c9.goto_safe(well_plate_new_grid[i])
# nr.c9.goto_safe(well_plate_new_grid[0])
# time.sleep(2)
# nr.c9.goto_safe(well_plate_new_grid[1])
# time.sleep(2)
# nr.c9.goto_safe(well_plate_new_grid[12])
# time.sleep(2)
# nr.c9.goto_safe(well_plate_new_grid[18])
# time.sleep(2)

# nr.set_robot_speed(50)
# nr.remove_pipet()

# c9.move_z(300)

# nr.c9 = None
# os._exit(0)


##**TEST CAPPING**
input("enter when ready")

try:
    for i in range(1):
        nr.move_vial_to_clamp(2) #open clamp at the end
        nr.uncap_clamp_vial() #opens clamp at the end 

        nr.recap_clamp_vial()
        nr.c9.open_clamp()
        nr.return_vial_from_clamp()
        nr.c9.move_z(292)
        time.sleep(5)
except KeyboardInterrupt:
    c9 = None

##**TEST vortexing before pipette
# nr.move_vial_to_clamp(5) #open clamp at the end
# nr.vortex_vial(5, 150000)
# nr.uncap_clamp_vial() #opens clamp at the end 
# nr.recap_clamp_vial()
# nr.c9.close_clamp()
# nr.return_vial_from_clamp(5)


##**TEST Quartz Wellplate
# nr.get_pipet()
# nr.c9.goto_safe(well_plate_grid[13])
# time.sleep(2)
# 
# nr.move_rel_z(-5)
# nr.move_rel_x(1.75)
# time.sleep(2)
# 
# nr.move_rel_x(-3.5)
# time.sleep(2)
# 
# nr.remove_pipet()



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






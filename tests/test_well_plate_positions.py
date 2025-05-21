import sys
sys.path.append("../utoronto_demo")
from north import NorthC9
from Locator import *
import time


c9 = NorthC9('A', network_serial='AU06CNCF')
c9.default_vel = 5  # percent

def goto_single_wp_position(well_index):
    c9.goto_safe(well_plate_new_grid[well_index])

def goto_all_wp_positions():
    for i in range(1,96):
        c9.goto(well_plate_new_grid[i])
        time.sleep(2)


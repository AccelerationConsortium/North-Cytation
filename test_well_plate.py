from north import NorthC9
from Locator import *
import time

c9 = NorthC9('A', network_serial='AU06CNCF')
c9.default_vel = 25  # percent

#c9.goto_safe(p_capture_grid[0])

c9.goto_safe(well_plate_grid[0])

for i in range (1, 96):
    c9.goto(well_plate_grid[i])
    time.sleep(0.5)

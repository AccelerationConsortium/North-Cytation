from north import NorthC9
from Locator import *
import time

c9 = NorthC9('A', network_serial='AU06CNCF')

# for i in range (0,3):
#     c9.goto(well_plate_new_grid[i])
#     time.sleep(2)

from North_Safe import North_Track

nr_track = North_Track(c9)
# nr_track.return_well_plate_to_nr(1, quartz_wp=True)
nr_track.origin()


#TODO: update

import sys
sys.path.append("../utoronto_demo")
from biotek import Biotek
from North_Safe import North_Robot
from North_Safe import North_Track
from north import NorthC9
import os

#North Robot
c9 = NorthC9('A', network_serial='AU06CNCF')
nr_track = North_Track(c9)
nr_robot = North_Robot(c9)

try:
    nr_track.open_gripper()
    nr_track.close_gripper()
except KeyboardInterrupt:
    os._exit(0)
    c9 = None

c9 = None
os._exit(0)
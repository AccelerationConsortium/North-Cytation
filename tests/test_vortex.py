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


c9 = NorthC9('A', network_serial='AU06CNCF')
nr = North_Safe.North_Robot(c9,VIAL_FILE,PIPET_FILE)
nr_track = North_Safe.North_Track(c9)
nr.set_robot_speed(20)

nr.reset_after_initialization() ##turn back on the home carousel & zerosscale

try:
    #checking get_pipet (change pipette tip type for the runs)
    nr_track.origin()
    nr.reset_robot()

    for i in range (1, 4):
        nr.vortex_vial(0, i)
    nr.c9.move_z(292)
except Exception as e:
    print(e)
    nr.reset_robot()


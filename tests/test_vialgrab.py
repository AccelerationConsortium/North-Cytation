import sys
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo")
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo\\status")
# print(sys.path)
import North_Safe
from Locator import *
from north import NorthC9
import pandas as pd
import time
import photoreactor_controller as pr

VIAL_FILE = "../utoronto_demo/status/vials_color.txt"  # Vials used
PIPET_FILE = "../utoronto_demo/status/pipets.txt"


c9 = NorthC9('A', network_serial='AU06CNCF')
nr = North_Safe.North_Robot(c9,VIAL_FILE,PIPET_FILE)
nr_track = North_Safe.North_Track(c9)
nr.set_robot_speed(20)
nr.reset_after_initialization() ##turn back on the home carousel & zerosscale

#checking get_pipet (change pipette tip type for the runs)
# nr.goto_location_if_not_there(rack[0]) #move to vial
# nr.c9.close_gripper() #grip vial
# nr.c9.move_z(292)

nr.move_vial_to_photoreactor(0,1)
pr.run_photoreactor(600,10,100,1)
nr.return_vial_from_photoreactor(0,1)

nr.move_vial_to_photoreactor(1,2)
pr.run_photoreactor(600,10,100,2)
nr.return_vial_from_photoreactor(1,2)
nr.reset_robot()

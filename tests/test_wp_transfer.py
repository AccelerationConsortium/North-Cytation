
import sys
sys.path.append("../utoronto_demo")
from biotek import Biotek
from North_Safe import North_Robot
from North_Safe import North_Track
from north import NorthC9
import os

def transfer_wp_to_cytation_from_nr():
    nr_track.grab_well_plate_from_nr()
    nr_track.move_gripper_to_cytation()
    gen5.CarrierOut()
    nr_track.release_well_plate_in_cytation()
    gen5.CarrierIn()

def transfer_wp_to_nr_from_cytation(): 
    gen5.CarrierOut()
    nr_track.grab_well_plate_from_cytation()
    gen5.CarrierIn()
    nr_track.return_well_plate_to_nr()

# Cytation 5
readerType = 21
ComPort = 4
appName = 'Gen5.Application'
gen5 = Biotek(readerType, ComPort, appName)

#North Robot
c9 = NorthC9('A', network_serial='AU06CNCF')
nr = North_Robot(c9)
nr_track = North_Track(c9)

try:
    transfer_wp_to_cytation_from_nr()
    transfer_wp_to_nr_from_cytation()
except KeyboardInterrupt:
    os._exit(0)
    gen5.close()
    nr_track.c9 = None

gen5.close()
nr_track.c9 = None
os._exit(0)


'''
nr_track.grab_well_plate_from_nr()
nr_track.move_gripper_to_cytation()
gen5.CarrierOut()
input("Please open Cytation...")
nr_track.release_well_plate_in_cytation()
nr_track.grab_well_plate_from_cytation()
gen5.CarrierIn()
input("Please close Cytation")
nr_track.return_well_plate_to_nr()
'''
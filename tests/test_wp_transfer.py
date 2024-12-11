
import sys
sys.path.append("../utoronto_demo")
from biotek import Biotek
from North_Safe import North_Robot
from North_Safe import North_Track
from north import NorthC9
import os
#from prefect import flow,serve

'''
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
'''

def open_and_close():
    # Cytation 5
    readerType = 21
    ComPort = 4
    appName = 'Gen5.Application'
    gen5 = Biotek()

    #North Robot
    c9 = NorthC9('A', network_serial='AU06CNCF')
    nr = North_Robot(c9)
    nr_track = North_Track(c9)
    nr_robot = North_Robot(c9)

    try:
        nr_track.grab_well_plate_from_nr(0)
        #nr_track.return_well_plate_to_nr(1)
        nr_track.move_gripper_to_cytation()
        gen5.CarrierOut()
        nr_track.release_well_plate_in_cytation()
        gen5.CarrierIn()
        gen5.CarrierOut()
        nr_track.grab_well_plate_from_cytation()
        gen5.CarrierIn()
        nr_track.return_well_plate_to_nr(0)
    except KeyboardInterrupt:
        #os._exit(0)
        #gen5.close()
        nr_track.c9 = None

    #gen5.close()
    c9 = None
    #os._exit(0)

if __name__ == "__main__":
    open_and_close()
    #flow1 = open_and_close.to_deployment(name="open_and_close")
    #serve(flow1)
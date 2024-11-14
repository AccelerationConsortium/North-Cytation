
import sys
sys.path.append("../")
from biotek import Biotek
from North_Safe import North_Robot
from North_Safe import North_Track
from north import NorthC9
from prefect import flow,serve
import time

def move_wp_to_cyt_to_test_then_back(well_plate_num=1,lid_storage_num=2):
    
    # Cytation 5 connection
    gen5 = Biotek()

    #North Robot
    c9 = NorthC9('A', network_serial='AU06CNCF')
    nr_track = North_Track(c9)

    try:
        #Set the speed for the track (horizontal and vertical)
        nr_track.set_horizontal_speed(50)
        nr_track.set_vertical_speed(50)

        #Move lid
        nr_track.grab_well_plate_from_nr(well_plate_num,grab_lid=True)
        nr_track.return_well_plate_to_nr(lid_storage_num,grab_lid=True)

        #Move well-plate to cytation
        nr_track.grab_well_plate_from_nr(well_plate_num)
        nr_track.move_gripper_to_cytation()
        gen5.CarrierOut()
        nr_track.release_well_plate_in_cytation()
        gen5.CarrierIn() 

        #Run cytation protocol
        '''
        plate = gen5.load_protocol(r"C:\Protocols\Stanley_Degradation.prt")
        run = gen5.run_protocol(plate)
        while gen5.protocol_in_progress(run):
            print("Read in progress...")
            time.sleep(10)
        '''

        #Return well-plate to storage
        gen5.CarrierOut()
        nr_track.grab_well_plate_from_cytation()
        gen5.CarrierIn()
        nr_track.return_well_plate_to_nr(well_plate_num)

        #Return lid
        nr_track.grab_well_plate_from_nr(lid_storage_num,grab_lid=True)
        nr_track.return_well_plate_to_nr(well_plate_num,grab_lid=True)

        nr_track.origin()

        #Should move track away from north robot
        gen5.close()
        c9 = None

    except Exception as e:
        print("Error occured during operation",e)
        gen5.close()
        c9 = None
    except KeyboardInterrupt:
        gen5.close()
        c9 = None

if __name__ == "__main__":
    move_wp_to_cyt_to_test_then_back()
    #flow1 = open_and_close.to_deployment(name="open_and_close")
    #serve(flow1)
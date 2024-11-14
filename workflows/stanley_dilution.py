import sys
sys.path.append("../utoronto_demo")
from North_Safe import North_Robot
from North_Safe import North_Track
from north import NorthC9
import time
import pandas as pd

def main():
    

    c9 = NorthC9('A', network_serial='AU06CNCF')
    VIAL_FILE = "../utoronto_demo/status/water.txt" #Vials used
    vial_df = pd.read_csv(VIAL_FILE, delimiter='\t')
    nr_robot = North_Robot(c9, vial_df)
    nr_track = North_Track(c9)

    nr_robot.remove_pipet()
    nr_robot.reset_after_initialization()
    nr_robot.set_pipet_tip_type(North_Robot.BLUE_DIMS,0) #This isn't really accurate

    #nr_track.grab_well_plate_from_nr(0)
    #nr_track.return_well_plate_to_nr(1)  

    nr_robot.aspirate_from_vial(0, 0.1)

    nr_robot.set_pipet_tip_type(North_Robot.DEFAULT_DIMS,0) #Cheating here
    nr_robot.dispense_into_wellplate([4,5,6],[0.03,0.02,0.01])

    nr_robot.save_vial_status("../utoronto_demo/status/water.txt")

    nr_robot.remove_pipet()

    

main()




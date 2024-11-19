import sys

sys.path.append("../utoronto_demo")
from North_Safe import North_Robot
from North_Safe import North_Track
from north import NorthC9
import time
import pandas as pd


def main():

    c9 = NorthC9("A", network_serial="AU06CNCF")
    VIAL_FILE = "../utoronto_demo/status/stanley_polyelectrolyte.csv"  # Vials used
    vial_df = pd.read_csv(VIAL_FILE, delimiter=",")
    nr_robot = North_Robot(c9, vial_df, [0,1,3,4,6,7])
    nr_track = North_Track(c9)

    #nr_robot.remove_pipet()
    nr_robot.reset_after_initialization()
    nr_robot.set_pipet_tip_type(North_Robot.BLUE_DIMS, 0)  # This isn't really accurate

    # nr_track.grab_well_plate_from_nr(0)
    # nr_track.return_well_plate_to_nr(1)

    # Dilute polyelectrolyte from 0.5mg/mL to 0.05mg/mL
    nr_robot.aspirate_from_vial(1, 0.1)  # aspirate from polyelectrolyte
    nr_robot.set_pipet_tip_type(North_Robot.DEFAULT_DIMS, 0)  # Cheating here
    nr_robot.dispense_into_vial(3, 0.1)  # dispense polyelectrolyte into vial 3
    nr_robot.remove_pipet()
    nr_robot.set_pipet_tip_type(North_Robot.BLUE_DIMS, 0)  # This isn't really accurate

    # Add water to dilute polyelectrolyte
    nr_robot.aspirate_from_vial(0, 0.9)  # aspirate from HPLC water with new tip
    nr_robot.set_pipet_tip_type(North_Robot.DEFAULT_DIMS, 0)  # Cheating here
    nr_robot.dispense_into_vial(3, 0.9)  # dispense polyelectrolyte into vial 3
    nr_robot.remove_pipet()
    nr_robot.set_pipet_tip_type(North_Robot.BLUE_DIMS, 0)  # This isn't really accurate

    # Dispense blank (HPLC_Water) into wellplate
    nr_robot.aspirate_from_vial(0, 0.25)  # aspirate from HPLC water
    nr_robot.set_pipet_tip_type(North_Robot.DEFAULT_DIMS, 0)  # Cheating here
    nr_robot.dispense_into_wellplate([1], [0.25])  # dispense HPLC water into well plate
    nr_robot.remove_pipet()
    nr_robot.set_pipet_tip_type(North_Robot.BLUE_DIMS, 0)  # This isn't really accurate

    # Dispense polyelectrolyte into well plate
    nr_robot.aspirate_from_vial(3, 0.6)  # aspirate from polyelectrolyte
    nr_robot.set_pipet_tip_type(North_Robot.DEFAULT_DIMS, 0)  # Cheating here
    nr_robot.dispense_into_wellplate(
        [2, 3, 4], [0.25, 0.2, 0.15]
    )  # dispense polyelectrolyte into well plate
    nr_robot.remove_pipet()
    nr_robot.set_pipet_tip_type(North_Robot.BLUE_DIMS, 0)  # This isn't really accurate

    # Dispense acid into well plate
    nr_robot.aspirate_from_vial(2, 0.15)  # aspirate from polyelectrolyte
    nr_robot.set_pipet_tip_type(North_Robot.DEFAULT_DIMS, 0)  # Cheating here
    nr_robot.dispense_into_wellplate(
        [3, 4], [0.05, 0.1]
    )  # dispense polyelectrolyte into well plate
    
    nr_robot.save_vial_status("../utoronto_demo/status/stanley_polyelectrolyte.csv")
    nr_robot.remove_pipet()


main()

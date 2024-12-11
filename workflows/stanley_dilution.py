import sys

sys.path.append("../utoronto_demo")
from North_Safe import North_Robot
from North_Safe import North_Track
from north import NorthC9
import time
import pandas as pd
from biotek import Biotek 

def main():

    c9 = NorthC9("A", network_serial="AU06CNCF")
    VIAL_FILE = "../utoronto_demo/status/stanley_polyelectrolyte.csv"  # Vials used
    #vial_df = pd.read_csv(VIAL_FILE, delimiter=",")
    nr_robot = North_Robot(c9, VIAL_FILE)
    nr_track = North_Track(c9)

    vial_dict = {"A1": "H2O_blank", "A2": "20pct_HCL_blank", "A3": "40pct_HCL_blank",
                 "B1": "AL-1-47D", "B2": "AL-1-47D-20pct_HCL", "B3": "AL-1-47D-40pct_HCL",
                 "C1": "AL-1-45D-BD", "C2": "AL-1-45D-BD-20pct_HCL", "C3": "AL-1-45D-BD-40pct_HCL",
                 "D1": "AL-1-45D-AD", "D2": "AL-1-45D-AD-20pct_HCL", "D3": "AL-1-45D-AD-40pct_HCL",
                 "E1": "AL-1-45E-BD", "E2": "AL-1-45E-BD-20pct_HCL", "E3": "AL-1-45E-BD-40pct_HCL"}
    vial_df = pd.DataFrame.from_dict(vial_dict, orient="columns")
    vial_df.to_csv("2024_11_26_polyelectrolyte_degradation.csv")

    try:
        nr_robot.reset_after_initialization()
        nr_robot.set_pipet_tip_type(North_Robot.DEFAULT_DIMS, 0) 

        # nr_track.grab_well_plate_from_nr(0)
        # nr_track.return_well_plate_to_nr(1)

        ### BLANKS
        # Dispense blank (HPLC_Water) into wellplate
        nr_robot.aspirate_from_vial(0, 0.6)  # aspirate from HPLC water
        nr_robot.dispense_into_wellplate([0, 1, 2], [0.25, 0.2, 0.15])  # dispense HPLC water into well plate
        nr_robot.remove_pipet()

        # Dispense acid into wellplate for blanks
        nr_robot.aspirate_from_vial(1, 0.15)  # aspirate from HCL
        nr_robot.dispense_into_wellplate([1, 2], [0.05, 0.1])  # dispense HPLC water into well plate
        nr_robot.remove_pipet()


        # Dispense polyelectrolyte into well plate
        nr_robot.aspirate_from_vial(2, 0.15)  # aspirate from monomer (0.25mg/mL)
        nr_robot.dispense_into_wellplate(
            [3, 4, 5], [0.05, 0.05, 0.05]
        )  # dispense polyelectrolyte into well plate
        nr_robot.remove_pipet()

        nr_robot.aspirate_from_vial(3, 0.15)  # aspirate from polyelectrolyte
        nr_robot.dispense_into_wellplate(
            [6, 7, 8], [0.05, 0.05, 0.05]
        )  # dispense polyelectrolyte into well plate
        nr_robot.remove_pipet()

        nr_robot.aspirate_from_vial(4, 0.15)  # aspirate from polyelectrolyte
        nr_robot.dispense_into_wellplate(
            [9, 10, 11], [0.05, 0.05, 0.05]
        )  # dispense polyelectrolyte into well plate
        nr_robot.remove_pipet()

        nr_robot.aspirate_from_vial(5, 0.15)  # aspirate from polyelectrolyte
        nr_robot.dispense_into_wellplate(
            [12, 13, 14], [0.05, 0.05, 0.05]
        )  # dispense polyelectrolyte into well plate
        nr_robot.remove_pipet()

        # Dispense H2O into well plate
        nr_robot.aspirate_from_vial(0, 0.45)  # aspirate from water
        nr_robot.dispense_into_wellplate(
            [3, 4, 5], [0.2, 0.15, 0.10]
        )  # dispense polyelectrolyte into well plate
        nr_robot.remove_pipet()

        nr_robot.aspirate_from_vial(0, 0.45)  # aspirate from polyelectrolyte
        nr_robot.dispense_into_wellplate(
            [6, 7, 8], [0.2, 0.15, 0.10]
        )  # dispense polyelectrolyte into well plate
        nr_robot.remove_pipet()

        nr_robot.aspirate_from_vial(0, 0.45)  # aspirate from polyelectrolyte
        nr_robot.dispense_into_wellplate(
            [9, 10, 11], [0.2, 0.15, 0.10]
        )  # dispense polyelectrolyte into well plate
        nr_robot.remove_pipet()

        nr_robot.aspirate_from_vial(0, 0.45)  # aspirate from polyelectrolyte
        nr_robot.dispense_into_wellplate(
            [12, 13, 14], [0.2, 0.15, 0.10]
        )  # dispense polyelectrolyte into well plate
        nr_robot.remove_pipet()

        # Dispense acid into well plate
        nr_robot.aspirate_from_vial(1, 0.15)  # aspirate from acid
        nr_robot.dispense_into_wellplate(
            [4], [0.05]
        )  # dispense polyelectrolyte into well plate
        time.sleep(85)
        nr_robot.dispense_into_wellplate([5], [0.1])
        nr_robot.remove_pipet()

        nr_robot.aspirate_from_vial(1, 0.15)  # aspirate from acid
        nr_robot.dispense_into_wellplate(
            [7, 8], [0.05, 0.1]
        )  # dispense polyelectrolyte into well plate
        nr_robot.remove_pipet()

        nr_robot.aspirate_from_vial(1, 0.15)  # aspirate from acid
        nr_robot.dispense_into_wellplate(
            [10, 11], [0.05, 0.1]
        )  # dispense polyelectrolyte into well plate
        nr_robot.remove_pipet()

        nr_robot.aspirate_from_vial(1, 0.15)  # aspirate from acid
        nr_robot.dispense_into_wellplate(
            [13, 14], [0.05, 0.1]
        )  # dispense polyelectrolyte into well plate
        nr_robot.remove_pipet()
        
        nr_robot.save_vial_status("../utoronto_demo/status/stanley_polyelectrolyte.csv")
        nr_robot.remove_pipet()
    except Exception as e:
        print("Error", e)
        nr_robot.reset_robot()

    gen5 = Biotek()
    try:
        # Set the speed for the track (horizontal and vertical)
        nr_track.set_horizontal_speed(50)
        nr_track.set_vertical_speed(50)

        # Move well plate to cytation
        nr_track.grab_well_plate_from_nr(0, quartz_wp=True)
        nr_track.move_gripper_to_cytation()
        gen5.CarrierOut()
        nr_track.release_well_plate_in_cytation(quartz_wp=True)
        gen5.CarrierIn()

        # Run cytation protocol

        plate = gen5.load_protocol(r"C:\Protocols\Stanley_Degradation.prt")
        run = gen5.run_protocol(plate)
        while gen5.protocol_in_progress(run):
             print("Read in progress...")
             time.sleep(10)

        # Return well-plate to storage
        gen5.CarrierOut()
        nr_track.grab_well_plate_from_cytation(quartz_wp=True)
        gen5.CarrierIn()
        nr_track.return_well_plate_to_nr(1, quartz_wp=True)

        # Return lid
        nr_track.grab_well_plate_from_nr(2, grab_lid=True)
        nr_track.return_well_plate_to_nr(1, grab_lid=True)

        nr_track.origin()

        # Should move track away from north robot
        gen5.close()
        c9 = None

    except Exception as e:
        print("Error occured during operation", e)
        gen5.close()
        c9 = None
    except KeyboardInterrupt:
        gen5.close()
        c9 = None

main()

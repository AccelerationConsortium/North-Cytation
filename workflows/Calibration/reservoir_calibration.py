import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import numpy as np


EXPORT_FILEPATH = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\other\Calibration Data\Calibration May 2025\calibration_reservoir_water.txt"
DISPENSE_VOLUMES = [0.1, 0.25, 0.5, 1, 1.5, 2, 2.5] #volumes in mL

REPLICATES = 3
VIAL_INDICES = [0, 1, 2, 3]
WATER_RESERVOIR_INDEX = 1 #index of the reservoir containing water (change if calibrating with a different solvent)

lash_e = Lash_E("../utoronto_demo/status/sample_input_vials.csv", simulate=False, initialize_biotek=False)

lash_e.nr_robot.check_input_file()

lash_e.nr_robot.prime_reservoir_line(1, overflow_vial=0)

input("Finished priming, Enter to continue")
calibration_data = [['Expected Volume (mL)', 'Actual mass (g)']]
curr_vial_index = 0
try:
    for volume in DISPENSE_VOLUMES: #loop through volumes to dispense
        curr_volume = lash_e.nr_robot.get_vial_info(curr_vial_index, "vial_volume")
        if curr_volume + volume*REPLICATES > 8.0 and curr_vial_index < len(VIAL_INDICES) - 1: #move to next vial if not enough volume to add the next set of volumes *replicates
            curr_vial_index += 1
        elif curr_volume + volume*REPLICATES > 8.0 and curr_vial_index == len(VIAL_INDICES) - 1:
            print("Not enough volume in the current vial to dispense the required volume. Please empty vials")
            break #TODO: potentially better error handling (ex. let user empty vial and continue)

        for replicate in range(REPLICATES): #loop through number of replicates
            mass = lash_e.nr_robot.dispense_into_vial_from_reservoir(WATER_RESERVOIR_INDEX, curr_vial_index, volume, measure_weight=True)
            print(f"Dispensing {volume}mL, rep {replicate+1}: mass = {mass}g")
            calibration_data.append([volume, str(mass)[0:5]]) #round mass to 5 digits

    np.savetxt(EXPORT_FILEPATH, calibration_data,delimiter='\t', fmt ='% s') #export txt file
    print(f"Calibration data saved to {EXPORT_FILEPATH}")
    lash_e.nr_robot.move_home()

except KeyboardInterrupt:
        lash_e.nr_robot.c9 = None

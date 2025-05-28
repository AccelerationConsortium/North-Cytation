import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import numpy as np

EXPORT_FILEPATH = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\Calibration Data\Calibration May 2025\calibration_reservoir_water"
DISPENSE_VOLUMES = [0.1, 0.25, 0.5, 1, 1.5, 2, 2.5] #volumes in mL
#DISPENSE_VOLUMES = [1, 2] #testing

REPLICATES = 3
VIAL_INDICES = [0, 1, 2, 3]

#TODO: test w/ 1mL volume first?

lash_e = Lash_E("../utoronto_demo/status/sample_input_vials.csv")

lash_e.nr_robot.check_input_file()

calibration_data = [['Expected Volume (mL)', 'Actual mass (g)']]
curr_vial_index = 0
try:
    for volume in DISPENSE_VOLUMES:
        curr_volume = lash_e.nr_robot.get_vial_info(curr_vial_index, "vial_volume")
        if curr_volume + volume*REPLICATES > 8.0 and curr_vial_index < len(VIAL_INDICES) - 1: #move to next vial if not enough volume to add the next set of volumes *replicates
            curr_vial_index += 1
        elif curr_volume + volume*REPLICATES > 8.0 and curr_vial_index == len(VIAL_INDICES) - 1:
            print("Not enough volume in the current vial to dispense the required volume. Please empty vials")
            break

        for replicate in range(REPLICATES):
            mass = lash_e.nr_robot.dispense_into_vial_from_reservoir(0, curr_vial_index, volume, measure_weight=True)
            print(f"Dispensing {volume}mL,  rep {replicate+1}: mass = {mass}g")
            calibration_data.append([volume, str(mass)[0:5]]) #round mass to 5 digits

    np.savetxt(EXPORT_FILEPATH, calibration_data,delimiter='\t', fmt ='% s') #export txt file

except KeyboardInterrupt:
        lash_e.nr_robot.c9 = None

import North_Safe
import numpy as np
import pandas as pd

#Input data
calibration_file_name = 'Calibration_File_PEG_Water_25_percent.txt'
volumes = [0.500, 0.400, 0.300, 0.200, 0.100, 0.050]
pipet_length = [0.5, 0.2] #Need to measure these distances
repeats = 3

#Get the input data about what vials you are using for your experiment
vial_df = pd.read_csv("vial_status.txt", delimiter='\t', index_col='vial index')
#print(vial_df)

nr = North_Safe.North_Robot(vial_df, pipet_length)

nr.reset_after_initialization()
nr.move_vial_to_clamp(0)
nr.uncap_clamp_vial()

#Calibration Routine. Pipet solvent from one vial to another, measure the mass
calibration_data = [['Volume (mL)', 'Mass (g)']]
for volume in volumes:
    print("\nVolume calibrating: " + str(volume) + " mL\n")
    for i in range (0, repeats):

        mass_pipetted = nr.pipet_from_vial_into_vial(1, 0, volume,measure_weight=True, aspirate_conditioning=False, track_height=True)

        mass_formatted = str(mass_pipetted)[0:5]
        print("Try #" + str(i+1) + " mass : " + mass_formatted + " g")
        calibration_data.append([volume, mass_formatted])
        
nr.remove_pipet()
nr.recap_clamp_vial()
nr.return_vial_from_clamp(0)
nr.c9.move_z(292)

np.savetxt(calibration_file_name, calibration_data, delimiter='\t', fmt ='% s')
import North_Safe
import numpy as np
import pandas as pd

#Input data
calibration_file_name = 'peg_50_pump_4speeds.txt'
volume = 0.5
pipet_length = [0.5, 0.2] #Need to measure these distances
repeats = 4

speeds = [10]


vial_df = pd.read_csv("vial_status.txt", delimiter='\t', index_col='vial index') #Edit this


nr = North_Safe.North_Robot(vial_df, pipet_length)

nr.reset_after_initialization()

nr.move_vial_to_clamp(0)
nr.uncap_clamp_vial()

nr.c9.set_pump_speed(0, 30)

#Calibration Routine. Pipet solvent from one vial to another, measure the mass
calibration_data = [['Volume (mL)', 'Mass (g)']]

print("\nVolume calibrating: " + str(volume) + " mL\n")
for i in range (0, len(speeds)):
    
    speed = speeds[i]

    mass_pipetted = nr.pipet_from_vial_into_vial(1, 0, volume,
                    measure_weight=True, wait_over_vial=True, track_height=True, aspirate_speed=speed, dispense_speed=speed)

    mass_formatted = str(mass_pipetted)[0:5]
    print("Speed" + str(speeds[i]) + " mass : " + mass_formatted + " g")
    calibration_data.append([volume, mass_formatted])
    
nr.remove_pipet()
nr.recap_clamp_vial()
nr.return_vial_from_clamp(0)
nr.c9.move_z(292)

np.savetxt(calibration_file_name, calibration_data, delimiter='\t', fmt ='% s')
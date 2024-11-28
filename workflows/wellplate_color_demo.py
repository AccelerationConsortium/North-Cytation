import numpy as np
import pandas as pd
import math
import sys
import os
import time
sys.path.append("../utoronto_demo")

def generate_random_matrix(rows, cols, row_sum, divisible_by, min_value):
    if row_sum < (cols - 1) * divisible_by:
        raise ValueError("Row sum is too small to meet the divisibility constraint.")

    matrix = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        remaining_sum = row_sum
        values = []

        for j in range(cols - 1):
            # Allow zero as a choice, but enforce minimum value for non-zero entries
            max_val = remaining_sum - (cols - j - 1) * min_value
            max_val = max(max_val, 0)

            if max_val == 0:
                values.append(0)
                continue

            # Generate possible values (including zero)
            choices = np.arange(0, max_val + divisible_by, divisible_by)
            if remaining_sum > min_value:
                choices = np.concatenate(([0], choices[choices >= min_value]))
            value = np.random.choice(choices)
            values.append(value)
            remaining_sum -= value

        # Last column takes the remaining sum (ensure it's divisible by divisible_by)
        if remaining_sum % divisible_by != 0:
            raise ValueError("Row sum constraints cannot be satisfied.")
        values.append(remaining_sum)

        # Shuffle to randomize placement of zeros and values
        np.random.shuffle(values)
        matrix[i, :] = values

    return matrix


# Parameters
rows = 8 #number of samples generated
replicates = 1 #number of replicates
cols = 4 #number of colors

row_sum = 250 #max per well 
divisible_by = 10 #resolution
min_value = 40 #It's actually divis_by + this value

data_colors_uL = generate_random_matrix(rows, cols, row_sum, divisible_by, min_value)/1000
data_colors_uL = np.insert(data_colors_uL,0,[0,0,0,0.25],0)
data_colors_uL = np.insert(data_colors_uL,0,[0,0,0.25,0],0)
data_colors_uL = np.insert(data_colors_uL,0,[0,0.25,0,0],0)
data_colors_uL = np.insert(data_colors_uL,0,[0.25,0,0,0],0)

print(data_colors_uL)
print("Row sums:", np.sum(data_colors_uL * 1000, axis=1))  # Should all equal 250

data_colors_uL = np.repeat(data_colors_uL, replicates, axis=0)
print(data_colors_uL)

sum_colors = np.sum(data_colors_uL,0)
print(sum_colors) #how much of each volume is used

input()

with open("output_colors.txt", "w") as file:
    for row in data_colors_uL:
        file.write(", ".join(map(str, row)) + "\n")



from north import NorthC9
from North_Safe import North_Robot
from North_Safe import North_Track
from Locator import *
from biotek import Biotek

#needed files: 1. vial_status 2.wellplate_recipe
VIAL_FILE = "../utoronto_demo/status/vials_color.txt" #txt

DEFAULT_DIMS = [25,85]
c9 = NorthC9('A', network_serial='AU06CNCF')
nr = North_Robot(c9,VIAL_FILE)              
nr.set_pipet_tip_type(DEFAULT_DIMS,0)
nr.PIPETS_USED = [0,0]
gen5 = Biotek()
nr_track = North_Track(c9)

try:
    c9.move_z(292)
    nr.reset_after_initialization()

    for i in range (0,4):
        nr.move_vial_to_clamp(i)
        nr.uncap_clamp_vial()
        vol_needed = round(sum_colors[i],3)
        vol_dispensed = 0
        array_index = 0
        VOL_LIMIT = 0.90 #max amt to pipet
        VOL_BUFFER = 0.05 #extra amount to pipet
        vol_buffer_n = VOL_BUFFER

        nr.get_pipet()
        
        print("Color ", i)
        print("Total volume: ", vol_needed)

        last_index = 0
        while round(vol_dispensed,3) < round(vol_needed,3):
            dispense_vol=0
            dispense_array = []
            processing=True
            while processing:
                try:
                    volume = round(data_colors_uL[last_index,i],3)

                    if dispense_vol+volume<=VOL_LIMIT:
                        dispense_vol+=volume
                        dispense_array.append(round(float(volume),3))
                        last_index+=1
                    else:
                        processing=False
                except:
                    processing=False
            print(f"Amount to Dispense:{dispense_vol}")
            print(f"Aspirating solution {i}: {dispense_vol+vol_buffer_n} uL")
            nr.aspirate_from_vial(i,dispense_vol+vol_buffer_n)
            vol_buffer_n=0

            well_plate_array = np.arange((last_index-len(dispense_array)),last_index,1)
            
            well_plate_array = [int(x) for x in well_plate_array]

            print("Indices dipensed:", well_plate_array)
            print("Dispense volumes:", dispense_array)
            print("Dispense sum", np.sum(dispense_array))

            nr.dispense_into_wellplate(well_plate_array,dispense_array)

            vol_dispensed += dispense_vol

            print(f"Solution Dispensed {i}: {vol_dispensed} uL")     
        
        nr.dispense_into_vial(i,VOL_BUFFER)

        nr.remove_pipet()
        nr.recap_clamp_vial()
        nr.return_vial_from_clamp()

    c9.move_z(292)

    #Move well-plate to cytation
    nr_track.grab_well_plate_from_nr(0)
    nr_track.move_gripper_to_cytation()
    gen5.CarrierOut()
    nr_track.release_well_plate_in_cytation()
    gen5.CarrierIn() 

    #Run cytation protocol
    plate = gen5.load_protocol(r"C:\Protocols\Spectra_WholePlate.prt") #Insert name here
    run = gen5.run_protocol(plate)
    while gen5.protocol_in_progress(run):
        print("Read in progress...")
        time.sleep(10)

    #Return well-plate to storage
    gen5.CarrierOut()
    nr_track.grab_well_plate_from_cytation()
    gen5.CarrierIn()
    nr_track.return_well_plate_to_nr(0)

    nr_track.origin()
    
except KeyboardInterrupt:
    nr.reset_robot()
    c9 = None
    

c9 = None

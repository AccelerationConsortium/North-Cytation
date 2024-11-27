import numpy as np
import pandas as pd
import math
import sys
import os
import time
sys.path.append("../utoronto_demo")

#ChatGPT generated
def generate_random_matrix(rows, cols, row_sum, divisible_by, min_value):
    matrix = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        # Adjust row_sum to account for the minimum value requirement
        adjusted_sum = row_sum - min_value * cols
        if adjusted_sum < 0:
            raise ValueError("Row sum is too small to satisfy the minimum value constraint.")
        
        # Generate random numbers divisible by the specified value
        options = np.arange(0, adjusted_sum + divisible_by, divisible_by)
        values = np.random.choice(options, size=cols - 1, replace=True)
        values.sort()
        values = np.concatenate(([0], values, [adjusted_sum]))
        differences = np.diff(values)
        np.random.shuffle(differences)

        # Add the minimum value to each element, ensuring non-zero elements meet the constraint
        matrix[i, :] = [max(value + min_value, 0) if value > 0 else 0 for value in differences]

    return matrix

# Parameters
rows = 32 #number of samples generated
replicates = 3 #number of replicates
cols = 5 #number of colors
row_sum = 200 #max per well 
divisible_by = 5 #resolution
min_value = 20 #It's actually divis_by + this value

data_colors_uL = generate_random_matrix(rows, cols, row_sum, divisible_by, min_value)/1000
print(data_colors_uL)
sum_colors = np.sum(data_colors_uL,0)*replicates
print(sum_colors)


from north import NorthC9
from North_Safe import North_Robot
from North_Safe import North_Track
from Locator import *
from biotek import Biotek

#needed files: 1. vial_status 2.wellplate_recipe
VIAL_FILE = "../utoronto_demo/status/vials_color.txt" #txt

BLUE_DIMS = [20,85] #Not accurate
vial_df = pd.read_csv(VIAL_FILE, delimiter='\t')
c9 = NorthC9('A', network_serial='AU06CNCF')
nr = North_Robot(c9,vial_df)              
nr.set_pipet_tip_type(BLUE_DIMS,0)
nr.PIPETS_USED = [0,0]
gen5 = Biotek()
nr_track = North_Track()

try:
    c9.move_z(300)
    nr.reset_after_initialization()

    for i in range (0,5):
        #nr.move_vial_to_clamp(i)
        #nr.uncap_clamp_vial()
        vol_needed = sum_colors[i]
        vol_dispensed = 0
        array_index = 0
        VOL_LIMIT = 0.95 #max amt to pipet
        VOL_BUFFER = 0.05 #extra amount to pipet
        
        print("Color ", i)
        print("Total volume: ", vol_needed)

        last_index = 0
        while vol_dispensed < vol_needed:
            dispense_vol=0
            dispense_array = []
            processing=True
            while processing:
                try:
                    volume = data_colors_uL[last_index,i]*replicates

                    if dispense_vol+volume<=VOL_LIMIT:
                        dispense_vol+=volume
                        for j in range (0, replicates):
                            dispense_array.append(float(volume)/replicates)
                        last_index+=1
                    else:
                        processing=False
                except:
                    processing=False
            print(f"Dispensing solution {i}: {dispense_vol} uL")
            #nr.get_pipet()
            #nr.aspirate_from_vial(i,dispense_vol+VOL_BUFFER)

            well_plate_array = np.arange((last_index-len(dispense_array)/replicates)*replicates,last_index*replicates,1)
            
            print("Indices dipensed:", well_plate_array)
            print("Dispense volumes:", dispense_array)
            print("Dispense sum", np.sum(dispense_array))

            #nr.dispense_into_wellplate(well_plate_array,dispense_array)

            vol_dispensed += dispense_vol

            print(f"Solution Dispensed {i}: {vol_dispensed} uL")       
        
        #nr.remove_pipet()
        #nr.recap_clamp_vial()
        #nr.return_vial_from_clamp()

    c9.move_z(300)

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
    
except KeyboardInterrupt:
    c9 = None

c9 = None

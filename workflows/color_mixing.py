import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np
import time

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/color_mixing_vials.txt"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Quick_Measurement.prt"

#Define your workflow! 
#In this case we have two parameters: 
def check_input_file(input_file):  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_file, sep=",")
    print(vial_status)
    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

def generate_random_matrix(rows, cols, row_sum, divisible_by):
    if row_sum % divisible_by != 0:
        raise ValueError("Row sum must be a multiple of divisible_by.")
    
    if row_sum < (cols - 1) * divisible_by:
        raise ValueError("Row sum is too small to distribute among columns.")
    
    matrix = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        remaining_sum = row_sum
        values = []

        for j in range(cols - 1):
            max_val = remaining_sum - (cols - j - 1) * divisible_by
            max_val = max(max_val, 0)
            
            choices = np.arange(0, max_val + divisible_by, divisible_by)
            value = np.random.choice(choices)
            values.append(value)
            remaining_sum -= value

        # Last column takes the remaining sum (guaranteed to be a multiple of divisible_by)
        values.append(remaining_sum)
        
        # Shuffle to randomize placement of zeros and values
        np.random.shuffle(values)
        matrix[i, :] = values
    
    return matrix

def sample_workflow(number_samples=12,replicates=1,colors=4,resolution_vol=10,well_volume=250):
  
    # Initial State of your Vials, so the robot can know where to pipet
    check_input_file(INPUT_VIAL_STATUS_FILE)

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE)

    #The vial indices are numbers that are used to track the vials. For the sake of clarity, these are stored in the input vial file but accessed here
    water_index = lash_e.nr_robot.get_vial_index_from_name('water') #Get the ID of our target reactor
    red_index = lash_e.nr_robot.get_vial_index_from_name('red')
    blue_index = lash_e.nr_robot.get_vial_index_from_name('blue')
    yellow_index = lash_e.nr_robot.get_vial_index_from_name('yellow')

    data_colors_uL = generate_random_matrix(number_samples, colors, well_volume, resolution_vol)/1000

    print("Row sums:", np.sum(data_colors_uL * 1000, axis=1))  # Should all equal 250

    data_colors_uL = np.repeat(data_colors_uL, replicates, axis=0)

    sum_colors = np.sum(data_colors_uL,0)
    print("Total volume per vial:", sum_colors) #how much of each volume is used

    data_pd = pd.DataFrame(data=data_colors_uL,columns=['water','red','blue','yellow'])
    print(data_pd)

    lash_e.nr_robot.dispense_from_vials_into_wellplate(data_pd,[water_index,red_index,blue_index,yellow_index])
    
    #Transfer the well plate to the cytation and measure
    #lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE)
    
#Execute the sample workflow.
#Specify that we are going to aspirate 0.6 from our two sample vials. We could also set the number of replicates to some other number than 3
#e.g. sample_workflow(aspiration_volume=0.6,replicates=5)
sample_workflow()

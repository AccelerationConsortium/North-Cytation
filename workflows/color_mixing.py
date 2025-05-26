import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np
import time

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/color_mixing_vials.csv"

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

def mix_wells(lash_e, wells, wash_index=4, wash_volume=0.150, repeats=1,replicates=6):
    for well in wells:
        stay_low=True
        if well%replicates==0:
            lash_e.nr_robot.aspirate_from_vial(wash_index,wash_volume)
            lash_e.nr_robot.dispense_into_vial(wash_index,wash_volume,initial_move=False)
            stay_low=False
        #for i in range (0,repeats):
        #    lash_e.nr_robot.dispense_from_vial_into_vial(wash_index,wash_index,wash_volume,move_to_aspirate=False,move_to_dispense=False,buffer_vol=0)

        lash_e.nr_robot.pipet_from_wellplate(well,wash_volume,stay_low=stay_low)
        lash_e.nr_robot.pipet_from_wellplate(well,wash_volume,aspirate=False,move_to_aspirate=False)
        for i in range (0, repeats):
            lash_e.nr_robot.pipet_from_wellplate(well,wash_volume,move_to_aspirate=False)
            lash_e.nr_robot.pipet_from_wellplate(well, wash_volume,aspirate=False,move_to_aspirate=False)

def sample_workflow(number_samples=6,replicates=2,colors=4,resolution_vol=10,well_volume=240):
  
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

    data_pd_save = data_colors_uL*1000
    data_pd_save = pd.DataFrame(data=data_pd_save,columns=['water','red','blue','yellow'])
    data_pd_save.to_csv("../utoronto_demo/output/color_mixing_composition.csv",sep=',')

    print("Row sums:", np.sum(data_colors_uL * 1000, axis=1))  # Should all equal 250

    data_colors_uL = np.repeat(data_colors_uL, replicates, axis=0)

    sum_colors = np.sum(data_colors_uL,0)
    print("Total volume per vial:", sum_colors) #how much of each volume is used

    data_pd = pd.DataFrame(data=data_colors_uL,columns=['water','red','blue','yellow'])
    print(data_pd)

    num_wells = data_colors_uL.shape[0]
    print("Number samples: ", num_wells)
    wells = range(0, num_wells)

    input("Waiting...")

    start_time = time.perf_counter()

    lash_e.nr_robot.dispense_from_vials_into_wellplate(data_pd,[water_index,red_index,blue_index,yellow_index],low_volume_cutoff=0.250,pipet_back_and_forth=True)
    #mix_wells(lash_e, wells,replicates=replicates)

    end_time = time.perf_counter()

    print("Time to complete: ", end_time - start_time)
    
    #Transfer the well plate to the cytation and measure
    #results = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE,wells_to_measure=range(0,96))

    #print(results)

    #results.to_csv('save_data.txt', sep=',')
    
#Execute the sample workflow.
sample_workflow()

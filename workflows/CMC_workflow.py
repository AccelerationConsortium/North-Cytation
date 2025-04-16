import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/CMC_workflow_input.csv"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\CMC_Fluorescence.prt" #Will need to create a measurement protocol

#Define your workflow! 
#In this case we have two parameters: 
def check_input_file(input_file):  
    # Initial State of your Vials, so the robot can know where to pipet
    vial_status = pd.read_csv(input_file, sep=",")
    print(vial_status)
    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

def split_volume(volume, max_volume=1.0):
    """Split a volume into chunks not exceeding max_volume"""
    if volume <= max_volume:
        return [volume]
    
    n_full = int(volume // max_volume)
    remainder = volume % max_volume
    parts = [max_volume] * n_full
    if remainder > 0:
        parts.append(remainder)
    return parts

def mix_surfactants(lash_e, surfactant_index_list, surfactant_volumes, target_vial_index, mix_ratio=0.75): #Mix the different surfactants + water into a new vial
    for i in range (0, len(surfactant_index_list)):
        print("\nCombining surfactants:")
        surfactant_index = surfactant_index_list[i]
        surfactant_volume = surfactant_volumes[i]
        #Split the volumes into pipetable chunks
        volumes = split_volume(surfactant_volume)
        print(f"Pipetable volumes: ", volumes)
        for volume in volumes:
            lash_e.nr_robot.dispense_from_vial_into_vial(surfactant_index,target_vial_index,volume)
        print("Mixing samples...")
        lash_e.nr_robot.mix_vial(target_vial_index,min(surfactant_volume*mix_ratio, 1.0))
        lash_e.nr_robot.remove_pipet()

def create_wellplate_samples(lash_e, surfactant_mixture_index, DMSO_pyrene_index,DMSO_pyrene_volume,water_index,concentration_ratios,replicates,last_wp_index): #Add the DMSO_pyrene and surfactant mixture to well plates
    print("\n Dispensing into Wellplate")
    well_indices = range (last_wp_index,last_wp_index+len(concentration_ratios)*replicates)
    dispense_indices = [surfactant_mixture_index,DMSO_pyrene_index,water_index]

    TOTAL_VOLUME = 1.0 #Volume per well
    surfactant_volumes = TOTAL_VOLUME*np.array(concentration_ratios)
    water_volumes = TOTAL_VOLUME - DMSO_volume - surfactant_volumes

    data = {'Surfactant volumes': surfactant_volumes*replicates, 'DMSO Volume': [DMSO_pyrene_volume]*len(concentration_ratios)*replicates, 'Water Volume': water_volumes*replicates}
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.index = well_indices
    print(df)
    lash_e.nr_robot.dispense_from_vials_into_wellplate(df,dispense_indices)
    

def sample_workflow(starting_wp_index,surfactant_volumes,surfactant_index_list,concentration_ratios,DMSO_VOLUME,replicates):
    #Step 0: Check the input to confirm that it's OK!
    check_input_file(INPUT_VIAL_STATUS_FILE)

    #Step 1: Mix the surfactants and Dilute with Water
    mix_surfactants(lash_e, surfactant_index_list,surfactant_volumes,surfactant_mixture_index)

    #Step 2: Perform the assay dilutions with water and the surfactant and the dye
    create_wellplate_samples(lash_e,surfactant_mixture_index,pyrene_DMSO_index,DMSO_VOLUME,water_index,concentration_ratios,replicates=replicates,last_wp_index=starting_wp_index)
    
    #Step 3: Transfer the well plate to the cytation and measure
    #resulting_data = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE,wells_to_measure=range(LAST_WP_INDEX,LAST_WP_INDEX+replicates))

    #Step 4: Analyze the results
    #Take the resulting_data and analyze it to determine the CMC
    
#Initialize the workstation, which includes the robot, track, cytation and photoreactors
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE,simulate=True)

#The vial indices are numbers that are used to track the vials. I will be implementing a dictionary system so this won't be needed
surfactant_1_index = lash_e.nr_robot.get_vial_index_from_name('surfactant_1_stock') #Get the ID of our target reactor
surfactant_2_index = lash_e.nr_robot.get_vial_index_from_name('surfactant_2_stock')
surfactant_mixture_index = lash_e.nr_robot.get_vial_index_from_name('surfactant_mixture_1')
pyrene_DMSO_index = lash_e.nr_robot.get_vial_index_from_name('pyrene_DMSO')
water_index = lash_e.nr_robot.get_vial_index_from_name('water')

#TODO: We need to be able to take the surfactant combinations from the algorithm here
#Note as written these can't exceed 1.0, but that's easy to change if you need larger volumes you can write a function that separates the volumes into <1 mL chunks
surfactant_index_list = [surfactant_1_index, surfactant_2_index, water_index]
surfactant_volumes = [1.0,1.0,3.0] #We need to make sure that we create enough for our assay

replicates = 1
DMSO_volume = 0.01
starting_wp_index = 0
concentration_ratios = [0,0.02,0.04,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.99] #Note I came up with these based on your 0, 0.5 to 25 mM as ratios, so it would be a similar sweep regardless of the concentrations. 

#Execute the sample workflow.
sample_workflow(starting_wp_index,surfactant_volumes,surfactant_index_list,concentration_ratios,DMSO_volume,replicates)
starting_wp_index+=replicates*len(concentration_ratios)
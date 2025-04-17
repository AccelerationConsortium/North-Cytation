from operator import sub
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np
import analysis.cmc_data_analysis as analyzer
import analysis.cmc_exp as experimental_planner
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
    """Split a volume into equal parts, each <= max_volume"""
    import math
    
    if volume <= max_volume:
        return [volume]
    
    # Calculate the number of parts needed, rounding up
    n_parts = math.ceil(volume / max_volume)
    
    # Each part is equal and less than or equal to max_volume
    part_volume = volume / n_parts
    
    return [part_volume] * n_parts

def mix_surfactants(lash_e, surfactant_index_list, sub_stock_vols, target_vial_index, mix_ratio=0.75): #Mix the different surfactants + water into a new vial
    print("\n Combining Surfactants: ")
    print("Stock solution composition: ", sub_stock_vols)
    for i in range (0, len(surfactant_index_list)):
        print("\nCombining surfactants:")
        surfactant_index = surfactant_index_list[i]
        surfactant_volume = list(sub_stock_vols.values())[i]/1000
        if surfactant_volume > 0:
            #Split the volumes into pipetable chunks
            volumes = split_volume(surfactant_volume)
            print(f"Pipetable volumes: ", volumes)
            for volume in volumes:
                lash_e.nr_robot.dispense_from_vial_into_vial(surfactant_index,target_vial_index,volume)
            print("Mixing samples...")
            lash_e.nr_robot.mix_vial(target_vial_index,min(surfactant_volume*mix_ratio, 1.0))
            lash_e.nr_robot.remove_pipet()

def create_wellplate_samples(lash_e, wellplate_data, substock_vial_index,DMSO_pyrene_index,water_index,last_wp_index): #Add the DMSO_pyrene and surfactant mixture to well plates
    print("\n Dispensing into Wellplate")
    samples_per_assay = wellplate_data.shape[0]
    well_indices = range (last_wp_index,last_wp_index+samples_per_assay)
    dispense_indices = [substock_vial_index,water_index,DMSO_pyrene_index]
    dispense_data = wellplate_data[['surfactant volume', 'water volume','probe volume']]/1000 #Convert to uL
    dispense_data.index = well_indices
    print(dispense_data)

    lash_e.nr_robot.dispense_from_vials_into_wellplate(dispense_data,dispense_indices)
    

def sample_workflow(starting_wp_index,surfactant_index_list,sub_stock_vols,substock_vial_index,water_index,pyrene_DMSO_index,wellplate_data):
    #Step 0: Check the input to confirm that it's OK!
    check_input_file(INPUT_VIAL_STATUS_FILE)

    #Step 1: Mix the surfactants and Dilute with Water
    mix_surfactants(lash_e, surfactant_index_list,sub_stock_vols,substock_vial_index)

    #Step 2: Perform the assay dilutions with water and the surfactant and the dye
    create_wellplate_samples(lash_e, wellplate_data, substock_vial_index,pyrene_DMSO_index,water_index,starting_wp_index)
    
    #Step 3: Transfer the well plate to the cytation and measure
    #resulting_data = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE,wells_to_measure=range(LAST_WP_INDEX,LAST_WP_INDEX+replicates))

    #Step 4: Analyze the results
    #Take the resulting_data and analyze it to determine the CMC
    concentrations = wellplate_data['concentration']
    ratio_data = None #This is determined from the resulting_data
    #CMC,r2 = analyzer.CMC_plot(ratio_data,concentrations)
    
#Initialize the workstation, which includes the robot, track, cytation and photoreactors
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE,simulate=True)

#The vial indices are numbers that are used to track the vials. I will be implementing a dictionary system so this won't be needed
substock_mixture_index = lash_e.nr_robot.get_vial_index_from_name('substock_1')
pyrene_DMSO_index = lash_e.nr_robot.get_vial_index_from_name('pyrene_DMSO')
water_index = lash_e.nr_robot.get_vial_index_from_name('water')

#These surfactants and ratios should be decided by something
surfactants = ['SDS', 'SLS', None]
ratios = [0.3, 0.7, 0]

surfactant_index_list = []
for surfactant in surfactants:
    surfactant_index_list.append(lash_e.nr_robot.get_vial_index_from_name(surfactant))
surfactant_index_list.append(water_index)

experiment,small_exp = experimental_planner.generate_exp(surfactants, ratios)

sub_stock_vols = experiment['surfactant_sub_stock_vols']
wellplate_data = experiment['df']
samples_per_assay = wellplate_data.shape[0]

starting_wp_index = 0

#Execute the sample workflow.
sample_workflow(starting_wp_index,surfactant_index_list,sub_stock_vols,substock_mixture_index,water_index,pyrene_DMSO_index,wellplate_data)
starting_wp_index+=samples_per_assay
from operator import sub
import sys
from turtle import delay
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np
from datetime import datetime
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

#Dilute surfactants with water
def mix_surfactants(lash_e, surfactant_index_list, sub_stock_vols, target_vial_index): #Mix the different surfactants + water into a new vial
    print("\n Combining Surfactants: ")
    print("Stock solution composition: ", sub_stock_vols)
    for i in range (0, len(surfactant_index_list)-1):
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
            lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.dispense_into_vial_from_reservoir(1, target_vial_index, sub_stock_vols['water'])
    lash_e.nr_robot.mix_vial(target_vial_index,0.9, repeats=5)
    lash_e.nr_robot.remove_pipet()

def fill_water_vial(water_index):
    vial_max_volume = 8 
    water_reservoir = 1
    lash_e.nr_robot.dispense_into_vial_from_reservoir(water_reservoir,water_index,vial_max_volume - lash_e.nr_robot.get_vial_info(water_index,'vial_volume')) #Fill up the water vial

#Dispense into wellplate
def create_wellplate_samples(lash_e, wellplate_data, substock_vial_index,DMSO_pyrene_index,water_index,last_wp_index): #Add the DMSO_pyrene and surfactant mixture to well plates
    print("\n Dispensing into Wellplate")
    samples_per_assay = wellplate_data.shape[0]
    well_indices = range (last_wp_index,last_wp_index+samples_per_assay)
    dispense_data = (wellplate_data[['surfactant volume', 'water volume','probe volume']]/1000).round(3) #Convert to uL
    dispense_data.index = well_indices
    print(dispense_data)

    df_surfactant = dispense_data[['surfactant volume']]
    df_water = dispense_data[['water volume']]
    df_dmso = dispense_data[['probe volume']]

    lash_e.nr_robot.move_vial_to_location(substock_vial_index,'main_8mL_rack', 42) #Safe location

    lash_e.nr_robot.dispense_from_vials_into_wellplate(df_dmso,[DMSO_pyrene_index],well_plate_type="48 WELL PLATE",dispense_speed=20,wait_time=5,asp_cycles=1,track_height=False, low_volume_cutoff = 0.04, buffer_vol = 0)
    lash_e.nr_robot.dispense_from_vials_into_wellplate(df_surfactant,[substock_vial_index],well_plate_type="48 WELL PLATE",dispense_speed=15)
    lash_e.nr_robot.dispense_from_vials_into_wellplate(df_water,[water_index],well_plate_type="48 WELL PLATE",dispense_speed=11)

    lash_e.nr_robot.return_vial_home(substock_vial_index) #return home from safe location

    lash_e.nr_robot.get_pipet(0) #get big pipet tip
    for well in well_indices:
        lash_e.nr_robot.mix_well_in_wellplate(well,volume=0.3,well_plate_type="48 WELL PLATE")
    lash_e.nr_robot.remove_pipet()
    

def sample_workflow(starting_wp_index,surfactant_index_list,sub_stock_vols,substock_vial_index,water_index,pyrene_DMSO_index,wellplate_data, surfactant_name=""):
    
    #Step 1: Mix the surfactants and Dilute with Water
    mix_surfactants(lash_e, surfactant_index_list,sub_stock_vols,substock_vial_index)

    #Step 2: Perform the assay dilutions with water and the surfactant and the dye
    fill_water_vial(water_index)
    create_wellplate_samples(lash_e, wellplate_data, substock_vial_index,pyrene_DMSO_index,water_index,starting_wp_index)
    
    #Step 3: Transfer the well plate to the cytation and measure
    samples_per_assay = wellplate_data.shape[0]
    if not simulate:
        resulting_data = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE,wells_to_measure=range(starting_wp_index,starting_wp_index+samples_per_assay),plate_type="48 WELL PLATE")
        resulting_data['ratio'] = resulting_data['1']/resulting_data['2']

        print(resulting_data)

        #Step 4: Analyze the results
        #Take the resulting_data and analyze it to determine the CMC
        concentrations = wellplate_data['concentration']
        ratio_data = resulting_data['ratio'].values #This is determined from the resulting_data
        CMC,r2 = analyzer.CMC_plot(ratio_data,concentrations)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        wellplate_data.to_csv(f'C:/Users/Imaging Controller/Desktop/CMC/{timestamp}_wellplate_data_{surfactant_name}.csv', index=False)
        resulting_data.to_csv(f'C:/Users/Imaging Controller/Desktop/CMC/{timestamp}_output_data_{surfactant_name}.csv', index=False)
        with open(f'C:/Users/Imaging Controller/Desktop/CMC/{timestamp}_wellplate_data_results_{surfactant_name}.txt', "w") as f:
            f.write(f"CMC: {CMC}, r2: {r2}")

        print("CMC (mMol): ", CMC)
        print("R-squared: ", r2)
    else:
        print("Skipping analysis for simulation")
    

#Step 0: Check the input to confirm that it's OK!
check_input_file(INPUT_VIAL_STATUS_FILE)

simulate = True

#Initialize the workstation, which includes the robot, track, cytation and photoreactors
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate, initialize_biotek=False)

#The vial indices are numbers that are used to track the vials. I will be implementing a dictionary system so this won't be needed
pyrene_DMSO_index = lash_e.nr_robot.get_vial_index_from_name('pyrene_DMSO')

starting_wp_index = 0 #CHANGE THIS AS NEEDED!!!

#These surfactants and ratios should be decided by something
surfactants = ['SDS', 'DTAB', 'TTAB', 'CAPB'] 
surfactant_index_list = []
for surfactant in surfactants:
    surfactant_index_list.append(lash_e.nr_robot.get_vial_index_from_name(surfactant))
ratios = [[0.5, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] #Must total 1
substock_name_list = ['substock_1','substock_2','substock_3', 'substock_4'] #For each set of surfactants

for i in range (0, len(ratios)):
    ratio = ratios[i]
    substock_mixture_index = lash_e.nr_robot.get_vial_index_from_name(substock_name_list[i])
    experiment,small_exp = experimental_planner.generate_exp(surfactants, ratio, sub_stock_volume=6000)
    sub_stock_vols = experiment['surfactant_sub_stock_vols']
    wellplate_data = experiment['df']
    samples_per_assay = wellplate_data.shape[0]
    surfactant_name = surfactants[i] #probably only works for the current set-up (1 aligns with i)
    water_vial_index = lash_e.nr_robot.get_vial_index_from_name('water')

    #Execute the sample workflow.
    sample_workflow(starting_wp_index,surfactant_index_list,sub_stock_vols,substock_mixture_index,water_vial_index,pyrene_DMSO_index,wellplate_data, surfactant_name=surfactant_name)
    starting_wp_index+=samples_per_assay

    #Check to see if we need a new well_plate
    if starting_wp_index >= 48:
        None
        #Return the wellplate to the disposal
        #Grab a new wellplate
        starting_wp_index = 0

    input("****Press enter to continue to next surfactant")
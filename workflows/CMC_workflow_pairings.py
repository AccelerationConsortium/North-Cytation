import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
from datetime import datetime
import analysis.cmc_data_analysis as analyzer
import analysis.cmc_exp_new as experimental_planner
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/CMC_workflow_input.csv"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\CMC_Fluorescence.prt" #Will need to create a measurement protocol
import numpy as np
import os
import slack_agent

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
# Mix multiple surfactants from index list into a substock vial
def mix_surfactants(lash_e, sub_stock_vols, substock_vial):
    """
    Mix surfactants from sub_stock_vols into substock_vial using robot logic.
    
    Args:
        lash_e: Lab automation object
        surfactant_index_dict: dict of {surfactant_name: vial_index}
        sub_stock_vols: dict of {surfactant_name: volume in µL}, including 'water'
        substock_vial: target vial index for the substock
    """
    print("\nCombining Surfactants:")
    print("Stock solution composition:", sub_stock_vols)

    for surfactant, volume_uL in sub_stock_vols.items():
        if surfactant == 'water':
            continue  # skip water here if water is handled elsewhere

        volume_mL = volume_uL / 1000

        print(f"\nHandling {surfactant}: {volume_mL:.3f} mL")

        if volume_mL <= 0.200:
            lash_e.nr_robot.move_vial_to_location(surfactant, 'main_8mL_rack', 43)  # Safe location

        if volume_mL > 0:
            volumes = split_volume(volume_mL)
            print("Pipetable volumes:", volumes)
            for v in volumes:
                lash_e.nr_robot.dispense_from_vial_into_vial(surfactant, substock_vial, v)
            print("Mixing samples...")
            lash_e.nr_robot.remove_pipet()

        if volume_mL <= 0.200:
            lash_e.nr_robot.return_vial_home(surfactant)

    if sub_stock_vols['water'] > 0:
        lash_e.nr_robot.dispense_into_vial_from_reservoir(1, substock_vial, sub_stock_vols['water']/1000)
        lash_e.nr_robot.move_vial_to_location(substock_vial, 'main_8mL_rack', 43) #move to safe location
        lash_e.nr_robot.mix_vial(substock_vial,0.9, repeats=10)
    #lash_e.nr_robot.vortex_vial(substock_vial,10,50) #Mix the surfactants
    lash_e.nr_robot.remove_pipet()

def fill_water_vial():
    vial_max_volume = 8 
    water_reservoir = 1
    current_water_volume = lash_e.nr_robot.get_vial_info('water','vial_volume')
    if current_water_volume < vial_max_volume:
        print(f"Filling water vial from {current_water_volume} mL to {vial_max_volume} mL")
        lash_e.nr_robot.dispense_into_vial_from_reservoir(water_reservoir,'water',vial_max_volume - current_water_volume) #Fill up the water vial

def split_water_batches(df_water, max_volume=7.0):
    batches = []
    current_batch = []
    cumulative_vol = 0.0

    for i, row in df_water.iterrows():
        vol = row['water volume']
        if cumulative_vol + vol > max_volume:
            # Commit the current batch
            if current_batch:
                batch_df = pd.DataFrame(current_batch)
                batches.append(batch_df)
            # Start new batch
            current_batch = [row]
            cumulative_vol = vol
        else:
            current_batch.append(row)
            cumulative_vol += vol

    # Add final batch
    if current_batch:
        batch_df = pd.DataFrame(current_batch)
        batches.append(batch_df)

    return batches

#Dispense into wellplate
def create_wellplate_samples(lash_e, wellplate_data, substock_vial_index,last_wp_index): #Add the DMSO_pyrene and surfactant mixture to well plates
    print("\n Dispensing into Wellplate")
    samples_per_assay = wellplate_data.shape[0]
    well_indices = range (last_wp_index,last_wp_index+samples_per_assay)
    dispense_data = (wellplate_data[['surfactant volume', 'water volume','probe volume']]/1000).round(3) #Convert to uL
    dispense_data.index = well_indices
    print(dispense_data)

    df_surfactant = dispense_data[['surfactant volume']]
    df_water = dispense_data[['water volume']]
    df_dmso = dispense_data[['probe volume']]

    water_batch_df = split_water_batches(df_water)

    lash_e.nr_robot.move_vial_to_location(substock_vial_index,'main_8mL_rack', 43) #Safe location

    lash_e.nr_robot.dispense_from_vials_into_wellplate(df_dmso,['pyrene_DMSO'],well_plate_type="48 WELL PLATE",dispense_speed=20,wait_time=2,asp_cycles=1,low_volume_cutoff = 0.04, buffer_vol = 0,pipet_back_and_forth=True,blowout_vol=0.1)
    lash_e.nr_robot.dispense_from_vials_into_wellplate(df_surfactant,[substock_vial_index],well_plate_type="48 WELL PLATE",dispense_speed=15)
    
    for water_batch in water_batch_df:
        print (water_batch)
        lash_e.nr_robot.dispense_from_vials_into_wellplate(water_batch,['water'],well_plate_type="48 WELL PLATE",dispense_speed=11)
        fill_water_vial()


    lash_e.nr_robot.return_vial_home(substock_vial_index) #return home from safe location

    lash_e.nr_robot.get_pipet(0) #get big pipet tip
    for well in well_indices:
        lash_e.nr_robot.mix_well_in_wellplate(well,volume=0.3,well_plate_type="48 WELL PLATE")
    lash_e.nr_robot.remove_pipet()
    
def sample_workflow(starting_wp_index,sub_stock_vols,substock_vial_index,wellplate_data,folder,save_modifier):
    
    lash_e.nr_robot.prime_reservoir_line(1,'water',0.5)

    #Step 1: Mix the surfactants and Dilute with Water
    mix_surfactants(lash_e,sub_stock_vols,substock_vial_index)

    #Step 2: Perform the assay dilutions with water and the surfactant and the dye
    fill_water_vial()
    create_wellplate_samples(lash_e, wellplate_data, substock_vial_index,starting_wp_index)
    
    #Step 3: Transfer the well plate to the cytation and measure
    samples_per_assay = wellplate_data.shape[0]
    if not simulate:
        resulting_data = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE,wells_to_measure=range(starting_wp_index,starting_wp_index+samples_per_assay),plate_type="48 WELL PLATE")
        
        
        resulting_data['ratio'] = resulting_data['1']/resulting_data['2']

        if np.mean(resulting_data['ratio']) > 1.0:
            print("Inverting data!")
            resulting_data['ratio'] = resulting_data['2'] / resulting_data['1']

        print(resulting_data)

        details = "_".join(f"{k}{int(v)}" for k, v in sub_stock_vols.items()) + '_' + save_modifier

        #Step 4: Analyze the results
        #Take the resulting_data and analyze it to determine the CMC
        concentrations = wellplate_data['concentration']
        ratio_data = resulting_data['ratio'].values #This is determined from the resulting_data
        figure_name = folder+f'CMC_plot_{details}.png'
        A1, A2, x0, dx, r_squared = analyzer.CMC_plot(ratio_data,concentrations,figure_name)
        
        wellplate_data.to_csv(folder+f'wellplate_data_{details}.csv', index=False)
        resulting_data.to_csv(folder+f'output_data_{details}.csv',index=False)
        with open(folder+f'wellplate_data_results_{details}.txt', "w") as f:
            f.write(f"CMC: {x0}, r2: {r_squared}")

        print("CMC (mMol): ", x0)
        print("R-squared: ", r_squared)

        slack_agent.send_slack_message(f"CMC Workflow complete! CMC={x0}, R-squared={r_squared}")
        slack_agent.upload_and_post_file(figure_name,'CMC Image')

    else:
        print("Skipping analysis for simulation")
    

simulate = False
logging = False

if logging:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = open(f"../utoronto_demo/logs/experiment_log_{timestamp}_sim{simulate}.txt", "w")
    sys.stdout = sys.stderr = log_file

#Initialize the workstation, which includes the robot, track, cytation and photoreactors
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
lash_e.nr_robot.check_input_file()
lash_e.nr_track.check_input_file()

if not simulate:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f'C:/Users/Imaging Controller/Desktop/CMC/{timestamp}/'
    os.makedirs(folder, exist_ok=True)
    print(f"Folder created at: {folder}")
else:
    folder = None

lash_e.nr_track.get_new_wellplate() #Get a new well plate #get wellplate from stack
starting_wp_index = 0 #CHANGE THIS AS NEEDED!!!

#These surfactants and ratios should be decided by something
surfactants = ['SDS', 'NaDC', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'P188',"P407", 'CAPB', 'CHAPS'] #refers to stock solutions
item_to_index = {item: i for i, item in enumerate(surfactants)}# Create a mapping from item to index
n = len(surfactants)  
substock_name_list = [f'substock_{i}' for i in range(1, n + 1)] #refers to substock solutions

#pairings_and_ratios = [(['P188'], [1])]
#pairings_and_ratios = [(['P188'], [1]), (['P407'],[1])]

# pairings_and_ratios = [
# (['SDS', 'NaDC'], [0.7, 0.3]),
# (['NaC', 'CTAB'], [0.4, 0.6]),
# (['P188', 'P407'], [0.5, 0.5]),
# (['DTAB', 'CHAPS'], [0.8, 0.2]),]
pairings_and_ratios = [
(['SDS', 'NaDC'], [0.7, 0.3]),]
padded_ratio_vectors = []
pairing_labels = []

for (pair, ratios) in pairings_and_ratios:
    ratio_vector = [0.0] * len(surfactants)
    for surf, ratio in zip(pair, ratios):
        idx = item_to_index[surf]
        ratio_vector[idx] = ratio
    padded_ratio_vectors.append(ratio_vector)
   
    if len(pair) == 1:
        label = f"{pair[0]}-{ratios[0]:.1f}"
    elif len(pair) == 2:
        label = f"{pair[0]}-{ratios[0]:.1f}_{pair[1]}-{ratios[1]:.1f}"
    else:
        raise ValueError("Only 1 or 2 surfactants per pairing are supported.")
    
pairing_labels.append(label)

ratios = padded_ratio_vectors

for i in range (0, len(ratios)):
    ratio = ratios[i]
    substock_mixture_index = substock_name_list[i]

    #Rough CMC
    experiment,small_exp = experimental_planner.generate_exp_flexible(surfactants, ratio, rough_screen=True)
    sub_stock_vols = experiment['surfactant_sub_stock_vols']
    wellplate_data = experiment['df']
    samples_per_assay = wellplate_data.shape[0]

    #Execute the sample workflow.
    cmc_rough = sample_workflow(starting_wp_index,sub_stock_vols,substock_mixture_index,wellplate_data,folder,save_modifier='rough')
    print("Rough CMC: ", cmc_rough)
    starting_wp_index+=samples_per_assay

    #Refined CMC
    experiment,small_exp = experimental_planner.generate_exp_flexible(surfactants, ratio,rough_screen=False,estimated_CMC=cmc_rough)
    sub_stock_vols = experiment['surfactant_sub_stock_vols']
    wellplate_data = experiment['df']
    samples_per_assay = wellplate_data.shape[0]

    cmc_refined = sample_workflow(starting_wp_index,sub_stock_vols,substock_mixture_index,wellplate_data,folder, save_modifier='fine')
    print("Refined CMC: ", cmc_refined)
    starting_wp_index+=samples_per_assay


    #Check to see if we need a new well_plate
    if starting_wp_index >= 48:
        lash_e.discard_used_wellplate()
        lash_e.grab_new_wellplate()
        #Return the wellplate to the disposal
        #Grab a new wellplate
        starting_wp_index = 0

    #input("****Press enter to continue to next surfactant")
    print("Moving on to next surfactant")

if lash_e.nr_track.NR_OCCUPIED == True:
    lash_e.discard_used_wellplate() #Discard the wellplate if it is occupied
    print("Workflow complete and wellplate discarded")

if logging:
    log_file.close()
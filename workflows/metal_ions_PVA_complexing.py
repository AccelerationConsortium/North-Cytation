from re import search
import sys

sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np
import analysis.spectral_difference as spec_dif
import recommenders.metal_ion_PVA_optimizer as recommender
from datetime import datetime
import north_gui
import os
import slack_agent
import itertools

def analyze_data(reference_df, target_df):
    #Take the reference data, target data, return the wavelengths
    spec_dif.get_absolute_peak_wavelength_difference()
    return None

#Create some reference samples to draw the color from
def get_reference_solutions(reference_solution,replicates):
    reference_df = pd.DataFrame({reference_solution: [METAL_ION_VOLUME_PER_WELL ], 'water': [TOTAL_VOLUME_PER_WELL-METAL_ION_VOLUME_PER_WELL]})
    reference_df  = reference_df.loc[reference_df.index.repeat(replicates)].reset_index(drop=True)
    new_index = range(wells_created,replicates)
    reference_df = reference_df.set_index(pd.Index(new_index))
    lash_e.nr_robot.dispense_from_vials_into_wellplate(reference_df,[reference_solution,'water'])
    wells_created += replicates

def mix_PVA_ion_solutions(reference_solution,suggested_data,replicates):
    duplicated_df = pd.concat([suggested_data]*replicates, axis=1, ignore_index=True)
    duplicated_df[reference_solution]=METAL_ION_VOLUME_PER_WELL
    new_index = range(wells_created,duplicated_df.shape[0]+wells_created)
    reference_df = reference_df.set_index(pd.Index(new_index))
    lash_e.nr_robot.dispense_from_vials_into_wellplate(reference_df,[reference_solution,'Water', 'PVA_1', 'PVA_2', 'PVA_3', 'HCl', 'NaOH'])
    wells_created += replicates

#Start program
input_vial_status_file="../utoronto_demo/status/metal_ion_PVA_vials.txt"
vial_status = pd.read_csv(input_vial_status_file, sep=",")
print(vial_status)
active_vials = vial_status['vial_index'].values[1:5]
print("Active vials: ", active_vials) #Should be 1,2,3,4 which is only their ID not their location
#Initialize the workstation, which includes the robot, track, cytation and photoreactors
lash_e = Lash_E(input_vial_status_file)
input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

#List of input parameters for the experiment
PROTOCOL_FILE=r"C:\Protocols\Color_Matching\Sweep_A1A6.prt"
random_recs = False #Change this
seed = 3 #No need to change this
robotics_on = False #Do we want to skip the actuation?
upper_bound = 100
starting_samples = 5
number_cycles = 5
samples_per_cycle = 6
replicates = 2
METAL_ION_VOLUME_PER_WELL = 0.08
TOTAL_VOLUME_PER_WELL = 0.24
wells_created = 0

# #Get initial recs
campaign,searchspace = recommender.initialize_campaign(upper_bound,seed,random_recs=random_recs) 

campaign,recommendations = recommender.get_initial_recommendations(campaign,starting_samples)
print(recommendations/1000)
print(f"Random: {random_recs}, seed #: {seed}")

print("Searchspace Size: ", searchspace)

#input("Pausing to wait for enter...")

#Experimental workflow and data gathering
if robotics_on:
    get_reference_solution(replicates)

#Start the GUI
plotter = north_gui.RealTimePlot(num_subplots=3, 
                                 titles=['Target Spectra', 'Closest Spectra Per Batch','Optimization Target'],
                                   x_titles=['Wavelength (nm)', 'Wavelength (nm)', 'Batch #'],
                                   y_titles=['Intensity', 'Intensity', 'Spectral Match (0 optimal)'],
                                   styles=[{}, {}, {"marker": "o", "linestyle": "None"}])
colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
graph_color = next(colors)

# # #Get analysis
results,ref_file = analyze_data(SOURCE_DATA_FOLDER)
print("Results: ", results)
recommendations['output']=results
campaign_data = recommendations

#Add data to GUI
plotter.add_data(2,[1]*5,results,plot_type='o',color=graph_color)

#Subsequent measurements: TODO add in condition for stopping... Probably best to use Ilya's code
for i in range (0,num_files):
     campaign,recommendations = recommender.get_new_recs_from_results(campaign,recommendations,6)
     graph_color = next(colors)

     print("New Recs: ", recommendations/1000)
     if robotics_on:
        find_closer_color_match(file_list[i],6*(i+1),recommendations/1000)
     
     results,ref_file = analyze_data(SOURCE_DATA_FOLDER,ref_file,dif_type=analysis_type)
     print("Results: ", results)
     recommendations['output']=results
     campaign_data = pd.concat([campaign_data, recommendations], ignore_index=True)
     plotter.add_data(2,[i+2]*6,results,plot_type='o',color=graph_color)

print("Final data:\n", campaign_data)
# Get current date and time
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

try:
    # Define filename... Eventually add the random number seed
    filename = folder_path+f"/data_{timestamp}_seed_{seed}.csv"
    campaign_data.to_csv(filename)
    print(f"Saved CSV as: {filename}")
except:
    print ("Issue saving data")
try: 
    file_name = plotter.save_figure(folder_path)
    plotter.upload_file(file_name,"Color matching demo is complete: Here is the visual summary: ")
except Exception as e:
    print ("Issue saving figure", e)





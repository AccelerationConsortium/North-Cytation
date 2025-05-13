from re import search
import sys
import wave
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np
import analysis.spectral_analyzer as analyzer
import recommenders.color_matching_optimizer as recommender
from datetime import datetime
import north_gui
import os
import itertools

def mix_wells(wells, wash_index=5, wash_volume=0.1, repeats=2):
    lash_e.nr_robot.get_pipet(1)
    for well in wells:
        lash_e.nr_robot.pipet_from_wellplate(well,wash_volume)
        lash_e.nr_robot.pipet_from_wellplate(well,wash_volume,aspirate=False,move_to_aspirate=False)
        for i in range (0, repeats):
            lash_e.nr_robot.pipet_from_wellplate(well,wash_volume,move_to_aspirate=False)
            lash_e.nr_robot.pipet_from_wellplate(well, wash_volume,aspirate=False,move_to_aspirate=False)
        
        lash_e.nr_robot.aspirate_from_vial(wash_index,wash_volume)
        lash_e.nr_robot.dispense_into_vial(wash_index,wash_volume,initial_move=False)
        for i in range (0,repeats):
            lash_e.nr_robot.dispense_from_vial_into_vial(wash_index,wash_index,wash_volume,move_to_aspirate=False,move_to_dispense=False,buffer_vol=0)

#Define your workflow! Make sure that it has parameters that can be changed!
def create_initial_colors(measurement_file, initial_guesses, initial_volumes):
    #lash_e.grab_new_wellplate()
    
    well_volume = 0.24

    lash_e.nr_robot.aspirate_from_vial(0,well_volume)
    lash_e.nr_robot.dispense_into_wellplate([0], [well_volume])
    lash_e.nr_robot.remove_pipet()

    initial_wells = range(1,1+initial_guesses)

    initial_volumes.index = initial_wells #Set wells
    print("Initial:\n", initial_volumes)

    lash_e.nr_robot.dispense_from_vials_into_wellplate(initial_volumes,active_vials)
    mix_wells(initial_wells)
    lash_e.nr_robot.remove_pipet()

    print("Measurement file: ", measurement_file)
    data = lash_e.measure_wellplate(measurement_file,range(0,initial_guesses+1))
    return data

def analyze_data(data, dif_type, reference_spectra=None,plotter=None):
    
    wavelengths = analyzer.get_wavelengths_from_df(data)
    differences_list = []

    if reference_spectra is None:
        comparison_index_list = list(range(1, 1+initial_recs)) #Make more robust
        reference_spectra = analyzer.get_spectra_from_df_using_index(data,0)
        print(reference_spectra)
        if plotter is not None:
            plotter.add_data(0,wavelengths,reference_spectra,color=graph_color)
    else:
        comparison_index_list = list(range(0, recs_per_cycle)) #Make more robust

    for i in comparison_index_list:
        spectra = analyzer.get_spectra_from_df_using_index(data,i)
        if dif_type == "spectral":
            difference = analyzer.get_absolute_spectral_difference(wavelengths,reference_spectra,wavelengths,spectra)
            differences_list.append(difference)
    
    top_spectra_index = differences_list.index(min(differences_list))
    top_spectra = analyzer.get_spectra_from_df_using_index(data,top_spectra_index)
    plotter.add_data(1,wavelengths,top_spectra,color=graph_color)

    return differences_list,reference_spectra 

def find_closer_color_match(measurement_file,wells,volumes):
    volumes.index = wells
    print("New Volumes:\n", volumes)

    lash_e.nr_robot.dispense_from_vials_into_wellplate(volumes,active_vials)
    mix_wells(wells)
    lash_e.nr_robot.remove_pipet()

    print("Measurement file: ", measurement_file)
    data = lash_e.measure_wellplate(measurement_file, wells)
    return data


#Start program
input_vial_status_file="../utoronto_demo/status/color_matching_vials.csv"
vial_status = pd.read_csv(input_vial_status_file, sep=",")
print(vial_status)
active_vials = vial_status['vial_index'].values[1:5]
print("Active vials: ", active_vials) #Should be 1,2,3,4 which is only their ID not their location
input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

#Initialize the workstation, which includes the robot, track, cytation and photoreactors
lash_e = Lash_E(input_vial_status_file)
lash_e.nr_robot.c9.home_robot()

# Create the full path and the folder
# Get current date and time as a string
SOURCE_DATA_FOLDER = "C://Users//Imaging Controller//Desktop//Color_Matching"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_path = os.path.join(SOURCE_DATA_FOLDER, timestamp)
os.makedirs(folder_path, exist_ok=True)
print(f"Folder created at: {folder_path}")

#List of measurement files for Cytation. Unfortunately inflexible
measurement_file=r"C:\Protocols\Color_Matching\Sweep_A1A6.prt"

# #Get initial recs
method = "method_a" #Change this
random_recs = False #Change this
seed = 12 #No need to change this
recreate_color_at_end = True #Do we want to make the vial at the end?
robotics_on = True #Do we want to skip the actuation?
num_cycles = 2 #6
initial_recs = 5
recs_per_cycle = 3 

if method == "method_a":
    upper_bound = 50
    analysis_type = "spectral"
elif method == "method_b":   
    upper_bound = 5
    analysis_type = "discrete"

campaign,searchspace = recommender.initialize_campaign(upper_bound,seed,random_recs=random_recs) 

campaign,recommendations = recommender.get_initial_recommendations(campaign,initial_recs)
print(recommendations/1000)
print(f"Model method: {method}, random: {random_recs}, seed #: {seed}")

#print("Searchspace Size: ", searchspace)

#input("Pausing to wait for enter...")

#Experimental workflow and data gathering
if robotics_on:
    data = create_initial_colors(measurement_file,initial_recs,recommendations/1000)
    data.to_csv(folder_path+"/initial_spectra.csv",index=False)
else:
    data = pd.read_csv('C:\\Users\\Imaging Controller\\Desktop\\Color_Matching\\2025-04-24_12-17-08\\initial_spectra.csv',sep=',',index_col=False)
    input()

#Start the GUI
plotter = north_gui.RealTimePlot(num_subplots=3, 
                                 titles=['Target Spectra', 'Closest Spectra Per Batch','Optimization Target'],
                                   x_titles=['Wavelength (nm)', 'Wavelength (nm)', 'Batch #'],
                                   y_titles=['Intensity', 'Intensity', 'Spectral Match (0 optimal)'],
                                   styles=[{}, {}, {"marker": "o", "linestyle": "None"}])
colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
graph_color = next(colors)

# # #Get analysis
results,reference_spectra= analyze_data(data, analysis_type,plotter=plotter)
print("Results: ", results)
recommendations['output']=results
campaign_data = recommendations

#Add data to GUI
plotter.add_data(2,[1]*initial_recs,results,plot_type='o',color=graph_color)

lash_e.nr_robot.move_home()
lash_e.nr_robot.c9.home_robot()

#Subsequent measurements
for i in range (0,num_cycles):
     campaign,recommendations = recommender.get_new_recs_from_results(campaign,recommendations,recs_per_cycle)
     graph_color = next(colors)

     print("New Recs: ", recommendations/1000)
     if robotics_on:
        wells = range(initial_recs+1+recs_per_cycle*i,initial_recs+1+recs_per_cycle*(i+1))
        print("New wells, ", wells)
        data = find_closer_color_match(measurement_file,wells,recommendations/1000)
        data.to_csv(folder_path+f"/spectra_cycle_{i}.csv",index=False)
     
     results,reference_spectra = analyze_data(data,analysis_type,reference_spectra,plotter)
     print("Results: ", results)
     recommendations['output']=results
     campaign_data = pd.concat([campaign_data, recommendations], ignore_index=True)
     plotter.add_data(2,[i+2]*recs_per_cycle,results,plot_type='o',color=graph_color)

     lash_e.nr_robot.move_home()
     lash_e.nr_robot.c9.home_robot()

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

best_result_index = np.argmin(campaign_data['output'].values)
print("Best result index: ", best_result_index)
best_composition=campaign_data.iloc[best_result_index].tolist()


#Recreate the color in a vial at the end
if recreate_color_at_end:
    recreate_volume = 2.0
    recreate_vial = np.array(best_composition)*recreate_volume/240

    for i in range (0, len(active_vials)):
        color_volume = recreate_vial[i]
        if color_volume > 0:
            if color_volume < 1.0:
                lash_e.nr_robot.dispense_from_vial_into_vial(active_vials[i],6,color_volume)
            else:
                for j in range (0,2):
                    lash_e.nr_robot.dispense_from_vial_into_vial(active_vials[i],6,color_volume/2)
            lash_e.nr_robot.remove_pipet()

    lash_e.nr_robot.vortex_vial(6,5)
    lash_e.nr_robot.return_vial_home(6)
    lash_e.nr_robot.move_home()





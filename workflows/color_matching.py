import sys

sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np
import analysis.spectral_difference as spec_dif
import recommenders.color_matching_optimizer as recommender
import random
from datetime import datetime
import north_gui
import os
import slack_agent

def mix_wells(wells, wash_location=3, wash_volume=0.1, repeats=2):
    for well in wells:
        lash_e.nr_robot.aspirate_from_vial(wash_location,wash_volume)
        lash_e.nr_robot.dispense_into_vial(wash_location,wash_volume,initial_move=False)
        for i in range (0,repeats):
            lash_e.nr_robot.dispense_from_vial_into_vial(wash_location,wash_location,wash_volume,move_to_aspirate=False,move_to_dispense=False,buffer_vol=0)
        
        lash_e.nr_robot.pipet_from_wellplate(well,wash_volume)
        lash_e.nr_robot.pipet_from_wellplate(well,wash_volume,aspirate=False,move_to_aspirate=False)
        for i in range (0, repeats):
            lash_e.nr_robot.pipet_from_wellplate(well,wash_volume,move_to_aspirate=False)
            lash_e.nr_robot.pipet_from_wellplate(well, wash_volume,aspirate=False,move_to_aspirate=False)

#Define your workflow! Make sure that it has parameters that can be changed!
def create_initial_colors(measurement_file, initial_guesses, initial_volumes):
    #lash_e.grab_new_wellplate()
    
    well_volume = 0.24

    lash_e.nr_robot.aspirate_from_vial(0,well_volume)
    lash_e.nr_robot.dispense_into_wellplate([0], [well_volume])
    lash_e.nr_robot.finish_pipetting()

    initial_wells = range(1,1+initial_guesses)
    #active_vials = range(1,5)

    initial_volumes.index = initial_wells #Set wells
    print("Initial:\n", initial_volumes)

    lash_e.nr_robot.dispense_from_vials_into_wellplate(initial_volumes,active_vials)
    mix_wells(initial_wells)
    lash_e.nr_robot.finish_pipetting()

    print("Measurement file: ", measurement_file)
    lash_e.measure_wellplate(measurement_file)

def analyze_data(source_data_folder, reference_file=None,  reference_index=0):
    
    #Step 1: Look in folder for most recent file
    comparison_file = spec_dif.get_most_recent_file(source_data_folder)

    print("Newest file:", comparison_file.split('\\')[-1])

    if reference_file is None:
        reference_file = comparison_file
        comparison_index_list = list(range(1, 6)) #Make more robust
    else:
        comparison_index_list = list(range(0, 6)) #Make more robust

    print(comparison_index_list)

    differences_list = spec_dif.get_differences(reference_file, reference_index, comparison_file, comparison_index_list,plotter=plotter)

    return differences_list,reference_file   

def find_closer_color_match(measurement_file,start_index,volumes):

    wells = range(start_index,6+start_index)
    #active_vials = range(1,5)

    volumes.index = wells
    print("New Volumes:\n", volumes)

    lash_e.nr_robot.dispense_from_vials_into_wellplate(volumes,active_vials)
    mix_wells(wells)
    lash_e.nr_robot.finish_pipetting()

    print("Measurement file: ", measurement_file)
    lash_e.measure_wellplate(measurement_file)


#Start program
input_vial_status_file="../utoronto_demo/status/color_matching_vials.txt"
vial_status = pd.read_csv(input_vial_status_file, sep=r"\t", engine="python")
print(vial_status)
active_vials = vial_status['vial index'].values[1:5]
print("Active vials: ", active_vials)
input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

#Initialize the workstation, which includes the robot, track, cytation and photoreactors
lash_e = Lash_E(input_vial_status_file)

# Create the full path and the folder
# Get current date and time as a string
SOURCE_DATA_FOLDER = "C://Users//Imaging Controller//Desktop//Color_Matching"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_path = os.path.join(SOURCE_DATA_FOLDER, timestamp)
os.makedirs(folder_path, exist_ok=True)
print(f"Folder created at: {folder_path}")

#List of measurement files for Cytation. Unfortunately this will be annoying!
#TODO: Create this list of files,adjust paths, see if I can edit them
#r"C:\Protocols\Quick_Measurement.prt"

file_0=r"C:\Protocols\Color_Matching\Sweep_A1A6.prt"
file_1=r"C:\Protocols\Color_Matching\Sweep_A7A12.prt"
file_2=r"C:\Protocols\Color_Matching\Sweep_B1B6.prt"
file_3=r"C:\Protocols\Color_Matching\Sweep_B7B12.prt"
file_4=r"C:\Protocols\Color_Matching\Sweep_C1C6.prt"
file_5=r"C:\Protocols\Color_Matching\Sweep_C7C12.prt"
file_6=r"C:\Protocols\Color_Matching\Sweep_D1D6.prt"
file_7=r"C:\Protocols\Color_Matching\Sweep_D7D12.prt"
file_list = [file_1, file_2, file_3,file_4,file_5,file_6,file_7]
file_list = []
num_files = len(file_list)

#ref_file = r"C:\Users\Imaging Controller\Desktop\Color_Matching\Experiment1_250130_151633_.txt"
# #Get initial recs
campaign = recommender.initialize_campaign(50,1) #upper bound=50, seed=0
campaign,recommendations = recommender.get_initial_recommendations(campaign,5)
print(recommendations/1000)

# #Experimental workflow and data gathering
create_initial_colors(file_0,5,recommendations/1000)

plotter = north_gui.RealTimePlot(num_subplots=3, styles=[{"color": "r"}, {"color": "g"}, {"color": "b", "marker": "o", "linestyle": "None"}])

# # #Get analysis
results,ref_file = analyze_data(SOURCE_DATA_FOLDER)
print("Results: ", results)
recommendations['output']=results
campaign_data = recommendations

plotter.add_data(2,[1]*5,results,plot_type='o')


#Subsequent measurements: TODO add in condition for stopping... Probably best to use Ilya's code
for i in range (0,num_files):
     campaign,recommendations = recommender.get_new_recs_from_results(campaign,recommendations,6)
     
     print("New Recs: ", recommendations/1000)
     find_closer_color_match(file_list[i],6*(i+1),recommendations/1000)
     results,ref_file = analyze_data(SOURCE_DATA_FOLDER,ref_file)
     print("Results: ", results)

     recommendations['output']=results
     campaign_data = pd.concat([campaign_data, recommendations], ignore_index=True)
     plotter.add_data(2,[i+2]*6,results,plot_type='o')

print("Final data:\n", campaign_data)
# Get current date and time
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Define filename... Eventually add the random number seed
filename = f"data_{timestamp}.csv"

try:
    campaign_data.to_csv(filename)
    print(f"Saved CSV as: {filename}")
except:
    print ("Issue saving data")
try: 
    file_name = plotter.save_figure()
    slack_agent.send_slack_message("Color matching demo is complete: Here is the visual summary")
    plotter.upload_file(file_name)
except:
    print ("Issue saving figure")

best_result_index = np.argmin(campaign_data['output'].values)
print("Best result index: ", best_result_index)
best_composition=campaign_data.iloc[best_result_index].tolist()
print("Best composition: ", best_composition)

recreate_volume = 2.0

recreate_vial = np.array(best_composition)*recreate_volume/1000

for i in range (0, len(active_vials)):
    color_volume = recreate_vial[i]
    if color_volume < 1.0:
        lash_e.nr_robot.dispense_from_vial_into_vial(active_vials[i],6,color_volume)
    else:
        lash_e.nr_robot.dispense_from_vial_into_vial(active_vials[i],6,color_volume-1)
        lash_e.nr_robot.dispense_from_vial_into_vial(active_vials[i],6,1)
    lash_e.nr_robot.remove_pipet()



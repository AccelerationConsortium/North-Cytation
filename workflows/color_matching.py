import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import numpy as np
import analysis.spectral_difference as spec_dif
import recommenders.color_matching_optimizer as recommender
import random
import datetime
import north_gui

#Define your workflow! Make sure that it has parameters that can be changed!
def create_initial_colors(measurement_file, initial_guesses, initial_volumes):
    #lash_e.grab_new_wellplate()
    
    well_volume = 0.24

    lash_e.nr_robot.aspirate_from_vial(0,well_volume)
    lash_e.nr_robot.dispense_into_wellplate([0], [well_volume])
    lash_e.nr_robot.finish_pipetting()

    initial_wells = range(1,1+initial_guesses)
    active_vials = range(1,5)

    initial_volumes.index = initial_wells #Set wells
    print("Initial:\n", initial_volumes)

    lash_e.nr_robot.dispense_from_vials_into_wellplate(initial_volumes,active_vials)
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

    differences_list = spec_dif.get_differences(reference_file, reference_index, comparison_file, comparison_index_list,plotter)

    return differences_list,reference_file   

def find_closer_color_match(measurement_file,start_index,volumes):

    wells = range(start_index,6+start_index)
    active_vials = range(1,5)

    volumes.index = wells
    print("New Volumes:\n", volumes)

    lash_e.nr_robot.dispense_from_vials_into_wellplate(volumes,active_vials)
    lash_e.nr_robot.finish_pipetting()

    print("Measurement file: ", measurement_file)
    lash_e.measure_wellplate(measurement_file)


#Start program
input_vial_status_file="../utoronto_demo/status/color_matching_vials.txt"
vial_status = pd.read_csv(input_vial_status_file, sep=r"\t", engine="python")
print(vial_status)
input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

#Initialize the workstation, which includes the robot, track, cytation and photoreactors
lash_e = Lash_E(input_vial_status_file)

#List of measurement files for Cytation. Unfortunately this will be annoying!
#TODO: Create this list of files,adjust paths, see if I can edit them
#r"C:\Protocols\Quick_Measurement.prt"

file_0=r"C:\Protocols\Color_Matching\Sweep_A1A6.prt"
file_1=r"C:\Protocols\Color_Matching\Sweep_A7A12.prt"
file_2=r"C:\Protocols\Color_Matching\Sweep_B1B6.prt"
file_3=r"C:\Protocols\Color_Matching\Sweep_B7B12.prt"
file_list = [file_1, file_2, file_3]

#TODO: Change this to the folder we save to upstairs
SOURCE_DATA_FOLDER = "C://Users//Imaging Controller//Desktop//Color_Matching"
#ref_file = r"C:\Users\Imaging Controller\Desktop\Color_Matching\Experiment1_250129_174113_.txt"
# #Get initial recs
campaign = recommender.initialize_campaign(20)
campaign,recommendations = recommender.get_initial_recommendations(campaign,5)
print(recommendations/1000)

# #Experimental workflow and data gathering
create_initial_colors(file_0,5,recommendations/1000)

plotter = north_gui.RealTimePlot(num_subplots=3, styles=[{"color": "r"}, {"color": "g"}, {"color": "b", "linestyle": "--"}])

# # #Get analysis
results,ref_file = analyze_data(SOURCE_DATA_FOLDER)
print("Results: ", results)
recommendations['output']=results
campaign_data = recommendations

plotter.add_data(2,[0]*5,results)

# #Subsequent measurements: TODO add in condition for stopping... Probably best to use Ilya's code
for i in range (0,3):
     campaign,recommendations = recommender.get_new_recs_from_results(campaign,recommendations,6)
     
     print("New Recs: ", recommendations/1000)
     find_closer_color_match(file_list[i],6*(i+1),recommendations/1000)
     results,ref_file = analyze_data(SOURCE_DATA_FOLDER,ref_file)
     print("Results: ", results)

     recommendations['output']=results
     campaign_data = pd.concat([campaign_data, recommendations], ignore_index=True)
     plotter.add_data(2,[i+1]*6,results)

print("Final data:\n", campaign_data)
# Get current date and time
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Define filename... Eventually add the random number seed
filename = f"data_{timestamp}.csv"

pd.to_csv(campaign_data)
print(f"Saved CSV as: {filename}")
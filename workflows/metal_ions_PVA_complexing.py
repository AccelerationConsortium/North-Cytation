import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import pandas as pd
import analysis.spectral_analyzer as spectral_analyzer
import recommenders.metal_ion_PVA_optimizer as recommender
from datetime import datetime
import north_gui
import slack_agent
import itertools

#Analyze the spectra we've got to see the peak difference in wavelength from the reference
def analyze_data(wavelengths, reference_df, target_df):
    results = []
    reference_spectra = reference_df.iloc[:, 0].values
    for col in target_df.columns:
        target_spectra = target_df[col].tolist()
        result = spectral_analyzer.get_peak_wavelength_difference(wavelengths,reference_spectra,wavelengths,target_spectra)
        results.append(result)
    return results


def refill_water():
    vial_max_volume = 8
    lash_e.nr_robot.dispense_into_vial_from_reservoir(0,'Water',vial_max_volume - lash_e.nr_robot.get_vial_info('Water','vial_volume'))

def create_FeCl3_solution():
    lash_e.nr_robot.dispense_from_vial_into_vial('FeCl3_stock','FeCl3', 0.16)
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.dispense_into_vial_from_reservoir(0,'FeCl3', 7.84) #Add water to dilute the solution

#Create some reference samples to draw the color from
def get_reference_solutions(reference_solution,replicates, first_well):
    reference_df = pd.DataFrame({reference_solution: [METAL_ION_VOLUME_PER_WELL ], 'Water': [TOTAL_VOLUME_PER_WELL-METAL_ION_VOLUME_PER_WELL]})
    reference_df  = reference_df.loc[reference_df.index.repeat(replicates)].reset_index(drop=True)
    print(reference_df)
    new_index = range(first_well,replicates)
    reference_df = reference_df.set_index(pd.Index(new_index))
    lash_e.nr_robot.dispense_from_vials_into_wellplate(reference_df,[reference_solution,'Water'])


def mix_PVA_ion_solutions(reference_solution,suggested_data,replicates,first_well):
    duplicated_df = suggested_data.loc[suggested_data.index.repeat(replicates)].reset_index(drop=True) / 1000
    duplicated_df[reference_solution]=METAL_ION_VOLUME_PER_WELL
    new_index = range(first_well,duplicated_df.shape[0]+first_well)
    duplicated_df = duplicated_df.set_index(pd.Index(new_index))
    print (duplicated_df)
    lash_e.nr_robot.dispense_from_vials_into_wellplate(duplicated_df,[reference_solution,'Water', 'PVA_1', 'PVA_2', 'PVA_3', 'HCl', 'Acid_2', 'Acid_3', 'NaOH'])


simulate = True

#Initialize the workstation, which includes the robot, track, cytation and photoreactors
input_vial_status_file="../utoronto_demo/status/metal_ion_PVA_vials.csv"
lash_e = Lash_E(input_vial_status_file, simulate=simulate)
lash_e.nr_robot.check_input_file()  # Outputs the values in input_vial_status_file and user must confirm by typing Enter if everything looks ok to proceed

#List of input parameters for the experiment
PROTOCOL_FILE=r"C:\Protocols\PVA_Spectra.prt"
random_recs = False 
seed = 27
robotics_on = False
upper_bound = 100
starting_samples = 5
number_cycles = 5
samples_per_cycle = 6
replicates = 2
METAL_ION_VOLUME_PER_WELL = 0.120
TOTAL_VOLUME_PER_WELL = 0.24
wells_created = 0
starting_well = 0

# #Get initial recs
campaign,searchspace = recommender.initialize_campaign(upper_bound,seed,random_recs=random_recs) 
campaign,recommendations = recommender.get_initial_recommendations(campaign,starting_samples)
print("Initial Recommendations: ", recommendations/1000)
print(f"Random: {random_recs}, seed #: {seed}")
print("Searchspace Size: ", searchspace)
input("Pausing to wait for enter...")

#Start the GUI
plotter = north_gui.RealTimePlot(num_subplots=2, 
                                 titles=['Target Spectra', 'Optimization Target'],
                                   x_titles=['Wavelength (nm)','Batch #'],
                                   y_titles=['Intensity', 'Shift (nm)'],
                                   styles=[{}, {"marker": "o", "linestyle": "None"}])
colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
graph_color = next(colors)

#Create the FeCl3 solution
refill_water()
create_FeCl3_solution()

#Experimental workflow and data gathering
get_reference_solutions('FeCl3', replicates,wells_created)
wells_created += replicates

mix_PVA_ion_solutions('FeCl3', recommendations, replicates,wells_created)
wells_created += replicates*recommendations.shape[0]


wells = range(starting_well,wells_created)
data_wells = lash_e.measure_wellplate(PROTOCOL_FILE,wells)
starting_well=wells_created

if not simulate:
    #Separate out the data
    wavelengths = data_wells['Wavelengths'].values
    reference_data = data_wells.iloc[:, 1:1+replicates]
    initial_data = data_wells.iloc[:, 1+replicates:]
    average_reference_spectra = spectral_analyzer.average_spectra(reference_data,replicates)
    average_initial_spectra = spectral_analyzer.average_spectra(initial_data,replicates)
    plotter.add_data(0,wavelengths,average_reference_spectra.values)

    #Analyze the data  
    results = analyze_data(wavelengths, average_reference_spectra, average_initial_spectra)
    print("Results: ", results)
    max_index = results.index(max(results))
    best_initial_spectra = average_initial_spectra.iloc[:,max_index]
    graph_color = next(colors)
    plotter.add_data(0,wavelengths,best_initial_spectra.values)
    recommendations['output']=results
    campaign_data = recommendations
    plotter.add_data(1,[1]*starting_samples,results,plot_type='o',color=graph_color)

    #Subsequent measurements: TODO add in condition for stopping...
    for i in range (0,number_cycles):
        campaign,recommendations = recommender.get_new_recs_from_results(campaign,recommendations,samples_per_cycle)
        print("New Recs: ", recommendations/1000)
        mix_PVA_ion_solutions('FeCl3', recommendations, replicates)
        wells = range(starting_well,wells_created)
        data_wells = lash_e.measure_wellplate(PROTOCOL_FILE,wells)
        average_spectral_data = spectral_analyzer.average_spectra(data_wells.iloc[:, 1:])
        results = analyze_data(wavelengths, average_reference_spectra, average_spectral_data)
        print("Results: ", results)
        max_index = results.index(max(results))
        best_spectra = average_reference_spectra.iloc[:,max_index]
        graph_color = next(colors)
        plotter.add_data(0,wavelengths,best_spectra.values)
        recommendations['output']=results
        campaign_data = pd.concat([campaign_data, recommendations], ignore_index=True)
        plotter.add_data(1,[i+2]*samples_per_cycle,results,plot_type='o',color=graph_color)
        starting_well = wells_created





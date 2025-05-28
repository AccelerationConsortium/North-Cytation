import sys
import time
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E 
import pandas as pd

def stop_reaction_and_take_aliquot(lash_e,reaction_mixture_index,water_index,first_well_index,replicates,well_volume=0.2):
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=1)
    lash_e.photoreactor.turn_off_reactor_led(reactor_num=1)
    lash_e.move_vial_to_location(reaction_mixture_index,location='clamp',location_index=0)
    lash_e.nr_robot.uncap_clamp_vial()
    lash_e.nr_robot.aspirate_from_vial(reaction_mixture_index, well_volume*replicates,track_height=False)
    wells = range(first_well_index,first_well_index+replicates)
    lash_e.nr_robot.dispense_into_wellplate(wells, [well_volume]*replicates)
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.aspirate_from_vial(water_index, well_volume*replicates,track_height=False)
    lash_e.nr_robot.dispense_into_wellplate(wells, [well_volume]*replicates)
    mix_current_sample(lash_e,first_well_index,new_pipet=False,volume=0.2)

def dispense_from_vial_into_reaction(lash_e,reaction_mixture_index,sample_index,volume=0.02):
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=1)
    lash_e.nr_robot.dispense_from_vial_into_vial(reaction_mixture_index,sample_index,volume=volume)
    mix_current_sample(lash_e,sample_index,volume=0.8)
    lash_e.nr_robot.remove_pipet()
    lash_e.photoreactor.turn_on_reactor_fan(reactor_num=1,rpm=600)
    lash_e.nr_robot.move_home()
    lash_e.nr_robot.c9.home_robot()

def mix_current_sample(lash_e, sample_index, new_pipet=False,repeats=3, volume=0.25):
    if new_pipet:
        lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.dispense_from_vial_into_vial(sample_index,sample_index,volume=volume,move_to_dispense=False,buffer_vol=0)
    for _ in range (repeats-1):
        lash_e.nr_robot.dispense_from_vial_into_vial(sample_index,sample_index,volume=volume,move_to_aspirate=False,move_to_dispense=False,buffer_vol=0)
    lash_e.nr_robot.remove_pipet() # This step is for pipetting up and down *3 to simulate mixing.

def measure_wellplate(lash_e,sample_index,first_well_index,cytation_protocol_file_path,replicates,well_volume=0.2):
    lash_e.nr_robot.aspirate_from_vial(sample_index, well_volume*replicates,track_height=False)
    wells = range(first_well_index,first_well_index+replicates)
    lash_e.nr_robot.dispense_into_wellplate(wells, [well_volume]*replicates)
    lash_e.nr_robot.remove_pipet()
    data_out = lash_e.measure_wellplate(cytation_protocol_file_path, wells_to_measure=wells)
    output_file = r'C:\Users\Imaging Controller\Desktop\SQ\output_'+str(first_well_index)+'.txt'
    data_out.to_csv(output_file, sep=',')

#Define your workflow! Make sure that it has parameters that can be changed!
def peroxide_workflow(sample_incubation_time=30*60,replicates=3): #Reagent incubation time=20 mins; sample incubation time is 18 mins; sample platereading interval is 5 mins.

  #This section is simply to create easier to remember and read indices for the vials
    #vial_numbers = vial_status['vial_index'].values #Gives you the values
    reaction_mixture_index = lash_e.nr_robot.get_vial_index_from_name('Rxn_Mixture') #Get the ID of our target reactor
    Aniline_index = lash_e.nr_robot.get_vial_index_from_name('Aniline')
    water_index=lash_e.nr_robot.get_vial_index_from_name('Water')

    # Initial State of your Vials, so the robot can know where to pipet
    INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/peroxide_assay.csv"
    MEASUREMENT_PROTOCOL_FILE =r"C:\Protocols\SQ_Peroxide.prt"

    # Initial State of your Vials, so the robot can know where to pipet. pd DataFrame created from input txt file.
    vial_status = pd.read_csv(INPUT_VIAL_STATUS_FILE, sep=",")
    print(vial_status)

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=True)

  
    
    #Get the active indices
    num_samples = vial_status.shape[0]-3 #Gets the total number of samples from the input vial (-3 because the first three vials are reagent vials)
    sample_indices = vial_status.index.values[3:] #Gets the indices for the samples (3-8 inclusive)

    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")    

    lash_e.nr_robot.return_vial_home(reaction_mixture_index)
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=1)
    lash_e.photoreactor.turn_off_reactor_led(reactor_num=1)

    lash_e.nr_robot.move_home()
        
peroxide_workflow() #Run your workflow


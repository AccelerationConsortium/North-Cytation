import sys
import time
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E 
import pandas as pd

def dispense_from_photoreactor_into_sample(lash_e,reaction_mixture_index,sample_index,volume=0.2):
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=1)
    lash_e.nr_robot.dispense_from_vial_into_vial(reaction_mixture_index,sample_index,volume=volume)
    mix_current_sample(lash_e,sample_index,volume=0.8)
    lash_e.nr_robot.remove_pipet()
    lash_e.photoreactor.turn_on_reactor_fan(reactor_num=1,rpm=600)

def transfer_samples_into_wellplate_and_characterize(lash_e,sample_index,first_well_index,cytation_protocol_file_path,replicates,well_volume=0.2):
    lash_e.nr_robot.aspirate_from_vial(sample_index, well_volume*replicates,track_height=False)
    wells = range(first_well_index,first_well_index+replicates)
    lash_e.nr_robot.dispense_into_wellplate(wells, [well_volume]*replicates)
    lash_e.nr_robot.remove_pipet()
    lash_e.measure_wellplate(cytation_protocol_file_path, wells_to_measure=wells)

def mix_current_sample(lash_e, sample_index, new_pipet=False,repeats=3, volume=0.25):
    if new_pipet:
        lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.dispense_from_vial_into_vial(sample_index,sample_index,volume=volume,move_to_dispense=False,buffer_vol=0)
    for _ in range (repeats-1):
        lash_e.nr_robot.dispense_from_vial_into_vial(sample_index,sample_index,volume=volume,move_to_aspirate=False,move_to_dispense=False,buffer_vol=0)
    lash_e.nr_robot.remove_pipet() # This step is for pipetting up and down *3 to simulate mixing.

#Define your workflow! Make sure that it has parameters that can be changed!
def peroxide_workflow(reagent_incubation_time=20*60,sample_incubation_time=18*60,interval=5*60,replicates=3): #Reagent incubation time=20 mins; sample incubation time is 18 mins; sample platereading interval is 5 mins.
  
    # Initial State of your Vials, so the robot can know where to pipet
    INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/peroxide_assay.txt"
    MEASUREMENT_PROTOCOL_FILE =r"C:\Protocols\SQ_peroxide.prt"

    # Initial State of your Vials, so the robot can know where to pipet. pd DataFrame created from input txt file.
    vial_status = pd.read_csv(INPUT_VIAL_STATUS_FILE, sep=",")
    print(vial_status)

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE)

    #This section is simply to create easier to remember and read indices for the vials
    #vial_numbers = vial_status['vial_index'].values #Gives you the values
    reaction_mixture_index = lash_e.nr_robot.get_vial_index_from_name('Rxn_Mixture') #Get the ID of our target reactor
    reagent_AB_index = lash_e.nr_robot.get_vial_index_from_name('Reagent_A+B')
    water_index=lash_e.nr_robot.get_vial_index_from_name('Water')
    #reagent_B_index = lash_e.nr_robot.get_vial_index_from_name('Reagent_B')
    
    #Get the active indices
    num_samples = vial_status.shape[0]-3 #Gets the total number of samples from the input vial (-3 because the first three vials are reagent vials)
    sample_indices = vial_status.index.values[3:] #Gets the indices for the samples (3-8 inclusive)

    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")    
    
    #Step 1: Add 20 µL "reagent A" (vial_index 0) to "reagent B" (vial_index 1).
    #lash_e.nr_robot.dispense_from_vial_into_vial(reagent_A_index,reagent_B_index,volume=0.02)
    #mix_current_sample(lash_e,reagent_B_index,new_pipet=True,volume=0.8) #New method for mixing
    
    #Step 2: incubate reagent A + B = "Working Reagent" (vial_index 1) for 20 min
   # print("Incubating...")
   # time.sleep(reagent_incubation_time)
   # print("Incubation finished...!")

    # # -> Start from here! 
    # Step 2.5: Add 950 µL water from water_index (vial_index 45) into vial_index 0-5.
    for i in sample_indices: 
        lash_e.nr_robot.dispense_from_vial_into_vial(water_index,i,volume=0.950)

    #Step 3: Add 150 µL "Working Reagent(reagent A+B)" (vial_index 44) to 950 µL deionized water (vial_index 0-5) to dilute the Working Reagent.
    for i in sample_indices: 
        lash_e.nr_robot.dispense_from_vial_into_vial(reagent_AB_index,i,volume=0.150)
        mix_current_sample(lash_e,i,new_pipet=True,volume=0.8)
    
    #Step 4: Move the reaction mixture vial (vial_index 43) to the photoreactor to start the reaction.
    lash_e.nr_robot.move_vial_to_location(reaction_mixture_index, location="photoreactor_array", location_index=0)
    #Turn on photoreactor
    lash_e.photoreactor.turn_on_reactor_led(reactor_num=1,intensity=100)
    lash_e.photoreactor.turn_on_reactor_fan(reactor_num=1,rpm=600)

    #Step 5: Add 200 µL "reaction mixture" (vial in the photoreactor) to "Diluted Working Reagent" (vial_index 0-5). 
    # Six aliquots added at 0, 5, 10, 15, 20, 25 min time marks for incubation (incubation=18 min).
    
    #Create a schedule using the given timings (interval and incubation time)
    schedule = pd.DataFrame(columns=['start_time', 'action', 'sample_index'])
    for i in range (0,num_samples):
        sample_index = sample_indices[i] #Which vial?
        schedule.loc[2*i]=[i*interval, 'dispense_from_reactor', sample_index]
        schedule.loc[2*i+1]=[i*interval+sample_incubation_time, 'measure_samples', sample_index]

    schedule = schedule.sort_values(by='start_time') #sort in ascending time order
    print("Schedule: ", schedule)
    
    start_time = time.time()
    print("Starting timed portion at: ", start_time)
 
    #Let's complete the items one at a time
    items_completed = 0
    starting_well_index = 0
    time_increment = 60
    while items_completed < schedule.shape[0]: #While we still have items to complete in our schedule
        active_item = schedule.iloc[items_completed]
        time_required = active_item['start_time']
        action_required = active_item['action']
        sample_index = active_item['sample_index']
        current_time = time.time()
        measured_items = 0

        #If we reach the triggered item's required time:
        if current_time - start_time > time_required:
            print("Event triggered: " + action_required + f" from sample {sample_index}")
            print(f"Current Elapsed Time: {(current_time - start_time)/60} minutes")
            print(f"Intended Elapsed Time: {(time_required)/60} minutes")

            if action_required=="dispense_from_reactor":
                dispense_from_photoreactor_into_sample(lash_e,reaction_mixture_index,sample_index,volume=0.2)
                items_completed+=1
            elif action_required=="measure_samples":
                transfer_samples_into_wellplate_and_characterize(lash_e,sample_index,starting_well_index,MEASUREMENT_PROTOCOL_FILE,replicates)
                starting_well_index += replicates
                items_completed+=1
                measured_items+=1
        elif current_time - start_time > time_increment:
            print(f"I'm Alive! Current Elapsed Time: {(current_time - start_time)/60} minutes")
            time_increment=time_increment+60
        
        time.sleep(1)

    lash_e.nr_robot.return_vial_home(reaction_mixture_index)
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=1)
    lash_e.photoreactor.turn_off_reactor_led(reactor_num=1)

    lash_e.nr_robot.move_home()
        
peroxide_workflow() #Run your workflow


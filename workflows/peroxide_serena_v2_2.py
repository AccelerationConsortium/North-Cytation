import sys
import time
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E 
import pandas as pd
import slack_agent
import os
from pathlib import Path

def dispense_from_photoreactor_into_sample(lash_e,reaction_mixture_index,sample_index,volume=0.2):
    print("\nDispensing from photoreactor into sample: ", sample_index)
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=1)
    lash_e.nr_robot.dispense_from_vial_into_vial(reaction_mixture_index,sample_index,volume=volume)
    mix_current_sample(lash_e,sample_index,volume=0.8)
    lash_e.nr_robot.remove_pipet()
    lash_e.photoreactor.turn_on_reactor_fan(reactor_num=1,rpm=600)
    lash_e.nr_robot.move_home()
    lash_e.nr_robot.c9.home_robot()
    #for i in range (6,8):
       # lash_e.nr_robot.home_axis(i) #Home the track
    print()

def transfer_samples_into_wellplate_and_characterize(lash_e,sample_index,first_well_index,cytation_protocol_file_path,replicates,output_dir,well_volume=0.2,simulate=False):
    print("\nTransferring sample: ", sample_index, " to wellplate at well index: ", first_well_index)
    lash_e.nr_robot.move_vial_to_location(sample_index, location="main_8mL_rack", location_index=44) #Move sample to safe pipetting position
    lash_e.nr_robot.aspirate_from_vial(sample_index, well_volume*replicates,track_height=True)
    wells = range(first_well_index,first_well_index+replicates)
    lash_e.nr_robot.dispense_into_wellplate(wells, [well_volume]*replicates)
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.return_vial_home(sample_index) #Return sample to home position
    data_out = lash_e.measure_wellplate(cytation_protocol_file_path, wells_to_measure=wells)
    output_file = output_dir / f'output_{first_well_index}.txt'
    # output_file = r'C:\Users\Imaging Controller\Desktop\SQ\output_'+str(first_well_index)+'.txt'
    if not simulate:
        data_out.to_csv(output_file, sep=',')
    print()

def mix_current_sample(lash_e, sample_index, new_pipet=False,repeats=3, volume=0.25):
    print("\nMixing sample: ", sample_index)
    # if new_pipet:
    #     lash_e.nr_robot.remove_pipet()
    # lash_e.nr_robot.dispense_from_vial_into_vial(sample_index,sample_index,volume=volume,move_to_dispense=False,buffer_vol=0)
    # for _ in range (repeats-1):
    #     lash_e.nr_robot.dispense_from_vial_into_vial(sample_index,sample_index,volume=volume,move_to_aspirate=False,move_to_dispense=False,buffer_vol=0)
    # lash_e.nr_robot.remove_pipet() # This step is for pipetting up and down *3 to simulate mixing.
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.vortex_vial(sample_index, 5)
    print()


def get_time(simulate,current_time=None):
    if not simulate:
        return time.time()
    else:
        if current_time is None:
            return 0
        else:
            return current_time + 1

#Define your workflow! Make sure that it has parameters that can be changed!
def peroxide_workflow(reagent_incubation_time=20*60,sample_incubation_time=18*60,interval=5*60,replicates=3): #Reagent incubation time=20 mins; sample incubation time is 18 mins; sample platereading interval is 5 mins.
  
    # Initial State of your Vials, so the robot can know where to pipet
    INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/peroxide_assay_vial_status.csv"
    MEASUREMENT_PROTOCOL_FILE =r"C:\Protocols\SQ_Peroxide.prt"

    # Initial State of your Vials, so the robot can know where to pipet. pd DataFrame created from input txt file.
    vial_status = pd.read_csv(INPUT_VIAL_STATUS_FILE, sep=",")
    print(vial_status)

    #Initialize the workstation, which includes the robot, track, cytation and photoreactors
    SIMULATE = False #Set to True if you want to simulate the robot, False if you want to run it on the real robot
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)

    #This section is simply to create easier to remember and read indices for the vials
    #vial_numbers = vial_status['vial_index'].values #Gives you the values
    reaction_mixture_index = lash_e.nr_robot.get_vial_index_from_name('Rxn_Mixture') #Get the ID of our target reactor
    reagent_AB_index = lash_e.nr_robot.get_vial_index_from_name('Reagent_A+B')
    water_index=lash_e.nr_robot.get_vial_index_from_name('Water')
    #reagent_B_index = lash_e.nr_robot.get_vial_index_from_name('Reagent_B')
    
    #Get the active indices
    sample_indices = vial_status.index.values[3:] #Gets the indices for the samples (3-8 inclusive)

    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")    

    #create file name for the output data
    exp_name = input("Experiment name: ")
    output_dir = Path(r'C:\Users\Imaging Controller\Desktop\SQ') / exp_name #appends exp_name to the output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output directory created at:", output_dir)

    slack_agent.send_slack_message("Peroxide workflow started!")

    # #-> Start from here! 
    # #Step 2.5: Add 950 µL water from water_index (vial_index 45) into vial_index 0-5.
    for i in sample_indices:
        lash_e.nr_robot.dispense_from_vial_into_vial(water_index,i,volume=0.950)
    lash_e.nr_robot.remove_pipet()

    # # #Step 3: Add 150 µL "Working Reagent(reagent A+B)" (vial_index 44) to 950 µL deionized water (vial_index 0-5) to dilute the Working Reagent.
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
    
    SCHEDULE_FILE = r"C:\Users\Imaging Controller\Desktop\SQ\schedule.csv"
    schedule = pd.read_csv(SCHEDULE_FILE, sep=",") #Read the schedule file

    schedule = schedule.sort_values(by='start_time') #sort in ascending time order
    print("Schedule: ", schedule)

    start_time = current_time = get_time(SIMULATE)
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
        current_time = get_time(SIMULATE,current_time)
        measured_items = 0

        #If we reach the triggered item's required time:
        if current_time - start_time > time_required:
            print("\nEvent triggered: " + action_required + f" from sample {sample_index}")
            print(f"Current Elapsed Time: {(current_time - start_time)/60} minutes")
            print(f"Intended Elapsed Time: {(time_required)/60} minutes")

            if action_required=="dispense_from_reactor":
                dispense_from_photoreactor_into_sample(lash_e,reaction_mixture_index,sample_index,volume=0.2)
                items_completed+=1
            elif action_required=="measure_samples":
                transfer_samples_into_wellplate_and_characterize(lash_e,sample_index,starting_well_index,MEASUREMENT_PROTOCOL_FILE,replicates, output_dir,simulate=SIMULATE)
                starting_well_index += replicates
                items_completed+=1
                measured_items+=1
        elif current_time - start_time > time_increment:
            print(f"I'm Alive! Current Elapsed Time: {(current_time - start_time)/60} minutes")
            time_increment=time_increment+60
        
        if not SIMULATE:
            time.sleep(1)
    lash_e.nr_robot.move_home()   

    lash_e.nr_robot.return_vial_home(reaction_mixture_index)
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=1)
    lash_e.photoreactor.turn_off_reactor_led(reactor_num=1)
    lash_e.nr_robot.move_home()   
    slack_agent.send_slack_message("Peroxide workflow completed!")

peroxide_workflow() #Run your workflow


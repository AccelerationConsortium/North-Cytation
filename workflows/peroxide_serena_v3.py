import sys
import time
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E 
import pandas as pd
# import slack_agent
from pathlib import Path
import analysis.cof_analyzer as analyzer

def dispense_from_photoreactor_into_sample(lash_e,reaction_mixture_index,sample_index,volume=0.2):
    print("\nDispensing from photoreactor into sample: ", sample_index)
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=0)
    lash_e.nr_robot.dispense_from_vial_into_vial(reaction_mixture_index,sample_index,volume=volume)
    lash_e.photoreactor.turn_on_reactor_fan(reactor_num=0,rpm=600)
    mix_current_sample(lash_e,sample_index,volume=0.8)
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.move_home()
    # lash_e.nr_robot.c9.home_robot() #removed for now to save time
    #for i in range (6,8):
       # lash_e.nr_robot.home_axis(i) #Home the track
    print()

def transfer_samples_into_wellplate_and_characterize(lash_e,sample_index,first_well_index,cytation_protocol_file_path,replicates,output_dir,simulate=True,well_volume=0.2):
    print("\nTransferring sample: ", sample_index, " to wellplate at well index: ", first_well_index)
    lash_e.nr_robot.move_vial_to_location(sample_index, location="main_8mL_rack", location_index=44) #Move sample to safe pipetting position
    lash_e.nr_robot.aspirate_from_vial(sample_index, well_volume*replicates, track_height=True)
    wells = range(first_well_index,first_well_index+replicates)
    lash_e.nr_robot.dispense_into_wellplate(wells, [well_volume]*replicates)
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.return_vial_home(sample_index) #Return sample to home position
    data_out = lash_e.measure_wellplate(cytation_protocol_file_path, wells_to_measure=wells)
    # output_file = r'C:\Users\Imaging Controller\Desktop\SQ\output_'+str(first_well_index)+'.txt'
    if not simulate:
        output_file = output_dir / f'output_{first_well_index}.txt'
        data_out.to_csv(output_file, sep=',')
        #Use analyzer to analyze the data
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
def peroxide_workflow(lash_e, assay_reagent='Assay_reagent_1', cof_vial='COF_1', set_suffix='',interval=5*60,replicates=3):
  
    # Initial State of your Vials, so the robot can know where to pipet
    INPUT_VIAL_STATUS_FILE = "/Users/serenaqiu/Desktop/north_repository/utoronto_demo/status/peroxide_assay_vial_status_v3.csv"
    MEASUREMENT_PROTOCOL_FILE =r"C:\Protocols\SQ_Peroxide.prt"

    SIMULATE =  True #Set to True if you want to simulate the robot, False if you want to run it on the real robot

    sample_times = [1,5,10,20,30,45] #in minutes
    sample_indices = [f"{t}_min_Reaction{set_suffix}" for t in sample_times]

    #create file name for the output data
    if not SIMULATE:
        exp_name = input("Experiment name: ")
        output_dir = Path(r'C:\Users\Imaging Controller\Desktop\SQ') / exp_name #appends exp_name to the output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Output directory created at:", output_dir)
        # slack_agent.send_slack_message("Peroxide workflow started!")
    else:
        output_dir = None

#-> Start from here! 
    #Step 1: Add 1.9 mL "assay reagent" to sample vials
    for i in sample_indices: 
        lash_e.nr_robot.dispense_from_vial_into_vial(assay_reagent,i,use_safe_location=False, volume=1.9)
    
    #Step 2: Move the reaction mixture vial to the photoreactor to start the reaction.
    lash_e.nr_robot.move_vial_to_location(cof_vial, location="photoreactor_array", location_index=0)
    #Turn on photoreactor
    lash_e.photoreactor.turn_on_reactor_led(reactor_num=0,intensity=100)
    lash_e.photoreactor.turn_on_reactor_fan(reactor_num=0,rpm=600)

    #Step 3: Add 200 µL "reaction mixture" (vial in the photoreactor) to ("assay reagent")" (vial_index 1-6). 
    # 6 aliquots added at 1, 5, 10, 20, 30, 45 min time marks for measuerement
    
    SCHEDULE_FILE = "../utoronto_demo/status/peroxide_assay_schedule.csv"
    schedule = pd.read_csv(SCHEDULE_FILE, sep=",") #Read the schedule file

    schedule = schedule.sort_values(by='start_time') #sort in ascending time order
    print("Schedule: ", schedule)

    start_time = current_time = get_time(SIMULATE)
    print("Starting timed portion at: ", start_time)
    #Let's complete the items one at a time
    items_completed = 0
    current_well_index = 0
    time_increment = 60
    while items_completed < schedule.shape[0]: #While we still have items to complete in our schedule
        active_item = schedule.iloc[items_completed]
        time_required = active_item['start_time']
        action_required = active_item['action']
        sample_index = active_item['sample_index'] + set_suffix
        current_time = get_time(SIMULATE,current_time)
        measured_items = 0

        #If we reach the triggered item's required time:
        if current_time - start_time > time_required:
            print("\nEvent triggered: " + action_required + f" from sample {sample_index}")
            print(f"Current Elapsed Time: {(current_time - start_time)/60} minutes")
            print(f"Intended Elapsed Time: {(time_required)/60} minutes")

            if action_required=="dispense_from_reactor":
                dispense_from_photoreactor_into_sample(lash_e,cof_vial,sample_index,volume=0.2)
                items_completed+=1
            elif action_required=="measure_samples":
                transfer_samples_into_wellplate_and_characterize(lash_e,sample_index,current_well_index,MEASUREMENT_PROTOCOL_FILE,replicates, output_dir,simulate=SIMULATE)
                current_well_index += replicates
                items_completed+=1
                measured_items+=1
        elif current_time - start_time > time_increment:
            print(f"I'm Alive! Current Elapsed Time: {(current_time - start_time)/60} minutes")
            time_increment=time_increment+60
        
        if not SIMULATE:
            time.sleep(1)
    lash_e.nr_robot.move_home()   

    lash_e.nr_robot.return_vial_home(cof_vial)
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=0)
    lash_e.photoreactor.turn_off_reactor_led(reactor_num=0)
    lash_e.nr_robot.move_home()

    # if not SIMULATE:   
    #     # slack_agent.send_slack_message("Peroxide workflow completed!")

# Initialize the workstation ONCE before running all workflows
INPUT_VIAL_STATUS_FILE = "/Users/serenaqiu/Desktop/north_repository/utoronto_demo/status/peroxide_assay_vial_status_v3.csv"
SIMULATE = True
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
lash_e.nr_robot.check_input_file()

# Run the workflow 3 times with different Reagent+COF+sample sets, reusing the same lash_e instance
peroxide_workflow(lash_e, assay_reagent='Assay_reagent_1', cof_vial='COF_1', set_suffix='_Set1')
peroxide_workflow(lash_e, assay_reagent='Assay_reagent_2', cof_vial='COF_2', set_suffix='_Set2')
peroxide_workflow(lash_e, assay_reagent='Assay_reagent_3', cof_vial='COF_3', set_suffix='_Set3')
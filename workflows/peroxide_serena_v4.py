import sys
import time
import datetime
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E 
import pandas as pd
from pathlib import Path
import slack_agent
import pipetting_data.embedded_calibration_validation as pipette_validator  

# Global well tracking variables
WELLS_PER_PLATE = 96
current_well_count = 0
wellplate_count = 0

def check_and_get_wellplate_if_needed(lash_e, wells_needed):
    """Check if we need a new wellplate and get one if necessary"""
    global current_well_count, wellplate_count, WELLS_PER_PLATE
    
    if current_well_count + wells_needed > WELLS_PER_PLATE:
        print(f"Wells needed: {wells_needed}, Current well count: {current_well_count}")
        print("Need new wellplate - current plate is full")
        if current_well_count > 0:  # Only discard if we have a plate in use
            lash_e.discard_used_wellplate()
        lash_e.grab_new_wellplate()
        wellplate_count += 1
        current_well_count = 0
        print(f"Got new wellplate #{wellplate_count}")

def update_well_count(wells_used):
    """Update the global well count after using wells"""
    global current_well_count
    current_well_count += wells_used
    print(f"Used {wells_used} wells. Total wells used on current plate: {current_well_count}")

def dispense_from_photoreactor_into_sample(lash_e,reaction_mixture_index,sample_index,volume=0.05):
    print("\nDispensing from photoreactor into sample: ", sample_index)
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=0)
    lash_e.nr_robot.dispense_from_vial_into_vial(reaction_mixture_index,sample_index,volume=volume, liquid='water')
    lash_e.photoreactor.turn_on_reactor_fan(reactor_num=0,rpm=600)
    mix_current_sample(lash_e,sample_index)
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.move_home()
    # lash_e.nr_robot.c9.home_robot() #removed for now to save time
    #for i in range (6,8):
       # lash_e.nr_robot.home_axis(i) #Home the track
    print()

def transfer_samples_into_wellplate_and_characterize(lash_e,sample_index,first_well_index,cytation_protocol_file_path,replicates,output_dir,simulate=True,well_volume=0.2):
    print("\nTransferring sample: ", sample_index, " to wellplate at well index: ", first_well_index)
    lash_e.nr_robot.move_vial_to_location(sample_index, location="main_8mL_rack", location_index=44) #Move sample to safe pipetting position
    
    wells = range(first_well_index,first_well_index+replicates)
    for well in wells:
        lash_e.nr_robot.aspirate_from_vial(sample_index, well_volume, track_height=True, liquid='water')
        lash_e.nr_robot.dispense_into_wellplate([well], [well_volume], liquid='water')
    
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.return_vial_home(sample_index) #Return sample to home position
    data_out = lash_e.measure_wellplate(cytation_protocol_file_path, wells_to_measure=wells)
    # output_file = r'C:\Users\Imaging Controller\Desktop\SQ\output_'+str(first_well_index)+'.txt'
    if not simulate:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f'output_{first_well_index}_{timestamp}.txt'
        data_out.to_csv(output_file, sep=',')
        #Use analyzer to analyze the data
    
    # Update well count after using wells
    update_well_count(replicates)
    print()

def mix_current_sample(lash_e, sample_index):
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
def peroxide_workflow(lash_e, assay_reagent='Assay_reagent_1', cof_vial='COF_1', set_suffix='',replicates=3, simulate=True, output_dir=None, global_well_index=0):
  
    MEASUREMENT_PROTOCOL_FILE =r"C:\Protocols\SQ_Peroxide.prt"

    SIMULATE =  simulate #Set to True if you want to simulate the robot, False if you want to run it on the real robot

    # Create subdirectory based on COF_vial name
    if output_dir is not None:
        cof_output_dir = output_dir / cof_vial
        cof_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created subdirectory for {cof_vial}: {cof_output_dir}")
    else:
        cof_output_dir = None

    #OAM Note: These times need to exactly correspond to the schedule and vials!
    sample_times = [1,6,11,20,30,45] #in minutes <----- These numbers need to match the vials and the schedule!!!!!
    sample_indices = [f"{t}_min_Reaction{set_suffix}" for t in sample_times]

#-> Start from here! 
    # Check if we need a new wellplate at the start of each workflow
    total_wells_needed = len(sample_indices) * replicates
    check_and_get_wellplate_if_needed(lash_e, total_wells_needed)
    
    #Step 1: Add 1.95 mL "assay reagent" to sample vials
    for i in sample_indices:  #May want to use liquid calibration eg water
        lash_e.nr_robot.dispense_from_vial_into_vial(assay_reagent,i,use_safe_location=False, volume=1.95, liquid='water')
    
    # #Step 2: Move the reaction mixture vial to the photoreactor to start the reaction.
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
    current_well_index = global_well_index  # Use the global well index passed in
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
                dispense_from_photoreactor_into_sample(lash_e,cof_vial,sample_index,volume=0.05)
                items_completed+=1
            elif action_required=="measure_samples":
                # Check if we need a new wellplate before measuring
                check_and_get_wellplate_if_needed(lash_e, replicates)
                # Recalculate well index based on current plate position
                actual_well_index = current_well_count
                transfer_samples_into_wellplate_and_characterize(lash_e,sample_index,actual_well_index,MEASUREMENT_PROTOCOL_FILE,replicates, cof_output_dir,simulate=SIMULATE)
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
    # Don't discard wellplate here - let the well counting system handle it
    if not SIMULATE: 
        slack_agent.send_slack_message("Peroxide workflow completed!")
    
    return current_well_index  # Return the updated well index

# Initialize the workstation ONCE before running all workflows

NUMBER_OF_SAMPLES = 3
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/peroxide_assay_vial_status_v3.csv"
SIMULATE = False

# Get experiment name at the start
if not SIMULATE:
    exp_name = input("Experiment name: ")
else:
    exp_name = "simulation_test"

lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
lash_e.nr_robot.check_input_file()
lash_e.nr_track.check_input_file()

lash_e.nr_robot.home_robot_components()

#create file name for the output data
if not SIMULATE:
    output_dir = Path(r'C:\Users\Imaging Controller\Desktop\SQ') / exp_name #appends exp_name to the output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output directory created at:", output_dir)
    slack_agent.send_slack_message(f"Peroxide workflow '{exp_name}' started!")
else:
    output_dir = None

# Validate Pipetting Accuracy
for cof in [f'COF_{i}' for i in range(1,NUMBER_OF_SAMPLES+1)]:
    vial_name = cof
    if output_dir is not None:
        cof_subdir = output_dir / vial_name
        cof_subdir.mkdir(parents=True, exist_ok=True)
        validation_folder = cof_subdir / f'Pipetting_Validation_{vial_name}'
    else:
        validation_folder = None
    results = pipette_validator.validate_pipetting_accuracy(
                    lash_e=lash_e,
                    source_vial=vial_name,
                    destination_vial=vial_name,
                    liquid_type="water",
                    volumes_ml=[0.05],  # Convert 10 µL to 0.01 mL
                    replicates=5,
                    output_folder=validation_folder,
                    plot_title=f"Pipetting Validation - {vial_name}",
                    condition_tip_enabled=True,
                    conditioning_volume_ul=100
                )

# Water validation - put in a general validation folder
vial_name = 'water'
if output_dir is not None:
    validation_folder = output_dir / f'Pipetting_Validation_{vial_name}'
else:
    validation_folder = None
results = pipette_validator.validate_pipetting_accuracy(
                lash_e=lash_e,
                source_vial=vial_name,
                destination_vial=vial_name,
                liquid_type="water",
                volumes_ml=[0.2, 0.95],  # Convert 10 µL to 0.01 mL
                replicates=5,
                output_folder=validation_folder,
                plot_title=f"Pipetting Validation - {vial_name}",
                condition_tip_enabled=True,
                conditioning_volume_ul=800
            )

# Initialize well tracking
global_well_index = 0
print(f"Starting experiment with {NUMBER_OF_SAMPLES} COF samples")
print(f"Each COF will use approximately {6 * 3} wells (6 samples x 3 replicates)")
print(f"Total wells needed: {NUMBER_OF_SAMPLES * 6 * 3} wells")

# Run the workflow N times with different Reagent+COF+sample sets, reusing the same lash_e instance
for i in range(1, NUMBER_OF_SAMPLES+1):
    assay_reagent = f'Assay_reagent_{i}'
    cof_vial = f'COF_{i}'
    set_suffix = f'_Set{i}'
    print(f"\n=== Starting COF {i} workflow ===")
    global_well_index = peroxide_workflow(lash_e, assay_reagent=assay_reagent, cof_vial=cof_vial, set_suffix=set_suffix, simulate=SIMULATE, output_dir=output_dir, global_well_index=global_well_index)
    print(f"=== COF {i} workflow completed. Global well index now at: {global_well_index} ===")

# Discard final wellplate if one is in use
if current_well_count > 0:
    print("Discarding final wellplate")
    lash_e.discard_used_wellplate()

print(f"\nExperiment completed!")
print(f"Total wellplates used: {wellplate_count}")
print(f"Final well count: {current_well_count}")
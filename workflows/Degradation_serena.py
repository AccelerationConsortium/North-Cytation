import sys
import time
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E 
import pandas as pd
import os
from pathlib import Path

# Take aliquot -> Wellplate measurement (1 replicate)-> Put sample back to wellplate
def measure_absorbance(lash_e,sample_index,first_well_index,cytation_protocol_file_path,well_volume=0.2):
    print("\nTransferring sample: ", sample_index, " to wellplate at well index: ", first_well_index)
    lash_e.temp_controller.turn_off_stirring()
    lash_e.nr_robot.aspirate_from_vial(sample_index, well_volume,track_height=True) # should automatically move vial to clamp for aspiration?
    wells = range(first_well_index)
    lash_e.nr_robot.dispense_into_wellplate(wells, well_volume)
    lash_e.nr_robot.remove_pipet()
    data_out = lash_e.measure_wellplate(cytation_protocol_file_path, wells_to_measure=wells)
    return data_out

def save_data(data_out,output_dir,first_well_index,simulate):
    output_file = output_dir / f'output_{first_well_index}.txt'
    # output_file = r'C:\Users\Imaging Controller\Desktop\SQ\output_'+str(first_well_index)+'.txt'
    if not simulate:
        output_file = output_dir / f'output_{first_well_index}.txt'
        data_out.to_csv(output_file, sep=',')
    print()

# Clean well plate = Solvent wash *1 + Acetone wash *2
def wash_wellplate(lash_e,first_well_index,solvent_repeats=1, acetone_repeats=2,volume=0.3):
    print("\nWashing well plate: ", first_well_index)
    for _ in range (solvent_repeats):
        lash_e.nr_robot.aspirate_from_vial(solvent_index,volume,track_height=True)
        lash_e.nr_robot.mix_well_in_wellplate(lash_e,first_well_index,volume=volume,repeats=2,well_plate_type="96 WELL PLATE")
    lash_e.nr_robot.remove_pipet()
    for _ in range (acetone_repeats):
        lash_e.nr_robot.pipet_from_wellplate(lash_e,first_well_index,volume=volume,aspirate=True,move_to_aspirate=False,well_plate_type="96 WELL PLATE")
        lash_e.nr_robot.aspirate_from_vial(acetone_index,volume,track_height=True)
        lash_e.nr_robot.mix_well_in_wellplate(lash_e,first_well_index,volume=volume,repeats=2,well_plate_type="96 WELL PLATE")
    lash_e.nr_robot.remove_pipet()
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
def degradation_workflow():
  
    # a. Initial State of your Vials, so the robot can know where to pipet:
    INPUT_VIAL_STATUS_FILE = ("../utoronto_demo/status/degradation_vial_status.csv")

    # b. Cytation 5 UV-Vis Measurement protocol:
    CYTATION_PROTOCOL_FILE = (r"C:\Protocols\degradation_protocol.prt") 

    # c. Time schedule for UV-VIS measurements: 
    SCHEDULE_FILE = r"C:\Users\Imaging Controller\Desktop\SQ\degradation\schedule.csv"

    # d. Polymer dilution calculation:
    sample_volume = 3.0 # Total volume of each polymer sample (mL)
    df = pd.read_csv(INPUT_VIAL_STATUS_FILE)
    stock_conc = df.loc[df['sample'] == 'stock', 'concentration (mg/mL)'].iloc[0] # Selects row [sample==stock]
    samples = df[df['sample'].str.contains("sample", case=False, na=False)].copy()
    samples['stock_volume'] = (samples['concentration (mg/mL)'] / stock_conc) * sample_volume
    samples['solvent_volume'] = sample_volume - samples['stock_volume']
    print(df) # Prints the vial status & location
    print(samples) # Prints the samples with calculated volumes of stock and solvent to be added

    # e. Initialize the workstation, which includes the robot, track, cytation and photoreactors
    SIMULATE = True #Set to True if you want to simulate the robot, False if you want to run it on the real robot
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)

    #Location indices: vial_numbers = vial_status['vial_index'].values #Gives you the values
    polymer_stock = lash_e.nr_robot.get_vial_index_from_name('polymer_stock') 
    solvent_index = lash_e.nr_robot.get_vial_index_from_name('solvent')
    acid_index = lash_e.nr_robot.get_vial_index_from_name('acid')
    acetone_index = lash_e.nr_robot.get_vial_index_from_name('acetone')

    #Sample incides: 
    sample_indices = vial_status.index.values[3:]

    #Variables: 
    stock_volume = df.loc[df['sample'] == 'stock_volume'].iloc[0]
    solvent_volume = df.loc[df['sample'] == 'solvent_volume'].iloc[0]
    acid_volume = 0.05 


    # input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")    
    lash_e.nr_robot.check_input_file()

    #create file name for the output data
    if not SIMULATE:
        import slack_agent
        exp_name = input("Experiment name: ")
        output_dir = Path(r'C:\Users\Imaging Controller\Desktop\SQ') / exp_name #appends exp_name to the output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Output directory created at:", output_dir)
        slack_agent.send_slack_message("Degradation workflow started!")
    else:
        output_dir = None

    # > Workflow starts from here! 
    # 1. Polymer sample preparation: Add stock polymer solution then solvent to each sample vial.
    for i in sample_indices:
        lash_e.nr_robot.dispense_from_vial_into_vial(polymer_stock,i,stock_volume)
        lash_e.nr_robot.remove_pipet()
    for i in sample_indices:
        print("\npreparing polymer sample: ", i)
        lash_e.nr_robot.dispense_from_vial_into_vial(solvent,i,solvent_volume)
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.mix_vial(vial_name=i, vortex_time=5)

    # 2. Add acid to the polymer samples to initiate degradation and take scheduled UV-VIS measurements.
    t0_map = {} #Dictionary to store the start time for each sample
    first_well_index = 0 #This is the first well index to use for the first sample

    for i in sample_indices: 
        lash_e.nr_robot.dispense_from_vial_into_vial(acid_index,i,acid_volume)
        lash_e.nr_robot.remove_pipet()
        print("\nAdding acid to sample: ", i)
        t0_map[i] = time.time() #Record the start time for each sample
        measure_absorbance(lash_e,sample_index,first_well_index,CYTATION_PROTOCOL_FILE, output_dir,simulate=SIMULATE)
        try:
            lash_e.temp_controller.turn_off_stirring()
            time.sleep(1)
        except: 
            print("Stirring was already off.")
        finally:
            lash_e.nr_robot.move_vial_to_location(vial_name=i,location='heater',location_index=i)
            lash_e.temp_controller.turn_on_stirring()

    schedule = pd.read_csv(SCHEDULE_FILE, sep=",") #Read the schedule file
    schedule = schedule.sort_values(by='start_time') #sort in ascending time order
    print("Schedule: ", schedule)

    start_time = current_time = get_time(SIMULATE)
    print("Starting timed portion at: ", start_time)

    # complete the items one at a time
    items_completed = 0
    time_increment = 60
    while items_completed < schedule.shape[0]: #While we still have items to complete in our schedule
        active_item = schedule.iloc[items_completed]
        time_required = active_item['start_time']
        action_required = active_item['action']
        sample_index = active_item['sample_index']
        current_time = get_time(SIMULATE,current_time)
        measured_items = 0

        t0 = t0_map.get(sample_index, start_time) 
        elapsed_time = current_time - t0

        #If we reach the triggered item's required time:
        if elapsed_time > time_required:
            print("\nEvent triggered: " + action_required + f" from sample {sample_index}")
            print(f"Current Elapsed Time: {(current_time - start_time)/60} minutes")
            print(f"Intended Elapsed Time: {(time_required)/60} minutes")

            if action_required=="measure_samples":
                measure_absorbance(lash_e,sample_index,starting_well_index,CYTATION_PROTOCOL_FILE, output_dir,simulate=SIMULATE)
                starting_well_index
                items_completed+=1
                measured_items+=1
        elif current_time - start_time > time_increment:
            print(f"I'm Alive! Current Elapsed Time: {(current_time - start_time)/60} minutes")
            time_increment+=60
        
        if not SIMULATE:
            time.sleep(1)
    lash_e.nr_robot.move_home()   

    lash_e.nr_robot.return_vial_home(sample_index)
    lash_e.temp_controller.turn_off_stirring()
    lash_e.nr_robot.move_home()

    if not SIMULATE:   
        slack_agent.send_slack_message("Degradation workflow completed!")

degradation_workflow() #Run your workflow


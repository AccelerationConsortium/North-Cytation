from doctest import script_from_examples
import sys
import time
from venv import create
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E 
import pandas as pd
from pathlib import Path

# Take aliquot -> Wellplate measurement (1 replicate)-> Put sample back to wellplate

def create_samples_and_measure(lash_e,output_dir,first_well_index,cytation_protocol_file_path,simulate, solvent_vial, wash_vial, sample_name, replicates=1):
    create_samples_in_wellplate(lash_e, sample_name=sample_name, first_well_index=first_well_index, well_volume=0.2, replicates=replicates)
    wells = list(range(first_well_index, first_well_index + replicates))
    data_out = lash_e.measure_wellplate(cytation_protocol_file_path, wells_to_measure=wells)
    save_data(data_out,output_dir,first_well_index,simulate)
    wash_wellplate(lash_e,first_well_index,solvent_vial=solvent_vial, wash_vial=wash_vial, solvent_repeats=1, acetone_repeats=2, volume=0.3, replicates=replicates)

def create_samples_in_wellplate(lash_e,sample_name,first_well_index,well_volume=0.2,replicates=1):
    print(f"\nTransferring sample: {sample_name} to wellplate at wells {first_well_index} to {first_well_index + replicates - 1} ({replicates} replicates)")
    lash_e.temp_controller.turn_off_stirring()
    
    # Aspirate enough volume for all replicates
    total_volume = well_volume * replicates
    lash_e.nr_robot.aspirate_from_vial(sample_name, total_volume, track_height=True)
    
    # Dispense into multiple wells
    wells = list(range(first_well_index, first_well_index + replicates))
    volumes = [well_volume] * replicates  # Create array of volumes, one for each well
    lash_e.nr_robot.dispense_into_wellplate(wells, volumes)
    lash_e.nr_robot.remove_pipet()

def save_data(data_out,output_dir,first_well_index,simulate):
    if not simulate:
        output_file = output_dir / f'output_{first_well_index}.txt'
        data_out.to_csv(output_file, sep=',')
        print("Data saved to:", output_file)
    else:
        print(f"Simulation mode: Would save data for well index {first_well_index}")

# Clean well plate = Solvent wash *1 + Acetone wash *2
def wash_wellplate(lash_e,first_well_index, solvent_vial, wash_vial, solvent_repeats=1, acetone_repeats=2,volume=0.3,replicates=1):
    wells_to_wash = list(range(first_well_index, first_well_index + replicates))
    print(f"\nWashing wellplate wells: {wells_to_wash}")
    
    # Solvent wash for all wells
    for _ in range(solvent_repeats):
        for well in wells_to_wash:
            lash_e.nr_robot.aspirate_from_vial(solvent_vial,volume,track_height=True)
            lash_e.nr_robot.mix_well_in_wellplate(well,volume,repeats=2,well_plate_type="96 WELL PLATE")
        lash_e.nr_robot.remove_pipet()
    
    # Acetone wash for all wells  
    for _ in range(acetone_repeats):
        for well in wells_to_wash:
            lash_e.nr_robot.pipet_from_wellplate(well,volume,aspirate=True,move_to_aspirate=False,well_plate_type="96 WELL PLATE")
            lash_e.nr_robot.aspirate_from_vial(wash_vial,volume,track_height=True)
            lash_e.nr_robot.mix_well_in_wellplate(well,volume,repeats=2,well_plate_type="96 WELL PLATE")
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

    #Simulate mode True or False
    SIMULATE = True #Set to True if you want to simulate the robot, False if you want to run it on the real robot
    
    # Number of replicate measurements per timepoint
    REPLICATES = 3  # Number of wells to use for each measurement (default: 3)

    # d. Polymer dilution calculation:
    sample_volume = 3.0 # Total volume of each polymer sample (mL)
    df = pd.read_csv(INPUT_VIAL_STATUS_FILE)
    sample_col = 'vial_name'  # Use vial_name column from CSV
    
    # Get stock concentration
    stock_conc = df.loc[df[sample_col] == 'polymer_stock', 'concentration'].iloc[0]
    print(f"Stock concentration: {stock_conc} mg/mL")
    
    # Calculate dilution volumes for each sample
    samples = df[df[sample_col].str.contains("sample", case=False, na=False)].copy()
    samples['stock_volume'] = (samples['concentration'] / stock_conc) * sample_volume
    samples['solvent_volume'] = sample_volume - samples['stock_volume']
    print("Calculated volumes for each sample:")
    print(samples[[sample_col, 'concentration', 'stock_volume', 'solvent_volume']])

    # e. Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)

    #Samples
    sample_solutions = {'sample_' + str(i) for i in range(1,4)} #Set of sample names (sample_1, sample_2, sample_3...)

    # Create a dictionary to look up volumes by sample name
    volume_lookup = {}
    for _, row in samples.iterrows():
        sample_name = row[sample_col]
        volume_lookup[sample_name] = {
            'stock_volume': row['stock_volume'],
            'solvent_volume': row['solvent_volume']
        }
    
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
    for sample in sample_solutions:
        stock_vol = volume_lookup[sample]['stock_volume']
        print(f"\nAdding {stock_vol:.3f} mL stock solution to {sample}")
        lash_e.nr_robot.dispense_from_vial_into_vial('polymer_stock',sample,stock_vol)
        lash_e.nr_robot.remove_pipet()
    
    for sample in sample_solutions:
        solvent_vol = volume_lookup[sample]['solvent_volume']
        print(f"\nAdding {solvent_vol:.3f} mL solvent to {sample}")
        lash_e.nr_robot.dispense_from_vial_into_vial('2MeTHF',sample,solvent_vol)
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.vortex_vial(vial_name=sample, vortex_time=5)

    # 2. Add acid to the polymer samples to initiate degradation and take scheduled UV-VIS measurements.
    t0_map = {} #Dictionary to store the start time for each sample
    first_well_index = 0 #This is the first well index to use for the first sample

    i = 0
    for sample in sample_solutions:
        lash_e.nr_robot.dispense_from_vial_into_vial('6M HCl',sample,acid_volume)
        lash_e.nr_robot.remove_pipet()
        print("\nAdding acid to sample: ", sample)
        t0_map[sample] = time.time() #Record the start time for each sample
        create_samples_and_measure(lash_e,output_dir,first_well_index,CYTATION_PROTOCOL_FILE,SIMULATE, solvent_vial='2MeTHF', wash_vial='acetone', sample_name=sample, replicates=REPLICATES)
        try:
            lash_e.temp_controller.turn_off_stirring()
            if not SIMULATE:
                time.sleep(1)
        except: 
            print("Stirring was already off.")
        finally:
            lash_e.nr_robot.move_vial_to_location(vial_name=sample,location='heater',location_index=i)
            lash_e.temp_controller.turn_on_stirring()
        first_well_index += REPLICATES  # Move to next set of wells for next sample
        i+=1

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
                create_samples_and_measure(lash_e,output_dir,first_well_index,CYTATION_PROTOCOL_FILE,SIMULATE, solvent_vial='2MeTHF', wash_vial='acetone', sample_name=sample_index, replicates=REPLICATES)
                first_well_index += REPLICATES  # Move to next set of wells
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


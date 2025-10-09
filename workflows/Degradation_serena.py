import sys
import time
sys.path.append("../North-Cytation")
from master_usdl_coordinator import Lash_E 
import pandas as pd
from pathlib import Path

# Take aliquot -> Wellplate measurement (1 replicate)-> Put sample back to wellplate

def create_samples_and_measure(lash_e,output_dir,first_well_index,cytation_protocol_file_path,simulate, solvent_vial, wash_vial, sample_name, replicates=1):
    create_samples_in_wellplate(lash_e, sample_name=sample_name, first_well_index=first_well_index, well_volume=0.2, replicates=replicates)
    wells = list(range(first_well_index, first_well_index + replicates))
    data_out = lash_e.measure_wellplate(cytation_protocol_file_path, wells_to_measure=wells)
    save_data(data_out,output_dir,first_well_index,simulate)
    wash_wellplate(lash_e,first_well_index,solvent_vial=solvent_vial, wash_vial=wash_vial, solvent_repeats=1, acetone_repeats=2, volume=0.1, replicates=replicates)

def safe_pipet(source_vial, dest_vial, volume, lash_e):
    source_home_location_index = lash_e.nr_robot.get_vial_info(source_vial, 'location_index')
    dest_home_location_index = lash_e.nr_robot.get_vial_info(dest_vial, 'location_index')
    move_source = (source_home_location_index > 5)
    move_dest = (dest_home_location_index > 5)

    if move_source: 
        lash_e.nr_robot.move_vial_to_location(vial_name=source_vial, location='main_8mL_rack', location_index=5)
        vial_index = lash_e.nr_robot.get_vial_info(source_vial, 'vial_index')
        lash_e.nr_robot.VIAL_DF.at[vial_index, 'home_location_index'] = 5
    elif move_dest:
        lash_e.nr_robot.move_vial_to_location(vial_name=dest_vial, location='main_8mL_rack', location_index=5)
        vial_index = lash_e.nr_robot.get_vial_info(dest_vial, 'vial_index')
        lash_e.nr_robot.VIAL_DF.at[vial_index, 'home_location_index'] = 5

    lash_e.nr_robot.dispense_from_vial_into_vial(source_vial, dest_vial, volume)

    if move_source: 
        vial_index = lash_e.nr_robot.get_vial_info(source_vial, 'vial_index')
        lash_e.nr_robot.VIAL_DF.at[vial_index, 'home_location_index'] = int(source_home_location_index)
        lash_e.nr_robot.return_vial_home(vial_name=source_vial)
    elif move_dest:
        vial_index = lash_e.nr_robot.get_vial_info(dest_vial, 'vial_index')
        lash_e.nr_robot.VIAL_DF.at[vial_index, 'home_location_index'] = int(dest_home_location_index)
        lash_e.nr_robot.return_vial_home(vial_name=dest_vial)

def create_samples_in_wellplate(lash_e,sample_name,first_well_index,well_volume=0.2,replicates=1):
    print(f"\nTransferring sample: {sample_name} to wellplate at wells {first_well_index} to {first_well_index + replicates - 1} ({replicates} replicates)")
    lash_e.temp_controller.turn_off_stirring()
    
    # Aspirate enough volume for all replicates
    total_volume = well_volume * replicates
    lash_e.nr_robot.aspirate_from_vial(sample_name, total_volume, track_height=True)
    lash_e.nr_robot.return_vial_home(sample_name)
    
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
def wash_wellplate(lash_e,first_well_index, solvent_vial, wash_vial, solvent_repeats=1, acetone_repeats=2,volume=0.1,replicates=1):
    wells_to_wash = list(range(first_well_index, first_well_index + replicates))
    print(f"\nWashing wellplate wells: {wells_to_wash}")

    # Solvent wash for all wells
    lash_e.nr_robot.move_vial_to_location('waste', location='main_8mL_rack', location_index=4) #Added by OAM
    for _ in range(solvent_repeats):
        for well in wells_to_wash:
            lash_e.nr_robot.aspirate_from_vial(solvent_vial,volume,track_height=True)
            lash_e.nr_robot.move_vial_to_location(solvent_vial, location='main_8mL_rack', location_index=5) #Added by OAM
            lash_e.nr_robot.dispense_into_wellplate([well],[volume], well_plate_type="96 WELL PLATE") #Added by OAM
            lash_e.nr_robot.mix_well_in_wellplate(well,volume,repeats=2,well_plate_type="96 WELL PLATE")
            # pipet wash solution to Waste
            lash_e.nr_robot.pipet_from_wellplate(well, volume, aspirate=True, move_to_aspirate=False, well_plate_type="96 WELL PLATE")
            lash_e.nr_robot.dispense_into_vial("waste", volume)
            lash_e.nr_robot.move_vial_to_location('waste', location='main_8mL_rack', location_index=4)
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.return_vial_home(solvent_vial)
    # Acetone wash for all wells  
    for _ in range(acetone_repeats):
        for well in wells_to_wash:
            lash_e.nr_robot.aspirate_from_vial(wash_vial,volume,track_height=True)
            lash_e.nr_robot.move_vial_to_location(wash_vial, location='main_8mL_rack', location_index=5)#Added by OAM
            lash_e.nr_robot.dispense_into_wellplat([well],[volume], well_plate_type="96 WELL PLATE") #Added by OAM
            lash_e.nr_robot.mix_well_in_wellplate(well,volume,repeats=2,well_plate_type="96 WELL PLATE")
            # pipet wash solution to Waste
            # New step: pipet wash solution to Waste
            lash_e.nr_robot.pipet_from_wellplate(well, volume, aspirate=True, move_to_aspirate=False, well_plate_type="96 WELL PLATE")
            lash_e.nr_robot.dispense_into_vial("waste", volume)
            lash_e.nr_robot.move_vial_to_location('waste', location='main_8mL_rack', location_index=4) #Added by OAM
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.return_vial_home(wash_vial)
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
    INPUT_VIAL_STATUS_FILE = ("../North-Cytation/status/degradation_vial_status.csv")

    # b. Cytation 5 UV-Vis Measurement protocol:
    CYTATION_PROTOCOL_FILE = (r"C:\Protocols\degradation_protocol.prt") 

    # c. Time schedule for UV-VIS measurements: 
    SCHEDULE_FILE = ("../North-Cytation/status/degradation_vial_schedule.csv")

    # d. Simulate mode True or False
    SIMULATE = True #Set to True if you want to simulate the robot, False if you want to run it on the real robot
    
    # e. Number of replicate measurements per timepoint
    REPLICATES = 3  # Number of wells to use for each measurement (default: 3)

    # f. Polymer dilution calculation:
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

    # Samples: derive dynamically from status file (any vial_name containing 'sample')
    sample_solutions = set(samples[sample_col].unique())
    if not sample_solutions:
        raise ValueError("No sample vials found (expected vial_name containing 'sample').")

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
        safe_pipet('polymer_stock',sample,stock_vol, lash_e)
        lash_e.nr_robot.remove_pipet()
    
    for sample in sample_solutions:
        solvent_vol = volume_lookup[sample]['solvent_volume']
        print(f"\nAdding {solvent_vol:.3f} mL solvent to {sample}")
        safe_pipet('2MeTHF',sample,solvent_vol, lash_e)
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.vortex_vial(vial_name=sample, vortex_time=5)

    # 2. Add acid to the polymer samples to initiate degradation and take scheduled UV-VIS measurements.
    t0_map = {} #Dictionary: sample -> start time (same time basis as start_time/current_time)
    first_well_index = 0 #This is the first well index to use for the first sample

    i = 0
    for sample in sample_solutions:
        safe_pipet('6M_HCl',sample,acid_volume, lash_e)
        lash_e.nr_robot.remove_pipet()
        print("\nAdding acid to sample: ", sample)
        # Record per-sample start time using consistent basis (simulated clock = 0; real clock = wall time)
        t0_map[sample] = 0 if SIMULATE else time.time()
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

    # Establish timing model.
    # Simulation: use an integer counter (seconds). Real run: wall clock seconds.
    start_time = 0 if SIMULATE else time.time()
    current_time = start_time
    print("Starting timed portion at (secs): ", start_time)

    # complete the items one at a time
    items_completed = 0
    time_increment = 60  # Heartbeat every 60s simulated / real elapsed
    max_schedule_time = schedule['start_time'].max()
    safety_cutoff = max_schedule_time + 120  # 2 minute buffer beyond last scheduled event

    SIM_TICK_SECONDS = 1  # Advance by 1 simulated second per loop when SIMULATE=True

    while items_completed < schedule.shape[0]: #While we still have items to complete
        active_item = schedule.iloc[items_completed]
        time_required = active_item['start_time']
        action_required = active_item['action']
        sample_index = active_item['sample_index']
        # Advance time
        if SIMULATE:
            current_time += SIM_TICK_SECONDS
        else:
            current_time = time.time()

        elapsed_time = current_time - t0_map.get(sample_index, start_time)
        total_elapsed = current_time - start_time

        # Trigger condition (>= so exact match fires)
        if elapsed_time >= time_required:
            print(f"\n[TRIGGER] {action_required} for {sample_index} at elapsed {elapsed_time:.0f}s (target {time_required}s)")
            if action_required in ("create_samples_and_measure","measure_samples"):
                create_samples_and_measure(
                    lash_e, output_dir, first_well_index, CYTATION_PROTOCOL_FILE, SIMULATE,
                    solvent_vial='2MeTHF', wash_vial='acetone', sample_name=sample_index, replicates=REPLICATES
                )
                first_well_index += REPLICATES
            else:
                print(f"[WARN] Unknown action '{action_required}' – skipping (row {items_completed})")
            items_completed += 1
        else:
            # Heartbeat
            if total_elapsed >= time_increment:
                print(f"Heartbeat: total_elapsed={total_elapsed:.0f}s next_event_in={time_required - elapsed_time:.0f}s -> {sample_index}")
                time_increment += 60

        # Safety cutoff (simulation or real) to avoid runaway
        if total_elapsed > safety_cutoff:
            print(f"[SAFETY STOP] Elapsed {total_elapsed:.0f}s exceeded schedule max {max_schedule_time}s + buffer. Breaking loop.")
            break

        if not SIMULATE:
            time.sleep(0.25)
    lash_e.nr_robot.move_home()   

    lash_e.nr_robot.return_vial_home(sample_index)
    lash_e.temp_controller.turn_off_stirring()
    lash_e.nr_robot.move_home()

    if not SIMULATE:   
        slack_agent.send_slack_message("Degradation workflow completed!")

degradation_workflow() #Run your workflow


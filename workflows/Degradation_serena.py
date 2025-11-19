import sys
import time
from venv import create

from networkx import volume
sys.path.append("../North-Cytation")
from master_usdl_coordinator import Lash_E 
import pandas as pd
from pathlib import Path

# Take aliquot -> Wellplate measurement (1 replicate)-> Put sample back to wellplate
def create_samples_and_measure(lash_e, output_dir, first_well_index, cytation_protocol_file_path, simulate, sample_name, used_wells, replicates=1, timepoint=None, waste_state=None):
    current_waste = check_and_switch_waste_vial(lash_e, waste_state)
    lash_e.nr_robot.move_vial_to_location(current_waste, location='main_8mL_rack', location_index=4)
    create_samples_in_wellplate(lash_e, sample_name=sample_name, first_well_index=first_well_index, well_volume=0.2, replicates=replicates)
    wells = list(range(first_well_index, first_well_index + replicates))
    data_out = lash_e.measure_wellplate(cytation_protocol_file_path, wells_to_measure=wells)
    save_data(data_out, output_dir, first_well_index, simulate, sample_name=sample_name, timepoint=timepoint)
    used_wells.extend(wells)
    print("Used wells so far:", used_wells)
    
    # Empty wells to waste after measurement
    if waste_state:
        current_waste = check_and_switch_waste_vial(lash_e, waste_state)
        for well in wells:
            lash_e.nr_robot.pipet_from_wellplate(well, volume=0.2, aspirate=True, well_plate_type="96 WELL PLATE")
            lash_e.nr_robot.dispense_into_vial(current_waste, 0.2)
        lash_e.nr_robot.remove_pipet()

    # Return sample to heater position for continued degradation
    # Extract heater index from sample name (sample_1 -> 1, sample_2 -> 2, etc.)
    if 'sample_' in sample_name:
        try:
            heater_index = int(sample_name.split('_')[1]) - 1  # Convert to 0-based index
            print(f"Returning {sample_name} to heater position {heater_index}")
            lash_e.nr_robot.move_vial_to_location(vial_name=sample_name, location='heater', location_index=heater_index)
            lash_e.temp_controller.turn_on_stirring()
        except (ValueError, IndexError):
            print(f"Warning: Could not parse heater index from sample name: {sample_name}")
    else:
        print(f"Warning: Sample name {sample_name} does not follow expected format 'sample_#'")

    lash_e.nr_robot.return_vial_home(current_waste)

def pipet_sample_from_well_to_vial(lash_e, first_well_index, sample_name, replicates):
    wells = list(range(first_well_index, first_well_index + replicates))
    print(f"Returning samples from wells {wells} back to sample vial: {sample_name}")
    lash_e.nr_robot.move_vial_to_location(sample_name, location='main_8mL_rack', location_index=4)
    lash_e.nr_robot.pipet_from_wellplate(wells, volume=0.2, aspirate=True, well_plate_type="96 WELL PLATE")
    lash_e.nr_robot.dispense_into_vial(sample_name, 0.2)
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.return_vial_home(sample_name)

def safe_pipet(source_vial, dest_vial, volume, lash_e):
    source_home_location_index = lash_e.nr_robot.get_vial_info(source_vial, 'location_index')
    dest_home_location_index = lash_e.nr_robot.get_vial_info(dest_vial, 'location_index')
    move_source = (source_home_location_index > 5)
    move_dest = (dest_home_location_index > 5)

    if move_source: 
        #lash_e.nr_robot.move_vial_to_location(vial_name=source_vial, location='main_8mL_rack', location_index=5)
        vial_index = lash_e.nr_robot.get_vial_info(source_vial, 'vial_index')
        lash_e.nr_robot.VIAL_DF.at[vial_index, 'home_location_index'] = 4
        lash_e.logger.info(f"Setting home location of {source_vial} to 4 for safe pipetting")
    elif move_dest:
        #lash_e.nr_robot.move_vial_to_location(vial_name=dest_vial, location='main_8mL_rack', location_index=5)
        vial_index = lash_e.nr_robot.get_vial_info(dest_vial, 'vial_index')
        lash_e.nr_robot.VIAL_DF.at[vial_index, 'home_location_index'] = 5
        lash_e.logger.info(f"Setting home location of {dest_vial} to 5 for safe pipetting")

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
    
    # Create DataFrame for dispense_from_vials_into_wellplate method
    # Each row represents a well, each column represents a vial
    well_indices = list(range(first_well_index, first_well_index + replicates))
    well_plate_df = pd.DataFrame(index=well_indices, columns=[sample_name])
    well_plate_df[sample_name] = well_volume  # Same volume for each replicate well
    
    # Use serial strategy for precise dispensing with full parameter support
    lash_e.nr_robot.dispense_from_vials_into_wellplate(well_plate_df=well_plate_df, strategy="serial" )

def save_data(data_out, output_dir, first_well_index, simulate, sample_name=None, timepoint=None):
    if not simulate:
        if sample_name and timepoint is not None:
            # Descriptive filename with sample and timepoint
            output_file = output_dir / f'{sample_name}_time_{timepoint}s.csv'
        else:
            # Fallback to old naming if parameters missing
            output_file = output_dir / f'output_{first_well_index}.txt'
        data_out.to_csv(output_file, sep=',')
        print("Data saved to:", output_file)
    else:
        if sample_name and timepoint is not None:
            print(f"Simulation mode: Would save data as {sample_name}_time_{timepoint}s.csv")
        else:
            print(f"Simulation mode: Would save data for well index {first_well_index}")

def create_and_measure_initial_samples(lash_e, sample_solutions, acid_volume, output_dir, cytation_protocol_file, simulate, used_wells, replicates=1, stagger_time=0, waste_state=None):
    """
    Fast initial sample preparation: minimize time between acid addition and measurement.
    
    Sequence for each sample:
    1. Move acid to clamp and uncap
    2. Aspirate from acid 
    3. Return acid home
    4. Dispense into sample (t0 timestamp here!)
    5. Aspirate from sample
    6. Dispense into wellplate
    7. Remove pipet, return sample home
    8. Measure wellplate
    """
    t0_map = {}
    first_well_index = 0
    sim_time_counter = 0  # Simulated time counter for t0 timestamps
    
    current_waste = check_and_switch_waste_vial(lash_e, waste_state)
    lash_e.nr_robot.move_vial_to_location(current_waste, location='main_8mL_rack', location_index=4)

    i = 0
    for sample in sample_solutions:
        print(f"\nFast acid addition and measurement for {sample}")
        
        # Step 1-3: Prepare acid pipet
        lash_e.nr_robot.move_vial_to_location('6M_HCl', location='clamp', location_index=0)
        lash_e.nr_robot.aspirate_from_vial('6M_HCl', acid_volume, track_height=True)
        lash_e.nr_robot.return_vial_home('6M_HCl')
        
        # Step 4: Dispense into sample (CRITICAL TIMESTAMP!)
        lash_e.nr_robot.move_vial_to_location(sample, location='clamp', location_index=0)
        lash_e.nr_robot.dispense_into_vial(sample, acid_volume)
        if simulate:
            t0_map[sample] = sim_time_counter  # Each sample gets different simulated t0
            sim_time_counter += stagger_time  # Use stagger_time separation between samples
        else:
            t0_map[sample] = time.time()  # Real timestamp
        
        # Step 5-6: Transfer to wellplate immediately
        lash_e.nr_robot.aspirate_from_vial(sample, 0.2, track_height=True)  # Sample for measurement
        wells = list(range(first_well_index, first_well_index + replicates))
        lash_e.nr_robot.dispense_into_wellplate(wells, [0.2] * replicates, well_plate_type="96 WELL PLATE")
        
        #Cap clamp vial?
        lash_e.nr_robot.recap_clamp_vial()
        
        # Step 8: Measure immediately
        data_out = lash_e.measure_wellplate(cytation_protocol_file, wells_to_measure=wells)
        save_data(data_out, output_dir, first_well_index, simulate, sample_name=sample, timepoint=0)
        used_wells.extend(wells)

        # Step 7: Cleanup
        # Empty wells to waste after measurement
        for well in wells:
            lash_e.nr_robot.pipet_from_wellplate(well, volume=0.2, aspirate=True, well_plate_type="96 WELL PLATE")
            lash_e.nr_robot.dispense_into_vial(current_waste, 0.2)

        lash_e.nr_robot.remove_pipet()
        
        # Move sample to heater for degradation
        try:
            lash_e.temp_controller.turn_off_stirring()
            if not simulate:
                time.sleep(1)
        except: 
            print("Stirring was already off.")
        finally:
            lash_e.nr_robot.move_vial_to_location(vial_name=sample, location='heater', location_index=i)
            lash_e.temp_controller.turn_on_stirring()
            
        first_well_index += replicates
        i += 1
        if not simulate:
            time.sleep(stagger_time)

    lash_e.nr_robot.return_vial_home(current_waste)
    return t0_map, first_well_index

# Clean well plate = Solvent wash *1 + Acetone wash *2 <- DO it serially (ie wash 5 wells at a time, and wash all wells w solvent first before moving on to acetone)
def wash_wellplate(lash_e, used_wells, solvent_vial, wash_vial, waste_state, solvent_repeats=1, acetone_repeats=2, volume=1):
    used_wells=used_wells
    print(f"\nWashing wellplate wells: {used_wells}")

    # Check and switch waste vial if needed before washing
    current_waste = check_and_switch_waste_vial(lash_e, waste_state)
    
    lash_e.nr_robot.move_vial_to_location(current_waste, location='main_8mL_rack', location_index=4) #Added by OAM
    for _ in range(solvent_repeats):
        for well in used_wells:
            lash_e.nr_robot.aspirate_from_vial(solvent_vial,volume,track_height=True)
            lash_e.nr_robot.dispense_into_wellplate([well],[volume], well_plate_type="96 WELL PLATE") #Added by OAM
        lash_e.nr_robot.move_vial_to_location(solvent_vial, location='main_8mL_rack', location_index=5)
        for well in used_wells:
            lash_e.nr_robot.mix_well_in_wellplate(well,volume,repeats=2,well_plate_type="96 WELL PLATE") #Overkill?
            # pipet wash solution to current waste vial
        for well in used_wells:
            lash_e.nr_robot.pipet_from_wellplate(well, volume, aspirate=True, well_plate_type="96 WELL PLATE")
            lash_e.nr_robot.dispense_into_vial(current_waste, volume)  
        lash_e.nr_robot.remove_pipet()  
        lash_e.nr_robot.move_vial_to_location(current_waste, location='main_8mL_rack', location_index=4) #Added by OAM
        lash_e.nr_robot.return_vial_home(solvent_vial)
    # Acetone wash for all wells  
    for _ in range(acetone_repeats):
        lash_e.nr_robot.move_vial_to_location(current_waste, location='main_8mL_rack', location_index=4)
        for well in used_wells:
            lash_e.nr_robot.aspirate_from_vial(wash_vial,volume,track_height=True)
            lash_e.nr_robot.move_vial_to_location(wash_vial, location='main_8mL_rack', location_index=5)#Added by OAM
            lash_e.nr_robot.dispense_into_wellplate([well],[volume], well_plate_type="96 WELL PLATE") #Added by OAM
        for well in used_wells:
            lash_e.nr_robot.mix_well_in_wellplate(well,volume,repeats=2,well_plate_type="96 WELL PLATE") #Overkill?
            # pipet wash solution to current waste vial
        for well in used_wells:
            lash_e.nr_robot.pipet_from_wellplate(well, volume, aspirate=True, well_plate_type="96 WELL PLATE")
            lash_e.nr_robot.dispense_into_vial(current_waste, volume)
        lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.return_vial_home(current_waste)
    lash_e.nr_robot.return_vial_home(wash_vial)
    
    print("\nWashing completed.")

def get_time(simulate,current_time=None):
    if not simulate:
        return time.time()
    else:
        if current_time is None:
            return 0
        else:
            return current_time + 1

def check_and_switch_waste_vial(lash_e, waste_state):
    """Check if current waste vial is full and switch to next one if needed"""
    current_waste = waste_state["current_waste_vial"]
    try:
        current_vol = lash_e.nr_robot.get_vial_info(current_waste, "vial_volume")
        if current_vol > 7.0:  # Waste vial is full
            # Switch to next waste vial
            waste_state["waste_index"] += 1
            new_waste_vial = f"waste_{waste_state['waste_index']}"
            waste_state["current_waste_vial"] = new_waste_vial
            lash_e.logger.info(f"[info] Waste vial {current_waste} is full ({current_vol:.1f}mL), switching to {new_waste_vial}")
            print(f"Switching from {current_waste} to {new_waste_vial} (was {current_vol:.1f}mL)")
            return new_waste_vial
        else:
            return current_waste
    except Exception as e:
        lash_e.logger.warning(f"Could not check waste vial volume: {e}, continuing with {current_waste}")
        return current_waste

#Define your workflow! Make sure that it has parameters that can be changed!
def degradation_workflow():
  
    # a. Initial State of your Vials, so the robot can know where to pipet:
    INPUT_VIAL_STATUS_FILE = "../North-Cytation/status/degradation_vial_status.csv"

    # b. Cytation 5 UV-Vis Measurement protocol:
    #CYTATION_PROTOCOL_FILE = r"C:\Protocols\degradation_protocol.prt"
    CYTATION_PROTOCOL_FILE = r"C:\Protocols\Quick_Measurement.prt" 

    # c. Time schedule for UV-VIS measurements: 
    SCHEDULE_FILE = "../North-Cytation/status/degradation_vial_schedule.csv"

    # d. Simulate mode True or False
    SIMULATE = True #Set to True if you want to simulate the robot, False if you want to run it on the real robot
    
    # e. Number of replicate measurements per timepoint
    REPLICATES = 1  # Number of wells to use for each measurement (default: 3)

    # f. Polymer dilution calculation:
    sample_volume = 2.0 # Total volume of each polymer sample (mL)
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
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_t8=True) 

    # Initialize waste vial management
    waste_state = {
        "waste_index": 0,  # Start with waste_0
        "current_waste_vial": "waste_0"
    }

    # Samples: derive dynamically from status file (any vial_name containing 'sample')
    sample_solutions = sorted(samples[sample_col].unique())
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

    lash_e.nr_robot.home_robot_components()

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
        #lash_e.nr_robot.remove_pipet()
    
    for sample in sample_solutions:
        solvent_vol = volume_lookup[sample]['solvent_volume']
        print(f"\nAdding {solvent_vol:.3f} mL solvent to {sample}")
        safe_pipet('2MeTHF',sample,solvent_vol, lash_e)
        #lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.vortex_vial(vial_name=sample, vortex_time=5)

    # 2. Fast acid addition and initial measurements
    used_wells = []
    t0_map, first_well_index = create_and_measure_initial_samples(
        lash_e, sample_solutions, acid_volume, output_dir, 
        CYTATION_PROTOCOL_FILE, SIMULATE, used_wells, replicates=REPLICATES, waste_state=waste_state
    )

    schedule = pd.read_csv(SCHEDULE_FILE, sep=",") #Read the schedule file
    schedule = schedule.sort_values(by='start_time') #sort in ascending time order
    print("Schedule: ", schedule)
    print("\nTiming Setup:")
    print(f"t0_map: {t0_map}")
    
    # Calculate when each measurement should actually trigger
    print("\nPlanned trigger times (sample-relative):")
    for _, row in schedule.iterrows():
        sample = row['sample_index']
        target_time = row['start_time']
        sample_t0 = t0_map.get(sample, 0)
        workflow_trigger_time = sample_t0 + target_time
        print(f"  {sample} at {target_time}s → workflow time {workflow_trigger_time}s")

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

        # Check for valid sample timing
        if sample_index not in t0_map:
            print(f"ERROR: Sample '{sample_index}' not found in t0_map! Available samples: {list(t0_map.keys())}")
            print("This means the sample was never processed in initial measurements.")
            print("Skipping this scheduled measurement to prevent timing errors.")
            items_completed += 1
            continue
            
        elapsed_time = current_time - t0_map[sample_index]
        total_elapsed = current_time - start_time

        # Trigger condition (>= so exact match fires)
        if elapsed_time >= time_required:
            print(f"\n[TRIGGER] {action_required} for {sample_index} at elapsed {elapsed_time:.0f}s (target {time_required}s)")
            if action_required in ("create_samples_and_measure","measure_samples"):
                create_samples_and_measure(
                    lash_e, output_dir, first_well_index, CYTATION_PROTOCOL_FILE, SIMULATE, 
                    sample_name=sample_index, used_wells=used_wells, replicates=REPLICATES, 
                    timepoint=time_required, waste_state=waste_state)
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

    # 3. Clean the well plate after all measurements are done
    wash_wellplate(lash_e, used_wells, solvent_vial='2MeTHF', wash_vial='acetone', waste_state=waste_state, solvent_repeats=1, acetone_repeats=2, volume=0.2)

    for sample in sample_solutions:
        lash_e.nr_robot.return_vial_home(sample)
    
    lash_e.nr_robot.move_home()

    if not SIMULATE:   
        slack_agent.send_slack_message("Degradation workflow completed!")

degradation_workflow() #Run your workflow


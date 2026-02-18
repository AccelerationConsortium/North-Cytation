from shutil import move
import sys
import time
from tkinter import YES
from venv import create
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E 
from pipetting_data.pipetting_parameters import PipettingParameters
import pandas as pd
from pathlib import Path

# Configuration
VALIDATE_LIQUIDS = False  # Set to True to run pipetting validation

# Create custom pipetting parameters
small_tip_2MeTHF_params = PipettingParameters(
    aspirate_speed=15,           # Slower aspiration
    dispense_speed=15,          # Medium dispense speed
    post_asp_air_vol=0.05,     # Small air gap
    pre_asp_air_vol=0.5,
    asp_disp_cycles=0 # tested w/o the cycles and it worked ok (no droplet, no dripping) - maybe not needed for 2MeTHF
)

# Create custom pipetting parameters
large_tip_2MeTHF_params = PipettingParameters(
    aspirate_speed=13,           # Slower aspiration
    dispense_speed=13,          # Medium dispense speed
    post_asp_air_vol=0.05,     # Small air gap
    pre_asp_air_vol=0.3,
    asp_disp_cycles=3
)


def validate_key_liquids(lash_e, output_dir):
    """Compact validation of key liquids used in degradation workflow."""
    if not VALIDATE_LIQUIDS:
        return
    
    try:
        from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy
        
        validation_tests = [
            {'vial': 'water', 'liquid': 'water', 'volumes': [0.01], 'reps': 3, 'params': None},
            {'vial': '2MeTHF', 'liquid': '2MeTHF', 'volumes': [0.15, 0.95], 'reps': 3, 'params': large_tip_2MeTHF_params},
            {'vial': '6M_HCl', 'liquid': '6M_HCl', 'volumes': [0.025, 0.015, 0.005], 'reps': 3, 'params': None},
            {'vial': 'polymer_stock', 'liquid': '2MeTHF', 'volumes': [0.100], 'reps': 3, 'params': small_tip_2MeTHF_params},
        ]
        
        lash_e.logger.info("Running compact liquid validation...")
        for test in validation_tests:
            result = validate_pipetting_accuracy(
                lash_e=lash_e, source_vial=test['vial'], destination_vial=test['vial'],
                liquid_type=test['liquid'], volumes_ml=test['volumes'], replicates=test['reps'],
                parameters=test['params'], output_folder=output_dir, switch_pipet=False, save_raw_data=not lash_e.simulate,
                condition_tip_enabled=True, conditioning_volume_ul=max(test['volumes'])*1000
            )
            param_info = "custom params" if test['params'] else "default params"
            lash_e.logger.info(f"  {test['vial']} ({param_info}): R^2={result['r_squared']:.3f}, Accuracy={result['mean_accuracy_pct']:.1f}%")
        
    except ImportError:
        lash_e.logger.info("Validation system not available, skipping...")


# Global well tracking
current_well_count = 0

def update_well_count(wells_used):
    global current_well_count
    current_well_count += wells_used
    print(f"Used {wells_used} wells. Total wells used so far: {current_well_count}")

# Gets the next well index to use based on the global well count
def get_next_well_index():
    return current_well_count

# Take aliquot -> Wellplate measurement (3 replicate)-> Put sample back to wellplate
def create_samples_and_measure(lash_e, output_dir, first_well_index, cytation_protocol_file_path, simulate,  sample_name, used_wells, replicates=3):

    create_samples_in_wellplate(lash_e, sample_name=sample_name, first_well_index=first_well_index, well_volume=0.15, replicates=replicates)
    wells = list(range(first_well_index, first_well_index + replicates))
    data_out = lash_e.measure_wellplate(cytation_protocol_file_path, wells_to_measure=wells, plate_type='quartz')
    save_data(data_out, output_dir, first_well_index, simulate,lash_e)
    update_well_count(replicates)
    used_wells.extend(wells)
    lash_e.logger.info("Used wells so far: %s", used_wells)

def pipet_sample_from_well_to_vial(lash_e, wells, sample_name, well_volume=0.15, heater_slot=None):
    lash_e.logger.info(f"Returning samples from wells {wells} back to sample vial: {sample_name}")
    lash_e.nr_robot.move_vial_to_location(sample_name, location='main_8mL_rack', location_index=4)
    for well in wells:
        lash_e.nr_robot.pipet_from_wellplate(well, volume=well_volume, aspirate=True,move_to_aspirate=True, well_plate_type="quartz")
        lash_e.nr_robot.dispense_into_vial(sample_name, well_volume)
    lash_e.nr_robot.remove_pipet()
    if heater_slot is not None:
        lash_e.nr_robot.move_vial_to_location(vial_name=sample_name, location='heater', location_index=heater_slot)


def safe_pipet(source_vial, dest_vial, volume, lash_e, parameters=None, liquid='2MeTHF', return_home=True):
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

    lash_e.nr_robot.dispense_from_vial_into_vial(
        source_vial, dest_vial, volume, 
        parameters=parameters, 
        liquid=liquid, 
        return_vial_home=return_home
    )

    if move_source: 
        vial_index = lash_e.nr_robot.get_vial_info(source_vial, 'vial_index')
        lash_e.nr_robot.VIAL_DF.at[vial_index, 'home_location_index'] = int(source_home_location_index)
        lash_e.nr_robot.return_vial_home(vial_name=source_vial)
    elif move_dest:
        vial_index = lash_e.nr_robot.get_vial_info(dest_vial, 'vial_index')
        lash_e.nr_robot.VIAL_DF.at[vial_index, 'home_location_index'] = int(dest_home_location_index)
        lash_e.nr_robot.return_vial_home(vial_name=dest_vial)

def stage_vial_safe(lash_e, vial_name, safe_index):#temporarily home position for safe pipetting
    original_home_index = lash_e.nr_robot.get_vial_info(vial_name, "location_index")
    vial_index = lash_e.nr_robot.get_vial_info(vial_name, "vial_index")

    if original_home_index > 5:
        lash_e.nr_robot.VIAL_DF.at[vial_index, "home_location_index"] = int(safe_index)
        lash_e.logger.info(f"Setting home location of {vial_name} to {safe_index} for safe pipetting")
        lash_e.nr_robot.move_vial_to_location(vial_name, location="main_8mL_rack", location_index=int(safe_index))
    return int(original_home_index)

def restore_vial_home(lash_e, vial_name, original_home_index): #restore original home position after safe pipetting
    vial_index = lash_e.nr_robot.get_vial_info(vial_name, "vial_index")
    lash_e.nr_robot.VIAL_DF.at[vial_index, "home_location_index"] = int(original_home_index)
    lash_e.nr_robot.return_vial_home(vial_name=vial_name)

def move_lid_to_wellplate(lash_e):
    lash_e.nr_track.grab_wellplate_from_location('lid_storage', wellplate_type='quartz_lid', waypoint_locations=['cytation_safe_area'])
    lash_e.nr_track.release_wellplate_in_location('pipetting_area', wellplate_type='quartz_lid')
    lash_e.nr_track.origin()

def move_lid_to_storage(lash_e):
    lash_e.nr_track.grab_wellplate_from_location('pipetting_area', wellplate_type='quartz_lid')
    lash_e.nr_track.release_wellplate_in_location('lid_storage', wellplate_type='quartz_lid')


def create_samples_in_wellplate(lash_e,sample_name,first_well_index,well_volume=0.15,replicates=1):
    lash_e.logger.info(f"\nTransferring sample: {sample_name} to wellplate at wells {first_well_index} to {first_well_index + replicates - 1} ({replicates} replicates)")
    # Create DataFrame for dispense_from_vials_into_wellplate method
    # Each row represents a well, each column represents a vial
    well_indices = list(range(first_well_index, first_well_index + replicates))
    well_plate_df = pd.DataFrame(index=well_indices, columns=[sample_name])
    well_plate_df[sample_name] = well_volume  # Same volume for each replicate well

    move_lid_to_storage(lash_e)

    # Use serial strategy for precise dispensing with full parameter support
    lash_e.nr_robot.dispense_from_vials_into_wellplate(well_plate_df=well_plate_df, strategy="serial" )

    move_lid_to_wellplate(lash_e)
    
    # Pipette samples back from wells to vial
    well_indices = list(range(first_well_index, first_well_index + replicates))
    lash_e.logger.info(f"Returning samples from wells {well_indices} back to sample vial: {sample_name}")
    move_lid_to_storage(lash_e)
    for well in well_indices:
        lash_e.nr_robot.pipet_from_wellplate(well, volume=well_volume, aspirate=True, well_plate_type="96 WELL PLATE")
        lash_e.nr_robot.dispense_into_vial(sample_name, well_volume)
    lash_e.nr_robot.remove_pipet()
    move_lid_to_wellplate(lash_e)

def save_data(data_out,output_dir,first_well_index,simulate,lash_e):
    if not simulate:
        output_file = output_dir / f'output_{first_well_index}.txt'
        data_out.to_csv(output_file, sep=',')
        lash_e.logger.info("Data saved to: %s", output_file)
    else:
        lash_e.logger.info(f"Simulation mode: Would save data for well index {first_well_index}")


# Clean well plate = Solvent wash *1 + Acetone wash *1 <- DO it serially (ie wash 5 wells at a time, and wash all wells w solvent first before moving on to acetone)
def wash_wellplate(lash_e, used_wells, solvent_vial, acetone_vial, waste_state, well_volume=0.19, solvent_repeats=1, acetone_repeats=1):
    lash_e.logger.info(f"\nWashing wellplate wells: {used_wells}")

    PLATE = "96 WELL PLATE"

    # Pipette all measured samples from wells into waste
    current_waste = check_and_switch_waste_vial(lash_e, waste_state)
    move_lid_to_storage(lash_e)



    def chunk(used_wells, n=4):
        for i in range(0, len(used_wells), n):
            yield used_wells[i:i+n]


    # 1 * Solvent wash
    waste_temp_home = stage_vial_safe(lash_e, current_waste, safe_index=4)
    solvent_temp_home = stage_vial_safe(lash_e, solvent_vial,safe_index=5)

    for _ in range(solvent_repeats):
        for wells in chunk(used_wells, 4):
            total = well_volume * len(wells)            # total to aspirate for this chunk (<= 0.8 mL if well_volume=0.2)
            dispense_volume = [well_volume] * len(wells)  # one volume per well

        # 1) Aspirate once from solvent vial
            lash_e.nr_robot.aspirate_from_vial(solvent_vial, total, track_height=True)

        # 2) Dispense into the wells (multi-well dispense)
            lash_e.nr_robot.dispense_into_wellplate(wells, dispense_volume, well_plate_type=PLATE)

        # 3) Mix each well
            for w in wells:
                lash_e.nr_robot.mix_well_in_wellplate(w, well_volume, repeats=2, well_plate_type=PLATE)
        # 4) Empty each well into waste
            for w in wells:
                lash_e.nr_robot.pipet_from_wellplate(w, well_volume, aspirate=True, well_plate_type=PLATE)
                lash_e.nr_robot.dispense_into_vial(current_waste, well_volume)

    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.recap_clamp_vial()  # Recaps whatever vial is currently in clamp
    restore_vial_home(lash_e, solvent_vial, solvent_temp_home)
    restore_vial_home(lash_e, current_waste, waste_temp_home)

    lash_e.logger.info("\nSolvent wash completed.")

    # 2 * Acetone wash
    waste_temp_home = stage_vial_safe(lash_e, current_waste, safe_index=4)
    acetone_temp_home = stage_vial_safe(lash_e, acetone_vial, safe_index=5)
 
    for _ in range(acetone_repeats):
        for wells in chunk(used_wells, 4):
            total = well_volume * len(wells)           
            dispense_volume = [well_volume] * len(wells)

            lash_e.nr_robot.aspirate_from_vial(acetone_vial, total, track_height=True)
            lash_e.nr_robot.dispense_into_wellplate(wells, dispense_volume, well_plate_type=PLATE)
            for w in wells:
                lash_e.nr_robot.mix_well_in_wellplate(w, well_volume, repeats=2, well_plate_type=PLATE)
            for w in wells:
                lash_e.nr_robot.pipet_from_wellplate(w, well_volume, aspirate=True, move_to_aspirate=True, well_plate_type=PLATE)
                lash_e.nr_robot.dispense_into_vial(current_waste, well_volume)
    
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.recap_clamp_vial()  # Recaps whatever vial is currently in clamp
    restore_vial_home(lash_e, acetone_vial, acetone_temp_home) 
    restore_vial_home(lash_e, current_waste, waste_temp_home) 
  
    lash_e.logger.info("\nAcetone wash completed.")

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
            lash_e.logger.info(f"Switching from {current_waste} to {new_waste_vial} (was {current_vol:.1f}mL)")
            return new_waste_vial
        else:
            return current_waste
    except Exception as e:
        lash_e.logger.warning(f"Could not check waste vial volume: {e}, continuing with {current_waste}")
        return current_waste

#  -------------------------------------------------------------- Define my workflow here! ---------------------------------------------------
# Acid library: maps acid name to {'molar_mass': g/mol, 'molarity': mol/L}
ACID_LIBRARY = {
    '6M_HCl': {'molar_mass': 36.46, 'molarity': 6.0},
    'TFA': {'molar_mass': 114.02, 'molarity': 6.0},
}

def degradation_workflow(sample_number, acid_type, acid_molar_excess, water_volume, solvent='2MeTHF'):
  
    # a. Initial State of your Vials, so the robot can know where to pipet:
    INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/degradation_vial_status.csv"

    # b. Cytation 5 UV-Vis Measurement protocol:
    CYTATION_PROTOCOL_FILE = r"C:\Protocols\300_900_sweep.prt"

    # c. Time schedule for UV-VIS measurements: 
    SCHEDULE_FILE = "../utoronto_demo/status/degradation_vial_schedule.csv"

    # d. Simulate mode True or False
    SIMULATE = True #Set to True if you want to simulate the robot, False if you want to run it on the real robot
    
    # e. Number of replicate measurements per timepoint
    REPLICATES = 1  # Number of wells to use for each measurement (default: 3)

    # f. Initialize the workstation, which includes the robot, track, cytation and photoreactors
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_t8=True)

    # g. Polymer dilution calculation:
    sample_volume = 2.0 # Total volume of each polymer sample (mL)
    df = pd.read_csv(INPUT_VIAL_STATUS_FILE)
    sample_col = 'vial_name'  # Use vial_name column from CSV

    # h. Get stock concentration
    stock_conc = df.loc[df[sample_col] == 'polymer_stock', 'concentration(mg/mL)'].iloc[0] #This selects the row where vial_name == 'polymer_stock' and pulls its 'concentration(mg/mL)'
    lash_e.logger.info(f"Stock concentration: {stock_conc} mg/mL")


    
    # i. Calculate dilution volumes for each sample
    samples = df[df[sample_col].str.contains("sample", case=False, na=False)].copy() # case=false -> Select rows where vial_name contains 'sample' in any capitalization, na=false -> if vial_name is missing, treat it as not a match
    samples['stock_volume'] = (samples['concentration(mg/mL)'] / stock_conc) * sample_volume # creates a new column 'stock_volume' in 'samples' df
    samples['solvent_volume'] = sample_volume - samples['stock_volume']
    lash_e.logger.info("Calculated volumes for each sample:")
    lash_e.logger.info("%s", samples[[sample_col, 'concentration(mg/mL)', 'stock_volume', 'solvent_volume']])

    # j. acid amount calculation:
    polymer_molar_mass = 1353350 # molar mass in mg to simplify calculation afterwards
    
    # Get acid properties from library
    acid_molarity = ACID_LIBRARY[acid_type]['molarity']
    acid_molar_mass = ACID_LIBRARY[acid_type]['molar_mass']
    lash_e.logger.info(f"Using acid: {acid_type}")
  
    # Calculates the volume of acid to be added
    samples['acid_molar_excess'] = acid_molar_excess
    if acid_type == '6M_HCl':
        samples['acid_volume'] = ((((samples['concentration(mg/mL)'] * sample_volume) / polymer_molar_mass) * acid_molar_excess) / acid_molarity) * 1000 # in mL
    if acid_type != '6M_HCl':
        samples['acid_volume'] = ((((samples['concentration(mg/mL)'] * sample_volume) / polymer_molar_mass) * acid_molar_excess) * acid_molar_mass) * 1000 # in mL
    lash_e.logger.info("Calculated volumes for each sample:")
    lash_e.logger.info("%s", samples[[sample_col, 'acid_molar_excess', 'acid_volume']])

    # k. Initialize waste vial management
    waste_state = {
        "waste_index": 0,  # Start with waste_0
        "current_waste_vial": "waste_0"
    }

    # Select only the specified sample
    sample_name = f'sample_{sample_number}'
    sample_solutions = {sample_name}
    if sample_name not in samples[sample_col].values:
        raise ValueError(f"Sample {sample_name} not found in vial status file.")

    # Create a dictionary to look up volumes by sample name
    volume_lookup = {}
    for _, row in samples.iterrows():
        sample_name = row[sample_col]
        volume_lookup[sample_name] = {
            'stock_volume': row['stock_volume'],
            'solvent_volume': row['solvent_volume'],
            'acid_volume': row['acid_volume']
        }


    # l. Get stock concentration
    stock_conc = df.loc[df[sample_col] == 'polymer_stock', 'concentration(mg/mL)'].iloc[0] #This selects the row where vial_name == 'polymer_stock' and pulls its 'concentration(mg/mL)'
    lash_e.logger.info(f"Stock concentration: {stock_conc} mg/mL")


    # input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")    
    lash_e.nr_robot.check_input_file()

    #create file name for the output data
    if not SIMULATE:
        import slack_agent
        exp_name = input("Experiment name: ")
        output_dir = Path(r'C:\Users\Imaging Controller\Desktop\SQ') / exp_name #appends exp_name to the output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        lash_e.logger.info("Output directory created at: %s", output_dir)
        
        # Run validation if enabled
        validate_key_liquids(lash_e, output_dir)
        
        slack_agent.send_slack_message("Degradation workflow started!")

        lash_e.nr_robot.home_robot_components()
    else:
        output_dir = None
        validate_key_liquids(lash_e, output_dir)  # Also validate in simulation

    # -------------------------------------------------------------- Workflow starts from here! ---------------------------------------------------

    # 1. Polymer sample preparation: Add solvent then stock to each sample vial.
    for sample in sample_solutions:
        solvent_vol = volume_lookup[sample]['solvent_volume']
        lash_e.logger.info(f"\nAdding {solvent_vol:.3f} mL solvent to {sample}")
        safe_pipet(solvent, sample, solvent_vol, lash_e, 
                  parameters=large_tip_2MeTHF_params, liquid=solvent) 
    
    for sample in sample_solutions:
        stock_vol = volume_lookup[sample]['stock_volume']
        lash_e.logger.info(f"\nAdding {stock_vol:.3f} mL stock solution to {sample}")
        safe_pipet('polymer_stock', sample, stock_vol, lash_e, 
                  parameters=small_tip_2MeTHF_params, liquid=solvent)
        lash_e.nr_robot.vortex_vial(vial_name=sample, vortex_time=5)


    # 2. Add acid and water to the polymer samples to initiate degradation and take scheduled UV-VIS measurements.
    t0_map = {} #Dictionary: sample -> start time (same time basis as start_time/current_time)
    first_well_index = get_next_well_index() #Use the global well tracking to continue from where we left off
    # Tracks used wells as a list of integer indices
    used_wells = []

    i = 0

    heater_slot = {}  # sample_name -> heater index

    for sample in sample_solutions:
        lash_e.logger.info(f"\nAdding {water_volume} mL water to sample: {sample}")
        lash_e.nr_robot.dispense_from_vial_into_vial('water', sample, water_volume, use_safe_location=False, liquid='water')
        lash_e.nr_robot.remove_pipet()
        acid_volume = round(float(volume_lookup[sample]['acid_volume']), 4)
        lash_e.logger.info(f"\nAdding {acid_volume} mL acid to sample: {sample}")
        safe_pipet(acid_type,sample, acid_volume, lash_e, return_home=True)
        lash_e.nr_robot.remove_pipet()

        # Record per-sample start time using consistent basis (simulated clock = 0; real clock = wall time)
        t0_map[sample] = 0 if SIMULATE else time.time()
        create_samples_and_measure(lash_e, output_dir, first_well_index, CYTATION_PROTOCOL_FILE, SIMULATE, sample_name=sample, used_wells=used_wells, replicates=REPLICATES)
        try:
            lash_e.temp_controller.turn_off_stirring()
            if not SIMULATE:
                time.sleep(1)
        except: 
            lash_e.logger.info("Stirring was already off.")
        finally:
            lash_e.nr_robot.move_vial_to_location(vial_name=sample,location='heater',location_index=i)
            heater_slot[sample] = i
            lash_e.temp_controller.turn_on_stirring()
        first_well_index += REPLICATES  # Move to next set of wells for next sample
        i+=1

    schedule = pd.read_csv(SCHEDULE_FILE, sep=",") #Read the schedule file
    schedule = schedule[schedule['sample_index'] == sample_name]  # Filter to only this sample
    schedule = schedule.sort_values(by='start_time') #sort in ascending time order
    lash_e.logger.info("Schedule for %s: %s", sample_name, schedule)

    # Establish timing model.
    # Simulation: use an integer counter (seconds). Real run: wall clock seconds.
    start_time = 0 if SIMULATE else time.time()
    current_time = start_time
    lash_e.logger.info("Starting timed portion at (secs): %s", start_time)

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
            lash_e.logger.info(f"\n[TRIGGER] {action_required} for {sample_index} at elapsed {elapsed_time:.0f}s (target {time_required}s)")
            if action_required in ("create_samples_and_measure","measure_samples"):
                create_samples_and_measure(
                    lash_e, output_dir, first_well_index, CYTATION_PROTOCOL_FILE, SIMULATE, sample_name=sample_index,used_wells=used_wells,replicates=REPLICATES)
                first_well_index += REPLICATES
            else:
                lash_e.logger.info(f"[WARN] Unknown action '{action_required}' – skipping (row {items_completed})")
            items_completed += 1
        else:
            # Heartbeat
            if total_elapsed >= time_increment:
                lash_e.logger.info(f"Total time elapsed={total_elapsed:.0f}s; Next event in={time_required - elapsed_time:.0f}s -> {sample_index}")
                time_increment += 60

        # Safety cutoff (simulation or real) to avoid runaway
        if total_elapsed > safety_cutoff:
            lash_e.logger.info(f"[SAFETY STOP] Elapsed {total_elapsed:.0f}s exceeded schedule max {max_schedule_time}s + buffer. Breaking loop.")
            break

        if not SIMULATE:
            time.sleep(0.25)
    lash_e.nr_robot.move_home()   

    if 'sample_index' in locals():
        lash_e.nr_robot.return_vial_home(sample_index)
    else:
        lash_e.nr_robot.return_vial_home(sample_name)
    lash_e.temp_controller.turn_off_stirring()
    lash_e.nr_robot.move_home()

    # 3. Clean the well plate after all measurements are done
    wash_wellplate(lash_e, used_wells, solvent_vial=solvent, acetone_vial='acetone', waste_state=waste_state, solvent_repeats=1, acetone_repeats=1, well_volume=0.19)

    # 4. Home all components at the end of the workflow
    lash_e.nr_robot.move_home()
    
    if not SIMULATE:   
        slack_agent.send_slack_message("Degradation workflow completed!")

# Run my workflow here!
degradation_workflow(sample_number=1, acid_type='6M_HCl', solvent='2MeTHF', acid_molar_excess=1000, water_volume=0.010)


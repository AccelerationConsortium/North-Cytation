import sys
import time
from venv import create
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E 
import pandas as pd
from pathlib import Path

#Question: how to incorporate the pipetting params?
# Take aliquot -> Wellplate measurement (3 replicate)-> Put sample back to wellplate
def create_samples_and_measure(lash_e, output_dir, first_well_index, cytation_protocol_file_path, simulate, sample_name, used_wells, replicates=3):
    create_samples_in_wellplate(lash_e, sample_name=sample_name, first_well_index=first_well_index, well_volume=0.15, replicates=replicates)
    wells = list(range(first_well_index, first_well_index + replicates))
    pipet_sample_from_well_to_vial(lash_e, wells, sample_name, replicates)
    lash_e.nr_robot.pipet_from_wellplate(wells, volume=0.15, aspirate=True, move_to_aspirate=True, well_plate_type="96 WELL PLATE")
    data_out = lash_e.measure_wellplate(cytation_protocol_file_path, wells_to_measure=wells)
    save_data(data_out, output_dir, first_well_index, simulate)
    used_wells.extend(wells)
    print("Used wells so far:", used_wells)
    
def pipet_sample_from_well_to_vial(lash_e, wells, sample_name):
    print(f"Returning samples from wells {wells} back to sample vial: {sample_name}")
    lash_e.nr_robot.move_vial_to_location(sample_name, location='main_8mL_rack', location_index=4)
    lash_e.nr_robot.pipet_from_wellplate(wells, volume=0.2, aspirate=True, well_plate_type="96 WELL PLATE")
    lash_e.nr_robot.dispense_into_vial(sample_name, volume=0.2)
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

def create_samples_in_wellplate(lash_e,sample_name,first_well_index,well_volume=0.15,replicates=1):
    print(f"\nTransferring sample: {sample_name} to wellplate at wells {first_well_index} to {first_well_index + replicates - 1} ({replicates} replicates)")
    # Create DataFrame for dispense_from_vials_into_wellplate method
    # Each row represents a well, each column represents a vial
    well_indices = list(range(first_well_index, first_well_index + replicates))
    well_plate_df = pd.DataFrame(index=well_indices, columns=[sample_name])
    well_plate_df[sample_name] = well_volume  # Same volume for each replicate well
    
    # Use serial strategy for precise dispensing with full parameter support
    lash_e.nr_robot.dispense_from_vials_into_wellplate(well_plate_df=well_plate_df, strategy="serial" )

def save_data(data_out,output_dir,first_well_index,simulate):
    if not simulate:
        output_file = output_dir / f'output_{first_well_index}.txt'
        data_out.to_csv(output_file, sep=',')
        print("Data saved to:", output_file)
    else:
        print(f"Simulation mode: Would save data for well index {first_well_index}")

# Clean well plate = Solvent wash *1 + Acetone wash *2 <- DO it serially (ie wash 5 wells at a time, and wash all wells w solvent first before moving on to acetone)
def wash_wellplate(lash_e, used_wells, solvent_vial, wash_vial, waste_state,well_volume=0.2, solvent_repeats=1, acetone_repeats=2):
    print(f"\nWashing wellplate wells: {used_wells}")

    current_waste = check_and_switch_waste_vial(lash_e, waste_state)
    lash_e.nr_robot.move_vial_to_location(current_waste, location='main_8mL_rack', location_index=4)

    def chunk(used_wells, n=4):
        for i in range(0, len(used_wells), n):
            yield used_wells[i:i+n]

    PLATE = "96 WELL PLATE"

    # 1 * Solvent wash
    lash_e.nr_robot.move_vial_to_location(solvent_vial, location='main_8mL_rack', location_index=5)

    for wells in chunk(used_wells, 4):
        total = well_volume * len(wells)            # total to aspirate for this chunk (<= 0.8 mL if well_volume=0.2)
        dispense_vols = [well_volume] * len(wells)  # one volume per well

        # 1) Aspirate once from solvent vial
        lash_e.nr_robot.aspirate_from_vial(solvent_vial, total, track_height=True)

        # 2) Dispense into the wells (multi-well dispense)
        lash_e.nr_robot.dispense_into_wellplate(wells, dispense_vols, well_plate_type=PLATE)

        # 3) Mix each well
        for w in wells:
            lash_e.nr_robot.mix_well_in_wellplate(w, well_volume, repeats=2, well_plate_type=PLATE)

        # 4) Empty each well into waste
        for w in wells:
            lash_e.nr_robot.pipet_from_wellplate(w, well_volume, aspirate=True, move_to_aspirate=False, well_plate_type=PLATE)
            lash_e.nr_robot.dispense_into_vial(current_waste, well_volume)

    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.return_vial_home(solvent_vial)

    print("\nSolvent wash completed.")

    # 2 * Acetone wash
    for _ in range(acetone_repeats):
        lash_e.nr_robot.move_vial_to_location(wash_vial, location='main_8mL_rack', location_index=5)

        for wells in chunk(used_wells, 4):
            total = well_volume * len(wells)
            dispense_vols = [well_volume] * len(wells)
            
            lash_e.nr_robot.aspirate_from_vial(wash_vial, total, track_height=True)
            lash_e.nr_robot.dispense_into_wellplate(wells, dispense_vols, well_plate_type=PLATE)

            for w in wells:
                lash_e.nr_robot.mix_well_in_wellplate(w, well_volume, repeats=2, well_plate_type=PLATE)

            for w in wells:
                lash_e.nr_robot.pipet_from_wellplate(w, well_volume, aspirate=True, move_to_aspirate=False, well_plate_type=PLATE)
                lash_e.nr_robot.dispense_into_vial(current_waste, well_volume)

        lash_e.nr_robot.remove_pipet()

    lash_e.nr_robot.return_vial_home(wash_vial)
    lash_e.nr_robot.return_vial_home(current_waste)
    print("\nAcetone wash completed.")

def get_time(simulate,current_time=None):
    if not simulate:
        return time.time()
    else:
        if current_time is None:
            return 0

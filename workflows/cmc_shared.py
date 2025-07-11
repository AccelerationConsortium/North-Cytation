# --- cmc_shared.py ---
import sys
sys.path.append("../utoronto_demo")
import math
import pandas as pd
import slack_agent
import os
import numpy as np

def check_input_file(input_file):
    vial_status = pd.read_csv(input_file, sep=",")
    print(vial_status)
    input("Only hit enter if the status of the vials (including open/close) is correct, otherwise hit ctrl-c")

def split_volume(volume, max_volume=1.0):
    if volume <= max_volume:
        return [volume]
    n_parts = math.ceil(volume / max_volume)
    part_volume = volume / n_parts
    return [part_volume] * n_parts

def mix_surfactants(lash_e, sub_stock_vols, substock_vial):
    print("\nCombining Surfactants:")
    print("Stock solution composition:", sub_stock_vols)

    for surfactant, volume_uL in sub_stock_vols.items():
        if surfactant == 'water':
            continue
        volume_mL = volume_uL / 1000
        print(f"\nHandling {surfactant}: {volume_mL:.3f} mL")
        if volume_mL > 0:
            volumes = split_volume(volume_mL)
            print("Pipetable volumes:", volumes)
            vial_location = lash_e.nr_robot.get_vial_info(surfactant, 'location')
            if vial_location == 'main_8mL_rack':
                lash_e.nr_robot.move_vial_to_location(surfactant, 'main_8mL_rack', 43)
            for v in volumes:
                lash_e.nr_robot.dispense_from_vial_into_vial(surfactant, substock_vial, v, blowout_vol=0.1)
            lash_e.nr_robot.remove_pipet()
            if vial_location == 'main_8mL_rack':
                lash_e.nr_robot.return_vial_home(surfactant)
            

    lash_e.nr_robot.dispense_into_vial_from_reservoir(1, substock_vial, sub_stock_vols['water'] / 1000)
    lash_e.nr_robot.move_vial_to_location(substock_vial, 'main_8mL_rack', 43)
    lash_e.nr_robot.mix_vial(substock_vial, 0.9, repeats=10)
    lash_e.nr_robot.remove_pipet()

def fill_water_vial(lash_e):
    vial_max_volume = 7.5
    water_reservoir = 1
    current_water_volume = lash_e.nr_robot.get_vial_info('water', 'vial_volume')
    if current_water_volume < vial_max_volume:
        print(f"Filling water vial from {current_water_volume} mL to {vial_max_volume} mL")
        lash_e.nr_robot.dispense_into_vial_from_reservoir(water_reservoir, 'water', vial_max_volume - current_water_volume)

def split_water_batches(df_water, max_volume=7.0):
    batches = []
    current_batch = []
    cumulative_vol = 0.0
    for _, row in df_water.iterrows():
        vol = row['water volume']
        if cumulative_vol + vol > max_volume:
            if current_batch:
                batches.append(pd.DataFrame(current_batch))
            current_batch = [row]
            cumulative_vol = vol
        else:
            current_batch.append(row)
            cumulative_vol += vol
    if current_batch:
        batches.append(pd.DataFrame(current_batch))
    return batches

def create_wellplate_samples(lash_e, wellplate_data, substock_vial_index, last_wp_index):
    print("\n Dispensing into Wellplate")
    samples_per_assay = wellplate_data.shape[0]
    well_indices = range(last_wp_index, last_wp_index + samples_per_assay)
    dispense_data = (wellplate_data[['surfactant volume', 'water volume', 'probe volume']] / 1000).round(3)
    dispense_data.index = well_indices
    print(dispense_data)

    df_surfactant = dispense_data[['surfactant volume']]
    df_water = dispense_data[['water volume']]
    df_dmso = dispense_data[['probe volume']]

    water_batch_df = split_water_batches(df_water)

    lash_e.nr_robot.dispense_from_vials_into_wellplate(df_dmso, ['pyrene_DMSO'], well_plate_type="48 WELL PLATE", dispense_speed=20, wait_time=2, asp_cycles=1, low_volume_cutoff=0.04, buffer_vol=0, pipet_back_and_forth=True, blowout_vol=0.1)
    lash_e.nr_robot.dispense_from_vials_into_wellplate(df_surfactant, [substock_vial_index], well_plate_type="48 WELL PLATE", dispense_speed=15)

    # for water_batch in water_batch_df:
    #     print(water_batch)
    #     lash_e.nr_robot.dispense_from_vials_into_wellplate(water_batch, ['water'], well_plate_type="48 WELL PLATE", dispense_speed=11)
    #     fill_water_vial(lash_e)

    #Only run the second batch if it exists
    if len(water_batch_df) >= 2:
        print(water_batch_df[1])
        high_volume_df = water_batch_df[1][water_batch_df[1]['water volume'] > 0.05]
        low_volume_df = water_batch_df[1][water_batch_df[1]['water volume'] <= 0.05]
        lash_e.nr_robot.dispense_from_vials_into_wellplate(low_volume_df, ['water_large'], well_plate_type="48 WELL PLATE", dispense_speed=11)
        lash_e.nr_robot.dispense_from_vials_into_wellplate(high_volume_df, ['water_large'], well_plate_type="48 WELL PLATE", dispense_speed=11, remove_pipet=False)
    
    print(water_batch_df[0])
    lash_e.nr_robot.dispense_from_vials_into_wellplate(water_batch_df[0], ['water'], well_plate_type="48 WELL PLATE", dispense_speed=11)
    fill_water_vial(lash_e)


    lash_e.nr_robot.return_vial_home(substock_vial_index)

    lash_e.nr_robot.get_pipet(0)
    for well in well_indices:
        lash_e.nr_robot.mix_well_in_wellplate(well, volume=0.3, well_plate_type="48 WELL PLATE")
    lash_e.nr_robot.remove_pipet()

def analyze_and_save_results(folder, details, wellplate_data, resulting_data, analyzer, save_modifier, log=False):
    concentrations = wellplate_data['concentration']
    if len(resulting_data) > len(concentrations):
        n_reps = len(resulting_data) // len(concentrations)
    if len(resulting_data) % len(concentrations) != 0:
        print("⚠️ Warning: replicate count is not an integer — concentration alignment may be wrong.")
    concentrations = np.tile(concentrations.values, n_reps)

    # --- Calculate fluorescence ratio ---
    try:
        if '334_373' in resulting_data.columns and '334_384' in resulting_data.columns:
            resulting_data['ratio'] = resulting_data['334_373'] / resulting_data['334_384']
        else:
            raise KeyError("Missing required columns 334_373 or 334_384")
    except KeyError:
        print("⚠️ No fluorescence data found (missing 334_373 or 334_384)")
        return None

    # --- Check absorbance ---
    try:
        if (resulting_data['600'] > 0.1).any():
            print("⚠️ Warning: Some absorbance (600 nm) values exceed 0.1!")
    except KeyError:
        print("⚠️ No absorbance data (missing 600 nm column)")

    # --- Output folder setup ---
    parent_folder = os.path.dirname(folder)
    graphs_folder = os.path.join(parent_folder, "graphs")
    os.makedirs(graphs_folder, exist_ok=True)

    # --- Run CMC analysis using all ratios ---
    figure_name = os.path.join(graphs_folder, f'CMC_plot_{details}_{save_modifier}.png')

    try:
        A1, A2, x0, dx, r_squared = analyzer.CMC_plot(resulting_data['ratio'].values, concentrations, figure_name, log=log)
    except Exception as e:
        print(f"❌ CMC fit failed: {e}")
        return None

    # --- Save all relevant data ---
    wellplate_data.to_csv(os.path.join(folder, f'wellplate_data_{details}_{save_modifier}.csv'), index=False)
    resulting_data.to_csv(os.path.join(folder, f'output_data_{details}_{save_modifier}.csv'), index=False)
    with open(os.path.join(folder, f'wellplate_data_results_{details}_{save_modifier}.txt'), "w") as f:
        f.write(f"CMC: {x0}, r2: {r_squared}, A1: {A1}, A2: {A2}, dx: {dx}")

    print("CMC (mMol): ", x0)
    print("R-squared: ", r_squared)
    slack_agent.send_slack_message(f"CMC Workflow complete! CMC={x0}, R-squared={r_squared}")
    slack_agent.upload_and_post_file(figure_name, 'CMC Image')

    return {
        "CMC": x0,
        "r2": r_squared,
        "A1": A1,
        "A2": A2,
        "dx": dx
    }
def coalesce_replicates_long(df_wide):
    """
    Converts wide-format MultiIndex replicate data into long format:
    Each replicate becomes a new row, retaining all columns.
    """
    records = []

    for top_level in df_wide.columns.levels[0]:
        try:
            rep_str, protocol = top_level.split('_', 1)
            replicate = int(rep_str.replace('rep', ''))
        except Exception as e:
            print(f"⚠️ Skipping column group {top_level}: {e}")
            continue

        sub_df = df_wide[top_level].copy()
        sub_df = sub_df.copy()
        sub_df['replicate'] = replicate
        sub_df['protocol'] = protocol
        sub_df['sample_id'] = sub_df.index
        records.append(sub_df)

    df_long = pd.concat(records, ignore_index=True)
    return df_long

def merge_absorbance_and_fluorescence(df_long):
    """
    Merge absorbance and fluorescence data on sample_id and replicate.
    Only retains expected columns to avoid _x/_y duplication.
    """

    # Filter rows
    absorb = df_long[df_long['protocol'] == 'CMC_Absorbance'].copy()
    fluor  = df_long[df_long['protocol'] == 'CMC_Fluorescence'].copy()

    # Sanity check — no overlap in columns except keys
    print("Absorbance columns:", absorb.columns.tolist())
    print("Fluorescence columns:", fluor.columns.tolist())

    # Reduce to only necessary columns
    absorb = absorb[['sample_id', 'replicate', '600']]
    fluor  = fluor[['sample_id', 'replicate', '334_373', '334_384']]

    # Merge
    merged = pd.merge(fluor, absorb, on=['sample_id', 'replicate'], how='inner')

    return merged

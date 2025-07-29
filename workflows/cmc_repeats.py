# --- cmc_timed_repeat_workflow.py ---
import sys
sys.path.append("../utoronto_demo")
from cmc_shared import *
import analysis.cmc_exp_new as experimental_planner
import analysis.cmc_data_analysis as analyzer

from master_usdl_coordinator import Lash_E
from datetime import datetime, timedelta
import os
import time
import pandas as pd
from analysis.CMC_replicate_analysis import analyze_combined_replicates
from analysis.CMC_replicate_analysis import analyze_summary_variation

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/CMC_workflow_repeats_input.csv"
LOGGING_FOLDER = "../utoronto_demo/logs/"
MEASUREMENT_PROTOCOL_FILE = r"C:\\Protocols\\CMC_Fluorescence.prt"
simulate = enable_logging = False
repeats = 3  # Number of replicate measurements

REPEATS_PER_BATCH = 3
#SURFACTANTS_TO_RUN = ['SDS', 'NaDC', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'CAPB', 'CHAPS'] #Note that we might just do 4 at a time (each assay takes about 4 hours)
surf_labels = ['a', 'b', 'c']  # Must match REPEATS_PER_BATCH
SURFACTANTS_TO_RUN = ['CAPB']
delay_minutes = [0, 5, 10, 20, 30]


with Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate, logging=enable_logging) as lash_e:
    # Setup
    lash_e.nr_robot.check_input_file()
    lash_e.nr_track.check_input_file()
    timestamp_start = datetime.now().strftime("%Y%m%d_%H%M")
    main_folder = f'C:/Users/Imaging Controller/Desktop/CMC/{timestamp_start}/'
    raw_data_folder = os.path.join(main_folder, "raw_data")

    if not simulate:
        os.makedirs(main_folder, exist_ok=True)
        print(f"[Main Folder] Created at: {main_folder}")
        os.makedirs(raw_data_folder, exist_ok=True)
        print(f"[Raw Data Folder] Created at: {raw_data_folder}")
        
    lash_e.grab_new_wellplate()

    summary_records = []
    substock_counter = 1
    starting_wp_index = 0

    for surfactant in SURFACTANTS_TO_RUN:
        print(f"\n--- Starting workflow for surfactant: {surfactant} ---")
        surfactants = [surfactant]
        ratio_vector = [1]

        experiment, _ = experimental_planner.generate_exp_flexible(surfactants, ratio_vector, rough_screen=True)
        sub_stock_vols = experiment['surfactant_sub_stock_vols']
        wellplate_data = experiment['df']
        samples_per_assay = wellplate_data.shape[0]

        for repeat_index in range(REPEATS_PER_BATCH):
            substock_vial = f'substock_{substock_counter}'
            substock_counter += 1
            repeat_label = surf_labels[repeat_index]

            lash_e.nr_robot.prime_reservoir_line(1, 'water', 0.5)
            mix_surfactants(lash_e, sub_stock_vols, substock_vial)
            fill_water_vial(lash_e)
            create_wellplate_samples(lash_e, wellplate_data, substock_vial, starting_wp_index)

            assay_start_time = datetime.now()
            sample_indices = range(starting_wp_index, starting_wp_index + samples_per_assay)

            # Define relative measurement schedule (minutes)
            
            for delay in delay_minutes:
                if not simulate:
                    target_time = assay_start_time + timedelta(minutes=delay)
                    while datetime.now() < target_time:
                        time_to_wait = (target_time - datetime.now()).total_seconds()
                        print(f"Waiting {time_to_wait:.1f} seconds for assay {repeat_label}, timepoint {delay} minutes...", flush=True)
                        time.sleep(min(time_to_wait, 5))
                else:
                    print(f"[Simulate] Advancing time by {delay} minutes for assay {repeat_label}...", flush=True)
                    # Fast-forward simulated time conceptually
                    # No actual wait required

                label_prefix = f"{surfactant}_{delay}min_{repeat_label}"
                if not simulate:
                    results = lash_e.measure_wellplate(
                        MEASUREMENT_PROTOCOL_FILE,
                        wells_to_measure=sample_indices,
                        plate_type="48 WELL PLATE",
                        repeats=repeats
                    )
                    results.to_csv(os.path.join(raw_data_folder, f"{label_prefix}_raw_multiindex.csv"))

                    details = "_".join(f"{k}{int(v)}" for k, v in sub_stock_vols.items())

                    results_concat = merge_absorbance_and_fluorescence(coalesce_replicates_long(results))

                    metrics = analyze_and_save_results(
                        raw_data_folder, details, wellplate_data, results_concat, analyzer, label_prefix, log=True
                    )

                    summary_records.append({
                        "Surfactant": surfactant,
                        "Assay": repeat_label,
                        "Time_min": delay,
                           **metrics
                    })

            starting_wp_index += samples_per_assay
            print("Wellplate index: ", starting_wp_index)

        last_surfactant = surfactant == SURFACTANTS_TO_RUN[-1]

        lash_e.discard_used_wellplate()
        starting_wp_index = 0
        if not (last_surfactant):
            print("Getting new wellplate...")
            lash_e.grab_new_wellplate()
            

        print(f"All assays complete for {surfactant}.")

    # Save summary
    if not simulate:
        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(os.path.join(main_folder, "CMC_measurement_summary.csv"), index=False)

    if not simulate:
        replicates_dir = os.path.join(main_folder, "replicates")
        variation_dir = os.path.join(main_folder, "variation")
        os.makedirs(replicates_dir, exist_ok=True)
        os.makedirs(variation_dir, exist_ok=True)

        analyze_summary_variation(summary_df, variation_dir)
        analyze_combined_replicates(raw_data_folder, replicates_dir) 

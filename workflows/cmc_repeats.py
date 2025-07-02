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
import matplotlib.pyplot as plt

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/CMC_workflow_input.csv"
LOGGING_FOLDER = "../utoronto_demo/logs/"
MEASUREMENT_PROTOCOL_FILE = r"C:\\Protocols\\CMC_Fluorescence.prt"
simulate = True
enable_logging = True

REPEATS_PER_BATCH = 4
#SURFACTANTS_TO_RUN = ['SDS', 'NaDC', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'CAPB', 'CHAPS'] #Note that we might just do 4 at a time (each assay takes about 4 hours)
surf_labels = ['a', 'b', 'c', 'd']  # Must match REPEATS_PER_BATCH
SURFACTANTS_TO_RUN = ['SDS']
#delay_minutes = [0, 5, 10, 20, 30]
delay_minutes = [0, 5, 10]

# Setup
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
lash_e.nr_robot.check_input_file()
timestamp_start = datetime.now().strftime("%Y%m%d_%H%M")
main_folder = f'C:/Users/Imaging Controller/Desktop/CMC/{timestamp_start}/'

if not simulate:
    os.makedirs(main_folder, exist_ok=True)
    print(f"[Main Folder] Created at: {main_folder}")

if enable_logging:
    log_file_path = os.path.join(LOGGING_FOLDER, f"experiment_log_{timestamp_start}_sim{simulate}.txt")
    log_file = open(log_file_path, "w")
    sys.stdout = sys.stderr = log_file

lash_e.nr_robot.prime_reservoir_line(1, 'water', 0.5)

summary_records = []
substock_counter = 1
starting_wp_index = 0

for surfactant in SURFACTANTS_TO_RUN:
    print(f"\n--- Starting workflow for surfactant: {surfactant} ---")
    surfactants = [surfactant]
    ratio_vector = [1]

    experiment, _ = experimental_planner.generate_exp_flexible(surfactants, ratio_vector, sub_stock_volume=6000, probe_volume=25)
    sub_stock_vols = experiment['surfactant_sub_stock_vols']
    wellplate_data = experiment['df']
    samples_per_assay = wellplate_data.shape[0]

    for repeat_index in range(REPEATS_PER_BATCH):
        substock_vial = f'substock_{substock_counter}'
        substock_counter += 1
        repeat_label = surf_labels[repeat_index]

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

            for rep in range(3):
                label = f"{surfactant}_{delay}min_{repeat_label}_rep{rep+1}"
                if not simulate:
                    results = lash_e.measure_wellplate(
                        MEASUREMENT_PROTOCOL_FILE,
                        sample_indices,
                        plate_type="48 WELL PLATE"
                    )
                    details = "_".join(f"{k}{int(v)}" for k, v in sub_stock_vols.items())
                    metrics = analyze_and_save_results(
                        main_folder, details, wellplate_data, results, analyzer, label
                    )
                else:
                    metrics = {"CMC": 0.01, "r2": 0.95, "A1": 1000, "A2": 1500, "dx": 0.05}
                    print(f"[Simulate] Measuring wells for {surfactant} {repeat_label} rep {rep+1} at {delay} min")

                summary_records.append({
                    "Surfactant": surfactant,
                    "Assay": repeat_label,
                    "Timing": f"{delay} minutes",
                    "Replicate": rep + 1,
                    **metrics
                })

        starting_wp_index += samples_per_assay

        if starting_wp_index >= 48:
            lash_e.discard_used_wellplate()
            lash_e.grab_new_wellplate()
            starting_wp_index = 0

    print(f"All assays complete for {surfactant}.")

# Save summary
if not simulate:
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(main_folder, "CMC_measurement_summary.csv"), index=False)

if enable_logging:
    log_file.close()

import re

def plot_variation(data, surfactant_name, output_dir):
    df = data[data["Surfactant"] == surfactant_name].copy()

    # Convert Timing to numeric if possible (e.g., "30 minutes" -> 30)
    try:
        df["Timing_num"] = df["Timing"].str.extract(r'(\d+)').astype(float)
    except Exception:
        df["Timing_num"] = df["Timing"]

    metrics = ["CMC", "r2", "A1", "A2", "dx"]

    # 1. Variation across replicates
    for metric in metrics:
        plt.figure()
        for assay in sorted(df["Assay"].unique()):
            sub = df[df["Assay"] == assay]
            plt.plot(sub["Replicate"], sub[metric], 'o-', label=f'Assay {assay}')
        plt.title(f"{metric} variation across replicates ({surfactant_name})")
        plt.xlabel("Replicate")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fname = re.sub(r'[^\w\-_.]', '_', f"{surfactant_name}_{metric}_replicate_variation.png")
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

    # 2. Variation across assays
    for metric in metrics:
        grouped = df.groupby(["Timing", "Assay"])[metric].mean().unstack()
        fig, ax = plt.subplots()
        grouped.plot(marker='o', ax=ax)
        plt.title(f"{metric} variation across assays ({surfactant_name})")
        plt.xlabel("Timing")
        plt.ylabel(metric)
        plt.grid(True)
        plt.tight_layout()
        fname = re.sub(r'[^\w\-_.]', '_', f"{surfactant_name}_{metric}_assay_variation.png")
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

    # 3. Variation over time (mean ± std)
    for metric in metrics:
        grouped = df.groupby("Timing")[metric].agg(["mean", "std"]).reset_index()
        try:
            grouped["Timing_num"] = grouped["Timing"].str.extract(r'(\d+)').astype(float)
        except Exception:
            grouped["Timing_num"] = grouped["Timing"]
        grouped = grouped.sort_values("Timing_num")

        plt.figure()
        plt.errorbar(grouped["Timing_num"], grouped["mean"], yerr=grouped["std"], fmt='-o', capsize=4)
        plt.title(f"{metric} variation over time ({surfactant_name})")
        plt.xlabel("Time (minutes)")
        plt.ylabel(metric)
        plt.grid(True)
        plt.tight_layout()
        fname = re.sub(r'[^\w\-_.]', '_', f"{surfactant_name}_{metric}_time_variation.png")
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

    # --- Save variation data tables ---
    replicate_stats = df.groupby(["Timing", "Assay", "Replicate"])[metrics].mean().reset_index()
    assay_stats = df.groupby(["Timing", "Assay"])[metrics].agg(["mean", "std"]).reset_index()
    time_stats = df.groupby("Timing")[metrics].agg(["mean", "std"]).reset_index()

    # Flatten multi-index columns
    assay_stats.columns = ['_'.join(col).strip('_') for col in assay_stats.columns.values]
    time_stats.columns = ['_'.join(col).strip('_') for col in time_stats.columns.values]

    replicate_stats.to_csv(os.path.join(output_dir, f"{surfactant_name}_replicate_variation.csv"), index=False)
    assay_stats.to_csv(os.path.join(output_dir, f"{surfactant_name}_assay_variation.csv"), index=False)
    time_stats.to_csv(os.path.join(output_dir, f"{surfactant_name}_time_variation.csv"), index=False)



if not simulate:
    # Run visualization for each surfactant
    for surf in summary_df["Surfactant"].unique():
        plot_variation(summary_df, surf, main_folder)
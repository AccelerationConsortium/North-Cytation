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
TOTAL_DURATION_MINUTES = 180
REPEAT_INTERVAL_MINUTES = 30
SURFACTANTS_TO_RUN = ['SDS', 'NaDC', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'CAPB', 'CHAPS']
surf_labels = ['a', 'b', 'c', 'd']  # Must match REPEATS_PER_BATCH

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

        for rep in range(3):
            label = f"{surfactant}_Immediate_{repeat_label}_rep{rep+1}"
            if not simulate:
                results = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, range(starting_wp_index, starting_wp_index + samples_per_assay), plate_type="48 WELL PLATE")
                details = "_".join(f"{k}{int(v)}" for k, v in sub_stock_vols.items())
                metrics = analyze_and_save_results(main_folder, details, wellplate_data, results, analyzer, label)
            else:
                metrics = {"CMC": 0.01, "r2": 0.95, "A1": 1000, "A2": 1500, "dx": 0.05}
                print(f"[Simulate] Measuring wells (Immediate) for {surfactant} {repeat_label} rep {rep+1}")

            summary_records.append({
                "Surfactant": surfactant,
                "Assay": repeat_label,
                "Timing": "Immediate",
                "Replicate": rep + 1,
                **metrics
            })

        starting_wp_index += samples_per_assay

        if starting_wp_index >= 48:
            lash_e.discard_used_wellplate()
            lash_e.grab_new_wellplate()
            starting_wp_index = 0

    print(f"Initial sample preparation and immediate measurements complete for {surfactant}. Beginning timed measurements.")

    # Timed measurements
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=TOTAL_DURATION_MINUTES)
    batch_count = 0
    now = datetime.now()
    simulated_now = now

    while (not simulate and datetime.now() <= end_time) or (simulate and simulated_now <= end_time):
        current_time = datetime.now() if not simulate else simulated_now
        timestamp = current_time.strftime("%H%M")
        timing_label = f"{batch_count * REPEAT_INTERVAL_MINUTES} minutes"
        print(f"[{surfactant} - Batch {batch_count}] Starting at {timestamp}")

        wp_index = 0
        for repeat_index in range(REPEATS_PER_BATCH):
            repeat_label = surf_labels[repeat_index]

            for rep in range(3):
                label = f"{surfactant}_{timestamp}_{repeat_label}_rep{rep+1}"
                if not simulate:
                    results = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, range(wp_index, wp_index + samples_per_assay), plate_type="48 WELL PLATE")
                    details = "_".join(f"{k}{int(v)}" for k, v in sub_stock_vols.items())
                    metrics = analyze_and_save_results(main_folder, details, wellplate_data, results, analyzer, label)
                else:
                    metrics = {"CMC": 0.01, "r2": 0.95, "A1": 1000, "A2": 1500, "dx": 0.05}
                    print(f"[Simulate] Measuring wells for {surfactant} {repeat_label} rep {rep+1} at {timing_label}")

                summary_records.append({
                    "Surfactant": surfactant,
                    "Assay": repeat_label,
                    "Timing": timing_label,
                    "Replicate": rep + 1,
                    **metrics
                })

            wp_index += samples_per_assay

        batch_count += 1
        if not simulate and datetime.now() < end_time:
            print(f"Sleeping until next batch in {REPEAT_INTERVAL_MINUTES} minutes...")
            time.sleep(REPEAT_INTERVAL_MINUTES * 60)
        elif simulate:
            simulated_now += timedelta(minutes=REPEAT_INTERVAL_MINUTES)

    print(f"Timed repeat measurements complete for {surfactant}.")

# Save summary
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



# Run visualization for each surfactant
for surf in summary_df["Surfactant"].unique():
    plot_variation(summary_df, surf, main_folder)
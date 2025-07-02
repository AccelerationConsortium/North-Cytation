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

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/CMC_workflow_repeats_input.csv"
LOGGING_FOLDER = "../utoronto_demo/logs/"
MEASUREMENT_PROTOCOL_FILE = r"C:\\Protocols\\CMC_Fluorescence.prt"
simulate = True
enable_logging = True
repeats = 3  # Number of replicate measurements

REPEATS_PER_BATCH = 3
#SURFACTANTS_TO_RUN = ['SDS', 'NaDC', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'CAPB', 'CHAPS'] #Note that we might just do 4 at a time (each assay takes about 4 hours)
surf_labels = ['a', 'b', 'c']  # Must match REPEATS_PER_BATCH
SURFACTANTS_TO_RUN = ['SDS', 'NaDC', 'NaC']  
delay_minutes = [0, 5, 10, 20, 30]

# Setup
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
lash_e.nr_robot.check_input_file()
lash_e.nr_track.check_input_file()
timestamp_start = datetime.now().strftime("%Y%m%d_%H%M")
main_folder = f'C:/Users/Imaging Controller/Desktop/CMC/{timestamp_start}/'

if not simulate:
    os.makedirs(main_folder, exist_ok=True)
    print(f"[Main Folder] Created at: {main_folder}")

if enable_logging:
    log_file_path = os.path.join(LOGGING_FOLDER, f"experiment_log_{timestamp_start}_sim{simulate}.txt")
    log_file = open(log_file_path, "w")
    sys.stdout = sys.stderr = log_file

#lash_e.nr_robot.prime_reservoir_line(1, 'water', 0.5)
lash_e.grab_new_wellplate()

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

                details = "_".join(f"{k}{int(v)}" for k, v in sub_stock_vols.items())

                # Iterate over replicates inside the result
                for rep in range(repeats):
                    rep_label = f"rep{rep+1}"

                    if isinstance(results.columns, pd.MultiIndex):
                        rep_data = results[rep_label]
                    else:
                        # fallback if single rep or simulate edge case
                        rep_data = results

                    label = f"{label_prefix}_rep{rep+1}"
                    metrics = analyze_and_save_results(
                        main_folder, details, wellplate_data, rep_data, analyzer, label
                    )

                    summary_records.append({
                        "Surfactant": surfactant,
                        "Assay": repeat_label,
                        "Time_min": delay,
                        "Replicate": rep + 1,
                        **metrics
                    })
else:
    for rep in range(repeats):
        print(f"[Simulate] Measuring wells for {surfactant} {repeat_label} rep {rep+1} at {delay} min")
        metrics = {"CMC": 0.01, "r2": 0.95, "A1": 1000, "A2": 1500, "dx": 0.05}
        summary_records.append({
            "Surfactant": surfactant,
            "Assay": repeat_label,
            "Time_min": delay,
            "Replicate": rep + 1,
            **metrics
        })

    starting_wp_index += samples_per_assay

    plate_full = starting_wp_index >= 48
    last_surfactant = surfactant == SURFACTANTS_TO_RUN[-1]

    if plate_full:
        lash_e.discard_used_wellplate()
        if not (last_surfactant):
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

def plot_variation_clean(summary_df, surfactant_name, output_dir):
    df = summary_df[summary_df["Surfactant"] == surfactant_name].copy()
    
    # Extract numeric time
    #df["Time_min"] = df["Time_min"].str.extract(r'(\d+)').astype(float)
    
    metrics = ["CMC", "r2", "A1", "A2", "dx"]
    
    for metric in metrics:
        plt.figure()
        for assay in sorted(df["Assay"].unique()):
            sub = df[df["Assay"] == assay]
            
            # Scatter plot of replicates
            plt.scatter(sub["Time_min"], sub[metric], label=f"Trial {assay}", alpha=0.6)
            
            # Mean per time point
            mean_vals = sub.groupby("Time_min")[metric].mean()
            plt.plot(mean_vals.index, mean_vals.values, linestyle=":", marker='o', label=f"{assay} avg")

        plt.title(f"{metric} vs Time ({surfactant_name})")
        plt.xlabel("Time (minutes)")
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend(title="Trial")
        plt.tight_layout()

        fname = f"{surfactant_name}_{metric}_clean_variation.png".replace(" ", "_")
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()




if not simulate:
    # --- BACKWARD-COMPATIBLE TIME HANDLING ---
    if "Time_min" not in summary_df.columns and "Timing" in summary_df.columns:
        summary_df["Time_min"] = summary_df["Timing"].str.extract(r"(\d+)").astype(float)
    else:
        summary_df["Time_min"] = pd.to_numeric(summary_df["Time_min"], errors="coerce")

    summary_df["Assay"] = summary_df["Assay"].astype(str)

    # --- PER-SURFACTANT PLOTS (OPTIONAL) ---
    for surf in summary_df["Surfactant"].unique():
        df = summary_df[summary_df["Surfactant"] == surf].copy()
        metrics = ["CMC", "r2", "A1", "A2", "dx"]
        for metric in metrics:
            plt.figure()
            for assay in sorted(df["Assay"].unique()):
                sub = df[df["Assay"] == assay].sort_values("Time_min")
                plt.scatter(sub["Time_min"], sub[metric], alpha=0.6, label=f"Trial {assay}")
                mean_vals = sub.groupby("Time_min")[metric].mean()
                plt.plot(mean_vals.index, mean_vals.values, linestyle=":", marker='o', label=f"{assay} avg")
            plt.title(f"{metric} vs Time ({surf})")
            plt.xlabel("Time (minutes)")
            plt.ylabel(metric)
            plt.grid(True)
            plt.legend(title="Trial")
            plt.tight_layout()
            fname = f"{surf}_{metric}_clean_variation.png".replace(" ", "_")
            plt.savefig(os.path.join(main_folder, fname))
            plt.close()

    # --- REPLICATE / ASSAY / TIME VARIANCE ANALYSIS ---
    rep_avg = summary_df.groupby(["Time_min", "Assay"])["CMC"].mean().reset_index()

    replicate_std = summary_df.groupby(["Time_min", "Assay"])["CMC"].std().reset_index()
    replicate_std.rename(columns={"CMC": "replicate_std"}, inplace=True)

    assay_std = rep_avg.groupby("Time_min")["CMC"].std().reset_index()
    assay_std.rename(columns={"CMC": "assay_std"}, inplace=True)

    time_std = rep_avg.groupby("Assay")["CMC"].std().reset_index()
    time_std.rename(columns={"CMC": "time_std"}, inplace=True)

    replicate_std.to_csv(os.path.join(main_folder, "replicate_std.csv"), index=False)
    assay_std.to_csv(os.path.join(main_folder, "assay_std.csv"), index=False)
    time_std.to_csv(os.path.join(main_folder, "time_std.csv"), index=False)

    # --- PLOTS FOR VARIANCE COMPONENTS ---

    # Plot 1: Std Dev of Replicates vs Time (per Trial)
    plt.figure()
    for assay in replicate_std["Assay"].unique():
        sub = replicate_std[replicate_std["Assay"] == assay]
        plt.plot(sub["Time_min"], sub["replicate_std"], marker='o', label=f"Trial {assay}")
    plt.title("Std Dev of Replicates vs Time (per Trial)")
    plt.xlabel("Time (min)")
    plt.ylabel("Std Dev (CMC)")
    plt.legend(title="Trial")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(main_folder, "plot_replicate_std.png"))
    plt.close()

    # Plot 2: Std Dev Across Trials vs Time
    plt.figure()
    plt.plot(assay_std["Time_min"], assay_std["assay_std"], marker='o')
    plt.title("Std Dev Across Trials vs Time")
    plt.xlabel("Time (min)")
    plt.ylabel("Std Dev (CMC across trial averages)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(main_folder, "plot_assay_std.png"))
    plt.close()

    # Plot 3: Std Dev Across Timepoints per Trial
    plt.figure()
    plt.bar(time_std["Assay"], time_std["time_std"])
    plt.title("Std Dev Across Timepoints per Trial")
    plt.xlabel("Trial")
    plt.ylabel("Std Dev (CMC over time)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(main_folder, "plot_time_std.png"))
    plt.close()

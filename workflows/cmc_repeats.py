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

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/CMC_workflow_input.csv"
LOGGING_FOLDER = "../utoronto_demo/logs/"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\CMC_Fluorescence.prt"
simulate = True
enable_logging = True

REPEATS_PER_BATCH = 4
TOTAL_DURATION_MINUTES = 180
REPEAT_INTERVAL_MINUTES = 30

surf_labels = ['a', 'b', 'c', 'd']  # Must match REPEATS_PER_BATCH
SURFACTANT_NAME = 'SDS'  # Specify single surfactant for the study
surfactants = [SURFACTANT_NAME]
ratio_vector = [1]

# Set up
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
lash_e.nr_robot.check_input_file()

# Create main timestamped folder once
timestamp_start = datetime.now().strftime("%Y%m%d_%H%M")
main_folder = f'C:/Users/Imaging Controller/Desktop/CMC/{timestamp_start}/'

if not simulate:
    os.makedirs(main_folder, exist_ok=True)
    print(f"[Main Folder] Created at: {main_folder}")

if enable_logging:
    log_file_path = os.path.join(LOGGING_FOLDER, f"experiment_log_{timestamp_start}_sim{simulate}.txt")
    log_file = open(log_file_path, "w")
    sys.stdout = sys.stderr = log_file

starting_wp_index = 0

# Prepare all REPEATS_PER_BATCH samples
experiment, _ = experimental_planner.generate_exp_flexible(surfactants, ratio_vector, sub_stock_volume=6000, probe_volume=25)
sub_stock_vols = experiment['surfactant_sub_stock_vols']
wellplate_data = experiment['df']
samples_per_assay = wellplate_data.shape[0]

for repeat_index in range(REPEATS_PER_BATCH):
    substock_vial = f'substock_{repeat_index+1}'
    repeat_label = surf_labels[repeat_index]

    mix_surfactants(lash_e, sub_stock_vols, substock_vial)
    fill_water_vial(lash_e)
    create_wellplate_samples(lash_e, wellplate_data, substock_vial, starting_wp_index)

    starting_wp_index += samples_per_assay

    if starting_wp_index >= 48:
        lash_e.discard_used_wellplate()
        lash_e.grab_new_wellplate()
        starting_wp_index = 0

print("Initial sample preparation complete. Beginning timed measurements.")

# Begin timed measurement loop
start_time = datetime.now()
end_time = start_time + timedelta(minutes=TOTAL_DURATION_MINUTES)
batch_count = 0

# Setup real or simulated time tracking
now = datetime.now()
end_time = now + timedelta(minutes=TOTAL_DURATION_MINUTES)
simulated_now = now  # Track simulated time separately

while (not simulate and datetime.now() <= end_time) or (simulate and simulated_now <= end_time):
    timestamp = (datetime.now() if not simulate else simulated_now).strftime("%H%M")
    print(f"[Measurement Batch {batch_count}] Starting at {timestamp}")

    wp_index = 0
    for repeat_index in range(REPEATS_PER_BATCH):
        repeat_label = surf_labels[repeat_index]
        label = f"{timestamp}_{repeat_label}"

        if not simulate:
            results = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, range(wp_index, wp_index + samples_per_assay), "48 WELL PLATE")
            details = "_".join(f"{k}{int(v)}" for k, v in sub_stock_vols.items()) + f"_{label}"
            analyze_and_save_results(main_folder, details, wellplate_data, results, analyzer, label)
        else:
            # simulate a fake results object if needed
            results = pd.DataFrame({'1': [1]*samples_per_assay, '2': [1]*samples_per_assay})  # Dummy data
            print(f"[Simulate] Pretending to measure wells {list(range(wp_index, wp_index + samples_per_assay))}")

        wp_index += samples_per_assay

    batch_count += 1
    if not simulate and datetime.now() < end_time:
        print(f"Sleeping until next batch in {REPEAT_INTERVAL_MINUTES} minutes...")
        time.sleep(REPEAT_INTERVAL_MINUTES * 60)
    elif simulate:
        simulated_now += timedelta(minutes=REPEAT_INTERVAL_MINUTES)


print("Timed repeat measurement workflow complete.")

if enable_logging:
    log_file.close()
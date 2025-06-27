# --- cmc_workflow.py ---
import sys
sys.path.append("../utoronto_demo")
from cmc_shared import *
import analysis.cmc_exp_new as experimental_planner
import analysis.cmc_data_analysis as analyzer
from master_usdl_coordinator import Lash_E
from datetime import datetime
import os

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/CMC_workflow_input.csv"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\CMC_Fluorescence.prt"
LOGGING_FOLDER = "../utoronto_demo/logs/"
simulate = True
enable_logging = True

lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
lash_e.nr_robot.check_input_file()
lash_e.nr_track.check_input_file()

folder = None


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder = f'C:/Users/Imaging Controller/Desktop/CMC/{timestamp}/'

if not simulate:
    os.makedirs(folder, exist_ok=True)
    print(f"Folder created at: {folder}")

if enable_logging:
    log_file_path = os.path.join(LOGGING_FOLDER, f"experiment_log_{timestamp}_sim{simulate}.txt")
    log_file = open(log_file_path, "w")
    sys.stdout = sys.stderr = log_file

lash_e.nr_track.get_new_wellplate()
starting_wp_index = 0
lash_e.nr_robot.prime_reservoir_line(1, 'water', 0.5)

surfactants = ['SDS', 'NaDC', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'CAPB', 'CHAPS']
item_to_index = {item: i for i, item in enumerate(surfactants)}
substock_name_list = [f'substock_{i}' for i in range(1, len(surfactants) + 1)]

# One-hot encoded simple screening
ratios = [[1 if j == i else 0 for j in range(len(surfactants))] for i in range(len(surfactants))]

for i, ratio in enumerate(ratios):
    substock_vial = substock_name_list[i]
    experiment, _ = experimental_planner.generate_exp_flexible(surfactants, ratio, sub_stock_volume=6000, probe_volume=25)

    sub_stock_vols = experiment['surfactant_sub_stock_vols']
    wellplate_data = experiment['df']
    samples_per_assay = wellplate_data.shape[0]

    # Mix, dispense, measure, analyze
    mix_surfactants(lash_e, sub_stock_vols, substock_vial)
    fill_water_vial(lash_e)
    create_wellplate_samples(lash_e, wellplate_data, substock_vial, starting_wp_index)

    if not simulate:
        resulting_data = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, wells_to_measure=range(starting_wp_index, starting_wp_index + samples_per_assay), plate_type="48 WELL PLATE")
        details = "_".join(f"{k}{int(v)}" for k, v in sub_stock_vols.items())
        analyze_and_save_results(folder, details, wellplate_data, resulting_data, analyzer, save_modifier='')

    starting_wp_index += samples_per_assay

    if starting_wp_index >= 48:
        lash_e.discard_used_wellplate()
        lash_e.grab_new_wellplate()
        starting_wp_index = 0

if lash_e.nr_track.NR_OCCUPIED:
    lash_e.discard_used_wellplate()
    print("Workflow complete and wellplate discarded")

if enable_logging:
    log_file.close()
# --- cmc_pairings_workflow.py ---
import sys
sys.path.append("../utoronto_demo")
from cmc_shared import *
import analysis.cmc_exp_new as experimental_planner
import analysis.cmc_data_analysis as analyzer
from master_usdl_coordinator import Lash_E
from datetime import datetime
import os

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/CMC_workflow_input.csv"
LOGGING_FOLDER = "../utoronto_demo/logs/"
MEASUREMENT_PROTOCOL_FILE = [
    r"C:\Protocols\CMC_Fluorescence.prt",
    r"C:\Protocols\CMC_Absorbance.prt"
]
simulate = True
enable_logging = True

lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
lash_e.nr_robot.check_input_file()
lash_e.nr_track.check_input_file()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder = f'C:/Users/Imaging Controller/Desktop/CMC/{timestamp}/'

if not simulate:
    os.makedirs(folder, exist_ok=True)
    print(f"Folder created at: {folder}")
else:
    folder = None

if enable_logging:
    log_file_path = os.path.join(LOGGING_FOLDER, f"experiment_log_{timestamp}_sim{simulate}.txt")
    log_file = open(log_file_path, "w")
    sys.stdout = sys.stderr = log_file

lash_e.nr_track.get_new_wellplate()
starting_wp_index = 0
lash_e.nr_robot.prime_reservoir_line(1, 'water', 0.5)

surfactants = ['SDS', 'NaDC', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'CAPB', 'CHAPS']
item_to_index = {item: i for i, item in enumerate(surfactants)}
n_substocks = len(surfactants) * 2
substock_name_list = [f'substock_{i}' for i in range(1, n_substocks + 1)]

pairings_and_ratios = [
    (['SDS', 'CAPB'], [0.5, 0.5]),
    (['CTAB', 'TTAB'], [0.5, 0.5]),
    (['CAPB', 'CHAPS'], [0.5, 0.5]),
]

padded_ratios = []
pairing_labels = []
for pair, ratios in pairings_and_ratios:
    vector = [0.0] * len(surfactants)
    for surf, ratio in zip(pair, ratios):
        vector[item_to_index[surf]] = ratio
    padded_ratios.append(vector)
    label = "_".join(f"{s}-{r:.1f}" for s, r in zip(pair, ratios))
    pairing_labels.append(label)

substock_count = 0
for i, ratio in enumerate(padded_ratios):
    rough_vial = substock_name_list[substock_count]
    fine_vial = substock_name_list[substock_count + 1]
    substock_count += 2
    label = pairing_labels[i]

    # Rough search
    rough_exp, _ = experimental_planner.generate_exp_flexible(surfactants, ratio, rough_screen=True)
    sub_stock_vols = rough_exp['surfactant_sub_stock_vols']
    wellplate_data = rough_exp['df']
    samples_per_assay = wellplate_data.shape[0]

    mix_surfactants(lash_e, sub_stock_vols, rough_vial)
    fill_water_vial(lash_e)
    create_wellplate_samples(lash_e, wellplate_data, rough_vial, starting_wp_index)

    if not simulate:
        results = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, range(starting_wp_index, starting_wp_index + samples_per_assay), plate_type="48 WELL PLATE")
        cmc_rough = analyze_and_save_results(folder, label + '_rough', wellplate_data, results, analyzer, 'rough')
        print("Rough CMC:", cmc_rough)
    else:
        cmc_rough = None

    starting_wp_index += samples_per_assay

    # Fine search
    fine_exp, _ = experimental_planner.generate_exp_flexible(surfactants, ratio, rough_screen=False, estimated_CMC=cmc_rough)
    sub_stock_vols = fine_exp['surfactant_sub_stock_vols']
    wellplate_data = fine_exp['df']
    samples_per_assay = wellplate_data.shape[0]

    mix_surfactants(lash_e, sub_stock_vols, fine_vial)
    fill_water_vial(lash_e)
    create_wellplate_samples(lash_e, wellplate_data, fine_vial, starting_wp_index)

    if not simulate:
        results = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, range(starting_wp_index, starting_wp_index + samples_per_assay), plate_type="48 WELL PLATE")
        cmc_fine = analyze_and_save_results(folder, label + '_fine', wellplate_data, results, analyzer, 'fine')
        print("Refined CMC:", cmc_fine)

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
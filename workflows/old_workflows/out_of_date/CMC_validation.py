# --- cmc_pairings_workflow.py ---
from gc import enable
import sys
sys.path.append("../utoronto_demo")
from cmc_shared import *
import analysis.cmc_exp_new as experimental_planner
import analysis.cmc_data_analysis as analyzer
from master_usdl_coordinator import Lash_E
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb
import seaborn as sns
import time


LOGGING_FOLDER = "../utoronto_demo/logs/"
MEASUREMENT_PROTOCOL_FILE = [r"C:\Protocols\CMC_Fluorescence.prt",r"C:\Protocols\CMC_Absorbance.prt"] #Cytation measurement files
simulate = enable_logging = False
INPUT_VIAL_STATUS_FILE = f"../utoronto_demo/status/CMC_validation_vials.csv"

# Load pairing data from CSV
data_in = pd.read_csv("../utoronto_demo/analysis/CMC_validation_inputs.csv")  

# Extract pairings and ratios for this trial
pairings_and_ratios = []
for _, row in data_in.iterrows():
    pairings_and_ratios.append(
        ([row['Surfactant1'], row['Surfactant2']], [row['Ratio1'], row['Ratio2']], row['CMC_guess'])
    )

# Show user what’s being run and pause
print(f"Pairings and Ratios:")
for (sfs, ratios) in pairings_and_ratios:
    print(f"  {sfs[0]}:{ratios[0]} + {sfs[1]}:{ratios[1]}")

#Setup
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
lash_e.nr_robot.check_input_file()
lash_e.nr_track.check_input_file()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#Monitor surfactants used
surfactants_used = set()
for pair, _ in pairings_and_ratios:
    surfactants_used.update(pair)
surfactants_used.update(["water_large", "pyrene_DMSO"])
initial_volumes = {}
for surf in surfactants_used:
    try:
        initial_volumes[surf] = lash_e.nr_robot.get_vial_info(surf, 'vial_volume')
    except Exception as e:
        print(f"Warning: couldn't get initial volume for {surf}: {e}")

#Create folders for data storage
if not simulate:
    folder = f'C:/Users/Imaging Controller/Desktop/CMC/{timestamp}/'
    os.makedirs(folder, exist_ok=True)
    print(f"Folder created at: {folder}")
else:
    folder = None

#Log data
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
summary_data = []
if not simulate:
    summary_file = os.path.join(folder, f"CMC_summary_{timestamp}.csv")

summary_columns = [
    "Surfactant_1", "Surfactant_2", "conc_1", "conc_2", "cmc_target",
    "cmc_measured", "r2_fine", "A1_fine", "A2_fine", "dx_fine",
    "replicates"
]

if not simulate:
    with open(summary_file, "w") as f:
        f.write(",".join(summary_columns) + "\n")


for i, ratio in enumerate(padded_ratios):
    substock_vial = substock_name_list[i]
    label = pairing_labels[i]
    pair, ratios, cmc_guess = pairings_and_ratios[i]

    # Fine search
    fine_exp, _ = experimental_planner.generate_exp_flexible(surfactants, ratio, rough_screen=False, estimated_CMC=cmc_guess)
    sub_stock_vols = fine_exp['surfactant_sub_stock_vols']
    wellplate_data = fine_exp['df']
    samples_per_assay = wellplate_data.shape[0]

    mix_surfactants(lash_e, sub_stock_vols, substock_vial)
    fill_water_vial(lash_e)
    create_wellplate_samples(lash_e, wellplate_data, substock_vial, starting_wp_index)

    if not simulate:
        time.sleep(600)
        results = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, range(starting_wp_index, starting_wp_index + samples_per_assay), plate_type="48 WELL PLATE", repeats=3)
        results_concat = merge_absorbance_and_fluorescence(coalesce_replicates_long(results))
        metrics_fine = analyze_and_save_results(folder, label + '_fine', wellplate_data, results_concat, analyzer, 'fine')
        cmc_fine = metrics_fine["CMC"]
        print("Refined CMC:", cmc_fine)

    starting_wp_index += samples_per_assay

    if not simulate:
        with open(summary_file, "a") as f:
            f.write(",".join(str(x) for x in [
                pair[0], pair[1],
                ratios[0], ratios[1],
                cmc_guess,
                metrics_fine.get("CMC"),
                metrics_fine.get("r2"),
                metrics_fine.get("A1"),
                metrics_fine.get("A2"),
                metrics_fine.get("dx"),
                3  # replicates, make configurable if needed
            ]) + "\n")
        print(f"Saved summary for {pair[0]} + {pair[1]}")

    if starting_wp_index >= 48 and i < len(padded_ratios) - 1:
        lash_e.discard_used_wellplate()
        lash_e.grab_new_wellplate()
        starting_wp_index = 0

if lash_e.nr_track.NR_OCCUPIED:
    lash_e.discard_used_wellplate()
    print("Workflow complete and wellplate discarded")

print("\n--- Volume Usage Summary ---")
for surf in surfactants_used:
    try:
        final_vol = lash_e.nr_robot.get_vial_info(surf, 'vial_volume')
        initial_vol = initial_volumes.get(surf, final_vol)
        used_vol = initial_vol - final_vol
        print(f"{surf}: used {used_vol:.2f} mL (from {initial_vol:.2f} -> {final_vol:.2f})")
    except Exception as e:
        print(f"{surf}: Error retrieving final volume — {e}")

if enable_logging:
    log_file.close()
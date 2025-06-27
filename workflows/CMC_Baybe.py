# --- cmc_baybe_workflow.py ---

import sys
sys.path.append("../utoronto_demo")
import os
import pandas as pd
import random
from datetime import datetime
import pickle

from cmc_shared import *
import analysis.cmc_exp_new as experimental_planner
import analysis.cmc_data_analysis as analyzer
from master_usdl_coordinator import Lash_E

from baybe.targets import NumericalTarget, TargetMode
from baybe.objectives import SingleTargetObjective
from baybe import Campaign
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.constraints import ContinuousCardinalityConstraint
from baybe.utils.random import set_random_seed

# --- Config ---
simulate = True
LOGGING_FOLDER = "../utoronto_demo/logs/"
n_trials = 3
initial_batch_size = 0
random_seed = 42
CMC_target = 5.0 #What CMC are we trying to achieve?
CMC_tolerance = 0.5
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/CMC_workflow_input.csv"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\CMC_Fluorescence.prt"
surfactants = ['SDS', 'NaDC', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'CAPB', 'CHAPS']
enable_logging = True

# --- Setup output folder and logging ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
if not simulate:
    main_folder = f"C:/Users/Imaging Controller/Desktop/CMC_BAYBE/{timestamp}/"
    os.makedirs(main_folder, exist_ok=True)

# --- Initialize BayBE ---
set_random_seed(random_seed)
target = NumericalTarget(name='CMC_difference', mode=TargetMode.MIN, bounds=(0, CMC_tolerance))
objective = SingleTargetObjective(target=target)

parameters = [
    NumericalContinuousParameter(name=s, bounds=(0.1, 20)) for s in surfactants
]
constraints = [
    ContinuousCardinalityConstraint(parameters=[p.name for p in parameters], min_cardinality=2, max_cardinality=2)
]
searchspace = SearchSpace.from_product(parameters, constraints)
campaign = Campaign(searchspace, objective)

# --- Lab setup ---
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
lash_e.nr_robot.check_input_file()
lash_e.nr_track.check_input_file()
lash_e.nr_track.get_new_wellplate()

if enable_logging:
    log_file_path = os.path.join(LOGGING_FOLDER, f"experiment_log_{timestamp}_sim{simulate}.txt")
    log_file = open(log_file_path, "w")
    sys.stdout = sys.stderr = log_file

# --- Experiment loop ---
starting_wp_index = 0
trial_index = 0
all_results = []

df = pd.read_csv("my_previous_data.csv")
campaign.add_measurements(df)

# Get initial recommendations
recommendations = campaign.recommend(batch_size=initial_batch_size)

while trial_index < n_trials:
    for _, rec in recommendations.iterrows():
        mixture = rec[surfactants].to_dict()
        total = sum(mixture.values())
        if total == 0:
            print(f"Error! Skipping trial {trial_index}: zero total concentration")
            continue

        ratios = [mixture[s] / total for s in surfactants]

        selected = {k: v for k, v in mixture.items() if v > 0}
        print(f"\n[Trial {trial_index + 1}] Recommended mixture:")
        for surf, conc in selected.items():
            print(f"  - {surf}: {conc:.3f} mM")

        print("[Normalized ratios:]")
        for s, r in zip(surfactants, ratios):
            if r > 0:
                print(f"  - {s}: {r:.3f}")

        experiment, _ = experimental_planner.generate_exp_flexible(surfactants, ratios, sub_stock_volume=6000, probe_volume=25)
        sub_stock_vols = experiment['surfactant_sub_stock_vols']
        wellplate_data = experiment['df']
        samples_per_assay = wellplate_data.shape[0]

        substock_vial = f"substock_{trial_index+1}"
        label = f"trial_{trial_index}"
        details = "_".join(f"{k}{int(v)}" for k, v in sub_stock_vols.items())

        # Execute workflow
        mix_surfactants(lash_e, sub_stock_vols, substock_vial)
        fill_water_vial(lash_e)
        create_wellplate_samples(lash_e, wellplate_data, substock_vial, starting_wp_index)

        if not simulate:
            results = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, range(starting_wp_index, starting_wp_index + samples_per_assay), "48 WELL PLATE")
            concentrations = wellplate_data['concentration']
            fig_name = main_folder + f"CMC_plot_{label}.png"
            x0 = analyze_and_save_results(main_folder, details, wellplate_data, results, analyzer, label=label)

            results.to_csv(main_folder + f"output_data_{label}.csv", index=False)
            wellplate_data.to_csv(main_folder + f"wellplate_data_{label}.csv", index=False)
        else:
            print(f"[Simulate] Generating dummy CMC for trial {trial_index}")
            x0 = CMC_target + random.uniform(-5, 5)  # or any desired simulation logic

        # Save and report
        df_result = pd.DataFrame([{**rec[surfactants].to_dict(), 'CMC_difference': (x0 - CMC_target) / CMC_target, 'CMC_estimate': x0}])
        campaign.add_measurements(df_result)
        all_results.append(df_result)

        print(f" Completed trial {trial_index + 1}/{n_trials}")

        starting_wp_index += samples_per_assay
        trial_index += 1

        if starting_wp_index >= 48:
            lash_e.discard_used_wellplate()
            lash_e.grab_new_wellplate()
            starting_wp_index = 0

        if trial_index >= n_trials:
            break

    if trial_index < n_trials:
        recommendations = campaign.recommend(batch_size=initial_batch_size)

# --- Save summary and campaign ---
if not simulate:
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(main_folder + "summary_all_trials.csv", index=False)

    with open(main_folder + "baybe_campaign.pkl", "wb") as f:
        pickle.dump(campaign, f)

print("Bayesian optimization workflow complete.")

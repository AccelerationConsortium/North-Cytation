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

from baybe import Campaign
from baybe.targets import NumericalTarget, TargetMode
from baybe.objectives import SingleTargetObjective
from baybe.parameters import CustomDiscreteParameter, NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.utils.random import set_random_seed

# --- Config ---
simulate = True
LOGGING_FOLDER = "../utoronto_demo/logs/"
n_trials = 3
initial_batch_size = 12
random_seed = 42
CMC_target = 5.0  # Desired CMC
CMC_tolerance = 0.5
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/CMC_workflow_input.csv"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\CMC_Fluorescence.prt"
enable_logging = True

set_random_seed(random_seed)

# --- Surfactant descriptor table ---
surfactant_library = pd.DataFrame({
    "CMC": [8.5, 5.3375, 14, 1.07, 15.85, 3.985, 0.627, 8],
    "MW": [289.39, 445.57, 431.56, 364.45, 308.34, 336.39, 342.52, 614.88],
    "Category": ["anionic", "anionic", "anionic", "cationic", "cationic", "cationic", "zwitterionic", "zwitterionic"]
}, index=["SDS", "NaDC", "NaC", "CTAB", "DTAB", "TTAB", "CAPB", "CHAPS"])

surf_desc = pd.get_dummies(surfactant_library, columns=["Category"])
surf_desc["name"] = surf_desc.index  # Move index to column
surf_desc = surf_desc.set_index("name")  # Set the correct label index

# --- Define BayBE parameters ---
surf_1 = CustomDiscreteParameter(name="surf_1", data=surf_desc)
surf_2 = CustomDiscreteParameter(name="surf_2", data=surf_desc)

conc_1 = NumericalContinuousParameter(name="conc_1", bounds=(0.1, 20))
conc_2 = NumericalContinuousParameter(name="conc_2", bounds=(0.1, 20))

# --- Search space (no constraints supported in 0.11.2) ---
searchspace = SearchSpace(parameters=[surf_1, surf_2, conc_1, conc_2])

# --- Objective ---
target = NumericalTarget(name="CMC_difference", mode=TargetMode.MIN, bounds=(0, CMC_tolerance))
objective = SingleTargetObjective(target=target)

# --- Campaign ---
campaign = Campaign(searchspace=searchspace, objective=objective)

# --- Lab setup ---
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
lash_e.nr_robot.check_input_file()
lash_e.nr_track.check_input_file()
lash_e.nr_track.get_new_wellplate()

# --- Logging ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
if enable_logging:
    log_file_path = os.path.join(LOGGING_FOLDER, f"experiment_log_{timestamp}_sim{simulate}.txt")
    log_file = open(log_file_path, "w")
    sys.stdout = sys.stderr = log_file

# --- Load prior data ---
if os.path.exists("my_previous_data.csv"):
    df = pd.read_csv("my_previous_data.csv")
    campaign.add_measurements(df)

# --- Main loop ---
starting_wp_index = 0
trial_index = 0
all_results = []

recommendations = campaign.recommend(batch_size=initial_batch_size)

while trial_index < n_trials:
    for _, rec in recommendations.iterrows():
        s1 = rec["surf_1"]
        s2 = rec["surf_2"]
        c1 = rec["conc_1"]
        c2 = rec["conc_2"]

        print(f"\n[Trial {trial_index + 1}] {s1} ({c1:.2f} mM) + {s2} ({c2:.2f} mM)")

        # Normalize and prep experiment
        total = c1 + c2
        ratios = {s1: c1 / total, s2: c2 / total}

        experiment, _ = experimental_planner.generate_exp_flexible(
            list(ratios.keys()), list(ratios.values()),
            sub_stock_volume=6000, probe_volume=25
        )
        sub_stock_vols = experiment['surfactant_sub_stock_vols']
        wellplate_data = experiment['df']
        samples_per_assay = wellplate_data.shape[0]

        substock_vial = f"substock_{trial_index+1}"
        label = f"trial_{trial_index}"

        mix_surfactants(lash_e, sub_stock_vols, substock_vial)
        fill_water_vial(lash_e)
        create_wellplate_samples(lash_e, wellplate_data, substock_vial, starting_wp_index)

        if not simulate:
            results = lash_e.measure_wellplate(
                MEASUREMENT_PROTOCOL_FILE,
                range(starting_wp_index, starting_wp_index + samples_per_assay),
                "48 WELL PLATE"
            )
            x0 = analyze_and_save_results(".", label, wellplate_data, results, analyzer, label=label)
        else:
            x0 = CMC_target + random.uniform(-5, 5)

        df_result = pd.DataFrame([{
            "surf_1": s1, "surf_2": s2,
            "conc_1": c1, "conc_2": c2,
            "CMC_difference": abs(x0 - CMC_target),
            "CMC_estimate": x0
        }])
        campaign.add_measurements(df_result)
        all_results.append(df_result)

        print(f"Completed trial {trial_index + 1}/{n_trials}")

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

# --- Save results ---
if not simulate:
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(f"summary_{timestamp}.csv", index=False)

    with open(f"baybe_campaign_{timestamp}.pkl", "wb") as f:
        pickle.dump(campaign, f)

print("Bayesian optimization workflow complete.")

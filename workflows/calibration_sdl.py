# calibration_sdl.py

import sys
sys.path.append("../utoronto_demo")
import numpy as np
import pandas as pd
from datetime import datetime
from master_usdl_coordinator import Lash_E
import time
from ax.core.observation import ObservationFeatures
import recommenders.pipeting_optimizer_honegumi as recommender  # adjust if needed
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Config ---
SEED = 5
SOBOL_CYCLES_PER_VOLUME = 2
BAYES_CYCLES_PER_VOLUME = 1
SIMULATE = True
REPLICATES = 5
VOLUMES = [0.01,0.02,0.05,0.1]
#VOLUMES = [0.01,0.02,0.05]
DENSITY_LIQUID = 0.997  # g/mL
EXPECTED_MASSES = [v * DENSITY_LIQUID for v in VOLUMES]
EXPECTED_TIME = [v * 10.146 + 9.5813 for v in VOLUMES]
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/calibration_vials.csv"

def pipet_and_measure_simulated(volume, params, expected_mass, expected_time):
    # Simulate a short delay to mimic a fast experiment
    time.sleep(0.2)

    # Simulated objective: deviation increases slightly if aspirate_speed is high
    deviation = np.abs(params["aspirate_speed"] - 15) + np.random.normal(0, 0.5)

    # Simulated variability: slightly better at moderate wait times
    variability = np.abs(params["wait_time"] - 30) * 0.1 + np.random.normal(0, 0.3)

    # Simulated time score: longer wait_time, longer time per replicate
    time_score = (params["wait_time"] / expected_time - 1) * 100 + np.random.normal(0, 5)

    results = {
        "deviation": deviation,
        "variability": variability,
        "time": time_score,
    }

    print (results)

    return results

def empty_vial_if_needed(vial_name, waste_vial_name):
    volume = lash_e.nr_robot.get_vial_volume(vial_name)
    if volume > 7.0:
        disp_volume = volume / np.ceil(volume)
        for i in range (int(np.ceil(volume))-1):
            lash_e.nr_robot.dispense_from_vial_into_vial(vial_name, waste_vial_name, disp_volume)

def pipet_and_measure(volume, params, expected_mass, expected_time):
    pre_air = params.get("pre_asp_air_vol", 0)
    post_air = params.get("post_asp_air_vol", 0)
    disp_vol = volume
    air_vol = pre_air + post_air

    aspirate_kwargs = {
        "aspirate_speed": params["aspirate_speed"],
        "wait_time": params["wait_time"],
        "retract_speed": params["retract_speed"],
        "pre_asp_air_vol": pre_air,
        "post_asp_air_vol": post_air,
    }

    dispense_kwargs = {
        "dispense_speed": params["aspirate_speed"],
        "wait_time": params["wait_time"],
        "blowout_vol": params["blowout_vol"],
        "measure_weight": True,
        "air_vol": air_vol,
    }

    masses = []
    start = datetime.now().timestamp()
    for _ in range(REPLICATES):
        lash_e.nr_robot.aspirate_from_vial("liquid_source", volume, **aspirate_kwargs)
        mass = lash_e.nr_robot.dispense_into_vial("measurement_vial", disp_vol, **dispense_kwargs)
        masses.append(mass)
    end = datetime.now().timestamp()
    empty_vial_if_needed("measurement_vial", "waste_vial")

    if not SIMULATE:
        mean_mass = np.mean(masses)
        std_mass = np.std(masses)
        deviation = (mean_mass - expected_mass) / expected_mass * 100
        time_score = ((end - start) / REPLICATES - expected_time) / expected_time * 100

        return {
            "deviation": deviation,
            "variability": std_mass,
            "time"
            : time_score,
        }
    else: return pipet_and_measure_simulated(volume, params, expected_mass, expected_time)


lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
lash_e.nr_robot.check_input_file()
lash_e.nr_robot.move_vial_to_location("measurement_vial", "clamp", 0)

# Run optimization
ax_client = recommender.create_model(SEED, SOBOL_CYCLES_PER_VOLUME*len(VOLUMES),VOLUMES)
all_results = []

# Prime each volume with at least 2 Sobol trials
for i,volume in enumerate(VOLUMES):
    expected_mass = EXPECTED_MASSES[i]
    expected_time = EXPECTED_TIME[i]
    for _ in range(SOBOL_CYCLES_PER_VOLUME):
        print(f"[INFO] Requesting SOBOL trial for volume {volume}...")
        params, trial_index = ax_client.get_next_trial(fixed_features=ObservationFeatures({"volume": volume}))
        print(f"[TRIAL {trial_index}] Parameters: {params}")
        result = pipet_and_measure(volume, params, expected_mass, expected_time)
        ax_client.complete_trial(trial_index=trial_index, raw_data=result)
        result.update(params)
        result["volume"] = volume
        result["trial_index"] = trial_index
        all_results.append(result)

# Continue with main optimization loop
for i, volume in enumerate(VOLUMES):
    expected_mass = EXPECTED_MASSES[i]
    expected_time = EXPECTED_TIME[i]

    for _ in range(BAYES_CYCLES_PER_VOLUME):
        print(f"[INFO] Requesting BAYESIAN trial for volume {volume}...")
        suggestions = recommender.get_suggestions(ax_client, volume, n=1)
        for params, trial_index in suggestions:
            print(f"[TRIAL {trial_index}] Parameters: {params}")
            results = pipet_and_measure(volume, params, expected_mass, expected_time)
            recommender.add_result(ax_client, trial_index, results)
            results.update(params)
            results["volume"] = volume
            results["trial_index"] = trial_index
            all_results.append(results)

# Save results
# Clean up any (value, None) tuples in results
for res in all_results:
    for key in ["deviation", "variability", "time"]:
        val = res[key]
        if isinstance(val, tuple):
            res[key] = val[0]  # Extract the numeric part
results_df = pd.DataFrame(all_results)

lash_e.nr_robot.remove_pipet()
lash_e.nr_robot.return_vial_home('measurement_vial')

print(results_df)

if not SIMULATE:
    # --- Analysis: Parameter vs Outcome Plots ---
    outcomes = ["deviation", "variability", "time"]
    param_cols = [
        "aspirate_speed",
        "wait_time",
        "retract_speed",
        "pre_asp_air_vol",
        "post_asp_air_vol",
        "blowout_vol",
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("output", f"experiment_calibration_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    results_df.to_csv(os.path.join(save_dir, "experiment_summary.csv"), index=False)
    print("Saved results to:", os.path.join(save_dir, "experiment_summary.csv"))

    for outcome in outcomes:
        for param in param_cols:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=results_df, x=param, y=outcome, hue="volume", palette="viridis")
            plt.title(f"{outcome} vs {param}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"param_effect_{param}_on_{outcome}.png"))
            plt.close()

    print("Saved parameter vs outcome plots.")
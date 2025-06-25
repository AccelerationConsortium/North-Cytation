# calibration_sdl.py

import sys

from torch import fill
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
SEED = 7
SOBOL_CYCLES_PER_VOLUME = 5
BAYES_CYCLES_PER_VOLUME = 27
SIMULATE = False
REPLICATES = 3
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
    variability = np.abs(params["aspirate_wait_time"] - 30) * 0.1 + np.random.normal(0, 0.3)

    # Simulated time score: longer wait_time, longer time per replicate
    time_score = (params["aspirate_wait_time"] / expected_time - 1) * 100 + np.random.normal(0, 5)

    results = {
        "deviation": deviation,
        "variability": variability,
        "time": time_score,
    }

    print (results)

    return results

def empty_vial_if_needed(vial_name, waste_vial_name):
    volume = lash_e.nr_robot.get_vial_info(vial_name, 'vial_volume')
    if volume > 7.0:
        lash_e.nr_robot.remove_pipet()
        disp_volume = volume / np.ceil(volume)
        for i in range (int(np.ceil(volume))-1):
            lash_e.nr_robot.dispense_from_vial_into_vial(vial_name, waste_vial_name, disp_volume)
        lash_e.nr_robot.remove_pipet()

def fill_liquid_if_needed(vial_name, liquid_source_name):
    volume = lash_e.nr_robot.get_vial_info(liquid_source_name, 'vial_volume')
    if volume < 4.0:
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.return_vial_home(vial_name)
        lash_e.nr_robot.dispense_into_vial_from_reservoir(1,liquid_source_name, 8 - volume)
        lash_e.nr_robot.return_vial_home(liquid_source_name)
        lash_e.nr_robot.move_vial_to_location(vial_name, "clamp", 0)

def pipet_and_measure(volume, params, expected_mass, expected_time):
    pre_air = params.get("pre_asp_air_vol", 0)
    post_air = params.get("post_asp_air_vol", 0)
    disp_vol = volume
    air_vol = pre_air + post_air

    aspirate_kwargs = {
        "aspirate_speed": params["aspirate_speed"],
        "wait_time": params["aspirate_wait_time"],
        "retract_speed": params["retract_speed"],
        "pre_asp_air_vol": pre_air,
        "post_asp_air_vol": post_air,
    }

    dispense_kwargs = {
        "dispense_speed": params["dispense_speed"],
        "wait_time": params["dispense_wait_time"],
        "blowout_vol": params["blowout_vol"],
        "measure_weight": True,
        "air_vol": air_vol,
    }

    masses = []
    start = time.time()
    for replicate_idx in range(REPLICATES):
        replicate_start = datetime.now().isoformat()
        lash_e.nr_robot.aspirate_from_vial("liquid_source", volume, **aspirate_kwargs)
        mass = lash_e.nr_robot.dispense_into_vial("measurement_vial", disp_vol, **dispense_kwargs)
        replicate_end = datetime.now().isoformat()
        print("Mass measured: ", mass)
        masses.append(mass)

        raw_entry = {
            "volume": volume,
            "replicate": replicate_idx,
            "mass": mass,
            "start_time": replicate_start,
            "end_time": replicate_end,
            **params,
        }
        raw_measurements.append(raw_entry)
    end = time.time()

    if not SIMULATE:
        mean_mass = np.mean(masses)
        print("Average mass: ", mean_mass)
        print("Average time per replicat (s): ", (end - start) / REPLICATES)
        std_mass = np.std(masses) * 1000
        deviation = np.abs((mean_mass - expected_mass) / expected_mass * 100)
        time_score = ((end - start) / REPLICATES - expected_time) / expected_time * 100

        return {
            "deviation": deviation,
            "variability": std_mass,
            "time"
            : time_score,
        }
    else: return pipet_and_measure_simulated(volume, params, expected_mass, expected_time)


lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False)
lash_e.nr_robot.check_input_file()
lash_e.nr_robot.move_vial_to_location("measurement_vial", "clamp", 0)

# Run optimization
ax_client = recommender.create_model(SEED, SOBOL_CYCLES_PER_VOLUME*len(VOLUMES),VOLUMES)
all_results = []
raw_measurements = [] 

# Prime each volume with at least 2 Sobol trials
for i,volume in enumerate(VOLUMES):
    expected_mass = EXPECTED_MASSES[i]
    expected_time = EXPECTED_TIME[i]
    for _ in range(SOBOL_CYCLES_PER_VOLUME):
        print(f"[INFO] Requesting SOBOL trial for volume {volume}...")
        params, trial_index = ax_client.get_next_trial(fixed_features=ObservationFeatures({"volume": volume}))
        print(f"[TRIAL {trial_index}] Parameters: {params}")
        fill_liquid_if_needed("measurement_vial","liquid_source")
        empty_vial_if_needed("measurement_vial", "waste_vial")
        result = pipet_and_measure(volume, params, expected_mass, expected_time)
        ax_client.complete_trial(trial_index=trial_index, raw_data=result)
        result.update(params)
        result["volume"] = volume
        result["trial_index"] = trial_index
        result["strategy"] = "SOBOL"
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
            fill_liquid_if_needed("measurement_vial","liquid_source")
            empty_vial_if_needed("measurement_vial", "waste_vial")
            results = pipet_and_measure(volume, params, expected_mass, expected_time)
            recommender.add_result(ax_client, trial_index, results)
            results.update(params)
            results["volume"] = volume
            results["trial_index"] = trial_index
            results["strategy"] = "BAYESIAN"
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
lash_e.nr_robot.move_home()

print(results_df)

if not SIMULATE:
    # --- Analysis: Parameter vs Outcome Plots ---
    outcomes = ["deviation", "variability", "time"]
    param_cols = [
        "aspirate_speed",
        "dispense_wait_time",
        "aspirate_wait_time",
        "dispense_speed",
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

    # --- Retrieve and display best parameters per volume ---
print("\nBest parameters per volume:")

# --- Save Best Parameters Per Volume ---
best_results = []

for volume in VOLUMES:
    try:
        best_parameters, values = ax_client.get_best_parameters(
            observation_features=ObservationFeatures({"volume": volume})
        )

        result_entry = {
            "volume": volume,
            **best_parameters,
            **{f"{k}_objective": v for k, v in values.items()},
        }
        best_results.append(result_entry)

        print(f"\n📏 Best for volume {volume} mL")
        for k, v in best_parameters.items():
            print(f"  {k}: {v}")
        for obj, val in values.items():
            print(f"  {obj}: {val}")

    except Exception as e:
        print(f"❌ Could not get best parameters for volume {volume}: {e}")

if best_results:
    best_df = pd.DataFrame(best_results)
    best_csv_path = os.path.join(save_dir, "best_parameters_by_volume.csv")
    best_txt_path = os.path.join(save_dir, "best_parameters_by_volume.txt")
    best_df.to_csv(best_csv_path, index=False)

    with open(best_txt_path, "w") as f:
        for row in best_results:
            f.write(f"Volume: {row['volume']} mL\n")
            for k, v in row.items():
                if k != "volume":
                    f.write(f"  {k}: {v}\n")
            f.write("\n")

    print(f"\n✅ Best parameter results saved to:\n - {best_csv_path}\n - {best_txt_path}")

# --- Save raw measurements ---
raw_df = pd.DataFrame(raw_measurements)
raw_csv_path = os.path.join(save_dir, "raw_replicate_data.csv")
raw_df.to_csv(raw_csv_path, index=False)
print("Saved raw replicate data to:", raw_csv_path)
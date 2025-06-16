import sys
sys.path.append("../utoronto_demo")
import numpy as np
import pandas as pd
from datetime import datetime
from master_usdl_coordinator import Lash_E
import time
from ax.core.observation import ObservationFeatures
import recommenders.pipeting_optimizer_honegumi as recommender  # adjust if needed

# --- Config ---
SEED = 5
NUM_INITIAL_RECS = 4
NUM_SUGGESTIONS_PER_CYCLE = 2
NUM_CYCLES = 3
SIMULATE = True
REPLICATES = 5
VOLUMES = [0.005, 0.01, 0.02, 0.05]
DENSITY_LIQUID = 0.997  # g/mL
EXPECTED_MASSES = [v * DENSITY_LIQUID * 1000 for v in VOLUMES]
EXPECTED_TIME = [4] * len(VOLUMES)
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

    return {
        "deviation": deviation,
        "variability": variability,
        "time": time_score,
    }


def pipet_and_measure(volume, params, expected_mass, expected_time):
    pre_air = params.get("pre_asp_air_vol", 0)
    post_air = params.get("post_asp_air_vol", 0)
    disp_vol = volume + pre_air + post_air

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
    }

    masses = []
    start = datetime.now().timestamp()
    for _ in range(REPLICATES):
        lash_e.nr_robot.aspirate_from_vial("liquid_source", volume, **aspirate_kwargs)
        mass = lash_e.nr_robot.dispense_into_vial("measurement_vial", disp_vol, **dispense_kwargs)
        masses.append(mass)
    end = datetime.now().timestamp()

    mean_mass = np.mean(masses)
    std_mass = np.std(masses)
    deviation = (mean_mass - expected_mass) / expected_mass * 100
    time_score = ((end - start) / REPLICATES - expected_time) / expected_time * 100

    return {
        "deviation": deviation,
        "variability": std_mass,
        "time": time_score,
    }


lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
lash_e.nr_robot.check_input_file()
lash_e.nr_robot.move_vial_to_location("measurement_vial", "clamp", 0)

# Run optimization
ax_client = recommender.create_model(SEED, NUM_INITIAL_RECS, NUM_SUGGESTIONS_PER_CYCLE)
all_results = []

# Prime each volume with at least 2 Sobol trials
for volume in VOLUMES:
    expected_mass = volume * DENSITY_LIQUID * 1000
    expected_time = 4
    for _ in range(2):
        params, trial_index = ax_client.get_next_trial(fixed_features=ObservationFeatures({"volume": volume}))
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

    for _ in range(TRIALS_PER_VOLUME):
        suggestions = recommender.get_suggestions(ax_client, volume, n=1)
        for params, trial_index in suggestions:
            results = pipet_and_measure(volume, params, expected_mass, expected_time)
            recommender.add_result(ax_client, trial_index, results)
            results.update(params)
            results["volume"] = volume
            results["trial_index"] = trial_index
            all_results.append(results)

# Save results
results_df = pd.DataFrame(all_results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df.to_csv(f"experiment_summary_{timestamp}.csv", index=False)
print("Saved results to:", f"experiment_summary_{timestamp}.csv")

lash_e.nr_robot.return_vial_home('measurement_vial')
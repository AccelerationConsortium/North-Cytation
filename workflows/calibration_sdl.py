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
import os
import analysis.calibration_analyzer as analyzer  # adjust if needed

# --- Config ---
SEED = 7
SOBOL_CYCLES_PER_VOLUME = 5
BAYES_CYCLES_PER_VOLUME = 27
SIMULATE = False
REPLICATES = 3
VOLUMES = [0.01,0.02,0.05,0.1]
LIQUID = "glycerol" 
#VOLUMES = [0.01,0.02,0.05]
NEW_PIPET_EACH_TIME_SET = True  # If True, will remove pipet after each replicate
DENSITY_LIQUID = 1.26  # g/mL
EXPECTED_MASSES = [v * DENSITY_LIQUID for v in VOLUMES]
EXPECTED_TIME = [v * 10.146 + 9.5813 for v in VOLUMES]
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/calibration_vials.csv"
state = {"waste_vial_index": 0}

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

def empty_vial_if_needed(vial_name, waste_vial_name, state):
    volume = lash_e.nr_robot.get_vial_info(vial_name, 'vial_volume')
    if volume > 7.0:
        #input("Emptying vial...")
        lash_e.nr_robot.remove_pipet()
        disp_volume = volume / np.ceil(volume)
        for i in range (int(np.ceil(volume))-1):
            waste_vial_name = f"waste_vial_{state['waste_vial_index']}"
            if lash_e.nr_robot.get_vial_info(waste_vial_name, 'vial_volume') + disp_volume > 18.0:
                #input("Changing waste..")
                state['waste_vial_index'] += 1
                waste_vial_name = f"waste_vial_{state['waste_vial_index']}"
            lash_e.nr_robot.dispense_from_vial_into_vial(vial_name, waste_vial_name, disp_volume)
        lash_e.nr_robot.remove_pipet()

def fill_liquid_if_needed(vial_name, liquid_source_name):
    volume = lash_e.nr_robot.get_vial_info(liquid_source_name, 'vial_volume')
    if volume < 4.0:
        #input("Adding water...")
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.return_vial_home(vial_name)
        lash_e.nr_robot.dispense_into_vial_from_reservoir(1,liquid_source_name, 8 - volume)
        lash_e.nr_robot.return_vial_home(liquid_source_name)
        lash_e.nr_robot.move_vial_to_location(vial_name, "clamp", 0)

def pipet_and_measure(volume, params, expected_mass, expected_time, new_pipet_each_time=False):
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
 #       "blowout_vol": params["blowout_vol"],
        "measure_weight": True,
        "air_vol": air_vol,
    }

    masses = []
    start = time.time()
    for replicate_idx in range(REPLICATES):
        replicate_start = datetime.now().isoformat()
        lash_e.nr_robot.aspirate_from_vial("liquid_source", volume, **aspirate_kwargs)
        mass = lash_e.nr_robot.dispense_into_vial("measurement_vial", disp_vol, **dispense_kwargs)
        if new_pipet_each_time:
            lash_e.nr_robot.remove_pipet()
        replicate_end = datetime.now().isoformat()
        print("Mass measured: ", mass)
        masses.append(mass)

        raw_entry = {
            "volume": volume,
            "replicate": replicate_idx,
            "mass": mass,
            "start_time": replicate_start,
            "end_time": replicate_end,
            "liquid": LIQUID,
            **params,
        }
        raw_measurements.append(raw_entry)
        pd.DataFrame([raw_entry]).to_csv(autosave_raw_path, mode='a', index=False, header=not os.path.exists(autosave_raw_path))
    end = time.time()

    if not SIMULATE:
        mean_mass = np.mean(masses)
        print("Average mass: ", mean_mass)
        print("Average time per replicate (s): ", (end - start) / REPLICATES)
        
        std_mass = np.std(masses) * 1000  # Variability in mg

        # Deviation: average absolute percent error across replicates
        percent_errors = [abs((m - expected_mass) / expected_mass * 100) for m in masses]
        deviation = np.mean(percent_errors)

        # Time score: relative percent difference from expected time
        time_score = ((end - start) / REPLICATES)

        return {
            "deviation": deviation,
            "variability": std_mass,
            "time": time_score,
        }

    else: return pipet_and_measure_simulated(volume, params, expected_mass, expected_time)


lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False)
lash_e.nr_robot.check_input_file()
lash_e.nr_robot.move_vial_to_location("measurement_vial", "clamp", 0)

if not SIMULATE:
    autosave_dir = os.path.join("output", "autosave_calibration")
    os.makedirs(autosave_dir, exist_ok=True)

    autosave_summary_path = os.path.join(autosave_dir, "experiment_summary.csv")
    autosave_raw_path = os.path.join(autosave_dir, "raw_replicate_data.csv")

# Run optimization
ax_client = recommender.create_model(SEED, SOBOL_CYCLES_PER_VOLUME*len(VOLUMES),VOLUMES, model_type="exploit")
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
        empty_vial_if_needed("measurement_vial", "waste_vial", state)
        result = pipet_and_measure(volume, params, expected_mass, expected_time, NEW_PIPET_EACH_TIME_SET)
        ax_client.complete_trial(trial_index=trial_index, raw_data=result)
        result.update(params)
        result["volume"] = volume
        result["trial_index"] = trial_index
        result["strategy"] = "SOBOL"
        result["liquid"] = LIQUID
        result["time_reported"] = datetime.now().isoformat()
        all_results.append(result)
        # Autosave summary after each trial
        if not SIMULATE:
            pd.DataFrame([result]).to_csv(autosave_summary_path, mode='a', index=False, header=not os.path.exists(autosave_summary_path))


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
            empty_vial_if_needed("measurement_vial", "waste_vial", state)
            results = pipet_and_measure(volume, params, expected_mass, expected_time,NEW_PIPET_EACH_TIME_SET)
            recommender.add_result(ax_client, trial_index, results)
            results.update(params)
            results["volume"] = volume
            results["trial_index"] = trial_index
            results["strategy"] = "BAYESIAN"
            results["liquid"] = LIQUID
            results["time_reported"] = datetime.now().isoformat()
            all_results.append(results)
            if not SIMULATE:
                pd.DataFrame([results]).to_csv(autosave_raw_path, mode='a', index=False, header=not os.path.exists(autosave_raw_path))


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

    # --- Save raw measurements ---
    raw_df = pd.DataFrame(raw_measurements)
    raw_csv_path = os.path.join(save_dir, "raw_replicate_data.csv")
    raw_df.to_csv(raw_csv_path, index=False)
    print("Saved raw replicate data to:", raw_csv_path)

    # 1. Run SHAP analysis and save plots
    analyzer.run_shap_analysis(results_df, save_dir)

    # 2. Get top 3 trials per volume based on weighted normalized scores
    best_trials = analyzer.get_top_trials(
        results_df,
        save_dir,
        weight_time=1.0,
        weight_deviation=0.5,
        weight_variability=2.0,
        top_n=3
    )

    # 3. Plot histograms of parameter distributions for those top trials
    analyzer.plot_top_trial_histograms(best_trials, save_dir)

    # 4. Time/Deviation/Variability scatter plots
    analyzer.plot_time_vs_deviation(results_df, save_dir)

    analyzer.plot_boxplots(results_df, save_dir)
    analyzer.plot_pairplot(results_df, save_dir)
    analyzer.plot_learning_curves(results_df, save_dir)
    analyzer.plot_improvement_summary(results_df, save_dir)
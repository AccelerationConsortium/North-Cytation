# calibration_sdl_long.py

from calibration_sdl_base import *
import sys
sys.path.append("../utoronto_demo")
from ax.core.observation import ObservationFeatures
from master_usdl_coordinator import Lash_E
import recommenders.pipeting_optimizer_honegumi as recommender

# --- Experiment Config ---
SEED = 7
SOBOL_CYCLES_PER_VOLUME = 30
BAYES_CYCLES_PER_VOLUME = 120
SIMULATE = False
REPLICATES = 3
VOLUMES = [0.01, 0.02, 0.05, 0.1]

LIQUID = "water" #<------------------- CHANGE THIS!
DENSITY_LIQUID = LIQUIDS[LIQUID]["density"]
NEW_PIPET_EACH_TIME_SET = LIQUIDS[LIQUID]["refill_pipets"]

MODE = "explore"
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/calibration_vials_long.csv"
EXPECTED_MASSES = [v * DENSITY_LIQUID for v in VOLUMES]
EXPECTED_TIME = [v * 10.146 + 9.5813 for v in VOLUMES]

# --- Init ---
state = {"waste_vial_index": 0}
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False)
lash_e.nr_robot.check_input_file()
lash_e.nr_robot.move_vial_to_location("measurement_vial", "clamp", 0)

if not SIMULATE:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S" + f"_{LIQUID}")
    base_autosave_dir = r"C:\Users\Imaging Controller\Desktop\Calibration_SDL_Output\autosave_calibration"
    autosave_dir = os.path.join(base_autosave_dir, timestamp)
    os.makedirs(autosave_dir, exist_ok=True)
    autosave_summary_path = os.path.join(autosave_dir, "experiment_summary_autosave.csv")
    autosave_raw_path = os.path.join(autosave_dir, "raw_replicate_data_autosave.csv")

# --- Optimization Loop ---
ax_client = recommender.create_model(SEED, SOBOL_CYCLES_PER_VOLUME * len(VOLUMES), VOLUMES, model_type=MODE)
all_results = []
raw_measurements = []

for i, volume in enumerate(VOLUMES):
    expected_mass = EXPECTED_MASSES[i]
    expected_time = EXPECTED_TIME[i]
    for _ in range(SOBOL_CYCLES_PER_VOLUME):
        params, trial_index = ax_client.get_next_trial(fixed_features=ObservationFeatures({"volume": volume}))
        fill_liquid_if_needed(lash_e, "measurement_vial", "liquid_source")
        empty_vial_if_needed(lash_e, "measurement_vial", state)
        result = pipet_and_measure(lash_e, 'liquid_source', 'measurement_vial', volume, params, expected_mass, expected_time, REPLICATES, SIMULATE, autosave_raw_path, raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET)
        ax_client.complete_trial(trial_index=trial_index, raw_data=result)
        result.update(params)
        result.update({"volume": volume, "trial_index": trial_index, "strategy": "SOBOL", "liquid": LIQUID, "time_reported": datetime.now().isoformat()})
        all_results.append(result)
        if not SIMULATE:
            pd.DataFrame([result]).to_csv(autosave_summary_path, mode='a', index=False, header=not os.path.exists(autosave_summary_path))

for i, volume in enumerate(VOLUMES):
    expected_mass = EXPECTED_MASSES[i]
    expected_time = EXPECTED_TIME[i]
    for _ in range(BAYES_CYCLES_PER_VOLUME):
        suggestions = recommender.get_suggestions(ax_client, volume, n=1)
        for params, trial_index in suggestions:
            fill_liquid_if_needed(lash_e, "measurement_vial", "liquid_source")
            empty_vial_if_needed(lash_e, "measurement_vial", state)
            results = pipet_and_measure(lash_e,'liquid_source', 'measurement_vial', volume, params, expected_mass, expected_time, REPLICATES, SIMULATE, autosave_raw_path, raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET)
            recommender.add_result(ax_client, trial_index, results)
            results.update(params)
            results.update({"volume": volume, "trial_index": trial_index, "strategy": "BAYESIAN", "liquid": LIQUID, "time_reported": datetime.now().isoformat()})
            all_results.append(results)
            if not SIMULATE:
                pd.DataFrame([results]).to_csv(autosave_summary_path, mode='a', index=False, header=not os.path.exists(autosave_summary_path))

results_df = pd.DataFrame(all_results)
lash_e.nr_robot.remove_pipet()
lash_e.nr_robot.return_vial_home('measurement_vial')
lash_e.nr_robot.move_home()
print(results_df)

if not SIMULATE:
    save_analysis(results_df, pd.DataFrame(raw_measurements), autosave_dir)
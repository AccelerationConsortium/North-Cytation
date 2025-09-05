# calibration_sdl_short.py


from matplotlib.pylab import f
from calibration_sdl_base import *
import sys
sys.path.append("../utoronto_demo")
from ax.core.observation import ObservationFeatures
from master_usdl_coordinator import Lash_E
import recommenders.pipeting_optimizer_v2 as recommender

# --- Experiment Config ---
LIQUID = "glycerol"  #<------------------- CHANGE THIS!
SIMULATE = True #<--------- CHANGE THIS!

DENSITY_LIQUID = LIQUIDS[LIQUID]["density"]
NEW_PIPET_EACH_TIME_SET = LIQUIDS[LIQUID]["refill_pipets"]

SEED = 7
SOBOL_CYCLES_PER_VOLUME = 5
BAYES_CYCLES_PER_VOLUME = 27
REPLICATES = 3
BAYESIAN_BATCH_SIZE = 1
VOLUMES = [0.05] #If time try different volumes! Eg 0.01 0.02 0.1
#MODELS = ['qEI', 'qLogEI', 'qNEHVI']
MODELS = ['qNEHVI'] #Change this!

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/calibration_vials_short.csv"
EXPECTED_MASSES = [v * DENSITY_LIQUID for v in VOLUMES]
EXPECTED_TIME = [v * 10.146 + 9.5813 for v in VOLUMES]
EXPECTED_ABSORBANCE = []

state = {
    "measurement_vial_index": 0,
    "measurement_vial_name": "measurement_vial_0"
}

# --- Init ---
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False)
lash_e.nr_robot.check_input_file()
lash_e.nr_robot.move_vial_to_location("measurement_vial_0", "clamp", 0)

lash_e.logger.info("Liquid: ", LIQUIDS[LIQUID])


for model_type in MODELS:
    if not SIMULATE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S" + f"_{LIQUID}"+f"_{model_type}")
        base_autosave_dir = r"C:\Users\Imaging Controller\Desktop\Calibration_SDL_Output\autosave_calibration"
        autosave_dir = os.path.join(base_autosave_dir, timestamp)
        os.makedirs(autosave_dir, exist_ok=True)
        autosave_summary_path = os.path.join(autosave_dir, "experiment_summary_autosave.csv")
        autosave_raw_path = os.path.join(autosave_dir, "raw_replicate_data_autosave.csv")
    else:
        autosave_raw_path=None
        autosave_summary_path=None

    # --- Optimization Loop ---
    ax_client = recommender.create_model(SEED, SOBOL_CYCLES_PER_VOLUME * len(VOLUMES), bayesian_batch_size=BAYESIAN_BATCH_SIZE, volume=VOLUMES, model_type=model_type)
    all_results = []
    raw_measurements = []

    def check_if_measurement_vial_full():
        global state
        current_vial = state["measurement_vial_name"]
        vol = lash_e.nr_robot.get_vial_info(current_vial, "vial_volume")
        if vol > 7.0:
            lash_e.nr_robot.remove_pipet()
            lash_e.nr_robot.return_vial_home(current_vial)
            state["measurement_vial_index"] += 1
            new_vial_name = f"measurement_vial_{state['measurement_vial_index']}"
            state["measurement_vial_name"] = new_vial_name
            lash_e.logger.info(f"[info] Switching to new measurement vial: {new_vial_name}")
            lash_e.nr_robot.move_vial_to_location(new_vial_name, "clamp", 0)


    for i, volume in enumerate(VOLUMES):
        expected_mass = EXPECTED_MASSES[i]
        expected_time = EXPECTED_TIME[i]
        for _ in range(SOBOL_CYCLES_PER_VOLUME):
            params, trial_index = ax_client.get_next_trial()
            check_if_measurement_vial_full()
            result = pipet_and_measure(lash_e, 'liquid_source', state["measurement_vial_name"], volume, params, expected_mass, expected_time, REPLICATES, SIMULATE, autosave_raw_path, raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET)
            ax_client.complete_trial(trial_index=trial_index, raw_data=result)
            result.update(params)
            result.update({"volume": volume, "trial_index": trial_index, "strategy": "SOBOL", "liquid": LIQUID, "time_reported": datetime.now().isoformat()})
            result = strip_tuples(result)
            all_results.append(result)
            if not SIMULATE:
                pd.DataFrame([result]).to_csv(autosave_summary_path, mode='a', index=False, header=not os.path.exists(autosave_summary_path))

    for i, volume in enumerate(VOLUMES):
        expected_mass = EXPECTED_MASSES[i]
        expected_time = EXPECTED_TIME[i]
        for _ in range(BAYES_CYCLES_PER_VOLUME):
            suggestions = recommender.get_suggestions(ax_client, volume, n=1)
            for params, trial_index in suggestions:
                check_if_measurement_vial_full()
                results = pipet_and_measure(lash_e, 'liquid_source', state["measurement_vial_name"], volume, params, expected_mass, expected_time, REPLICATES, SIMULATE, autosave_raw_path, raw_measurements, LIQUID, NEW_PIPET_EACH_TIME_SET)
                recommender.add_result(ax_client, trial_index, results)
                results.update(params)
                results.update({"volume": volume, "trial_index": trial_index, "strategy": "BAYESIAN", "liquid": LIQUID, "time_reported": datetime.now().isoformat()})
                results = strip_tuples(results)
                all_results.append(results)
                if not SIMULATE:
                    pd.DataFrame([results]).to_csv(autosave_summary_path, mode='a', index=False, header=not os.path.exists(autosave_summary_path))

    results_df = pd.DataFrame(all_results)
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.return_vial_home(state["measurement_vial_name"])
    lash_e.nr_robot.move_home()
    lash_e.logger.info(results_df)

    if not SIMULATE:
        save_analysis(results_df, pd.DataFrame(raw_measurements), autosave_dir)

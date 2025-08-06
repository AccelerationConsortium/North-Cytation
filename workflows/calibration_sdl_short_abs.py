# calibration_sdl_short.py

from calibration_sdl_base import *
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import recommenders.pipeting_optimizer_honegumi as recommender
import time

# --- Experiment Config ---
LIQUID = "water"  #<------------------- CHANGE THIS!
SIMULATE = LOGGING = True #<--------- CHANGE THIS!

DENSITY_LIQUID = LIQUIDS[LIQUID]["density"]
NEW_PIPET_EACH_TIME_SET = LIQUIDS[LIQUID]["refill_pipets"]

calib_data = {
    "water": {"intercept": 0.081702253157804, "slope": 0.000399821977780455},
    "ethanol": {"intercept": 0.0772742779640708, "slope": 0.000394283351777735},
    "glycerol": {"intercept": 0.0822068049753275, "slope": 0.000389037817262831}
}

SEED = 7
SOBOL_CYCLES_PER_VOLUME = 5
BAYES_CYCLES_PER_VOLUME = 9
BAYES_CONDITIONS_PER_CYCLE = 3
REPLICATES = 3
VOLUMES = [0.07, 0.100, 0.150, 0.200]

MODE = "exploit"
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/calibration_vials_short.csv"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Ilya_Measurement.prt" #<------------------- CHECK THIS
EXPECTED_ABSORBANCE = []

print("Liquid: ", LIQUIDS[LIQUID])

lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False, logging=LOGGING)
lash_e.nr_robot.check_input_file()
lash_e.nr_track.check_input_file()

initial_volume = lash_e.nr_robot.get_vial_info('liquid_source', 'vial_volume')

if not SIMULATE:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S" + f"_{LIQUID}")
    base_autosave_dir = r"C:\Users\Imaging Controller\Desktop\Calibration_SDL_Output\autosave_calibration"
    autosave_dir = os.path.join(base_autosave_dir, timestamp)
    os.makedirs(autosave_dir, exist_ok=True)
    autosave_summary_path = os.path.join(autosave_dir, "experiment_summary_autosave.csv")
    autosave_raw_path = os.path.join(autosave_dir, "raw_replicate_data_autosave.csv")
else:
    autosave_raw_path=None
    autosave_summary_path=None

# --- Optimization Loop ---
ax_client = recommender.create_model(SEED, SOBOL_CYCLES_PER_VOLUME * len(VOLUMES), VOLUMES, model_type=MODE, simulate=SIMULATE)
all_results = []
raw_measurements = []

def dispense_into_wp(lash_e, source_vial, volume, params, wells, new_pipet_each_time):
    pre_air = params.get("pre_asp_air_vol", 0)
    post_air = params.get("post_asp_air_vol", 0)
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
        "air_vol": air_vol,
    }
    start = time.time()
    for well in wells:
        lash_e.nr_robot.aspirate_from_vial(source_vial, volume, **aspirate_kwargs)
        lash_e.nr_robot.dispense_into_wellplate([well], [volume], **dispense_kwargs)
        if new_pipet_each_time:
            lash_e.nr_robot.remove_pipet()

    end = time.time()
    return (end - start) / len(wells)


def measure_wellplate(lash_e, wells, simulate, raw_path, raw_measurements): #Save the data and return the results
    if not simulate:
        measurements = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, wells) 
        #SAVE THAT DATA
    return measurements

def get_results(measurements_all: pd.DataFrame, times: list, expected_absorbances: list): #May need to play with this with real data
    results = []
    num_replicates = 3

    for i in range(len(times)):
        # Get the i-th group of replicates
        start = i * num_replicates
        end = (i + 1) * num_replicates
        group = measurements_all.iloc[start:end]
        values = group['tbd'].values #This depends on the column names in the DataFrame
        measurement = np.mean(values)
        variability = np.std(values) / measurement * 100
        deviation = (measurement - expected_absorbances[i]) / expected_absorbances[i] * 100
        time_rec = times[i]

        results.append({
            "deviation": deviation,
            "variability": variability,
            "time": time_rec
        })

    return pd.DataFrame(results)


for volume in VOLUMES:
    well_index = 0
    lash_e.nr_track.get_new_wellplate()

    suggestions = recommender.get_suggestions(ax_client, volume, n=SOBOL_CYCLES_PER_VOLUME)
    batch_wells = range (well_index, well_index + REPLICATES*SOBOL_CYCLES_PER_VOLUME) #Wells for measurement
    times = []
    expected_absorbances = []
    trials = []
    for params, trial_index in suggestions:
        print("Sobol Trial: ", trial_index, "Params: ", params)
        wells = range(well_index, well_index + REPLICATES)
        well_index += REPLICATES
        time_rec = dispense_into_wp(lash_e, 'liquid_source', volume, params, wells, NEW_PIPET_EACH_TIME_SET)
        expected_abs = calib_data[LIQUID]["intercept"] + calib_data[LIQUID]["slope"] * volume
        times.append(time_rec)
        expected_absorbances.append(expected_abs)
        trials.append(trial_index)
    
    if not SIMULATE:
        measurements = measure_wellplate(lash_e, batch_wells, SIMULATE, autosave_raw_path, raw_measurements)
        results = get_results(measurements, times, expected_absorbances)
    else:
        results = [pipet_and_measure_simulated(volume, params, expected_absorbances[0], times[0])for _ in range(SOBOL_CYCLES_PER_VOLUME)]
    
    for i in range (0, SOBOL_CYCLES_PER_VOLUME):
        result = results[i]
        print("Result: ", result)
        ax_client.complete_trial(trial_index=trials[i], raw_data=result)
        result.update(params)
        result.update({"volume": volume, "trial_index": trials[i], "strategy": "SOBOL", "liquid": LIQUID, "time_reported": datetime.now().isoformat()})
        result = strip_tuples(result)
        all_results.append(result)

        if not SIMULATE:
            pd.DataFrame([result]).to_csv(autosave_summary_path, mode='a', index=False, header=not os.path.exists(autosave_summary_path))

    for _ in range(0, BAYES_CYCLES_PER_VOLUME):
        suggestions = recommender.get_suggestions(ax_client, volume, n=BAYES_CONDITIONS_PER_CYCLE)

        batch_wells = range (well_index, well_index + REPLICATES*BAYES_CONDITIONS_PER_CYCLE) #Wells for measurement
        times = []
        expected_absorbances = []
        trials = []
        for params, trial_index in suggestions:
            print("Bayesian Trial: ", trial_index, "Params: ", params)
            wells = range(well_index, well_index + REPLICATES)
            well_index += REPLICATES
            time_rec = dispense_into_wp(lash_e, 'liquid_source', volume, params, wells, NEW_PIPET_EACH_TIME_SET)
            expected_abs = calib_data[LIQUID]["intercept"] + calib_data[LIQUID]["slope"] * volume
            times.append(time_rec)
            trials.append(trial_index)
            expected_absorbances.append(expected_abs)
        
        if not SIMULATE:
            measurements = measure_wellplate(lash_e, batch_wells, SIMULATE, autosave_raw_path, raw_measurements)
            results = get_results(measurements, times, expected_absorbances)
        else:
            results = [pipet_and_measure_simulated(volume, params, expected_absorbances[0], times[0])for _ in range(SOBOL_CYCLES_PER_VOLUME)]

        for i in range (0, BAYES_CONDITIONS_PER_CYCLE):
            result = results[i]
            print("Result: ", result)    
            recommender.add_result(ax_client, trials[i], result)
            result.update(params)
            result.update({"volume": volume, "trial_index": trials[i], "strategy": "BAYESIAN", "liquid": LIQUID, "time_reported": datetime.now().isoformat()})
            result = strip_tuples(result)
            all_results.append(result)
            if not SIMULATE:
                pd.DataFrame([result]).to_csv(autosave_summary_path, mode='a', index=False, header=not os.path.exists(autosave_summary_path))

    lash_e.nr_track.discard_wellplate()

results_df = pd.DataFrame(all_results)
lash_e.nr_robot.remove_pipet()
lash_e.nr_robot.move_home()
print(results_df)

final_volume = lash_e.nr_robot.get_vial_info('liquid_source', 'vial_volume')
total_volume_used = initial_volume - final_volume

print(f"Total volume used: {total_volume_used} mL")

if not SIMULATE:
    save_analysis(results_df, pd.DataFrame(raw_measurements), autosave_dir)

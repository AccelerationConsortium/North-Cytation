# calibration_sdl_base.py
import sys
sys.path.append("../utoronto_demo")

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import analysis.calibration_analyzer as analyzer

LIQUIDS = {
    "water": {"density": 1.00, "refill_pipets": False},
    "ethanol": {"density": 0.789, "refill_pipets": False},
    "glycerol": {"density": 1.26, "refill_pipets": True},
    "PEG_Water": {"density": 1.05, "refill_pipets": True},
}

# --- Utility Functions ---
def pipet_and_measure_simulated(volume, params, expected_mass, expected_time):
    time.sleep(0.2)
    deviation = np.abs(params["aspirate_speed"] - 15) + np.random.normal(0, 0.5)
    variability = np.abs(params["aspirate_wait_time"] - 30) * 0.1 + np.random.normal(0, 0.3)
    time_score = (params["aspirate_wait_time"] - 1) * 100 + np.random.normal(0, 5)
    return {"deviation": deviation, "variability": variability, "time": time_score}

def empty_vial_if_needed(lash_e, vial_name, state):
    volume = lash_e.nr_robot.get_vial_info(vial_name, 'vial_volume')
    if volume > 7.0:
        lash_e.nr_robot.remove_pipet()
        disp_volume = volume / np.ceil(volume)
        for _ in range(int(np.ceil(volume)) - 1):
            waste_vial_name = f"waste_vial_{state['waste_vial_index']}"
            if lash_e.nr_robot.get_vial_info(waste_vial_name, 'vial_volume') + disp_volume > 18.0:
                state['waste_vial_index'] += 1
                waste_vial_name = f"waste_vial_{state['waste_vial_index']}"
            lash_e.nr_robot.dispense_from_vial_into_vial(vial_name, waste_vial_name, disp_volume)
        lash_e.nr_robot.remove_pipet()

def fill_liquid_if_needed(lash_e, vial_name, liquid_source_name):
    volume = lash_e.nr_robot.get_vial_info(liquid_source_name, 'vial_volume')
    if volume < 4.0:
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.return_vial_home(vial_name)
        lash_e.nr_robot.dispense_into_vial_from_reservoir(1, liquid_source_name, 8 - volume)
        lash_e.nr_robot.return_vial_home(liquid_source_name)
        lash_e.nr_robot.move_vial_to_location(vial_name, "clamp", 0)

def pipet_and_measure(lash_e, source_vial, dest_vial, volume, params, expected_measurement, expected_time, replicate_count, simulate, raw_path, raw_measurements, liquid, new_pipet_each_time):
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
        "measure_weight": True,
        "air_vol": air_vol,
    }
    

    measurements = []
    start = time.time()
    for replicate_idx in range(replicate_count):
        replicate_start = datetime.now().isoformat()
        lash_e.nr_robot.aspirate_from_vial(source_vial, volume, **aspirate_kwargs)
        measurement = lash_e.nr_robot.dispense_into_vial(dest_vial, volume, **dispense_kwargs)
        if new_pipet_each_time:
            lash_e.nr_robot.remove_pipet()
        replicate_end = datetime.now().isoformat()
        raw_entry = {"volume": volume, "replicate": replicate_idx, "mass": measurement, "start_time": replicate_start, "end_time": replicate_end, "liquid": liquid, **params}
        raw_measurements.append(raw_entry) 
        if not simulate:
            pd.DataFrame([raw_entry]).to_csv(raw_path, mode='a', index=False, header=not os.path.exists(raw_path))
        measurements.append(measurement)
    end = time.time()

    avg_measurement = np.mean(measurements)
    std_measurement = np.std(measurements) / avg_measurement * 100  # % relative std
    percent_errors = [abs((m - expected_measurement) / expected_measurement * 100) for m in measurements]
    deviation = np.mean(percent_errors)
    time_score = ((end - start) / replicate_count)

    if simulate:
        return pipet_and_measure_simulated(volume, params, expected_measurement, expected_time)
    else:
        return {"deviation": deviation, "variability": std_measurement , "time": time_score}

def strip_tuples(d):
    """Convert any (x, None) → x in a flat dict."""
    return {k: (v if not (isinstance(v, tuple) and v[1] is None) else v[0]) for k, v in d.items()}

def save_analysis(results_df, raw_df, save_dir):
    results_df.to_csv(os.path.join(save_dir, "experiment_summary.csv"), index=False)
    raw_df.to_csv(os.path.join(save_dir, "raw_replicate_data.csv"), index=False)

    for metric in ['deviation', 'time', 'variability']:
        if metric in results_df.columns:
            results_df[metric] = pd.to_numeric(results_df[metric], errors='coerce')

    analyzer.run_shap_analysis(results_df, save_dir)
    best_trials = analyzer.get_top_trials(results_df, save_dir, weight_time=1.0, weight_deviation=0.5, weight_variability=2.0, top_n=3)
    analyzer.plot_top_trial_histograms(best_trials, save_dir)
    analyzer.plot_time_vs_deviation(results_df, save_dir)
    analyzer.plot_boxplots(results_df, save_dir)
    analyzer.plot_pairplot(results_df, save_dir)
    analyzer.plot_learning_curves(results_df, save_dir)
    analyzer.plot_improvement_summary(results_df, save_dir)

def flatten_measurements(raw_data: dict) -> pd.DataFrame:
    records = []
    for (replicate, wavelength), series in raw_data.items():
        for well, absorbance in series.items():
            records.append({
                'replicate': replicate,
                'well': well,
                'wavelength': int(wavelength),  # Convert string like '590' to int
                'absorbance': absorbance
            })
    return pd.DataFrame(records)
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime

sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/calibration_vials.csv"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Ilya_Measurement.prt"
SIMULATE = True
REPLICATES = 5
VOLUMES = [0.005, 0.01, 0.02, 0.05]  # In mL
ARGS = [
    # {
    #     "aspirate_speed": 8,
    #     "dispense_speed": 8,
    #     "wait_time": 0.5,
    #     "retract_speed": 3,
    #     "pre_asp_air_vol": 0.01,
    #     "post_asp_air_vol": 0.01,
    #     "blowout_vol": 0.005
    # },
    {
        "blowout_vol": 0.1
    },
    {}
]

DENSITY_LIQUID = 0.997  # g/mL
EXPECTED_MASSES = [v * DENSITY_LIQUID * 1000 for v in VOLUMES]  # in mg
EXPECTED_ABS = None  # Placeholder if needed later
EXPECTED_TIME = [4] * len(VOLUMES)  # Example expected time per replicate in seconds
method = "mass"
well_count = 0

# Initialize a dataframe to collect results
results_df = pd.DataFrame(columns=[
    "volume_mL", "mean_mass_a", "std_mass_a", "deviation_a",
    "mean_mass_b", "std_mass_b", "deviation_b",
    "time_per_replicate", "time_score"
])

raw_data = []  # To store raw measurements

def pipet_and_measure(args): #Take a single argument
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
    lash_e.nr_robot.check_input_file()
    lash_e.nr_robot.move_vial_to_location('measurement_vial', 'clamp', 0)

    for args in ARGS:
        print("Conditions: ", args)

        for i, volume in enumerate(VOLUMES):
            print("Volume being measured (mL): ", volume)

            measurement_results_a = []
            start_time = time.time()

            for j in range(REPLICATES):
                # Calculate dispense volume if there is pre/post air volume
                pre_asp_air = args.get("pre_asp_air_vol", 0)
                post_asp_air = args.get("post_asp_air_vol", 0)
                dispense_volume = volume + pre_asp_air + post_asp_air

                # Prepare aspirate kwargs
                aspirate_kwargs = {
                    "aspirate_speed": args.get("aspirate_speed", 11),
                    "wait_time": args.get("wait_time", 0),
                    "retract_speed": args.get("retract_speed", 5),
                    "pre_asp_air_vol": pre_asp_air,
                    "post_asp_air_vol": post_asp_air
                }

                # Prepare dispense kwargs
                dispense_kwargs = {
                    "dispense_speed": args.get("dispense_speed", 11),
                    "wait_time": args.get("wait_time", 0),
                    "blowout_vol": args.get("blowout_vol", 0),
                    "measure_weight": True
                }

                lash_e.nr_robot.aspirate_from_vial('liquid_source', volume, **aspirate_kwargs)
                mass = lash_e.nr_robot.dispense_into_vial('measurement_vial', dispense_volume, **dispense_kwargs)

                measurement_results_a.append(mass)
                raw_data.append({"volume_mL": volume, "replicate": j+1, "method": "A", "measured_mass_mg": mass})

            end_time = time.time()
            time_elapsed = (end_time - start_time) / REPLICATES
            time_score = ( time_elapsed - EXPECTED_TIME[i] ) / EXPECTED_TIME[i] * 100 #May not be the best kind of score

            mean_a = np.mean(measurement_results_a)
            std_a = np.std(measurement_results_a)
            dev_a = (mean_a - EXPECTED_MASSES[i]) / EXPECTED_MASSES[i] * 100

            # Dummy placeholders for Method B for now
            mean_b, std_b, dev_b = 0, 0, 0

            results_df.loc[len(results_df)] = [
                volume, mean_a, std_a, dev_a, mean_b, std_b, dev_b, time_elapsed, time_score
            ]

            print(f"Volume deposited: {volume} mL with {REPLICATES} replicates")
            print(f"Measurement type: {method}")
            print(f"Method A: Mean={mean_a:.2f}, Std={std_a:.2f}, Deviation={dev_a:.2f}%")
            print(f"Elapsed Time: {time_elapsed:.2f} sec/replicate | Time Score: {time_score:.2f}")
            print("------------------------------")

    return results_df

#Create workflow
NUM_CYCLES = 3
SEED = 5
NUM_SUGGESTIONS_PER_CYCLE = 5

#Step 1: Define initial model and suggestions
model = create_model()
initial_suggestions = get_initial_suggestions()

#Step 2: Get initial data
for suggested_parameters in initial_suggestions:
    results = pipet_and_measure(suggested_parameters)
    #Add the new results to the model data

#Step 3: Iterate
for i in range (0, NUM_CYCLES):
    next_suggestions = get_suggestions()

    for suggested_parameters in next_suggestions:
        results = pipet_and_measure(suggested_parameters)
        #Add the new results to the model data

#Step 4: Save Data
if not SIMULATE:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"../utoronto_demo/output/{timestamp}_summary.csv", index=False)
    raw_df = pd.DataFrame(raw_data)
    raw_df.to_csv(f"../utoronto_demo/output/{timestamp}_raw_data.csv", index=False)
    print("Results saved.")



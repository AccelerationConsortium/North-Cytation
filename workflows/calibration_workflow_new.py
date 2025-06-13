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
VOLUMES = [0.02, 0.05, 0.10, 0.15, 0.20]  # In mL
ARGS = [] #TODO
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

def sample_workflow():
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
    lash_e.nr_robot.check_input_file()

    lash_e.nr_robot.move_vial_to_location('measurement_vial', 'clamp', 0)

    for args in ARGS:
        print("Conditions: ", args)

        for i, volume in enumerate(VOLUMES):
            print("Volume being measured (mL): ", volume)

            measurement_results_a = []
            start_time = time.time()

            # Method A: No air gap
            for j in range(REPLICATES):
                lash_e.nr_robot.aspirate_from_vial('liquid_source', volume)
                mass = lash_e.nr_robot.dispense_into_vial('measurement_vial', volume, measure_weight=True)
                measurement_results_a.append(mass)
                raw_data.append({"volume_mL": volume, "replicate": j+1, "method": "A", "measured_mass_mg": mass})

            end_time = time.time()
            time_elapsed = (end_time - start_time) / REPLICATES

            mean_a = np.mean(measurement_results_a)
            std_a = np.std(measurement_results_a)
            dev_a = (mean_a - EXPECTED_MASSES[i]) / EXPECTED_MASSES[i] * 100

            results_df.loc[len(results_df)] = [volume, mean_a, std_a, dev_a, time_elapsed]

            print(f"Volume deposited: {volume} mL with {REPLICATES} replicates")
            print(f"Measurement type: {method}")
            print(f"Method A: Mean={mean_a:.2f}, Std={std_a:.2f}, Deviation={dev_a:.2f}%")
            print(f"Elapsed Time: {time_elapsed:.2f} sec/replicate")
            print("------------------------------")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f"../utoronto_demo/output/{timestamp}_summmary.csv", index=False)
        raw_df = pd.DataFrame(raw_data)
        raw_df.to_csv( f"../utoronto_demo/output/{timestamp}_raw_data.csv", index=False)
        print("Results saved to volume_mass_comparison.csv and raw_mass_measurements.csv")

sample_workflow()

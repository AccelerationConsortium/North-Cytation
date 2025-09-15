import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime

sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/calibration_vials.csv"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Ilya_Measurement.prt"
SIMULATE = False
REPLICATES = 5
VOLUMES = [0.025, 0.05, 0.1, 0.2] #[0.005, 0.01, 0.02, 0.03, 0.05]  # In mL
ARGS = [
    {}
]

# DENSITY_LIQUID = 0.997  # g/mL
# EXPECTED_MASSES = [v * DENSITY_LIQUID * 1000 for v in VOLUMES]  # in mg
# EXPECTED_ABS = None  # Placeholder if needed later
# EXPECTED_TIME = [4] * len(VOLUMES)  # Example expected time per replicate in seconds
# method = "mass"
well_count = 0

# Initialize a dataframe to collect results
results_df = pd.DataFrame(columns=[
    "volume_mL", 
    "time"
])

raw_data = []  # To store raw measurements

def sample_workflow():
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
    
    lash_e.nr_robot.check_input_file()

    well_index = 0
    lash_e.nr_robot.get_pipet(lash_e.nr_robot.HIGHER_PIPET_ARRAY_INDEX) #get small pipet tip
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
                    #"measure_weight": True
                }

                start_time = time.time()
                lash_e.nr_robot.aspirate_from_vial('liquid_source', volume, **aspirate_kwargs)
                lash_e.nr_robot.dispense_into_wellplate(dest_wp_num_array=[well_index], amount_mL_array=[dispense_volume], **dispense_kwargs)
                end_time = time.time()
                time_elapsed = end_time - start_time
                measurement_results_a.append(time_elapsed)

                well_index += 1
                
                raw_data.append({"volume_mL": volume, "replicate": j+1, "method": "A", "time": time_elapsed})


            mean_time = np.mean(measurement_results_a)
            results_df.loc[len(results_df)] = [
                volume,mean_time
            ]

            print(f"Volume deposited: {volume} mL with {REPLICATES} replicates")
            print(f"Mean Time: {mean_time:.2f} sec/replicate")
            print("------------------------------")

        # Save results
        if not SIMULATE:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_df.to_csv(f"../utoronto_demo/output/{timestamp}_vial_to_wp_time_summary.csv", index=False)
            raw_df = pd.DataFrame(raw_data)
            raw_df.to_csv(f"../utoronto_demo/output/{timestamp}_vial_to_wp_time_raw_data.csv", index=False)
            print("Results saved.")
    
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.move_home()


sample_workflow()

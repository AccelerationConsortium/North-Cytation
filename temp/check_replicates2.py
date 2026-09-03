import pandas as pd
import os

BASE = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign"

param_cols = ["aspirate_speed", "dispense_speed", "aspirate_wait_time", "dispense_wait_time",
              "pre_asp_air_vol_uL", "post_asp_air_vol_uL", "overaspirate_vol_uL",
              "blowout_vol_uL", "retract_speed", "post_retract_wait_time", "volume_ul_target"]

for c in ["200uL", "1000uL"]:
    df = pd.read_csv(os.path.join(BASE, c, "incremental_results.csv"))
    ok = df[df["status"] == "ok"].copy()

    reps = ok.groupby(param_cols).size()
    print(f"=== {c} ===")
    print(f"Total ok rows: {len(ok)}")
    print(f"Unique parameter combinations: {len(reps)}")
    print(f"Replicate distribution (replicates per condition):")
    print(reps.value_counts().sort_index().to_string())
    print()

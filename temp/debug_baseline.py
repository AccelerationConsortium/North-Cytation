import pandas as pd
from pathlib import Path

FOCUS = Path("output/two_point_focus_glycerol_25_50_75_100")

trial = pd.read_csv(FOCUS / "all_trial_results.csv")
raw   = pd.read_csv(FOCUS / "all_raw_measurements.csv")

print("=== Baseline rows in trial file ===")
base_trial = trial[trial["source_folder"] == "baseline"]
print(base_trial[["source_folder","trial_id","liquid","volume_target_ul"]].to_string(index=False))

print("\n=== Unique source_folder values in raw file ===")
print(raw["source_folder"].unique())

print("\n=== Does raw have any baseline source_folder? ===")
print(raw[raw["source_folder"] == "baseline"][["source_folder","trial_id","liquid","timestamp"]].head(10).to_string(index=False))

print("\n=== Raw rows for same liquids in trial_id 1-6 range ===")
print(raw[raw["trial_id"].isin(["trial_1","trial_2","trial_3","trial_4","trial_5","trial_6"])][
    ["source_folder","trial_id","liquid","timestamp"]].head(20).to_string(index=False))

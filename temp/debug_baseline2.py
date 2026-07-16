import pandas as pd
from pathlib import Path

FOCUS = Path("output/two_point_focus_glycerol_25_50_75_100")
raw = pd.read_csv(FOCUS / "all_raw_measurements.csv")

print("=== Raw rows where source_folder is NaN ===")
null_rows = raw[raw["source_folder"].isna()]
print(f"Count: {len(null_rows)}")
print(null_rows[["source_folder","trial_id","liquid","volume_target_ul","timestamp"]].to_string(index=False))

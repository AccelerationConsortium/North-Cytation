"""
Verify MQTT timestamp matching quality:
1. Check time deltas are genuinely small (seconds, not hours)
2. Spot-check: show actual measurement timestamp vs matched env_timestamp side by side
3. Check if any systematic offset exists (timezone issue would show as consistent ~Nh offset)
"""
import pandas as pd
from pathlib import Path

FOCUS = Path("output/two_point_focus_glycerol_25_50_75_100")

print("=== all_raw_measurements.csv spot-check ===")
raw = pd.read_csv(FOCUS / "all_raw_measurements.csv")
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
raw["env_timestamp"] = pd.to_datetime(raw["env_timestamp"])

print(f"env_time_delta_min stats:")
print(raw["env_time_delta_min"].describe().round(4))
print()

# Show a sample spread across the date range
sample = raw.sort_values("timestamp").iloc[::raw.shape[0]//10][
    ["liquid", "timestamp", "env_timestamp", "env_sht_temp_c", "env_sht_rh", "env_time_delta_min"]
]
print("Sample rows (evenly spaced across dataset):")
print(sample.to_string(index=False))

print()
print("=== two_point_combined_raw_data.csv spot-check ===")
tp = pd.read_csv(FOCUS / "two_point_combined_raw_data.csv")
tp["timestamp"] = pd.to_datetime(tp["timestamp"])
tp["env_timestamp"] = pd.to_datetime(tp["env_timestamp"])

print(f"env_time_delta_min stats:")
print(tp["env_time_delta_min"].describe().round(4))
print()
sample_tp = tp.sort_values("timestamp").iloc[::tp.shape[0]//8][
    ["liquid", "timestamp", "env_timestamp", "env_sht_temp_c", "env_sht_rh", "env_time_delta_min"]
]
print("Sample rows:")
print(sample_tp.to_string(index=False))

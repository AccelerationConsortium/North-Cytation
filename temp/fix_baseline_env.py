"""
Fix: source_folder NaN -> 'baseline' in all_raw_measurements.csv,
then re-derive mean timestamps for baseline trial rows and update env columns
in all_trial_results.csv.
"""
import pandas as pd
from pathlib import Path

FOCUS = Path("output/two_point_focus_glycerol_25_50_75_100")
MQTT_PATH = Path(r"C:\Users\Imaging Controller\Desktop\m5stack\mqtt_log.csv")
THRESHOLD_MIN = 30.0

# ── Fix raw file: NaN -> 'baseline' ──────────────────────────────────────────
raw = pd.read_csv(FOCUS / "all_raw_measurements.csv")
n_fixed = raw["source_folder"].isna().sum()
raw["source_folder"] = raw["source_folder"].fillna("baseline")
raw.to_csv(FOCUS / "all_raw_measurements.csv", index=False)
print(f"Fixed {n_fixed} NaN source_folder -> 'baseline' in all_raw_measurements.csv")

# ── Load MQTT ─────────────────────────────────────────────────────────────────
print("Loading MQTT log...")
mqtt = pd.read_csv(MQTT_PATH, usecols=["Timestamp", "data_quality", "sht_temp_c", "sht_rh", "bmp_pa"])
mqtt = mqtt[mqtt["data_quality"] == "valid"].copy()
mqtt["Timestamp"] = pd.to_datetime(mqtt["Timestamp"])
mqtt = mqtt.sort_values("Timestamp").reset_index(drop=True)
print(f"  {len(mqtt):,} valid readings")

# ── Re-derive timestamps for baseline trial rows ──────────────────────────────
trial = pd.read_csv(FOCUS / "all_trial_results.csv")
join_keys = ["source_folder", "trial_id", "liquid"]

raw_ts = raw[join_keys + ["timestamp"]].copy()
raw_ts["timestamp"] = pd.to_datetime(raw_ts["timestamp"])
mean_ts = raw_ts.groupby(join_keys)["timestamp"].mean().reset_index()
mean_ts = mean_ts.rename(columns={"timestamp": "_mean_ts"})

# Only update the baseline rows
baseline_mask = trial["source_folder"] == "baseline"
baseline_rows = trial[baseline_mask].copy()
# Drop existing env columns to avoid rename conflicts in merge_asof
existing_env = [c for c in baseline_rows.columns if c.startswith("env_")]
baseline_rows = baseline_rows.drop(columns=existing_env)
baseline_rows = baseline_rows.merge(mean_ts, on=join_keys, how="left")

missing = baseline_rows["_mean_ts"].isna().sum()
if missing:
    print(f"WARNING: {missing} baseline trial rows still have no timestamp after fix")

# Nearest-timestamp merge for baseline rows only
def attach_env_to_subset(df, ts_col):
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df_sorted = df.sort_values(ts_col).reset_index()
    merged = pd.merge_asof(
        df_sorted,
        mqtt.rename(columns={
            "Timestamp":  "env_timestamp",
            "sht_temp_c": "env_sht_temp_c",
            "sht_rh":     "env_sht_rh",
            "bmp_pa":     "env_bmp_pa",
        }),
        left_on=ts_col,
        right_on="env_timestamp",
        direction="nearest",
    )
    merged["env_time_delta_min"] = (
        (merged["env_timestamp"] - merged[ts_col]).abs().dt.total_seconds() / 60.0
    ).round(2)
    merged["env_within_30min"] = merged["env_time_delta_min"] <= THRESHOLD_MIN
    return merged.set_index("index").sort_index()

merged_baseline = attach_env_to_subset(baseline_rows, "_mean_ts")
merged_baseline = merged_baseline.drop(columns=["_mean_ts"])

env_cols = ["env_timestamp", "env_sht_temp_c", "env_sht_rh", "env_bmp_pa", "env_time_delta_min", "env_within_30min"]
for col in env_cols:
    trial.loc[baseline_mask, col] = merged_baseline[col].values

n_invalid = (~trial["env_within_30min"].astype(bool)).sum()
print(f"\nall_trial_results.csv: {len(trial)} rows | {n_invalid} rows outside 30-min window")
print("\nBaseline rows after fix:")
print(trial[baseline_mask][["liquid", "env_timestamp", "env_sht_temp_c", "env_sht_rh", "env_time_delta_min", "env_within_30min"]].to_string(index=False))

trial.to_csv(FOCUS / "all_trial_results.csv", index=False)
print("\nSaved all_trial_results.csv")

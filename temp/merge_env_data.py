"""
Merges MQTT environmental log (temperature, humidity, pressure) into all four
calibration data files using nearest-timestamp matching.

Adds to each file:
    env_timestamp       - nearest MQTT reading timestamp
    env_sht_temp_c      - SHT sensor temperature (C)
    env_sht_rh          - SHT sensor relative humidity (%)
    env_bmp_pa          - BMP pressure (Pa)
    env_time_delta_min  - absolute time difference to nearest reading (minutes)
    env_within_30min    - True if delta <= 30 minutes

For raw files: matches on the row's 'timestamp' column.
For trial files: uses mean timestamp of corresponding raw rows as the anchor.
"""
import pandas as pd
from pathlib import Path

FOCUS = Path("output/two_point_focus_glycerol_25_50_75_100")
MQTT_PATH = Path(r"C:\Users\Imaging Controller\Desktop\m5stack\mqtt_log.csv")
THRESHOLD_MIN = 30.0

# ── Load and prepare MQTT log ─────────────────────────────────────────────────
print("Loading MQTT log...")
mqtt = pd.read_csv(MQTT_PATH, usecols=["Timestamp", "data_quality", "sht_temp_c", "sht_rh", "bmp_pa"])
mqtt = mqtt[mqtt["data_quality"] == "valid"].copy()
mqtt["Timestamp"] = pd.to_datetime(mqtt["Timestamp"])
mqtt = mqtt.sort_values("Timestamp").reset_index(drop=True)
print(f"  {len(mqtt):,} valid readings | {mqtt['Timestamp'].iloc[0]} to {mqtt['Timestamp'].iloc[-1]}")


def attach_env(df: pd.DataFrame, ts_col: str, label: str) -> pd.DataFrame:
    """Nearest-timestamp join against MQTT log. Returns df with env columns added."""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df_sorted = df.sort_values(ts_col).reset_index(drop=True)
    orig_index = df_sorted.index  # will re-order at end

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

    n_invalid = (~merged["env_within_30min"]).sum()
    print(f"  {label}: {len(merged)} rows | {n_invalid} rows OUTSIDE 30-min window")
    if n_invalid > 0:
        print(f"    Max delta: {merged['env_time_delta_min'].max():.1f} min")
        bad = merged[~merged["env_within_30min"]][["liquid", ts_col, "env_timestamp", "env_time_delta_min"]]
        print(bad.to_string(index=False))

    return merged


# ── 1. two_point_combined_raw_data.csv ───────────────────────────────────────
print("\n--- two_point_combined_raw_data.csv ---")
raw_tp = pd.read_csv(FOCUS / "two_point_combined_raw_data.csv")
raw_tp = attach_env(raw_tp, "timestamp", "two_point_raw")
raw_tp.to_csv(FOCUS / "two_point_combined_raw_data.csv", index=False)
print("  Saved.")

# ── 2. two_point_combined_trial_data.csv ─────────────────────────────────────
print("\n--- two_point_combined_trial_data.csv ---")
trial_tp = pd.read_csv(FOCUS / "two_point_combined_trial_data.csv")
# Derive mean timestamp per (run_id, source_timestamp, trial_id, liquid) from raw
join_keys = ["run_id", "source_timestamp", "trial_id", "liquid"]
raw_tp_ts = raw_tp[join_keys + ["timestamp"]].copy()
raw_tp_ts["timestamp"] = pd.to_datetime(raw_tp_ts["timestamp"])
mean_ts = raw_tp_ts.groupby(join_keys)["timestamp"].mean().reset_index()
mean_ts = mean_ts.rename(columns={"timestamp": "_mean_ts"})
trial_tp = trial_tp.merge(mean_ts, on=join_keys, how="left")
missing = trial_tp["_mean_ts"].isna().sum()
if missing:
    print(f"  WARNING: {missing} trial rows could not be matched to a raw timestamp")
trial_tp = attach_env(trial_tp, "_mean_ts", "two_point_trial")
trial_tp = trial_tp.drop(columns=["_mean_ts"])
trial_tp.to_csv(FOCUS / "two_point_combined_trial_data.csv", index=False)
print("  Saved.")

# ── 3. all_raw_measurements.csv ───────────────────────────────────────────────
print("\n--- all_raw_measurements.csv ---")
raw_all = pd.read_csv(FOCUS / "all_raw_measurements.csv")
raw_all = attach_env(raw_all, "timestamp", "all_raw")
raw_all.to_csv(FOCUS / "all_raw_measurements.csv", index=False)
print("  Saved.")

# ── 4. all_trial_results.csv ──────────────────────────────────────────────────
print("\n--- all_trial_results.csv ---")
trial_all = pd.read_csv(FOCUS / "all_trial_results.csv")
join_keys_all = ["source_folder", "trial_id", "liquid"]
raw_all_ts = raw_all[join_keys_all + ["timestamp"]].copy()
raw_all_ts["timestamp"] = pd.to_datetime(raw_all_ts["timestamp"])
mean_ts_all = raw_all_ts.groupby(join_keys_all)["timestamp"].mean().reset_index()
mean_ts_all = mean_ts_all.rename(columns={"timestamp": "_mean_ts"})
trial_all = trial_all.merge(mean_ts_all, on=join_keys_all, how="left")
missing_all = trial_all["_mean_ts"].isna().sum()
if missing_all:
    print(f"  WARNING: {missing_all} trial rows have no raw timestamp — env columns will be NaN for those rows")
    print(trial_all[trial_all["_mean_ts"].isna()][join_keys_all].to_string(index=False))

# Split: match only rows that have a timestamp, leave the rest with NaN env cols
has_ts = trial_all["_mean_ts"].notna()
matched = attach_env(trial_all[has_ts].copy(), "_mean_ts", "all_trial (matched)")
unmatched = trial_all[~has_ts].copy()
for col in ["env_timestamp", "env_sht_temp_c", "env_sht_rh", "env_bmp_pa", "env_time_delta_min", "env_within_30min"]:
    unmatched[col] = float("nan")
trial_all = pd.concat([matched, unmatched], ignore_index=True)
trial_all = trial_all.drop(columns=["_mean_ts"])
trial_all.to_csv(FOCUS / "all_trial_results.csv", index=False)
print("  Saved.")

print("\nDone.")

"""Fix: process only all_trial_results.csv (the other 3 files already have env columns)."""
import pandas as pd
from pathlib import Path

FOCUS = Path("output/two_point_focus_glycerol_25_50_75_100")
MQTT_PATH = Path(r"C:\Users\Imaging Controller\Desktop\m5stack\mqtt_log.csv")
THRESHOLD_MIN = 30.0

print("Loading MQTT log...")
mqtt = pd.read_csv(MQTT_PATH, usecols=["Timestamp", "data_quality", "sht_temp_c", "sht_rh", "bmp_pa"])
mqtt = mqtt[mqtt["data_quality"] == "valid"].copy()
mqtt["Timestamp"] = pd.to_datetime(mqtt["Timestamp"])
mqtt = mqtt.sort_values("Timestamp").reset_index(drop=True)
print(f"  {len(mqtt):,} valid readings")


def attach_env(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    orig_order = df.index.tolist()
    df_sorted = df.sort_values(ts_col).reset_index(drop=True)

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
    return merged


print("\n--- all_trial_results.csv ---")
trial_all = pd.read_csv(FOCUS / "all_trial_results.csv")
raw_all   = pd.read_csv(FOCUS / "all_raw_measurements.csv")

join_keys_all = ["source_folder", "trial_id", "liquid"]
raw_ts = raw_all[join_keys_all + ["timestamp"]].copy()
raw_ts["timestamp"] = pd.to_datetime(raw_ts["timestamp"])
mean_ts_all = raw_ts.groupby(join_keys_all)["timestamp"].mean().reset_index()
mean_ts_all = mean_ts_all.rename(columns={"timestamp": "_mean_ts"})

trial_all = trial_all.merge(mean_ts_all, on=join_keys_all, how="left")
missing = trial_all["_mean_ts"].isna().sum()
print(f"  Rows missing a raw timestamp: {missing}")
if missing:
    print(trial_all[trial_all["_mean_ts"].isna()][join_keys_all + ["volume_target_ul"]].to_string(index=False))

has_ts = trial_all["_mean_ts"].notna()
matched   = attach_env(trial_all[has_ts].copy(), "_mean_ts")
unmatched = trial_all[~has_ts].copy()
for col in ["env_timestamp", "env_sht_temp_c", "env_sht_rh", "env_bmp_pa", "env_time_delta_min", "env_within_30min"]:
    unmatched[col] = float("nan")

trial_all = pd.concat([matched, unmatched], ignore_index=True)
trial_all = trial_all.drop(columns=["_mean_ts"])

within = trial_all["env_within_30min"].map(lambda x: bool(x) if x == x else False)
n_invalid = (~within).sum()
print(f"  {len(trial_all)} rows | {n_invalid} rows OUTSIDE 30-min window (including {missing} with no timestamp)")
if n_invalid > missing:
    bad = trial_all[within == False][["liquid", "env_timestamp", "env_time_delta_min"]]
    print(bad.to_string(index=False))

trial_all.to_csv(FOCUS / "all_trial_results.csv", index=False)
print("  Saved.")
print("\nDone.")

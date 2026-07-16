"""
Full env merge (clean run) - adds env_timestamp, env_sht_temp_c, env_sht_rh,
env_bmp_pa, env_time_delta_min, env_within_30min to all 4 data files.

Safe to re-run: drops any existing env_* columns first.
"""
import pandas as pd
from pathlib import Path

FOCUS = Path("output/two_point_focus_glycerol_25_50_75_100")
MQTT_PATH = Path(r"C:\Users\Imaging Controller\Desktop\m5stack\mqtt_log.csv")
THRESHOLD_MIN = 30.0

ENV_COLS = ["env_timestamp", "env_sht_temp_c", "env_sht_rh", "env_bmp_pa",
            "env_time_delta_min", "env_within_30min"]

# ── Load MQTT ─────────────────────────────────────────────────────────────────
print("Loading MQTT log...")
mqtt = pd.read_csv(MQTT_PATH, usecols=["Timestamp", "data_quality", "sht_temp_c", "sht_rh", "bmp_pa"])
mqtt = mqtt[mqtt["data_quality"] == "valid"].copy()
mqtt["Timestamp"] = pd.to_datetime(mqtt["Timestamp"])
mqtt = mqtt.sort_values("Timestamp").reset_index(drop=True)
print(f"  {len(mqtt):,} valid readings | {mqtt['Timestamp'].iloc[0]} -> {mqtt['Timestamp'].iloc[-1]}")


def nearest_env(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Nearest-timestamp join. Returns df with env columns appended."""
    df = df.copy()
    # Drop existing env cols to avoid conflicts
    df = df.drop(columns=[c for c in ENV_COLS if c in df.columns], errors="ignore")
    df[ts_col] = pd.to_datetime(df[ts_col])
    original_index = df.index
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


def report(df: pd.DataFrame, label: str):
    n_out = (~df["env_within_30min"].astype(bool)).sum()
    max_d = df["env_time_delta_min"].max()
    print(f"  {label}: {len(df)} rows | outside 30min: {n_out} | max delta: {max_d:.2f} min")
    if n_out > 0:
        bad = df[~df["env_within_30min"].astype(bool)]
        print(bad[["liquid", "env_timestamp", "env_time_delta_min"]].to_string(index=False))


# ── 1. all_raw_measurements.csv  (fix NaN source_folder first) ───────────────
print("\n--- all_raw_measurements.csv ---")
raw_all = pd.read_csv(FOCUS / "all_raw_measurements.csv")
n_fixed = raw_all["source_folder"].isna().sum()
raw_all["source_folder"] = raw_all["source_folder"].fillna("baseline")
print(f"  Fixed {n_fixed} NaN source_folder -> 'baseline'")
raw_all = nearest_env(raw_all, "timestamp")
report(raw_all, "all_raw")
raw_all.to_csv(FOCUS / "all_raw_measurements.csv", index=False)
print("  Saved.")

# ── 2. two_point_combined_raw_data.csv ────────────────────────────────────────
print("\n--- two_point_combined_raw_data.csv ---")
raw_tp = pd.read_csv(FOCUS / "two_point_combined_raw_data.csv")
raw_tp = nearest_env(raw_tp, "timestamp")
report(raw_tp, "two_point_raw")
raw_tp.to_csv(FOCUS / "two_point_combined_raw_data.csv", index=False)
print("  Saved.")

# ── 3. all_trial_results.csv  (derive mean timestamp from raw rows) ───────────
print("\n--- all_trial_results.csv ---")
trial_all = pd.read_csv(FOCUS / "all_trial_results.csv")
trial_all = trial_all.drop(columns=[c for c in ENV_COLS if c in trial_all.columns], errors="ignore")

join_keys = ["source_folder", "trial_id", "liquid"]
mean_ts = (
    raw_all[join_keys + ["timestamp"]]
    .copy()
    .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]))
    .groupby(join_keys)["timestamp"].mean()
    .reset_index()
    .rename(columns={"timestamp": "_mean_ts"})
)
trial_all = trial_all.merge(mean_ts, on=join_keys, how="left")
n_missing = trial_all["_mean_ts"].isna().sum()
if n_missing:
    print(f"  {n_missing} trial rows have no matching raw rows (will get NaN env):")
    print(trial_all[trial_all["_mean_ts"].isna()][join_keys + ["volume_target_ul"]].to_string(index=False))

has_ts = trial_all["_mean_ts"].notna()
matched   = nearest_env(trial_all[has_ts].copy(), "_mean_ts").drop(columns=["_mean_ts"])
unmatched = trial_all[~has_ts].drop(columns=["_mean_ts"])
for c in ENV_COLS:
    unmatched[c] = None

trial_all = pd.concat([matched, unmatched], ignore_index=True)
trial_all["env_within_30min"] = trial_all["env_within_30min"].astype("boolean")
report(trial_all, "all_trial")
trial_all.to_csv(FOCUS / "all_trial_results.csv", index=False)
print("  Saved.")

# ── 4. two_point_combined_trial_data.csv  (derive mean ts from raw_tp rows) ──
print("\n--- two_point_combined_trial_data.csv ---")
trial_tp = pd.read_csv(FOCUS / "two_point_combined_trial_data.csv")
trial_tp = trial_tp.drop(columns=[c for c in ENV_COLS if c in trial_tp.columns], errors="ignore")

join_keys_tp = ["run_id", "source_timestamp", "trial_id", "liquid"]
mean_ts_tp = (
    raw_tp[join_keys_tp + ["timestamp"]]
    .copy()
    .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]))
    .groupby(join_keys_tp)["timestamp"].mean()
    .reset_index()
    .rename(columns={"timestamp": "_mean_ts"})
)
trial_tp = trial_tp.merge(mean_ts_tp, on=join_keys_tp, how="left")
n_missing_tp = trial_tp["_mean_ts"].isna().sum()
if n_missing_tp:
    print(f"  WARNING: {n_missing_tp} rows have no matching raw timestamp")

has_ts_tp = trial_tp["_mean_ts"].notna()
matched_tp   = nearest_env(trial_tp[has_ts_tp].copy(), "_mean_ts").drop(columns=["_mean_ts"])
unmatched_tp = trial_tp[~has_ts_tp].drop(columns=["_mean_ts"])
for c in ENV_COLS:
    unmatched_tp[c] = None

trial_tp = pd.concat([matched_tp, unmatched_tp], ignore_index=True)
trial_tp["env_within_30min"] = trial_tp["env_within_30min"].astype("boolean")
report(trial_tp, "two_point_trial")
trial_tp.to_csv(FOCUS / "two_point_combined_trial_data.csv", index=False)
print("  Saved.")

print("\n=== All done ===")

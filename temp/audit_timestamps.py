"""
Timestamp audit across all calibration data files and the MQTT log.
Reports format, range, nulls, and any anomalies. No modifications made.
"""
import pandas as pd
from pathlib import Path

FOCUS = Path("output/two_point_focus_glycerol_25_50_75_100")
MQTT_PATH = Path(r"C:\Users\Imaging Controller\Desktop\m5stack\mqtt_log.csv")

SEP = "=" * 70

def audit_ts_col(series: pd.Series, col_name: str):
    print(f"  [{col_name}]")
    print(f"    Total rows : {len(series)}")
    print(f"    Nulls/NaN  : {series.isna().sum()}")
    non_null = series.dropna().astype(str)
    print(f"    First value: {non_null.iloc[0] if len(non_null) else 'N/A'}")
    print(f"    Last value : {non_null.iloc[-1] if len(non_null) else 'N/A'}")
    # Try parse
    try:
        parsed = pd.to_datetime(series, errors="coerce")
        bad = parsed.isna() & series.notna()
        print(f"    Parse fails: {bad.sum()}")
        valid = parsed.dropna()
        if len(valid):
            print(f"    Parsed min : {valid.min()}")
            print(f"    Parsed max : {valid.max()}")
            # Check for timezone info
            has_tz = str(valid.iloc[0].tzinfo) if valid.iloc[0].tzinfo else "naive (no tz)"
            print(f"    Timezone   : {has_tz}")
    except Exception as e:
        print(f"    Parse error: {e}")
    # Format distribution
    fmt_dist = non_null.str[:10].value_counts()
    if len(fmt_dist) > 1:
        print(f"    *** MIXED DATE PREFIXES ***")
        print(fmt_dist.to_string())
    else:
        print(f"    Date prefix: {fmt_dist.index[0] if len(fmt_dist) else 'N/A'} ({fmt_dist.iloc[0] if len(fmt_dist) else 0} rows)")

# ── 1. all_raw_measurements.csv ───────────────────────────────────────────────
print(SEP)
print("FILE: all_raw_measurements.csv")
print(SEP)
df = pd.read_csv(FOCUS / "all_raw_measurements.csv")
print(f"Shape: {df.shape}")
ts_cols = [c for c in df.columns if "time" in c.lower() or "stamp" in c.lower()]
print(f"Timestamp-like cols: {ts_cols}")
for c in ts_cols:
    audit_ts_col(df[c], c)
    print()

# ── 2. all_trial_results.csv ──────────────────────────────────────────────────
print(SEP)
print("FILE: all_trial_results.csv")
print(SEP)
df2 = pd.read_csv(FOCUS / "all_trial_results.csv")
print(f"Shape: {df2.shape}")
ts_cols2 = [c for c in df2.columns if "time" in c.lower() or "stamp" in c.lower()]
print(f"Timestamp-like cols: {ts_cols2}")
for c in ts_cols2:
    audit_ts_col(df2[c], c)
    print()

# ── 3. two_point_combined_raw_data.csv ───────────────────────────────────────
print(SEP)
print("FILE: two_point_combined_raw_data.csv")
print(SEP)
df3 = pd.read_csv(FOCUS / "two_point_combined_raw_data.csv")
print(f"Shape: {df3.shape}")
ts_cols3 = [c for c in df3.columns if "time" in c.lower() or "stamp" in c.lower()]
ts_cols3 = [c for c in ts_cols3 if "hardware" not in c]
print(f"Timestamp-like cols: {ts_cols3}")
for c in ts_cols3:
    audit_ts_col(df3[c], c)
    print()

# ── 4. two_point_combined_trial_data.csv ─────────────────────────────────────
print(SEP)
print("FILE: two_point_combined_trial_data.csv")
print(SEP)
df4 = pd.read_csv(FOCUS / "two_point_combined_trial_data.csv")
print(f"Shape: {df4.shape}")
ts_cols4 = [c for c in df4.columns if "time" in c.lower() or "stamp" in c.lower()]
ts_cols4 = [c for c in ts_cols4 if "hardware" not in c]
print(f"Timestamp-like cols: {ts_cols4}")
for c in ts_cols4:
    audit_ts_col(df4[c], c)
    print()

# ── 5. MQTT log ───────────────────────────────────────────────────────────────
print(SEP)
print("FILE: mqtt_log.csv")
print(SEP)
mqtt = pd.read_csv(MQTT_PATH, usecols=["Timestamp", "data_quality"])
print(f"Shape: {mqtt.shape}")
audit_ts_col(mqtt["Timestamp"], "Timestamp")
print()
valid_count = (mqtt["data_quality"] == "valid").sum()
print(f"  data_quality='valid': {valid_count:,} / {len(mqtt):,} rows")

# ── 6. Cross-check: do experiment dates fall within MQTT range? ───────────────
print()
print(SEP)
print("CROSS-CHECK: Experiment timestamps vs MQTT range")
print(SEP)
mqtt_ts = pd.to_datetime(mqtt["Timestamp"], errors="coerce").dropna()
mqtt_min, mqtt_max = mqtt_ts.min(), mqtt_ts.max()
print(f"MQTT range: {mqtt_min}  -->  {mqtt_max}")
print()
for name, df_x, col in [
    ("all_raw", df, "timestamp"),
    ("two_point_raw", df3, "timestamp"),
]:
    exp_ts = pd.to_datetime(df_x[col], errors="coerce").dropna()
    exp_min, exp_max = exp_ts.min(), exp_ts.max()
    in_range = ((exp_ts >= mqtt_min) & (exp_ts <= mqtt_max)).sum()
    out_range = len(exp_ts) - in_range
    print(f"  {name}: {exp_min}  -->  {exp_max}")
    print(f"    In MQTT range: {in_range}/{len(exp_ts)}  |  Out of range: {out_range}")
    print()

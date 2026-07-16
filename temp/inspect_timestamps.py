import pandas as pd

print('=== two_point_combined_raw_data.csv ===')
df = pd.read_csv('output/two_point_focus_glycerol_25_50_75_100/two_point_combined_raw_data.csv')
ts_cols = [c for c in df.columns if 'time' in c.lower() or 'stamp' in c.lower()]
print('Timestamp-like cols:', ts_cols)
for c in ts_cols:
    print(f'  {c}: first={df[c].iloc[0]}  last={df[c].iloc[-1]}')
print()

print('=== all_raw_measurements.csv ===')
df2 = pd.read_csv('output/two_point_focus_glycerol_25_50_75_100/all_raw_measurements.csv')
ts_cols2 = [c for c in df2.columns if 'time' in c.lower() or 'stamp' in c.lower()]
print('Timestamp-like cols:', ts_cols2)
for c in ts_cols2:
    print(f'  {c}: first={df2[c].iloc[0]}  last={df2[c].iloc[-1]}')
print()

print('=== all_trial_results.csv ===')
df3 = pd.read_csv('output/two_point_focus_glycerol_25_50_75_100/all_trial_results.csv')
ts_cols3 = [c for c in df3.columns if 'time' in c.lower() or 'stamp' in c.lower()]
print('Timestamp-like cols:', ts_cols3)
for c in ts_cols3:
    print(f'  {c}: first={df3[c].iloc[0]}  last={df3[c].iloc[-1]}')

import pandas as pd
import numpy as np

df = pd.read_csv('calibration_modular_v2/output/run_1780417119_ethanol/trial_results.csv')
for c in ['deviation_pct','precision_cv_pct','duration_mean_s','measurement_count']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['trial_num'] = df['trial_id'].str.extract(r'(\d+)').astype(int)
df = df.sort_values('trial_num').reset_index(drop=True)
df_valid = df[df['measurement_count'] >= 2].reset_index(drop=True)

acc_std  = max(df_valid['deviation_pct'].std(), 0.1)
prec_std = max(df_valid['precision_cv_pct'].std(), 0.1)
time_std = max(df_valid['duration_mean_s'].std(), 1.0)

df_valid['score'] = (0.4*df_valid['deviation_pct']/acc_std*100 +
                     0.5*df_valid['precision_cv_pct']/prec_std*100 +
                     0.1*df_valid['duration_mean_s']/time_std*100)

pd.set_option('display.width', 200)
print(df_valid[['trial_num','deviation_pct','precision_cv_pct','duration_mean_s','measurement_count','score']].to_string())
print()
print('Best trial:')
print(df_valid.loc[df_valid['score'].idxmin(), ['trial_num','deviation_pct','precision_cv_pct','duration_mean_s','score']])

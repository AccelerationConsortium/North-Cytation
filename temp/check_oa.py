import pandas as pd
df = pd.read_csv('output/two_point_focus_glycerol_25_50_75_100/two_point_combined_raw_data.csv')
hw_oa = 'hardware_parameters_overaspirate_vol'
print('hardware_parameters_overaspirate_vol exists:', hw_oa in df.columns)
print('calibration_overaspirate_vol exists:', 'calibration_overaspirate_vol' in df.columns)
print()
print(df[['liquid','trial_id','calibration_overaspirate_vol']].groupby(['liquid','trial_id'])['calibration_overaspirate_vol'].mean().round(6))

import pandas as pd
t = pd.read_csv('output/two_point_focus_glycerol_25_50_75_100/two_point_combined_trial_data.csv')
print('two_point_combined_trial_data cols:', list(t.columns))
print('rows:', len(t))

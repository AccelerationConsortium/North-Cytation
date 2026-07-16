import pandas as pd
raw = pd.read_csv('output/two_point_focus_glycerol_25_50_75_100/all_raw_measurements.csv')
trial = pd.read_csv('output/two_point_focus_glycerol_25_50_75_100/all_trial_results.csv')
print('all_raw_measurements:', raw.shape)
print('cols:', list(raw.columns))
print()
print('all_trial_results:', trial.shape)
print('cols:', list(trial.columns))
print()
liq_col = 'liquid' if 'liquid' in raw.columns else 'liquid_name' if 'liquid_name' in raw.columns else None
if liq_col:
    print('raw liquids:', sorted(raw[liq_col].unique()))
liq_col2 = 'liquid' if 'liquid' in trial.columns else 'liquid_name' if 'liquid_name' in trial.columns else None
if liq_col2:
    print('trial liquids:', sorted(trial[liq_col2].unique()))

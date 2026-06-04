import pandas as pd
df = pd.read_csv('calibration_modular_v2/output/run_1780417119_ethanol/trial_results.csv')
print('ALL COLUMNS:')
for c in df.columns:
    print(f"  {c}")
print()
print('FULL DATA (all rows, all columns):')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 40)
print(df.to_string())

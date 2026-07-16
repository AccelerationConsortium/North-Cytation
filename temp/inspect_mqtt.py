import pandas as pd
df = pd.read_csv(r'C:\Users\Imaging Controller\Desktop\m5stack\mqtt_log.csv')
print('Shape:', df.shape)
print('Cols:', list(df.columns))
print()
print(df.head(8).to_string())
print()
print('dtypes:')
print(df.dtypes)
print()
print('Timestamp range:')
ts_col = df.columns[0]
print(f'  first: {df[ts_col].iloc[0]}')
print(f'  last:  {df[ts_col].iloc[-1]}')

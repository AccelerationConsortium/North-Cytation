import pandas as pd
raw = pd.read_csv('output/two_point_focus_glycerol_25_50_75_100/all_raw_measurements.csv')
bad_mask = raw['timestamp'].astype(str).str.match(r'^\d{2}:\d{2}')
bad = raw[bad_mask]
print('Malformed timestamp rows:', len(bad))
if len(bad):
    print(bad[['source_folder','liquid','timestamp','env_timestamp']].head(10).to_string())
print()
print('Unique timestamp formats (first 10 chars):')
print(raw['timestamp'].astype(str).str[:10].value_counts().head(20))

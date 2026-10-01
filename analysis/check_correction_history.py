import pandas as pd
import csv

# parameter_corrections.csv has a known schema-drift bug (header defines 10 columns,
# but ~196/270 rows have 13 - three extra unlabeled env columns appended later without
# updating the header) - parse manually, keeping only the first 10 (named) fields.
HEADER = ['timestamp', 'session_id', 'validation_type', 'liquid_type', 'target_volume_ml',
          'target_volume_ul', 'old_overaspirate_ml', 'new_overaspirate_ml', 'correction_ml', 'correction_ul']
with open('pipetting_data/parameter_corrections.csv', newline='') as f:
    rows = list(csv.reader(f))
rows = rows[1:]  # drop header line
records = [row[:10] for row in rows]
df = pd.DataFrame(records, columns=HEADER)
for col in ['target_volume_ml', 'target_volume_ul', 'old_overaspirate_ml', 'new_overaspirate_ml', 'correction_ml', 'correction_ul']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print('Total correction log entries:', len(df))
print('validation_type value counts:')
print(df['validation_type'].value_counts())
print()

targets = {
    'water': [0.05, 0.1, 0.15, 0.5, 0.8, 0.9],
    'SDS': [0.5, 0.8],
}

for liquid, vols in targets.items():
    sub = df[(df['liquid_type'] == liquid) & (df['target_volume_ml'].isin(vols))].copy()
    print('=' * 78)
    print(f'{liquid}: {len(sub)} correction attempts at volumes {vols}')
    if len(sub) == 0:
        print('  (NO correction attempts ever logged at these volumes)')
        continue
    sub['applied'] = ~sub['validation_type'].str.contains('rejected')
    print(sub.groupby('target_volume_ml')['applied'].agg(['sum', 'count']))
    print()
    print(sub[['timestamp', 'target_volume_ml', 'validation_type', 'old_overaspirate_ml', 'new_overaspirate_ml']].to_string(index=False))

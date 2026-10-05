import pandas as pd
import numpy as np
import csv

HEADER = ['timestamp', 'session_id', 'validation_type', 'liquid_type', 'target_volume_ml',
          'target_volume_ul', 'old_overaspirate_ml', 'new_overaspirate_ml', 'correction_ml', 'correction_ul']
with open('pipetting_data/parameter_corrections.csv', newline='') as f:
    rows = list(csv.reader(f))
rows = rows[1:]
records = [row[:10] for row in rows]
df = pd.DataFrame(records, columns=HEADER)
for col in ['target_volume_ml', 'old_overaspirate_ml', 'new_overaspirate_ml', 'correction_ml']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

for liquid, vol in [('water', 0.8), ('water', 0.5), ('SDS', 0.5), ('SDS', 0.8)]:
    sub = df[(df['liquid_type'] == liquid) & (df['target_volume_ml'] == vol)].sort_values('timestamp')
    if len(sub) < 5:
        continue
    new_vals = sub['new_overaspirate_ml'].values
    corrections = sub['correction_ml'].values

    # Lag-1 autocorrelation of the resulting OAV series: strongly negative => oscillation/overshoot,
    # near zero => random walk / independent noise, strongly positive => smooth drift.
    def lag1_autocorr(x):
        x = x - x.mean()
        return np.sum(x[:-1] * x[1:]) / np.sum(x ** 2)

    ac_val = lag1_autocorr(new_vals)

    # Sign of consecutive corrections: if the process is overshooting/oscillating, a positive
    # correction tends to be followed by a negative one (and vice versa) more often than not.
    signs = np.sign(corrections)
    sign_flips = np.sum(signs[:-1] != signs[1:]) / max(len(signs) - 1, 1)

    print(f'{liquid} {vol}mL (n={len(sub)}): lag-1 autocorr(new_OAV)={ac_val:+.3f}, '
          f'fraction of sign flips between consecutive corrections={sign_flips:.2f}')
    print('  new_overaspirate_ml sequence (uL):', [f'{v*1000:.1f}' for v in new_vals])

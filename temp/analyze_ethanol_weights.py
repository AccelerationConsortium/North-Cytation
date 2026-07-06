import pandas as pd
import numpy as np

df = pd.read_csv('calibration_modular_v2/output/run_1780412080_ethanol/trial_results.csv')
for c in ['deviation_pct','precision_cv_pct','duration_mean_s','measurement_count']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['trial_num'] = df['trial_id'].str.extract(r'(\d+)').astype(int)
df = df.sort_values('trial_num').reset_index(drop=True)

acc_std  = max(df['deviation_pct'].std(), 0.1)
prec_std = max(df['precision_cv_pct'].std(), 0.1)
time_std = max(df['duration_mean_s'].std(), 1.0)

print(f"Population stds: acc={acc_std:.2f}, prec={prec_std:.2f}, time={time_std:.2f}")
print()

df['score_equal']   = 1.0*df['deviation_pct']/acc_std + 1.0*df['precision_cv_pct']/prec_std + 1.0*df['duration_mean_s']/time_std
df['score_correct'] = 0.4*df['deviation_pct']/acc_std + 0.5*df['precision_cv_pct']/prec_std + 0.1*df['duration_mean_s']/time_std

for name, col in [('EQUAL 1:1:1 (current plots)', 'score_equal'), ('CORRECT 0.4:0.5:0.1 (v2 config)', 'score_correct')]:
    best_score = np.inf
    waypoints = []
    for i, row in df.iterrows():
        if row[col] < best_score:
            best_score = row[col]
            waypoints.append(row)
    best = pd.DataFrame(waypoints)
    print(f"--- {name} ---")
    print(f"Trajectory endpoint: Trial {int(best.iloc[-1]['trial_num'])} | "
          f"dev={best.iloc[-1]['deviation_pct']:.2f}% | "
          f"prec={best.iloc[-1]['precision_cv_pct']:.2f}% | "
          f"time={best.iloc[-1]['duration_mean_s']:.1f}s")
    print(best[['trial_num','deviation_pct','precision_cv_pct','duration_mean_s',col]].to_string())
    print()

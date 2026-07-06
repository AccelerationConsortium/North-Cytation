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

print(f"Population stds used for normalization:")
print(f"  acc_std  = {acc_std:.4f}  (deviation_pct)")
print(f"  prec_std = {prec_std:.4f}  (precision_cv_pct)")
print(f"  time_std = {time_std:.4f}  (duration_mean_s)")
print()

df_valid['acc_term']   = 0.4 * df_valid['deviation_pct']    / acc_std  * 100
df_valid['prec_term']  = 0.5 * df_valid['precision_cv_pct'] / prec_std * 100
df_valid['time_term']  = 0.1 * df_valid['duration_mean_s']  / time_std * 100
df_valid['score']      = df_valid['acc_term'] + df_valid['prec_term'] + df_valid['time_term']

# Sort by score to show top 10
top10 = df_valid.nsmallest(10, 'score')
print("Top 10 trials by composite score:")
print(f"{'T#':>3}  {'dev%':>7}  {'cv%':>7}  {'time_s':>7}  | {'acc_term':>9}  {'prec_term':>10}  {'time_term':>10}  | {'score':>8}")
print("-" * 85)
for _, r in top10.iterrows():
    print(f"{int(r['trial_num']):>3}  {r['deviation_pct']:>7.3f}  {r['precision_cv_pct']:>7.3f}  {r['duration_mean_s']:>7.1f}  | "
          f"{r['acc_term']:>9.2f}  {r['prec_term']:>10.2f}  {r['time_term']:>10.2f}  | {r['score']:>8.2f}")

print()
print("Full breakdown for every trial in top 10:")
print(f"  Columns: raw values | normalized*100 (pre-weight) | weighted terms (w*norm*100) | final score | % contribution")
print()
print(f"{'T#':>3}  {'dev%':>7}  {'cv%':>7}  {'t(s)':>6}  |  "
      f"{'norm_acc*100':>12}  {'norm_prec*100':>13}  {'norm_time*100':>13}  |  "
      f"{'w=0.4':>6}  {'w=0.5':>6}  {'w=0.1':>6}  |  {'SCORE':>7}  |  acc%  prec%  time%")
print("-" * 145)
for _, r in top10.iterrows():
    norm_dev  = r['deviation_pct']    / acc_std
    norm_cv   = r['precision_cv_pct'] / prec_std
    norm_t    = r['duration_mean_s']  / time_std
    pct_acc   = r['acc_term']  / r['score'] * 100
    pct_prec  = r['prec_term'] / r['score'] * 100
    pct_time  = r['time_term'] / r['score'] * 100
    print(f"{int(r['trial_num']):>3}  {r['deviation_pct']:>7.3f}  {r['precision_cv_pct']:>7.3f}  {r['duration_mean_s']:>6.1f}  |  "
          f"{norm_dev*100:>12.2f}  {norm_cv*100:>13.2f}  {norm_t*100:>13.2f}  |  "
          f"{r['acc_term']:>6.2f}  {r['prec_term']:>6.2f}  {r['time_term']:>6.2f}  |  "
          f"{r['score']:>7.2f}  |  {pct_acc:>4.0f}%  {pct_prec:>4.0f}%  {pct_time:>4.0f}%")

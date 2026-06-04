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

print("=" * 60)
print("STEP 1: POPULATION STATISTICS (n=2+ trials only)")
print("=" * 60)
for label, col, std in [
    ("deviation_pct   (accuracy)", "deviation_pct",    acc_std),
    ("precision_cv_pct (precision)", "precision_cv_pct", prec_std),
    ("duration_mean_s  (time)",    "duration_mean_s",  time_std),
]:
    print(f"\n  {label}")
    print(f"    min    = {df_valid[col].min():.4f}")
    print(f"    max    = {df_valid[col].max():.4f}")
    print(f"    mean   = {df_valid[col].mean():.4f}")
    print(f"    std    = {std:.4f}  <-- used as divisor")
    print(f"    range  = {df_valid[col].max() - df_valid[col].min():.4f}")
    print(f"    mean/std ratio = {df_valid[col].mean()/std:.2f}  <-- this is the baseline normalized*100 scale")

print()
print("=" * 60)
print("STEP 2: FOR EACH TRIAL - FULL CALCULATION CHAIN")
print("  formula: score = 0.4*(dev/acc_std)*100 + 0.5*(cv/prec_std)*100 + 0.1*(t/time_std)*100")
print("=" * 60)

df_valid['n_acc']  = df_valid['deviation_pct']    / acc_std
df_valid['n_prec'] = df_valid['precision_cv_pct'] / prec_std
df_valid['n_time'] = df_valid['duration_mean_s']  / time_std
df_valid['s_acc']  = df_valid['n_acc']  * 100
df_valid['s_prec'] = df_valid['n_prec'] * 100
df_valid['s_time'] = df_valid['n_time'] * 100
df_valid['w_acc']  = 0.4 * df_valid['s_acc']
df_valid['w_prec'] = 0.5 * df_valid['s_prec']
df_valid['w_time'] = 0.1 * df_valid['s_time']
df_valid['score']  = df_valid['w_acc'] + df_valid['w_prec'] + df_valid['w_time']

top10 = df_valid.nsmallest(10, 'score')

for _, r in top10.iterrows():
    pct_acc  = r['w_acc']  / r['score'] * 100
    pct_prec = r['w_prec'] / r['score'] * 100
    pct_time = r['w_time'] / r['score'] * 100
    print(f"\n  Trial {int(r['trial_num'])}  |  strategy={df.loc[df['trial_num']==int(r['trial_num']),'strategy'].values[0]}")
    print(f"    RAW:        dev={r['deviation_pct']:7.4f}%    cv={r['precision_cv_pct']:7.4f}%    time={r['duration_mean_s']:6.2f}s")
    print(f"    /std:       dev={r['n_acc']:7.4f}       cv={r['n_prec']:7.4f}       time={r['n_time']:6.4f}")
    print(f"    x100:       dev={r['s_acc']:7.2f}       cv={r['s_prec']:7.2f}       time={r['s_time']:6.2f}")
    print(f"    x weight:   dev={r['w_acc']:7.2f}(x0.4)  cv={r['w_prec']:7.2f}(x0.5)  time={r['w_time']:6.2f}(x0.1)")
    print(f"    SCORE = {r['w_acc']:.2f} + {r['w_prec']:.2f} + {r['w_time']:.2f} = {r['score']:.2f}  |  acc:{pct_acc:.0f}%  prec:{pct_prec:.0f}%  time:{pct_time:.0f}%")

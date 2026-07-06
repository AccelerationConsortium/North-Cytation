import pandas as pd
import numpy as np

df = pd.read_csv('calibration_modular_v2/output/run_1780417119_ethanol/trial_results.csv')
for c in ['deviation_pct','precision_cv_pct','duration_mean_s','measurement_count']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['trial_num'] = df['trial_id'].str.extract(r'(\d+)').astype(int)
df = df.sort_values('trial_num').reset_index(drop=True)
df_valid = df[df['measurement_count'] >= 2].reset_index(drop=True)

# ── Desirability scoring (new approach) ─────────────────────────────────────
# Ethanol run: 50uL target -> tolerance = 3% (from experiment_config.yaml)
TOLERANCE_PCT = 3.0
S = 2  # shape: convex curve (diminishing returns near 0)

def desirability(metric, tolerance, s=2):
    """1=perfect, 0.5=at tolerance, still >0 beyond tolerance (soft boundary)."""
    return 1.0 / (1.0 + (metric / tolerance) ** s)

df_valid['d_acc']  = df_valid['deviation_pct'].apply(lambda x: desirability(x, TOLERANCE_PCT, S))
df_valid['d_prec'] = df_valid['precision_cv_pct'].apply(lambda x: desirability(x, TOLERANCE_PCT, S))

t_min = df_valid['duration_mean_s'].min()
t_max = df_valid['duration_mean_s'].max()
df_valid['d_time'] = (t_max - df_valid['duration_mean_s']) / (t_max - t_min)

w_acc, w_prec, w_time = 0.4, 0.5, 0.1
df_valid['desirability'] = w_acc * df_valid['d_acc'] + w_prec * df_valid['d_prec'] + w_time * df_valid['d_time']

print(f"Desirability scoring  (higher=better, ideal=1.0)")
print(f"Tolerance={TOLERANCE_PCT}%, shape s={S}, weights acc={w_acc} prec={w_prec} time={w_time}")
print(f"d_acc = 1/(1+(dev/tol)^s)   d_prec = 1/(1+(cv/tol)^s)   d_time = (tmax-t)/(tmax-tmin)")
print()
print(f"{'T#':>3}  {'dev%':>6}  {'cv%':>6}  {'t(s)':>6}  |  {'d_acc':>6}  {'d_prec':>7}  {'d_time':>7}  |  {'DESIR':>7}")
print("-"*75)
for _, r in df_valid.sort_values('desirability', ascending=False).iterrows():
    print(f"{int(r['trial_num']):>3}  {r['deviation_pct']:>6.3f}  {r['precision_cv_pct']:>6.3f}  {r['duration_mean_s']:>6.1f}  |  "
          f"{r['d_acc']:>6.3f}  {r['d_prec']:>7.3f}  {r['d_time']:>7.3f}  |  "
          f"{r['desirability']:>7.4f}")
print("-"*75)
print(f"{'':>3}  {'':>6}  {'':>6}  {'':>6}  |  {'':>6}  {'IDEAL':>7}  {'':>7}  |  {'1.0000':>7}")
print()
print("-- Old stdev scoring (lower=better) --")

acc_std  = max(df_valid['deviation_pct'].std(), 0.1)
prec_std = max(df_valid['precision_cv_pct'].std(), 0.1)
time_std = max(df_valid['duration_mean_s'].std(), 1.0)

df_valid['s_acc']  = df_valid['deviation_pct']    / acc_std  * 100
df_valid['s_prec'] = df_valid['precision_cv_pct'] / prec_std * 100
df_valid['s_time'] = df_valid['duration_mean_s']  / time_std * 100
df_valid['w_acc']  = 0.4 * df_valid['s_acc']
df_valid['w_prec'] = 0.5 * df_valid['s_prec']
df_valid['w_time'] = 0.1 * df_valid['s_time']
df_valid['score']  = df_valid['w_acc'] + df_valid['w_prec'] + df_valid['w_time']
df_valid['pct_acc']  = 100 * df_valid['w_acc']  / df_valid['score']
df_valid['pct_prec'] = 100 * df_valid['w_prec'] / df_valid['score']
df_valid['pct_time'] = 100 * df_valid['w_time'] / df_valid['score']

out_cols = ['trial_num','deviation_pct','precision_cv_pct','duration_mean_s',
            'w_acc','w_prec','w_time','score','pct_acc','pct_prec','pct_time']
out_path = 'temp/ethanol_sb_score_breakdown.csv'
df_valid[out_cols].sort_values('score').to_csv(out_path, index=False, float_format='%.4f')
print(f"CSV saved: {out_path}")

print(f"Population stds:  acc_std={acc_std:.3f}  prec_std={prec_std:.3f}  time_std={time_std:.3f}")
print(f"SCORE = w_acc + w_prec + w_time  where  w_acc=0.4*(dev/acc_std)*100  w_prec=0.5*(cv/prec_std)*100  w_time=0.1*(t/time_std)*100")
print()
print(f"{'T#':>3}  {'dev%':>7}  {'cv%':>7}  {'t(s)':>6}  |  {'w_acc':>6}  {'w_prec':>7}  {'w_time':>7}  |  {'TOTAL':>7}  |  {'%acc':>5}  {'%prec':>6}  {'%time':>6}")
print("-"*85)
df_sorted = df_valid.sort_values('score')
for _, r in df_sorted.iterrows():
    pct_acc  = 100 * r['w_acc']  / r['score']
    pct_prec = 100 * r['w_prec'] / r['score']
    pct_time = 100 * r['w_time'] / r['score']
    print(f"{int(r['trial_num']):>3}  {r['deviation_pct']:>7.3f}  {r['precision_cv_pct']:>7.3f}  {r['duration_mean_s']:>6.1f}  |  "
          f"{r['w_acc']:>6.2f}  {r['w_prec']:>7.2f}  {r['w_time']:>7.2f}  |  "
          f"{r['score']:>7.2f}  |  "
          f"{pct_acc:>5.1f}  {pct_prec:>6.1f}  {pct_time:>6.1f}")
print("-"*85)
print("Lower score = better.  %acc + %prec + %time = 100% per trial.")
print()
print("NOTE: The configured weights (acc=40%, prec=50%, time=10%) describe how much each")
print("      metric's *spread* (std) contributes to score differences between trials.")
print("      They do NOT control the absolute level — time has a high baseline mean,")
print("      so it adds a large fixed cost to every trial's score regardless of weight.")

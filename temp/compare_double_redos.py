import pandas as pd
df = pd.read_csv(r'C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign\1000uL\incremental_results.csv')
double_redo = [1674, 1731, 1855, 2041, 3175, 3836, 3884, 4028, 4205, 4649, 4963]
for idx in double_redo:
    rows = df[df['row_index']==idx][['timestamp','run_type','measured_volume_ul','accuracy_pct','volume_ul_target']].sort_values('timestamp')
    target = rows.iloc[0]['volume_ul_target']
    print(f"idx {idx}  target={target:.1f}uL")
    for _, r in rows.iterrows():
        print(f"  {r['timestamp']}  {r['run_type']:8s}  measured={r['measured_volume_ul']:8.1f}uL  acc={r['accuracy_pct']:6.1f}%")

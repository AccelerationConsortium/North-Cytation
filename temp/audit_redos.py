import pandas as pd

for campaign in ['200uL', '1000uL']:
    path = rf'C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign\{campaign}\incremental_results.csv'
    df = pd.read_csv(path)

    dup_indices = sorted(df[df.duplicated(['row_index'], keep=False)]['row_index'].dropna().unique())

    print(f'\n{"="*70}')
    print(f'{campaign} — {len(dup_indices)} indices with multiple measurements')
    print(f'{"="*70}')

    for idx in dup_indices:
        group = df[df['row_index'] == idx].sort_values('timestamp').copy()
        cols = ['timestamp', 'run_type', 'status', 'measured_volume_ul', 'accuracy_pct',
                'pre_stable_pct', 'pre_baseline_std', 'tip_type']
        # only show cols that exist
        cols = [c for c in cols if c in group.columns]
        print(f'\n--- row_index {int(idx)} ---')
        print(group[cols].to_string(index=False))

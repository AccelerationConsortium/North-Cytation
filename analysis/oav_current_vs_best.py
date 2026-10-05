"""Compare the CURRENT optimal_conditions OAV (the wizard's live source file) against
the BEST empirical OAV estimated from historical measurements (inverse-variance-weighted
mean from master_pipetting_measurements.csv), for the regimes with enough data:
  - water: large tier only (0.2-0.9 mL) - small tier skipped per instruction (less/noisier data)
  - SDS: both tiers (0.02-0.9 mL) - SDS never had the contamination issue
"""
import pandas as pd
import numpy as np

EPS_UL = 0.5
MIN_SAMPLES_PER_VOLUME = 100
fixed_cols = ['aspirate_speed', 'dispense_speed', 'retract_speed', 'blowout_speed',
              'aspirate_wait_time', 'dispense_wait_time', 'post_retract_wait_time',
              'pre_asp_air_vol', 'post_asp_air_vol', 'blowout_vol', 'asp_disp_cycles']

df = pd.read_csv('pipetting_data/master_pipetting_measurements.csv')


def mode_filter(sub):
    keep_idx = []
    for volume, vsub in sub.groupby('target_volume_ml'):
        counts = vsub.groupby(fixed_cols, dropna=False).size().sort_values(ascending=False)
        mode_key = counts.index[0]
        mask = np.ones(len(vsub), dtype=bool)
        for col, val in zip(fixed_cols, mode_key):
            mask &= (vsub[col] == val).values
        keep_idx.append(vsub.index[mask])
    return sub.loc[np.concatenate([np.array(i) for i in keep_idx])].copy()


def best_oav_per_volume(sub):
    sub = sub.copy()
    sub['error_ul'] = (sub['measured_volume_ml'] - sub['target_volume_ml']) * 1000
    sub['oav_ul'] = sub['overaspirate_vol'] * 1000
    iv_w = 1.0 / (sub['error_ul'] ** 2 + EPS_UL ** 2)
    out = {}
    for vol, g in sub.groupby('target_volume_ml'):
        ww = iv_w.loc[g.index]
        wmean = np.sum(ww * g['oav_ul']) / np.sum(ww)
        wse = np.sqrt(1.0 / np.sum(ww))
        out[vol] = (wmean, wse, len(g))
    return out


configs = {
    'water': {
        'csv': 'pipetting_data/optimal_conditions_water_complete.csv',
        'volumes_ml': [0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8, 0.9],  # excludes 0.01mL (<20uL, less data)
    },
    'SDS': {
        'csv': 'pipetting_data/optimal_conditions_SDS.csv',
        'volumes_ml': [0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8, 0.9],  # both tiers
    },
}

for liquid, cfg in configs.items():
    sub = df[df['liquid_type'] == liquid].copy()
    vol_counts = sub.groupby('target_volume_ml').size()
    valid_volumes = vol_counts[vol_counts >= MIN_SAMPLES_PER_VOLUME].index
    sub = sub[sub['target_volume_ml'].isin(valid_volumes)].copy()
    sub = mode_filter(sub)
    best = best_oav_per_volume(sub)

    ref = pd.read_csv(cfg['csv'])
    ref = ref.sort_values('volume_target').reset_index(drop=True)

    print('=' * 78)
    print(f'{liquid}')
    print(f'{"volume_uL":>10} | {"current_OAV_uL":>14} | {"best_OAV_uL":>12} | {"+/- SE":>8} | {"diff_uL":>8} | {"within 95% CI?":>14} | n')
    for vol_ml in cfg['volumes_ml']:
        vol_ul = vol_ml * 1000
        row = ref[np.isclose(ref['volume_target'], vol_ul, atol=0.5)]
        if row.empty:
            print(f'{vol_ul:>10.1f} | (no exact row in {cfg["csv"]})')
            continue
        current_oav_ul = row.iloc[0]['overaspirate_vol'] * 1000

        if vol_ml not in best:
            print(f'{vol_ul:>10.1f} | {current_oav_ul:>14.3f} | (no measurement data)')
            continue
        wmean, wse, n = best[vol_ml]
        diff = current_oav_ul - wmean
        within_ci = abs(diff) <= 1.96 * wse
        print(f'{vol_ul:>10.1f} | {current_oav_ul:>14.3f} | {wmean:>12.3f} | {wse:>8.3f} | {diff:>+8.3f} | {str(within_ci):>14} | {n}')

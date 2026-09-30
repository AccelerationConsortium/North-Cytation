"""OAV vs. target volume analysis for water and SDS, using master_pipetting_measurements.csv.

Filters out cross-regime contamination caused by a historical logging bug (fixed in
pipetting_data/embedded_calibration_validation.py) where a session's non-OAV settings
were sometimes stamped with a different volume's parameters. For each target volume,
only rows matching that volume's most common (mode) non-OAV parameter set are kept.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('pipetting_data/master_pipetting_measurements.csv')
fixed_cols = ['aspirate_speed', 'dispense_speed', 'retract_speed', 'blowout_speed',
              'aspirate_wait_time', 'dispense_wait_time', 'post_retract_wait_time',
              'pre_asp_air_vol', 'post_asp_air_vol', 'blowout_vol', 'asp_disp_cycles']

EPS_UL = 0.5  # small floor to avoid divide-by-zero for near-perfect measurements
MIN_SAMPLES_PER_VOLUME = 100  # drop one-off/stray test volumes (a handful of rows), not part of the real calibration grid
OUTPUT_DIR = 'analysis/output'


def mode_filter(sub):
    """Keep, per target volume, only rows matching that volume's most common non-OAV parameter set.

    Also returns {target_volume_ml: mode_key} so callers can detect where the mode
    parameter set actually changes between neighboring volumes (a real regime boundary).
    """
    keep_masks = []
    mode_key_by_volume = {}
    for volume, vsub in sub.groupby('target_volume_ml'):
        counts = vsub.groupby(fixed_cols, dropna=False).size().sort_values(ascending=False)
        mode_key = counts.index[0]
        mode_key_by_volume[volume] = mode_key
        mask = np.ones(len(vsub), dtype=bool)
        for col, val in zip(fixed_cols, mode_key):
            mask &= (vsub[col] == val).values
        keep_masks.append(vsub.index[mask])
    keep_idx = np.concatenate([np.array(m) for m in keep_masks])
    return sub.loc[keep_idx].copy(), mode_key_by_volume


results = {}
mode_keys_by_liquid = {}
for liquid in ['water', 'SDS']:
    sub = df[df['liquid_type'] == liquid].copy()

    # Drop one-off/stray test volumes with too few rows to trust (e.g. a single
    # exploratory measurement) - these aren't part of the real calibration grid
    # and were producing spurious "regime change" lines.
    vol_counts = sub.groupby('target_volume_ml').size()
    valid_volumes = vol_counts[vol_counts >= MIN_SAMPLES_PER_VOLUME].index
    dropped_volumes = sorted(set(vol_counts.index) - set(valid_volumes))
    if dropped_volumes:
        print(f'{liquid}: dropping stray low-sample volumes (mL): {dropped_volumes}')
    sub = sub[sub['target_volume_ml'].isin(valid_volumes)].copy()

    before_n = len(sub)
    sub, mode_key_by_volume = mode_filter(sub)
    after_n = len(sub)
    print(f'{liquid}: {before_n} -> {after_n} rows after mode-filter ({before_n - after_n} excluded)')
    results[liquid] = sub
    mode_keys_by_liquid[liquid] = mode_key_by_volume

for liquid, sub in results.items():
    sub = sub.copy()
    sub['error_ul'] = (sub['measured_volume_ml'] - sub['target_volume_ml']) * 1000
    sub['abs_error_ul'] = sub['error_ul'].abs()
    sub['oav_ul'] = sub['overaspirate_vol'] * 1000
    sub['target_ul'] = sub['target_volume_ml'] * 1000

    # Emphasis weight: low error -> large/opaque, high (often deliberate) error -> small/faint
    w = 1.0 / (sub['abs_error_ul'] + EPS_UL)
    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-12)
    sizes = 15 + w_norm * 200
    alphas = 0.12 + w_norm * 0.75

    # Clip the color scale at the 90th percentile so a handful of deliberately-bad
    # Stage 2 "probe" points (up to ~200uL error) don't wash out the gradient across
    # the bulk of the (mostly accurate) data.
    color_vmax = np.percentile(sub['abs_error_ul'], 90)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(sub['target_ul'], sub['oav_ul'], s=sizes, c=sub['abs_error_ul'],
                     cmap='viridis_r', vmin=0, vmax=color_vmax, alpha=None, edgecolors='none')
    # apply per-point alpha manually (matplotlib scatter doesn't support array alpha directly pre-3.4 easily with cmap)
    sc.set_alpha(alphas.values)
    cbar = plt.colorbar(sc, ax=ax, extend='max')
    cbar.set_label('|error| (uL), clipped at 90th pct -- darker/larger = lower error')

    # Inverse-variance-weighted mean OAV trend per target volume
    iv_w = 1.0 / (sub['error_ul'] ** 2 + EPS_UL ** 2)
    grp = sub.groupby('target_ul')
    trend_x, trend_y, trend_se = [], [], []
    for vol, g in grp:
        ww = iv_w.loc[g.index]
        wmean = np.sum(ww * g['oav_ul']) / np.sum(ww)
        wse = np.sqrt(1.0 / np.sum(ww))
        trend_x.append(vol)
        trend_y.append(wmean)
        trend_se.append(wse)
    trend_x = np.array(trend_x)
    trend_y = np.array(trend_y)
    trend_se = np.array(trend_se)
    order = np.argsort(trend_x)
    trend_x, trend_y, trend_se = trend_x[order], trend_y[order], trend_se[order]

    ax.plot(trend_x, trend_y, color='red', linewidth=2, marker='o', label='inverse-variance-weighted optimal OAV')
    ax.fill_between(trend_x, trend_y - 1.96 * trend_se, trend_y + 1.96 * trend_se, color='red', alpha=0.2)

    # Mark regime boundaries: draw a dotted line wherever the mode (non-OAV) parameter
    # set actually changes between two neighboring tested volumes - data-driven, not assumed.
    mode_key_by_volume = mode_keys_by_liquid[liquid]
    sorted_vols = sorted(mode_key_by_volume.keys())
    first_boundary = True
    for v1, v2 in zip(sorted_vols, sorted_vols[1:]):
        if mode_key_by_volume[v1] != mode_key_by_volume[v2]:
            boundary_ul = np.sqrt(v1 * v2) * 1000  # geometric mean, consistent with log x-axis
            ax.axvline(boundary_ul, linestyle=':', color='black', linewidth=1.5,
                       label='regime change' if first_boundary else None)
            first_boundary = False

    ax.set_xlabel('Target volume (uL)')
    ax.set_ylabel('Overaspirate volume, OAV (uL)')
    ax.set_title(f'{liquid}: OAV vs target volume (point size/color = accuracy emphasis)')
    ax.legend()
    ax.set_xscale('log')
    fig.tight_layout()
    out_path = f'{OUTPUT_DIR}/oav_analysis_{liquid}.png'
    fig.savefig(out_path, dpi=130)
    print(f'Saved {out_path}')
    plt.close(fig)

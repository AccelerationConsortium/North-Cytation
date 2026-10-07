"""Import a sdl_pipette_calibration run's optimal_conditions CSV (new format with
hardware_parameters_* columns) into a legacy pipetting_data calibration CSV
(the format PipettingWizard reads), merging by volume_target.

Usage:
    python import_calibration_run.py <new_run_optimal_conditions.csv> <legacy_csv>
"""
import sys
import pandas as pd

COLUMN_MAP = {
    'volume_target_ul': 'volume_target',
    'volume_measured_ul': 'volume_measured',
    'deviation_pct': 'average_deviation',
    'precision_cv_pct': 'variability',
    'duration_s': 'time',
    'calibration_overaspirate_vol': 'overaspirate_vol',
    'volume_target_ml': 'volume_ml',
    'hardware_parameters_aspirate_speed': 'aspirate_speed',
    'hardware_parameters_dispense_speed': 'dispense_speed',
    'hardware_parameters_aspirate_wait_time': 'aspirate_wait_time',
    'hardware_parameters_dispense_wait_time': 'dispense_wait_time',
    'hardware_parameters_retract_speed': 'retract_speed',
    'hardware_parameters_blowout_vol': 'blowout_vol',
    'hardware_parameters_post_asp_air_vol': 'post_asp_air_vol',
    'hardware_parameters_pre_asp_air_vol': 'pre_asp_air_vol',
    'hardware_parameters_post_retract_wait_time': 'post_retract_wait_time',
    'hardware_parameters_asp_disp_cycles': 'asp_disp_cycles',
}


def convert(new_run_csv, legacy_csv):
    new_df = pd.read_csv(new_run_csv).rename(columns=COLUMN_MAP)
    new_df['volume_ul'] = new_df['volume_target']

    legacy_df = pd.read_csv(legacy_csv)
    # keep the legacy file's column set; new-only columns (e.g. pre_asp_air_vol) are dropped
    # unless the legacy file already tracks them
    new_df = new_df.reindex(columns=legacy_df.columns)

    merged = pd.concat([legacy_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset='volume_target', keep='last')
    merged = merged.sort_values('volume_target').reset_index(drop=True)
    merged.to_csv(legacy_csv, index=False)
    print(f"Merged {len(new_df)} rows into {legacy_csv} ({len(merged)} total rows)")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])

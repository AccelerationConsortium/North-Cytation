"""Import a sdl_pipette_calibration run's optimal_conditions CSV (new format with
hardware_parameters_* columns) into a legacy pipetting_data calibration CSV
(the format PipettingWizard reads), merging by volume_target.

The legacy file is backed up (timestamped copy in its backups/ folder) before being
overwritten, since this is the live file PipettingWizard reads for real
pipetting parameters.

Usage:
    python import_calibration_run.py <new_run_optimal_conditions.csv> <legacy_csv>
"""
import shutil
from dataclasses import asdict
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipetting_parameters import PipettingParameters

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
    'hardware_parameters_blowout_speed': 'blowout_speed',
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
    # Preserve the legacy schema and add recognized calibration columns supplied
    # by this run. Never drop a mapped hardware parameter merely because the
    # production file predates it. Fill missing hardware values below using
    # the same defaults as the robot; measurement columns remain untouched.
    # None is a runtime fallback (e.g. blowout_speed uses aspirate_speed),
    # not a numeric value to write into a calibration CSV.
    defaults = {name: value for name, value in asdict(PipettingParameters()).items()
                if value is not None}
    columns = list(legacy_df.columns)
    for column in dict.fromkeys([*COLUMN_MAP.values(), "volume_ul", *defaults]):
        if (column in new_df.columns or column in defaults) and column not in columns:
            columns.append(column)
    new_df = new_df.reindex(columns=columns)
    legacy_df = legacy_df.reindex(columns=columns)

    legacy_path = Path(legacy_csv)
    backup_dir = legacy_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / (
        f"{legacy_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{legacy_path.suffix}"
    )
    shutil.copy2(legacy_path, backup_path)

    merged = pd.concat([legacy_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset='volume_target', keep='last')
    merged = merged.sort_values('volume_target').reset_index(drop=True)
    defaulted = {}
    for column, default in defaults.items():
        missing = merged[column].isna()
        if missing.any():
            defaulted[column] = int(missing.sum())
            merged[column] = merged[column].fillna(default)
    merged.to_csv(legacy_csv, index=False)
    print(f"Backed up {legacy_csv} to {backup_path}")
    print(f"Merged {len(new_df)} rows into {legacy_csv} ({len(merged)} total rows)")
    if defaulted:
        print("Filled missing parameters with PipettingParameters defaults: "
              + ", ".join(f"{name}={defaults[name]} ({count} rows)"
                          for name, count in defaulted.items()))
    return backup_path


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])

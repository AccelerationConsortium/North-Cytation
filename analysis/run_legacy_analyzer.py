"""Quick legacy calibration plotting.

Minimal usage (PowerShell):
    python analysis/run_legacy_analyzer.py --run-dir "C:\\Users\\owenm\\OneDrive\\Desktop\\Calibration_SDL\\20251007_151100_water_success"

Optional subfolder:
    python analysis/run_legacy_analyzer.py --run-dir "<run_dir>" --out-subdir quick_plots

Looks for inside run_dir:
    experiment_summary_autosave.csv  (required)
    raw_replicate_data_autosave.csv  (optional)

Outputs (saved in run dir unless --out-subdir used):
    time_vs_deviation_scatter.png / _abs
    measured_volume_over_time.png (if raw replicate data)
    learning_curves_deviation.png / learning_curves_time.png
    improvement_summary.png / improvement_summary.csv

Auto column handling:
    deviation <- deviation | percent_deviation | (abs_deviation_ul + volume)
    time      <- time | time_seconds
    volume    <- volume | target_volume (µL auto-converted when values >10)
    trial_index added if missing
    raw replicate volume from calculated_volume(_ul) or mass (density=1.0 fallback)
"""
from __future__ import annotations
import argparse, os, sys, pandas as pd
from typing import Optional
from calibration_analyzer import (
    plot_time_vs_deviation,
    plot_measured_volume_over_time,
    plot_learning_curves,
    plot_improvement_summary,
)
import glob

def normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        'time_seconds': 'time',
        'percent_deviation': 'deviation',
        'deviation_percent': 'deviation',
        'target_volume': 'volume',
        'std_dev_ul': 'variability'
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    if 'deviation' not in df.columns:
        if 'abs_deviation_ul' in df.columns and 'volume' in df.columns:
            # Ensure volume in mL for computation
            vol_mL = df['volume'].copy()
            if vol_mL.median() > 10:  # assume µL
                vol_mL = vol_mL / 1000.0
                df['volume'] = vol_mL
            df['deviation'] = df['abs_deviation_ul'] / (vol_mL * 1000) * 100
        else:
            raise ValueError("Cannot determine deviation column (need deviation or abs_deviation_ul + volume).")
    if 'volume' in df.columns and df['volume'].median() > 10:
        df['volume'] = df['volume'] / 1000.0
    if 'trial_index' not in df.columns:
        df['trial_index'] = range(1, len(df)+1)
    return df

def load_raw(raw_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(raw_path):
        return None
    raw = pd.read_csv(raw_path)
    if 'volume' not in raw.columns and 'target_volume' in raw.columns:
        raw.rename(columns={'target_volume': 'volume'}, inplace=True)
    # Volume unit heuristic
    if 'volume' in raw.columns and raw['volume'].median() > 10:
        raw['volume'] = raw['volume'] / 1000.0
    if 'calculated_volume' not in raw.columns and 'calculated_volume_ul' not in raw.columns:
        if 'mass' in raw.columns:
            raw['calculated_volume'] = raw['mass'] / 1.0  # density=1.0 fallback
    return raw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default='', help='Path to run directory (defaults to CWD if omitted).')
    ap.add_argument('--out-subdir', default='', help='If provided and non-empty, create/use this subfolder inside run dir. Blank = save directly in run dir.')
    ap.add_argument('--full', action='store_true', help='Generate main graph set (time vs deviation + learning curves).')
    ap.add_argument('--include-improvement', action='store_true', help='Also generate improvement summary plot & CSV (off by default).')
    args = ap.parse_args()

    run_dir = args.run_dir or os.getcwd()
    if not os.path.isdir(run_dir):
        print(f"[ERROR] Run directory not found: {run_dir}")
        sys.exit(1)

    # Locate summary file: prefer direct file, else choose most recently modified among recursive candidates
    summary_path = os.path.join(run_dir, 'experiment_summary_autosave.csv')
    if not os.path.exists(summary_path):
        candidates = glob.glob(os.path.join(run_dir, '**', 'experiment_summary_autosave.csv'), recursive=True)
        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            summary_path = candidates[0]
            run_dir = os.path.dirname(summary_path)
            print(f"[INFO] Using latest summary file: {summary_path}")
            if len(candidates) > 1:
                print(f"[INFO] ({len(candidates)-1} other summary files ignored)")

    out_dir = run_dir if not args.out_subdir else os.path.join(run_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(summary_path):
        print(f"[ERROR] Missing {summary_path}")
        sys.exit(1)

    summary = pd.read_csv(summary_path)
    try:
        summary = normalize_summary(summary)  # type: ignore
    except Exception as e:
        print(f"[ERROR] Normalization failed: {e}")
        sys.exit(1)

    raw_path = os.path.join(run_dir, 'raw_replicate_data_autosave.csv')
    raw = None
    if os.path.exists(raw_path):
        raw = load_raw(raw_path)  # type: ignore
    else:
        # Prefer raw file in same parent as summary if multiple exist
        parent = os.path.dirname(summary_path)
        candidate_same = os.path.join(parent, 'raw_replicate_data_autosave.csv')
        if os.path.exists(candidate_same):
            raw = load_raw(candidate_same)  # type: ignore
        else:
            raw_candidates = glob.glob(os.path.join(run_dir, '**', 'raw_replicate_data_autosave.csv'), recursive=True)
            if raw_candidates:
                raw_candidates.sort(key=os.path.getmtime, reverse=True)
                chosen = raw_candidates[0]
                print(f"[INFO] Using raw replicate file: {chosen}")
                raw = load_raw(chosen)  # type: ignore
            else:
                print('[INFO] No raw replicate file found anywhere; will skip measured volume plot.')

    print('[INFO] Plot: time vs deviation (percent)')
    plot_time_vs_deviation(summary, out_dir, show_absolute=False)
    print('[INFO] Plot: time vs deviation (absolute)')
    plot_time_vs_deviation(summary, out_dir, show_absolute=True)

    if raw is not None and len(raw):
        print('[INFO] Plot: measured volume over time')
        plot_measured_volume_over_time(raw, out_dir)
    else:
        print('[WARN] Skipping measured volume plot (no raw replicate data).')

    if args.full:
        try:
            print('[INFO] Plot: learning curves (deviation,time)')
            plot_learning_curves(summary, out_dir, metrics=['deviation','time'])
        except Exception as e:
            print(f"[WARN] Learning curves failed: {e}")

    if args.include_improvement:
        try:
            print('[INFO] Plot: improvement summary')
            plot_improvement_summary(summary, out_dir)
        except Exception as e:
            print(f"[WARN] Improvement summary failed: {e}")

    print(f"[DONE] Graphs written to {out_dir}")

if __name__ == '__main__':
    main()

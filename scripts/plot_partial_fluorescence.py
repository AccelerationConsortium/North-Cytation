"""One-off: plot partial results for an in-progress fluorescence calibration run.

Usage: python scripts/plot_partial_fluorescence.py <output_dir>
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

from workflows.fluorescence_calibration_workflow import per_well_means, summarize, plot_calibration, plot_kinetics


def main(output_dir):
    output_dir = Path(output_dir)
    results = pd.read_csv(output_dir / "fluorescence_results.csv")
    channels = [c for c in results.columns if c[0].isdigit()]
    wells = per_well_means(results, channels)
    summary = summarize(results, channels)
    plot_calibration(wells, summary, channels, output_dir)
    plot_kinetics(wells, channels, output_dir)
    timepoints = sorted(results.timepoint_min.unique())
    print(f"Plotted {len(timepoints)} timepoint(s): {timepoints}")
    print(f"Saved PNGs to {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")

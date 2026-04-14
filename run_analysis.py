#!/usr/bin/env python3
"""
Run spectral analysis on the IDT-TIT heptane samples
"""

import sys
sys.path.insert(0, '/Users/serenaqiu/Desktop/north_repository/utoronto_demo')

from degradation_spectral_analyzer_program_copy import analyze_spectral_data

# Target folder
folder_path = "/Users/serenaqiu/Downloads/p(IDT-TIT)_1000x_HCl_heptane/p(IDT-TIT)_2000x_HCl_heptane"

# Create processed data subdirectory in the target folder
processed_data_dir = f"{folder_path}/processed_data"

# Run the analysis
# The timepoints are in seconds (0, 300, 600, 900, 1200, 1800, 2400, 3600)
# Which are 0, 5, 10, 15, 20, 30, 40, 60 minutes
manual_timepoints = [0, 5, 10, 15, 20, 30, 40, 60]

print(f"Starting analysis of files in: {folder_path}")
print(f"Timepoints (minutes): {manual_timepoints}")
print()

results = analyze_spectral_data(
    folder_path=folder_path,
    processed_data_dir=processed_data_dir,
    sample_name=None,
    manual_timepoints=manual_timepoints
)

if results:
    print("\n✓ Analysis completed successfully!")
    print(f"Results saved to: {processed_data_dir}")
else:
    print("\n✗ Analysis failed!")

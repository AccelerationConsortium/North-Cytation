#!/usr/bin/env python3
"""
Run spectral analysis on the provided output files
"""

import sys
sys.path.insert(0, '/Users/serenaqiu/Desktop/north_repository/utoronto_demo')

from spectral_analyzer_program import process_degradation_spectral_data
from pathlib import Path

# Use the directory containing your output files
folder_path = '/Users/serenaqiu/Downloads/p(IDT-TIT)-SQ003E-1000xHCl-10uLH2O_ALL/p(IDT-TIT)-SQ003E-1000xHCl-10uLH2O-WorkedWorkflow'

# Run the analysis
results = process_degradation_spectral_data(folder_path)

if results is not None:
    print("\n" + "="*60)
    print("SPECTRAL ANALYSIS COMPLETE!")
    print("="*60)
    if isinstance(results, tuple):
        combined_df, time_series_df = results
    else:
        combined_df = results
        time_series_df = None
    
    print("Files generated in processed_data folder:")
    print(f"  1. spectral_analysis_plot.png - Full spectral time series (purple->pink->orange->yellow)")
    print(f"  2. combined_spectral_data.csv - Combined dataset")
    if time_series_df is not None:
        print(f"  3. wavelength_time_analysis.png - 555nm (green), 458nm (yellow), ratio (orange)")
        print(f"  4. wavelength_time_series.csv - Time series in minutes")
else:
    print("\nAnalysis failed - no data processed.")

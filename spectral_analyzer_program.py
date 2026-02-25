#!/usr/bin/env python3
"""
Spectral Data Analyzer for output_# files

This program:
1. Finds all files labeled output_# in a specified folder
2. Graphs the spectral data with color series for different time points
3. Combines data into single DataFrame and saves it
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob
import re
from pathlib import Path


def find_output_files(folder_path, sample_name=None):
    """
    Find all output_# files in the specified folder, optionally filtered by sample name
    """
    # Updated pattern to handle sample prefixes like "sample_1_output_*.txt" 
    pattern = os.path.join(folder_path, "*output_*.txt")
    files = glob.glob(pattern)
    
    # Filter by sample name if specified
    if sample_name is not None:
        files = [f for f in files if f'sample_{sample_name}' in f or f'sample_{sample_name}_' in os.path.basename(f)]
    
    # Sort files by the number after output_
    def extract_number(filename):
        match = re.search(r'output_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    files.sort(key=extract_number)
    return files


def extract_sample_name(filename):
    """
    Extract sample name from filename (e.g., 'sample_1' from 'sample_1_output_5.txt')
    """
    basename = os.path.basename(filename)
    match = re.search(r'sample_(\d+)', basename)
    if match:
        return int(match.group(1))
    return None


def get_all_samples(folder_path):
    """
    Get list of unique sample numbers in the folder
    """
    pattern = os.path.join(folder_path, "*output_*.txt")
    files = glob.glob(pattern)
    
    # If there are any output files, just process them all as one dataset
    if files:
        return [None]  # Process all files together
    else:
        return []  # No files found


def read_spectral_file(filepath):
    """
    Read a single spectral data file and return wavelength, averaged absorbance, and individual replicates
    Returns:
        wavelength: array of wavelengths
        absorbance_avg: averaged absorbance across all replicates
        absorbance_replicates: list of individual replicate absorbance arrays
    """
    # Read the file, skip the first 2 header rows
    df = pd.read_csv(filepath, skiprows=2, header=None)
    
    # Extract wavelength (column 1)
    wavelength = df.iloc[:, 1].values
    
    # Extract all replicate columns (column 2 onwards)
    replicate_columns = df.iloc[:, 2:].values  # All columns from index 2 onwards
    
    # Calculate average absorbance across replicates for spectral plots
    absorbance_avg = np.mean(replicate_columns, axis=1)
    
    # Individual replicates for wavelength-specific analysis
    absorbance_replicates = [replicate_columns[:, i] for i in range(replicate_columns.shape[1])]
    
    return wavelength, absorbance_avg, absorbance_replicates


def get_absorbance_at_wavelength(wavelength_array, absorbance_array, target_wavelength):
    """
    Extract absorbance value at a specific wavelength using interpolation
    """
    # Find the closest wavelength values
    idx = np.argmin(np.abs(wavelength_array - target_wavelength))
    
    # If exact match, return the value
    if wavelength_array[idx] == target_wavelength:
        return absorbance_array[idx]
    
    # Otherwise, interpolate between closest points
    return np.interp(target_wavelength, wavelength_array, absorbance_array)


def create_wavelength_time_plots_with_sample(processed_data_dir, combined_data, files, sample_name=None, plot_filename=None, csv_filename=None, wavelength1=555, wavelength2=458):
    """
    Create time series plots for specific wavelengths and their ratio
    Args:
        wavelength1: Primary wavelength (default: 555nm)  
        wavelength2: Secondary wavelength (default: 458nm)
    """
    print(f"\nCreating wavelength-specific time series plots for {wavelength1}nm and {wavelength2}nm...")
    
    # Initialize data storage for time series
    timepoints = []
    abs_wl1 = []
    abs_wl2 = []
    
    # Extract timepoints and absorbance values at specific wavelengths
    timepoints_all_reps = []  # Will store timepoint for each replicate measurement
    abs_wl1 = []
    abs_wl2 = []
    
    for i, filepath in enumerate(files):
        timepoint = re.search(r'output_(\d+)', os.path.basename(filepath))
        timepoint_num = int(timepoint.group(1)) if timepoint else i * 300
        
        # Get wavelength and absorbance data (including replicates)
        wavelength, absorbance_avg, absorbance_replicates = read_spectral_file(filepath)
        
        # Extract absorbance at specified wavelengths for each replicate
        for abs_replicate in absorbance_replicates:
            abs_1 = get_absorbance_at_wavelength(wavelength, abs_replicate, wavelength1)
            abs_2 = get_absorbance_at_wavelength(wavelength, abs_replicate, wavelength2)
            
            abs_wl1.append(abs_1)
            abs_wl2.append(abs_2)
            timepoints_all_reps.append(timepoint_num)
    
    # Update timepoints to use the expanded list
    timepoints = timepoints_all_reps
    
    # Convert timepoints from seconds to minutes
    timepoints = [t / 60.0 for t in timepoints]
    
    # Calculate ratio
    ratio_wl2_wl1 = np.array(abs_wl2) / np.array(abs_wl1)
    
    # Create the three plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sample_label = f" - Sample {sample_name}" if sample_name is not None else ""
    fig.suptitle(f'Wavelength-Specific Analysis Over Time{sample_label}', fontsize=18, fontweight='bold')
    
    # Plot 1: wavelength1 over time (scatter plot for individual replicates)
    axes[0, 0].scatter(timepoints, abs_wl1, color='#4A3A7F', s=50, alpha=0.7)
    axes[0, 0].set_xlabel('Time (min)', fontsize=14)
    axes[0, 0].set_ylabel('Absorbance (a.u.)', fontsize=14)
    axes[0, 0].set_title(f'{wavelength1} nm Absorbance vs Time', fontsize=14)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=12, direction='in')
    axes[0, 0].grid(False)
    
    # Plot 2: wavelength2 over time (scatter plot for individual replicates)
    axes[0, 1].scatter(timepoints, abs_wl2, color="#859DE6", s=50, alpha=0.7)
    axes[0, 1].set_xlabel('Time (min)', fontsize=14)
    axes[0, 1].set_ylabel('Absorbance (a.u.)', fontsize=14)
    axes[0, 1].set_title(f'{wavelength2} nm Absorbance vs Time', fontsize=14)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=12, direction='in')
    axes[0, 1].grid(False)
    
    # Plot 3: ratio over time (scatter plot for individual replicates)
    axes[1, 0].scatter(timepoints, ratio_wl2_wl1, color="#9B6BA8", s=50, alpha=0.7)
    axes[1, 0].set_xlabel('Time (min)', fontsize=14)
    axes[1, 0].set_ylabel(f'Absorbance Ratio ({wavelength2}/{wavelength1} nm)', fontsize=14)
    axes[1, 0].set_title(f'{wavelength2}/{wavelength1} nm Ratio vs Time', fontsize=14)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=12, direction='in')
    axes[1, 0].grid(False)
    
    # Plot 4: Combined comparison (scatter plots for individual replicates)
    axes[1, 1].scatter(timepoints, abs_wl1, color='#4A3A7F', s=40, alpha=0.7, label=f'{wavelength1} nm')
    axes[1, 1].scatter(timepoints, abs_wl2, color='#859DE6', s=40, alpha=0.7, label=f'{wavelength2} nm')
    ax2 = axes[1, 1].twinx()
    ax2.scatter(timepoints, ratio_wl2_wl1, color='#9B6BA8', s=40, alpha=0.7, label=f'{wavelength2}/{wavelength1} nm Ratio', marker='s')
    axes[1, 1].set_xlabel('Time (min)', fontsize=14)
    axes[1, 1].set_ylabel('Absorbance (a.u.)', fontsize=14)
    ax2.set_ylabel(f'Absorbance Ratio ({wavelength2}/{wavelength1} nm)', color="#000000", fontsize=14)
    axes[1, 1].set_title('Combined Analysis (Individual Replicates)', fontsize=14)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=12, direction='in')
    ax2.tick_params(axis='both', which='major', labelsize=12, direction='in')
    axes[1, 1].legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    axes[1, 1].grid(False)
    
    plt.tight_layout()
    
    # Use provided filenames or defaults
    if plot_filename is None:
        plot_filename = os.path.join(processed_data_dir, 'wavelength_time_analysis.png')
    if csv_filename is None:
        csv_filename = os.path.join(processed_data_dir, 'wavelength_time_series.csv')
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Wavelength time series plots saved as: {plot_filename}")
    
    # Create and save time series data with all individual replicate measurements
    time_series_data = pd.DataFrame({
        'timepoint_min': timepoints,
        'timepoint_sec': [t * 60 for t in timepoints],  # Also save in seconds
        f'abs_{wavelength1}nm': abs_wl1,
        f'abs_{wavelength2}nm': abs_wl2,
        f'ratio_{wavelength2}_{wavelength1}': ratio_wl2_wl1
    })
    
    time_series_data.to_csv(csv_filename, index=False)
    print(f"✓ Time series data saved as: {csv_filename}")
    
    # Print summary statistics
    print(f"\nWavelength Analysis Summary:")
    print(f"  {wavelength1}nm - Range: {min(abs_wl1):.4f} to {max(abs_wl1):.4f}")
    print(f"  {wavelength2}nm - Range: {min(abs_wl2):.4f} to {max(abs_wl2):.4f}") 
    print(f"  {wavelength2}/{wavelength1} Ratio - Range: {min(ratio_wl2_wl1):.4f} to {max(ratio_wl2_wl1):.4f}")
    
    return time_series_data


def analyze_spectral_data(folder_path, processed_data_dir=None, sample_name=None, wavelength1=555, wavelength2=458):
    """
    Main function to analyze spectral data from output_# files
    Args:
        folder_path: Directory containing output_# files
        processed_data_dir: Optional directory to save processed results (defaults to folder_path/processed_data)
        sample_name: Optional sample number to filter (e.g., 1 for sample_1)
        wavelength1: Primary wavelength for analysis (default: 555nm)
        wavelength2: Secondary wavelength for analysis (default: 458nm)
    """
    sample_label = f" (Sample {sample_name})" if sample_name is not None else ""
    print(f"Analyzing spectral data in: {folder_path}{sample_label}")
    
    # Create processed data directory if not specified
    if processed_data_dir is None:
        processed_data_dir = os.path.join(folder_path, 'processed_data')
    
    os.makedirs(processed_data_dir, exist_ok=True)
    print(f"Saving processed results to: {processed_data_dir}")
    
    # Find all output files (filtered by sample if specified)
    files = find_output_files(folder_path, sample_name=sample_name)
    
    if not files:
        print("No output_*.txt files found in the specified directory!")
        return
    
    print(f"Found {len(files)} output files:")
    for file in files:
        print(f"  - {os.path.basename(file)}")
    
    # Initialize combined DataFrame
    combined_data = None
    
    # Set up a new plot for this analysis (ensures separate graphs for each sample)
    fig = plt.figure(figsize=(12, 8))
    # Create custom gradient: dark blue-purple -> purple -> magenta -> pink -> peachy cream
    custom_colors = ['#4A3A7F', '#9B6BA8', '#D485C0', "#E0A5C9", "#F1CE9A"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('peach_pink_purple_blue', custom_colors)
    colors = custom_cmap(np.linspace(0, 1, len(files)))  # Color series
    
    print("\nReading and processing files...")
    
    for i, filepath in enumerate(files):
        try:
            # Extract timepoint number from filename
            timepoint = re.search(r'output_(\d+)', os.path.basename(filepath))
            timepoint_num = int(timepoint.group(1)) if timepoint else i * 300
            timepoint_min = timepoint_num / 60.0  # Convert seconds to minutes
            
            # Read spectral data (use averaged absorbance for clean spectral plot)
            wavelength, absorbance_avg, absorbance_replicates = read_spectral_file(filepath)
            
            # Plot with color series using averaged data
            plt.plot(wavelength, absorbance_avg, color=colors[i], 
                    label=f'{int(timepoint_min)} min', linewidth=2)
            
            # Create or update combined DataFrame using averaged data
            if combined_data is None:
                combined_data = pd.DataFrame({'wavelength': wavelength})
            
            combined_data[f'absorbance_t{timepoint_num}'] = absorbance_avg
            
            print(f"  ✓ Processed {os.path.basename(filepath)} (timepoint {timepoint_num}, {len(absorbance_replicates)} replicates)")
            
        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(filepath)}: {e}")
    
    # Customize the plot
    title_suffix = f" - Sample {sample_name}" if sample_name is not None else ""
    plt.xlabel('Wavelength (nm)', fontsize=16)
    plt.ylabel('Absorbance (a.u.)', fontsize=16)
    plt.xlim(300, 900)
    plt.title(f'Spectral Data Over Time{title_suffix}', fontsize=18, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14, direction='in')
    plt.grid(False)
    plt.tight_layout()
    
    # Save the plot to processed data directory with sample name in filename
    if sample_name is not None:
        plot_filename = os.path.join(processed_data_dir, f'spectral_analysis_plot_sample_{sample_name}.png')
        csv_filename = os.path.join(processed_data_dir, f'combined_spectral_data_sample_{sample_name}.csv')
        ts_filename = os.path.join(processed_data_dir, f'wavelength_time_analysis_sample_{sample_name}.png')
        ts_csv_filename = os.path.join(processed_data_dir, f'wavelength_time_series_sample_{sample_name}.csv')
    else:
        plot_filename = os.path.join(processed_data_dir, 'spectral_analysis_plot.png')
        csv_filename = os.path.join(processed_data_dir, 'combined_spectral_data.csv')
        ts_filename = os.path.join(processed_data_dir, 'wavelength_time_analysis.png')
        ts_csv_filename = os.path.join(processed_data_dir, 'wavelength_time_series.csv')
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved as: {plot_filename}")
    
    # Save combined data
    if combined_data is not None:
        combined_data.to_csv(csv_filename, index=False)
        print(f"✓ Combined data saved as: {csv_filename}")
        
        # Display summary
        print(f"\nData Summary:")
        print(f"  - Wavelength range: {combined_data['wavelength'].min():.1f} - {combined_data['wavelength'].max():.1f} nm")
        print(f"  - Number of data points: {len(combined_data)}")
        print(f"  - Number of timepoints: {len(files)}")
        print(f"\nCombined DataFrame shape: {combined_data.shape}")
        print(f"Columns: {list(combined_data.columns)}")
        
        # Create wavelength-specific time series plots
        time_series_data = create_wavelength_time_plots_with_sample(processed_data_dir, combined_data, files, sample_name, ts_filename, ts_csv_filename, wavelength1, wavelength2)
        
        return combined_data, time_series_data
    
    return None


def process_workflow_spectral_data(output_dir, logger=None, sample_name=None, wavelength1=555, wavelength2=458):
    """
    Process spectral data from workflow output directory
    Args:
        output_dir: Path or string - directory containing output_# files
        logger: Optional logger instance for consistent logging
        sample_name: Optional sample number to process (e.g., 1, 2)
        wavelength1: Primary wavelength for analysis (default: 555nm)
        wavelength2: Secondary wavelength for analysis (default: 458nm)
    Returns:
        tuple: (combined_df, time_series_df) or None if failed
    """
    try:
        folder_path = str(output_dir)  # Convert Path object to string if needed
        processed_data_dir = os.path.join(folder_path, 'processed_data')
        
        sample_label = f" - Sample {sample_name}" if sample_name else ""
        log_msg = f"Starting spectral data analysis for: {folder_path}{sample_label}"
        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            error_msg = f"Error: Output directory not found: {folder_path}"
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            return None
        
        # Analyze the spectral data (with sample filter if specified)
        results = analyze_spectral_data(folder_path, processed_data_dir, sample_name=sample_name, wavelength1=wavelength1, wavelength2=wavelength2)
        
        if results is not None:
            if isinstance(results, tuple):
                combined_df, time_series_df = results
            else:
                combined_df = results
                time_series_df = None
                
            success_msg = "Spectral analysis completed successfully!"
            if logger:
                logger.info(success_msg)
                logger.info(f"Results saved to: {processed_data_dir}")
            else:
                print(success_msg)
                print(f"Results saved to: {processed_data_dir}")
            
            return results
        else:
            error_msg = "Spectral analysis failed - no data processed"
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            return None
            
    except Exception as e:
        error_msg = f"Error during spectral analysis: {str(e)}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None


def main():
    """
    Main function - specify your folder path here (for standalone testing)
    """
    # User-specified folder path
    folder_path = r'C:\Users\Imaging Controller\Desktop\SQ\PL _COF_ph_new_white_pierce_20260225_124615\COF_2'
    
    # Get all samples in the folder
    samples = get_all_samples(folder_path)
    
    if not samples:
        print("No samples found in the folder.")
        return
    
    print(f"Found samples: {samples}")
    print(f"Processing {len(samples)} sample(s)...\n")
    
    # Process each sample separately
    all_results = {}
    for sample_num in samples:
        if sample_num is None:
            print(f"\n{'='*60}")
            print(f"Processing All Files (No Sample Prefixes)")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Processing Sample {sample_num}")
            print(f"{'='*60}")
        
        # Use the new function for processing this specific sample (using default 555nm/458nm wavelengths)
        results = process_workflow_spectral_data(folder_path, sample_name=sample_num, wavelength1=595, wavelength2=450)
        all_results[sample_num] = results
        
        if results is not None:
            if isinstance(results, tuple):
                combined_df, time_series_df = results
            else:
                combined_df = results
                time_series_df = None
                
            if sample_num is None:
                print(f"\nAll Files - Generated in processed_data folder:")
                print(f"  1. spectral_analysis_plot.png - Full spectral time series")
                print(f"  2. combined_spectral_data.csv - Combined dataset")
                if time_series_df is not None:
                    print(f"  3. wavelength_time_analysis.png - Wavelength-specific and ratio plots")
                    print(f"  4. wavelength_time_series.csv - Time series for specific wavelengths")
            else:
                print(f"\nSample {sample_num} - Files generated in processed_data folder:")
                print(f"  1. spectral_analysis_plot_sample_{sample_num}.png - Full spectral time series")
                print(f"  2. combined_spectral_data_sample_{sample_num}.csv - Combined dataset")
                if time_series_df is not None:
                    print(f"  3. wavelength_time_analysis_sample_{sample_num}.png - Wavelength-specific and ratio plots")
                    print(f"  4. wavelength_time_series_sample_{sample_num}.csv - Time series for specific wavelengths")
        else:
            if sample_num is None:
                print(f"\nAll files analysis failed - no data processed.")
            else:
                print(f"\nSample {sample_num} analysis failed - no data processed.")
    
    print("\n" + "="*60)
    print("ALL SAMPLES PROCESSED!")
    print("="*60)


if __name__ == "__main__":
    main()
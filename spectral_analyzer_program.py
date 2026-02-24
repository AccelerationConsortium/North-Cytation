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
    
    samples = set()
    for f in files:
        sample_num = extract_sample_name(f)
        if sample_num is not None:
            samples.add(sample_num)
    
    return sorted(list(samples))


def read_spectral_file(filepath):
    """
    Read a single spectral data file and return wavelength, absorbance arrays
    """
    # Read the file, skip the first 2 header rows
    df = pd.read_csv(filepath, skiprows=2, header=None)
    
    # Extract wavelength (column 1) and absorbance (column 2) 
    wavelength = df.iloc[:, 1].values
    absorbance = df.iloc[:, 2].values
    
    return wavelength, absorbance


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


def create_wavelength_time_plots_with_sample(processed_data_dir, combined_data, files, sample_name=None, plot_filename=None, csv_filename=None):
    """
    Create time series plots for specific wavelengths (555 nm, 458 nm, and 458/555 ratio)
    """
    print("\nCreating wavelength-specific time series plots...")
    
    # Initialize data storage for time series
    timepoints = []
    abs_555nm = []
    abs_458nm = []
    
    # Extract timepoints and absorbance values at specific wavelengths
    for i, filepath in enumerate(files):
        timepoint = re.search(r'output_(\d+)', os.path.basename(filepath))
        timepoint_num = int(timepoint.group(1)) if timepoint else i
        timepoints.append(timepoint_num)
        
        # Get wavelength and absorbance data
        wavelength, absorbance = read_spectral_file(filepath)
        
        # Extract absorbance at 555 nm and 458 nm
        abs_555 = get_absorbance_at_wavelength(wavelength, absorbance, 555)
        abs_458 = get_absorbance_at_wavelength(wavelength, absorbance, 458)
        
        abs_555nm.append(abs_555)
        abs_458nm.append(abs_458)
    
    # Convert timepoints from seconds to minutes
    timepoints = [t / 60.0 for t in timepoints]
    
    # Calculate 458/555 ratio
    ratio_458_555 = np.array(abs_458nm) / np.array(abs_555nm)
    
    # Create the three plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sample_label = f" - Sample {sample_name}" if sample_name is not None else ""
    fig.suptitle(f'Wavelength-Specific Analysis Over Time{sample_label}', fontsize=18, fontweight='bold')
    
    # Plot 1: 555 nm over time
    axes[0, 0].plot(timepoints, abs_555nm, 'o-', color='#4A3A7F', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Time (min)', fontsize=14)
    axes[0, 0].set_ylabel('Absorbance (a.u.)', fontsize=14)
    axes[0, 0].set_title('555 nm Absorbance vs Time', fontsize=14)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=12, direction='in')
    axes[0, 0].grid(False)
    
    # Plot 2: 458 nm over time
    axes[0, 1].plot(timepoints, abs_458nm, 'o-', color="#859DE6", linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Time (min)', fontsize=14)
    axes[0, 1].set_ylabel('Absorbance (a.u.)', fontsize=14)
    axes[0, 1].set_title('458 nm Absorbance vs Time', fontsize=14)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=12, direction='in')
    axes[0, 1].grid(False)
    
    # Plot 3: 458/555 ratio over time
    axes[1, 0].plot(timepoints, ratio_458_555, 'o-', color="#9B6BA8", linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Time (min)', fontsize=14)
    axes[1, 0].set_ylabel('Absorbance Ratio (458/555 nm)', fontsize=14)
    axes[1, 0].set_title('458/555 nm Ratio vs Time', fontsize=14)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=12, direction='in')
    axes[1, 0].grid(False)
    
    # Plot 4: Combined comparison
    axes[1, 1].plot(timepoints, abs_555nm, 'o-', color='#4A3A7F', linewidth=2, markersize=6, label='555 nm')
    axes[1, 1].plot(timepoints, abs_458nm, 'o-', color='#859DE6', linewidth=2, markersize=6, label='458 nm')
    ax2 = axes[1, 1].twinx()
    ax2.plot(timepoints, ratio_458_555, 's-', color='#9B6BA8', linewidth=2, markersize=6, label='458/555 nm Ratio')
    axes[1, 1].set_xlabel('Time (min)', fontsize=14)
    axes[1, 1].set_ylabel('Absorbance (a.u.)', fontsize=14)
    ax2.set_ylabel('Absorbance Ratio (458/555 nm)', color="#000000", fontsize=14)
    axes[1, 1].set_title('Combined Analysis', fontsize=14)
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
    plt.show()
    print(f"✓ Wavelength time series plots saved as: {plot_filename}")
    
    # Create and save time series data
    time_series_data = pd.DataFrame({
        'timepoint': timepoints,
        'abs_555nm': abs_555nm,
        'abs_458nm': abs_458nm,
        'ratio_458_555': ratio_458_555
    })
    
    time_series_data.to_csv(csv_filename, index=False)
    print(f"✓ Time series data saved as: {csv_filename}")
    
    # Print summary statistics
    print(f"\nWavelength Analysis Summary:")
    print(f"  555 nm - Range: {min(abs_555nm):.4f} to {max(abs_555nm):.4f}")
    print(f"  458 nm - Range: {min(abs_458nm):.4f} to {max(abs_458nm):.4f}") 
    print(f"  458/555 Ratio - Range: {min(ratio_458_555):.4f} to {max(ratio_458_555):.4f}")
    
    return time_series_data


def analyze_spectral_data(folder_path, processed_data_dir=None, sample_name=None):
    """
    Main function to analyze spectral data from output_# files
    Args:
        folder_path: Directory containing output_# files
        processed_data_dir: Optional directory to save processed results (defaults to folder_path/processed_data)
        sample_name: Optional sample number to filter (e.g., 1 for sample_1)
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
            timepoint_num = int(timepoint.group(1)) if timepoint else i
            timepoint_min = timepoint_num / 60.0  # Convert to minutes
            
            # Read spectral data
            wavelength, absorbance = read_spectral_file(filepath)
            
            # Plot with color series
            plt.plot(wavelength, absorbance, color=colors[i], 
                    label=f'{int(timepoint_min)} min', linewidth=2)
            
            # Create or update combined DataFrame
            if combined_data is None:
                combined_data = pd.DataFrame({'wavelength': wavelength})
            
            combined_data[f'absorbance_t{timepoint_num}'] = absorbance
            
            print(f"  ✓ Processed {os.path.basename(filepath)} (timepoint {timepoint_num})")
            
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
    plt.show()
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
        time_series_data = create_wavelength_time_plots_with_sample(processed_data_dir, combined_data, files, sample_name, ts_filename, ts_csv_filename)
        
        return combined_data, time_series_data
    
    return None


def process_degradation_spectral_data(output_dir, logger=None, sample_name=None):
    """
    Process spectral data from degradation workflow output directory
    Args:
        output_dir: Path or string - directory containing output_# files
        logger: Optional logger instance for consistent logging
        sample_name: Optional sample number to process (e.g., 1, 2)
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
        results = analyze_spectral_data(folder_path, processed_data_dir, sample_name=sample_name)
        
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
    folder_path = r"C:\Users\Imaging Controller\Desktop\SQ\p(IDT-TIT)-SQ003F-1000xHCl-0.05mgml"
    
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
        print(f"\n{'='*60}")
        print(f"Processing Sample {sample_num}")
        print(f"{'='*60}")
        
        # Use the new function for processing this specific sample
        results = process_degradation_spectral_data(folder_path, sample_name=sample_num)
        all_results[sample_num] = results
        
        if results is not None:
            if isinstance(results, tuple):
                combined_df, time_series_df = results
            else:
                combined_df = results
                time_series_df = None
                
            print(f"\nSample {sample_num} - Files generated in processed_data folder:")
            print(f"  1. spectral_analysis_plot_sample_{sample_num}.png - Full spectral time series")
            print(f"  2. combined_spectral_data_sample_{sample_num}.csv - Combined dataset")
            if time_series_df is not None:
                print(f"  3. wavelength_time_analysis_sample_{sample_num}.png - 555nm, 458nm, and ratio plots")
                print(f"  4. wavelength_time_series_sample_{sample_num}.csv - Time series for specific wavelengths")
        else:
            print(f"\nSample {sample_num} analysis failed - no data processed.")
    
    print("\n" + "="*60)
    print("ALL SAMPLES PROCESSED!")
    print("="*60)


if __name__ == "__main__":
    main()
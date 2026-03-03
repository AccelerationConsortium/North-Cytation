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
    Find all output files in the specified folder, handling both naming conventions:
    1. Peroxide workflow: output_1080_timestamp.txt 
    2. Degradation workflow: sample_1_output_000.txt
    """
    # Try peroxide pattern first: output_*.txt
    pattern1 = os.path.join(folder_path, "output_*.txt")
    files = glob.glob(pattern1)
    
    if files:
        print(f"Found {len(files)} files using peroxide pattern (output_*.txt)")
        # Sort files by the number after output_
        def extract_number(filename):
            match = re.search(r'output_(\d+)', filename)
            return int(match.group(1)) if match else 0
        files.sort(key=extract_number)
        return files
    
    # Try degradation pattern: sample_*_output_*.txt  
    pattern2 = os.path.join(folder_path, "sample_*_output_*.txt")
    files = glob.glob(pattern2)
    
    if files:
        print(f"Found {len(files)} files using degradation pattern (sample_*_output_*.txt)")
        # Filter by sample if specified
        if sample_name:
            files = [f for f in files if sample_name in os.path.basename(f)]
        
        # Sort files by the number after output_
        def extract_number_degradation(filename):
            match = re.search(r'output_(\d+)', filename)
            return int(match.group(1)) if match else 0
        files.sort(key=extract_number_degradation)
        return files
    
    print(f"No output files found in {folder_path}")
    return []


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
    Get list of unique sample numbers in the folder - simplified for single sample
    """
    pattern = os.path.join(folder_path, "output_*.txt")
    files = glob.glob(pattern)
    
    # For files like "output_1080_20260227_133740.txt", treat as single sample
    if files:
        return [1]  # Return single sample number
    return []


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


def create_wavelength_time_plots_with_sample(processed_data_dir, combined_data, files, sample_name=None, plot_filename=None, csv_filename=None, manual_timepoints=None):
    """
    Create time series plots for specific wavelengths (595 nm, 450 nm, and 450/595 ratio)
    """
    print("\nCreating wavelength-specific time series plots...")
    
    # Initialize data storage for time series
    timepoints = []
    abs_595nm = []
    abs_450nm = []
    
    # Extract timepoints and absorbance values at specific wavelengths
    for i, filepath in enumerate(files):
        if manual_timepoints is not None:
            # Use manual timepoints if provided
            timepoint_num = manual_timepoints[i] if i < len(manual_timepoints) else i
        else:
            # Extract from filename if no manual timepoints
            timepoint = re.search(r'output_(\d+)', os.path.basename(filepath))
            timepoint_num = int(timepoint.group(1)) if timepoint else i
        timepoints.append(timepoint_num)
        
        # Get wavelength and absorbance data
        wavelength, absorbance = read_spectral_file(filepath)
        
        # Extract absorbance at 595 nm and 450 nm
        abs_595 = get_absorbance_at_wavelength(wavelength, absorbance, 595)
        abs_450 = get_absorbance_at_wavelength(wavelength, absorbance, 450)
        
        abs_595nm.append(abs_595)
        abs_450nm.append(abs_450)
    
    # Convert timepoints from seconds to minutes (auto-detect format) - only if not using manual timepoints
    if manual_timepoints is not None:
        timepoints = [float(t) for t in timepoints]  # Manual timepoints assumed to be in minutes already
        print(f"Using manual timepoints: {timepoints} minutes")
    else:
        max_timepoint = max(timepoints) if timepoints else 0
        if max_timepoint > 60:  # Likely in seconds, convert to minutes
            timepoints = [t / 60.0 for t in timepoints]
            print(f"Timepoints detected as seconds, converted to minutes (max: {max(timepoints):.1f} min)")
        else:  # Likely already in minutes, keep as-is
            timepoints = [float(t) for t in timepoints]
            print(f"Timepoints detected as minutes (max: {max(timepoints):.1f} min)")
        print(f"Timepoints detected as minutes, kept as-is (max: {max(timepoints):.1f} min)")
    
    # Calculate 595/450 ratio (product formation)
    ratio_595_450 = np.array(abs_595nm) / np.array(abs_450nm)
    
    # Create the three plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sample_label = f" - Sample {sample_name}" if sample_name is not None else ""
    fig.suptitle(f'Wavelength-Specific Analysis Over Time{sample_label}', fontsize=18, fontweight='bold')
    
    # Plot 1: 595 nm over time
    axes[0, 0].plot(timepoints, abs_595nm, 'o-', color='#4A3A7F', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Time (min)', fontsize=14)
    axes[0, 0].set_ylabel('Absorbance (a.u.)', fontsize=14)
    axes[0, 0].set_title('595 nm Absorbance vs Time', fontsize=14)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=12, direction='in')
    axes[0, 0].grid(False)
    
    # Plot 2: 450 nm over time
    axes[0, 1].plot(timepoints, abs_450nm, 'o-', color="#859DE6", linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Time (min)', fontsize=14)
    axes[0, 1].set_ylabel('Absorbance (a.u.)', fontsize=14)
    axes[0, 1].set_title('450 nm Absorbance vs Time', fontsize=14)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=12, direction='in')
    axes[0, 1].grid(False)
    
    # Plot 3: 595/450 ratio over time
    axes[1, 0].plot(timepoints, ratio_595_450, 'o-', color="#9B6BA8", linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Time (min)', fontsize=14)
    axes[1, 0].set_ylabel('Product Ratio (595/450 nm)', fontsize=14)
    axes[1, 0].set_title('595/450 nm Ratio vs Time (Product Formation)', fontsize=14)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=12, direction='in')
    axes[1, 0].grid(False)
    
    # Plot 4: Combined comparison
    axes[1, 1].plot(timepoints, abs_595nm, 'o-', color='#4A3A7F', linewidth=2, markersize=6, label='595 nm')
    axes[1, 1].plot(timepoints, abs_450nm, 'o-', color='#859DE6', linewidth=2, markersize=6, label='450 nm')
    ax2 = axes[1, 1].twinx()
    ax2.plot(timepoints, ratio_595_450, 's-', color='#9B6BA8', linewidth=2, markersize=6, label='595/450 nm Ratio')
    axes[1, 1].set_xlabel('Time (min)', fontsize=14)
    axes[1, 1].set_ylabel('Absorbance (a.u.)', fontsize=14)
    ax2.set_ylabel('Product Ratio (595/450 nm)', color="#000000", fontsize=14)
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
    print(f"✓ Wavelength time series plots saved as: {plot_filename}")
    
    # Create and save time series data
    time_series_data = pd.DataFrame({
        'timepoint': timepoints,
        'abs_595nm': abs_595nm,
        'abs_450nm': abs_450nm,
        'ratio_595_450': ratio_595_450
    })
    
    time_series_data.to_csv(csv_filename, index=False)
    print(f"✓ Time series data saved as: {csv_filename}")
    
    # Print summary statistics
    print(f"\nWavelength Analysis Summary:")
    print(f"  595 nm - Range: {min(abs_595nm):.4f} to {max(abs_595nm):.4f}")
    print(f"  450 nm - Range: {min(abs_450nm):.4f} to {max(abs_450nm):.4f}") 
    print(f"  595/450 Ratio - Range: {min(ratio_595_450):.4f} to {max(ratio_595_450):.4f}")
    
    return time_series_data


def analyze_spectral_data(folder_path, processed_data_dir=None, sample_name=None, manual_timepoints=None):
    """
    Main function to analyze spectral data from output_# files
    Args:
        folder_path: Directory containing output_# files
        processed_data_dir: Optional directory to save processed results (defaults to folder_path/processed_data)
        sample_name: Optional sample number to filter (e.g., 1 for sample_1)
        manual_timepoints: List of actual timepoints in minutes (e.g., [0, 5, 10, 15, 20, 25, 45])
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
    
    # First pass: extract all timepoints to detect format
    all_timepoints = []
    for i, filepath in enumerate(files):
        # Use manual timepoints if provided
        if manual_timepoints and i < len(manual_timepoints):
            timepoint_num = manual_timepoints[i]
        else:
            timepoint = re.search(r'output_(\d+)', os.path.basename(filepath))
            timepoint_num = int(timepoint.group(1)) if timepoint else 0
        all_timepoints.append(timepoint_num)
    
    # Handle time conversion
    if manual_timepoints:
        print(f"Using manual timepoints: {all_timepoints} minutes")
        convert_to_minutes = False  # Already in correct format
    else:
        # Auto-detect if timepoints are in seconds or minutes
        max_timepoint = max(all_timepoints) if all_timepoints else 0
        convert_to_minutes = max_timepoint > 60  # If max > 60, assume seconds
        
        if convert_to_minutes:
            print(f"Timepoints detected as seconds (max: {max_timepoint}s), will convert to minutes")
        else:
            print(f"Timepoints detected as minutes (max: {max_timepoint}min), will keep as-is")
    
    for i, filepath in enumerate(files):
        try:
            # Use manual timepoints if provided
            if manual_timepoints and i < len(manual_timepoints):
                timepoint_num = manual_timepoints[i]
                timepoint_min = float(timepoint_num)  # Already in minutes
            else:
                # Extract timepoint number from filename
                timepoint = re.search(r'output_(\d+)', os.path.basename(filepath))
                timepoint_num = int(timepoint.group(1)) if timepoint else i
                
                # Smart conversion based on detection
                if convert_to_minutes:
                    timepoint_min = timepoint_num / 60.0  # Convert seconds to minutes
                else:
                    timepoint_min = float(timepoint_num)  # Already in minutes
            
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
        time_series_data = create_wavelength_time_plots_with_sample(processed_data_dir, combined_data, files, sample_name, ts_filename, ts_csv_filename, manual_timepoints=manual_timepoints)
        
        return combined_data, time_series_data
    
    return None


def process_degradation_spectral_data(output_dir, logger=None, sample_name=None, manual_timepoints=None):
    """
    Process spectral data from degradation workflow output directory
    Args:
        output_dir: Path or string - directory containing output_# files
        logger: Optional logger instance for consistent logging
        sample_name: Optional sample number to process (e.g., 1, 2)
        manual_timepoints: List of actual timepoints in minutes (e.g., [0, 5, 10, 15, 20, 25, 45])
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
        results = analyze_spectral_data(folder_path, processed_data_dir, sample_name=sample_name, manual_timepoints=manual_timepoints)
        
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


def process_multiple_cof_folders(base_path, cof_folders, output_dir=None, manual_timepoints=None):
    """
    Process multiple COF folders and create replicate comparison plots.
    
    Args:
        base_path: Base directory containing COF folders
        cof_folders: List of COF folder names (e.g., ['COF_1', 'COF_2', 'COF_3'])
        output_dir: Optional output directory for combined results
        manual_timepoints: List of timepoints in minutes for analysis
    """
    if output_dir is None:
        output_dir = os.path.join(base_path, 'combined_replicates')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(cof_folders)} COF folders for replicate analysis...")
    print(f"Output directory: {output_dir}")
    
    all_replicate_data = {}
    
    # Process each COF folder
    for i, cof_folder in enumerate(cof_folders):
        folder_path = os.path.join(base_path, cof_folder)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
            
        print(f"\nProcessing {cof_folder}...")
        
        # Analyze this folder
        results = process_degradation_spectral_data(folder_path, sample_name=None, manual_timepoints=manual_timepoints)
        
        if results is not None and isinstance(results, tuple):
            combined_df, time_series_df = results
            
            if time_series_df is not None:
                # Store the time series data with replicate label
                all_replicate_data[f'{cof_folder}'] = time_series_df
                print(f"✓ Successfully processed {cof_folder}")
            else:
                print(f"✗ No time series data for {cof_folder}")
        else:
            print(f"✗ Failed to process {cof_folder}")
    
    if not all_replicate_data:
        print("No data processed successfully!")
        return None
    
    # Create combined plots
    print(f"\nCreating combined replicate plots...")
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('COF Product Formation Analysis - Replicates Comparison', fontsize=18, fontweight='bold')
    
    # Colors for different replicates
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot 1: 595 nm comparison
    for i, (replicate_name, data) in enumerate(all_replicate_data.items()):
        color = colors[i % len(colors)]
        axes[0, 0].plot(data['timepoint'], data['abs_595nm'], 'o-', 
                       color=color, linewidth=2, markersize=6, label=replicate_name)
    
    axes[0, 0].set_xlabel('Time (min)', fontsize=14)
    axes[0, 0].set_ylabel('Absorbance (a.u.)', fontsize=14)
    axes[0, 0].set_title('595 nm Absorbance - All Replicates', fontsize=14)
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: 450 nm comparison
    for i, (replicate_name, data) in enumerate(all_replicate_data.items()):
        color = colors[i % len(colors)]
        axes[0, 1].plot(data['timepoint'], data['abs_450nm'], 'o-', 
                       color=color, linewidth=2, markersize=6, label=replicate_name)
    
    axes[0, 1].set_xlabel('Time (min)', fontsize=14)
    axes[0, 1].set_ylabel('Absorbance (a.u.)', fontsize=14)
    axes[0, 1].set_title('450 nm Absorbance - All Replicates', fontsize=14)
    axes[0, 1].legend(fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Ratio comparison (main plot of interest)
    for i, (replicate_name, data) in enumerate(all_replicate_data.items()):
        color = colors[i % len(colors)]
        axes[1, 0].plot(data['timepoint'], data['ratio_595_450'], 'o-', 
                       color=color, linewidth=3, markersize=8, label=replicate_name)
    
    axes[1, 0].set_xlabel('Time (min)', fontsize=14)
    axes[1, 0].set_ylabel('595/450 nm Ratio', fontsize=14)
    axes[1, 0].set_title('595/450 nm Ratio - All Replicates (Product Formation)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistical summary (mean ± std)
    if len(all_replicate_data) > 1:
        # Calculate mean and std for ratio
        max_timepoints = max(len(data['timepoint']) for data in all_replicate_data.values())
        
        # Find common timepoints (assuming similar sampling)
        first_data = list(all_replicate_data.values())[0]
        timepoints = first_data['timepoint'].values
        
        ratio_values = []
        for data in all_replicate_data.values():
            # Interpolate to common timepoints if needed
            if len(data['timepoint']) == len(timepoints):
                ratio_values.append(data['ratio_595_450'].values)
        
        if ratio_values:
            ratio_array = np.array(ratio_values)
            mean_ratio = np.mean(ratio_array, axis=0)
            std_ratio = np.std(ratio_array, axis=0)
            
            axes[1, 1].plot(timepoints, mean_ratio, 'k-', linewidth=3, label='Mean')
            axes[1, 1].fill_between(timepoints, mean_ratio - std_ratio, mean_ratio + std_ratio, 
                                   alpha=0.3, color='gray', label='±1 SD')
            
            axes[1, 1].set_xlabel('Time (min)', fontsize=14)
            axes[1, 1].set_ylabel('450/595 nm Ratio', fontsize=14)
            axes[1, 1].set_title('Statistical Summary (Mean ± SD)', fontsize=14)
            axes[1, 1].legend(fontsize=12)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor statistics', 
                           transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=14)
    else:
        axes[1, 1].text(0.5, 0.5, 'Need >1 replicate\nfor statistics', 
                       transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    
    # Save combined plot
    combined_plot_path = os.path.join(output_dir, 'combined_replicates_analysis.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Combined replicate plot saved: {combined_plot_path}")
    
    # Save combined data to CSV
    combined_csv_data = []
    for replicate_name, data in all_replicate_data.items():
        for _, row in data.iterrows():
            combined_csv_data.append({
                'replicate': replicate_name,
                'timepoint_min': row['timepoint'],
                'abs_595nm': row['abs_595nm'],
                'abs_450nm': row['abs_450nm'],
                'ratio_595_450': row['ratio_595_450']
            })
    
    combined_df = pd.DataFrame(combined_csv_data)
    combined_csv_path = os.path.join(output_dir, 'combined_replicates_data.csv')
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"✓ Combined data saved: {combined_csv_path}")
    
    # Print summary
    print(f"\nReplicate Analysis Summary:")
    print(f"  Processed replicates: {len(all_replicate_data)}")
    for replicate_name in all_replicate_data.keys():
        print(f"    - {replicate_name}")
    
    return all_replicate_data


def process_replicate_paths(replicate_paths, output_dir=None, replicate_names=None, manual_timepoints=None):
    """
    Process replicates from different folder paths and create combined plots
    
    Args:
        replicate_paths: List of full paths to replicate folders
        output_dir: Optional output directory for combined results
        replicate_names: Optional list of names for replicates (defaults to folder names)
        manual_timepoints: List of actual timepoints in minutes (e.g., [0, 5, 10, 15, 20, 25, 45])
    """
    if output_dir is None:
        # Create output in parent directory of first replicate
        first_path_parent = os.path.dirname(os.path.dirname(replicate_paths[0]))
        output_dir = os.path.join(first_path_parent, 'combined_replicates_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(replicate_paths)} replicate folders for analysis...")
    print(f"Output directory: {output_dir}")
    
    all_replicate_data = {}
    
    # Process each replicate path
    for i, folder_path in enumerate(replicate_paths):
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
        
        # Generate replicate name
        if replicate_names and i < len(replicate_names):
            replicate_name = replicate_names[i]
        else:
            # Use parent folder name + subfolder name for clarity
            parent_name = os.path.basename(os.path.dirname(folder_path))
            folder_name = os.path.basename(folder_path)
            replicate_name = f"{parent_name[-8:]}_{folder_name}"  # Last 8 chars of parent + folder
            
        print(f"\nProcessing {replicate_name}...")
        print(f"  Path: {folder_path}")
        
        # Analyze this folder
        results = process_degradation_spectral_data(folder_path, sample_name=None, manual_timepoints=manual_timepoints)
        
        if results is not None and isinstance(results, tuple):
            combined_df, time_series_df = results
            
            if time_series_df is not None:
                # Store the time series data with replicate label
                all_replicate_data[replicate_name] = time_series_df
                print(f"✓ Successfully processed {replicate_name}")
            else:
                print(f"✗ No time series data for {replicate_name}")
        else:
            print(f"✗ Failed to process {replicate_name}")
    
    if not all_replicate_data:
        print("No data processed successfully!")
        return None
    
    # Create combined plots
    print(f"\nCreating combined replicate plots...")
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('COF Product Formation Analysis - Replicates Comparison', fontsize=18, fontweight='bold')
    
    # Colors for different replicates
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot 1: 595 nm comparison
    for i, (replicate_name, data) in enumerate(all_replicate_data.items()):
        color = colors[i % len(colors)]
        axes[0, 0].plot(data['timepoint'], data['abs_595nm'], 'o-', 
                       color=color, linewidth=2, markersize=6, label=replicate_name)
    
    axes[0, 0].set_xlabel('Time (min)', fontsize=14)
    axes[0, 0].set_ylabel('Absorbance (a.u.)', fontsize=14)
    axes[0, 0].set_title('595 nm Absorbance - All Replicates', fontsize=14)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: 450 nm comparison
    for i, (replicate_name, data) in enumerate(all_replicate_data.items()):
        color = colors[i % len(colors)]
        axes[0, 1].plot(data['timepoint'], data['abs_450nm'], 'o-', 
                       color=color, linewidth=2, markersize=6, label=replicate_name)
    
    axes[0, 1].set_xlabel('Time (min)', fontsize=14)
    axes[0, 1].set_ylabel('Absorbance (a.u.)', fontsize=14)
    axes[0, 1].set_title('450 nm Absorbance - All Replicates', fontsize=14)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Ratio comparison (main plot of interest)
    for i, (replicate_name, data) in enumerate(all_replicate_data.items()):
        color = colors[i % len(colors)]
        axes[1, 0].plot(data['timepoint'], data['ratio_595_450'], 'o-', 
                       color=color, linewidth=3, markersize=8, label=replicate_name)
    
    axes[1, 0].set_xlabel('Time (min)', fontsize=14)
    axes[1, 0].set_ylabel('450/595 nm Ratio', fontsize=14)
    axes[1, 0].set_title('450/595 nm Ratio - All Replicates (Product Formation)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistical summary (mean ± std)
    if len(all_replicate_data) > 1:
        # Calculate mean and std for ratio
        max_timepoints = max(len(data['timepoint']) for data in all_replicate_data.values())
        
        # Find common timepoints (assuming similar sampling)
        first_data = list(all_replicate_data.values())[0]
        timepoints = first_data['timepoint'].values
        
        ratio_values = []
        for data in all_replicate_data.values():
            # Interpolate to common timepoints if needed
            if len(data['timepoint']) == len(timepoints):
                ratio_values.append(data['ratio_595_450'].values)
        
        if ratio_values:
            ratio_array = np.array(ratio_values)
            mean_ratio = np.mean(ratio_array, axis=0)
            std_ratio = np.std(ratio_array, axis=0)
            
            axes[1, 1].plot(timepoints, mean_ratio, 'k-', linewidth=3, label='Mean')
            axes[1, 1].fill_between(timepoints, mean_ratio - std_ratio, mean_ratio + std_ratio, 
                                   alpha=0.3, color='gray', label='±1 SD')
            
            axes[1, 1].set_xlabel('Time (min)', fontsize=14)
            axes[1, 1].set_ylabel('450/595 nm Ratio', fontsize=14)
            axes[1, 1].set_title('Statistical Summary (Mean ± SD)', fontsize=14)
            axes[1, 1].legend(fontsize=12)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor statistics', 
                           transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=14)
    else:
        axes[1, 1].text(0.5, 0.5, 'Need >1 replicate\nfor statistics', 
                       transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    
    # Save combined plot
    combined_plot_path = os.path.join(output_dir, 'combined_replicates_analysis.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Combined replicate plot saved: {combined_plot_path}")
    
    # Save combined data to CSV
    combined_csv_data = []
    for replicate_name, data in all_replicate_data.items():
        for _, row in data.iterrows():
            combined_csv_data.append({
                'replicate': replicate_name,
                'timepoint_min': row['timepoint'],
                'abs_595nm': row['abs_595nm'],
                'abs_450nm': row['abs_450nm'],
                'ratio_595_450': row['ratio_595_450']
            })
    
    combined_df = pd.DataFrame(combined_csv_data)
    combined_csv_path = os.path.join(output_dir, 'combined_replicates_data.csv')
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"✓ Combined data saved: {combined_csv_path}")
    
    # Print summary
    print(f"\nReplicate Analysis Summary:")
    print(f"  Processed replicates: {len(all_replicate_data)}")
    for replicate_name in all_replicate_data.keys():
        print(f"    - {replicate_name}")
    
    return all_replicate_data


def main():
    """
    Main function - can process single folder, same-base replicates, or cross-directory replicates
    """
    # Option 1: Process single COF folder
    # folder_path = r"C:\Users\Imaging Controller\Desktop\SQ\PL _COF_bs_new_white_pierce_20260227_131106\COF_1"
    # manual_timepoints = [0, 5, 10, 15, 20, 25, 45] 
    # results = process_degradation_spectral_data(folder_path, sample_name=None, manual_timepoints=manual_timepoints)
    
    # ANALYSIS GROUP 1: Same-base directory COF folders (bs folder)
    base_path_bs = r"C:\Users\Imaging Controller\Desktop\SQ\PL _COF_bs_new_white_pierce_20260227_131106"
    cof_folders = ['COF_1', 'COF_2', 'COF_3']
    manual_timepoints = [0, 5, 10, 15, 20, 25, 45]
    
    print(f"ANALYSIS GROUP 1: BS folder COF analysis")
    print(f"Base path: {base_path_bs}")
    print(f"COF folders: {cof_folders}")
    print(f"Using manual timepoints: {manual_timepoints} minutes")
    print(f"{'='*60}")
    
    replicate_results_1 = process_multiple_cof_folders(base_path_bs, cof_folders, manual_timepoints=manual_timepoints)
    
    if replicate_results_1 is not None:
        print(f"\n✓ Group 1 (BS folder) analysis completed!")
        print(f"Check the 'combined_replicates' folder for results")
    else:
        print(f"\n✗ Group 1 (BS folder) analysis failed!")
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS GROUP 2: Cross-directory PH folder replicates")
    print(f"{'='*60}")
    
    print(f"CROSS-DIRECTORY REPLICATE ANALYSIS")
    
    # Option 3: Process replicates from different base directories (PH folders)
    replicate_paths = [
        r"C:\Users\Imaging Controller\Desktop\SQ\PL _COF_ph_new_white_pierce_33_20260225_152033\COF_1",
        r"C:\Users\Imaging Controller\Desktop\SQ\PL _COF_ph_new_white_pierce_20260225_124615\COF_1", 
        r"C:\Users\Imaging Controller\Desktop\SQ\PL _COF_ph_new_white_pierce_20260225_124615\COF_2"
    ]
    
    # Specify the correct timepoints for your experiment (in minutes)
    manual_timepoints = [0, 5, 10, 15, 20, 25, 45]  # Actual experimental timepoints
    
    # Optional: Custom names for replicates (if not provided, auto-generated from paths)
    replicate_names = ["Replicate_1", "Replicate_2", "Replicate_3"]
    
    print(f"Processing replicates from different folders...")
    print(f"Using manual timepoints: {manual_timepoints} minutes")
    print(f"Replicate paths:")
    for i, path in enumerate(replicate_paths):
        print(f"  {i+1}. {path}")
    print(f"{'='*60}")
    
    # Process replicates from different paths
    replicate_results_2 = process_replicate_paths(replicate_paths, replicate_names=replicate_names, manual_timepoints=manual_timepoints)
    
    if replicate_results_2 is not None:
        print(f"\n✓ Group 2 (PH folders) cross-directory analysis completed!")
        print(f"Check the 'combined_replicates_analysis' folder for:")
        print(f"  1. combined_replicates_analysis.png - Replicate comparison plots")
        print(f"  2. combined_replicates_data.csv - All replicate data")
    else:
        print(f"\n✗ Group 2 (PH folders) analysis failed!")
    
    print("\n" + "="*60)
    print("BOTH ANALYSIS GROUPS COMPLETED!")
    print("Results Summary:")
    if replicate_results_1 is not None:
        print("  ✓ Group 1: BS folder COF analysis (COF_1, COF_2, COF_3) - SUCCESS")
    else:
        print("  ✗ Group 1: BS folder COF analysis - FAILED")
    
    if replicate_results_2 is not None:
        print("  ✓ Group 2: PH folders cross-directory replicate analysis - SUCCESS")
    else:
        print("  ✗ Group 2: PH folders cross-directory replicate analysis - FAILED")
    print("="*60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-Sample Spectral Comparison

Creates 3 plots comparing samples 1, 2, and 3:
1. Absorbance at 555 nm over time
2. Absorbance at 458 nm over time
3. Ratio 458/555 nm over time

Each plot shows all 3 samples with standard deviation bands
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re


def read_spectral_file(filepath):
    """
    Read a spectral data file and return wavelength, absorbance arrays
    """
    df = pd.read_csv(filepath, skiprows=2, header=None)
    wavelength = df.iloc[:, 1].values
    absorbance = df.iloc[:, 2].values
    return wavelength, absorbance


def get_absorbance_at_wavelength(wavelength_array, absorbance_array, target_wavelength):
    """
    Extract absorbance value at a specific wavelength using interpolation
    """
    idx = np.argmin(np.abs(wavelength_array - target_wavelength))
    if wavelength_array[idx] == target_wavelength:
        return absorbance_array[idx]
    return np.interp(target_wavelength, wavelength_array, absorbance_array)


def find_sample_files(folder_path, sample_num):
    """
    Find all output files for a specific sample
    """
    pattern = os.path.join(folder_path, f"sample_{sample_num}_output_*.txt")
    files = glob.glob(pattern)
    
    # Sort by timepoint number
    def extract_number(filename):
        match = re.search(r'output_(\d+(?:\.\d+)?)', filename)
        return float(match.group(1)) if match else 0
    
    files.sort(key=extract_number)
    return files


def process_sample_data(folder_path, sample_num):
    """
    Process all spectral files for a sample and extract 555nm, 458nm values
    """
    files = find_sample_files(folder_path, sample_num if sample_num else None)
    
    if not files:
        print(f"Warning: No files found for sample {sample_num if sample_num else 1}")
        return None
    
    timepoints = []
    abs_555nm = []
    abs_458nm = []
    
    for filepath in files:
        # Extract timepoint from filename (e.g., output_300.0 -> 300.0 seconds)
        timepoint = re.search(r'output_(\d+(?:\.\d+)?)', os.path.basename(filepath))
        timepoint_num = float(timepoint.group(1)) if timepoint else 0
        timepoints.append(timepoint_num)
        
        # Read and extract values at specific wavelengths
        wavelength, absorbance = read_spectral_file(filepath)
        abs_555 = get_absorbance_at_wavelength(wavelength, absorbance, 555)
        abs_458 = get_absorbance_at_wavelength(wavelength, absorbance, 458)
        
        abs_555nm.append(abs_555)
        abs_458nm.append(abs_458)
    
    # Sort by timepoint
    sorted_indices = np.argsort(timepoints)
    timepoints = np.array(timepoints)[sorted_indices] / 60.0  # Convert to minutes
    abs_555nm = np.array(abs_555nm)[sorted_indices]
    abs_458nm = np.array(abs_458nm)[sorted_indices]
    
    ratio_458_555 = abs_458nm / abs_555nm
    
    return {
        'timepoints': timepoints,
        'abs_555nm': abs_555nm,
        'abs_458nm': abs_458nm,
        'ratio_458_555': ratio_458_555
    }


def create_comparison_plots(folder_path, output_dir=None):
    """
    Create 3 subplots in a single figure comparing all samples with error bars
    """
    print("Loading spectral data from all samples...")
    
    # Load data for all three samples
    sample1_data = process_sample_data(folder_path, 1)
    sample2_data = process_sample_data(folder_path, 2)
    sample3_data = process_sample_data(folder_path, 3)
    
    samples_loaded = []
    data_list = []
    
    if sample1_data is not None:
        samples_loaded.append(1)
        data_list.append(sample1_data)
    if sample2_data is not None:
        samples_loaded.append(2)
        data_list.append(sample2_data)
    if sample3_data is not None:
        samples_loaded.append(3)
        data_list.append(sample3_data)
    
    if not data_list:
        print("Error: Could not load any sample data!")
        return
    
    print(f"Successfully loaded samples: {samples_loaded}\n")
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(folder_path, 'comparison_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timepoints (use first sample as reference)
    timepoints = data_list[0]['timepoints']
    
    # Find minimum timepoints across all samples
    min_timepoints = min(len(data['timepoints']) for data in data_list)
    timepoints = data_list[0]['timepoints'][:min_timepoints]
    
    # Prepare arrays for all metrics, keeping only common timepoints
    all_abs_555nm = []
    all_abs_458nm = []
    all_ratio_458_555 = []
    
    for data in data_list:
        all_abs_555nm.append(data['abs_555nm'][:min_timepoints])
        all_abs_458nm.append(data['abs_458nm'][:min_timepoints])
        all_ratio_458_555.append(data['ratio_458_555'][:min_timepoints])
    
    all_abs_555nm = np.array(all_abs_555nm)
    all_abs_458nm = np.array(all_abs_458nm)
    all_ratio_458_555 = np.array(all_ratio_458_555)
    
    # Calculate statistics
    mean_abs_555nm = np.mean(all_abs_555nm, axis=0)
    std_abs_555nm = np.std(all_abs_555nm, axis=0)
    
    mean_abs_458nm = np.mean(all_abs_458nm, axis=0)
    std_abs_458nm = np.std(all_abs_458nm, axis=0)
    
    mean_ratio_458_555 = np.mean(all_ratio_458_555, axis=0)
    std_ratio_458_555 = np.std(all_ratio_458_555, axis=0)
    
    # Define colors for each sample
    sample_colors = {1: '#4A3A7F', 2: '#859DE6', 3: '#9B6BA8'}
    
    # Create figure with 2 rows: measurements and standard deviations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Multi-Sample Spectral Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    # ===== Subplot 1: 555 nm Absorbance =====
    ax = axes[0, 0]
    
    # Plot individual samples
    for idx, sample_num in enumerate(samples_loaded):
        ax.plot(timepoints, all_abs_555nm[idx], 'o-', color=sample_colors[sample_num], 
                linewidth=2.5, markersize=6, label=f'Sample {sample_num}', alpha=0.8)
    
    # Add error bars only (no mean line)
    ax.errorbar(timepoints, mean_abs_555nm, yerr=std_abs_555nm, fmt='none', 
                color='black', elinewidth=2, capsize=5, capthick=2, alpha=0.95, zorder=10)
    
    ax.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Absorbance (a.u.)', fontsize=11, fontweight='bold')
    ax.set_title('555 nm Absorbance', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 61)
    ax.set_xticks(np.arange(0, 65, 5))
    ax.legend(fontsize=9, loc='best', framealpha=0.95)
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in')
    
    # ===== Subplot 2: 458 nm Absorbance =====
    ax = axes[0, 1]
    
    # Plot individual samples
    for idx, sample_num in enumerate(samples_loaded):
        ax.plot(timepoints, all_abs_458nm[idx], 'o-', color=sample_colors[sample_num], 
                linewidth=2.5, markersize=6, label=f'Sample {sample_num}', alpha=0.8)
    
    # Add error bars only (no mean line)
    ax.errorbar(timepoints, mean_abs_458nm, yerr=std_abs_458nm, fmt='none', 
                color='black', elinewidth=2, capsize=5, capthick=2, alpha=0.95, zorder=10)
    
    ax.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Absorbance (a.u.)', fontsize=11, fontweight='bold')
    ax.set_title('458 nm Absorbance', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 61)
    ax.set_xticks(np.arange(0, 65, 5))
    ax.legend(fontsize=9, loc='best', framealpha=0.95)
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in')
    ax = axes[0, 2]
    
    # Plot individual samples
    for idx, sample_num in enumerate(samples_loaded):
        ax.plot(timepoints, all_ratio_458_555[idx], 's-', color=sample_colors[sample_num], 
                linewidth=2, markersize=5, label=f'Sample {sample_num}', alpha=0.7)
    
    # Add error bars only (no mean line)
    ax.errorbar(timepoints, mean_ratio_458_555, yerr=std_ratio_458_555, fmt='none', 
                color='black', elinewidth=2, capsize=5, capthick=2, alpha=0.95, zorder=10)
    
    ax.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Absorbance Ratio (458/555 nm)', fontsize=11, fontweight='bold')
    ax.set_title('458/555 nm Ratio', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 61)
    ax.set_xticks(np.arange(0, 65, 5))
    ax.legend(fontsize=9, loc='best', framealpha=0.95)
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in')
    
    # ===== Subplot 4: 555 nm Standard Deviation =====
    ax = axes[1, 0]
    # Exclude timepoint 50 for bar chart
    bar_mask = timepoints != 50
    timepoints_bar = timepoints[bar_mask]
    std_percent_555nm_bar = (std_abs_555nm / mean_abs_555nm)[bar_mask] * 100
    x_positions = np.arange(len(timepoints_bar))
    ax.bar(x_positions, std_percent_555nm_bar, width=0.6, color='#4A3A7F', alpha=0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{int(tp)}' for tp in timepoints_bar], fontsize=10)
    ax.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=11, fontweight='bold')
    ax.set_title('555 nm Std Dev', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in')
    
    # ===== Subplot 5: 458 nm Standard Deviation =====
    ax = axes[1, 1]
    std_percent_458nm_bar = (std_abs_458nm / mean_abs_458nm)[bar_mask] * 100
    ax.bar(x_positions, std_percent_458nm_bar, width=0.6, color='#859DE6', alpha=0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{int(tp)}' for tp in timepoints_bar], fontsize=10)
    ax.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=11, fontweight='bold')
    ax.set_title('458 nm Std Dev', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in')
    
    # ===== Subplot 6: 458/555 Ratio Standard Deviation =====
    ax = axes[1, 2]
    std_percent_ratio_bar = (std_ratio_458_555 / mean_ratio_458_555)[bar_mask] * 100
    ax.bar(x_positions, std_percent_ratio_bar, width=0.6, color='#9B6BA8', alpha=0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{int(tp)}' for tp in timepoints_bar], fontsize=10)
    ax.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=11, fontweight='bold')
    ax.set_title('458/555 Ratio Std Dev', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in')
    
    plt.tight_layout()
    
    # Save the combined plot
    combined_plot_filename = os.path.join(output_dir, 'combined_comparison.png')
    plt.savefig(combined_plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Combined comparison plot saved as: {combined_plot_filename}")
    
    # Display the plot
    plt.show()
    
    # Calculate CV for table and summary
    cv_abs_555nm = (std_abs_555nm / mean_abs_555nm) * 100
    cv_abs_458nm = (std_abs_458nm / mean_abs_458nm) * 100
    cv_ratio_458_555 = (std_ratio_458_555 / mean_ratio_458_555) * 100
    
    # Create and save table as figure
    fig_table, ax_table = plt.subplots(figsize=(16, 10))
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Prepare table data with headers
    table_data = []
    table_data.append(['Time (min)', '555nm Mean', '555nm SD', '555nm CV%', '458nm Mean', '458nm SD', '458nm CV%', 'Ratio Mean', 'Ratio SD', 'Ratio CV%'])
    
    for i, tp in enumerate(timepoints):
        table_data.append([
            f'{tp:.1f}',
            f'{mean_abs_555nm[i]:.4f}',
            f'{std_abs_555nm[i]:.4f}',
            f'{cv_abs_555nm[i]:.2f}',
            f'{mean_abs_458nm[i]:.4f}',
            f'{std_abs_458nm[i]:.4f}',
            f'{cv_abs_458nm[i]:.2f}',
            f'{mean_ratio_458_555[i]:.4f}',
            f'{std_ratio_458_555[i]:.4f}',
            f'{cv_ratio_458_555[i]:.2f}'
        ])
    
    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                           colWidths=[0.08, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4A3A7F')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    plt.title('Multi-Sample Spectral Comparison - Detailed Statistics', fontsize=14, fontweight='bold', pad=20)
    
    table_filename = os.path.join(output_dir, 'statistics_table.png')
    plt.savefig(table_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Statistics table saved as: {table_filename}")
    
    # Display the table
    plt.show()
    
    # Create summary statistics CSV
    # Create comprehensive summary statistics with CV
    summary_data = pd.DataFrame({
        'timepoint_min': timepoints,
        'abs_555nm_mean': mean_abs_555nm,
        'abs_555nm_std': std_abs_555nm,
        'abs_555nm_cv_percent': cv_abs_555nm,
        'abs_458nm_mean': mean_abs_458nm,
        'abs_458nm_std': std_abs_458nm,
        'abs_458nm_cv_percent': cv_abs_458nm,
        'ratio_458_555_mean': mean_ratio_458_555,
        'ratio_458_555_std': std_ratio_458_555,
        'ratio_458_555_cv_percent': cv_ratio_458_555
    })
    
    summary_filename = os.path.join(output_dir, 'comparison_summary_statistics.csv')
    summary_data.to_csv(summary_filename, index=False)
    print(f"✓ Summary statistics saved as: {summary_filename}")
    
    # Print summary
    print("\n" + "="*100)
    print("MULTI-SAMPLE COMPARISON SUMMARY - DETAILED STATISTICS")
    print("="*100)
    print(f"Samples analyzed: {samples_loaded}\n")
    
    # Print table with all metrics
    print(f"{'Time(min)':<12} {'555nm Mean':<12} {'555nm SD':<12} {'555nm CV%':<12} {'458nm Mean':<12} {'458nm SD':<12} {'458nm CV%':<12} {'Ratio Mean':<12} {'Ratio SD':<12} {'Ratio CV%':<12}")
    print("-" * 100)
    for i, tp in enumerate(timepoints):
        print(f"{tp:<12.1f} {mean_abs_555nm[i]:<12.4f} {std_abs_555nm[i]:<12.4f} {cv_abs_555nm[i]:<12.2f} {mean_abs_458nm[i]:<12.4f} {std_abs_458nm[i]:<12.4f} {cv_abs_458nm[i]:<12.2f} {mean_ratio_458_555[i]:<12.4f} {std_ratio_458_555[i]:<12.4f} {cv_ratio_458_555[i]:<12.2f}")
    print("="*100)


def main():
    """
    Main function
    """
    folder_path = "/Users/serenaqiu/Downloads/p(IDT-TIT)_1000xHCl_2MeTHF_3Replicates"
    
    print(f"Analyzing spectral data from: {folder_path}\n")
    
    # Create comparison plots
    create_comparison_plots(folder_path)
    
    print("\nAll comparisons complete!")


if __name__ == "__main__":
    main()

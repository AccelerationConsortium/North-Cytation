#!/usr/bin/env python3
"""
Spectral Comparison Between Two Datasets

Creates 3 plots comparing two different spectral datasets:
1. Absorbance at 556 nm over time
2. Absorbance at 428 nm over time
3. Ratio 428/556 nm over time

Each plot shows both datasets overlaid for comparison
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
    Process all spectral files for a sample and extract 556nm, 428nm values
    """
    files = find_sample_files(folder_path, sample_num if sample_num else None)
    
    if not files:
        print(f"Warning: No files found for sample {sample_num if sample_num else 1}")
        return None
    
    timepoints = []
    abs_556nm = []
    abs_428nm = []
    
    for filepath in files:
        # Extract timepoint from filename (e.g., output_300.0 -> 300.0 seconds)
        timepoint = re.search(r'output_(\d+(?:\.\d+)?)', os.path.basename(filepath))
        timepoint_num = float(timepoint.group(1)) if timepoint else 0
        timepoints.append(timepoint_num)
        
        # Read and extract values at specific wavelengths
        wavelength, absorbance = read_spectral_file(filepath)
        abs_556 = get_absorbance_at_wavelength(wavelength, absorbance, 556)
        abs_428 = get_absorbance_at_wavelength(wavelength, absorbance, 428)
        
        abs_556nm.append(abs_556)
        abs_428nm.append(abs_428)
    
    # Sort by timepoint
    sorted_indices = np.argsort(timepoints)
    timepoints = np.array(timepoints)[sorted_indices] / 60.0  # Convert to minutes
    abs_556nm = np.array(abs_556nm)[sorted_indices]
    abs_428nm = np.array(abs_428nm)[sorted_indices]
    
    ratio_428_556 = abs_428nm / abs_556nm
    
    return {
        'timepoints': timepoints,
        'abs_556nm': abs_556nm,
        'abs_428nm': abs_428nm,
        'ratio_428_556': ratio_428_556
    }


def create_comparison_plots(folder_path_1, folder_path_2, output_dir=None):
    """
    Create 3 subplots comparing two different spectral datasets
    """
    print("Loading spectral data from dataset 1...")
    
    # Load data for all three samples from folder 1
    sample1_data_1 = process_sample_data(folder_path_1, 1)
    sample2_data_1 = process_sample_data(folder_path_1, 2)
    sample3_data_1 = process_sample_data(folder_path_1, 3)
    
    data_list_1 = []
    if sample1_data_1 is not None:
        data_list_1.append(sample1_data_1)
    if sample2_data_1 is not None:
        data_list_1.append(sample2_data_1)
    if sample3_data_1 is not None:
        data_list_1.append(sample3_data_1)
    
    if not data_list_1:
        print("Error: Could not load any sample data from dataset 1!")
        return
    
    print(f"Successfully loaded 3 samples from dataset 1\n")
    
    print("Loading spectral data from dataset 2...")
    
    # Load data for all three samples from folder 2
    sample1_data_2 = process_sample_data(folder_path_2, 1)
    sample2_data_2 = process_sample_data(folder_path_2, 2)
    sample3_data_2 = process_sample_data(folder_path_2, 3)
    
    data_list_2 = []
    if sample1_data_2 is not None:
        data_list_2.append(sample1_data_2)
    if sample2_data_2 is not None:
        data_list_2.append(sample2_data_2)
    if sample3_data_2 is not None:
        data_list_2.append(sample3_data_2)
    
    if not data_list_2:
        print("Error: Could not load any sample data from dataset 2!")
        return
    
    print(f"Successfully loaded 3 samples from dataset 2\n")
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = "/Users/serenaqiu/Desktop/Human vs Robot Data/Degradation_Automated_Data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timepoints from both datasets (use first sample as reference)
    timepoints_1 = data_list_1[0]['timepoints']
    timepoints_2 = data_list_2[0]['timepoints']
    
    # Find minimum timepoints across samples within each dataset
    min_timepoints_1 = min(len(data['timepoints']) for data in data_list_1)
    min_timepoints_2 = min(len(data['timepoints']) for data in data_list_2)
    
    timepoints_1 = data_list_1[0]['timepoints'][:min_timepoints_1]
    timepoints_2 = data_list_2[0]['timepoints'][:min_timepoints_2]
    
    # Prepare arrays for dataset 1
    all_abs_556nm_1 = np.array([data['abs_556nm'][:min_timepoints_1] for data in data_list_1])
    all_abs_428nm_1 = np.array([data['abs_428nm'][:min_timepoints_1] for data in data_list_1])
    all_ratio_428_556_1 = np.array([data['ratio_428_556'][:min_timepoints_1] for data in data_list_1])
    
    # Prepare arrays for dataset 2
    all_abs_556nm_2 = np.array([data['abs_556nm'][:min_timepoints_2] for data in data_list_2])
    all_abs_428nm_2 = np.array([data['abs_428nm'][:min_timepoints_2] for data in data_list_2])
    all_ratio_428_556_2 = np.array([data['ratio_428_556'][:min_timepoints_2] for data in data_list_2])
    
    # Calculate means across samples
    mean_abs_556nm_1 = np.mean(all_abs_556nm_1, axis=0)
    mean_abs_428nm_1 = np.mean(all_abs_428nm_1, axis=0)
    mean_ratio_428_556_1 = np.mean(all_ratio_428_556_1, axis=0)
    
    mean_abs_556nm_2 = np.mean(all_abs_556nm_2, axis=0)
    mean_abs_428nm_2 = np.mean(all_abs_428nm_2, axis=0)
    mean_ratio_428_556_2 = np.mean(all_ratio_428_556_2, axis=0)
    
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('p(IDT-TIT) in 2MeTHF Degradation', fontsize=16, fontweight='bold', y=0.995)
    
    # Define colors for each dataset
    color_1 = '#B79CEE'
    color_2 = "#4A3A7F"
    
    # ===== Subplot 1: 556 nm Absorbance =====
    ax = axes[0]
    ax.plot(timepoints_1, mean_abs_556nm_1, 'o-', color=color_1, 
            linewidth=2.5, markersize=7, label='500x HCl', alpha=0.85)
    ax.plot(timepoints_2, mean_abs_556nm_2, 's-', color=color_2, 
            linewidth=2.5, markersize=7, label='1000x HCl', alpha=0.85)
    
    ax.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Absorbance (a.u.)', fontsize=11, fontweight='bold')
    ax.set_title('556 nm Absorbance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ===== Subplot 2: 428 nm Absorbance =====
    ax = axes[1]
    ax.plot(timepoints_1, mean_abs_428nm_1, 'o-', color=color_1, 
            linewidth=2.5, markersize=7, label='500x HCl', alpha=0.85)
    ax.plot(timepoints_2, mean_abs_428nm_2, 's-', color=color_2, 
            linewidth=2.5, markersize=7, label='1000x HCl', alpha=0.85)
    
    ax.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Absorbance (a.u.)', fontsize=11, fontweight='bold')
    ax.set_title('428 nm Absorbance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ===== Subplot 3: 428/556 nm Ratio =====
    ax = axes[2]
    ax.plot(timepoints_1, mean_ratio_428_556_1, 'o-', color=color_1, 
            linewidth=2.5, markersize=7, label='500x HCl', alpha=0.85)
    ax.plot(timepoints_2, mean_ratio_428_556_2, 's-', color=color_2, 
            linewidth=2.5, markersize=7, label='1000x HCl', alpha=0.85)
    
    ax.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Absorbance Ratio (428/556 nm)', fontsize=11, fontweight='bold')
    ax.set_title('428/556 nm Ratio', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save the combined plot
    combined_plot_filename = os.path.join(output_dir, 'two_dataset_comparison.png')
    plt.savefig(combined_plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved as: {combined_plot_filename}")
    
    # Display the plot
    plt.show()
    
    # Create summary table
    fig_table, ax_table = plt.subplots(figsize=(16, 8))
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Prepare table data with headers
    table_data = []
    table_data.append(['Dataset', 'Time (min)', '556nm (a.u.)', '428nm (a.u.)', 'Ratio 428/556'])
    
    for i, tp in enumerate(timepoints_1):
        table_data.append([
            '500x HCl',
            f'{tp:.1f}',
            f'{mean_abs_556nm_1[i]:.4f}',
            f'{mean_abs_428nm_1[i]:.4f}',
            f'{mean_ratio_428_556_1[i]:.4f}',
        ])
    
    # Add empty row for spacing
    table_data.append(['', '', '', '', ''])
    
    for i, tp in enumerate(timepoints_2):
        table_data.append([
            '1000x HCl',
            f'{tp:.1f}',
            f'{mean_abs_556nm_2[i]:.4f}',
            f'{mean_abs_428nm_2[i]:.4f}',
            f'{mean_ratio_428_556_2[i]:.4f}',
        ])
    
    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                           colWidths=[0.15, 0.15, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4A3A7F')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if table_data[i][0] == '':  # Empty spacer row
                table[(i, j)].set_facecolor('#CCCCCC')
            elif table_data[i][0] == '500x HCl':
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E8F0F8')
                else:
                    table[(i, j)].set_facecolor('#F5F5F5')
            else:
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#FFF4E6')
                else:
                    table[(i, j)].set_facecolor('#F5F5F5')
    
    plt.title('p(IDT-TIT) in 2MeTHF Degradation - Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    table_filename = os.path.join(output_dir, 'two_dataset_comparison_table.png')
    plt.savefig(table_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison table saved as: {table_filename}")
    plt.show()
    
    # Print summary
    print("\n" + "="*90)
    print("p(IDT-TIT) IN 2MeTHF DEGRADATION SUMMARY")
    print("="*90)
    
    print("\n500x HCl:")
    print(f"{'Time(min)':<12} {'556nm':<15} {'428nm':<15} {'Ratio 428/556':<15}")
    print("-" * 60)
    for i, tp in enumerate(timepoints_1):
        print(f"{tp:<12.1f} {mean_abs_556nm_1[i]:<15.4f} {mean_abs_428nm_1[i]:<15.4f} {mean_ratio_428_556_1[i]:<15.4f}")
    
    print("\n1000x HCl:")
    print(f"{'Time(min)':<12} {'556nm':<15} {'428nm':<15} {'Ratio 428/556':<15}")
    print("-" * 60)
    for i, tp in enumerate(timepoints_2):
        print(f"{tp:<12.1f} {mean_abs_556nm_2[i]:<15.4f} {mean_abs_428nm_2[i]:<15.4f} {mean_ratio_428_556_2[i]:<15.4f}")
    
    print("="*90)


def main():
    """
    Main function - compare two spectral datasets
    """
    folder_path_1 = "/Users/serenaqiu/Desktop/Human vs Robot Data/Degradation_Automated_Data/p(IDT-TIT)-0.05mgml-500xHCl All"
    folder_path_2 = "/Users/serenaqiu/Desktop/Human vs Robot Data/Degradation_Automated_Data/p(IDT-TIT)_1000xHCl_2MeTHF_All"
    
    print(f"Comparing spectral data from two datasets:\n")
    print(f"Dataset 1: {folder_path_1}\n")
    print(f"Dataset 2: {folder_path_2}\n")
    
    # Create comparison plots
    create_comparison_plots(folder_path_1, folder_path_2)
    
    print("\nAll comparisons complete!")


if __name__ == "__main__":
    main()

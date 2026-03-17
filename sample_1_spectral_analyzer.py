#!/usr/bin/env python3
"""
Sample 1 Spectral Analyzer

Analyzes and graphs spectral data for sample 1 from degradation experiments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import os
import glob
import re
from pathlib import Path


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


def find_sample_1_files(folder_path):
    """
    Find all output files for sample 1
    """
    pattern = os.path.join(folder_path, "sample_1_output_*.txt")
    files = glob.glob(pattern)
    
    # Sort by timepoint number
    def extract_number(filename):
        match = re.search(r'output_(\d+(?:\.\d+)?)', filename)
        return float(match.group(1)) if match else 0
    
    files.sort(key=extract_number)
    return files


def analyze_sample_1(folder_path, output_dir=None):
    """
    Analyze and graph sample 1 spectral data
    """
    print(f"Analyzing Sample 1 from: {folder_path}\n")
    
    # Use the main folder for output (save directly there)
    if output_dir is None:
        output_dir = folder_path
    os.makedirs(output_dir, exist_ok=True)
    
    # Find sample 1 files
    files = find_sample_1_files(folder_path)
    
    if not files:
        print("Error: No sample_1 output files found!")
        return
    
    print(f"Found {len(files)} sample 1 files")
    
    # Set font to Helvetica Neue
    plt.rcParams['font.sans-serif'] = ['Helvetica Neue']
    plt.rcParams['font.family'] = 'sans-serif'
    
    # ===== Plot 1: Full Spectral Data Over Time =====
    fig1 = plt.figure(figsize=(12, 8))
    
    # Create custom color gradient
    custom_colors = ['#4A3A7F', '#9B6BA8', '#D485C0', "#E0A5C9", "#F1CE9A"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('peach_pink_purple_blue', custom_colors)
    colors = custom_cmap(np.linspace(0, 1, len(files)))
    
    timepoints_list = []
    for i, filepath in enumerate(files):
        wavelength, absorbance = read_spectral_file(filepath)
        
        # Extract timepoint
        timepoint = re.search(r'output_(\d+(?:\.\d+)?)', os.path.basename(filepath))
        timepoint_num = float(timepoint.group(1)) if timepoint else i
        timepoint_min = timepoint_num / 60.0
        timepoints_list.append(timepoint_min)
        
        # Plot
        plt.plot(wavelength, absorbance, color=colors[i],
                label=f'{int(timepoint_min)} min', linewidth=2)
    
    plt.xlabel('wavelength (nm)', fontsize=24)
    plt.ylabel('absorbance (a.u.)', fontsize=24)
    plt.title('Sample 1 - Spectral Data Over Time', fontsize=24, fontweight='bold')
    plt.xlim(300, 900)
    plt.legend(fontsize=23, loc='best', frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=23, direction='in', length=12, width=1.2)
    plt.tick_params(axis='both', which='minor', direction='in', length=6, width=1.2)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.7)
    ax.spines['bottom'].set_linewidth(1.7)
    plt.tight_layout()
    
    plot1_filename = os.path.join(output_dir, 'sample_1_full_spectral.png')
    plt.savefig(plot1_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Full spectral plot saved: {plot1_filename}")
    plt.show()
    
    # ===== Plot 2: Wavelength-Specific Time Series =====
    timepoints = np.array(timepoints_list)
    abs_556nm = []
    abs_428nm = []
    
    for filepath in files:
        wavelength, absorbance = read_spectral_file(filepath)
        abs_556 = get_absorbance_at_wavelength(wavelength, absorbance, 556)
        abs_428 = get_absorbance_at_wavelength(wavelength, absorbance, 428)
        abs_556nm.append(abs_556)
        abs_428nm.append(abs_428)
    
    abs_556nm = np.array(abs_556nm)
    abs_428nm = np.array(abs_428nm)
    ratio_556_428 = abs_556nm / abs_428nm
    
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('Sample 1 - Wavelength-Specific Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Subplot 1: 556 nm
    axes[0].plot(timepoints, abs_556nm, 'o-', color='#4A3A7F', linewidth=2.5, markersize=6)
    axes[0].set_xlabel('time (min)', fontsize=24)
    axes[0].set_ylabel('absorbance (a.u.)', fontsize=24)
    axes[0].set_title('556 nm absorbance', fontsize=22)
    axes[0].set_ylim(bottom=0)
    axes[0].tick_params(axis='both', which='major', labelsize=23, direction='in', length=12, width=1.2)
    axes[0].tick_params(axis='both', which='minor', direction='in', length=6, width=1.2)
    axes[0].xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axes[0].yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['left'].set_linewidth(1.7)
    axes[0].spines['bottom'].set_linewidth(1.7)
    # Remove 0 tick
    yticks = axes[0].get_yticks()
    axes[0].set_yticks(yticks[np.abs(yticks) > 1e-10])
    
    # Subplot 2: 428 nm
    axes[1].plot(timepoints, abs_428nm, 'o-', color='#859DE6', linewidth=2.5, markersize=6)
    axes[1].set_xlabel('time (min)', fontsize=24)
    axes[1].set_ylabel('absorbance (a.u.)', fontsize=24)
    axes[1].set_title('428 nm absorbance', fontsize=22)
    axes[1].set_ylim(bottom=0)
    axes[1].tick_params(axis='both', which='major', labelsize=23, direction='in', length=12, width=1.2)
    axes[1].tick_params(axis='both', which='minor', direction='in', length=6, width=1.2)
    axes[1].xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axes[1].yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_linewidth(1.7)
    axes[1].spines['bottom'].set_linewidth(1.7)
    # Remove 0 tick
    yticks = axes[1].get_yticks()
    axes[1].set_yticks(yticks[np.abs(yticks) > 1e-10])
    
    # Subplot 3: 556/428 ratio
    axes[2].plot(timepoints, ratio_556_428, 's-', color='#9B6BA8', linewidth=2, markersize=5)
    axes[2].set_xlabel('time (min)', fontsize=24)
    axes[2].set_ylabel('absorbance ratio (556/428 nm)', fontsize=24)
    axes[2].set_title('556/428 nm ratio', fontsize=22)
    axes[2].set_ylim(bottom=0, top=2.5)
    axes[2].tick_params(axis='both', which='major', labelsize=23, direction='in', length=12, width=1.2)
    axes[2].tick_params(axis='both', which='minor', direction='in', length=6, width=1.2)
    axes[2].xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axes[2].yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['left'].set_linewidth(1.7)
    axes[2].spines['bottom'].set_linewidth(1.7)
    # Remove 0 tick
    yticks = axes[2].get_yticks()
    axes[2].set_yticks(yticks[np.abs(yticks) > 1e-10])
    
    plt.tight_layout()
    
    plot2_filename = os.path.join(output_dir, 'sample_1_wavelength_analysis.png')
    plt.savefig(plot2_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Wavelength analysis plot saved: {plot2_filename}")
    plt.show()
    
    # ===== Save Data to CSV =====
    data_df = pd.DataFrame({
        'timepoint (min)': timepoints,
        'abs_556nm': abs_556nm,
        'abs_428nm': abs_428nm,
        'ratio_556_428': ratio_556_428
    })
    
    csv_filename = os.path.join(output_dir, 'sample_1_wavelength_data.csv')
    data_df.to_csv(csv_filename, index=False)
    print(f"✓ Data saved: {csv_filename}")
    
    print("\n" + "="*60)
    print("Sample 1 Analysis Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}")


def main():
    """
    Main function
    """
    folder_path = "/Users/serenaqiu/Desktop/Human vs Robot Data/Degradation_Automated_Data/p(IDT-TIT)-0.05mgml-500xHCl All"
    
    analyze_sample_1(folder_path)


if __name__ == "__main__":
    main()

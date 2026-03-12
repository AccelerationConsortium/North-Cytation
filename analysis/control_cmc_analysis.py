#!/usr/bin/env python3
"""
CMC Control Data Analysis
Analyzes control wells to plot ratio vs concentration curves for SDS and TTAB surfactants
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import warnings

# Define the Boltzmann sigmoid function for CMC fitting
def boltzmann(x, A1, A2, x0, dx):
    """Boltzmann sigmoid function for CMC determination"""
    return A2 + (A1 - A2) / (1 + np.exp((x - x0) / dx))

def fit_cmc_curve(concentrations, ratios, log_scale=True):
    """Fit Boltzmann curve to determine CMC"""
    try:
        if log_scale:
            x_data = np.log10(concentrations)
        else:
            x_data = concentrations
            
        # Initial guess for parameters [A1, A2, x0, dx]
        p0 = [max(ratios), min(ratios), np.mean(x_data), (max(x_data) - min(x_data)) / 5]
        
        # Fit the data
        popt, pcov = curve_fit(boltzmann, x_data, ratios, p0, maxfev=5000)
        A1, A2, x0, dx = popt
        
        # Convert back from log scale
        if log_scale:
            cmc_estimate = 10**x0
        else:
            cmc_estimate = x0
            
        # Calculate R-squared
        residuals = ratios - boltzmann(x_data, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ratios - np.mean(ratios))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return cmc_estimate, r_squared, popt, x_data
        
    except Exception as e:
        warnings.warn(f"CMC fitting failed: {e}")
        return None, 0, None, None

def analyze_cmc_controls(csv_file_path, surfactant_a_name="SurfA", surfactant_b_name="SurfB"):
    """
    Analyze CMC control data and create ratio vs concentration plots
    
    Parameters:
    csv_file_path (str): Path to the iterative experiment results CSV
    """
    
    # Read the CSV data
    df = pd.read_csv(csv_file_path)
    
    # Filter for control data only
    control_data = df[df['well_type'] == 'control'].copy()
    
    print(f"Total control wells: {len(control_data)}")
    print(f"Control types found: {sorted(control_data['control_type'].unique())}")
    
    # Extract surfactant A CMC controls (cmc_{surfactant_a_name}_1 through cmc_{surfactant_a_name}_8)
    surfactant_a_controls = control_data[control_data['control_type'].str.startswith(f'cmc_{surfactant_a_name}_')].copy()
    
    # Extract surfactant B CMC controls (cmc_{surfactant_b_name}_1 through cmc_{surfactant_b_name}_8)  
    surfactant_b_controls = control_data[control_data['control_type'].str.startswith(f'cmc_{surfactant_b_name}_')].copy()
    
    # Get reference controls
    water_blank = control_data[control_data['control_type'] == 'water_blank']
    surfactant_a_stock = control_data[control_data['control_type'] == 'surfactant_A_stock']
    surfactant_b_stock = control_data[control_data['control_type'] == 'surfactant_B_stock']
    
    print(f"{surfactant_a_name} CMC points: {len(surfactant_a_controls)}")
    print(f"{surfactant_b_name} CMC points: {len(surfactant_b_controls)}")
    print(f"Water blanks: {len(water_blank)}")
    print(f"{surfactant_a_name} stock: {len(surfactant_a_stock)}")
    print(f"{surfactant_b_name} stock: {len(surfactant_b_stock)}")
    
    # Set up plotting
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Surfactant A CMC Analysis (Left Plot) ---
    if len(surfactant_a_controls) > 0:
        # Sort by concentration for proper line connection
        surfactant_a_controls_sorted = surfactant_a_controls.sort_values('surf_A_conc_mm')
        
        ax1.semilogx(surfactant_a_controls_sorted['surf_A_conc_mm'], surfactant_a_controls_sorted['ratio'], 
                     'o-', color='red', linewidth=2, markersize=8, alpha=0.8, 
                     label=f'{surfactant_a_name} CMC Series (n={len(surfactant_a_controls)})')
        
        # Add individual point labels
        for _, row in surfactant_a_controls_sorted.iterrows():
            ax1.annotate(row['control_type'].replace(f'cmc_{surfactant_a_name}_', ''), 
                        (row['surf_A_conc_mm'], row['ratio']),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
    
    # Add reference points for Surfactant A plot
    if len(water_blank) > 0:
        water_ratio = water_blank['ratio'].mean()
        ax1.axhline(y=water_ratio, color='gray', linestyle='--', alpha=0.6,
                   label=f'Water baseline ({water_ratio:.3f})')
    
    if len(surfactant_a_stock) > 0:
        surfactant_a_stock_ratio = surfactant_a_stock['ratio'].mean()
        ax1.axhline(y=surfactant_a_stock_ratio, color='darkred', linestyle='-', alpha=0.7, linewidth=2,
                   label=f'{surfactant_a_name} Stock ({surfactant_a_stock_ratio:.3f})')
    
    # Fit CMC curve for Surfactant A if enough points
    cmc_surfactant_a, r2_surfactant_a = None, 0
    if len(surfactant_a_controls) >= 4:
        cmc_surfactant_a, r2_surfactant_a, popt_surfactant_a, x_data_surfactant_a = fit_cmc_curve(
            surfactant_a_controls_sorted['surf_A_conc_mm'].values, 
            surfactant_a_controls_sorted['ratio'].values
        )
        
        if cmc_surfactant_a is not None:
            # Plot fitted curve
            x_fit_log = np.linspace(x_data_surfactant_a.min(), x_data_surfactant_a.max(), 100)
            y_fit = boltzmann(x_fit_log, *popt_surfactant_a)
            x_fit_linear = 10**x_fit_log
            ax1.plot(x_fit_linear, y_fit, '--', color='orange', alpha=0.8, linewidth=2,
                    label=f'Boltzmann Fit (R²={r2_surfactant_a:.3f})')
            
            # Add CMC vertical line
            ax1.axvline(x=cmc_surfactant_a, color='green', linestyle=':', linewidth=2, alpha=0.8,
                       label=f'CMC = {cmc_surfactant_a:.2f} mM')
    
    ax1.set_xlabel(f'{surfactant_a_name} Concentration [mM] (log scale)', fontsize=12)
    ax1.set_ylabel('Fluorescence Ratio (F373/F384)', fontsize=12)
    ax1.set_title(f'{surfactant_a_name} CMC Analysis', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Set reasonable x-axis limits for Surfactant A
    if len(surfactant_a_controls) > 0:
        surfactant_a_conc_min = surfactant_a_controls['surf_A_conc_mm'].min() * 0.5
        surfactant_a_conc_max = surfactant_a_controls['surf_A_conc_mm'].max() * 2
        ax1.set_xlim(surfactant_a_conc_min, surfactant_a_conc_max)
    
    # --- Surfactant B CMC Analysis (Right Plot) ---
    if len(surfactant_b_controls) > 0:
        # Sort by concentration for proper line connection
        surfactant_b_controls_sorted = surfactant_b_controls.sort_values('surf_B_conc_mm')
        
        ax2.semilogx(surfactant_b_controls_sorted['surf_B_conc_mm'], surfactant_b_controls_sorted['ratio'], 
                     'o-', color='blue', linewidth=2, markersize=8, alpha=0.8,
                     label=f'{surfactant_b_name} CMC Series (n={len(surfactant_b_controls)})')
        
        # Add individual point labels
        for _, row in surfactant_b_controls_sorted.iterrows():
            ax2.annotate(row['control_type'].replace(f'cmc_{surfactant_b_name}_', ''), 
                        (row['surf_B_conc_mm'], row['ratio']),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
    
    # Add reference points for Surfactant B plot
    if len(water_blank) > 0:
        ax2.axhline(y=water_ratio, color='gray', linestyle='--', alpha=0.6,
                   label=f'Water baseline ({water_ratio:.3f})')
    
    if len(surfactant_b_stock) > 0:
        surfactant_b_stock_ratio = surfactant_b_stock['ratio'].mean()
        ax2.axhline(y=surfactant_b_stock_ratio, color='darkblue', linestyle='-', alpha=0.7, linewidth=2,
                   label=f'{surfactant_b_name} Stock ({surfactant_b_stock_ratio:.3f})')
    
    # Fit CMC curve for Surfactant B if enough points
    cmc_surfactant_b, r2_surfactant_b = None, 0
    if len(surfactant_b_controls) >= 4:
        cmc_surfactant_b, r2_surfactant_b, popt_surfactant_b, x_data_surfactant_b = fit_cmc_curve(
            surfactant_b_controls_sorted['surf_B_conc_mm'].values, 
            surfactant_b_controls_sorted['ratio'].values
        )
        
        if cmc_surfactant_b is not None:
            # Plot fitted curve
            x_fit_log = np.linspace(x_data_surfactant_b.min(), x_data_surfactant_b.max(), 100)
            y_fit = boltzmann(x_fit_log, *popt_surfactant_b)
            x_fit_linear = 10**x_fit_log
            ax2.plot(x_fit_linear, y_fit, '--', color='orange', alpha=0.8, linewidth=2,
                    label=f'Boltzmann Fit (R²={r2_surfactant_b:.3f})')
            
            # Add CMC vertical line
            ax2.axvline(x=cmc_surfactant_b, color='green', linestyle=':', linewidth=2, alpha=0.8,
                       label=f'CMC = {cmc_surfactant_b:.2f} mM')
    
    ax2.set_xlabel(f'{surfactant_b_name} Concentration [mM] (log scale)', fontsize=12)
    ax2.set_ylabel('Fluorescence Ratio (F373/F384)', fontsize=12)
    ax2.set_title(f'{surfactant_b_name} CMC Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Set reasonable x-axis limits for Surfactant B
    if len(surfactant_b_controls) > 0:
        surfactant_b_conc_min = surfactant_b_controls['surf_B_conc_mm'].min() * 0.5
        surfactant_b_conc_max = surfactant_b_controls['surf_B_conc_mm'].max() * 2
        ax2.set_xlim(surfactant_b_conc_min, surfactant_b_conc_max)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    try:
        output_path = csv_file_path.replace('iterative_experiment_results.csv', 'cmc_control_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"CMC control analysis saved to: {output_path}")
    except Exception as e:
        backup_path = "cmc_control_analysis.png"
        plt.savefig(backup_path, dpi=300, bbox_inches='tight')
        print(f"Saved to backup location: {backup_path}")
    
    # Close the figure to prevent display and free memory
    plt.close(fig)
    
    # Print data summary with CMC estimates
    print("\n--- CMC Control Data Summary ---")
    if len(surfactant_a_controls) > 0:
        print(f"{surfactant_a_name} concentrations: {surfactant_a_controls['surf_A_conc_mm'].min():.4f} - {surfactant_a_controls['surf_A_conc_mm'].max():.2f} mM")
        print(f"{surfactant_a_name} ratios: {surfactant_a_controls['ratio'].min():.4f} - {surfactant_a_controls['ratio'].max():.4f}")
        if cmc_surfactant_a is not None:
            print(f"{surfactant_a_name} CMC estimate: {cmc_surfactant_a:.2f} mM (R²={r2_surfactant_a:.3f})")
    
    if len(surfactant_b_controls) > 0:
        print(f"{surfactant_b_name} concentrations: {surfactant_b_controls['surf_B_conc_mm'].min():.4f} - {surfactant_b_controls['surf_B_conc_mm'].max():.2f} mM")
        print(f"{surfactant_b_name} ratios: {surfactant_b_controls['ratio'].min():.4f} - {surfactant_b_controls['ratio'].max():.4f}")
        if cmc_surfactant_b is not None:
            print(f"{surfactant_b_name} CMC estimate: {cmc_surfactant_b:.2f} mM (R²={r2_surfactant_b:.3f})")
    
    if len(water_blank) > 0:
        print(f"Water baseline ratio: {water_blank['ratio'].mean():.4f} ± {water_blank['ratio'].std():.4f}")
    
    print("CMC control analysis completed successfully!")
    
    return fig, ax1, ax2

if __name__ == "__main__":
    # This program is now called from workflows, not run directly
    # For testing, you can uncomment and modify the path below:
    # csv_path = r"path/to/your/iterative_experiment_results.csv"
    # fig, ax1, ax2 = analyze_cmc_controls(csv_path, "SDS", "DTAB") 
    print("CMC analysis program - import this module to use analyze_cmc_controls()")
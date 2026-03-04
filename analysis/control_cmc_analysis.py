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
    
    # Extract SDS CMC controls (cmc_SDS_1 through cmc_SDS_8)
    sds_controls = control_data[control_data['control_type'].str.startswith('cmc_SDS_')].copy()
    
    # Extract TTAB CMC controls (cmc_TTAB_1 through cmc_TTAB_8)  
    ttab_controls = control_data[control_data['control_type'].str.startswith('cmc_TTAB_')].copy()
    
    # Get reference controls
    water_blank = control_data[control_data['control_type'] == 'water_blank']
    sds_stock = control_data[control_data['control_type'] == 'surfactant_A_stock']
    ttab_stock = control_data[control_data['control_type'] == 'surfactant_B_stock']
    
    print(f"SDS CMC points: {len(sds_controls)}")
    print(f"TTAB CMC points: {len(ttab_controls)}")
    print(f"Water blanks: {len(water_blank)}")
    print(f"SDS stock: {len(sds_stock)}")
    print(f"TTAB stock: {len(ttab_stock)}")
    
    # Set up plotting
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- SDS CMC Analysis (Left Plot) ---
    if len(sds_controls) > 0:
        # Sort by concentration for proper line connection
        sds_controls_sorted = sds_controls.sort_values('surf_A_conc_mm')
        
        ax1.semilogx(sds_controls_sorted['surf_A_conc_mm'], sds_controls_sorted['ratio'], 
                     'o-', color='red', linewidth=2, markersize=8, alpha=0.8, 
                     label=f'SDS CMC Series (n={len(sds_controls)})')
        
        # Add individual point labels
        for _, row in sds_controls_sorted.iterrows():
            ax1.annotate(row['control_type'].replace('cmc_SDS_', ''), 
                        (row['surf_A_conc_mm'], row['ratio']),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
    
    # Add reference points for SDS plot
    if len(water_blank) > 0:
        water_ratio = water_blank['ratio'].mean()
        ax1.axhline(y=water_ratio, color='gray', linestyle='--', alpha=0.6,
                   label=f'Water baseline ({water_ratio:.3f})')
    
    if len(sds_stock) > 0:
        sds_stock_ratio = sds_stock['ratio'].mean()
        ax1.axhline(y=sds_stock_ratio, color='darkred', linestyle='-', alpha=0.7, linewidth=2,
                   label=f'SDS Stock ({sds_stock_ratio:.3f})')
    
    # Fit CMC curve for SDS if enough points
    cmc_sds, r2_sds = None, 0
    if len(sds_controls) >= 4:
        cmc_sds, r2_sds, popt_sds, x_data_sds = fit_cmc_curve(
            sds_controls_sorted['surf_A_conc_mm'].values, 
            sds_controls_sorted['ratio'].values
        )
        
        if cmc_sds is not None:
            # Plot fitted curve
            x_fit_log = np.linspace(x_data_sds.min(), x_data_sds.max(), 100)
            y_fit = boltzmann(x_fit_log, *popt_sds)
            x_fit_linear = 10**x_fit_log
            ax1.plot(x_fit_linear, y_fit, '--', color='orange', alpha=0.8, linewidth=2,
                    label=f'Boltzmann Fit (R²={r2_sds:.3f})')
            
            # Add CMC vertical line
            ax1.axvline(x=cmc_sds, color='green', linestyle=':', linewidth=2, alpha=0.8,
                       label=f'CMC = {cmc_sds:.2f} mM')
    
    ax1.set_xlabel(f'{surfactant_a_name} Concentration [mM] (log scale)', fontsize=12)
    ax1.set_ylabel('Fluorescence Ratio (F373/F384)', fontsize=12)
    ax1.set_title(f'{surfactant_a_name} CMC Analysis', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Set reasonable x-axis limits for SDS
    if len(sds_controls) > 0:
        sds_conc_min = sds_controls['surf_A_conc_mm'].min() * 0.5
        sds_conc_max = sds_controls['surf_A_conc_mm'].max() * 2
        ax1.set_xlim(sds_conc_min, sds_conc_max)
    
    # --- TTAB CMC Analysis (Right Plot) ---
    if len(ttab_controls) > 0:
        # Sort by concentration for proper line connection
        ttab_controls_sorted = ttab_controls.sort_values('surf_B_conc_mm')
        
        ax2.semilogx(ttab_controls_sorted['surf_B_conc_mm'], ttab_controls_sorted['ratio'], 
                     'o-', color='blue', linewidth=2, markersize=8, alpha=0.8,
                     label=f'TTAB CMC Series (n={len(ttab_controls)})')
        
        # Add individual point labels
        for _, row in ttab_controls_sorted.iterrows():
            ax2.annotate(row['control_type'].replace('cmc_TTAB_', ''), 
                        (row['surf_B_conc_mm'], row['ratio']),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
    
    # Add reference points for TTAB plot
    if len(water_blank) > 0:
        ax2.axhline(y=water_ratio, color='gray', linestyle='--', alpha=0.6,
                   label=f'Water baseline ({water_ratio:.3f})')
    
    if len(ttab_stock) > 0:
        ttab_stock_ratio = ttab_stock['ratio'].mean()
        ax2.axhline(y=ttab_stock_ratio, color='darkblue', linestyle='-', alpha=0.7, linewidth=2,
                   label=f'TTAB Stock ({ttab_stock_ratio:.3f})')
    
    # Fit CMC curve for TTAB if enough points
    cmc_ttab, r2_ttab = None, 0
    if len(ttab_controls) >= 4:
        cmc_ttab, r2_ttab, popt_ttab, x_data_ttab = fit_cmc_curve(
            ttab_controls_sorted['surf_B_conc_mm'].values, 
            ttab_controls_sorted['ratio'].values
        )
        
        if cmc_ttab is not None:
            # Plot fitted curve
            x_fit_log = np.linspace(x_data_ttab.min(), x_data_ttab.max(), 100)
            y_fit = boltzmann(x_fit_log, *popt_ttab)
            x_fit_linear = 10**x_fit_log
            ax2.plot(x_fit_linear, y_fit, '--', color='orange', alpha=0.8, linewidth=2,
                    label=f'Boltzmann Fit (R²={r2_ttab:.3f})')
            
            # Add CMC vertical line
            ax2.axvline(x=cmc_ttab, color='green', linestyle=':', linewidth=2, alpha=0.8,
                       label=f'CMC = {cmc_ttab:.2f} mM')
    
    ax2.set_xlabel(f'{surfactant_b_name} Concentration [mM] (log scale)', fontsize=12)
    ax2.set_ylabel('Fluorescence Ratio (F373/F384)', fontsize=12)
    ax2.set_title(f'{surfactant_b_name} CMC Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Set reasonable x-axis limits for TTAB
    if len(ttab_controls) > 0:
        ttab_conc_min = ttab_controls['surf_B_conc_mm'].min() * 0.5
        ttab_conc_max = ttab_controls['surf_B_conc_mm'].max() * 2
        ax2.set_xlim(ttab_conc_min, ttab_conc_max)
    
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
    if len(sds_controls) > 0:
        print(f"SDS concentrations: {sds_controls['surf_A_conc_mm'].min():.4f} - {sds_controls['surf_A_conc_mm'].max():.2f} mM")
        print(f"SDS ratios: {sds_controls['ratio'].min():.4f} - {sds_controls['ratio'].max():.4f}")
        if cmc_sds is not None:
            print(f"SDS CMC estimate: {cmc_sds:.2f} mM (R²={r2_sds:.3f})")
    
    if len(ttab_controls) > 0:
        print(f"TTAB concentrations: {ttab_controls['surf_B_conc_mm'].min():.4f} - {ttab_controls['surf_B_conc_mm'].max():.2f} mM")
        print(f"TTAB ratios: {ttab_controls['ratio'].min():.4f} - {ttab_controls['ratio'].max():.4f}")
        if cmc_ttab is not None:
            print(f"TTAB CMC estimate: {cmc_ttab:.2f} mM (R²={r2_ttab:.3f})")
    
    if len(water_blank) > 0:
        print(f"Water baseline ratio: {water_blank['ratio'].mean():.4f} ± {water_blank['ratio'].std():.4f}")
    
    print("CMC control analysis completed successfully!")
    
    return fig, ax1, ax2

if __name__ == "__main__":
    # This program is now called from workflows, not run directly
    # For testing, you can uncomment and modify the path below:
    # csv_path = r"path/to/your/iterative_experiment_results.csv"
    # fig, ax1, ax2 = analyze_cmc_controls(csv_path, "SDS", "TTAB") 
    print("CMC analysis program - import this module to use analyze_cmc_controls()")
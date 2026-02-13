# -*- coding: utf-8 -*-
"""
Test Visualization for Adaptive/Irregular Concentration Data
Handles non-uniform, adaptively chosen concentration points instead of regular grids.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def visualize_adaptive_surfactant_data(csv_file_path, output_dir=None, show_plots=True):
    """
    Create scatter plots and interpolated heatmaps for irregular concentration data.
    
    Args:
        csv_file_path: Path to CSV with irregular concentration data
        output_dir: Directory to save plots (optional)
        show_plots: Whether to display plots interactively
    """
    print(f"Loading data from: {csv_file_path}")
    
    # Read and filter data
    df = pd.read_csv(csv_file_path)
    experiment_data = df[df['well_type'] == 'experiment'].copy()
    
    print(f"Found {len(experiment_data)} experimental points")
    print(f"SDS concentration range: {experiment_data['surf_A_conc_mm'].min():.6f} - {experiment_data['surf_A_conc_mm'].max():.6f} mM")
    print(f"TTAB concentration range: {experiment_data['surf_B_conc_mm'].min():.6f} - {experiment_data['surf_B_conc_mm'].max():.6f} mM")
    
    # Set up the plot style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Adaptive Surfactant Screening Results (Irregular Grid)', fontsize=16, fontweight='bold')
    
    # Get concentration data
    x = experiment_data['surf_A_conc_mm'].values
    y = experiment_data['surf_B_conc_mm'].values
    
    # Convert to log scale for better visualization
    log_x = np.log10(x)
    log_y = np.log10(y)
    
    # Plot 1: Turbidity Scatter
    ax1 = axes[0, 0]
    turbidity = experiment_data['turbidity_600'].values
    scatter1 = ax1.scatter(log_x, log_y, c=turbidity, cmap='viridis', s=50, alpha=0.8)
    ax1.set_xlabel('log10(SDS Concentration [mM])')
    ax1.set_ylabel('log10(TTAB Concentration [mM])')
    ax1.set_title('Turbidity 600nm')
    plt.colorbar(scatter1, ax=ax1, label='Turbidity')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ratio Scatter
    ax2 = axes[0, 1]
    ratio = experiment_data['ratio'].values
    scatter2 = ax2.scatter(log_x, log_y, c=ratio, cmap='RdYlBu', s=50, alpha=0.8)
    ax2.set_xlabel('log10(SDS Concentration [mM])')
    ax2.set_ylabel('log10(TTAB Concentration [mM])')
    ax2.set_title('F373/F384 Ratio')
    plt.colorbar(scatter2, ax=ax2, label='Ratio')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Interpolated Turbidity Heatmap
    ax3 = axes[1, 0]
    try:
        from scipy.interpolate import griddata
        
        # Create regular grid for interpolation
        xi = np.linspace(log_x.min(), log_x.max(), 50)
        yi = np.linspace(log_y.min(), log_y.max(), 50)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate turbidity data
        zi_turbidity = griddata((log_x, log_y), turbidity, (xi_grid, yi_grid), method='cubic')
        
        # Create heatmap
        heatmap1 = ax3.imshow(zi_turbidity, extent=[log_x.min(), log_x.max(), log_y.min(), log_y.max()], 
                             origin='lower', cmap='viridis', aspect='auto', alpha=0.8)
        
        # Overlay the actual data points
        ax3.scatter(log_x, log_y, c='white', s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        ax3.set_xlabel('log10(SDS Concentration [mM])')
        ax3.set_ylabel('log10(TTAB Concentration [mM])')
        ax3.set_title('Turbidity 600nm (Interpolated)')
        plt.colorbar(heatmap1, ax=ax3, label='Turbidity')
        
    except ImportError:
        # Fallback if scipy not available
        ax3.scatter(log_x, log_y, c=turbidity, cmap='viridis', s=50, alpha=0.8)
        ax3.set_xlabel('log10(SDS Concentration [mM])')
        ax3.set_ylabel('log10(TTAB Concentration [mM])')
        ax3.set_title('Turbidity 600nm (No Interpolation)')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Interpolated Ratio Heatmap
    ax4 = axes[1, 1]
    try:
        from scipy.interpolate import griddata
        
        # Interpolate ratio data
        zi_ratio = griddata((log_x, log_y), ratio, (xi_grid, yi_grid), method='cubic')
        
        # Create heatmap
        heatmap2 = ax4.imshow(zi_ratio, extent=[log_x.min(), log_x.max(), log_y.min(), log_y.max()], 
                             origin='lower', cmap='RdYlBu', aspect='auto', alpha=0.8)
        
        # Overlay the actual data points
        ax4.scatter(log_x, log_y, c='white', s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        ax4.set_xlabel('log10(SDS Concentration [mM])')
        ax4.set_ylabel('log10(TTAB Concentration [mM])')
        ax4.set_title('F373/F384 Ratio (Interpolated)')
        plt.colorbar(heatmap2, ax=ax4, label='Ratio')
        
    except ImportError:
        # Fallback if scipy not available
        ax4.scatter(log_x, log_y, c=ratio, cmap='RdYlBu', s=50, alpha=0.8)
        ax4.set_xlabel('log10(SDS Concentration [mM])')
        ax4.set_ylabel('log10(TTAB Concentration [mM])')
        ax4.set_title('F373/F384 Ratio (No Interpolation)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'adaptive_surfactant_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    if show_plots:
        plt.show()
    
    return fig

def analyze_concentration_distribution(csv_file_path):
    """Analyze how the adaptive algorithm distributed concentration points."""
    df = pd.read_csv(csv_file_path)
    experiment_data = df[df['well_type'] == 'experiment'].copy()
    
    # Concentration distribution analysis
    print("\n=== ADAPTIVE CONCENTRATION ANALYSIS ===")
    print(f"Total experimental points: {len(experiment_data)}")
    
    # SDS analysis
    sds_concs = experiment_data['surf_A_conc_mm'].values
    unique_sds = sorted(set(sds_concs))
    print(f"\nSDS concentrations used ({len(unique_sds)} unique):")
    for i, conc in enumerate(unique_sds[:10]):  # Show first 10
        count = sum(sds_concs == conc)
        print(f"  {conc:.6f} mM (used {count} times)")
    if len(unique_sds) > 10:
        print(f"  ... and {len(unique_sds) - 10} more")
    
    # TTAB analysis  
    ttab_concs = experiment_data['surf_B_conc_mm'].values
    unique_ttab = sorted(set(ttab_concs))
    print(f"\nTTAB concentrations used ({len(unique_ttab)} unique):")
    for i, conc in enumerate(unique_ttab[:10]):  # Show first 10
        count = sum(ttab_concs == conc)
        print(f"  {conc:.6f} mM (used {count} times)")
    if len(unique_ttab) > 10:
        print(f"  ... and {len(unique_ttab) - 10} more")
    
    # Grid coverage analysis
    total_cells = len(unique_sds) * len(unique_ttab)
    coverage = len(experiment_data) / total_cells * 100
    print(f"\nGrid coverage: {len(experiment_data)}/{total_cells} = {coverage:.1f}%")
    print("(This shows how sparse/dense the adaptive sampling was)")
    
    return unique_sds, unique_ttab

if __name__ == "__main__":
    # Test with your latest data
    csv_file = "output/surfactant_grid_SDS_TTAB_20260212_165101/iterative_experiment_results.csv"
    output_folder = "test_analysis"
    
    if os.path.exists(csv_file):
        print("Testing adaptive data visualization...")
        
        # Analyze concentration distribution
        analyze_concentration_distribution(csv_file)
        
        # Create visualizations
        visualize_adaptive_surfactant_data(csv_file, output_folder, show_plots=True)
        
    else:
        print(f"CSV file not found: {csv_file}")
        print("Please update the file path or run your adaptive workflow first.")
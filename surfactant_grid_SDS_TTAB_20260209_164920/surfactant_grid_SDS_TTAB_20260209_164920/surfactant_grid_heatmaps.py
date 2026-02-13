"""
Surfactant Grid Heatmap Generator

This script creates 2D heatmaps for surfactant concentration grid experiments.
Generates separate plots for turbidity and fluorescence ratio measurements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def create_grid_heatmaps(csv_file_path):
    """
    Create heatmaps for turbidity and ratio from surfactant grid experiment data.
    
    Parameters:
    csv_file_path (str): Path to the CSV file containing experiment results
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Filter for experiment data only (exclude controls)
    experiment_data = df[df['well_type'] == 'experiment'].copy()
    
    # Get control data
    control_data = df[df['well_type'] == 'control'].copy()
    
    print(f"Processing {len(experiment_data)} experiment data points and {len(control_data)} control points...")
    
    # Process control data - handle NaN values
    control_processed = []
    for _, row in control_data.iterrows():
        control_info = {
            'control_type': row['control_type'],
            'surf_A_conc_mm': row['surf_A_conc_mm'] if not pd.isna(row['surf_A_conc_mm']) else 0,
            'surf_B_conc_mm': row['surf_B_conc_mm'] if not pd.isna(row['surf_B_conc_mm']) else 0,
            'turbidity_600': row['turbidity_600'],
            'ratio': row['ratio']
        }
        control_processed.append(control_info)
    
    control_df = pd.DataFrame(control_processed)
    print("Controls found:")
    for _, row in control_df.iterrows():
        print(f"  {row['control_type']}: SDS={row['surf_A_conc_mm']:.1f}, TTAB={row['surf_B_conc_mm']:.1f}, Turbidity={row['turbidity_600']:.4f}, Ratio={row['ratio']:.4f}")
    
    # Get unique concentration values and sort them
    surf_A_concs = sorted(experiment_data['surf_A_conc_mm'].unique())
    surf_B_concs = sorted(experiment_data['surf_B_conc_mm'].unique())
    
    print(f"Concentration grid: {len(surf_A_concs)} x {len(surf_B_concs)}")
    print(f"surf_A_conc_mm range: {min(surf_A_concs):.6f} to {max(surf_A_concs):.1f}")
    print(f"surf_B_conc_mm range: {min(surf_B_concs):.6f} to {max(surf_B_concs):.1f}")
    
    # Create pivot tables for heatmap data
    # Note: surf_A on x-axis, surf_B on y-axis, with high concentrations top-right
    turbidity_grid = experiment_data.pivot_table(
        index='surf_B_conc_mm', 
        columns='surf_A_conc_mm', 
        values='turbidity_600',
        aggfunc='mean'  # Use mean in case of multiple measurements
    )
    
    ratio_grid = experiment_data.pivot_table(
        index='surf_B_conc_mm', 
        columns='surf_A_conc_mm', 
        values='ratio',
        aggfunc='mean'
    )
    
    # Sort indices to ensure proper orientation (high concentrations top-right)
    turbidity_grid = turbidity_grid.reindex(sorted(turbidity_grid.index, reverse=True))
    turbidity_grid = turbidity_grid.reindex(sorted(turbidity_grid.columns), axis=1)
    
    ratio_grid = ratio_grid.reindex(sorted(ratio_grid.index, reverse=True))
    ratio_grid = ratio_grid.reindex(sorted(ratio_grid.columns), axis=1)
    
    # Create control grids (3 controls arranged horizontally)
    control_labels = ['Water\n(0,0)', 'SDS Only\n(50,0)', 'TTAB Only\n(0,50)']
    
    # Create 1x3 control grids
    turbidity_controls = np.array([[
        control_df[control_df['control_type'] == 'water_blank']['turbidity_600'].iloc[0],
        control_df[control_df['control_type'] == 'surfactant_A_stock']['turbidity_600'].iloc[0], 
        control_df[control_df['control_type'] == 'surfactant_B_stock']['turbidity_600'].iloc[0]
    ]])
    
    ratio_controls = np.array([[
        control_df[control_df['control_type'] == 'water_blank']['ratio'].iloc[0],
        control_df[control_df['control_type'] == 'surfactant_A_stock']['ratio'].iloc[0], 
        control_df[control_df['control_type'] == 'surfactant_B_stock']['ratio'].iloc[0]
    ]])
    
    # Determine color scale ranges to include both experiment and control data
    all_turbidity = list(experiment_data['turbidity_600']) + list(control_df['turbidity_600'])
    all_ratio = list(experiment_data['ratio']) + list(control_df['ratio'])
    
    turbidity_vmin, turbidity_vmax = min(all_turbidity), max(all_turbidity)
    ratio_vmin, ratio_vmax = min(all_ratio), max(all_ratio)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Create figure with subplots - main grids plus control rows
    fig = plt.figure(figsize=(20, 14))
    
    # Create grid layout: 2 columns x 2 rows, with main plots much larger than controls
    gs = fig.add_gridspec(2, 2, height_ratios=[15, 1], hspace=0.15, wspace=0.25)
    
    # Main turbidity plot
    ax1_main = fig.add_subplot(gs[0, 0])
    turbidity_plot = sns.heatmap(
        turbidity_grid,
        ax=ax1_main,
        cmap='viridis',
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Turbidity (600 nm)'},
        xticklabels=[f'{x:.6f}' if x < 0.001 else f'{x:.4f}' if x < 1 else f'{x:.1f}' for x in turbidity_grid.columns],
        yticklabels=[f'{y:.6f}' if y < 0.001 else f'{y:.4f}' if y < 1 else f'{y:.1f}' for y in turbidity_grid.index],
        square=True,
        vmin=turbidity_vmin,
        vmax=turbidity_vmax
    )
    
    ax1_main.set_title('Turbidity vs Surfactant Concentrations\n(SDS vs TTAB)', fontsize=16, fontweight='bold')
    ax1_main.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax1_main.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax1_main.tick_params(axis='x', rotation=45)
    ax1_main.tick_params(axis='y', rotation=0)
    
    # Main ratio plot  
    ax2_main = fig.add_subplot(gs[0, 1])
    ratio_plot = sns.heatmap(
        ratio_grid,
        ax=ax2_main,
        cmap='plasma',
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Fluorescence Ratio (334/373 : 334/384)'},
        xticklabels=[f'{x:.6f}' if x < 0.001 else f'{x:.4f}' if x < 1 else f'{x:.1f}' for x in ratio_grid.columns],
        yticklabels=[f'{y:.6f}' if y < 0.001 else f'{y:.4f}' if y < 1 else f'{y:.1f}' for y in ratio_grid.index],
        square=True,
        vmin=ratio_vmin,
        vmax=ratio_vmax
    )
    
    ax2_main.set_title('Fluorescence Ratio vs Surfactant Concentrations\n(SDS vs TTAB)', fontsize=16, fontweight='bold')
    ax2_main.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax2_main.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax2_main.tick_params(axis='x', rotation=45)
    ax2_main.tick_params(axis='y', rotation=0)
    
    # Turbidity controls (smaller bar below)
    ax1_ctrl = fig.add_subplot(gs[1, 0])
    sns.heatmap(
        turbidity_controls,
        ax=ax1_ctrl,
        cmap='viridis',
        annot=True,
        fmt='.3f',
        cbar=False,
        xticklabels=control_labels,
        yticklabels=['Controls'],
        vmin=turbidity_vmin,
        vmax=turbidity_vmax
    )
    ax1_ctrl.set_title('Control Samples', fontsize=11)
    ax1_ctrl.tick_params(axis='x', rotation=0, labelsize=10)
    ax1_ctrl.tick_params(axis='y', rotation=0, labelsize=10)
    
    # Ratio controls (smaller bar below)
    ax2_ctrl = fig.add_subplot(gs[1, 1])
    sns.heatmap(
        ratio_controls,
        ax=ax2_ctrl,
        cmap='plasma',
        annot=True,
        fmt='.3f',
        cbar=False,
        xticklabels=control_labels,
        yticklabels=['Controls'],
        vmin=ratio_vmin,
        vmax=ratio_vmax
    )
    ax2_ctrl.set_title('Control Samples', fontsize=11)
    ax2_ctrl.tick_params(axis='x', rotation=0, labelsize=10)
    ax2_ctrl.tick_params(axis='y', rotation=0, labelsize=10)
    
    # Save the plot in the same directory as the CSV file
    output_dir = Path(csv_file_path).parent
    plot_filename = output_dir / 'surfactant_grid_heatmaps_with_controls.png'
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nHeatmaps with controls saved to: {plot_filename}")
    
    # Also save as separate plots for individual use
    
    # Individual turbidity plot with controls
    fig1, (ax1_solo, ax1_ctrl_solo) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [15, 1]}, sharex=False)
    plt.subplots_adjust(hspace=0.15)
    
    sns.heatmap(
        turbidity_grid,
        ax=ax1_solo,
        cmap='viridis',
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Turbidity (600 nm)'},
        xticklabels=[f'{x:.6f}' if x < 0.001 else f'{x:.4f}' if x < 1 else f'{x:.1f}' for x in turbidity_grid.columns],
        yticklabels=[f'{y:.6f}' if y < 0.001 else f'{y:.4f}' if y < 1 else f'{y:.1f}' for y in turbidity_grid.index],
        square=True,
        vmin=turbidity_vmin,
        vmax=turbidity_vmax
    )
    ax1_solo.set_title('Turbidity vs Surfactant Concentrations (SDS vs TTAB)', fontsize=18, fontweight='bold')
    ax1_solo.set_xlabel('SDS Concentration (mM)', fontsize=14)
    ax1_solo.set_ylabel('TTAB Concentration (mM)', fontsize=14)
    ax1_solo.tick_params(axis='x', rotation=45)
    ax1_solo.tick_params(axis='y', rotation=0)
    
    sns.heatmap(
        turbidity_controls,
        ax=ax1_ctrl_solo,
        cmap='viridis',
        annot=True,
        fmt='.3f',
        cbar=False,
        xticklabels=control_labels,
        yticklabels=['Controls'],
        vmin=turbidity_vmin,
        vmax=turbidity_vmax
    )
    ax1_ctrl_solo.set_title('Control Samples', fontsize=12)
    ax1_ctrl_solo.tick_params(axis='x', rotation=0, labelsize=10)
    ax1_ctrl_solo.tick_params(axis='y', rotation=0, labelsize=10)
    
    turbidity_filename = output_dir / 'surfactant_grid_turbidity_with_controls.png'
    plt.savefig(turbidity_filename, dpi=300, bbox_inches='tight')
    print(f"Turbidity heatmap with controls saved to: {turbidity_filename}")
    plt.close()
    
    # Individual ratio plot with controls
    fig2, (ax2_solo, ax2_ctrl_solo) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [15, 1]}, sharex=False)
    plt.subplots_adjust(hspace=0.15)
    
    sns.heatmap(
        ratio_grid,
        ax=ax2_solo,
        cmap='plasma',
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Fluorescence Ratio (334/373 : 334/384)'},
        xticklabels=[f'{x:.6f}' if x < 0.001 else f'{x:.4f}' if x < 1 else f'{x:.1f}' for x in ratio_grid.columns],
        yticklabels=[f'{y:.6f}' if y < 0.001 else f'{y:.4f}' if y < 1 else f'{y:.1f}' for y in ratio_grid.index],
        square=True,
        vmin=ratio_vmin,
        vmax=ratio_vmax
    )
    ax2_solo.set_title('Fluorescence Ratio vs Surfactant Concentrations (SDS vs TTAB)', fontsize=18, fontweight='bold')
    ax2_solo.set_xlabel('SDS Concentration (mM)', fontsize=14)
    ax2_solo.set_ylabel('TTAB Concentration (mM)', fontsize=14)
    ax2_solo.tick_params(axis='x', rotation=45)
    ax2_solo.tick_params(axis='y', rotation=0)
    
    sns.heatmap(
        ratio_controls,
        ax=ax2_ctrl_solo,
        cmap='plasma',
        annot=True,
        fmt='.3f',
        cbar=False,
        xticklabels=control_labels,
        yticklabels=['Controls'],
        vmin=ratio_vmin,
        vmax=ratio_vmax
    )
    ax2_ctrl_solo.set_title('Control Samples', fontsize=12)
    ax2_ctrl_solo.tick_params(axis='x', rotation=0, labelsize=10)
    ax2_ctrl_solo.tick_params(axis='y', rotation=0, labelsize=10)
    
    ratio_filename = output_dir / 'surfactant_grid_ratio_with_controls.png'
    plt.savefig(ratio_filename, dpi=300, bbox_inches='tight')
    print(f"Ratio heatmap with controls saved to: {ratio_filename}")
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Experiment data:")
    print(f"  Turbidity range: {experiment_data['turbidity_600'].min():.4f} to {experiment_data['turbidity_600'].max():.4f}")
    print(f"  Ratio range: {experiment_data['ratio'].min():.4f} to {experiment_data['ratio'].max():.4f}")
    print(f"Control data:")
    print(f"  Turbidity range: {control_df['turbidity_600'].min():.4f} to {control_df['turbidity_600'].max():.4f}")
    print(f"  Ratio range: {control_df['ratio'].min():.4f} to {control_df['ratio'].max():.4f}")
    print(f"Combined color scale:")
    print(f"  Turbidity: {turbidity_vmin:.4f} to {turbidity_vmax:.4f}")
    print(f"  Ratio: {ratio_vmin:.4f} to {ratio_vmax:.4f}")
    
    # Show the combined plot
    plt.show()
    
    return turbidity_grid, ratio_grid

def main():
    """Main function to run the heatmap generation."""
    
    # Get the current script directory
    script_dir = Path(__file__).parent
    
    # Look for CSV file in the same directory
    csv_files = list(script_dir.glob("complete_experiment_results.csv"))
    
    if not csv_files:
        print("Error: No 'complete_experiment_results.csv' found in the current directory.")
        print(f"Current directory: {script_dir}")
        print("Available files:", list(script_dir.iterdir()))
        return
    
    csv_file_path = csv_files[0]
    print(f"Found CSV file: {csv_file_path}")
    
    # Create the heatmaps
    try:
        turbidity_grid, ratio_grid = create_grid_heatmaps(csv_file_path)
        print("\n✓ Heatmap generation completed successfully!")
        
        # Print grid dimensions for verification
        print(f"\nGrid verification:")
        print(f"Turbidity grid shape: {turbidity_grid.shape}")
        print(f"Ratio grid shape: {ratio_grid.shape}")
        
    except Exception as e:
        print(f"Error generating heatmaps: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
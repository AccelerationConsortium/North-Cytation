#!/usr/bin/env python3
"""
Visualization script for surfactant grid experiment
Creates 2D heatmaps of turbidity and fluorescence ratio vs concentrations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
import os

def load_data(csv_path):
    """Load the recovered experimental data"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} data points")
    print(f"Concentration range A: {df['conc_a_mm'].min():.6f} to {df['conc_a_mm'].max():.1f} mM")
    print(f"Concentration range B: {df['conc_b_mm'].min():.6f} to {df['conc_b_mm'].max():.1f} mM")
    print(f"Turbidity range: {df['turbidity_600'].min():.4f} to {df['turbidity_600'].max():.4f}")
    print(f"Ratio range: {df['ratio'].min():.3f} to {df['ratio'].max():.3f}")
    return df

def create_2d_grid(df, value_column):
    """Convert data to 2D grid for heatmap"""
    # Get unique concentrations (sorted)
    conc_a_unique = sorted(df['conc_a_mm'].unique())
    conc_b_unique = sorted(df['conc_b_mm'].unique())
    
    print(f"Grid size: {len(conc_a_unique)} x {len(conc_b_unique)}")
    
    # Create 2D array
    grid = np.full((len(conc_a_unique), len(conc_b_unique)), np.nan)
    
    # Fill grid with values
    for _, row in df.iterrows():
        i = conc_a_unique.index(row['conc_a_mm'])
        j = conc_b_unique.index(row['conc_b_mm'])
        grid[i, j] = row[value_column]
    
    return grid, conc_a_unique, conc_b_unique

def create_turbidity_heatmap(df, output_dir):
    """Create turbidity heatmap with non-linear scaling"""
    grid, conc_a, conc_b = create_2d_grid(df, 'turbidity_600')
    
    # Custom normalization for turbidity (exaggerate differences above 0.04)
    def turbidity_transform(x):
        """Transform turbidity values to exaggerate differences above 0.04"""
        x_clipped = np.clip(x, 0.04, None)  # Set minimum to 0.04
        # Use log scale above 0.04 to exaggerate differences
        return np.log10(x_clipped / 0.04 + 1)
    
    # Transform the data
    grid_transformed = turbidity_transform(grid)
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    ax = sns.heatmap(grid_transformed, 
                     xticklabels=[f'{c:.1e}' if c < 0.1 else f'{c:.1f}' for c in conc_b],
                     yticklabels=[f'{c:.1e}' if c < 0.1 else f'{c:.1f}' for c in conc_a],
                     cmap='viridis',
                     cbar_kws={'label': 'Turbidity (600nm) [transformed]'})
    
    plt.title('Turbidity Heatmap\n(Non-linear scale, values <0.04 treated as baseline)', fontsize=14)
    plt.xlabel('TTAB Concentration (mM)', fontsize=12)
    plt.ylabel('SDS Concentration (mM)', fontsize=12)
    
    # Rotate tick labels for readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'turbidity_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved turbidity heatmap: {output_path}")
    
    plt.show()

def create_ratio_heatmap(df, output_dir):
    """Create fluorescence ratio heatmap with linear scaling"""
    grid, conc_a, conc_b = create_2d_grid(df, 'ratio')
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap with linear scaling
    ax = sns.heatmap(grid, 
                     xticklabels=[f'{c:.1e}' if c < 0.1 else f'{c:.1f}' for c in conc_b],
                     yticklabels=[f'{c:.1e}' if c < 0.1 else f'{c:.1f}' for c in conc_a],
                     cmap='RdYlBu_r',
                     cbar_kws={'label': 'Fluorescence Ratio (334_373/334_384)'})
    
    plt.title('Fluorescence Ratio Heatmap\n(Linear scale - indicates micelle formation)', fontsize=14)
    plt.xlabel('TTAB Concentration (mM)', fontsize=12)
    plt.ylabel('SDS Concentration (mM)', fontsize=12)
    
    # Rotate tick labels for readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'ratio_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved ratio heatmap: {output_path}")
    
    plt.show()

def create_combined_figure(df, output_dir):
    """Create a combined figure with both heatmaps"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Turbidity heatmap
    grid_turb, conc_a, conc_b = create_2d_grid(df, 'turbidity_600')
    
    # Transform turbidity data
    def turbidity_transform(x):
        x_clipped = np.clip(x, 0.04, None)
        return np.log10(x_clipped / 0.04 + 1)
    
    grid_turb_transformed = turbidity_transform(grid_turb)
    
    sns.heatmap(grid_turb_transformed, 
                xticklabels=[f'{c:.1e}' if c < 0.1 else f'{c:.1f}' for c in conc_b],
                yticklabels=[f'{c:.1e}' if c < 0.1 else f'{c:.1f}' for c in conc_a],
                cmap='viridis',
                cbar_kws={'label': 'Turbidity [transformed]'},
                ax=ax1)
    
    ax1.set_title('Turbidity (600nm)\nNon-linear scale', fontsize=14)
    ax1.set_xlabel('TTAB Concentration (mM)', fontsize=12)
    ax1.set_ylabel('SDS Concentration (mM)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Ratio heatmap
    grid_ratio, _, _ = create_2d_grid(df, 'ratio')
    
    sns.heatmap(grid_ratio, 
                xticklabels=[f'{c:.1e}' if c < 0.1 else f'{c:.1f}' for c in conc_b],
                yticklabels=[f'{c:.1e}' if c < 0.1 else f'{c:.1f}' for c in conc_a],
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Fluorescence Ratio'},
                ax=ax2)
    
    ax2.set_title('Fluorescence Ratio (334_373/334_384)\nLinear scale', fontsize=14)
    ax2.set_xlabel('TTAB Concentration (mM)', fontsize=12)
    ax2.set_ylabel('SDS Concentration (mM)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save combined plot
    output_path = os.path.join(output_dir, 'combined_heatmaps.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined heatmaps: {output_path}")
    
    plt.show()

def main():
    """Main visualization function"""
    # Paths
    experiment_dir = r"c:\Users\Imaging Controller\Desktop\utoronto_demo\output\surfactant_grid_SDS_TTAB_20260203_200428"
    csv_path = os.path.join(experiment_dir, "consolidated_measurements_RECOVERED.csv")
    output_dir = experiment_dir
    
    print("=== Surfactant Grid Visualization ===")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"ERROR: Data file not found: {csv_path}")
        return
    
    # Load data
    df = load_data(csv_path)
    
    # Create visualizations
    print("\n1. Creating turbidity heatmap...")
    create_turbidity_heatmap(df, output_dir)
    
    print("\n2. Creating ratio heatmap...")
    create_ratio_heatmap(df, output_dir)
    
    print("\n3. Creating combined figure...")
    create_combined_figure(df, output_dir)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
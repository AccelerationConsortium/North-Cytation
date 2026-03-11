"""
Tweakable Contour Map Generator with Adjustable Detail Levels
Modify parameters below to customize your contour plots
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm

# === TWEAKABLE PARAMETERS ===
CSV_FILE = r"C:\Users\Owen\Documents\GitHub\North-Cytation\New folder\iterative_experiment_results.csv"

# Surfactant names for plot titles
SURFACTANT_A_NAME = "SDS"
SURFACTANT_B_NAME = "DTAB"

# Contour level settings - ADJUST THESE TO REDUCE/INCREASE DETAIL
TURBIDITY_LEVELS = 5        # Fewer levels = less false detail (try 4-6)
RATIO_LEVELS = 6           # Fewer levels = cleaner gradients (try 4-7)
FLUORESCENCE_LEVELS = 5    # Fewer levels = smoother appearance (try 4-6)

# Grid resolution (higher = smoother but slower)
GRID_RESOLUTION = 80       # Try 50-100 (80 is good balance)

# Output filename
OUTPUT_NAME = "adjustable_contour_maps.png"

def create_adjustable_contour_maps(csv_file_path, surfactant_a_name="SurfA", surfactant_b_name="SurfB", 
                                  turb_levels=5, ratio_levels=6, fluor_levels=5, grid_res=80):
    """
    Create contour maps with adjustable detail levels to match data density
    """
    # Read and filter data
    df = pd.read_csv(csv_file_path)
    exp_data = df[df['well_type'] == 'experiment'].copy()
    
    # Control data for baselines
    control_data = df[df['well_type'] == 'control'].copy()
    water_control = control_data[control_data['control_type'] == 'water_blank']
    
    if len(water_control) > 0:
        water_baseline_ratio = water_control['ratio'].max()
        water_baseline_turbidity = water_control['turbidity_600'].iloc[0]
        print(f"Water control baseline: Ratio: {water_baseline_ratio:.4f}, Turbidity: {water_baseline_turbidity:.4f}")
    else:
        water_baseline_ratio = 0.83
        water_baseline_turbidity = 0.04
        print(f"Using fallback baseline - Ratio: {water_baseline_ratio:.4f}")

    print(f"Data points: {len(exp_data)}")
    print(f"Contour levels: Turbidity={turb_levels}, Ratio={ratio_levels}, Fluorescence={fluor_levels}")
    
    # Extract concentration and measurement data
    x = np.log10(exp_data['surf_A_conc_mm'])
    y = np.log10(exp_data['surf_B_conc_mm'])
    turbidity = exp_data['turbidity_600']
    ratio = exp_data['ratio']
    avg_fluorescence = (exp_data['fluorescence_334_373'] + exp_data['fluorescence_334_384']) / 2
    
    # Create interpolation grid
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Improved interpolation to reduce white patches
    print("Interpolating data (reducing white patches)...")
    
    # Use linear interpolation first, then fill gaps with nearest neighbor
    turbidity_grid_linear = griddata((x, y), turbidity, (Xi, Yi), method='linear')
    turbidity_grid_nearest = griddata((x, y), turbidity, (Xi, Yi), method='nearest')
    turbidity_grid = np.where(np.isnan(turbidity_grid_linear), turbidity_grid_nearest, turbidity_grid_linear)
    
    ratio_grid_linear = griddata((x, y), ratio, (Xi, Yi), method='linear')
    ratio_grid_nearest = griddata((x, y), ratio, (Xi, Yi), method='nearest')
    ratio_grid = np.where(np.isnan(ratio_grid_linear), ratio_grid_nearest, ratio_grid_linear)
    
    fluor_grid_linear = griddata((x, y), avg_fluorescence, (Xi, Yi), method='linear')
    fluor_grid_nearest = griddata((x, y), avg_fluorescence, (Xi, Yi), method='nearest')
    fluor_grid = np.where(np.isnan(fluor_grid_linear), fluor_grid_nearest, fluor_grid_linear)
    
    # Create the plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- TURBIDITY PLOT (with adjustable levels) ---
    ax1.set_title(f'Turbidity (600nm) vs Surfactant Concentrations\\n({surfactant_a_name} vs {surfactant_b_name})', 
                 fontsize=14, fontweight='bold')
    
    turb_min, turb_max = exp_data['turbidity_600'].min(), exp_data['turbidity_600'].max()
    turb_min_adj = max(turb_min, 0.01)  # Avoid log(0)
    
    if turb_max > turb_min_adj:
        # Create log-spaced levels with user-specified number
        turb_level_values = np.logspace(np.log10(turb_min_adj), np.log10(turb_max), turb_levels)
        
        cs1 = ax1.contourf(Xi, Yi, turbidity_grid, levels=turb_level_values, 
                          norm=LogNorm(), cmap='viridis', alpha=0.8, extend='both')
        
        # Add fewer contour lines for cleaner look
        if turb_levels > 3:
            line_levels = turb_level_values[::2]  # Every other level
            cs1_lines = ax1.contour(Xi, Yi, turbidity_grid, levels=line_levels, 
                                   colors='black', alpha=0.3, linewidths=0.6)
            ax1.clabel(cs1_lines, inline=True, fontsize=8, fmt='%.3f')
        
        plt.colorbar(cs1, ax=ax1, label='Turbidity (600nm)')
    else:
        # Fallback for problematic data
        im1 = ax1.imshow(turbidity_grid, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(im1, ax=ax1, label='Turbidity (600nm)')
    
    # Plot data points
    ax1.scatter(x, y, c=turbidity, s=20, edgecolors='white', linewidth=0.3, cmap='viridis')
    
    # --- RATIO PLOT (with adjustable levels) ---
    ax2.set_title(f'Fluorescence Ratio vs Surfactant Concentrations\\n({surfactant_a_name} vs {surfactant_b_name})', 
                 fontsize=14, fontweight='bold')
    
    ratio_min, ratio_max = exp_data['ratio'].min(), exp_data['ratio'].max()
    
    if ratio_max > ratio_min:
        ratio_level_values = np.linspace(ratio_min, ratio_max, ratio_levels)
        
        cs2 = ax2.contourf(Xi, Yi, ratio_grid, levels=ratio_level_values, 
                          cmap='plasma', alpha=0.8, extend='both')
        
        # Add fewer contour lines
        if ratio_levels > 3:
            line_levels = ratio_level_values[::2]
            cs2_lines = ax2.contour(Xi, Yi, ratio_grid, levels=line_levels, 
                                   colors='black', alpha=0.3, linewidths=0.6)
            ax2.clabel(cs2_lines, inline=True, fontsize=8, fmt='%.3f')
        
        plt.colorbar(cs2, ax=ax2, label='Fluorescence Ratio')
    else:
        im2 = ax2.imshow(ratio_grid, aspect='auto', cmap='plasma', origin='lower')
        plt.colorbar(im2, ax=ax2, label='Fluorescence Ratio')
    
    ax2.scatter(x, y, c=ratio, s=20, edgecolors='white', linewidth=0.3, cmap='plasma')
    
    # --- FLUORESCENCE PLOT (with adjustable levels) ---
    ax3.set_title(f'Average Fluorescence Intensity vs Surfactant Concentrations\\n({surfactant_a_name} vs {surfactant_b_name})', 
                 fontsize=14, fontweight='bold')
    
    fluor_min, fluor_max = avg_fluorescence.min(), avg_fluorescence.max()
    
    if fluor_max > fluor_min:
        fluor_level_values = np.linspace(fluor_min, fluor_max, fluor_levels)
        
        cs3 = ax3.contourf(Xi, Yi, fluor_grid, levels=fluor_level_values, 
                          cmap='inferno', alpha=0.8, extend='both')
        
        # Add fewer contour lines
        if fluor_levels > 3:
            line_levels = fluor_level_values[::2]
            cs3_lines = ax3.contour(Xi, Yi, fluor_grid, levels=line_levels, 
                                   colors='black', alpha=0.3, linewidths=0.6)
            ax3.clabel(cs3_lines, inline=True, fontsize=8, fmt='%.0f')
        
        plt.colorbar(cs3, ax=ax3, label='Average Fluorescence Intensity')
    else:
        im3 = ax3.imshow(fluor_grid, aspect='auto', cmap='inferno', origin='lower')
        plt.colorbar(im3, ax=ax3, label='Average Fluorescence Intensity')
    
    ax3.scatter(x, y, c=avg_fluorescence, s=20, edgecolors='white', linewidth=0.3, cmap='inferno')
    
    # Format all axes
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel(f'{surfactant_a_name} Concentration (log10[mM])', fontsize=12)
        ax.set_ylabel(f'{surfactant_b_name} Concentration (log10[mM])', fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='box')
        
        # Cleaner axis labels
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        x_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in x_ticks]
        y_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in y_ticks]
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_yticklabels(y_labels, rotation=45)
    
    plt.tight_layout()
    
    # Save with informative filename
    output_path = csv_file_path.replace('.csv', f'_levels_{turb_levels}_{ratio_levels}_{fluor_levels}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Contour plot saved: {output_path}")
    
    plt.close(fig)
    return output_path

# === EXECUTION ===
if __name__ == "__main__":
    print(f"Creating adjustable contour maps: {SURFACTANT_A_NAME} + {SURFACTANT_B_NAME}")
    print(f"Input: {CSV_FILE}")
    print(f"Recommended levels for {173} data points: Turbidity=4-6, Ratio=5-7, Fluorescence=4-6")
    
    output_path = create_adjustable_contour_maps(
        CSV_FILE, 
        surfactant_a_name=SURFACTANT_A_NAME, 
        surfactant_b_name=SURFACTANT_B_NAME,
        turb_levels=TURBIDITY_LEVELS,
        ratio_levels=RATIO_LEVELS,
        fluor_levels=FLUORESCENCE_LEVELS,
        grid_res=GRID_RESOLUTION
    )
    
    print(f"🎯 EXPERIMENT: Try changing the level values at the top of this script!")
    print(f"   For your 173 points, try: 4-6 levels each for cleaner visualization")
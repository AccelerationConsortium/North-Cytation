"""
Detailed Contour Maps with Cubic Interpolation
Uses cubic interpolation for smoother, more detailed surfaces
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm

# === DETAILED SETTINGS ===
CSV_FILE = r"C:\Users\Owen\Documents\GitHub\North-Cytation\New folder\iterative_experiment_results.csv"

# Surfactant names
SURFACTANT_A_NAME = "SDS"  
SURFACTANT_B_NAME = "DTAB"

# Detailed contour levels (as preferred)
TURBIDITY_LEVELS = 8        # Detailed
RATIO_LEVELS = 10          # Detailed  
FLUORESCENCE_LEVELS = 8    # Detailed

# High resolution grid for smooth cubic interpolation
GRID_RESOLUTION = 120      # Higher resolution for cubic

def create_detailed_cubic_contours(csv_file_path, surfactant_a_name="SurfA", surfactant_b_name="SurfB"):
    """
    Create detailed contour maps using cubic interpolation for smoother surfaces
    """
    # Read and filter data
    df = pd.read_csv(csv_file_path)
    exp_data = df[df['well_type'] == 'experiment'].copy()
    
    # Control data
    control_data = df[df['well_type'] == 'control'].copy()
    water_control = control_data[control_data['control_type'] == 'water_blank']
    sds_control = control_data[control_data['control_type'] == 'surfactant_A_stock']
    ttab_control = control_data[control_data['control_type'] == 'surfactant_B_stock']
    
    if len(water_control) > 0:
        water_baseline_ratio = water_control['ratio'].max()
        water_baseline_turbidity = water_control['turbidity_600'].iloc[0]
        print(f"Water control baseline: Ratio: {water_baseline_ratio:.4f}, Turbidity: {water_baseline_turbidity:.4f}")
    else:
        water_baseline_ratio = 0.83
        water_baseline_turbidity = 0.04

    # Store control measurements for display
    control_measurements = {}
    
    if len(water_control) > 0:
        control_measurements['Water'] = {
            'turbidity': water_control['turbidity_600'].iloc[0],
            'ratio': water_control['ratio'].max()
        }
    
    if len(sds_control) > 0:
        control_measurements['SDS Stock'] = {
            'turbidity': sds_control['turbidity_600'].iloc[0],
            'ratio': sds_control['ratio'].iloc[0]
        }
    
    if len(ttab_control) > 0:
        control_measurements['TTAB Stock'] = {
            'turbidity': ttab_control['turbidity_600'].iloc[0],
            'ratio': ttab_control['ratio'].iloc[0]
        }

    print(f"Data points: {len(exp_data)}")
    print(f"Using CUBIC interpolation with detailed levels: T={TURBIDITY_LEVELS}, R={RATIO_LEVELS}, F={FLUORESCENCE_LEVELS}")
    
    # Extract data
    x = np.log10(exp_data['surf_A_conc_mm'])
    y = np.log10(exp_data['surf_B_conc_mm'])
    turbidity = exp_data['turbidity_600']
    ratio = exp_data['ratio']
    avg_fluorescence = (exp_data['fluorescence_334_373'] + exp_data['fluorescence_334_384']) / 2
    
    # High resolution grid for cubic interpolation
    xi = np.linspace(x.min(), x.max(), GRID_RESOLUTION)
    yi = np.linspace(y.min(), y.max(), GRID_RESOLUTION)
    Xi, Yi = np.meshgrid(xi, yi)
    
    print("Creating cubic interpolation surfaces (smooth and detailed)...")
    
    # CUBIC INTERPOLATION - Smoother but may create some artifacts
    turbidity_grid = griddata((x, y), turbidity, (Xi, Yi), method='cubic')
    ratio_grid = griddata((x, y), ratio, (Xi, Yi), method='cubic')
    avg_fluor_grid = griddata((x, y), avg_fluorescence, (Xi, Yi), method='cubic')
    
    # Fill any NaN areas with linear interpolation as backup
    turbidity_linear = griddata((x, y), turbidity, (Xi, Yi), method='linear')
    ratio_linear = griddata((x, y), ratio, (Xi, Yi), method='linear')
    fluor_linear = griddata((x, y), avg_fluorescence, (Xi, Yi), method='linear')
    
    turbidity_grid = np.where(np.isnan(turbidity_grid), turbidity_linear, turbidity_grid)
    ratio_grid = np.where(np.isnan(ratio_grid), ratio_linear, ratio_grid)
    avg_fluor_grid = np.where(np.isnan(avg_fluor_grid), fluor_linear, avg_fluor_grid)
    
    # Create the plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))  # Slightly wider for detailed view
    
    # --- DETAILED TURBIDITY PLOT ---
    ax1.set_title(f'Turbidity (600nm) - CUBIC Interpolation\\n({surfactant_a_name} vs {surfactant_b_name})', 
                 fontsize=14, fontweight='bold')
    
    # Ensure positive values for log scale and use original data range
    turb_min, turb_max = exp_data['turbidity_600'].min(), exp_data['turbidity_600'].max()
    turb_min_adj = max(turb_min, 0.001)
    
    if turb_max > turb_min_adj:
        # Detailed log-spaced levels
        turb_level_values = np.logspace(np.log10(turb_min_adj), np.log10(turb_max), TURBIDITY_LEVELS)
        
        # Ensure grid values are positive for log scale
        turbidity_grid_safe = np.maximum(turbidity_grid, turb_min_adj * 0.1)
        
        cs1 = ax1.contourf(Xi, Yi, turbidity_grid_safe, levels=turb_level_values, 
                          norm=LogNorm(), cmap='viridis', alpha=0.85)
        
        # Detailed contour lines
        cs1_lines = ax1.contour(Xi, Yi, turbidity_grid_safe, levels=turb_level_values, 
                               colors='black', alpha=0.4, linewidths=0.5)
        ax1.clabel(cs1_lines, inline=True, fontsize=8, fmt='%.3f')
        
        plt.colorbar(cs1, ax=ax1, label='Turbidity (600nm)')
    else:
        im1 = ax1.imshow(turbidity_grid, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(im1, ax=ax1, label='Turbidity (600nm)')
    
    # Plot data points
    ax1.scatter(x, y, c=turbidity, s=25, edgecolors='white', linewidth=0.6, cmap='viridis', zorder=5)
    
    # --- DETAILED RATIO PLOT ---
    ax2.set_title(f'Fluorescence Ratio - CUBIC Interpolation\\n({surfactant_a_name} vs {surfactant_b_name})', 
                 fontsize=14, fontweight='bold')
    
    ratio_min, ratio_max = exp_data['ratio'].min(), exp_data['ratio'].max()
    
    if ratio_max > ratio_min:
        ratio_level_values = np.linspace(ratio_min, ratio_max, RATIO_LEVELS)
        
        cs2 = ax2.contourf(Xi, Yi, ratio_grid, levels=ratio_level_values, 
                          cmap='plasma', alpha=0.85)
        
        # Detailed contour lines
        cs2_lines = ax2.contour(Xi, Yi, ratio_grid, levels=ratio_level_values, 
                               colors='black', alpha=0.4, linewidths=0.5)
        ax2.clabel(cs2_lines, inline=True, fontsize=8, fmt='%.3f')
        
        plt.colorbar(cs2, ax=ax2, label='Fluorescence Ratio')
    else:
        im2 = ax2.imshow(ratio_grid, aspect='auto', cmap='plasma', origin='lower')
        plt.colorbar(im2, ax=ax2, label='Fluorescence Ratio')
    
    ax2.scatter(x, y, c=ratio, s=25, edgecolors='white', linewidth=0.6, cmap='plasma', zorder=5)
    
    # --- DETAILED FLUORESCENCE PLOT ---
    ax3.set_title(f'Average Fluorescence - CUBIC Interpolation\\n({surfactant_a_name} vs {surfactant_b_name})', 
                 fontsize=14, fontweight='bold')
    
    fluor_min, fluor_max = avg_fluorescence.min(), avg_fluorescence.max()
    
    if fluor_max > fluor_min:
        fluor_level_values = np.linspace(fluor_min, fluor_max, FLUORESCENCE_LEVELS)
        
        cs3 = ax3.contourf(Xi, Yi, avg_fluor_grid, levels=fluor_level_values, 
                          cmap='inferno', alpha=0.85)
        
        # Detailed contour lines
        cs3_lines = ax3.contour(Xi, Yi, avg_fluor_grid, levels=fluor_level_values, 
                               colors='black', alpha=0.4, linewidths=0.5)
        ax3.clabel(cs3_lines, inline=True, fontsize=8, fmt='%.0f')
        
        plt.colorbar(cs3, ax=ax3, label='Average Fluorescence Intensity')
    else:
        im3 = ax3.imshow(avg_fluor_grid, aspect='auto', cmap='inferno', origin='lower')
        plt.colorbar(im3, ax=ax3, label='Average Fluorescence Intensity')
    
    ax3.scatter(x, y, c=avg_fluorescence, s=25, edgecolors='white', linewidth=0.6, cmap='inferno', zorder=5)
    
    # Format all axes
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel(f'{surfactant_a_name} Concentration (log10[mM])', fontsize=12)
        ax.set_ylabel(f'{surfactant_b_name} Concentration (log10[mM])', fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='box')
        
        # Add concentration labels
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        x_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in x_ticks]
        y_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in y_ticks]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_yticklabels(y_labels, rotation=45)
    
    # Add control blocks (same as before)
    if control_measurements:
        import matplotlib.cm as cm
        viridis_cmap = plt.cm.viridis
        plasma_cmap = plt.cm.plasma
        
        # Normalize values to get colors  
        turb_norm = LogNorm(vmin=max(turb_min, 0.001), vmax=turb_max)
        ratio_norm = plt.Normalize(vmin=ratio_min, vmax=ratio_max)
        
        for i, (control_name, values) in enumerate(control_measurements.items()):
            turb_val = values['turbidity']
            ratio_val = values['ratio']
            
            # Get colors from colormaps
            turb_color = viridis_cmap(turb_norm(max(turb_val, 0.001)))
            ratio_color = plasma_cmap(ratio_norm(ratio_val))
            
            # Create control blocks
            turb_block_x = 0.125 + i * 0.12
            turb_ax = fig.add_axes([turb_block_x, 0.08, 0.10, 0.06])
            turb_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=turb_color, 
                                          edgecolor='black', linewidth=1.5))
            turb_ax.set_xlim(0, 1)
            turb_ax.set_ylim(0, 1)
            turb_ax.set_xticks([])
            turb_ax.set_yticks([])
            
            ratio_block_x = 0.525 + i * 0.12  # Adjusted for wider figure
            ratio_ax = fig.add_axes([ratio_block_x, 0.08, 0.10, 0.06])  
            ratio_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=ratio_color,
                                           edgecolor='black', linewidth=1.5))
            ratio_ax.set_xlim(0, 1)
            ratio_ax.set_ylim(0, 1)
            ratio_ax.set_xticks([])
            ratio_ax.set_yticks([])
            
            # Add labels
            fig.text(turb_block_x + 0.05, 0.06, control_name, 
                    ha='center', va='top', fontsize=9, fontweight='bold')
            fig.text(turb_block_x + 0.05, 0.04, f'{turb_val:.3f}', 
                    ha='center', va='top', fontsize=8)
            
            fig.text(ratio_block_x + 0.05, 0.06, control_name, 
                    ha='center', va='top', fontsize=9, fontweight='bold')
            fig.text(ratio_block_x + 0.05, 0.04, f'{ratio_val:.3f}', 
                    ha='center', va='top', fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)  # Room for control blocks
    
    # Save the cubic interpolation version
    output_path = csv_file_path.replace('.csv', '_DETAILED_CUBIC.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Detailed cubic contour plot saved: {output_path}")
    
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    print("🎯 Creating DETAILED contour maps with CUBIC interpolation...")
    print(f"Levels: Turbidity={TURBIDITY_LEVELS}, Ratio={RATIO_LEVELS}, Fluorescence={FLUORESCENCE_LEVELS}")
    print(f"Grid resolution: {GRID_RESOLUTION}x{GRID_RESOLUTION}")
    
    output_path = create_detailed_cubic_contours(
        CSV_FILE,
        surfactant_a_name=SURFACTANT_A_NAME, 
        surfactant_b_name=SURFACTANT_B_NAME
    )
    
    print("🚀 CUBIC interpolation complete!")
    print("   This version should show smoother, more detailed surfaces")
    print("   Note: Cubic may create some interpolation artifacts in sparse areas")
"""
Improved contour mapping with no white patches
Uses better interpolation methods and fills gaps properly
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
import seaborn as sns

def create_smooth_contour_maps(csv_file_path, surfactant_a_name="SurfA", surfactant_b_name="SurfB"):
    """
    Create contour maps with improved interpolation to eliminate white patches
    """
    
    # Read the CSV data
    df = pd.read_csv(csv_file_path)
    
    # Filter for experiment data only (exclude controls)
    exp_data = df[df['well_type'] == 'experiment'].copy()
    
    # Extract control values for baseline reference
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
        print(f"No water control found, using fallback baseline")
    
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
    
    print(f"Total data points: {len(exp_data)}")
    print(f"SDS concentration range: {exp_data['surf_A_conc_mm'].min():.6f} - {exp_data['surf_A_conc_mm'].max():.2f} mM")
    print(f"TTAB concentration range: {exp_data['surf_B_conc_mm'].min():.6f} - {exp_data['surf_B_conc_mm'].max():.2f} mM")
    print(f"Turbidity range: {exp_data['turbidity_600'].min():.4f} - {exp_data['turbidity_600'].max():.4f}")
    print(f"Ratio range: {exp_data['ratio'].min():.4f} - {exp_data['ratio'].max():.4f}")
    
    # Extract data for plotting
    x = np.log10(exp_data['surf_A_conc_mm'])  # SDS concentration (log scale)
    y = np.log10(exp_data['surf_B_conc_mm'])  # TTAB concentration (log scale)
    turbidity = exp_data['turbidity_600']
    ratio = exp_data['ratio']
    
    # Calculate average fluorescence
    avg_fluorescence = (exp_data['fluorescence_334_373'] + exp_data['fluorescence_334_384']) / 2
    
    # Create higher resolution grid for smoother interpolation
    xi = np.linspace(x.min(), x.max(), 100)  # Increased resolution
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # IMPROVED INTERPOLATION - Use linear first, then fill with nearest neighbor
    print("Creating smooth interpolated surfaces...")
    
    # Method 1: Linear interpolation (more stable than cubic)
    turbidity_grid_linear = griddata((x, y), turbidity, (Xi, Yi), method='linear')
    ratio_grid_linear = griddata((x, y), ratio, (Xi, Yi), method='linear')
    avg_fluor_grid_linear = griddata((x, y), avg_fluorescence, (Xi, Yi), method='linear')
    
    # Method 2: Fill NaN areas with nearest neighbor
    turbidity_grid_nearest = griddata((x, y), turbidity, (Xi, Yi), method='nearest')
    ratio_grid_nearest = griddata((x, y), ratio, (Xi, Yi), method='nearest')
    avg_fluor_grid_nearest = griddata((x, y), avg_fluorescence, (Xi, Yi), method='nearest')
    
    # Combine: Use linear where available, nearest neighbor to fill gaps
    turbidity_grid = np.where(np.isnan(turbidity_grid_linear), turbidity_grid_nearest, turbidity_grid_linear)
    ratio_grid = np.where(np.isnan(ratio_grid_linear), ratio_grid_nearest, ratio_grid_linear)
    avg_fluor_grid = np.where(np.isnan(avg_fluor_grid_linear), avg_fluor_grid_nearest, avg_fluor_grid_linear)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- Improved Turbidity Contour Plot ---
    ax1.set_title(f'Turbidity (600nm) vs Surfactant Concentrations\\n({surfactant_a_name} vs {surfactant_b_name})', fontsize=14, fontweight='bold')
    
    # Ensure all values are positive for log scale
    turbidity_grid_safe = np.maximum(turbidity_grid, 0.001)  # Set minimum value
    turb_min, turb_max = np.nanmin(turbidity_grid_safe), np.nanmax(turbidity_grid_safe)
    
    # Create smooth contour levels
    turb_levels = np.logspace(np.log10(turb_min), np.log10(turb_max), 15)
    
    # Create filled contour plot
    cs1 = ax1.contourf(Xi, Yi, turbidity_grid_safe, levels=turb_levels, 
                       norm=LogNorm(), cmap='viridis', alpha=0.9, extend='both')
    
    # Add contour lines
    cs1_lines = ax1.contour(Xi, Yi, turbidity_grid_safe, levels=turb_levels, 
                           colors='black', alpha=0.3, linewidths=0.5)
    
    # Add colorbar
    cbar1 = plt.colorbar(cs1, ax=ax1)
    cbar1.set_label('Turbidity (600nm)', fontsize=12)
    
    # Plot actual data points
    scatter1 = ax1.scatter(x, y, c=turbidity, s=30, edgecolors='white', 
                          linewidth=0.8, cmap='viridis', zorder=5)
    
    # --- Improved Ratio Contour Plot ---
    ax2.set_title(f'Fluorescence Ratio vs Surfactant Concentrations\\n({surfactant_a_name} vs {surfactant_b_name})', fontsize=14, fontweight='bold')
    
    ratio_min, ratio_max = np.nanmin(ratio_grid), np.nanmax(ratio_grid)
    ratio_levels = np.linspace(ratio_min, ratio_max, 15)
    
    # Create filled contour plot
    cs2 = ax2.contourf(Xi, Yi, ratio_grid, levels=ratio_levels, 
                       cmap='plasma', alpha=0.9, extend='both')
    
    # Add contour lines
    cs2_lines = ax2.contour(Xi, Yi, ratio_grid, levels=ratio_levels, 
                           colors='black', alpha=0.3, linewidths=0.5)
    
    # Add colorbar
    cbar2 = plt.colorbar(cs2, ax=ax2)
    cbar2.set_label('Fluorescence Ratio', fontsize=12)
    
    # Plot actual data points
    scatter2 = ax2.scatter(x, y, c=ratio, s=30, edgecolors='white', 
                          linewidth=0.8, cmap='plasma', zorder=5)
    
    # --- Improved Average Fluorescence Plot ---
    ax3.set_title(f'Average Fluorescence Intensity vs Surfactant Concentrations\\n({surfactant_a_name} vs {surfactant_b_name})', fontsize=14, fontweight='bold')
    
    fluor_min, fluor_max = np.nanmin(avg_fluor_grid), np.nanmax(avg_fluor_grid)
    fluor_levels = np.linspace(fluor_min, fluor_max, 15)
    
    # Create filled contour plot
    cs3 = ax3.contourf(Xi, Yi, avg_fluor_grid, levels=fluor_levels, 
                       cmap='inferno', alpha=0.9, extend='both')
    
    # Add contour lines
    cs3_lines = ax3.contour(Xi, Yi, avg_fluor_grid, levels=fluor_levels, 
                           colors='black', alpha=0.3, linewidths=0.5)
    
    # Add colorbar
    cbar3 = plt.colorbar(cs3, ax=ax3)
    cbar3.set_label('Average Fluorescence Intensity', fontsize=12)
    
    # Plot actual data points
    scatter3 = ax3.scatter(x, y, c=avg_fluorescence, s=30, edgecolors='white', 
                          linewidth=0.8, cmap='inferno', zorder=5)
    
    # Format all axes consistently
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel(f'{surfactant_a_name} Concentration (log10[mM])', fontsize=12)
        ax.set_ylabel(f'{surfactant_b_name} Concentration (log10[mM])', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add concentration labels on axes
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        x_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in x_ticks]
        y_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in y_ticks]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_yticklabels(y_labels, rotation=45)
    
    # Add colored control blocks (same as original)
    if control_measurements:
        import matplotlib.cm as cm
        viridis_cmap = plt.cm.viridis
        plasma_cmap = plt.cm.plasma
        
        # Normalize values to get colors
        turb_norm = LogNorm(vmin=max(turb_min, 0.001), vmax=turb_max)
        ratio_norm = plt.Normalize(vmin=ratio_min, vmax=ratio_max)
        
        # Create control blocks
        for i, (control_name, values) in enumerate(control_measurements.items()):
            turb_val = values['turbidity']
            ratio_val = values['ratio']
            
            # Get colors from colormaps
            turb_color = viridis_cmap(turb_norm(max(turb_val, 0.001)))
            ratio_color = plasma_cmap(ratio_norm(ratio_val))
            
            # Create axes for control blocks
            turb_block_x = 0.125 + i * 0.12
            turb_ax = fig.add_axes([turb_block_x, 0.08, 0.10, 0.06])
            turb_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=turb_color, 
                                          edgecolor='black', linewidth=1.5))
            turb_ax.set_xlim(0, 1)
            turb_ax.set_ylim(0, 1)
            turb_ax.set_xticks([])
            turb_ax.set_yticks([])
            
            ratio_block_x = 0.575 + i * 0.12
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
    plt.subplots_adjust(bottom=0.20)
    
    # Save the plot
    try:
        output_path = csv_file_path.replace('iterative_experiment_results.csv', 'smooth_contour_maps.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Smooth contour plots saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        backup_path = "smooth_surfactant_contour_maps.png"
        plt.savefig(backup_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to backup location: {backup_path}")
    
    plt.close(fig)
    
    print("\n--- Smooth Contour Analysis Complete ---")
    print(f"Data points plotted: {len(exp_data)}")
    print(f"White patches eliminated using improved interpolation")
    print(f"Method: Linear + Nearest Neighbor gap filling")
    
    return fig, ax1, ax2, ax3

if __name__ == "__main__":
    # Path to your CSV data
    csv_file_path = r"C:\Users\Owen\Documents\GitHub\North-Cytation\New folder\iterative_experiment_results.csv"
    
    print("Creating SMOOTH contour maps (no white patches)...")
    print(f"Input CSV: {csv_file_path}")
    
    # Create the improved contour maps
    fig, ax1, ax2, ax3 = create_smooth_contour_maps(
        csv_file_path, 
        surfactant_a_name="SDS", 
        surfactant_b_name="DTAB"
    )
    
    print("Smooth contour mapping complete!")
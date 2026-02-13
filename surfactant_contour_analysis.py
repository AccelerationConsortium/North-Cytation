"""
Contour mapping for surfactant concentration vs turbidity and ratio
Creates colored contour plots with log-scale concentrations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
import seaborn as sns

def create_contour_maps(csv_file_path):
    """
    Create contour maps for turbidity and ratio vs surfactant concentrations
    
    Parameters:
    csv_file_path (str): Path to the iterative experiment results CSV
    """
    
    # Read the CSV data
    df = pd.read_csv(csv_file_path)
    
    # Filter for experiment data only (exclude controls)
    exp_data = df[df['well_type'] == 'experiment'].copy()
    
    # Extract control values for baseline reference
    control_data = df[df['well_type'] == 'control'].copy()
    water_control = control_data[control_data['control_type'] == 'water_blank']
    
    if len(water_control) > 0:
        water_baseline_ratio = water_control['ratio'].iloc[0]
        water_baseline_turbidity = water_control['turbidity_600'].iloc[0]
        print(f"Water control baseline - Ratio: {water_baseline_ratio:.4f}, Turbidity: {water_baseline_turbidity:.4f}")
    else:
        water_baseline_ratio = 0.83  # Fallback if no water control found
        water_baseline_turbidity = 0.04
        print(f"No water control found, using fallback baseline - Ratio: {water_baseline_ratio:.4f}")
    
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
    
    # Create grid for interpolation
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate data onto grid
    turbidity_grid = griddata((x, y), turbidity, (Xi, Yi), method='cubic')
    ratio_grid = griddata((x, y), ratio, (Xi, Yi), method='cubic')
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Turbidity Contour Plot (Log Scale) ---
    ax1.set_title('Turbidity (600nm) vs Surfactant Concentrations', fontsize=14, fontweight='bold')
    
    # Create log-spaced contour levels for turbidity
    turb_min, turb_max = np.nanmin(turbidity_grid), np.nanmax(turbidity_grid)
    # Use fewer contour levels since we only have 96 data points
    turb_levels = np.logspace(np.log10(max(turb_min, 0.01)), np.log10(turb_max), 8)
    
    # Create filled contour plot with log norm
    cs1 = ax1.contourf(Xi, Yi, turbidity_grid, levels=turb_levels, 
                       norm=LogNorm(), cmap='viridis', alpha=0.8)
    
    # Add contour lines
    cs1_lines = ax1.contour(Xi, Yi, turbidity_grid, levels=turb_levels, 
                           colors='black', alpha=0.4, linewidths=0.8)
    ax1.clabel(cs1_lines, inline=True, fontsize=9, fmt='%.3f')
    
    # Add colorbar
    cbar1 = plt.colorbar(cs1, ax=ax1)
    cbar1.set_label('Turbidity (600nm)', fontsize=12)
    
    # Plot actual data points
    scatter1 = ax1.scatter(x, y, c=turbidity, s=25, edgecolors='white', 
                          linewidth=0.5, cmap='viridis', norm=LogNorm())
    
    # --- Ratio Contour Plot (Linear Scale) - Capped at water baseline ---
    ax2.set_title('Fluorescence Ratio vs Surfactant Concentrations', fontsize=14, fontweight='bold')
    
    # Create linear contour levels for ratio - cap at water baseline
    ratio_min, ratio_max = np.nanmin(ratio_grid), min(np.nanmax(ratio_grid), water_baseline_ratio)
    ratio_levels = np.linspace(ratio_min, ratio_max, 10)  # Fewer levels for 96 points
    
    # Clip ratio values to max of water baseline for better scaling
    ratio_grid_clipped = np.clip(ratio_grid, None, water_baseline_ratio)
    
    # Create filled contour plot
    cs2 = ax2.contourf(Xi, Yi, ratio_grid_clipped, levels=ratio_levels, 
                       cmap='plasma', alpha=0.8)
    
    # Add contour lines
    cs2_lines = ax2.contour(Xi, Yi, ratio_grid_clipped, levels=ratio_levels, 
                           colors='black', alpha=0.4, linewidths=0.8)
    ax2.clabel(cs2_lines, inline=True, fontsize=9, fmt='%.3f')
    
    # Add colorbar
    cbar2 = plt.colorbar(cs2, ax=ax2)
    cbar2.set_label(f'Fluorescence Ratio (capped at water baseline: {water_baseline_ratio:.3f})', fontsize=12)
    
    # Plot actual data points - also clip for consistent coloring
    ratio_clipped = np.clip(ratio, None, water_baseline_ratio)
    scatter2 = ax2.scatter(x, y, c=ratio_clipped, s=25, edgecolors='white', 
                          linewidth=0.5, cmap='plasma')
    
    # --- Format both axes ---
    for ax in [ax1, ax2]:
        ax.set_xlabel('SDS Concentration (log10[mM])', fontsize=12)
        ax.set_ylabel('TTAB Concentration (log10[mM])', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add concentration labels on axes
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        x_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in x_ticks]
        y_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in y_ticks]
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_yticklabels(y_labels, rotation=45)
    
    plt.tight_layout()
    
    # Save the plot with error handling
    try:
        output_path = csv_file_path.replace('iterative_experiment_results.csv', 'contour_maps.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Contour plots saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        # Try saving to current directory as backup
        backup_path = "surfactant_contour_maps.png"
        plt.savefig(backup_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to backup location: {backup_path}")
    
    plt.show()
    
    # Print some statistics
    print("\n--- Data Statistics ---")
    print(f"Data points plotted: {len(exp_data)}")
    print(f"Turbidity - Min: {turbidity.min():.4f}, Max: {turbidity.max():.4f}")
    print(f"Ratio - Min: {ratio.min():.4f}, Max: {ratio.max():.4f}")
    print(f"Water baseline used for ratio cap: {water_baseline_ratio:.4f}")
    print("Contour mapping completed successfully!")
    
    return fig, ax1, ax2

def create_regular_plots(x, y, turbidity, ratio, exp_data, csv_file_path):
    """Create the original full-scale contour plots"""
    # Create grid for interpolation
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate data onto grid
    turbidity_grid = griddata((x, y), turbidity, (Xi, Yi), method='cubic')
    ratio_grid = griddata((x, y), ratio, (Xi, Yi), method='cubic')
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Turbidity Contour Plot (Log Scale) ---
    ax1.set_title('Turbidity (600nm) vs Surfactant Concentrations', fontsize=14, fontweight='bold')
    
    # Create log-spaced contour levels for turbidity
    turb_min, turb_max = np.nanmin(turbidity_grid), np.nanmax(turbidity_grid)
    # Use fewer contour levels since we only have 96 data points
    turb_levels = np.logspace(np.log10(max(turb_min, 0.01)), np.log10(turb_max), 8)
    
    # Create filled contour plot with log norm
    cs1 = ax1.contourf(Xi, Yi, turbidity_grid, levels=turb_levels, 
                       norm=LogNorm(), cmap='viridis', alpha=0.8)
    
    # Add contour lines
    cs1_lines = ax1.contour(Xi, Yi, turbidity_grid, levels=turb_levels, 
                           colors='black', alpha=0.4, linewidths=0.8)
    ax1.clabel(cs1_lines, inline=True, fontsize=9, fmt='%.3f')
    
    # Add colorbar
    cbar1 = plt.colorbar(cs1, ax=ax1)
    cbar1.set_label('Turbidity (600nm)', fontsize=12)
    
    # Plot actual data points
    scatter1 = ax1.scatter(x, y, c=turbidity, s=25, edgecolors='white', 
                          linewidth=0.5, cmap='viridis', norm=LogNorm())
    
    # --- Ratio Contour Plot (Linear Scale) ---
    ax2.set_title('Fluorescence Ratio vs Surfactant Concentrations', fontsize=14, fontweight='bold')
    
    # Create linear contour levels for ratio
    ratio_min, ratio_max = np.nanmin(ratio_grid), np.nanmax(ratio_grid)
    ratio_levels = np.linspace(ratio_min, ratio_max, 10)  # Fewer levels for 96 points
    
    # Create filled contour plot
    cs2 = ax2.contourf(Xi, Yi, ratio_grid, levels=ratio_levels, 
                       cmap='plasma', alpha=0.8)
    
    # Add contour lines
    cs2_lines = ax2.contour(Xi, Yi, ratio_grid, levels=ratio_levels, 
                           colors='black', alpha=0.4, linewidths=0.8)
    ax2.clabel(cs2_lines, inline=True, fontsize=9, fmt='%.3f')
    
    # Add colorbar
    cbar2 = plt.colorbar(cs2, ax=ax2)
    cbar2.set_label('Fluorescence Ratio', fontsize=12)
    
    # Plot actual data points
    scatter2 = ax2.scatter(x, y, c=ratio, s=25, edgecolors='white', 
                          linewidth=0.5, cmap='plasma')
    
    # --- Format both axes ---
    for ax in [ax1, ax2]:
        ax.set_xlabel('SDS Concentration (log10[mM])', fontsize=12)
        ax.set_ylabel('TTAB Concentration (log10[mM])', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add concentration labels on axes
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        x_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in x_ticks]
        y_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in y_ticks]
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_yticklabels(y_labels, rotation=45)
    
    plt.tight_layout()
    
    # Save the plot with error handling
    try:
        output_path = csv_file_path.replace('iterative_experiment_results.csv', 'contour_maps.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Regular contour plots saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        # Try saving to current directory as backup
        backup_path = "surfactant_contour_maps.png"
        plt.savefig(backup_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to backup location: {backup_path}")
    
    plt.show()
    return fig

def create_zoom_plots(x, y, turbidity, ratio, exp_data, csv_file_path, water_baseline_ratio, water_baseline_turbidity):
    """Create zoomed-in contour plots focusing on interesting regions"""
    
    # For turbidity zoom: focus on regions where turbidity > water baseline + margin
    turbidity_threshold = water_baseline_turbidity * 2  # 2x water baseline
    high_turb_mask = turbidity > turbidity_threshold
    
    # For ratio zoom: focus on regions that deviate significantly from water baseline
    ratio_deviation_threshold = 0.02  # 2% deviation from water baseline
    interesting_ratio_mask = np.abs(ratio - water_baseline_ratio) > ratio_deviation_threshold
    
    print(f"\nZoom regions:")
    print(f"High turbidity points (>{turbidity_threshold:.4f}, 2x water baseline): {high_turb_mask.sum()}")
    print(f"Significant ratio deviation (>{ratio_deviation_threshold:.3f} from water baseline {water_baseline_ratio:.4f}): {interesting_ratio_mask.sum()}")
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    
    # Regular plots (top row)
    create_single_contour(ax1, x, y, turbidity, 'Turbidity (600nm) - Full Scale', 
                         'viridis', log_scale=True)
    create_single_contour(ax2, x, y, ratio, 'Fluorescence Ratio - Full Scale', 
                         'plasma', log_scale=False)
    
    # Zoomed plots (bottom row)
    if high_turb_mask.sum() > 5:  # Need at least 5 points for good interpolation
        x_turb, y_turb, turb_zoom = x[high_turb_mask], y[high_turb_mask], turbidity[high_turb_mask]
        create_single_contour(ax3, x_turb, y_turb, turb_zoom, 
                             f'Turbidity Zoom (>2x Water Baseline)', 
                             'viridis', log_scale=True)
    else:
        ax3.text(0.5, 0.5, f'Insufficient high-turbidity\ndata for zoom\n(>{turbidity_threshold:.4f})', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Turbidity Zoom - Insufficient Data')
    
    if interesting_ratio_mask.sum() > 5:
        x_ratio, y_ratio, ratio_zoom = x[interesting_ratio_mask], y[interesting_ratio_mask], ratio[interesting_ratio_mask]
        create_single_contour(ax4, x_ratio, y_ratio, ratio_zoom, 
                             f'Ratio Zoom (Deviation from Water Baseline)', 
                             'plasma', log_scale=False)
    else:
        # Alternative: zoom on highest and lowest ratio regions
        n_points = len(ratio)
        high_ratio_idx = np.argsort(ratio)[-n_points//4:]  # Top 25%
        low_ratio_idx = np.argsort(ratio)[:n_points//4]   # Bottom 25%
        extreme_idx = np.concatenate([high_ratio_idx, low_ratio_idx])
        
        x_ratio, y_ratio, ratio_zoom = x[extreme_idx], y[extreme_idx], ratio[extreme_idx]
        create_single_contour(ax4, x_ratio, y_ratio, ratio_zoom, 
                             'Ratio Zoom (Extreme Values)', 
                             'plasma', log_scale=False)
    
    # Format all axes
    for ax in [ax1, ax2, ax3, ax4]:
        format_contour_axis(ax)
    
    plt.tight_layout()
    
    # Save zoom plots
    try:
        zoom_path = csv_file_path.replace('iterative_experiment_results.csv', 'contour_maps_zoom.png')
        plt.savefig(zoom_path, dpi=300, bbox_inches='tight')
        print(f"Zoom contour plots saved to: {zoom_path}")
    except Exception as e:
        backup_path = "surfactant_contour_maps_zoom.png"
        plt.savefig(backup_path, dpi=300, bbox_inches='tight')
        print(f"Zoom plots saved to backup: {backup_path}")
    
    plt.show()
    return fig

def create_single_contour(ax, x, y, z, title, cmap, log_scale=False):
    """Create a single contour plot"""
    if len(x) < 4:  # Need minimum points for interpolation
        ax.text(0.5, 0.5, 'Insufficient data\nfor interpolation', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return
    
    # Apply 0.85 cap for ratio plots (detect by title or value range)
    is_ratio_plot = 'ratio' in title.lower() or (z.max() < 1.0 and z.min() > 0.7)
    if is_ratio_plot:
        z = np.clip(z, None, 0.85)
    
    # Create grid
    xi = np.linspace(x.min(), x.max(), 30)
    yi = np.linspace(y.min(), y.max(), 30)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate
    zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    
    # Apply cap to interpolated grid if ratio plot
    if is_ratio_plot:
        zi = np.clip(zi, None, 0.85)
    
    # Create contour levels
    z_min, z_max = np.nanmin(zi), np.nanmax(zi)
    if log_scale and z_min > 0:
        levels = np.logspace(np.log10(max(z_min, 0.01)), np.log10(z_max), 6)
        norm = LogNorm()
    else:
        levels = np.linspace(z_min, z_max, 8)
        norm = None
    
    # Plot
    cs = ax.contourf(Xi, Yi, zi, levels=levels, cmap=cmap, alpha=0.8, norm=norm)
    cs_lines = ax.contour(Xi, Yi, zi, levels=levels, colors='black', alpha=0.4, linewidths=0.6)
    ax.clabel(cs_lines, inline=True, fontsize=8, fmt='%.3f')
    
    # Add data points
    if log_scale and z.min() > 0:
        scatter = ax.scatter(x, y, c=z, s=30, edgecolors='white', linewidth=0.5, 
                           cmap=cmap, norm=LogNorm())
    else:
        scatter = ax.scatter(x, y, c=z, s=30, edgecolors='white', linewidth=0.5, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax)
    cbar.ax.tick_params(labelsize=9)
    
    # Update colorbar label for ratio plots
    if is_ratio_plot:
        cbar.set_label('Ratio (capped at 0.85)', fontsize=9)
    
    ax.set_title(title, fontsize=12, fontweight='bold')

def format_contour_axis(ax):
    """Format contour plot axis with proper labels"""
    ax.set_xlabel('SDS Concentration (log10[mM])', fontsize=10)
    ax.set_ylabel('TTAB Concentration (log10[mM])', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add concentration labels on axes
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    x_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in x_ticks]
    y_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in y_ticks]
    ax.set_xticklabels(x_labels, rotation=45, fontsize=9)
    ax.set_yticklabels(y_labels, rotation=45, fontsize=9)

if __name__ == "__main__":
    # Path to your data file
    csv_path = r"C:\Users\owenm\OneDrive\Desktop\North Robotics\utoronto_demo\utoronto_demo\output\surfactant_grid_SDS_TTAB_20260212_205209_HARDWARE\iterative_experiment_results.csv"
    
    # Create both regular and zoom contour maps
    create_contour_maps(csv_path)
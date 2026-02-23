"""
Contour mapping for surfactant concentration vs turbidity and ratio
Creates colored contour plots with log-scale concentrations
Uses water control baseline for ratio scaling 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
import seaborn as sns#

def create_contour_maps(csv_file_path, surfactant_a_name="SurfA", surfactant_b_name="SurfB"):
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
    sds_control = control_data[control_data['control_type'] == 'surfactant_A_stock']
    ttab_control = control_data[control_data['control_type'] == 'surfactant_B_stock']
    
    if len(water_control) > 0:
        # Use the higher ratio value if there are multiple water blanks
        water_baseline_ratio = water_control['ratio'].max()
        water_baseline_turbidity = water_control['turbidity_600'].iloc[0]
        print(f"Water control baseline (using max ratio): Ratio: {water_baseline_ratio:.4f}, Turbidity: {water_baseline_turbidity:.4f}")
        if len(water_control) > 1:
            print(f"Found {len(water_control)} water blanks, using max ratio: {water_baseline_ratio:.4f}")
    else:
        water_baseline_ratio = 0.83  # Fallback if no water control found
        water_baseline_turbidity = 0.04
        print(f"No water control found, using fallback baseline - Ratio: {water_baseline_ratio:.4f}")
    
    # Store control measurements for display
    control_measurements = {}
    
    if len(water_control) > 0:
        control_measurements['Water'] = {
            'turbidity': water_control['turbidity_600'].iloc[0],
            'ratio': water_control['ratio'].max()  # Use max ratio
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
    
    # Create grid for interpolation
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate data onto grid
    turbidity_grid = griddata((x, y), turbidity, (Xi, Yi), method='cubic')
    ratio_grid = griddata((x, y), ratio, (Xi, Yi), method='cubic')
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))  # Three plots side by side
    
    # --- Turbidity Contour Plot (Log Scale) ---
    ax1.set_title(f'Turbidity (600nm) vs Surfactant Concentrations\\n({surfactant_a_name} vs {surfactant_b_name})', fontsize=14, fontweight='bold')
    
    # Create log-spaced contour levels for turbidity with better error handling
    turb_min, turb_max = np.nanmin(turbidity_grid), np.nanmax(turbidity_grid)
    
    # Safety check for contour levels
    if turb_max <= turb_min or turb_max <= 0 or np.isnan(turb_min) or np.isnan(turb_max):
        # Fallback to simple visualization if contour levels can't be calculated
        im1 = ax1.imshow(turbidity_grid, aspect='auto', cmap='viridis', origin='lower')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Turbidity (600nm)', fontsize=12)
        ax1.set_title(f'Turbidity (600nm) - Image View\\n({surfactant_a_name} vs {surfactant_b_name})', fontsize=14, fontweight='bold')
    else:
        # Use fewer contour levels and ensure proper range
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
                          linewidth=0.5, cmap='viridis')
    
    # --- Ratio Contour Plot (Linear Scale) - Capped at water baseline ---
    ax2.set_title(f'Fluorescence Ratio vs Surfactant Concentrations\\n({surfactant_a_name} vs {surfactant_b_name})', fontsize=14, fontweight='bold')
    
    # Create linear contour levels for ratio - cap at water baseline with error handling
    ratio_min, ratio_max = np.nanmin(ratio_grid), min(np.nanmax(ratio_grid), water_baseline_ratio)
    
    # Safety check for contour levels
    if ratio_max <= ratio_min or np.isnan(ratio_min) or np.isnan(ratio_max) or (ratio_max - ratio_min) < 1e-10:
        # Fallback to simple visualization if contour levels can't be calculated
        ratio_grid_clipped = np.clip(ratio_grid, None, water_baseline_ratio)
        im2 = ax2.imshow(ratio_grid_clipped, aspect='auto', cmap='plasma', origin='lower')
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label(f'Fluorescence Ratio (capped at water baseline: {water_baseline_ratio:.3f})', fontsize=12)
        ax2.set_title(f'Fluorescence Ratio - Image View\\n({surfactant_a_name} vs {surfactant_b_name})', fontsize=14, fontweight='bold')
    else:
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
    
    # --- Average Fluorescence Intensity Contour Plot ---
    ax3.set_title(f'Average Fluorescence Intensity vs Surfactant Concentrations\\n({surfactant_a_name} vs {surfactant_b_name})', fontsize=14, fontweight='bold')
    
    # Calculate average fluorescence (F373 + F384)/2
    avg_fluorescence = (exp_data['fluorescence_334_373'] + exp_data['fluorescence_334_384']) / 2
    
    # Interpolate average fluorescence onto grid
    avg_fluor_grid = griddata((x, y), avg_fluorescence, (Xi, Yi), method='cubic')
    
    # Create linear contour levels for average fluorescence with error handling
    fluor_min, fluor_max = np.nanmin(avg_fluor_grid), np.nanmax(avg_fluor_grid)
    
    # Safety check for contour levels
    if fluor_max <= fluor_min or np.isnan(fluor_min) or np.isnan(fluor_max) or (fluor_max - fluor_min) < 1e-10:
        # Fallback to simple visualization if contour levels can't be calculated
        im3 = ax3.imshow(avg_fluor_grid, aspect='auto', cmap='inferno', origin='lower')
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('Average Fluorescence Intensity', fontsize=12)
        ax3.set_title(f'Average Fluorescence - Image View\\n({surfactant_a_name} vs {surfactant_b_name})', fontsize=14, fontweight='bold')
    else:
        fluor_levels = np.linspace(fluor_min, fluor_max, 10)
        
        # Create filled contour plot
        cs3 = ax3.contourf(Xi, Yi, avg_fluor_grid, levels=fluor_levels, 
                           cmap='inferno', alpha=0.8)
        
        # Add contour lines
        cs3_lines = ax3.contour(Xi, Yi, avg_fluor_grid, levels=fluor_levels, 
                               colors='black', alpha=0.4, linewidths=0.8)
        ax3.clabel(cs3_lines, inline=True, fontsize=9, fmt='%.0f')
        
        # Add colorbar
        cbar3 = plt.colorbar(cs3, ax=ax3)
        cbar3.set_label('Average Fluorescence Intensity', fontsize=12)
    
    # Plot actual data points
    scatter3 = ax3.scatter(x, y, c=avg_fluorescence, s=25, edgecolors='white', 
                          linewidth=0.5, cmap='inferno')
    
    # Format third axis
    ax3.set_xlabel(f'{surfactant_a_name} Concentration (log10[mM])', fontsize=12)
    ax3.set_ylabel(f'{surfactant_b_name} Concentration (log10[mM])', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')  # Make plot square
    
    # Add concentration labels on axes
    x_ticks_3 = ax3.get_xticks()
    y_ticks_3 = ax3.get_yticks()
    x_labels_3 = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in x_ticks_3]
    y_labels_3 = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in y_ticks_3]
    ax3.set_xticklabels(x_labels_3, rotation=45)
    ax3.set_yticklabels(y_labels_3, rotation=45)
    
    # --- Format first two axes ---
    for ax in [ax1, ax2]:
        ax.set_xlabel(f'{surfactant_a_name} Concentration (log10[mM])', fontsize=12)
        ax.set_ylabel(f'{surfactant_b_name} Concentration (log10[mM])', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')  # Make plots square
        
        # Add concentration labels on axes
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        x_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in x_ticks]
        y_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in y_ticks]
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_yticklabels(y_labels, rotation=45)
    
    # Add colored control blocks beneath each graph
    if control_measurements:
        # Import colormaps
        import matplotlib.cm as cm
        viridis_cmap = plt.cm.viridis
        plasma_cmap = plt.cm.plasma
        
        # Normalize values to get colors
        turb_norm = LogNorm(vmin=max(turb_min, 0.01), vmax=turb_max)
        ratio_norm = plt.Normalize(vmin=ratio_min, vmax=ratio_max)
        
        # Get control names and values
        control_names = list(control_measurements.keys())
        n_controls = len(control_names)
        
        # Create small subplot axes for control blocks
        # Turbidity control blocks (beneath left plot)
        for i, (control_name, values) in enumerate(control_measurements.items()):
            turb_val = values['turbidity']
            ratio_val = values['ratio']
            
            # Get colors from colormaps
            if turb_val > 0:
                turb_color = viridis_cmap(turb_norm(turb_val))
            else:
                turb_color = viridis_cmap(0)
            
            ratio_color = plasma_cmap(ratio_norm(min(ratio_val, water_baseline_ratio)))
            
            # Create axes for turbidity control block
            turb_block_x = 0.125 + i * 0.12  # Position under left plot
            turb_ax = fig.add_axes([turb_block_x, 0.08, 0.10, 0.06])
            turb_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=turb_color, 
                                          edgecolor='black', linewidth=1.5))
            turb_ax.set_xlim(0, 1)
            turb_ax.set_ylim(0, 1)
            turb_ax.set_xticks([])
            turb_ax.set_yticks([])
            
            # Create axes for ratio control block  
            ratio_block_x = 0.575 + i * 0.12  # Position under right plot
            ratio_ax = fig.add_axes([ratio_block_x, 0.08, 0.10, 0.06])
            ratio_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=ratio_color,
                                           edgecolor='black', linewidth=1.5))
            ratio_ax.set_xlim(0, 1)
            ratio_ax.set_ylim(0, 1)
            ratio_ax.set_xticks([])
            ratio_ax.set_yticks([])
            
            # Add labels beneath the blocks
            fig.text(turb_block_x + 0.05, 0.06, control_name, 
                    ha='center', va='top', fontsize=9, fontweight='bold')
            fig.text(turb_block_x + 0.05, 0.04, f'{turb_val:.3f}', 
                    ha='center', va='top', fontsize=8)
            
            fig.text(ratio_block_x + 0.05, 0.06, control_name, 
                    ha='center', va='top', fontsize=9, fontweight='bold')
            fig.text(ratio_block_x + 0.05, 0.04, f'{ratio_val:.3f}', 
                    ha='center', va='top', fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)  # Make room for control blocks
    
    # Save the plot with error handling
    try:
        output_path = csv_file_path.replace('iterative_experiment_results.csv', 'contour_maps_with_fluorescence.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Contour plots saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        # Try saving to current directory as backup
        backup_path = "surfactant_contour_maps.png"
        plt.savefig(backup_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to backup location: {backup_path}")
    
    # Close the figure to prevent display and free memory
    plt.close(fig)
    
    # Print some statistics
    print("\n--- Data Statistics ---")
    print(f"Data points plotted: {len(exp_data)}")
    print(f"Turbidity - Min: {turbidity.min():.4f}, Max: {turbidity.max():.4f}")
    print(f"Ratio - Min: {ratio.min():.4f}, Max: {ratio.max():.4f}")
    print(f"Water baseline used for ratio cap: {water_baseline_ratio:.4f}")
    print("Contour mapping with average fluorescence analysis completed successfully!")
    
    return fig, ax1, ax2, ax3

if __name__ == "__main__":
    # This program is now called from workflows, not run directly
    # For testing, you can uncomment and modify the path below:
    # csv_path = r"path/to/your/iterative_experiment_results.csv"
    # fig, ax1, ax2, ax3 = create_contour_maps(csv_path, "SDS", "TTAB")
    print("Contour analysis program - import this module to use create_contour_maps()")
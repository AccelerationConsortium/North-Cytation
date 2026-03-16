"""
Complete Linear Contour Map Generator  
Full-featured version with regular linear interpolation styling (subdued, clean)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm

# === TWEAKABLE PARAMETERS ===
CSV_FILE = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\surfactant_grid_SDS_BDDAC_March_12_Overnight__SDS_CTAB_BDDAC_20260313_012112\iterative_experiment_results.csv"

# Surfactant names for plot titles
SURFACTANT_A_NAME = "SDS"
SURFACTANT_B_NAME = "BDDAC"

# Contour level settings - ADJUST THESE TO REDUCE/INCREASE DETAIL
TURBIDITY_LEVELS = 8        # Detailed levels (as you preferred)
RATIO_LEVELS = 10          # Detailed levels
FLUORESCENCE_LEVELS = 8    # Detailed levels

# Grid resolution (regular linear quality)
GRID_RESOLUTION = 100       # Good balance - not too high

# Visual settings (regular linear style - more subdued)
ALPHA_CONTOURS = 0.8        # Nice shading (not too strong)
ALPHA_POINTS = 0.8          # Subdued data points
POINT_SIZE = 20             # Same as good version (smaller, more subdued)
POINT_EDGE_WIDTH = 0.5      # Thin edges

def create_complete_linear_contours(csv_file_path, surfactant_a_name="SurfA", surfactant_b_name="SurfB"):
    """
    Create complete contour maps with regular linear interpolation styling
    Includes all 3 plots + control blocks with subdued, clean appearance
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
        print(f"Water control baseline: Ratio: {water_baseline_ratio:.4f}, Turbidity: {water_baseline_turbidity:.4f}")
    else:
        water_baseline_ratio = 0.83  # Fallback if no water control found
        water_baseline_turbidity = 0.04
        print(f"No water control found, using fallback baseline")
    
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
    print(f"Using REGULAR LINEAR interpolation with clean styling")
    
    # Extract data for plotting
    x = np.log10(exp_data['surf_A_conc_mm'])  # SDS concentration (log scale)
    y = np.log10(exp_data['surf_B_conc_mm'])  # TTAB concentration (log scale)
    turbidity = exp_data['turbidity_600']
    ratio = exp_data['ratio']
    
    # Calculate average fluorescence (F373 + F384)/2
    avg_fluorescence = (exp_data['fluorescence_334_373'] + exp_data['fluorescence_334_384']) / 2
    
    # Create grid for interpolation
    xi = np.linspace(x.min(), x.max(), GRID_RESOLUTION)
    yi = np.linspace(y.min(), y.max(), GRID_RESOLUTION)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # REGULAR LINEAR interpolation (reliable, no artifacts)
    print("Creating linear interpolation surfaces...")
    turbidity_grid_linear = griddata((x, y), turbidity, (Xi, Yi), method='linear')
    ratio_grid_linear = griddata((x, y), ratio, (Xi, Yi), method='linear')
    avg_fluor_grid_linear = griddata((x, y), avg_fluorescence, (Xi, Yi), method='linear')
    
    # Fill NaN areas with nearest neighbor (no artifacts)
    turbidity_grid_nearest = griddata((x, y), turbidity, (Xi, Yi), method='nearest')
    ratio_grid_nearest = griddata((x, y), ratio, (Xi, Yi), method='nearest')
    avg_fluor_grid_nearest = griddata((x, y), avg_fluorescence, (Xi, Yi), method='nearest')
    
    # Combine: Use linear where available, nearest neighbor to fill gaps
    turbidity_grid = np.where(np.isnan(turbidity_grid_linear), turbidity_grid_nearest, turbidity_grid_linear)
    ratio_grid = np.where(np.isnan(ratio_grid_linear), ratio_grid_nearest, ratio_grid_linear)
    avg_fluor_grid = np.where(np.isnan(avg_fluor_grid_linear), avg_fluor_grid_nearest, avg_fluor_grid_linear)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))  # Three plots side by side
    
    # --- Turbidity Contour Plot ---
    ax1.set_title(f'Turbidity (600nm) vs Surfactant Concentrations\\n({surfactant_a_name} vs {surfactant_b_name})', fontsize=14, fontweight='bold')
    
    # Use original data range for contour levels
    turb_min, turb_max = exp_data['turbidity_600'].min(), exp_data['turbidity_600'].max()
    turb_min_adj = max(turb_min, 0.01)  # Minimum for log scale
    
    if turb_max > turb_min_adj:
        # Create log-spaced contour levels
        turb_levels = np.logspace(np.log10(turb_min_adj), np.log10(turb_max), TURBIDITY_LEVELS)
        
        # Ensure grid values are positive for log scale
        turbidity_grid_safe = np.maximum(turbidity_grid, turb_min_adj * 0.1)
        
        # Create filled contour plot with regular styling
        cs1 = ax1.contourf(Xi, Yi, turbidity_grid_safe, levels=turb_levels, 
                           norm=LogNorm(), cmap='viridis', alpha=ALPHA_CONTOURS)
        
        # Add colorbar
        cbar1 = plt.colorbar(cs1, ax=ax1)
        cbar1.set_label('Turbidity (600nm)', fontsize=12)
    
    # Plot actual data points with subdued styling
    scatter1 = ax1.scatter(x, y, c=turbidity, s=POINT_SIZE, edgecolors='white', 
                          linewidth=POINT_EDGE_WIDTH, cmap='viridis', alpha=ALPHA_POINTS)
    
    # --- Ratio Contour Plot ---
    ax2.set_title(f'Fluorescence Ratio vs Surfactant Concentrations\\n({surfactant_a_name} vs {surfactant_b_name})', fontsize=14, fontweight='bold')
    
    ratio_min, ratio_max = exp_data['ratio'].min(), exp_data['ratio'].max()
    
    if ratio_max > ratio_min:
        ratio_levels = np.linspace(ratio_min, ratio_max, RATIO_LEVELS)
        
        # Create filled contour plot
        cs2 = ax2.contourf(Xi, Yi, ratio_grid, levels=ratio_levels, 
                           cmap='plasma', alpha=ALPHA_CONTOURS)
        
        # Add colorbar
        cbar2 = plt.colorbar(cs2, ax=ax2)
        cbar2.set_label('Fluorescence Ratio', fontsize=12)
    
    # Plot actual data points
    scatter2 = ax2.scatter(x, y, c=ratio, s=POINT_SIZE, edgecolors='white', 
                          linewidth=POINT_EDGE_WIDTH, cmap='plasma', alpha=ALPHA_POINTS)
    
    # --- Average Fluorescence Intensity Contour Plot ---
    ax3.set_title(f'Average Fluorescence Intensity vs Surfactant Concentrations\\n({surfactant_a_name} vs {surfactant_b_name})', fontsize=14, fontweight='bold')
    
    fluor_min, fluor_max = avg_fluorescence.min(), avg_fluorescence.max()
    
    if fluor_max > fluor_min:
        fluor_levels = np.linspace(fluor_min, fluor_max, FLUORESCENCE_LEVELS)
        
        # Create filled contour plot
        cs3 = ax3.contourf(Xi, Yi, avg_fluor_grid, levels=fluor_levels, 
                           cmap='inferno', alpha=ALPHA_CONTOURS)
        
        # Add colorbar
        cbar3 = plt.colorbar(cs3, ax=ax3)
        cbar3.set_label('Average Fluorescence Intensity', fontsize=12)
    
    # Plot actual data points
    scatter3 = ax3.scatter(x, y, c=avg_fluorescence, s=POINT_SIZE, edgecolors='white', 
                          linewidth=POINT_EDGE_WIDTH, cmap='inferno', alpha=ALPHA_POINTS)
    
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
    
    # Add colored control blocks
    if control_measurements:
        # Import colormaps
        import matplotlib.cm as cm
        viridis_cmap = plt.cm.viridis
        plasma_cmap = plt.cm.plasma
        
        # Normalize values to get colors
        turb_norm = LogNorm(vmin=max(turb_min, 0.01), vmax=turb_max)
        ratio_norm = plt.Normalize(vmin=ratio_min, vmax=ratio_max)
        
        # Create control blocks
        for i, (control_name, values) in enumerate(control_measurements.items()):
            turb_val = values['turbidity']
            ratio_val = values['ratio']
            
            # Get colors from colormaps
            turb_color = viridis_cmap(turb_norm(max(turb_val, 0.01)))
            ratio_color = plasma_cmap(ratio_norm(ratio_val))
            
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
            ratio_block_x = 0.575 + i * 0.12  # Position under middle plot
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
    
    # Save the plot
    output_path = csv_file_path.replace('iterative_experiment_results.csv', 'COMPLETE_LINEAR_CONTOURS.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Complete linear contour plots saved to: {output_path}")
    
    # Close the figure to prevent display and free memory
    plt.close(fig)
    
    return output_path

# === EXECUTION ===
if __name__ == "__main__":
    print("🎯 Creating COMPLETE LINEAR contour maps...")
    print("Features: 3 plots + control blocks + regular linear styling")
    print("Style: Subdued data points, nice shading, clean appearance")
    print("="*60)
    
    output_path = create_complete_linear_contours(
        CSV_FILE, 
        surfactant_a_name=SURFACTANT_A_NAME, 
        surfactant_b_name=SURFACTANT_B_NAME
    )
    
    print("\\n✅ COMPLETE LINEAR VERSION READY!")
    print("This combines:")
    print("  • Regular linear interpolation (reliable, no artifacts)")  
    print("  • All 3 plots (turbidity, ratio, fluorescence)")
    print("  • Control blocks (colored reference values)")
    print("  • Subdued styling (clean, not distracting)")
    print("\\n🎯 Perfect for scientific presentations with clean aesthetics!")
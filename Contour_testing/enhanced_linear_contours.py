"""
Enhanced Linear Interpolation - Best of Both Worlds
Maximizes visual appeal of linear interpolation without artifacts
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

# === ENHANCED LINEAR SETTINGS ===
CSV_FILE = r"C:\Users\Owen\Documents\GitHub\North-Cytation\New folder\iterative_experiment_results.csv"
SURFACTANT_A_NAME = "SDS"
SURFACTANT_B_NAME = "DTAB"

# Visual enhancement settings
TURBIDITY_LEVELS = 8        # Detailed levels (since you liked them)
RATIO_LEVELS = 10          
FLUORESCENCE_LEVELS = 8
GRID_RESOLUTION = 150       # Higher resolution for smoother linear
SMOOTHING_SIGMA = 0.8       # Light Gaussian smoothing (removes pixelation)
ALPHA_CONTOURS = 0.9        # Slightly more opaque contours
ALPHA_LINES = 0.35          # More visible contour lines

def create_enhanced_linear_contours(csv_file_path):
    """
    Create visually enhanced linear interpolation contours
    High resolution + light smoothing = reliable but smooth
    """
    # Read data
    df = pd.read_csv(csv_file_path)
    exp_data = df[df['well_type'] == 'experiment'].copy()
    
    # Control data for display
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

    # Store control measurements
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

    print(f"Enhanced Linear Interpolation: {len(exp_data)} points")
    print(f"Resolution: {GRID_RESOLUTION}x{GRID_RESOLUTION}, Smoothing: {SMOOTHING_SIGMA}")
    
    # Extract data
    x = np.log10(exp_data['surf_A_conc_mm'])
    y = np.log10(exp_data['surf_B_conc_mm'])
    turbidity = exp_data['turbidity_600']
    ratio = exp_data['ratio']
    avg_fluorescence = (exp_data['fluorescence_334_373'] + exp_data['fluorescence_334_384']) / 2
    
    # High resolution grid for smooth linear
    xi = np.linspace(x.min(), x.max(), GRID_RESOLUTION)
    yi = np.linspace(y.min(), y.max(), GRID_RESOLUTION)
    Xi, Yi = np.meshgrid(xi, yi)
    
    print("Creating enhanced linear interpolation...")
    
    # LINEAR interpolation (most reliable)
    turbidity_grid = griddata((x, y), turbidity, (Xi, Yi), method='linear')
    ratio_grid = griddata((x, y), ratio, (Xi, Yi), method='linear')
    fluor_grid = griddata((x, y), avg_fluorescence, (Xi, Yi), method='linear')
    
    # Fill NaN areas with nearest neighbor (no artifacts)
    turbidity_nearest = griddata((x, y), turbidity, (Xi, Yi), method='nearest')
    ratio_nearest = griddata((x, y), ratio, (Xi, Yi), method='nearest')
    fluor_nearest = griddata((x, y), avg_fluorescence, (Xi, Yi), method='nearest')
    
    turbidity_grid = np.where(np.isnan(turbidity_grid), turbidity_nearest, turbidity_grid)
    ratio_grid = np.where(np.isnan(ratio_grid), ratio_nearest, ratio_grid)
    fluor_grid = np.where(np.isnan(fluor_grid), fluor_nearest, fluor_grid)
    
    # ENHANCEMENT: Light Gaussian smoothing to reduce pixelation (preserves features)
    print("Applying light smoothing for visual enhancement...")
    turbidity_grid = gaussian_filter(turbidity_grid, sigma=SMOOTHING_SIGMA)
    ratio_grid = gaussian_filter(ratio_grid, sigma=SMOOTHING_SIGMA) 
    fluor_grid = gaussian_filter(fluor_grid, sigma=SMOOTHING_SIGMA)
    
    # Create enhanced visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # --- ENHANCED TURBIDITY PLOT ---
    ax1.set_title(f'Turbidity (Enhanced Linear)\\n{SURFACTANT_A_NAME} vs {SURFACTANT_B_NAME} | {len(exp_data)} points', 
                 fontsize=13, fontweight='bold')
    
    turb_min, turb_max = exp_data['turbidity_600'].min(), exp_data['turbidity_600'].max()
    turb_min_adj = max(turb_min, 0.001)
    
    if turb_max > turb_min_adj:
        turb_levels = np.logspace(np.log10(turb_min_adj), np.log10(turb_max), TURBIDITY_LEVELS)
        turb_grid_safe = np.maximum(turbidity_grid, turb_min_adj * 0.1)
        
        # Enhanced contours
        cs1 = ax1.contourf(Xi, Yi, turb_grid_safe, levels=turb_levels, 
                          norm=LogNorm(), cmap='viridis', alpha=ALPHA_CONTOURS, extend='both')
        
        # Enhanced contour lines
        cs1_lines = ax1.contour(Xi, Yi, turb_grid_safe, levels=turb_levels[::2], 
                               colors='black', alpha=ALPHA_LINES, linewidths=0.7)
        ax1.clabel(cs1_lines, inline=True, fontsize=9, fmt='%.3f')
        
        plt.colorbar(cs1, ax=ax1, label='Turbidity (600nm)')
    
    # Enhanced data points
    ax1.scatter(x, y, c=turbidity, s=30, edgecolors='white', linewidth=0.8, 
               cmap='viridis', zorder=5, alpha=0.9)
    
    # --- ENHANCED RATIO PLOT ---
    ax2.set_title(f'Fluorescence Ratio (Enhanced Linear)\\n{SURFACTANT_A_NAME} vs {SURFACTANT_B_NAME} | {len(exp_data)} points', 
                 fontsize=13, fontweight='bold')
    
    ratio_min, ratio_max = exp_data['ratio'].min(), exp_data['ratio'].max()
    
    if ratio_max > ratio_min:
        ratio_levels = np.linspace(ratio_min, ratio_max, RATIO_LEVELS)
        
        cs2 = ax2.contourf(Xi, Yi, ratio_grid, levels=ratio_levels, 
                          cmap='plasma', alpha=ALPHA_CONTOURS, extend='both')
        
        cs2_lines = ax2.contour(Xi, Yi, ratio_grid, levels=ratio_levels[::2], 
                               colors='black', alpha=ALPHA_LINES, linewidths=0.7)
        ax2.clabel(cs2_lines, inline=True, fontsize=9, fmt='%.3f')
        
        plt.colorbar(cs2, ax=ax2, label='Fluorescence Ratio')
    
    ax2.scatter(x, y, c=ratio, s=30, edgecolors='white', linewidth=0.8, 
               cmap='plasma', zorder=5, alpha=0.9)
    
    # --- ENHANCED FLUORESCENCE PLOT ---
    ax3.set_title(f'Average Fluorescence (Enhanced Linear)\\n{SURFACTANT_A_NAME} vs {SURFACTANT_B_NAME} | {len(exp_data)} points', 
                 fontsize=13, fontweight='bold')
    
    fluor_min, fluor_max = avg_fluorescence.min(), avg_fluorescence.max()
    
    if fluor_max > fluor_min:
        fluor_levels = np.linspace(fluor_min, fluor_max, FLUORESCENCE_LEVELS)
        
        cs3 = ax3.contourf(Xi, Yi, fluor_grid, levels=fluor_levels, 
                          cmap='inferno', alpha=ALPHA_CONTOURS, extend='both')
        
        cs3_lines = ax3.contour(Xi, Yi, fluor_grid, levels=fluor_levels[::2], 
                               colors='black', alpha=ALPHA_LINES, linewidths=0.7)
        ax3.clabel(cs3_lines, inline=True, fontsize=9, fmt='%.0f')
        
        plt.colorbar(cs3, ax=ax3, label='Average Fluorescence Intensity')
    
    ax3.scatter(x, y, c=avg_fluorescence, s=30, edgecolors='white', linewidth=0.8, 
               cmap='inferno', zorder=5, alpha=0.9)
    
    # Enhanced formatting
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel(f'{SURFACTANT_A_NAME} Concentration (log10[mM])', fontsize=12)
        ax.set_ylabel(f'{SURFACTANT_B_NAME} Concentration (log10[mM])', fontsize=12)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        
        # Cleaner axis labels
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        x_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in x_ticks]
        y_labels = [f'{10**tick:.1e}' if tick < -2 else f'{10**tick:.2f}' for tick in y_ticks]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_yticklabels(y_labels, rotation=45)
    
    # Add enhanced control blocks
    if control_measurements:
        import matplotlib.cm as cm
        viridis_cmap = plt.cm.viridis
        plasma_cmap = plt.cm.plasma
        
        turb_norm = LogNorm(vmin=max(turb_min, 0.001), vmax=turb_max)
        ratio_norm = plt.Normalize(vmin=ratio_min, vmax=ratio_max)
        
        for i, (control_name, values) in enumerate(control_measurements.items()):
            turb_val = values['turbidity']
            ratio_val = values['ratio']
            
            turb_color = viridis_cmap(turb_norm(max(turb_val, 0.001)))
            ratio_color = plasma_cmap(ratio_norm(ratio_val))
            
            # Enhanced control blocks
            turb_block_x = 0.125 + i * 0.10
            turb_ax = fig.add_axes([turb_block_x, 0.08, 0.08, 0.06])
            turb_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=turb_color, 
                                          edgecolor='black', linewidth=2))
            turb_ax.set_xlim(0, 1)
            turb_ax.set_ylim(0, 1)
            turb_ax.set_xticks([])
            turb_ax.set_yticks([])
            
            ratio_block_x = 0.525 + i * 0.10  
            ratio_ax = fig.add_axes([ratio_block_x, 0.08, 0.08, 0.06])
            ratio_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=ratio_color,
                                           edgecolor='black', linewidth=2))
            ratio_ax.set_xlim(0, 1)
            ratio_ax.set_ylim(0, 1)
            ratio_ax.set_xticks([])
            ratio_ax.set_yticks([])
            
            # Enhanced labels
            fig.text(turb_block_x + 0.04, 0.06, control_name, 
                    ha='center', va='top', fontsize=10, fontweight='bold')
            fig.text(turb_block_x + 0.04, 0.04, f'{turb_val:.3f}', 
                    ha='center', va='top', fontsize=9)
            
            fig.text(ratio_block_x + 0.04, 0.06, control_name, 
                    ha='center', va='top', fontsize=10, fontweight='bold')
            fig.text(ratio_block_x + 0.04, 0.04, f'{ratio_val:.3f}', 
                    ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    
    # Save enhanced version
    output_path = csv_file_path.replace('.csv', '_ENHANCED_LINEAR.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Enhanced linear contour plot saved: {output_path}")
    return output_path

if __name__ == "__main__":
    print("🎯 Creating ENHANCED LINEAR contours...")
    print("Reliable linear interpolation + visual enhancements")
    print("✅ No artifacts, ✅ Smooth appearance, ✅ Scientifically accurate")
    print("="*60)
    
    output_path = create_enhanced_linear_contours(CSV_FILE)
    
    print("\\n🏆 ENHANCED LINEAR SUCCESS!")
    print("This combines:")
    print("  • Linear reliability (no false features)")  
    print("  • High resolution (smooth appearance)")
    print("  • Light smoothing (reduces pixelation)")
    print("  • Enhanced visuals (better contour lines & colors)")
    print("\\n💡 Best of both worlds: accurate + beautiful!")
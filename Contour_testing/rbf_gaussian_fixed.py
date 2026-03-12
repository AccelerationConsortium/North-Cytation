"""
RBF Gaussian Interpolation - Fixed Version
Create ultra-smooth contours using Gaussian radial basis functions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

# === SETTINGS ===
CSV_FILE = r"C:\Users\Owen\Documents\GitHub\North-Cytation\New folder\iterative_experiment_results.csv"
SURFACTANT_A_NAME = "SDS"
SURFACTANT_B_NAME = "DTAB"

# Detailed levels
TURBIDITY_LEVELS = 8
RATIO_LEVELS = 10  
FLUORESCENCE_LEVELS = 8
GRID_RESOLUTION = 100

def create_rbf_gaussian_contours(csv_file_path):
    """
    Create ultra-smooth contours using RBF Gaussian interpolation
    """
    # Read data
    df = pd.read_csv(csv_file_path)
    exp_data = df[df['well_type'] == 'experiment'].copy()
    
    print(f"Creating RBF Gaussian interpolation for {len(exp_data)} data points...")
    
    # Extract data
    x = np.log10(exp_data['surf_A_conc_mm'])
    y = np.log10(exp_data['surf_B_conc_mm']) 
    turbidity = exp_data['turbidity_600']
    ratio = exp_data['ratio']
    
    # Create grid
    xi = np.linspace(x.min(), x.max(), GRID_RESOLUTION)
    yi = np.linspace(y.min(), y.max(), GRID_RESOLUTION)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_points = np.column_stack([Xi.ravel(), Yi.ravel()])
    data_points = np.column_stack([x.values, y.values])
    
    # Calculate appropriate epsilon for Gaussian RBF
    # Use median distance between points as a starting point
    from scipy.spatial.distance import pdist
    distances = pdist(data_points)
    epsilon = np.median(distances) * 0.5  # Scale factor for smoothness
    
    print(f"Using epsilon = {epsilon:.4f} for Gaussian RBF")
    
    try:
        # RBF with Gaussian kernel (fixed with epsilon parameter)
        rbf_turb = RBFInterpolator(
            data_points, 
            turbidity, 
            kernel='gaussian',
            epsilon=epsilon,
            smoothing=0.02  # Small amount of smoothing
        )
        
        rbf_ratio = RBFInterpolator(
            data_points, 
            ratio, 
            kernel='gaussian',
            epsilon=epsilon,
            smoothing=0.02
        )
        
        print("Interpolating surfaces with Gaussian RBF...")
        turb_grid = rbf_turb(grid_points).reshape(Xi.shape)
        ratio_grid = rbf_ratio(grid_points).reshape(Xi.shape)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Turbidity plot
        ax1.set_title(f'Turbidity - RBF Gaussian (Ultra Smooth)\\nEpsilon={epsilon:.4f}, {len(exp_data)} points', 
                     fontsize=13, fontweight='bold')
        
        turb_min, turb_max = exp_data['turbidity_600'].min(), exp_data['turbidity_600'].max()
        turb_min_adj = max(turb_min, 0.001)
        
        if turb_max > turb_min_adj:
            turb_levels = np.logspace(np.log10(turb_min_adj), np.log10(turb_max), TURBIDITY_LEVELS)
            turb_grid_safe = np.maximum(turb_grid, turb_min_adj * 0.1)
            
            cs1 = ax1.contourf(Xi, Yi, turb_grid_safe, levels=turb_levels, 
                              norm=LogNorm(), cmap='viridis', alpha=0.85)
            
            # Smooth contour lines
            cs1_lines = ax1.contour(Xi, Yi, turb_grid_safe, levels=turb_levels[::2], 
                                   colors='black', alpha=0.3, linewidths=0.6)
            ax1.clabel(cs1_lines, inline=True, fontsize=8, fmt='%.3f')
            
            plt.colorbar(cs1, ax=ax1, label='Turbidity (600nm)')
        
        ax1.scatter(x, y, c=turbidity, s=25, edgecolors='white', linewidth=0.6, cmap='viridis', zorder=5)
        
        # Ratio plot
        ax2.set_title(f'Fluorescence Ratio - RBF Gaussian (Ultra Smooth)\\nEpsilon={epsilon:.4f}, {len(exp_data)} points', 
                     fontsize=13, fontweight='bold')
        
        ratio_min, ratio_max = exp_data['ratio'].min(), exp_data['ratio'].max()
        
        if ratio_max > ratio_min:
            ratio_levels = np.linspace(ratio_min, ratio_max, RATIO_LEVELS)
            
            cs2 = ax2.contourf(Xi, Yi, ratio_grid, levels=ratio_levels, 
                              cmap='plasma', alpha=0.85)
            
            # Smooth contour lines
            cs2_lines = ax2.contour(Xi, Yi, ratio_grid, levels=ratio_levels[::2], 
                                   colors='black', alpha=0.3, linewidths=0.6) 
            ax2.clabel(cs2_lines, inline=True, fontsize=8, fmt='%.3f')
            
            plt.colorbar(cs2, ax=ax2, label='Fluorescence Ratio')
        
        ax2.scatter(x, y, c=ratio, s=25, edgecolors='white', linewidth=0.6, cmap='plasma', zorder=5)
        
        # Format axes
        for ax in [ax1, ax2]:
            ax.set_xlabel(f'{SURFACTANT_A_NAME} Concentration (log10[mM])', fontsize=11)
            ax.set_ylabel(f'{SURFACTANT_B_NAME} Concentration (log10[mM])', fontsize=11)
            ax.grid(True, alpha=0.2)
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save
        output_path = csv_file_path.replace('.csv', '_METHOD_RBF_GAUSSIAN_FIXED.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ RBF Gaussian contour plot saved: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ RBF Gaussian failed: {e}")
        return None

if __name__ == "__main__":
    print("🌊 Creating ULTRA-SMOOTH contours with RBF Gaussian...")
    print("This method creates the smoothest possible surfaces")
    print("="*50)
    
    output_path = create_rbf_gaussian_contours(CSV_FILE)
    
    if output_path:
        print("\\n🎯 RBF GAUSSIAN SUCCESS!")
        print("This should be the smoothest interpolation method available")
        print("Perfect for creating publication-quality smooth contours")
    else:
        print("\\n❌ RBF Gaussian failed - stick with other methods")
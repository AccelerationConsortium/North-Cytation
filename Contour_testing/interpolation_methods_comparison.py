"""
Multiple Interpolation Methods Comparison
Test different interpolation approaches to find the best one for your data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial import distance_matrix
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

# === SETTINGS ===
CSV_FILE = r"C:\Users\Owen\Documents\GitHub\North-Cytation\New folder\iterative_experiment_results.csv"
SURFACTANT_A_NAME = "SDS"
SURFACTANT_B_NAME = "DTAB"

# Detailed levels (since you liked them)
TURBIDITY_LEVELS = 8
RATIO_LEVELS = 10  
FLUORESCENCE_LEVELS = 8
GRID_RESOLUTION = 100

def create_interpolation_comparison(csv_file_path):
    """
    Compare different interpolation methods for contour mapping
    """
    # Read data
    df = pd.read_csv(csv_file_path)
    exp_data = df[df['well_type'] == 'experiment'].copy()
    
    print(f"Testing {len(exp_data)} data points with different interpolation methods...")
    
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
    
    # Test different interpolation methods
    methods = {
        'linear': {
            'name': 'Linear (Stable)',
            'description': 'Most stable, no artifacts'
        },
        'nearest': {
            'name': 'Nearest Neighbor',
            'description': 'Blocky but artifact-free'
        },
        'rbf_thin_plate': {
            'name': 'RBF Thin Plate',
            'description': 'Smooth, good for scattered data'
        },
        'rbf_gaussian': {
            'name': 'RBF Gaussian',
            'description': 'Very smooth, natural looking'
        },
        'inverse_distance': {
            'name': 'Inverse Distance',
            'description': 'Weighted by distance, smooth'
        }
    }
    
    results = {}
    
    for method_key, method_info in methods.items():
        print(f"\\n🔄 Testing: {method_info['name']} - {method_info['description']}")
        
        try:
            if method_key == 'linear':
                turb_grid = griddata(data_points, turbidity, (Xi, Yi), method='linear')
                ratio_grid = griddata(data_points, ratio, (Xi, Yi), method='linear')
                
                # Fill NaN with nearest
                turb_nearest = griddata(data_points, turbidity, (Xi, Yi), method='nearest')
                ratio_nearest = griddata(data_points, ratio, (Xi, Yi), method='nearest')
                turb_grid = np.where(np.isnan(turb_grid), turb_nearest, turb_grid)  
                ratio_grid = np.where(np.isnan(ratio_grid), ratio_nearest, ratio_grid)
                
            elif method_key == 'nearest':
                turb_grid = griddata(data_points, turbidity, (Xi, Yi), method='nearest')
                ratio_grid = griddata(data_points, ratio, (Xi, Yi), method='nearest')
                
            elif method_key == 'rbf_thin_plate':
                # RBF with thin plate spline kernel
                rbf_turb = RBFInterpolator(data_points, turbidity, kernel='thin_plate_spline', smoothing=0.1)
                rbf_ratio = RBFInterpolator(data_points, ratio, kernel='thin_plate_spline', smoothing=0.1)
                
                turb_grid = rbf_turb(grid_points).reshape(Xi.shape)
                ratio_grid = rbf_ratio(grid_points).reshape(Xi.shape)
                
            elif method_key == 'rbf_gaussian':
                # RBF with Gaussian kernel
                rbf_turb = RBFInterpolator(data_points, turbidity, kernel='gaussian', smoothing=0.05)
                rbf_ratio = RBFInterpolator(data_points, ratio, kernel='gaussian', smoothing=0.05)
                
                turb_grid = rbf_turb(grid_points).reshape(Xi.shape)
                ratio_grid = rbf_ratio(grid_points).reshape(Xi.shape)
                
            elif method_key == 'inverse_distance':
                # Inverse distance weighting
                turb_grid = inverse_distance_weighting(data_points, turbidity.values, grid_points, power=2).reshape(Xi.shape)
                ratio_grid = inverse_distance_weighting(data_points, ratio.values, grid_points, power=2).reshape(Xi.shape)
            
            # Store results
            results[method_key] = {
                'info': method_info,
                'turbidity_grid': turb_grid,
                'ratio_grid': ratio_grid,
                'success': True
            }
            print(f"   ✅ Success")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:50]}...")
            results[method_key] = {'success': False, 'info': method_info}
    
    return results, exp_data, (x, y, Xi, Yi, turbidity, ratio)

def inverse_distance_weighting(data_points, values, grid_points, power=2):
    """
    Custom inverse distance weighting interpolation
    """
    distances = distance_matrix(grid_points, data_points)
    
    # Avoid division by zero for points that are exactly on data points
    distances = np.maximum(distances, 1e-10)
    
    # Calculate weights (inverse distance to the power)
    weights = 1.0 / (distances ** power)
    
    # Normalize weights
    weights_sum = np.sum(weights, axis=1)
    weights_normalized = weights / weights_sum[:, np.newaxis]
    
    # Calculate interpolated values
    interpolated = np.sum(weights_normalized * values[np.newaxis, :], axis=1)
    
    return interpolated

def create_method_visualization(method_key, method_data, exp_data, coords, output_suffix=""):
    """
    Create visualization for a specific interpolation method
    """
    x, y, Xi, Yi, turbidity, ratio = coords
    turb_grid = method_data['turbidity_grid']
    ratio_grid = method_data['ratio_grid']
    method_info = method_data['info']
    
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Turbidity plot
    ax1.set_title(f'Turbidity - {method_info["name"]}\\n{method_info["description"]}', 
                 fontsize=13, fontweight='bold')
    
    turb_min, turb_max = exp_data['turbidity_600'].min(), exp_data['turbidity_600'].max()
    turb_min_adj = max(turb_min, 0.001)
    
    if turb_max > turb_min_adj:
        turb_levels = np.logspace(np.log10(turb_min_adj), np.log10(turb_max), TURBIDITY_LEVELS)
        turb_grid_safe = np.maximum(turb_grid, turb_min_adj * 0.1)
        
        cs1 = ax1.contourf(Xi, Yi, turb_grid_safe, levels=turb_levels, 
                          norm=LogNorm(), cmap='viridis', alpha=0.8)
        plt.colorbar(cs1, ax=ax1, label='Turbidity (600nm)')
    else:
        im1 = ax1.imshow(turb_grid, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(im1, ax=ax1, label='Turbidity (600nm)')
    
    ax1.scatter(x, y, c=turbidity, s=20, edgecolors='white', linewidth=0.5, cmap='viridis', zorder=5)
    
    # Ratio plot
    ax2.set_title(f'Fluorescence Ratio - {method_info["name"]}\\n{method_info["description"]}', 
                 fontsize=13, fontweight='bold')
    
    ratio_min, ratio_max = exp_data['ratio'].min(), exp_data['ratio'].max()
    
    if ratio_max > ratio_min:
        ratio_levels = np.linspace(ratio_min, ratio_max, RATIO_LEVELS)
        
        cs2 = ax2.contourf(Xi, Yi, ratio_grid, levels=ratio_levels, 
                          cmap='plasma', alpha=0.8)
        plt.colorbar(cs2, ax=ax2, label='Fluorescence Ratio')
    else:
        im2 = ax2.imshow(ratio_grid, aspect='auto', cmap='plasma', origin='lower')
        plt.colorbar(im2, ax=ax2, label='Fluorescence Ratio')
    
    ax2.scatter(x, y, c=ratio, s=20, edgecolors='white', linewidth=0.5, cmap='plasma', zorder=5)
    
    # Format axes
    for ax in [ax1, ax2]:
        ax.set_xlabel(f'{SURFACTANT_A_NAME} Concentration (log10[mM])', fontsize=11)
        ax.set_ylabel(f'{SURFACTANT_B_NAME} Concentration (log10[mM])', fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save
    output_path = CSV_FILE.replace('.csv', f'_METHOD_{method_key.upper()}{output_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

if __name__ == "__main__":
    print("🧪 TESTING MULTIPLE INTERPOLATION METHODS...")
    print("This will try 5 different approaches to find the best one")
    print("="*60)
    
    # Run comparison
    results, exp_data, coords = create_interpolation_comparison(CSV_FILE)
    
    # Create visualizations for successful methods
    successful_methods = []
    
    print("\\n📊 GENERATING VISUALIZATIONS...")
    for method_key, method_data in results.items():
        if method_data['success']:
            print(f"\\n🎨 Creating plots for: {method_data['info']['name']}")
            
            try:
                output_path = create_method_visualization(method_key, method_data, exp_data, coords)
                successful_methods.append({
                    'method': method_key,
                    'name': method_data['info']['name'],
                    'description': method_data['info']['description'],
                    'file': output_path
                })
                print(f"   ✅ Saved: {output_path.split('/')[-1]}")
                
            except Exception as e:
                print(f"   ❌ Visualization failed: {str(e)[:50]}...")
    
    # Summary
    print("\\n" + "="*60)
    print("🏆 INTERPOLATION METHOD COMPARISON COMPLETE!")
    print("="*60)
    
    for method in successful_methods:
        print(f"✅ {method['name']}: {method['description']}")
    
    print("\\n💡 RECOMMENDATIONS:")
    print("• LINEAR: Most stable, no artifacts, good for scientific data")
    print("• RBF GAUSSIAN: Smoothest appearance, natural gradients")  
    print("• INVERSE DISTANCE: Good balance of smooth + stable")
    print("• Compare all versions to see which looks best for your data!")
    
    print(f"\\n📁 All files saved in: {'/'.join(CSV_FILE.split('/')[:-1])}")
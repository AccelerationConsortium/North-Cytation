"""
Gradient Magnitude Sampling for Boundary Detection
Calculates gradients of normalized turbidity and ratio at each grid point
and selects points with highest gradient magnitudes for sampling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os

def load_and_prepare_data(csv_file_path):
    """Load experimental data and prepare for gradient analysis."""
    df = pd.read_csv(csv_file_path)
    experiment_data = df[df['well_type'] == 'experiment'].copy()
    
    print(f"Loaded {len(experiment_data)} experimental data points")
    print(f"Turbidity range: {experiment_data['turbidity_600'].min():.4f} - {experiment_data['turbidity_600'].max():.3f}")
    print(f"Ratio range: {experiment_data['ratio'].min():.4f} - {experiment_data['ratio'].max():.4f}")
    
    return experiment_data

def normalize_fields(experiment_data):
    """
    Normalize turbidity and ratio to comparable scales:
    - Log turbidity + z-score
    - Z-score ratio
    """
    
    # Log transform turbidity
    epsilon = 1e-6
    log_turbidity = np.log10(experiment_data['turbidity_600'] + epsilon)
    
    # Z-score both fields
    scaler_turb = StandardScaler()
    turb_normalized = scaler_turb.fit_transform(log_turbidity.values.reshape(-1, 1)).flatten()
    
    scaler_ratio = StandardScaler()
    ratio_normalized = scaler_ratio.fit_transform(experiment_data['ratio'].values.reshape(-1, 1)).flatten()
    
    print(f"\nNormalization completed:")
    print(f"Log turbidity: {log_turbidity.min():.3f} - {log_turbidity.max():.3f} → z-score: {turb_normalized.min():.3f} - {turb_normalized.max():.3f}")
    print(f"Ratio: {experiment_data['ratio'].min():.4f} - {experiment_data['ratio'].max():.4f} → z-score: {ratio_normalized.min():.3f} - {ratio_normalized.max():.3f}")
    
    # Add normalized values and log concentrations
    experiment_data = experiment_data.copy()
    experiment_data['log_turbidity'] = log_turbidity
    experiment_data['turb_normalized'] = turb_normalized
    experiment_data['ratio_normalized'] = ratio_normalized
    experiment_data['log_sds'] = np.log10(experiment_data['surf_A_conc_mm'])
    experiment_data['log_ttab'] = np.log10(experiment_data['surf_B_conc_mm'])
    
    return experiment_data

def create_grid_structure(experiment_data):
    """Create 2D grid structure with interpolated fields."""
    
    # Get unique concentrations
    sds_concs = sorted(experiment_data['surf_A_conc_mm'].unique())
    ttab_concs = sorted(experiment_data['surf_B_conc_mm'].unique())
    
    print(f"\nGrid structure: {len(sds_concs)} SDS × {len(ttab_concs)} TTAB = {len(sds_concs) * len(ttab_concs)} grid positions")
    
    n_sds = len(sds_concs)
    n_ttab = len(ttab_concs)
    
    # Create 2D grids
    turb_grid = np.zeros((n_sds, n_ttab))
    ratio_grid = np.zeros((n_sds, n_ttab))
    log_sds_grid = np.zeros((n_sds, n_ttab))
    log_ttab_grid = np.zeros((n_sds, n_ttab))
    
    # Fill grids with data
    for _, row in experiment_data.iterrows():
        sds_idx = sds_concs.index(row['surf_A_conc_mm'])
        ttab_idx = ttab_concs.index(row['surf_B_conc_mm'])
        
        turb_grid[sds_idx, ttab_idx] = row['turb_normalized']
        ratio_grid[sds_idx, ttab_idx] = row['ratio_normalized']
        log_sds_grid[sds_idx, ttab_idx] = row['log_sds']
        log_ttab_grid[sds_idx, ttab_idx] = row['log_ttab']
    
    return {
        'turb_grid': turb_grid,
        'ratio_grid': ratio_grid,
        'log_sds_grid': log_sds_grid,
        'log_ttab_grid': log_ttab_grid,
        'sds_concs': sds_concs,
        'ttab_concs': ttab_concs
    }

def calculate_gradients(grid_data):
    """Calculate gradients for turbidity and ratio fields."""
    
    turb_grid = grid_data['turb_grid']
    ratio_grid = grid_data['ratio_grid']
    log_sds_grid = grid_data['log_sds_grid']
    log_ttab_grid = grid_data['log_ttab_grid']
    
    n_sds, n_ttab = turb_grid.shape
    
    # Calculate gradients using numpy gradient (central differences where possible)
    # Gradient returns [gradient_along_axis0, gradient_along_axis1]
    turb_grad = np.gradient(turb_grid)
    ratio_grad = np.gradient(ratio_grid)
    
    # Calculate spacing for proper gradient scaling
    # Use log concentrations for proper spacing
    log_sds_spacing = np.gradient(log_sds_grid, axis=0)
    log_ttab_spacing = np.gradient(log_ttab_grid, axis=1)
    
    # Scale gradients by spacing (convert to derivatives w.r.t log concentration)
    turb_grad_sds = np.divide(turb_grad[0], log_sds_spacing, 
                              out=np.zeros_like(turb_grad[0]), where=log_sds_spacing!=0)
    turb_grad_ttab = np.divide(turb_grad[1], log_ttab_spacing, 
                               out=np.zeros_like(turb_grad[1]), where=log_ttab_spacing!=0)
    
    ratio_grad_sds = np.divide(ratio_grad[0], log_sds_spacing, 
                               out=np.zeros_like(ratio_grad[0]), where=log_sds_spacing!=0)
    ratio_grad_ttab = np.divide(ratio_grad[1], log_ttab_spacing, 
                                out=np.zeros_like(ratio_grad[1]), where=log_ttab_spacing!=0)
    
    print(f"\nGradient calculation completed:")
    print(f"Turbidity gradient SDS range: {turb_grad_sds.min():.3f} - {turb_grad_sds.max():.3f}")
    print(f"Turbidity gradient TTAB range: {turb_grad_ttab.min():.3f} - {turb_grad_ttab.max():.3f}")
    print(f"Ratio gradient SDS range: {ratio_grad_sds.min():.3f} - {ratio_grad_sds.max():.3f}")
    print(f"Ratio gradient TTAB range: {ratio_grad_ttab.min():.3f} - {ratio_grad_ttab.max():.3f}")
    
    return {
        'turb_grad_sds': turb_grad_sds,
        'turb_grad_ttab': turb_grad_ttab,
        'ratio_grad_sds': ratio_grad_sds,
        'ratio_grad_ttab': ratio_grad_ttab
    }

def calculate_gradient_magnitudes(gradients):
    """Calculate gradient magnitude for each measurement and combined."""
    
    # Turbidity gradient magnitude
    turb_mag = np.sqrt(gradients['turb_grad_sds']**2 + gradients['turb_grad_ttab']**2)
    
    # Ratio gradient magnitude
    ratio_mag = np.sqrt(gradients['ratio_grad_sds']**2 + gradients['ratio_grad_ttab']**2)
    
    # Combined gradient magnitude (treating as vector field)
    combined_mag = np.sqrt(turb_mag**2 + ratio_mag**2)
    
    print(f"\nGradient magnitudes:")
    print(f"Turbidity magnitude range: {turb_mag.min():.3f} - {turb_mag.max():.3f}")
    print(f"Ratio magnitude range: {ratio_mag.min():.3f} - {ratio_mag.max():.3f}")
    print(f"Combined magnitude range: {combined_mag.min():.3f} - {combined_mag.max():.3f}")
    
    return {
        'turb_magnitude': turb_mag,
        'ratio_magnitude': ratio_mag,
        'combined_magnitude': combined_mag
    }

def select_sampling_points(grid_data, magnitudes, n_points=32, min_spacing=0.335):
    """Select sampling points based on highest gradient magnitudes with spacing enforcement."""
    
    combined_mag = magnitudes['combined_magnitude']
    log_sds_grid = grid_data['log_sds_grid']
    log_ttab_grid = grid_data['log_ttab_grid']
    
    n_sds, n_ttab = combined_mag.shape
    
    # Create list of all grid points with their scores
    candidates = []
    for i in range(n_sds):
        for j in range(n_ttab):
            candidates.append({
                'sds_idx': i,
                'ttab_idx': j,
                'log_sds': log_sds_grid[i, j],
                'log_ttab': log_ttab_grid[i, j],
                'score': combined_mag[i, j]
            })
    
    # Sort by gradient magnitude (highest first)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\nSelecting {n_points} points with minimum spacing {min_spacing:.3f} log units:")
    print(f"Top 5 gradient scores: {[c['score'] for c in candidates[:5]]}")
    
    # Select points with spacing enforcement
    selected_points = []
    
    for candidate in candidates:
        # Check spacing constraint
        if len(selected_points) == 0:
            # First point
            selected_points.append(candidate)
            continue
        
        # Calculate distances to all selected points
        candidate_pos = np.array([[candidate['log_sds'], candidate['log_ttab']]])
        selected_pos = np.array([[p['log_sds'], p['log_ttab']] for p in selected_points])
        
        distances = cdist(candidate_pos, selected_pos, 'euclidean')[0]
        min_distance = distances.min()
        
        if min_distance >= min_spacing:
            selected_points.append(candidate)
            
            if len(selected_points) >= n_points:
                break
    
    print(f"Selected {len(selected_points)} points")
    print(f"Score range: {selected_points[0]['score']:.3f} - {selected_points[-1]['score']:.3f}")
    
    return selected_points

def create_visualization(grid_data, magnitudes, selected_points, output_path):
    """Create visualization of gradient magnitudes and selected sampling points."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    log_sds_grid = grid_data['log_sds_grid']
    log_ttab_grid = grid_data['log_ttab_grid']
    
    # Plot turbidity gradient magnitude
    im1 = axes[0, 0].imshow(magnitudes['turb_magnitude'].T, origin='lower', 
                           extent=[log_sds_grid.min(), log_sds_grid.max(), 
                                  log_ttab_grid.min(), log_ttab_grid.max()],
                           cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Turbidity Gradient Magnitude')
    axes[0, 0].set_xlabel('Log SDS Concentration')
    axes[0, 0].set_ylabel('Log TTAB Concentration')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot ratio gradient magnitude
    im2 = axes[0, 1].imshow(magnitudes['ratio_magnitude'].T, origin='lower',
                           extent=[log_sds_grid.min(), log_sds_grid.max(), 
                                  log_ttab_grid.min(), log_ttab_grid.max()],
                           cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Ratio Gradient Magnitude')
    axes[0, 1].set_xlabel('Log SDS Concentration')
    axes[0, 1].set_ylabel('Log TTAB Concentration')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot combined gradient magnitude
    im3 = axes[1, 0].imshow(magnitudes['combined_magnitude'].T, origin='lower',
                           extent=[log_sds_grid.min(), log_sds_grid.max(), 
                                  log_ttab_grid.min(), log_ttab_grid.max()],
                           cmap='viridis', aspect='auto')
    axes[1, 0].set_title('Combined Gradient Magnitude')
    axes[1, 0].set_xlabel('Log SDS Concentration')
    axes[1, 0].set_ylabel('Log TTAB Concentration')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot selected sampling points
    axes[1, 1].imshow(magnitudes['combined_magnitude'].T, origin='lower',
                     extent=[log_sds_grid.min(), log_sds_grid.max(), 
                            log_ttab_grid.min(), log_ttab_grid.max()],
                     cmap='viridis', aspect='auto', alpha=0.7)
    
    # Add selected points
    selected_sds = [p['log_sds'] for p in selected_points]
    selected_ttab = [p['log_ttab'] for p in selected_points]
    scatter = axes[1, 1].scatter(selected_sds, selected_ttab, 
                               c=[p['score'] for p in selected_points],
                               cmap='Reds', s=100, edgecolors='white', linewidth=2)
    axes[1, 1].set_title(f'Selected {len(selected_points)} Sampling Points')
    axes[1, 1].set_xlabel('Log SDS Concentration')
    axes[1, 1].set_ylabel('Log TTAB Concentration')
    plt.colorbar(scatter, ax=axes[1, 1], label='Gradient Score')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()
    return fig

def main():
    """Main execution function."""
    
    # Define paths - use the correct data file with turbidity and ratio measurements
    data_file = r"c:\Users\owenm\OneDrive\Desktop\North Robotics\utoronto_demo\utoronto_demo\surfactant_grid_SDS_TTAB_20260209_164920\surfactant_grid_SDS_TTAB_20260209_164920\complete_experiment_results.csv"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = data_file
    output_path = os.path.join(current_dir, "gradient_magnitude_sampling.png")
    
    print("=== Gradient Magnitude Sampling Analysis ===")
    print(f"Loading data from: {csv_path}")
    
    # Load and prepare data
    experiment_data = load_and_prepare_data(csv_path)
    experiment_data = normalize_fields(experiment_data)
    
    # Create grid structure
    grid_data = create_grid_structure(experiment_data)
    
    # Calculate gradients
    gradients = calculate_gradients(grid_data)
    
    # Calculate gradient magnitudes
    magnitudes = calculate_gradient_magnitudes(gradients)
    
    # Select sampling points
    selected_points = select_sampling_points(grid_data, magnitudes, n_points=32, min_spacing=0.335)
    
    # Create visualization
    fig = create_visualization(grid_data, magnitudes, selected_points, output_path)
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Selected {len(selected_points)} sampling points")
    print(f"Gradient score range: {selected_points[0]['score']:.3f} - {selected_points[-1]['score']:.3f}")
    print(f"Visualization saved to: gradient_magnitude_sampling.png")
    
    # Show top 10 recommendations
    print(f"\nTop 10 recommended sampling points:")
    print("Rank | SDS_idx | TTAB_idx | Log_SDS | Log_TTAB | Score")
    print("-" * 60)
    for i, point in enumerate(selected_points[:10], 1):
        print(f"{i:4d} | {point['sds_idx']:7d} | {point['ttab_idx']:8d} | {point['log_sds']:7.3f} | {point['log_ttab']:8.3f} | {point['score']:5.3f}")

if __name__ == "__main__":
    main()
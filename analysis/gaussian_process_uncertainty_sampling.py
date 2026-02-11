"""
Gaussian Process Gradient-Based Sampling
Model ratio = f(SDS, TTAB) and sample points where GP gradient magnitude is highest.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
import os

def load_experiment_data(csv_file_path):
    """Load the existing experimental data."""
    df = pd.read_csv(csv_file_path)
    experiment_data = df[df['well_type'] == 'experiment'].copy()
    
    # Convert to log scale for modeling (GP works better with log concentrations)
    experiment_data['log_sds'] = np.log10(experiment_data['surf_A_conc_mm'])
    experiment_data['log_ttab'] = np.log10(experiment_data['surf_B_conc_mm'])
    
    return experiment_data

def train_gaussian_process(experiment_data, target_variable='turbidity_600'):
    """
    Train a Gaussian Process to model target = f(log_SDS, log_TTAB).
    """
    
    # Prepare training data
    X_train = experiment_data[['log_sds', 'log_ttab']].values
    y_train = experiment_data[target_variable].values
    
    print(f"Training GP on {len(X_train)} data points...")
    print(f"SDS range: {np.exp(X_train[:, 0]).min():.4f} - {np.exp(X_train[:, 0]).max():.1f} mM")
    print(f"TTAB range: {np.exp(X_train[:, 1]).min():.4f} - {np.exp(X_train[:, 1]).max():.1f} mM")
    print(f"Ratio range: {y_train.min():.4f} - {y_train.max():.4f}")
    
    # Standardize features for better GP performance
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # Define GP kernel
    # RBF kernel for smooth function modeling + White kernel for noise
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
    
    # Train Gaussian Process
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,  # Small regularization
        n_restarts_optimizer=5,  # Multiple random starts for optimization
        normalize_y=False  # We're doing our own normalization
    )
    
    gp.fit(X_train_scaled, y_train_scaled)
    
    print(f"✓ GP trained successfully")
    print(f"✓ Optimized kernel: {gp.kernel_}")
    print(f"✓ Log-marginal-likelihood: {gp.log_marginal_likelihood():.3f}")
    
    return gp, scaler_X, scaler_y, X_train, y_train

def generate_candidate_grid(experiment_data, grid_density=50):
    """
    Generate a dense grid of candidate points for uncertainty evaluation.
    """
    
    # Define concentration ranges (slightly extend beyond existing data)
    log_sds_min = experiment_data['log_sds'].min() - 0.1
    log_sds_max = experiment_data['log_sds'].max() + 0.1
    log_ttab_min = experiment_data['log_ttab'].min() - 0.1
    log_ttab_max = experiment_data['log_ttab'].max() + 0.1
    
    # Create grid
    log_sds_grid = np.linspace(log_sds_min, log_sds_max, grid_density)
    log_ttab_grid = np.linspace(log_ttab_min, log_ttab_max, grid_density)
    
    log_sds_mesh, log_ttab_mesh = np.meshgrid(log_sds_grid, log_ttab_grid)
    
    # Flatten for GP prediction
    candidate_points = np.column_stack([
        log_sds_mesh.ravel(),
        log_ttab_mesh.ravel()
    ])
    
    print(f"Generated {len(candidate_points)} candidate points")
    print(f"SDS range: {np.exp(log_sds_min):.4f} - {np.exp(log_sds_max):.1f} mM")
    print(f"TTAB range: {np.exp(log_ttab_min):.4f} - {np.exp(log_ttab_max):.1f} mM")
    
    return candidate_points, log_sds_grid, log_ttab_grid

def predict_gradient_landscape(gp, scaler_X, scaler_y, candidate_points, grid_shape):
    """
    Use trained GP to predict gradients at all candidate points.
    """
    
    print("Predicting gradient landscape...")
    
    # Scale candidate points
    candidate_points_scaled = scaler_X.transform(candidate_points)
    
    # GP predictions
    y_pred_scaled = gp.predict(candidate_points_scaled, return_std=False)
    
    # Transform back to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate gradients using finite differences on the grid
    y_pred_grid = y_pred.reshape(grid_shape)
    
    # Calculate gradients in both directions
    grad_sds, grad_ttab = np.gradient(y_pred_grid)
    
    # Convert back to log-concentration gradients (accounting for scaling)
    # We need to account for the grid spacing in log space
    log_sds_spacing = (candidate_points[:, 0].max() - candidate_points[:, 0].min()) / (grid_shape[1] - 1)
    log_ttab_spacing = (candidate_points[:, 1].max() - candidate_points[:, 1].min()) / (grid_shape[0] - 1)
    
    grad_sds = grad_sds / log_sds_spacing  # ∂y/∂(log_SDS)
    grad_ttab = grad_ttab / log_ttab_spacing  # ∂y/∂(log_TTAB)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_sds**2 + grad_ttab**2)
    
    # Flatten back to match candidate_points
    grad_sds_flat = grad_sds.ravel()
    grad_ttab_flat = grad_ttab.ravel()
    gradient_magnitude_flat = gradient_magnitude.ravel()
    
    print(f"✓ Gradient predictions completed")
    print(f"✓ Predicted ratio range: {y_pred.min():.4f} - {y_pred.max():.4f}")
    print(f"✓ Gradient SDS range: {grad_sds_flat.min():.4f} - {grad_sds_flat.max():.4f}")
    print(f"✓ Gradient TTAB range: {grad_ttab_flat.min():.4f} - {grad_ttab_flat.max():.4f}")
    print(f"✓ Gradient magnitude range: {gradient_magnitude_flat.min():.4f} - {gradient_magnitude_flat.max():.4f}")
    
    return y_pred, grad_sds_flat, grad_ttab_flat, gradient_magnitude_flat

def select_high_gradient_points(candidate_points, y_pred, gradient_magnitude, 
                              grad_sds, grad_ttab, n_points=96, 
                              exclude_existing=True, experiment_data=None):
    """
    Select points with highest GP gradient magnitude for sampling.
    """
    
    print(f"Selecting top {n_points} high-gradient points...")
    
    if exclude_existing and experiment_data is not None:
        # Remove points too close to existing experiments
        existing_points = experiment_data[['log_sds', 'log_ttab']].values
        
        # Calculate minimum distance to existing points
        from scipy.spatial.distance import cdist
        distances = cdist(candidate_points, existing_points)
        min_distances = np.min(distances, axis=1)
        
        # Exclude points within threshold distance (e.g., 0.1 in log space ≈ 25% concentration difference)
        distance_threshold = 0.05  # Adjust as needed
        valid_candidates = min_distances > distance_threshold
        
        print(f"✓ Excluded {np.sum(~valid_candidates)} points too close to existing experiments")
    else:
        valid_candidates = np.ones(len(candidate_points), dtype=bool)
    
    # Apply validity filter
    valid_points = candidate_points[valid_candidates]
    valid_gradients = gradient_magnitude[valid_candidates]
    valid_grad_sds = grad_sds[valid_candidates]
    valid_grad_ttab = grad_ttab[valid_candidates]
    valid_predictions = y_pred[valid_candidates]
    
    # Sort by gradient magnitude (highest first)
    gradient_order = np.argsort(valid_gradients)[::-1]
    
    # Select top points
    n_select = min(n_points, len(valid_points))
    selected_indices = gradient_order[:n_select]
    
    selected_points = valid_points[selected_indices]
    selected_gradients = valid_gradients[selected_indices]
    selected_grad_sds = valid_grad_sds[selected_indices]
    selected_grad_ttab = valid_grad_ttab[selected_indices]
    selected_predictions = valid_predictions[selected_indices]
    
    # Convert back to concentration space
    selected_sds = np.exp(selected_points[:, 0])
    selected_ttab = np.exp(selected_points[:, 1])
    
    print(f"✓ Selected {len(selected_points)} points")
    print(f"✓ Top gradient magnitude: {selected_gradients[0]:.4f}")
    print(f"✓ Average gradient magnitude: {np.mean(selected_gradients):.4f}")
    
    return {
        'points_log': selected_points,
        'points_conc': np.column_stack([selected_sds, selected_ttab]),
        'gradients': selected_gradients,
        'grad_sds': selected_grad_sds,
        'grad_ttab': selected_grad_ttab,
        'predictions': selected_predictions,
        'sds_conc': selected_sds,
        'ttab_conc': selected_ttab
    }

def visualize_gp_uncertainty_sampling(experiment_data, candidate_points, y_pred, y_std, 
                                    selected_points, log_sds_grid, log_ttab_grid,
                                    output_dir='output/gp_uncertainty_sampling'):
    """
    Visualize the GP uncertainty landscape and selected sampling points.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Reshape predictions for plotting
    grid_shape = (len(log_ttab_grid), len(log_sds_grid))
    y_pred_grid = y_pred.reshape(grid_shape)
    y_std_grid = y_std.reshape(grid_shape)
    
    # Convert grid to concentration space for plotting
    sds_conc_grid = np.exp(log_sds_grid)
    ttab_conc_grid = np.exp(log_ttab_grid)
    
    # Plot 1: Existing data and selected points
    ax1.scatter(experiment_data['surf_A_conc_mm'], experiment_data['surf_B_conc_mm'],
               c=experiment_data['ratio'], s=60, cmap='viridis', alpha=0.8, 
               edgecolors='black', linewidth=1, label='Existing data')
    
    # Plot selected high-gradient points
    sc = ax1.scatter(selected_points['sds_conc'], selected_points['ttab_conc'],
                    c=selected_points['gradients'], s=80, cmap='Reds', alpha=0.9,
                    edgecolors='darkred', linewidth=2, marker='^', 
                    label=f'Selected points (n={len(selected_points["sds_conc"])})')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax1.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax1.set_title('GP Uncertainty-Based Sampling Points', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    cbar1 = plt.colorbar(sc, ax=ax1)
    cbar1.set_label('GP Gradient Magnitude', fontsize=10)
    
    # Plot 2: GP predicted ratio landscape
    X_mesh, Y_mesh = np.meshgrid(sds_conc_grid, ttab_conc_grid)
    im2 = ax2.contourf(X_mesh, Y_mesh, y_pred_grid, levels=20, cmap='viridis', alpha=0.8)
    ax2.contour(X_mesh, Y_mesh, y_pred_grid, levels=10, colors='black', alpha=0.4, linewidths=0.5)
    
    # Overlay existing data
    ax2.scatter(experiment_data['surf_A_conc_mm'], experiment_data['surf_B_conc_mm'],
               c='white', s=40, alpha=0.8, edgecolors='black', linewidth=1)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax2.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax2.set_title('GP Predicted Ratio Landscape', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Predicted Ratio', fontsize=10)
    
    # Plot 3: GP uncertainty landscape  
    im3 = ax3.contourf(X_mesh, Y_mesh, y_std_grid, levels=20, cmap='Reds', alpha=0.8)
    ax3.contour(X_mesh, Y_mesh, y_std_grid, levels=10, colors='black', alpha=0.4, linewidths=0.5)
    
    # Overlay selected points
    ax3.scatter(selected_points['sds_conc'], selected_points['ttab_conc'],
               c='blue', s=60, alpha=1.0, edgecolors='darkblue', linewidth=2, marker='^')
    
    # Overlay existing data
    ax3.scatter(experiment_data['surf_A_conc_mm'], experiment_data['surf_B_conc_mm'],
               c='white', s=40, alpha=0.8, edgecolors='black', linewidth=1)
    
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax3.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax3.set_title('GP Uncertainty Landscape', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Gradient Magnitude', fontsize=10)
    
    # Plot 4: Uncertainty distribution and statistics
    ax4.hist(y_std, bins=50, alpha=0.7, color='red', label=f'All points (n={len(y_std)})')
    ax4.axvline(np.mean(selected_points['gradients']), color='blue', linestyle='--', linewidth=2,
               label=f'Selected avg: {np.mean(selected_points["gradients"]):.4f}')
    ax4.axvline(np.mean(y_std), color='red', linestyle='--', linewidth=2,
               label=f'Overall avg: {np.mean(y_std):.4f}')
    
    ax4.set_xlabel('GP Gradient Magnitude', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Gradient Distribution', fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    output_file = f'{output_dir}/gp_uncertainty_sampling.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\\nVisualization saved: {output_file}")
    
    plt.show()

def save_sampling_recommendations(selected_points, experiment_data, output_dir='output/gp_uncertainty_sampling'):
    """
    Save the recommended sampling points to CSV.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame with recommended points
    recommendations_df = pd.DataFrame({
        'sds_conc_mm': selected_points['sds_conc'],
        'ttab_conc_mm': selected_points['ttab_conc'],
        'gp_uncertainty': selected_points['uncertainties'],
        'gp_prediction': selected_points['predictions'],
        'rank': range(1, len(selected_points['sds_conc']) + 1)
    })
    
    # Round for practical use
    recommendations_df['sds_conc_mm'] = recommendations_df['sds_conc_mm'].round(4)
    recommendations_df['ttab_conc_mm'] = recommendations_df['ttab_conc_mm'].round(4)
    recommendations_df['gp_uncertainty'] = recommendations_df['gp_uncertainty'].round(5)
    recommendations_df['gp_prediction'] = recommendations_df['gp_prediction'].round(5)
    
    output_file = f'{output_dir}/gp_uncertainty_recommendations.csv'
    recommendations_df.to_csv(output_file, index=False)
    
    print(f"\\nRecommendations saved: {output_file}")
    
    # Print top 10 recommendations
    print("\\nTop 10 recommended sampling points:")
    print(recommendations_df.head(10).to_string(index=False))
    
    return recommendations_df

def main():
    """Main function for GP uncertainty-based sampling."""
    
    data_file = r"C:\\Users\\owenm\\OneDrive\\Desktop\\North Robotics\\utoronto_demo\\utoronto_demo\\surfactant_grid_SDS_TTAB_20260209_164920\\surfactant_grid_SDS_TTAB_20260209_164920\\complete_experiment_results.csv"
    
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    print("="*60)
    print("GAUSSIAN PROCESS GRADIENT-BASED SAMPLING")
    print("="*60)
    
    print("\\n1. Loading experimental data...")
    experiment_data = load_experiment_data(data_file)
    
    print("\\n2. Training Gaussian Process...")
    gp, scaler_X, scaler_y, X_train, y_train = train_gaussian_process(experiment_data, target_variable='turbidity_600')
    
    print("\\n3. Generating candidate grid...")
    candidate_points, log_sds_grid, log_ttab_grid = generate_candidate_grid(experiment_data, grid_density=40)
    grid_shape = (len(log_ttab_grid), len(log_sds_grid))
    
    print("\n4. Predicting gradient landscape...")
    y_pred, grad_sds, grad_ttab, gradient_magnitude = predict_gradient_landscape(gp, scaler_X, scaler_y, candidate_points, grid_shape)
    
    print("\n5. Selecting high-gradient points...")
    selected_points = select_high_gradient_points(candidate_points, y_pred, gradient_magnitude,
                                                grad_sds, grad_ttab, n_points=96, 
                                                exclude_existing=True, experiment_data=experiment_data)
    
    print("\\n6. Visualizing results...")
    visualize_gp_uncertainty_sampling(experiment_data, candidate_points, y_pred, gradient_magnitude,
                                     selected_points, log_sds_grid, log_ttab_grid)
    
    print("\\n7. Saving recommendations...")
    recommendations_df = save_sampling_recommendations(selected_points, experiment_data)
    
    print("\\n" + "="*60)
    print("GAUSSIAN PROCESS GRADIENT RESULTS")
    print("="*60)
    print(f"✓ GP trained on {len(experiment_data)} existing experiments")
    print(f"✓ {len(selected_points['sds_conc'])} high-gradient points selected")
    print(f"✓ Average gradient magnitude: {np.mean(selected_points['gradients']):.4f}")
    print(f"✓ Max gradient magnitude: {np.max(selected_points['gradients']):.4f}")
    print(f"✓ SDS range: {np.min(selected_points['sds_conc']):.4f} - {np.max(selected_points['sds_conc']):.2f} mM")
    print(f"✓ TTAB range: {np.min(selected_points['ttab_conc']):.4f} - {np.max(selected_points['ttab_conc']):.2f} mM")
    
    print("\\n✓ Strategy: Sample where GP gradient magnitude is HIGHEST")
    print("✓ Benefit: Target regions of fastest change/maximum information content")
    print("✓ Output: Ranked list of 96 optimal sampling points")

if __name__ == "__main__":
    main()
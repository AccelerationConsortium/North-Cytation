"""
Gradient-Based Rectangle Boundary Optimization
Define 3 rectangles and calculate optimal x-y bounds based on gradient information.
High gradient regions get tighter bounds, low gradient regions get wider bounds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

def load_and_classify_data(csv_file_path):
    """Load data and apply baseline classification."""
    df = pd.read_csv(csv_file_path)
    experiment_data = df[df['well_type'] == 'experiment'].copy()
    
    # Apply AND logic classification 
    turbidity_threshold = 0.1
    turbidity_baseline = experiment_data['turbidity_600'] <= turbidity_threshold
    
    # K-means on ratio
    ratio_values = experiment_data['ratio'].values.reshape(-1, 1)
    scaler = StandardScaler()
    ratio_scaled = scaler.fit_transform(ratio_values)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    ratio_clusters = kmeans.fit_predict(ratio_scaled)
    
    cluster_0_mean = experiment_data[ratio_clusters == 0]['ratio'].mean()
    cluster_1_mean = experiment_data[ratio_clusters == 1]['ratio'].mean()
    baseline_cluster = 0 if cluster_0_mean > cluster_1_mean else 1
    ratio_baseline = (ratio_clusters == baseline_cluster)
    
    # Combined AND classification
    combined_baseline = turbidity_baseline & ratio_baseline
    experiment_data['is_baseline'] = combined_baseline
    
    # Add log concentrations 
    experiment_data['log_sds'] = np.log10(experiment_data['surf_A_conc_mm'])
    experiment_data['log_ttab'] = np.log10(experiment_data['surf_B_conc_mm'])
    
    return experiment_data

def train_gradient_gp(experiment_data):
    """Train GP to get gradient information."""
    
    # Prepare training data
    X_train = experiment_data[['log_sds', 'log_ttab']].values
    y_train = experiment_data['turbidity_600'].values
    
    # Standardize
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # Train GP
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5, normalize_y=False)
    gp.fit(X_train_scaled, y_train_scaled)
    
    print(f"✓ GP trained for gradient calculation")
    print(f"✓ Kernel: {gp.kernel_}")
    
    return gp, scaler_X, scaler_y

def calculate_gradient_field(experiment_data, gp, scaler_X, scaler_y):
    """Calculate gradient at each experimental point."""
    
    gradients = []
    
    for _, well in experiment_data.iterrows():
        log_sds = well['log_sds']
        log_ttab = well['log_ttab']
        
        # Create small perturbations for finite difference gradient
        delta = 0.01  # Small step in log space
        
        points = np.array([
            [log_sds, log_ttab],           # Center
            [log_sds + delta, log_ttab],   # +SDS
            [log_sds - delta, log_ttab],   # -SDS  
            [log_sds, log_ttab + delta],   # +TTAB
            [log_sds, log_ttab - delta]    # -TTAB
        ])
        
        # Scale and predict
        points_scaled = scaler_X.transform(points)
        preds_scaled = gp.predict(points_scaled)
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        
        # Calculate gradients 
        grad_sds = (preds[1] - preds[2]) / (2 * delta)  # ∂turbidity/∂(log_SDS)
        grad_ttab = (preds[3] - preds[4]) / (2 * delta)  # ∂turbidity/∂(log_TTAB)
        grad_magnitude = np.sqrt(grad_sds**2 + grad_ttab**2)
        
        gradients.append({
            'log_sds': log_sds,
            'log_ttab': log_ttab,
            'sds_conc': well['surf_A_conc_mm'],
            'ttab_conc': well['surf_B_conc_mm'],
            'grad_sds': grad_sds,
            'grad_ttab': grad_ttab,
            'grad_magnitude': grad_magnitude,
            'is_baseline': well['is_baseline']
        })
    
    gradient_df = pd.DataFrame(gradients)
    
    print(f"✓ Calculated gradients for {len(gradients)} points")
    print(f"✓ Gradient magnitude range: {gradient_df['grad_magnitude'].min():.3f} - {gradient_df['grad_magnitude'].max():.3f}")
    
    return gradient_df

def define_rectangle_regions(experiment_data, gradient_df):
    """
    Define the 3 main rectangular regions based on baseline classification.
    """
    
    # Find baseline boundary from the attachment image
    baseline_wells = experiment_data[experiment_data['is_baseline']]
    nonbaseline_wells = experiment_data[~experiment_data['is_baseline']]
    
    # From the image, we can see approximate boundaries:
    # SDS boundary around 0.2214 mM (log ≈ -0.65)
    # TTAB boundary around 1.0331 mM (log ≈ 0.01)
    
    sds_boundary_log = np.log10(0.2214)  # From red dashed line in image
    ttab_boundary_log = np.log10(1.0331)  # From red dashed line in image
    
    print(f"Boundary thresholds:")
    print(f"  SDS: {np.exp(sds_boundary_log):.4f} mM (log: {sds_boundary_log:.3f})")
    print(f"  TTAB: {np.exp(ttab_boundary_log):.4f} mM (log: {ttab_boundary_log:.3f})")
    
    # Define the 3 rectangles
    rectangles = [
        {
            'name': 'High_SDS_Region',
            'description': 'High SDS concentrations (top region)',
            'base_bounds': {
                'log_sds_min': sds_boundary_log,
                'log_sds_max': experiment_data['log_sds'].max(),
                'log_ttab_min': experiment_data['log_ttab'].min(),
                'log_ttab_max': ttab_boundary_log
            },
            'priority': 'high'
        },
        {
            'name': 'High_TTAB_Region', 
            'description': 'High TTAB concentrations (right region)',
            'base_bounds': {
                'log_sds_min': experiment_data['log_sds'].min(),
                'log_sds_max': sds_boundary_log,
                'log_ttab_min': ttab_boundary_log,
                'log_ttab_max': experiment_data['log_ttab'].max()
            },
            'priority': 'high'
        },
        {
            'name': 'Interaction_Region',
            'description': 'High SDS + High TTAB (top-right corner)',
            'base_bounds': {
                'log_sds_min': sds_boundary_log,
                'log_sds_max': experiment_data['log_sds'].max(),
                'log_ttab_min': ttab_boundary_log,
                'log_ttab_max': experiment_data['log_ttab'].max()
            },
            'priority': 'very_high'
        }
    ]
    
    return rectangles

def optimize_rectangle_bounds(rectangles, gradient_df, target_wells_per_rectangle=32):
    """
    Calculate optimal x-y bounds for each rectangle based on gradient information.
    High gradient = tighter bounds, Low gradient = wider bounds.
    """
    
    optimized_rectangles = []
    
    for rect in rectangles:
        print(f"\\nOptimizing bounds for {rect['name']}...")
        
        # Get base bounds
        base_bounds = rect['base_bounds']
        
        # Find gradient points in this region
        in_region = (
            (gradient_df['log_sds'] >= base_bounds['log_sds_min']) &
            (gradient_df['log_sds'] <= base_bounds['log_sds_max']) &
            (gradient_df['log_ttab'] >= base_bounds['log_ttab_min']) &
            (gradient_df['log_ttab'] <= base_bounds['log_ttab_max'])
        )
        
        region_gradients = gradient_df[in_region].copy()
        
        if len(region_gradients) == 0:
            print(f"  No gradient points in region - using base bounds")
            optimized_bounds = base_bounds.copy()
        else:
            # Calculate gradient statistics for this region
            avg_gradient = region_gradients['grad_magnitude'].mean()
            max_gradient = region_gradients['grad_magnitude'].max()
            
            print(f"  Points in region: {len(region_gradients)}")
            print(f"  Average gradient: {avg_gradient:.3f}")
            print(f"  Max gradient: {max_gradient:.3f}")
            
            # Gradient-based bound adjustment
            # High gradient → tighter bounds (focus sampling)
            # Low gradient → wider bounds (broader coverage)
            
            gradient_scale = avg_gradient / max_gradient if max_gradient > 0 else 0.5
            
            # Adjust bounds based on gradient intensity
            # Higher gradient = smaller adjustment (tighter bounds)
            # Lower gradient = larger adjustment (wider bounds)
            adjustment_factor = 1.0 - (gradient_scale * 0.5)  # Range: 0.5 to 1.0
            
            sds_range = base_bounds['log_sds_max'] - base_bounds['log_sds_min']
            ttab_range = base_bounds['log_ttab_max'] - base_bounds['log_ttab_min']
            
            sds_adjustment = sds_range * adjustment_factor * 0.2  # Max 20% adjustment
            ttab_adjustment = ttab_range * adjustment_factor * 0.2
            
            optimized_bounds = {
                'log_sds_min': base_bounds['log_sds_min'] + sds_adjustment,
                'log_sds_max': base_bounds['log_sds_max'] - sds_adjustment,
                'log_ttab_min': base_bounds['log_ttab_min'] + ttab_adjustment,
                'log_ttab_max': base_bounds['log_ttab_max'] - ttab_adjustment
            }
            
            print(f"  Gradient scale: {gradient_scale:.3f}")
            print(f"  Adjustment factor: {adjustment_factor:.3f}")
            print(f"  SDS adjustment: ±{sds_adjustment:.3f}")
            print(f"  TTAB adjustment: ±{ttab_adjustment:.3f}")
        
        # Convert back to concentration space
        optimized_rect = {
            'name': rect['name'],
            'description': rect['description'],
            'priority': rect['priority'],
            'log_bounds': optimized_bounds,
            'conc_bounds': {
                'sds_min': 10**optimized_bounds['log_sds_min'],
                'sds_max': 10**optimized_bounds['log_sds_max'],
                'ttab_min': 10**optimized_bounds['log_ttab_min'],
                'ttab_max': 10**optimized_bounds['log_ttab_max']
            },
            'gradient_stats': {
                'n_points': len(region_gradients),
                'avg_gradient': avg_gradient if len(region_gradients) > 0 else 0,
                'max_gradient': max_gradient if len(region_gradients) > 0 else 0
            }
        }
        
        # Calculate area and well density
        log_area = ((optimized_bounds['log_sds_max'] - optimized_bounds['log_sds_min']) * 
                   (optimized_bounds['log_ttab_max'] - optimized_bounds['log_ttab_min']))
        
        optimized_rect['log_area'] = log_area
        optimized_rect['target_wells'] = target_wells_per_rectangle
        optimized_rect['well_density'] = target_wells_per_rectangle / log_area if log_area > 0 else 0
        
        optimized_rectangles.append(optimized_rect)
        
        print(f"  Optimized SDS: {optimized_rect['conc_bounds']['sds_min']:.4f} - {optimized_rect['conc_bounds']['sds_max']:.3f} mM")
        print(f"  Optimized TTAB: {optimized_rect['conc_bounds']['ttab_min']:.4f} - {optimized_rect['conc_bounds']['ttab_max']:.3f} mM")
        print(f"  Log area: {log_area:.3f}")
        print(f"  Target wells: {target_wells_per_rectangle}")
        print(f"  Well density: {optimized_rect['well_density']:.1f} wells per log-unit²")
    
    return optimized_rectangles

def visualize_optimized_rectangles(experiment_data, gradient_df, optimized_rectangles, 
                                 output_dir='output/gradient_rectangle_bounds'):
    """
    Visualize the gradient-optimized rectangular bounds.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Separate baseline and non-baseline wells
    baseline_wells = experiment_data[experiment_data['is_baseline']]
    nonbaseline_wells = experiment_data[~experiment_data['is_baseline']]
    
    # Plot 1: Original data with optimized rectangles
    ax1.scatter(baseline_wells['surf_A_conc_mm'], baseline_wells['surf_B_conc_mm'],
               c='lightblue', s=50, alpha=0.7, label=f'Baseline wells (n={len(baseline_wells)})', 
               marker='o', edgecolors='blue', linewidth=1)
    ax1.scatter(nonbaseline_wells['surf_A_conc_mm'], nonbaseline_wells['surf_B_conc_mm'],
               c='red', s=60, alpha=0.8, label=f'Non-baseline wells (n={len(nonbaseline_wells)})', 
               marker='s', edgecolors='darkred', linewidth=1)
    
    # Draw optimized rectangles
    colors = ['orange', 'green', 'purple']
    for i, rect in enumerate(optimized_rectangles):
        bounds = rect['conc_bounds']
        
        from matplotlib.patches import Rectangle as MPLRectangle
        rect_patch = MPLRectangle(
            (bounds['sds_min'], bounds['ttab_min']), 
            bounds['sds_max'] - bounds['sds_min'], 
            bounds['ttab_max'] - bounds['ttab_min'],
            linewidth=3, edgecolor=colors[i], facecolor=colors[i], alpha=0.3,
            label=f"{rect['name']} ({rect['target_wells']} wells)"
        )
        ax1.add_patch(rect_patch)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax1.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax1.set_title('Gradient-Optimized Rectangle Bounds', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Gradient magnitude field
    sc = ax2.scatter(gradient_df['sds_conc'], gradient_df['ttab_conc'], 
                    c=gradient_df['grad_magnitude'], s=60, cmap='viridis', alpha=0.8,
                    edgecolors='black', linewidth=1)
    
    # Overlay rectangle bounds
    for i, rect in enumerate(optimized_rectangles):
        bounds = rect['conc_bounds']
        rect_patch = MPLRectangle(
            (bounds['sds_min'], bounds['ttab_min']), 
            bounds['sds_max'] - bounds['sds_min'], 
            bounds['ttab_max'] - bounds['ttab_min'],
            linewidth=3, edgecolor=colors[i], facecolor='none', alpha=1.0
        )
        ax2.add_patch(rect_patch)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax2.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax2.set_title('Gradient Field with Optimized Bounds', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(sc, ax=ax2)
    cbar2.set_label('Gradient Magnitude', fontsize=10)
    
    # Plot 3: Rectangle statistics
    rect_names = [rect['name'].replace('_', '\\n') for rect in optimized_rectangles]
    avg_gradients = [rect['gradient_stats']['avg_gradient'] for rect in optimized_rectangles]
    well_densities = [rect['well_density'] for rect in optimized_rectangles]
    
    x_pos = range(len(rect_names))
    ax3_twin = ax3.twinx()
    
    bars1 = ax3.bar([x - 0.2 for x in x_pos], avg_gradients, 0.4, 
                   label='Avg Gradient', color='red', alpha=0.7)
    bars2 = ax3_twin.bar([x + 0.2 for x in x_pos], well_densities, 0.4,
                        label='Well Density', color='blue', alpha=0.7)
    
    ax3.set_xlabel('Rectangle', fontsize=12)
    ax3.set_ylabel('Average Gradient', fontsize=12, color='red')
    ax3_twin.set_ylabel('Well Density (wells/log-unit²)', fontsize=12, color='blue')
    ax3.set_title('Rectangle Optimization Metrics', fontweight='bold', fontsize=14)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(rect_names)
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 4: Bounds comparison
    rect_data = []
    for rect in optimized_rectangles:
        rect_data.append({
            'Rectangle': rect['name'].replace('_', '\\n'),
            'SDS Range': f"{rect['conc_bounds']['sds_min']:.3f} - {rect['conc_bounds']['sds_max']:.2f}",
            'TTAB Range': f"{rect['conc_bounds']['ttab_min']:.3f} - {rect['conc_bounds']['ttab_max']:.2f}",
            'Wells': rect['target_wells'],
            'Avg Gradient': f"{rect['gradient_stats']['avg_gradient']:.3f}"
        })
    
    # Create table
    table_data = pd.DataFrame(rect_data)
    ax4.axis('tight')
    ax4.axis('off')
    ax4.table(cellText=table_data.values, colLabels=table_data.columns,
             cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    ax4.set_title('Rectangle Bounds Summary', fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    output_file = f'{output_dir}/gradient_optimized_rectangles.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\\nVisualization saved: {output_file}")
    
    plt.show()

def main():
    """Main function for gradient-optimized rectangle bounds."""
    
    data_file = r"C:\\Users\\owenm\\OneDrive\\Desktop\\North Robotics\\utoronto_demo\\utoronto_demo\\surfactant_grid_SDS_TTAB_20260209_164920\\surfactant_grid_SDS_TTAB_20260209_164920\\complete_experiment_results.csv"
    
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    print("="*60)
    print("GRADIENT-BASED RECTANGLE BOUNDARY OPTIMIZATION")
    print("="*60)
    
    print("\\n1. Loading and classifying data...")
    experiment_data = load_and_classify_data(data_file)
    
    print("\\n2. Training GP for gradient calculation...")
    gp, scaler_X, scaler_y = train_gradient_gp(experiment_data)
    
    print("\\n3. Calculating gradient field...")
    gradient_df = calculate_gradient_field(experiment_data, gp, scaler_X, scaler_y)
    
    print("\\n4. Defining rectangle regions...")
    rectangles = define_rectangle_regions(experiment_data, gradient_df)
    
    print(f"\\nDefined {len(rectangles)} rectangles:")
    for rect in rectangles:
        print(f"  - {rect['name']}: {rect['description']} ({rect['priority']} priority)")
    
    print("\\n5. Optimizing rectangle bounds using gradients...")
    optimized_rectangles = optimize_rectangle_bounds(rectangles, gradient_df, target_wells_per_rectangle=32)
    
    print("\\n6. Visualizing optimized rectangles...")
    visualize_optimized_rectangles(experiment_data, gradient_df, optimized_rectangles)
    
    total_wells = sum([rect['target_wells'] for rect in optimized_rectangles])
    
    print("\\n" + "="*60)
    print("GRADIENT-OPTIMIZED RECTANGLE RESULTS")
    print("="*60)
    print(f"✓ {len(optimized_rectangles)} rectangles defined")
    print(f"✓ {total_wells} total wells allocated")
    print("✓ Rectangle bounds optimized using gradient information:")
    print("  → High gradient regions: Tighter bounds (focused sampling)")
    print("  → Low gradient regions: Wider bounds (broader coverage)")
    
    print("\\nOptimized rectangles:")
    for rect in optimized_rectangles:
        print(f"\\n  {rect['name']}:")
        print(f"    SDS: {rect['conc_bounds']['sds_min']:.4f} - {rect['conc_bounds']['sds_max']:.3f} mM")
        print(f"    TTAB: {rect['conc_bounds']['ttab_min']:.4f} - {rect['conc_bounds']['ttab_max']:.3f} mM")
        print(f"    Wells: {rect['target_wells']}, Density: {rect['well_density']:.1f} wells/log-unit²")
        print(f"    Avg gradient: {rect['gradient_stats']['avg_gradient']:.3f}")

if __name__ == "__main__":
    main()
"""
Gradient Field Visualization
Create maps showing gradients in each direction for turbidity and ratio.
Visualize where measurements change most steeply in SDS vs TTAB directions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import os

def load_baseline_classification(csv_file_path):
    """Load and classify the existing data."""
    df = pd.read_csv(csv_file_path)
    experiment_data = df[df['well_type'] == 'experiment'].copy()
    
    # Apply the AND logic classification
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
    
    return experiment_data

def classify_high_info_wells(baseline_data):
    """Get high-info well classification."""
    
    # Get unique concentrations for grid mapping
    sds_concentrations = sorted(baseline_data['surf_A_conc_mm'].unique())
    ttab_concentrations = sorted(baseline_data['surf_B_conc_mm'].unique())
    
    # Create grid mapping
    grid_map = {}
    for _, well in baseline_data.iterrows():
        sds_idx = sds_concentrations.index(well['surf_A_conc_mm'])
        ttab_idx = ttab_concentrations.index(well['surf_B_conc_mm'])
        grid_map[(sds_idx, ttab_idx)] = well
    
    # Apply high-info classification
    high_info_wells = []
    n_sds = len(sds_concentrations)
    n_ttab = len(ttab_concentrations)
    
    for sds_idx in range(n_sds):
        for ttab_idx in range(n_ttab):
            if (sds_idx, ttab_idx) not in grid_map:
                continue
                
            current_well = grid_map[(sds_idx, ttab_idx)]
            is_high_info = False
            
            # Rule 1: Non-baseline wells are high-info
            if not current_well['is_baseline']:
                is_high_info = True
            
            # Rule 2: Wells adjacent to non-baseline are high-info
            if not is_high_info:
                adjacent_positions = [
                    (sds_idx-1, ttab_idx-1), (sds_idx-1, ttab_idx), (sds_idx-1, ttab_idx+1),
                    (sds_idx, ttab_idx-1),                          (sds_idx, ttab_idx+1),
                    (sds_idx+1, ttab_idx-1), (sds_idx+1, ttab_idx), (sds_idx+1, ttab_idx+1)
                ]
                
                for adj_sds, adj_ttab in adjacent_positions:
                    if 0 <= adj_sds < n_sds and 0 <= adj_ttab < n_ttab:
                        if (adj_sds, adj_ttab) in grid_map:
                            adj_well = grid_map[(adj_sds, adj_ttab)]
                            if not adj_well['is_baseline']:
                                is_high_info = True
                                break
            
            if is_high_info:
                high_info_wells.append(current_well)
    
    return high_info_wells, sds_concentrations, ttab_concentrations

def calculate_gradients_at_wells(high_info_wells, max_neighbors=4, max_distance_log=0.7):
    """
    Calculate gradients in each direction (SDS, TTAB) for each measurement (turbidity, ratio).
    """
    
    if len(high_info_wells) < 3:
        return []
    
    # Convert to log coordinates for distance calculations
    existing_coords_log = np.array([[np.log10(w['surf_A_conc_mm']), np.log10(w['surf_B_conc_mm'])] 
                                   for w in high_info_wells])
    existing_turbidity = np.array([w['turbidity_600'] for w in high_info_wells])
    existing_ratio = np.array([w['ratio'] for w in high_info_wells])
    
    gradient_results = []
    
    print(f"Calculating gradients for {len(high_info_wells)} high-info wells...")
    
    for i, well in enumerate(high_info_wells):
        point_log = existing_coords_log[i]
        
        # Find nearest existing wells
        distances = np.linalg.norm(existing_coords_log - point_log, axis=1)
        # Exclude self (distance = 0)
        distances[i] = float('inf')
        
        nearby_indices = np.where(distances <= max_distance_log)[0]
        
        if len(nearby_indices) < 2:
            continue
        
        # Take closest neighbors
        nearby_indices = nearby_indices[np.argsort(distances[nearby_indices])][:max_neighbors]
        nearby_coords = existing_coords_log[nearby_indices]
        nearby_turbidity = existing_turbidity[nearby_indices]
        nearby_ratio = existing_ratio[nearby_indices]
        
        # Calculate gradients using local linear fit
        try:
            # Relative coordinates centered on current point
            rel_coords = nearby_coords - point_log
            
            # Fit plane: measurement = a*sds_log + b*ttab_log + c
            A = np.column_stack([rel_coords, np.ones(len(rel_coords))])
            
            # Turbidity gradients
            turbidity_coeffs = np.linalg.lstsq(A, nearby_turbidity, rcond=None)[0]
            dturbidity_dsds = turbidity_coeffs[0]  # ∂turbidity/∂(log SDS)
            dturbidity_dttab = turbidity_coeffs[1]  # ∂turbidity/∂(log TTAB)
            
            # Ratio gradients
            ratio_coeffs = np.linalg.lstsq(A, nearby_ratio, rcond=None)[0]
            dratio_dsds = ratio_coeffs[0]  # ∂ratio/∂(log SDS)
            dratio_dttab = ratio_coeffs[1]  # ∂ratio/∂(log TTAB)
            
        except np.linalg.LinAlgError:
            dturbidity_dsds = 0.0
            dturbidity_dttab = 0.0
            dratio_dsds = 0.0
            dratio_dttab = 0.0
        
        gradient_results.append({
            'well': well,
            'position': [well['surf_A_conc_mm'], well['surf_B_conc_mm']],
            'position_log': point_log,
            'turbidity_value': well['turbidity_600'],
            'ratio_value': well['ratio'],
            'dturbidity_dsds': dturbidity_dsds,
            'dturbidity_dttab': dturbidity_dttab,
            'dratio_dsds': dratio_dsds,
            'dratio_dttab': dratio_dttab,
            'turbidity_grad_magnitude': np.sqrt(dturbidity_dsds**2 + dturbidity_dttab**2),
            'ratio_grad_magnitude': np.sqrt(dratio_dsds**2 + dratio_dttab**2),
            'nearby_count': len(nearby_indices)
        })
    
    print(f"Calculated gradients for {len(gradient_results)} locations")
    
    return gradient_results

def create_gradient_field_maps(gradient_results, output_dir='output/gradient_field_maps'):
    """
    Create 4 gradient field maps showing directional gradients.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    if len(gradient_results) == 0:
        print("No gradient data to visualize!")
        return
    
    # Extract data for plotting
    positions = np.array([g['position'] for g in gradient_results])
    sds_coords = positions[:, 0]
    ttab_coords = positions[:, 1]
    
    # Gradient components
    dturbidity_dsds = np.array([g['dturbidity_dsds'] for g in gradient_results])
    dturbidity_dttab = np.array([g['dturbidity_dttab'] for g in gradient_results])
    dratio_dsds = np.array([g['dratio_dsds'] for g in gradient_results])
    dratio_dttab = np.array([g['dratio_dttab'] for g in gradient_results])
    
    # Also get the actual measurement values for reference
    turbidity_values = np.array([g['turbidity_value'] for g in gradient_results])
    ratio_values = np.array([g['ratio_value'] for g in gradient_results])
    
    # Separate baseline and non-baseline for coloring
    is_baseline = np.array([g['well']['is_baseline'] for g in gradient_results])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: ∂turbidity/∂(log SDS)
    scatter1 = ax1.scatter(sds_coords, ttab_coords, c=dturbidity_dsds, 
                          s=80, cmap='RdBu_r', alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Mark baseline vs non-baseline with different markers
    baseline_mask = is_baseline
    nonbaseline_mask = ~is_baseline
    
    if np.any(baseline_mask):
        ax1.scatter(sds_coords[baseline_mask], ttab_coords[baseline_mask], 
                   s=100, marker='o', facecolors='none', edgecolors='blue', linewidth=2, alpha=0.8)
    
    if np.any(nonbaseline_mask):
        ax1.scatter(sds_coords[nonbaseline_mask], ttab_coords[nonbaseline_mask], 
                   s=120, marker='s', facecolors='none', edgecolors='red', linewidth=2, alpha=0.8)
    
    plt.colorbar(scatter1, ax=ax1, label='∂turbidity/∂(log SDS)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax1.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax1.set_title('Turbidity Gradient: SDS Direction\\n∂turbidity/∂(log SDS)', 
                 fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ∂turbidity/∂(log TTAB) 
    scatter2 = ax2.scatter(sds_coords, ttab_coords, c=dturbidity_dttab, 
                          s=80, cmap='RdBu_r', alpha=0.8, edgecolors='black', linewidth=0.5)
    
    if np.any(baseline_mask):
        ax2.scatter(sds_coords[baseline_mask], ttab_coords[baseline_mask], 
                   s=100, marker='o', facecolors='none', edgecolors='blue', linewidth=2, alpha=0.8)
    
    if np.any(nonbaseline_mask):
        ax2.scatter(sds_coords[nonbaseline_mask], ttab_coords[nonbaseline_mask], 
                   s=120, marker='s', facecolors='none', edgecolors='red', linewidth=2, alpha=0.8)
    
    plt.colorbar(scatter2, ax=ax2, label='∂turbidity/∂(log TTAB)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax2.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax2.set_title('Turbidity Gradient: TTAB Direction\\n∂turbidity/∂(log TTAB)', 
                 fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ∂ratio/∂(log SDS)
    scatter3 = ax3.scatter(sds_coords, ttab_coords, c=dratio_dsds, 
                          s=80, cmap='RdBu_r', alpha=0.8, edgecolors='black', linewidth=0.5)
    
    if np.any(baseline_mask):
        ax3.scatter(sds_coords[baseline_mask], ttab_coords[baseline_mask], 
                   s=100, marker='o', facecolors='none', edgecolors='blue', linewidth=2, alpha=0.8)
    
    if np.any(nonbaseline_mask):
        ax3.scatter(sds_coords[nonbaseline_mask], ttab_coords[nonbaseline_mask], 
                   s=120, marker='s', facecolors='none', edgecolors='red', linewidth=2, alpha=0.8)
    
    plt.colorbar(scatter3, ax=ax3, label='∂ratio/∂(log SDS)')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax3.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax3.set_title('Ratio Gradient: SDS Direction\\n∂ratio/∂(log SDS)', 
                 fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: ∂ratio/∂(log TTAB)
    scatter4 = ax4.scatter(sds_coords, ttab_coords, c=dratio_dttab, 
                          s=80, cmap='RdBu_r', alpha=0.8, edgecolors='black', linewidth=0.5)
    
    if np.any(baseline_mask):
        ax4.scatter(sds_coords[baseline_mask], ttab_coords[baseline_mask], 
                   s=100, marker='o', facecolors='none', edgecolors='blue', linewidth=2, alpha=0.8)
    
    if np.any(nonbaseline_mask):
        ax4.scatter(sds_coords[nonbaseline_mask], ttab_coords[nonbaseline_mask], 
                   s=120, marker='s', facecolors='none', edgecolors='red', linewidth=2, alpha=0.8)
    
    plt.colorbar(scatter4, ax=ax4, label='∂ratio/∂(log TTAB)')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax4.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax4.set_title('Ratio Gradient: TTAB Direction\\n∂ratio/∂(log TTAB)', 
                 fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # Add legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
               markeredgecolor='blue', markersize=10, markeredgewidth=2, label='Baseline wells'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='none', 
               markeredgecolor='red', markersize=10, markeredgewidth=2, label='Non-baseline wells')
    ]
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    output_file = f'{output_dir}/gradient_field_maps.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\\nVisualization saved: {output_file}")
    
    plt.show()
    
    # Print gradient statistics
    print("\\n" + "="*60)
    print("GRADIENT FIELD STATISTICS")
    print("="*60)
    
    print(f"\\nTurbidity gradients:")
    print(f"  ∂turbidity/∂(log SDS):  min={dturbidity_dsds.min():.3f}, max={dturbidity_dsds.max():.3f}")
    print(f"  ∂turbidity/∂(log TTAB): min={dturbidity_dttab.min():.3f}, max={dturbidity_dttab.max():.3f}")
    
    print(f"\\nRatio gradients:")
    print(f"  ∂ratio/∂(log SDS):      min={dratio_dsds.min():.3f}, max={dratio_dsds.max():.3f}")
    print(f"  ∂ratio/∂(log TTAB):     min={dratio_dttab.min():.3f}, max={dratio_dttab.max():.3f}")
    
    # Gradient magnitudes
    turbidity_grad_magnitudes = np.array([g['turbidity_grad_magnitude'] for g in gradient_results])
    ratio_grad_magnitudes = np.array([g['ratio_grad_magnitude'] for g in gradient_results])
    
    print(f"\\nGradient magnitudes:")
    print(f"  Turbidity: min={turbidity_grad_magnitudes.min():.3f}, max={turbidity_grad_magnitudes.max():.3f}")
    print(f"  Ratio:     min={ratio_grad_magnitudes.min():.3f}, max={ratio_grad_magnitudes.max():.3f}")
    
    # Find locations with highest gradients
    print(f"\\nTop 5 turbidity gradient locations:")
    top_turb_indices = np.argsort(turbidity_grad_magnitudes)[-5:]
    for i, idx in enumerate(top_turb_indices[::-1]):
        grad = gradient_results[idx]
        print(f"  {i+1}. SDS={grad['position'][0]:.2e}, TTAB={grad['position'][1]:.2e} "
              f"(mag: {grad['turbidity_grad_magnitude']:.3f})")
    
    print(f"\\nTop 5 ratio gradient locations:")
    top_ratio_indices = np.argsort(ratio_grad_magnitudes)[-5:]
    for i, idx in enumerate(top_ratio_indices[::-1]):
        grad = gradient_results[idx]
        print(f"  {i+1}. SDS={grad['position'][0]:.2e}, TTAB={grad['position'][1]:.2e} "
              f"(mag: {grad['ratio_grad_magnitude']:.3f})")

def main():
    """Main function for gradient field visualization."""
    
    data_file = r"C:\\Users\\owenm\\OneDrive\\Desktop\\North Robotics\\utoronto_demo\\utoronto_demo\\surfactant_grid_SDS_TTAB_20260209_164920\\surfactant_grid_SDS_TTAB_20260209_164920\\complete_experiment_results.csv"
    
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    print("Loading baseline classification...")
    baseline_data = load_baseline_classification(data_file)
    
    print("\\nClassifying high-info wells...")
    high_info_wells, sds_concentrations, ttab_concentrations = classify_high_info_wells(baseline_data)
    print(f"High-info wells: {len(high_info_wells)}")
    
    print("\\nCalculating gradient fields...")
    gradient_results = calculate_gradients_at_wells(high_info_wells)
    
    if len(gradient_results) == 0:
        print("No gradient calculations possible!")
        return
    
    print("\\nCreating gradient field maps...")
    create_gradient_field_maps(gradient_results)

if __name__ == "__main__":
    main()
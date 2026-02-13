"""
Gradient-Based Adaptive Sampling
Find unsampled locations in high-info region with highest gradients.
Sample where measurements change most rapidly for better resolution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
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
    """Get high-info well classification from previous analysis."""
    
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

def generate_fine_grid(high_info_wells, sds_concentrations, ttab_concentrations, 
                      fine_grid_spacing_log=0.3):
    """
    Generate a fine grid across the high-info region.
    """
    
    if len(high_info_wells) == 0:
        return np.array([]), {}
    
    # Find bounds of high-info region in log space
    sds_coords = [w['surf_A_conc_mm'] for w in high_info_wells]
    ttab_coords = [w['surf_B_conc_mm'] for w in high_info_wells]
    
    sds_min_log = np.log10(min(sds_coords))
    sds_max_log = np.log10(max(sds_coords))
    ttab_min_log = np.log10(min(ttab_coords))
    ttab_max_log = np.log10(max(ttab_coords))
    
    # Generate fine grid
    n_sds_fine = int((sds_max_log - sds_min_log) / fine_grid_spacing_log) + 1
    n_ttab_fine = int((ttab_max_log - ttab_min_log) / fine_grid_spacing_log) + 1
    
    sds_fine_log = np.linspace(sds_min_log, sds_max_log, n_sds_fine)
    ttab_fine_log = np.linspace(ttab_min_log, ttab_max_log, n_ttab_fine)
    
    # Convert back to linear space
    sds_fine = 10**sds_fine_log
    ttab_fine = 10**ttab_fine_log
    
    # Create all fine grid points
    fine_grid_points = []
    for sds in sds_fine:
        for ttab in ttab_fine:
            fine_grid_points.append([sds, ttab])
    
    fine_grid_points = np.array(fine_grid_points)
    
    print(f"Fine grid: {n_sds_fine} × {n_ttab_fine} = {len(fine_grid_points)} points")
    print(f"Fine grid spacing: {fine_grid_spacing_log:.2f} log units = {10**fine_grid_spacing_log:.2f}× factors")
    
    # Create existing sample mapping for fast lookup
    existing_samples = {}
    tolerance_log = 0.05  # Within 0.05 log units considered "same location"
    
    for well in high_info_wells:
        sds_log = np.log10(well['surf_A_conc_mm'])
        ttab_log = np.log10(well['surf_B_conc_mm'])
        existing_samples[(sds_log, ttab_log)] = well
    
    return fine_grid_points, existing_samples, tolerance_log

def find_unsampled_locations(fine_grid_points, existing_samples, tolerance_log):
    """
    Find fine grid points that don't correspond to existing samples.
    """
    
    unsampled_points = []
    
    for point in fine_grid_points:
        sds, ttab = point
        sds_log, ttab_log = np.log10(sds), np.log10(ttab)
        
        # Check if this point is close to any existing sample
        is_sampled = False
        for (existing_sds_log, existing_ttab_log) in existing_samples.keys():
            sds_diff = abs(sds_log - existing_sds_log)
            ttab_diff = abs(ttab_log - existing_ttab_log)
            
            if sds_diff < tolerance_log and ttab_diff < tolerance_log:
                is_sampled = True
                break
        
        if not is_sampled:
            unsampled_points.append([sds, ttab])
    
    unsampled_points = np.array(unsampled_points)
    print(f"Unsampled locations: {len(unsampled_points)}/{len(fine_grid_points)} fine grid points")
    
    return unsampled_points

def calculate_gradients_at_unsampled_points(unsampled_points, high_info_wells, 
                                          max_neighbors=4, max_distance_log=0.7):
    """
    Calculate gradients at unsampled points based on nearby existing wells.
    Higher gradients indicate locations where sampling would provide better resolution.
    """
    
    if len(unsampled_points) == 0 or len(high_info_wells) == 0:
        return []
    
    # Convert existing wells to log coordinates for distance calculations
    existing_coords_log = np.array([[np.log10(w['surf_A_conc_mm']), np.log10(w['surf_B_conc_mm'])] 
                                   for w in high_info_wells])
    existing_turbidity = np.array([w['turbidity_600'] for w in high_info_wells])
    existing_ratio = np.array([w['ratio'] for w in high_info_wells])
    
    gradient_results = []
    
    print(f"\\nCalculating gradients for {len(unsampled_points)} unsampled points...")
    
    for i, point in enumerate(unsampled_points):
        sds, ttab = point
        point_log = np.array([np.log10(sds), np.log10(ttab)])
        
        # Find nearest existing wells
        distances = np.linalg.norm(existing_coords_log - point_log, axis=1)
        nearby_indices = np.where(distances <= max_distance_log)[0]
        
        if len(nearby_indices) < 2:
            # Not enough nearby points for gradient calculation
            continue
        
        # Take closest neighbors
        nearby_indices = nearby_indices[np.argsort(distances[nearby_indices])][:max_neighbors]
        nearby_distances = distances[nearby_indices]
        nearby_turbidity = existing_turbidity[nearby_indices]
        nearby_ratio = existing_ratio[nearby_indices]
        
        # Calculate gradient scores
        # Method 1: Range/span in nearby values (higher = more variable)
        turbidity_range = np.max(nearby_turbidity) - np.min(nearby_turbidity)
        ratio_range = np.max(nearby_ratio) - np.min(nearby_ratio)
        
        # Method 2: Weighted variance (closer points have more influence)
        weights = np.exp(-nearby_distances / 0.3)  # Exponential weighting
        weights = weights / np.sum(weights)
        
        turbidity_weighted_var = np.average((nearby_turbidity - np.average(nearby_turbidity, weights=weights))**2, weights=weights)
        ratio_weighted_var = np.average((nearby_ratio - np.average(nearby_ratio, weights=weights))**2, weights=weights)
        
        # Method 3: Directional gradient estimate
        if len(nearby_indices) >= 3:
            # Fit a simple plane to estimate gradient magnitude
            try:
                coords = existing_coords_log[nearby_indices]
                
                # Gradient in turbidity
                A = np.column_stack([coords, np.ones(len(coords))])
                turbidity_coeffs = np.linalg.lstsq(A, nearby_turbidity, rcond=None)[0]
                turbidity_gradient_mag = np.linalg.norm(turbidity_coeffs[:2])
                
                # Gradient in ratio
                ratio_coeffs = np.linalg.lstsq(A, nearby_ratio, rcond=None)[0]
                ratio_gradient_mag = np.linalg.norm(ratio_coeffs[:2])
                
            except np.linalg.LinAlgError:
                turbidity_gradient_mag = 0
                ratio_gradient_mag = 0
        else:
            turbidity_gradient_mag = 0
            ratio_gradient_mag = 0
        
        # Combined gradient score - ONLY RATIO GRADIENTS
        # Focus exclusively on ratio changes (interaction effects)
        
        # Ignore turbidity completely
        turbidity_score = 0.0
        
        # Only consider ratio components (typical ratio range ~0-0.1) 
        ratio_score = (ratio_range / 0.1 + np.sqrt(ratio_weighted_var) / 0.05 + \
                      ratio_gradient_mag / 0.1)
        
        # Combined score is just the ratio score
        combined_score = ratio_score
        
        gradient_results.append({
            'point': point,
            'point_log': point_log,
            'nearby_count': len(nearby_indices),
            'avg_distance': np.mean(nearby_distances),
            'turbidity_range': turbidity_range,
            'ratio_range': ratio_range,
            'turbidity_weighted_var': turbidity_weighted_var,
            'ratio_weighted_var': ratio_weighted_var,
            'turbidity_gradient_mag': turbidity_gradient_mag,
            'ratio_gradient_mag': ratio_gradient_mag,
            'turbidity_score': turbidity_score,
            'ratio_score': ratio_score,
            'combined_score': combined_score,
            'nearby_turbidity': nearby_turbidity,
            'nearby_ratio': nearby_ratio
        })
    
    # Sort by combined gradient score (highest first)
    gradient_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    print(f"Calculated gradients for {len(gradient_results)} points")
    print(f"Top gradient score: {gradient_results[0]['combined_score']:.3f}" if gradient_results else "No results")
    print(f"Median gradient score: {gradient_results[len(gradient_results)//2]['combined_score']:.3f}" if gradient_results else "No results")
    
    return gradient_results

def select_optimal_sampling_points(gradient_results, target_new_wells=96):
    """
    Select the top gradient locations for new sampling.
    """
    
    if len(gradient_results) == 0:
        return []
    
    # Take top N points by gradient score
    n_select = min(target_new_wells, len(gradient_results))
    selected_points = gradient_results[:n_select]
    
    print(f"\\nSelected {n_select} points for new sampling:")
    print(f"Score range: {selected_points[0]['combined_score']:.3f} to {selected_points[-1]['combined_score']:.3f}")
    
    # Statistics
    avg_turbidity_score = np.mean([p['turbidity_score'] for p in selected_points])
    avg_ratio_score = np.mean([p['ratio_score'] for p in selected_points])
    avg_nearby = np.mean([p['nearby_count'] for p in selected_points])
    
    print(f"Average turbidity score: {avg_turbidity_score:.3f}")
    print(f"Average ratio score: {avg_ratio_score:.3f}")
    print(f"Average nearby wells: {avg_nearby:.1f}")
    
    return selected_points

def visualize_gradient_based_sampling(high_info_wells, gradient_results, selected_points,
                                    output_dir='output/gradient_sampling'):
    """
    Visualize the gradient-based adaptive sampling strategy.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Separate existing wells for plotting
    baseline_wells = [w for w in high_info_wells if w['is_baseline']]
    nonbaseline_wells = [w for w in high_info_wells if not w['is_baseline']]
    
    # Plot 1: Overview with all gradient scores
    if baseline_wells:
        ax1.scatter([w['surf_A_conc_mm'] for w in baseline_wells], 
                   [w['surf_B_conc_mm'] for w in baseline_wells],
                   c='lightblue', s=50, alpha=0.7, marker='o', 
                   label=f'Existing baseline (n={len(baseline_wells)})')
    
    if nonbaseline_wells:
        ax1.scatter([w['surf_A_conc_mm'] for w in nonbaseline_wells], 
                   [w['surf_B_conc_mm'] for w in nonbaseline_wells],
                   c='red', s=60, alpha=0.8, marker='s', 
                   label=f'Existing non-baseline (n={len(nonbaseline_wells)})')
    
    # Show all gradient points with color-coded scores
    if gradient_results:
        points = np.array([g['point'] for g in gradient_results])
        scores = np.array([g['combined_score'] for g in gradient_results])
        
        # Normalize scores for color mapping
        if len(scores) > 1:
            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores_normalized = np.array([1.0])
        
        scatter = ax1.scatter(points[:, 0], points[:, 1], c=scores_normalized, 
                            s=30, alpha=0.6, marker='^', cmap='viridis',
                            label=f'Gradient candidates (n={len(points)})')
        plt.colorbar(scatter, ax=ax1, label='Normalized Gradient Score')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax1.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax1.set_title('Gradient-Based Sampling Overview', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Selected points (top gradients)
    if baseline_wells:
        ax2.scatter([w['surf_A_conc_mm'] for w in baseline_wells], 
                   [w['surf_B_conc_mm'] for w in baseline_wells],
                   c='lightblue', s=50, alpha=0.7, marker='o', 
                   label=f'Existing baseline')
    
    if nonbaseline_wells:
        ax2.scatter([w['surf_A_conc_mm'] for w in nonbaseline_wells], 
                   [w['surf_B_conc_mm'] for w in nonbaseline_wells],
                   c='red', s=60, alpha=0.8, marker='s', 
                   label=f'Existing non-baseline')
    
    if selected_points:
        selected_coords = np.array([p['point'] for p in selected_points])
        ax2.scatter(selected_coords[:, 0], selected_coords[:, 1], 
                   c='darkgreen', s=80, alpha=0.9, marker='^',
                   edgecolors='black', linewidth=0.5,
                   label=f'Selected for sampling (n={len(selected_points)})')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax2.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax2.set_title(f'Selected High-Gradient Points\\n{len(selected_points)} new sampling locations', 
                 fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Gradient score distribution
    if gradient_results:
        scores = [g['combined_score'] for g in gradient_results]
        ax3.hist(scores, bins=20, alpha=0.7, color='darkgreen', edgecolor='black')
        
        if selected_points:
            selected_scores = [p['combined_score'] for p in selected_points]
            ax3.axvline(x=min(selected_scores), color='red', linestyle='--', 
                       label=f'Selection threshold')
            ax3.hist(selected_scores, bins=10, alpha=0.8, color='orange', 
                    edgecolor='black', label='Selected points')
    
    ax3.set_xlabel('Combined Gradient Score', fontsize=12)
    ax3.set_ylabel('Number of Points', fontsize=12)
    ax3.set_title('Gradient Score Distribution', fontweight='bold', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Score components breakdown
    if selected_points:
        turbidity_scores = [p['turbidity_score'] for p in selected_points]
        ratio_scores = [p['ratio_score'] for p in selected_points]
        
        ax4.scatter(turbidity_scores, ratio_scores, c='darkgreen', s=60, alpha=0.7)
        ax4.set_xlabel('Turbidity Score', fontsize=12)
        ax4.set_ylabel('Ratio Score', fontsize=12)
        ax4.set_title('Score Components for Selected Points', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Add diagonal reference line
        max_score = max(max(turbidity_scores), max(ratio_scores))
        ax4.plot([0, max_score], [0, max_score], 'k--', alpha=0.3, 
                label='Equal contribution')
        ax4.legend()
    
    plt.tight_layout()
    
    output_file = f'{output_dir}/gradient_adaptive_sampling.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\\nVisualization saved: {output_file}")
    
    plt.show()

def main():
    """Main function for gradient-based adaptive sampling."""
    
    data_file = r"C:\\Users\\owenm\\OneDrive\\Desktop\\North Robotics\\utoronto_demo\\utoronto_demo\\surfactant_grid_SDS_TTAB_20260209_164920\\surfactant_grid_SDS_TTAB_20260209_164920\\complete_experiment_results.csv"
    
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    print("Loading baseline classification...")
    baseline_data = load_baseline_classification(data_file)
    
    print("\\nClassifying high-info wells...")
    high_info_wells, sds_concentrations, ttab_concentrations = classify_high_info_wells(baseline_data)
    print(f"High-info wells: {len(high_info_wells)}")
    
    print("\\nGenerating fine grid...")
    fine_grid_points, existing_samples, tolerance_log = generate_fine_grid(
        high_info_wells, sds_concentrations, ttab_concentrations, 
        fine_grid_spacing_log=0.25  # 1.78× spacing (denser than current ~4.67×)
    )
    
    print("\\nFinding unsampled locations...")
    unsampled_points = find_unsampled_locations(fine_grid_points, existing_samples, tolerance_log)
    
    if len(unsampled_points) == 0:
        print("No unsampled locations found!")
        return
    
    print("\\nCalculating gradients...")
    gradient_results = calculate_gradients_at_unsampled_points(unsampled_points, high_info_wells)
    
    if len(gradient_results) == 0:
        print("No gradient calculations possible!")
        return
    
    print("\\nSelecting optimal sampling points...")
    selected_points = select_optimal_sampling_points(gradient_results, target_new_wells=96)
    
    print("\\nVisualizing results...")
    visualize_gradient_based_sampling(high_info_wells, gradient_results, selected_points)
    
    print("\\n" + "="*60)
    print("GRADIENT-BASED ADAPTIVE SAMPLING RESULTS")
    print("="*60)
    print(f"✓ {len(high_info_wells)} existing high-info wells")
    print(f"✓ {len(fine_grid_points)} fine grid points generated")
    print(f"✓ {len(unsampled_points)} unsampled locations identified")
    print(f"✓ {len(gradient_results)} gradient calculations completed")
    print(f"✓ {len(selected_points)} high-gradient points selected for sampling")
    print("\\nStrategy:")
    print("  • Generate fine grid across high-info region")
    print("  • Calculate gradients at unsampled locations") 
    print("  • Prioritize locations with highest measurement variability")
    print("  • Add samples where resolution is needed most")
    
    if selected_points:
        print("\\nTop 5 selected points:")
        for i, point in enumerate(selected_points[:5]):
            sds, ttab = point['point']
            print(f"  {i+1}. SDS={sds:.2e}, TTAB={ttab:.2e} (score: {point['combined_score']:.3f})")

if __name__ == "__main__":
    main()
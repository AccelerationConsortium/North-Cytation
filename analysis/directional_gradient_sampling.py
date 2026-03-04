"""
Directional Gradient Sampling
Generate sampling points along gradient directions for maximum information density.
Dense along gradients, sparse perpendicular to gradients.
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

def calculate_gradient_vectors_and_magnitude(high_info_wells, max_neighbors=4, max_distance_log=0.7):
    """
    Calculate gradient vectors (direction) and magnitude at existing high-info locations.
    """
    
    if len(high_info_wells) < 3:
        return []
    
    # Convert to log coordinates for distance calculations
    existing_coords_log = np.array([[np.log10(w['surf_A_conc_mm']), np.log10(w['surf_B_conc_mm'])] 
                                   for w in high_info_wells])
    existing_turbidity = np.array([w['turbidity_600'] for w in high_info_wells])
    existing_ratio = np.array([w['ratio'] for w in high_info_wells])
    
    gradient_info = []
    
    print(f"Calculating gradient vectors for {len(high_info_wells)} high-info wells...")
    
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
        nearby_distances = distances[nearby_indices]
        nearby_turbidity = existing_turbidity[nearby_indices]
        nearby_ratio = existing_ratio[nearby_indices]
        
        # Calculate gradient vectors using local linear fit
        try:
            # Relative coordinates centered on current point
            rel_coords = nearby_coords - point_log
            
            # Fit plane: measurement = a*sds + b*ttab + c
            A = np.column_stack([rel_coords, np.ones(len(rel_coords))])
            
            # Gradient vector for turbidity
            turbidity_coeffs = np.linalg.lstsq(A, nearby_turbidity, rcond=None)[0]
            turbidity_grad_vector = turbidity_coeffs[:2]  # [d/d(sds_log), d/d(ttab_log)]
            turbidity_grad_mag = np.linalg.norm(turbidity_grad_vector)
            
            # Gradient vector for ratio  
            ratio_coeffs = np.linalg.lstsq(A, nearby_ratio, rcond=None)[0]
            ratio_grad_vector = ratio_coeffs[:2]  # [d/d(sds_log), d/d(ttab_log)]
            ratio_grad_mag = np.linalg.norm(ratio_grad_vector)
            
            # Normalize gradient vectors
            if turbidity_grad_mag > 1e-10:
                turbidity_grad_dir = turbidity_grad_vector / turbidity_grad_mag
            else:
                turbidity_grad_dir = np.array([0.0, 0.0])
                
            if ratio_grad_mag > 1e-10:
                ratio_grad_dir = ratio_grad_vector / ratio_grad_mag
            else:
                ratio_grad_dir = np.array([0.0, 0.0])
            
            # Combined gradient (focus on ratio as requested)
            combined_grad_vector = ratio_grad_vector  # Only ratio gradients
            combined_grad_mag = ratio_grad_mag
            
            if combined_grad_mag > 1e-10:
                combined_grad_dir = combined_grad_vector / combined_grad_mag
            else:
                combined_grad_dir = np.array([0.0, 0.0])
            
        except np.linalg.LinAlgError:
            turbidity_grad_vector = np.array([0.0, 0.0])
            turbidity_grad_dir = np.array([0.0, 0.0])
            turbidity_grad_mag = 0.0
            ratio_grad_vector = np.array([0.0, 0.0])
            ratio_grad_dir = np.array([0.0, 0.0])
            ratio_grad_mag = 0.0
            combined_grad_vector = np.array([0.0, 0.0])
            combined_grad_dir = np.array([0.0, 0.0])
            combined_grad_mag = 0.0
        
        gradient_info.append({
            'well': well,
            'position_log': point_log,
            'position': [well['surf_A_conc_mm'], well['surf_B_conc_mm']],
            'turbidity_grad_vector': turbidity_grad_vector,
            'turbidity_grad_dir': turbidity_grad_dir,
            'turbidity_grad_mag': turbidity_grad_mag,
            'ratio_grad_vector': ratio_grad_vector,
            'ratio_grad_dir': ratio_grad_dir,
            'ratio_grad_mag': ratio_grad_mag,
            'combined_grad_vector': combined_grad_vector,
            'combined_grad_dir': combined_grad_dir,
            'combined_grad_mag': combined_grad_mag,
            'nearby_count': len(nearby_indices)
        })
    
    # Sort by gradient magnitude (highest first)
    gradient_info.sort(key=lambda x: x['combined_grad_mag'], reverse=True)
    
    print(f"Calculated gradient vectors for {len(gradient_info)} locations")
    if gradient_info:
        print(f"Top gradient magnitude: {gradient_info[0]['combined_grad_mag']:.3f}")
        print(f"Median gradient magnitude: {gradient_info[len(gradient_info)//2]['combined_grad_mag']:.3f}")
    
    return gradient_info

def generate_directional_sampling_lines(gradient_info, target_wells=96, 
                                       line_length_log=0.8, points_per_line=5):
    """
    Generate sampling lines along gradient directions.
    Dense along gradient, sparse perpendicular.
    """
    
    if len(gradient_info) == 0:
        return []
    
    # Select top gradient locations for line generation
    n_lines = min(target_wells // points_per_line, len(gradient_info))
    top_gradients = gradient_info[:n_lines]
    
    sampling_lines = []
    all_sampling_points = []
    
    print(f"\\nGenerating {n_lines} directional sampling lines...")
    print(f"Line length: {line_length_log:.2f} log units")
    print(f"Points per line: {points_per_line}")
    
    for i, grad_info in enumerate(top_gradients):
        center_log = grad_info['position_log']
        grad_dir = grad_info['combined_grad_dir']
        grad_mag = grad_info['combined_grad_mag']
        
        if grad_mag < 1e-6:  # Skip very low gradient locations
            continue
        
        # Create sampling line along gradient direction
        # Line extends in both directions from center
        half_length = line_length_log / 2
        
        line_points_log = []
        for j in range(points_per_line):
            # Parameter from -1 to +1 along the line
            t = -1 + 2 * j / (points_per_line - 1) if points_per_line > 1 else 0
            
            point_log = center_log + t * half_length * grad_dir
            line_points_log.append(point_log)
        
        # Convert back to linear space
        line_points = []
        for point_log in line_points_log:
            sds = 10**point_log[0]
            ttab = 10**point_log[1]
            
            # Check if point is within reasonable bounds
            if 1e-5 <= sds <= 100 and 1e-5 <= ttab <= 100:
                line_points.append([sds, ttab])
                all_sampling_points.append([sds, ttab])
        
        sampling_lines.append({
            'center': grad_info['position'],
            'center_log': center_log,
            'gradient_direction': grad_dir,
            'gradient_magnitude': grad_mag,
            'line_points_log': line_points_log,
            'line_points': line_points,
            'source_well': grad_info['well']
        })
        
        print(f"Line {i+1}: {len(line_points)} points, grad_mag={grad_mag:.3f}, "
              f"dir=({grad_dir[0]:.2f}, {grad_dir[1]:.2f})")
    
    print(f"\\nTotal sampling points generated: {len(all_sampling_points)}")
    
    return sampling_lines, np.array(all_sampling_points)

def visualize_directional_sampling(high_info_wells, gradient_info, sampling_lines, 
                                 all_sampling_points, output_dir='output/directional_sampling'):
    """
    Visualize the directional gradient-based sampling strategy.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Separate existing wells for plotting
    baseline_wells = [w for w in high_info_wells if w['is_baseline']]
    nonbaseline_wells = [w for w in high_info_wells if not w['is_baseline']]
    
    # Plot 1: Overview with gradient vectors
    if baseline_wells:
        ax1.scatter([w['surf_A_conc_mm'] for w in baseline_wells], 
                   [w['surf_B_conc_mm'] for w in baseline_wells],
                   c='lightblue', s=50, alpha=0.7, marker='o', 
                   label=f'Baseline wells (n={len(baseline_wells)})')
    
    if nonbaseline_wells:
        ax1.scatter([w['surf_A_conc_mm'] for w in nonbaseline_wells], 
                   [w['surf_B_conc_mm'] for w in nonbaseline_wells],
                   c='red', s=60, alpha=0.8, marker='s', 
                   label=f'Non-baseline wells (n={len(nonbaseline_wells)})')
    
    # Show gradient vectors
    for grad_info in gradient_info[:20]:  # Top 20 gradients
        pos = grad_info['position']
        grad_dir = grad_info['combined_grad_dir']
        grad_mag = grad_info['combined_grad_mag']
        
        if grad_mag > 1e-6:
            # Scale arrow length by gradient magnitude
            arrow_length_sds = grad_mag * 0.3 * pos[0]  # Log scale consideration
            arrow_length_ttab = grad_mag * 0.3 * pos[1]
            
            ax1.arrow(pos[0], pos[1], 
                     arrow_length_sds * grad_dir[0], 
                     arrow_length_ttab * grad_dir[1],
                     head_width=pos[0]*0.08, head_length=pos[1]*0.08, 
                     fc='purple', ec='purple', alpha=0.7)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax1.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax1.set_title('Gradient Vectors (Purple Arrows)', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Directional sampling lines
    if baseline_wells:
        ax2.scatter([w['surf_A_conc_mm'] for w in baseline_wells], 
                   [w['surf_B_conc_mm'] for w in baseline_wells],
                   c='lightblue', s=30, alpha=0.5, marker='o', 
                   label=f'Baseline wells')
    
    if nonbaseline_wells:
        ax2.scatter([w['surf_A_conc_mm'] for w in nonbaseline_wells], 
                   [w['surf_B_conc_mm'] for w in nonbaseline_wells],
                   c='red', s=40, alpha=0.6, marker='s', 
                   label=f'Non-baseline wells')
    
    # Draw sampling lines
    colors = plt.cm.Set1(np.linspace(0, 1, len(sampling_lines)))
    
    for i, line_info in enumerate(sampling_lines[:15]):  # Show first 15 lines
        line_points = line_info['line_points']
        if len(line_points) > 0:
            line_points_array = np.array(line_points)
            color = colors[i % len(colors)]
            
            # Draw line
            ax2.plot(line_points_array[:, 0], line_points_array[:, 1], 
                    color=color, linewidth=2, alpha=0.7)
            
            # Mark sampling points
            ax2.scatter(line_points_array[:, 0], line_points_array[:, 1], 
                       c=color, s=60, marker='^', alpha=0.9,
                       edgecolors='black', linewidth=0.5)
            
            # Mark center
            center = line_info['center']
            ax2.scatter([center[0]], [center[1]], 
                       c='black', s=80, marker='*', alpha=0.8)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax2.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax2.set_title(f'Directional Sampling Lines\\n{len(all_sampling_points)} points along gradients', 
                 fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Gradient magnitude distribution
    if gradient_info:
        grad_mags = [g['combined_grad_mag'] for g in gradient_info]
        ax3.hist(grad_mags, bins=20, alpha=0.7, color='purple', edgecolor='black')
        
        # Mark where lines were generated
        if sampling_lines:
            line_grad_mags = [line['gradient_magnitude'] for line in sampling_lines]
            if line_grad_mags:
                ax3.axvline(x=min(line_grad_mags), color='red', linestyle='--', 
                           label=f'Line generation threshold')
    
    ax3.set_xlabel('Gradient Magnitude', fontsize=12)
    ax3.set_ylabel('Number of Wells', fontsize=12)
    ax3.set_title('Gradient Magnitude Distribution', fontweight='bold', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Gradient direction compass
    if gradient_info:
        # Show gradient directions as vectors on unit circle
        for grad_info in gradient_info[:30]:  # Top 30
            grad_dir = grad_info['combined_grad_dir']
            grad_mag = grad_info['combined_grad_mag']
            
            if grad_mag > 1e-6:
                # Scale by magnitude for visualization
                scaled_dir = grad_dir * (grad_mag / max([g['combined_grad_mag'] for g in gradient_info]))
                ax4.arrow(0, 0, scaled_dir[0], scaled_dir[1], 
                         head_width=0.05, head_length=0.05,
                         fc='purple', ec='purple', alpha=0.6)
    
    ax4.set_xlim(-1.2, 1.2)
    ax4.set_ylim(-1.2, 1.2)
    ax4.set_xlabel('SDS Gradient Direction', fontsize=12)
    ax4.set_ylabel('TTAB Gradient Direction', fontsize=12)
    ax4.set_title('Gradient Direction Compass', fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # Add unit circle reference
    circle = plt.Circle((0, 0), 1, fill=False, color='black', alpha=0.3, linestyle='--')
    ax4.add_patch(circle)
    
    plt.tight_layout()
    
    output_file = f'{output_dir}/directional_gradient_sampling.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\\nVisualization saved: {output_file}")
    
    plt.show()

def main():
    """Main function for directional gradient-based sampling."""
    
    data_file = r"C:\\Users\\owenm\\OneDrive\\Desktop\\North Robotics\\utoronto_demo\\utoronto_demo\\surfactant_grid_SDS_TTAB_20260209_164920\\surfactant_grid_SDS_TTAB_20260209_164920\\complete_experiment_results.csv"
    
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    print("Loading baseline classification...")
    baseline_data = load_baseline_classification(data_file)
    
    print("\\nClassifying high-info wells...")
    high_info_wells, sds_concentrations, ttab_concentrations = classify_high_info_wells(baseline_data)
    print(f"High-info wells: {len(high_info_wells)}")
    
    print("\\nCalculating gradient vectors and magnitudes...")
    gradient_info = calculate_gradient_vectors_and_magnitude(high_info_wells)
    
    if len(gradient_info) == 0:
        print("No gradient calculations possible!")
        return
    
    print("\\nGenerating directional sampling lines...")
    sampling_lines, all_sampling_points = generate_directional_sampling_lines(
        gradient_info, target_wells=96, line_length_log=0.6, points_per_line=5)
    
    print("\\nVisualizing results...")
    visualize_directional_sampling(high_info_wells, gradient_info, sampling_lines, 
                                  all_sampling_points)
    
    print("\\n" + "="*60)
    print("DIRECTIONAL GRADIENT-BASED SAMPLING RESULTS")
    print("="*60)
    print(f"✓ {len(high_info_wells)} existing high-info wells")
    print(f"✓ {len(gradient_info)} gradient vectors calculated")
    print(f"✓ {len(sampling_lines)} directional sampling lines generated")
    print(f"✓ {len(all_sampling_points)} total sampling points")
    print("\\nStrategy:")
    print("  • Calculate gradient direction vectors at existing wells")
    print("  • Generate sampling lines along gradient directions")
    print("  • Dense sampling along gradients (where change is rapid)")
    print("  • Sparse sampling perpendicular to gradients")
    print("  • Anisotropic sampling for maximum information density")
    
    if sampling_lines:
        avg_grad_mag = np.mean([line['gradient_magnitude'] for line in sampling_lines])
        print(f"\\nAverage gradient magnitude: {avg_grad_mag:.3f}")
        print(f"Line length: 0.6 log units")
        print("Top 5 sampling lines:")
        for i, line in enumerate(sampling_lines[:5]):
            center = line['center']
            grad_mag = line['gradient_magnitude']
            n_points = len(line['line_points'])
            print(f"  {i+1}. Center: SDS={center[0]:.2e}, TTAB={center[1]:.2e} "
                  f"(grad: {grad_mag:.3f}, {n_points} points)")

if __name__ == "__main__":
    main()
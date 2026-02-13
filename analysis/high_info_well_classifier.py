"""
High-Info Well Classification Algorithm
Simple approach: classify each well as high_info or not.
high_info = non-baseline OR adjacent to any non-baseline well.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

def organize_data_by_grid(baseline_data):
    """Organize data into a grid structure by unique concentrations."""
    
    # Get unique concentrations
    sds_concentrations = sorted(baseline_data['surf_A_conc_mm'].unique())
    ttab_concentrations = sorted(baseline_data['surf_B_conc_mm'].unique())
    
    print(f"Grid structure: {len(sds_concentrations)} SDS levels × {len(ttab_concentrations)} TTAB levels")
    
    # Create grid mapping: (sds_idx, ttab_idx) -> well_data
    grid_map = {}
    
    for _, well in baseline_data.iterrows():
        sds_idx = sds_concentrations.index(well['surf_A_conc_mm'])
        ttab_idx = ttab_concentrations.index(well['surf_B_conc_mm'])
        
        grid_map[(sds_idx, ttab_idx)] = {
            'well_data': well,
            'is_baseline': well['is_baseline'],
            'sds_conc': well['surf_A_conc_mm'],
            'ttab_conc': well['surf_B_conc_mm']
        }
    
    return grid_map, sds_concentrations, ttab_concentrations

def classify_high_info_wells(grid_map, sds_concentrations, ttab_concentrations):
    """
    Classify each well as high_info or not.
    high_info = non-baseline OR adjacent to any non-baseline well
    """
    
    n_sds = len(sds_concentrations)
    n_ttab = len(ttab_concentrations)
    
    high_info_wells = []
    high_info_count = 0
    
    print(f"\\nClassifying wells as high_info...")
    print(f"Rule: high_info = non-baseline OR adjacent to any non-baseline")
    
    for sds_idx in range(n_sds):
        for ttab_idx in range(n_ttab):
            
            # Skip if no well at this position
            if (sds_idx, ttab_idx) not in grid_map:
                continue
            
            current_well = grid_map[(sds_idx, ttab_idx)]
            is_high_info = False
            reason = []
            
            # Rule 1: If non-baseline, it's high_info
            if not current_well['is_baseline']:
                is_high_info = True
                reason.append("non-baseline")
            
            # Rule 2: If adjacent to any non-baseline well, it's high_info
            if not is_high_info:  # Only check if not already high_info
                # Check all 8 adjacent positions (including diagonals)
                adjacent_positions = [
                    (sds_idx-1, ttab_idx-1), (sds_idx-1, ttab_idx), (sds_idx-1, ttab_idx+1),
                    (sds_idx, ttab_idx-1),                          (sds_idx, ttab_idx+1),
                    (sds_idx+1, ttab_idx-1), (sds_idx+1, ttab_idx), (sds_idx+1, ttab_idx+1)
                ]
                
                for adj_sds, adj_ttab in adjacent_positions:
                    # Check bounds
                    if 0 <= adj_sds < n_sds and 0 <= adj_ttab < n_ttab:
                        # Check if adjacent well exists and is non-baseline
                        if (adj_sds, adj_ttab) in grid_map:
                            adj_well = grid_map[(adj_sds, adj_ttab)]
                            if not adj_well['is_baseline']:
                                is_high_info = True
                                reason.append("adjacent to non-baseline")
                                break  # Found one adjacent non-baseline, that's enough
            
            # Store result
            well_info = current_well.copy()
            well_info['is_high_info'] = is_high_info
            well_info['high_info_reason'] = "; ".join(reason) if reason else "baseline + not adjacent"
            well_info['grid_position'] = (sds_idx, ttab_idx)
            
            high_info_wells.append(well_info)
            
            if is_high_info:
                high_info_count += 1
    
    print(f"Results: {high_info_count}/{len(high_info_wells)} wells classified as high_info")
    
    return high_info_wells

def calculate_high_info_area_and_sampling(high_info_wells, sds_concentrations, ttab_concentrations):
    """
    Calculate the area covered by high-info wells and determine sampling density.
    """
    
    # Get high-info wells only
    high_info_data = [w for w in high_info_wells if w['is_high_info']]
    
    if len(high_info_data) == 0:
        return {}
    
    # Find bounding box in concentration space
    sds_coords = [w['sds_conc'] for w in high_info_data]
    ttab_coords = [w['ttab_conc'] for w in high_info_data]
    
    sds_min, sds_max = min(sds_coords), max(sds_coords)
    ttab_min, ttab_max = min(ttab_coords), max(ttab_coords)
    
    # Find bounding box in grid indices
    sds_indices = [sds_concentrations.index(w['sds_conc']) for w in high_info_data]
    ttab_indices = [ttab_concentrations.index(w['ttab_conc']) for w in high_info_data]
    
    sds_idx_min, sds_idx_max = min(sds_indices), max(sds_indices)
    ttab_idx_min, ttab_idx_max = min(ttab_indices), max(ttab_indices)
    
    # Calculate bounding box dimensions
    grid_width = ttab_idx_max - ttab_idx_min + 1
    grid_height = sds_idx_max - sds_idx_min + 1
    bounding_box_area = grid_width * grid_height
    
    # Calculate actual coverage (non-rectangular)
    actual_coverage = len(high_info_data)
    coverage_efficiency = actual_coverage / bounding_box_area
    
    # Calculate area in log space (more meaningful for concentrations)
    log_sds_min, log_sds_max = np.log10(sds_min), np.log10(sds_max)
    log_ttab_min, log_ttab_max = np.log10(ttab_min), np.log10(ttab_max)
    log_area = (log_sds_max - log_sds_min) * (log_ttab_max - log_ttab_min)
    
    # Calculate concentration ratios (fold changes)
    sds_fold_range = sds_max / sds_min
    ttab_fold_range = ttab_max / ttab_min
    
    print(f"\\n" + "="*50)
    print("HIGH-INFO REGION AREA ANALYSIS")
    print("="*50)
    print(f"Bounding box (grid indices):")
    print(f"  SDS: {sds_idx_min} to {sds_idx_max} ({grid_height} levels)")
    print(f"  TTAB: {ttab_idx_min} to {ttab_idx_max} ({grid_width} levels)")
    print(f"  Bounding box area: {grid_width} × {grid_height} = {bounding_box_area} grid positions")
    
    print(f"\\nActual coverage:")
    print(f"  High-info wells: {actual_coverage}/{bounding_box_area} positions")
    print(f"  Coverage efficiency: {coverage_efficiency:.1%}")
    
    print(f"\\nConcentration ranges:")
    print(f"  SDS: {sds_min:.2e} to {sds_max:.2e} mM ({sds_fold_range:.1f}-fold range)")
    print(f"  TTAB: {ttab_min:.2e} to {ttab_max:.2e} mM ({ttab_fold_range:.1f}-fold range)")
    
    print(f"\\nLog-space coverage:")
    print(f"  SDS log range: {log_sds_max - log_sds_min:.2f} log units")
    print(f"  TTAB log range: {log_ttab_max - log_ttab_min:.2f} log units") 
    print(f"  Total log area: {log_area:.2f} square log units")
    
    # Sampling density analysis
    current_grid_spacing_sds = np.log10(sds_concentrations[1] / sds_concentrations[0]) if len(sds_concentrations) > 1 else 0
    current_grid_spacing_ttab = np.log10(ttab_concentrations[1] / ttab_concentrations[0]) if len(ttab_concentrations) > 1 else 0
    
    print(f"\\nCurrent grid spacing:")
    print(f"  SDS: {current_grid_spacing_sds:.3f} log units = {10**current_grid_spacing_sds:.2f}× factor")
    print(f"  TTAB: {current_grid_spacing_ttab:.3f} log units = {10**current_grid_spacing_ttab:.2f}× factor")
    
    return {
        'bounding_box_grid': (grid_width, grid_height, bounding_box_area),
        'actual_coverage': actual_coverage,
        'coverage_efficiency': coverage_efficiency,
        'concentration_ranges': ((sds_min, sds_max), (ttab_min, ttab_max)),
        'fold_ranges': (sds_fold_range, ttab_fold_range),
        'log_area': log_area,
        'log_ranges': ((log_sds_min, log_sds_max), (log_ttab_min, log_ttab_max)),
        'current_spacing': (current_grid_spacing_sds, current_grid_spacing_ttab),
        'high_info_data': high_info_data
    }

def calculate_optimal_sampling_density(area_info, target_wells=96):
    """
    Calculate how many wells we can fit in the high-info region at different densities.
    """
    
    if not area_info:
        return {}
    
    log_area = area_info['log_area']
    current_spacing_sds, current_spacing_ttab = area_info['current_spacing']
    
    print(f"\\n" + "="*50)
    print("SAMPLING DENSITY OPTIONS")
    print("="*50)
    
    # Test different grid spacings
    spacing_options = [
        (0.5, "2× denser than current"),
        (0.25, "4× denser than current"), 
        (0.2, "5× denser than current"),
        (0.15, "6.7× denser than current"),
        (0.1, "10× denser than current")
    ]
    
    sampling_results = []
    
    for log_spacing, description in spacing_options:
        # Calculate number of points in each dimension
        sds_log_min, sds_log_max = area_info['log_ranges'][0]
        ttab_log_min, ttab_log_max = area_info['log_ranges'][1]
        
        n_sds_points = int((sds_log_max - sds_log_min) / log_spacing) + 1
        n_ttab_points = int((ttab_log_max - ttab_log_min) / log_spacing) + 1
        
        total_wells = n_sds_points * n_ttab_points
        
        # Account for coverage efficiency (not all bounding box positions are high-info)
        expected_wells_in_region = int(total_wells * area_info['coverage_efficiency'])
        
        sampling_results.append({
            'log_spacing': log_spacing,
            'description': description,
            'grid_dimensions': (n_sds_points, n_ttab_points),
            'total_bounding_box_wells': total_wells,
            'expected_high_info_wells': expected_wells_in_region,
            'concentration_factor': 10**log_spacing
        })
        
        print(f"Spacing {log_spacing:.2f} log units ({description}):")
        print(f"  Grid: {n_sds_points} × {n_ttab_points} = {total_wells} wells (bounding box)")
        print(f"  Expected high-info wells: ~{expected_wells_in_region} (after masking)")
        print(f"  Concentration factor: {10**log_spacing:.2f}×")
        print()
    
    # Find closest to target
    best_option = min(sampling_results, 
                     key=lambda x: abs(x['expected_high_info_wells'] - target_wells))
    
    print(f"Closest to {target_wells} wells: {best_option['description']}")
    print(f"  Expected: {best_option['expected_high_info_wells']} wells")
    print(f"  Spacing: {best_option['log_spacing']:.2f} log units")
    
    return sampling_results, best_option

def visualize_high_info_classification(high_info_wells, baseline_data, 
                                     sds_concentrations, ttab_concentrations,
                                     output_dir='output/high_info_classification'):
    """Visualize the high_info well classification."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate high_info and low_info wells
    high_info_data = [w for w in high_info_wells if w['is_high_info']]
    low_info_data = [w for w in high_info_wells if not w['is_high_info']]
    
    # Separate by baseline status for coloring
    baseline_low_info = [w for w in low_info_data if w['is_baseline']]
    baseline_high_info = [w for w in high_info_data if w['is_baseline']]
    nonbaseline_high_info = [w for w in high_info_data if not w['is_baseline']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Concentration space visualization
    if baseline_low_info:
        sds_coords = [w['sds_conc'] for w in baseline_low_info]
        ttab_coords = [w['ttab_conc'] for w in baseline_low_info] 
        ax1.scatter(sds_coords, ttab_coords, c='lightblue', s=50, alpha=0.7, 
                   marker='o', edgecolors='blue', linewidth=1,
                   label=f'Low-info baseline (n={len(baseline_low_info)})')
    
    if baseline_high_info:
        sds_coords = [w['sds_conc'] for w in baseline_high_info]
        ttab_coords = [w['ttab_conc'] for w in baseline_high_info]
        ax1.scatter(sds_coords, ttab_coords, c='orange', s=60, alpha=0.8, 
                   marker='^', edgecolors='darkorange', linewidth=1,
                   label=f'High-info baseline (n={len(baseline_high_info)})')
    
    if nonbaseline_high_info:
        sds_coords = [w['sds_conc'] for w in nonbaseline_high_info]
        ttab_coords = [w['ttab_conc'] for w in nonbaseline_high_info]
        ax1.scatter(sds_coords, ttab_coords, c='red', s=60, alpha=0.9, 
                   marker='s', edgecolors='darkred', linewidth=1,
                   label=f'High-info non-baseline (n={len(nonbaseline_high_info)})')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax1.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax1.set_title(f'High-Info Well Classification\\n{len(high_info_data)} high-info wells total', 
                 fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Grid structure visualization
    n_sds = len(sds_concentrations)
    n_ttab = len(ttab_concentrations)
    
    for well in high_info_wells:
        sds_idx, ttab_idx = well['grid_position']
        y_pos = n_sds - sds_idx - 1  # Flip for plotting
        
        if well['is_high_info']:
            if well['is_baseline']:
                color = 'orange'
                marker = '^'
                size = 80
            else:
                color = 'red' 
                marker = 's'
                size = 100
        else:
            color = 'lightblue'
            marker = 'o'
            size = 60
        
        ax2.scatter(ttab_idx, y_pos, c=color, s=size, marker=marker, alpha=0.8,
                   edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('TTAB Index', fontsize=12)
    ax2.set_ylabel('SDS Index (flipped)', fontsize=12)
    ax2.set_title('Grid Structure: High-Info Classification', 
                 fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add grid lines
    for i in range(n_ttab + 1):
        ax2.axvline(x=i-0.5, color='gray', alpha=0.3, linewidth=0.5)
    for j in range(n_sds + 1):
        ax2.axhline(y=j-0.5, color='gray', alpha=0.3, linewidth=0.5)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markersize=8, label='Low-info baseline'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', 
               markersize=10, label='High-info baseline'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
               markersize=10, label='High-info non-baseline')
    ]
    ax2.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    output_file = f'{output_dir}/high_info_classification.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\\nVisualization saved: {output_file}")
    
    plt.show()
    
    return len(high_info_data)

def main():
    """Main function to run high-info well classification."""
    
    data_file = r"C:\\Users\\owenm\\OneDrive\\Desktop\\North Robotics\\utoronto_demo\\utoronto_demo\\surfactant_grid_SDS_TTAB_20260209_164920\\surfactant_grid_SDS_TTAB_20260209_164920\\complete_experiment_results.csv"
    
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    print("Loading baseline classification...")
    baseline_data = load_baseline_classification(data_file)
    
    print("\\nOrganizing data by grid structure...")
    grid_map, sds_concentrations, ttab_concentrations = organize_data_by_grid(baseline_data)
    
    print("\\nClassifying high-info wells...")
    high_info_wells = classify_high_info_wells(grid_map, sds_concentrations, ttab_concentrations)
    
    print("\\nVisualizing high-info classification...")
    total_high_info = visualize_high_info_classification(high_info_wells, baseline_data, 
                                                        sds_concentrations, ttab_concentrations)
    
    print("\\nCalculating high-info region area...")
    area_info = calculate_high_info_area_and_sampling(high_info_wells, sds_concentrations, ttab_concentrations)
    
    print("\\nCalculating optimal sampling density...")
    sampling_results, best_option = calculate_optimal_sampling_density(area_info, target_wells=96)
    
    # Print classification breakdown
    nonbaseline_high_info = sum(1 for w in high_info_wells if w['is_high_info'] and not w['is_baseline'])
    baseline_high_info = sum(1 for w in high_info_wells if w['is_high_info'] and w['is_baseline'])
    
    print("\\n" + "="*60)
    print("HIGH-INFO WELL CLASSIFICATION RESULTS") 
    print("="*60)
    print(f"✓ {total_high_info}/{len(high_info_wells)} wells classified as high-info")
    print(f"✓ {nonbaseline_high_info} non-baseline wells (all high-info by definition)")
    print(f"✓ {baseline_high_info} baseline wells adjacent to non-baseline (high-info)")
    print(f"✓ {len(high_info_wells) - total_high_info} baseline wells not adjacent (low-info)")
    print("\\nClassification rule:")
    print("  high_info = non-baseline OR adjacent to any non-baseline well")
    
    if area_info and best_option:
        print("\\n" + "="*60)
        print("AREA & SAMPLING RECOMMENDATIONS")
        print("="*60)
        bounding_box = area_info['bounding_box_grid']
        print(f"✓ High-info region: {bounding_box[0]}×{bounding_box[1]} bounding box ({bounding_box[2]} positions)")
        print(f"✓ Actual coverage: {area_info['actual_coverage']} wells ({area_info['coverage_efficiency']:.1%} efficiency)")
        print(f"✓ Concentration ranges: {area_info['fold_ranges'][0]:.1f}× SDS, {area_info['fold_ranges'][1]:.1f}× TTAB")
        print(f"\\nRecommended sampling for ~96 wells:")
        print(f"✓ Grid spacing: {best_option['log_spacing']:.2f} log units ({best_option['concentration_factor']:.2f}× factors)")
        print(f"✓ Expected wells: {best_option['expected_high_info_wells']} in high-info region")
        print(f"✓ Grid dimensions: {best_option['grid_dimensions'][0]}×{best_option['grid_dimensions'][1]}")
    
    # Show some examples
    print("\\nExample classifications:")
    for i, well in enumerate(high_info_wells[:10]):  # Show first 10
        print(f"  Well {i+1}: {well['high_info_reason']}")
    
    if len(high_info_wells) > 10:
        print(f"  ... and {len(high_info_wells) - 10} more wells")

if __name__ == "__main__":
    main()
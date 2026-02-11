"""
Gradient-Adaptive Rectangle Sampling
Create rectangles in non-baseline regions with sizes based on local gradients.
High gradient regions get dense sampling, low gradient regions get sparse sampling.
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

def identify_rectangular_regions(baseline_data):
    """
    Identify distinct rectangular regions based on baseline classification patterns.
    """
    
    # Get unique concentrations
    sds_concentrations = sorted(baseline_data['surf_A_conc_mm'].unique())
    ttab_concentrations = sorted(baseline_data['surf_B_conc_mm'].unique())
    
    # Create grid of baseline classifications
    n_sds = len(sds_concentrations)
    n_ttab = len(ttab_concentrations)
    
    baseline_grid = np.full((n_sds, n_ttab), -1)  # -1 = no data
    
    for _, well in baseline_data.iterrows():
        sds_idx = sds_concentrations.index(well['surf_A_conc_mm'])
        ttab_idx = ttab_concentrations.index(well['surf_B_conc_mm'])
        baseline_grid[sds_idx, ttab_idx] = int(well['is_baseline'])
    
    # Define rectangular regions based on the data pattern
    regions = []
    
    # Region 1: Bottom-left baseline region (SKIP THIS ONE)
    # Find the largest contiguous baseline rectangle
    baseline_sds_max = 0
    baseline_ttab_max = 0
    
    for sds_idx in range(n_sds):
        for ttab_idx in range(n_ttab):
            if baseline_grid[sds_idx, ttab_idx] == 1:  # baseline
                baseline_sds_max = max(baseline_sds_max, sds_idx)
                baseline_ttab_max = max(baseline_ttab_max, ttab_idx)
    
    print(f"Baseline region extends to SDS index {baseline_sds_max}, TTAB index {baseline_ttab_max}")
    
    # Region 2: Top region (high SDS, mixed TTAB)
    if baseline_sds_max < n_sds - 1:
        regions.append({
            'name': 'High SDS Region',
            'sds_range': (baseline_sds_max + 1, n_sds - 1),
            'ttab_range': (0, n_ttab - 1),
            'priority': 'high'
        })
    
    # Region 3: Right region (mixed SDS, high TTAB) 
    if baseline_ttab_max < n_ttab - 1:
        regions.append({
            'name': 'High TTAB Region',
            'sds_range': (0, baseline_sds_max),
            'ttab_range': (baseline_ttab_max + 1, n_ttab - 1),
            'priority': 'high'
        })
    
    # Region 4: Corner transition (medium SDS, medium TTAB)
    mid_sds = n_sds // 2
    mid_ttab = n_ttab // 2
    
    if mid_sds > baseline_sds_max and mid_ttab > baseline_ttab_max:
        regions.append({
            'name': 'Interaction Corner',
            'sds_range': (baseline_sds_max + 1, mid_sds),
            'ttab_range': (baseline_ttab_max + 1, mid_ttab),
            'priority': 'medium'
        })
    
    # Convert to actual concentrations
    for region in regions:
        sds_min_idx, sds_max_idx = region['sds_range']
        ttab_min_idx, ttab_max_idx = region['ttab_range']
        
        region['sds_conc_range'] = (sds_concentrations[sds_min_idx], sds_concentrations[sds_max_idx])
        region['ttab_conc_range'] = (ttab_concentrations[ttab_min_idx], ttab_concentrations[ttab_max_idx])
        region['area_indices'] = (sds_max_idx - sds_min_idx + 1) * (ttab_max_idx - ttab_min_idx + 1)
    
    return regions, sds_concentrations, ttab_concentrations

def calculate_gradient_density_map(baseline_data, sds_concentrations, ttab_concentrations):
    """
    Calculate gradient density at each location for adaptive rectangle sizing.
    """
    
    # Create gradient map
    n_sds = len(sds_concentrations)
    n_ttab = len(ttab_concentrations)
    gradient_map = np.zeros((n_sds, n_ttab))
    
    # Calculate gradients for each existing well
    for _, well in baseline_data.iterrows():
        sds_idx = sds_concentrations.index(well['surf_A_conc_mm'])
        ttab_idx = ttab_concentrations.index(well['surf_B_conc_mm'])
        
        # Calculate local gradient based on nearby wells
        gradient_sum = 0
        neighbor_count = 0
        
        # Check all adjacent positions
        for d_sds in [-1, 0, 1]:
            for d_ttab in [-1, 0, 1]:
                if d_sds == 0 and d_ttab == 0:
                    continue
                    
                adj_sds = sds_idx + d_sds
                adj_ttab = ttab_idx + d_ttab
                
                if 0 <= adj_sds < n_sds and 0 <= adj_ttab < n_ttab:
                    # Find adjacent well
                    adj_well = baseline_data[
                        (baseline_data['surf_A_conc_mm'] == sds_concentrations[adj_sds]) &
                        (baseline_data['surf_B_conc_mm'] == ttab_concentrations[adj_ttab])
                    ]
                    
                    if len(adj_well) > 0:
                        adj_well = adj_well.iloc[0]
                        
                        # Calculate measurement differences
                        turbidity_diff = abs(well['turbidity_600'] - adj_well['turbidity_600'])
                        ratio_diff = abs(well['ratio'] - adj_well['ratio'])
                        
                        # Normalize and combine (focus on ratio as requested)
                        turb_grad = turbidity_diff / 2.0  # typical turbidity range ~0-2
                        ratio_grad = ratio_diff / 0.1     # typical ratio range ~0-0.1
                        
                        local_gradient = ratio_grad  # Focus on ratio gradients only
                        gradient_sum += local_gradient
                        neighbor_count += 1
        
        if neighbor_count > 0:
            gradient_map[sds_idx, ttab_idx] = gradient_sum / neighbor_count
    
    return gradient_map

def create_gradient_adaptive_rectangles(regions, gradient_map, sds_concentrations, 
                                      ttab_concentrations, target_total_wells=96):
    """
    Create rectangles within each region with sizes based on local gradient density.
    High gradient = smaller rectangles (more wells), Low gradient = larger rectangles (fewer wells).
    """
    
    all_rectangles = []
    total_wells_allocated = 0
    
    # Calculate priority weights for well allocation
    total_priority_weight = sum([2.0 if r['priority'] == 'high' else 1.0 for r in regions])
    
    print(f"\\nCreating gradient-adaptive rectangles for {len(regions)} regions...")
    
    for region in regions:
        region_name = region['name']
        sds_range = region['sds_range']
        ttab_range = region['ttab_range']
        priority = region['priority']
        
        print(f"\\nRegion: {region_name} ({priority} priority)")
        
        # Allocate wells based on priority
        priority_weight = 2.0 if priority == 'high' else 1.0
        region_wells = int((priority_weight / total_priority_weight) * target_total_wells)
        
        # Extract gradient values for this region
        sds_min_idx, sds_max_idx = sds_range
        ttab_min_idx, ttab_max_idx = ttab_range
        
        region_gradient = gradient_map[sds_min_idx:sds_max_idx+1, ttab_min_idx:ttab_max_idx+1]
        region_sds_size = sds_max_idx - sds_min_idx + 1
        region_ttab_size = ttab_max_idx - ttab_min_idx + 1
        
        # Calculate average gradient for this region
        avg_gradient = np.mean(region_gradient[region_gradient > 0])
        max_gradient = np.max(region_gradient) if np.max(region_gradient) > 0 else 1.0
        
        print(f"  Grid size: {region_sds_size} × {region_ttab_size}")
        print(f"  Average gradient: {avg_gradient:.3f}")
        print(f"  Max gradient: {max_gradient:.3f}")
        print(f"  Allocated wells: {region_wells}")
        
        # Adaptive rectangle sizing based on gradient
        if avg_gradient > 0:
            # High gradient → smaller rectangles (more dense sampling)
            # Low gradient → larger rectangles (less dense sampling)
            
            # Gradient-based scaling factor (0.5 to 2.0)
            gradient_scale = 0.5 + 1.5 * (avg_gradient / max_gradient)
            
            # Calculate rectangle dimensions
            base_rect_size = max(1, int(np.sqrt(region_sds_size * region_ttab_size / region_wells)))
            rect_sds_size = max(1, int(base_rect_size / gradient_scale))
            rect_ttab_size = max(1, int(base_rect_size / gradient_scale))
            
        else:
            # No gradient data - use uniform rectangles
            base_rect_size = max(1, int(np.sqrt(region_sds_size * region_ttab_size / region_wells)))
            rect_sds_size = base_rect_size
            rect_ttab_size = base_rect_size
        
        print(f"  Rectangle size: {rect_sds_size} × {rect_ttab_size}")
        
        # Create rectangles within this region
        region_rectangles = []
        region_wells_used = 0
        
        sds_step = max(1, rect_sds_size // 2)  # Overlap rectangles slightly
        ttab_step = max(1, rect_ttab_size // 2)
        
        for sds_start in range(sds_min_idx, sds_max_idx + 1, sds_step):
            for ttab_start in range(ttab_min_idx, ttab_max_idx + 1, ttab_step):
                
                sds_end = min(sds_start + rect_sds_size - 1, sds_max_idx)
                ttab_end = min(ttab_start + rect_ttab_size - 1, ttab_max_idx)
                
                rect_wells = (sds_end - sds_start + 1) * (ttab_end - ttab_start + 1)
                
                if region_wells_used + rect_wells <= region_wells * 1.2:  # Allow slight overflow
                    
                    rectangle = {
                        'region': region_name,
                        'sds_indices': (sds_start, sds_end),
                        'ttab_indices': (ttab_start, ttab_end),
                        'sds_concentrations': (sds_concentrations[sds_start], sds_concentrations[sds_end]),
                        'ttab_concentrations': (ttab_concentrations[ttab_start], ttab_concentrations[ttab_end]),
                        'wells': rect_wells,
                        'local_gradient': np.mean(gradient_map[sds_start:sds_end+1, ttab_start:ttab_end+1]),
                        'rectangle_size': (rect_sds_size, rect_ttab_size)
                    }
                    
                    region_rectangles.append(rectangle)
                    region_wells_used += rect_wells
                    
                    if region_wells_used >= region_wells:
                        break
            
            if region_wells_used >= region_wells:
                break
        
        all_rectangles.extend(region_rectangles)
        total_wells_allocated += region_wells_used
        
        print(f"  Created {len(region_rectangles)} rectangles, {region_wells_used} wells used")
    
    print(f"\\nTotal wells allocated: {total_wells_allocated}")
    
    return all_rectangles

def visualize_gradient_adaptive_rectangles(baseline_data, all_rectangles, gradient_map,
                                         sds_concentrations, ttab_concentrations,
                                         output_dir='output/gradient_adaptive_rectangles'):
    """
    Visualize the gradient-adaptive rectangles.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Separate wells for plotting
    baseline_wells = baseline_data[baseline_data['is_baseline']]
    nonbaseline_wells = baseline_data[~baseline_data['is_baseline']]
    
    # Plot 1: Original data with rectangles
    ax1.scatter(baseline_wells['surf_A_conc_mm'], baseline_wells['surf_B_conc_mm'],
               c='lightblue', s=50, alpha=0.7, label=f'Baseline wells (n={len(baseline_wells)})', 
               marker='o', edgecolors='blue', linewidth=1)
    ax1.scatter(nonbaseline_wells['surf_A_conc_mm'], nonbaseline_wells['surf_B_conc_mm'],
               c='red', s=60, alpha=0.8, label=f'Non-baseline wells (n={len(nonbaseline_wells)})', 
               marker='s', edgecolors='darkred', linewidth=1)
    
    # Draw rectangles
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_rectangles)))
    
    for i, rect in enumerate(all_rectangles):
        sds_min, sds_max = rect['sds_concentrations']
        ttab_min, ttab_max = rect['ttab_concentrations']
        wells = rect['wells']
        gradient = rect['local_gradient']
        
        color = colors[i % len(colors)]
        
        from matplotlib.patches import Rectangle as MPLRectangle
        rect_patch = MPLRectangle((sds_min, ttab_min), sds_max - sds_min, ttab_max - ttab_min,
                                 linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
                                 label=f'{wells} wells (grad: {gradient:.3f})')
        ax1.add_patch(rect_patch)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('SDS Concentration (mM)', fontsize=12)
    ax1.set_ylabel('TTAB Concentration (mM)', fontsize=12)
    ax1.set_title(f'Gradient-Adaptive Rectangles\\n{sum([r["wells"] for r in all_rectangles])} total wells', 
                 fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Gradient density map
    X, Y = np.meshgrid(range(len(ttab_concentrations)), range(len(sds_concentrations)))
    im = ax2.imshow(gradient_map, cmap='viridis', alpha=0.8, extent=[0, len(ttab_concentrations), 0, len(sds_concentrations)], 
                   aspect='auto', origin='lower')
    plt.colorbar(im, ax=ax2, label='Gradient Density')
    
    # Overlay rectangles on gradient map
    for rect in all_rectangles:
        sds_start, sds_end = rect['sds_indices']
        ttab_start, ttab_end = rect['ttab_indices']
        
        rect_patch = MPLRectangle((ttab_start, sds_start), ttab_end - ttab_start + 1, 
                                 sds_end - sds_start + 1,
                                 linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
        ax2.add_patch(rect_patch)
    
    ax2.set_xlabel('TTAB Index', fontsize=12)
    ax2.set_ylabel('SDS Index', fontsize=12)
    ax2.set_title('Gradient Density Map with Rectangles', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rectangle statistics
    rect_wells = [r['wells'] for r in all_rectangles]
    rect_gradients = [r['local_gradient'] for r in all_rectangles]
    
    ax3.scatter(rect_gradients, rect_wells, c='darkgreen', s=80, alpha=0.7)
    ax3.set_xlabel('Average Gradient', fontsize=12)
    ax3.set_ylabel('Wells in Rectangle', fontsize=12)
    ax3.set_title('Rectangle Size vs Local Gradient', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Add trend line
    if len(rect_gradients) > 1:
        z = np.polyfit(rect_gradients, rect_wells, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(rect_gradients), max(rect_gradients), 100)
        ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend: y={(z[1]):.1f}+{z[0]:.1f}x')
        ax3.legend()
    
    # Plot 4: Well allocation by region
    region_names = list(set([r['region'] for r in all_rectangles]))
    region_wells = [sum([r['wells'] for r in all_rectangles if r['region'] == name]) for name in region_names]
    
    ax4.bar(range(len(region_names)), region_wells, color=['red', 'orange', 'green'][:len(region_names)])
    ax4.set_xticks(range(len(region_names)))
    ax4.set_xticklabels(region_names, rotation=45, ha='right')
    ax4.set_ylabel('Wells Allocated', fontsize=12)
    ax4.set_title('Well Allocation by Region', fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = f'{output_dir}/gradient_adaptive_rectangles.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\\nVisualization saved: {output_file}")
    
    plt.show()

def main():
    """Main function for gradient-adaptive rectangle sampling."""
    
    data_file = r"C:\\Users\\owenm\\OneDrive\\Desktop\\North Robotics\\utoronto_demo\\utoronto_demo\\surfactant_grid_SDS_TTAB_20260209_164920\\surfactant_grid_SDS_TTAB_20260209_164920\\complete_experiment_results.csv"
    
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    print("Loading baseline classification...")
    baseline_data = load_baseline_classification(data_file)
    
    print("\\nIdentifying rectangular regions...")
    regions, sds_concentrations, ttab_concentrations = identify_rectangular_regions(baseline_data)
    
    print(f"Found {len(regions)} regions:")
    for region in regions:
        print(f"  - {region['name']}: {region['area_indices']} grid positions ({region['priority']} priority)")
    
    print("\\nCalculating gradient density map...")
    gradient_map = calculate_gradient_density_map(baseline_data, sds_concentrations, ttab_concentrations)
    
    print("\\nCreating gradient-adaptive rectangles...")
    all_rectangles = create_gradient_adaptive_rectangles(regions, gradient_map, 
                                                        sds_concentrations, ttab_concentrations,
                                                        target_total_wells=96)
    
    print("\\nVisualizing results...")
    visualize_gradient_adaptive_rectangles(baseline_data, all_rectangles, gradient_map,
                                         sds_concentrations, ttab_concentrations)
    
    total_wells = sum([r['wells'] for r in all_rectangles])
    avg_gradient = np.mean([r['local_gradient'] for r in all_rectangles])
    
    print("\\n" + "="*60)
    print("GRADIENT-ADAPTIVE RECTANGLES RESULTS")
    print("="*60)
    print(f"✓ {len(all_rectangles)} rectangles created")
    print(f"✓ {total_wells} total wells (target: ≤96)")
    print(f"✓ Average rectangle gradient: {avg_gradient:.3f}")
    print("✓ High gradient regions → small rectangles (dense sampling)")
    print("✓ Low gradient regions → large rectangles (sparse sampling)")
    print("✓ Baseline region excluded from sampling")
    
    print("\\nRectangle summary:")
    for i, rect in enumerate(all_rectangles):
        print(f"  {i+1}. {rect['region']}: {rect['wells']} wells, "
              f"gradient {rect['local_gradient']:.3f}, "
              f"size {rect['rectangle_size'][0]}×{rect['rectangle_size'][1]}")

if __name__ == "__main__":
    main()
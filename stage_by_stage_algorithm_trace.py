# -*- coding: utf-8 -*-
"""
Stage-by-Stage Algorithm Trace with REAL Simulation Function
Shows exactly what the vector edge algorithm recommends using the actual simulation function from the workflow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def simulate_surfactant_measurements(surf_a_conc, surf_b_conc, add_noise=True):
    """
    REAL simulation function copied from the workflow - NOT FAKE DATA!
    Generate realistic simulation data for turbidity and ratio based on surfactant concentrations.
    """
    import numpy as np
    
    # Work in log space for realistic concentration effects
    log_a = np.log10(surf_a_conc)
    log_b = np.log10(surf_b_conc)
    
    # RATIO SIMULATION: Diagonal boundary (sigmoid transition)
    # Boundary runs from (-6, -4) to (-3, -1) in log space
    diagonal_distance = (log_a + log_b + 5.0) / np.sqrt(2)  # Distance from diagonal line
    ratio_transition = 1.0 / (1.0 + np.exp(-8.0 * diagonal_distance))  # Sharp sigmoid
    ratio_baseline = 0.6   # Low ratio state
    ratio_elevated = 1.4   # High ratio state 
    simulated_ratio = ratio_baseline + (ratio_elevated - ratio_baseline) * ratio_transition
    
    # TURBIDITY SIMULATION: Circular boundary (different center)
    # Circle centered at (-4.5, -2.5) in log space with radius 1.2
    center_a, center_b = -4.5, -2.5
    radius_distance = np.sqrt((log_a - center_a)**2 + (log_b - center_b)**2)
    turbidity_transition = 1.0 / (1.0 + np.exp(-5.0 * (radius_distance - 1.2)))
    turbidity_baseline = 0.15  # Low turbidity state
    turbidity_elevated = 0.85  # High turbidity state
    simulated_turbidity = turbidity_baseline + (turbidity_elevated - turbidity_baseline) * turbidity_transition
    
    # FLUORESCENCE: Derive from ratio (realistic relationship)
    # F384 stays relatively constant, F373 varies with ratio
    f384_base = 95.0 + 10.0 * np.sin(log_a + log_b)  # Slight spatial variation
    f373_base = simulated_ratio * f384_base  # F373/F384 = ratio
    
    # Add realistic experimental noise if requested
    if add_noise:
        noise_scale = 0.01  # 1% coefficient of variation (reduced from 5%)
        ratio_noise = 1.0 + np.random.normal(0, noise_scale)
        turbidity_noise = 1.0 + np.random.normal(0, noise_scale)
        fluorescence_noise = 1.0 + np.random.normal(0, noise_scale * 0.5)  # Lower noise for fluorescence
        
        simulated_ratio *= ratio_noise
        simulated_turbidity *= turbidity_noise
        f373_base *= fluorescence_noise
        f384_base *= fluorescence_noise
        
        # Recalculate ratio from potentially noisy fluorescence
        simulated_ratio = f373_base / f384_base if f384_base > 0 else simulated_ratio
    
    # Ensure physically reasonable bounds
    simulated_ratio = max(0.1, min(3.0, simulated_ratio))
    simulated_turbidity = max(0.01, min(1.5, simulated_turbidity))
    f373_base = max(10.0, min(300.0, f373_base))
    f384_base = max(10.0, min(300.0, f384_base))
    
    return {
        'turbidity_600': round(simulated_turbidity, 4),
        'fluorescence_334_373': round(f373_base, 2),
        'fluorescence_334_384': round(f384_base, 2),
        'ratio': round(simulated_ratio, 4)
    }

def create_initial_5x5_grid():
    """
    Create the initial 5x5 concentration grid using the same parameters as the workflow.
    Uses log spacing from MIN_CONC (10^-4 mM) to ~25 mM for both surfactants.
    """
    MIN_CONC = 10**-4  # 0.0001 mM minimum concentration (from workflow)
    MAX_CONC = 22.5    # Representative max from your data
    
    # Create log-spaced concentrations (5 points each)
    log_min = np.log10(MIN_CONC)
    log_max = np.log10(MAX_CONC)
    concentrations = np.logspace(log_min, log_max, 5)
    
    print(f"5x5 Grid Concentrations (log-spaced from {MIN_CONC:.1e} to {MAX_CONC:.1f} mM):")
    print(f"  {[f'{c:.6f}' for c in concentrations]}")
    
    # Create all combinations (5x5 = 25 points)
    surf_a_list = []
    surf_b_list = []
    
    for i, surf_a_conc in enumerate(concentrations):
        for j, surf_b_conc in enumerate(concentrations):
            surf_a_list.append(surf_a_conc)
            surf_b_list.append(surf_b_conc)
    
    return surf_a_list, surf_b_list

def simulate_measurements_for_grid(surf_a_list, surf_b_list):
    """Simulate measurements for a list of concentration pairs using the REAL simulation function."""
    data_rows = []
    
    for surf_a_conc, surf_b_conc in zip(surf_a_list, surf_b_list):
        # Use the REAL simulation function from the workflow
        measurements = simulate_surfactant_measurements(surf_a_conc, surf_b_conc, add_noise=True)
        
        # Create data row in same format as workflow CSV
        row = {
            'surf_A_conc_mm': surf_a_conc,
            'surf_B_conc_mm': surf_b_conc,
            'turbidity_600': measurements['turbidity_600'],
            'fluorescence_334_373': measurements['fluorescence_334_373'],
            'fluorescence_334_384': measurements['fluorescence_334_384'],
            'ratio': measurements['ratio'],
            'well_type': 'experiment'
        }
        data_rows.append(row)
    
    return pd.DataFrame(data_rows)

def trace_real_iterative_process():
    """
    Trace the algorithm through multiple iterative cycles with edge visualizations.
    Shows exactly what the algorithm "sees" at each stage with edge score plots.
    """
    print("=== REAL MULTI-CYCLE ALGORITHM TRACE WITH EDGE VISUALIZATION ===")
    print("Using ACTUAL simulation function from workflow")
    print("Will show edge visualizations at each cycle")
    
    # Import the recommender
    try:
        sys.path.append(".")
        from recommenders.generalized_vector_edge_recommender import GeneralizedVectorEdgeRecommender
        print("✓ Successfully imported GeneralizedVectorEdgeRecommender")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return
    
    # Configure recommender (same as your workflow)
    recommender = GeneralizedVectorEdgeRecommender(
        input_columns=['surf_A_conc_mm', 'surf_B_conc_mm'],
        output_columns=['ratio'],  # Focus on ratio only
        log_transform_inputs=True,
        normalization_method='log_zscore'
    )
    
    # === CYCLE 1: Initial 5x5 Grid ===
    print(f"\n" + "="*70)
    print(f"CYCLE 1: Initial 5x5 Grid → First Algorithm Recommendations")
    print(f"="*70)
    
    # Create initial 5x5 grid 
    surf_a_list, surf_b_list = create_initial_5x5_grid()
    cycle1_data = simulate_measurements_for_grid(surf_a_list, surf_b_list)
    
    print(f"Cycle 1 data: {len(cycle1_data)} points")
    print(f"SDS range: {cycle1_data['surf_A_conc_mm'].min():.6f} - {cycle1_data['surf_A_conc_mm'].max():.6f} mM")
    print(f"Ratio range: {cycle1_data['ratio'].min():.3f} - {cycle1_data['ratio'].max():.3f}")
    
    # Show edge visualization for cycle 1
    cycle1_edges = visualize_edges_for_cycle(cycle1_data, recommender, cycle_num=1, recommendations_df=None)
    
    # Get cycle 1 recommendations  
    print(f"\n--- Getting 8 Recommendations from Cycle 1 ---")
    cycle1_recs = recommender.get_recommendations(
        cycle1_data,
        n_points=8,  # Reduced from 12
        min_spacing_factor=0.5,  # Allow boundary clustering
        output_dir=None,
        create_visualization=False
    )
    
    print(f"✓ Cycle 1: Algorithm returned {len(cycle1_recs)} recommendations")
    if len(cycle1_recs) > 0:
        print(f"Cycle 1 Top Recommendations:")
        for i, row in cycle1_recs.head(6).iterrows():
            sds = row['surf_A_conc_mm']
            ttab = row['surf_B_conc_mm']
            score = row.get('boundary_score', 'N/A')
            print(f"  {i+1}: SDS={sds:.6f} mM, TTAB={ttab:.6f} mM (Score: {score})")
        
        # Update cycle 1 visualization to show the picks
        print(f"\n--- Updating Cycle 1 Visualization with Algorithm Picks ---")
        cycle1_edges = visualize_edges_for_cycle(cycle1_data, recommender, cycle_num=1, recommendations_df=cycle1_recs)
    else:
        print("✗ No recommendations from Cycle 1!")
        return
    
    # === CYCLE 2: Add Cycle 1 Measurements ===
    print(f"\n" + "="*70)
    print(f"CYCLE 2: Measure Cycle 1 Picks → Next Recommendations")
    print(f"="*70)
    
    # Measure cycle 1 recommendations
    cycle1_measured = simulate_measurements_for_grid(
        cycle1_recs['surf_A_conc_mm'].tolist(),
        cycle1_recs['surf_B_conc_mm'].tolist()
    )
    
    cycle2_data = pd.concat([cycle1_data, cycle1_measured], ignore_index=True)
    print(f"Cycle 2 data: {len(cycle2_data)} points ({len(cycle1_data)} + {len(cycle1_measured)} new)")
    
    # Get cycle 2 recommendations
    print(f"\n--- Getting 8 Recommendations from Cycle 2 ---")
    cycle2_recs = recommender.get_recommendations(
        cycle2_data,
        n_points=8,
        min_spacing_factor=0.5,
        output_dir=None,
        create_visualization=False
    )
    
    print(f"✓ Cycle 2: Algorithm returned {len(cycle2_recs)} recommendations")
    if len(cycle2_recs) > 0:
        print(f"Cycle 2 Top Recommendations:")
        for i, row in cycle2_recs.head(6).iterrows():
            sds = row['surf_A_conc_mm']
            ttab = row['surf_B_conc_mm']
            score = row.get('boundary_score', 'N/A')
            print(f"  {i+1}: SDS={sds:.6f} mM, TTAB={ttab:.6f} mM (Score: {score})")
    
    # Show edge visualization for cycle 2 WITH recommendations
    cycle2_edges = visualize_edges_for_cycle(cycle2_data, recommender, cycle_num=2, recommendations_df=cycle2_recs)
    
    # === CYCLE 3: Add Cycle 2 Measurements ===
    print(f"\n" + "="*70)
    print(f"CYCLE 3: Measure Cycle 2 Picks → Further Refinement")
    print(f"="*70)
    
    # Measure cycle 2 recommendations  
    cycle2_measured = simulate_measurements_for_grid(
        cycle2_recs['surf_A_conc_mm'].tolist(),
        cycle2_recs['surf_B_conc_mm'].tolist()
    )
    
    cycle3_data = pd.concat([cycle2_data, cycle2_measured], ignore_index=True)
    print(f"Cycle 3 data: {len(cycle3_data)} points (total accumulated)")
    
    # Get cycle 3 recommendations
    print(f"\n--- Getting 8 Recommendations from Cycle 3 ---")
    cycle3_recs = recommender.get_recommendations(
        cycle3_data,
        n_points=8,
        min_spacing_factor=0.5,
        output_dir=None,
        create_visualization=False
    )
    
    print(f"✓ Cycle 3: Algorithm returned {len(cycle3_recs)} recommendations")
    if len(cycle3_recs) > 0:
        print(f"Cycle 3 Top Recommendations:")
        for i, row in cycle3_recs.head(6).iterrows():
            sds = row['surf_A_conc_mm']
            ttab = row['surf_B_conc_mm']
            score = row.get('boundary_score', 'N/A')
            print(f"  {i+1}: SDS={sds:.6f} mM, TTAB={ttab:.6f} mM (Score: {score})")
    
    # Show edge visualization for cycle 3 WITH recommendations
    cycle3_edges = visualize_edges_for_cycle(cycle3_data, recommender, cycle_num=3, recommendations_df=cycle3_recs)
    
    # === CYCLE 4: Final Refinement ===
    print(f"\n" + "="*70)
    print(f"CYCLE 4: Final Boundary Refinement")
    print(f"="*70)
    
    # Measure cycle 3 recommendations  
    cycle3_measured = simulate_measurements_for_grid(
        cycle3_recs['surf_A_conc_mm'].tolist(),
        cycle3_recs['surf_B_conc_mm'].tolist()
    )
    
    cycle4_data = pd.concat([cycle3_data, cycle3_measured], ignore_index=True)
    print(f"Cycle 4 data: {len(cycle4_data)} points (total accumulated)")
    
    # Get final recommendations
    print(f"\n--- Getting Final 8 Recommendations from Cycle 4 ---")
    cycle4_recs = recommender.get_recommendations(
        cycle4_data,
        n_points=8,
        min_spacing_factor=0.5,
        output_dir=None,
        create_visualization=False
    )
    
    print(f"✓ Cycle 4: Algorithm returned {len(cycle4_recs)} recommendations")
    if len(cycle4_recs) > 0:
        print(f"Cycle 4 Final Recommendations:")
        for i, row in cycle4_recs.head(6).iterrows():
            sds = row['surf_A_conc_mm']
            ttab = row['surf_B_conc_mm']
            score = row.get('boundary_score', 'N/A')
            print(f"  {i+1}: SDS={sds:.6f} mM, TTAB={ttab:.6f} mM (Score: {score})")
    
    # Show edge visualization for cycle 4 WITH recommendations
    cycle4_edges = visualize_edges_for_cycle(cycle4_data, recommender, cycle_num=4, recommendations_df=cycle4_recs)
    
    print(f"\n" + "="*70)
    print(f"MULTI-CYCLE SUMMARY")
    print(f"="*70)
    print(f"Cycle 1: Started with 5x5 grid → Found {len(cycle1_recs)} boundary points")
    print(f"Cycle 2: Added {len(cycle1_recs)} → Found {len(cycle2_recs)} refined points")
    print(f"Cycle 3: Added {len(cycle2_recs)} → Found {len(cycle3_recs)} further refinements")
    print(f"Cycle 4: Added {len(cycle3_recs)} → Found {len(cycle4_recs)} final recommendations")
    print(f"Total data points accumulated: {len(cycle4_data)}")
    print(f"Algorithm successfully adapted and focused on boundary regions!")
    
    return cycle1_recs, cycle2_recs, cycle3_recs, cycle4_recs

def visualize_edges_for_cycle(data_df, recommender, cycle_num, recommendations_df=None):
    """
    Create edge visualization for a specific cycle showing algorithm's boundary detection.
    Shows data points + edges colored by score intensity + top algorithm picks.
    """
    print(f"\nCycle {cycle_num} Edge Visualization:")
    
    # Get normalized data and internal grid structure  
    normalized_data = recommender._normalize_outputs(data_df)
    
    if recommender.log_transform_inputs:
        normalized_data['log_surf_A_conc_mm'] = np.log10(normalized_data['surf_A_conc_mm'])
        normalized_data['log_surf_B_conc_mm'] = np.log10(normalized_data['surf_B_conc_mm'])
    
    # Get internal grid data and calculate edges (same as algorithm)
    grid_data, grid_coordinates = recommender._create_grid_structure(normalized_data)
    
    # Calculate edges exactly like the algorithm does
    edges = []
    grid_sizes = [len(grid_coordinates[col]) for col in recommender.input_columns]
    
    for dim in range(recommender.n_inputs):
        dim_name = recommender.input_columns[dim]
        
        for grid_indices in np.ndindex(*grid_sizes):
            if grid_indices[dim] >= grid_sizes[dim] - 1:
                continue
            
            neighbor_indices = list(grid_indices)
            neighbor_indices[dim] += 1
            neighbor_indices = tuple(neighbor_indices)
            
            if grid_indices in grid_data and neighbor_indices in grid_data:
                data1 = grid_data[grid_indices]
                data2 = grid_data[neighbor_indices]
                
                # Calculate vector difference (same as algorithm)
                output_diffs = []
                for col in recommender.output_columns:
                    norm_col = f'{col}_normalized'
                    diff = data2[norm_col] - data1[norm_col]
                    output_diffs.append(diff)
                
                score = np.sqrt(sum(diff**2 for diff in output_diffs))
                
                edge = {
                    'x1': data1['log_surf_A_conc_mm'],
                    'y1': data1['log_surf_B_conc_mm'], 
                    'x2': data2['log_surf_A_conc_mm'],
                    'y2': data2['log_surf_B_conc_mm'],
                    'score': score,
                    'direction': dim_name,
                    'midpoint_x': (data1['log_surf_A_conc_mm'] + data2['log_surf_A_conc_mm']) / 2,
                    'midpoint_y': (data1['log_surf_B_conc_mm'] + data2['log_surf_B_conc_mm']) / 2,
                }
                edges.append(edge)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot data points
    log_sds = normalized_data['log_surf_A_conc_mm']
    log_ttab = normalized_data['log_surf_B_conc_mm']
    
    scatter = ax.scatter(log_sds, log_ttab, c=normalized_data['ratio'], 
                        cmap='RdYlBu', s=150, alpha=0.8, edgecolors='black', 
                        linewidth=1.5, zorder=5, label='Data Points')
    
    # Plot edges colored by algorithm score
    if edges:
        max_score = max(edge['score'] for edge in edges)
        min_score = min(edge['score'] for edge in edges)
        
        for edge in edges:
            # Color based on score (red = high score = strong boundary)
            if max_score > min_score:
                score_norm = (edge['score'] - min_score) / (max_score - min_score) 
            else:
                score_norm = 0
            color_intensity = plt.cm.Reds(0.3 + 0.7 * score_norm)
            linewidth = 1 + 6 * score_norm  # Thicker lines = higher scores
            
            ax.plot([edge['x1'], edge['x2']], [edge['y1'], edge['y2']], 
                    color=color_intensity, linewidth=linewidth, alpha=0.7, zorder=3)
            
            # Add score text on high-score edges
            if edge['score'] > 0.7 * max_score:
                ax.text(edge['midpoint_x'], edge['midpoint_y'], f'{edge["score"]:.2f}', 
                        fontsize=9, fontweight='bold', color='darkred', 
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8),
                        ha='center', va='center', zorder=6)
    
    # Draw human-visible boundary for reference
    boundary_x = np.linspace(log_sds.min(), log_sds.max(), 100)
    boundary_y = -4.5 + 0.5 * boundary_x  # Approximate diagonal
    mask = (boundary_y >= log_ttab.min()) & (boundary_y <= log_ttab.max())
    ax.plot(boundary_x[mask], boundary_y[mask], 'k--', linewidth=2, 
            label='Expected Boundary', alpha=0.7, zorder=2)
    
    # Plot algorithm recommendations if provided
    if recommendations_df is not None and len(recommendations_df) > 0:
        rec_log_sds = np.log10(recommendations_df['surf_A_conc_mm'])
        rec_log_ttab = np.log10(recommendations_df['surf_B_conc_mm'])
        
        # Plot recommendations as big red X's
        ax.scatter(rec_log_sds, rec_log_ttab, c='red', s=200, marker='X', 
                  linewidth=3, label=f'Algorithm Picks ({len(recommendations_df)})', 
                  edgecolors='darkred', zorder=10)
        
        # Add numbers to recommendations
        for i, (x, y) in enumerate(zip(rec_log_sds, rec_log_ttab)):
            if i < 8:  # Only label first 8
                ax.annotate(f'{i+1}', (x, y), xytext=(10, 10), textcoords='offset points',
                          fontsize=12, fontweight='bold', color='white', 
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8),
                          ha='center', va='center', zorder=11)
        
        print(f"  ✓ Added {len(recommendations_df)} algorithm recommendations to plot")
    
    ax.set_xlabel('log10(SDS Concentration [mM])')
    ax.set_ylabel('log10(TTAB Concentration [mM])')
    ax.set_title(f'Cycle {cycle_num}: Algorithm Edge Detection + Picks\\n(Red/Thick = High Boundary Score, Red X = Algorithm Pick)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Ratio')
    
    # Save plot
    os.makedirs("debug_analysis", exist_ok=True)
    plot_filename = f"cycle{cycle_num}_edges_and_picks.png"
    plt.savefig(f"debug_analysis/{plot_filename}", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: debug_analysis/{plot_filename}")
    
    # Print edge summary
    if edges:
        max_score = max(edge['score'] for edge in edges)
        min_score = min(edge['score'] for edge in edges)
        high_score_edges = [e for e in edges if e['score'] > 0.7 * max_score]
        print(f"  Edge Summary: {len(edges)} total, {len(high_score_edges)} high-scoring (>{0.7*max_score:.2f})")
        print(f"  Score range: {min_score:.3f} - {max_score:.3f}")
    
    try:
        plt.show(block=False)  # Don't block execution
        plt.pause(1)  # Brief pause to display
    except:
        print("  Could not display plot interactively")
    
    return edges

if __name__ == "__main__":
    print("Running MULTI-CYCLE algorithm trace with edge visualizations...")
    print("Using actual simulate_surfactant_measurements function from workflow!")
    print("8 recommendations per cycle across 4 cycles")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    cycle1_recs, cycle2_recs, cycle3_recs, cycle4_recs = trace_real_iterative_process()
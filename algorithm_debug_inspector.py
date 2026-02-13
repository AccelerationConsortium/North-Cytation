# -*- coding: utf-8 -*-
"""
Algorithm Debug Inspector - Shows Internal Calculations
Exposes exactly how the vector edge algorithm calculates gradients and scores
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def simulate_surfactant_measurements(surf_a_conc, surf_b_conc, add_noise=True):
    """REAL simulation function from workflow"""
    import numpy as np
    
    log_a = np.log10(surf_a_conc)
    log_b = np.log10(surf_b_conc)
    
    # RATIO SIMULATION: Diagonal boundary
    diagonal_distance = (log_a + log_b + 5.0) / np.sqrt(2)
    ratio_transition = 1.0 / (1.0 + np.exp(-8.0 * diagonal_distance))
    ratio_baseline = 0.6
    ratio_elevated = 1.4
    simulated_ratio = ratio_baseline + (ratio_elevated - ratio_baseline) * ratio_transition
    
    # TURBIDITY SIMULATION: Circular boundary 
    center_a, center_b = -4.5, -2.5
    radius_distance = np.sqrt((log_a - center_a)**2 + (log_b - center_b)**2)
    turbidity_transition = 1.0 / (1.0 + np.exp(-5.0 * (radius_distance - 1.2)))
    turbidity_baseline = 0.15
    turbidity_elevated = 0.85
    simulated_turbidity = turbidity_baseline + (turbidity_elevated - turbidity_baseline) * turbidity_transition
    
    # FLUORESCENCE: Derive from ratio
    f384_base = 95.0 + 10.0 * np.sin(log_a + log_b)
    f373_base = simulated_ratio * f384_base
    
    # Add noise
    if add_noise:
        noise_scale = 0.01
        ratio_noise = 1.0 + np.random.normal(0, noise_scale)
        turbidity_noise = 1.0 + np.random.normal(0, noise_scale)
        fluorescence_noise = 1.0 + np.random.normal(0, noise_scale * 0.5)
        
        simulated_ratio *= ratio_noise
        simulated_turbidity *= turbidity_noise
        f373_base *= fluorescence_noise
        f384_base *= fluorescence_noise
        
        simulated_ratio = f373_base / f384_base if f384_base > 0 else simulated_ratio
    
    # Bounds
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

class AlgorithmDebugger:
    """Debug the vector edge algorithm's internal calculations step by step."""
    
    def __init__(self):
        """Initialize debugger."""
        pass
    
    def debug_algorithm_decisions(self, data_df):
        """
        Debug the algorithm's internal calculations to see why it picks certain points.
        Shows grid structure, normalization, edge calculations, and scoring step-by-step.
        """
        sys.path.append(".")
        from recommenders.generalized_vector_edge_recommender import GeneralizedVectorEdgeRecommender
        
        print("="*70)
        print("ALGORITHM DEBUG: Internal Calculation Inspector")
        print("="*70)
        
        # Initialize recommender like in workflow
        recommender = GeneralizedVectorEdgeRecommender(
            input_columns=['surf_A_conc_mm', 'surf_B_conc_mm'],
            output_columns=['ratio'],
            log_transform_inputs=True,
            normalization_method='log_zscore'
        )
        
        print(f"\\nStep 1: Input Data Summary")
        print(f"  {len(data_df)} data points")
        print(f"  SDS: {data_df['surf_A_conc_mm'].min():.6f} - {data_df['surf_A_conc_mm'].max():.6f} mM")
        print(f"  TTAB: {data_df['surf_B_conc_mm'].min():.6f} - {data_df['surf_B_conc_mm'].max():.6f} mM")
        print(f"  Ratio: {data_df['ratio'].min():.3f} - {data_df['ratio'].max():.3f}")
        
        # Step 2: Show normalization process
        print(f"\\nStep 2: Normalization Process")
        normalized_data = recommender._normalize_outputs(data_df)
        
        print(f"  Original ratio range: {data_df['ratio'].min():.3f} - {data_df['ratio'].max():.3f}")
        
        if 'ratio_log' in normalized_data.columns:
            print(f"  Log-transformed: {normalized_data['ratio_log'].min():.3f} - {normalized_data['ratio_log'].max():.3f}")
        
        if 'ratio_normalized' in normalized_data.columns:
            print(f"  Z-score normalized: {normalized_data['ratio_normalized'].min():.3f} - {normalized_data['ratio_normalized'].max():.3f}")
            print(f"  Normalization stats: mean={normalized_data['ratio_normalized'].mean():.3f}, std={normalized_data['ratio_normalized'].std():.3f}")
        
        # Step 3: Show grid structure
        print(f"\\nStep 3: Grid Structure Analysis")
        
        if recommender.log_transform_inputs:
            normalized_data['log_surf_A_conc_mm'] = np.log10(normalized_data['surf_A_conc_mm'])
            normalized_data['log_surf_B_conc_mm'] = np.log10(normalized_data['surf_B_conc_mm'])
        
        unique_sds = sorted(normalized_data['surf_A_conc_mm'].unique())
        unique_ttab = sorted(normalized_data['surf_B_conc_mm'].unique())
        
        print(f"  SDS grid ({len(unique_sds)} values): {[f'{x:.6f}' for x in unique_sds]}")
        print(f"  TTAB grid ({len(unique_ttab)} values): {[f'{x:.6f}' for x in unique_ttab]}")
        print(f"  Grid size: {len(unique_sds)} × {len(unique_ttab)} = {len(unique_sds)*len(unique_ttab)} total positions")
        print(f"  Data coverage: {len(data_df)}/{len(unique_sds)*len(unique_ttab)} = {100*len(data_df)/(len(unique_sds)*len(unique_ttab)):.1f}%")
        
        # Step 4: Show individual grid points with their values
        print(f"\\nStep 4: Grid Point Values (showing first 10)")
        print(f"{'SDS':>12} {'TTAB':>12} {'Ratio':>8} {'Norm':>8} {'LogSDS':>8} {'LogTTAB':>8}")
        print("-"*70)
        
        for i, row in normalized_data.head(10).iterrows():
            sds = row['surf_A_conc_mm']
            ttab = row['surf_B_conc_mm']
            ratio = row['ratio']
            norm_ratio = row.get('ratio_normalized', 'N/A')
            log_sds = row.get('log_surf_A_conc_mm', 'N/A')
            log_ttab = row.get('log_surf_B_conc_mm', 'N/A')
            
            print(f"{sds:>12.6f} {ttab:>12.6f} {ratio:>8.3f} {norm_ratio:>8.3f} {log_sds:>8.3f} {log_ttab:>8.3f}")
        
        # Step 5: Calculate edges manually to show process
        print(f"\\nStep 5: Edge Calculation Process")
        
        # Create grid lookup for easier edge calculation
        grid_lookup = {}
        for _, row in normalized_data.iterrows():
            sds = row['surf_A_conc_mm']
            ttab = row['surf_B_conc_mm']
            grid_lookup[(sds, ttab)] = row
        
        print(f"  Created grid lookup with {len(grid_lookup)} points")
        
        # Calculate some example edges
        edges_calculated = []
        edge_count = 0
        
        print(f"\\n  Example Edge Calculations:")
        print(f"  {'Edge':>20} {'Point 1':>25} {'Point 2':>25} {'Δ Norm':>10} {'Score':>8}")
        print("-"*100)
        
        for i, sds1 in enumerate(unique_sds[:-1]):
            sds2 = unique_sds[i + 1]
            for ttab in unique_ttab:
                if (sds1, ttab) in grid_lookup and (sds2, ttab) in grid_lookup:
                    r1 = grid_lookup[(sds1, ttab)]
                    r2 = grid_lookup[(sds2, ttab)]
                    
                    norm1 = r1.get('ratio_normalized', r1['ratio'])
                    norm2 = r2.get('ratio_normalized', r2['ratio'])
                    diff = norm2 - norm1
                    score = abs(diff)  # Simplified edge score
                    
                    edge_name = f"SDS {i}->{i+1}"
                    point1_desc = f"({sds1:.6f}, {ttab:.6f})"
                    point2_desc = f"({sds2:.6f}, {ttab:.6f})"
                    
                    edges_calculated.append({
                        'edge_type': 'SDS',
                        'sds1': sds1, 'sds2': sds2, 'ttab': ttab,
                        'norm1': norm1, 'norm2': norm2,
                        'diff': diff, 'score': score
                    })
                    
                    if edge_count < 10:  # Show first 10 edges
                        print(f"  {edge_name:>20} {point1_desc:>25} {point2_desc:>25} {diff:>10.3f} {score:>8.3f}")
                    
                    edge_count += 1
        
        # Show which edges have highest scores
        if edges_calculated:
            edges_df = pd.DataFrame(edges_calculated)
            top_edges = edges_df.nlargest(5, 'score')
            
            print(f"\\n  TOP 5 EDGE SCORES:")
            print(f"  {'Type':>8} {'SDS1':>12} {'SDS2':>12} {'TTAB':>12} {'Norm1':>8} {'Norm2':>8} {'Score':>8}")
            print("-"*80)
            
            for _, edge in top_edges.iterrows():
                print(f"  {edge['edge_type']:>8} {edge['sds1']:>12.6f} {edge['sds2']:>12.6f} {edge['ttab']:>12.6f} {edge['norm1']:>8.3f} {edge['norm2']:>8.3f} {edge['score']:>8.3f}")
        
        # Step 6: Run actual algorithm and show discrepancy
        print(f"\\nStep 6: Compare with Actual Algorithm Results")
        
        actual_recs = recommender.get_recommendations(
            data_df,
            n_points=12,
            min_spacing_factor=2.0,
            output_dir=None,
            create_visualization=False
        )
        
        print(f"\\n  Algorithm returned {len(actual_recs)} recommendations")
        if len(actual_recs) > 0:
            print(f"  Top 5 algorithm picks:")
            for i, row in actual_recs.head(5).iterrows():
                sds = row['surf_A_conc_mm']
                ttab = row['surf_B_conc_mm']
                score = row.get('boundary_score', 'N/A')
                print(f"    {i+1}: SDS={sds:.6f} mM, TTAB={ttab:.6f} mM, Score={score}")
        
        print(f"\\n" + "="*70)
        print(f"SUMMARY: Algorithm Decision Analysis")
        print(f"="*70)
        print(f"The algorithm picks points based on normalized vector differences.")
        print(f"If picks don't match visible boundaries, possible issues:")
        print(f"1. Normalization is hiding the signal")
        print(f"2. Log transform is distorting relationships") 
        print(f"3. Grid structure doesn't capture boundary well")
        print(f"4. Edge scoring method doesn't match visual patterns")
        
        return edges_calculated, actual_recs, normalized_data
    
    def visualize_edge_scores(self, data_df):
        """
        Visualize edge scores to see exactly what the algorithm detects as boundaries.
        Shows data points + edges colored by score intensity.
        """
        sys.path.append(".")
        from recommenders.generalized_vector_edge_recommender import GeneralizedVectorEdgeRecommender
        
        # Initialize recommender
        recommender = GeneralizedVectorEdgeRecommender(
            input_columns=['surf_A_conc_mm', 'surf_B_conc_mm'],
            output_columns=['ratio'],
            log_transform_inputs=True,
            normalization_method='log_zscore'
        )
        
        # Get normalized data and internal grid structure
        normalized_data = recommender._normalize_outputs(data_df)
        
        if recommender.log_transform_inputs:
            normalized_data['log_surf_A_conc_mm'] = np.log10(normalized_data['surf_A_conc_mm'])
            normalized_data['log_surf_B_conc_mm'] = np.log10(normalized_data['surf_B_conc_mm'])
        
        # Get internal grid data (same as algorithm uses)
        grid_data, grid_coordinates = recommender._create_grid_structure(normalized_data)
        
        # Calculate edge scores (same as algorithm)
        edges = []
        grid_sizes = [len(grid_coordinates[col]) for col in recommender.input_columns]
        
        # Generate edges exactly like the algorithm does
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
                    
                    # Calculate Euclidean distance in normalized space
                    score = np.sqrt(sum(diff**2 for diff in output_diffs))
                    
                    # Store edge info for visualization
                    edge = {
                        'x1': data1['log_surf_A_conc_mm'],
                        'y1': data1['log_surf_B_conc_mm'], 
                        'x2': data2['log_surf_A_conc_mm'],
                        'y2': data2['log_surf_B_conc_mm'],
                        'score': score,
                        'direction': dim_name,
                        'midpoint_x': (data1['log_surf_A_conc_mm'] + data2['log_surf_A_conc_mm']) / 2,
                        'midpoint_y': (data1['log_surf_B_conc_mm'] + data2['log_surf_B_conc_mm']) / 2,
                        'ratio1': data1['ratio'],
                        'ratio2': data2['ratio'],
                        'ratio_diff': abs(data2['ratio'] - data1['ratio'])
                    }
                    edges.append(edge)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left plot: Data points + edges colored by algorithm scores
        log_sds = normalized_data['log_surf_A_conc_mm']
        log_ttab = normalized_data['log_surf_B_conc_mm']
        
        # Plot data points
        scatter1 = ax1.scatter(log_sds, log_ttab, c=normalized_data['ratio'], 
                              cmap='RdYlBu', s=200, alpha=0.8, edgecolors='black', 
                              linewidth=2, zorder=5, label='Data Points')
        
        # Plot edges colored by algorithm score
        if edges:
            max_score = max(edge['score'] for edge in edges)
            min_score = min(edge['score'] for edge in edges)
            
            for edge in edges:
                # Color based on score (red = high score = strong boundary)
                score_norm = (edge['score'] - min_score) / (max_score - min_score) if max_score > min_score else 0
                color_intensity = plt.cm.Reds(0.3 + 0.7 * score_norm)  # Red scale
                linewidth = 2 + 8 * score_norm  # Thicker lines = higher scores
                
                ax1.plot([edge['x1'], edge['x2']], [edge['y1'], edge['y2']], 
                        color=color_intensity, linewidth=linewidth, alpha=0.8, zorder=3)
                
                # Add score text on high-score edges
                if edge['score'] > 0.5 * max_score:
                    ax1.text(edge['midpoint_x'], edge['midpoint_y'], f'{edge["score"]:.2f}', 
                            fontsize=10, fontweight='bold', color='darkred', 
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                            ha='center', va='center', zorder=6)
        
        ax1.set_xlabel('log10(SDS Concentration [mM])')
        ax1.set_ylabel('log10(TTAB Concentration [mM])')
        ax1.set_title('Algorithm Edge Scores\\n(Red = High Score, Thick = Strong Boundary)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.colorbar(scatter1, ax=ax1, label='Ratio (Actual)')
        
        # Right plot: Human-visible boundary vs algorithm edges
        scatter2 = ax2.scatter(log_sds, log_ttab, c=normalized_data['ratio'], 
                              cmap='RdYlBu', s=200, alpha=0.8, edgecolors='black', 
                              linewidth=2, zorder=5)
        
        # Draw human-visible boundary (diagonal)
        boundary_x = np.linspace(log_sds.min(), log_sds.max(), 100)
        boundary_y = -4.5 + 0.5 * boundary_x  # Approximate diagonal from visual inspection
        mask = (boundary_y >= log_ttab.min()) & (boundary_y <= log_ttab.max())
        ax2.plot(boundary_x[mask], boundary_y[mask], 'k--', linewidth=3, 
                label='Human-Visible Boundary', alpha=0.7)
        
        # Highlight top algorithm edges
        if edges:
            top_edges = sorted(edges, key=lambda e: e['score'], reverse=True)[:3]
            for i, edge in enumerate(top_edges):
                ax2.plot([edge['x1'], edge['x2']], [edge['y1'], edge['y2']], 
                        color='red', linewidth=6, alpha=0.9, zorder=4,
                        label=f'Top Edge {i+1} (Score: {edge["score"]:.2f})' if i < 1 else '')
                
                # Mark midpoints where algorithm would recommend
                ax2.scatter(edge['midpoint_x'], edge['midpoint_y'], c='red', s=300, 
                           marker='X', edgecolors='darkred', linewidth=3, zorder=6)
                ax2.text(edge['midpoint_x'], edge['midpoint_y'] + 0.2, f'Rec {i+1}', 
                        fontsize=11, fontweight='bold', color='darkred', ha='center')
        
        ax2.set_xlabel('log10(SDS Concentration [mM])')
        ax2.set_ylabel('log10(TTAB Concentration [mM])')
        ax2.set_title('Algorithm Recommendations vs Human Boundary\\n(Red X = Where Algorithm Picks)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.colorbar(scatter2, ax=ax2, label='Ratio (Actual)')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs("debug_analysis", exist_ok=True)
        plt.savefig("debug_analysis/edge_score_visualization.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved edge score visualization: debug_analysis/edge_score_visualization.png")
        
        try:
            plt.show()
        except:
            print("Could not display plot interactively")
        
        # Print edge score summary
        print(f"\\nEDGE SCORE SUMMARY:")
        print(f"Total edges calculated: {len(edges)}")
        if edges:
            max_score = max(edge['score'] for edge in edges)
            min_score = min(edge['score'] for edge in edges)
            print(f"Score range: {min_score:.3f} - {max_score:.3f}")
            print(f"\\nTop 5 Edges by Score:")
            top_edges = sorted(edges, key=lambda e: e['score'], reverse=True)
            for i, edge in enumerate(top_edges[:5]):
                ratio_change = edge['ratio_diff']
                print(f"  {i+1}: Score={edge['score']:.3f}, Ratio Δ={ratio_change:.3f}, Dir={edge['direction']}")
                print(f"      From ({edge['x1']:.2f}, {edge['y1']:.2f}) to ({edge['x2']:.2f}, {edge['y2']:.2f})")
        
        return edges

def create_simple_test_grid():
    """Create a simple 3x3 test grid for debugging."""
    MIN_CONC = 10**-4
    MAX_CONC = 22.5
    
    # Simple 3x3 grid for easier debugging
    concentrations = np.logspace(np.log10(MIN_CONC), np.log10(MAX_CONC), 3)
    
    surf_a_list = []
    surf_b_list = []
    
    for surf_a_conc in concentrations:
        for surf_b_conc in concentrations:
            surf_a_list.append(surf_a_conc)
            surf_b_list.append(surf_b_conc)
    
    return surf_a_list, surf_b_list

def simulate_measurements_for_grid(surf_a_list, surf_b_list):
    """Simulate measurements for grid."""
    data_rows = []
    
    for surf_a_conc, surf_b_conc in zip(surf_a_list, surf_b_list):
        measurements = simulate_surfactant_measurements(surf_a_conc, surf_b_conc, add_noise=False)  # No noise for debugging
        
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

if __name__ == "__main__":
    print("Running Algorithm Edge Score Visualizer...")
    
    # Create 5x5 test grid (like your original workflow)
    MIN_CONC = 10**-4
    MAX_CONC = 22.5
    concentrations = np.logspace(np.log10(MIN_CONC), np.log10(MAX_CONC), 5)  # Back to 5x5
    
    surf_a_list = []
    surf_b_list = []
    for surf_a_conc in concentrations:
        for surf_b_conc in concentrations:
            surf_a_list.append(surf_a_conc)
            surf_b_list.append(surf_b_conc)
    
    test_data = simulate_measurements_for_grid(surf_a_list, surf_b_list)
    
    print(f"Created 5x5 grid with {len(test_data)} points")
    print(f"Ratio range: {test_data['ratio'].min():.3f} - {test_data['ratio'].max():.3f}")
    
    # Visualize edge scores
    debugger = AlgorithmDebugger()
    edges = debugger.visualize_edge_scores(test_data)
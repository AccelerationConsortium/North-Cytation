#!/usr/bin/env python3
"""
Debug script to understand why the triangle algorithm isn't exploring the 1.0->1.4 boundary.
"""

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

def main():
    # Create the same 5x5 grid from the test
    surf_a_concs = np.logspace(-6, -2, 5)  # 1e-6 to 1e-2
    surf_b_concs = np.logspace(-5, -1, 5)  # 1e-5 to 1e-1

    # Generate grid data
    initial_data = []
    for i, surf_a in enumerate(surf_a_concs):
        for j, surf_b in enumerate(surf_b_concs):
            log_a = np.log10(surf_a)
            log_b = np.log10(surf_b)
            diagonal_distance = (log_a + log_b + 5.0) / np.sqrt(2)
            ratio_transition = 1.0 / (1.0 + np.exp(-8.0 * diagonal_distance))
            simulated_ratio = 0.6 + (1.4 - 0.6) * ratio_transition
            
            initial_data.append({
                'surf_A_conc_mm': surf_a,
                'surf_B_conc_mm': surf_b,
                'ratio': simulated_ratio
            })

    grid_df = pd.DataFrame(initial_data)

    # Build triangulation
    inputs_log = np.column_stack([
        np.log10(grid_df['surf_A_conc_mm'].values),
        np.log10(grid_df['surf_B_conc_mm'].values)
    ])
    triangulation = Delaunay(inputs_log)

    # Analyze triangles in original space
    triangles_analysis = []
    for t_idx, triangle in enumerate(triangulation.simplices):
        vertex_ratios = [grid_df.iloc[v]['ratio'] for v in triangle]
        disagreement = max(vertex_ratios) - min(vertex_ratios)
        triangles_analysis.append({
            'triangle_idx': t_idx,
            'vertices': triangle,
            'min_ratio': min(vertex_ratios),
            'max_ratio': max(vertex_ratios),
            'disagreement': disagreement
        })

    triangles_analysis.sort(key=lambda x: x['disagreement'], reverse=True)

    # Test the normalization that the algorithm uses
    ratio_values = grid_df['ratio'].values
    print("NORMALIZATION EFFECTS ON TRIANGLE SCORING")
    print("=" * 60)
    
    print("Original ratio values:")
    print(f"  Range: {ratio_values.min():.3f} - {ratio_values.max():.3f}")
    print(f"  Span: {ratio_values.max() - ratio_values.min():.3f}")

    # Step 1: Log transform (if > 0)
    log_ratios = np.log(ratio_values)
    print(f"\nAfter log transform:")
    print(f"  Range: {log_ratios.min():.3f} - {log_ratios.max():.3f}")
    print(f"  Span: {log_ratios.max() - log_ratios.min():.3f}")

    # Step 2: Z-score normalization 
    mean_log = np.mean(log_ratios)
    std_log = np.std(log_ratios)
    normalized_ratios = (log_ratios - mean_log) / std_log
    print(f"\nAfter z-score normalization:")
    print(f"  Range: {normalized_ratios.min():.3f} - {normalized_ratios.max():.3f}")
    print(f"  Span: {normalized_ratios.max() - normalized_ratios.min():.3f}")

    # Compare triangle scores before and after normalization
    print(f"\n\nTOPMOST TRIANGLES COMPARISON:")
    print("="*60)
    print(f"{'Rank':<4} {'Vertices':<12} {'Original Range':<15} {'Orig Score':<12} {'Norm Range':<15} {'Norm Score':<12}")
    print("-"*80)
    
    for i, t in enumerate(triangles_analysis[:10]):
        # Get normalized values for triangle vertices
        vertex_indices = t['vertices']
        norm_values = [normalized_ratios[v] for v in vertex_indices]
        norm_disagreement = max(norm_values) - min(norm_values)
        
        orig_range = f"{t['min_ratio']:.3f}-{t['max_ratio']:.3f}"
        norm_range = f"{min(norm_values):.3f}-{max(norm_values):.3f}"
        
        print(f"{i+1:<4} {str(vertex_indices):<12} {orig_range:<15} {t['disagreement']:<12.3f} {norm_range:<15} {norm_disagreement:<12.3f}")

    # Look for specific problematic triangles
    print(f"\n\nTRIANGLES SPANNING 1.0 -> 1.4 BOUNDARY:")
    print("="*60)
    spanning_triangles = [t for t in triangles_analysis if t['min_ratio'] <= 1.0 and t['max_ratio'] >= 1.3]
    
    if spanning_triangles:
        print(f"Found {len(spanning_triangles)} triangles that should explore toward yellow region:")
        for i, t in enumerate(spanning_triangles):
            vertex_indices = t['vertices']
            norm_values = [normalized_ratios[v] for v in vertex_indices]
            norm_disagreement = max(norm_values) - min(norm_values)
            
            print(f"\n  Triangle {i+1}: vertices {vertex_indices}")
            print(f"    Original range: {t['min_ratio']:.3f} - {t['max_ratio']:.3f} (score: {t['disagreement']:.3f})")
            print(f"    Normalized range: {min(norm_values):.3f} - {max(norm_values):.3f} (score: {norm_disagreement:.3f})")
            
            # Check if this would be in top recommendations 
            rank_orig = triangles_analysis.index(t) + 1
            norm_ranks = [(max(normalized_ratios[list(tri['vertices'])]) - min(normalized_ratios[list(tri['vertices'])])) for tri in triangles_analysis]
            norm_ranks.sort(reverse=True)
            rank_norm = norm_ranks.index(norm_disagreement) + 1
            print(f"    Ranking: {rank_orig} (original) → {rank_norm} (normalized)")
    else:
        print("  No triangles span from ~1.0 to ~1.4! Geometric issue.")

if __name__ == "__main__":
    main()
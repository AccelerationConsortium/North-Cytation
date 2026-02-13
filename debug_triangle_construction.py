#!/usr/bin/env python3
"""
Debug triangle construction to see if Delaunay triangulation is creating the right connections.
"""

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def main():
    # Create the same 5x5 grid
    surf_a_concs = np.logspace(-6, -2, 5)  # 1e-6 to 1e-2
    surf_b_concs = np.logspace(-5, -1, 5)  # 1e-5 to 1e-1

    # Generate grid with position tracking
    initial_data = []
    point_map = {}  # Map (i,j) grid position to point index
    
    for i, surf_a in enumerate(surf_a_concs):
        for j, surf_b in enumerate(surf_b_concs):
            log_a = np.log10(surf_a)
            log_b = np.log10(surf_b)
            diagonal_distance = (log_a + log_b + 5.0) / np.sqrt(2)
            ratio_transition = 1.0 / (1.0 + np.exp(-8.0 * diagonal_distance))
            simulated_ratio = 0.6 + (1.4 - 0.6) * ratio_transition
            
            point_idx = len(initial_data)
            point_map[(i, j)] = point_idx
            
            initial_data.append({
                'surf_A_conc_mm': surf_a,
                'surf_B_conc_mm': surf_b,
                'ratio': simulated_ratio,
                'grid_i': i,
                'grid_j': j,
                'point_idx': point_idx
            })

    grid_df = pd.DataFrame(initial_data)

    # Build triangulation in log space (like the algorithm does)
    inputs_log = np.column_stack([
        np.log10(grid_df['surf_A_conc_mm'].values),
        np.log10(grid_df['surf_B_conc_mm'].values)
    ])
    triangulation = Delaunay(inputs_log)

    print("GRID LAYOUT AND TRIANGULATION ANALYSIS")
    print("=" * 60)
    
    # Show the grid layout with ratio values
    print("\nGrid layout (ratio values):")
    print("   j=0    j=1    j=2    j=3    j=4")
    for i in range(5):
        row_str = f"i={i}"
        for j in range(5):
            point_idx = point_map[(i, j)]
            ratio = grid_df.iloc[point_idx]['ratio']
            row_str += f"  {ratio:.3f}"
        print(row_str)
    
    # Key points of interest
    key_points = {
        'bottom-left (0.600)': point_map[(0, 0)],
        'top-right (1.400)': point_map[(4, 4)],
        'center (0.600)': point_map[(2, 2)],
        'transition (1.000)': point_map[(2, 4)]  # Should be ~1.0
    }
    
    print(f"\nKey points:")
    for desc, idx in key_points.items():
        row = grid_df.iloc[idx]
        print(f"  {desc}: index {idx}, grid ({row['grid_i']},{row['grid_j']}), ratio={row['ratio']:.3f}")
    
    # Find triangles that connect critical regions
    print(f"\nTriangles connecting transition regions:")
    
    transition_triangles = []
    for t_idx, triangle in enumerate(triangulation.simplices):
        vertex_ratios = [grid_df.iloc[v]['ratio'] for v in triangle]
        vertex_positions = [(grid_df.iloc[v]['grid_i'], grid_df.iloc[v]['grid_j']) for v in triangle]
        
        min_ratio = min(vertex_ratios)
        max_ratio = max(vertex_ratios)
        
        # Look for triangles that span significant ratio differences
        if max_ratio - min_ratio > 0.5:
            transition_triangles.append({
                'triangle_idx': t_idx,
                'vertices': triangle,
                'positions': vertex_positions,
                'ratios': vertex_ratios,
                'disagreement': max_ratio - min_ratio
            })
    
    transition_triangles.sort(key=lambda x: x['disagreement'], reverse=True)
    
    print(f"Found {len(transition_triangles)} triangles with significant transitions:")
    for i, t in enumerate(transition_triangles[:8]):
        print(f"\n  Triangle {i+1}: vertices {t['vertices']}")
        print(f"    Grid positions: {t['positions']}")
        print(f"    Ratios: {[f'{r:.3f}' for r in t['ratios']]}")
        print(f"    Disagreement: {t['disagreement']:.3f}")
        
        # Check if this connects expected regions
        if any(r >= 1.3 for r in t['ratios']) and any(r <= 1.0 for r in t['ratios']):
            print(f"    ** This triangle connects low->high regions! **")
    
    # Check specific connections we expect to see
    print(f"\n\nCHECKING EXPECTED CONNECTIONS:")
    print("=" * 40)
    
    # Should there be a triangle connecting center (0.600) to top-right (1.400)?
    center_idx = point_map[(2, 2)]  # Center: ratio 0.600
    top_right_idx = point_map[(4, 4)]  # Top-right: ratio 1.400
    
    connected = False
    connecting_triangle = None
    for triangle in triangulation.simplices:
        if center_idx in triangle and top_right_idx in triangle:
            connected = True
            connecting_triangle = triangle
            break
    
    print(f"Center ({center_idx}) directly connected to top-right ({top_right_idx})? {connected}")
    if connected:
        third_vertex = [v for v in connecting_triangle if v not in [center_idx, top_right_idx]][0]
        third_ratio = grid_df.iloc[third_vertex]['ratio']
        print(f"  Triangle: {connecting_triangle}")
        print(f"  Third vertex {third_vertex}: ratio {third_ratio:.3f}")
    
    # What about transition points?
    transition_idx = point_map[(2, 4)]  # Should be ~1.000
    print(f"\nTransition point ({transition_idx}, ratio={grid_df.iloc[transition_idx]['ratio']:.3f}) connected to top-right? ", end="")
    
    connected_trans = False
    for triangle in triangulation.simplices:
        if transition_idx in triangle and top_right_idx in triangle:
            connected_trans = True
            third_vertex = [v for v in triangle if v not in [transition_idx, top_right_idx]][0]
            third_ratio = grid_df.iloc[third_vertex]['ratio']
            print(f"YES")
            print(f"  Triangle: {triangle}")
            print(f"  Third vertex {third_vertex}: ratio {third_ratio:.3f}")
            break
    
    if not connected_trans:
        print("NO")
        print("  This might explain why the boundary isn't being explored!")

if __name__ == "__main__":
    main()
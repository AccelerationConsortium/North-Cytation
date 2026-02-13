#!/usr/bin/env python3
"""
Delaunay Triangle Refinement Test Program
=========================================

Test the new Delaunay Triangle Refinement algorithm using simulated surfactant data.
Demonstrates how the algorithm identifies high-variation triangles and recommends 
sampling at triangle centroids through multiple optimization cycles.

The test uses the simulate_surfactant_measurements function to create realistic 
boundary patterns in ratio measurements and shows how triangulation adapts.
"""

import sys
import os
sys.path.append("../utoronto_demo")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import the new recommender
from recommenders.delaunay_triangle_recommender import DelaunayTriangleRecommender

def simulate_surfactant_measurements(surf_a_conc, surf_b_conc, add_noise=True):
    """
    Generate realistic simulation data for turbidity and ratio based on surfactant concentrations.
    
    Creates 2D boundary patterns that transition between baseline and elevated states:
    - Ratio: Diagonal boundary from bottom-left to top-right
    - Turbidity: Circular boundary with different center
    
    Args:
        surf_a_conc: Surfactant A concentration in mM
        surf_b_conc: Surfactant B concentration in mM 
        add_noise: Whether to add realistic experimental noise
        
    Returns:
        dict: {'turbidity_600': float, 'fluorescence_334_373': float, 
               'fluorescence_334_384': float, 'ratio': float}
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

def create_initial_grid_data(n_points_per_dim=5):
    """
    Create initial COMPLETE regular grid data for testing triangulation.
    Uses a larger grid for better visualization of algorithm behavior.
    
    Args:
        n_points_per_dim: Number of concentration points per dimension
        
    Returns:
        pd.DataFrame: Complete grid measurements (all points measured)
    """
    
    print(f"Creating COMPLETE {n_points_per_dim}x{n_points_per_dim} concentration grid...")
    
    # Create logarithmic concentration ranges (broader for more interesting patterns)
    surf_a_concs = np.logspace(-6, -2, n_points_per_dim)  # 1e-6 to 1e-2 mM  
    surf_b_concs = np.logspace(-5, -1, n_points_per_dim)  # 1e-5 to 1e-1 mM
    
    print(f"  Surf A concentrations: {[f'{c:.1e}' for c in surf_a_concs]}")
    print(f"  Surf B concentrations: {[f'{c:.1e}' for c in surf_b_concs]}")
    
    grid_data = []
    
    for i, conc_a in enumerate(surf_a_concs):
        for j, conc_b in enumerate(surf_b_concs):
            # Simulate measurements for this point
            sim_data = simulate_surfactant_measurements(conc_a, conc_b, add_noise=True)
            
            grid_data.append({
                'surf_A_conc_mm': conc_a,
                'surf_B_conc_mm': conc_b,
                'turbidity_600': sim_data['turbidity_600'],
                'ratio': sim_data['ratio'],
                'well_type': 'experiment',
                'grid_i': i,
                'grid_j': j
            })
    
    df = pd.DataFrame(grid_data)
    
    print(f"Generated {len(df)} initial grid points")
    print(f"  Expected: {n_points_per_dim}x{n_points_per_dim} = {n_points_per_dim**2}")
    print(f"  Surf A range: {df['surf_A_conc_mm'].min():.2e} - {df['surf_A_conc_mm'].max():.2e} mM")
    print(f"  Surf B range: {df['surf_B_conc_mm'].min():.2e} - {df['surf_B_conc_mm'].max():.2e} mM")
    print(f"  Ratio range: {df['ratio'].min():.3f} - {df['ratio'].max():.3f}")
    print(f"  Unique Surf A values: {len(df['surf_A_conc_mm'].unique())}")
    print(f"  Unique Surf B values: {len(df['surf_B_conc_mm'].unique())}")
    
    return df

def run_triangle_refinement_cycle(experiment_data, cycle_num, n_suggestions=6):
    """
    Run one cycle of Delaunay triangle refinement.
    
    Args:
        experiment_data: Current experimental data DataFrame
        cycle_num: Cycle number for logging
        n_suggestions: Number of new triangle centroids to suggest
        
    Returns:
        tuple: (updated_experiment_data, recommendations_df)
    """
    
    print(f"\\n{'='*60}")
    print(f"CYCLE {cycle_num}: DELAUNAY TRIANGLE REFINEMENT")
    print(f"{'='*60}")
    
    print(f"Input: {len(experiment_data)} measured points")
    print(f"  Ratio range: {experiment_data['ratio'].min():.3f} - {experiment_data['ratio'].max():.3f}")
    
    # Initialize recommender 
    recommender = DelaunayTriangleRecommender(
        input_columns=['surf_A_conc_mm', 'surf_B_conc_mm'],
        output_columns=['ratio'],  # Focus on ratio only for clarity
        log_transform_inputs=True,
        normalization_method='log_zscore'
    )
    
    # Get recommendations
    recommendations_df = recommender.get_recommendations(
        experiment_data,
        n_points=n_suggestions,
        min_spacing_factor=0.3,  # Allow some closer packing
        tol_factor=0.05,        # Reasonable duplicate tolerance  
        triangle_score_method='max',
        create_visualization=False
    )
    
    if len(recommendations_df) == 0:
        print("WARNING: No recommendations generated!")
        return experiment_data, recommendations_df
    
    # Simulate new measurements at recommended points
    print(f"\\nSimulating measurements at {len(recommendations_df)} recommended triangle centroids...")
    new_data = []
    
    for idx, row in recommendations_df.iterrows():
        conc_a = row['surf_A_conc_mm']
        conc_b = row['surf_B_conc_mm']
        
        sim_data = simulate_surfactant_measurements(conc_a, conc_b, add_noise=True)
        
        new_data.append({
            'surf_A_conc_mm': conc_a,
            'surf_B_conc_mm': conc_b,
            'turbidity_600': sim_data['turbidity_600'],
            'ratio': sim_data['ratio'],
            'well_type': 'experiment',
            'cycle': cycle_num,
            'triangle_score': row['triangle_score']
        })
        
        print(f"  {idx+1:2d}: {conc_a:.2e}, {conc_b:.2e} → ratio={sim_data['ratio']:.3f} (score={row['triangle_score']:.3f})")
    
    # Combine with existing data
    new_df = pd.DataFrame(new_data)
    updated_data = pd.concat([experiment_data, new_df], ignore_index=True)
    
    print(f"\\nCycle {cycle_num} complete: {len(new_df)} new points added, {len(updated_data)} total")
    
    return updated_data, recommendations_df

def visualize_triangle_results(all_data, all_recommendations):
    """
    Create visualization showing progression of Delaunay triangle refinement.
    """
    
    print("\\nCreating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Delaunay Triangle Refinement Progress', fontsize=14)
    
    # Separate data by cycle
    initial_data = all_data[~all_data['cycle'].notna()]
    cycle_data = all_data[all_data['cycle'].notna()]
    
    # Plot 1: Initial grid points (ratio)
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(initial_data['surf_A_conc_mm'], initial_data['surf_B_conc_mm'], 
                          c=initial_data['ratio'], s=80, cmap='viridis', alpha=0.8, 
                          edgecolors='black', linewidth=0.5, marker='s')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Surf A Concentration (mM)')
    ax1.set_ylabel('Surf B Concentration (mM)')
    ax1.set_title('Initial Grid Points (Ratio)')
    plt.colorbar(scatter1, ax=ax1, label='Ratio')
    
    # Plot 2: All measurements (ratio with triangulation-guided refinement)
    ax2 = axes[0, 1]
    # Initial grid points
    ax2.scatter(initial_data['surf_A_conc_mm'], initial_data['surf_B_conc_mm'], 
               c=initial_data['ratio'], s=60, cmap='viridis', alpha=0.6, marker='s', label='Initial Grid')
    # New points from triangle centroids
    if len(cycle_data) > 0:
        scatter2 = ax2.scatter(cycle_data['surf_A_conc_mm'], cycle_data['surf_B_conc_mm'], 
                             c=cycle_data['ratio'], s=120, cmap='viridis', alpha=0.9, 
                             marker='*', edgecolors='red', linewidth=2, label='Triangle Centroids')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Surf A Concentration (mM)')
    ax2.set_ylabel('Surf B Concentration (mM)')
    ax2.set_title('Grid + Triangle Refinement')
    ax2.legend()
    if len(cycle_data) > 0:
        plt.colorbar(scatter2, ax=ax2, label='Ratio')
    
    # Plot 3: Triangle scores progression
    ax3 = axes[1, 0]
    if len(all_recommendations) > 0:
        for i, rec_df in enumerate(all_recommendations):
            if len(rec_df) > 0:
                ax3.scatter(rec_df['surf_A_conc_mm'], rec_df['surf_B_conc_mm'], 
                           s=rec_df['triangle_score']*500, alpha=0.6, label=f'Cycle {i+1}')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Surf A Concentration (mM)')
        ax3.set_ylabel('Surf B Concentration (mM)')
        ax3.set_title('Triangle Scores (Size ∝ Score)')
        ax3.legend()
    
    # Plot 4: Distribution comparison
    ax4 = axes[1, 1]
    ax4.hist(initial_data['ratio'], bins=8, alpha=0.6, label='Initial Grid', density=True)
    if len(cycle_data) > 0:
        ax4.hist(cycle_data['ratio'], bins=8, alpha=0.6, label='Triangle Centroids', density=True)
    ax4.set_xlabel('Ratio')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution of Ratio Values')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"test_analysis/delaunay_triangle_test_{timestamp}.png"
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {plot_filename}")
    
    plt.show()

def main():
    """
    Main test function demonstrating Delaunay triangle refinement algorithm.
    """
    
    print("="*70)
    print("DELAUNAY TRIANGLE REFINEMENT TEST")
    print("="*70)
    print("Testing the triangle-based algorithm using simulated surfactant data.")
    print("Shows how triangulation identifies boundaries starting from a regular grid.")
    
    # Step 1: Create initial grid data (larger grid for better visualization)
    initial_data = create_initial_grid_data(n_points_per_dim=5)  # 5x5 = 25 points grid
    initial_data['cycle'] = np.nan  # Mark as initial data
    
    # Step 2: Run multiple cycles of refinement
    all_data = initial_data.copy()
    all_recommendations = []
    
    n_cycles = 10  # Many cycles to see long-term evolution
    n_suggestions_per_cycle = 8  # More suggestions to see pattern
    
    for cycle in range(1, n_cycles + 1):
        updated_data, recommendations = run_triangle_refinement_cycle(
            all_data, cycle, n_suggestions=n_suggestions_per_cycle
        )
        
        all_data = updated_data
        all_recommendations.append(recommendations)
        
        # Show summary
        cycle_points = all_data[all_data['cycle'] == cycle]
        if len(cycle_points) > 0:
            print(f"\\nCycle {cycle} Summary:")
            print(f"  Points added: {len(cycle_points)}")
            print(f"  Ratio range of new points: {cycle_points['ratio'].min():.3f} - {cycle_points['ratio'].max():.3f}")
            print(f"  Average triangle score: {cycle_points['triangle_score'].mean():.4f}")
    
    # Step 3: Final summary
    print(f"\\n{'='*70}")
    print("TRIANGLE REFINEMENT TEST COMPLETE")
    print(f"{'='*70}")
    
    total_initial = len(initial_data)
    total_added = len(all_data) - len(initial_data)
    total_final = len(all_data)
    
    print(f"Initial grid points: {total_initial}")
    print(f"Triangle centroids added: {total_added}")
    print(f"Final total points: {total_final}")
    
    # Show how ratio coverage improved
    initial_ratio_range = initial_data['ratio'].max() - initial_data['ratio'].min()
    final_ratio_range = all_data['ratio'].max() - all_data['ratio'].min()
    
    print(f"\\nRatio coverage:")
    print(f"  Initial range: {initial_data['ratio'].min():.3f} - {initial_data['ratio'].max():.3f} (span: {initial_ratio_range:.3f})")
    print(f"  Final range: {all_data['ratio'].min():.3f} - {all_data['ratio'].max():.3f} (span: {final_ratio_range:.3f})")
    
    # Show boundary detection effectiveness
    boundary_points = all_data[all_data['cycle'].notna()]
    if len(boundary_points) > 0:
        print(f"\\nBoundary detection:")
        print(f"  Average triangle score: {boundary_points['triangle_score'].mean():.4f}")
        print(f"  Score range: {boundary_points['triangle_score'].min():.4f} - {boundary_points['triangle_score'].max():.4f}")
    
    print(f"\\nTriangle algorithm characteristics:")
    print(f"  ✓ Starts with regular grid (familiar base case)")
    print(f"  ✓ No missing corners (triangulation uses only measured points)")
    print(f"  ✓ Adapts naturally from grid to irregular distributions")
    print(f"  ✓ Clean geometric foundation with Delaunay triangulation")
    print(f"  ✓ Perfect for 2D spaces like surfactant concentrations")
    
    # Step 4: Create visualization
    visualize_triangle_results(all_data, all_recommendations)
    
    # Step 5: Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_analysis/delaunay_triangle_test_data_{timestamp}.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    all_data.to_csv(results_file, index=False)
    print(f"\\nTest data saved: {results_file}")
    
    print("\\nDelaunay triangle refinement test completed successfully!")

if __name__ == "__main__":
    main()
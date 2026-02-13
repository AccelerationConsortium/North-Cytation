#!/usr/bin/env python3
"""
Cell Variation Refinement Test Program
======================================

Test the new Cell-Variation Refinement algorithm using simulated surfactant data.
Demonstrates how the algorithm identifies high-variation grid cells and recommends 
sampling at their centers through multiple optimization cycles.

The test uses the simulate_surfactant_measurements function from the main workflow
to create realistic boundary patterns in ratio measurements.
"""

import sys
import os
sys.path.append("../utoronto_demo")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import the new recommender
from recommenders.cell_variation_refinement_recommender import CellVariationRefinementRecommender

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
    
    # Add realistic experimental noise if requested
    if add_noise:
        noise_scale = 0.02  # 2% coefficient of variation 
        ratio_noise = 1.0 + np.random.normal(0, noise_scale)
        turbidity_noise = 1.0 + np.random.normal(0, noise_scale)
        
        simulated_ratio *= ratio_noise
        simulated_turbidity *= turbidity_noise
    
    # Ensure physically reasonable bounds
    simulated_ratio = max(0.1, min(3.0, simulated_ratio))
    simulated_turbidity = max(0.01, min(1.5, simulated_turbidity))
    
    return {
        'turbidity_600': round(simulated_turbidity, 4),
        'ratio': round(simulated_ratio, 4)
    }

def create_initial_grid_data(n_points_per_dim=4):
    """
    Create initial COMPLETE regular grid data for testing.
    
    Args:
        n_points_per_dim: Number of concentration points per dimension
        
    Returns:
        pd.DataFrame: Complete grid measurements (all points measured)
    """
    
    print(f"Creating COMPLETE {n_points_per_dim}x{n_points_per_dim} concentration grid...")
    
    # Create logarithmic concentration ranges
    surf_a_concs = np.logspace(-5, -3, n_points_per_dim)  # 1e-5 to 1e-3 mM  
    surf_b_concs = np.logspace(-4, -2, n_points_per_dim)  # 1e-4 to 1e-2 mM
    
    print(f"  Surf A concentrations: {[f'{c:.2e}' for c in surf_a_concs]}")
    print(f"  Surf B concentrations: {[f'{c:.2e}' for c in surf_b_concs]}")
    
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

def run_cell_variation_cycle(experiment_data, cycle_num, n_suggestions=6):
    """
    Run one cycle of cell variation refinement.
    
    Args:
        experiment_data: Current experimental data DataFrame
        cycle_num: Cycle number for logging
        n_suggestions: Number of new points to suggest
        
    Returns:
        tuple: (updated_experiment_data, recommendations_df)
    """
    
    print(f"\\n{'='*60}")
    print(f"CYCLE {cycle_num}: CELL VARIATION REFINEMENT")
    print(f"{'='*60}")
    
    print(f"Input: {len(experiment_data)} measured points")
    
    # Initialize recommender 
    recommender = CellVariationRefinementRecommender(
        input_columns=['surf_A_conc_mm', 'surf_B_conc_mm'],
        output_columns=['ratio'],  # Focus on ratio only for clarity
        log_transform_inputs=True,
        normalization_method='log_zscore'
    )
    
    # Get recommendations
    recommendations_df = recommender.get_recommendations(
        experiment_data,
        n_points=n_suggestions,
        min_spacing_factor=0.5,
        create_visualization=False
    )
    
    if len(recommendations_df) == 0:
        print("WARNING: No recommendations generated!")
        return experiment_data, recommendations_df
    
    # Simulate new measurements at recommended points
    print(f"\\nSimulating measurements at {len(recommendations_df)} recommended points...")
    new_data = []
    
    for _, row in recommendations_df.iterrows():
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
            'cell_score': row['cell_score']
        })
        
        print(f"  {conc_a:.3e}, {conc_b:.3e} → ratio={sim_data['ratio']:.3f}, score={row['cell_score']:.4f}")
    
    # Combine with existing data
    new_df = pd.DataFrame(new_data)
    updated_data = pd.concat([experiment_data, new_df], ignore_index=True)
    
    print(f"\\nCycle {cycle_num} complete: {len(new_df)} new points added, {len(updated_data)} total")
    
    return updated_data, recommendations_df

def visualize_results(all_data, all_recommendations):
    """
    Create visualization showing progression of cell variation refinement.
    """
    
    print("\\nCreating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Cell Variation Refinement Progress', fontsize=14)
    
    # Separate data by cycle
    initial_data = all_data[~all_data['cycle'].notna()]
    cycle_data = all_data[all_data['cycle'].notna()]
    
    # Plot 1: Initial grid (ratio)
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(initial_data['surf_A_conc_mm'], initial_data['surf_B_conc_mm'], 
                          c=initial_data['ratio'], s=60, cmap='viridis', alpha=0.8)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Surf A Concentration (mM)')
    ax1.set_ylabel('Surf B Concentration (mM)')
    ax1.set_title('Initial Grid (Ratio)')
    plt.colorbar(scatter1, ax=ax1, label='Ratio')
    
    # Plot 2: All measurements (ratio)
    ax2 = axes[0, 1]
    # Initial points
    ax2.scatter(initial_data['surf_A_conc_mm'], initial_data['surf_B_conc_mm'], 
               c=initial_data['ratio'], s=40, cmap='viridis', alpha=0.6, marker='s', label='Initial')
    # New points from cycles
    if len(cycle_data) > 0:
        scatter2 = ax2.scatter(cycle_data['surf_A_conc_mm'], cycle_data['surf_B_conc_mm'], 
                             c=cycle_data['ratio'], s=80, cmap='viridis', alpha=0.9, 
                             marker='o', edgecolors='red', linewidth=1, label='Cell Centers')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Surf A Concentration (mM)')
    ax2.set_ylabel('Surf B Concentration (mM)')
    ax2.set_title('All Measurements (Ratio)')
    ax2.legend()
    if len(cycle_data) > 0:
        plt.colorbar(scatter2, ax=ax2, label='Ratio')
    
    # Plot 3: Cell scores progression
    ax3 = axes[1, 0]
    if len(all_recommendations) > 0:
        for i, rec_df in enumerate(all_recommendations):
            if len(rec_df) > 0:
                ax3.scatter(rec_df['surf_A_conc_mm'], rec_df['surf_B_conc_mm'], 
                           s=rec_df['cell_score']*500, alpha=0.6, label=f'Cycle {i+1}')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Surf A Concentration (mM)')
        ax3.set_ylabel('Surf B Concentration (mM)')
        ax3.set_title('Cell Scores (Size ∝ Score)')
        ax3.legend()
    
    # Plot 4: Distribution of measurements
    ax4 = axes[1, 1]
    ax4.hist(initial_data['ratio'], bins=10, alpha=0.6, label='Initial Grid', density=True)
    if len(cycle_data) > 0:
        ax4.hist(cycle_data['ratio'], bins=10, alpha=0.6, label='Cell Centers', density=True)
    ax4.set_xlabel('Ratio')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution of Ratio Values')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"test_analysis/cell_variation_test_{timestamp}.png"
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, dpi=150)
    print(f"Visualization saved: {plot_filename}")
    
    plt.show()

def main():
    """
    Main test function demonstrating cell variation refinement algorithm.
    """
    
    print("="*70)
    print("CELL VARIATION REFINEMENT TEST")
    print("="*70)
    print("Testing the new Cell-Variation algorithm using simulated surfactant data.")
    print("Shows how the algorithm identifies boundaries by finding high-variation cells.")
    
    # Step 1: Create initial grid data
    initial_data = create_initial_grid_data(n_points_per_dim=3)  # Smaller 3x3 = 9 points for debugging
    initial_data['cycle'] = np.nan  # Mark as initial data
    
    # Step 2: Run multiple cycles of refinement
    all_data = initial_data.copy()
    all_recommendations = []
    
    n_cycles = 2  # Reduced for debugging
    n_suggestions_per_cycle = 2  # Reduced for debugging
    
    for cycle in range(1, n_cycles + 1):
        updated_data, recommendations = run_cell_variation_cycle(
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
            print(f"  Average cell score: {cycle_points['cell_score'].mean():.4f}")
    
    # Step 3: Final summary
    print(f"\\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")
    
    total_initial = len(initial_data)
    total_added = len(all_data) - len(initial_data)
    total_final = len(all_data)
    
    print(f"Initial grid points: {total_initial}")
    print(f"Cell centers added: {total_added}")
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
        print(f"  Average cell score: {boundary_points['cell_score'].mean():.4f}")
        print(f"  Score range: {boundary_points['cell_score'].min():.4f} - {boundary_points['cell_score'].max():.4f}")
    
    # Step 4: Create visualization
    visualize_results(all_data, all_recommendations)
    
    # Step 5: Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_analysis/cell_variation_test_data_{timestamp}.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    all_data.to_csv(results_file, index=False)
    print(f"\\nTest data saved: {results_file}")
    
    print("\\nTest completed successfully!")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Vector Edge Algorithm Debugger
Shows step-by-step what the vector edge algorithm "sees" and why it picks certain points.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

def debug_vector_edge_algorithm(csv_file_path, output_dir="debug_analysis"):
    """
    STEP-BY-STEP gradient analysis to see why the algorithm is failing.
    
    This will show:
    1. Raw gradient calculations at each data point
    2. How the algorithm ranks/scores potential boundary points  
    3. What the algorithm THINKS vs what humans SEE
    4. Save detailed results with explanations
    """
    print(f"=== STEP-BY-STEP GRADIENT ALGORITHM DEBUGGER ===")
    print(f"Loading data from: {csv_file_path}")
    print(f"Goal: Figure out why algorithm ignores obvious diagonal boundary")
    
    # Import the recommender to replicate its analysis
    try:
        sys.path.append(".")
        from recommenders.generalized_vector_edge_recommender import GeneralizedVectorEdgeRecommender
        print("✓ Successfully imported GeneralizedVectorEdgeRecommender")
    except ImportError as e:
        print(f"✗ Failed to import GeneralizedVectorEdgeRecommender: {e}")
        return
    
    # Load and filter data
    df = pd.read_csv(csv_file_path)
    experiment_data = df[df['well_type'] == 'experiment'].copy()
    
    print(f"✓ Loaded {len(experiment_data)} experimental points")
    print(f"  SDS range: {experiment_data['surf_A_conc_mm'].min():.6f} - {experiment_data['surf_A_conc_mm'].max():.6f} mM")
    print(f"  TTAB range: {experiment_data['surf_B_conc_mm'].min():.6f} - {experiment_data['surf_B_conc_mm'].max():.6f} mM")
    print(f"  Turbidity range: {experiment_data['turbidity_600'].min():.3f} - {experiment_data['turbidity_600'].max():.3f}")
    print(f"  Ratio range: {experiment_data['ratio'].min():.3f} - {experiment_data['ratio'].max():.3f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the same recommender configuration as your workflow
    print(f"\n=== INITIALIZING ALGORITHM (Same Config as Workflow) ===")
    recommender = GeneralizedVectorEdgeRecommender(
        input_columns=['surf_A_conc_mm', 'surf_B_conc_mm'],
        output_columns=['ratio'],  # Focus on ratio only (like your workflow)
        log_transform_inputs=True,
        normalization_method='log_zscore'
    )
    print(f"✓ Configured: inputs={recommender.input_columns}, outputs={recommender.output_columns}")
    print(f"✓ Log transform: {recommender.log_transform_inputs}, normalization: {recommender.normalization_method}")
    
    # Create detailed debugging visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Vector Edge Algorithm Step-by-Step Debug Analysis', fontsize=16, fontweight='bold')
    
    # Convert to log space for visualization (same as algorithm)
    log_sds = np.log10(experiment_data['surf_A_conc_mm'])
    log_ttab = np.log10(experiment_data['surf_B_conc_mm'])
    
    # Plot 1: Raw input data (linear space)
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(experiment_data['surf_A_conc_mm'], experiment_data['surf_B_conc_mm'], 
                          c=experiment_data['ratio'], cmap='RdYlBu', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('SDS Concentration [mM] (Linear)')
    ax1.set_ylabel('TTAB Concentration [mM] (Linear)')  
    ax1.set_title('1. Raw Input Data (Linear Scale)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.colorbar(scatter1, ax=ax1, label='Ratio')
    
    # Plot 2: Log-transformed data (algorithm's view)
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(log_sds, log_ttab, c=experiment_data['ratio'], cmap='RdYlBu', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('log10(SDS Concentration [mM])')
    ax2.set_ylabel('log10(TTAB Concentration [mM])')
    ax2.set_title('2. Algorithm\'s View (Log Space)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Ratio')
    
    # Plot 3: Normalized data (algorithm's internal representation) 
    ax3 = axes[0, 2]
    
    # Manually replicate the normalization the algorithm uses
    try:
        # Apply the same normalization as the algorithm
        test_data = experiment_data.copy()
        normalized_data = recommender._normalize_outputs(test_data)
        
        if 'ratio_normalized' in normalized_data.columns:
            scatter3 = ax3.scatter(log_sds, log_ttab, c=normalized_data['ratio_normalized'], cmap='RdYlBu', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
            ax3.set_title('3. Normalized Data (Algorithm Internal)')
            plt.colorbar(scatter3, ax=ax3, label='Normalized Ratio')
        else:
            ax3.text(0.5, 0.5, 'Normalization Failed', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('3. Normalization Error')
            
    except Exception as e:
        ax3.text(0.5, 0.5, f'Debug Error:\\n{e}', ha='center', va='center', transform=ax3.transAxes, fontsize=8)
        ax3.set_title('3. Debug Error')
    
    ax3.set_xlabel('log10(SDS Concentration [mM])')
    ax3.set_ylabel('log10(TTAB Concentration [mM])')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Edge detection analysis
    ax4 = axes[0, 3]
    
    try:
        # Get the algorithm's recommendations with debugging info
        print(f"\\n=== RUNNING ALGORITHM ANALYSIS ===")
        
        recommendations_df = recommender.get_recommendations(
            experiment_data,
            n_points=12,
            min_spacing_factor=2.0,
            output_dir=None,
            create_visualization=False
        )
        
        print(f"✓ Algorithm returned {len(recommendations_df)} recommendations")
        
        if len(recommendations_df) > 0:
            # Plot the recommendations
            rec_log_sds = np.log10(recommendations_df['surf_A_conc_mm'])
            rec_log_ttab = np.log10(recommendations_df['surf_B_conc_mm'])
            
            ax4.scatter(log_sds, log_ttab, c=experiment_data['ratio'], cmap='RdYlBu', s=30, alpha=0.6, edgecolors='gray', linewidth=0.5)
            ax4.scatter(rec_log_sds, rec_log_ttab, c='red', s=100, marker='x', linewidth=3, label='Algorithm Picks')
            
            # Add numbers to show selection order
            for i, (x, y) in enumerate(zip(rec_log_sds, rec_log_ttab)):
                ax4.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                           fontsize=10, fontweight='bold', color='darkred')
            
            ax4.set_title('4. Algorithm Selections')
            ax4.legend()
            
        else:
            ax4.text(0.5, 0.5, 'No Recommendations\\nGenerated', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('4. Algorithm Failed')
            
    except Exception as e:
        ax4.text(0.5, 0.5, f'Algorithm Error:\\n{e}', ha='center', va='center', transform=ax4.transAxes, fontsize=8)
        ax4.set_title('4. Algorithm Error')
        
    ax4.set_xlabel('log10(SDS Concentration [mM])')
    ax4.set_ylabel('log10(TTAB Concentration [mM])')
    ax4.grid(True, alpha=0.3)
    
    # Bottom row: Analysis of what the algorithm "thinks"
    
    # Plot 5: Gradient magnitude visualization
    ax5 = axes[1, 0]
    
    try:
        # Create a regular grid to estimate gradients like the algorithm might
        log_sds_grid = np.linspace(log_sds.min(), log_sds.max(), 20)
        log_ttab_grid = np.linspace(log_ttab.min(), log_ttab.max(), 20)
        grid_sds, grid_ttab = np.meshgrid(log_sds_grid, log_ttab_grid)
        
        # Interpolate ratio values onto this grid
        from scipy.interpolate import griddata
        grid_ratios = griddata((log_sds, log_ttab), experiment_data['ratio'], 
                              (grid_sds, grid_ttab), method='cubic', fill_value=np.nan)
        
        # Calculate gradients
        grad_y, grad_x = np.gradient(grid_ratios)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        contour = ax5.contourf(grid_sds, grid_ttab, gradient_magnitude, levels=20, cmap='hot', alpha=0.7)
        ax5.scatter(log_sds, log_ttab, c='white', s=20, edgecolors='black', linewidth=0.5)
        ax5.set_title('5. Estimated Gradient Magnitude')
        plt.colorbar(contour, ax=ax5, label='Gradient Magnitude')
        
    except Exception as e:
        ax5.text(0.5, 0.5, f'Gradient Calc Error:\\n{e}', ha='center', va='center', transform=ax5.transAxes, fontsize=8)
        ax5.set_title('5. Gradient Error')
    
    ax5.set_xlabel('log10(SDS Concentration [mM])')
    ax5.set_ylabel('log10(TTAB Concentration [mM])')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Data density analysis
    ax6 = axes[1, 1] 
    
    try:
        # Show where data points are clustered vs sparse
        from scipy.stats import gaussian_kde
        
        # Create density estimate
        xy = np.vstack([log_sds, log_ttab])
        density = gaussian_kde(xy)(xy)
        
        scatter6 = ax6.scatter(log_sds, log_ttab, c=density, cmap='viridis', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax6.set_title('6. Data Point Density')
        plt.colorbar(scatter6, ax=ax6, label='Density')
        
    except Exception as e:
        ax6.text(0.5, 0.5, f'Density Error:\\n{e}', ha='center', va='center', transform=ax6.transAxes, fontsize=8)
        ax6.set_title('6. Density Error')
    
    ax6.set_xlabel('log10(SDS Concentration [mM])')
    ax6.set_ylabel('log10(TTAB Concentration [mM])')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Human-visible boundary analysis  
    ax7 = axes[1, 2]
    
    # Show what a human would identify as the boundary
    ax7.scatter(log_sds, log_ttab, c=experiment_data['ratio'], cmap='RdYlBu', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Draw approximate boundary lines where human sees transitions
    boundary_sds = np.linspace(log_sds.min(), log_sds.max(), 100)
    boundary_1 = -4.5 + 0.5 * boundary_sds  # Approximate diagonal
    ax7.plot(boundary_sds, boundary_1, 'k--', linewidth=2, alpha=0.7, label='Human-Visible Boundary')
    
    ax7.set_title('7. Human-Identified Boundary')
    ax7.set_xlabel('log10(SDS Concentration [mM])')
    ax7.set_ylabel('log10(TTAB Concentration [mM])')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Plot 8: Algorithm vs Human comparison
    ax8 = axes[1, 3]
    
    # Overlay algorithm picks on human boundary
    ax8.scatter(log_sds, log_ttab, c=experiment_data['ratio'], cmap='RdYlBu', s=30, alpha=0.6, edgecolors='gray', linewidth=0.5)
    ax8.plot(boundary_sds, boundary_1, 'k--', linewidth=2, alpha=0.7, label='Human Boundary')
    
    try:
        if len(recommendations_df) > 0:
            ax8.scatter(rec_log_sds, rec_log_ttab, c='red', s=100, marker='x', linewidth=3, label='Algorithm Picks')
    except:
        pass
        
    ax8.set_title('8. Algorithm vs Human')
    ax8.set_xlabel('log10(SDS Concentration [mM])')
    ax8.set_ylabel('log10(TTAB Concentration [mM])')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    plt.tight_layout()
    
    # Save the analysis
    debug_plot_path = os.path.join(output_dir, 'vector_edge_algorithm_debug.png')
    plt.savefig(debug_plot_path, dpi=300, bbox_inches='tight')
    print(f"\\n✓ Saved debug visualization: {debug_plot_path}")
    
    if hasattr(plt, 'show'):
        plt.show()
    
    # Print detailed analysis
    print(f"\\n=== HUMAN VS ALGORITHM ANALYSIS ===")
    print(f"Human observation: Clear diagonal boundary from low-SDS/high-TTAB to high-SDS/low-TTAB")
    print(f"Ratio transition: ~0.6 (red) to ~1.4 (blue) along diagonal")
    
    if len(recommendations_df) > 0:
        print(f"\\nAlgorithm recommendations:")
        for i, row in recommendations_df.head(5).iterrows():
            sds_conc = row['surf_A_conc_mm'] 
            ttab_conc = row['surf_B_conc_mm']
            print(f"  {i+1}: SDS={sds_conc:.3e} mM, TTAB={ttab_conc:.3e} mM")
            
            # Check if this point is near the human-visible boundary
            log_sds_pt = np.log10(sds_conc)
            log_ttab_pt = np.log10(ttab_conc) 
            expected_boundary_ttab = -4.5 + 0.5 * log_sds_pt
            distance_from_boundary = abs(log_ttab_pt - expected_boundary_ttab)
            
            if distance_from_boundary < 0.5:
                print(f"      → NEAR human boundary (distance: {distance_from_boundary:.2f})")
            else:
                print(f"      → FAR from human boundary (distance: {distance_from_boundary:.2f})")
    else:
        print(f"\\nAlgorithm generated 0 recommendations - this is the problem!")
    
    return debug_plot_path

if __name__ == "__main__":
    # Test with your data
    csv_file = "output/surfactant_grid_SDS_TTAB_20260212_165101/iterative_experiment_results.csv"
    
    if os.path.exists(csv_file):
        print("Testing vector edge algorithm debugging...")
        debug_vector_edge_algorithm(csv_file, "debug_analysis")
    else:
        print(f"CSV file not found: {csv_file}")
        print("Please update the file path to your latest results.")
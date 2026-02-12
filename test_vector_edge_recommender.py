"""
Test Program for Generalized Vector Edge Refinement Recommender
================================================================

Creates synthetic data with known boundary patterns and tests the recommender's ability
to identify these boundaries. Uses various mathematical functions that transition from
baseline to non-baseline behavior.

Test Functions:
- Step Function: Sharp diagonal boundary
- Circular Boundary: Radial transition 
- Sigmoid Transition: Smooth boundary
- Multi-Region: Different behaviors in different quadrants
- Combination: Multiple outputs with different boundary patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import pdist
import os
import sys

# Add recommender to path
recommender_path = os.path.join(os.path.dirname(__file__), 'recommenders')
if os.path.exists(recommender_path):
    sys.path.append(recommender_path)
else:
    # Try parent directory structure
    recommender_path = os.path.join(os.path.dirname(__file__), '..', 'recommenders')
    sys.path.append(recommender_path)

from generalized_vector_edge_recommender import GeneralizedVectorEdgeRecommender

class SyntheticDataGenerator:
    """Generate synthetic data with known boundary patterns."""
    
    def __init__(self, x_range=(-2, 2), y_range=(-2, 2), grid_size=7, noise_level=0.05):
        """
        Initialize synthetic data generator.
        
        Parameters:
        -----------
        x_range : tuple
            (min, max) for x coordinates
        y_range : tuple  
            (min, max) for y coordinates
        grid_size : int
            Number of points along each axis (default: 7 for coarser grid)
        noise_level : float
            Standard deviation of Gaussian noise to add
        """
        self.x_range = x_range
        self.y_range = y_range
        self.grid_size = grid_size
        self.noise_level = noise_level
        
        # Create regular grid
        x_vals = np.linspace(x_range[0], x_range[1], grid_size)
        y_vals = np.linspace(y_range[0], y_range[1], grid_size)
        self.X, self.Y = np.meshgrid(x_vals, y_vals)
        self.x_flat = self.X.flatten()
        self.y_flat = self.Y.flatten()
        
        print(f"Generated {grid_size}x{grid_size} = {len(self.x_flat)} grid points")
        print(f"X range: {x_range}, Y range: {y_range}")
        print(f"Noise level: {noise_level}")
    
    def add_noise(self, values):
        """Add Gaussian noise to values."""
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, values.shape)
            return values + noise
        return values
    
    def step_function_boundary(self, steep_factor=10):
        """
        Create a sharp step function boundary along the diagonal y = x.
        Output A: Low baseline below line, high above line
        Output B: Inverted step pattern
        """
        # Distance from diagonal line y = x
        distance_from_diagonal = self.y_flat - self.x_flat
        
        # Sharp step function using tanh for smoothing
        output_A = 0.1 + 0.8 * (0.5 + 0.5 * np.tanh(steep_factor * distance_from_diagonal))
        output_B = 0.9 - 0.7 * (0.5 + 0.5 * np.tanh(steep_factor * distance_from_diagonal))
        
        return self.add_noise(output_A), self.add_noise(output_B)
    
    def circular_boundary(self, center=(0, 0), radius=1.0, steep_factor=8):
        """
        Create a circular boundary around a center point.
        Output A: Low inside circle, high outside
        Output B: High inside circle, low outside  
        """
        # Distance from center
        dx = self.x_flat - center[0]
        dy = self.y_flat - center[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Step function based on radius
        output_A = 0.1 + 0.8 * (0.5 + 0.5 * np.tanh(steep_factor * (distance - radius)))
        output_B = 0.9 - 0.7 * (0.5 + 0.5 * np.tanh(steep_factor * (distance - radius)))
        
        return self.add_noise(output_A), self.add_noise(output_B)
    
    def sigmoid_transition(self, direction='vertical', position=0, steep_factor=5):
        """
        Create a smooth sigmoid transition.
        Output A: Smooth transition along specified direction
        Output B: Different sigmoid pattern
        """
        if direction == 'vertical':
            transition_var = self.x_flat - position
        else:  # horizontal
            transition_var = self.y_flat - position
        
        output_A = 0.1 + 0.8 * (0.5 + 0.5 * np.tanh(steep_factor * transition_var))
        
        # Different pattern for output B (more complex)
        if direction == 'vertical':
            transition_var_B = self.y_flat - position * 0.5
        else:
            transition_var_B = self.x_flat - position * 0.5
        output_B = 0.5 + 0.4 * np.sin(3 * transition_var_B) * (0.5 + 0.5 * np.tanh(steep_factor * transition_var))
        
        return self.add_noise(output_A), self.add_noise(output_B)
    
    def multi_region_patterns(self):
        """
        Create different patterns in different quadrants.
        Output A: Quadrant-based step function
        Output B: Radial gradient pattern
        """
        # Quadrant-based pattern for output A
        output_A = np.zeros_like(self.x_flat)
        
        # Different values for each quadrant
        mask_q1 = (self.x_flat >= 0) & (self.y_flat >= 0)  # Top-right
        mask_q2 = (self.x_flat < 0) & (self.y_flat >= 0)   # Top-left  
        mask_q3 = (self.x_flat < 0) & (self.y_flat < 0)    # Bottom-left
        mask_q4 = (self.x_flat >= 0) & (self.y_flat < 0)   # Bottom-right
        
        output_A[mask_q1] = 0.8
        output_A[mask_q2] = 0.2
        output_A[mask_q3] = 0.6
        output_A[mask_q4] = 0.4
        
        # Radial gradient for output B
        distance_from_origin = np.sqrt(self.x_flat**2 + self.y_flat**2)
        max_distance = np.sqrt(max(self.x_range)**2 + max(self.y_range)**2)
        output_B = 0.1 + 0.8 * (distance_from_origin / max_distance)
        
        return self.add_noise(output_A), self.add_noise(output_B)
    
    def complex_combination(self):
        """
        Complex combination of multiple boundary types.
        Output A: Combination of circular and linear boundaries
        Output B: Wave pattern with boundaries
        Output C: Checkerboard-like pattern
        """
        # Output A: Circular + diagonal
        circle_A, _ = self.circular_boundary(center=(-0.5, 0.5), radius=0.8)
        step_A, _ = self.step_function_boundary(steep_factor=8)
        output_A = 0.5 * circle_A + 0.5 * step_A
        
        # Output B: Wave pattern
        output_B = 0.5 + 0.3 * np.sin(3 * self.x_flat) * np.cos(3 * self.y_flat)
        output_B = np.clip(output_B, 0.1, 0.9)
        
        # Output C: Checkerboard-like with smooth transitions
        checker_x = np.sin(4 * np.pi * self.x_flat / (self.x_range[1] - self.x_range[0]))
        checker_y = np.sin(4 * np.pi * self.y_flat / (self.y_range[1] - self.y_range[0]))
        output_C = 0.5 + 0.3 * checker_x * checker_y
        output_C = np.clip(output_C, 0.1, 0.9)
        
        return (self.add_noise(output_A), self.add_noise(output_B), self.add_noise(output_C))


class RecommenderTester:
    """Test the recommender on synthetic data and evaluate performance."""
    
    def __init__(self, output_dir='test_recommender_output'):
        """Initialize tester with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def run_test_case(self, test_name, data_generator, output_function, input_cols, output_cols,
                      n_points=20, log_transform=False, normalization='zscore'):
        """
        Run a single test case.
        
        Parameters:
        -----------
        test_name : str
            Name for this test case
        data_generator : SyntheticDataGenerator
            Generator instance  
        output_function : callable
            Function to generate outputs (returns tuple of arrays)
        input_cols : list
            Input column names
        output_cols : list
            Output column names
        n_points : int
            Number of boundary points to recommend
        log_transform : bool
            Whether to log transform inputs
        normalization : str
            Normalization method
        """
        
        print(f"\\n{'='*60}")
        print(f"TEST CASE: {test_name}")
        print(f"{'='*60}")
        
        # Generate synthetic data
        outputs = output_function()
        
        # Create DataFrame
        data_dict = {
            input_cols[0]: data_generator.x_flat,
            input_cols[1]: data_generator.y_flat
        }
        
        for i, col in enumerate(output_cols):
            data_dict[col] = outputs[i]
        
        # Add well_type for compatibility
        data_dict['well_type'] = 'experiment'
        
        synthetic_df = pd.DataFrame(data_dict)
        
        print(f"Generated {len(synthetic_df)} synthetic data points")
        for col in output_cols:
            print(f"  {col}: {synthetic_df[col].min():.3f} - {synthetic_df[col].max():.3f}")
        
        # Initialize recommender
        recommender = GeneralizedVectorEdgeRecommender(
            input_columns=input_cols,
            output_columns=output_cols,
            log_transform_inputs=log_transform,
            normalization_method=normalization
        )
        
        # Get recommendations
        test_output_dir = os.path.join(self.output_dir, test_name)
        recommendations = recommender.get_recommendations(
            synthetic_df,
            n_points=n_points,
            output_dir=test_output_dir,
            create_visualization=True
        )
        
        # Save synthetic data for reference
        synthetic_file = os.path.join(test_output_dir, 'synthetic_data.csv')
        synthetic_df.to_csv(synthetic_file, index=False)
        
        # Create custom analysis plots
        self._create_analysis_plots(test_name, synthetic_df, recommendations, 
                                   input_cols, output_cols, test_output_dir)
        
        print(f"\n📁 FILES SAVED TO: {test_output_dir}")
        print(f"   - Look for: {test_name}_analysis.png (shows Output A and Output B side by side)")
        print(f"   - Also generated: vector_edge_refinement.png (default recommender viz)")
        
        # Calculate boundary detection metrics
        metrics = self._evaluate_boundary_detection(synthetic_df, recommendations, 
                                                  input_cols, output_cols)
        
        print(f"\\nTest Results for {test_name}:")
        print(f"  Recommended {len(recommendations)} boundary points")
        print(f"  Score range: {recommendations['boundary_score'].min():.4f} - {recommendations['boundary_score'].max():.4f}")
        print(f"  Coverage metrics: {metrics}")
        
        return {
            'test_name': test_name,
            'synthetic_data': synthetic_df,
            'recommendations': recommendations,
            'metrics': metrics,
            'output_dir': test_output_dir
        }
    
    def _create_analysis_plots(self, test_name, synthetic_df, recommendations, 
                             input_cols, output_cols, output_dir):
        """Create SIMPLE side-by-side plots showing Output A vs Output B."""
        
        x_col, y_col = input_cols[0], input_cols[1]
        
        # Simple side-by-side layout: just Output A vs Output B
        if len(output_cols) >= 2:
            fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 6))
            
            # LEFT: Output A
            scatter_a = ax_a.scatter(synthetic_df[x_col], synthetic_df[y_col], 
                                   c=synthetic_df[output_cols[0]], cmap='viridis', 
                                   s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
            ax_a.scatter(recommendations[x_col], recommendations[y_col],
                        c='red', s=150, marker='^', edgecolors='darkred', 
                        linewidth=2, alpha=1.0, label=f'Boundary Points (n={len(recommendations)})')
            ax_a.set_xlabel(x_col, fontsize=12)
            ax_a.set_ylabel(y_col, fontsize=12)
            ax_a.set_title(f'OUTPUT A: {output_cols[0]}\n(What you see in main viz)', fontsize=14, fontweight='bold')
            ax_a.grid(True, alpha=0.3)
            ax_a.legend(fontsize=10)
            cbar_a = plt.colorbar(scatter_a, ax=ax_a)
            cbar_a.set_label(output_cols[0], fontsize=11)
            
            # RIGHT: Output B  
            scatter_b = ax_b.scatter(synthetic_df[x_col], synthetic_df[y_col],
                                   c=synthetic_df[output_cols[1]], cmap='plasma',
                                   s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
            ax_b.scatter(recommendations[x_col], recommendations[y_col],
                        c='red', s=150, marker='^', edgecolors='darkred',
                        linewidth=2, alpha=1.0, label=f'Same Boundary Points')
            ax_b.set_xlabel(x_col, fontsize=12)
            ax_b.set_ylabel(y_col, fontsize=12)
            ax_b.set_title(f'OUTPUT B: {output_cols[1]}\n(Hidden dimension!)', fontsize=14, fontweight='bold')
            ax_b.grid(True, alpha=0.3)
            ax_b.legend(fontsize=10)
            cbar_b = plt.colorbar(scatter_b, ax=ax_b)
            cbar_b.set_label(output_cols[1], fontsize=11)
            
        else:
            # Single output fallback
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            scatter = ax.scatter(synthetic_df[x_col], synthetic_df[y_col], 
                               c=synthetic_df[output_cols[0]], cmap='viridis', 
                               s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
            ax.scatter(recommendations[x_col], recommendations[y_col],
                      c='red', s=150, marker='^', edgecolors='darkred', 
                      linewidth=2, alpha=1.0, label=f'Recommended (n={len(recommendations)})')
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f'{output_cols[0]} with Boundary Points', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(output_cols[0])
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'{test_name}_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ SIMPLE A vs B comparison saved: {output_file}")
        return output_file
    
    def _evaluate_boundary_detection(self, synthetic_df, recommendations, input_cols, output_cols):
        """Evaluate how well the recommender detected boundaries."""
        
        # Simple metrics for boundary detection quality
        metrics = {}
        
        # Coverage: How well distributed are the recommendations?
        x_col, y_col = input_cols[0], input_cols[1]
        
        x_range = synthetic_df[x_col].max() - synthetic_df[x_col].min()
        y_range = synthetic_df[y_col].max() - synthetic_df[y_col].min()
        
        rec_x_range = recommendations[x_col].max() - recommendations[x_col].min()
        rec_y_range = recommendations[y_col].max() - recommendations[y_col].min()
        
        metrics['x_coverage'] = rec_x_range / x_range if x_range > 0 else 0
        metrics['y_coverage'] = rec_y_range / y_range if y_range > 0 else 0
        
        # Diversity: Average distance between recommended points
        if len(recommendations) > 1:
            points = recommendations[[x_col, y_col]].values
            distances = pdist(points)
            metrics['avg_spacing'] = np.mean(distances)
            metrics['min_spacing'] = np.min(distances)
        else:
            metrics['avg_spacing'] = 0
            metrics['min_spacing'] = 0
        
        # Score distribution
        metrics['score_mean'] = recommendations['boundary_score'].mean()
        metrics['score_std'] = recommendations['boundary_score'].std()
        
        return metrics
    
    def run_iterative_exploration(self, test_name, data_generator, output_function, 
                                input_cols, output_cols, n_cycles=4, points_per_cycle=8,
                                log_transform=False, normalization='zscore'):
        """
        Run iterative boundary exploration - start coarse and progressively refine.
        
        Parameters:
        -----------
        test_name : str
            Name for this test case
        data_generator : SyntheticDataGenerator
            Generator instance (should have coarse initial grid)
        output_function : callable
            Function to generate outputs
        input_cols : list
            Input column names
        output_cols : list
            Output column names  
        n_cycles : int
            Number of iterative cycles
        points_per_cycle : int
            New points to add each cycle
        log_transform : bool
            Whether to log transform inputs
        normalization : str
            Normalization method
        """
        
        print(f"\\n{'='*70}")
        print(f"ITERATIVE EXPLORATION: {test_name}")
        print(f"{'='*70}")
        print(f"Cycles: {n_cycles}, Points per cycle: {points_per_cycle}")
        print(f"Starting grid: {data_generator.grid_size}x{data_generator.grid_size} = {data_generator.grid_size**2} points")
        
        # Generate the ground truth synthetic data for the entire function
        outputs = output_function()
        
        # Start with initial coarse grid
        current_data = {
            input_cols[0]: data_generator.x_flat.copy(),
            input_cols[1]: data_generator.y_flat.copy()
        }
        
        for i, col in enumerate(output_cols):
            current_data[col] = outputs[i].copy()
        
        current_data['well_type'] = ['initial'] * len(data_generator.x_flat)
        current_data['cycle'] = [0] * len(data_generator.x_flat)
        
        current_df = pd.DataFrame(current_data)
        
        # Track results for each cycle
        cycle_results = []
        test_output_dir = os.path.join(self.output_dir, f'{test_name}_iterative')
        os.makedirs(test_output_dir, exist_ok=True)
        
        print(f"\\nCYCLE 0 (Initial): {len(current_df)} points")
        
        # Run iterative cycles
        for cycle in range(1, n_cycles + 1):
            print(f"\\n--- CYCLE {cycle} ---")
            
            # Initialize recommender
            recommender = GeneralizedVectorEdgeRecommender(
                input_columns=input_cols,
                output_columns=output_cols,
                log_transform_inputs=log_transform,
                normalization_method=normalization
            )
            
            # Set well_type for current cycle analysis
            analysis_df = current_df.copy()
            analysis_df['well_type'] = 'experiment'  # For compatibility
            
            # Get recommendations
            cycle_output_dir = os.path.join(test_output_dir, f'cycle_{cycle}')
            recommendations = recommender.get_recommendations(
                analysis_df,
                n_points=points_per_cycle,
                output_dir=cycle_output_dir,
                create_visualization=True
            )
            
            print(f"  Recommended {len(recommendations)} new boundary points")
            if len(recommendations) > 0:
                print(f"  Score range: {recommendations['boundary_score'].min():.4f} - {recommendations['boundary_score'].max():.4f}")
            
            # Add recommended points to dataset
            if len(recommendations) > 0:
                # Sample the true function at recommended points
                new_points = self._sample_function_at_points(
                    recommendations, input_cols, output_function, data_generator, output_cols
                )
                
                new_points['well_type'] = ['recommended'] * len(new_points)
                new_points['cycle'] = [cycle] * len(new_points)
                
                # Add to cumulative dataset
                current_df = pd.concat([current_df, new_points], ignore_index=True)
                
                print(f"  Added {len(new_points)} new points. Total dataset: {len(current_df)} points")
            
            # Store cycle results
            cycle_results.append({
                'cycle': cycle,
                'recommendations': recommendations,
                'cumulative_data': current_df.copy(),
                'new_points_added': len(recommendations)
            })
        
        # Create comprehensive visualization
        self._create_iterative_visualization(
            test_name, cycle_results, input_cols, output_cols, test_output_dir
        )
        
        # Save final dataset
        final_file = os.path.join(test_output_dir, 'final_iterative_dataset.csv')
        current_df.to_csv(final_file, index=False)
        
        print(f"\\nIterative exploration complete for {test_name}")
        print(f"Final dataset: {len(current_df)} points across {n_cycles} cycles")
        print(f"Results saved to: {test_output_dir}")
        
        return {
            'test_name': test_name,
            'cycle_results': cycle_results,
            'final_dataset': current_df,
            'output_dir': test_output_dir
        }
    
    def _sample_function_at_points(self, recommendations, input_cols, output_function, data_generator, output_cols):
        """Sample the true function at recommended points."""
        
        # Create temporary data generator with recommended points
        x_vals = recommendations[input_cols[0]].values
        y_vals = recommendations[input_cols[1]].values
        
        # Temporarily modify the data generator
        original_x = data_generator.x_flat
        original_y = data_generator.y_flat
        
        data_generator.x_flat = x_vals
        data_generator.y_flat = y_vals
        
        # Sample the function
        outputs = output_function()
        
        # Restore original values
        data_generator.x_flat = original_x
        data_generator.y_flat = original_y
        
        # Create dataframe
        new_data = {
            input_cols[0]: x_vals,
            input_cols[1]: y_vals
        }
        
        for i, col in enumerate(output_cols):
            new_data[col] = outputs[i]
        
        return pd.DataFrame(new_data)
    
    def _create_iterative_visualization(self, test_name, cycle_results, input_cols, output_cols, output_dir):
        """Create visualization showing iterative boundary exploration."""
        
        n_cycles = len(cycle_results)
        fig, axes = plt.subplots(2, n_cycles, figsize=(5*n_cycles, 10))
        
        if n_cycles == 1:
            axes = axes.reshape(-1, 1)
        
        x_col, y_col = input_cols[0], input_cols[1]
        main_output = output_cols[0]
        
        # Plot evolution of dataset
        for i, cycle_result in enumerate(cycle_results):
            cycle = cycle_result['cycle']
            cumulative_data = cycle_result['cumulative_data']
            recommendations = cycle_result['recommendations']
            
            # Top row: Dataset evolution
            ax_data = axes[0, i]
            
            # Plot initial points
            initial_data = cumulative_data[cumulative_data['cycle'] == 0]
            ax_data.scatter(initial_data[x_col], initial_data[y_col],
                          c=initial_data[main_output], cmap='viridis',
                          s=40, alpha=0.6, marker='o', label='Initial grid')
            
            # Plot recommended points from previous cycles
            colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray']
            for prev_cycle in range(1, cycle + 1):
                prev_data = cumulative_data[cumulative_data['cycle'] == prev_cycle]
                if len(prev_data) > 0:
                    color = colors[(prev_cycle-1) % len(colors)]
                    ax_data.scatter(prev_data[x_col], prev_data[y_col],
                                  c=color, s=80, alpha=0.8, marker='^',
                                  label=f'Cycle {prev_cycle}' if prev_cycle == cycle else '')
            
            ax_data.set_xlabel(x_col)
            ax_data.set_ylabel(y_col)
            ax_data.set_title(f'After Cycle {cycle}\\n{len(cumulative_data)} total points')
            ax_data.grid(True, alpha=0.3)
            ax_data.legend(fontsize=8)
            
            # Bottom row: Current cycle recommendations with scores
            ax_rec = axes[1, i]
            
            # Background: all current data
            ax_rec.scatter(cumulative_data[x_col], cumulative_data[y_col],
                         c=cumulative_data[main_output], cmap='viridis',
                         s=30, alpha=0.4)
            
            # Highlight current cycle recommendations
            if len(recommendations) > 0:
                scatter = ax_rec.scatter(recommendations[x_col], recommendations[y_col],
                                       c=recommendations['boundary_score'], cmap='Reds',
                                       s=150, marker='^', edgecolors='darkred',
                                       linewidth=2, alpha=1.0)
                
                # Add colorbar for scores
                cbar = plt.colorbar(scatter, ax=ax_rec)
                cbar.set_label('Boundary Score')
                
                ax_rec.set_title(f'Cycle {cycle} Recommendations\\nScores: {recommendations["boundary_score"].min():.3f}-{recommendations["boundary_score"].max():.3f}')
            else:
                ax_rec.set_title(f'Cycle {cycle}: No recommendations')
            
            ax_rec.set_xlabel(x_col)
            ax_rec.set_ylabel(y_col)
            ax_rec.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'{test_name}_iterative_evolution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Iterative visualization saved: {output_file}")
    
    def run_iterative_exploration(self, test_name, data_generator, output_function, 
                                input_cols, output_cols, n_cycles=4, points_per_cycle=8,
                                log_transform=False, normalization='zscore'):
        """
        Run iterative boundary exploration - start coarse and progressively refine.
        
        Parameters:
        -----------
        test_name : str
            Name for this test case
        data_generator : SyntheticDataGenerator
            Generator instance (should have coarse initial grid)
        output_function : callable
            Function to generate outputs
        input_cols : list
            Input column names
        output_cols : list
            Output column names  
        n_cycles : int
            Number of iterative cycles
        points_per_cycle : int
            New points to add each cycle
        log_transform : bool
            Whether to log transform inputs
        normalization : str
            Normalization method
        """
        
        print(f"\n{'='*70}")
        print(f"ITERATIVE EXPLORATION: {test_name}")
        print(f"{'='*70}")
        print(f"Cycles: {n_cycles}, Points per cycle: {points_per_cycle}")
        print(f"Starting grid: {data_generator.grid_size}x{data_generator.grid_size} = {data_generator.grid_size**2} points")
        
        # Generate the ground truth synthetic data for the entire function
        outputs = output_function()
        
        # Start with initial coarse grid
        current_data = {
            input_cols[0]: data_generator.x_flat.copy(),
            input_cols[1]: data_generator.y_flat.copy()
        }
        
        for i, col in enumerate(output_cols):
            current_data[col] = outputs[i].copy()
        
        current_data['well_type'] = ['initial'] * len(data_generator.x_flat)
        current_data['cycle'] = [0] * len(data_generator.x_flat)
        
        current_df = pd.DataFrame(current_data)
        
        # Track results for each cycle
        cycle_results = []
        test_output_dir = os.path.join(self.output_dir, f'{test_name}_iterative')
        os.makedirs(test_output_dir, exist_ok=True)
        
        print(f"\nCYCLE 0 (Initial): {len(current_df)} points")
        
        # Run iterative cycles
        for cycle in range(1, n_cycles + 1):
            print(f"\n--- CYCLE {cycle} ---")
            
            # Initialize recommender
            recommender = GeneralizedVectorEdgeRecommender(
                input_columns=input_cols,
                output_columns=output_cols,
                log_transform_inputs=log_transform,
                normalization_method=normalization
            )
            
            # Set well_type for current cycle analysis
            analysis_df = current_df.copy()
            analysis_df['well_type'] = 'experiment'  # For compatibility
            
            # Get recommendations
            cycle_output_dir = os.path.join(test_output_dir, f'cycle_{cycle}')
            recommendations = recommender.get_recommendations(
                analysis_df,
                n_points=points_per_cycle,
                output_dir=cycle_output_dir,
                create_visualization=True
            )
            
            print(f"  Recommended {len(recommendations)} new boundary points")
            if len(recommendations) > 0:
                print(f"  Score range: {recommendations['boundary_score'].min():.4f} - {recommendations['boundary_score'].max():.4f}")
            
            # Add recommended points to dataset
            if len(recommendations) > 0:
                # Sample the true function at recommended points
                new_points = self._sample_function_at_points(
                    recommendations, input_cols, output_function, data_generator
                )
                
                new_points['well_type'] = ['recommended'] * len(new_points)
                new_points['cycle'] = [cycle] * len(new_points)
                
                # Add to cumulative dataset
                current_df = pd.concat([current_df, new_points], ignore_index=True)
                
                print(f"  Added {len(new_points)} new points. Total dataset: {len(current_df)} points")
            
            # Store cycle results
            cycle_results.append({
                'cycle': cycle,
                'recommendations': recommendations,
                'cumulative_data': current_df.copy(),
                'new_points_added': len(recommendations)
            })
        
        # Create comprehensive visualization
        self._create_iterative_visualization(
            test_name, cycle_results, input_cols, output_cols, test_output_dir
        )
        
        # Save final dataset
        final_file = os.path.join(test_output_dir, 'final_iterative_dataset.csv')
        current_df.to_csv(final_file, index=False)
        
        print(f"\nIterative exploration complete for {test_name}")
        print(f"Final dataset: {len(current_df)} points across {n_cycles} cycles")
        print(f"Results saved to: {test_output_dir}")
        
        return {
            'test_name': test_name,
            'cycle_results': cycle_results,
            'final_dataset': current_df,
            'output_dir': test_output_dir
        }
    
    def _sample_function_at_points(self, recommendations, input_cols, output_function, data_generator):
        """Sample the true function at recommended points."""
        
        # Create temporary data generator with recommended points
        x_vals = recommendations[input_cols[0]].values
        y_vals = recommendations[input_cols[1]].values
        
        # Temporarily modify the data generator
        original_x = data_generator.x_flat
        original_y = data_generator.y_flat
        
        data_generator.x_flat = x_vals
        data_generator.y_flat = y_vals
        
        # Sample the function
        outputs = output_function()
        
        # Restore original values
        data_generator.x_flat = original_x
        data_generator.y_flat = original_y
        
        # Create dataframe
        new_data = {
            input_cols[0]: x_vals,
            input_cols[1]: y_vals
        }
        
        output_cols = [col for col in recommendations.columns if col.startswith('output_') or col in ['turbidity_600', 'ratio']]
        # Infer output columns from function
        if len(outputs) == 2:
            output_cols = ['output_A', 'output_B']
        elif len(outputs) == 3:
            output_cols = ['output_A', 'output_B', 'output_C']
        
        for i, col in enumerate(output_cols):
            new_data[col] = outputs[i]
        
        return pd.DataFrame(new_data)
    
    def _create_iterative_visualization(self, test_name, cycle_results, input_cols, output_cols, output_dir):
        """Create visualization showing iterative boundary exploration."""
        
        n_cycles = len(cycle_results)
        fig, axes = plt.subplots(2, n_cycles, figsize=(5*n_cycles, 10))
        
        if n_cycles == 1:
            axes = axes.reshape(-1, 1)
        
        x_col, y_col = input_cols[0], input_cols[1]
        main_output = output_cols[0]
        
        # Plot evolution of dataset
        for i, cycle_result in enumerate(cycle_results):
            cycle = cycle_result['cycle']
            cumulative_data = cycle_result['cumulative_data']
            recommendations = cycle_result['recommendations']
            
            # Top row: Dataset evolution
            ax_data = axes[0, i]
            
            # Plot initial points
            initial_data = cumulative_data[cumulative_data['cycle'] == 0]
            ax_data.scatter(initial_data[x_col], initial_data[y_col],
                          c=initial_data[main_output], cmap='viridis',
                          s=40, alpha=0.6, marker='o', label='Initial grid')
            
            # Plot recommended points from previous cycles
            for prev_cycle in range(1, cycle + 1):
                prev_data = cumulative_data[cumulative_data['cycle'] == prev_cycle]
                if len(prev_data) > 0:
                    ax_data.scatter(prev_data[x_col], prev_data[y_col],
                                  c='red', s=80, alpha=0.8, marker='^',
                                  label=f'Cycle {prev_cycle}' if prev_cycle == cycle else '')
            
            ax_data.set_xlabel(x_col)
            ax_data.set_ylabel(y_col)
            ax_data.set_title(f'After Cycle {cycle}\n{len(cumulative_data)} total points')
            ax_data.grid(True, alpha=0.3)
            ax_data.legend()
            
            # Bottom row: Current cycle recommendations with scores
            ax_rec = axes[1, i]
            
            # Background: all current data
            ax_rec.scatter(cumulative_data[x_col], cumulative_data[y_col],
                         c=cumulative_data[main_output], cmap='viridis',
                         s=30, alpha=0.4)
            
            # Highlight current cycle recommendations
            if len(recommendations) > 0:
                scatter = ax_rec.scatter(recommendations[x_col], recommendations[y_col],
                                       c=recommendations['boundary_score'], cmap='Reds',
                                       s=150, marker='^', edgecolors='darkred',
                                       linewidth=2, alpha=1.0)
                
                # Add colorbar for scores
                cbar = plt.colorbar(scatter, ax=ax_rec)
                cbar.set_label('Boundary Score')
                
                ax_rec.set_title(f'Cycle {cycle} Recommendations\nScores: {recommendations["boundary_score"].min():.3f}-{recommendations["boundary_score"].max():.3f}')
            else:
                ax_rec.set_title(f'Cycle {cycle}: No recommendations')
            
            ax_rec.set_xlabel(x_col)
            ax_rec.set_ylabel(y_col)
            ax_rec.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'{test_name}_iterative_evolution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Iterative visualization saved: {output_file}")


def run_all_tests():
    """Run comprehensive test suite."""
    
    print("="*80)
    print("GENERALIZED VECTOR EDGE REFINEMENT RECOMMENDER - TEST SUITE")
    print("="*80)
    
    # Initialize with coarser grid
    data_gen = SyntheticDataGenerator(
        x_range=(-2, 2), 
        y_range=(-2, 2), 
        grid_size=7,  # Coarser starting grid
        noise_level=0.03
    )
    
    tester = RecommenderTester('test_recommender_results')
    
    print(f"\n🗂️  ALL FILES WILL BE SAVED TO: {os.path.abspath('test_recommender_results')}")
    print(f"   Look for files ending in '_analysis.png' to see Output A vs Output B comparison!")
    
    # Test iterative exploration first
    print("\n" + "="*60)
    print("ITERATIVE BOUNDARY EXPLORATION TESTS")
    print("="*60)
    
    iterative_results = []
    
    # Iterative Test 1: Step Function (MAIN DEMO)
    iter_result1 = tester.run_iterative_exploration(
        test_name='step_function_iterative',
        data_generator=data_gen,
        output_function=lambda: data_gen.step_function_boundary(steep_factor=12),
        input_cols=['x', 'y'],
        output_cols=['output_A', 'output_B'],
        n_cycles=3,
        points_per_cycle=4,
        log_transform=False,
        normalization='zscore'
    )
    iterative_results.append(iter_result1)
    
    # Iterative Test 2: Circular Boundary (SIMPLIFIED)
    iter_result2 = tester.run_iterative_exploration(
        test_name='circular_iterative',
        data_generator=data_gen,
        output_function=lambda: data_gen.circular_boundary(center=(0, 0), radius=1.2, steep_factor=10),
        input_cols=['x', 'y'],
        output_cols=['output_A', 'output_B'],
        n_cycles=3,
        points_per_cycle=4,
        log_transform=False,
        normalization='zscore'
    )
    iterative_results.append(iter_result2)
    
    print("\n" + "="*60)
    print("SINGLE-SHOT COMPARISON TESTS")
    print("="*60)
    
    test_results = []
    
    # Test 1: Step Function Boundary (MAIN DEMO)
    result1 = tester.run_test_case(
        test_name='step_function_boundary',
        data_generator=data_gen,
        output_function=lambda: data_gen.step_function_boundary(steep_factor=12),
        input_cols=['x', 'y'],
        output_cols=['output_A', 'output_B'],
        n_points=8,
        log_transform=False,
        normalization='zscore'
    )
    test_results.append(result1)
    
    # Test 2: Circular Boundary (COMPARISON)
    result2 = tester.run_test_case(
        test_name='circular_boundary',
        data_generator=data_gen,
        output_function=lambda: data_gen.circular_boundary(center=(0, 0), radius=1.2, steep_factor=10),
        input_cols=['x', 'y'],
        output_cols=['output_A', 'output_B'],
        n_points=8,
        log_transform=False,
        normalization='zscore'
    )
    test_results.append(result2)
    
    # Summary report
    print("\\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    
    print("\\nITERATIVE EXPLORATION RESULTS:")
    for result in iterative_results:
        final_points = len(result['final_dataset'])
        n_cycles = len(result['cycle_results'])
        print(f"\\n{result['test_name']}:")
        print(f"  Cycles: {n_cycles}, Final points: {final_points}")
        print(f"  Points added per cycle: {[r['new_points_added'] for r in result['cycle_results']]}")
        
    print("\\nSINGLE-SHOT COMPARISON RESULTS:")
    for result in test_results:
        print(f"\\n{result['test_name']}:")
        print(f"  Points recommended: {len(result['recommendations'])}")
        if len(result['recommendations']) > 0:
            print(f"  Score range: {result['recommendations']['boundary_score'].min():.4f} - {result['recommendations']['boundary_score'].max():.4f}")
        print(f"  Coverage: X={result['metrics']['x_coverage']:.2f}, Y={result['metrics']['y_coverage']:.2f}")
        print(f"  Avg spacing: {result['metrics']['avg_spacing']:.3f}")
    
    print(f"\\nAll test results saved to: test_recommender_results/")
    print("\\nINTERPRETATION:")
    print("- ITERATIVE: Shows progressive boundary refinement over multiple cycles")
    print("- Each cycle finds new boundary points based on existing data")
    print("- Higher boundary scores = stronger transitions detected")
    print("- Good coverage (close to 1.0) = recommendations span the space")
    print("- Reasonable spacing = recommendations are well distributed")
    print("- Visualizations show where boundaries were detected vs actual patterns")
    print("- Red triangles show recommended boundary points from each cycle")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    run_all_tests()
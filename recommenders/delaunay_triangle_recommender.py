"""
Triangle-Based Adaptive Refinement using Delaunay Triangulation
===============================================================

A new recommender that uses Delaunay triangulation to adaptively refine
measurement points by finding triangles with high output variation and
proposing new points at triangle centroids.

Algorithm:
1. Work in log-space for concentration inputs  
2. Normalize outputs to comparable scales
3. Build Delaunay triangulation over measured points
4. Score triangles by output disagreement among vertices
5. Propose sampling at centroids of highest-scoring triangles
6. Apply minimum spacing constraints

Key advantages:
- No "missing corners" since vertices are always measured points
- Natural adaptation to irregular point distributions
- Works perfectly for 2D input spaces (surfactant concentrations)
- Clean geometric foundation

Usage:
    recommender = DelaunayTriangleRecommender(
        input_columns=['surf_A_conc_mm', 'surf_B_conc_mm'],  # X, Y (2D input)
        output_columns=['ratio'],                             # Focus on ratio
        log_transform_inputs=True,    # Use log space for inputs
        normalization_method='log_zscore'  # Normalize outputs
    )
    
    recommendations = recommender.get_recommendations(
        data_df, 
        n_points=12,
        min_spacing_factor=0.5
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial import Delaunay
import os
import warnings

class DelaunayTriangleRecommender:
    """
    Triangle-based refinement for 2D input spaces using Delaunay triangulation.
    
    Builds triangulation over measured points and scores triangles by output
    variation among vertices. Recommends sampling at triangle centroids.
    """
    
    def __init__(self, input_columns, output_columns, 
                 log_transform_inputs=True, normalization_method='log_zscore'):
        """
        Initialize the triangle-based recommender.
        
        Parameters:
        -----------
        input_columns : list of str
            Names of input variables [X, Y]. Must be exactly 2 dimensions for Delaunay.
        output_columns : list of str  
            Names of output variables (A, B, C, ...). Will be normalized.
        log_transform_inputs : bool
            Whether to work in log space for input variables (recommended for concentrations).
        normalization_method : str
            Method for normalizing outputs: 'log_zscore', 'zscore', 'minmax'
        """
        
        if len(input_columns) != 2:
            raise ValueError(f"Delaunay triangulation requires exactly 2 input dimensions, got {len(input_columns)}")
        
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.log_transform_inputs = log_transform_inputs
        self.normalization_method = normalization_method
        self.n_inputs = len(input_columns)
        self.n_outputs = len(output_columns)
        
        # Storage for fitted scalers
        self.scalers = {}
        
        print(f"Initialized DelaunayTriangleRecommender:")
        print(f"  Input variables (2D): {input_columns}")
        print(f"  Output variables ({self.n_outputs}): {output_columns}")
        print(f"  Log transform inputs: {log_transform_inputs}")
        print(f"  Normalization method: {normalization_method}")
    
    def get_recommendations(self, data_df, n_points=12, min_spacing_factor=0.5, 
                          tol_factor=0.1, triangle_score_method='max',
                          beta=0.5, output_dir=None, create_visualization=True):
        """
        Get triangle-based refinement recommendations from experimental data.
        
        Parameters:
        -----------
        data_df : pd.DataFrame
            Experimental data with input_columns and output_columns
        n_points : int
            Number of triangle centroids to recommend
        min_spacing_factor : float  
            Minimum spacing between points as fraction of nearest-neighbor distance
        tol_factor : float
            Tolerance for rejecting duplicate points as fraction of median NN distance
        triangle_score_method : str
            'max' = max pairwise distance, 'second_largest' = second largest distance
        beta : float
            Power factor for area scaling: score_adj = score * (area/median_area)^beta
            (0.0 = no area influence, 0.5 = modest influence, 1.0 = linear scaling)
        output_dir : str, optional
            Directory for saving visualizations and results
        create_visualization : bool
            Whether to create and save visualization plots
            
        Returns:
        --------
        pd.DataFrame
            Recommended sampling points with scores and metadata
        """
        
        print("\\n" + "="*70)
        print("DELAUNAY TRIANGLE REFINEMENT")
        print("="*70)
        
        # Step 1: Prepare and validate data
        print("\\n1. Preparing experimental data...")
        experiment_data = self._prepare_data(data_df)
        
        if len(experiment_data) < 3:
            raise RuntimeError(f"Need at least 3 points for triangulation, got {len(experiment_data)}")
        
        # Step 2: Transform inputs to log space
        print("\\n2. Transforming inputs to log space...")
        U = self._transform_inputs(experiment_data)
        
        # Step 3: Normalize outputs
        print("\\n3. Normalizing output variables...")
        Y_norm = self._normalize_outputs(experiment_data)
        
        # Step 4: Build Delaunay triangulation
        print("\\n4. Building Delaunay triangulation...")
        tri, triangles = self._build_triangulation(U)
        
        # Step 4.5: Debug triangulation (early visualization)
        print("\\n4.5. Debugging triangulation structure...")
        self._debug_triangulation(U, tri, triangles, experiment_data, output_dir)
        
        # Step 5: Score triangles
        print("\\n5. Scoring triangles by output disagreement...")
        triangle_scores = self._score_triangles(triangles, Y_norm, U, triangle_score_method, beta)
        
        # Step 5.5: Create visualization with scores overlaid
        print("\\n5.5. Creating scored triangulation visualization...")
        self._create_scored_visualization(U, tri, triangle_scores, experiment_data, output_dir)
        
        # Step 6: Select top triangle centroids with spacing
        print("\\n6. Selecting triangle centroids with spacing...")
        selected_centroids = self._select_centroids(triangle_scores, U, n_points, 
                                                   min_spacing_factor, tol_factor)
        
        # Step 7: Format recommendations
        print("\\n7. Formatting recommendations...")
        recommendations = self._format_recommendations(selected_centroids, output_dir)
        
        # Step 8: Visualization (optional)
        if create_visualization and output_dir:
            print("\\n8. Creating visualization...")
            self._create_visualization(U, Y_norm, tri, triangle_scores, 
                                     selected_centroids, experiment_data, output_dir)
        
        self._print_summary(triangle_scores, selected_centroids)
        
        return recommendations
    
    def _prepare_data(self, data_df):
        """Prepare and validate experimental data."""
        
        # Filter to experimental data if 'well_type' column exists
        if 'well_type' in data_df.columns:
            experiment_data = data_df[data_df['well_type'] == 'experiment'].copy()
            print(f"  Filtered to {len(experiment_data)} experimental points")
        else:
            experiment_data = data_df.copy()
            print(f"  Using all {len(experiment_data)} data points")
        
        # Validate required columns
        missing_inputs = [col for col in self.input_columns if col not in experiment_data.columns]
        missing_outputs = [col for col in self.output_columns if col not in experiment_data.columns]
        
        if missing_inputs:
            raise ValueError(f"Missing input columns: {missing_inputs}")
        if missing_outputs:
            raise ValueError(f"Missing output columns: {missing_outputs}")
        
        # Show data ranges
        for col in self.input_columns:
            min_val, max_val = experiment_data[col].min(), experiment_data[col].max()
            print(f"  {col}: {min_val:.4e} - {max_val:.4e}")
        
        for col in self.output_columns:
            min_val, max_val = experiment_data[col].min(), experiment_data[col].max()
            print(f"  {col}: {min_val:.4f} - {max_val:.4f}")
        
        return experiment_data
    
    def _transform_inputs(self, experiment_data):
        """Transform inputs to log space if requested."""
        
        if self.log_transform_inputs:
            U = np.zeros((len(experiment_data), 2))
            for i, col in enumerate(self.input_columns):
                values = experiment_data[col].values
                U[:, i] = np.log10(values)
                print(f"  {col}: log range {U[:, i].min():.3f} to {U[:, i].max():.3f}")
        else:
            U = experiment_data[self.input_columns].values
            print(f"  Using original input space")
        
        return U
    
    def _normalize_outputs(self, experiment_data):
        """Normalize output variables using specified method."""
        
        Y_norm = np.zeros((len(experiment_data), self.n_outputs))
        self.normalized_output_columns = []
        
        for j, col in enumerate(self.output_columns):
            values = experiment_data[col].values
            values = np.asarray(values, dtype=np.float64)
            
            if self.normalization_method == 'log_zscore':
                # Log transform + z-score
                epsilon = 1e-6
                values_with_epsilon = values + epsilon
                log_values = np.log10(values_with_epsilon)
                
                scaler = StandardScaler()
                normalized = scaler.fit_transform(log_values.reshape(-1, 1)).flatten()
                
                Y_norm[:, j] = normalized
                norm_col = f'{col}_normalized'
                self.normalized_output_columns.append(norm_col)
                self.scalers[col] = scaler
                
                print(f"  {col}: {values.min():.4f}-{values.max():.4f} → log → z-score: {normalized.min():.3f}-{normalized.max():.3f}")
                
                # Store ranges for debugging
                if not hasattr(self, '_debug_norm_ranges'):
                    self._debug_norm_ranges = {}
                self._debug_norm_ranges[col] = (normalized.min(), normalized.max())
                
            elif self.normalization_method == 'zscore':
                scaler = StandardScaler() 
                normalized = scaler.fit_transform(values.reshape(-1, 1)).flatten()
                
                Y_norm[:, j] = normalized
                norm_col = f'{col}_normalized'
                self.normalized_output_columns.append(norm_col)
                self.scalers[col] = scaler
                
                print(f"  {col}: {values.min():.4f}-{values.max():.4f} → z-score: {normalized.min():.3f}-{normalized.max():.3f}")
                
            elif self.normalization_method == 'minmax':
                normalized = (values - values.min()) / (values.max() - values.min())
                
                Y_norm[:, j] = normalized
                norm_col = f'{col}_normalized'
                self.normalized_output_columns.append(norm_col)
                
                print(f"  {col}: {values.min():.4f}-{values.max():.4f} → minmax: {normalized.min():.3f}-{normalized.max():.3f}")
        
        return Y_norm
    
    def _build_triangulation(self, U):
        """Build Delaunay triangulation over input points."""
        
        try:
            tri = Delaunay(U)
            triangles = tri.simplices
            
            print(f"  Built triangulation: {len(U)} points → {len(triangles)} triangles")
            
            # Basic quality check
            areas = []
            for triangle in triangles:
                i, j, k = triangle
                # Triangle area = 0.5 * |cross product|
                v1 = U[j] - U[i]
                v2 = U[k] - U[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
            
            areas = np.array(areas)
            print(f"  Triangle areas: min={areas.min():.4f}, max={areas.max():.4f}, median={np.median(areas):.4f}")
            
            return tri, triangles
            
        except Exception as e:
            raise RuntimeError(f"Delaunay triangulation failed: {e}")
    
    def _debug_triangulation(self, U, tri, triangles, experiment_data, output_dir=None):
        """Debug triangulation by showing triangle connections and creating early visualization."""
        
        print(f"  Triangulation details:")
        print(f"    Points: {len(U)}")
        print(f"    Triangles: {len(triangles)}")
        
        # Show first few triangles in detail
        print(f"\n  First 15 triangles (vertex indices -> grid positions -> original ratios -> orig diffs):")
        for t_idx, triangle in enumerate(triangles[:15]):
            i, j, k = triangle
            
            # Get grid positions if available
            grid_info = ""
            if 'grid_i' in experiment_data.columns and 'grid_j' in experiment_data.columns:
                try:
                    pos_i = f"({experiment_data.iloc[i]['grid_i']:.0f},{experiment_data.iloc[i]['grid_j']:.0f})"
                    pos_j = f"({experiment_data.iloc[j]['grid_i']:.0f},{experiment_data.iloc[j]['grid_j']:.0f})"
                    pos_k = f"({experiment_data.iloc[k]['grid_i']:.0f},{experiment_data.iloc[k]['grid_j']:.0f})"
                    grid_info = f" -> grid[{pos_i},{pos_j},{pos_k}]"
                except:
                    grid_info = " -> grid[?,?,?]"  # Handle new points without grid positions
            
            # Get output values and calculate original differences
            if 'ratio' in experiment_data.columns:
                ratio_i = experiment_data.iloc[i]['ratio']
                ratio_j = experiment_data.iloc[j]['ratio']
                ratio_k = experiment_data.iloc[k]['ratio']
                
                # Calculate original space differences
                diff_ij = abs(ratio_i - ratio_j)
                diff_ik = abs(ratio_i - ratio_k)
                diff_jk = abs(ratio_j - ratio_k)
                max_orig_diff = max(diff_ij, diff_ik, diff_jk)
                
                output_info = f" -> ratios[{ratio_i:.3f},{ratio_j:.3f},{ratio_k:.3f}] -> max_diff:{max_orig_diff:.3f}"
                
                # Flag significant differences
                flag = " **HIGH DIFF**" if max_orig_diff > 0.3 else ""
                output_info += flag
            else:
                output_info = ""
            
            print(f"    {t_idx+1:2d}. vertices[{i},{j},{k}]{grid_info}{output_info}")
        
        if len(triangles) > 15:
            print(f"    ... {len(triangles)-15} more triangles")
        
        # SPECIFIC DEBUG: Look for triangles connecting high-value regions
        print(f"\n  SEARCHING FOR HIGH-DIFFERENCE TRIANGLES:")
        high_diff_triangles = []
        for t_idx, triangle in enumerate(triangles):
            i, j, k = triangle
            if 'ratio' in experiment_data.columns:
                ratio_i = experiment_data.iloc[i]['ratio']
                ratio_j = experiment_data.iloc[j]['ratio']
                ratio_k = experiment_data.iloc[k]['ratio']
                
                # Calculate original space differences
                diff_ij = abs(ratio_i - ratio_j)
                diff_ik = abs(ratio_i - ratio_k)
                diff_jk = abs(ratio_j - ratio_k)
                max_orig_diff = max(diff_ij, diff_ik, diff_jk)
                
                if max_orig_diff > 0.3:  # Significant difference
                    high_diff_triangles.append((t_idx, triangle, max_orig_diff, [ratio_i, ratio_j, ratio_k]))
        
        if high_diff_triangles:
            print(f"    Found {len(high_diff_triangles)} triangles with significant differences (>0.3):")
            for t_idx, triangle, max_diff, ratios in high_diff_triangles:
                print(f"      Triangle {t_idx}: vertices {triangle} -> ratios {[f'{r:.3f}' for r in ratios]} -> diff: {max_diff:.3f}")
        else:
            print(f"    NO triangles found with differences >0.3!")
            print(f"    This suggests Delaunay triangulation isn't connecting distant regions.")
            
            # Sample some boundary points to understand connections
            print(f"\n    Checking specific point connections:")
            # Find points with ratio ~1.4 and ~1.0
            high_points = [i for i, row in experiment_data.iterrows() if row['ratio'] > 1.3]
            mid_points = [i for i, row in experiment_data.iterrows() if 0.9 < row['ratio'] < 1.1]
            
            print(f"      High ratio points (>1.3): {high_points}")
            print(f"      Mid ratio points (0.9-1.1): {mid_points}")
            
            # Check if any triangles connect these regions
            connecting_found = False
            for triangle in triangles:
                vertices = set(triangle)
                has_high = any(v in high_points for v in vertices)
                has_mid = any(v in mid_points for v in vertices)
                if has_high and has_mid:
                    connecting_found = True
                    print(f"      FOUND connecting triangle: {triangle}")
                    break
            
            if not connecting_found:
                print(f"      No triangles connect high (>1.3) to mid (0.9-1.1) regions!")
        
        # Create immediate visualization
        print(f"\n  Creating early triangulation visualization...")
        self._create_early_visualization(U, tri, experiment_data, output_dir)
    
    def _create_early_visualization(self, U, tri, experiment_data, output_dir=None):
        """Create simple triangulation visualization for debugging."""
        
        try:
            import matplotlib.pyplot as plt
            import os
            from datetime import datetime
            
            # Create output directory structure
            if output_dir:
                viz_dir = os.path.join(output_dir, 'recommender_visualizations')
                os.makedirs(viz_dir, exist_ok=True)
            else:
                viz_dir = os.getcwd()
            
            # Create separate visualization for each output variable
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for output_var in self.output_columns:
                if output_var not in experiment_data.columns:
                    print(f"      Warning: {output_var} not found in data, skipping...")
                    continue
                    
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                
                # Plot triangles
                ax.triplot(U[:, 0], U[:, 1], tri.simplices, 'b-', alpha=0.4, linewidth=1)
                
                # Get values for this output variable
                output_values = experiment_data[output_var].values
                
                # Choose appropriate colormap based on variable type
                if 'turbidity' in output_var.lower():
                    cmap = 'plasma'  # Good for turbidity (purple to yellow)
                    label = 'Turbidity (600nm)'
                elif 'ratio' in output_var.lower():
                    cmap = 'viridis'  # Good for ratio (blue to yellow)
                    label = 'Fluorescence Ratio'
                else:
                    cmap = 'coolwarm'  # Default
                    label = output_var
                
                # Plot points colored by output values
                scatter = ax.scatter(U[:, 0], U[:, 1], c=output_values, s=100, 
                                   cmap=cmap, alpha=0.8, edgecolors='black', linewidth=1)
                plt.colorbar(scatter, ax=ax, label=label)
                
                # Add point labels with indices and values
                for i, (x, y) in enumerate(U):
                    ax.annotate(f'{i}\n{output_values[i]:.3f}', (x, y), 
                              textcoords="offset points", xytext=(0,10), ha='center',
                              fontsize=8, alpha=0.7)
                
                ax.set_xlabel(f'{self.input_columns[0]} (log space)' if self.log_transform_inputs else self.input_columns[0])
                ax.set_ylabel(f'{self.input_columns[1]} (log space)' if self.log_transform_inputs else self.input_columns[1])
                ax.set_title(f'Early Debug: Delaunay Triangulation - {label}')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save with descriptive filename
                clean_var_name = output_var.replace('_', '-')
                debug_filename = os.path.join(viz_dir, f'debug_triangulation_{clean_var_name}_{timestamp}.png')
                plt.savefig(debug_filename, dpi=150, bbox_inches='tight')
                print(f"    Debug visualization saved: {debug_filename}")
                
                plt.close(fig)  # Close to free memory
                
            # plt.show()  # Disabled - save to output instead
        except ImportError:
            print(f"    Warning: matplotlib not available for debug visualization")
        except Exception as e:
            print(f"    Warning: Could not create debug visualization: {e}")
    
    def _create_combined_early_visualization(self, U, tri, experiment_data, viz_dir, timestamp):
        """Create combined visualization showing multiple outputs side by side."""
        
        try:
            import matplotlib.pyplot as plt
            
            n_outputs = len(self.output_columns)
            fig, axes = plt.subplots(1, n_outputs, figsize=(6 * n_outputs, 6))
            
            if n_outputs == 1:
                axes = [axes]  # Make it iterable
            
            for idx, output_var in enumerate(self.output_columns):
                ax = axes[idx]
                
                if output_var not in experiment_data.columns:
                    ax.text(0.5, 0.5, f'{output_var}\\nnot available', 
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Plot triangles
                ax.triplot(U[:, 0], U[:, 1], tri.simplices, 'b-', alpha=0.4, linewidth=1)
                
                # Get values and colormap
                output_values = experiment_data[output_var].values
                
                if 'turbidity' in output_var.lower():
                    cmap = 'plasma'
                    label = 'Turbidity (600nm)'
                elif 'ratio' in output_var.lower():
                    cmap = 'viridis'
                    label = 'Fluorescence Ratio'
                else:
                    cmap = 'coolwarm'
                    label = output_var
                
                # Plot points
                scatter = ax.scatter(U[:, 0], U[:, 1], c=output_values, s=80, 
                                   cmap=cmap, alpha=0.8, edgecolors='black', linewidth=1)
                plt.colorbar(scatter, ax=ax, label=label)
                
                ax.set_xlabel(f'{self.input_columns[0]} (log space)' if self.log_transform_inputs else self.input_columns[0])
                ax.set_ylabel(f'{self.input_columns[1]} (log space)' if self.log_transform_inputs else self.input_columns[1])
                ax.set_title(f'{label}')
                ax.grid(True, alpha=0.3)
            
            plt.suptitle('Early Debug: Delaunay Triangulation - All Outputs', fontsize=14)
            plt.tight_layout()
            
            combined_filename = os.path.join(viz_dir, f'debug_triangulation_combined_{timestamp}.png')
            plt.savefig(combined_filename, dpi=150, bbox_inches='tight')
            print(f"    Combined debug visualization saved: {combined_filename}")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"    Warning: Could not create combined debug visualization: {e}")
    
    def _create_scored_visualization(self, U, tri, triangle_scores, experiment_data, output_dir=None):
        """Create triangulation visualization with triangle scores displayed on each triangle."""
        
        try:
            import matplotlib.pyplot as plt
            import os
            from datetime import datetime
            
            # Create output directory structure
            if output_dir:
                viz_dir = os.path.join(output_dir, 'recommender_visualizations')
                os.makedirs(viz_dir, exist_ok=True)
            else:
                viz_dir = os.getcwd()
            
            # Create separate scored visualization for each output variable
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for output_var in self.output_columns:
                if output_var not in experiment_data.columns:
                    print(f"      Warning: {output_var} not found in data, skipping scored visualization...")
                    continue
                    
                fig, ax = plt.subplots(1, 1, figsize=(14, 10))
                
                # Plot triangles
                ax.triplot(U[:, 0], U[:, 1], tri.simplices, 'b-', alpha=0.4, linewidth=1)
                
                # Get values for this output variable
                output_values = experiment_data[output_var].values
                
                # Choose appropriate colormap and label
                if 'turbidity' in output_var.lower():
                    cmap = 'plasma'
                    label = 'Turbidity (600nm)'
                elif 'ratio' in output_var.lower():
                    cmap = 'viridis'
                    label = 'Fluorescence Ratio'
                else:
                    cmap = 'coolwarm'
                    label = output_var
                
                # Plot points colored by output values
                scatter = ax.scatter(U[:, 0], U[:, 1], c=output_values, s=120, 
                                   cmap=cmap, alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
                plt.colorbar(scatter, ax=ax, label=label)
                
                # Add point labels with indices and values (smaller font)
                for i, (x, y) in enumerate(U):
                    ax.annotate(f'{i}\\n{output_values[i]:.3f}', (x, y), 
                              textcoords="offset points", xytext=(0,12), ha='center',
                              fontsize=7, alpha=0.8, zorder=6)
                
                # Add triangle scores at triangle centroids
                print(f"    Adding triangle scores to {output_var} visualization...")
                for ts in triangle_scores:
                    triangle_idx = ts['triangle_idx']
                    score = ts['score']
                    vertices = ts['vertices']
                    
                    # Calculate triangle centroid for label placement
                    i, j, k = vertices
                    centroid_x = (U[i, 0] + U[j, 0] + U[k, 0]) / 3.0
                    centroid_y = (U[i, 1] + U[j, 1] + U[k, 1]) / 3.0
                    
                    # Color code by score level
                    if score > 2.0:
                        color = 'red'
                        fontweight = 'bold'
                    elif score > 1.0:
                        color = 'orange'
                        fontweight = 'bold'
                    elif score > 0.1:
                        color = 'blue'
                        fontweight = 'normal'
                    else:
                        color = 'gray'
                        fontweight = 'normal'
                    
                    # Add score label with background
                    ax.annotate(f'{score:.2f}', (centroid_x, centroid_y),
                              ha='center', va='center',
                              fontsize=10, fontweight=fontweight, color=color,
                              bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8),
                              zorder=7)
                    
                    # Also add a small triangle index number
                    ax.annotate(f'T{triangle_idx}', (centroid_x, centroid_y),
                              ha='center', va='bottom',
                              textcoords="offset points", xytext=(0, 15),
                              fontsize=6, alpha=0.6, zorder=8)
                
                ax.set_xlabel(f'{self.input_columns[0]} (log space)' if self.log_transform_inputs else self.input_columns[0])
                ax.set_ylabel(f'{self.input_columns[1]} (log space)' if self.log_transform_inputs else self.input_columns[1])
                ax.set_title(f'Delaunay Triangulation with Triangle Scores - {label}')
                ax.grid(True, alpha=0.3)
                
                # Add score legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Score > 2.0'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Score > 1.0'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Score > 0.1'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Score ≤ 0.1')
                ]
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
                
                plt.tight_layout()
                
                # Save with descriptive filename
                clean_var_name = output_var.replace('_', '-')
                scored_filename = os.path.join(viz_dir, f'scored_triangulation_{clean_var_name}_{timestamp}.png')
                plt.savefig(scored_filename, dpi=150, bbox_inches='tight')
                print(f"    Scored visualization saved: {scored_filename}")
                
                plt.close(fig)  # Close to free memory
            
            # plt.show()  # Disabled - save to output instead
        except ImportError:
            print(f"    Warning: matplotlib not available for scored visualization")
        except Exception as e:
            print(f"    Warning: Could not create scored visualization: {e}")
    
    def _score_triangles(self, triangles, Y_norm, U, score_method='max', beta=0.5):
        """Score each triangle by output disagreement among vertices with power-based area scaling."""
        
        triangle_scores = []
        all_areas = []
        
        # First pass: calculate all basic scores and areas
        for t_idx, triangle in enumerate(triangles):
            i, j, k = triangle
            
            # Get output vectors for the three vertices
            a = Y_norm[i]  # shape (n_outputs,)
            b = Y_norm[j]
            c = Y_norm[k]
            
            # Calculate pairwise distances in output space
            d_ab = np.linalg.norm(a - b)
            d_ac = np.linalg.norm(a - c)
            d_bc = np.linalg.norm(b - c)
            
            distances = [d_ab, d_ac, d_bc]
            
            # Base triangle score based on method
            if score_method == 'max':
                base_score = max(distances)
            elif score_method == 'second_largest':
                sorted_distances = sorted(distances, reverse=True)
                base_score = sorted_distances[1] if len(sorted_distances) >= 2 else sorted_distances[0]
            else:
                raise ValueError(f"Unknown score method: {score_method}")
            
            # Calculate triangle area
            v1 = U[j] - U[i]
            v2 = U[k] - U[i]
            area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
            all_areas.append(area)
            
            # Calculate centroid in input space
            centroid = (U[i] + U[j] + U[k]) / 3.0
            
            triangle_scores.append({
                'triangle_idx': t_idx,
                'vertices': (i, j, k),
                'base_score': base_score,
                'area': area,
                'max_distance': max(distances),
                'second_largest_distance': sorted(distances, reverse=True)[1] if len(distances) >= 2 else max(distances),
                'centroid_log': centroid,
                'distances': {'d_ab': d_ab, 'd_ac': d_ac, 'd_bc': d_bc},
                'normalized_outputs': {'a': a, 'b': b, 'c': c}  # Store for debugging
            })
        
        # Second pass: apply power-based area scaling
        areas = np.array(all_areas)
        median_area = np.median(areas)
        max_area = np.max(areas)
        
        print(f"  Area statistics: min={areas.min():.4f}, max={max_area:.4f}, median={median_area:.4f}")
        print(f"  Applying power-based area scaling with beta: {beta:.2f}")
        
        for ts in triangle_scores:
            area = ts['area']
            base_score = ts['base_score']
            
            if beta == 0.0:
                # No area influence
                area_factor = 1.0
                area_adjustment = 0.0
            else:
                # Apply power-based scaling: score_adj = score * (area/median_area)^beta
                area_ratio = area / median_area if median_area > 0 else 1.0
                area_factor = area_ratio ** beta
                area_adjustment = area_factor - 1.0  # For debugging/display
            
            final_score = base_score * area_factor
            
            ts['area_factor'] = area_factor
            ts['area_adjustment'] = area_adjustment
            ts['score'] = final_score
        
        # Sort by final score descending
        triangle_scores = sorted(triangle_scores, key=lambda x: x['score'], reverse=True)
        
        scores = [ts['score'] for ts in triangle_scores]
        print(f"  Triangle scores (with area scaling): min={min(scores):.4f}, max={max(scores):.4f}, median={np.median(scores):.4f}")
        
        # DEBUG: Show detailed scoring 
        print(f"\n  DETAILED TRIANGLE SCORING DEBUG:")
        self._debug_triangle_scores(triangle_scores[:10])
        
        return triangle_scores
    
    def _debug_triangle_scores(self, top_triangle_scores):
        """Debug triangle scoring to understand why certain triangles aren't selected."""
        
        print(f"  Top 10 triangles with detailed scoring:")
        print(f"  {'Rank':<4} {'Vertices':<12} {'Base':<8} {'Area':<8} {'Factor':<8} {'Final':<8} {'Issue?':<15}")
        print(f"  {'-'*75}")
        
        for rank, ts in enumerate(top_triangle_scores, 1):
            vertices = ts['vertices']
            base_score = ts.get('base_score', ts['score'])  # fallback for compatibility
            area = ts['area']
            area_factor = ts.get('area_factor', 1.0)
            final_score = ts['score']
            
            # Check for potential issues
            issue = ""
            if final_score < 0.1:
                issue = "Very low score"
            elif area_factor > 2.0:
                issue = "High area factor"
            elif area_factor < 0.5:
                issue = "Low area factor"
            
            print(f"  {rank:<4} {str(vertices):<12} {base_score:<8.3f} {area:<8.3f} {area_factor:<8.3f} {final_score:<8.3f} {issue:<15}")
        
        print(f"\n  Normalized output ranges for reference:")
        if hasattr(self, '_debug_norm_ranges'):
            for col, (min_norm, max_norm) in self._debug_norm_ranges.items():
                print(f"    {col}: {min_norm:.3f} to {max_norm:.3f}")
    
    def _select_centroids(self, triangle_scores, U, n_points, min_spacing_factor, tol_factor):
        """Select top triangle centroids with spacing constraints."""
        
        if len(triangle_scores) == 0:
            raise RuntimeError("No triangle scores available for selection")
        
        # Calculate characteristic spacing from nearest neighbors
        nn_distances = []
        for i in range(len(U)):
            distances = np.linalg.norm(U - U[i], axis=1)
            distances[i] = np.inf  # Exclude self
            nn_distances.append(np.min(distances))
        
        median_nn_dist = np.median(nn_distances)
        d_min = min_spacing_factor * median_nn_dist
        tol = tol_factor * median_nn_dist
        
        print(f"  Median nearest-neighbor distance: {median_nn_dist:.4f}")
        print(f"  Minimum spacing constraint: {d_min:.4f}")
        print(f"  Duplicate tolerance: {tol:.4f}")
        
        selected_centroids = []
        rejected_too_close = 0
        rejected_duplicate = 0
        
        for triangle_data in triangle_scores:
            if len(selected_centroids) >= n_points:
                break
            
            centroid_log = triangle_data['centroid_log']
            
            # Check for duplicates (too close to existing points)
            too_close_to_existing = False
            for existing_point in U:
                distance = np.linalg.norm(centroid_log - existing_point)
                if distance < tol:
                    too_close_to_existing = True
                    break
            
            if too_close_to_existing:
                rejected_duplicate += 1
                continue
            
            # Check spacing with already selected centroids
            too_close_to_selected = False
            for selected in selected_centroids:
                distance = np.linalg.norm(centroid_log - selected['centroid_log'])
                if distance < d_min:
                    too_close_to_selected = True
                    break
            
            if too_close_to_selected:
                rejected_too_close += 1
                continue
            
            # Accept this centroid
            selected_centroids.append(triangle_data)
        
        print(f"  Selected {len(selected_centroids)} centroids (requested {n_points})")
        print(f"  Rejected {rejected_duplicate} duplicates, {rejected_too_close} for spacing")
        
        return selected_centroids
    
    def _format_recommendations(self, selected_centroids, output_dir):
        """Format selected centroids as recommendations dataframe."""
        
        if len(selected_centroids) == 0:
            # Return empty dataframe with correct columns
            columns = self.input_columns + ['triangle_score', 'max_distance', 'second_largest_distance', 'triangle_area']
            return pd.DataFrame(columns=columns)
        
        recommendations_data = []
        
        for i, centroid_data in enumerate(selected_centroids):
            rec_dict = {}
            
            # Convert centroid back to original input space
            centroid_log = centroid_data['centroid_log']
            
            if self.log_transform_inputs:
                # Convert from log space back to original
                for j, col in enumerate(self.input_columns):
                    rec_dict[col] = 10 ** centroid_log[j]
            else:
                # Already in original space
                for j, col in enumerate(self.input_columns):
                    rec_dict[col] = centroid_log[j]
            
            # Add scores and metadata
            rec_dict['triangle_score'] = centroid_data['score']
            rec_dict['max_distance'] = centroid_data['max_distance']
            rec_dict['second_largest_distance'] = centroid_data['second_largest_distance']
            rec_dict['triangle_area'] = centroid_data['area']
            rec_dict['triangle_vertices'] = str(centroid_data['vertices'])
            rec_dict['rank'] = i + 1
            
            recommendations_data.append(rec_dict)
        
        recommendations_df = pd.DataFrame(recommendations_data)
        
        # Save to file if output directory specified
        if output_dir and len(recommendations_df) > 0:
            os.makedirs(output_dir, exist_ok=True)
            recommendations_file = os.path.join(output_dir, 'delaunay_triangle_recommendations.csv')
            recommendations_df.to_csv(recommendations_file, index=False)
            print(f"  Saved recommendations to: {recommendations_file}")
        
        return recommendations_df
    
    def _create_visualization(self, U, Y_norm, tri, triangle_scores, selected_centroids, 
                            experiment_data, output_dir):
        """Create triangulation visualization with selected centroids."""
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Delaunay Triangle Refinement Visualization', fontsize=14)
        
        # Plot 1: Triangulation with scores
        ax1 = axes[0]
        
        # Plot triangulation
        ax1.triplot(U[:, 0], U[:, 1], tri.simplices, 'k-', alpha=0.3, linewidth=0.5)
        
        # Color existing points by first output
        if self.n_outputs >= 1:
            scatter = ax1.scatter(U[:, 0], U[:, 1], c=Y_norm[:, 0], s=60, 
                                cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax1, label=f'{self.output_columns[0]} (normalized)')
        
        # Highlight selected triangle centroids
        if len(selected_centroids) > 0:
            centroids_log = np.array([sc['centroid_log'] for sc in selected_centroids])
            ax1.scatter(centroids_log[:, 0], centroids_log[:, 1], 
                       c='red', s=100, marker='*', edgecolors='black', linewidth=1,
                       label='Selected Centroids', zorder=5)
        
        ax1.set_xlabel(f'{self.input_columns[0]} (log space)' if self.log_transform_inputs else self.input_columns[0])
        ax1.set_ylabel(f'{self.input_columns[1]} (log space)' if self.log_transform_inputs else self.input_columns[1])
        ax1.set_title('Triangulation & Selected Centroids')
        ax1.legend()
        
        # Plot 2: Triangle scores
        ax2 = axes[1]
        
        # Plot triangulation
        ax2.triplot(U[:, 0], U[:, 1], tri.simplices, 'k-', alpha=0.3, linewidth=0.5)
        
        # Color triangles by score (show centroids of all triangles)
        all_centroids = np.array([ts['centroid_log'] for ts in triangle_scores])
        all_scores = np.array([ts['score'] for ts in triangle_scores])
        
        scatter2 = ax2.scatter(all_centroids[:, 0], all_centroids[:, 1], 
                             c=all_scores, s=80, cmap='plasma', alpha=0.7,
                             edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter2, ax=ax2, label='Triangle Score')
        
        # Highlight top triangles
        top_n = min(len(selected_centroids), 5)
        if top_n > 0:
            top_centroids = all_centroids[:top_n]
            ax2.scatter(top_centroids[:, 0], top_centroids[:, 1], 
                       c='red', s=120, marker='s', edgecolors='white', linewidth=2,
                       label=f'Top {top_n} Triangles', zorder=5)
        
        ax2.set_xlabel(f'{self.input_columns[0]} (log space)' if self.log_transform_inputs else self.input_columns[0])
        ax2.set_ylabel(f'{self.input_columns[1]} (log space)' if self.log_transform_inputs else self.input_columns[1])
        ax2.set_title('Triangle Scores')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory for visualization
        viz_dir = os.path.join(output_dir, 'recommender_visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        plot_filename = os.path.join(viz_dir, f'delaunay_triangulation_{timestamp}.png')
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"  Visualization saved: {plot_filename}")
        
        # plt.show()  # Disabled - save to output instead
    
    def _print_summary(self, triangle_scores, selected_centroids):
        """Print summary of triangle analysis results."""
        
        print("\\n" + "="*70)
        print("DELAUNAY TRIANGLE REFINEMENT SUMMARY")
        print("="*70)
        
        if len(triangle_scores) > 0:
            print(f"Total triangles evaluated: {len(triangle_scores)}")
            scores = [ts['score'] for ts in triangle_scores]
            print(f"Triangle score range: {min(scores):.4f} to {max(scores):.4f}")
            
            print(f"\\nTop 5 triangles by disagreement score:")
            for i, triangle_data in enumerate(triangle_scores[:5]):
                vertices = triangle_data['vertices']
                score = triangle_data['score']
                area = triangle_data['area']
                print(f"  {i+1}. Vertices {vertices}, Score: {score:.4f}, Area: {area:.4f}")
        
        if len(selected_centroids) > 0:
            print(f"\\nSelected {len(selected_centroids)} triangle centroids:")
            for i, centroid_data in enumerate(selected_centroids):
                centroid_log = centroid_data['centroid_log']
                score = centroid_data['score']
                
                if self.log_transform_inputs:
                    coords_str = ", ".join([f"{self.input_columns[j]}={10**centroid_log[j]:.3e}" 
                                          for j in range(len(self.input_columns))])
                else:
                    coords_str = ", ".join([f"{self.input_columns[j]}={centroid_log[j]:.3e}" 
                                          for j in range(len(self.input_columns))])
                
                print(f"  {i+1}. {coords_str} (score: {score:.4f})")
        else:
            print("\\nNo triangle centroids selected!")
        
        print("="*70)
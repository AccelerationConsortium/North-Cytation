"""
Generalized Vector Edge Refinement Recommender
===============================================

A generalized version of vector edge refinement that works with:
- Arbitrary n-dimensional input variables (X, Y, Z, ...)  
- Arbitrary output variables (A, B, C, ...)
- Automatic normalization and vector field analysis
- Returns top boundary refinement recommendations

Usage:
    recommender = GeneralizedVectorEdgeRecommender(
        input_columns=['surf_A_conc_mm', 'surf_B_conc_mm'],  # X, Y
        output_columns=['turbidity_600', 'ratio'],           # A, B
        log_transform_inputs=True,    # Use log space for inputs
        normalization_method='log_zscore'  # Normalize outputs
    )
    
    recommendations = recommender.get_recommendations(
        data_df, 
        n_points=32,
        min_spacing_factor=0.5
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os
from itertools import combinations
import warnings

class GeneralizedVectorEdgeRecommender:
    """
    Generalized vector edge refinement for n-dimensional input/output spaces.
    
    Treats multiple output variables as components of a vector field and finds
    the strongest boundaries by measuring vector changes across edges between
    neighboring grid points in n-dimensional input space.
    """
    
    def __init__(self, input_columns, output_columns, 
                 log_transform_inputs=True, normalization_method='log_zscore'):
        """
        Initialize the generalized recommender.
        
        Parameters:
        -----------
        input_columns : list of str
            Names of input variables (X, Y, Z, ...). Must form a regular grid.
        output_columns : list of str  
            Names of output variables (A, B, C, ...). Will be normalized.
        log_transform_inputs : bool
            Whether to work in log space for input variables (recommended for concentrations).
        normalization_method : str
            Method for normalizing outputs: 'log_zscore', 'zscore', 'minmax'
        """
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.log_transform_inputs = log_transform_inputs
        self.normalization_method = normalization_method
        self.n_inputs = len(input_columns)
        self.n_outputs = len(output_columns)
        
        # Storage for fitted scalers
        self.scalers = {}
        
        print(f"Initialized GeneralizedVectorEdgeRecommender:")
        print(f"  Input variables ({self.n_inputs}): {input_columns}")
        print(f"  Output variables ({self.n_outputs}): {output_columns}")
        print(f"  Log transform inputs: {log_transform_inputs}")
        print(f"  Normalization method: {normalization_method}")
    
    def get_recommendations(self, data_df, n_points=32, min_spacing_factor=0.5, 
                          output_dir=None, create_visualization=True):
        """
        Get boundary refinement recommendations from experimental data.
        
        Parameters:
        -----------
        data_df : pd.DataFrame
            Experimental data with input_columns and output_columns
        n_points : int
            Number of boundary points to recommend
        min_spacing_factor : float  
            Minimum spacing between points as fraction of grid step
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
        print("GENERALIZED VECTOR EDGE REFINEMENT")
        print("="*70)
        
        # Step 1: Prepare and validate data
        print("\\n1. Preparing experimental data...")
        experiment_data = self._prepare_data(data_df)
        
        # Step 2: Normalize outputs
        print("\\n2. Normalizing vector field outputs...")
        experiment_data = self._normalize_outputs(experiment_data)
        
        # Step 3: Create grid structure  
        print("\\n3. Creating n-dimensional grid structure...")
        grid_data, grid_coordinates = self._create_grid_structure(experiment_data)
        
        # Step 4: Calculate edge scores
        print("\\n4. Calculating edge scores (vector changes)...")
        edges_df = self._calculate_edge_scores(grid_data, grid_coordinates)
        
        # Step 5: Select boundary points
        print("\\n5. Selecting boundary refinement points...")
        selected_df = self._select_boundary_points(edges_df, grid_coordinates, 
                                                 n_points, min_spacing_factor)
        
        # Step 6: Visualization
        if create_visualization and output_dir:
            print("\\n6. Creating visualization...")
            self._create_visualization(experiment_data, edges_df, selected_df, output_dir)
        
        # Step 7: Format recommendations
        print("\\n7. Formatting recommendations...")
        recommendations = self._format_recommendations(selected_df, output_dir)
        
        self._print_summary(edges_df, selected_df)
        
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
            print(f"  {col}: {min_val:.4f} - {max_val:.4f}")
        
        for col in self.output_columns:
            min_val, max_val = experiment_data[col].min(), experiment_data[col].max()
            print(f"  {col}: {min_val:.4f} - {max_val:.4f}")
        
        return experiment_data
    
    def _normalize_outputs(self, experiment_data):
        """Normalize output variables using specified method."""
        
        normalized_columns = []
        
        for col in self.output_columns:
            values = experiment_data[col].values
            
            # Ensure values is always an array (fix for single-element case)
            values = np.asarray(values)
            
            if self.normalization_method == 'log_zscore':
                # Log transform + z-score
                epsilon = 1e-6
                
                # Ensure proper numpy array handling (fixes dtype issues)
                values_array = np.asarray(values, dtype=np.float64)
                values_with_epsilon = values_array + epsilon
                
                log_values = np.log10(values_with_epsilon)
                
                scaler = StandardScaler()
                normalized = scaler.fit_transform(log_values.reshape(-1, 1)).flatten()
                
                log_col = f'log_{col}'
                norm_col = f'{col}_normalized'
                experiment_data[log_col] = log_values
                experiment_data[norm_col] = normalized
                normalized_columns.append(norm_col)
                
                print(f"  {col}: {values.min():.4f}-{values.max():.4f} → log → z-score: {normalized.min():.3f}-{normalized.max():.3f}")
                
            elif self.normalization_method == 'zscore':
                # Z-score normalization
                scaler = StandardScaler()
                normalized = scaler.fit_transform(values.reshape(-1, 1)).flatten()
                
                norm_col = f'{col}_normalized'
                experiment_data[norm_col] = normalized
                normalized_columns.append(norm_col)
                
                print(f"  {col}: {values.min():.4f}-{values.max():.4f} → z-score: {normalized.min():.3f}-{normalized.max():.3f}")
                
            elif self.normalization_method == 'minmax':
                # Min-max normalization
                normalized = (values - values.min()) / (values.max() - values.min())
                
                norm_col = f'{col}_normalized'
                experiment_data[norm_col] = normalized
                normalized_columns.append(norm_col)
                
                print(f"  {col}: {values.min():.4f}-{values.max():.4f} → minmax: {normalized.min():.3f}-{normalized.max():.3f}")
                
                # Create dummy scaler for minmax (for consistency)
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                scaler.fit(values.reshape(-1, 1))
            
            # Store scaler for potential future use
            self.scalers[col] = scaler
        
        # Store normalized column names
        self.normalized_columns = normalized_columns
        
        # Add log-transformed inputs if requested
        if self.log_transform_inputs:
            for col in self.input_columns:
                log_col = f'log_{col}'
                experiment_data[log_col] = np.log10(experiment_data[col])
        
        return experiment_data
    
    def _create_ghost_points(self, grid_data, grid_sizes, grid_coordinates):
        """Create ghost points for missing grid positions using neighbor interpolation."""
        
        complete_grid_data = grid_data.copy()
        ghost_count = 0
        
        # Iterate through all possible grid positions
        for grid_indices in np.ndindex(*grid_sizes):
            if grid_indices not in grid_data:
                # This is a missing position - create a ghost point
                neighbors = self._find_grid_neighbors(grid_indices, grid_sizes, grid_data)
                
                if len(neighbors) >= 2:  # Need at least 2 neighbors for interpolation
                    ghost_point = self._interpolate_ghost_point(grid_indices, neighbors, grid_coordinates)
                    ghost_point['is_ghost'] = True
                    complete_grid_data[grid_indices] = ghost_point
                    ghost_count += 1
        
        print(f"    Created {ghost_count} ghost points from neighbor interpolation")
        return complete_grid_data
    
    def _find_grid_neighbors(self, grid_indices, grid_sizes, grid_data):
        """Find real data neighbors within 1-2 grid steps for interpolation."""
        
        neighbors = []
        
        # Search in expanding radius (1 step, then 2 steps)
        for radius in [1, 2]:
            for dim in range(len(grid_sizes)):
                # Check negative direction
                if grid_indices[dim] >= radius:
                    neighbor_idx = list(grid_indices)
                    neighbor_idx[dim] -= radius
                    neighbor_idx = tuple(neighbor_idx)
                    if neighbor_idx in grid_data:
                        neighbors.append((neighbor_idx, grid_data[neighbor_idx]))
                
                # Check positive direction
                if grid_indices[dim] < grid_sizes[dim] - radius:
                    neighbor_idx = list(grid_indices)
                    neighbor_idx[dim] += radius
                    neighbor_idx = tuple(neighbor_idx)
                    if neighbor_idx in grid_data:
                        neighbors.append((neighbor_idx, grid_data[neighbor_idx]))
            
            # If we found enough neighbors at this radius, stop searching
            if len(neighbors) >= 2:
                break
        
        return neighbors
    
    def _interpolate_ghost_point(self, grid_indices, neighbors, grid_coordinates):
        """Interpolate values for a ghost point using inverse distance weighting."""
        
        # Calculate position of ghost point
        ghost_coords = {}
        for i, col in enumerate(self.input_columns):
            coord_list = grid_coordinates[col]
            ghost_coords[col] = coord_list[grid_indices[i]]
        
        # If log transform, calculate log coordinates too
        if self.log_transform_inputs:
            for col in self.input_columns:
                ghost_coords[f'log_{col}'] = np.log10(ghost_coords[col])
        
        # Interpolate output values using inverse distance weighting
        total_weight = 0
        weighted_outputs = {col: 0 for col in self.output_columns}
        weighted_normalized = {f'{col}_normalized': 0 for col in self.output_columns}
        
        for neighbor_idx, neighbor_data in neighbors:
            # Calculate distance in log space (if using log transform)
            if self.log_transform_inputs:
                ghost_pos = np.array([ghost_coords[f'log_{col}'] for col in self.input_columns])
                neighbor_pos = np.array([neighbor_data[f'log_{col}'] for col in self.input_columns])
            else:
                ghost_pos = np.array([ghost_coords[col] for col in self.input_columns])
                neighbor_pos = np.array([neighbor_data[col] for col in self.input_columns])
            
            distance = np.linalg.norm(ghost_pos - neighbor_pos)
            if distance < 1e-10:  # Very close, give high weight
                weight = 1e10
            else:
                weight = 1.0 / (distance ** 2)  # Inverse distance squared
            
            total_weight += weight
            
            # Weight the output values
            for col in self.output_columns:
                weighted_outputs[col] += weight * neighbor_data[col]
                weighted_normalized[f'{col}_normalized'] += weight * neighbor_data[f'{col}_normalized']
        
        # Create interpolated ghost point
        ghost_point = ghost_coords.copy()
        
        for col in self.output_columns:
            ghost_point[col] = weighted_outputs[col] / total_weight
            ghost_point[f'{col}_normalized'] = weighted_normalized[f'{col}_normalized'] / total_weight
        
        return ghost_point

    def _create_grid_structure(self, experiment_data):
        """Create n-dimensional grid structure for neighbor identification."""
        
        # Get unique coordinates for each input dimension
        grid_coordinates = {}
        grid_sizes = []
        
        for col in self.input_columns:
            unique_vals = sorted(experiment_data[col].unique())
            grid_coordinates[col] = unique_vals
            grid_sizes.append(len(unique_vals))
            print(f"  {col}: {len(unique_vals)} unique values")
        
        total_grid_points = np.prod(grid_sizes)
        print(f"  Grid structure: {' × '.join(map(str, grid_sizes))} = {total_grid_points} total positions")
        print(f"  Data coverage: {len(experiment_data)}/{total_grid_points} = {len(experiment_data)/total_grid_points:.1%}")
        
        # Create grid mapping from coordinates to data
        grid_data = {}
        
        for _, row in experiment_data.iterrows():
            # Create grid key from input coordinates
            grid_key = tuple(
                grid_coordinates[col].index(row[col]) 
                for col in self.input_columns
            )
            
            # Store all relevant data for this grid point
            grid_point = {}
            
            # Original input values
            for col in self.input_columns:
                grid_point[col] = row[col]
                if self.log_transform_inputs:
                    grid_point[f'log_{col}'] = row[f'log_{col}']
            
            # Original output values
            for col in self.output_columns:
                grid_point[col] = row[col]
                grid_point[f'{col}_normalized'] = row[f'{col}_normalized']
            
            grid_data[grid_key] = grid_point
        
        return grid_data, grid_coordinates
    
    def _calculate_edge_scores(self, grid_data, grid_coordinates):
        """Calculate vector difference scores for edges between neighboring points."""
        
        edges = []
        grid_sizes = [len(grid_coordinates[col]) for col in self.input_columns]
        
        print(f"  Calculating edges for {self.n_inputs}-dimensional grid...")
        print(f"  Original data coverage: {len(grid_data)}/{np.prod(grid_sizes)}")
        
        # STEP 1: Create ghost points for missing grid positions
        print(f"  Step 1: Creating ghost points for missing grid positions...")
        complete_grid_data = self._create_ghost_points(grid_data, grid_sizes, grid_coordinates)
        print(f"  Enhanced data coverage: {len(complete_grid_data)}/{np.prod(grid_sizes)} (includes ghost points)")
        
        # STEP 2A: Generate axis-aligned edges (existing behavior)
        print(f"  Step 2A: Creating axis-aligned edges...")
        axis_edges = 0
        
        for dim in range(self.n_inputs):
            dim_name = self.input_columns[dim]
            
            # Iterate through all possible positions
            for grid_indices in np.ndindex(*grid_sizes):
                # Skip if we're at the edge of this dimension
                if grid_indices[dim] >= grid_sizes[dim] - 1:
                    continue
                
                # Create neighboring position by moving one step in current dimension
                neighbor_indices = list(grid_indices)
                neighbor_indices[dim] += 1
                neighbor_indices = tuple(neighbor_indices)
                
                # Check both positions exist
                if grid_indices in complete_grid_data and neighbor_indices in complete_grid_data:
                    edge = self._create_edge(grid_indices, neighbor_indices, complete_grid_data, 
                                           [dim], f"axis_{dim_name}")
                    if edge:
                        edges.append(edge)
                        axis_edges += 1
        
        print(f"    Created {axis_edges} axis-aligned edges")
        
        # STEP 2B: Generate diagonal edges (NEW - fixes stair-step boundaries)
        print(f"  Step 2B: Creating diagonal edges...")
        diagonal_edges = 0
        
        # Add positive diagonal edges (+1, +1)
        for dim1, dim2 in combinations(range(self.n_inputs), 2):
            dim1_name = self.input_columns[dim1]
            dim2_name = self.input_columns[dim2]
            
            for grid_indices in np.ndindex(*grid_sizes):
                # Check if we can move +1 in both dimensions
                if (grid_indices[dim1] < grid_sizes[dim1] - 1 and 
                    grid_indices[dim2] < grid_sizes[dim2] - 1):
                    
                    neighbor_indices = list(grid_indices)
                    neighbor_indices[dim1] += 1
                    neighbor_indices[dim2] += 1
                    neighbor_indices = tuple(neighbor_indices)
                    
                    # Check both positions exist
                    if grid_indices in complete_grid_data and neighbor_indices in complete_grid_data:
                        edge = self._create_edge(grid_indices, neighbor_indices, complete_grid_data,
                                               [dim1, dim2], f"diag_{dim1_name}+{dim2_name}")
                        if edge:
                            edges.append(edge)
                            diagonal_edges += 1
        
        # Add negative diagonal edges (+1, -1) 
        for dim1, dim2 in combinations(range(self.n_inputs), 2):
            dim1_name = self.input_columns[dim1]
            dim2_name = self.input_columns[dim2]
            
            for grid_indices in np.ndindex(*grid_sizes):
                # Check if we can move +1 in dim1 and -1 in dim2
                if (grid_indices[dim1] < grid_sizes[dim1] - 1 and 
                    grid_indices[dim2] > 0):
                    
                    neighbor_indices = list(grid_indices)
                    neighbor_indices[dim1] += 1
                    neighbor_indices[dim2] -= 1
                    neighbor_indices = tuple(neighbor_indices)
                    
                    # Check both positions exist
                    if grid_indices in complete_grid_data and neighbor_indices in complete_grid_data:
                        edge = self._create_edge(grid_indices, neighbor_indices, complete_grid_data,
                                               [dim1, dim2], f"diag_{dim1_name}-{dim2_name}")
                        if edge:
                            edges.append(edge)
                            diagonal_edges += 1
        
        print(f"    Created {diagonal_edges} diagonal edges")
        
        edges_df = pd.DataFrame(edges)
        
        print(f"  Total edges: {len(edges_df)} ({axis_edges} axis + {diagonal_edges} diagonal)")
        if len(edges_df) > 0:
            print(f"  Weighted score range: {edges_df['score'].min():.4f} - {edges_df['score'].max():.4f}")
            print(f"  Base score range: {edges_df['base_score'].min():.4f} - {edges_df['base_score'].max():.4f}")
            print(f"  Length-weighted mean score: {edges_df['score'].mean():.4f}")
            print(f"  Edges with ghost points: {edges_df['has_ghost'].sum()}/{len(edges_df)} ({100*edges_df['has_ghost'].mean():.1f}%)")
            
            # Show distribution by dimension
            dim_counts = edges_df['dimension_name'].value_counts()
            for dim_name, count in dim_counts.items():
                print(f"    {dim_name}: {count} edges")
        
        return edges_df
    
    def _create_edge(self, pos1, pos2, grid_data, changed_dims, dimension_name):
        """Create edge between two grid positions with proper length calculation."""
        
        data1 = grid_data[pos1]
        data2 = grid_data[pos2]
        
        # Ghost point filtering: don't allow ghost-ghost edges
        is_ghost1 = data1.get('is_ghost', False)
        is_ghost2 = data2.get('is_ghost', False)
        
        if is_ghost1 and is_ghost2:
            return None  # Skip ghost-ghost edges
        
        # Calculate vector difference in normalized output space
        output_diffs = []
        normalized_diffs_dict = {}
        
        for col in self.output_columns:
            norm_col = f'{col}_normalized'
            diff = data2[norm_col] - data1[norm_col]
            output_diffs.append(diff)
            normalized_diffs_dict[f'{col}_diff'] = diff
        
        # Calculate base boundary score (Euclidean distance in normalized space)
        base_score = np.sqrt(sum(diff**2 for diff in output_diffs))
        
        # Calculate proper edge length for all changed dimensions (Euclidean)
        edge_deltas = []
        for dim in changed_dims:
            col = self.input_columns[dim]
            if self.log_transform_inputs:
                delta = data2[f'log_{col}'] - data1[f'log_{col}']
            else:
                delta = data2[col] - data1[col]
            edge_deltas.append(delta)
        
        # Euclidean edge length in appropriate space
        edge_length = np.sqrt(sum(delta**2 for delta in edge_deltas))
        
        # Apply length weighting (reduced bonus for diagonals to avoid over-favoring)
        if len(changed_dims) > 1:  # Diagonal edge
            length_weight = np.sqrt(edge_length) * 0.5  # Reduced bonus for diagonals
        else:  # Axis edge
            length_weight = np.sqrt(edge_length)
        
        weighted_score = base_score * (1.0 + 0.3 * length_weight)  # Reduced from 0.5 to 0.3
        
        # Stronger penalty for ghost points (increased from 10% to 40%)
        if is_ghost1 or is_ghost2:
            weighted_score *= 0.6  # 40% penalty for ghost interpolation
        
        # Calculate midpoint coordinates
        midpoint = {}
        for dim in range(self.n_inputs):
            col = self.input_columns[dim]
            if self.log_transform_inputs:
                # Geometric mean (proper log-space midpoint)
                midpoint[col] = np.sqrt(data1[col] * data2[col])
                midpoint[f'log_{col}'] = (data1[f'log_{col}'] + data2[f'log_{col}']) / 2
            else:
                # Arithmetic mean
                midpoint[col] = (data1[col] + data2[col]) / 2
        
        # Create edge record
        edge = {
            'pos1': pos1,
            'pos2': pos2,
            'dimension': changed_dims[0] if len(changed_dims) == 1 else -1,  # -1 for diagonals
            'dimension_name': dimension_name,
            'changed_dims': changed_dims,
            'score': weighted_score,
            'base_score': base_score,
            'edge_length': edge_length,
            'length_weight': length_weight,
            'has_ghost': is_ghost1 or is_ghost2,
            **midpoint,
            **normalized_diffs_dict
        }
        
        return edge
    
    def _select_boundary_points(self, edges_df, grid_coordinates, n_points, min_spacing_factor):
        """Select high-scoring edge midpoints with minimum spacing enforcement."""
        
        if len(edges_df) == 0:
            print("  Warning: No edges found!")
            return pd.DataFrame()
        
        # Sort edges by score (highest first)
        edges_sorted = edges_df.sort_values('score', ascending=False).reset_index(drop=True)
        
        # Calculate minimum spacing in log space (if applicable) or regular space
        min_spacings = []
        for col in self.input_columns:
            coords = grid_coordinates[col]
            if len(coords) > 1:
                if self.log_transform_inputs:
                    log_coords = [np.log10(c) for c in coords]
                    spacing = min(log_coords[i+1] - log_coords[i] for i in range(len(log_coords)-1))
                    min_spacings.append(spacing)
                else:
                    spacing = min(coords[i+1] - coords[i] for i in range(len(coords)-1))
                    min_spacings.append(spacing)
        
        if min_spacings:
            d_min = min_spacing_factor * min(min_spacings)
        else:
            d_min = 0.1  # Default fallback
        
        print(f"  Selecting up to {n_points} points with minimum spacing: {d_min:.3f}")
        
        selected_points = []
        
        for _, edge in edges_sorted.iterrows():
            # Create candidate position vector
            if self.log_transform_inputs:
                candidate_pos = np.array([edge[f'log_{col}'] for col in self.input_columns])
            else:
                candidate_pos = np.array([edge[col] for col in self.input_columns])
            
            # Check spacing against already selected points
            if len(selected_points) > 0:
                if self.log_transform_inputs:
                    selected_positions = np.array([[p[f'log_{col}'] for col in self.input_columns] 
                                                 for p in selected_points])
                else:
                    selected_positions = np.array([[p[col] for col in self.input_columns] 
                                                 for p in selected_points])
                
                distances = cdist([candidate_pos], selected_positions)[0]
                
                if np.min(distances) < d_min:
                    continue  # Skip - too close to existing point
            
            # Add point to selection
            point = {col: edge[col] for col in self.input_columns}
            if self.log_transform_inputs:
                for col in self.input_columns:
                    point[f'log_{col}'] = edge[f'log_{col}']
            
            point.update({
                'score': edge['score'],
                'dimension': edge['dimension'],
                'dimension_name': edge['dimension_name']
            })
            
            # Add output differences  
            for col in self.output_columns:
                if f'{col}_diff' in edge:
                    point[f'{col}_diff'] = edge[f'{col}_diff']
            
            selected_points.append(point)
            
            if len(selected_points) >= n_points:
                break
        
        selected_df = pd.DataFrame(selected_points)
        
        print(f"  Selected {len(selected_df)} points (requested {n_points})")
        if len(selected_df) > 0:
            print(f"  Score range: {selected_df['score'].min():.4f} - {selected_df['score'].max():.4f}")
        
        return selected_df
    
    def _create_visualization(self, experiment_data, edges_df, selected_df, output_dir):
        """Create visualization plots (implementation depends on dimensionality)."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.n_inputs <= 2 and self.n_outputs >= 1:
            # Can create meaningful 2D plots
            self._create_2d_visualization(experiment_data, edges_df, selected_df, output_dir)
        else:
            # Create summary plots for higher dimensions
            self._create_summary_visualization(edges_df, selected_df, output_dir)
    
    def _create_2d_visualization(self, experiment_data, edges_df, selected_df, output_dir):
        """Create 2D visualization for 1-2 input dimensions."""
        
        if self.n_inputs == 1:
            # 1D case - plot along single input dimension
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            input_col = self.input_columns[0]
            output_col = self.output_columns[0]
            
            # Plot 1: Data with selected points
            ax1.scatter(experiment_data[input_col], experiment_data[output_col],
                       c='blue', s=60, alpha=0.7, label='Existing data')
            ax1.scatter(selected_df[input_col], [0]*len(selected_df),  # Plot at y=0 for visibility
                       c='red', s=100, marker='^', label='Selected points')
            
            if self.log_transform_inputs:
                ax1.set_xscale('log')
            ax1.set_xlabel(input_col)
            ax1.set_ylabel(output_col)
            ax1.set_title('Vector Edge Refinement - 1D Case')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Score distribution
            ax2.hist(edges_df['score'], bins=30, alpha=0.7, color='gray')
            ax2.axvline(selected_df['score'].min(), color='red', linestyle='--', 
                       label=f'Selection threshold: {selected_df["score"].min():.3f}')
            ax2.set_xlabel('Edge Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Edge Score Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        elif self.n_inputs == 2:
            # 2D case - can create full visualization similar to original
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            
            x_col, y_col = self.input_columns[0], self.input_columns[1]
            
            # Plot 1: Original data with selected points
            if self.n_outputs > 0:
                color_col = self.output_columns[0]
                sc1 = ax1.scatter(experiment_data[x_col], experiment_data[y_col],
                                c=experiment_data[color_col], s=60, cmap='viridis', alpha=0.8,
                                edgecolors='black', linewidth=1, label='Existing data')
                cbar1 = plt.colorbar(sc1, ax=ax1)
                cbar1.set_label(color_col)
            
            ax1.scatter(selected_df[x_col], selected_df[y_col],
                       c='red', s=100, alpha=1.0, marker='^', edgecolors='darkred', 
                       linewidth=2, label=f'Selected points (n={len(selected_df)})')
            
            if self.log_transform_inputs:
                ax1.set_xscale('log')
                ax1.set_yscale('log')
            ax1.set_xlabel(x_col)
            ax1.set_ylabel(y_col)
            ax1.set_title('Vector Edge Refinement - Selected Points')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Edge scores
            sc2 = ax2.scatter(edges_df[x_col], edges_df[y_col],
                            c=edges_df['score'], s=30, cmap='Reds', alpha=0.7,
                            edgecolors='black', linewidth=0.5)
            ax2.scatter(selected_df[x_col], selected_df[y_col],
                       c='blue', s=80, alpha=1.0, marker='s', edgecolors='darkblue', 
                       linewidth=2, label='Selected edges')
            
            if self.log_transform_inputs:
                ax2.set_xscale('log')
                ax2.set_yscale('log')
            ax2.set_xlabel(x_col)
            ax2.set_ylabel(y_col)
            ax2.set_title('Edge Scores')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            cbar2 = plt.colorbar(sc2, ax=ax2)
            cbar2.set_label('Vector Change Score')
            
            # Plot 3: Score distribution
            ax3.hist(edges_df['score'], bins=50, alpha=0.7, color='gray', 
                    label=f'All edges (n={len(edges_df)})')
            if len(selected_df) > 0:
                ax3.axvline(selected_df['score'].min(), color='red', linestyle='--', 
                           linewidth=2, label=f'Selection threshold: {selected_df["score"].min():.3f}')
                ax3.axvline(selected_df['score'].mean(), color='blue', linestyle='-', 
                           linewidth=2, label=f'Selected avg: {selected_df["score"].mean():.3f}')
            
            ax3.set_xlabel('Vector Change Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Edge Score Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Dimension analysis
            dim_counts = selected_df['dimension_name'].value_counts()
            ax4.bar(dim_counts.index, dim_counts.values, alpha=0.7)
            ax4.set_xlabel('Edge Direction')
            ax4.set_ylabel('Number of Selected Edges')
            ax4.set_title('Selected Edges by Dimension')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (dim, count) in enumerate(dim_counts.items()):
                ax4.text(i, count + max(dim_counts.values)*0.01, str(count), 
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_file = f'{output_dir}/vector_edge_refinement.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Visualization saved: {output_file}")
        plt.show()
    
    def _create_summary_visualization(self, edges_df, selected_df, output_dir):
        """Create summary visualization for higher dimensional cases."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Score distribution
        ax1.hist(edges_df['score'], bins=50, alpha=0.7, color='gray', 
                label=f'All edges (n={len(edges_df)})')
        if len(selected_df) > 0:
            ax1.axvline(selected_df['score'].min(), color='red', linestyle='--', 
                       linewidth=2, label=f'Selection threshold: {selected_df["score"].min():.3f}')
            ax1.axvline(selected_df['score'].mean(), color='blue', linestyle='-', 
                       linewidth=2, label=f'Selected avg: {selected_df["score"].mean():.3f}')
        
        ax1.set_xlabel('Vector Change Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Edge Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Dimension analysis
        dim_counts = selected_df['dimension_name'].value_counts()
        ax2.bar(dim_counts.index, dim_counts.values, alpha=0.7)
        ax2.set_xlabel('Edge Direction')
        ax2.set_ylabel('Number of Selected Edges')
        ax2.set_title('Selected Edges by Dimension')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 3: Score vs rank
        selected_sorted = selected_df.sort_values('score', ascending=False)
        ax3.plot(range(1, len(selected_sorted)+1), selected_sorted['score'], 'bo-')
        ax3.set_xlabel('Rank')
        ax3.set_ylabel('Score')
        ax3.set_title('Selected Points Score vs Rank')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Output variable differences
        if self.n_outputs > 1:
            diff_cols = [f'{col}_diff' for col in self.output_columns if f'{col}_diff' in selected_df.columns]
            if diff_cols:
                ax4.boxplot([selected_df[col] for col in diff_cols], labels=self.output_columns)
                ax4.set_ylabel('Normalized Output Difference')
                ax4.set_title('Output Variable Changes')
                ax4.grid(True, alpha=0.3)
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        else:
            ax4.text(0.5, 0.5, 'Single output variable', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Output Analysis')
        
        plt.tight_layout()
        output_file = f'{output_dir}/vector_edge_refinement_summary.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Summary visualization saved: {output_file}")
        plt.show()
    
    def _format_recommendations(self, selected_df, output_dir=None):
        """Format recommendations for output."""
        
        if len(selected_df) == 0:
            print("  No recommendations to format")
            return pd.DataFrame()
        
        # Create recommendations dataframe
        recommendations = selected_df.copy()
        recommendations['rank'] = range(1, len(selected_df) + 1)
        
        # Select and rename columns
        columns_to_keep = ['rank'] + self.input_columns + ['score', 'dimension_name']
        
        # Add output differences if available
        for col in self.output_columns:
            diff_col = f'{col}_diff'
            if diff_col in recommendations.columns:
                columns_to_keep.append(diff_col)
        
        recommendations = recommendations[columns_to_keep].round(6)
        
        # Rename for clarity
        rename_dict = {
            'score': 'boundary_score',
            'dimension_name': 'edge_direction'
        }
        for col in self.output_columns:
            diff_col = f'{col}_diff'
            if diff_col in recommendations.columns:
                rename_dict[diff_col] = f'{col}_change'
        
        recommendations.rename(columns=rename_dict, inplace=True)
        
        # Save to file if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = f'{output_dir}/vector_edge_recommendations.csv'
            recommendations.to_csv(output_file, index=False)
            print(f"  Recommendations saved: {output_file}")
        
        print("\\n  Top 10 recommended sampling points:")
        print(recommendations.head(10).to_string(index=False))
        
        return recommendations
    
    def _print_summary(self, edges_df, selected_df):
        """Print analysis summary."""
        
        print("\\n" + "="*70)
        print("GENERALIZED VECTOR EDGE REFINEMENT RESULTS")  
        print("="*70)
        print(f"✓ Input dimensions: {self.n_inputs} ({', '.join(self.input_columns)})")
        print(f"✓ Output variables: {self.n_outputs} ({', '.join(self.output_columns)})")
        print(f"✓ Normalization: {self.normalization_method}")
        print(f"✓ Log transform inputs: {self.log_transform_inputs}")
        print()
        
        if len(edges_df) > 0:
            print(f"✓ Analyzed {len(edges_df)} edges between neighboring grid points")
            print(f"✓ Vector field scoring: Euclidean distance in normalized output space")
            if len(selected_df) > 0:
                print(f"✓ Selected {len(selected_df)} boundary points with spacing enforcement")
                print(f"✓ Score range: {selected_df['score'].min():.4f} - {selected_df['score'].max():.4f}")
                
                # Show distribution by dimension
                dim_counts = selected_df['dimension_name'].value_counts()
                print("\\n✓ Edge directions:")
                for dim_name, count in dim_counts.items():
                    print(f"   {count} edges along {dim_name}")
                
                print("\\n✓ Strategy: Sample at midpoints of strongest vector field boundaries")
                print("✓ Benefit: Automatically finds transitions in any combination of output variables")
            else:
                print("✗ No points selected (may need to adjust parameters)")
        else:
            print("✗ No edges found in data (check grid structure)")


# ============================================================================
# CONVENIENT WRAPPER FUNCTIONS FOR WORKFLOW INTEGRATION
# ============================================================================

def get_surfactant_boundary_recommendations(data_file_path, n_points=32, 
                                          output_dir=None, create_visualization=True):
    """
    Convenient function for surfactant screening boundary recommendations.
    
    Parameters:
    -----------
    data_file_path : str
        Path to CSV file with experimental results
    n_points : int
        Number of boundary points to recommend  
    output_dir : str, optional
        Directory for saving outputs. If None, uses data file directory.
    create_visualization : bool
        Whether to create visualization plots
        
    Returns:
    --------
    pd.DataFrame
        Recommended sampling points with scores and metadata
    """
    
    # Load data
    data_df = pd.read_csv(data_file_path)
    
    # Set output directory
    if output_dir is None:
        import os
        output_dir = os.path.join(os.path.dirname(data_file_path), 'vector_edge_refinement')
    
    # Initialize recommender for surfactant screening
    recommender = GeneralizedVectorEdgeRecommender(
        input_columns=['surf_A_conc_mm', 'surf_B_conc_mm'],
        output_columns=['turbidity_600', 'ratio'],
        log_transform_inputs=True,
        normalization_method='log_zscore'
    )
    
    # Get recommendations
    recommendations = recommender.get_recommendations(
        data_df, 
        n_points=n_points,
        output_dir=output_dir,
        create_visualization=create_visualization
    )
    
    return recommendations

def get_custom_boundary_recommendations(data_file_path, input_columns, output_columns,
                                      n_points=32, log_transform_inputs=True,
                                      normalization_method='log_zscore', output_dir=None,
                                      create_visualization=True):
    """
    Convenient function for custom boundary recommendations with arbitrary inputs/outputs.
    
    Parameters:
    -----------
    data_file_path : str
        Path to CSV file with experimental results
    input_columns : list of str
        Names of input variables (X, Y, Z, ...)
    output_columns : list of str  
        Names of output variables (A, B, C, ...)
    n_points : int
        Number of boundary points to recommend
    log_transform_inputs : bool
        Whether to work in log space for inputs
    normalization_method : str
        'log_zscore', 'zscore', or 'minmax'
    output_dir : str, optional
        Directory for saving outputs
    create_visualization : bool
        Whether to create visualization plots
        
    Returns:
    --------
    pd.DataFrame
        Recommended sampling points with scores and metadata
    """
    
    # Load data
    data_df = pd.read_csv(data_file_path)
    
    # Set output directory
    if output_dir is None:
        import os
        output_dir = os.path.join(os.path.dirname(data_file_path), 'vector_edge_refinement')
    
    # Initialize recommender
    recommender = GeneralizedVectorEdgeRecommender(
        input_columns=input_columns,
        output_columns=output_columns,
        log_transform_inputs=log_transform_inputs,
        normalization_method=normalization_method
    )
    
    # Get recommendations
    recommendations = recommender.get_recommendations(
        data_df,
        n_points=n_points,
        output_dir=output_dir,
        create_visualization=create_visualization
    )
    
    return recommendations

def recommendations_to_well_recipes(recommendations, surfactant_library, 
                                  stock_volume_ul=200.0, final_volume_ul=200.0):
    """
    Convert boundary recommendations to well_recipes_df format for workflow execution.
    
    Parameters:
    -----------
    recommendations : pd.DataFrame
        Output from vector edge recommender
    surfactant_library : dict
        Surfactant library with stock concentrations
    stock_volume_ul : float
        Stock volume per well (uL)
    final_volume_ul : float  
        Final volume per well (uL)
        
    Returns:
    --------
    pd.DataFrame
        Well recipes in workflow format
    """
    
    well_recipes = []
    
    for i, row in recommendations.iterrows():
        recipe = {
            'condition_id': f'boundary_{i+1}',
            'surf_A_conc_mm': row['surf_A_conc_mm'],
            'surf_B_conc_mm': row['surf_B_conc_mm'],
            'stock_volume_ul': stock_volume_ul,
            'final_volume_ul': final_volume_ul,
            'recommendation_source': 'vector_edge_refinement',
            'boundary_score': row['boundary_score'],
            'edge_direction': row['edge_direction']
        }
        well_recipes.append(recipe)
    
    well_recipes_df = pd.DataFrame(well_recipes)
    
    return well_recipes_df

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of how to use the generalized recommender."""
    
    # Example 1: 2D surfactant screening (like original)
    recommender = GeneralizedVectorEdgeRecommender(
        input_columns=['surf_A_conc_mm', 'surf_B_conc_mm'],
        output_columns=['turbidity_600', 'ratio'],
        log_transform_inputs=True,
        normalization_method='log_zscore'
    )
    
    # Example 2: 3D optimization with different outputs
    # recommender = GeneralizedVectorEdgeRecommender(
    #     input_columns=['temperature', 'pH', 'concentration'],
    #     output_columns=['yield', 'purity', 'reaction_time'],
    #     log_transform_inputs=False,
    #     normalization_method='zscore'
    # )
    
    # Example 3: Single output optimization
    # recommender = GeneralizedVectorEdgeRecommender(
    #     input_columns=['x', 'y', 'z'],
    #     output_columns=['performance'],
    #     log_transform_inputs=False,
    #     normalization_method='minmax'
    # )
    
    print("Example recommender initialized. Use get_recommendations(data_df) to run.")
    
    # Quick usage examples:
    print("\nQuick usage for surfactant screening:")
    print("recommendations = get_surfactant_boundary_recommendations('data.csv', n_points=12)")
    print("well_recipes = recommendations_to_well_recipes(recommendations, surfactant_library)")
    
    print("\nCustom usage:")
    print("recommendations = get_custom_boundary_recommendations(")
    print("    'data.csv',")
    print("    input_columns=['x', 'y'],")
    print("    output_columns=['yield', 'purity'],")
    print("    n_points=20")
    print(")")

if __name__ == "__main__":
    example_usage()
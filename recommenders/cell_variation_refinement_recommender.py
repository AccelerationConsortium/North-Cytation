"""
Cell-Variation Refinement Recommender
=====================================

A new recommender that uses cell-based boundary detection to find high-variation regions
on a discrete grid without interpolation or ghost points.

Algorithm:
1. Work in log-space for concentration inputs
2. Normalize outputs to comparable scales
3. Build grid coordinate sets from discrete measured points
4. Enumerate cells (hypercubes) and score by corner variation
5. Propose sampling at the centers of highest-scoring cells
6. Apply minimum spacing constraints

Usage:
    recommender = CellVariationRefinementRecommender(
        input_columns=['surf_A_conc_mm', 'surf_B_conc_mm'],  # X, Y
        output_columns=['ratio'],                             # Focus on ratio only 
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
from scipy.spatial.distance import cdist
import os
from itertools import product, combinations
import warnings

class CellVariationRefinementRecommender:
    """
    Cell-variation refinement for n-dimensional input/output spaces.
    
    Finds grid cells (hypercubes) whose corner outputs vary the most, indicating
    boundaries/transitions. Recommends sampling at the centers of those cells.
    """
    
    def __init__(self, input_columns, output_columns, 
                 log_transform_inputs=True, normalization_method='log_zscore'):
        """
        Initialize the cell-variation recommender.
        
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
        
        print(f"Initialized CellVariationRefinementRecommender:")
        print(f"  Input variables ({self.n_inputs}): {input_columns}")
        print(f"  Output variables ({self.n_outputs}): {output_columns}")
        print(f"  Log transform inputs: {log_transform_inputs}")
        print(f"  Normalization method: {normalization_method}")
    
    def get_recommendations(self, data_df, n_points=12, min_spacing_factor=0.5, 
                          output_dir=None, create_visualization=True):
        """
        Get cell-variation refinement recommendations from experimental data.
        
        Parameters:
        -----------
        data_df : pd.DataFrame
            Experimental data with input_columns and output_columns
        n_points : int
            Number of cell centers to recommend
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
        print("CELL-VARIATION REFINEMENT")
        print("="*70)
        
        # Step 1: Prepare and validate data
        print("\\n1. Preparing experimental data...")
        experiment_data = self._prepare_data(data_df)
        
        # Step 2: Normalize outputs
        print("\\n2. Normalizing output variables...")
        experiment_data = self._normalize_outputs(experiment_data)
        
        # Step 3: Build grid structure and data lookup
        print("\\n3. Building grid coordinate structure...")
        grid_coordinates, data_lookup = self._build_grid_structure(experiment_data)
        
        # Step 4: Enumerate and score cells
        print("\\n4. Enumerating cells and calculating variation scores...")
        cell_scores = self._calculate_cell_scores(grid_coordinates, data_lookup)
        
        # Step 5: Select best cell centers with spacing
        print("\\n5. Selecting cell centers with minimum spacing...")
        selected_centers = self._select_cell_centers(cell_scores, grid_coordinates, 
                                                   n_points, min_spacing_factor)
        
        # Step 6: Format recommendations
        print("\\n6. Formatting recommendations...")
        recommendations = self._format_recommendations(selected_centers, output_dir)
        
        self._print_summary(cell_scores, selected_centers)
        
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
    
    def _normalize_outputs(self, experiment_data):
        """Normalize output variables using specified method."""
        
        normalized_columns = []
        
        for col in self.output_columns:
            values = experiment_data[col].values
            
            # Ensure values is always an array
            values = np.asarray(values, dtype=np.float64)
            
            if self.normalization_method == 'log_zscore':
                # Log transform + z-score
                epsilon = 1e-6
                values_with_epsilon = values + epsilon
                log_values = np.log10(values_with_epsilon)
                
                scaler = StandardScaler()
                normalized = scaler.fit_transform(log_values.reshape(-1, 1)).flatten()
                
                norm_col = f'{col}_normalized'
                experiment_data[norm_col] = normalized
                normalized_columns.append(norm_col)
                
                # Store scaler for potential future use
                self.scalers[col] = scaler
                
                print(f"  {col}: {values.min():.4f}-{values.max():.4f} → log → z-score: {normalized.min():.3f}-{normalized.max():.3f}")
                
            elif self.normalization_method == 'zscore':
                scaler = StandardScaler() 
                normalized = scaler.fit_transform(values.reshape(-1, 1)).flatten()
                
                norm_col = f'{col}_normalized'
                experiment_data[norm_col] = normalized
                normalized_columns.append(norm_col)
                self.scalers[col] = scaler
                
                print(f"  {col}: {values.min():.4f}-{values.max():.4f} → z-score: {normalized.min():.3f}-{normalized.max():.3f}")
                
            elif self.normalization_method == 'minmax':
                normalized = (values - values.min()) / (values.max() - values.min())
                
                norm_col = f'{col}_normalized'
                experiment_data[norm_col] = normalized
                normalized_columns.append(norm_col)
                
                print(f"  {col}: {values.min():.4f}-{values.max():.4f} → minmax: {normalized.min():.3f}-{normalized.max():.3f}")
        
        self.normalized_output_columns = normalized_columns
        return experiment_data
    
    def _build_grid_structure(self, experiment_data):
        """
        Build grid coordinate sets and create data lookup dictionary.
        
        For irregular grids (after adding cell centers), this identifies the base grid
        structure from points that lie on regular grid positions.
        
        Returns:
        --------
        grid_coordinates : dict
            For each input dimension, sorted unique values from the base grid
        data_lookup : dict
            Maps grid index tuple to normalized output vector (includes all points)
        """
        
        # Step 1: Identify base grid structure from regular grid points
        print("  Identifying base grid structure...")
        
        # For a regular grid, we expect logarithmic spacing to be approximately constant
        grid_coordinates = {}
        
        for i, col in enumerate(self.input_columns):
            if self.log_transform_inputs:
                log_values = np.log10(experiment_data[col].values)
                unique_log_vals = np.unique(log_values)
                
                # Find the base grid by identifying the most common spacing
                if len(unique_log_vals) >= 3:
                    spacings = np.diff(np.sort(unique_log_vals))
                    # Find the mode spacing (most common)
                    spacing_counts = {}
                    tolerance = 1e-6
                    for spacing in spacings:
                        found_match = False
                        for existing_spacing in spacing_counts:
                            if abs(spacing - existing_spacing) < tolerance:
                                spacing_counts[existing_spacing] += 1
                                found_match = True
                                break
                        if not found_match:
                            spacing_counts[spacing] = 1
                    
                    # Use the most common spacing to identify base grid
                    base_spacing = max(spacing_counts.keys(), key=lambda x: spacing_counts[x])
                    
                    # Find base grid points that align with this spacing
                    base_log_values = []
                    min_log = min(unique_log_vals)
                    current_log = min_log
                    
                    for log_val in sorted(unique_log_vals):
                        # Check if this value is close to a base grid position
                        grid_position = round((log_val - min_log) / base_spacing)
                        expected_log = min_log + grid_position * base_spacing
                        if abs(log_val - expected_log) < tolerance:
                            base_log_values.append(log_val)
                
                else:
                    # Too few points, use all unique values
                    base_log_values = unique_log_vals.tolist()
                
                grid_coordinates[col] = {
                    'log_values': sorted(base_log_values),
                    'original_values': [10**log_val for log_val in sorted(base_log_values)]
                }
                
                print(f"  {col}: {len(base_log_values)} base grid values from {min(base_log_values):.3f} to {max(base_log_values):.3f}")
                print(f"         All unique values: {len(unique_log_vals)} (includes cell centers)")
            
            else:
                # Similar logic for original space
                unique_vals = np.unique(experiment_data[col].values)
                grid_coordinates[col] = {'values': sorted(unique_vals)}
                print(f"  {col}: {len(unique_vals)} unique values from {unique_vals[0]:.3e} to {unique_vals[-1]:.3e}")
        
        # Step 2: Build data lookup for ALL measured points (not just base grid)
        print("  Building data lookup for ALL measured points...")
        data_lookup = {}
        
        for idx, row in experiment_data.iterrows():
            # Find nearest grid indices for ALL points (including cell centers)
            if self.log_transform_inputs:
                indices = []
                for col in self.input_columns:
                    log_val = np.log10(row[col])
                    base_log_values = grid_coordinates[col]['log_values']
                    
                    # Find closest base grid index
                    distances = np.abs(np.array(base_log_values) - log_val)
                    closest_idx = np.argmin(distances)
                    
                    # If it's exactly on the grid, use that index
                    if distances[closest_idx] < 1e-10:
                        indices.append(closest_idx)
                    else:
                        # For cell centers, we need to assign a fractional index
                        # Map to the cell it belongs to
                        if closest_idx > 0 and log_val < base_log_values[closest_idx]:
                            # It's between closest_idx-1 and closest_idx
                            cell_lower = closest_idx - 1
                        else:
                            # It's between closest_idx and closest_idx+1
                            cell_lower = closest_idx
                        
                        # Use a tuple (cell_lower, fractional_position) as key
                        if cell_lower < len(base_log_values) - 1:
                            fraction = (log_val - base_log_values[cell_lower]) / (base_log_values[cell_lower + 1] - base_log_values[cell_lower])
                            indices.append((cell_lower, fraction))
                        else:
                            indices.append(closest_idx)
            else:
                # Similar for original space
                indices = []
                for col in self.input_columns:
                    val = row[col]
                    values = grid_coordinates[col]['values']
                    distances = np.abs(np.array(values) - val)
                    closest_idx = np.argmin(distances)
                    indices.append(closest_idx)
            
            # Store with actual coordinates as key for cell center points
            coord_key = tuple([row[col] for col in self.input_columns])
            output_vector = np.array([row[col] for col in self.normalized_output_columns])
            data_lookup[coord_key] = output_vector
        
        # Also create a grid index lookup for base grid points only
        self.grid_index_lookup = {}
        for idx, row in experiment_data.iterrows():
            if self.log_transform_inputs:
                indices = []
                is_base_grid_point = True
                
                for col in self.input_columns:
                    log_val = np.log10(row[col])
                    base_log_values = grid_coordinates[col]['log_values']
                    
                    # Check if this is exactly on the base grid
                    distances = np.abs(np.array(base_log_values) - log_val)
                    closest_idx = np.argmin(distances)
                    
                    if distances[closest_idx] < 1e-10:
                        indices.append(closest_idx)
                    else:
                        is_base_grid_point = False
                        break
                
                if is_base_grid_point:
                    idx_tuple = tuple(indices)
                    coord_key = tuple([row[col] for col in self.input_columns])
                    self.grid_index_lookup[idx_tuple] = coord_key
        
        print(f"  Built data lookup with {len(data_lookup)} total measured points")
        print(f"  Base grid points: {len(self.grid_index_lookup)} on regular grid positions")
        
        return grid_coordinates, data_lookup
    
    def _calculate_cell_scores(self, grid_coordinates, data_lookup):
        """
        Enumerate cells based on base grid structure and calculate variation scores
        using ALL measured points (including cell centers) within each cell.
        
        A cell is defined by its lower corner on the base grid, but we include
        any measured points that fall within the cell boundaries for scoring.
        """
        
        # Use base grid dimensions
        grid_dims = []
        for col in self.input_columns:
            if self.log_transform_inputs:
                grid_dims.append(len(grid_coordinates[col]['log_values']))
            else:
                grid_dims.append(len(grid_coordinates[col]['values']))
        
        print(f"  Base grid dimensions: {grid_dims}")
        
        cell_scores = []
        
        # Enumerate all possible cells in the base grid
        total_cells = np.prod([dim - 1 for dim in grid_dims])
        print(f"  Evaluating up to {total_cells} possible base grid cells...")
        
        evaluated_cells = 0
        skipped_incomplete = 0
        
        for lower_corner in product(*[range(dim - 1) for dim in grid_dims]):
            # Define cell boundaries in log space (or original space)
            cell_bounds = []
            for i, col in enumerate(self.input_columns):
                lower_idx = lower_corner[i]
                upper_idx = lower_corner[i] + 1
                
                if self.log_transform_inputs:
                    log_values = grid_coordinates[col]['log_values']
                    lower_bound = log_values[lower_idx]
                    upper_bound = log_values[upper_idx]
                    cell_bounds.append((lower_bound, upper_bound))
                else:
                    values = grid_coordinates[col]['values']
                    lower_bound = values[lower_idx]
                    upper_bound = values[upper_idx]
                    cell_bounds.append((lower_bound, upper_bound))
            
            # Find ALL measured points that fall within this cell
            points_in_cell = []
            
            for coord_key, output_vector in data_lookup.items():
                point_coords = coord_key  # These are actual concentration values
                
                # Convert to log space if needed for bounds checking
                if self.log_transform_inputs:
                    log_coords = [np.log10(coord) for coord in point_coords]
                    coords_to_check = log_coords
                else:
                    coords_to_check = point_coords
                
                # Check if this point falls within the cell bounds
                point_in_cell = True
                for i, coord in enumerate(coords_to_check):
                    lower_bound, upper_bound = cell_bounds[i]
                    if not (lower_bound <= coord <= upper_bound):
                        point_in_cell = False
                        break
                
                if point_in_cell:
                    points_in_cell.append(output_vector)
            
            # Need at least 2 points to calculate variation
            if len(points_in_cell) < 2:
                skipped_incomplete += 1
                if skipped_incomplete <= 3:
                    print(f"    Cell {lower_corner}: Only {len(points_in_cell)} points in cell")
                continue
            
            # Calculate cell variation score
            if len(points_in_cell) >= 2:
                # Convert to numpy array for distance calculations
                points_matrix = np.array(points_in_cell)
                
                # Calculate all pairwise distances
                distances = []
                for i in range(len(points_in_cell)):
                    for j in range(i + 1, len(points_in_cell)):
                        dist = np.linalg.norm(points_matrix[i] - points_matrix[j])
                        distances.append(dist)
                
                # Score options:
                # A) Max distance (most sensitive to boundaries)
                max_distance = max(distances) if distances else 0.0
                
                # B) Second-largest distance (more robust to noise)
                second_largest = sorted(distances)[-2] if len(distances) >= 2 else max_distance
                
                # Use max distance as primary score
                cell_score = max_distance
                
                cell_scores.append({
                    'lower_corner': lower_corner,
                    'cell_score': cell_score,
                    'max_distance': max_distance,
                    'second_largest_distance': second_largest,
                    'n_points_in_cell': len(points_in_cell),
                    'cell_bounds': cell_bounds
                })
                
                evaluated_cells += 1
                
                if evaluated_cells <= 5:  # Debug first few cells
                    print(f"    Cell {lower_corner}: {len(points_in_cell)} points, score={cell_score:.4f}")
        
        print(f"  Evaluated {evaluated_cells} cells with sufficient data, skipped {skipped_incomplete} sparse cells")
        
        if evaluated_cells == 0:
            raise RuntimeError("No cells with sufficient data found! Need at least 2 points per cell.")
        
        # Sort by score descending
        cell_scores = sorted(cell_scores, key=lambda x: x['cell_score'], reverse=True)
        
        print(f"  Cell scores range: {cell_scores[-1]['cell_score']:.4f} to {cell_scores[0]['cell_score']:.4f}")
        
        return cell_scores
    
    def _select_cell_centers(self, cell_scores, grid_coordinates, n_points, min_spacing_factor):
        """
        Select top cell centers with minimum spacing constraint.
        """
        
        if len(cell_scores) == 0:
            raise RuntimeError("No cell scores available for selection")
        
        # Calculate typical grid step for spacing constraint
        typical_steps = []
        for col in self.input_columns:
            if self.log_transform_inputs:
                log_values = grid_coordinates[col]['log_values']
                steps = [log_values[i+1] - log_values[i] for i in range(len(log_values) - 1)]
            else:
                values = grid_coordinates[col]['values']
                steps = [np.log10(values[i+1]) - np.log10(values[i]) for i in range(len(values) - 1)]
            typical_steps.extend(steps)
        
        min_step = min(typical_steps) if typical_steps else 0.1
        d_min = min_spacing_factor * min_step
        
        print(f"  Minimum spacing constraint: {d_min:.4f} (factor={min_spacing_factor}, min_step={min_step:.4f})")
        
        selected_centers = []
        
        # Greedily select centers with spacing constraint
        for cell_data in cell_scores:
            if len(selected_centers) >= n_points:
                break
                
            # Calculate cell center in log space (or original space if not log transformed)
            lower_corner = cell_data['lower_corner']
            center = self._calculate_cell_center(lower_corner, grid_coordinates)
            
            # Check spacing with already selected centers
            too_close = False
            for existing_center in selected_centers:
                existing_coords = existing_center['center_log'] if self.log_transform_inputs else existing_center['center_original']
                current_coords = center['log'] if self.log_transform_inputs else center['original']
                
                distance = np.linalg.norm(np.array(current_coords) - np.array(existing_coords))
                if distance < d_min:
                    too_close = True
                    break
            
            if not too_close:
                selected_centers.append({
                    'cell_score': cell_data['cell_score'],
                    'lower_corner': lower_corner,
                    'center_original': center['original'],
                    'center_log': center['log'] if self.log_transform_inputs else None,
                    'max_distance': cell_data['max_distance'],
                    'second_largest_distance': cell_data['second_largest_distance']
                })
        
        print(f"  Selected {len(selected_centers)} cell centers (requested {n_points})")
        
        return selected_centers
    
    def _calculate_cell_center(self, lower_corner, grid_coordinates):
        """
        Calculate the center of a cell given its lower corner.
        
        Returns both original and log coordinates.
        """
        
        center_original = []
        center_log = []
        
        for i, col in enumerate(self.input_columns):
            lower_idx = lower_corner[i]
            upper_idx = lower_corner[i] + 1
            
            if self.log_transform_inputs:
                log_values = grid_coordinates[col]['log_values']
                original_values = grid_coordinates[col]['original_values']
                
                # Center in log space
                log_center = (log_values[lower_idx] + log_values[upper_idx]) / 2.0
                
                # Convert back to original space
                original_center = 10 ** log_center
                
                center_log.append(log_center)
                center_original.append(original_center)
            else:
                values = grid_coordinates[col]['values']
                # Geometric mean for concentrations (equivalent to arithmetic mean in log space)
                original_center = np.sqrt(values[lower_idx] * values[upper_idx])
                center_original.append(original_center)
        
        return {
            'original': center_original,
            'log': center_log if self.log_transform_inputs else None
        }
    
    def _format_recommendations(self, selected_centers, output_dir):
        """Format selected centers as recommendations dataframe."""
        
        if len(selected_centers) == 0:
            # Return empty dataframe with correct columns
            columns = self.input_columns + ['cell_score', 'max_distance', 'second_largest_distance']
            return pd.DataFrame(columns=columns)
        
        recommendations_data = []
        
        for i, center_data in enumerate(selected_centers):
            rec_dict = {}
            
            # Add input coordinates
            for j, col in enumerate(self.input_columns):
                rec_dict[col] = center_data['center_original'][j]
            
            # Add scores
            rec_dict['cell_score'] = center_data['cell_score']
            rec_dict['max_distance'] = center_data['max_distance']
            rec_dict['second_largest_distance'] = center_data['second_largest_distance']
            rec_dict['rank'] = i + 1
            
            recommendations_data.append(rec_dict)
        
        recommendations_df = pd.DataFrame(recommendations_data)
        
        # Save to file if output directory specified
        if output_dir and len(recommendations_df) > 0:
            os.makedirs(output_dir, exist_ok=True)
            recommendations_file = os.path.join(output_dir, 'cell_variation_recommendations.csv')
            recommendations_df.to_csv(recommendations_file, index=False)
            print(f"  Saved recommendations to: {recommendations_file}")
        
        return recommendations_df
    
    def _print_summary(self, cell_scores, selected_centers):
        """Print summary of cell analysis results."""
        
        print("\\n" + "="*70)
        print("CELL-VARIATION REFINEMENT SUMMARY")
        print("="*70)
        
        if len(cell_scores) > 0:
            print(f"Total cells evaluated: {len(cell_scores)}")
            print(f"Cell score range: {cell_scores[-1]['cell_score']:.4f} to {cell_scores[0]['cell_score']:.4f}")
            
            print(f"\\nTop 5 cells by variation score:")
            for i, cell_data in enumerate(cell_scores[:5]):
            n_points = cell_data.get('n_points_in_cell', cell_data.get('n_corners', 0))
            print(f"  {i+1}. Score: {cell_data['cell_score']:.4f}, Lower corner: {cell_data['lower_corner']}, Points: {n_points}")
            print(f"\\nSelected {len(selected_centers)} cell centers:")
            for i, center_data in enumerate(selected_centers):
                coords_str = ", ".join([f"{self.input_columns[j]}={val:.3e}" 
                                      for j, val in enumerate(center_data['center_original'])])
                print(f"  {i+1}. {coords_str} (score: {center_data['cell_score']:.4f})")
        else:
            print("\\nNo cell centers selected!")
        
        print("="*70)
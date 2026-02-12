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
            
            if self.normalization_method == 'log_zscore':
                # Log transform + z-score
                epsilon = 1e-6
                log_values = np.log10(values + epsilon)
                
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
        
        # Generate edges along each dimension
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
                
                # Check if both positions have data
                if grid_indices in grid_data and neighbor_indices in grid_data:
                    data1 = grid_data[grid_indices]
                    data2 = grid_data[neighbor_indices]
                    
                    # Calculate vector difference in normalized output space
                    output_diffs = []
                    normalized_diffs_dict = {}
                    
                    for col in self.output_columns:
                        norm_col = f'{col}_normalized'
                        diff = data2[norm_col] - data1[norm_col]
                        output_diffs.append(diff)
                        normalized_diffs_dict[f'{col}_diff'] = diff
                    
                    # Calculate Euclidean distance in normalized space
                    score = np.sqrt(sum(diff**2 for diff in output_diffs))
                    
                    # Calculate midpoint coordinates
                    midpoint = {}
                    for col in self.input_columns:
                        midpoint[col] = (data1[col] + data2[col]) / 2
                        if self.log_transform_inputs:
                            midpoint[f'log_{col}'] = (data1[f'log_{col}'] + data2[f'log_{col}']) / 2
                    
                    # Create edge record
                    edge = {
                        'pos1': grid_indices,
                        'pos2': neighbor_indices,
                        'dimension': dim,
                        'dimension_name': dim_name,
                        'score': score,
                        **midpoint,
                        **normalized_diffs_dict
                    }
                    
                    edges.append(edge)
        
        edges_df = pd.DataFrame(edges)
        
        print(f"  Calculated {len(edges_df)} edge scores")
        if len(edges_df) > 0:
            print(f"  Score range: {edges_df['score'].min():.4f} - {edges_df['score'].max():.4f}")
            print(f"  Mean score: {edges_df['score'].mean():.4f}")
            
            # Show distribution by dimension
            for dim, dim_name in enumerate(self.input_columns):
                dim_edges = edges_df[edges_df['dimension'] == dim]
                if len(dim_edges) > 0:
                    print(f"    {dim_name} direction: {len(dim_edges)} edges, avg score: {dim_edges['score'].mean():.4f}")
        
        return edges_df
    
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
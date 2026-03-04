"""
Bayesian Active Learning for Transition Region Discovery
======================================================

A Bayesian optimization recommender that uses BoTorch/GPyTorch to discover
transition regions in 2-5D surfactant concentration spaces with dual outputs.

Uses a "local contrast" acquisition function that detects sharp transitions
by evaluating directional gradients plus uncertainty-based exploration.

Key Features:
- Handles 2-5D input spaces (surfactant concentrations)
- Dual scalar outputs (e.g., ratio, turbidity)
- Log-transform for concentration data
- Independent GP per output (ModelListGP)
- Sequential greedy batch selection with diversity enforcement
- Sobol initialization for new experiments

Algorithm:
1. Normalize inputs to [0,1]^d (with optional log transform)
2. Standardize outputs independently 
3. Fit separate GP per output using SingleTaskGP
4. Local contrast acquisition: score(x) = mean over K random directions of
   sum_k |mu_k(x + delta*u) - mu_k(x)| + lambda * sum_k sigma_k(x)
5. Sequential batch selection with local penalization for diversity

Usage:
    recommender = BayesianTransitionRecommender(
        input_columns=['surf_A_conc_mm', 'surf_B_conc_mm'],
        output_columns=['ratio', 'turbidity_600nm'], 
        log_transform_inputs=True,
        delta=0.05,
        K=10,
        lam=0.3
    )
    
    recommendations = recommender.get_recommendations(
        data_df,
        n_points=14,
        min_distance=0.1
    )
"""

import pandas as pd
import numpy as np
import torch
import gpytorch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize, standardize
from botorch.acquisition.optimizer import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Configuration Section
DEFAULT_CONFIG = {
    'delta': 0.05,           # Step size for local contrast evaluation
    'K': 10,                 # Number of random directions for local contrast
    'lam': 0.3,             # Exploration bonus weight (uncertainty)  
    'q': 14,                # Default batch size
    'min_distance': 0.1,    # Minimum distance between points (in normalized space)
    'device': 'cpu',        # Device for PyTorch operations
    'dtype': torch.double   # Data type for numerical stability
}

class BayesianTransitionRecommender:
    """
    Bayesian optimization recommender for discovering transition regions 
    in multi-dimensional surfactant concentration spaces.
    """
    
    def __init__(self, input_columns: List[str], output_columns: List[str],
                 log_transform_inputs: bool = True, 
                 delta: float = 0.05, K: int = 10, lam: float = 0.3,
                 device: str = 'cpu', dtype = torch.double):
        """
        Initialize Bayesian transition recommender.
        
        Parameters:
        -----------
        input_columns : list of str
            Names of input variables (2-5D surfactant concentrations)
        output_columns : list of str
            Names of output variables (exactly 2 outputs)
        log_transform_inputs : bool
            Whether to apply log10 transform to input concentrations
        delta : float
            Step size for local contrast evaluation
        K : int  
            Number of random directions for local contrast sampling
        lam : float
            Exploration bonus weight (λ in acquisition function)
        device : str
            PyTorch device ('cpu' or 'cuda')
        dtype : torch.dtype
            Data type for numerical operations
        """
        
        if not (2 <= len(input_columns) <= 5):
            raise ValueError(f"Input dimensions must be 2-5, got {len(input_columns)}")
        
        if len(output_columns) != 2:
            raise ValueError(f"Must have exactly 2 outputs, got {len(output_columns)}")
        
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.log_transform_inputs = log_transform_inputs
        self.delta = delta
        self.K = K
        self.lam = lam
        self.device = torch.device(device)
        self.dtype = dtype
        
        self.n_inputs = len(input_columns)
        self.n_outputs = len(output_columns)
        
        # Storage for fitted scalers and bounds
        self.input_bounds = None  # Original space bounds
        self.output_scalers = {}
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        print(f"Initialized BayesianTransitionRecommender:")
        print(f"  Input variables ({self.n_inputs}D): {input_columns}")
        print(f"  Output variables (2D): {output_columns}")
        print(f"  Log transform inputs: {log_transform_inputs}")
        print(f"  Acquisition params: delta={delta:.3f}, K={K}, lambda={lam:.3f}")
        print(f"  Device: {self.device}, dtype: {dtype}")
    
    def get_recommendations(self, data_df: pd.DataFrame, n_points: int = 14,
                          min_distance: float = 0.1, use_sobol_init: bool = True,
                          n_sobol: int = None) -> pd.DataFrame:
        """
        Get Bayesian optimization recommendations for next experiments.
        
        Parameters:
        -----------
        data_df : pd.DataFrame
            Experimental data with input_columns and output_columns
        n_points : int
            Number of points to recommend
        min_distance : float
            Minimum distance between recommended points (normalized space)
        use_sobol_init : bool
            Whether to include Sobol sequence points if data is sparse
        n_sobol : int, optional
            Number of Sobol points to add (default: min(5, n_points//2))
            
        Returns:
        --------
        pd.DataFrame
            Recommended experiments with acquisition scores
        """
        
        print("\n" + "="*70)
        print("BAYESIAN TRANSITION REGION DISCOVERY")
        print("="*70)
        
        # Step 1: Prepare and validate data
        print("\n1. Preparing experimental data...")
        experiment_data = self._prepare_data(data_df)
        
        if len(experiment_data) < 3:
            print(f"  Warning: Only {len(experiment_data)} points available.")
            if use_sobol_init:
                print("  Will use Sobol initialization for sparse data.")
            
        # Step 2: Transform and normalize inputs
        print("\n2. Processing input variables...")
        X_raw, X_normalized = self._process_inputs(experiment_data)
        
        # Step 3: Standardize outputs  
        print("\n3. Processing output variables...")
        Y_raw, Y_standardized = self._process_outputs(experiment_data)
        
        # Handle sparse data with Sobol initialization
        if len(experiment_data) < 10 and use_sobol_init:
            print(f"\n3.5. Adding Sobol initialization points...")
            if n_sobol is None:
                n_sobol = min(5, max(1, n_points // 2))
            X_normalized, Y_standardized = self._add_sobol_points(
                X_normalized, Y_standardized, n_sobol)
        
        # Step 4: Fit Gaussian Process models
        print("\n4. Fitting Gaussian Process models...")
        models = self._fit_models(X_normalized, Y_standardized)
        
        # Step 5: Generate batch recommendations  
        print("\n5. Optimizing acquisition function...")
        X_candidates = self._propose_batch(models, X_normalized, n_points, min_distance)
        
        # Step 6: Convert back to original space and format
        print("\n6. Converting to original space...")
        recommendations = self._format_recommendations(X_candidates)
        
        self._print_summary(recommendations, experiment_data)
        
        return recommendations
    
    def _prepare_data(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate experimental data."""
        
        # Filter to experimental data if available
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
        
        # Remove any rows with NaN values
        initial_len = len(experiment_data)
        experiment_data = experiment_data.dropna(subset=self.input_columns + self.output_columns)
        if len(experiment_data) < initial_len:
            print(f"  Removed {initial_len - len(experiment_data)} rows with NaN values")
        
        # Show data ranges
        for col in self.input_columns:
            min_val, max_val = experiment_data[col].min(), experiment_data[col].max()
            print(f"  {col}: {min_val:.4e} - {max_val:.4e}")
        
        for col in self.output_columns:
            min_val, max_val = experiment_data[col].min(), experiment_data[col].max()
            print(f"  {col}: {min_val:.4f} - {max_val:.4f}")
        
        return experiment_data
    
    def _process_inputs(self, experiment_data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform and normalize input variables."""
        
        X_raw = experiment_data[self.input_columns].values.astype(np.float64)
        
        if self.log_transform_inputs:
            # Apply log10 transform to concentrations
            X_log = np.log10(X_raw)
            print(f"  Applied log10 transform:")
            for i, col in enumerate(self.input_columns):
                print(f"    {col}: [{X_log[:, i].min():.3f}, {X_log[:, i].max():.3f}]")
            X_transformed = X_log
        else:
            X_transformed = X_raw
            print(f"  Using original input space")
        
        # Store bounds for this dataset (in transformed space)
        self.input_bounds = torch.tensor([
            [X_transformed[:, i].min(), X_transformed[:, i].max()] 
            for i in range(self.n_inputs)
        ], dtype=self.dtype, device=self.device).T  # Shape: (2, n_inputs)
        
        # Normalize to [0, 1]^d using bounds
        X_torch = torch.tensor(X_transformed, dtype=self.dtype, device=self.device)
        X_normalized = normalize(X_torch, bounds=self.input_bounds)
        
        print(f"  Normalized to [0,1]^{self.n_inputs}: {X_normalized.shape}")
        print(f"    Range check - min: {X_normalized.min():.6f}, max: {X_normalized.max():.6f}")
        
        return torch.tensor(X_transformed, dtype=self.dtype, device=self.device), X_normalized
    
    def _process_outputs(self, experiment_data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standardize output variables independently."""
        
        Y_raw = experiment_data[self.output_columns].values.astype(np.float64)
        Y_standardized = np.zeros_like(Y_raw)
        
        for i, col in enumerate(self.output_columns):
            values = Y_raw[:, i]
            
            # Fit StandardScaler for this output
            scaler = StandardScaler()
            standardized = scaler.fit_transform(values.reshape(-1, 1)).flatten()
            
            Y_standardized[:, i] = standardized
            self.output_scalers[col] = scaler
            
            print(f"  {col}: [{values.min():.4f}, {values.max():.4f}] -> "
                  f"standardized: [{standardized.min():.3f}, {standardized.max():.3f}]")
        
        Y_raw_torch = torch.tensor(Y_raw, dtype=self.dtype, device=self.device)
        Y_std_torch = torch.tensor(Y_standardized, dtype=self.dtype, device=self.device)
        
        return Y_raw_torch, Y_std_torch
    
    def _add_sobol_points(self, X_existing: torch.Tensor, Y_existing: torch.Tensor, 
                         n_sobol: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add Sobol sequence points for sparse data initialization."""
        
        # Generate Sobol points in [0,1]^d
        bounds_01 = torch.tensor([[0.0] * self.n_inputs, [1.0] * self.n_inputs], 
                                dtype=self.dtype, device=self.device)
        
        X_sobol = draw_sobol_samples(bounds=bounds_01, n=n_sobol, q=1).squeeze(1)
        
        # Create synthetic outputs (zeros for now - GPs will handle uncertainty)
        Y_sobol = torch.zeros(n_sobol, 2, dtype=self.dtype, device=self.device)
        
        # Combine with existing data
        X_combined = torch.cat([X_existing, X_sobol], dim=0)
        Y_combined = torch.cat([Y_existing, Y_sobol], dim=0)
        
        print(f"  Added {n_sobol} Sobol points -> total: {len(X_combined)} points")
        
        return X_combined, Y_combined
    
    def _fit_models(self, X: torch.Tensor, Y: torch.Tensor) -> ModelListGP:
        """Fit independent Gaussian Process models for each output."""
        
        models = []
        
        for i, col in enumerate(self.output_columns):
            print(f"  Fitting GP for {col}...")
            
            # Create SingleTaskGP for this output
            y_i = Y[:, i:i+1]  # Keep as column vector
            
            model = SingleTaskGP(X, y_i)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            
            try:
                fit_gpytorch_mll(mll)
                print(f"    ✓ Fitted successfully")
            except Exception as e:
                print(f"    Warning: Fit issues for {col}: {e}")
                # Continue with default parameters if fitting fails
            
            models.append(model)
        
        model_list = ModelListGP(*models)
        return model_list
    
    def _propose_batch(self, models: ModelListGP, X_existing: torch.Tensor, 
                      n_points: int, min_distance: float) -> torch.Tensor:
        """Propose batch of points using sequential greedy selection."""
        
        bounds_01 = torch.tensor([[0.0] * self.n_inputs, [1.0] * self.n_inputs],
                                dtype=self.dtype, device=self.device)
        
        candidates = []
        selected_points = X_existing.clone()  # Track all selected points
        
        for i in range(n_points):
            print(f"  Selecting point {i+1}/{n_points}...")
            
            # Define acquisition function with current selected points
            def acquisition_fn(X_batch):
                return self._local_contrast_acquisition(models, X_batch, selected_points, min_distance)
            
            # Optimize acquisition function
            try:
                candidate, acq_value = optimize_acqf(
                    acq_function=acquisition_fn,
                    bounds=bounds_01,
                    q=1,
                    num_restarts=20,
                    raw_samples=500,
                )
                
                candidate_point = candidate.squeeze(0)  # Remove batch dimension
                candidates.append(candidate_point)
                selected_points = torch.cat([selected_points, candidate_point.unsqueeze(0)], dim=0)
                
                print(f"    Point {i+1}: acq_value = {acq_value:.4f}")
                
            except Exception as e:
                print(f"    Warning: Acquisition optimization failed: {e}")
                # Fallback to random point
                fallback = torch.rand(1, self.n_inputs, dtype=self.dtype, device=self.device)
                candidates.append(fallback.squeeze(0))
                selected_points = torch.cat([selected_points, fallback], dim=0)
        
        return torch.stack(candidates)
    
    def _local_contrast_acquisition(self, models: ModelListGP, X_batch: torch.Tensor,
                                  selected_points: torch.Tensor, min_distance: float) -> torch.Tensor:
        """
        Local contrast acquisition function with exploration bonus.
        
        Score(x) = mean over K random directions of sum_k |mu_k(x + delta*u) - mu_k(x)|
                  + lambda * sum_k sigma_k(x)
                  - diversity penalty for points too close to existing selections
        """
        
        batch_size = X_batch.shape[0]
        scores = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        
        for i in range(batch_size):
            x = X_batch[i:i+1]  # Shape: (1, n_inputs)
            
            # Generate K random unit directions
            directions = torch.randn(self.K, self.n_inputs, dtype=self.dtype, device=self.device)
            directions = directions / torch.norm(directions, dim=1, keepdim=True)
            
            # Evaluate local contrast
            contrast_sum = 0.0
            uncertainty_sum = 0.0
            
            with torch.no_grad():
                # Base prediction at x
                posterior_x = models.posterior(x)
                mean_x = posterior_x.mean.squeeze(-1)  # Shape: (1, n_outputs)
                var_x = posterior_x.variance.squeeze(-1)
                
                for k in range(self.K):
                    # Evaluate at x + delta * direction
                    x_perturb = x + self.delta * directions[k:k+1]
                    
                    # Clamp to [0,1] bounds
                    x_perturb = torch.clamp(x_perturb, 0.0, 1.0)
                    
                    posterior_perturb = models.posterior(x_perturb)
                    mean_perturb = posterior_perturb.mean.squeeze(-1)
                    
                    # Local contrast: sum over outputs of |mu_k(x+δu) - mu_k(x)|
                    contrast = torch.sum(torch.abs(mean_perturb - mean_x))
                    contrast_sum += contrast
                
                # Average over directions
                avg_contrast = contrast_sum / self.K
                
                # Exploration bonus: λ * sum_k σ_k(x)
                exploration_bonus = self.lam * torch.sum(torch.sqrt(var_x))
                
                # Diversity penalty: penalize points too close to existing selections
                min_dist_to_existing = float('inf')
                if len(selected_points) > 0:
                    distances = torch.norm(selected_points - x, dim=1)
                    min_dist_to_existing = torch.min(distances).item()
                
                diversity_penalty = 0.0
                if min_dist_to_existing < min_distance:
                    # Strong penalty for being too close
                    diversity_penalty = 10.0 * (min_distance - min_dist_to_existing)
                
                # Final score
                scores[i] = avg_contrast + exploration_bonus - diversity_penalty
        
        return scores
    
    def _format_recommendations(self, X_candidates: torch.Tensor) -> pd.DataFrame:
        """Convert normalized candidates back to original space and format as DataFrame."""
        
        # Denormalize from [0,1] to transformed space
        X_denorm = X_candidates * (self.input_bounds[1] - self.input_bounds[0]) + self.input_bounds[0]
        
        # Convert back to original space if log transformed
        if self.log_transform_inputs:
            X_original = torch.pow(10, X_denorm)
        else:
            X_original = X_denorm
        
        # Create DataFrame
        recommendations = pd.DataFrame(
            X_original.detach().cpu().numpy(),
            columns=self.input_columns
        )
        
        # Add metadata
        recommendations['recommendation_id'] = range(1, len(recommendations) + 1)
        recommendations['method'] = 'bayesian_transition'
        recommendations['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return recommendations
    
    def _print_summary(self, recommendations: pd.DataFrame, experiment_data: pd.DataFrame):
        """Print recommendation summary."""
        
        print(f"\n" + "="*50)
        print(f"BAYESIAN RECOMMENDATIONS SUMMARY")
        print(f"="*50)
        print(f"Generated {len(recommendations)} recommendations")
        print(f"Based on {len(experiment_data)} experimental points")
        print(f"Input space: {self.n_inputs}D, Output space: {self.n_outputs}D")
        print(f"Acquisition parameters: δ={self.delta:.3f}, K={self.K}, λ={self.lam:.3f}")
        
        print(f"\nRecommended concentration ranges:")
        for col in self.input_columns:
            min_val = recommendations[col].min()
            max_val = recommendations[col].max()
            print(f"  {col}: {min_val:.4e} - {max_val:.4e}")


def main():
    """
    Example usage with synthetic transition surface data.
    
    Creates a 3D test case with sharp transitions and runs 5 iterations
    of Bayesian active learning.
    """
    
    print("BAYESIAN TRANSITION RECOMMENDER - SYNTHETIC TEST")
    print("=" * 60)
    
    # Set seeds
    torch.manual_seed(123)
    np.random.seed(123)
    
    # Create synthetic 3D data with transition surface
    def synthetic_transition_function(X):
        """
        Synthetic function with sharp transitions.
        X: (n, 3) array of concentrations [SDS, CTAB, NaCl]
        Returns: (ratio, turbidity)
        """
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        
        # Ratio: sharp transition around x1 + x2 = 10 mM
        total_surf = x1 + x2
        ratio = 1.0 + 0.6 * np.tanh(5 * (total_surf - 10))  # Sharp transition at 10 mM
        ratio += 0.1 * x3 / 50  # Weak salt effect
        ratio += 0.05 * np.random.randn(len(X))  # Noise
        
        # Turbidity: transition around x3 = 25 mM salt
        turbidity = 0.2 + 0.4 * (1 / (1 + np.exp(-10 * (x3 - 25))))  # Sigmoid transition
        turbidity += 0.02 * total_surf / 20  # Weak surfactant effect  
        turbidity += 0.02 * np.random.randn(len(X))  # Noise
        
        return np.column_stack([ratio, turbidity])
    
    # Generate initial experimental design (Sobol)
    print("\n1. Generating initial experimental design...")
    
    # Concentration bounds: [0.1, 50] mM for surfactants, [0, 100] mM for salt
    n_initial = 15
    X_init_raw = np.column_stack([
        np.random.uniform(0.1, 50, n_initial),  # SDS
        np.random.uniform(0.1, 50, n_initial),  # CTAB  
        np.random.uniform(0, 100, n_initial)    # NaCl
    ])
    
    Y_init = synthetic_transition_function(X_init_raw)
    
    # Create initial DataFrame
    initial_data = pd.DataFrame({
        'SDS_conc_mm': X_init_raw[:, 0],
        'CTAB_conc_mm': X_init_raw[:, 1], 
        'NaCl_conc_mm': X_init_raw[:, 2],
        'ratio': Y_init[:, 0],
        'turbidity_600nm': Y_init[:, 1],
        'well_type': 'experiment'
    })
    
    print(f"Initial design: {len(initial_data)} points")
    print(f"Ratio range: [{Y_init[:, 0].min():.3f}, {Y_init[:, 0].max():.3f}]")  
    print(f"Turbidity range: [{Y_init[:, 1].min():.3f}, {Y_init[:, 1].max():.3f}]")
    
    # Initialize recommender
    recommender = BayesianTransitionRecommender(
        input_columns=['SDS_conc_mm', 'CTAB_conc_mm', 'NaCl_conc_mm'],
        output_columns=['ratio', 'turbidity_600nm'],
        log_transform_inputs=True,  # Log space for concentrations
        delta=0.08,    # Larger step for 3D
        K=8,           # Fewer directions for efficiency
        lam=0.4        # Higher exploration
    )
    
    # Run iterative optimization
    all_data = initial_data.copy()
    
    for iteration in range(5):
        print(f"\n" + "="*60)
        print(f"ITERATION {iteration + 1}/5")
        print(f"="*60)
        
        # Get recommendations
        recommendations = recommender.get_recommendations(
            all_data,
            n_points=4,        # Smaller batches
            min_distance=0.15  # Larger diversity spacing
        )
        
        print(f"\nRecommended experiments for iteration {iteration + 1}:")
        for i, row in recommendations.iterrows():
            print(f"  {i+1}. SDS: {row['SDS_conc_mm']:.3f} mM, "
                  f"CTAB: {row['CTAB_conc_mm']:.3f} mM, "
                  f"NaCl: {row['NaCl_conc_mm']:.3f} mM")
        
        # "Run" experiments (evaluate synthetic function)
        X_new = recommendations[['SDS_conc_mm', 'CTAB_conc_mm', 'NaCl_conc_mm']].values
        Y_new = synthetic_transition_function(X_new)
        
        # Add to dataset
        new_data = pd.DataFrame({
            'SDS_conc_mm': X_new[:, 0],
            'CTAB_conc_mm': X_new[:, 1],
            'NaCl_conc_mm': X_new[:, 2], 
            'ratio': Y_new[:, 0],
            'turbidity_600nm': Y_new[:, 1],
            'well_type': 'experiment'
        })
        
        all_data = pd.concat([all_data, new_data], ignore_index=True)
        
        print(f"New measurements:")
        for i in range(len(new_data)):
            print(f"  {i+1}. Ratio: {Y_new[i, 0]:.3f}, Turbidity: {Y_new[i, 1]:.3f}")
        
        print(f"Total data points: {len(all_data)}")
    
    print(f"\n" + "="*60)
    print(f"OPTIMIZATION COMPLETE")
    print(f"="*60)
    print(f"Final dataset: {len(all_data)} total measurements")
    print(f"Explored ratio range: [{all_data['ratio'].min():.3f}, {all_data['ratio'].max():.3f}]")
    print(f"Explored turbidity range: [{all_data['turbidity_600nm'].min():.3f}, {all_data['turbidity_600nm'].max():.3f}]")
    
    # Check if we found transitions
    ratio_std = all_data['ratio'].std()
    turb_std = all_data['turbidity_600nm'].std()
    print(f"Output variation captured - Ratio σ: {ratio_std:.3f}, Turbidity σ: {turb_std:.3f}")
    
    return all_data


if __name__ == "__main__":
    results = main()
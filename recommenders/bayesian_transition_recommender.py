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
- Candidate pool + greedy batch selection for boundary spreading
- Comprehensive diagnostics and performance tracking

Algorithm:
1. Normalize inputs to [0,1]^d (with optional log transform)
2. Standardize outputs independently 
3. Fit separate GP per output using SingleTaskGP
4. Generate large Sobol candidate pool
5. Evaluate acquisition on all candidates (vectorized)
6. Select batch using greedy max-score with hard min_distance constraint
7. Track boundary exploration metrics and performance

Usage:
    recommender = BayesianTransitionRecommender(
        input_columns=['surf_A_conc_mm', 'surf_B_conc_mm'],
        output_columns=['ratio', 'turbidity_600nm'], 
        log_transform_inputs=True,
        delta=0.05,
        K=24,
        lam=0.2
    )
    
    recommendations = recommender.get_recommendations(
        data_df,
        n_points=14,
        min_distance=0.06
    )
"""

import pandas as pd
import numpy as np
import torch
import gpytorch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize, standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ========================================================================
# CONFIGURATION SECTION
# ========================================================================

# Core algorithm parameters
Q_BATCH = 14              # Default batch size
CANDIDATE_POOL = 50_000   # Size of Sobol candidate pool for greedy selection
# MIN_DIST: Now computed adaptively based on points + dimensions
ALPHA_SPACING = 0.5       # Boundary density factor (0.5 = balanced spreading)

# Experimental design parameters
N_INITIAL = 50            # Initial Sobol points
N_ITERATIONS = 10         # Optimization iterations

# Local contrast acquisition parameters
DELTA = 0.03             # Step size for directional gradients (smaller = finer resolution)
K = 24                   # Number of random directions
LAM = 0.05               # Exploration weight (λ) - LOWER = focus on boundaries over exploration

# Performance metrics thresholds
EPS_NEAR = 0.05          # Distance threshold for "near boundary" points
EPS_FAR = 0.2            # Distance threshold for "far from boundary" points

# ========================================================================
# SYNTHETIC FUNCTION PARAMETERS
# ========================================================================

# Ellipsoidal boundary parameters
A_ELLIPSE = 0.3           # Semi-axis a
B_ELLIPSE = 0.25          # Semi-axis b  
C_ELLIPSE = 0.2           # Semi-axis c
CENTER_X = 0.5            # Center x-coordinate
CENTER_Y = 0.5            # Center y-coordinate
CENTER_Z = 0.5            # Center z-coordinate

# Saddle boundary parameters
ALPHA_SADDLE = 0.8        # Curvature parameter
Z0_SADDLE = 0.5           # Saddle center height

# Output function parameters
F1_INSIDE = 1.0           # f1 value inside boundary
F1_OUTSIDE = 3.0          # f1 value outside boundary
F2_INSIDE = 0.2           # f2 value inside boundary
F2_OUTSIDE = 0.8          # f2 value outside boundary
SMOOTH_EPS = 0.08         # Smoothing width for transitions (wider = less sharp)
LAM = 0.1                # Uncertainty bonus weight (λ in acquisition function) - LOW = focus on boundaries

# Boundary diagnostics parameters  
EPS_NEAR = 0.05          # Near-boundary threshold (normalized units)
EPS_FAR = 0.15           # Far-boundary threshold for waste calculation
SEED = 0                 # Random seed for reproducibility

# Algorithm behavior
USE_OPTIMIZE_ACQF = False  # Flag to use old optimize_acqf method (not recommended)

DEFAULT_CONFIG = {
    'delta': DELTA,         
    'K': K,             
    'lam': LAM,             
    'q': Q_BATCH,           
    'min_distance': 0.06,  # Placeholder - will be overridden by adaptive calculation
    'candidate_pool': CANDIDATE_POOL,
    'device': 'cpu',        
    'dtype': torch.double   
}

class BayesianTransitionRecommender:
    """
    Bayesian optimization recommender for discovering transition regions 
    in multi-dimensional surfactant concentration spaces.
    """
    
    def __init__(self, input_columns: List[str], output_columns: List[str],
                 log_transform_inputs: bool = True, 
                 delta: float = DELTA, K: int = K, lam: float = LAM,
                 min_distance: float = 0.06, candidate_pool: int = CANDIDATE_POOL,
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
        min_distance : float
            Minimum distance between selected points (normalized space)
        candidate_pool : int
            Size of Sobol candidate pool for greedy selection
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
        self.min_distance = min_distance
        self.candidate_pool = candidate_pool
        self.device = torch.device(device)
        self.dtype = dtype
        
        self.n_inputs = len(input_columns)
        self.n_outputs = len(output_columns)
        
        # Storage for fitted scalers and bounds
        self.input_bounds = None  # Original space bounds
        self.output_scalers = {}
        
        # Metrics tracking
        self.metrics_data = []
        
        # Set random seeds for reproducibility
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        
        print(f"Initialized BayesianTransitionRecommender:")
        print(f"  Input variables ({self.n_inputs}D): {input_columns}")
        print(f"  Output variables (2D): {output_columns}")
        print(f"  Log transform inputs: {log_transform_inputs}")
        print(f"  Acquisition params: delta={delta:.3f}, K={K}, lambda={lam:.3f}")
        min_dist_str = "adaptive" if min_distance is None else f"{min_distance:.3f}"
        print(f"  Batch selection: candidate_pool={candidate_pool:,}, min_dist={min_dist_str}")
        print(f"  Device: {self.device}, dtype: {dtype}")
    
    def get_recommendations(self, data_df: pd.DataFrame, n_points: int = Q_BATCH,
                          min_distance: float = None, iteration: int = None,
                          boundary_func: callable = None) -> pd.DataFrame:
        """
        Get Bayesian optimization recommendations for next experiments.
        
        Parameters:
        -----------
        data_df : pd.DataFrame
            Experimental data with input_columns and output_columns
        n_points : int
            Number of points to recommend
        min_distance : float, optional
            Minimum distance between recommended points (default: self.min_distance)
        iteration : int, optional
            Current iteration number (for metrics tracking)
        boundary_func : callable, optional
            Function h(x) where boundary is h(x)=0 (for diagnostics)
            
        Returns:
        --------
        pd.DataFrame
            Recommended experiments with acquisition scores
        """
        
        if min_distance is None:
            min_distance = self.min_distance
            
        print(f"\n{'='*70}")
        print(f"BAYESIAN TRANSITION REGION DISCOVERY")
        print(f"{'='*70}")
        
        # Step 1: Prepare and validate data
        print(f"\n1. Preparing experimental data...")
        experiment_data = self._prepare_data(data_df)
        
        # Step 2: Transform and normalize inputs
        print(f"\n2. Processing input variables...")
        X_raw, X_normalized = self._process_inputs(experiment_data)
        
        # Step 3: Standardize outputs  
        print(f"\n3. Processing output variables...")
        Y_raw, Y_standardized = self._process_outputs(experiment_data)
        
        # Step 4: Fit Gaussian Process models
        print(f"\n4. Fitting Gaussian Process models...")
        models = self._fit_models(X_normalized, Y_standardized)
        
        # Step 5: Generate batch recommendations using candidate pool + greedy selection
        print(f"\n5. Generating candidate pool and selecting batch...")
        X_candidates = self._propose_batch_greedy(models, X_normalized, n_points, min_distance, boundary_func)
        
        # Step 6: Convert back to original space and format
        print(f"\n6. Converting to original space...")
        recommendations = self._format_recommendations(X_candidates)
        
        # Step 7: Compute and track metrics (if boundary function provided)
        if boundary_func is not None and iteration is not None:
            print(f"\n7. Computing performance metrics...")
            self._compute_metrics(X_candidates, X_normalized, iteration, boundary_func, min_distance)
        
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
    
    def _compute_adaptive_min_distance(self, n_existing: int, n_batch: int) -> float:
        """
        Compute adaptive minimum distance based on point density and dimensionality.
        
        For boundary-focused sampling in d dimensions:
        min_dist = α * (1/n_total)^(1/d)
        
        Where α < 1 gives denser spacing than uniform space-filling.
        """
        n_total = n_existing + n_batch
        
        # Theoretical uniform spacing for n_total points in d-dimensional unit cube
        uniform_spacing = (1.0 / n_total) ** (1.0 / self.n_inputs)
        
        # Apply boundary density factor (α < 1 = denser than uniform)
        adaptive_min_dist = ALPHA_SPACING * uniform_spacing
        
        # Reasonable bounds to prevent extreme values
        min_allowed = 0.02  # Don't allow points too close
        max_allowed = 0.25  # Don't force points too far apart
        
        final_min_dist = max(min_allowed, min(max_allowed, adaptive_min_dist))
        
        print(f"    Adaptive spacing: n_total={n_total}, d={self.n_inputs}D → "
              f"uniform={uniform_spacing:.4f}, α={ALPHA_SPACING} → min_dist={final_min_dist:.4f}")
        
        return final_min_dist
    
    def _propose_batch_greedy(self, models: ModelListGP, X_existing: torch.Tensor, 
                            n_points: int, requested_min_distance: float, boundary_func: callable = None) -> torch.Tensor:
        """
        Two-stage batch selection with sigmoid-gated uncertainty and auto-scaling.
        
        Stage 1: Select high-contrast points (pure boundary detection)
        Stage 2: Select high-uncertainty points from boundary regions
        """
        
        # Compute adaptive minimum distance (overrides requested value)
        min_distance = self._compute_adaptive_min_distance(len(X_existing), n_points)
        
        bounds_01 = torch.tensor([[0.0] * self.n_inputs, [1.0] * self.n_inputs],
                                dtype=self.dtype, device=self.device)
        
        # Step 1: Generate large Sobol candidate pool
        print(f"  Generating {self.candidate_pool:,} candidate points...")
        X_candidates_pool = draw_sobol_samples(
            bounds=bounds_01, n=self.candidate_pool, q=1
        ).squeeze(1)  # Shape: (candidate_pool, n_inputs)
        
        # Step 2: Compute contrast and uncertainty for all candidates
        print(f"  Computing contrast and uncertainty on {self.candidate_pool:,} candidates...")
        contrast, uncertainty = self._compute_contrast_uncertainty(models, X_candidates_pool)
        
        # Step 3: Sigmoid gating and auto-scaling with detailed diagnostics
        c0 = torch.quantile(contrast, 0.70)
        s = 0.10 * torch.std(contrast) + 1e-12
        gate = torch.sigmoid((contrast - c0) / s)
        
        # DIAGNOSTICS: Analyze contrast distribution and filtering effectiveness
        contrast_min, contrast_max = torch.min(contrast), torch.max(contrast)
        contrast_median = torch.median(contrast)
        contrast_std = torch.std(contrast)
        
        # Check how many candidates would pass Stage 2 filter
        stage2_mask = contrast >= c0
        n_stage2_candidates = torch.sum(stage2_mask).item()
        
        print(f"  CONTRAST DIAGNOSTICS:")
        print(f"    Range: [{contrast_min:.6f}, {contrast_max:.6f}], median={contrast_median:.6f}, std={contrast_std:.6f}")
        print(f"    c0 (70th percentile): {c0:.6f}")
        print(f"    Stage2 candidates: {n_stage2_candidates}/{len(contrast)} ({100*n_stage2_candidates/len(contrast):.1f}%) pass contrast >= c0")
        
        # Sample some gate values to see sigmoid effectiveness
        sample_indices = torch.randperm(len(gate))[:5]
        print(f"    Sample gates: {[f'{gate[i].item():.3f}' for i in sample_indices]}")
        
        # Check if we need more aggressive filtering
        if c0 < 0.05:  # Threshold too low
            print(f"    WARNING: c0={c0:.6f} is very low - might let baseline points through")
        if n_stage2_candidates > 0.5 * len(contrast):  # Too many candidates pass
            print(f"    WARNING: {100*n_stage2_candidates/len(contrast):.1f}% candidates pass filter - threshold may be too permissive")
        
        # ALTERNATIVE FILTERING IDEAS (if current approach fails):
        # Option 1: Absolute threshold: stage2_mask = contrast >= 0.1  
        # Option 2: Higher percentile: c0 = torch.quantile(contrast, 0.90)
        # Option 3: Multiplicative gate: stage2_scores = uncertainty * (gate ** 2)  # More aggressive
        # Option 4: Boundary validation: only allow candidates with |h(x)| < eps_near
        
        # Auto-scale lambda
        k = 0.2
        median_contrast = torch.median(contrast)
        median_unc = torch.median(uncertainty)
        lam_eff = k * median_contrast / (median_unc + 1e-12)
        lam_eff = torch.clamp(lam_eff, min=0.01, max=0.5)
        
        print(f"  Adaptive parameters | c0={c0:.4f} | lam_eff={lam_eff:.4f}")
        
        # UNIFIED ACQUISITION with soft distance weighting (inspired by Delaunay approach)
        print(f"  Computing unified acquisition function with soft distance penalties...")
        
        # Step 1: Compute distance weights for all candidates
        print(f"    Computing distance weights to {len(X_existing)} existing points...")
        if len(X_existing) > 0:
            # Calculate minimum distance from each candidate to existing points
            distances = torch.cdist(X_candidates_pool, X_existing)  # Shape: (n_candidates, n_existing)
            min_distances = torch.min(distances, dim=1)[0]  # Shape: (n_candidates,)
        else:
            # No existing points - all distances are infinite (weight = 1.0)
            min_distances = torch.full((len(X_candidates_pool),), float('inf'), device=self.device)
        
        # Soft distance weighting: sigmoid penalty based on proximity
        target_distance = min_distance  # Use adaptive min_distance as target
        distance_scale = target_distance * 0.5  # Controls sigmoid steepness
        distance_weights = torch.sigmoid((min_distances - target_distance) / (distance_scale + 1e-12))
        
        # Step 2: Compute unified acquisition scores
        # Enhanced contrast gating: only boost uncertainty in high-contrast regions
        c0_strict = torch.quantile(contrast, 0.90)  # Use 90th percentile for tighter gating
        gate_strict = torch.sigmoid((contrast - c0_strict) / (0.20 * torch.std(contrast) + 1e-12))
        
        # Unified acquisition function
        # exploration_bonus = lam_eff * uncertainty * gate_strict  # Temporarily disabled
        exploration_bonus = torch.zeros_like(uncertainty)  # TEMP: Pure contrast-based selection
        raw_acquisition = contrast + exploration_bonus
        final_acquisition = raw_acquisition * distance_weights
        
        print(f"    Acquisition components:")
        print(f"      Contrast range: [{torch.min(contrast):.6f}, {torch.max(contrast):.6f}]")
        print(f"      Distance weights range: [{torch.min(distance_weights):.3f}, {torch.max(distance_weights):.3f}]") 
        print(f"      c0_strict (90th percentile): {c0_strict:.6f}")
        print(f"      Final acquisition range: [{torch.min(final_acquisition):.6f}, {torch.max(final_acquisition):.6f}]")
        
        # Get eps_near for boundary validation (diagnostics only)
        eps_near = 0.1  # Default
        if boundary_func is not None:
            func_name = boundary_func.__name__
            if 'ellipse' in func_name:
                eps_near = 0.3
            elif 'saddle' in func_name:
                eps_near = 0.05
            elif 'spiral' in func_name:
                eps_near = 0.1

        # Step 3: Greedy selection with unified scores (no hard distance constraints)
        print(f"  Selecting {n_points} points by unified acquisition (contrast+exploration)*distance_weight...")
        selected_indices = []
        
        # Sort all candidates by unified acquisition score (descending)
        sorted_indices = torch.argsort(final_acquisition, descending=True)
        
        for i in range(n_points):
            if i >= len(sorted_indices):
                print(f"    Warning: Only {i} candidates available, stopping selection")
                break
                
            idx = sorted_indices[i].item()
            candidate = X_candidates_pool[idx:idx+1]
            selected_indices.append(idx)
            
            # Diagnostic output with boundary validation
            boundary_info = ""
            if boundary_func is not None:
                candidate_np = candidate.detach().cpu().numpy()[0]
                h_val = boundary_func(candidate_np)
                is_boundary = abs(h_val) < eps_near
                status = "✓BOUNDARY" if is_boundary else "✗BASELINE"
                boundary_info = f", h={h_val:.3f} {status}"
            
            print(f"    Point {i+1}/{n_points}: unified_acq={final_acquisition[idx]:.6f} "
                  f"(contrast={contrast[idx]:.4f}, dist_weight={distance_weights[idx]:.3f})"
                  f"{boundary_info}")
            
            # Update distance weights for remaining candidates based on this selection
            if i < n_points - 1:  # Not the last point
                # Recalculate distances including this newly selected point
                remaining_mask = torch.ones(len(X_candidates_pool), dtype=torch.bool)
                remaining_mask[selected_indices] = False
                
                if torch.any(remaining_mask):
                    remaining_candidates = X_candidates_pool[remaining_mask]
                    all_selected = torch.cat([X_existing, X_candidates_pool[selected_indices]], dim=0)
                    
                    new_distances = torch.cdist(remaining_candidates, all_selected)
                    new_min_distances = torch.min(new_distances, dim=1)[0]
                    new_distance_weights = torch.sigmoid((new_min_distances - target_distance) / (distance_scale + 1e-12))
                    
                    # Update acquisition scores for remaining candidates
                    remaining_raw_acq = raw_acquisition[remaining_mask]
                    remaining_final_acq = remaining_raw_acq * new_distance_weights
                    final_acquisition[remaining_mask] = remaining_final_acq
                    
                    # Re-sort remaining candidates
                    remaining_indices = torch.where(remaining_mask)[0]
                    remaining_sorted = torch.argsort(remaining_final_acq, descending=True)
                    sorted_indices[i+1:] = remaining_indices[remaining_sorted][:len(sorted_indices)-(i+1)]
        
        # Step 4: Return selected points and diagnostics
        if selected_indices:
            X_new_batch = X_candidates_pool[selected_indices]
            
            # Compute unified diagnostics
            selected_contrast = contrast[selected_indices] 
            selected_dist_weights = distance_weights[selected_indices]
            
            if len(selected_indices) > 1:
                from scipy.spatial.distance import pdist
                batch_dists = pdist(X_new_batch.detach().cpu().numpy())
                min_pairwise_dist = np.min(batch_dists)
                mean_pairwise_dist = np.mean(batch_dists)
            else:
                min_pairwise_dist = float('nan')
                mean_pairwise_dist = float('nan')
            
            print(f"  UNIFIED SELECTION SUMMARY:")
            print(f"    Selected contrast range: [{torch.min(selected_contrast):.4f}, {torch.max(selected_contrast):.4f}]")
            print(f"    Selected distance weights: [{torch.min(selected_dist_weights):.3f}, {torch.max(selected_dist_weights):.3f}]")
            print(f"    Pairwise distances: min={min_pairwise_dist:.3f}, mean={mean_pairwise_dist:.3f}")
            
        else:
            # Fallback: select top contrast points  
            print("    Fallback: Selecting top contrast points")
            top_indices = torch.argsort(contrast, descending=True)[:n_points]
            X_new_batch = X_candidates_pool[top_indices]
        
        return X_new_batch
    
    def _compute_contrast_uncertainty(self, models: ModelListGP, X_candidates: torch.Tensor) -> tuple:
        """
        Compute contrast and uncertainty arrays for all candidates.
        
        Returns:
            contrast: torch.Tensor of shape (n_candidates,)
            uncertainty: torch.Tensor of shape (n_candidates,) 
        """
        
        n_candidates = X_candidates.shape[0]
        
        # Pre-generate K random unit directions (fixed for all candidates)
        directions = torch.randn(self.K, self.n_inputs, dtype=self.dtype, device=self.device)
        directions = directions / torch.norm(directions, dim=1, keepdim=True)
        
        # Evaluate base predictions for all candidates
        with torch.no_grad():  # No gradients needed for candidate evaluation
            posterior = models.posterior(X_candidates)
            mean_base = posterior.mean  # Shape: (n_candidates, n_outputs)
            var_base = posterior.variance  # Shape: (n_candidates, n_outputs)
        
        # Compute local contrast scores
        contrast_scores = torch.zeros(n_candidates, dtype=self.dtype, device=self.device)
        
        for k in range(self.K):
            # Perturb all candidates in direction k
            X_perturbed = X_candidates + self.delta * directions[k:k+1]  # Broadcasting
            X_perturbed = torch.clamp(X_perturbed, 0.0, 1.0)  # Stay in [0,1]^d
            
            # Evaluate perturbed predictions
            with torch.no_grad():
                posterior_perturb = models.posterior(X_perturbed)
                mean_perturb = posterior_perturb.mean  # Shape: (n_candidates, n_outputs)
            
            # Local contrast: sum over outputs of |mu_k(x+δu) - mu_k(x)|
            contrast = torch.sum(torch.abs(mean_perturb - mean_base), dim=1)  # Shape: (n_candidates,)
            contrast_scores += contrast
        
        # Average over directions
        avg_contrast = contrast_scores / self.K
        
        # Uncertainty: sum_k σ_k(x)
        uncertainty_scores = torch.sum(torch.sqrt(var_base), dim=1)
        
        return avg_contrast, uncertainty_scores
    
    def _vectorized_acquisition(self, models: ModelListGP, X_candidates: torch.Tensor) -> torch.Tensor:
        """
        Legacy acquisition function - kept for compatibility but not used in two-stage selection.
        """
        contrast, uncertainty = self._compute_contrast_uncertainty(models, X_candidates)
        return contrast + self.lam * uncertainty
    
    def _compute_metrics(self, X_new: torch.Tensor, X_all: torch.Tensor, iteration: int,
                        boundary_func: callable, min_dist_used: float):
        """
        Compute clean cumulative metrics for boundary exploration performance.
        
        4 key metrics:
        1. frac_near_cumulative: Overall boundary hit rate  
        2. frac_far_cumulative: Overall waste rate
        3. nn_spacing_cumulative: Boundary point spacing quality
        4. min_batch_dist: Distance constraint validation
        """
        
        X_new_np = X_new.detach().cpu().numpy()
        X_all_np = X_all.detach().cpu().numpy()
        
        # Compute boundary distances for all points (cumulative)
        h_all = np.array([boundary_func(x) for x in X_all_np])
        
        # Set fixed thresholds based on boundary function name
        func_name = boundary_func.__name__
        if 'ellipse' in func_name:
            eps_near = 0.3    # Ellipse scale: ~±12 range
            eps_far = 1.0
        elif 'saddle' in func_name:
            eps_near = 0.05   # Saddle scale: ~±0.85 range 
            eps_far = 0.15
        elif 'spiral' in func_name:
            eps_near = 0.1    # Spiral scale: intermediate
            eps_far = 0.3
        else:
            eps_near = 0.1    # Default fallback
            eps_far = 0.3
            print(f"  Warning: Unknown boundary function {func_name}, using default thresholds")
        
        # 1. Cumulative boundary hit rates
        near_mask_all = np.abs(h_all) < eps_near
        far_mask_all = np.abs(h_all) > eps_far
        
        frac_near_cumulative = np.mean(near_mask_all)  # Overall boundary hit rate
        frac_far_cumulative = np.mean(far_mask_all)    # Overall waste rate
        
        # 2. Boundary point spacing (cumulative)
        nn_spacing_cumulative = np.nan
        n_near = np.sum(near_mask_all)
        
        if n_near >= 2:
            X_near = X_all_np[near_mask_all]
            from scipy.spatial import cKDTree
            tree = cKDTree(X_near)
            nn_dists, _ = tree.query(X_near, k=2)  # k=2 to exclude self-distance
            actual_nn = nn_dists[:, 1]  # Actual nearest-neighbor distances
            nn_spacing_cumulative = np.median(actual_nn)
        
        # 3. Batch distance constraint validation
        min_batch_dist = np.nan
        if len(X_new) > 1:
            from scipy.spatial.distance import pdist
            batch_dists = pdist(X_new_np)
            min_batch_dist = np.min(batch_dists)
        
        # Store simplified metrics
        metrics = {
            'iteration': iteration,
            'n_total': len(X_all),
            'n_near': n_near,
            'frac_near_cumulative': frac_near_cumulative,
            'frac_far_cumulative': frac_far_cumulative,
            'nn_spacing_cumulative': nn_spacing_cumulative,
            'min_batch_dist': min_batch_dist,
            'eps_near': eps_near,
            'eps_far': eps_far
        }
        
        self.metrics_data.append(metrics)
        
        # Clean console logging
        print(f"  Metrics | iter {iteration} | "
              f"near={frac_near_cumulative:.3f} | far={frac_far_cumulative:.3f} | "
              f"spacing={nn_spacing_cumulative:.3f} | batch_dist={min_batch_dist:.3f} | "
              f"n_near={n_near}")
        
        # Show threshold info on first iteration
        if iteration <= 1:
            print(f"  Thresholds | {func_name} | eps_near={eps_near} | eps_far={eps_far}")
    
    def get_metrics_df(self) -> pd.DataFrame:
        """Return metrics as pandas DataFrame."""
        return pd.DataFrame(self.metrics_data)
    
    def plot_diagnostics(self, out_dir: str = None, show: bool = True):
        """
        Create diagnostic plots for the new simplified metrics.
        """
        
        if not self.metrics_data:
            print("No metrics data available for plotting")
            return
        
        metrics_df = self.get_metrics_df()
        
        # Create output directory if specified
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Boundary performance (cumulative)
        ax1.plot(metrics_df['iteration'], 100*metrics_df['frac_near_cumulative'], 'g-o', label='Boundary hits (%)', linewidth=2)
        ax1.plot(metrics_df['iteration'], 100*metrics_df['frac_far_cumulative'], 'r-o', label='Waste (%)', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Percentage')
        ax1.set_title('Boundary Targeting Performance (Higher near = better)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Point count and coverage
        ax2.plot(metrics_df['iteration'], metrics_df['n_total'], 'b-o', label='Total points')
        ax2.plot(metrics_df['iteration'], metrics_df['n_near'], 'g-s', label='Near boundary')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Count')
        ax2.set_title('Point Accumulation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Spacing quality
        if 'nn_spacing_cumulative' in metrics_df.columns:
            valid_spacing = metrics_df['nn_spacing_cumulative'].dropna()
            if len(valid_spacing) > 0:
                ax3.plot(valid_spacing.index, valid_spacing.values, 'purple', marker='o', label='NN spacing')
        ax3.plot(metrics_df['iteration'], metrics_df['min_batch_dist'], 'orange', marker='s', label='Min batch dist')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Distance')
        ax3.set_title('Point Spacing Quality') 
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Threshold info
        if 'eps_near' in metrics_df.columns and 'eps_far' in metrics_df.columns:
            ax4.axhline(y=metrics_df['eps_near'].iloc[0], color='green', linestyle='--', label=f"eps_near={metrics_df['eps_near'].iloc[0]:.3f}")
            ax4.axhline(y=metrics_df['eps_far'].iloc[0], color='red', linestyle='--', label=f"eps_far={metrics_df['eps_far'].iloc[0]:.3f}")
            ax4.set_xlabel('Threshold Type')
            ax4.set_ylabel('Distance Value')
            ax4.set_title('Boundary Distance Thresholds Used')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if out_dir:
            plt.savefig(os.path.join(out_dir, 'boundary_performance_diagnostics.png'), dpi=150)
        if show:
            plt.show()
        
        plt.close('all')  # Clean up
    
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

# =====================
# SYNTHETIC BOUNDARY FUNCTIONS (with signed distance h(x))
# =====================

def ellipse_boundary_3d(x: np.ndarray) -> float:
    """
    Ellipsoidal boundary: (x/a)^2 + (y/b)^2 + (z/c)^2 = 1
    
    Defines smooth transition surface in 3D space.
    Returns signed distance: negative inside, positive outside.
    """
    a, b, c = A_ELLIPSE, B_ELLIPSE, C_ELLIPSE
    cx, cy, cz = CENTER_X, CENTER_Y, CENTER_Z
    
    # Ellipsoid equation: (x-cx)^2/a^2 + (y-cy)^2/b^2 + (z-cz)^2/c^2
    ellipse_val = ((x[0] - cx) / a) ** 2 + ((x[1] - cy) / b) ** 2 + ((x[2] - cz) / c) ** 2
    
    # h(x) > 0 outside ellipsoid, h(x) < 0 inside  
    return ellipse_val - 1.0

def saddle_boundary_3d(x: np.ndarray) -> float:
    """
    Complex wavy saddle surface: z = 0.5 + 0.15*sin(4πx)*cos(4πy) + 0.1*(x²-y²)
    
    Creates a complex curved transition surface with multiple undulations.
    Much more visually interesting than simple ellipsoid!
    Returns signed distance from surface.
    """
    # Main saddle component
    saddle_base = Z0_SADDLE + ALPHA_SADDLE * (x[0] ** 2 - x[1] ** 2)
    
    # Add wavy oscillations for visual complexity
    wavy_component = 0.15 * np.sin(4 * np.pi * x[0]) * np.cos(4 * np.pi * x[1])
    
    # Final complex surface
    complex_surface_z = saddle_base + wavy_component
    
    # h(x) > 0 above surface, h(x) < 0 below
    return x[2] - complex_surface_z

def spiral_boundary_3d(x: np.ndarray) -> float:
    """
    Spiral shell boundary: r = 0.2 + 0.1*θ where θ = arctan(y/x) + 2πz
    
    Creates a beautiful 3D spiral shell - very complex curved boundary!
    """
    # Convert to cylindrical coordinates
    r = np.sqrt(x[0]**2 + x[1]**2)
    theta = np.arctan2(x[1], x[0]) + 2 * np.pi * x[2]  # Spiral with z
    
    # Spiral shell equation
    r_boundary = 0.2 + 0.1 * theta
    
    # h(x) > 0 outside spiral, h(x) < 0 inside
    return r - r_boundary

def synthetic_f1(x: np.ndarray, boundary_func: callable) -> float:
    """Output 1: Sharp transition based on boundary function."""
    h_val = boundary_func(x)
    
    if h_val > SMOOTH_EPS:
        return F1_OUTSIDE
    elif h_val < -SMOOTH_EPS:
        return F1_INSIDE  
    else:
        # Smooth transition in [-ε, ε]
        t = h_val / SMOOTH_EPS  # t ∈ [-1, 1]
        smooth_interp = 0.5 * (1 + t + (2/np.pi) * np.arctan(t/0.1))
        return F1_INSIDE + (F1_OUTSIDE - F1_INSIDE) * smooth_interp

def synthetic_f2(x: np.ndarray, boundary_func: callable) -> float:
    """Output 2: Smoother transition with different baseline."""
    h_val = boundary_func(x)
    
    # Sigmoid-like smooth transition
    transition = 1 / (1 + np.exp(-10 * h_val))  # Steepness factor = 10
    return F2_INSIDE + (F2_OUTSIDE - F2_INSIDE) * transition

def evaluate_synthetic_problem(X: np.ndarray, boundary_func: callable) -> np.ndarray:
    """
    Evaluate synthetic dual-output optimization problem.
    
    Args:
        X: Input points, shape (n_points, n_dims)
        boundary_func: Function computing h(x) signed distance
        
    Returns:
        Y: Output values, shape (n_points, 2)
    """
    n_points = X.shape[0]
    Y = np.zeros((n_points, 2))
    
    for i in range(n_points):
        Y[i, 0] = synthetic_f1(X[i], boundary_func)
        Y[i, 1] = synthetic_f2(X[i], boundary_func)
    
    return Y

def main():
    """
    Comprehensive test of boundary-exploring Bayesian recommender.
    
    Tests BOTH ellipsoidal and saddle boundary functions with:
    - Candidate pool + greedy selection approach
    - Performance metrics tracking (boundary distance, coverage, waste)
    - Diagnostic plotting showing algorithm effectiveness
    """
    
    print("BAYESIAN BOUNDARY EXPLORER - COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test multiple boundary types for visual comparison
    boundary_types = [
        ("Ellipsoidal", ellipse_boundary_3d, "Simple 3D ellipsoid"),
        ("Complex_Saddle", saddle_boundary_3d, "Wavy saddle with oscillations"),
        ("Spiral_Shell", spiral_boundary_3d, "3D spiral shell boundary")
    ]
    
    for boundary_name, boundary_func, description in boundary_types[:2]:  # Test first 2 types
        
        print(f"\n{'='*60}")
        print(f"TESTING {boundary_name.upper()} BOUNDARY ({description})")
        print(f"{'='*60}")
        
        # Step 1: Initialize with Sobol sampling
        print(f"\n1. Generating initial Sobol design...")
        
        # Create Sobol samples in [0,1]^3, then scale to problem domain  
        bounds_01 = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)
        X_sobol_01 = draw_sobol_samples(bounds=bounds_01, n=N_INITIAL, q=1).squeeze(1)
        X_sobol_np = X_sobol_01.detach().cpu().numpy()
        
        print(f"  Generated {N_INITIAL} Sobol points in [0,1]³")
        
        # Evaluate synthetic problem
        Y_sobol = evaluate_synthetic_problem(X_sobol_np, boundary_func)
        
        print(f"  Output ranges:")
        print(f"    f1: [{Y_sobol[:, 0].min():.3f}, {Y_sobol[:, 0].max():.3f}]")
        print(f"    f2: [{Y_sobol[:, 1].min():.3f}, {Y_sobol[:, 1].max():.3f}]")
        
        # Create initial DataFrame
        initial_data = pd.DataFrame({
            'x1': X_sobol_np[:, 0], 
            'x2': X_sobol_np[:, 1],
            'x3': X_sobol_np[:, 2],
            'f1': Y_sobol[:, 0],
            'f2': Y_sobol[:, 1],
            'iteration': 0
        })
        
        # Step 2: Initialize recommender with boundary-focused configuration
        print(f"\n2. Initializing Bayesian recommender...")
        recommender = BayesianTransitionRecommender(
            input_columns=['x1', 'x2', 'x3'],
            output_columns=['f1', 'f2'],
            log_transform_inputs=False,  # Already in [0,1]
            candidate_pool=CANDIDATE_POOL,
            min_distance=None,  # Use adaptive spacing
            delta=DELTA,
            K=K,
            lam=LAM
        )
        
        print(f"  Configuration:")
        print(f"    Candidate pool: {CANDIDATE_POOL:,} points")
        print(f"    Min distance: adaptive (density-based)")
        print(f"    Local contrast: δ={DELTA:.3f}, K={K}")
        print(f"    Exploration: λ={LAM:.2f}")
        
        # Step 3: Iterative optimization with metrics tracking
        all_data = initial_data.copy()
        
        print(f"\n3. Running {N_ITERATIONS} optimization iterations...")
        
        for iteration in range(1, N_ITERATIONS + 1):
            print(f"\n--- Iteration {iteration}/{N_ITERATIONS} ---")
            print(f"Current dataset: {len(all_data)} points")
            
            # Get batch of recommendations
            print(f"Getting {Q_BATCH} recommendations...")
            recommendations = recommender.get_recommendations(
                all_data, 
                n_points=Q_BATCH,
                iteration=iteration,        # Add iteration for metrics tracking
                boundary_func=boundary_func  # Pass for metrics computation
            )
            
            # Evaluate synthetic experiments  
            X_new = recommendations[['x1', 'x2', 'x3']].values
            Y_new = evaluate_synthetic_problem(X_new, boundary_func)
            
            # Add to dataset
            new_data = pd.DataFrame({
                'x1': X_new[:, 0],
                'x2': X_new[:, 1], 
                'x3': X_new[:, 2],
                'f1': Y_new[:, 0],
                'f2': Y_new[:, 1],
                'iteration': iteration
            })
            
            all_data = pd.concat([all_data, new_data], ignore_index=True)
            print(f"Updated dataset: {len(all_data)} total points")
        
        # Step 4: Performance analysis and diagnostics
        print(f"\n4. Performance Analysis ({boundary_name})...")
        
        # Get metrics DataFrame
        metrics_df = recommender.get_metrics_df()
        
        if len(metrics_df) > 0:
            final_metrics = metrics_df.iloc[-1]
            initial_metrics = metrics_df.iloc[0] if len(metrics_df) > 1 else final_metrics
            
            print(f"  Final Performance:")
            print(f"    Boundary hit rate: {100*final_metrics['frac_near_cumulative']:.1f}% (target: high)")
            print(f"    Points near boundary: {final_metrics['n_near']}/{final_metrics['n_total']} ({100*final_metrics['n_near']/final_metrics['n_total']:.1f}%)")
            print(f"    Sampling waste (far): {100*final_metrics['frac_far_cumulative']:.1f}% (target: low)")
            
            if not np.isnan(final_metrics['nn_spacing_cumulative']):
                print(f"    Boundary coverage (NN spacing): {final_metrics['nn_spacing_cumulative']:.3f}")
            
            # Show improvement in boundary targeting
            near_improvement = final_metrics['frac_near_cumulative'] - initial_metrics['frac_near_cumulative']
            waste_improvement = initial_metrics['frac_far_cumulative'] - final_metrics['frac_far_cumulative']
            print(f"  Improvement: +{100*near_improvement:.1f}% boundary hits, -{100*waste_improvement:.1f}% waste reduction")
        
        # Step 5: Visualization and diagnostics
        print(f"\n5. Creating diagnostic plots...")
        output_dir = f"boundary_diagnostics_{boundary_name.lower()}"
        
        try:
            recommender.plot_diagnostics(out_dir=output_dir, show=True)
            print(f"  ✓ Diagnostic plots saved to {output_dir}/")
        except Exception as e:
            print(f"  Warning: Plotting failed: {e}")
        
        # Step 6: Boundary visualization (if 3D)
        print(f"\n6. Creating 3D boundary visualization...")
        
        try:
            # Create TWO 3D plots: one colored by output value, one by iteration
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={'projection': '3d'})
            
            # Plot 1: Colored by f1 output (shows transition boundary)
            f1_values = all_data['f1'].values
            scatter1 = ax1.scatter(
                all_data['x1'], all_data['x2'], all_data['x3'],
                c=f1_values, cmap='RdYlBu_r', s=40, alpha=0.8, edgecolors='black', linewidth=0.5
            )
            
            ax1.set_xlabel('x1')
            ax1.set_ylabel('x2') 
            ax1.set_zlabel('x3')
            ax1.set_title(f'{boundary_name} - Output f1 Values\n(Shows Transition Boundary)')
            plt.colorbar(scatter1, ax=ax1, label='f1 Output', shrink=0.6)
            
            # Plot 2: Colored by iteration (shows search progress)
            iterations = all_data['iteration'].values
            scatter2 = ax2.scatter(
                all_data['x1'], all_data['x2'], all_data['x3'],
                c=iterations, cmap='viridis', s=40, alpha=0.8, edgecolors='black', linewidth=0.5
            )
            
            ax2.set_xlabel('x1')
            ax2.set_ylabel('x2') 
            ax2.set_zlabel('x3')
            ax2.set_title(f'{boundary_name} - Search Progress\n(Iteration Order)')
            plt.colorbar(scatter2, ax=ax2, label='Iteration', shrink=0.6)
            
            # Add boundary surface to BOTH plots
            if boundary_name == "Ellipsoidal":
                # Create ellipsoid surface
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x_ellipse = CENTER_X + A_ELLIPSE * np.outer(np.cos(u), np.sin(v))
                y_ellipse = CENTER_Y + B_ELLIPSE * np.outer(np.sin(u), np.sin(v)) 
                z_ellipse = CENTER_Z + C_ELLIPSE * np.outer(np.ones(np.size(u)), np.cos(v))
                
                # Only show the part within [0,1]³
                mask = ((x_ellipse >= 0) & (x_ellipse <= 1) & 
                       (y_ellipse >= 0) & (y_ellipse <= 1) &
                       (z_ellipse >= 0) & (z_ellipse <= 1))
                
                ax1.plot_surface(x_ellipse, y_ellipse, z_ellipse, 
                              alpha=0.2, color='red', label='True Boundary')
                ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, 
                              alpha=0.2, color='red', label='True Boundary')
                              
            elif boundary_name == "Complex_Saddle":
                # Create wavy saddle surface
                x_surf = np.linspace(0, 1, 30)
                y_surf = np.linspace(0, 1, 30)
                X_surf, Y_surf = np.meshgrid(x_surf, y_surf)
                
                # Complex wavy saddle: z = 0.5 + 0.8*(x²-y²) + 0.15*sin(4πx)*cos(4πy)
                Z_surf = (Z0_SADDLE + ALPHA_SADDLE * (X_surf**2 - Y_surf**2) + 
                         0.15 * np.sin(4 * np.pi * X_surf) * np.cos(4 * np.pi * Y_surf))
                
                ax1.plot_surface(X_surf, Y_surf, Z_surf, alpha=0.3, color='orange', label='True Boundary')
                ax2.plot_surface(X_surf, Y_surf, Z_surf, alpha=0.3, color='orange', label='True Boundary')
            
            plt.tight_layout()
            
            # Print analysis of boundary finding effectiveness
            f1_min, f1_max = f1_values.min(), f1_values.max()
            f1_range = f1_max - f1_min
            transition_found = f1_range > (F1_OUTSIDE - F1_INSIDE) * 0.7  # Found 70% of expected range
            
            print(f"  f1 Output Analysis:")
            print(f"    Range found: [{f1_min:.3f}, {f1_max:.3f}] (span: {f1_range:.3f})")
            print(f"    Expected range: [{F1_INSIDE:.3f}, {F1_OUTSIDE:.3f}] (span: {F1_OUTSIDE-F1_INSIDE:.3f})")
            print(f"    Transition discovered: {'✓ YES' if transition_found else '✗ PARTIAL'} ({100*f1_range/(F1_OUTSIDE-F1_INSIDE):.1f}% of expected)")
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)  # Create directory if needed
                plt.savefig(f"{output_dir}/3d_exploration_progress.png", dpi=150, bbox_inches='tight')
                print(f"  ✓ 3D plot saved to {output_dir}/3d_exploration_progress.png")
            
            plt.show()
            
        except Exception as e:
            print(f"  Warning: 3D visualization failed: {e}")
        
        print(f"\n{boundary_name} boundary test complete!")
    
    print(f"\n" + "="*60)
    print(f"COMPREHENSIVE BOUNDARY TEST COMPLETE")  
    print(f"="*60)
    print(f"Final dataset: {len(all_data)} points")
    print(f"Check diagnostic plots in boundary_diagnostics_*/ directories")
    
    return all_data


if __name__ == "__main__":
    results = main()
"""
TransitionRecommenderBase
=========================

Shared plumbing for transition-region recommenders. Handles data validation,
input log-transform + normalization, output standardization, GP-agnostic
orchestration, metrics tracking, and diagnostic plotting.

Subclasses must implement two hooks:
- `_fit_models(X, Y) -> ModelListGP`
- `_propose_batch(models, X_existing, n_points, boundary_func) -> torch.Tensor`

This module is acquisition-agnostic. No Sobol pool, no exclusion logic, no
gradient or contrast computation lives here.
"""

import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from botorch.models import ModelListGP
from botorch.utils.transforms import normalize
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


class TransitionRecommenderBase:
    """Base class providing the data pipeline and orchestration shared by
    transition-region recommenders. Acquisition-specific behaviour lives in
    `_fit_models` and `_propose_batch`, which subclasses override.
    """

    def __init__(self, input_columns: List[str], output_columns: List[str],
                 log_transform_inputs: bool = True,
                 candidate_pool: int = 50_000,
                 device: str = 'cpu', dtype=torch.double):
        if not (2 <= len(input_columns) <= 5):
            raise ValueError(f"Input dimensions must be 2-5, got {len(input_columns)}")
        if len(output_columns) != 2:
            raise ValueError(f"Must have exactly 2 outputs, got {len(output_columns)}")

        self.input_columns = input_columns
        self.output_columns = output_columns
        self.log_transform_inputs = log_transform_inputs
        self.candidate_pool = candidate_pool
        self.device = torch.device(device)
        self.dtype = dtype

        self.n_inputs = len(input_columns)
        self.n_outputs = len(output_columns)

        # Populated during processing
        self.input_bounds = None
        self.output_scalers = {}
        self.metrics_data = []

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #

    def _fit_models(self, X: torch.Tensor, Y: torch.Tensor) -> ModelListGP:
        raise NotImplementedError("Subclass must implement _fit_models")

    def _propose_batch(self, models: ModelListGP, X_existing: torch.Tensor,
                       n_points: int, boundary_func: callable = None) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement _propose_batch")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_recommendations(self, data_df: pd.DataFrame, n_points: int,
                            iteration: int = None,
                            boundary_func: callable = None,
                            feasibility_fn: callable = None) -> pd.DataFrame:
        """Run the full data -> GP -> acquisition -> recommendations pipeline.

        Parameters
        ----------
        feasibility_fn : callable or None
            Optional vectorized feasibility filter applied to the Sobol
            candidate pool inside _propose_batch.  Must accept an ndarray of
            shape (N_candidates, n_inputs) of concentrations in original units
            (mM) and return a boolean ndarray of shape (N_candidates,).
            Infeasible candidates are removed before acquisition scoring.
        """

        print(f"\n{'='*70}")
        print(f"TRANSITION RECOMMENDER ({self.__class__.__name__})")
        print(f"{'='*70}")

        # Store feasibility function so _propose_batch can access it via self.
        self._feasibility_fn = feasibility_fn

        print(f"\n1. Preparing experimental data...")
        experiment_data = self._prepare_data(data_df)

        print(f"\n2. Processing input variables...")
        X_raw, X_normalized = self._process_inputs(experiment_data)

        print(f"\n3. Processing output variables...")
        Y_raw, Y_standardized = self._process_outputs(experiment_data)

        print(f"\n4. Fitting Gaussian Process models...")
        models = self._fit_models(X_normalized, Y_standardized)

        print(f"\n5. Selecting batch via {self.__class__.__name__}._propose_batch...")
        X_candidates = self._propose_batch(models, X_normalized, n_points, boundary_func)

        print(f"\n6. Converting to original space...")
        recommendations = self._format_recommendations(X_candidates)

        if boundary_func is not None and iteration is not None:
            print(f"\n7. Computing performance metrics...")
            self._compute_metrics(X_candidates, X_normalized, iteration, boundary_func)

        self._print_summary(recommendations, experiment_data)
        return recommendations

    def get_metrics_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics_data)

    # ------------------------------------------------------------------ #
    # Data pipeline (copied verbatim from BayesianTransitionRecommender)
    # ------------------------------------------------------------------ #

    def _to_concentration_space(self, X_normalized: torch.Tensor) -> np.ndarray:
        """Back-transform from normalized [0,1]^N to original concentration
        space (mM).  Uses self.input_bounds set by _process_inputs.
        Returns a numpy array of shape (N_candidates, n_inputs)."""
        X_denorm = (X_normalized
                    * (self.input_bounds[1] - self.input_bounds[0])
                    + self.input_bounds[0])
        if self.log_transform_inputs:
            X_conc = torch.pow(10, X_denorm)
        else:
            X_conc = X_denorm
        return X_conc.detach().cpu().numpy()

    def _prepare_data(self, data_df: pd.DataFrame) -> pd.DataFrame:
        if 'well_type' in data_df.columns:
            experiment_data = data_df[data_df['well_type'] == 'experiment'].copy()
            print(f"  Filtered to {len(experiment_data)} experimental points")
        else:
            experiment_data = data_df.copy()
            print(f"  Using all {len(experiment_data)} data points")

        missing_inputs = [c for c in self.input_columns if c not in experiment_data.columns]
        missing_outputs = [c for c in self.output_columns if c not in experiment_data.columns]
        if missing_inputs:
            raise ValueError(f"Missing input columns: {missing_inputs}")
        if missing_outputs:
            raise ValueError(f"Missing output columns: {missing_outputs}")

        initial_len = len(experiment_data)
        experiment_data = experiment_data.dropna(
            subset=self.input_columns + self.output_columns)
        if len(experiment_data) < initial_len:
            print(f"  Removed {initial_len - len(experiment_data)} rows with NaN values")

        for col in self.input_columns:
            print(f"  {col}: {experiment_data[col].min():.4e} - {experiment_data[col].max():.4e}")
        for col in self.output_columns:
            print(f"  {col}: {experiment_data[col].min():.4f} - {experiment_data[col].max():.4f}")
        return experiment_data

    def _process_inputs(self, experiment_data: pd.DataFrame
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        X_raw = experiment_data[self.input_columns].values.astype(np.float64)
        if self.log_transform_inputs:
            X_transformed = np.log10(X_raw)
            print(f"  Applied log10 transform:")
            for i, col in enumerate(self.input_columns):
                print(f"    {col}: [{X_transformed[:, i].min():.3f}, "
                      f"{X_transformed[:, i].max():.3f}]")
        else:
            X_transformed = X_raw
            print(f"  Using original input space")

        self.input_bounds = torch.tensor(
            [[X_transformed[:, i].min(), X_transformed[:, i].max()]
             for i in range(self.n_inputs)],
            dtype=self.dtype, device=self.device).T  # (2, n_inputs)

        X_torch = torch.tensor(X_transformed, dtype=self.dtype, device=self.device)
        X_normalized = normalize(X_torch, bounds=self.input_bounds)
        print(f"  Normalized to [0,1]^{self.n_inputs}: {X_normalized.shape}")
        return X_torch, X_normalized

    def _process_outputs(self, experiment_data: pd.DataFrame
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        Y_raw = experiment_data[self.output_columns].values.astype(np.float64)
        Y_standardized = np.zeros_like(Y_raw)
        for i, col in enumerate(self.output_columns):
            scaler = StandardScaler()
            Y_standardized[:, i] = scaler.fit_transform(
                Y_raw[:, i].reshape(-1, 1)).flatten()
            self.output_scalers[col] = scaler
            print(f"  {col}: [{Y_raw[:, i].min():.4f}, {Y_raw[:, i].max():.4f}] -> "
                  f"standardized: [{Y_standardized[:, i].min():.3f}, "
                  f"{Y_standardized[:, i].max():.3f}]")
        return (torch.tensor(Y_raw, dtype=self.dtype, device=self.device),
                torch.tensor(Y_standardized, dtype=self.dtype, device=self.device))

    def _format_recommendations(self, X_candidates: torch.Tensor) -> pd.DataFrame:
        X_denorm = (X_candidates * (self.input_bounds[1] - self.input_bounds[0])
                    + self.input_bounds[0])
        if self.log_transform_inputs:
            X_original = torch.pow(10, X_denorm)
        else:
            X_original = X_denorm

        recommendations = pd.DataFrame(
            X_original.detach().cpu().numpy(), columns=self.input_columns)
        recommendations['recommendation_id'] = range(1, len(recommendations) + 1)
        recommendations['method'] = self.__class__.__name__
        recommendations['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        return recommendations

    def _print_summary(self, recommendations: pd.DataFrame,
                       experiment_data: pd.DataFrame):
        print(f"\n{'='*50}")
        print(f"RECOMMENDATIONS SUMMARY ({self.__class__.__name__})")
        print(f"{'='*50}")
        print(f"Generated {len(recommendations)} recommendations")
        print(f"Based on {len(experiment_data)} experimental points")
        print(f"Input space: {self.n_inputs}D, Output space: {self.n_outputs}D")
        print(f"\nRecommended ranges:")
        for col in self.input_columns:
            print(f"  {col}: {recommendations[col].min():.4e} - "
                  f"{recommendations[col].max():.4e}")

    # ------------------------------------------------------------------ #
    # Metrics + diagnostics (copied verbatim from BayesianTransitionRecommender)
    # ------------------------------------------------------------------ #

    def _compute_metrics(self, X_new: torch.Tensor, X_all: torch.Tensor,
                         iteration: int, boundary_func: callable):
        """Cumulative boundary-exploration metrics."""
        X_new_np = X_new.detach().cpu().numpy()
        X_all_np = X_all.detach().cpu().numpy()
        h_all = np.array([boundary_func(x) for x in X_all_np])

        func_name = boundary_func.__name__
        if 'ellipse' in func_name:
            eps_near, eps_far = 0.3, 1.0
        elif 'saddle' in func_name:
            eps_near, eps_far = 0.05, 0.15
        elif 'spiral' in func_name:
            eps_near, eps_far = 0.1, 0.3
        elif 'circle' in func_name or 'step' in func_name:
            eps_near, eps_far = 0.05, 0.15
        else:
            eps_near, eps_far = 0.1, 0.3
            print(f"  Warning: unknown boundary function {func_name}, "
                  f"using default thresholds")

        near_mask = np.abs(h_all) < eps_near
        far_mask = np.abs(h_all) > eps_far
        frac_near = float(np.mean(near_mask))
        frac_far = float(np.mean(far_mask))

        nn_spacing = np.nan
        n_near = int(np.sum(near_mask))
        if n_near >= 2:
            from scipy.spatial import cKDTree
            tree = cKDTree(X_all_np[near_mask])
            nn_dists, _ = tree.query(X_all_np[near_mask], k=2)
            nn_spacing = float(np.median(nn_dists[:, 1]))

        min_batch_dist = np.nan
        if len(X_new) > 1:
            from scipy.spatial.distance import pdist
            min_batch_dist = float(np.min(pdist(X_new_np)))

        metrics = {
            'iteration': iteration,
            'n_total': len(X_all),
            'n_near': n_near,
            'frac_near_cumulative': frac_near,
            'frac_far_cumulative': frac_far,
            'nn_spacing_cumulative': nn_spacing,
            'min_batch_dist': min_batch_dist,
            'eps_near': eps_near,
            'eps_far': eps_far,
        }
        self.metrics_data.append(metrics)
        print(f"  Metrics | iter {iteration} | near={frac_near:.3f} | "
              f"far={frac_far:.3f} | spacing={nn_spacing:.3f} | "
              f"batch_dist={min_batch_dist:.3f} | n_near={n_near}")

    def plot_diagnostics(self, out_dir: str = None, show: bool = True):
        if not self.metrics_data:
            print("No metrics data available for plotting")
            return
        df = self.get_metrics_df()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        ax1.plot(df['iteration'], 100*df['frac_near_cumulative'], 'g-o',
                 label='Boundary hits (%)', linewidth=2)
        ax1.plot(df['iteration'], 100*df['frac_far_cumulative'], 'r-o',
                 label='Waste (%)', linewidth=2)
        ax1.set_xlabel('Iteration'); ax1.set_ylabel('Percentage')
        ax1.set_title('Boundary Targeting Performance')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(df['iteration'], df['n_total'], 'b-o', label='Total points')
        ax2.plot(df['iteration'], df['n_near'], 'g-s', label='Near boundary')
        ax2.set_xlabel('Iteration'); ax2.set_ylabel('Count')
        ax2.set_title('Point Accumulation')
        ax2.legend(); ax2.grid(True, alpha=0.3)

        valid = df['nn_spacing_cumulative'].dropna()
        if len(valid) > 0:
            ax3.plot(valid.index, valid.values, 'purple', marker='o',
                     label='NN spacing')
        ax3.plot(df['iteration'], df['min_batch_dist'], 'orange', marker='s',
                 label='Min batch dist')
        ax3.set_xlabel('Iteration'); ax3.set_ylabel('Distance')
        ax3.set_title('Point Spacing Quality')
        ax3.legend(); ax3.grid(True, alpha=0.3)

        ax4.axhline(y=df['eps_near'].iloc[0], color='green', linestyle='--',
                    label=f"eps_near={df['eps_near'].iloc[0]:.3f}")
        ax4.axhline(y=df['eps_far'].iloc[0], color='red', linestyle='--',
                    label=f"eps_far={df['eps_far'].iloc[0]:.3f}")
        ax4.set_title('Boundary Distance Thresholds Used')
        ax4.legend(); ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if out_dir:
            plt.savefig(os.path.join(out_dir, 'boundary_performance_diagnostics.png'),
                        dpi=150)
        if show:
            plt.show()
        plt.close('all')

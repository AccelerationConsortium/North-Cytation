"""
Bayesian Local-Contrast Transition Recommender
==============================================

Boundary-discovery recommender for 2-5D input / 2-output problems.

Algorithm
---------
1. Fit one SingleTaskGP per output (RBF + ARD, BoTorch MLE).
2. Generate a Sobol candidate pool over [0,1]^d.
3. Score each candidate by *local contrast*: the average over K random
   unit-direction perturbations of |mu(x+delta*u) - mu(x)|, summed across
   outputs. High where the GP posterior changes fast = boundary.
4. Greedy batch selection with a soft min-distance penalty:
       score *= sigmoid((min_dist_to_existing - target) / scale)
   Re-computed against the growing batch after every pick. Soft penalty
   means picks gracefully spread without hard cutoffs.

The spacing target adapts to the actual point density (median nearest-
neighbor distance of historical points), inspired by the deployed
Delaunay triangle recommender.
"""

from typing import List

import numpy as np
import torch
import gpytorch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples

from recommenders._transition_base import TransitionRecommenderBase


# Defaults
DELTA = 0.03                # finite-difference step for contrast (normalized)
K = 24                      # # random directions for contrast
ALPHA_SPACING = 0.7         # spacing factor (target = ALPHA * (1/n_total)^(1/d))
DISTANCE_SCALE = 0.4        # sigmoid steepness as a fraction of target
MIN_TARGET = 0.02           # floor for spacing target
MAX_TARGET = 0.25           # ceiling for spacing target
CANDIDATE_POOL = 50_000


class BayesianTransitionRecommender(TransitionRecommenderBase):
    """Local-contrast acquisition with soft-distance greedy batch selection."""

    def __init__(self, input_columns: List[str], output_columns: List[str],
                 log_transform_inputs: bool = True,
                 delta: float = DELTA, K: int = K,
                 alpha_spacing: float = ALPHA_SPACING,
                 distance_scale: float = DISTANCE_SCALE,
                 candidate_pool: int = CANDIDATE_POOL,
                 explore_beta: float = 0.0,
                 output_normalization: str = 'log_zscore',
                 device: str = 'cpu', dtype=torch.double):
        """
        Parameters
        ----------
        delta : float
            Step size (in normalized [0,1]^d) for the directional contrast
            finite difference.
        K : int
            Number of random unit directions per candidate.
        alpha_spacing : float
            Soft-distance target = alpha_spacing * (1/n_total)^(1/d),
            clamped to [MIN_TARGET, MAX_TARGET]. n_total = n_existing +
            n_batch. Smaller -> tighter packing allowed; larger -> picks
            forced further apart.
        distance_scale : float
            Sigmoid steepness for the distance penalty, expressed as a
            fraction of the target distance. Lower = sharper penalty.
        explore_beta : float
            Optional UCB-style exploration bonus on the *contrast itself*.
            When > 0 the acquisition becomes
                acq = mean_k(|d_k| + explore_beta * sqrt(Var(d_k)))
            where d_k = mu(x + delta*u_k) - mu(x). Set to 0 (default) for
            the original pure-mean contrast. A common UCB value is 1.0.
        output_normalization : str
            How to normalise outputs before GP fitting and contrast scoring.
            'log_zscore' (default): log10-transform then z-score. Prevents
                wide-range outputs (turbidity) from dominating tighter ones
                (ratio). Matches the original 2D workflow.
            'zscore': z-score only (no log).
        """
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            log_transform_inputs=log_transform_inputs,
            candidate_pool=candidate_pool,
            device=device,
            dtype=dtype,
        )
        self.delta = float(delta)
        self.K = int(K)
        self.alpha_spacing = float(alpha_spacing)
        self.distance_scale = float(distance_scale)
        self.explore_beta = float(explore_beta)
        self.output_normalization = str(output_normalization)

        print(f"Initialized BayesianTransitionRecommender (local-contrast):")
        print(f"  Inputs ({self.n_inputs}D): {input_columns}")
        print(f"  Outputs ({self.n_outputs}): {output_columns}")
        print(f"  log_transform_inputs={log_transform_inputs}")
        print(f"  output_normalization={output_normalization}")
        print(f"  contrast: delta={delta:.3f}, K={K}")
        print(f"  explore_beta={self.explore_beta} "
              f"({'UCB on contrast' if self.explore_beta > 0 else 'pure mean'})")
        print(f"  spacing: alpha={alpha_spacing:.2f} * (1/n_total)^(1/d), "
              f"sigmoid_scale={distance_scale:.2f} * target")
        print(f"  candidate_pool={candidate_pool:,}, device={self.device}")

    # ------------------------------------------------------------------ #
    # NaN-tolerant data pipeline overrides
    # Mirrors the Delaunay overrides: points with partial NaN outputs
    # (e.g. ratio masked for high-turbidity wells) stay in X_existing so
    # the spacing penalty still accounts for them, but each GP is fit only
    # on rows where its own output is valid.
    # ------------------------------------------------------------------ #

    def _prepare_data(self, data_df):
        import pandas as pd
        if 'well_type' in data_df.columns:
            experiment_data = data_df[data_df['well_type'] == 'experiment'].copy()
            print(f"  Filtered to {len(experiment_data)} experimental points")
        else:
            experiment_data = data_df.copy()
            print(f"  Using all {len(experiment_data)} data points")

        missing_inputs = [c for c in self.input_columns
                          if c not in experiment_data.columns]
        missing_outputs = [c for c in self.output_columns
                           if c not in experiment_data.columns]
        if missing_inputs:
            raise ValueError(f"Missing input columns: {missing_inputs}")
        if missing_outputs:
            raise ValueError(f"Missing output columns: {missing_outputs}")

        initial_len = len(experiment_data)
        experiment_data = experiment_data.dropna(subset=self.input_columns)
        output_all_nan = experiment_data[self.output_columns].isna().all(axis=1)
        experiment_data = experiment_data[~output_all_nan]
        dropped = initial_len - len(experiment_data)
        if dropped:
            print(f"  Removed {dropped} rows with no valid outputs")
        n_partial = int(
            experiment_data[self.output_columns].isna().any(axis=1).sum())
        if n_partial:
            print(f"  {n_partial} rows have partial NaN outputs "
                  f"(GP fit per output on valid rows only)")

        for col in self.input_columns:
            print(f"  {col}: {experiment_data[col].min():.4e} "
                  f"- {experiment_data[col].max():.4e}")
        for col in self.output_columns:
            n_valid = int(experiment_data[col].notna().sum())
            print(f"  {col}: [{experiment_data[col].min():.4f}, "
                  f"{experiment_data[col].max():.4f}] ({n_valid} valid)")
        return experiment_data

    def _process_outputs(self, experiment_data):
        """NaN-aware output standardisation.

        Behaviour controlled by self.output_normalization:
          'log_zscore' (default): log10-transform then z-score. Prevents
              wide-range outputs (turbidity) from dominating tighter ones
              (ratio) in the GP contrast signal.
          'zscore': z-score only (no log).
        """
        Y_raw = experiment_data[self.output_columns].values.astype(np.float64)
        Y_standardized = np.full_like(Y_raw, np.nan)
        use_log = (self.output_normalization == 'log_zscore')
        for i, col in enumerate(self.output_columns):
            col_vals = Y_raw[:, i]
            valid = ~np.isnan(col_vals)
            if use_log:
                vals_to_scale = np.full_like(col_vals, np.nan)
                vals_to_scale[valid] = np.log10(col_vals[valid] + 1e-6)
            else:
                vals_to_scale = col_vals.copy()
            mean_val = float(np.nanmean(vals_to_scale))
            std_val = float(np.nanstd(vals_to_scale))
            if std_val == 0:
                std_val = 1.0
            Y_standardized[valid, i] = (vals_to_scale[valid] - mean_val) / std_val
            self.output_scalers[col] = (mean_val, std_val)
            n_valid = int(valid.sum())
            label = 'log+zscore' if use_log else 'zscore'
            print(f"  {col}: [{np.nanmin(col_vals):.4f}, {np.nanmax(col_vals):.4f}] "
                  f"-> {label} ({n_valid}/{len(col_vals)} valid): "
                  f"[{np.nanmin(Y_standardized[:, i]):.3f}, "
                  f"{np.nanmax(Y_standardized[:, i]):.3f}]")
        return (torch.tensor(Y_raw, dtype=self.dtype, device=self.device),
                torch.tensor(Y_standardized, dtype=self.dtype, device=self.device))

    # ------------------------------------------------------------------ #
    # GP fitting
    # ------------------------------------------------------------------ #

    def _fit_models(self, X: torch.Tensor, Y: torch.Tensor) -> ModelListGP:
        models = []
        for i, col in enumerate(self.output_columns):
            print(f"  Fitting GP for {col}...")
            # Use only rows where this output is not NaN
            valid_mask = ~torch.isnan(Y[:, i])
            X_fit = X[valid_mask]
            Y_fit = Y[valid_mask, i:i + 1]
            n_valid = int(valid_mask.sum())
            if n_valid < X_fit.shape[1] + 2:
                print(f"    Warning: only {n_valid} valid rows for {col} — skipping GP fit")
                # Fit on dummy data so ModelListGP structure is intact
                model = SingleTaskGP(X_fit[:1], Y_fit[:1])
            else:
                print(f"    Fitting on {n_valid}/{X.shape[0]} valid rows")
                model = SingleTaskGP(X_fit, Y_fit)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                    model.likelihood, model)
                try:
                    fit_gpytorch_mll(mll)
                except Exception as e:
                    print(f"    Warning: fit issue for {col}: {e}")
            models.append(model)
        return ModelListGP(*models)

    # ------------------------------------------------------------------ #
    # Local-contrast acquisition
    # ------------------------------------------------------------------ #

    def _compute_contrast(self, models: ModelListGP,
                          X: torch.Tensor) -> torch.Tensor:
        """Average over K random-direction perturbations of
            score_k = sum_outputs |mu(x+delta*u_k) - mu(x)|
                       + explore_beta * sqrt(Var(mu(x+delta*u_k) - mu(x)))
        Returns (N,) tensor. When explore_beta == 0 the second term is
        skipped entirely (cheaper, identical to the original behavior).
        """
        contrast_per_output, _ = self._compute_contrast_per_output(models, X)
        return contrast_per_output.sum(dim=1)   # (N,)

    def _compute_contrast_per_output(self, models: ModelListGP,
                                     X: torch.Tensor):
        """Same as _compute_contrast but returns per-output contributions.

        Returns
        -------
        contrast_per_output : (N, n_outputs) tensor
            Average |delta_mu| per output, averaged over K directions.
        mean_at_X : (N, n_outputs) tensor
            GP posterior mean at the candidate points (standardised space).
        """
        N = X.shape[0]
        directions = torch.randn(self.K, self.n_inputs,
                                 dtype=self.dtype, device=self.device)
        directions = directions / torch.norm(directions, dim=1, keepdim=True)

        with torch.no_grad():
            mean_base = models.posterior(X).mean  # (N, n_out)

        total_per_out = torch.zeros(N, self.n_outputs,
                                    dtype=self.dtype, device=self.device)
        beta = self.explore_beta
        use_var = beta > 0.0
        for k in range(self.K):
            X_pert = torch.clamp(
                X + self.delta * directions[k:k + 1], 0.0, 1.0)
            with torch.no_grad():
                mean_pert = models.posterior(X_pert).mean
                d_mean = torch.abs(mean_pert - mean_base)     # (N, n_out)

                if use_var:
                    X_pair = torch.stack([X, X_pert], dim=1)
                    for j, m in enumerate(models.models):
                        post_j = m.posterior(X_pair)
                        cov = post_j.mvn.covariance_matrix
                        v_jk = (cov[:, 0, 0] + cov[:, 1, 1]
                                - 2.0 * cov[:, 0, 1]).clamp_min(0.0)
                        d_mean[:, j] = d_mean[:, j] + beta * torch.sqrt(v_jk)

                total_per_out = total_per_out + d_mean

        return total_per_out / self.K, mean_base

    # ------------------------------------------------------------------ #
    # Spacing target (adapts to actual point density)
    # ------------------------------------------------------------------ #

    def _spacing_target(self, X_existing: torch.Tensor,
                        n_batch: int) -> float:
        """target = alpha_spacing * (1 / n_total)^(1/d), the expected
        spacing for n_total points uniformly filling [0,1]^d. Clamped to
        [MIN_TARGET, MAX_TARGET].

        We use the count-based formula instead of median nearest-neighbor
        distance because the latter creates a feedback loop: if the
        recommender clumps, median_nn shrinks, target shrinks, future
        picks can clump even tighter."""
        n_total = max(1, X_existing.shape[0] + n_batch)
        target = self.alpha_spacing * (1.0 / n_total) ** (1.0 / self.n_inputs)
        target = float(np.clip(target, MIN_TARGET, MAX_TARGET))
        print(f"    spacing: n_total={n_total}, d={self.n_inputs}, "
              f"target={target:.4f} (clamped to "
              f"[{MIN_TARGET}, {MAX_TARGET}])")
        return target

    # ------------------------------------------------------------------ #
    # Batch selection
    # ------------------------------------------------------------------ #

    def _propose_batch(self, models: ModelListGP, X_existing: torch.Tensor,
                       n_points: int,
                       boundary_func: callable = None) -> torch.Tensor:
        bounds = torch.tensor([[0.0] * self.n_inputs, [1.0] * self.n_inputs],
                              dtype=self.dtype, device=self.device)

        print(f"  Generating {self.candidate_pool:,} Sobol candidates (persistent)...")
        X_pool = self._draw_sobol_pool(self.candidate_pool)

        # Feasibility filter: remove candidates that violate the physical
        # dispensing constraint (e.g. simplex budget).  Applied before
        # acquisition scoring so the GP never proposes infeasible points.
        feasibility_fn = getattr(self, '_feasibility_fn', None)
        if feasibility_fn is not None:
            X_conc = self._to_concentration_space(X_pool)
            feasible = torch.tensor(
                feasibility_fn(X_conc),
                dtype=torch.bool, device=self.device)
            n_total = X_pool.shape[0]
            X_pool = X_pool[feasible]
            print(f"  Simplex feasibility filter: {X_pool.shape[0]:,}/{n_total:,} "
                  f"candidates kept")
            if X_pool.shape[0] < n_points * 5:
                print(f"  WARNING: Only {X_pool.shape[0]} feasible candidates remain. "
                      f"Consider increasing candidate_pool.")

        print(f"  Computing local contrast on {X_pool.shape[0]:,} "
              f"candidates...")
        contrast_per_output, mean_at_pool = self._compute_contrast_per_output(
            models, X_pool)
        # Normalise per-output contrast by its 95th percentile before summing
        # so no output can dominate by having steeper GP gradients.
        # Mirrors the triangle recommender where L2 within a triangle caps
        # each output's contribution to the same scale.
        p95 = torch.quantile(contrast_per_output, 0.95, dim=0, keepdim=True)  # (1, n_out)
        p95 = torch.clamp(p95, min=1e-12)
        contrast_per_output_norm = contrast_per_output / p95
        contrast = contrast_per_output_norm.sum(dim=1)  # (N,)
        for oi, col in enumerate(self.output_columns):
            print(f"    {col} contrast: p95={p95[0, oi].item():.4e}, "
                  f"range [{contrast_per_output[:, oi].min():.4e}, "
                  f"{contrast_per_output[:, oi].max():.4e}]")
        print(f"    normalised total contrast: range [{contrast.min():.4e}, "
              f"{contrast.max():.4e}], median {contrast.median():.4e}")

        target = self._spacing_target(X_existing, n_points)
        scale = max(self.distance_scale * target, 1e-12)

        # Initial distance to existing experimental points
        if X_existing.shape[0] > 0:
            min_d = torch.cdist(X_pool, X_existing).min(dim=1).values
        else:
            min_d = torch.full((X_pool.shape[0],), float('inf'),
                               dtype=self.dtype, device=self.device)

        weights = torch.sigmoid((min_d - target) / scale)
        turb_penalty = self._turbidity_penalty(
            models, X_pool,
            threshold=getattr(self, 'turbidity_penalty_threshold', 0.2),
            decay=getattr(self, 'turbidity_penalty_decay', 0.15),
        )
        score = contrast * weights * turb_penalty

        # Greedy with re-computed distances after each pick
        selected = []
        chosen_X = []
        for i in range(n_points):
            idx = int(torch.argmax(score).item())
            x_star = X_pool[idx]
            selected.append(idx)
            chosen_X.append(x_star)
            print(f"    pick {i+1}/{n_points}: contrast="
                  f"{contrast[idx].item():.4e}, "
                  f"weight={weights[idx].item():.3f}, "
                  f"min_d={min_d[idx].item():.4f}")

            # Update min_d against newest pick
            d_new = torch.norm(X_pool - x_star, dim=1)
            min_d = torch.minimum(min_d, d_new)
            weights = torch.sigmoid((min_d - target) / scale)
            score = contrast * weights
            score[selected] = float('-inf')  # never re-pick

        X_selected = torch.stack(chosen_X)

        # Save state for workflow-side visualization (any dimensionality).
        b_min = self.input_bounds[0].cpu().numpy()
        b_max = self.input_bounds[1].cpu().numpy()
        self._viz_state = dict(
            X_pool=X_pool.detach().cpu().numpy(),
            contrast_per_output=contrast_per_output.detach().cpu().numpy(),
            contrast_total=contrast.detach().cpu().numpy(),
            mean_at_pool=mean_at_pool.detach().cpu().numpy(),
            X_existing=X_existing.detach().cpu().numpy(),
            X_selected=X_selected.detach().cpu().numpy(),
            b_min=b_min,
            b_max=b_max,
            input_columns=self.input_columns,
            output_columns=self.output_columns,
        )

        return X_selected

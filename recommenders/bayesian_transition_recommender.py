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

        print(f"Initialized BayesianTransitionRecommender (local-contrast):")
        print(f"  Inputs ({self.n_inputs}D): {input_columns}")
        print(f"  Outputs ({self.n_outputs}): {output_columns}")
        print(f"  log_transform_inputs={log_transform_inputs}")
        print(f"  contrast: delta={delta:.3f}, K={K}")
        print(f"  explore_beta={self.explore_beta} "
              f"({'UCB on contrast' if self.explore_beta > 0 else 'pure mean'})")
        print(f"  spacing: alpha={alpha_spacing:.2f} * (1/n_total)^(1/d), "
              f"sigmoid_scale={distance_scale:.2f} * target")
        print(f"  candidate_pool={candidate_pool:,}, device={self.device}")

    # ------------------------------------------------------------------ #
    # GP fitting
    # ------------------------------------------------------------------ #

    def _fit_models(self, X: torch.Tensor, Y: torch.Tensor) -> ModelListGP:
        models = []
        for i, col in enumerate(self.output_columns):
            print(f"  Fitting GP for {col}...")
            model = SingleTaskGP(X, Y[:, i:i + 1])
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
        N = X.shape[0]
        directions = torch.randn(self.K, self.n_inputs,
                                 dtype=self.dtype, device=self.device)
        directions = directions / torch.norm(directions, dim=1, keepdim=True)

        with torch.no_grad():
            mean_base = models.posterior(X).mean  # (N, n_out)

        total = torch.zeros(N, dtype=self.dtype, device=self.device)
        beta = self.explore_beta
        use_var = beta > 0.0
        for k in range(self.K):
            X_pert = torch.clamp(
                X + self.delta * directions[k:k + 1], 0.0, 1.0)
            with torch.no_grad():
                mean_pert = models.posterior(X_pert).mean
                d_mean = mean_pert - mean_base                # (N, n_out)
                step = torch.abs(d_mean).sum(dim=1)           # (N,)

                if use_var:
                    # Variance of (mu(x+du) - mu(x)) per output, using the
                    # joint posterior so the covariance between paired
                    # points is included. ModelListGP with q=2 inputs:
                    #   X_pair: (N, 2, d) -> per-output joint posterior
                    X_pair = torch.stack([X, X_pert], dim=1)  # (N, 2, d)
                    var_diff = torch.zeros(N, dtype=self.dtype,
                                           device=self.device)
                    for j, m in enumerate(models.models):
                        post_j = m.posterior(X_pair)
                        cov = post_j.mvn.covariance_matrix    # (N, 2, 2)
                        v_jk = (cov[:, 0, 0] + cov[:, 1, 1]
                                - 2.0 * cov[:, 0, 1]).clamp_min(0.0)
                        var_diff = var_diff + v_jk
                    step = step + beta * torch.sqrt(var_diff)

                total = total + step
        return total / self.K

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

        print(f"  Generating {self.candidate_pool:,} Sobol candidates...")
        X_pool = draw_sobol_samples(
            bounds=bounds, n=self.candidate_pool, q=1).squeeze(1)

        print(f"  Computing local contrast on {self.candidate_pool:,} "
              f"candidates...")
        contrast = self._compute_contrast(models, X_pool)
        print(f"    contrast: range [{contrast.min():.4e}, "
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
        score = contrast * weights

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

        return torch.stack(chosen_X)

"""
Level-Set Tracing Transition Recommender
========================================

Boundary-discovery recommender that explicitly samples along level-set
contours, rather than just where |grad y| is large. Designed to capture
narrow and tapering features (e.g. the tip of a sharp spike) that
gradient-magnitude / contrast acquisitions tend to miss.

Algorithm
---------
1. Fit one SingleTaskGP per output (RBF + ARD, BoTorch MLE).
2. For each output, choose a small bank of LEVELS spread over the
   observed value range (excluding the saturated extremes).
3. Generate a Sobol candidate pool over [0,1]^d.
4. For each candidate, compute the Bryan-Schneider STRADDLE score per
   output / level:
       straddle(x, c) = 1.96 * std(x) - |mu(x) - c|
   then take per-output max over the level bank, then sum across
   outputs. High where the GP confidence interval straddles a level
   crossing -> point is on (or near) a contour AND uncertain.
5. Greedy batch selection with the same soft min-distance penalty used
   by the other recommenders, so spacing behavior is comparable.

Reference: Bryan, B. and Schneider, J. (2008). Actively learning
level-sets of composite functions. ICML.
"""

from typing import List

import numpy as np
import torch
import gpytorch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples

from recommenders._transition_base import TransitionRecommenderBase


# Defaults (match the other two recommenders for the spacing piece)
N_LEVELS = 5                # # contour levels per output
LEVEL_LOW_Q = 0.10          # don't put levels in the saturated tails
LEVEL_HIGH_Q = 0.90
BETA = 1.96                 # straddle confidence multiplier
ALPHA_SPACING = 0.7
DISTANCE_SCALE = 0.4
MIN_TARGET = 0.02
MAX_TARGET = 0.25
CANDIDATE_POOL = 50_000


class LevelSetTransitionRecommender(TransitionRecommenderBase):
    """Multi-level straddle acquisition with soft-distance greedy batch."""

    def __init__(self, input_columns: List[str], output_columns: List[str],
                 log_transform_inputs: bool = True,
                 n_levels: int = N_LEVELS, beta: float = BETA,
                 level_low_q: float = LEVEL_LOW_Q,
                 level_high_q: float = LEVEL_HIGH_Q,
                 alpha_spacing: float = ALPHA_SPACING,
                 distance_scale: float = DISTANCE_SCALE,
                 candidate_pool: int = CANDIDATE_POOL,
                 device: str = 'cpu', dtype=torch.double):
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            log_transform_inputs=log_transform_inputs,
            candidate_pool=candidate_pool,
            device=device,
            dtype=dtype,
        )
        self.n_levels = int(n_levels)
        self.beta = float(beta)
        self.level_low_q = float(level_low_q)
        self.level_high_q = float(level_high_q)
        self.alpha_spacing = float(alpha_spacing)
        self.distance_scale = float(distance_scale)

        print(f"Initialized LevelSetTransitionRecommender (straddle):")
        print(f"  Inputs ({self.n_inputs}D): {input_columns}")
        print(f"  Outputs ({self.n_outputs}): {output_columns}")
        print(f"  log_transform_inputs={log_transform_inputs}")
        print(f"  n_levels={n_levels}, beta={beta}, "
              f"level quantiles=[{level_low_q}, {level_high_q}]")
        print(f"  spacing: alpha={alpha_spacing:.2f} * (1/n_total)^(1/d), "
              f"sigmoid_scale={distance_scale:.2f} * target")
        print(f"  candidate_pool={candidate_pool:,}, device={self.device}")

    # -------------------------------------------------------------- #
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

    # -------------------------------------------------------------- #
    def _spacing_target(self, X_existing: torch.Tensor,
                        n_batch: int) -> float:
        n_total = max(1, X_existing.shape[0] + n_batch)
        target = self.alpha_spacing * (1.0 / n_total) ** (1.0 / self.n_inputs)
        target = float(np.clip(target, MIN_TARGET, MAX_TARGET))
        print(f"    spacing: n_total={n_total}, d={self.n_inputs}, "
              f"target={target:.4f} (clamped to "
              f"[{MIN_TARGET}, {MAX_TARGET}])")
        return target

    # -------------------------------------------------------------- #
    def _level_bank(self, Y: torch.Tensor) -> torch.Tensor:
        """Pick n_levels evenly spaced levels for each output, between
        the level_low_q and level_high_q quantiles of the observed Y.
        Returns (n_outputs, n_levels) tensor in standardized space.
        """
        # Y here is the standardized training Y (mean 0, var 1 per col)
        n_out = Y.shape[1]
        levels = torch.zeros(n_out, self.n_levels,
                             dtype=self.dtype, device=self.device)
        qs = torch.linspace(self.level_low_q, self.level_high_q,
                            self.n_levels)
        for j in range(n_out):
            yj = Y[:, j]
            vals = torch.tensor(
                np.quantile(yj.detach().cpu().numpy(), qs.numpy()),
                dtype=self.dtype, device=self.device)
            levels[j] = vals
        return levels

    # -------------------------------------------------------------- #
    def _straddle_score(self, models: ModelListGP, X: torch.Tensor,
                        levels: torch.Tensor) -> torch.Tensor:
        """Compute sum_j max_l (beta * std_j(x) - |mu_j(x) - level_jl|).
        Returns (N,) tensor.
        """
        with torch.no_grad():
            post = models.posterior(X)
            mu = post.mean                          # (N, n_out)
            std = post.variance.clamp_min(1e-12).sqrt()
        N = X.shape[0]
        total = torch.zeros(N, dtype=self.dtype, device=self.device)
        for j in range(self.n_outputs):
            mu_j = mu[:, j:j + 1]                   # (N, 1)
            std_j = std[:, j:j + 1]                 # (N, 1)
            lv = levels[j].view(1, -1)              # (1, n_levels)
            straddle_jl = self.beta * std_j - torch.abs(mu_j - lv)
            best = straddle_jl.max(dim=1).values    # (N,)
            total = total + best
        return total

    # -------------------------------------------------------------- #
    def _propose_batch(self, models: ModelListGP, X_existing: torch.Tensor,
                       n_points: int,
                       boundary_func: callable = None) -> torch.Tensor:
        bounds = torch.tensor([[0.0] * self.n_inputs, [1.0] * self.n_inputs],
                              dtype=self.dtype, device=self.device)

        print(f"  Generating {self.candidate_pool:,} Sobol candidates...")
        X_pool = draw_sobol_samples(
            bounds=bounds, n=self.candidate_pool, q=1).squeeze(1)

        # Build level bank from training Y stored on the model
        # (BoTorch standardizes internally only via Outcome transforms; here
        # the GPs see standardized Y because TransitionRecommenderBase
        # standardizes before calling _fit_models, and the same standardized
        # values are stored on each model's train_targets.)
        Y_train = torch.stack(
            [m.train_targets for m in models.models], dim=1)
        levels = self._level_bank(Y_train)
        print(f"  Levels per output (standardized):")
        for j, col in enumerate(self.output_columns):
            print(f"    {col}: "
                  f"{levels[j].cpu().numpy().round(3).tolist()}")

        print(f"  Computing straddle on {self.candidate_pool:,} "
              f"candidates...")
        acq = self._straddle_score(models, X_pool, levels)
        print(f"    straddle: range [{acq.min():.3e}, {acq.max():.3e}], "
              f"median {acq.median():.3e}")

        target = self._spacing_target(X_existing, n_points)
        scale = max(self.distance_scale * target, 1e-12)

        if X_existing.shape[0] > 0:
            min_d = torch.cdist(X_pool, X_existing).min(dim=1).values
        else:
            min_d = torch.full((X_pool.shape[0],), float('inf'),
                               dtype=self.dtype, device=self.device)

        weights = torch.sigmoid((min_d - target) / scale)
        # Straddle can be negative; shift so weighting is meaningful.
        acq_shift = acq - acq.min() + 1e-12
        score = acq_shift * weights

        selected = []
        chosen_X = []
        for i in range(n_points):
            idx = int(torch.argmax(score).item())
            x_star = X_pool[idx]
            selected.append(idx)
            chosen_X.append(x_star)
            print(f"    pick {i+1}/{n_points}: straddle="
                  f"{acq[idx].item():.3e}, "
                  f"weight={weights[idx].item():.3f}, "
                  f"min_d={min_d[idx].item():.4f}")
            d_new = torch.norm(X_pool - x_star, dim=1)
            min_d = torch.minimum(min_d, d_new)
            weights = torch.sigmoid((min_d - target) / scale)
            score = acq_shift * weights
            score[selected] = float('-inf')

        return torch.stack(chosen_X)

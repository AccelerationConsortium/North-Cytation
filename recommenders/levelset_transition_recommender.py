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
BOUNDARY_FOCUS = 1.0        # 0=pure straddle (exploration), 1=pure proximity (always on boundary)
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
                 boundary_focus: float = BOUNDARY_FOCUS,
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
        self.boundary_focus = float(boundary_focus)
        self.alpha_spacing = float(alpha_spacing)
        self.distance_scale = float(distance_scale)
        self.output_normalization = 'log_zscore'

        print(f"Initialized LevelSetTransitionRecommender (straddle):")
        print(f"  Inputs ({self.n_inputs}D): {input_columns}")
        print(f"  Outputs ({self.n_outputs}): {output_columns}")
        print(f"  log_transform_inputs={log_transform_inputs}, output_normalization=log_zscore")
        print(f"  n_levels={n_levels}, beta={beta}, boundary_focus={boundary_focus:.2f}, "
              f"level quantiles=[{level_low_q}, {level_high_q}]")
        print(f"  spacing: alpha={alpha_spacing:.2f} * (1/n_total)^(1/d), "
              f"sigmoid_scale={distance_scale:.2f} * target")
        print(f"  candidate_pool={candidate_pool:,}, device={self.device}")

    # -------------------------------------------------------------- #
    def _prepare_data(self, data_df):
        import pandas as pd
        if 'well_type' in data_df.columns:
            experiment_data = data_df[data_df['well_type'] == 'experiment'].copy()
        else:
            experiment_data = data_df.copy()
        missing_inputs = [c for c in self.input_columns if c not in experiment_data.columns]
        missing_outputs = [c for c in self.output_columns if c not in experiment_data.columns]
        if missing_inputs:
            raise ValueError(f"Missing input columns: {missing_inputs}")
        if missing_outputs:
            raise ValueError(f"Missing output columns: {missing_outputs}")
        experiment_data = experiment_data.dropna(subset=self.input_columns)
        output_all_nan = experiment_data[self.output_columns].isna().all(axis=1)
        experiment_data = experiment_data[~output_all_nan]
        n_partial = int(experiment_data[self.output_columns].isna().any(axis=1).sum())
        if n_partial:
            print(f"  {n_partial} rows have partial NaN outputs (GP fit per output on valid rows only)")
        return experiment_data

    def _process_outputs(self, experiment_data):
        """log+zscore normalisation — prevents wide-range outputs from dominating."""
        Y_raw = experiment_data[self.output_columns].values.astype(np.float64)
        Y_standardized = np.full_like(Y_raw, np.nan)
        for i, col in enumerate(self.output_columns):
            col_vals = Y_raw[:, i]
            valid = ~np.isnan(col_vals)
            log_vals = np.full_like(col_vals, np.nan)
            log_vals[valid] = np.log10(col_vals[valid] + 1e-6)
            mean_val = float(np.nanmean(log_vals))
            std_val = float(np.nanstd(log_vals))
            if std_val == 0:
                std_val = 1.0
            Y_standardized[valid, i] = (log_vals[valid] - mean_val) / std_val
            self.output_scalers[col] = (mean_val, std_val)
            n_valid = int(valid.sum())
            print(f"  {col}: [{np.nanmin(col_vals):.4f}, {np.nanmax(col_vals):.4f}] "
                  f"-> log+zscore ({n_valid}/{len(col_vals)} valid): "
                  f"[{np.nanmin(Y_standardized[:, i]):.3f}, {np.nanmax(Y_standardized[:, i]):.3f}]")
        return (torch.tensor(Y_raw, dtype=self.dtype, device=self.device),
                torch.tensor(Y_standardized, dtype=self.dtype, device=self.device))

    # -------------------------------------------------------------- #
    def _fit_models(self, X: torch.Tensor, Y: torch.Tensor) -> ModelListGP:
        models = []
        for i, col in enumerate(self.output_columns):
            print(f"  Fitting GP for {col}...")
            valid_mask = ~torch.isnan(Y[:, i])
            X_fit = X[valid_mask]
            Y_fit = Y[valid_mask, i:i + 1]
            n_valid = int(valid_mask.sum())
            if n_valid < X_fit.shape[1] + 2:
                print(f"    Warning: only {n_valid} valid rows for {col} — using dummy fit")
                model = SingleTaskGP(X_fit[:1], Y_fit[:1])
                models.append(model)
                continue
            model = SingleTaskGP(X_fit, Y_fit)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"    Warning: fit issue for {col}: {e}")
            models.append(model)
        # Store full Y so _propose_batch can build level bank without
        # reading per-output train_targets (which differ in length when
        # rows have partial NaN outputs).
        self._Y_train = Y
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
                np.nanquantile(yj.detach().cpu().numpy(), qs.numpy()),
                dtype=self.dtype, device=self.device)
            levels[j] = vals
        return levels

    # -------------------------------------------------------------- #
    def _straddle_score(self, models: ModelListGP, X: torch.Tensor,
                        levels: torch.Tensor) -> torch.Tensor:
        """Blended acquisition: straddle (exploration) + level proximity (boundary focus).

        boundary_focus controls the blend:
          0.0 = pure straddle (original behavior: picks uncertain regions)
          1.0 = pure proximity (always picks near the boundary contour)
          0.5 = equal blend (recommended default)

        Level proximity = exp(-|mu - c_nearest| / sigma) where sigma is the
        inter-level spacing.  This is always high at the transition boundary
        regardless of GP confidence, so the algorithm keeps picking there
        even after the boundary is well-characterised.

        Both components are p95-normalised per output before summing.
        Returns (N,) tensor.
        """
        with torch.no_grad():
            post = models.posterior(X)
            mu = post.mean                          # (N, n_out)
            std = post.variance.clamp_min(1e-12).sqrt()
        N = X.shape[0]
        straddle_part = torch.zeros(N, self.n_outputs,
                                    dtype=self.dtype, device=self.device)
        proximity_part = torch.zeros(N, self.n_outputs,
                                     dtype=self.dtype, device=self.device)
        for j in range(self.n_outputs):
            mu_j = mu[:, j:j + 1]                   # (N, 1)
            std_j = std[:, j:j + 1]                 # (N, 1)
            lv = levels[j].view(1, -1)              # (1, n_levels)
            dist_to_levels = torch.abs(mu_j - lv)   # (N, n_levels)
            # Straddle: positive where GP straddles a level (uncertain + close)
            straddle_jl = self.beta * std_j - dist_to_levels
            straddle_part[:, j] = straddle_jl.max(dim=1).values
            # Proximity: always high near the closest level contour
            # sigma = median gap between adjacent levels
            lv_sorted = levels[j].sort().values
            if self.n_levels > 1:
                sigma = (lv_sorted[1:] - lv_sorted[:-1]).mean().clamp(min=1e-6)
            else:
                sigma = torch.tensor(0.5, dtype=self.dtype, device=self.device)
            proximity_part[:, j] = torch.exp(-dist_to_levels.min(dim=1).values / sigma)

        # p95-normalise each component per output before blending
        alpha = self.boundary_focus
        if alpha < 1.0:
            p95_s = torch.quantile(straddle_part, 0.95, dim=0, keepdim=True).clamp(min=1e-12)
            straddle_norm = straddle_part / p95_s
        else:
            straddle_norm = torch.zeros_like(straddle_part)
        if alpha > 0.0:
            p95_p = torch.quantile(proximity_part, 0.95, dim=0, keepdim=True).clamp(min=1e-12)
            proximity_norm = proximity_part / p95_p
        else:
            proximity_norm = torch.zeros_like(proximity_part)

        per_output_blended = (1.0 - alpha) * straddle_norm + alpha * proximity_norm
        return per_output_blended.sum(dim=1)

    # -------------------------------------------------------------- #
    def _propose_batch(self, models: ModelListGP, X_existing: torch.Tensor,
                       n_points: int,
                       boundary_func: callable = None) -> torch.Tensor:
        bounds = torch.tensor([[0.0] * self.n_inputs, [1.0] * self.n_inputs],
                              dtype=self.dtype, device=self.device)

        print(f"  Generating {self.candidate_pool:,} Sobol candidates...")
        X_pool = draw_sobol_samples(
            bounds=bounds, n=self.candidate_pool, q=1).squeeze(1)

        # Feasibility filter: remove candidates that violate the physical
        # dispensing constraint (e.g. simplex budget).
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

        # Build level bank from Y stored during _fit_models.
        # Cannot use model.train_targets here: with partial NaN outputs,
        # each GP is fit on a different number of rows so the tensors have
        # different lengths and cannot be stacked.
        Y_train = self._Y_train
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
        turb_penalty = self._turbidity_penalty(
            models, X_pool,
            threshold=getattr(self, 'turbidity_penalty_threshold', 0.2),
            decay=getattr(self, 'turbidity_penalty_decay', 0.15),
        )
        # Invert: blue (minimum straddle = GP mean on the boundary) is what we want.
        acq_shift = acq.max() - acq + 1e-12
        score = acq_shift * weights * turb_penalty

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

        X_selected = torch.stack(chosen_X)

        # viz_state for workflow-side plotting (any dimensionality)
        b_min = self.input_bounds[0].cpu().numpy()
        b_max = self.input_bounds[1].cpu().numpy()
        with torch.no_grad():
            post = models.posterior(X_pool)
            mu = post.mean
            std = post.variance.clamp_min(1e-12).sqrt()
        levels_viz = self._level_bank(self._Y_train)
        # Show the BLENDED acquisition per output (same as what drives selection),
        # not the raw straddle — so yellow = ON boundary, blue = away from boundary.
        per_output_viz = torch.zeros(X_pool.shape[0], self.n_outputs,
                                     dtype=self.dtype, device=self.device)
        for j in range(self.n_outputs):
            mu_j = mu[:, j:j+1]
            std_j = std[:, j:j+1]
            lv = levels_viz[j].view(1, -1)
            dist = torch.abs(mu_j - lv)               # (N, n_levels)
            # Proximity component: high exactly at the boundary contour
            lv_sorted = levels_viz[j].sort().values
            sigma = ((lv_sorted[1:] - lv_sorted[:-1]).mean().clamp(min=1e-6)
                     if self.n_levels > 1
                     else torch.tensor(0.5, dtype=self.dtype, device=self.device))
            proximity = torch.exp(-dist.min(dim=1).values / sigma)
            # Straddle component: high where GP straddles the level
            straddle = (self.beta * std_j - dist).max(dim=1).values
            alpha = self.boundary_focus
            per_output_viz[:, j] = (1.0 - alpha) * straddle + alpha * proximity

        self._viz_state = dict(
            X_pool=X_pool.detach().cpu().numpy(),
            acq_per_output=(per_output_viz.max(dim=0).values - per_output_viz).detach().cpu().numpy(),
            acq_total=(acq.max() - acq).detach().cpu().numpy(),
            mean_at_pool=mu.detach().cpu().numpy(),
            X_existing=X_existing.detach().cpu().numpy(),
            X_selected=X_selected.detach().cpu().numpy(),
            b_min=b_min, b_max=b_max,
            input_columns=self.input_columns,
            output_columns=self.output_columns,
            acq_label='boundary proximity',
        )

        return X_selected

"""
Gradient-Based Transition Recommender
======================================

Active-learning recommender that picks experiments along sharp transition
boundaries by maximising a UCB on the *derivative* of a Gaussian Process
posterior. Implements the algorithm from:

    Vadhavkar, Bajracharya, Zaman, Padmanabhan, Wang.
    "Gradient-based active learning for intelligent discovery of colloidal
    phase diagrams." Mol. Syst. Des. Eng. 2026 (DOI: 10.1039/d5me00233h).

Algorithm
---------
1. Per-output GP with ARD-RBF kernel (BoTorch SingleTaskGP defaults, MLE).
2. For each candidate x and each input dim j, compute the analytical
   posterior of the partial derivative dmu/dx_j (mean via autograd, variance
   via the closed-form RBF derivative-posterior, paper eqns 9-10).
3. UCB acquisition per (output, dim):
       a_kj(x) = |dmu_k/dx_j| + beta * sqrt(Sigma_grad,jj_k)
   Reduce across outputs (sum / max / standardized_sum).
   Per candidate, take the BEST-dim score: alpha(x) = max_j a_j(x).
4. Greedy batch selection with the SAME soft-distance penalty as
   BayesianTransitionRecommender:
       score *= sigmoid((min_dist_to_existing - target) / scale)
   Spacing target = alpha_spacing * median_nn_dist(existing).

Differences from the paper (intentional)
----------------------------------------
- Multi-output: paper has a single output. We add a reduction step.
- Batch selection: paper uses per-dim greedy + axis-aligned ellipse
  exclusion. We replaced that with soft-distance greedy because
  axis-aligned exclusion fails on diagonal/curved boundaries (clumps
  picks on one ridge direction) and hard exclusion can starve the pool.
"""

from typing import List

import numpy as np
import torch
import gpytorch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples

from recommenders._transition_base import TransitionRecommenderBase


# Defaults shared with BayesianTransitionRecommender for fair comparison
ALPHA_SPACING = 0.7         # spacing factor (target = ALPHA * (1/n_total)^(1/d))
DISTANCE_SCALE = 0.4        # sigmoid steepness as fraction of target
MIN_TARGET = 0.02
MAX_TARGET = 0.25
CANDIDATE_POOL = 50_000


class GradientTransitionRecommender(TransitionRecommenderBase):
    """Per-dim gradient-UCB acquisition with soft-distance batch selection."""

    def __init__(self, input_columns: List[str], output_columns: List[str],
                 log_transform_inputs: bool = True,
                 beta: float = 1.0,
                 multi_output_reduce: str = 'standardized_sum',
                 alpha_spacing: float = ALPHA_SPACING,
                 distance_scale: float = DISTANCE_SCALE,
                 candidate_pool: int = CANDIDATE_POOL,
                 device: str = 'cpu', dtype=torch.double):
        """
        Parameters
        ----------
        beta : float
            UCB weight on the gradient-posterior std term. Paper uses 1.0.
        multi_output_reduce : {'sum', 'max', 'standardized_sum'}
            How to combine per-output acquisition into one (N, d) tensor:
              - 'sum': paper-style; risks one loud output dominating
              - 'max': winner-takes-all per dim
              - 'standardized_sum': normalize each output by its
                90th-percentile pool magnitude, then sum. Best when
                outputs have very different scales.
        alpha_spacing, distance_scale : see BayesianTransitionRecommender.
        """
        if multi_output_reduce not in ('sum', 'max', 'standardized_sum'):
            raise ValueError(
                f"multi_output_reduce must be one of "
                f"'sum'|'max'|'standardized_sum', got "
                f"{multi_output_reduce!r}")
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            log_transform_inputs=log_transform_inputs,
            candidate_pool=candidate_pool,
            device=device,
            dtype=dtype,
        )
        self.beta = float(beta)
        self.multi_output_reduce = multi_output_reduce
        self.alpha_spacing = float(alpha_spacing)
        self.distance_scale = float(distance_scale)
        self.output_normalization = 'log_zscore'

        print(f"Initialized GradientTransitionRecommender (gradient-UCB):")
        print(f"  Inputs ({self.n_inputs}D): {input_columns}")
        print(f"  Outputs ({self.n_outputs}): {output_columns}")
        print(f"  log_transform_inputs={log_transform_inputs}, output_normalization=log_zscore")
        print(f"  acq: max_j {multi_output_reduce}_k(|grad mu_kj| "
              f"+ beta*sqrt(Sigma_grad,jj_k))")
        print(f"  beta={beta}")
        print(f"  spacing: alpha={alpha_spacing:.2f} * (1/n_total)^(1/d), "
              f"sigmoid_scale={distance_scale:.2f} * target")
        print(f"  candidate_pool={candidate_pool:,}, device={self.device}")

    # ------------------------------------------------------------------ #
    # NaN-tolerant data pipeline overrides (same pattern as Bayesian)
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # GP fitting
    # ------------------------------------------------------------------ #

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
            print(f"    Fitting on {n_valid}/{X.shape[0]} valid rows")
            model = SingleTaskGP(X_fit, Y_fit)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
                ls, os_, ns = self._kernel_params(model)
                print(f"    lengthscale={ls.cpu().numpy().flatten()}, "
                      f"outputscale={float(os_):.4f}, noise={float(ns):.4e}")
            except Exception as e:
                print(f"    Warning: fit issue for {col}: {e}")
            models.append(model)
        return ModelListGP(*models)

    def _kernel_params(self, model):
        """Return (lengthscale (d,), outputscale, noise) on this dtype/device.
        Handles both raw RBFKernel and ScaleKernel(RBFKernel) layouts."""
        cov = model.covar_module
        if hasattr(cov, "base_kernel"):
            base = cov.base_kernel
            outputscale = cov.outputscale.detach()
        else:
            base = cov
            outputscale = torch.tensor(1.0, dtype=self.dtype,
                                       device=self.device)
        ls = base.lengthscale.detach().to(self.dtype).to(self.device).view(-1)
        outputscale = outputscale.to(self.dtype).to(self.device)
        noise = model.likelihood.noise.detach().to(
            self.dtype).to(self.device).mean()
        return ls, outputscale, noise

    # ------------------------------------------------------------------ #
    # Posterior gradient mean (autograd) and variance (analytical RBF)
    # ------------------------------------------------------------------ #

    def _grad_mu(self, models: ModelListGP, X: torch.Tensor) -> torch.Tensor:
        """Posterior mean gradient via autograd. Returns (N, n_out, d)."""
        X_req = X.detach().clone().requires_grad_(True)
        posterior = models.posterior(X_req)
        mean = posterior.mean  # (N, n_out)
        grads = []
        for k in range(self.n_outputs):
            (g,) = torch.autograd.grad(
                mean[:, k].sum(), X_req,
                retain_graph=(k < self.n_outputs - 1),
            )
            grads.append(g)
        return torch.stack(grads, dim=1).detach()

    def _grad_var(self, models: ModelListGP, X: torch.Tensor) -> torch.Tensor:
        """Analytical posterior variance of dmu/dx_j per output and dim.
        Paper eqns 9-10. Returns (N, n_out, d), clamped >= 0."""
        N, d = X.shape
        X = X.to(self.dtype).to(self.device)
        all_var = torch.zeros((N, self.n_outputs, d),
                              dtype=self.dtype, device=self.device)

        for k, model in enumerate(models.models):
            X_train = model.train_inputs[0].to(self.dtype).to(self.device)
            n_train = X_train.shape[0]
            ls, outputscale, noise = self._kernel_params(model)

            with torch.no_grad():
                K_train = model.covar_module(X_train).evaluate().to(self.dtype)
            K_noisy = K_train + noise * torch.eye(
                n_train, dtype=self.dtype, device=self.device)

            jitter = 1e-6
            L = None
            for _ in range(5):
                try:
                    L = torch.linalg.cholesky(
                        K_noisy + jitter * torch.eye(
                            n_train, dtype=self.dtype, device=self.device))
                    break
                except Exception:
                    jitter *= 10
            if L is None:
                K_inv = torch.linalg.pinv(K_noisy)

            with torch.no_grad():
                k_star = model.covar_module(X_train, X).evaluate().to(
                    self.dtype)

            diag_term = outputscale / (ls ** 2)  # (d,)

            for j in range(d):
                diff_j = X_train[:, j].unsqueeze(1) - X[:, j].unsqueeze(0)
                dk_star = -(diff_j / (ls[j] ** 2)) * k_star
                if L is not None:
                    K_inv_dk = torch.cholesky_solve(dk_star, L)
                else:
                    K_inv_dk = K_inv @ dk_star
                quad = (dk_star * K_inv_dk).sum(dim=0)
                all_var[:, k, j] = diag_term[j] - quad

        return torch.clamp(all_var, min=0.0)

    # ------------------------------------------------------------------ #
    # Acquisition reduction
    # ------------------------------------------------------------------ #

    def _acquisition(self, grad_mu: torch.Tensor,
                     grad_var: torch.Tensor) -> torch.Tensor:
        """Combine per-(output, dim) acquisition into a single (N,) score
        by reducing across outputs (per self.multi_output_reduce) and then
        taking the best dim per candidate."""
        per_output = (torch.abs(grad_mu)
                      + self.beta * torch.sqrt(grad_var))  # (N, n_out, d)

        mode = self.multi_output_reduce
        if mode == 'sum':
            per_dim = per_output.sum(dim=1)               # (N, d)
        elif mode == 'max':
            per_dim = per_output.max(dim=1).values        # (N, d)
        elif mode == 'standardized_sum':
            scale = torch.quantile(per_output.sum(dim=2), 0.9, dim=0)
            scale = torch.clamp(scale, min=1e-12)
            per_dim = (per_output / scale.view(1, -1, 1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown reduce mode: {mode!r}")

        # Best dim per candidate -> single (N,) score
        return per_dim.max(dim=1).values

    # ------------------------------------------------------------------ #
    # Spacing target (same as BayesianTransitionRecommender)
    # ------------------------------------------------------------------ #

    def _spacing_target(self, X_existing: torch.Tensor,
                        n_batch: int) -> float:
        """target = alpha_spacing * (1 / n_total)^(1/d). See
        BayesianTransitionRecommender._spacing_target for rationale."""
        n_total = max(1, X_existing.shape[0] + n_batch)
        target = self.alpha_spacing * (1.0 / n_total) ** (1.0 / self.n_inputs)
        target = float(np.clip(target, MIN_TARGET, MAX_TARGET))
        print(f"    spacing: n_total={n_total}, d={self.n_inputs}, "
              f"target={target:.4f} (clamped to "
              f"[{MIN_TARGET}, {MAX_TARGET}])")
        return target

    # ------------------------------------------------------------------ #
    # Batch selection (soft-distance greedy)
    # ------------------------------------------------------------------ #

    def _propose_batch(self, models: ModelListGP, X_existing: torch.Tensor,
                       n_points: int,
                       boundary_func: callable = None) -> torch.Tensor:
        bounds = torch.tensor([[0.0] * self.n_inputs, [1.0] * self.n_inputs],
                              dtype=self.dtype, device=self.device)

        print(f"  Generating {self.candidate_pool:,} Sobol candidates (persistent)...")
        X_pool = self._draw_sobol_pool(self.candidate_pool)

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

        print(f"  Computing posterior gradient mean (autograd)...")
        grad_mu = self._grad_mu(models, X_pool)
        print(f"  Computing posterior gradient variance (analytical)...")
        grad_var = self._grad_var(models, X_pool)
        score_raw = self._acquisition(grad_mu, grad_var)
        print(f"    acq: range [{score_raw.min():.4e}, "
              f"{score_raw.max():.4e}], median {score_raw.median():.4e}")

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
        score = score_raw * weights * turb_penalty

        selected = []
        chosen_X = []
        for i in range(n_points):
            idx = int(torch.argmax(score).item())
            x_star = X_pool[idx]
            selected.append(idx)
            chosen_X.append(x_star)
            print(f"    pick {i+1}/{n_points}: acq="
                  f"{score_raw[idx].item():.4e}, "
                  f"weight={weights[idx].item():.3f}, "
                  f"min_d={min_d[idx].item():.4f}")

            d_new = torch.norm(X_pool - x_star, dim=1)
            min_d = torch.minimum(min_d, d_new)
            weights = torch.sigmoid((min_d - target) / scale)
            score = score_raw * weights
            score[selected] = float('-inf')

        X_selected = torch.stack(chosen_X)

        # viz_state for workflow-side plotting (any dimensionality)
        b_min = self.input_bounds[0].cpu().numpy()
        b_max = self.input_bounds[1].cpu().numpy()
        # Per-output gradient magnitude: max over dims of |grad_mu| (N, n_out)
        grad_mag_per_output = torch.abs(grad_mu).max(dim=2).values  # (N, n_out)
        self._viz_state = dict(
            X_pool=X_pool.detach().cpu().numpy(),
            acq_per_output=grad_mag_per_output.detach().cpu().numpy(),
            acq_total=score_raw.detach().cpu().numpy(),
            X_existing=X_existing.detach().cpu().numpy(),
            X_selected=X_selected.detach().cpu().numpy(),
            b_min=b_min, b_max=b_max,
            input_columns=self.input_columns,
            output_columns=self.output_columns,
            acq_label='gradient-UCB magnitude',
        )

        return X_selected

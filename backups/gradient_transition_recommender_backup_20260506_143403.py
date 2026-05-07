"""
Gradient-Based Transition Recommender
======================================

Active-learning recommender that picks experiments along sharp transition
boundaries by maximising a per-dimension UCB on the *derivative* of a
Gaussian Process posterior. Implements the algorithm from:

    Vadhavkar, Bajracharya, Zaman, Padmanabhan, Wang.
    "Gradient-based active learning for intelligent discovery of colloidal
    phase diagrams." Mol. Syst. Des. Eng. 2026 (DOI: 10.1039/d5me00233h).

Key ideas
---------
1. Per-output GP with ARD-RBF kernel (BoTorch SingleTaskGP defaults, fit by MLE).
2. For each candidate x and each input dimension j, compute the analytical
   posterior of the partial derivative dmu/dx_j (mean via autograd, variance
   via the closed-form RBF derivative-posterior, paper eqns 9-10).
3. Per-dimension UCB acquisition:
        alpha_j(x) = sum_k [ |dmu_k/dx_j| + beta * sqrt(Sigma_grad,jj_k) ]
   where the sum is over the n_outputs GPs.
4. Greedy selection: in each sub-round, iterate over dims (random order),
   pick argmax of alpha_j on remaining (non-excluded) candidates, then add
   an axis-aligned anisotropic exclusion ellipse with `small_radius` along
   the chosen dim and `large_radius` along all others. This forces
   subsequent picks to spread along the boundary manifold.

Differences from the paper (intentional, v1)
--------------------------------------------
- Multi-output: paper has a single output. Here we sum |grad| + beta*std
  contributions across both surfactant outputs (ratio, turbidity), per dim.
- Adaptive lengthscale: not implemented in v1. BoTorch MLE is used as-is.
  Add only if GPs are observed to collapse.
- Stopping criterion: not implemented; runs for a fixed n_points per call.

Usage
-----
    from recommenders.gradient_transition_recommender import (
        GradientTransitionRecommender,
    )
    rec = GradientTransitionRecommender(
        input_columns=['surf_A_mm', 'surf_B_mm'],
        output_columns=['ratio', 'turbidity_600'],
        log_transform_inputs=True,
    )
    df = rec.get_recommendations(data_df, n_points=14)
"""

from typing import List

import numpy as np
import pandas as pd
import torch
import gpytorch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples

from recommenders._transition_base import TransitionRecommenderBase


class GradientTransitionRecommender(TransitionRecommenderBase):
    """Per-dimension gradient-UCB recommender with axis-aligned ellipse
    exclusion, after Vadhavkar et al. (MSDE 2026)."""

    def __init__(self, input_columns: List[str], output_columns: List[str],
                 log_transform_inputs: bool = True,
                 candidate_pool: int = 50_000,
                 small_radius: float = 0.05,
                 large_radius: float = 0.20,
                 beta: float = 1.0,
                 multi_output_reduce: str = 'sum',
                 device: str = 'cpu', dtype=torch.double):
        """
        Parameters
        ----------
        input_columns, output_columns, log_transform_inputs, candidate_pool,
        device, dtype : see TransitionRecommenderBase.
        small_radius : float
            Exclusion ellipse radius along the *selected* input dimension
            (in normalised [0,1] space). Smaller -> picks can be tighter
            across the boundary.
        large_radius : float
            Exclusion ellipse radius along all *other* input dimensions.
            Larger -> spreads picks further along the boundary tangent.
        beta : float
            UCB weight on the gradient-posterior std term. Paper uses 1.0.
        multi_output_reduce : {'sum', 'max', 'standardized_sum'}
            How to combine the per-output acquisition scores into a single
            per-dim score:
              - 'sum' : alpha_j = sum_k (|grad mu_kj| + beta*sigma_kj)
                        (paper-style for single output; risks one loud output
                         dominating when outputs have very different scales)
              - 'max' : alpha_j = max_k (|grad mu_kj| + beta*sigma_kj)
                        (winner-takes-all; a sharp gradient in EITHER output
                         is enough to attract picks)
              - 'standardized_sum' : divide each output's per-pool acquisition
                        by its 90th-percentile across the candidate pool, then
                        sum. Equalizes per-output influence regardless of raw
                        magnitude.
        """
        if multi_output_reduce not in ('sum', 'max', 'standardized_sum'):
            raise ValueError(
                f"multi_output_reduce must be one of "
                f"'sum'|'max'|'standardized_sum', got {multi_output_reduce!r}")
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            log_transform_inputs=log_transform_inputs,
            candidate_pool=candidate_pool,
            device=device,
            dtype=dtype,
        )
        self.small_radius = float(small_radius)
        self.large_radius = float(large_radius)
        self.beta = float(beta)
        self.multi_output_reduce = multi_output_reduce
        self.exclusion_regions = []  # reset per _propose_batch call

        print(f"Initialized GradientTransitionRecommender (per-dim grad UCB):")
        print(f"  Input variables ({self.n_inputs}D): {input_columns}")
        print(f"  Output variables ({self.n_outputs}D): {output_columns}")
        print(f"  Log transform inputs: {log_transform_inputs}")
        print(f"  Acquisition: alpha_j = {multi_output_reduce}_k(|grad_j mu_k| + beta*sqrt(Sigma_grad,jj_k))")
        print(f"  beta = {self.beta}")
        print(f"  Exclusion radii (normalised space): "
              f"small={small_radius:.3f}, large={large_radius:.3f}")
        print(f"  Candidate pool: {candidate_pool:,}")
        print(f"  Device: {self.device}, dtype: {dtype}")

    # ------------------------------------------------------------------ #
    # GP fitting
    # ------------------------------------------------------------------ #

    def _fit_models(self, X: torch.Tensor, Y: torch.Tensor) -> ModelListGP:
        """Fit one SingleTaskGP per output (RBF + ARD, GaussianLikelihood,
        BoTorch MLE for hyperparameters)."""
        models = []
        for i, col in enumerate(self.output_columns):
            print(f"  Fitting GP for {col}...")
            model = SingleTaskGP(X, Y[:, i:i+1])
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
                ls, os_, ns = self._kernel_params(model)
                print(f"    Fitted: lengthscale={ls.cpu().numpy().flatten()}, "
                      f"outputscale={float(os_):.4f}, noise={float(ns):.4e}")
            except Exception as e:
                print(f"    Warning: fit issues for {col}: {e}")
            models.append(model)
        return ModelListGP(*models)

    def _kernel_params(self, model):
        """Return (lengthscale (d,), outputscale scalar, noise scalar) tensors
        on this recommender's dtype/device. Handles both raw RBFKernel and
        ScaleKernel(RBFKernel) layouts.
        """
        cov = model.covar_module
        if hasattr(cov, "base_kernel"):
            base = cov.base_kernel
            outputscale = cov.outputscale.detach()
        else:
            base = cov
            outputscale = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        ls = base.lengthscale.detach().to(self.dtype).to(self.device).view(-1)
        outputscale = outputscale.to(self.dtype).to(self.device)
        noise = model.likelihood.noise.detach().to(
            self.dtype).to(self.device).mean()
        return ls, outputscale, noise

    # ------------------------------------------------------------------ #
    # Gradient mean (autograd)
    # ------------------------------------------------------------------ #

    def _grad_mu(self, models: ModelListGP, X: torch.Tensor) -> torch.Tensor:
        """Posterior mean gradient via autograd.

        Parameters
        ----------
        X : (N, d) tensor in normalised [0,1] space.

        Returns
        -------
        grad_mu : (N, n_outputs, d) tensor.
        """
        X_req = X.detach().clone().requires_grad_(True)
        posterior = models.posterior(X_req)
        mean = posterior.mean  # (N, n_outputs)
        grads = []
        for k in range(self.n_outputs):
            (g,) = torch.autograd.grad(
                mean[:, k].sum(), X_req,
                retain_graph=(k < self.n_outputs - 1),
            )
            grads.append(g)
        return torch.stack(grads, dim=1).detach()  # (N, n_outputs, d)

    # ------------------------------------------------------------------ #
    # Gradient variance (analytical RBF derivative posterior)
    # ------------------------------------------------------------------ #

    def _grad_var(self, models: ModelListGP, X: torch.Tensor) -> torch.Tensor:
        """Analytical posterior variance of dmu/dx_j for each output and dim.

        Implements paper eqns 9-10. For RBF kernel with ARD lengthscales l_j,
        outputscale s_f^2, noise s_n^2, training inputs X_train (n, d):

            K       = k(X_train, X_train) + s_n^2 I        (n, n)
            k_*     = k(X_train, x*)                       (n,)
            dk_*/dx*_j = -((X_train[:, j] - x*[j]) / l_j^2) * k_*
            d2k(x*,x*)/dx*_j^2 = s_f^2 / l_j^2             (zero-distance)
            Sigma_grad,jj(x*)  = s_f^2 / l_j^2
                                 - (dk_*/dx*_j)^T K^-1 (dk_*/dx*_j)

        Parameters
        ----------
        X : (N, d) tensor in normalised [0,1] space.

        Returns
        -------
        grad_var : (N, n_outputs, d) tensor, clamped >= 0.
        """
        N, d = X.shape
        X = X.to(self.dtype).to(self.device)
        all_var = torch.zeros((N, self.n_outputs, d),
                              dtype=self.dtype, device=self.device)

        for k, model in enumerate(models.models):
            X_train = model.train_inputs[0].to(self.dtype).to(self.device)  # (n, d)
            n_train = X_train.shape[0]

            # Hyperparameters (move to our dtype/device)
            ls, outputscale, noise = self._kernel_params(model)            # (d,), scalar, scalar

            # K(X_train, X_train) + noise * I
            with torch.no_grad():
                K_train = model.covar_module(X_train).evaluate().to(self.dtype)
            K_noisy = K_train + noise * torch.eye(
                n_train, dtype=self.dtype, device=self.device)

            # Cholesky for stable solves; jitter if needed
            jitter = 1e-6
            for attempt in range(5):
                try:
                    L = torch.linalg.cholesky(
                        K_noisy + jitter * torch.eye(
                            n_train, dtype=self.dtype, device=self.device))
                    break
                except Exception:
                    jitter *= 10
            else:
                # Last resort: pseudo-inverse (slow, but correct)
                K_inv = torch.linalg.pinv(K_noisy)
                L = None

            # k_* for all test points: shape (n_train, N)
            with torch.no_grad():
                k_star = model.covar_module(X_train, X).evaluate().to(self.dtype)

            # Diagonal of d2k/dx*_j^2 = outputscale / l_j^2  (constant in x*)
            diag_term = outputscale / (ls ** 2)  # (d,)

            # For each dim j compute the derivative covector and the variance
            for j in range(d):
                # dk_*/dx*_j has shape (n_train, N)
                # dk_*/dx*_j[:, i] = -((X_train[:, j] - X[i, j]) / l_j^2) * k_star[:, i]
                diff_j = X_train[:, j].unsqueeze(1) - X[:, j].unsqueeze(0)  # (n_train, N)
                dk_star = -(diff_j / (ls[j] ** 2)) * k_star                  # (n_train, N)

                # solve K_inv @ dk_star
                if L is not None:
                    K_inv_dk = torch.cholesky_solve(dk_star, L)              # (n_train, N)
                else:
                    K_inv_dk = K_inv @ dk_star

                # Quadratic form per test point: sum over n_train of dk * (K^-1 dk)
                quad = (dk_star * K_inv_dk).sum(dim=0)                       # (N,)
                var_kj = diag_term[j] - quad                                  # (N,)
                all_var[:, k, j] = var_kj

        return torch.clamp(all_var, min=0.0)

    # ------------------------------------------------------------------ #
    # Acquisition + exclusion
    # ------------------------------------------------------------------ #

    def _acquisition_per_dim(self, grad_mu: torch.Tensor,
                             grad_var: torch.Tensor) -> torch.Tensor:
        """Combine per-(output, dim) acquisition scores into one (N, d) tensor.
        Reduction across outputs controlled by self.multi_output_reduce.
        """
        per_output = torch.abs(grad_mu) + self.beta * torch.sqrt(grad_var)  # (N, n_out, d)
        mode = self.multi_output_reduce
        if mode == 'sum':
            return per_output.sum(dim=1)            # (N, d)
        if mode == 'max':
            return per_output.max(dim=1).values     # (N, d)
        if mode == 'standardized_sum':
            # Per-output normalize by 90th-percentile magnitude across the
            # candidate pool (summed over input dims to get a single scale).
            # Avoids one loud output dominating the per-dim sum.
            scale = torch.quantile(per_output.sum(dim=2), 0.9, dim=0)  # (n_out,)
            scale = torch.clamp(scale, min=1e-12)
            return (per_output / scale.view(1, -1, 1)).sum(dim=1)   # (N, d)
        raise ValueError(f"Unknown multi_output_reduce: {mode!r}")

    def _is_excluded(self, X_pool: torch.Tensor) -> torch.Tensor:
        """Boolean mask (N,): True where pool points fall inside any region."""
        if not self.exclusion_regions:
            return torch.zeros(X_pool.shape[0], dtype=torch.bool,
                               device=self.device)
        mask = torch.zeros(X_pool.shape[0], dtype=torch.bool,
                           device=self.device)
        for region in self.exclusion_regions:
            diff = (X_pool - region["center"]) / region["radii"]
            mask |= ((diff ** 2).sum(dim=1) <= 1.0)
        return mask

    def _add_exclusion(self, x_selected: torch.Tensor, dim_j: int):
        """Axis-aligned ellipse: small radius along dim_j, large elsewhere."""
        radii = torch.full((self.n_inputs,), self.large_radius,
                           dtype=self.dtype, device=self.device)
        radii[dim_j] = self.small_radius
        self.exclusion_regions.append({
            "center": x_selected.detach().clone(),
            "radii": radii,
        })

    # ------------------------------------------------------------------ #
    # Batch selection
    # ------------------------------------------------------------------ #

    def _propose_batch(self, models: ModelListGP, X_existing: torch.Tensor,
                       n_points: int,
                       boundary_func: callable = None) -> torch.Tensor:
        # Reset for this call
        self.exclusion_regions = []

        bounds = torch.tensor([[0.0] * self.n_inputs, [1.0] * self.n_inputs],
                              dtype=self.dtype, device=self.device)

        print(f"  Generating {self.candidate_pool:,} Sobol candidate points...")
        X_pool = draw_sobol_samples(bounds=bounds, n=self.candidate_pool,
                                    q=1).squeeze(1)

        print(f"  Computing posterior gradient mean (autograd) on pool...")
        grad_mu = self._grad_mu(models, X_pool)
        print(f"  Computing posterior gradient variance (analytical) on pool...")
        grad_var = self._grad_var(models, X_pool)
        alpha = self._acquisition_per_dim(grad_mu, grad_var)  # (N, d)

        # Diagnostics: per-dim acquisition stats
        for j in range(self.n_inputs):
            a = alpha[:, j]
            print(f"    dim {j}: alpha range [{a.min().item():.4e}, "
                  f"{a.max().item():.4e}], median {a.median().item():.4e}")

        # Pre-exclude isotropic regions around existing experimental points
        # using the same `small_radius` as intra-batch exclusion. Previously
        # this was clamped to 0.02 to avoid pool starvation, but that allowed
        # cross-iteration clumping (new picks landing 0.02 from old picks
        # while intra-batch picks were never closer than 0.05).
        existing_radius = self.small_radius
        if len(X_existing) > 0:
            iso_radii = torch.full((self.n_inputs,), existing_radius,
                                   dtype=self.dtype, device=self.device)
            for x_e in X_existing:
                self.exclusion_regions.append({
                    "center": x_e.to(self.dtype).to(self.device).detach(),
                    "radii": iso_radii.clone(),
                })
            n_excluded_init = int(self._is_excluded(X_pool).sum().item())
            print(f"  Pre-excluded {n_excluded_init}/{self.candidate_pool} "
                  f"candidates near {len(X_existing)} existing points "
                  f"(isotropic radius {existing_radius:.3f}).")

        selected = []
        rng = np.random.default_rng(0)
        small_r0, large_r0 = self.small_radius, self.large_radius
        relax_count = 0

        while len(selected) < n_points:
            dims = list(range(self.n_inputs))
            rng.shuffle(dims)
            progressed = False
            for j in dims:
                if len(selected) >= n_points:
                    break
                mask = self._is_excluded(X_pool)
                if bool(mask.all().item()):
                    # Pool starved -> shrink radii and retry
                    if relax_count >= 5 or self.small_radius < 1e-3:
                        print(f"  WARNING: pool exhausted after "
                              f"{relax_count} relaxations; returning "
                              f"{len(selected)} picks.")
                        if not selected:
                            return X_pool[:1]
                        return torch.stack(selected)
                    self.small_radius *= 0.5
                    self.large_radius *= 0.5
                    relax_count += 1
                    print(f"  Pool exhausted -> halving radii "
                          f"(small={self.small_radius:.4f}, "
                          f"large={self.large_radius:.4f}). Relaxation "
                          f"#{relax_count}.")
                    # Rebuild exclusion radii on existing regions too
                    for region in self.exclusion_regions:
                        region["radii"] *= 0.5
                    continue
                scores = alpha[:, j].clone()
                scores[mask] = float('-inf')
                idx = int(torch.argmax(scores).item())
                x_star = X_pool[idx]
                selected.append(x_star)
                self._add_exclusion(x_star, j)
                progressed = True
                hval_str = ""
                if boundary_func is not None:
                    h = float(boundary_func(x_star.detach().cpu().numpy()))
                    hval_str = f"  boundary h={h:+.3f}"
                print(f"    pick {len(selected)}/{n_points}: "
                      f"dim={j}, alpha={scores[idx].item():.4e}{hval_str}")
            if not progressed:
                # Should be unreachable given the all-mask handling above,
                # but guard against infinite loops.
                print(f"  WARNING: no progress in sub-round; returning "
                      f"{len(selected)} picks.")
                break

        # Restore radii for next call
        self.small_radius, self.large_radius = small_r0, large_r0

        return torch.stack(selected) if selected else X_pool[:1]

# Instructions: Gradient-Based Transition Recommender (v1)

## Goal
Implement the gradient-based active-learning acquisition from Vadhavkar et al.
(MSDE 2026, d5me00233h) as a new recommender for finding sharp transition
boundaries in 2-D surfactant phase diagrams. Will replace the existing
contrast-based acquisition for this use case.

## Files to create / edit
1. **NEW** `recommenders/_transition_base.py` — extract shared helpers
2. **EDIT** `recommenders/bayesian_transition_recommender.py` — refactor to inherit from base; keep contrast acquisition
3. **NEW** `recommenders/gradient_transition_recommender.py` — paper method
4. **NEW** `recommenders/test_gradient_transition_recommender.py` — head-to-head harness, **2D first**, then 3D

**Do NOT touch:** any workflow files, any other recommender.

## Hard constraints
- No CLI subagent. Edit files directly in the workspace.
- No silent defaults (see repo copilot-instructions.md). If a config value is required, look it up; if missing, raise.
- All logging via `print()` is fine for now (matches existing style); no Unicode characters (mu, +/-, ->).
- Do not add adaptive lengthscale — leave BoTorch MLE default. Revisit if GPs collapse on later iterations.
- Do not add soft penalties, distance weighting, sigmoid gates, random direction perturbations, or any of the contrast-recommender machinery.

---

## Phase 1 — Extract base class

**File:** `recommenders/_transition_base.py`

Create `class TransitionRecommenderBase` containing only the input/output
plumbing that has nothing to do with acquisition:

- `__init__(input_columns, output_columns, log_transform_inputs, candidate_pool, device, dtype)` — store config; validate `2 <= len(input_columns) <= 5` and `len(output_columns) == 2`.
- `_prepare_data`, `_process_inputs`, `_process_outputs`, `_format_recommendations`, `_print_summary` — copy from existing recommender unchanged.
- `_compute_metrics`, `get_metrics_df`, `plot_diagnostics` — copy unchanged.
- Two abstract hook methods subclasses must implement:
  - `_fit_models(self, X, Y) -> ModelListGP`
  - `_propose_batch(self, models, X_existing, n_points, boundary_func) -> torch.Tensor`
- `get_recommendations(data_df, n_points, iteration=None, boundary_func=None)` — orchestrator that calls the hooks. Copy structure from existing.

**Verify Phase 1:** existing recommender's `main()` runs and produces same outputs as before refactor.

---

## Phase 2 — Refactor existing recommender

**File:** `recommenders/bayesian_transition_recommender.py`

- `BayesianTransitionRecommender(TransitionRecommenderBase)`.
- Keep only: `__init__` (calls super, stores `delta`, `K`, `lam`, `min_distance`), `_fit_models`, `_compute_contrast_uncertainty`, `_compute_adaptive_min_distance`, `_propose_batch` (renamed from `_propose_batch_greedy`), the synthetic-boundary helpers + `main()`.
- Delete duplicated plumbing methods and `_vectorized_acquisition` (unused legacy).

**Verify Phase 2:** workflow import path still works; `main()` runs.

---

## Phase 3 — Gradient recommender

**File:** `recommenders/gradient_transition_recommender.py`

### 3.1 Class signature
```python
class GradientTransitionRecommender(TransitionRecommenderBase):
    def __init__(self, input_columns, output_columns,
                 log_transform_inputs=True,
                 candidate_pool=50_000,
                 small_radius=0.05,    # ellipse radius along selected dim
                 large_radius=0.20,    # ellipse radius along other dims
                 beta=1.0,             # UCB weight on grad std (paper: 1.0)
                 device='cpu', dtype=torch.double):
        super().__init__(...)
        self.exclusion_regions = []   # reset per _propose_batch call
```

### 3.2 GP fitting
`_fit_models(X, Y)`: per-output `SingleTaskGP` wrapped in `ModelListGP`, RBF
kernel with ARD lengthscales, `GaussianLikelihood`. Fit with
`fit_gpytorch_mll`. **No** lengthscale prior in v1.

### 3.3 Gradient mean (autograd)
```python
def _grad_mu(self, models, X):
    """X: (N, d) normalized. Returns grad_mu: (N, n_outputs, d)."""
    X_req = X.detach().clone().requires_grad_(True)
    posterior = models.posterior(X_req)
    mean = posterior.mean  # (N, n_outputs)
    grads = []
    for k in range(self.n_outputs):
        g, = torch.autograd.grad(mean[:, k].sum(), X_req,
                                 retain_graph=(k < self.n_outputs - 1))
        grads.append(g)
    return torch.stack(grads, dim=1).detach()  # (N, n_outputs, d)
```

### 3.4 Gradient variance (analytical RBF derivative posterior, paper eqns 9-10)
For each output GP separately, with RBF kernel of lengthscales `l_j`,
outputscale `s_f^2`, training inputs `X_train (n, d)`:

For test point `x*`:
- `K = k(X_train, X_train) + s_n^2 I` shape `(n, n)`
- `k_* = k(X_train, x*)` shape `(n,)`
- `dk_* / dx*_j = -((X_train[:, j] - x*[j]) / l_j^2) * k_*` shape `(n,)`
- `d2k(x*, x*) / dx*_j^2 = s_f^2 / l_j^2` (zero-distance diagonal)
- `Sigma_grad,jj(x*) = s_f^2 / l_j^2 - (dk_*/dx*_j)^T K^-1 (dk_*/dx*_j)`

Vectorize over batch of test points and all dims `j`. Returns `grad_var: (N,
n_outputs, d)`. Clamp `>= 0`.

Get hyperparameters from:
- `model.covar_module.base_kernel.lengthscale.squeeze()` -> `(d,)`
- `model.covar_module.outputscale` -> scalar
- `model.likelihood.noise` -> scalar

### 3.5 Per-dimension acquisition
```python
def _acquisition_per_dim(self, grad_mu, grad_var):
    alpha_per_output = torch.abs(grad_mu) + self.beta * torch.sqrt(grad_var)
    return alpha_per_output.sum(dim=1)   # (N, d)
```

### 3.6 Ellipse exclusion (axis-aligned, anisotropic, generalizes to d-D)
```python
def _is_excluded(self, X_pool):
    if not self.exclusion_regions:
        return torch.zeros(X_pool.shape[0], dtype=torch.bool, device=self.device)
    mask = torch.zeros(X_pool.shape[0], dtype=torch.bool, device=self.device)
    for region in self.exclusion_regions:
        diff = (X_pool - region["center"]) / region["radii"]
        mask |= ((diff ** 2).sum(dim=1) <= 1.0)
    return mask

def _add_exclusion(self, x_selected, dim_j):
    radii = torch.full((self.n_inputs,), self.large_radius,
                       dtype=self.dtype, device=self.device)
    radii[dim_j] = self.small_radius
    self.exclusion_regions.append({"center": x_selected.detach().clone(),
                                   "radii": radii})
```

### 3.7 Batch selection
```python
def _propose_batch(self, models, X_existing, n_points, boundary_func=None):
    self.exclusion_regions = []

    bounds = torch.tensor([[0.]*self.n_inputs, [1.]*self.n_inputs],
                          dtype=self.dtype, device=self.device)
    X_pool = draw_sobol_samples(bounds=bounds, n=self.candidate_pool, q=1).squeeze(1)

    grad_mu = self._grad_mu(models, X_pool)
    grad_var = self._grad_var(models, X_pool)
    alpha = self._acquisition_per_dim(grad_mu, grad_var)   # (N, d)

    # Pre-exclude isotropic regions around existing experimental points
    for x_e in X_existing:
        radii = torch.full((self.n_inputs,), self.large_radius,
                           dtype=self.dtype, device=self.device)
        self.exclusion_regions.append({"center": x_e, "radii": radii})

    selected = []
    rng = np.random.default_rng(0)
    while len(selected) < n_points:
        dims = list(range(self.n_inputs))
        rng.shuffle(dims)
        for j in dims:
            if len(selected) >= n_points:
                break
            mask = self._is_excluded(X_pool)
            if mask.all():
                self.small_radius *= 0.5
                self.large_radius *= 0.5
                if self.small_radius < 1e-3:
                    print(f"  WARNING: pool exhausted, returning {len(selected)} picks")
                    return torch.stack(selected) if selected else X_pool[:1]
                continue
            scores = alpha[:, j].clone()
            scores[mask] = float('-inf')
            idx = int(torch.argmax(scores))
            x_star = X_pool[idx]
            selected.append(x_star)
            self._add_exclusion(x_star, j)

    return torch.stack(selected)
```

GP is **not** refit between picks. `alpha` is computed once per
`get_recommendations` call.

---

## Phase 4 — 2D tests (DO THIS FIRST, BEFORE ANY 3D)

**File:** `recommenders/test_gradient_transition_recommender.py`

CLI dispatcher pattern:
```
python -m recommenders.test_gradient_transition_recommender --test step2d
python -m recommenders.test_gradient_transition_recommender --test circle2d
python -m recommenders.test_gradient_transition_recommender --test surfactant2d
```

Three required 2D tests, in this order:

### 4.1 `step2d` — line boundary, smooth (NOT abrupt)
- Boundary: `h(x) = x[0] - 0.5`.
- Outputs: smooth sigmoid in `h(x)` with width 0.05.
- Purpose: simplest possible test. Confirms gradient acquisition concentrates picks on `x[0] ~ 0.5` and spreads them along `x[1]`.

### 4.2 `circle2d` — circular boundary, smooth
- Boundary: `h(x) = sqrt((x[0]-0.5)^2 + (x[1]-0.5)^2) - 0.3`.
- Outputs: smooth sigmoid in `h(x)` with width 0.05.
- Purpose: confirms picks trace a curved boundary (not just a straight line). Per-dim acquisition matters here — picks should cover both x and y portions.

### 4.3 `surfactant2d` — real simulator
- Calls `simulate_surfactant_measurements` from `workflows/surfactant_grid_adaptive_concentrations.py` (returns `ratio` and `turbidity_600`).
- Concentration range: `[0.01, 25.0]` mM, log-transformed by recommender.
- Purpose: end-to-end check on actual dual-output behavior the lab will see.

For each test:
- Initial Sobol design (~10 points), `Q_BATCH=8`, `N_ITERATIONS=5`.
- Run **both** `BayesianTransitionRecommender` (existing contrast) and `GradientTransitionRecommender` (new) on identical seeds.
- Outputs to `recommenders/test_outputs/<testname>/`:
  - `all_data_<recommender>.csv`
  - `metrics_<recommender>.csv`
  - `2d_exploration_<recommender>.png` — scatter colored by iteration, true boundary contour overlaid
  - `gradient_map_<recommender>.png` — heatmap of `||grad mu||` over `[0,1]^2` (gradient recommender only)
  - `comparison_hd.png` — HD-vs-iteration for both recommenders side by side

**Do not move to Phase 5 (3D) until all 2D tests pass.**

---

## Phase 5 — 3D tests (only after Phase 4 passes)

Add to the same test file:
- `ellipse3d` — `(x/a)^2 + (y/b)^2 + (z/c)^2 = 1` with smooth transition.
- `saddle3d` — saddle boundary with oscillations.

Outputs: `3d_exploration_progress_<recommender>.png` (per-iteration scatter slices).

---

## Verification checklist (in order)

1. **Phase 1**: existing recommender's `main()` produces same selected points as before refactor.
2. **Phase 2**: workflow import works, `main()` runs.
3. **Phase 3 unit checks**:
   - `_grad_mu` on a 1D GP fit to `y = sin(x)`: returns approx `cos(x)` at training points (vs. `numpy.gradient`).
   - `_grad_var` on the same GP: variance is **near zero at training points**, grows away. If this fails, the analytical formula is wrong — fix before continuing.
   - `_is_excluded` on hand-built 2D case with one exclusion at `(0.5, 0.5)` radii `[0.05, 0.2]`: confirms a horizontal strip excluded, not a circle.
4. **Phase 4 outcomes**:
   - `step2d`: gradient recommender's points concentrate at `x[0] ~ 0.5`, spread evenly along `x[1]`. Contrast recommender clusters or spreads uniformly.
   - `circle2d`: gradient recommender's points trace the circle (both x and y axes hit). Contrast recommender misses parts of the circle.
   - `surfactant2d`: gradient recommender finds the dual-transition region.
5. **Phase 5**: 3D HD curves drop below 0.1 within ~17-24 simulations on `ellipse3d` (paper Fig 4 envelope).
6. **No silent fallbacks**: pool-exhaustion path prints WARNING; doesn't return a fabricated batch.

---

## Recovery if implementation fails

- Each phase is independently reversible via `git checkout -- recommenders/`.
- Existing `BayesianTransitionRecommender` is untouched until Phase 2; workflow import keeps working.
- If only Phase 3 fails: base class + refactored existing recommender are still useful.
- Most error-prone single piece: `_grad_var`. If unit check fails, the rest is meaningless. Check: noisy kernel `K + s_n^2 I` for `K^-1`? Lengthscales right shape? Sign on `dk/dx` correct?

---

## Deliberately NOT in v1
- Adaptive lengthscale (paper section 4.1) — only if GP visibly collapses
- Stopping criterion via rolling mean — fixed iteration count
- Per-(output, dim) acquisition picks — collapsed to per-dim summed across outputs
- Soft exclusion penalties

---

## Comparison: paper vs. user-pasted instructions vs. failed CLI attempt

| Item | Paper | User instructions | What CLI built (FAILED) | This v1 |
|---|---|---|---|---|
| Gradient mean | analytical | autograd | autograd | autograd |
| Gradient variance | analytical Sigma_grad in AF | "optional later" | posterior.variance (WRONG) | analytical Sigma_grad |
| Per-dim AF | yes | yes | single global score (WRONG) | yes |
| Anisotropic ellipses | axis-aligned | hardcoded 2D | gradient-frame (different) | axis-aligned, d-dim |
| Beta | 1.0 | not specified | 0.0 default (disabled) | 1.0 |
| Distance weighting | absent | removed | removed | removed |
| Multi-output | single output | not addressed | sum across outputs (max-norm) | sum across outputs |
| Compartmentalization | n/a | "compartmentalize" | edited in place, no base class | base class extracted |

"""
Delaunay Simplex Transition Recommender (n-dim)
================================================

n-dimensional generalization of the 2D Delaunay triangle recommender.
Builds a Delaunay tessellation of the existing experimental points
(triangles in 2D, tetrahedra in 3D, simplices in higher d) and proposes
new points at simplex centroids, scored by output disagreement among
the simplex vertices and weighted by simplex volume.

Algorithm
---------
1. Standardize outputs (z-score per output) so disparate scales are
   comparable.
2. Build a Delaunay tessellation in normalized [0,1]^d input space.
3. Score each simplex by:
       base_score = max pairwise || y_i - y_j || over its (d+1) vertices,
                    in standardized output space
       final_score = base_score * (volume / median_volume) ** beta
   beta>0 boosts large simplices; beta=0 disables volume influence.
4. Sort simplices by final_score, select top n_points centroids subject
   to a soft min-spacing constraint (factor of median nearest-neighbor
   distance among the existing points).

Notes
-----
- Cost of scipy.spatial.Delaunay is ~ O(n log n) in 3D; ~ms for n<200.
- This recommender does NOT use a GP. It relies purely on the geometry
  and the observed output values, so it is robust to noisy GP fits and
  has no acquisition tuning.
- Inherits from TransitionRecommenderBase but does not implement
  _fit_models (returns a dummy); _propose_batch ignores the models.
"""

from typing import List

import numpy as np
import torch
from math import factorial
from scipy.spatial import Delaunay

from recommenders._transition_base import TransitionRecommenderBase


DEFAULT_BETA = 0.5
DEFAULT_MIN_SPACING_FACTOR = 0.5
DEFAULT_TOL_FACTOR = 0.1
SCORE_METHOD = "max"            # "max" or "second_largest"


class _NoopModelList:
    """Stand-in returned by _fit_models since this recommender ignores GPs."""
    pass


class DelaunaySimplexTransitionRecommender(TransitionRecommenderBase):
    """n-dim Delaunay-simplex recommender."""

    def __init__(self, input_columns: List[str], output_columns: List[str],
                 log_transform_inputs: bool = True,
                 beta: float = DEFAULT_BETA,
                 min_spacing_factor: float = DEFAULT_MIN_SPACING_FACTOR,
                 tol_factor: float = DEFAULT_TOL_FACTOR,
                 score_method: str = SCORE_METHOD,
                 candidate_pool: int = 0,
                 device: str = "cpu", dtype=torch.double):
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            log_transform_inputs=log_transform_inputs,
            candidate_pool=candidate_pool,
            device=device,
            dtype=dtype,
        )
        self.beta = float(beta)
        self.min_spacing_factor = float(min_spacing_factor)
        self.tol_factor = float(tol_factor)
        self.score_method = str(score_method)

        print(f"Initialized DelaunaySimplexTransitionRecommender (n-dim):")
        print(f"  Inputs ({self.n_inputs}D): {input_columns}")
        print(f"  Outputs ({self.n_outputs}): {output_columns}")
        print(f"  log_transform_inputs={log_transform_inputs}")
        print(f"  beta={beta} (volume scaling), score_method={score_method}")
        print(f"  spacing: min_spacing_factor={min_spacing_factor} of "
              f"median NN distance, tol_factor={tol_factor}")

    # --------------------------------------------------------------- #
    def _fit_models(self, X, Y):
        # No GP needed; return placeholder. _propose_batch uses (X, Y).
        self._cached_X = X
        self._cached_Y = Y
        return _NoopModelList()

    # --------------------------------------------------------------- #
    @staticmethod
    def _simplex_volume(verts: np.ndarray) -> float:
        """Volume of a d-simplex with d+1 vertices in d-space."""
        n_v, d = verts.shape
        assert n_v == d + 1
        M = verts[1:] - verts[0]                   # (d, d)
        return abs(np.linalg.det(M)) / factorial(d)

    @staticmethod
    def _max_pairwise_dist(Yvals: np.ndarray) -> float:
        """Max pairwise L2 distance among rows of Yvals."""
        n = Yvals.shape[0]
        best = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(Yvals[i] - Yvals[j]))
                if d > best:
                    best = d
        return best

    @staticmethod
    def _second_largest_pairwise(Yvals: np.ndarray) -> float:
        n = Yvals.shape[0]
        ds = []
        for i in range(n):
            for j in range(i + 1, n):
                ds.append(float(np.linalg.norm(Yvals[i] - Yvals[j])))
        ds.sort(reverse=True)
        if len(ds) >= 2:
            return ds[1]
        return ds[0] if ds else 0.0

    # --------------------------------------------------------------- #
    def _propose_batch(self, models, X_existing: torch.Tensor,
                       n_points: int,
                       boundary_func: callable = None) -> torch.Tensor:
        # Pull cached X (normalized) and Y (standardized) saved by _fit_models
        X = self._cached_X.detach().cpu().numpy()
        Y = self._cached_Y.detach().cpu().numpy()
        d = X.shape[1]
        n_x = X.shape[0]

        # ---- 1. Delaunay tessellation in normalized input space
        try:
            tri = Delaunay(X)
        except Exception as e:
            raise RuntimeError(f"Delaunay failed in {d}D with {n_x} points: {e}")
        simplices = tri.simplices                        # (n_s, d+1)
        n_s = simplices.shape[0]
        print(f"  Built tessellation: {n_x} pts -> {n_s} simplices in {d}D")

        # ---- 2. Score every simplex
        vols = np.empty(n_s)
        base_scores = np.empty(n_s)
        centroids = np.empty((n_s, d))
        for s_idx in range(n_s):
            verts_idx = simplices[s_idx]
            verts = X[verts_idx]                         # (d+1, d)
            yvals = Y[verts_idx]                         # (d+1, n_out)
            vols[s_idx] = self._simplex_volume(verts)
            if self.score_method == "second_largest":
                base_scores[s_idx] = self._second_largest_pairwise(yvals)
            else:
                base_scores[s_idx] = self._max_pairwise_dist(yvals)
            centroids[s_idx] = verts.mean(axis=0)

        median_vol = float(np.median(vols))
        if median_vol <= 0:
            volume_factor = np.ones(n_s)
        else:
            volume_factor = (vols / median_vol) ** self.beta
        scores = base_scores * volume_factor
        print(f"  Volumes: min={vols.min():.4e}, median={median_vol:.4e}, "
              f"max={vols.max():.4e}")
        print(f"  Base scores: min={base_scores.min():.3f}, "
              f"median={np.median(base_scores):.3f}, "
              f"max={base_scores.max():.3f}")
        print(f"  Final scores: min={scores.min():.3f}, "
              f"median={np.median(scores):.3f}, max={scores.max():.3f}")

        # ---- 3. Spacing constraint (factor of median NN among existing X)
        nn_dists = []
        for i in range(n_x):
            d2 = np.linalg.norm(X - X[i], axis=1)
            d2[i] = np.inf
            nn_dists.append(d2.min())
        median_nn = float(np.median(nn_dists))
        d_min = self.min_spacing_factor * median_nn
        tol = self.tol_factor * median_nn
        print(f"  Median NN distance: {median_nn:.4f} -> "
              f"d_min={d_min:.4f}, tol={tol:.4f}")

        # ---- 4. Greedy selection by score with spacing
        order = np.argsort(-scores)                      # high to low
        chosen_centroids = []
        rejected_dup = 0
        rejected_close = 0
        for s_idx in order:
            if len(chosen_centroids) >= n_points:
                break
            c = centroids[s_idx]
            # not too close to any existing point
            d_to_x = np.linalg.norm(X - c, axis=1)
            if d_to_x.min() < tol:
                rejected_dup += 1
                continue
            # not too close to any already chosen centroid
            if len(chosen_centroids) > 0:
                arr = np.stack(chosen_centroids, axis=0)
                d_to_chosen = np.linalg.norm(arr - c, axis=1)
                if d_to_chosen.min() < d_min:
                    rejected_close += 1
                    continue
            chosen_centroids.append(c)
        print(f"  Selected {len(chosen_centroids)}/{n_points} "
              f"(rejected {rejected_dup} dup, {rejected_close} too close)")

        if len(chosen_centroids) == 0:
            # Fall back to top-score centroids ignoring spacing
            print("  WARNING: no centroids passed spacing; falling back.")
            chosen_centroids = [centroids[i] for i in order[:n_points]]

        out = np.clip(np.stack(chosen_centroids, axis=0), 0.0, 1.0)
        return torch.tensor(out, dtype=self.dtype, device=self.device)

"""
Metrics for evaluating boundary-tracing recommenders.

Two groups:
  boundary_metrics  - per output, given a fine grid of ground-truth y values
                      and the recommender's picks; reports hit rate and
                      coverage statistics over the boundary mask.
  clumping_metrics  - boundary-agnostic; describes how clustered the picks are
                      versus a uniform reference.

All distances are in NORMALIZED coordinates: caller maps picks and the grid
into the same [0,1]^d cube before calling these functions, so values are
scale-free and comparable across datasets.

Boundary definition (decided by inspection of recommenders/test_outputs/
boundary_diag/*.png on May 2026):
    ratio:         top 30% of |grad y| on the fine grid
    turbidity_600: top 10% of |grad y|
A per-output presence check skips outputs whose value range is below
MIN_RANGE.
"""

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

# np.trapezoid was added in NumPy 2.0; fall back to np.trapz for older installs
_trapz = getattr(np, 'trapezoid', np.trapz)


TOP_FRAC = {"ratio": 0.30, "turbidity_600": 0.10}
MIN_RANGE = 0.10


# -----------------------------------------------------------------
# Boundary mask
# -----------------------------------------------------------------

def grad_magnitude(F, X, Y):
    """Central-difference |grad F| on a 2D grid using grid spacing."""
    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]
    gx, gy = np.gradient(F, dx, dy)
    return np.sqrt(gx ** 2 + gy ** 2)


def define_boundary(F, X, Y, top_frac, min_range=MIN_RANGE):
    """Return (mask, has_boundary).
    mask is a boolean array same shape as F, True on the top top_frac of
    |grad F|. has_boundary is False (mask=None) when max(F)-min(F) < min_range.
    """
    rng = float(F.max() - F.min())
    if rng < min_range:
        return None, False
    G = grad_magnitude(F, X, Y)
    thresh = np.percentile(G, 100.0 * (1.0 - top_frac))
    return G >= thresh, True


# -----------------------------------------------------------------
# Boundary-coverage metrics (per output)
# -----------------------------------------------------------------

def boundary_metrics(picks_norm, mask, X_norm, Y_norm):
    """All inputs in normalized [0,1]^2 coordinates.
      picks_norm  (N, 2)
      mask        (n_grid, n_grid) boolean - boundary cells
      X_norm      (n_grid, n_grid) - grid x in [0,1]
      Y_norm      (n_grid, n_grid) - grid y in [0,1]
    Returns dict of metrics:
      hit_rate          frac of picks landing on a boundary cell
      coverage_auc      area under (frac boundary cells covered) vs r curve
      n_boundary_cells  size of the boundary mask
    """
    n_picks = picks_norm.shape[0]
    bnd_pts = np.stack([X_norm[mask], Y_norm[mask]], axis=1)
    n_bnd = bnd_pts.shape[0]
    if n_picks == 0 or n_bnd == 0:
        return {
            "hit_rate": float("nan"),
            "coverage_auc": float("nan"),
            "n_boundary_cells": int(n_bnd),
        }

    # "hit rate" = which picks land on a boundary cell (nearest-cell lookup)
    n = X_norm.shape[0]
    ix = np.clip((picks_norm[:, 0] * (n - 1)).round().astype(int), 0, n - 1)
    iy = np.clip((picks_norm[:, 1] * (n - 1)).round().astype(int), 0, n - 1)
    on_bnd = mask[ix, iy]
    hit_rate = float(on_bnd.mean())

    # For each boundary cell, distance to nearest pick.
    pick_tree = cKDTree(picks_norm)
    nn_dist, _ = pick_tree.query(bnd_pts, k=1)

    # Coverage AUC: integrate frac{cells covered within r} over r in [0, 1].
    rs = np.linspace(0.0, 1.0, 101)
    covered = np.array([(nn_dist <= r).mean() for r in rs])
    coverage_auc = float(_trapz(covered, rs))

    return {
        "hit_rate": hit_rate,
        "coverage_auc": coverage_auc,
        "n_boundary_cells": int(n_bnd),
    }


# -----------------------------------------------------------------
# 3D boundary helpers (same definitions, generalized to a cube grid)
# -----------------------------------------------------------------

def grad_magnitude_3d(F, axes):
    """Central-difference |grad F| on a 3D grid.
    F     : (n, n, n) array
    axes  : (g, g, g) length-3 tuple of 1D arrays (each of length n)
    """
    dx = axes[0][1] - axes[0][0]
    dy = axes[1][1] - axes[1][0]
    dz = axes[2][1] - axes[2][0]
    g1, g2, g3 = np.gradient(F, dx, dy, dz)
    return np.sqrt(g1 ** 2 + g2 ** 2 + g3 ** 2)


def define_boundary_3d(F, axes, top_frac, min_range=MIN_RANGE):
    rng = float(F.max() - F.min())
    if rng < min_range:
        return None, False
    G = grad_magnitude_3d(F, axes)
    thresh = np.percentile(G, 100.0 * (1.0 - top_frac))
    return G >= thresh, True


def define_boundary_levelset_3d(F, threshold, shell_frac=0.12):
    """Return a boolean mask for the level-set shell at F == threshold.

    Cells within the shell are those where |F - threshold| is in the
    smallest shell_frac fraction of |F - threshold| values across the
    whole grid.  This gives a thin surface of consistent fractional
    thickness regardless of the gradient steepness.

    shell_frac=0.12 means ~12% of grid cells form the shell — wide
    enough that picks can realistically land on it, narrow enough to
    be physically meaningful.

    Returns (mask, has_boundary):
      mask           boolean (n,n,n), True on the shell
      has_boundary   False if the threshold is outside the data range
    """
    fmin, fmax = float(F.min()), float(F.max())
    if threshold <= fmin or threshold >= fmax:
        return None, False
    dist = np.abs(F - threshold)
    cutoff = np.percentile(dist, 100.0 * shell_frac)
    return dist <= cutoff, True


def boundary_metrics_3d(picks_norm, mask, axes):
    """3D version. picks_norm (N,3), mask (n,n,n), axes len-3 of (n,)."""
    n_picks = picks_norm.shape[0]
    g1, g2, g3 = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    bnd_pts = np.stack([g1[mask], g2[mask], g3[mask]], axis=1)
    n_bnd = bnd_pts.shape[0]
    if n_picks == 0 or n_bnd == 0:
        return {
            "hit_rate": float("nan"),
            "coverage_auc": float("nan"),
            "n_boundary_cells": int(n_bnd),
        }
    n = len(axes[0])
    ix = np.clip((picks_norm[:, 0] * (n - 1)).round().astype(int), 0, n - 1)
    iy = np.clip((picks_norm[:, 1] * (n - 1)).round().astype(int), 0, n - 1)
    iz = np.clip((picks_norm[:, 2] * (n - 1)).round().astype(int), 0, n - 1)
    on_bnd = mask[ix, iy, iz]
    hit_rate = float(on_bnd.mean())
    pick_tree = cKDTree(picks_norm)
    nn_dist, _ = pick_tree.query(bnd_pts, k=1)
    rs = np.linspace(0.0, 1.0, 101)
    covered = np.array([(nn_dist <= r).mean() for r in rs])
    coverage_auc = float(_trapz(covered, rs))
    return {
        "hit_rate": hit_rate,
        "coverage_auc": coverage_auc,
        "n_boundary_cells": int(n_bnd),
    }


# -----------------------------------------------------------------
# Clumping metrics (boundary-agnostic)
# -----------------------------------------------------------------

def clumping_metrics(picks_norm, d=None):
    """Picks in normalized [0,1]^d. Returns dict with:
      cv_nn           std/mean of nearest-neighbor distances
                      (~0.52 for uniform random in 2D, 0 for perfect grid,
                       larger for clumpy)
      clumping_ratio  median(d_nn) / d_uniform
                      d_uniform = (1/N)^(1/d)  (expected NN spacing for grid)
                      ~ 1.0 means grid-like spacing
      min_pairwise    minimum nearest-neighbor distance (worst clump)
    """
    n = picks_norm.shape[0]
    if d is None:
        d = picks_norm.shape[1]
    if n < 2:
        return {
            "cv_nn": float("nan"),
            "clumping_ratio": float("nan"),
            "min_pairwise": float("nan"),
        }
    tree = cKDTree(picks_norm)
    nn, _ = tree.query(picks_norm, k=2)
    nn_dist = nn[:, 1]
    median_nn = float(np.median(nn_dist))
    d_uniform = (1.0 / n) ** (1.0 / d)
    return {
        "cv_nn": float(nn_dist.std() / nn_dist.mean()),
        "clumping_ratio": float(median_nn / d_uniform),
        "min_pairwise": float(nn_dist.min()),
    }

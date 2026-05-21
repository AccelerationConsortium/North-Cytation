# -*- coding: utf-8 -*-
"""
Surfactant Multidimensional Workflow (N surfactants, N >= 2)
=============================================================

Generalizes workflows/surfactant_grid_adaptive_concentrations.py from 2D
(SDS + TTAB) to N-D (any 2-5 surfactants from SURFACTANT_LIBRARY) using
the same substock prep, dispensing primitives, and measurement helpers.

Differs from the 2D file in three places only:
  1. SURFACTANTS list drives column generation (N concentration cols, N
     volume cols, N substock-name cols).
  2. Initial sampling is a log-spaced N-D grid (cube approach) instead of
     a 2D meshgrid; per-axis max = stock_conc * (max_vol_per_surf / well_vol)
     where max_vol_per_surf = budget / N. Every cube point is feasible by
     construction (no constraint solver needed).
  3. Recommender call uses the N-D-capable transition recommenders
     (DelaunaySimplexTransitionRecommender, BayesianTransitionRecommender,
     GradientTransitionRecommender, LevelSetTransitionRecommender, or
     Sobol/Random baselines) via the shared TransitionRecommenderBase interface.

Excluded vs 2D file: kinetics, CMC controls, adaptive baseline-rectangle
re-bounding, 2D heatmaps, contour plots. Post-experiment N-D analysis is
left to a separate script that reads the saved CSV.

Volume budget per well: WELL_VOLUME_UL = PYRENE + sum(surf vols) + water.
Per-surfactant cap is BUDGET / N where BUDGET = WELL_VOLUME_UL - PYRENE.
This guarantees every (conc_1, ..., conc_N) combination in the cube fits.
"""

import sys
sys.path.append("../utoronto_demo")

import os
import numpy as np
import pandas as pd
from datetime import datetime

from master_usdl_coordinator import Lash_E

# Reuse all the 2D primitives. Per-surfactant dispensing uses a column-rename
# shim so we can keep the existing surf_A_* / surf_B_* code paths intact.
from workflows.surfactant_grid_adaptive_concentrations import (
    SURFACTANT_LIBRARY,
    NUM_SUBSTOCKS,
    MIN_WELL_PIPETTE_VOLUME_UL,
    fill_water_vial,
    refill_surfactant_vial,
    calculate_smart_dilution_plan,
    create_plan_from_existing_stocks,
    calculate_dilution_recipes,
    create_substocks_from_recipes,
    dispense_component_to_wellplate,
    position_surfactant_vials_by_concentration,
    return_surfactant_vials_home,
    return_water_vial_home,
    dispense_dmso,
    measure_turbidity,
    measure_fluorescence,
    validate_pipetting_system,
)

# ================================================================================
# CONFIGURATION
# ================================================================================

# Pick any 2-5 surfactants present in SURFACTANT_LIBRARY. The whole workflow
# is parameterized over this list.
SURFACTANTS = ["TTAB", "DTAB", "SDS"]

EXPERIMENT_TAG = "multidim_v1"
SIMULATE = True
VALIDATE_LIQUIDS = False

# Recommender selection
RECOMMENDER_TYPE = "bayesian"  # 'triangle' | 'bayesian' | 'gradient' | 'levelset' | 'sobol' | 'random'

# Turbidity filtering for recommender inputs
# - FILTER_UNRELIABLE_RATIOS: null out ratio for high-turbidity wells before passing
#   to the recommender so unreliable fluorescence data does not corrupt GP fits.
# - TURBIDITY_FILTER_THRESHOLD: per-point cutoff (above = ratio is unreliable).
# - FLAT_TURBIDITY_MAX: if max(turbidity_600) across ALL data is below this value,
#   the turbidity landscape is considered flat/uninformative and is dropped from
#   output_columns so the recommender only sees ratio.
# - TURBIDITY_PLOT_THRESHOLD: threshold used only for the final 3D turbidity
#   cloud visualization (does not affect active-learning decisions).
# - OUTPUT_COLUMNS_OVERRIDE: when set to a list (e.g. ['turbidity_600'] or ['ratio']),
#   bypasses all automatic output selection logic and forces these columns.
#   Set to None to use the automatic turbidity/ratio logic.
FILTER_UNRELIABLE_RATIOS = True
TURBIDITY_FILTER_THRESHOLD = 0.2   # Wells above this turbidity have unreliable ratio
FLAT_TURBIDITY_MAX = 0.08          # Whole-dataset gate: drop turbidity if max <= this
TURBIDITY_PLOT_THRESHOLD = 0.08    # 3D cloud threshold for post-run visualization only
OUTPUT_COLUMNS_OVERRIDE = None     # e.g. ['turbidity_600'] to test turbidity-only
# Turbidity acquisition penalty — discourages recommenders from picking deep inside
# the turbid (precipitated) zone where the boundary is already clear.
# Penalty = exp(-(turbidity - threshold) / decay) for turbidity > threshold, else 1.0.
# threshold=0.2: no penalty below this (allows sampling the boundary zone freely)
# decay=0.15: weight at turbidity 0.4 -> 0.26, at 0.7 -> 0.08
TURBIDITY_PENALTY_THRESHOLD = 0.2
TURBIDITY_PENALTY_DECAY = 0.15
TURBIDITY_PENALTY_ENABLED = False  # Set False to disable penalty for all recommenders

# Bayesian recommender spacing control
# alpha_spacing scales the soft repulsion target: target = alpha * (1/n_total)^(1/d)
# Lower values allow more clustering; 0.7 is the default (uniform-fill heuristic).
BAYESIAN_ALPHA_SPACING = 0.7

# Volume budget.
# WATER_RESERVE_UL is always held back so that even at max concentration for
# every surfactant there is still water to fill the well.
WELL_VOLUME_UL = 250
PYRENE_VOLUME_UL = 5
WATER_RESERVE_UL = 20          # Minimum water in every well (uL)
SURFACTANT_BUDGET_UL = WELL_VOLUME_UL - PYRENE_VOLUME_UL - WATER_RESERVE_UL  # 225 uL

# Initialization strategy
# 'simplex' (recommended): stratified design that covers the true physical
#   simplex (sum of volumes <= budget).  Initialization uses:
#     - boundary simplex samples (sum of surf volumes = budget exactly)
#     - interior Sobol samples (sum of surf volumes < budget)
# 'grid' (legacy): hypercube with equal per-surf cap (BUDGET/N).  Safe for
#   2D; may not reach CMC for low-CMC surfactants in 4D.
INIT_STRATEGY = 'grid'  # 'simplex' | 'grid'
GRID_POINTS_PER_AXIS = 3   # Used only when INIT_STRATEGY='grid': 3^N grid
INIT_BOUNDARY_LEVELS = 3    # Simplex: boundary lattice density on the outer simplex facet
INIT_INTERIOR_PTS = 20      # Simplex: Sobol-in-simplex interior points
MIN_CONC_MM = 1e-2          # Lower bound for all concentration axes
FEASIBLE_MAX_CONC_MULTIPLIER = 1.0  # Multiplier for max concentration in feasibility grid (1.0 = no smoothing; <1.0 = smooth)
FEASIBLE_DIAGNOSTIC_GRID_N = 140  # Grid resolution for feasibility mask (140=default, 300/500 for artifact diagnostic)

# Active-learning loop
GRADIENT_SUGGESTIONS_PER_ITERATION = 16
RECOMMEND_OVERSAMPLE_FACTOR = 10  # Ask recommender for this many times the target; filter down after actual-volume check
TARGET_TOTAL_WELLS = 192       # Measurement budget (experiment wells only): stop after this many wells total
MAX_ITERATIONS = 20            # Safety cap; loop also stops if TARGET_TOTAL_WELLS reached

# Plate handling
MAX_WELLS_PER_PLATE = 96

# Controls (placed in the first plate before active learning starts)
# Regular controls: 1 water blank + 1 pure-stock well per surfactant
# 1D controls:      CMC_CONTROL_POINTS log-spaced wells per surfactant sweeping the CMC transition
# Both groups use well_type='control' and are excluded from the AL recommender.
ADD_REGULAR_CONTROLS = True
ADD_1D_CONTROLS = True
CMC_CONTROL_POINTS = 8         # wells per surfactant in the 1D CMC sweep
STOCK_CONTROL_VOL_UL = 200     # uL of undiluted stock per stock-reference control well

# File paths
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/surfactant_multidim_vials.csv"

# ================================================================================
# DERIVED CONFIG (column names, per-surf caps)
# Lowercase so the config manager treats them as computed — not user-settable.
# ================================================================================

n_surfactants = len(SURFACTANTS)
if not (2 <= n_surfactants <= 5):
    raise ValueError(f"SURFACTANTS must have 2-5 entries (TransitionRecommenderBase limit), got {n_surfactants}")
for s in SURFACTANTS:
    if s not in SURFACTANT_LIBRARY:
        raise ValueError(f"Surfactant '{s}' not in SURFACTANT_LIBRARY. Available: {list(SURFACTANT_LIBRARY.keys())}")

# Cube per-surf caps (used by grid init and as axis clip in grid mode):
#   each surf gets an equal share of the budget
_cube_vol_per_surf = SURFACTANT_BUDGET_UL / n_surfactants
cube_max_conc_mm = {
    s: SURFACTANT_LIBRARY[s]["stock_conc"] * (_cube_vol_per_surf / WELL_VOLUME_UL)
    for s in SURFACTANTS
}

# Simplex per-surf caps: how much of the budget is available to ONE surfactant
# when all others must still be present at a pipettable minimum.
# The lowest achievable volume at MIN_CONC_MM comes from the lowest substock,
# which calculate_systematic_dilution_series always places at 10*MIN_CONC_MM:
#   min_vol = MIN_CONC_MM * WELL_VOLUME_UL / (10*MIN_CONC_MM) = WELL_VOLUME_UL/10
_min_vol_per_other_surf_ul = WELL_VOLUME_UL / 10.0  # 25 uL for 250 uL wells
_simplex_budget_for_one = max(
    SURFACTANT_BUDGET_UL - (n_surfactants - 1) * _min_vol_per_other_surf_ul, 0.0
)
simplex_max_conc_mm = {
    s: SURFACTANT_LIBRARY[s]["stock_conc"] * (_simplex_budget_for_one / WELL_VOLUME_UL)
    for s in SURFACTANTS
}

# Active caps depend on init strategy.  max_conc_mm drives the axis clip in
# get_next_batch and the axis range in generate_log_grid.
if INIT_STRATEGY == 'simplex':
    max_conc_mm = simplex_max_conc_mm
    max_vol_per_surf_ul = SURFACTANT_BUDGET_UL   # substock planner range
else:
    max_conc_mm = cube_max_conc_mm
    max_vol_per_surf_ul = _cube_vol_per_surf

# Column names used throughout the recipe DataFrame
input_cols = [f"{s}_conc_mm" for s in SURFACTANTS]
vol_cols = [f"{s}_volume_ul" for s in SURFACTANTS]
substock_name_cols = [f"{s}_substock_name" for s in SURFACTANTS]
substock_conc_cols = [f"{s}_substock_conc_mm" for s in SURFACTANTS]

# Pre-filtered Sobol pool — built once at the start of each run.
# All iterations (including simplex interior init) draw from this ordered list
# so the full experiment is one continuous low-discrepancy sequence.
_sobol_pool = []       # list of {surfactant: conc_mm} dicts, feasibility-checked
_sobol_pool_idx = 0    # next index to hand out


# ================================================================================
# GRID GENERATION
# ================================================================================

def is_feasible(point):
    """Return True if the total surfactant volume fits within the dispensing budget.

    The simplex constraint: sum(conc_mm[s] * WELL_VOLUME_UL / stock_conc[s]) <= BUDGET.
    A 1 uL tolerance is used to absorb the negligible overage that can arise
    when MIN_CONC_MM clamping nudges points slightly past the boundary.
    """
    total_vol = sum(
        point[s] * WELL_VOLUME_UL / SURFACTANT_LIBRARY[s]["stock_conc"]
        for s in SURFACTANTS
    )
    return total_vol <= SURFACTANT_BUDGET_UL + 1.0  # 1 uL tolerance


def project_to_simplex(point):
    """Scale all concentrations proportionally to fit within the volume budget.

    Preserves the relative surfactant composition (volume ratios between surfs).
    Does NOT clamp to MIN_CONC_MM — that is the caller's responsibility.
    The resulting total vol equals SURFACTANT_BUDGET_UL exactly after projection.

    Returns a new dict; does not modify point in place.
    """
    total_vol = sum(
        point[s] * WELL_VOLUME_UL / SURFACTANT_LIBRARY[s]["stock_conc"]
        for s in SURFACTANTS
    )
    if total_vol <= SURFACTANT_BUDGET_UL + 1e-9:
        return dict(point)
    scale = SURFACTANT_BUDGET_UL / total_vol
    return {s: point[s] * scale for s in SURFACTANTS}


def generate_log_grid(n_per_axis=GRID_POINTS_PER_AXIS):
    """Log-spaced N-D hypercube grid over [MIN_CONC_MM, cube_max_conc_mm[s]].

    Used when INIT_STRATEGY='grid'.  Each axis uses the cube per-surf cap
    (BUDGET/N) so that all grid corners are feasible by construction.
    Returns a list of dicts {surfactant_name: target_conc_mm} of length n_per_axis^N.
    """
    axes = [
        np.logspace(np.log10(MIN_CONC_MM), np.log10(cube_max_conc_mm[s]), n_per_axis)
        for s in SURFACTANTS
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    flat = [m.ravel() for m in mesh]
    points = []
    for i in range(flat[0].size):
        points.append({s: float(flat[k][i]) for k, s in enumerate(SURFACTANTS)})
    return points


def generate_achievable_boundary_samples(n_boundary_points, plans=None, logger=None):
    """Sample uniformly spaced points along the source-achievable boundary.
    
    Algorithm (dimension-agnostic):
    1. Generate log-space grid across full concentration range
    2. Evaluate source-achievable feasibility (via joint_select_sources) at every grid point
    3. Boundary = achievable points with at least one unachievable axis-aligned neighbor or at grid edge
    4. Greedy max-spacing in log-space to select final points
    
    Args:
        n_boundary_points: Target number of boundary points to select
        plans: List of source plans (required for source-achievable checking)
        logger: Optional logger instance
    """
    import logging as _log
    _logger = logger or _log.getLogger(__name__)

    levels = max(15, int(n_boundary_points * 3))
    _logger.info(f"Boundary sampling: {levels}^{n_surfactants} grid...")

    axes = [
        np.logspace(np.log10(MIN_CONC_MM), np.log10(simplex_max_conc_mm[s]), levels)
        for s in SURFACTANTS
    ]

    # Helper: check if point is source-achievable
    def _is_achievable(pt):
        if plans is None:
            return True  # No plans available, assume achievable
        try:
            result = joint_select_sources(pt, plans)
            return result is not None  # None = not achievable
        except:
            return False

    # Evaluate achievability on entire grid (dimension-agnostic via np.ndindex)
    grid_shape = tuple(levels for _ in SURFACTANTS)
    feasible_grid = np.zeros(grid_shape, dtype=bool)

    for idx in np.ndindex(grid_shape):
        pt = {s: float(axes[i][idx[i]]) for i, s in enumerate(SURFACTANTS)}
        try:
            feasible_grid[idx] = _is_achievable(pt)
        except:
            feasible_grid[idx] = False

    _logger.info(f"  Feasible: {int(feasible_grid.sum())}/{feasible_grid.size} grid points")

    # Boundary = feasible points with any infeasible axis-aligned neighbor or at grid edge
    boundary_points = []
    for idx in np.ndindex(grid_shape):
        if not feasible_grid[idx]:
            continue

        on_boundary = False
        for dim in range(n_surfactants):
            for delta in [-1, 1]:
                neighbor = list(idx)
                neighbor[dim] += delta
                if neighbor[dim] < 0 or neighbor[dim] >= grid_shape[dim]:
                    on_boundary = True  # Grid edge = min/max conc wall
                    break
                if not feasible_grid[tuple(neighbor)]:
                    on_boundary = True  # Infeasible neighbor = feasibility boundary
                    break
            if on_boundary:
                break

        if on_boundary:
            pt = {s: float(axes[i][idx[i]]) for i, s in enumerate(SURFACTANTS)}
            boundary_points.append(pt)

    _logger.info(f"  Boundary: {len(boundary_points)} points")

    if len(boundary_points) == 0:
        _logger.warning("  No boundary points found!")
        return []

    if len(boundary_points) <= n_boundary_points:
        result = [pt.copy() for pt in boundary_points]
        for pt in result:
            pt["_init_source_type"] = "boundary"
        return result

    # Greedy max-spacing in log-space (dimension-agnostic)
    def log_distance(pt1, pt2):
        return np.sqrt(sum((np.log10(pt1[s]) - np.log10(pt2[s]))**2 for s in SURFACTANTS))

    # Force-select corners first: 2^n candidates (min/max per dimension)
    # For each corner, pick the nearest feasible boundary point
    from itertools import product as _product
    corner_extremes = [[MIN_CONC_MM, simplex_max_conc_mm[s]] for s in SURFACTANTS]
    selected = []
    selected_set = set()
    for corner_vals in _product(*corner_extremes):
        corner = {s: corner_vals[i] for i, s in enumerate(SURFACTANTS)}
        # Find nearest boundary point to this corner in log-space
        best_j = min(range(len(boundary_points)), key=lambda j: log_distance(corner, boundary_points[j]))
        if best_j not in selected_set:
            selected.append(best_j)
            selected_set.add(best_j)
    _logger.info(f"  Corner-seeded: {len(selected)} points from {2**n_surfactants} corners")

    remaining = [j for j in range(len(boundary_points)) if j not in selected_set]
    min_dists = [
        min(log_distance(boundary_points[j], boundary_points[s]) for s in selected)
        for j in remaining
    ]

    while len(selected) < n_boundary_points and remaining:
        best_pos = int(np.argmax(min_dists))
        best_idx = remaining[best_pos]
        selected.append(best_idx)
        remaining.pop(best_pos)
        min_dists.pop(best_pos)
        for k in range(len(remaining)):
            d = log_distance(boundary_points[best_idx], boundary_points[remaining[k]])
            if d < min_dists[k]:
                min_dists[k] = d

    result = []
    for idx in selected:
        pt = boundary_points[idx].copy()
        pt["_init_source_type"] = "boundary"
        result.append(pt)

    _logger.info(f"  Selected {len(result)} boundary points via greedy max-spacing")
    return result


def generate_simplex_init(plans=None, logger=None):
    """Stratified initialization with intelligent boundary wrapping.

    Two layers:

    1. Boundary wrapping (loose outline of source-achievable region)
       Uses coarse grid to detect the source-achievable boundary, then uniformly
       samples points along it. This gives good coverage of the achievable region's
       shape in any dimensionality.

    2. Sobol interior (INIT_INTERIOR_PTS points)
       Quasi-random points sampled inside the simplex to seed interior coverage.

    Duplicate points are removed. All returned points have _init_source_type 
    metadata for plotting (boundary points blue, sobol points red).
    
    Args:
        plans: List of source plans (required for source-achievable checking)
        logger: Optional logger instance
    """
    global _sobol_pool_idx
    import logging as _log
    _logger = logger or _log.getLogger(__name__)

    pts = []
    seen = set()

    def _add(pt, source):
        # Round to 4 sig figs for dedup
        key = tuple(float(f"{pt[s]:.4g}") for s in SURFACTANTS)
        if key not in seen:
            seen.add(key)
            pt["_init_source_type"] = source  # For plotting
            pts.append(pt)

    # Helper: check if point is source-achievable
    def _is_achievable(pt):
        if plans is None:
            return True  # No plans available, assume achievable
        try:
            result = joint_select_sources(pt, plans)
            return result is not None  # None = not achievable
        except:
            return False

    # 1. Boundary wrapping: outline the source-achievable region
    # Scale as LEVELS * n * (n-1) so boundary coverage grows with the manifold dimension:
    # 2D->LEVELS*2, 3D->LEVELS*6, 4D->LEVELS*12
    n_boundary = max(5, INIT_BOUNDARY_LEVELS * n_surfactants * max(1, n_surfactants - 1))
    boundary_pts = generate_achievable_boundary_samples(n_boundary, plans=plans, logger=_logger)
    for pt in boundary_pts:
        _add(pt, "boundary")

    # 2. Interior points — quasi-random Sobol sampling
    if INIT_INTERIOR_PTS > 0:
        if RECOMMENDER_TYPE in ('sobol', 'random'):
            for _ in range(INIT_INTERIOR_PTS):
                if _sobol_pool_idx < len(_sobol_pool):
                    _add(_sobol_pool[_sobol_pool_idx], "sobol")
                    _sobol_pool_idx += 1
        else:
            from scipy.stats.qmc import Sobol as _SobolInt
            seq = _SobolInt(d=n_surfactants, scramble=True, seed=7)
            n_raw = int(2 ** np.ceil(np.log2(max(INIT_INTERIOR_PTS * 4, 16))))
            raw = seq.random(n_raw)
            added = 0
            for row in raw:
                if added >= INIT_INTERIOR_PTS:
                    break
                pt = {s: float(10 ** (np.log10(MIN_CONC_MM) + row[i] * (
                          np.log10(simplex_max_conc_mm[s]) - np.log10(MIN_CONC_MM))))
                      for i, s in enumerate(SURFACTANTS)}
                if _is_achievable(pt):
                    _add(pt, "sobol")
                    added += 1

    return pts


def generate_simplex_boundary_points(levels=INIT_BOUNDARY_LEVELS):
    """Generate geometric boundary-lattice points on the concentration simplex.

    Boundary points are parameterized between MIN_CONC_MM and simplex_max_conc_mm
    using weak compositions in n dimensions. This preserves the intended low-
    concentration corners in pairwise projections while still enforcing the
    global budget constraint via is_feasible().
    """
    levels = max(int(levels), 1)

    def _weak_compositions(total, parts):
        if parts == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for tail in _weak_compositions(total - i, parts - 1):
                yield (i,) + tail

    points = []
    all_candidates = []  # Track all for diagnostics
    for comp in _weak_compositions(levels, n_surfactants):
        pt = {}
        for idx, s in enumerate(SURFACTANTS):
            frac = comp[idx] / levels
            conc_mm = MIN_CONC_MM + frac * (simplex_max_conc_mm[s] - MIN_CONC_MM)
            pt[s] = float(conc_mm)
        all_candidates.append(pt)
        if is_feasible(pt):
            points.append(pt)
    
    # Store diagnostic info for logging
    if not hasattr(generate_simplex_boundary_points, '_diagnostic'):
        generate_simplex_boundary_points._diagnostic = {}
    generate_simplex_boundary_points._diagnostic['candidates_generated'] = len(all_candidates)
    generate_simplex_boundary_points._diagnostic['survived_feasible'] = len(points)
    
    return points


def collect_unique_targets_per_surf(points):
    """Collapse list-of-dicts to {surfactant: sorted unique target concentrations}."""
    targets = {s: set() for s in SURFACTANTS}
    for pt in points:
        for s in SURFACTANTS:
            targets[s].add(pt[s])
    return {s: sorted(targets[s]) for s in SURFACTANTS}


# ================================================================================
# DILUTION PLANNING (per surfactant; reuses 2D helpers)
# ================================================================================

def build_plans_for_surfactants(lash_e, target_concs_per_surf, existing_stock_solutions=None):
    """Run calculate_smart_dilution_plan once per surfactant.

    Returns: dict {surfactant_name: plan} where each plan has 'concentration_map'
    and 'substocks_needed' (matches the structure used by the 2D helpers).
    """
    plans = {}
    for s in SURFACTANTS:
        targets = target_concs_per_surf[s]
        if existing_stock_solutions is not None:
            plans[s] = create_plan_from_existing_stocks(
                existing_stock_solutions, s, targets, lash_e,
                max_volume_ul=max_vol_per_surf_ul,
            )
        else:
            plan, _tracker = calculate_smart_dilution_plan(
                lash_e, s, targets, num_substocks=NUM_SUBSTOCKS,
                max_surfactant_volume_ul=max_vol_per_surf_ul,
            )
            plans[s] = plan
    return plans


def calculate_dilution_recipes_nd(lash_e, plans):
    """Generic N-D version of calculate_dilution_recipes.

    The 2D helper takes (plan_a, plan_b, name_a, name_b) and pairs them into a
    single loop. We invoke it pairwise to avoid duplicating its substock-sourcing
    logic; for N>=2 we feed (plan_i, plan_i) so each surfactant is processed
    independently. Then dedup by Vial_Name.
    """
    all_recipes = []
    seen = set()
    for s in SURFACTANTS:
        recipes = calculate_dilution_recipes(lash_e, plans[s], plans[s], s, s)
        for r in recipes:
            key = r["Vial_Name"]
            if key in seen:
                continue
            seen.add(key)
            all_recipes.append(r)
    return all_recipes


def stock_solutions_needed_from_plans(plans, dilution_recipes):
    """Mirror the 2D 'stock_solutions_needed' list of dicts so it can be passed
    back into create_plan_from_existing_stocks() on subsequent iterations."""
    recipe_lookup = {r["Vial_Name"]: r for r in dilution_recipes}
    out = []
    seen_vials = set()
    for s in SURFACTANTS:
        for substock in plans[s]["substocks_needed"]:
            vial = substock["vial_name"]
            if vial in seen_vials:
                continue
            seen_vials.add(vial)
            r = recipe_lookup.get(vial, {})
            out.append({
                "vial_name": vial,
                "surfactant": s,
                "target_concentration_mm": substock["concentration_mm"],
                "needed_for_concentrations": ", ".join(
                    f"{c:.2e}" for c in substock["needed_for"]),
                "source_vial": r.get("Source_Vial", "Unknown"),
                "source_concentration_mm": r.get("Source_Conc_mM", "Unknown"),
                "source_volume_ml": r.get("Source_Volume_mL", "Unknown"),
                "water_volume_ml": r.get("Water_Volume_mL", "Unknown"),
                "final_volume_ml": r.get("Final_Volume_mL", 7.5),
                "dilution_factor": r.get("Dilution_Factor", "Unknown"),
            })
    return out


# ================================================================================
# WELL RECIPES
# ================================================================================

def diagnose_white_dot(target_concs, plans, logger=None):
    """Detailed diagnostic: explain why this point is infeasible.
    
    For each surfactant, report available sources, then show why joint_select_sources fails.
    """
    import logging as _log
    _logger = logger or _log.getLogger(__name__)
    
    _logger.info(f"  WHITE DOT DIAGNOSTIC: {', '.join(f'{s}={target_concs[s]:.3e}mM' for s in SURFACTANTS)}")
    
    # Show individual source options per surfactant
    for s in SURFACTANTS:
        target = target_concs[s]
        _logger.info(f"    {s} (target={target:.3e} mM):")
        
        # Stock solution volume
        stock_conc = SURFACTANT_LIBRARY[s]["stock_conc"]
        stock_vol = target * WELL_VOLUME_UL / stock_conc
        status = "OK" if MIN_WELL_PIPETTE_VOLUME_UL <= stock_vol <= SURFACTANT_BUDGET_UL + 1.0 else "FAIL"
        _logger.info(f"      Stock ({stock_conc:.2f} mM): {stock_vol:.1f} uL [{status}]")
        
        # Substocks
        for sub in plans[s]["substocks_needed"]:
            src_conc = sub["concentration_mm"]
            if src_conc <= 0:
                continue
            vol = target * WELL_VOLUME_UL / src_conc
            status = "OK" if MIN_WELL_PIPETTE_VOLUME_UL <= vol <= SURFACTANT_BUDGET_UL + 1.0 else "FAIL"
            reason = ""
            if vol < MIN_WELL_PIPETTE_VOLUME_UL:
                reason = f"(too small, needs {MIN_WELL_PIPETTE_VOLUME_UL:.0f} uL min)"
            elif vol > SURFACTANT_BUDGET_UL + 1.0:
                reason = f"(too large, only {SURFACTANT_BUDGET_UL:.0f} uL budget)"
            _logger.info(f"      {sub['vial_name']} ({src_conc:.2f} mM): {vol:.1f} uL [{status}] {reason}")
    
    # Now try joint selection and show the actual error
    try:
        sources = joint_select_sources(target_concs, plans)
        total_vol = sum(sources[s]["volume_needed_ul"] for s in SURFACTANTS)
        _logger.info(f"    JOINT SELECTION RESULT: {total_vol:.1f} uL total [UNEXPECTED SUCCESS?]")
    except ValueError as e:
        _logger.info(f"    JOINT SELECTION FAILED: {e}")


def joint_select_sources(target_concs, plans):

    """Jointly select dispensing sources for all surfactants at one target point.

    Enumerates every combination of available sources (substocks + stock) across
    all surfactants, keeps only combinations whose total volume fits within
    SURFACTANT_BUDGET_UL, and among those picks the one that:
      1. Maximises the minimum individual volume (balance - no surfactant too low)
      2. Tiebreaks by maximising total volume (pipetting accuracy)

    This avoids the single-surfactant greedy approach where each surfactant
    independently picks its highest-volume source, potentially exhausting the
    shared well budget.

    Returns dict {surfactant: solution_dict} with keys:
      vial_name, concentration_mm, volume_needed_ul, volume_needed_ml, is_stock.

    Raises ValueError if no valid joint combination exists for this point.
    """
    import itertools as _it

    all_candidates = []
    for s in SURFACTANTS:
        target = target_concs[s]
        cands = []
        # 1 uL tolerance on the upper bound absorbs floating-point overshoot
        # when simplex axis points are generated via log10/10** arithmetic.
        _vol_upper = SURFACTANT_BUDGET_UL + 1.0
        # Substocks in the plan
        for sub in plans[s]["substocks_needed"]:
            src_conc = sub["concentration_mm"]
            if src_conc <= 0:
                continue
            vol_ul = target * WELL_VOLUME_UL / src_conc
            if MIN_WELL_PIPETTE_VOLUME_UL <= vol_ul <= _vol_upper:
                cands.append({
                    "vial_name": sub["vial_name"],
                    "concentration_mm": src_conc,
                    "volume_needed_ul": min(vol_ul, SURFACTANT_BUDGET_UL),
                    "volume_needed_ml": min(vol_ul, SURFACTANT_BUDGET_UL) / 1000.0,
                    "is_stock": False,
                })
        # Stock solution
        stock_conc = SURFACTANT_LIBRARY[s]["stock_conc"]
        vol_ul = target * WELL_VOLUME_UL / stock_conc
        if MIN_WELL_PIPETTE_VOLUME_UL <= vol_ul <= _vol_upper:
            cands.append({
                "vial_name": f"{s}_stock",
                "concentration_mm": stock_conc,
                "volume_needed_ul": min(vol_ul, SURFACTANT_BUDGET_UL),
                "volume_needed_ml": min(vol_ul, SURFACTANT_BUDGET_UL) / 1000.0,
                "is_stock": True,
            })
        if not cands:
            raise ValueError(
                f"No valid source for {s} at {target:.3e} mM: need volume in "
                f"[{MIN_WELL_PIPETTE_VOLUME_UL:.0f}, {SURFACTANT_BUDGET_UL:.0f}] uL. "
                f"Concentration may be outside the pipettable range."
            )
        all_candidates.append(cands)

    # Joint search: find combination that fits budget and maximises min(vol)
    best_combo = None
    best_score = (-1.0, -1.0)
    for combo in _it.product(*all_candidates):
        total_vol = sum(c["volume_needed_ul"] for c in combo)
        if total_vol > SURFACTANT_BUDGET_UL + 1.0:
            continue
        min_vol = min(c["volume_needed_ul"] for c in combo)
        score = (min_vol, total_vol)  # maximise min first, then total
        if score > best_score:
            best_score = score
            best_combo = combo

    if best_combo is None:
        raise ValueError(
            f"No joint source combination fits budget {SURFACTANT_BUDGET_UL:.0f} uL for: "
            + ", ".join(f"{s}={target_concs[s]:.3e}mM" for s in SURFACTANTS)
        )

    return {s: best_combo[i] for i, s in enumerate(SURFACTANTS)}


def build_well_recipe(target_concs, plans, well_index, replicate=1):
    """Translate one N-D target point into a recipe row.

    Uses joint_select_sources() to pick dispensing sources across all
    surfactants together, ensuring the shared volume budget is respected
    while maximising individual pipette volumes for accuracy.
    """
    recipe = {
        "wellplate_index": int(well_index),
        "well_type": "experiment",
        "control_type": "experiment",
        "replicate": replicate,
    }
    
    # Preserve metadata from target_concs (e.g., _init_source_type for plotting)
    if "_init_source_type" in target_concs:
        recipe["_init_source_type"] = target_concs["_init_source_type"]
    
    total_surf_vol = 0.0
    sources = joint_select_sources(target_concs, plans)
    for s in SURFACTANTS:
        sol = sources[s]
        recipe[f"{s}_conc_mm"] = float(target_concs[s])
        recipe[f"{s}_substock_name"] = sol["vial_name"]
        recipe[f"{s}_substock_conc_mm"] = sol["concentration_mm"]
        recipe[f"{s}_volume_ul"] = float(sol["volume_needed_ul"])
        total_surf_vol += float(sol["volume_needed_ul"])

    water_vol = WELL_VOLUME_UL - PYRENE_VOLUME_UL - total_surf_vol
    if water_vol < 0:
        raise ValueError(
            f"Negative water volume ({water_vol:.2f} uL) at well {well_index}: "
            f"surfactant total {total_surf_vol:.2f} uL exceeds budget "
            f"{SURFACTANT_BUDGET_UL:.2f} uL."
        )
    recipe["water_volume_ul"] = water_vol
    recipe["pyrene_volume_ul"] = PYRENE_VOLUME_UL
    # Buffer disabled in v1; legacy primitives expect this column to exist.
    recipe["buffer_volume_ul"] = 0.0
    recipe["buffer_used"] = None
    return recipe


def plot_1d_cmc_controls(well_recipes_df, output_folder, logger=None):
    """Plot ratio and turbidity vs concentration for each surfactant's 1D CMC controls.

    One column per surfactant, two rows (ratio top, turbidity bottom).
    The known CMC value is marked with a vertical dashed line.
    Saved as 'cmc_1d_controls.png' in output_folder.

    Returns the path to the saved file, or None if no control data is found.
    """
    import logging as _log
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _logger = logger or _log.getLogger(__name__)

    ctrl = well_recipes_df[
        (well_recipes_df['well_type'] == 'control') &
        (well_recipes_df['control_type'].str.startswith('cmc_1d_'))
    ].copy()

    if ctrl.empty:
        _logger.info("No 1D CMC controls found — skipping CMC plot.")
        return None

    n_surfs = len(SURFACTANTS)
    fig, axes = plt.subplots(2, n_surfs, figsize=(4.5 * n_surfs, 7), squeeze=False)
    fig.suptitle("1D CMC Control Series", fontsize=13, y=1.01)

    for col_idx, s in enumerate(SURFACTANTS):
        s_ctrl = ctrl[ctrl['control_type'].str.startswith(f'cmc_1d_{s}_')].copy()
        conc_col = f'{s}_conc_mm'
        if s_ctrl.empty or conc_col not in s_ctrl.columns:
            for row in range(2):
                axes[row][col_idx].set_visible(False)
            continue

        s_ctrl = s_ctrl.sort_values(conc_col)
        concs = s_ctrl[conc_col].values.astype(float)
        cmc_val = SURFACTANT_LIBRARY[s]['cmc_mm']

        for row_idx, (metric, ylabel, color) in enumerate([
            ('ratio', 'I1/I3 Ratio', '#2166ac'),
            ('turbidity_600', 'Turbidity (600 nm)', '#d6604d'),
        ]):
            ax = axes[row_idx][col_idx]
            if metric in s_ctrl.columns:
                vals = s_ctrl[metric].values.astype(float)
                ax.semilogx(concs, vals, 'o-', color=color, lw=1.8, ms=6)
            ax.axvline(cmc_val, color='gray', linestyle='--', lw=1.2, label=f'CMC={cmc_val:.2f} mM')
            ax.set_xlabel(f'{s} (mM)', fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(s if row_idx == 0 else '', fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.25, linestyle='--')

    fig.tight_layout()
    out_path = os.path.join(output_folder, 'cmc_1d_controls.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _select_source_for_1d_control(surf_name, target_conc_mm, plans):
    """Select the best dispensing source for a single-surfactant 1D control well.

    Unlike joint_select_sources, this handles one surfactant at a time so the
    other surfactants can be set to zero volume without raising an error.

    Returns a dict {vial_name, concentration_mm, volume_ul} or None if the
    target concentration is unreachable with available sources.
    """
    best = None
    best_vol = None
    vol_upper = SURFACTANT_BUDGET_UL + 1.0

    for sub in plans[surf_name]["substocks_needed"]:
        src_conc = sub["concentration_mm"]
        if src_conc <= 0:
            continue
        vol_ul = target_conc_mm * WELL_VOLUME_UL / src_conc
        if MIN_WELL_PIPETTE_VOLUME_UL <= vol_ul <= vol_upper:
            vol_ul = min(vol_ul, SURFACTANT_BUDGET_UL)
            if best is None or vol_ul > best_vol:
                best = {
                    "vial_name": sub["vial_name"],
                    "concentration_mm": src_conc,
                    "volume_ul": vol_ul,
                }
                best_vol = vol_ul

    stock_conc = SURFACTANT_LIBRARY[surf_name]["stock_conc"]
    vol_ul = target_conc_mm * WELL_VOLUME_UL / stock_conc
    if MIN_WELL_PIPETTE_VOLUME_UL <= vol_ul <= vol_upper:
        vol_ul = min(vol_ul, SURFACTANT_BUDGET_UL)
        if best is None or vol_ul > best_vol:
            best = {
                "vial_name": f"{surf_name}_stock",
                "concentration_mm": stock_conc,
                "volume_ul": vol_ul,
            }
    return best


def build_control_wells_df(plans, starting_well_index=0):
    """Build control well recipes to populate the first plate.

    Creates two groups of controls governed by ADD_REGULAR_CONTROLS and
    ADD_1D_CONTROLS:

    1. Regular controls (ADD_REGULAR_CONTROLS=True):
       - One water blank (control_type='water_blank')
       - One pure-stock reference well per surfactant (control_type='stock_{s}')

    2. 1D CMC controls (ADD_1D_CONTROLS=True):
       - CMC_CONTROL_POINTS log-spaced wells per surfactant sweeping the CMC
         transition (control_type='cmc_1d_{s}_{i}')

    All rows carry well_type='control' so they are excluded from the active-
    learning recommender data (see get_next_batch filter).  They ARE counted
    toward current_wellplate_wells for plate-rotation bookkeeping but NOT
    toward measured_count (the experiment-only budget).

    Returns a (possibly empty) pd.DataFrame.
    """
    rows = []
    well_idx = starting_well_index

    def _zero_surf_cols():
        """Return zero-filled recipe fragments for every surfactant."""
        d = {}
        for _s in SURFACTANTS:
            d[f"{_s}_conc_mm"] = 0.0
            d[f"{_s}_substock_name"] = None
            d[f"{_s}_substock_conc_mm"] = None
            d[f"{_s}_volume_ul"] = 0.0
        return d

    if ADD_REGULAR_CONTROLS:
        # Water blank
        rec = {
            "wellplate_index": well_idx,
            "well_type": "control",
            "control_type": "water_blank",
            "replicate": 1,
            **_zero_surf_cols(),
            "water_volume_ul": WELL_VOLUME_UL - PYRENE_VOLUME_UL,
            "pyrene_volume_ul": PYRENE_VOLUME_UL,
            "buffer_volume_ul": 0.0,
            "buffer_used": None,
        }
        rows.append(rec)
        well_idx += 1

        # Pure-stock reference per surfactant
        for s in SURFACTANTS:
            stock_conc = SURFACTANT_LIBRARY[s]["stock_conc"]
            rec = {
                "wellplate_index": well_idx,
                "well_type": "control",
                "control_type": f"stock_{s}",
                "replicate": 1,
                **_zero_surf_cols(),
                f"{s}_conc_mm": stock_conc,
                f"{s}_substock_name": f"{s}_stock",
                f"{s}_substock_conc_mm": stock_conc,
                f"{s}_volume_ul": float(STOCK_CONTROL_VOL_UL),
                "water_volume_ul": WELL_VOLUME_UL - PYRENE_VOLUME_UL - STOCK_CONTROL_VOL_UL,
                "pyrene_volume_ul": PYRENE_VOLUME_UL,
                "buffer_volume_ul": 0.0,
                "buffer_used": None,
            }
            rows.append(rec)
            well_idx += 1

    if ADD_1D_CONTROLS:
        for s in SURFACTANTS:
            cmc = SURFACTANT_LIBRARY[s]["cmc_mm"]
            # Log-spaced from CMC/20 to CMC*5, clipped to the active search region.
            c_lo = max(cmc / 20.0, MIN_CONC_MM)
            c_hi = min(cmc * 5.0, max_conc_mm[s])
            if c_lo >= c_hi:
                c_lo = MIN_CONC_MM
                c_hi = max_conc_mm[s]
            concs = np.logspace(np.log10(c_lo), np.log10(c_hi), CMC_CONTROL_POINTS)
            for i, conc in enumerate(concs):
                src = _select_source_for_1d_control(s, conc, plans)
                if src is None:
                    continue  # concentration unreachable with available stocks
                water_vol = WELL_VOLUME_UL - PYRENE_VOLUME_UL - src["volume_ul"]
                if water_vol < 0:
                    continue  # volume infeasible
                rec = {
                    "wellplate_index": well_idx,
                    "well_type": "control",
                    "control_type": f"cmc_1d_{s}_{i + 1}",
                    "replicate": 1,
                    **_zero_surf_cols(),
                    f"{s}_conc_mm": float(conc),
                    f"{s}_substock_name": src["vial_name"],
                    f"{s}_substock_conc_mm": float(src["concentration_mm"]),
                    f"{s}_volume_ul": float(src["volume_ul"]),
                    "water_volume_ul": float(water_vol),
                    "pyrene_volume_ul": PYRENE_VOLUME_UL,
                    "buffer_volume_ul": 0.0,
                    "buffer_used": None,
                }
                rows.append(rec)
                well_idx += 1

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_well_recipes_df(target_points, plans, starting_well_index=0):
    rows = []
    for i, pt in enumerate(target_points):
        rows.append(build_well_recipe(pt, plans, starting_well_index + i))
    return pd.DataFrame(rows)


def filter_points_by_actual_volumes(points, plans, logger=None):
    """Drop any points where no joint source selection can fit within the budget.

    Uses joint_select_sources() which jointly optimises source selection across
    all surfactants.  A point is infeasible only if no combination of available
    sources for every surfactant sums to <= SURFACTANT_BUDGET_UL.

    Returns the filtered list and logs a warning per dropped point.
    Also tracks and reports how many boundary vs interior points are dropped.
    """
    import logging as _log
    _logger = logger or _log.getLogger(__name__)
    feasible = []
    dropped = []
    drop_reasons = {}  # Count reasons for drops
    
    for pt in points:
        try:
            sources = joint_select_sources(pt, plans)
            total_vol = sum(sources[s]["volume_needed_ul"] for s in SURFACTANTS)
            if total_vol <= SURFACTANT_BUDGET_UL + 1.0:  # Match tolerance in joint_select_sources()
                feasible.append(pt)
            else:
                reason = f"vol_exceeded: {total_vol:.1f}uL > {SURFACTANT_BUDGET_UL + 1.0:.1f}uL"
                dropped.append((pt, reason))
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
        except ValueError as exc:
            reason = f"no_sources: {str(exc)[:50]}"
            dropped.append((pt, reason))
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
    
    n_dropped = len(dropped)
    if n_dropped > 0:
        _logger.info(f"Filtered points: {n_dropped}/{len(points)} dropped, {len(feasible)} kept ({100*len(feasible)/len(points):.0f}%)")
        for reason, count in drop_reasons.items():
            _logger.info(f"  {count} dropped: {reason}")
        # Log first 3 dropped points as examples
        _logger.info(f"  Examples of dropped points:")
        for i, (pt, reason) in enumerate(dropped[:3]):
            conc_str = ", ".join(f"{s}={pt[s]:.3e}mM" for s in SURFACTANTS)
            _logger.info(f"    [{i+1}] {conc_str} ({reason})")
        if n_dropped > 3:
            _logger.info(f"  ... and {n_dropped - 3} more dropped points")
    
    return feasible


# ================================================================================
# DISPENSING (per surfactant pass via column-rename shim)
# ================================================================================

def _validate_and_convert_volumes_nd(df):
    """Same as 2D validate_and_convert_recipe_volumes but parameterized over
    SURFACTANTS so it covers all N volume columns."""
    cols = vol_cols + ["water_volume_ul", "buffer_volume_ul", "pyrene_volume_ul"]
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        converted = pd.to_numeric(out[col], errors="coerce")
        if converted.isna().any():
            bad = out[converted.isna()][col].unique()
            raise ValueError(f"Column '{col}' contains non-numeric values: {bad}")
        if (converted < 0).any():
            raise ValueError(f"Column '{col}' contains negative volumes: {converted[converted < 0].values}")
        out[col] = converted
    return out


def _shim_df_for_surfactant(df, surf_name):
    """Build a temporary DataFrame that aliases this surfactant's columns to
    surf_A_* so we can hand it to the legacy 2D dispense helpers unchanged.

    surf_B_* columns are zeroed so the helpers' B-branch is a no-op for this
    pass. Only the A-branch fires.
    """
    shim = df.copy()
    shim["surf_A_volume_ul"] = shim[f"{surf_name}_volume_ul"]
    shim["substock_A_name"] = shim[f"{surf_name}_substock_name"]
    shim["substock_A_conc_mm"] = shim[f"{surf_name}_substock_conc_mm"]
    shim["surf_B_volume_ul"] = 0.0
    shim["substock_B_name"] = None
    shim["substock_B_conc_mm"] = None
    return shim


def execute_dispensing_nd(lash_e, well_recipes_df):
    """N-D analog of execute_dispensing.

    Called once per plate (the orchestrator ensures well_recipes_df contains
    only the wells for the current plate, so no internal batching is needed).
    For each surfactant: position its substock vials, dispense, return home.
    Then dispense water split across two water vials for tip-life balance.
    """
    lash_e.logger.info(f"Executing N-D dispensing for {len(well_recipes_df)} wells...")
    well_recipes_df = _validate_and_convert_volumes_nd(well_recipes_df)

    for s in SURFACTANTS:
        shim = _shim_df_for_surfactant(well_recipes_df, s)
        vials_for_s = shim[shim["surf_A_volume_ul"] > 0]["substock_A_name"].dropna().unique()
        if len(vials_for_s) == 0:
            continue
        sorted_vials = position_surfactant_vials_by_concentration(
            lash_e, vials_for_s, shim, "A",
        )
        for i, vial in enumerate(sorted_vials):
            should_condition = (i == 0)
            dispense_component_to_wellplate(
                lash_e, shim, vial, "SDS", "surf_A_volume_ul",
                should_condition_tip=should_condition,
            )
        lash_e.nr_robot.remove_pipet()
        return_surfactant_vials_home(lash_e, sorted_vials, "A")

    # Water makeup (split across water + water_2 for tip-life balance)
    water_wells = well_recipes_df[well_recipes_df["water_volume_ul"] > 0]
    if len(water_wells) > 0:
        water_wells = water_wells.sort_values("water_volume_ul", ascending=True)
        mid = len(water_wells) // 2
        w1 = water_wells.iloc[:mid]
        w2 = water_wells.iloc[mid:]
        lash_e.nr_robot.move_vial_to_location("water", "main_8mL_rack", 44)
        lash_e.nr_robot.move_vial_to_location("water_2", "main_8mL_rack", 45)
        if len(w1) > 0:
            dispense_component_to_wellplate(lash_e, w1, "water", "water", "water_volume_ul")
        if len(w2) > 0:
            dispense_component_to_wellplate(lash_e, w2, "water_2", "water", "water_volume_ul")
        lash_e.nr_robot.remove_pipet()
        return_water_vial_home(lash_e, "water")
        return_water_vial_home(lash_e, "water_2")

    return well_recipes_df


# ================================================================================
# MEASUREMENT (delegates to legacy primitives + N-D simulator)
# ================================================================================

def simulate_measurements_nd(target_concs):
    """Physically motivated N-D simulator.

    Turbidity: catanionic precipitation model.
      Peaks sharply when sum(anionic concs) == sum(cationic concs).
      Width controlled by SIGMA_LOG (in log10 space, ~0.3 = within factor of 2).
      Suppressed at very low total concentration.

    Ratio: mixed-CMC pyrene probe.
      High (~0.85) in aqueous; drops to ~0.70 as Sigma(c_i / CMC_i) exceeds 1.
      Broad, continuous transition across the whole space.
    """
    SIGMA_LOG = 0.35   # catanionic band width in log10 space

    # Classify surfactants by charge
    sum_anionic = 1e-10
    sum_cationic = 1e-10
    for s in SURFACTANTS:
        c = target_concs[s]
        cat = SURFACTANT_LIBRARY[s].get("category", "nonionic")
        if cat == "anionic":
            sum_anionic += c
        elif cat == "cationic":
            sum_cationic += c

    # --- Turbidity: Gaussian in log-charge-ratio space ---
    log_imbalance = np.log10(sum_anionic / sum_cationic)   # 0 at charge balance
    turb_peak = np.exp(-(log_imbalance / SIGMA_LOG) ** 2)
    # Suppress at very low total concentration (need > ~0.5 mM total to precipitate)
    total_conc = sum_anionic + sum_cationic
    conc_gate = 1.0 / (1.0 + np.exp(-4.0 * (np.log10(total_conc) - np.log10(0.5))))
    simulated_turbidity = 0.04 + (1.0 - 0.04) * turb_peak * conc_gate

    # --- Ratio: mixed-CMC transition (broad) ---
    # Sigma(c_i / CMC_i) = 1 defines the mixed CMC surface.
    # Ratio drops as this sum rises above 1.
    mixed_cmc_sum = sum(
        target_concs[s] / SURFACTANT_LIBRARY[s]["cmc_mm"]
        for s in SURFACTANTS
    )
    # Sigmoid centred at mixed_cmc_sum = 1; width ~0.5 decades
    ratio_drop = 1.0 / (1.0 + np.exp(-5.0 * (np.log10(max(mixed_cmc_sum, 1e-9)))))
    simulated_ratio = 0.85 - (0.85 - 0.70) * ratio_drop

    # Noise
    rng = np.random.default_rng()   # thread-safe, no global state
    simulated_ratio *= 1.0 + rng.normal(0, 0.015)
    simulated_turbidity *= 1.0 + rng.normal(0, 0.025)
    simulated_ratio = float(np.clip(simulated_ratio, 0.70, 0.95))
    simulated_turbidity = float(np.clip(simulated_turbidity, 0.02, 1.0))

    # Fluorescence backfill (consistent with ratio)
    f384 = float(95.0 + rng.normal(0, 1.0))
    f373 = simulated_ratio * f384

    return {
        "turbidity_600": round(simulated_turbidity, 4),
        "fluorescence_334_373": round(f373, 2),
        "fluorescence_334_384": round(f384, 2),
        "ratio": round(simulated_ratio, 4),
    }


def measure_and_process_turbidity_nd(lash_e, well_recipes_df):
    """Measure turbidity for all wells in well_recipes_df (one plate at a time).

    In hardware mode, delegates to the 2D measure_turbidity primitive.
    In simulation, generates measurements from simulate_measurements_nd.
    """
    lash_e.logger.info(f"Measuring turbidity for {len(well_recipes_df)} wells...")
    if "turbidity_600" not in well_recipes_df.columns:
        well_recipes_df["turbidity_600"] = None

    wells = well_recipes_df["wellplate_index"].tolist()

    if lash_e.simulate:
        for _, row in well_recipes_df.iterrows():
            target_concs = {s: float(row[f"{s}_conc_mm"]) for s in SURFACTANTS}
            sim = simulate_measurements_nd(target_concs)
            idx = int(row["wellplate_index"])
            well_recipes_df.loc[
                well_recipes_df["wellplate_index"] == idx, "turbidity_600"
            ] = sim["turbidity_600"]
        lash_e.logger.info(f"  [SIMULATED] turbidity for {len(wells)} wells")
    else:
        data = measure_turbidity(
            lash_e, wells, well_recipes_df, shake_and_wait=True, return_wellplate=False,
        )
        if data is not None and "wellplate_index" in data.columns:
            col = "turbidity_600" if "turbidity_600" in data.columns else data.columns[-1]
            for _, drow in data.iterrows():
                idx = int(drow["wellplate_index"])
                well_recipes_df.loc[
                    well_recipes_df["wellplate_index"] == idx, "turbidity_600"
                ] = drow[col]
    return well_recipes_df


def measure_and_process_fluorescence_nd(lash_e, well_recipes_df):
    """Measure fluorescence for all wells in well_recipes_df (one plate at a time)."""
    lash_e.logger.info(f"Measuring fluorescence for {len(well_recipes_df)} wells...")
    for col in ("fluorescence_334_373", "fluorescence_334_384", "ratio"):
        if col not in well_recipes_df.columns:
            well_recipes_df[col] = None

    wells = well_recipes_df["wellplate_index"].tolist()

    if lash_e.simulate:
        for _, row in well_recipes_df.iterrows():
            target_concs = {s: float(row[f"{s}_conc_mm"]) for s in SURFACTANTS}
            sim = simulate_measurements_nd(target_concs)
            idx = int(row["wellplate_index"])
            mask = well_recipes_df["wellplate_index"] == idx
            well_recipes_df.loc[mask, "fluorescence_334_373"] = sim["fluorescence_334_373"]
            well_recipes_df.loc[mask, "fluorescence_334_384"] = sim["fluorescence_334_384"]
            well_recipes_df.loc[mask, "ratio"] = sim["ratio"]
        lash_e.logger.info(f"  [SIMULATED] fluorescence for {len(wells)} wells")
    else:
        data = measure_fluorescence(
            lash_e, wells, well_recipes_df, shake_and_wait=False, return_wellplate=True,
        )
        if data is not None and "wellplate_index" in data.columns:
            for _, drow in data.iterrows():
                idx = int(drow["wellplate_index"])
                mask = well_recipes_df["wellplate_index"] == idx
                if "fluorescence_334_373" in data.columns:
                    well_recipes_df.loc[mask, "fluorescence_334_373"] = drow["fluorescence_334_373"]
                if "fluorescence_334_384" in data.columns:
                    well_recipes_df.loc[mask, "fluorescence_334_384"] = drow["fluorescence_334_384"]
                if "ratio" in data.columns:
                    well_recipes_df.loc[mask, "ratio"] = drow["ratio"]
    return well_recipes_df


# ================================================================================
# RECOMMENDER (N-D capable)
# ================================================================================

def _plot_delaunay_decisions(viz_state, surviving_points, iteration, output_folder):
    """2-panel Delaunay iteration plot showing only the post-filter selected points.
    Called from the workflow loop after volume filtering."""
    import matplotlib.pyplot as plt
    b_min = viz_state['b_min']
    b_max = viz_state['b_max']
    X_norm = viz_state['X_norm']
    Y_std = viz_state['Y_std']
    tri = viz_state['tri']
    scores = viz_state['scores']
    centroids = viz_state['centroids']
    n_existing = viz_state['n_existing']
    input_columns = viz_state['input_columns']
    output_columns = viz_state['output_columns']

    # Denormalize: [0,1] -> log10 space
    log_X = X_norm * (b_max - b_min) + b_min
    log_centroids = centroids * (b_max - b_min) + b_min

    # Convert surviving concentration dicts to log10 coords
    if surviving_points:
        surfs = list(surviving_points[0].keys())
        surviving_log = np.array([
            [np.log10(pt[s]) for s in surfs]
            for pt in surviving_points
        ])
    else:
        surviving_log = np.empty((0, 2))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f'Delaunay Triangle Decisions — Iteration {iteration} '
        f'({n_existing} pts, {len(surviving_log)} selected)',
        fontsize=13,
    )

    for ax in axes:
        ax.triplot(log_X[:, 0], log_X[:, 1], tri.simplices,
                   'k-', alpha=0.25, linewidth=0.5)

    # Left: existing points coloured by first output
    ax1 = axes[0]
    sc1 = ax1.scatter(log_X[:, 0], log_X[:, 1], c=Y_std[:, 0],
                      cmap='coolwarm', s=50, edgecolors='k',
                      linewidth=0.4, zorder=3)
    plt.colorbar(sc1, ax=ax1, label=f'{output_columns[0]} (std)')
    if len(surviving_log):
        ax1.scatter(surviving_log[:, 0], surviving_log[:, 1],
                    c='red', s=180, marker='*', edgecolors='k',
                    linewidth=0.8, label='Selected', zorder=5)
    ax1.set_xlabel(f'log10({input_columns[0]})')
    ax1.set_ylabel(f'log10({input_columns[1]})')
    ax1.set_title('Output values + selected centroids')
    ax1.legend(fontsize=8)

    # Right: all simplex scores
    ax2 = axes[1]
    sc2 = ax2.scatter(log_centroids[:, 0], log_centroids[:, 1],
                      c=scores, cmap='plasma', s=60, alpha=0.7,
                      edgecolors='k', linewidth=0.3)
    plt.colorbar(sc2, ax=ax2, label='Simplex score')
    if len(surviving_log):
        ax2.scatter(surviving_log[:, 0], surviving_log[:, 1],
                    c='red', s=180, marker='*', edgecolors='k',
                    linewidth=0.8, label='Selected', zorder=5)
    ax2.set_xlabel(f'log10({input_columns[0]})')
    ax2.set_ylabel(f'log10({input_columns[1]})')
    ax2.set_title('Simplex scores (all triangles)')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_dir = os.path.join(output_folder, 'triangle_decisions')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'triangle_iter{iteration}_{ts}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_bayesian_decisions(viz_state, surviving_points, iteration, output_folder):
    """Multi-panel Bayesian iteration plot showing per-output contrast maps
    and the final selected points after volume filtering.

    Panels (per output + 1 total):
      - One panel per output: GP posterior mean (existing pts) + contrast heatmap
      - Final panel: total contrast heatmap + selected points marked as stars
    Called from the workflow loop after volume filtering.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    b_min = viz_state['b_min']
    b_max = viz_state['b_max']
    X_pool = viz_state['X_pool']                          # (N, 2) normalized
    contrast_per_output = viz_state['contrast_per_output']  # (N, n_out)
    contrast_total = viz_state['contrast_total']           # (N,)
    mean_at_pool = viz_state['mean_at_pool']               # (N, n_out)
    X_existing = viz_state['X_existing']                   # (n_existing, 2)
    X_selected = viz_state['X_selected']                   # (n_picks, 2)
    input_columns = viz_state['input_columns']
    output_columns = viz_state['output_columns']
    n_out = len(output_columns)

    def denorm(X_norm):
        """[0,1]^2 -> log10 space."""
        return X_norm * (b_max - b_min) + b_min

    log_pool = denorm(X_pool)
    log_existing = denorm(X_existing) if len(X_existing) else np.empty((0, 2))
    log_selected = denorm(X_selected) if len(X_selected) else np.empty((0, 2))

    # Surviving concentration dicts -> log10 coords (post-volume-filter)
    if surviving_points:
        surfs = list(surviving_points[0].keys())
        surviving_log = np.array([
            [np.log10(pt[s]) for s in surfs]
            for pt in surviving_points
        ])
    else:
        surviving_log = np.empty((0, 2))

    n_panels = n_out * 2 + 1   # per-output: (GP mean | contrast) + total contrast
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(
        f'Bayesian Decisions — Iteration {iteration} '
        f'({len(X_existing)} existing pts, {len(surviving_log)} selected)',
        fontsize=12,
    )

    scatter_kw = dict(s=8, alpha=0.5, linewidths=0)
    existing_kw = dict(c='white', s=40, edgecolors='k', linewidth=0.6, zorder=4)
    star_kw = dict(c='red', s=200, marker='*', edgecolors='k',
                   linewidth=0.8, zorder=5, label='Selected')

    ax_idx = 0
    for oi, col in enumerate(output_columns):
        # --- GP posterior mean panel ---
        ax = axes[ax_idx]; ax_idx += 1
        sc = ax.scatter(log_pool[:, 0], log_pool[:, 1],
                        c=mean_at_pool[:, oi], cmap='viridis',
                        norm=Normalize(), **scatter_kw)
        plt.colorbar(sc, ax=ax, label=f'{col} GP mean (std)')
        if len(log_existing):
            ax.scatter(log_existing[:, 0], log_existing[:, 1], **existing_kw)
        if len(surviving_log):
            ax.scatter(surviving_log[:, 0], surviving_log[:, 1], **star_kw)
        ax.set_xlabel(f'log10({input_columns[0]})')
        ax.set_ylabel(f'log10({input_columns[1]})')
        ax.set_title(f'{col}: GP posterior mean')
        ax.legend(fontsize=7)

        # --- Per-output contrast panel ---
        ax = axes[ax_idx]; ax_idx += 1
        sc = ax.scatter(log_pool[:, 0], log_pool[:, 1],
                        c=contrast_per_output[:, oi], cmap='plasma',
                        norm=Normalize(), **scatter_kw)
        plt.colorbar(sc, ax=ax, label=f'{col} contrast')
        if len(log_existing):
            ax.scatter(log_existing[:, 0], log_existing[:, 1], **existing_kw)
        if len(surviving_log):
            ax.scatter(surviving_log[:, 0], surviving_log[:, 1], **star_kw)
        ax.set_xlabel(f'log10({input_columns[0]})')
        ax.set_ylabel(f'log10({input_columns[1]})')
        ax.set_title(f'{col}: local contrast (boundary signal)')
        ax.legend(fontsize=7)

    # --- Total contrast + selections ---
    ax = axes[ax_idx]
    sc = ax.scatter(log_pool[:, 0], log_pool[:, 1],
                    c=contrast_total, cmap='inferno',
                    norm=Normalize(), **scatter_kw)
    plt.colorbar(sc, ax=ax, label='Total contrast (sum outputs)')
    if len(log_existing):
        ax.scatter(log_existing[:, 0], log_existing[:, 1], **existing_kw)
    if len(surviving_log):
        ax.scatter(surviving_log[:, 0], surviving_log[:, 1], **star_kw)
    ax.set_xlabel(f'log10({input_columns[0]})')
    ax.set_ylabel(f'log10({input_columns[1]})')
    ax.set_title('Total contrast + final selections')
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_dir = os.path.join(output_folder, 'bayesian_decisions')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'bayesian_iter{iteration}_{ts}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_gp_decisions(viz_state, surviving_points, iteration, output_folder,
                       subfolder='gp_decisions'):
    """Shared acquisition-map plot for gradient and levelset recommenders.

    Panels (per 2D input space):
      - If GP mean is available (mean_at_pool in viz_state): one GP-mean panel
        per output showing what the model predicts — makes the boundary visible
        as a transition in model output.
      - One acquisition heatmap per output.
      - One total acquisition panel with final selected stars.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    b_min = viz_state['b_min']
    b_max = viz_state['b_max']
    X_pool = viz_state['X_pool']                    # (N, 2) normalized
    acq_per_output = viz_state['acq_per_output']    # (N, n_out)
    acq_total = viz_state['acq_total']              # (N,)
    mean_at_pool = viz_state.get('mean_at_pool')    # (N, n_out) or None
    X_existing = viz_state['X_existing']
    X_selected = viz_state['X_selected']
    input_columns = viz_state['input_columns']
    output_columns = viz_state['output_columns']
    acq_label = viz_state.get('acq_label', 'acquisition')
    n_out = len(output_columns)

    def denorm(X_norm):
        return X_norm * (b_max - b_min) + b_min

    log_pool = denorm(X_pool)
    log_existing = denorm(X_existing) if len(X_existing) else np.empty((0, 2))
    log_selected = denorm(X_selected) if len(X_selected) else np.empty((0, 2))

    if surviving_points:
        surfs = list(surviving_points[0].keys())
        surviving_log = np.array([
            [np.log10(pt[s]) for s in surfs]
            for pt in surviving_points
        ])
    else:
        surviving_log = np.empty((0, 2))

    has_mean = mean_at_pool is not None
    # Panels: [GP mean per output (if available)] + [acq per output] + [total acq]
    n_panels = (n_out if has_mean else 0) + n_out + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(
        f'{acq_label.title()} Decisions — Iteration {iteration} '
        f'({len(X_existing)} existing pts, {len(surviving_log)} selected)',
        fontsize=12,
    )

    scatter_kw = dict(s=8, alpha=0.5, linewidths=0)
    existing_kw = dict(c='white', s=40, edgecolors='k', linewidth=0.6, zorder=4)
    star_kw = dict(c='red', s=200, marker='*', edgecolors='k',
                   linewidth=0.8, zorder=5, label='Selected')

    ax_idx = 0

    # GP mean panels (show what the model predicts — boundary = sharp color transition)
    if has_mean:
        for oi, col in enumerate(output_columns):
            ax = axes[ax_idx]; ax_idx += 1
            sc = ax.scatter(log_pool[:, 0], log_pool[:, 1],
                            c=mean_at_pool[:, oi], cmap='viridis',
                            norm=Normalize(), **scatter_kw)
            plt.colorbar(sc, ax=ax, label=f'{col} GP mean (std-space)')
            if len(log_existing):
                ax.scatter(log_existing[:, 0], log_existing[:, 1], **existing_kw)
            if len(surviving_log):
                ax.scatter(surviving_log[:, 0], surviving_log[:, 1], **star_kw)
            ax.set_xlabel(f'log10({input_columns[0]})')
            ax.set_ylabel(f'log10({input_columns[1]})')
            ax.set_title(f'{col}: GP predicted mean\n(boundary = color transition)')
            ax.legend(fontsize=7)

    for oi, col in enumerate(output_columns):
        ax = axes[ax_idx]; ax_idx += 1
        sc = ax.scatter(log_pool[:, 0], log_pool[:, 1],
                        c=acq_per_output[:, oi], cmap='plasma',
                        norm=Normalize(), **scatter_kw)
        plt.colorbar(sc, ax=ax, label=f'{col} {acq_label}')
        if len(log_existing):
            ax.scatter(log_existing[:, 0], log_existing[:, 1], **existing_kw)
        if len(surviving_log):
            ax.scatter(surviving_log[:, 0], surviving_log[:, 1], **star_kw)
        ax.set_xlabel(f'log10({input_columns[0]})')
        ax.set_ylabel(f'log10({input_columns[1]})')
        ax.set_title(f'{col}: {acq_label}')
        ax.legend(fontsize=7)

    ax = axes[ax_idx]
    sc = ax.scatter(log_pool[:, 0], log_pool[:, 1],
                    c=acq_total, cmap='inferno',
                    norm=Normalize(), **scatter_kw)
    plt.colorbar(sc, ax=ax, label=f'Total {acq_label}')
    if len(log_existing):
        ax.scatter(log_existing[:, 0], log_existing[:, 1], **existing_kw)
    if len(surviving_log):
        ax.scatter(surviving_log[:, 0], surviving_log[:, 1], **star_kw)
    ax.set_xlabel(f'log10({input_columns[0]})')
    ax.set_ylabel(f'log10({input_columns[1]})')
    ax.set_title(f'Total {acq_label} + final selections')
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_dir = os.path.join(output_folder, subfolder)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'iter{iteration}_{ts}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_nd_decisions(viz_state, surviving_points, iteration, output_folder):
    """N-D acquisition visualization for 3D+ spaces.

    Produces two figures per iteration saved to {output_folder}/nd_decisions/:

    1. pairwise_iter{N}.png
       One scatter panel per pair of input dimensions, pool points colored
       by total acquisition score. Shows which 2D projection of concentration
       space is being targeted — if the algorithm is tracing a boundary,
       you'll see a concentrated band in every pairwise view.

    2. parallel_iter{N}.png
       Each selected point is a polyline across all input dimensions.
       Lines are colored by the acquisition value at that point
       (hot = algorithm was most confident it's on the boundary).
       Existing data shown in gray. This makes it easy to see whether
       selections are converging on a narrow region across all dimensions.
    """
    import itertools
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from scipy.spatial.distance import cdist as _cdist

    b_min = viz_state['b_min']   # (d,) log10 space
    b_max = viz_state['b_max']   # (d,) log10 space
    X_pool_norm = viz_state['X_pool']
    X_existing_norm = viz_state['X_existing']
    X_selected_norm = viz_state['X_selected']
    input_columns = viz_state['input_columns']
    # Support both naming conventions (bayesian vs gradient/levelset)
    acq_total = viz_state.get('acq_total', viz_state.get('contrast_total'))
    acq_label = viz_state.get('acq_label', 'contrast')
    n_dim = X_pool_norm.shape[1]

    def denorm(X_norm):
        return X_norm * (b_max - b_min) + b_min

    log_pool = denorm(X_pool_norm)
    log_existing = denorm(X_existing_norm) if len(X_existing_norm) else np.empty((0, n_dim))
    log_selected = denorm(X_selected_norm) if len(X_selected_norm) else np.empty((0, n_dim))

    if surviving_points:
        surf_keys = input_columns
        surviving_log = np.array([
            [np.log10(max(pt[s], 1e-12)) for s in surf_keys]
            for pt in surviving_points
        ])
    else:
        surviving_log = np.empty((0, n_dim))

    out_dir = os.path.join(output_folder, 'nd_decisions')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    paths = []

    # ---- Figure 1: Pairwise projections ----
    pairs = list(itertools.combinations(range(n_dim), 2))
    n_pairs = len(pairs)

    # Subsample pool for speed (50k points is too many to render well)
    MAX_PLOT = 5000
    if len(log_pool) > MAX_PLOT:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(log_pool), MAX_PLOT, replace=False)
        log_pool_plot = log_pool[idx]
        acq_plot = acq_total[idx]
    else:
        log_pool_plot = log_pool
        acq_plot = acq_total

    vmin = float(np.percentile(acq_total, 5))
    vmax = float(np.percentile(acq_total, 95))
    norm_acq = Normalize(vmin=vmin, vmax=vmax)

    ncols = min(n_pairs, 3)
    nrows = int(np.ceil(n_pairs / ncols))
    fig1, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    fig1.suptitle(
        f'Pairwise Projections ({acq_label}) — Iteration {iteration}  '
        f'({len(log_existing)} existing, {len(surviving_log)} selected)',
        fontsize=12,
    )

    for pi, (i, j) in enumerate(pairs):
        ax = axes[pi // ncols][pi % ncols]
        sc = ax.scatter(log_pool_plot[:, i], log_pool_plot[:, j],
                        c=acq_plot, cmap='plasma', norm=norm_acq,
                        s=6, alpha=0.45, linewidths=0, rasterized=True)
        plt.colorbar(sc, ax=ax, label=acq_label)
        if len(log_existing):
            ax.scatter(log_existing[:, i], log_existing[:, j],
                       c='white', s=40, edgecolors='k', linewidth=0.6, zorder=4,
                       label='Existing')
        if len(surviving_log):
            ax.scatter(surviving_log[:, i], surviving_log[:, j],
                       c='red', s=180, marker='*', edgecolors='k',
                       linewidth=0.7, zorder=5, label='Selected')
        ax.set_xlabel(f'log10({input_columns[i]})')
        ax.set_ylabel(f'log10({input_columns[j]})')
        ax.set_title(f'{input_columns[i]} vs {input_columns[j]}')
        ax.legend(fontsize=7)

    for pi in range(n_pairs, nrows * ncols):
        axes[pi // ncols][pi % ncols].set_visible(False)

    plt.tight_layout()
    path1 = os.path.join(out_dir, f'pairwise_iter{iteration}_{ts}.png')
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    paths.append(path1)

    # ---- Figure 2: Parallel coordinates ----
    fig2, ax2 = plt.subplots(figsize=(max(8, 2.5 * n_dim), 5))

    log_all = (np.vstack([log_existing, log_selected])
               if len(log_existing) else log_selected)
    if len(log_all) == 0:
        plt.close(fig2)
        return paths

    col_min = log_all.min(axis=0)
    col_max = log_all.max(axis=0)
    col_range = np.where(col_max > col_min, col_max - col_min, 1.0)
    x_ticks = np.arange(n_dim, dtype=float)

    # Existing points in gray
    for row in log_existing:
        y = (row - col_min) / col_range
        ax2.plot(x_ticks, y, color='lightgray', linewidth=0.8, alpha=0.6, zorder=1)

    # Selected points colored by acquisition value at nearest pool point
    if len(log_selected) > 0 and len(log_pool) > 0:
        dists = _cdist(log_selected, log_pool)
        nearest_idx = dists.argmin(axis=1)
        sel_acq = acq_total[nearest_idx]
        norm_sel = Normalize(vmin=float(sel_acq.min()), vmax=float(sel_acq.max()))
        cmap_sel = cm.get_cmap('plasma')
        for k, row in enumerate(log_selected):
            y = (row - col_min) / col_range
            color = cmap_sel(norm_sel(sel_acq[k]))
            ax2.plot(x_ticks, y, color=color, linewidth=2.5, alpha=0.9, zorder=3)
            ax2.scatter(x_ticks, y, color=color, s=50, zorder=4, linewidths=0)
        sm = cm.ScalarMappable(cmap='plasma', norm=norm_sel)
        sm.set_array([])
        plt.colorbar(sm, ax=ax2, label=f'{acq_label} at selected point')

    # Post-filter surviving points as red stars
    if len(surviving_log):
        for row in surviving_log:
            y = (row - col_min) / col_range
            ax2.plot(x_ticks, y, 'r-', linewidth=1.5, alpha=0.75, zorder=5)
            ax2.scatter(x_ticks, y, c='red', s=80, marker='*', zorder=6)

    # Vertical axis lines and concentration labels
    for di in range(n_dim):
        ax2.axvline(di, color='black', linewidth=0.5, alpha=0.3)
        # Show actual concentration (mM) at bottom and top of each axis
        ax2.text(di, -0.07, f'{10**col_min[di]:.3g} mM', ha='center', va='top',
                 fontsize=7, transform=ax2.get_xaxis_transform())
        ax2.text(di, 1.07, f'{10**col_max[di]:.3g} mM', ha='center', va='bottom',
                 fontsize=7, transform=ax2.get_xaxis_transform())

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(input_columns, fontsize=10)
    ax2.set_ylabel('Normalized value (per axis)', fontsize=9)
    ax2.set_ylim(-0.12, 1.12)
    ax2.set_title(
        f'Parallel Coordinates — Iteration {iteration}  '
        f'(gray=existing, colored=selected by {acq_label}, red*=post-filter)',
        fontsize=11,
    )

    # Add legend patches
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color='lightgray', label='Existing data'),
        mpatches.Patch(color='red', label='Post-filter selected'),
    ]
    ax2.legend(handles=handles, fontsize=8, loc='upper right')

    plt.tight_layout()
    path2 = os.path.join(out_dir, f'parallel_iter{iteration}_{ts}.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    paths.append(path2)

    return paths


def get_next_batch(experiment_data_df, n_points=GRADIENT_SUGGESTIONS_PER_ITERATION,
                   iteration=None, output_dir=None):
    """Run the configured N-D recommender. Returns a list of dicts
    {surfactant_name: target_conc_mm} of length <= n_points.

    All transition recommenders inherit from TransitionRecommenderBase and
    expose the same get_recommendations(data_df, n_points) interface.
    Sobol and Random are non-adaptive baselines that ignore observed data.

    RECOMMENDER_TYPE options
    ------------------------
    'triangle'  : DelaunaySimplexTransitionRecommender - geometric, no GP
    'bayesian'  : BayesianTransitionRecommender        - GP + local contrast
    'gradient'  : GradientTransitionRecommender        - GP, follows gradient-UCB
    'levelset'  : LevelSetTransitionRecommender        - GP, finds phase boundaries
    'sobol'     : Sobol quasi-random sequence           - pure space-filling baseline
    'random'    : Uniform random                        - pure random baseline

    Turbidity gates (applied before passing data to recommender)
    ------------------------------------------------------------
    1. Flat-turbidity gate: if max(turbidity_600) <= FLAT_TURBIDITY_MAX across all
       data, turbidity_600 is dropped from output_columns (ratio-only mode).
    2. Per-point reliability mask: if FILTER_UNRELIABLE_RATIOS is True, ratio is
       set to NaN for wells where turbidity_600 > TURBIDITY_FILTER_THRESHOLD so
       those points are excluded from ratio GP fitting.
    """
    # --- build output_columns and clean data copy for recommender ---
    # Re-derive from SURFACTANTS at call time — ConfigManager may have updated
    # SURFACTANTS in globals after the module-level input_cols was computed.
    _input_cols = [f"{s}_conc_mm" for s in SURFACTANTS]
    _n_surfs = len(SURFACTANTS)

    # Filter controls out before passing data to the recommender.  Controls have
    # well_type='control' and must not bias the active-learning model.
    import logging as _log_ctrl
    if 'well_type' in experiment_data_df.columns:
        rec_data = experiment_data_df[experiment_data_df['well_type'] == 'experiment'].copy()
        n_filtered = len(experiment_data_df) - len(rec_data)
        if n_filtered > 0:
            _log_ctrl.getLogger(__name__).info(
                f"get_next_batch: excluded {n_filtered} control wells from recommender "
                f"({len(rec_data)} experiment wells used)"
            )
    else:
        rec_data = experiment_data_df.copy()

    output_columns = ["ratio", "turbidity_600"]

    if OUTPUT_COLUMNS_OVERRIDE is not None:
        # Manual override — skip all automatic turbidity/ratio logic.
        output_columns = list(OUTPUT_COLUMNS_OVERRIDE)
        import logging as _logging
        _logging.getLogger(__name__).info(
            f"OUTPUT_COLUMNS_OVERRIDE active: using {output_columns}"
        )
    elif "turbidity_600" in rec_data.columns and rec_data["turbidity_600"].notna().any():
        max_turb = rec_data["turbidity_600"].max()
        if max_turb <= FLAT_TURBIDITY_MAX:
            # Turbidity landscape is flat - drop it so the recommender isn't
            # distracted by noise masquerading as signal.
            output_columns = ["ratio"]
            import logging as _logging
            _logging.getLogger(__name__).info(
                f"Flat turbidity gate: max={max_turb:.4f} <= {FLAT_TURBIDITY_MAX} "
                f"- dropping turbidity_600 from recommender outputs (ratio-only mode)"
            )
        elif FILTER_UNRELIABLE_RATIOS:
            # Per-point mask: null out ratio for high-turbidity wells so the ratio
            # GP is not corrupted by unreliable fluorescence readings.  The
            # base-class _prepare_data will drop those rows from ratio fitting.
            mask = rec_data["turbidity_600"] > TURBIDITY_FILTER_THRESHOLD
            n_masked = int(mask.sum())
            if n_masked > 0:
                rec_data.loc[mask, "ratio"] = float("nan")
                import logging as _logging
                _logging.getLogger(__name__).info(
                    f"Turbidity mask: {n_masked} wells with turbidity_600 > "
                    f"{TURBIDITY_FILTER_THRESHOLD} have ratio set to NaN "
                    f"(excluded from ratio GP)"
                )
    else:
        # No turbidity data yet - fall back to ratio only
        output_columns = ["ratio"]

    # Scale candidate pool with dimensionality to maintain ~37 pts/axis in 3D.
    # 2D: 50k, 3D: 50k, 4D: 200k, 5D: 500k
    _candidate_pool = max(50000, int(37 ** _n_surfs))

    if RECOMMENDER_TYPE in ("delaunay", "triangle"):
        from recommenders.delaunay_simplex_recommender import (
            DelaunaySimplexTransitionRecommender,
        )
        recommender = DelaunaySimplexTransitionRecommender(
            input_columns=_input_cols,
            output_columns=output_columns,
            log_transform_inputs=True,
        )
        if output_dir:
            recommender._output_dir = os.path.join(output_dir, 'triangle_decisions')
    elif RECOMMENDER_TYPE == "bayesian":
        from recommenders.bayesian_transition_recommender import (
            BayesianTransitionRecommender,
        )
        recommender = BayesianTransitionRecommender(
            input_columns=_input_cols,
            output_columns=output_columns,
            log_transform_inputs=True,
            delta=0.03,
            K=24,
            candidate_pool=_candidate_pool,
            alpha_spacing=BAYESIAN_ALPHA_SPACING,
        )
        recommender.turbidity_penalty_threshold = TURBIDITY_PENALTY_THRESHOLD if TURBIDITY_PENALTY_ENABLED else 999.0
        recommender.turbidity_penalty_decay = TURBIDITY_PENALTY_DECAY
    elif RECOMMENDER_TYPE == "gradient":
        from recommenders.gradient_transition_recommender import (
            GradientTransitionRecommender,
        )
        recommender = GradientTransitionRecommender(
            input_columns=_input_cols,
            output_columns=output_columns,
            log_transform_inputs=True,
            beta=1.0,
            multi_output_reduce="standardized_sum",
            candidate_pool=_candidate_pool,
        )
        recommender.turbidity_penalty_threshold = TURBIDITY_PENALTY_THRESHOLD if TURBIDITY_PENALTY_ENABLED else 999.0
        recommender.turbidity_penalty_decay = TURBIDITY_PENALTY_DECAY
    elif RECOMMENDER_TYPE == "levelset":
        from recommenders.levelset_transition_recommender import (
            LevelSetTransitionRecommender,
        )
        recommender = LevelSetTransitionRecommender(
            input_columns=_input_cols,
            output_columns=output_columns,
            log_transform_inputs=True,
            candidate_pool=_candidate_pool,
        )
        recommender.turbidity_penalty_threshold = TURBIDITY_PENALTY_THRESHOLD if TURBIDITY_PENALTY_ENABLED else 999.0
        recommender.turbidity_penalty_decay = TURBIDITY_PENALTY_DECAY
    elif RECOMMENDER_TYPE in ("sobol", "random"):
        global _sobol_pool_idx
        # Hand out the next n_points from the pre-filtered pool.
        # Pool was built once at run start so this is a pure O(n) slice.
        points = _sobol_pool[_sobol_pool_idx: _sobol_pool_idx + n_points]
        _sobol_pool_idx += len(points)
        if not points:
            raise RuntimeError("Sobol pool exhausted. Increase TARGET_TOTAL_WELLS pool size.")
        return points, None
    else:
        raise ValueError(
            f"RECOMMENDER_TYPE must be one of 'triangle', 'bayesian', 'gradient', "
            f"'levelset', 'sobol', 'random'; got {RECOMMENDER_TYPE!r}"
        )

    # Build a vectorized feasibility function for simplex mode so the
    # recommender's Sobol candidate pool is pre-filtered before scoring.
    feasibility_fn = None
    if INIT_STRATEGY == 'simplex':
        import numpy as _np
        _stock_concs = _np.array(
            [SURFACTANT_LIBRARY[s]["stock_conc"] for s in SURFACTANTS])
        def feasibility_fn(X_conc_mm):   # noqa: E306
            vols = X_conc_mm * WELL_VOLUME_UL / _stock_concs[_np.newaxis, :]
            return vols.sum(axis=1) <= SURFACTANT_BUDGET_UL

    recs_df = recommender.get_recommendations(
        rec_data, n_points=n_points, iteration=iteration,
        feasibility_fn=feasibility_fn,
    )
    if len(recs_df) == 0:
        raise RuntimeError(
            f"{recommender.__class__.__name__} returned 0 recommendations "
            f"(input shape={rec_data.shape})."
        )

    # Clip to per-axis bounds and apply simplex projection as a final safety net.
    # Use max_conc_mm (cube cap in grid mode, simplex cap in simplex mode) so
    # GP recommendations don't escape the active search region.
    _active_max = {
        s: max_conc_mm[s]
        for s in SURFACTANTS
    }
    points = []
    for _, row in recs_df.iterrows():
        pt = {}
        for s in SURFACTANTS:
            c = float(row[f"{s}_conc_mm"])
            c = min(max(c, MIN_CONC_MM), _active_max[s])
            pt[s] = c
        points.append(project_to_simplex(pt))
    viz_state = getattr(recommender, '_viz_state', None)
    return points, viz_state


# ================================================================================
# OUTPUT FOLDER + EXPERIMENT NAME
# ================================================================================

def setup_experiment_folder(lash_e):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    surf_tag = "_".join(SURFACTANTS)
    name = f"multidim_{n_surfactants}D_{surf_tag}_{RECOMMENDER_TYPE}_{INIT_STRATEGY}_{EXPERIMENT_TAG}_{timestamp}"
    lash_e.current_experiment_name = name
    sim_folder = "simulated_surfactant_grid" if lash_e.simulate else "experimental_surfactant_grid"
    base = os.path.join("output", sim_folder, name)
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "measurement_backups"), exist_ok=True)
    os.makedirs(os.path.join(base, "calibration_validation"), exist_ok=True)
    lash_e.logger.info(f"Experiment folder: {base}")
    return base, name


def save_results(experiment_df, output_folder, label="results"):
    path = os.path.join(output_folder, f"{label}.csv")
    experiment_df.to_csv(path, index=False)
    return path


# ================================================================================
# MAIN WORKFLOW
# ================================================================================

def run_multidim_workflow(lash_e):
    """Top-level orchestrator. Steps:
       1. Setup folder + refill water/stock vials.
       2. Generate initial N-D log grid.
       3. Build per-surf dilution plans + create physical substocks.
       4. Build well recipes for grid, dispense, measure.
       5. Bayesian loop: recommender -> recipes (using existing stocks) ->
          dispense -> measure -> save -> rotate plate when full -> repeat
          until TARGET_TOTAL_WELLS reached.
       6. Final save + summary.
    """
    # Recompute all derived config from the live SURFACTANTS list.
    # The ConfigManager may have updated SURFACTANTS in globals after module load,
    # so module-level derived values (n_surfactants, cube_max_conc_mm, etc.) can
    # be stale. Recomputing here ensures they always reflect the actual run config.
    global n_surfactants, _cube_vol_per_surf, cube_max_conc_mm, simplex_max_conc_mm
    global max_conc_mm, max_vol_per_surf_ul
    global input_cols, vol_cols, substock_name_cols, substock_conc_cols
    global _sobol_pool, _sobol_pool_idx

    # Reset pool so each new experiment starts fresh.
    _sobol_pool = []
    _sobol_pool_idx = 0

    n_surfactants = len(SURFACTANTS)
    if not (2 <= n_surfactants <= 5):
        raise ValueError(f"SURFACTANTS must have 2-5 entries, got {n_surfactants}: {SURFACTANTS}")
    for s in SURFACTANTS:
        if s not in SURFACTANT_LIBRARY:
            raise ValueError(f"Surfactant '{s}' not in SURFACTANT_LIBRARY. Available: {list(SURFACTANT_LIBRARY.keys())}")

    _cube_vol_per_surf = SURFACTANT_BUDGET_UL / n_surfactants
    cube_max_conc_mm = {
        s: SURFACTANT_LIBRARY[s]["stock_conc"] * (_cube_vol_per_surf / WELL_VOLUME_UL)
        for s in SURFACTANTS
    }
    _min_vol_per_other_surf_ul = WELL_VOLUME_UL / 10.0
    _simplex_budget_for_one = max(
        SURFACTANT_BUDGET_UL - (n_surfactants - 1) * _min_vol_per_other_surf_ul, 0.0
    )
    simplex_max_conc_mm = {
        s: SURFACTANT_LIBRARY[s]["stock_conc"] * (_simplex_budget_for_one / WELL_VOLUME_UL)
        for s in SURFACTANTS
    }
    if INIT_STRATEGY == 'simplex':
        max_conc_mm = simplex_max_conc_mm
        max_vol_per_surf_ul = SURFACTANT_BUDGET_UL
    else:
        max_conc_mm = cube_max_conc_mm
        max_vol_per_surf_ul = _cube_vol_per_surf

    input_cols = [f"{s}_conc_mm" for s in SURFACTANTS]
    vol_cols = [f"{s}_volume_ul" for s in SURFACTANTS]
    substock_name_cols = [f"{s}_substock_name" for s in SURFACTANTS]
    substock_conc_cols = [f"{s}_substock_conc_mm" for s in SURFACTANTS]

    lash_e.logger.info("=" * 80)
    lash_e.logger.info(f"N-D SURFACTANT WORKFLOW: {n_surfactants} surfactants {SURFACTANTS}")
    lash_e.logger.info(f"  init strategy: {INIT_STRATEGY}")
    lash_e.logger.info(f"  simplex vertex maxima: " + ", ".join(
        f"{s}={simplex_max_conc_mm[s]:.2f} mM" for s in SURFACTANTS))
    lash_e.logger.info(f"  cube maxima (grid mode): " + ", ".join(
        f"{s}={cube_max_conc_mm[s]:.2f} mM" for s in SURFACTANTS))
    lash_e.logger.info(f"  recommender: {RECOMMENDER_TYPE}")
    lash_e.logger.info(f"  simulate: {lash_e.simulate}")
    lash_e.logger.info("=" * 80)

    output_folder, experiment_name = setup_experiment_folder(lash_e)

    # 1. Refill vials
    lash_e.nr_robot.home_robot_components()
    fill_water_vial(lash_e, "water")
    fill_water_vial(lash_e, "water_2")
    for s in SURFACTANTS:
        refill_surfactant_vial(lash_e, f"{s}_stock", liquid="SDS")

    # 2. Initial sampling
    # Build the Sobol pool once now. Interior init + all iterations draw from
    # this pre-filtered list in order — one continuous low-discrepancy sequence.
    if RECOMMENDER_TYPE in ('sobol', 'random'):
        from scipy.stats.qmc import Sobol as _SobolInit
        import numpy as np
        _sample_max = simplex_max_conc_mm if INIT_STRATEGY == 'simplex' else cube_max_conc_mm
        # Pool size: must cover all oversampled draws across all iterations.
        # Each iteration requests up to GRADIENT_SUGGESTIONS_PER_ITERATION * RECOMMEND_OVERSAMPLE_FACTOR
        # feasible points, so the raw pool needs enough to yield that many after feasibility filtering.
        _pool_size = max(TARGET_TOTAL_WELLS * RECOMMEND_OVERSAMPLE_FACTOR * 4, 8192)
        _pool_size = int(2 ** np.ceil(np.log2(_pool_size)))  # power of 2 for Sobol
        seq = _SobolInit(d=n_surfactants, scramble=True, seed=42)
        raw = seq.random(_pool_size)
        for row in raw:
            pt = {s: float(10 ** (np.log10(MIN_CONC_MM) + row[i] * (np.log10(_sample_max[s]) - np.log10(MIN_CONC_MM))))
                  for i, s in enumerate(SURFACTANTS)}
            if is_feasible(pt):
                _sobol_pool.append(pt)
        lash_e.logger.info(f"Sobol pool: {len(_sobol_pool)}/{_pool_size} feasible points pre-computed")

    if INIT_STRATEGY == 'simplex':
        # Build bootstrap plans early so boundary/interior selection can use source-achievable checks
        bootstrap_targets_per_surf = {
            s: sorted(set([MIN_CONC_MM, simplex_max_conc_mm[s]]))
            for s in SURFACTANTS
        }
        bootstrap_plans = build_plans_for_surfactants(lash_e, bootstrap_targets_per_surf, existing_stock_solutions=None)
        grid_points = generate_simplex_init(plans=bootstrap_plans, logger=lash_e.logger)
        lash_e.logger.info(
            f"Simplex init: {len(grid_points)} points "
            f"({n_surfactants}D: boundary levels={INIT_BOUNDARY_LEVELS} + "
            f"{INIT_INTERIOR_PTS} interior)"
        )
    else:
        grid_points = generate_log_grid(GRID_POINTS_PER_AXIS)
        lash_e.logger.info(
            f"Grid init: {len(grid_points)} points "
            f"(= {GRID_POINTS_PER_AXIS}^{n_surfactants})"
        )
    # Keep a copy so we can diagnose points dropped by plan-based volume filtering.
    init_grid_candidates = [dict(pt) for pt in grid_points]
    targets_per_surf = collect_unique_targets_per_surf(grid_points)

    # 3. Plans + physical substocks
    plans = build_plans_for_surfactants(lash_e, targets_per_surf, existing_stock_solutions=None)
    dilution_recipes = calculate_dilution_recipes_nd(lash_e, plans)
    dilution_recipes.sort(key=lambda r: r["Target_Conc_mM"], reverse=True)
    if dilution_recipes:
        create_substocks_from_recipes(lash_e, dilution_recipes)
    stock_solutions = stock_solutions_needed_from_plans(plans, dilution_recipes)

    # 4. Controls (first plate) + initial experiment grid -> dispense -> measure
    grid_points_before_filter = len(grid_points)
    grid_points = filter_points_by_actual_volumes(grid_points, plans, lash_e.logger)
    if not grid_points:
        raise RuntimeError("All initial grid points were infeasible after volume check.")
    lash_e.logger.info(f"  Initial grid: {len(grid_points)}/{grid_points_before_filter} points survived volume filter ({100*len(grid_points)/grid_points_before_filter:.0f}%)")

    # Diagnose dropped points for visualization/debugging.
    def _pt_key(pt):
        return tuple(float(f"{pt[s]:.4g}") for s in SURFACTANTS)

    kept_keys = {_pt_key(pt) for pt in grid_points}
    dropped_init_points = [pt for pt in init_grid_candidates if _pt_key(pt) not in kept_keys]
    n_dropped_boundary = sum(1 for pt in dropped_init_points if pt.get('_init_source_type') == 'boundary')
    n_dropped_sobol = sum(1 for pt in dropped_init_points if pt.get('_init_source_type') == 'sobol')
    lash_e.logger.info(
        f"  Dropped init candidates: {len(dropped_init_points)} "
        f"(boundary={n_dropped_boundary}, sobol={n_dropped_sobol})"
    )
    dropped_csv = os.path.join(output_folder, "dropped_init_candidates.csv")
    pd.DataFrame(dropped_init_points).to_csv(dropped_csv, index=False)
    lash_e.logger.info(f"  Dropped init candidates saved: {dropped_csv}")

    # Build controls first so well indices are contiguous on the plate.
    controls_df = build_control_wells_df(plans, starting_well_index=0)
    n_controls = len(controls_df)
    if n_controls > 0:
        lash_e.logger.info(
            f"  Built {n_controls} control wells for first plate "
            f"({sum(controls_df['control_type'] == 'water_blank')} water blank, "
            f"{sum(controls_df['control_type'].str.startswith('stock_'))} stock refs, "
            f"{sum(controls_df['control_type'].str.startswith('cmc_1d_'))} 1D CMC)."
        )

    # Experiment grid wells start after the control block.
    init_exp_df = build_well_recipes_df(grid_points, plans, starting_well_index=n_controls)

    # Combine for the first plate (controls are always first).
    if n_controls > 0:
        first_plate_df = pd.concat([controls_df, init_exp_df], ignore_index=True)
    else:
        first_plate_df = init_exp_df

    lash_e.grab_new_wellplate()
    save_results(first_plate_df, output_folder, "experiment_plan_initial")
    well_recipes_df = execute_dispensing_nd(lash_e, first_plate_df)
    well_recipes_df = dispense_dmso(lash_e, well_recipes_df)
    well_recipes_df = measure_and_process_turbidity_nd(lash_e, well_recipes_df)
    well_recipes_df = measure_and_process_fluorescence_nd(lash_e, well_recipes_df)
    well_recipes_df['iteration'] = 0
    save_results(well_recipes_df, output_folder, "results_after_initial_grid")

    # Plot initial grid so it can be compared against the final distribution.
    try:
        from analysis.multidim_visualizer import plot_pairwise_maps, plot_pairwise_feasible_overlay
        from analysis.plot_3d_interactive import plot_init_feasible_overlay_3d
        init_plot_folder = os.path.join(output_folder, "plots_initial_grid")
        os.makedirs(init_plot_folder, exist_ok=True)
        # Only plot experiment wells for the initial grid view.
        init_exp_measured = well_recipes_df[well_recipes_df['well_type'] == 'experiment']
        saved = plot_pairwise_maps(init_exp_measured, SURFACTANTS, init_plot_folder)
        for k, p in saved.items():
            lash_e.logger.info(f"  Initial grid plot saved: {p}")

        if INIT_STRATEGY == 'simplex':
            # 2D overlay should only show actual init picks (no extra boundary artifact series).
            feasible_boundary_points = None
            # 3D overlay needs boundary points to render the envelope mesh.
            feasible_boundary_points_3d = [pt for pt in grid_points if pt.get('_init_source_type') == 'boundary']
        else:
            feasible_boundary_points = generate_log_grid(GRID_POINTS_PER_AXIS)
            feasible_boundary_points_3d = feasible_boundary_points

        # Wrap achievable_fn to log white dot diagnostics on first few failures
        white_dot_log_count = [0]  # mutable counter in closure
        MAX_WHITE_DOT_DIAGNOSTICS = 5  # Log details for first 5 white dots per panel
        
        def achievable_with_diagnostics(pt):
            is_achievable = len(filter_points_by_actual_volumes([pt], plans)) == 1
            if not is_achievable and white_dot_log_count[0] < MAX_WHITE_DOT_DIAGNOSTICS:
                white_dot_log_count[0] += 1
                diagnose_white_dot(pt, plans, logger=lash_e.logger)
            return is_achievable
        
        p_overlay_2d = plot_pairwise_feasible_overlay(
            init_exp_measured,
            SURFACTANTS,
            feasible_boundary_points,
            init_plot_folder,
            show_all_experiment=False,
            feasible_config={
                "stock_concs": {s: SURFACTANT_LIBRARY[s]["stock_conc"] for s in SURFACTANTS},
                "well_volume_ul": WELL_VOLUME_UL,
                "surfactant_budget_ul": SURFACTANT_BUDGET_UL,
                "min_conc_mm": MIN_CONC_MM,
                "max_conc_multiplier": FEASIBLE_MAX_CONC_MULTIPLIER,
            },
            achievable_fn=achievable_with_diagnostics,
            grid_n=FEASIBLE_DIAGNOSTIC_GRID_N,
            logger=lash_e.logger,
            dropped_points=dropped_init_points,
        )
        if p_overlay_2d:
            lash_e.logger.info(f"  Initial feasible-overlay 2D plot saved: {p_overlay_2d}")

        if len(SURFACTANTS) == 3:
            p_overlay_3d = plot_init_feasible_overlay_3d(
                init_exp_measured,
                SURFACTANTS,
                feasible_boundary_points_3d,
                output_dir=init_plot_folder,
                feasible_config={
                    "stock_concs": {s: SURFACTANT_LIBRARY[s]["stock_conc"] for s in SURFACTANTS},
                    "well_volume_ul": WELL_VOLUME_UL,
                    "surfactant_budget_ul": SURFACTANT_BUDGET_UL,
                    "min_conc_mm": MIN_CONC_MM,
                    "max_conc_multiplier": FEASIBLE_MAX_CONC_MULTIPLIER,
                },
                dropped_points=dropped_init_points,
            )
            if p_overlay_3d:
                lash_e.logger.info(f"  Initial feasible-overlay 3D plot saved: {p_overlay_3d}")
    except Exception as e:
        lash_e.logger.warning(f"Initial grid plotting failed (non-fatal): {e}")

    current_wellplate_wells = len(well_recipes_df)
    n_exp_wells = len(init_exp_df)
    lash_e.logger.info(
        f"Initial grid complete: {n_exp_wells} experiment wells + {n_controls} controls "
        f"({current_wellplate_wells}/{MAX_WELLS_PER_PLATE} on current plate, "
        f"{current_wellplate_wells}/{TARGET_TOTAL_WELLS} total budget used)."
    )

    # Plot 1D CMC controls immediately after first measurement round.
    try:
        cmc_plot_path = plot_1d_cmc_controls(well_recipes_df, output_folder, lash_e.logger)
        if cmc_plot_path:
            lash_e.logger.info(f"  1D CMC control plot saved: {cmc_plot_path}")
    except Exception as e:
        lash_e.logger.warning(f"1D CMC control plot failed (non-fatal): {e}")

    # 5. Bayesian iterations
    iteration = 1
    while len(well_recipes_df) < TARGET_TOTAL_WELLS and iteration <= MAX_ITERATIONS:
        lash_e.logger.info("-" * 80)
        lash_e.logger.info(f"Iteration {iteration} ({len(well_recipes_df)}/{TARGET_TOTAL_WELLS} wells)")

        # Refill consumables between iterations
        fill_water_vial(lash_e, "water")
        fill_water_vial(lash_e, "water_2")
        for s in SURFACTANTS:
            refill_surfactant_vial(lash_e, f"{s}_stock", liquid="SDS")
        lash_e.nr_robot.home_robot_components()

        # Top up substocks if needed
        if dilution_recipes:
            create_substocks_from_recipes(lash_e, dilution_recipes)

        wells_remaining_in_plate = MAX_WELLS_PER_PLATE - current_wellplate_wells
        wells_remaining_to_target = TARGET_TOTAL_WELLS - len(well_recipes_df)
        max_this_iter = min(GRADIENT_SUGGESTIONS_PER_ITERATION,
                            wells_remaining_in_plate, wells_remaining_to_target)
        if max_this_iter <= 0:
            lash_e.logger.info("No room to dispense this iteration; rotating plate.")
            lash_e.discard_used_wellplate()
            lash_e.grab_new_wellplate()
            current_wellplate_wells = 0
            continue

        # Recommender -> N-D points
        # No oversampling needed: joint_select rejects virtually nothing, so
        # every point drawn from the persistent Sobol sequence gets used.
        points, viz_state = get_next_batch(
            well_recipes_df, n_points=max_this_iter * RECOMMEND_OVERSAMPLE_FACTOR,
            iteration=iteration, output_dir=output_folder,
        )
        lash_e.logger.info(
            f"Iteration {iteration}: recommender returned {len(points)} candidates "
            f"(requested {max_this_iter * RECOMMEND_OVERSAMPLE_FACTOR}, target {max_this_iter})"
        )

        # Map suggested concentrations to existing substocks
        targets_per_surf = collect_unique_targets_per_surf(points)
        plans = build_plans_for_surfactants(
            lash_e, targets_per_surf, existing_stock_solutions=stock_solutions,
        )

        # Drop any point whose actual substock volumes exceed the dispensing
        # budget (the pre-filter uses stock concs; substocks may be more dilute).
        points = filter_points_by_actual_volumes(points, plans, lash_e.logger)
        lash_e.logger.info(
            f"Iteration {iteration}: {len(points)} candidates survived volume filter; "
            f"keeping first {min(len(points), max_this_iter)}"
        )
        if not points:
            lash_e.logger.warning(
                f"Iteration {iteration}: all recommended points were infeasible "
                f"after volume check; skipping iteration."
            )
            iteration += 1
            continue

        # Trim to the number of wells that actually fit this iteration
        if len(points) > max_this_iter:
            points = points[:max_this_iter]

        # Intermediate diagnostic plots — 2D only (3D+ uses end-of-run HTML plots instead)
        if len(SURFACTANTS) == 2:
            if RECOMMENDER_TYPE in ('delaunay', 'triangle') and viz_state is not None:
                try:
                    path = _plot_delaunay_decisions(
                        viz_state, points[:max_this_iter], iteration, output_folder)
                    lash_e.logger.info(f"Triangle viz saved: {path}")
                except Exception as _ve:
                    lash_e.logger.warning(f"Triangle viz failed (non-fatal): {_ve}")

            if RECOMMENDER_TYPE == 'bayesian' and viz_state is not None:
                try:
                    path = _plot_bayesian_decisions(
                        viz_state, points[:max_this_iter], iteration, output_folder)
                    lash_e.logger.info(f"Bayesian viz saved: {path}")
                except Exception as _ve:
                    lash_e.logger.warning(f"Bayesian viz failed (non-fatal): {_ve}")

            if RECOMMENDER_TYPE in ('gradient', 'levelset') and viz_state is not None:
                try:
                    subfolder = f'{RECOMMENDER_TYPE}_decisions'
                    path = _plot_gp_decisions(
                        viz_state, points[:max_this_iter], iteration, output_folder,
                        subfolder=subfolder)
                    lash_e.logger.info(f"{RECOMMENDER_TYPE.title()} viz saved: {path}")
                except Exception as _ve:
                    lash_e.logger.warning(f"{RECOMMENDER_TYPE.title()} viz failed (non-fatal): {_ve}")

            if viz_state is not None and len(viz_state.get('input_columns', [])) >= 3:
                try:
                    paths = _plot_nd_decisions(
                        viz_state, points[:max_this_iter], iteration, output_folder)
                    for p in paths:
                        lash_e.logger.info(f"N-D viz saved: {p}")
                except Exception as _ve:
                    lash_e.logger.warning(f"N-D viz failed (non-fatal): {_ve}")

        # Build new recipe rows continuing the well numbering
        next_recipes_df = build_well_recipes_df(
            points, plans, starting_well_index=current_wellplate_wells,
        )
        next_recipes_df = execute_dispensing_nd(lash_e, next_recipes_df)
        next_recipes_df = dispense_dmso(lash_e, next_recipes_df)
        next_recipes_df = measure_and_process_turbidity_nd(lash_e, next_recipes_df)
        next_recipes_df = measure_and_process_fluorescence_nd(lash_e, next_recipes_df)

        next_recipes_df['iteration'] = iteration
        well_recipes_df = pd.concat([well_recipes_df, next_recipes_df], ignore_index=True)
        current_wellplate_wells += len(next_recipes_df)
        save_results(well_recipes_df, output_folder, "results_iterative")

        if current_wellplate_wells >= MAX_WELLS_PER_PLATE and len(well_recipes_df) < TARGET_TOTAL_WELLS:
            lash_e.logger.info("Plate full; rotating to a new plate.")
            lash_e.discard_used_wellplate()
            lash_e.grab_new_wellplate()
            current_wellplate_wells = 0

        iteration += 1

    # 6. Final cleanup + summary
    lash_e.discard_used_wellplate()
    final_path = save_results(well_recipes_df, output_folder, "results_final")

    # Generate pairwise marginal maps
    try:
        from analysis.multidim_visualizer import plot_pairwise_maps
        lash_e.logger.info("Generating pairwise contour maps...")
        saved = plot_pairwise_maps(well_recipes_df, SURFACTANTS, output_folder)
        for k, p in saved.items():
            lash_e.logger.info(f"  Plot saved: {p}")
    except Exception as e:
        lash_e.logger.warning(f"Plotting failed (non-fatal): {e}")

    # Generate interactive 3D HTML scatter plots (only for 3D spaces)
    if len(SURFACTANTS) == 3:
        try:
            from analysis.plot_3d_interactive import plot_3d_interactive, plot_isosurface, plot_ratio_phases
            lash_e.logger.info("Generating interactive 3D scatter plots...")
            p_turb, p_ratio = plot_3d_interactive(final_path, output_dir=output_folder)
            lash_e.logger.info(f"  3D turbidity plot: {p_turb}")
            lash_e.logger.info(f"  3D ratio plot: {p_ratio}")
            lash_e.logger.info("Generating turbidity cloud plot...")
            p_iso = plot_isosurface(final_path, threshold=TURBIDITY_PLOT_THRESHOLD, output_dir=output_folder)
            lash_e.logger.info(f"  Turbidity cloud: {p_iso}")
            lash_e.logger.info("Generating ratio phase map...")
            p_phases = plot_ratio_phases(final_path, output_dir=output_folder)
            lash_e.logger.info(f"  Ratio phase map: {p_phases}")
        except Exception as e:
            lash_e.logger.warning(f"Interactive 3D plot failed (non-fatal): {e}")

    if len(SURFACTANTS) == 3:
        try:
            from analysis.iteration_metrics import compute_iteration_metrics, save_iteration_metrics
            lash_e.logger.info("Computing per-iteration metrics...")
            iter_metrics = compute_iteration_metrics(final_path, SURFACTANTS, output_folder)
            csv_m, png_m = save_iteration_metrics(iter_metrics, output_folder)
            lash_e.logger.info(f"  Iteration metrics CSV: {csv_m}")
            lash_e.logger.info(f"  Iteration metrics plot: {png_m}")
        except Exception as e:
            lash_e.logger.warning(f"Iteration metrics failed (non-fatal): {e}")

    lash_e.logger.info("=" * 80)
    lash_e.logger.info("N-D WORKFLOW COMPLETE")
    n_exp_final = len(well_recipes_df[well_recipes_df['well_type'] == 'experiment'])
    n_ctrl_final = len(well_recipes_df[well_recipes_df['well_type'] == 'control'])
    lash_e.logger.info(f"  Total wells: {len(well_recipes_df)} ({n_exp_final} experiment + {n_ctrl_final} control)")
    lash_e.logger.info(f"  Iterations: {iteration - 1}")
    lash_e.logger.info(f"  Output: {final_path}")
    lash_e.logger.info("=" * 80)
    return {
        "experiment_name": experiment_name,
        "output_folder": output_folder,
        "well_recipes_df": well_recipes_df,
        "iterations": iteration - 1,
    }


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.ioff()

    lash_e = Lash_E(
        INPUT_VIAL_STATUS_FILE, simulate=SIMULATE,
        workflow_globals=globals(), workflow_name="surfactant_multidimensional_workflow",
        show_gui=True,
    )

    fill_water_vial(lash_e, "water")
    fill_water_vial(lash_e, "water_2")

    if VALIDATE_LIQUIDS:
        validation_folder = os.path.join(
            "output", f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(validation_folder, exist_ok=True)
        validate_pipetting_system(
            lash_e, validation_folder, surfactant_names=[SURFACTANTS[0]],
        )

    run_multidim_workflow(lash_e)

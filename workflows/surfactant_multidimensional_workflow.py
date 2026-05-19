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
SURFACTANTS = ["SDS", "DSS", "TTAB", "DTAB"]

EXPERIMENT_TAG = "multidim_v1"
SIMULATE = True
VALIDATE_LIQUIDS = False

# Recommender selection
RECOMMENDER_TYPE = "delaunay"  # 'delaunay' | 'bayesian' | 'gradient' | 'levelset' | 'sobol' | 'random'

# Turbidity filtering for recommender inputs
# - FILTER_UNRELIABLE_RATIOS: null out ratio for high-turbidity wells before passing
#   to the recommender so unreliable fluorescence data does not corrupt GP fits.
# - TURBIDITY_FILTER_THRESHOLD: per-point cutoff (above = ratio is unreliable).
# - FLAT_TURBIDITY_MAX: if max(turbidity_600) across ALL data is below this value,
#   the turbidity landscape is considered flat/uninformative and is dropped from
#   output_columns so the recommender only sees ratio.
FILTER_UNRELIABLE_RATIOS = True
TURBIDITY_FILTER_THRESHOLD = 0.2   # Wells above this turbidity have unreliable ratio
FLAT_TURBIDITY_MAX = 0.08          # Whole-dataset gate: drop turbidity if max <= this

# Volume budget.
# WATER_RESERVE_UL is always held back so that even at max concentration for
# every surfactant there is still water to fill the well.
WELL_VOLUME_UL = 250
PYRENE_VOLUME_UL = 5
WATER_RESERVE_UL = 20          # Minimum water in every well (uL)
SURFACTANT_BUDGET_UL = WELL_VOLUME_UL - PYRENE_VOLUME_UL - WATER_RESERVE_UL  # 225 uL

# Initialization strategy
# 'simplex' (recommended): stratified design that covers the true physical
#   simplex (sum of volumes <= budget).  Allows each surfactant to reach its
#   individual CMC even when used with others.  Three layers:
#     - axis rays  : one surfactant varies, others at MIN_CONC_MM
#     - pair faces : log-grid on every (s_i, s_j) pair, others at MIN_CONC_MM
#     - interior   : Sobol points mapped into simplex via stick-breaking
# 'grid' (legacy): hypercube with equal per-surf cap (BUDGET/N).  Safe for
#   2D; may not reach CMC for low-CMC surfactants in 4D.
INIT_STRATEGY = 'simplex'  # 'simplex' | 'grid'
GRID_POINTS_PER_AXIS = 3   # Used only when INIT_STRATEGY='grid': 3^N grid
INIT_AXIS_PTS = 5           # Simplex: points on each 1D axis ray
INIT_FACE_PTS = 3           # Simplex: points per axis on each 2D pairwise face (3x3=9)
INIT_INTERIOR_PTS = 20      # Simplex: Sobol-in-simplex interior points
MIN_CONC_MM = 1e-2          # Lower bound for all concentration axes

# Active-learning loop
GRADIENT_SUGGESTIONS_PER_ITERATION = 14
TARGET_TOTAL_WELLS = 96        # Measurement budget: stop after this many wells total
MAX_ITERATIONS = 20            # Safety cap; loop also stops if TARGET_TOTAL_WELLS reached

# Plate handling
MAX_WELLS_PER_PLATE = 96

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

# Simplex per-surf caps (each surf can use the FULL budget when alone):
#   these are the simplex vertices — the physically achievable maxima
simplex_max_conc_mm = {
    s: SURFACTANT_LIBRARY[s]["stock_conc"] * (SURFACTANT_BUDGET_UL / WELL_VOLUME_UL)
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


def generate_simplex_init():
    """Stratified initialization that covers the physical volume simplex.

    Three layers (all using log-spaced concentrations along each active axis):

    1. Axis rays  (N * INIT_AXIS_PTS points)
       Vary one surfactant from MIN_CONC_MM to its simplex vertex; all others
       at MIN_CONC_MM.  Covers each surf's individual CMC crossing and ensures
       the recommender has data near every simplex edge.

    2. Pairwise faces  (C(N,2) * INIT_FACE_PTS^2 points)
       Log-grid on each (s_i, s_j) pair; remaining surfs at MIN_CONC_MM.
       Points at the far corner are projected onto the simplex face.
       Covers catanionic pair interactions across the full concentration range.

    3. Sobol interior  (INIT_INTERIOR_PTS points)
       Quasi-random volume fractions mapped into the simplex via stick-breaking:
       draw Sobol in (N-1) dimensions, convert to N volume fractions (sum=1),
       scale by budget, convert to concentrations.  Uniform coverage of interior.

    Duplicate points (axis-face overlaps at corners) are removed.
    All returned points satisfy is_feasible() to within float tolerance.
    Returns a list of dicts {surfactant_name: target_conc_mm}.
    """
    from itertools import combinations as _comb

    pts = []
    seen = set()

    def _add(pt):
        # Round to 4 sig figs for dedup — enough to collapse true duplicates
        # (axis-face overlaps) without collapsing nearby distinct points.
        key = tuple(float(f"{pt[s]:.4g}") for s in SURFACTANTS)
        if key not in seen:
            seen.add(key)
            pts.append(pt)

    # 1. Axis rays
    for s in SURFACTANTS:
        concs = np.logspace(
            np.log10(MIN_CONC_MM), np.log10(simplex_max_conc_mm[s]),
            INIT_AXIS_PTS,
        )
        for c in concs:
            pt = {surf: MIN_CONC_MM for surf in SURFACTANTS}
            pt[s] = float(c)
            _add(pt)

    # 2. Pairwise faces
    for s_i, s_j in _comb(SURFACTANTS, 2):
        ci_vals = np.logspace(
            np.log10(MIN_CONC_MM), np.log10(simplex_max_conc_mm[s_i]),
            INIT_FACE_PTS,
        )
        cj_vals = np.logspace(
            np.log10(MIN_CONC_MM), np.log10(simplex_max_conc_mm[s_j]),
            INIT_FACE_PTS,
        )
        for ci in ci_vals:
            for cj in cj_vals:
                pt = {surf: MIN_CONC_MM for surf in SURFACTANTS}
                pt[s_i] = float(ci)
                pt[s_j] = float(cj)
                _add(project_to_simplex(pt))

    # 3. Sobol interior via stick-breaking
    if INIT_INTERIOR_PTS > 0:
        from scipy.stats.qmc import Sobol as _Sobol
        n_dim = n_surfactants - 1
        # Sobol requires power-of-2 samples
        n_draw = int(2 ** np.ceil(np.log2(max(INIT_INTERIOR_PTS, 2))))
        raw = _Sobol(d=n_dim, scramble=True, seed=42).random(n_draw)[:INIT_INTERIOR_PTS]
        # Stick-breaking -> uniform distribution over the standard simplex
        weights = np.zeros((len(raw), n_surfactants))
        remaining = np.ones(len(raw))
        for k in range(n_surfactants - 1):
            weights[:, k] = raw[:, k] * remaining
            remaining -= weights[:, k]
        weights[:, -1] = remaining
        # Map volume fractions to concentrations.
        # Clamp to MIN_CONC_MM first, then project to re-enforce budget.
        for w_row in weights:
            pt = {}
            for i, s in enumerate(SURFACTANTS):
                vol_ul = float(w_row[i]) * SURFACTANT_BUDGET_UL
                conc = vol_ul * SURFACTANT_LIBRARY[s]["stock_conc"] / WELL_VOLUME_UL
                pt[s] = max(conc, MIN_CONC_MM)
            # project_to_simplex scales without re-clamping, so this is exact
            _add(project_to_simplex(pt))

    return pts


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

def build_well_recipe(target_concs, plans, well_index, replicate=1):
    """Translate one N-D target point into a recipe row.

    Looks up substock + volume from each surfactant's plan, sums to compute the
    water makeup, and returns a dict with all column names.
    """
    recipe = {
        "wellplate_index": int(well_index),
        "well_type": "experiment",
        "control_type": "experiment",
        "replicate": replicate,
    }
    total_surf_vol = 0.0
    for s in SURFACTANTS:
        target = target_concs[s]
        sol = plans[s]["concentration_map"].get(target)
        if not sol:
            raise ValueError(
                f"No dilution solution found for {s} at {target:.3e} mM "
                f"(check stock matching / pipettable volume range)."
            )
        recipe[f"{s}_conc_mm"] = float(target)
        recipe[f"{s}_substock_name"] = sol["vial_name"]
        recipe[f"{s}_substock_conc_mm"] = sol["concentration_mm"]
        recipe[f"{s}_volume_ul"] = float(sol["volume_needed_ul"])
        total_surf_vol += float(sol["volume_needed_ul"])

    water_vol = WELL_VOLUME_UL - PYRENE_VOLUME_UL - total_surf_vol
    if water_vol < 0:
        raise ValueError(
            f"Negative water volume ({water_vol:.2f} uL) at well {well_index}: "
            f"surfactant total {total_surf_vol:.2f} uL exceeds budget "
            f"{SURFACTANT_BUDGET_UL:.2f} uL. Check that the point is inside "
            f"the simplex (sum of volumes <= budget)."
        )
    recipe["water_volume_ul"] = water_vol
    recipe["pyrene_volume_ul"] = PYRENE_VOLUME_UL
    # Buffer disabled in v1; legacy primitives expect this column to exist.
    recipe["buffer_volume_ul"] = 0.0
    recipe["buffer_used"] = None
    return recipe


def build_well_recipes_df(target_points, plans, starting_well_index=0):
    rows = []
    for i, pt in enumerate(target_points):
        rows.append(build_well_recipe(pt, plans, starting_well_index + i))
    return pd.DataFrame(rows)


def filter_points_by_actual_volumes(points, plans, logger=None):
    """Drop any points whose actual planned substock volumes would exceed the budget.

    The feasibility check in get_next_batch uses stock concentrations (best case).
    In practice the planner may choose a diluted substock, which requires more
    physical volume for the same target concentration.  This check uses the
    actual volume from the plan so infeasible points are discarded before we
    attempt to build recipes (which would crash with a negative water error).

    Returns the filtered list and logs a warning per dropped point.
    """
    import logging as _log
    _logger = logger or _log.getLogger(__name__)
    feasible = []
    for pt in points:
        total_vol = 0.0
        ok = True
        for s in SURFACTANTS:
            target = pt[s]
            sol = plans[s]["concentration_map"].get(target)
            if sol is None:
                ok = False
                break
            total_vol += sol["volume_needed_ul"]
        if ok and total_vol <= SURFACTANT_BUDGET_UL + 1e-9:
            feasible.append(pt)
        else:
            _logger.warning(
                f"Dropping recommended point (actual total vol={total_vol:.1f} uL > "
                f"budget {SURFACTANT_BUDGET_UL:.1f} uL — substock more dilute than "
                f"stock assumption): "
                + ", ".join(f"{s}={pt[s]:.3g}mM" for s in SURFACTANTS)
            )
    n_dropped = len(points) - len(feasible)
    if n_dropped:
        _logger.warning(
            f"Filtered {n_dropped}/{len(points)} recommended points infeasible "
            f"due to substock volume mismatch."
        )
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

def get_next_batch(experiment_data_df, n_points=GRADIENT_SUGGESTIONS_PER_ITERATION,
                   iteration=None):
    """Run the configured N-D recommender. Returns a list of dicts
    {surfactant_name: target_conc_mm} of length <= n_points.

    All transition recommenders inherit from TransitionRecommenderBase and
    expose the same get_recommendations(data_df, n_points) interface.
    Sobol and Random are non-adaptive baselines that ignore observed data.

    RECOMMENDER_TYPE options
    ------------------------
    'delaunay'  : DelaunaySimplexTransitionRecommender - geometric, no GP
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

    rec_data = experiment_data_df.copy()
    output_columns = ["ratio", "turbidity_600"]

    if "turbidity_600" in rec_data.columns and rec_data["turbidity_600"].notna().any():
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

    if RECOMMENDER_TYPE == "delaunay":
        from recommenders.delaunay_simplex_recommender import (
            DelaunaySimplexTransitionRecommender,
        )
        recommender = DelaunaySimplexTransitionRecommender(
            input_columns=_input_cols,
            output_columns=output_columns,
            log_transform_inputs=True,
        )
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
            candidate_pool=50000,
        )
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
            candidate_pool=50000,
        )
    elif RECOMMENDER_TYPE == "levelset":
        from recommenders.levelset_transition_recommender import (
            LevelSetTransitionRecommender,
        )
        recommender = LevelSetTransitionRecommender(
            input_columns=_input_cols,
            output_columns=output_columns,
            log_transform_inputs=True,
            candidate_pool=50000,
        )
    elif RECOMMENDER_TYPE in ("sobol", "random"):
        # Non-adaptive baselines: sample directly in concentration space.
        # Sample log-uniformly in each axis independently (same approach as
        # the GP recommenders' Sobol candidate pool), then reject points that
        # violate the simplex constraint.  This gives true log-space coverage
        # of the feasible region, equivalent to what the GP algorithms see.
        # seed is offset by iteration so successive calls don't repeat.
        import numpy as np
        _seed = 0 if iteration is None else int(iteration)

        rng = np.random.default_rng(_seed)
        points = []
        max_attempts = n_points * 200  # safety cap
        attempts = 0
        while len(points) < n_points and attempts < max_attempts:
            batch_size = min((n_points - len(points)) * 10, 1000)
            if RECOMMENDER_TYPE == "sobol":
                from scipy.stats.qmc import Sobol as _Sobol
                n_draw = int(2 ** np.ceil(np.log2(max(batch_size, 2))))
                raw = _Sobol(d=n_surfactants, scramble=True,
                             seed=_seed + attempts).random(n_draw)[:batch_size]
            else:
                raw = rng.uniform(0, 1, (batch_size, n_surfactants))
            attempts += batch_size
            for row in raw:
                pt = {}
                for i, s in enumerate(SURFACTANTS):
                    log_lo = np.log10(MIN_CONC_MM)
                    log_hi = np.log10(simplex_max_conc_mm[s])
                    pt[s] = float(10 ** (log_lo + row[i] * (log_hi - log_lo)))
                if is_feasible(pt):
                    points.append(pt)
                if len(points) >= n_points:
                    break
        if not points:
            raise RuntimeError(
                f"sobol/random: could not generate any feasible points after "
                f"{attempts} attempts. Check simplex_max_conc_mm and is_feasible."
            )
        return points[:n_points]
    else:
        raise ValueError(
            f"RECOMMENDER_TYPE must be one of 'delaunay', 'bayesian', 'gradient', "
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
    # Recompute simplex_max_conc_mm from live SURFACTANTS in case ConfigManager
    # updated the global after module-level derived dicts were computed.
    _simplex_max = {
        s: SURFACTANT_LIBRARY[s]["stock_conc"] * (SURFACTANT_BUDGET_UL / WELL_VOLUME_UL)
        for s in SURFACTANTS
    }
    points = []
    for _, row in recs_df.iterrows():
        pt = {}
        for s in SURFACTANTS:
            c = float(row[f"{s}_conc_mm"])
            c = min(max(c, MIN_CONC_MM), _simplex_max[s])
            pt[s] = c
        points.append(project_to_simplex(pt))
    return points


# ================================================================================
# OUTPUT FOLDER + EXPERIMENT NAME
# ================================================================================

def setup_experiment_folder(lash_e):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    surf_tag = "_".join(SURFACTANTS)
    name = f"multidim_{n_surfactants}D_{surf_tag}_{EXPERIMENT_TAG}_{timestamp}"
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
    if INIT_STRATEGY == 'simplex':
        grid_points = generate_simplex_init()
        lash_e.logger.info(
            f"Simplex init: {len(grid_points)} points "
            f"({n_surfactants}D: {n_surfactants}x{INIT_AXIS_PTS} axis + "
            f"{n_surfactants*(n_surfactants-1)//2}x{INIT_FACE_PTS}^2 faces + "
            f"{INIT_INTERIOR_PTS} interior)"
        )
    else:
        grid_points = generate_log_grid(GRID_POINTS_PER_AXIS)
        lash_e.logger.info(
            f"Grid init: {len(grid_points)} points "
            f"(= {GRID_POINTS_PER_AXIS}^{n_surfactants})"
        )
    targets_per_surf = collect_unique_targets_per_surf(grid_points)

    # 3. Plans + physical substocks
    plans = build_plans_for_surfactants(lash_e, targets_per_surf, existing_stock_solutions=None)
    dilution_recipes = calculate_dilution_recipes_nd(lash_e, plans)
    dilution_recipes.sort(key=lambda r: r["Target_Conc_mM"], reverse=True)
    if dilution_recipes:
        create_substocks_from_recipes(lash_e, dilution_recipes)
    stock_solutions = stock_solutions_needed_from_plans(plans, dilution_recipes)

    # 4. Initial recipes -> dispense -> measure
    lash_e.grab_new_wellplate()
    well_recipes_df = build_well_recipes_df(grid_points, plans, starting_well_index=0)
    save_results(well_recipes_df, output_folder, "experiment_plan_initial")
    well_recipes_df = execute_dispensing_nd(lash_e, well_recipes_df)
    well_recipes_df = dispense_dmso(lash_e, well_recipes_df)
    well_recipes_df = measure_and_process_turbidity_nd(lash_e, well_recipes_df)
    well_recipes_df = measure_and_process_fluorescence_nd(lash_e, well_recipes_df)
    save_results(well_recipes_df, output_folder, "results_after_initial_grid")

    current_wellplate_wells = len(well_recipes_df)
    measured_count = len(well_recipes_df)
    lash_e.logger.info(f"Initial grid complete: {measured_count} wells measured, "
                       f"{current_wellplate_wells}/96 on current plate.")

    # 5. Bayesian iterations
    iteration = 1
    while measured_count < TARGET_TOTAL_WELLS and iteration <= MAX_ITERATIONS:
        lash_e.logger.info("-" * 80)
        lash_e.logger.info(f"Iteration {iteration} ({measured_count}/{TARGET_TOTAL_WELLS} wells)")

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
        wells_remaining_to_target = TARGET_TOTAL_WELLS - measured_count
        max_this_iter = min(GRADIENT_SUGGESTIONS_PER_ITERATION,
                            wells_remaining_in_plate, wells_remaining_to_target)
        if max_this_iter <= 0:
            lash_e.logger.warning("No room to dispense this iteration; rotating plate.")
            lash_e.discard_used_wellplate()
            lash_e.grab_new_wellplate()
            current_wellplate_wells = 0
            continue

        # Recommender -> N-D points (always request the full count; cap at fit)
        points = get_next_batch(
            well_recipes_df, n_points=GRADIENT_SUGGESTIONS_PER_ITERATION,
            iteration=iteration,
        )
        if len(points) > max_this_iter:
            points = points[:max_this_iter]

        # Map suggested concentrations to existing substocks
        targets_per_surf = collect_unique_targets_per_surf(points)
        plans = build_plans_for_surfactants(
            lash_e, targets_per_surf, existing_stock_solutions=stock_solutions,
        )

        # Drop any point whose actual substock volumes exceed the dispensing
        # budget (the pre-filter uses stock concs; substocks may be more dilute).
        points = filter_points_by_actual_volumes(points, plans, lash_e.logger)
        if not points:
            lash_e.logger.warning(
                f"Iteration {iteration}: all recommended points were infeasible "
                f"after volume check; skipping iteration."
            )
            iteration += 1
            continue

        # Build new recipe rows continuing the well numbering
        next_recipes_df = build_well_recipes_df(
            points, plans, starting_well_index=current_wellplate_wells,
        )
        next_recipes_df = execute_dispensing_nd(lash_e, next_recipes_df)
        next_recipes_df = dispense_dmso(lash_e, next_recipes_df)
        next_recipes_df = measure_and_process_turbidity_nd(lash_e, next_recipes_df)
        next_recipes_df = measure_and_process_fluorescence_nd(lash_e, next_recipes_df)

        well_recipes_df = pd.concat([well_recipes_df, next_recipes_df], ignore_index=True)
        measured_count = len(well_recipes_df)
        current_wellplate_wells += len(next_recipes_df)
        save_results(well_recipes_df, output_folder, "results_iterative")

        if current_wellplate_wells >= MAX_WELLS_PER_PLATE and measured_count < TARGET_TOTAL_WELLS:
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
    lash_e.logger.info("=" * 80)
    lash_e.logger.info("N-D WORKFLOW COMPLETE")
    lash_e.logger.info(f"  Total wells: {len(well_recipes_df)}")
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

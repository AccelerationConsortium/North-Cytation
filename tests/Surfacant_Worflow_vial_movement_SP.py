"""Movement-only replay for the multidimensional surfactant workflow first plate.

This script mirrors the vial movement pattern from
workflows/surfactant_multidimensional_workflow.py for the initial plate,
without dispensing to wells.

It performs two stages:
1. Build first-plate recipes (controls + initial experiment grid) using the
   same planning helpers as the workflow.
2. Replay vial placement/return movements only for all surfactant sources,
   water/water_2, and optional buffer.

Finally, it runs a water-vial small-tip aspiration check at each safe position
to verify down-travel and aspiration access.
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.append(str(REPO_ROOT))

from master_usdl_coordinator import Lash_E
from workflows.surfactant_grid_adaptive_concentrations import (
	position_surfactant_vials_by_concentration,
	return_surfactant_vials_home,
	return_water_vial_home,
)
import workflows.surfactant_multidimensional_workflow as multidim


# ============================================================
# CONFIGURATION
# ============================================================

SIMULATE = False
INITIALIZE_BIOTEK = False

VIAL_FILE = str(REPO_ROOT / "status" / "surfactant_multidim_vials.csv")

# Use only vials currently present in VIAL_FILE for this movement test.
USE_AVAILABLE_RACK_VIALS_ONLY = True
MAX_SURFACTANTS_FROM_RACK = 3
PREFERRED_SURFACTANTS = None  # Example: ["TTAB", "SDS", "CHAPS"]
BUFFER_CANDIDATE_ORDER = ["MES", "HEPES", "CAPS", "NaCl"]

HOME_ROBOT_AT_START = True
HOME_BETWEEN_STAGES = False
HOME_AT_END = True

# Set to True to skip all surfactant/recipe logic and only move the water vial
# through SAFE_POSITIONS_FOR_WATER_TEST.
WATER_VIAL_MOVEMENT_ONLY = True

RUN_WATER_SAFE_POSITION_ASPIRATION_TEST = True
WATER_TEST_VIAL = "water"
SMALL_TIP_ASPIRATE_VOLUME_ML = 0.02  # 20 uL

# Safe position set used by position_surfactant_vials_by_concentration.
SAFE_POSITIONS_FOR_WATER_TEST = [
	("main_8mL_rack", 47),
	("main_8mL_rack", 46),
	("main_8mL_rack", 45),
	("main_8mL_rack", 44),
	("clamp", 0),
	("main_8mL_rack", 43),
	("heater", 2),
]


def _position_label(location, location_index):
	if location == "clamp":
		return "clamp[0]"
	return f"{location}[{location_index}]"


def _assert_required_vials_exist(lash_e, vial_names):
	available = set(lash_e.nr_robot.VIAL_DF["vial_name"].tolist())
	missing = sorted(set(vial_names) - available)
	if missing:
		raise ValueError(
			"Required vials are missing from the vial status file: "
			+ ", ".join(missing)
		)


def _refresh_multidim_derived_config():
	"""Recompute multidim module derived globals after SURFACTANTS changes."""
	multidim._sobol_pool = []
	multidim._sobol_pool_idx = 0

	multidim.n_surfactants = len(multidim.SURFACTANTS)
	if not (2 <= multidim.n_surfactants <= 5):
		raise ValueError(
			f"SURFACTANTS must have 2-5 entries, got {multidim.n_surfactants}: {multidim.SURFACTANTS}"
		)

	for surf_name in multidim.SURFACTANTS:
		if surf_name not in multidim.SURFACTANT_LIBRARY:
			raise ValueError(
				f"Surfactant '{surf_name}' not in SURFACTANT_LIBRARY. "
				f"Available: {list(multidim.SURFACTANT_LIBRARY.keys())}"
			)

	multidim._cube_vol_per_surf = multidim.SURFACTANT_BUDGET_UL / multidim.n_surfactants
	multidim.cube_max_conc_mm = {
		s: multidim.SURFACTANT_LIBRARY[s]["stock_conc"] * (multidim._cube_vol_per_surf / multidim.WELL_VOLUME_UL)
		for s in multidim.SURFACTANTS
	}
	_min_vol_per_other_surf_ul = multidim.WELL_VOLUME_UL / 10.0
	_simplex_budget_for_one = max(
		multidim.SURFACTANT_BUDGET_UL - (multidim.n_surfactants - 1) * _min_vol_per_other_surf_ul,
		0.0,
	)
	multidim.simplex_max_conc_mm = {
		s: multidim.SURFACTANT_LIBRARY[s]["stock_conc"] * (_simplex_budget_for_one / multidim.WELL_VOLUME_UL)
		for s in multidim.SURFACTANTS
	}

	if multidim.INIT_STRATEGY == "simplex":
		multidim.max_conc_mm = multidim.simplex_max_conc_mm
		multidim.max_vol_per_surf_ul = multidim.SURFACTANT_BUDGET_UL
	else:
		multidim.max_conc_mm = multidim.cube_max_conc_mm
		multidim.max_vol_per_surf_ul = multidim._cube_vol_per_surf

	multidim.input_cols = [f"{s}_conc_mm" for s in multidim.SURFACTANTS]
	multidim.vol_cols = [f"{s}_volume_ul" for s in multidim.SURFACTANTS]
	multidim.substock_name_cols = [f"{s}_substock_name" for s in multidim.SURFACTANTS]
	multidim.substock_conc_cols = [f"{s}_substock_conc_mm" for s in multidim.SURFACTANTS]


def _configure_multidim_from_available_vials(logger):
	"""Use only currently available rack vials for this test script."""
	if not USE_AVAILABLE_RACK_VIALS_ONLY:
		return

	vials_df = pd.read_csv(VIAL_FILE)
	available_vials = set(vials_df["vial_name"].astype(str).tolist())

	available_surfactants = sorted(
		{
			vial_name[:-6]
			for vial_name in available_vials
			if vial_name.endswith("_stock") and vial_name[:-6] in multidim.SURFACTANT_LIBRARY
		}
	)

	if PREFERRED_SURFACTANTS is not None:
		selected = list(PREFERRED_SURFACTANTS)
		missing_preferred = [s for s in selected if s not in available_surfactants]
		if missing_preferred:
			raise ValueError(
				"PREFERRED_SURFACTANTS missing in vial file: " + ", ".join(missing_preferred)
			)
	else:
		selected = []
		for surf_name in multidim.SURFACTANTS:
			if surf_name in available_surfactants and surf_name not in selected:
				selected.append(surf_name)
		for surf_name in available_surfactants:
			if surf_name not in selected:
				selected.append(surf_name)
		selected = selected[:MAX_SURFACTANTS_FROM_RACK]

	if len(selected) < 2:
		raise ValueError(
			"Need at least 2 available surfactant stock vials for this workflow test. "
			f"Found: {available_surfactants}"
		)

	multidim.SURFACTANTS = selected

	selected_buffer = None
	if multidim.ADD_BUFFER:
		for candidate in BUFFER_CANDIDATE_ORDER:
			if candidate in available_vials:
				selected_buffer = candidate
				break
		if selected_buffer is None:
			logger.info(
				"No configured buffer vial found in rack; disabling buffer moves for this test run."
			)
			multidim.ADD_BUFFER = False
		else:
			multidim.SELECTED_BUFFER = selected_buffer

	_refresh_multidim_derived_config()

	logger.info("Rack-only configuration enabled for this test")
	logger.info(f"Selected surfactants: {multidim.SURFACTANTS}")
	if multidim.ADD_BUFFER:
		logger.info(f"Selected buffer: {multidim.SELECTED_BUFFER}")
	else:
		logger.info("Buffer disabled for this run")


def _seed_sobol_pool_if_needed(logger):
	if multidim.RECOMMENDER_TYPE not in ("sobol", "random"):
		return

	from scipy.stats.qmc import Sobol as SobolInit

	multidim._sobol_pool.clear()
	multidim._sobol_pool_idx = 0

	sample_max = (
		multidim.simplex_max_conc_mm
		if multidim.INIT_STRATEGY == "simplex"
		else multidim.cube_max_conc_mm
	)
	pool_size = max(multidim.TARGET_TOTAL_WELLS * multidim.RECOMMEND_OVERSAMPLE_FACTOR * 4, 8192)
	pool_size = int(2 ** multidim.np.ceil(multidim.np.log2(pool_size)))

	seq = SobolInit(d=multidim.n_surfactants, scramble=True, seed=42)
	raw = seq.random(pool_size)
	for row in raw:
		pt = {
			s: float(
				10
				** (
					multidim.np.log10(multidim.MIN_CONC_MM)
					+ row[i]
					* (
						multidim.np.log10(sample_max[s])
						- multidim.np.log10(multidim.MIN_CONC_MM)
					)
				)
			)
			for i, s in enumerate(multidim.SURFACTANTS)
		}
		if multidim.is_feasible(pt):
			multidim._sobol_pool.append(pt)

	logger.info(
		f"Sobol pool prepared: {len(multidim._sobol_pool)}/{pool_size} feasible points"
	)


def _build_first_plate_recipes(lash_e):
	logger = lash_e.logger

	_seed_sobol_pool_if_needed(logger)

	if multidim.INIT_STRATEGY == "simplex":
		bootstrap_targets = {
			s: sorted(set([multidim.MIN_CONC_MM, multidim.simplex_max_conc_mm[s]]))
			for s in multidim.SURFACTANTS
		}
		bootstrap_plans = multidim.build_plans_for_surfactants(
			lash_e, bootstrap_targets, existing_stock_solutions=None
		)
		grid_points = multidim.generate_simplex_init(plans=bootstrap_plans, logger=logger)
		logger.info(
			f"Simplex init prepared: {len(grid_points)} points "
			f"({multidim.n_surfactants}D)"
		)
	else:
		grid_points = multidim.generate_log_grid(multidim.GRID_POINTS_PER_AXIS)
		logger.info(
			f"Grid init prepared: {len(grid_points)} points "
			f"(= {multidim.GRID_POINTS_PER_AXIS}^{multidim.n_surfactants})"
		)

	targets_per_surf = multidim.collect_unique_targets_per_surf(grid_points)
	plans = multidim.build_plans_for_surfactants(
		lash_e, targets_per_surf, existing_stock_solutions=None
	)

	points_before = len(grid_points)
	grid_points = multidim.filter_points_by_actual_volumes(grid_points, plans, logger)
	if not grid_points:
		raise RuntimeError("All initial points were filtered out by source/volume feasibility.")
	logger.info(f"Initial points kept: {len(grid_points)}/{points_before}")

	controls_df = multidim.build_control_wells_df(plans, starting_well_index=0)
	n_controls = len(controls_df)
	init_exp_df = multidim.build_well_recipes_df(grid_points, plans, starting_well_index=n_controls)

	if n_controls > 0:
		first_plate_df = pd.concat([controls_df, init_exp_df], ignore_index=True)
	else:
		first_plate_df = init_exp_df

	return first_plate_df


def _replay_first_plate_vial_movements(lash_e, first_plate_df):
	logger = lash_e.logger
	logger.info("")
	logger.info("=== Stage 1: Replaying first-plate vial placement movements ===")
	logger.info(f"First plate rows: {len(first_plate_df)}")

	for surf_name in multidim.SURFACTANTS:
		shim = multidim._shim_df_for_surfactant(first_plate_df, surf_name)
		vials_for_surf = (
			shim[shim["surf_A_volume_ul"] > 0]["substock_A_name"].dropna().unique().tolist()
		)
		if not vials_for_surf:
			logger.info(f"Skipping {surf_name}: no source vials needed on first plate")
			continue

		logger.info(f"Replaying movement for {surf_name}: {len(vials_for_surf)} vials")
		sorted_vials = position_surfactant_vials_by_concentration(
			lash_e, vials_for_surf, shim, "A"
		)
		return_surfactant_vials_home(lash_e, sorted_vials, "A")

	water_wells = first_plate_df[first_plate_df["water_volume_ul"] > 0]
	if len(water_wells) > 0:
		logger.info("Replaying water/buffer placement movements")
		lash_e.nr_robot.move_vial_to_location("water", "main_8mL_rack", 44)
		lash_e.nr_robot.move_vial_to_location("water_2", "main_8mL_rack", 45)
		if multidim.ADD_BUFFER:
			lash_e.nr_robot.move_vial_to_location(multidim.SELECTED_BUFFER, "main_8mL_rack", 47)

		return_water_vial_home(lash_e, "water")
		return_water_vial_home(lash_e, "water_2")
		if multidim.ADD_BUFFER:
			lash_e.nr_robot.return_vial_home(multidim.SELECTED_BUFFER)

	logger.info("Stage 1 complete")


def _run_water_safe_position_aspiration_test(lash_e):
	logger = lash_e.logger
	robot = lash_e.nr_robot

	logger.info("")
	logger.info("=== Stage 2: Water vial safe-position aspiration check (small tip) ===")
	logger.info(
		f"Aspirate volume per position: {SMALL_TIP_ASPIRATE_VOLUME_ML * 1000:.0f}uL"
	)

	for idx, (location, location_index) in enumerate(SAFE_POSITIONS_FOR_WATER_TEST, start=1):
		pos_label = _position_label(location, location_index)
		logger.info("")
		logger.info(f"Position {idx}/{len(SAFE_POSITIONS_FOR_WATER_TEST)}: {pos_label}")

		robot.move_vial_to_location(WATER_TEST_VIAL, location, location_index)
		robot.get_pipet("small_tip")

		# Aspirate then dispense back into the same vial to verify down-travel
		# and liquid pickup while preserving vial inventory.
		robot.aspirate_from_vial(
			WATER_TEST_VIAL,
			SMALL_TIP_ASPIRATE_VOLUME_ML,
			liquid="water",
			specified_tip="small_tip",
		)
		robot.dispense_into_vial(
			WATER_TEST_VIAL,
			SMALL_TIP_ASPIRATE_VOLUME_ML,
			liquid="water",
			initial_move=True,
		)

		robot.remove_pipet()
		robot.return_vial_home(WATER_TEST_VIAL)

	logger.info("Stage 2 complete")


def run_surfactant_workflow_movement_test():
	lash_e = Lash_E(
		vial_file=VIAL_FILE,
		simulate=SIMULATE,
		initialize_biotek=INITIALIZE_BIOTEK,
	)

	logger = lash_e.logger
	logger.info("Starting surfactant workflow movement replay test")
	logger.info(f"Simulation mode: {SIMULATE}")
	logger.info(f"Workflow init strategy: {multidim.INIT_STRATEGY}")

	_configure_multidim_from_available_vials(logger)
	logger.info(f"Workflow surfactants: {multidim.SURFACTANTS}")

	if HOME_ROBOT_AT_START:
		logger.info("Homing robot components before test")
		lash_e.nr_robot.home_robot_components()

	if WATER_VIAL_MOVEMENT_ONLY:
		logger.info("WATER_VIAL_MOVEMENT_ONLY=True: skipping surfactant recipe build and vial movements")
		_assert_required_vials_exist(lash_e, [WATER_TEST_VIAL])
		_run_water_safe_position_aspiration_test(lash_e)
	else:
		first_plate_df = _build_first_plate_recipes(lash_e)

		required_vials = set()
		for surf_name in multidim.SURFACTANTS:
			substock_col = f"{surf_name}_substock_name"
			if substock_col in first_plate_df.columns:
				required_vials.update(first_plate_df[substock_col].dropna().tolist())
		required_vials.update(["water", "water_2", WATER_TEST_VIAL])
		if multidim.ADD_BUFFER:
			required_vials.add(multidim.SELECTED_BUFFER)
		_assert_required_vials_exist(lash_e, required_vials)

		_replay_first_plate_vial_movements(lash_e, first_plate_df)

		if HOME_BETWEEN_STAGES:
			logger.info("Homing robot components between stage 1 and stage 2")
			lash_e.nr_robot.home_robot_components()

		if RUN_WATER_SAFE_POSITION_ASPIRATION_TEST:
			_run_water_safe_position_aspiration_test(lash_e)

	if HOME_AT_END:
		logger.info("Homing robot components at end of test")
		lash_e.nr_robot.home_robot_components()

	logger.info("Movement replay test complete")


if __name__ == "__main__":
	run_surfactant_workflow_movement_test()

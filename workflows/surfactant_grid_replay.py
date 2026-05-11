# -*- coding: utf-8 -*-
"""
Surfactant Grid Replay Workflow

Re-runs a previous 192-row surfactant grid experiment from its CSV outputs in
one continuous pass (no optimization, no iterations). Refills are triggered
mid-run by a simple volume threshold check between sub-chunks of each
component dispense, so a 192-well grid can be completed without manual
intervention.

Inputs (from a prior iterative run's output folder):
  - iterative_experiment_results.csv (or complete_experiment_results.csv)
      Per-well recipes: 192 rows with substock_A/B_name, volumes, etc.
  - experiment_plan_stock_solutions.csv
      Substock dilution recipes (Vial_Name, Source_Vial, source/water volumes).

Refill rules:
  - Check current vial volume after every REFILL_CHECK_CHUNK_SIZE wells of a
    given component dispense (and before the first chunk).
  - If volume < REFILL_THRESHOLD_ML, refill to max for that vial kind:
      water    -> 8 mL via fill_water_vial
      stock    -> 7.5 mL via refill_surfactant_vial
      substock -> FINAL_SUBSTOCK_VOLUME_ML via create_substocks_from_recipes
                  filtered to that single recipe
      buffer   -> raises (no automated refill source)

This file imports all heavy lifting from
workflows.surfactant_grid_adaptive_concentrations to avoid duplication.
"""

import sys
sys.path.append("../utoronto_demo")
import os
import pandas as pd
from datetime import datetime

from master_usdl_coordinator import Lash_E

from workflows.surfactant_grid_adaptive_concentrations import (
    # constants
    MEASUREMENT_INTERVAL,
    WELL_VOLUME_UL,
    BUFFER_VOLUME_UL,
    PYRENE_VOLUME_UL,
    ADD_BUFFER,
    SELECTED_BUFFER,
    INPUT_VIAL_STATUS_FILE,
    FINAL_SUBSTOCK_VOLUME_ML,
    REFILL_THRESHOLD_ML,
    # helpers reused as-is
    setup_experiment_environment,
    create_substocks_from_recipes,
    dispense_component_to_wellplate,
    position_surfactant_vials_by_concentration,
    return_surfactant_vials_home,
    return_water_vial_home,
    dispense_dmso,
    measure_and_process_turbidity,
    measure_and_process_fluorescence,
    fill_water_vial,
    refill_surfactant_vial,
    get_pipette_usage_breakdown,
    run_post_experiment_analysis,
    generate_surfactant_grid_heatmaps,
)

# ============================================================================
# CONFIG
# ============================================================================

# Source CSVs (override at __main__ entry point if desired)
SOURCE_EXPERIMENT_FOLDER = (
    "output/surfactant_grid_DSS_BZT_May_06_Experiment_20260506_160331"
)
RECIPES_CSV = "iterative_experiment_results.csv"
STOCKS_CSV = "experiment_plan_stock_solutions.csv"

SIMULATE = True

# Refill cadence within each component dispense.
REFILL_CHECK_CHUNK_SIZE = 24

# Tag appended to the new experiment folder name.
REPLAY_TAG = "replay"


# ============================================================================
# INPUT LOADING
# ============================================================================

def load_replay_inputs(recipes_path, stocks_path):
    """Load 192-row recipes and the substock dilution recipes used to build them.

    Returns:
        well_recipes_df: DataFrame matching the schema of
            iterative_experiment_results.csv (wellplate_index, well_type,
            substock_A_name, surf_A_volume_ul, ...).
        dilution_recipes: list of dicts in the format expected by
            create_substocks_from_recipes (Vial_Name, Source_Vial,
            Source_Volume_mL, Water_Volume_mL, Final_Volume_mL, Surfactant,
            Target_Conc_mM, Source_Conc_mM).
    """
    well_recipes_df = pd.read_csv(recipes_path)

    stocks_df = pd.read_csv(stocks_path)
    # Translate the stock_solutions CSV column names to the keys
    # create_substocks_from_recipes expects.
    rename = {
        "vial_name": "Vial_Name",
        "surfactant": "Surfactant",
        "target_concentration_mm": "Target_Conc_mM",
        "source_vial": "Source_Vial",
        "source_concentration_mm": "Source_Conc_mM",
        "source_volume_ml": "Source_Volume_mL",
        "water_volume_ml": "Water_Volume_mL",
        "final_volume_ml": "Final_Volume_mL",
    }
    stocks_df = stocks_df.rename(columns=rename)
    dilution_recipes = stocks_df.to_dict(orient="records")
    # Sort highest-concentration first so dependencies are made before children
    # (matches the order create_substocks_from_recipes is called with elsewhere).
    dilution_recipes.sort(key=lambda r: float(r["Target_Conc_mM"]), reverse=True)
    return well_recipes_df, dilution_recipes


def split_into_plates(well_recipes_df):
    """Split the recipes DataFrame into one DataFrame per physical wellplate.

    The iterative workflow writes wellplate_index in 0..95 ranges, with the
    index resetting to 0 when a new plate is started. We reproduce that.
    """
    plates = []
    start = 0
    for i in range(1, len(well_recipes_df)):
        prev_idx = well_recipes_df.iloc[i - 1]["wellplate_index"]
        curr_idx = well_recipes_df.iloc[i]["wellplate_index"]
        if curr_idx <= prev_idx:
            plates.append(well_recipes_df.iloc[start:i].reset_index(drop=True))
            start = i
    plates.append(well_recipes_df.iloc[start:].reset_index(drop=True))
    return plates


# ============================================================================
# REFILL ENGINE
# ============================================================================

def _classify_vial(vial_name, dilution_recipes):
    """Return ('water'|'stock'|'substock'|'buffer', recipe_or_none)."""
    if vial_name in ("water", "water_2"):
        return "water", None
    if vial_name.endswith("_stock"):
        return "stock", None
    for r in dilution_recipes:
        if r["Vial_Name"] == vial_name:
            return "substock", r
    if ADD_BUFFER and vial_name == SELECTED_BUFFER:
        return "buffer", None
    raise ValueError(f"Cannot classify vial '{vial_name}' for refill routing")


def ensure_vial_above_threshold(lash_e, vial_name, dilution_recipes):
    """Check current volume; refill to max if < REFILL_THRESHOLD_ML.

    Returns True if a refill was performed (caller should re-condition tip).
    Raises if a buffer vial is below threshold (no auto-refill source).
    """
    current_ml = lash_e.nr_robot.get_vial_info(vial_name, "vial_volume")
    if current_ml is None:
        raise ValueError(f"No volume tracked for vial '{vial_name}'")
    if current_ml >= REFILL_THRESHOLD_ML:
        return False

    kind, recipe = _classify_vial(vial_name, dilution_recipes)
    lash_e.logger.info(
        f"  REFILL: {vial_name} at {current_ml:.2f} mL < {REFILL_THRESHOLD_ML} mL "
        f"(kind={kind})"
    )

    # Drop tip before any vial-moving operation.
    lash_e.nr_robot.remove_pipet()

    if kind == "water":
        fill_water_vial(lash_e, vial_name)
    elif kind == "stock":
        refill_surfactant_vial(lash_e, vial_name, liquid="SDS")
    elif kind == "substock":
        create_substocks_from_recipes(lash_e, [recipe])
    elif kind == "buffer":
        raise RuntimeError(
            f"Buffer vial '{vial_name}' at {current_ml:.2f} mL is below "
            f"{REFILL_THRESHOLD_ML} mL and has no automated refill source"
        )
    return True


def _dispense_vial_in_chunks(
    lash_e,
    batch_df,
    vial_name,
    liquid_type,
    volume_column,
    dilution_recipes,
    should_condition_first,
):
    """Dispense one vial's wells in REFILL_CHECK_CHUNK_SIZE-row sub-chunks.

    A volume check + optional refill runs before each chunk. If a refill
    happens, the next chunk re-conditions the tip.
    """
    # Pre-filter to only the wells this vial serves, mirroring the filter
    # logic inside dispense_component_to_wellplate.
    if volume_column == "surf_A_volume_ul":
        wells = batch_df[
            (batch_df[volume_column] > 0) & (batch_df["substock_A_name"] == vial_name)
        ]
    elif volume_column == "surf_B_volume_ul":
        wells = batch_df[
            (batch_df[volume_column] > 0) & (batch_df["substock_B_name"] == vial_name)
        ]
    else:
        wells = batch_df[batch_df[volume_column] > 0]

    if len(wells) == 0:
        return

    should_condition = should_condition_first
    for chunk_start in range(0, len(wells), REFILL_CHECK_CHUNK_SIZE):
        chunk_df = wells.iloc[chunk_start : chunk_start + REFILL_CHECK_CHUNK_SIZE]
        refilled = ensure_vial_above_threshold(lash_e, vial_name, dilution_recipes)
        if refilled:
            should_condition = True
        dispense_component_to_wellplate(
            lash_e,
            chunk_df,
            vial_name,
            liquid_type,
            volume_column,
            should_condition_tip=should_condition,
        )
        should_condition = False


# ============================================================================
# PER-PLATE DISPENSING
# ============================================================================

def execute_dispensing_with_refills(lash_e, plate_df, dilution_recipes):
    """Dispense one plate of recipes (<= MEASUREMENT_INTERVAL wells).

    Mirrors the per-component structure of execute_dispensing in the original
    workflow but sub-chunks within each vial's dispense for refill checks.
    """
    lash_e.logger.info(
        f"Dispensing plate ({len(plate_df)} wells) with chunked refill checks "
        f"(chunk={REFILL_CHECK_CHUNK_SIZE}, threshold={REFILL_THRESHOLD_ML} mL)"
    )

    surf_a_vials = (
        plate_df[plate_df["surf_A_volume_ul"] > 0]["substock_A_name"].dropna().unique()
    )
    surf_b_vials = (
        plate_df[plate_df["surf_B_volume_ul"] > 0]["substock_B_name"].dropna().unique()
    )

    # ---- Surfactant A ----
    if len(surf_a_vials) > 0:
        sorted_a = position_surfactant_vials_by_concentration(
            lash_e, surf_a_vials, plate_df, "A"
        )
        for i, vial in enumerate(sorted_a):
            should_condition_first = i == 0
            _dispense_vial_in_chunks(
                lash_e,
                plate_df,
                vial,
                "SDS",
                "surf_A_volume_ul",
                dilution_recipes,
                should_condition_first=should_condition_first,
            )
        lash_e.nr_robot.remove_pipet()
        return_surfactant_vials_home(lash_e, sorted_a, "A")

    # ---- Water (+ optional buffer) ----
    water_wells = plate_df[plate_df["water_volume_ul"] > 0]
    if len(water_wells) > 0:
        # Sort ascending so any large-tip wells (>=200uL) end up in the second
        # half (water_2), matching the original logic.
        water_wells = water_wells.sort_values("water_volume_ul", ascending=True)
        mid = len(water_wells) // 2
        water_batch_1_idx = water_wells.iloc[:mid]["wellplate_index"].tolist()
        water_batch_2_idx = water_wells.iloc[mid:]["wellplate_index"].tolist()
        water_1_df = plate_df[plate_df["wellplate_index"].isin(water_batch_1_idx)]
        water_2_df = plate_df[plate_df["wellplate_index"].isin(water_batch_2_idx)]

        lash_e.nr_robot.move_vial_to_location("water", "main_8mL_rack", 44)
        lash_e.nr_robot.move_vial_to_location("water_2", "main_8mL_rack", 45)
        if ADD_BUFFER:
            lash_e.nr_robot.move_vial_to_location(SELECTED_BUFFER, "main_8mL_rack", 47)

        if len(water_1_df) > 0:
            _dispense_vial_in_chunks(
                lash_e,
                water_1_df,
                "water",
                "water",
                "water_volume_ul",
                dilution_recipes,
                should_condition_first=True,
            )
        if len(water_2_df) > 0:
            _dispense_vial_in_chunks(
                lash_e,
                water_2_df,
                "water_2",
                "water",
                "water_volume_ul",
                dilution_recipes,
                should_condition_first=True,
            )

        if ADD_BUFFER:
            _dispense_vial_in_chunks(
                lash_e,
                plate_df,
                SELECTED_BUFFER,
                "water",
                "buffer_volume_ul",
                dilution_recipes,
                should_condition_first=False,
            )

        lash_e.nr_robot.remove_pipet()
        return_water_vial_home(lash_e, "water")
        return_water_vial_home(lash_e, "water_2")
        if ADD_BUFFER:
            lash_e.nr_robot.return_vial_home(SELECTED_BUFFER)

    # ---- Surfactant B ----
    if len(surf_b_vials) > 0:
        sorted_b = position_surfactant_vials_by_concentration(
            lash_e, surf_b_vials, plate_df, "B"
        )
        for i, vial in enumerate(sorted_b):
            should_condition_first = i == 0
            _dispense_vial_in_chunks(
                lash_e,
                plate_df,
                vial,
                "SDS",
                "surf_B_volume_ul",
                dilution_recipes,
                should_condition_first=should_condition_first,
            )
        lash_e.nr_robot.remove_pipet()
        return_surfactant_vials_home(lash_e, sorted_b, "B")


# ============================================================================
# TOP-LEVEL WORKFLOW
# ============================================================================

def execute_replay_workflow(
    recipes_path,
    stocks_path,
    lash_e,
    simulate=True,
):
    """Replay a 192-row grid CSV in one pass with refill checks."""
    lash_e.logger.info("=" * 80)
    lash_e.logger.info("SURFACTANT GRID REPLAY WORKFLOW")
    lash_e.logger.info(f"Recipes:  {recipes_path}")
    lash_e.logger.info(f"Stocks:   {stocks_path}")
    lash_e.logger.info(f"Simulate: {simulate}")
    lash_e.logger.info("=" * 80)

    well_recipes_df, dilution_recipes = load_replay_inputs(recipes_path, stocks_path)
    surfactant_a_name = well_recipes_df["surf_A"].dropna().iloc[0]
    surfactant_b_name = well_recipes_df["surf_B"].dropna().iloc[0]
    lash_e.logger.info(
        f"Surfactants: {surfactant_a_name} + {surfactant_b_name}, "
        f"{len(well_recipes_df)} rows, {len(dilution_recipes)} substocks"
    )

    # Standard env setup (creates output folder + experiment_name on lash_e).
    experiment_output_folder, experiment_name = setup_experiment_environment(
        lash_e, surfactant_a_name, f"{surfactant_b_name}_{REPLAY_TAG}", simulate
    )

    # Top off water + stocks once at start.
    lash_e.logger.info("Pre-run: filling water + stock vials to capacity")
    fill_water_vial(lash_e, "water")
    fill_water_vial(lash_e, "water_2")
    refill_surfactant_vial(lash_e, f"{surfactant_a_name}_stock", liquid="SDS")
    refill_surfactant_vial(lash_e, f"{surfactant_b_name}_stock", liquid="SDS")

    # Build / top up substocks from the source CSV recipes.
    lash_e.logger.info("Pre-run: creating/refilling substocks from source recipes")
    create_substocks_from_recipes(lash_e, dilution_recipes)

    # Split into plates and run each plate end-to-end.
    plate_dfs = split_into_plates(well_recipes_df)
    lash_e.logger.info(f"Source CSV splits into {len(plate_dfs)} plate(s)")

    completed_plates = []
    lash_e.nr_robot.home_robot_components()

    for plate_idx, plate_df in enumerate(plate_dfs):
        lash_e.logger.info(
            f"\n--- PLATE {plate_idx + 1}/{len(plate_dfs)} "
            f"({len(plate_df)} wells) ---"
        )
        lash_e.grab_new_wellplate()

        execute_dispensing_with_refills(lash_e, plate_df, dilution_recipes)
        plate_df = dispense_dmso(lash_e, plate_df.copy())
        plate_df = measure_and_process_turbidity(
            lash_e, plate_df, shake_and_wait=True
        )
        plate_df = measure_and_process_fluorescence(
            lash_e, plate_df, shake_and_wait=False
        )

        completed_plates.append(plate_df)
        lash_e.discard_used_wellplate()

    # Stitch plates and save.
    final_df = pd.concat(completed_plates, ignore_index=True)
    final_csv = os.path.join(experiment_output_folder, "complete_experiment_results.csv")
    final_df.to_csv(final_csv, index=False)
    lash_e.logger.info(f"Saved replay results: {final_csv}")

    # Heatmaps + post-experiment analysis (best-effort).
    try:
        heatmap_folder = os.path.join(experiment_output_folder, "heatmap")
        os.makedirs(heatmap_folder, exist_ok=True)
        generate_surfactant_grid_heatmaps(
            final_csv, heatmap_folder, lash_e.logger,
            surfactant_a_name, surfactant_b_name,
        )
    except Exception as e:
        lash_e.logger.warning(f"Heatmap generation failed: {e}")

    try:
        run_post_experiment_analysis(
            final_csv, experiment_output_folder,
            surfactant_a_name, surfactant_b_name, lash_e.logger,
        )
    except Exception as e:
        lash_e.logger.warning(f"Post-experiment analysis failed: {e}")

    pipette = get_pipette_usage_breakdown(lash_e)
    lash_e.logger.info(
        f"Pipette tips used: large={pipette['large_tips']} small={pipette['small_tips']} "
        f"total={pipette['total']}"
    )
    lash_e.logger.info("REPLAY WORKFLOW COMPLETE")

    return {
        "well_recipes_df": final_df,
        "output_folder": experiment_output_folder,
        "experiment_name": experiment_name,
        "pipette_breakdown": pipette,
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    recipes_path = os.path.join(SOURCE_EXPERIMENT_FOLDER, RECIPES_CSV)
    stocks_path = os.path.join(SOURCE_EXPERIMENT_FOLDER, STOCKS_CSV)

    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
    execute_replay_workflow(recipes_path, stocks_path, lash_e, simulate=SIMULATE)

# -*- coding: utf-8 -*-
"""
Surfactant Grid Study Workflow

Re-runs a previous surfactant grid experiment from its CSV outputs in one
continuous pass. This workflow owns its top-level control flow so study metadata,
replicates, randomization, and dye/solvent behavior can be added here without
changing the baseline replay workflow.

Current status:
- DYE selects pyrene, nile-red, or coumarin-6, with its vial and protocol looked up automatically.
- Study config fields are loaded through the standard workflow config system.
- Dye dispensing is local to this workflow and uses DYE_SOLVENT as the dispense liquid.
- coumarin-6 and nile-red protocols are real/implemented for hardware runs, but
  simulate_dye_fluorescence() only generates synthetic data for DYE="pyrene" -
  running SIMULATE=True with those dyes will still raise NotImplementedError.
"""

import sys
sys.path.append("../utoronto_demo")
import os
import time
import math
import random
import pandas as pd

from master_usdl_coordinator import Lash_E, flatten_cytation_data

from workflows.surfactant_grid_adaptive_concentrations import (
    ADD_BUFFER,
    REFILL_THRESHOLD_ML,
    SHAKE_WAIT_PROTOCOL,
    SELECTED_BUFFER,
    TURBIDITY_PROTOCOL_FILE,
    create_substocks_from_recipes,
    dispense_component_to_wellplate,
    fill_water_vial,
    get_pipette_usage_breakdown,
    position_surfactant_vials_by_concentration,
    refill_surfactant_vial,
    return_surfactant_vials_home,
    return_water_vial_home,
    run_post_experiment_analysis,
    simulate_surfactant_measurements,
    setup_experiment_environment,
)

# ============================================================================
# CONFIG
# ============================================================================

# Dedicated vial layout for this workflow (SDS + BDDAC) - do not reuse the
# shared adaptive_concentrations vial file, which other experiments rely on.
INPUT_VIAL_STATUS_FILE = "status/surfactant_grid_ailsa_vials.csv"

# Per-well recipe CSV (override in workflow_configs/surfactant_grid_ailsa.yaml).
# Other candidate recipe files in inputs/: active_learning_first73_96well_recipe.csv,
# swap_refined73_scored_96well_recipe.csv.
RECIPES_CSV = "inputs/gradient_proposal_snapped_96well_recipe.csv"
STOCKS_CSV = "inputs/experiment_plan_stock_solutions_SDS_BDDAC.csv"

SIMULATE = False
VALIDATE_LIQUIDS = False
VALIDATION_ONLY = False

# Set to 0 to run all wells. Set to e.g. 96 to run only the first plate,
# or any other number to stop after that many wells.
MAX_WELLS = 96
WELLPLATE_TYPE = "96 WELL PLATE"
VOLUME_SCALE_BY_PLATE = {"96 WELL PLATE": 1, "48 WELL PLATE": 2, "24 WELL PLATE": 4}

_RECIPE_VOLUME_COLUMNS = (
    "surf_A_volume_ul", "surf_B_volume_ul", "water_volume_ul", "buffer_volume_ul",
)

# Select one dye for this run. Its vial and fluorescence protocol are looked
# up from the mappings below; solvent and dispense volume remain shared.
DYE = "pyrene"
DYE_SOLVENT = "DMSO"
DYE_VOLUME_UL = 5
# "buffer" omitted: this recipe set has ADD_BUFFER=False and buffer_volume_ul=0 throughout.
DISPENSE_ORDER = ["surfactant_A", "water", "surfactant_B", "dye"]

PREPARATION_REPLICATES = 3
MEASUREMENT_REPLICATES = 3
# Minutes since a block finishes dispensing; [0] preserves the current single read.
MEASUREMENT_SCHEDULE_MIN = [0]

RANDOMIZED_ORDER = False
RANDOMIZATION_SEED = None
RANDOMIZED_MEASUREMENT_ORDER = False
MEASUREMENT_RANDOMIZATION_SEED = None
EXECUTION_BLOCK_SIZE = 96

# Refill cadence within each component dispense.
REFILL_CHECK_CHUNK_SIZE = 24

# Water vials are topped up before every chunk. Setting the threshold to 8.0
# ensures the check always triggers (vial can never be at 8.0 after dispensing).
# fill_water_vial skips internally if already within 100uL of full, so no wasted
# movement if a chunk used very little water.
# Substocks/stocks continue using REFILL_THRESHOLD_ML = 4.0 mL.
WATER_REFILL_THRESHOLD_ML = 8.0

# Tag appended to the new experiment folder name.
EXPERIMENT_TAG = "study"

# Snapshot of the constants above (plus the imported ones ConfigManager has
# historically persisted for this workflow) used by execute()/ConfigManager.
_CONFIG_KEYS = [
    "INPUT_VIAL_STATUS_FILE", "RECIPES_CSV", "STOCKS_CSV", "SIMULATE",
<<<<<<< HEAD
    "MAX_WELLS", "WELLPLATE_TYPE", "VOLUME_SCALE_BY_PLATE", "DYE",
    "DYE_SOLVENT", "DYE_VOLUME_UL", "DISPENSE_ORDER", "PREPARATION_REPLICATES",
=======
    "MAX_WELLS", "WELLPLATE_TYPE", "VOLUME_SCALE_BY_PLATE", "DYE", "DYE_SOLVENT", "DYE_VIAL",
    "VALIDATE_LIQUIDS", "VALIDATION_ONLY",
    "DYE_VOLUME_UL", "DISPENSE_ORDER", "PREPARATION_REPLICATES",
>>>>>>> 5e52d64edb03cb5865f93a8e60353d0ff93adf1a
    "MEASUREMENT_REPLICATES", "MEASUREMENT_SCHEDULE_MIN", "RANDOMIZED_ORDER", "RANDOMIZATION_SEED",
    "RANDOMIZED_MEASUREMENT_ORDER", "MEASUREMENT_RANDOMIZATION_SEED",
    "EXECUTION_BLOCK_SIZE", "REFILL_CHECK_CHUNK_SIZE",
    "WATER_REFILL_THRESHOLD_ML", "EXPERIMENT_TAG",
    "ADD_BUFFER", "REFILL_THRESHOLD_ML", "SHAKE_WAIT_PROTOCOL",
    "SELECTED_BUFFER", "TURBIDITY_PROTOCOL_FILE",
]

# Safe rack positions assigned by position_surfactant_vials_by_concentration,
# in the same order that function uses (concentrated -> dilute).
_SURF_SAFE_POSITIONS = [47, 46, 45, 44, "clamp", 43, 36]

_DYE_PROTOCOLS = {
    "pyrene": {
        "implemented": True,
        "fluorescence_protocol_file": r"C:\Protocols\CMC_Fluorescence_96.prt",
        "raw_column_mapping": {
            "334_373": "fluorescence_334_373",
            "334_384": "fluorescence_334_384",
        },
        "primary_metric": "ratio",
        "metric_kind": "ratio",
        "measurement_value_columns": (
            "fluorescence_334_373",
            "fluorescence_334_384",
            "ratio",
        ),
    },
    "coumarin-6": {
        "implemented": True,
        "fluorescence_protocol_file": r"C:\Protocols\Coumarin_96.prt",
        "raw_column_mapping": {
            "485_530": "fluorescence_485_530",
        },
        "primary_metric": "fluorescence_485_530",
        "metric_kind": "intensity",
        "measurement_value_columns": ("fluorescence_485_530",),
    },
    "nile-red": {
        "implemented": True,
        "fluorescence_protocol_file": r"C:\Protocols\NileRed_96.prt",
        "raw_column_mapping": {
            "550_648": "fluorescence_550_648",
        },
        "primary_metric": "fluorescence_550_648",
        "metric_kind": "intensity",
        "measurement_value_columns": ("fluorescence_550_648",),
    },
}

_DYE_VIALS = {
    "pyrene": "dye_vial",
    "nile-red": "dye_nile_red",
    "coumarin-6": "dye_coumarin_6",
}


# ============================================================================
# STUDY METADATA
# ============================================================================

def get_dye_protocol():
    """Return the active dye protocol with runtime dispense settings."""
    if DYE not in _DYE_PROTOCOLS:
        valid_dyes = ", ".join(sorted(_DYE_PROTOCOLS))
        raise ValueError(f"Unknown DYE '{DYE}'. Valid dyes: {valid_dyes}")
    dye_vial = _DYE_VIALS[DYE]
    if DYE_VOLUME_UL <= 0:
        raise ValueError(f"DYE_VOLUME_UL must be positive, got {DYE_VOLUME_UL}")

    protocol = dict(_DYE_PROTOCOLS[DYE])
    protocol["dye"] = DYE
    protocol["dye_solvent"] = DYE_SOLVENT
    protocol["dye_vial"] = dye_vial
    protocol["dye_volume_ul"] = DYE_VOLUME_UL * get_volume_scale()
    return protocol


def get_volume_scale():
    """Scale the base 96-well recipe for the selected physical plate."""
    if WELLPLATE_TYPE not in VOLUME_SCALE_BY_PLATE:
        raise ValueError(f"No volume scale configured for WELLPLATE_TYPE '{WELLPLATE_TYPE}'")
    scale = VOLUME_SCALE_BY_PLATE[WELLPLATE_TYPE]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError(f"Invalid volume scale for {WELLPLATE_TYPE}: {scale}")
    return scale


def scale_well_recipes(well_recipes_df, lash_e):
    """Return a scaled copy; never change the source recipe file or stock recipes."""
    scale = get_volume_scale()
    missing = [column for column in _RECIPE_VOLUME_COLUMNS if column not in well_recipes_df]
    if missing:
        raise KeyError(f"Recipe missing volume columns: {missing}")
    scaled = well_recipes_df.copy()
    for column in _RECIPE_VOLUME_COLUMNS:
        scaled[column] = pd.to_numeric(scaled[column], errors="raise") * scale
    if "pyrene_volume_ul" in scaled:
        scaled["pyrene_volume_ul"] = pd.to_numeric(scaled["pyrene_volume_ul"], errors="raise") * scale
    total_ul = scaled[list(_RECIPE_VOLUME_COLUMNS)].fillna(0).sum(axis=1) + DYE_VOLUME_UL * scale
    capacity_ml = lash_e.nr_robot.WELLPLATES[WELLPLATE_TYPE].get("max_volume_per_well")
    if capacity_ml is not None and (total_ul > capacity_ml * 1000).any():
        raise ValueError(
            f"Scaled recipe exceeds {WELLPLATE_TYPE} capacity of {capacity_ml * 1000:g} uL "
            f"(maximum {total_ul.max():g} uL)"
        )
    return scaled


def get_wellplate_size(lash_e):
    """Return the configured wellplate capacity from robot_state/wellplates.yaml."""
    wellplates = getattr(lash_e.nr_robot, "WELLPLATES", {})
    if WELLPLATE_TYPE not in wellplates:
        valid_types = ", ".join(sorted(wellplates))
        raise ValueError(f"Unknown WELLPLATE_TYPE '{WELLPLATE_TYPE}'. Valid types: {valid_types}")
    num_wells = wellplates[WELLPLATE_TYPE]["num_wells"]
    if num_wells <= 0:
        raise ValueError(f"WELLPLATE_TYPE '{WELLPLATE_TYPE}' has invalid num_wells={num_wells}")
    return int(num_wells)


def get_wells_per_row():
    """Return the number of well columns per row for the configured wellplate type."""
    if WELLPLATE_TYPE in ("96 WELL PLATE", "96 QUARTZ LID", "quartz"):
        return 12
    if WELLPLATE_TYPE == "48 WELL PLATE":
        return 8
    if WELLPLATE_TYPE == "24 WELL PLATE":
        return 6
    raise ValueError(f"Unsupported WELLPLATE_TYPE for well label conversion: {WELLPLATE_TYPE}")


def well_position_to_index(well_position):
    """Convert a Cytation well label like A1 to a zero-based wellplate index."""
    wells_per_row = get_wells_per_row()
    return (ord(well_position[0]) - ord("A")) * wells_per_row + (int(well_position[1:]) - 1)


def validate_execution_block_size(lash_e):
    """Validate execution block size against sample and wellplate capacity."""
    wellplate_size = get_wellplate_size(lash_e)
    if EXECUTION_BLOCK_SIZE <= 0:
        raise ValueError(f"EXECUTION_BLOCK_SIZE must be positive, got {EXECUTION_BLOCK_SIZE}")
    return wellplate_size


def validate_measurement_schedule():
    """Require a first read at zero and increasing, finite minute offsets."""
    schedule = MEASUREMENT_SCHEDULE_MIN
    if (not isinstance(schedule, (list, tuple)) or not schedule or schedule[0] != 0
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in schedule)
            or any(later <= earlier for earlier, later in zip(schedule, schedule[1:]))):
        raise ValueError("MEASUREMENT_SCHEDULE_MIN must start at 0 and strictly increase")
    return schedule


def get_measurement_well_order(block_df, randomizer):
    """Choose one physical well sequence for both assays in a measurement read."""
    wells = sorted(block_df["wellplate_index"].tolist())
    if RANDOMIZED_MEASUREMENT_ORDER:
        randomizer.shuffle(wells)
    return wells


def get_measurement_value_columns():
    """Return numeric measurement columns used for per-measurement means."""
    measurement_columns = ["turbidity_600", *get_dye_protocol()["measurement_value_columns"]]
    if "fluorescence_metric" not in measurement_columns:
        measurement_columns.append("fluorescence_metric")
    return tuple(measurement_columns)


def add_fluorescence_metric(results_df):
    """Add the dye-generic fluorescence_metric analysis column."""
    dye_protocol = get_dye_protocol()
    primary_metric = dye_protocol["primary_metric"]
    if primary_metric not in results_df.columns:
        raise KeyError(
            f"Primary fluorescence metric '{primary_metric}' is missing. "
            f"Available columns: {list(results_df.columns)}"
        )

    results_df = results_df.copy()
    results_df["fluorescence_metric"] = results_df[primary_metric]
    results_df["fluorescence_metric_source"] = primary_metric
    return results_df


def get_study_config():
    """Return the current study configuration dictionary."""
    dye_protocol = get_dye_protocol()
    return {
        "dye": dye_protocol["dye"],
        "dye_solvent": DYE_SOLVENT,
        "dye_vial": dye_protocol["dye_vial"],
        "dye_volume_ul": dye_protocol["dye_volume_ul"],
        "base_dye_volume_ul": DYE_VOLUME_UL,
        "volume_scale": get_volume_scale(),
        "dispense_order": DISPENSE_ORDER,
        "wellplate_type": WELLPLATE_TYPE,
        "fluorescence_protocol_file": dye_protocol["fluorescence_protocol_file"],
        "primary_fluorescence_metric": dye_protocol["primary_metric"],
        "fluorescence_metric_kind": dye_protocol["metric_kind"],
        "preparation_replicates": PREPARATION_REPLICATES,
        "measurement_replicates": MEASUREMENT_REPLICATES,
        "measurement_schedule_min": MEASUREMENT_SCHEDULE_MIN,
        "randomized_order": RANDOMIZED_ORDER,
        "randomization_seed": RANDOMIZATION_SEED,
        "randomized_measurement_order": RANDOMIZED_MEASUREMENT_ORDER,
        "measurement_randomization_seed": MEASUREMENT_RANDOMIZATION_SEED,
        "execution_block_size": EXECUTION_BLOCK_SIZE,
    }


def add_study_metadata(results_df, preparation_replicate):
    """Add study metadata columns to the result table."""
    dye_protocol = get_dye_protocol()
    results_df = results_df.copy()
    results_df["dye"] = dye_protocol["dye"]
    results_df["dye_solvent"] = DYE_SOLVENT
    results_df["dye_vial"] = dye_protocol["dye_vial"]
    results_df["dye_volume_ul"] = dye_protocol["dye_volume_ul"]
    results_df["base_dye_volume_ul"] = DYE_VOLUME_UL
    results_df["volume_scale"] = get_volume_scale()
    results_df["dispense_order"] = "|".join(DISPENSE_ORDER)
    results_df["wellplate_type"] = WELLPLATE_TYPE
    results_df["preparation_replicates"] = PREPARATION_REPLICATES
    results_df["measurement_replicates"] = MEASUREMENT_REPLICATES
    results_df["measurement_schedule_min"] = "|".join(map(str, MEASUREMENT_SCHEDULE_MIN))
    results_df["preparation_replicate"] = preparation_replicate
    if "measurement_replicate" not in results_df.columns:
        results_df["measurement_replicate"] = 1
    results_df["randomized_order"] = RANDOMIZED_ORDER
    results_df["randomization_seed"] = RANDOMIZATION_SEED
    results_df["randomized_measurement_order"] = RANDOMIZED_MEASUREMENT_ORDER
    results_df["measurement_randomization_seed"] = MEASUREMENT_RANDOMIZATION_SEED
    results_df["execution_block_size"] = EXECUTION_BLOCK_SIZE
    return results_df


def save_measurement_analysis_tables(replicate_df, output_folder, logger):
    """Save per-measurement and mean-measurement CSVs for plotting/analysis."""
    measurement_value_columns = get_measurement_value_columns()
    required_columns = ["measurement_replicate", "timepoint_min", *measurement_value_columns]
    missing_columns = [col for col in required_columns if col not in replicate_df.columns]
    if missing_columns:
        raise KeyError(f"Missing measurement columns for analysis tables: {missing_columns}")

    analysis_folder = os.path.join(output_folder, "measurement_replicate_analysis")
    os.makedirs(analysis_folder, exist_ok=True)

    individual_csvs = {}
    for measurement_replicate in sorted(replicate_df["measurement_replicate"].dropna().unique()):
        measurement_df = replicate_df[
            replicate_df["measurement_replicate"] == measurement_replicate
        ].copy()
        measurement_label = int(measurement_replicate)
        measurement_csv = os.path.join(
            analysis_folder,
            f"measurement_replicate_{measurement_label}_results.csv",
        )
        measurement_df.to_csv(measurement_csv, index=False)
        individual_csvs[measurement_label] = measurement_csv
        logger.info(f"Saved measurement replicate table: {measurement_csv}")

    group_columns = [
        col for col in replicate_df.columns
        if col not in measurement_value_columns and col not in (
            "measurement_replicate", "elapsed_actual_min", "measurement_order_position"
        )
    ]
    mean_columns = list(measurement_value_columns) + ["elapsed_actual_min"]
    mean_df = replicate_df.groupby(
        group_columns,
        dropna=False,
        as_index=False,
    )[mean_columns].mean()
    mean_df["measurement_replicate"] = "mean"

    mean_csv = os.path.join(analysis_folder, "measurement_mean_results.csv")
    mean_df.to_csv(mean_csv, index=False)
    logger.info(f"Saved mean measurement table: {mean_csv}")

    return {
        "individual_csvs": individual_csvs,
        "mean_csv": mean_csv,
    }


def apply_well_order(well_recipes_df, logger):
    """Assign physical wells, optionally randomized within fixed-size blocks."""
    if EXECUTION_BLOCK_SIZE <= 0:
        raise ValueError(
            f"EXECUTION_BLOCK_SIZE must be positive, got {EXECUTION_BLOCK_SIZE}"
        )
    if "wellplate_index" not in well_recipes_df.columns:
        raise KeyError("well_recipes_df must include wellplate_index before randomization")

    ordered_df = well_recipes_df.copy().reset_index(drop=True)
    if "source_recipe_index" not in ordered_df.columns:
        ordered_df["source_recipe_index"] = ordered_df.index
    ordered_df["original_wellplate_index"] = ordered_df["wellplate_index"]
    if "source_plate_index" not in ordered_df.columns:
        ordered_df["source_plate_index"] = 0
    ordered_df["source_plate_row"] = ordered_df.index
    ordered_df["execution_block"] = ordered_df["source_plate_row"] // min(EXECUTION_BLOCK_SIZE, len(ordered_df))

    if not RANDOMIZED_ORDER:
        logger.info("RANDOMIZED_ORDER=False: preserving original wellplate_index order")
        return ordered_df

    import random
    randomizer = random.Random(RANDOMIZATION_SEED)
    randomized_wells = ordered_df["wellplate_index"].copy()

    for (plate_id, block_id), block_df in ordered_df.groupby(
        ["source_plate_index", "execution_block"], sort=True
    ):
        row_indices = block_df.index.tolist()
        physical_wells = block_df["wellplate_index"].tolist()
        shuffled_wells = physical_wells.copy()
        randomizer.shuffle(shuffled_wells)
        randomized_wells.loc[row_indices] = shuffled_wells
        logger.info(
            f"Randomized well order within plate {plate_id}, block {block_id}: "
            f"rows {row_indices[0]}-{row_indices[-1]}, "
            f"wells {min(physical_wells)}-{max(physical_wells)}"
        )

    ordered_df["wellplate_index"] = randomized_wells.astype(ordered_df["wellplate_index"].dtype)
    return ordered_df


def split_into_execution_blocks(plate_df):
    """Split one physical plate into execution blocks."""
    if "execution_block" in plate_df.columns:
        return [
            block_df.reset_index(drop=True)
            for _, block_df in plate_df.groupby("execution_block", sort=True)
        ]

    return [
        plate_df.iloc[start:start + EXECUTION_BLOCK_SIZE].reset_index(drop=True)
        for start in range(0, len(plate_df), EXECUTION_BLOCK_SIZE)
    ]


def measure_and_process_study_turbidity(lash_e, well_recipes_df, shake_and_wait=True, measurement_order=None):
    """Measure turbidity using the configured wellplate type."""
    lash_e.logger.info(f"Measuring and processing turbidity (shake_and_wait={shake_and_wait})...")
    result_df = well_recipes_df.copy()
    if "turbidity_600" not in result_df.columns:
        result_df["turbidity_600"] = None

    total_wells = len(result_df)
    lash_e.logger.info(f"Total wells to measure turbidity: {total_wells}")
    for batch_start in range(0, total_wells, EXECUTION_BLOCK_SIZE):
        batch_end = min(batch_start + EXECUTION_BLOCK_SIZE, total_wells)
        batch_df = result_df.iloc[batch_start:batch_end]
        wells_in_batch = batch_df["wellplate_index"].tolist()
        if measurement_order is not None:
            batch_wells = set(wells_in_batch)
            wells_in_batch = [well for well in measurement_order if well in batch_wells]
        lash_e.logger.info(
            f"\nMeasuring turbidity batch {batch_start // EXECUTION_BLOCK_SIZE + 1}: "
            f"wells {batch_start}-{batch_end - 1}"
        )

        turbidity_data = measure_study_turbidity(
            lash_e,
            wells_in_batch,
            batch_df,
            shake_and_wait=shake_and_wait,
            return_wellplate=False,
            one_well_at_a_time=measurement_order is not None,
        )
        if turbidity_data is None:
            continue

        turbidity_filename = None
        try:
            experiment_name = getattr(lash_e, "current_experiment_name", "unknown_experiment")
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            turbidity_filename = (
                f"turbidity_wells{wells_in_batch[0]}-{wells_in_batch[-1]}_{timestamp}.csv"
            )
            sim_folder = "surfactant_grid/simulated" if lash_e.simulate else "surfactant_grid/experimental"
            turbidity_path = os.path.join(
                "output",
                sim_folder,
                experiment_name,
                "measurement_backups",
                turbidity_filename,
            )
            os.makedirs(os.path.dirname(turbidity_path), exist_ok=True)
            turbidity_data.to_csv(turbidity_path, index=True)
            mode_label = "[SIMULATED]" if lash_e.simulate else "[HARDWARE]"
            lash_e.logger.info(f"    {mode_label} Saved raw turbidity data: {turbidity_filename}")
        except Exception as e:
            lash_e.logger.error(f"    WARNING: Failed to save turbidity CSV: {e}")

        if "wellplate_index" not in turbidity_data.columns:
            raise KeyError(
                f"Turbidity data missing wellplate_index. Available columns: {list(turbidity_data.columns)}"
            )
        turbidity_col = "turbidity_600" if "turbidity_600" in turbidity_data.columns else turbidity_data.columns[-1]
        for _, row in turbidity_data.iterrows():
            well_idx = int(row["wellplate_index"])
            result_df.loc[result_df["wellplate_index"] == well_idx, "turbidity_600"] = row[turbidity_col]

    return result_df


def measure_study_turbidity(lash_e, well_indices, batch_recipes, shake_and_wait=True, return_wellplate=True,
                            one_well_at_a_time=False):
    """Measure turbidity for selected wells using the configured wellplate type."""
    shake_msg = "with shake protocol" if shake_and_wait else "without shake protocol"
    lash_e.logger.info(f"Measuring turbidity in wells {well_indices} ({shake_msg})")

    position_study_wellplate_at_cytation(lash_e)
    if shake_and_wait:
        shake_study_wellplate(lash_e)
    else:
        lash_e.logger.info("Skipping shake protocol")

    if one_well_at_a_time:
        reads = [measure_study_turbidity_protocol_only(lash_e, [well], batch_recipes)
                 for well in well_indices]
        if any(read is None for read in reads):
            raise RuntimeError("Missing turbidity data during ordered measurement")
        turbidity_data = pd.concat(reads, ignore_index=True)
    else:
        turbidity_data = measure_study_turbidity_protocol_only(lash_e, well_indices, batch_recipes)

    if return_wellplate:
        position_study_wellplate_at_track(lash_e)

    return turbidity_data


def measure_study_turbidity_protocol_only(lash_e, well_indices, batch_recipes):
    """Run turbidity protocol using the configured wellplate type."""
    lash_e.logger.info(f"Running turbidity protocol: {TURBIDITY_PROTOCOL_FILE}")

    if not lash_e.simulate:
        turbidity_data = lash_e.cytation.run_protocol(
            TURBIDITY_PROTOCOL_FILE,
            well_indices,
            plate_type=WELLPLATE_TYPE,
        )
        turbidity_data = flatten_cytation_data(turbidity_data, "turbidity")
        if turbidity_data is None:
            return None

        if "well_position" in turbidity_data.columns and "wellplate_index" not in turbidity_data.columns:
            turbidity_data["wellplate_index"] = [
                well_position_to_index(pos)
                for pos in turbidity_data["well_position"]
            ]
        return turbidity_data

    simulated_data = []
    for well_idx in well_indices:
        well_recipe = batch_recipes[batch_recipes["wellplate_index"] == well_idx]
        if len(well_recipe) > 0:
            row = well_recipe.iloc[0]
            if (
                row["well_type"] == "experiment"
                and pd.notna(row["surf_A_conc_mm"])
                and pd.notna(row["surf_B_conc_mm"])
            ):
                sim_result = simulate_surfactant_measurements(
                    row["surf_A_conc_mm"],
                    row["surf_B_conc_mm"],
                    add_noise=True,
                )
                simulated_data.append(sim_result["turbidity_600"])
            elif "water" in str(row["control_type"]).lower():
                simulated_data.append(0.02)
            else:
                simulated_data.append(0.35)
        else:
            simulated_data.append(0.05 + (well_idx * 0.01) % 0.50)

    return pd.DataFrame({
        "wellplate_index": well_indices,
        "turbidity_600": simulated_data,
    })


def generate_study_heatmaps(csv_file_path, output_dir, logger, surfactant_a_name, surfactant_b_name):
    """Generate dye-generic turbidity and fluorescence_metric heatmaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file_path)
    required_columns = [
        "well_type",
        "surf_A_conc_mm",
        "surf_B_conc_mm",
        "turbidity_600",
        "fluorescence_metric",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns for study heatmaps: {missing_columns}")

    experiment_data = df[df["well_type"] == "experiment"].copy()
    if experiment_data.empty:
        logger.warning("No experiment data found - skipping study heatmaps")
        return []

    def _format_conc_label(conc):
        if conc < 0.001:
            return f"{conc:.6f}"
        if conc < 1:
            return f"{conc:.4f}"
        return f"{conc:.1f}"

    def _make_grid(value_column):
        grid = experiment_data.pivot_table(
            index="surf_B_conc_mm",
            columns="surf_A_conc_mm",
            values=value_column,
            aggfunc="mean",
        )
        grid = grid.reindex(sorted(experiment_data["surf_B_conc_mm"].unique(), reverse=True))
        grid = grid.reindex(sorted(experiment_data["surf_A_conc_mm"].unique()), axis=1)
        return grid

    turbidity_grid = _make_grid("turbidity_600")
    fluorescence_grid = _make_grid("fluorescence_metric")
    dye_protocol = get_dye_protocol()
    metric_label = dye_protocol["primary_metric"]

    saved_paths = []

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(
        turbidity_grid,
        ax=axes[0],
        cmap="viridis",
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Turbidity (600 nm)"},
        xticklabels=[_format_conc_label(x) for x in turbidity_grid.columns],
        yticklabels=[_format_conc_label(y) for y in turbidity_grid.index],
    )
    axes[0].set_title("Turbidity")
    axes[0].set_xlabel(f"{surfactant_a_name} Concentration (mM)")
    axes[0].set_ylabel(f"{surfactant_b_name} Concentration (mM)")
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(
        fluorescence_grid,
        ax=axes[1],
        cmap="plasma",
        annot=True,
        fmt=".3f",
        cbar_kws={"label": f"Fluorescence Metric ({metric_label})"},
        xticklabels=[_format_conc_label(x) for x in fluorescence_grid.columns],
        yticklabels=[_format_conc_label(y) for y in fluorescence_grid.index],
    )
    axes[1].set_title(f"Fluorescence Metric: {metric_label}")
    axes[1].set_xlabel(f"{surfactant_a_name} Concentration (mM)")
    axes[1].set_ylabel(f"{surfactant_b_name} Concentration (mM)")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    combined_path = os.path.join(output_dir, "study_heatmaps_turbidity_fluorescence_metric.png")
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(combined_path)

    for value_column, title, cmap, filename in (
        ("turbidity_600", "Turbidity (600 nm)", "viridis", "study_turbidity_heatmap.png"),
        ("fluorescence_metric", f"Fluorescence Metric ({metric_label})", "plasma", "study_fluorescence_metric_heatmap.png"),
    ):
        grid = _make_grid(value_column)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            grid,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=".3f",
            cbar_kws={"label": title},
            xticklabels=[_format_conc_label(x) for x in grid.columns],
            yticklabels=[_format_conc_label(y) for y in grid.index],
        )
        ax.set_title(title)
        ax.set_xlabel(f"{surfactant_a_name} Concentration (mM)")
        ax.set_ylabel(f"{surfactant_b_name} Concentration (mM)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    logger.info(f"Saved study heatmaps to {output_dir}")
    return saved_paths


def generate_study_1d_plots(csv_file_path, output_dir, logger, surfactant_a_name, surfactant_b_name):
    """Save 1D control data and CMC fits, then generate plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from analysis.control_cmc_analysis import fit_cmc_curve

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file_path)
    required_columns = ["well_type", "control_type", "turbidity_600", "fluorescence_metric"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns for 1D plots: {missing_columns}")

    control_data = df[df["well_type"] == "control"].copy()
    if control_data.empty:
        logger.info("No control data found - skipping 1D plots")
        return []

    dye_protocol = get_dye_protocol()
    metric_label = dye_protocol["primary_metric"]
    saved_paths = []

    for surfactant_name, concentration_column in (
        (surfactant_a_name, "surf_A_conc_mm"),
        (surfactant_b_name, "surf_B_conc_mm"),
    ):
        if concentration_column not in control_data.columns:
            logger.warning(
                f"Missing {concentration_column}; skipping 1D plot for {surfactant_name}"
            )
            continue

        cmc_prefix = f"cmc_{surfactant_name}_"
        cmc_data = control_data[
            control_data["control_type"].astype(str).str.startswith(cmc_prefix)
        ].copy()
        cmc_data = cmc_data.dropna(subset=[concentration_column])
        if cmc_data.empty:
            logger.info(f"No 1D CMC control data found for {surfactant_name}")
            continue

        summary = cmc_data.groupby(concentration_column, as_index=False)[
            ["turbidity_600", "fluorescence_metric"]
        ].mean()
        summary = summary.sort_values(concentration_column)

        safe_name = str(surfactant_name).replace(" ", "_")
        csv_path = os.path.join(output_dir, f"study_1d_{safe_name}_controls.csv")
        summary.to_csv(csv_path, index=False)
        saved_paths.append(csv_path)

        # Use the same log-concentration Boltzmann fit as the legacy CMC plot.
        fit_data = summary[[concentration_column, "fluorescence_metric"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        fit_data = fit_data[fit_data[concentration_column] > 0]
        fit_result = {
            "surfactant": surfactant_name,
            "metric": metric_label,
            "fit_status": "insufficient_points",
            "n_points": len(fit_data),
            "concentration_unit": "mM",
            "fit_x_scale": "log10_concentration_mm",
            "A1": np.nan, "A2": np.nan, "x0": np.nan, "dx": np.nan,
            "cmc_mm": np.nan, "r_squared": np.nan,
        }
        if len(fit_data) >= 4 and fit_data[concentration_column].nunique() >= 4:
            cmc, r_squared, parameters, _ = fit_cmc_curve(
                fit_data[concentration_column].to_numpy(),
                fit_data["fluorescence_metric"].to_numpy(),
            )
            if parameters is not None:
                fit_result.update(
                    fit_status="success", A1=parameters[0], A2=parameters[1],
                    x0=parameters[2], dx=parameters[3], cmc_mm=cmc,
                    r_squared=r_squared,
                )
            else:
                fit_result["fit_status"] = "fit_failed"
        fit_csv = os.path.join(output_dir, f"study_1d_{safe_name}_fit.csv")
        pd.DataFrame([fit_result]).to_csv(fit_csv, index=False)
        saved_paths.append(fit_csv)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(
            summary[concentration_column],
            summary["fluorescence_metric"],
            marker="o",
            label="Measured mean",
        )
        if fit_result["fit_status"] == "success":
            from analysis.control_cmc_analysis import boltzmann
            x_fit = np.linspace(
                np.log10(fit_data[concentration_column].min()),
                np.log10(fit_data[concentration_column].max()), 200,
            )
            axes[0].plot(10 ** x_fit, boltzmann(
                x_fit, fit_result["A1"], fit_result["A2"],
                fit_result["x0"], fit_result["dx"],
            ), linestyle="--", label=f"Boltzmann fit (R²={fit_result['r_squared']:.2f})")
            axes[0].axvline(fit_result["cmc_mm"], color="tab:red", linestyle=":",
                            label=f"CMC {fit_result['cmc_mm']:.2g} mM")
            axes[0].legend()
        axes[0].set_title(f"{surfactant_name} 1D Fluorescence")
        axes[0].set_xlabel(f"{surfactant_name} Concentration (mM)")
        axes[0].set_ylabel(f"Fluorescence Metric ({metric_label})")
        if (summary[concentration_column] > 0).all():
            axes[0].set_xscale("log")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            summary[concentration_column],
            summary["turbidity_600"],
            marker="o",
            color="tab:green",
        )
        axes[1].set_title(f"{surfactant_name} 1D Turbidity")
        axes[1].set_xlabel(f"{surfactant_name} Concentration (mM)")
        axes[1].set_ylabel("Turbidity (600 nm)")
        if (summary[concentration_column] > 0).all():
            axes[1].set_xscale("log")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = os.path.join(output_dir, f"study_1d_{safe_name}_controls.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(plot_path)

        logger.info(f"Saved 1D CMC/control plot for {surfactant_name}: {plot_path}")

    return saved_paths


# ============================================================================
# INPUT LOADING
# ============================================================================

def load_replay_inputs(recipes_path, stocks_path):
    """Load per-well recipes and substock dilution recipes used to build them."""
    well_recipes_df = pd.read_csv(recipes_path)

    stocks_df = pd.read_csv(stocks_path)
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
    dilution_recipes.sort(key=lambda r: float(r["Target_Conc_mM"]), reverse=True)
    return well_recipes_df, dilution_recipes


def split_into_plates(well_recipes_df, wellplate_size=None):
    """Split the recipes DataFrame into one DataFrame per physical wellplate."""
    if wellplate_size is not None:
        plates = []
        for plate_index, start in enumerate(range(0, len(well_recipes_df), wellplate_size)):
            plate_df = well_recipes_df.iloc[start:start + wellplate_size].reset_index(drop=True)
            plate_df["source_plate_index"] = plate_index
            plate_df["source_plate_row"] = plate_df.index
            plate_df["wellplate_index"] = plate_df.index
            plates.append(plate_df)
        return plates

    if "source_plate_index" in well_recipes_df.columns:
        return [
            plate_df.reset_index(drop=True)
            for _, plate_df in well_recipes_df.groupby("source_plate_index", sort=True)
        ]

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
# DYE DISPENSE AND MEASUREMENT
# ============================================================================

def dispense_dye(lash_e, well_recipes_df):
    """Dispense the configured dye solution to all wells in the plate DataFrame."""
    dye_protocol = get_dye_protocol()
    lash_e.logger.info(
        f"Dispensing dye {dye_protocol['dye']} from vial {dye_protocol['dye_vial']} "
        f"using solvent/liquid {dye_protocol['dye_solvent']} "
        f"at {dye_protocol['dye_volume_ul']} uL per well"
    )

    batch_df = well_recipes_df.copy()
    batch_df["dye_volume_ul"] = dye_protocol["dye_volume_ul"]

    lash_e.nr_robot.move_vial_to_location(
        dye_protocol["dye_vial"], "clamp", 0
    )
    dispense_study_component_to_wellplate(
        lash_e,
        batch_df,
        dye_protocol["dye_vial"],
        dye_protocol["dye_solvent"],
        "dye_volume_ul",
    )
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.return_vial_home(dye_protocol["dye_vial"])
    lash_e.logger.info(f"  Dye added to {len(well_recipes_df)} wells")

    return well_recipes_df


def dispense_study_component_to_wellplate(
    lash_e,
    batch_df,
    vial_name,
    liquid_type,
    volume_column,
    should_condition_tip=True,
):
    """Dispense one component into the configured wellplate type."""
    dispense_component_to_wellplate(
        lash_e,
        batch_df,
        vial_name,
        liquid_type,
        volume_column,
        should_condition_tip=should_condition_tip,
        well_plate_type=WELLPLATE_TYPE,
    )


def measure_and_process_dye_fluorescence(lash_e, well_recipes_df, shake_and_wait=True,
                                         measurement_order=None):
    """Measure fluorescence using the active dye protocol."""
    dye_protocol = get_dye_protocol()
    dye = dye_protocol["dye"]
    if not dye_protocol["implemented"]:
        raise NotImplementedError(
            f"Fluorescence measurement for dye '{dye}' is not implemented yet. "
            f"Add the Cytation protocol file and output-column mapping before running "
            f"this dye on hardware."
        )

    lash_e.logger.info(
        f"Measuring fluorescence for dye {dye} using protocol "
        f"{dye_protocol['fluorescence_protocol_file']} "
        f"(primary metric: {dye_protocol['primary_metric']})"
    )

    if "wellplate_index" not in well_recipes_df.columns:
        raise KeyError("well_recipes_df must include wellplate_index for fluorescence measurement")

    result_df = well_recipes_df.copy()
    for column in dye_protocol["measurement_value_columns"]:
        if column not in result_df.columns:
            result_df[column] = None

    total_wells = len(result_df)
    lash_e.logger.info(f"Total wells to measure dye fluorescence: {total_wells}")

    for batch_start in range(0, total_wells, EXECUTION_BLOCK_SIZE):
        batch_end = min(batch_start + EXECUTION_BLOCK_SIZE, total_wells)
        batch_df = result_df.iloc[batch_start:batch_end]
        wells_in_batch = batch_df["wellplate_index"].tolist()
        if measurement_order is not None:
            batch_wells = set(wells_in_batch)
            wells_in_batch = [well for well in measurement_order if well in batch_wells]
        lash_e.logger.info(
            f"\nMeasuring dye fluorescence batch {batch_start // EXECUTION_BLOCK_SIZE + 1}: "
            f"wells {batch_start}-{batch_end - 1}"
        )

        fluorescence_data = measure_dye_fluorescence(
            lash_e,
            wells_in_batch,
            batch_df,
            dye_protocol,
            shake_and_wait=shake_and_wait,
            return_wellplate=True,
            one_well_at_a_time=measurement_order is not None,
        )

        if fluorescence_data is None:
            continue

        fluorescence_filename = None
        try:
            experiment_name = getattr(lash_e, "current_experiment_name", "unknown_experiment")
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fluorescence_filename = (
                f"fluorescence_{dye}_wells{wells_in_batch[0]}-{wells_in_batch[-1]}_"
                f"{timestamp}.csv"
            )
            sim_folder = "surfactant_grid/simulated" if lash_e.simulate else "surfactant_grid/experimental"
            fluorescence_path = os.path.join(
                "output",
                sim_folder,
                experiment_name,
                "measurement_backups",
                fluorescence_filename,
            )
            os.makedirs(os.path.dirname(fluorescence_path), exist_ok=True)
            fluorescence_data.to_csv(fluorescence_path, index=True)
            mode_label = "[SIMULATED]" if lash_e.simulate else "[HARDWARE]"
            lash_e.logger.info(f"    {mode_label} Saved raw dye fluorescence data: {fluorescence_filename}")
        except Exception as e:
            lash_e.logger.error(f"    WARNING: Failed to save dye fluorescence CSV: {e}")

        if "wellplate_index" not in fluorescence_data.columns:
            available_cols = list(fluorescence_data.columns)
            raise KeyError(
                f"Dye fluorescence data missing wellplate_index. Available columns: {available_cols}"
            )

        for _, row in fluorescence_data.iterrows():
            well_idx = int(row["wellplate_index"])
            mask = result_df["wellplate_index"] == well_idx
            for column in dye_protocol["measurement_value_columns"]:
                if column in row:
                    result_df.loc[mask, column] = row[column]

    return result_df


def measure_dye_fluorescence(
    lash_e,
    well_indices,
    batch_recipes,
    dye_protocol,
    shake_and_wait=True,
    return_wellplate=True,
    one_well_at_a_time=False,
):
    """Measure fluorescence for the active dye using its configured protocol."""
    dye = dye_protocol["dye"]
    shake_msg = "with shake protocol" if shake_and_wait else "without shake protocol"
    lash_e.logger.info(
        f"Measuring {dye} fluorescence in wells {well_indices} ({shake_msg})"
    )

    position_study_wellplate_at_cytation(lash_e)
    if shake_and_wait:
        shake_study_wellplate(lash_e)
    else:
        lash_e.logger.info("Skipping shake protocol")

    if one_well_at_a_time:
        reads = [measure_dye_fluorescence_protocol_only(lash_e, [well], batch_recipes, dye_protocol)
                 for well in well_indices]
        if any(read is None for read in reads):
            raise RuntimeError("Missing fluorescence data during ordered measurement")
        fluorescence_data = pd.concat(reads, ignore_index=True)
    else:
        fluorescence_data = measure_dye_fluorescence_protocol_only(
            lash_e, well_indices, batch_recipes, dye_protocol,
        )

    if return_wellplate:
        position_study_wellplate_at_track(lash_e)
    else:
        lash_e.logger.info("Leaving wellplate at cytation for repeated measurements")

    return fluorescence_data


def position_study_wellplate_at_cytation(lash_e):
    """Move the active wellplate to Cytation using the configured wellplate type."""
    lash_e.nr_track.get_track_status()
    if lash_e.nr_track.ACTIVE_WELLPLATE_POSITION != "cytation":
        lash_e.move_wellplate_to_cytation(plate_type=WELLPLATE_TYPE)
        lash_e.nr_track.set_wellplate_position("cytation")


def position_study_wellplate_at_track(lash_e):
    """Return the active wellplate to the pipetting area using the configured type."""
    lash_e.nr_track.get_track_status()
    if lash_e.nr_track.ACTIVE_WELLPLATE_POSITION != "pipetting_area":
        lash_e.move_wellplate_back_from_cytation(plate_type=WELLPLATE_TYPE)
        lash_e.nr_track.origin()
        lash_e.nr_track.set_wellplate_position("pipetting_area")


def shake_study_wellplate(lash_e):
    """Run the configured shake protocol using the configured wellplate type."""
    if not lash_e.simulate:
        lash_e.logger.info(f"Running shake protocol: {SHAKE_WAIT_PROTOCOL}")
        lash_e.cytation.run_protocol(SHAKE_WAIT_PROTOCOL, [0], plate_type=WELLPLATE_TYPE)
    else:
        lash_e.logger.info("Shake protocol simulated")


def measure_dye_fluorescence_protocol_only(lash_e, well_indices, batch_recipes, dye_protocol):
    """Run the configured dye fluorescence protocol and normalize output columns."""
    protocol_file = dye_protocol["fluorescence_protocol_file"]
    lash_e.logger.info(f"Running dye fluorescence protocol: {protocol_file}")

    if not lash_e.simulate:
        fluorescence_data = lash_e.cytation.run_protocol(
            protocol_file,
            well_indices,
            plate_type=WELLPLATE_TYPE,
        )
        fluorescence_data = flatten_cytation_data(fluorescence_data, "fluorescence")
        if fluorescence_data is None:
            return None

        if "well_position" in fluorescence_data.columns and "wellplate_index" not in fluorescence_data.columns:
            fluorescence_data["wellplate_index"] = [
                well_position_to_index(pos)
                for pos in fluorescence_data["well_position"]
            ]

        fluorescence_data = fluorescence_data.rename(columns=dye_protocol["raw_column_mapping"])
        missing_columns = [
            column for column in dye_protocol["measurement_value_columns"]
            if column not in fluorescence_data.columns and column != dye_protocol["primary_metric"]
        ]
        if missing_columns:
            raise KeyError(
                f"Dye protocol {dye_protocol['dye']} did not produce expected columns {missing_columns}. "
                f"Available columns: {list(fluorescence_data.columns)}"
            )
    else:
        fluorescence_data = simulate_dye_fluorescence(well_indices, batch_recipes, dye_protocol)

    fluorescence_data = calculate_dye_metric(fluorescence_data, dye_protocol)
    lash_e.logger.info(
        f"Successfully measured {dye_protocol['dye']} fluorescence for {len(well_indices)} wells"
    )
    return fluorescence_data


def calculate_dye_metric(fluorescence_data, dye_protocol):
    """Calculate the configured primary fluorescence metric when needed."""
    if dye_protocol["metric_kind"] == "ratio":
        if dye_protocol["dye"] == "pyrene":
            numerator = "fluorescence_334_373"
            denominator = "fluorescence_334_384"
            if numerator in fluorescence_data.columns and denominator in fluorescence_data.columns:
                fluorescence_data["ratio"] = fluorescence_data[numerator] / fluorescence_data[denominator]
                return fluorescence_data
        raise KeyError(f"Cannot calculate ratio metric for dye '{dye_protocol['dye']}'")

    return fluorescence_data


def simulate_dye_fluorescence(well_indices, batch_recipes, dye_protocol):
    """Generate synthetic measurement data for the active dye protocol.

    Intensity-dye values use a seeded concentration proxy for workflow testing,
    not a calibrated model of the dye's response.
    """
    dye = dye_protocol["dye"]
    if dye != "pyrene":
        if dye_protocol["metric_kind"] != "intensity":
            raise NotImplementedError(
                f"Simulation for dye '{dye}' is not implemented for metric "
                f"'{dye_protocol['metric_kind']}'."
            )

        import numpy as np

        required_columns = (
            "wellplate_index", "control_type", "surf_A_conc_mm", "surf_B_conc_mm",
        )
        missing_columns = [column for column in required_columns if column not in batch_recipes]
        if missing_columns:
            raise KeyError(f"Recipes missing columns required for simulated fluorescence: {missing_columns}")
        if batch_recipes["wellplate_index"].duplicated().any():
            raise ValueError("Recipes must have unique wellplate_index values for simulated fluorescence")

        recipes = batch_recipes.set_index("wellplate_index", verify_integrity=True)
        missing_wells = pd.Index(well_indices).difference(recipes.index)
        if not missing_wells.empty:
            raise KeyError(f"Recipes missing requested wells: {missing_wells.tolist()}")

        concentrations = []
        for well_idx, row in recipes.iterrows():
            component_concentrations = [
                float(row[column])
                for column in ("surf_A_conc_mm", "surf_B_conc_mm")
                if pd.notna(row[column])
            ]
            if not component_concentrations:
                if "water" not in str(row["control_type"]).lower():
                    raise ValueError(
                        f"Well {well_idx} lacks surfactant concentrations for simulated fluorescence"
                    )
                concentration = 0.0
            else:
                concentration = sum(component_concentrations)
            if not math.isfinite(concentration) or concentration < 0:
                raise ValueError(
                    f"Well {well_idx} has invalid surfactant concentration {concentration}"
                )
            concentrations.append(concentration)

        concentration_scale = max(concentrations)
        if concentration_scale <= 0:
            raise ValueError("Simulated fluorescence requires at least one positive surfactant concentration")
        relative_concentration = pd.Series(
            [concentration / concentration_scale for concentration in concentrations],
            index=recipes.index,
        )

        simulated_data = {"wellplate_index": well_indices}
        for column in dye_protocol["measurement_value_columns"]:
            rng = np.random.default_rng(0)
            noise = pd.Series(
                rng.normal(0, 20, len(recipes)),
                index=recipes.index,
            )
            simulated_data[column] = [
                500 + relative_concentration.loc[well_idx] * 5000 + noise.loc[well_idx]
                for well_idx in well_indices
            ]
        return pd.DataFrame(simulated_data)

    simulated_373 = []
    simulated_384 = []
    for well_idx in well_indices:
        well_recipe = batch_recipes[batch_recipes["wellplate_index"] == well_idx]
        if len(well_recipe) > 0:
            row = well_recipe.iloc[0]
            if (
                row["well_type"] == "experiment"
                and pd.notna(row["surf_A_conc_mm"])
                and pd.notna(row["surf_B_conc_mm"])
            ):
                sim_result = simulate_surfactant_measurements(
                    row["surf_A_conc_mm"],
                    row["surf_B_conc_mm"],
                    add_noise=True,
                )
                simulated_373.append(sim_result["fluorescence_334_373"])
                simulated_384.append(sim_result["fluorescence_334_384"])
            elif "water" in str(row["control_type"]).lower():
                simulated_373.append(800)
                simulated_384.append(1200)
            elif str(row["control_type"]).startswith("cmc_"):
                # Simulated 1D controls need a concentration response so that
                # fitting and downstream local calibration can be exercised.
                import numpy as np
                concentration = (row["surf_A_conc_mm"] if pd.notna(row["surf_A_conc_mm"])
                                 else row["surf_B_conc_mm"])
                log_concentration = np.log10(float(concentration))
                simulated_ratio = 0.75 + 0.10 / (1 + np.exp((log_concentration - np.log10(8.8)) / 0.18))
                simulated_373.append(1500 * simulated_ratio)
                simulated_384.append(1500)
            else:
                simulated_373.append(1500)
                simulated_384.append(2200)
        else:
            simulated_373.append(800 + (well_idx * 50) % 2000)
            simulated_384.append(1200 + (well_idx * 75) % 3000)

    return pd.DataFrame({
        "wellplate_index": well_indices,
        "fluorescence_334_373": simulated_373,
        "fluorescence_334_384": simulated_384,
    })


# ============================================================================
# REFILL ENGINE
# ============================================================================

def _classify_vial(vial_name, dilution_recipes):
    """Return ('water'|'stock'|'substock'|'buffer', recipe_or_none)."""
    if vial_name in ("water", "water_2"):
        return "water", None
    if vial_name.endswith("_stock"):
        return "stock", None
    for recipe in dilution_recipes:
        if recipe["Vial_Name"] == vial_name:
            return "substock", recipe
    if ADD_BUFFER and vial_name == SELECTED_BUFFER:
        return "buffer", None
    raise ValueError(f"Cannot classify vial '{vial_name}' for refill routing")


def ensure_vial_above_threshold(lash_e, vial_name, dilution_recipes):
    """Check current volume; refill if below the kind-specific threshold."""
    kind, recipe = _classify_vial(vial_name, dilution_recipes)
    threshold = WATER_REFILL_THRESHOLD_ML if kind == "water" else REFILL_THRESHOLD_ML

    current_ml = lash_e.nr_robot.get_vial_info(vial_name, "vial_volume")
    if current_ml is None:
        raise ValueError(f"No volume tracked for vial '{vial_name}'")
    if current_ml >= threshold:
        return False

    lash_e.logger.info(
        f"  REFILL: {vial_name} at {current_ml:.2f} mL < {threshold} mL "
        f"(kind={kind})"
    )

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
    reposition_after_refill=None,
):
    """Dispense one vial's wells in REFILL_CHECK_CHUNK_SIZE-row sub-chunks."""
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
            if reposition_after_refill is not None:
                location, index = reposition_after_refill
                lash_e.logger.info(
                    f"  Repositioning {vial_name} to {location}[{index}] after refill"
                )
                lash_e.nr_robot.move_vial_to_location(vial_name, location, index)
        dispense_component_to_wellplate(
            lash_e,
            chunk_df,
            vial_name,
            liquid_type,
            volume_column,
            should_condition_tip=should_condition,
            well_plate_type=WELLPLATE_TYPE,
        )
        should_condition = False


# ============================================================================
# PER-PLATE DISPENSING
# ============================================================================

def validate_dispense_order():
    """Validate configured dispense order."""
    valid_components = {"surfactant_A", "water", "buffer", "surfactant_B", "dye"}
    required_components = {"surfactant_A", "water", "surfactant_B", "dye"}
    if ADD_BUFFER:
        required_components.add("buffer")

    unknown_components = [component for component in DISPENSE_ORDER if component not in valid_components]
    if unknown_components:
        raise ValueError(f"Unknown DISPENSE_ORDER component(s): {unknown_components}")

    duplicate_components = [
        component for component in valid_components
        if DISPENSE_ORDER.count(component) > 1
    ]
    if duplicate_components:
        raise ValueError(f"DISPENSE_ORDER contains duplicate component(s): {duplicate_components}")

    missing_components = [
        component for component in required_components
        if component not in DISPENSE_ORDER
    ]
    if missing_components:
        raise ValueError(f"DISPENSE_ORDER missing required component(s): {missing_components}")

    if not ADD_BUFFER and "buffer" in DISPENSE_ORDER:
        raise ValueError("DISPENSE_ORDER includes 'buffer' but ADD_BUFFER=False")


def dispense_surfactant_component(lash_e, plate_df, dilution_recipes, component):
    """Dispense surfactant A or B using existing refill/chunk logic."""
    if component == "surfactant_A":
        surfactant_label = "A"
        substock_column = "substock_A_name"
        volume_column = "surf_A_volume_ul"
    elif component == "surfactant_B":
        surfactant_label = "B"
        substock_column = "substock_B_name"
        volume_column = "surf_B_volume_ul"
    else:
        raise ValueError(f"Unsupported surfactant component: {component}")

    surfactant_vials = (
        plate_df[plate_df[volume_column] > 0][substock_column].dropna().unique()
    )
    if len(surfactant_vials) == 0:
        return

    sorted_vials = position_surfactant_vials_by_concentration(
        lash_e, surfactant_vials, plate_df, surfactant_label
    )
    n_vials = len(sorted_vials)
    for i, vial in enumerate(sorted_vials):
        raw_pos = _SURF_SAFE_POSITIONS[n_vials - 1 - i]
        reposition = ("clamp", 0) if raw_pos == "clamp" else ("main_8mL_rack", raw_pos)
        _dispense_vial_in_chunks(
            lash_e,
            plate_df,
            vial,
            "SDS",
            volume_column,
            dilution_recipes,
            should_condition_first=(i == 0),
            reposition_after_refill=reposition,
        )
    lash_e.nr_robot.remove_pipet()
    return_surfactant_vials_home(lash_e, sorted_vials, surfactant_label)


def dispense_water_component(lash_e, plate_df, dilution_recipes):
    """Dispense water using existing split-vial refill/chunk logic."""
    water_wells = plate_df[plate_df["water_volume_ul"] > 0]
    if len(water_wells) == 0:
        return

    water_wells = water_wells.sort_values("water_volume_ul", ascending=True)
    mid = len(water_wells) // 2
    water_batch_1_idx = water_wells.iloc[:mid]["wellplate_index"].tolist()
    water_batch_2_idx = water_wells.iloc[mid:]["wellplate_index"].tolist()
    water_1_df = plate_df[plate_df["wellplate_index"].isin(water_batch_1_idx)]
    water_2_df = plate_df[plate_df["wellplate_index"].isin(water_batch_2_idx)]

    lash_e.nr_robot.move_vial_to_location("water", "main_8mL_rack", 44)
    lash_e.nr_robot.move_vial_to_location("water_2", "main_8mL_rack", 45)

    if len(water_1_df) > 0:
        _dispense_vial_in_chunks(
            lash_e,
            water_1_df,
            "water",
            "water",
            "water_volume_ul",
            dilution_recipes,
            should_condition_first=True,
            reposition_after_refill=("main_8mL_rack", 44),
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
            reposition_after_refill=("main_8mL_rack", 45),
        )

    lash_e.nr_robot.remove_pipet()
    return_water_vial_home(lash_e, "water")
    return_water_vial_home(lash_e, "water_2")


def dispense_buffer_component(lash_e, plate_df, dilution_recipes):
    """Dispense configured buffer using existing refill/chunk logic."""
    if not ADD_BUFFER:
        return

    lash_e.nr_robot.move_vial_to_location(SELECTED_BUFFER, "main_8mL_rack", 47)
    _dispense_vial_in_chunks(
        lash_e,
        plate_df,
        SELECTED_BUFFER,
        "water",
        "buffer_volume_ul",
        dilution_recipes,
        should_condition_first=False,
        reposition_after_refill=("main_8mL_rack", 47),
    )
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.return_vial_home(SELECTED_BUFFER)


def execute_dispense_order(lash_e, plate_df, dilution_recipes):
    """Dispense one execution block according to DISPENSE_ORDER."""
    validate_dispense_order()
    lash_e.logger.info(
        f"Dispensing block ({len(plate_df)} wells) with order {DISPENSE_ORDER} "
        f"and refill chunk={REFILL_CHECK_CHUNK_SIZE}"
    )

    for component in DISPENSE_ORDER:
        lash_e.logger.info(f"Dispense step: {component}")
        if component == "surfactant_A":
            dispense_surfactant_component(lash_e, plate_df, dilution_recipes, component)
        elif component == "water":
            dispense_water_component(lash_e, plate_df, dilution_recipes)
        elif component == "buffer":
            dispense_buffer_component(lash_e, plate_df, dilution_recipes)
        elif component == "surfactant_B":
            dispense_surfactant_component(lash_e, plate_df, dilution_recipes, component)
        elif component == "dye":
            dispense_dye(lash_e, plate_df)


def execute_dispensing_with_refills(lash_e, plate_df, dilution_recipes):
    """Dispense one execution block with configurable order and refill checks."""
    execute_dispense_order(lash_e, plate_df, dilution_recipes)


# ============================================================================
# TOP-LEVEL WORKFLOW
# ============================================================================

def execute_preparation_replicate(
    lash_e,
    plate_dfs,
    dilution_recipes,
    surfactant_a_name,
    surfactant_b_name,
    preparation_replicate,
    simulate=True,
):
    """Run one preparation replicate and save it to its own output folder."""
    lash_e.logger.info("=" * 80)
    lash_e.logger.info(
        f"PREPARATION REPLICATE {preparation_replicate}/{PREPARATION_REPLICATES}"
    )
    lash_e.logger.info("=" * 80)

    experiment_output_folder, experiment_name = setup_experiment_environment(
        lash_e,
        surfactant_a_name,
        f"{surfactant_b_name}_{EXPERIMENT_TAG}_prep{preparation_replicate}",
        simulate,
    )

    lash_e.logger.info("Replicate setup: filling water + stock vials to capacity")
    fill_water_vial(lash_e, "water")
    fill_water_vial(lash_e, "water_2")
    refill_surfactant_vial(lash_e, f"{surfactant_a_name}_stock", liquid="SDS")
    refill_surfactant_vial(lash_e, f"{surfactant_b_name}_stock", liquid="SDS")

    lash_e.logger.info("Replicate setup: checking/refilling substocks from source recipes")
    create_substocks_from_recipes(lash_e, dilution_recipes)

    completed_plates = []
    measurement_randomizer = random.Random(MEASUREMENT_RANDOMIZATION_SEED)
    lash_e.nr_robot.home_robot_components()

    for plate_idx, source_plate_df in enumerate(plate_dfs):
        plate_df = source_plate_df.copy()
        lash_e.logger.info(
            f"\n--- PREP {preparation_replicate}, PLATE {plate_idx + 1}/{len(plate_dfs)} "
            f"({len(plate_df)} wells) ---"
        )
        lash_e.grab_new_wellplate()

        execution_blocks = split_into_execution_blocks(plate_df)
        for block_idx, source_block_df in enumerate(execution_blocks):
            block_df = source_block_df.copy()
            lash_e.logger.info(
                f"\n--- PREP {preparation_replicate}, PLATE {plate_idx + 1}, "
                f"BLOCK {block_idx + 1}/{len(execution_blocks)} "
                f"({len(block_df)} wells) ---"
            )
            execute_dispensing_with_refills(lash_e, block_df, dilution_recipes)

            measurement_start = time.monotonic()
            for timepoint_min in validate_measurement_schedule():
                wait_seconds = measurement_start + timepoint_min * 60 - time.monotonic()
                if wait_seconds > 0 and not simulate:
                    time.sleep(wait_seconds)
                for measurement_replicate in range(1, MEASUREMENT_REPLICATES + 1):
                    well_order = get_measurement_well_order(block_df, measurement_randomizer)
                    ordered_reads = well_order if RANDOMIZED_MEASUREMENT_ORDER else None
                    lash_e.logger.info(
                        f"Measuring prep {preparation_replicate}, plate {plate_idx + 1}, "
                        f"block {block_idx + 1}, t={timepoint_min} min, "
                        f"measurement replicate {measurement_replicate}/{MEASUREMENT_REPLICATES}"
                    )
                    measurement_df = block_df.copy()
                    measurement_df = measure_and_process_study_turbidity(
                        lash_e, measurement_df, shake_and_wait=True,
                        measurement_order=ordered_reads,
                    )
                    measurement_df = measure_and_process_dye_fluorescence(
                        lash_e, measurement_df, shake_and_wait=False,
                        measurement_order=ordered_reads,
                    )
                    order_positions = {well: position for position, well in enumerate(well_order, 1)}
                    measurement_df["measurement_order_position"] = measurement_df["wellplate_index"].map(order_positions)
                    measurement_df["timepoint_min"] = timepoint_min
                    measurement_df["elapsed_actual_min"] = (time.monotonic() - measurement_start) / 60
                    measurement_df["measurement_replicate"] = measurement_replicate
                    completed_plates.append(measurement_df)

        lash_e.discard_used_wellplate()

    replicate_df = pd.concat(completed_plates, ignore_index=True)
    replicate_df = add_fluorescence_metric(replicate_df)
    replicate_df = add_study_metadata(replicate_df, preparation_replicate)
    final_csv = os.path.join(experiment_output_folder, "complete_experiment_results.csv")
    replicate_df.to_csv(final_csv, index=False)
    lash_e.logger.info(f"Saved preparation replicate results: {final_csv}")

    analysis_tables = save_measurement_analysis_tables(
        replicate_df,
        experiment_output_folder,
        lash_e.logger,
    )

    analysis_inputs = [
        (f"measurement_replicate_{replicate}", path)
        for replicate, path in analysis_tables["individual_csvs"].items()
    ]
    analysis_inputs.append(("measurement_mean", analysis_tables["mean_csv"]))
    if len(MEASUREMENT_SCHEDULE_MIN) > 1:
        timepoint_inputs = []
        for label, measurement_csv in analysis_inputs:
            table = pd.read_csv(measurement_csv)
            for timepoint_min, timepoint_table in table.groupby("timepoint_min", sort=True):
                timepoint_label = f"{label}_t{timepoint_min:g}min"
                timepoint_csv = os.path.join(
                    experiment_output_folder, "measurement_replicate_analysis",
                    f"{timepoint_label}_results.csv",
                )
                timepoint_table.to_csv(timepoint_csv, index=False)
                timepoint_inputs.append((timepoint_label, timepoint_csv))
        analysis_inputs = timepoint_inputs
    for label, measurement_csv in analysis_inputs:
        one_d_folder = os.path.join(experiment_output_folder, "one_dimensional", label)
        try:
            generate_study_1d_plots(
                measurement_csv, one_d_folder, lash_e.logger,
                surfactant_a_name, surfactant_b_name,
            )
        except Exception as e:
            lash_e.logger.warning(f"1D analysis failed for {label}: {e}")
        try:
            heatmap_folder = os.path.join(experiment_output_folder, "heatmap", label)
            generate_study_heatmaps(
                measurement_csv, heatmap_folder, lash_e.logger,
                surfactant_a_name, surfactant_b_name,
            )
        except Exception as e:
            lash_e.logger.warning(f"Heatmap generation failed for {label}: {e}")
        if get_dye_protocol()["primary_metric"] == "ratio":
            try:
                from analysis.surfactant_grid_scoring import score_surfactant_grid_2d
                score_surfactant_grid_2d(
                    measurement_csv, one_d_folder,
                    os.path.join(experiment_output_folder, "scoring", label),
                    surfactant_a_name, surfactant_b_name,
                )
            except Exception as e:
                lash_e.logger.warning(f"2D scoring failed for {label}: {e}")

    dye_protocol = get_dye_protocol()
    if dye_protocol["primary_metric"] == "ratio":
        mean_inputs = [
            (label, path) for label, path in analysis_inputs
            if label == "measurement_mean" or label.startswith("measurement_mean_t")
        ]
        for label, mean_csv in mean_inputs:
            destination = (experiment_output_folder if label == "measurement_mean" else
                           os.path.join(experiment_output_folder, "post_analysis", label))
            try:
                os.makedirs(destination, exist_ok=True)
                run_post_experiment_analysis(
                    mean_csv, destination,
                    surfactant_a_name, surfactant_b_name, lash_e.logger,
                )
            except Exception as e:
                lash_e.logger.warning(f"Post-experiment analysis failed for {label}: {e}")
    else:
        lash_e.logger.info(
            "Skipping legacy post-experiment analysis because it assumes pyrene ratio; "
            f"active primary metric is {dye_protocol['primary_metric']}"
        )

    return {
        "preparation_replicate": preparation_replicate,
        "well_recipes_df": replicate_df,
        "output_folder": experiment_output_folder,
        "experiment_name": experiment_name,
        "results_csv": final_csv,
        "analysis_tables": analysis_tables,
    }


def execute_study_workflow(
    recipes_path,
    stocks_path,
    lash_e,
    simulate=True,
    max_wells=0,
    validate_liquids=False,
    validation_only=False,
):
    """Replay a grid CSV in one pass with study config available locally."""
    study_config = get_study_config()
    validate_measurement_schedule()

    # Home robot once before the first experiment
    lash_e.nr_robot.home_robot_components()

    lash_e.logger.info("=" * 80)
    lash_e.logger.info("SURFACTANT GRID STUDY WORKFLOW")
    lash_e.logger.info(f"Recipes:  {recipes_path}")
    lash_e.logger.info(f"Stocks:   {stocks_path}")
    lash_e.logger.info(f"Simulate: {simulate}")
    lash_e.logger.info(f"Study config: {study_config}")
    lash_e.logger.info("=" * 80)

    well_recipes_df, dilution_recipes = load_replay_inputs(recipes_path, stocks_path)
    surfactant_a_name = well_recipes_df["surf_A"].dropna().iloc[0]
    surfactant_b_name = well_recipes_df["surf_B"].dropna().iloc[0]
    lash_e.logger.info(
        f"Surfactants: {surfactant_a_name} + {surfactant_b_name}, "
        f"{len(well_recipes_df)} rows, {len(dilution_recipes)} substocks"
    )

    if not simulate:
        import slack_agent
        slack_agent.send_slack_message(
            f"Starting surfactant grid study workflow: {surfactant_a_name}+{surfactant_b_name}, "
            f"{len(well_recipes_df)} wells, {PREPARATION_REPLICATES} prep replicate(s)"
        )

    if max_wells > 0:
        well_recipes_df = well_recipes_df.iloc[:max_wells].reset_index(drop=True)
        lash_e.logger.info(f"MAX_WELLS={max_wells}: running first {len(well_recipes_df)} wells only")

    wellplate_size = validate_execution_block_size(lash_e)
    well_recipes_df = scale_well_recipes(well_recipes_df, lash_e)
    if validate_liquids or validation_only:
        from workflows.embedded_liquid_validations import run_liquid_validations
        sources = [("water", "water", well_recipes_df["water_volume_ul"]),
                   (DYE_VIAL, DYE_SOLVENT, [get_dye_protocol()["dye_volume_ul"]])]
        for label, column in ((surfactant_a_name, "surf_A_volume_ul"),
                              (surfactant_b_name, "surf_B_volume_ul")):
            sources.append((f"{label}_stock", "SDS", well_recipes_df[column]))
        validation_results = run_liquid_validations(
            lash_e, sources, os.path.join("output", "surfactant_grid_ailsa"))
        if validation_only:
            return {"validation_results": validation_results, "study_config": study_config}
    plate_dfs = split_into_plates(well_recipes_df, wellplate_size=wellplate_size)
    plate_dfs = [apply_well_order(plate_df, lash_e.logger) for plate_df in plate_dfs]
    lash_e.logger.info(f"Source CSV splits into {len(plate_dfs)} plate(s)")

    replicate_results = []
    for preparation_replicate in range(1, PREPARATION_REPLICATES + 1):
        replicate_result = execute_preparation_replicate(
            lash_e,
            plate_dfs,
            dilution_recipes,
            surfactant_a_name,
            surfactant_b_name,
            preparation_replicate,
            simulate=simulate,
        )
        replicate_results.append(replicate_result)
        lash_e.logger.info(
            f"Completed preparation replicate {preparation_replicate}: "
            f"{replicate_result['output_folder']}"
        )
        if not simulate:
            import slack_agent
            slack_agent.send_slack_message(
                f"Preparation replicate {preparation_replicate}/{PREPARATION_REPLICATES} complete: "
                f"{replicate_result['output_folder']}"
            )

    pipette = get_pipette_usage_breakdown(lash_e)
    lash_e.logger.info(
        f"Pipette tips used: large={pipette['large_tips']} small={pipette['small_tips']} "
        f"total={pipette['total']}"
    )
    lash_e.logger.info("STUDY WORKFLOW COMPLETE")

    if not simulate:
        import slack_agent
        slack_agent.send_slack_message(
            f"Surfactant grid study workflow complete: {surfactant_a_name}+{surfactant_b_name}, "
            f"{PREPARATION_REPLICATES} prep replicate(s), "
            f"pipette tips used: large={pipette['large_tips']} small={pipette['small_tips']}"
        )

    return {
        "replicate_results": replicate_results,
        "study_config": study_config,
        "pipette_breakdown": pipette,
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

def execute(config=None, show_gui=True):
    """Use GUI-confirmed settings, or complete config with show_gui=False."""
    lash_e = Lash_E(
        workflow_globals=globals(), workflow_name="surfactant_grid_ailsa",
        config=config, show_gui=show_gui,
    )
    if not lash_e._workflow_should_continue:
        return None

    from workflows.interrupt_cleanup import cleanup_after_keyboard_interrupt
    try:
        c = lash_e.workflow_config
        return execute_study_workflow(
            c["RECIPES_CSV"],
            c["STOCKS_CSV"],
            lash_e,
            simulate=c["SIMULATE"],
            max_wells=c["MAX_WELLS"],
            validate_liquids=c["VALIDATE_LIQUIDS"],
            validation_only=c["VALIDATION_ONLY"],
        )
    except KeyboardInterrupt:
        lash_e.logger.warning("Workflow interrupted by keyboard; attempting cleanup")
        cleanup_after_keyboard_interrupt(lash_e)
        raise


if __name__ == "__main__":
    execute()

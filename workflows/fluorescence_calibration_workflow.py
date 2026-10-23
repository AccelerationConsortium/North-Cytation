"""Constant-solvent fluorescence standards in water, surfactant and solvent.

Run from the repository root: python -m workflows.fluorescence_calibration_workflow
Defaults execute the simulated workflow after startup GUI review.
Concentrations are relative to the parent dye stock unless its concentration is set.
"""

import sys
sys.path.append("../utoronto_demo")

from datetime import datetime
from pathlib import Path
import logging
import math
import re
import time

import numpy as np
import pandas as pd
import yaml

import slack_agent

logger = logging.getLogger(__name__)


# Workflow config constants, auto-detected by ConfigManager and persisted to
# workflow_configs/fluorescence_calibration_workflow.yaml (standard pattern:
# module-level UPPERCASE constants + workflow_globals=globals() at Lash_E init).
SIMULATE = True
VALIDATE_LIQUIDS = False
VALIDATION_ONLY = False
_SAVE_FILES_ON_SIMULATE = False
INPUT_VIAL_STATUS_FILE = "status/fluorescence_calibration_vials.csv"
DYE = "pyrene"
DYE_SOLVENT = "DMSO"
SOLVENT_SOURCE_VIALS = ["DMSO"]
SOLVENT_SOURCE_RESERVE_ML = 2.0
VIAL_NAME_OVERRIDES = {}
STOCK_CONCENTRATION_UM = None
DILUTION_FACTORS = [0.0, 0.1, 0.25, 0.5, 1.0]
DYE_VOLUME_UL = 5.0
TOTAL_VOLUME_UL = 200.0
VOLATILE_SOLVENT = False
SUBSTOCK_VOLUME_ML = 6.0
REPLICATES = 3
REPETITIONS = 1
FRESH_SUBSTOCKS = False
PREPARE_ONLY = False
MEASUREMENT_REPLICATES = 1
# Elapsed minutes (since the plate is ready) at which to read it; must start at 0.
MEASUREMENT_SCHEDULE_MIN = [0, 10, 30, 60, 120, 240, 360]
# Flat floor, not derived from MEASUREMENT_REPLICATES; widen by hand if needed.
MIN_SCHEDULE_GAP_MIN = 5
RANDOMIZED_ORDER = False
RANDOMIZATION_SEED = 42
WELLPLATE_TYPE = "96 WELL PLATE"
VOLUME_SCALE_BY_PLATE = {"96 WELL PLATE": 1, "48 WELL PLATE": 2, "24 WELL PLATE": 4}
DISPENSE_ORDER = ["medium", "dye"]
# Which base media to prepare wells for.
TEST_WATER = True
TEST_SURFACTANT = True
TEST_SOLVENT = True
SURFACTANT_NAME = "SDS"
SURFACTANT_CONCENTRATION_MM = None
SURFACTANT_CMC_MM = None
SURFACTANT_LIQUID = "SDS"
VORTEX_SECONDS = 5
PROTOCOL_FILE = None
RAW_CHANNELS = None
SHAKE_BEFORE_MEASUREMENT = False
SHAKE_PROTOCOL_FILE = r"C:\Protocols\shake_5.prt"
TWO_RUNS_PER_PLATE = False

# Names of the constants above; used to snapshot them into a plain config dict.
_CONFIG_KEYS = [
    "SIMULATE", "VALIDATE_LIQUIDS", "VALIDATION_ONLY", "INPUT_VIAL_STATUS_FILE", "DYE", "DYE_SOLVENT",
    "SOLVENT_SOURCE_VIALS", "SOLVENT_SOURCE_RESERVE_ML", "VIAL_NAME_OVERRIDES",
    "STOCK_CONCENTRATION_UM", "DILUTION_FACTORS", "DYE_VOLUME_UL",
    "TOTAL_VOLUME_UL", "VOLATILE_SOLVENT", "SUBSTOCK_VOLUME_ML", "REPLICATES", "REPETITIONS",
    "FRESH_SUBSTOCKS", "PREPARE_ONLY", "MEASUREMENT_REPLICATES", "MEASUREMENT_SCHEDULE_MIN",
    "MIN_SCHEDULE_GAP_MIN", "RANDOMIZED_ORDER",
    "RANDOMIZATION_SEED", "WELLPLATE_TYPE", "VOLUME_SCALE_BY_PLATE", "DISPENSE_ORDER",
    "TEST_WATER", "TEST_SURFACTANT", "TEST_SOLVENT",
    "SURFACTANT_NAME", "SURFACTANT_CONCENTRATION_MM", "SURFACTANT_CMC_MM",
    "SURFACTANT_LIQUID", "VORTEX_SECONDS", "PROTOCOL_FILE", "RAW_CHANNELS",
    "SHAKE_BEFORE_MEASUREMENT", "SHAKE_PROTOCOL_FILE",
    "TWO_RUNS_PER_PLATE",
]

# Leading underscore keeps these lookup tables out of ConfigManager's
# auto-detected config (it only picks up non-underscore constants).
# Pyrene settings match surfactant_grid_ailsa, pending Cytation verification of
# the proposed 355/373/383 settings. Other dyes remain placeholders.
_PROTOCOLS = {
    "pyrene": (r"C:\Protocols\CMC_Fluorescence_96.prt", ["334_373", "334_384"]),
    "coumarin-6": (r"C:\Protocols\Coumarin_96.prt", ["485_530"]),
    "nile-red": (r"C:\Protocols\NileRed_96.prt", ["550_648"]),
}
_PLATES = {"96 WELL PLATE": (96, 12), "48 WELL PLATE": (48, 8), "24 WELL PLATE": (24, 6)}
_STOCK_CONCENTRATIONS_UM = {"pyrene": 48.6, "nile-red": 50.0, "coumarin-6": 3.0}


def selected_pair(config):
    """Derive physical vial names from this run's single dye/solvent pair."""
    dye, solvent = config["DYE"], config["DYE_SOLVENT"]
    if not isinstance(dye, str) or dye.lower() not in _PROTOCOLS:
        raise ValueError("DYE must be pyrene, coumarin-6 or nile-red")
    if not isinstance(solvent, str) or not solvent.strip():
        raise ValueError("DYE_SOLVENT must name the solvent vial")
    pair_id = (re.sub(r"[^A-Za-z0-9]", "", dye.lower()) + "_" +
               re.sub(r"[^A-Za-z0-9]", "", solvent))
    if not pair_id.split("_", 1)[1]:
        raise ValueError("DYE_SOLVENT must contain letters or digits for vial naming")
    return pair_id, dye.lower(), solvent


def substock_vial_name(pair_id, batch, level, fresh_substocks):
    suffix = f"r{batch}_s{level}" if fresh_substocks else f"s{level}"
    return f"{pair_id}_{suffix}"


def configured_vial(config, generated_name):
    """Map a generated stock, substock or working name to physical inventory."""
    return config.get("VIAL_NAME_OVERRIDES", {}).get(generated_name, generated_name)


def build_plan(config):
    """Return well map and direct-from-parent substock recipes, without hardware."""
    c = config
    overrides = c.get("VIAL_NAME_OVERRIDES", {})
    if not isinstance(overrides, dict) or any(
        not isinstance(name, str) or not name.strip()
        for pair in overrides.items() for name in pair
    ):
        raise ValueError("VIAL_NAME_OVERRIDES must map nonempty vial names to nonempty vial names")
    sources = c["SOLVENT_SOURCE_VIALS"]
    if (not isinstance(sources, list) or not sources
            or any(not isinstance(source, str) or not source.strip() for source in sources)
            or len(set(sources)) != len(sources)):
        raise ValueError("SOLVENT_SOURCE_VIALS must be a nonempty list of distinct vial names")
    reserve = c["SOLVENT_SOURCE_RESERVE_ML"]
    if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("SOLVENT_SOURCE_RESERVE_ML must be finite and nonnegative")
    if type(c["SHAKE_BEFORE_MEASUREMENT"]) is not bool:
        raise ValueError("SHAKE_BEFORE_MEASUREMENT must be true or false")
    if c["SHAKE_BEFORE_MEASUREMENT"] and (
        not isinstance(c["SHAKE_PROTOCOL_FILE"], str) or not c["SHAKE_PROTOCOL_FILE"].strip()
    ):
        raise ValueError("Set SHAKE_PROTOCOL_FILE when SHAKE_BEFORE_MEASUREMENT is enabled")
    pair_id, dye, solvent = selected_pair(c)
    if type(c["PREPARE_ONLY"]) is not bool:
        raise ValueError("PREPARE_ONLY must be true or false")
    if c["WELLPLATE_TYPE"] not in _PLATES:
        raise ValueError("Only 24/48/96-well plate layouts are supported")
    scales = c["VOLUME_SCALE_BY_PLATE"]
    if not isinstance(scales, dict) or c["WELLPLATE_TYPE"] not in scales:
        raise ValueError(f"No volume scale configured for {c['WELLPLATE_TYPE']}")
    scale = scales[c["WELLPLATE_TYPE"]]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Volume scale for {c['WELLPLATE_TYPE']} must be finite and positive")
    for key in ("REPLICATES", "REPETITIONS", "MEASUREMENT_REPLICATES"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("DYE_VOLUME_UL", "TOTAL_VOLUME_UL", "SUBSTOCK_VOLUME_ML", "VORTEX_SECONDS"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if type(c["VOLATILE_SOLVENT"]) is not bool:
        raise ValueError("VOLATILE_SOLVENT must be true or false")
    schedule = c["MEASUREMENT_SCHEDULE_MIN"]
    if not schedule or schedule[0] != 0 or any(
        not math.isfinite(t) for t in schedule
    ) or list(schedule) != sorted(set(schedule)):
        raise ValueError("MEASUREMENT_SCHEDULE_MIN must start at 0 and strictly increase")
    if any(b - a < c["MIN_SCHEDULE_GAP_MIN"] for a, b in zip(schedule, schedule[1:])):
        raise ValueError(f"MEASUREMENT_SCHEDULE_MIN gaps must be >= MIN_SCHEDULE_GAP_MIN ({c['MIN_SCHEDULE_GAP_MIN']} min)")
    factors = c["DILUTION_FACTORS"]
    if len(factors) < 2 or len(set(factors)) != len(factors) or any(
        not math.isfinite(f) or not 0 <= f <= 1 for f in factors
    ):
        raise ValueError("Provide at least two distinct dilution factors in [0, 1]")
    if sorted(c["DISPENSE_ORDER"]) != ["dye", "medium"]:
        raise ValueError("DISPENSE_ORDER must contain medium and dye once each")
    fraction = c["DYE_VOLUME_UL"] / c["TOTAL_VOLUME_UL"]
    if not c["VOLATILE_SOLVENT"] and not 0 < fraction < 1:
        raise ValueError("Dye addition must be smaller than total well volume")
    for key in ("STOCK_CONCENTRATION_UM", "SURFACTANT_CONCENTRATION_MM", "SURFACTANT_CMC_MM"):
        if c[key] is not None and (not math.isfinite(c[key]) or c[key] <= 0):
            raise ValueError(f"{key} must be positive or null")
    surf = c["SURFACTANT_CONCENTRATION_MM"]
    cmc = c["SURFACTANT_CMC_MM"]
    if surf is not None and cmc is not None and surf * (1 if c["VOLATILE_SOLVENT"] else 1 - fraction) <= cmc:
        raise ValueError("Final surfactant concentration must remain above the configured CMC")
    stock_concentration = c["STOCK_CONCENTRATION_UM"]
    if stock_concentration is None:
        stock_concentration = _STOCK_CONCENTRATIONS_UM[dye]
    capacity, columns = _PLATES[c["WELLPLATE_TYPE"]]
    scaled_dye_ul = c["DYE_VOLUME_UL"] * scale
    scaled_total_ul = c["TOTAL_VOLUME_UL"] * scale
    rng = np.random.default_rng(c["RANDOMIZATION_SEED"])
    media = [medium for medium, enabled in (
        ("water", c["TEST_WATER"]), ("surfactant", c["TEST_SURFACTANT"]),
        ("solvent", c["TEST_SOLVENT"] and not c["VOLATILE_SOLVENT"])
    ) if enabled]
    if not media:
        raise ValueError("At least one compatible medium must be enabled")
    rows, recipes = [], []
    sets = c["REPETITIONS"] if c["FRESH_SUBSTOCKS"] else 1
    for batch in range(1, sets + 1):
        for level, factor in enumerate(factors):
            if 0 < factor < 1:
                recipes.append(dict(batch=batch, vial=configured_vial(c, substock_vial_name(pair_id, batch, level, c["FRESH_SUBSTOCKS"])),
                                    stock_ml=c["SUBSTOCK_VOLUME_ML"] * factor,
                                    solvent_ml=c["SUBSTOCK_VOLUME_ML"] * (1 - factor)))
    plate_offset = 0
    for repetition in range(1, c["REPETITIONS"] + 1):
        batch = repetition if c["FRESH_SUBSTOCKS"] else 1
        block = []
        for medium in media:
            for level, factor in enumerate(factors):
                source = f"{solvent}_working" if factor == 0 else (
                    f"{pair_id}_stock" if factor == 1 else
                    substock_vial_name(pair_id, batch, level, c["FRESH_SUBSTOCKS"]))
                source = configured_vial(c, source)
                for replicate in range(1, c["REPLICATES"] + 1):
                    block.append(dict(repetition=repetition, substock_batch=batch,
                        replicate=replicate, medium=medium, dye_source=source,
                        dye=dye, dye_solvent=solvent,
                        dilution_factor=factor, concentration_relative=factor * fraction,
                        concentration_um=stock_concentration * factor * fraction,
                        dye_volume_ul=scaled_dye_ul,
                        medium_volume_ul=scaled_total_ul if c["VOLATILE_SOLVENT"] else scaled_total_ul - scaled_dye_ul,
                        volume_scale=scale,
                        solvent_fraction=1.0 if medium == "solvent" else 0.0 if c["VOLATILE_SOLVENT"] else fraction,
                        surfactant_final_mm=surf * (1 if c["VOLATILE_SOLVENT"] else 1 - fraction)
                        if medium == "surfactant" and surf is not None else None))
        if c["RANDOMIZED_ORDER"]:
            rng.shuffle(block)
        for index, row in enumerate(block):
            well = index % capacity
            row.update(plate=plate_offset + index // capacity + 1, well_index=well,
                       well_position=f"{chr(65 + well // columns)}{well % columns + 1}")
        plate_offset += math.ceil(len(block) / capacity)
        rows.extend(block)
    plan = pd.DataFrame(rows)
    for recipe in recipes:
        required = (plan.dye_source == recipe["vial"]).sum() * scaled_dye_ul / 1000
        if required + 0.1 > c["SUBSTOCK_VOLUME_ML"]:
            raise ValueError(f"Insufficient substock volume for {recipe['vial']} (including 0.1 mL reserve)")
    return plan, pd.DataFrame(recipes, columns=["batch", "vial", "stock_ml", "solvent_ml"])


def per_well_means(results, channels):
    """Average repeat reads (measurement_replicate) per well, keeping wells separate."""
    keys = ["repetition", "medium", "timepoint_min", "concentration_relative"]
    return results.groupby(keys + ["plate", "well_index"], as_index=False)[channels].mean()


def summarize(results, channels):
    """Average repeat reads (measurement_replicate) per well/timepoint, then summarize independent preparation wells."""
    keys = ["repetition", "medium", "timepoint_min", "concentration_relative"]
    wells = per_well_means(results, channels)
    summary = wells.groupby(keys)[channels].agg(["mean", "std", "count"])
    summary.columns = [f"{channel}_{stat}" for channel, stat in summary.columns]
    summary = summary.reset_index()
    for channel in channels:
        blanks = summary[summary.concentration_relative == 0].set_index(
            ["repetition", "medium", "timepoint_min"])[f"{channel}_mean"]
        summary[f"{channel}_blank_corrected"] = [
            row[f"{channel}_mean"] - blanks.loc[(row.repetition, row.medium, row.timepoint_min)]
            for _, row in summary.iterrows()]
    return summary


def fit_calibration(summary, channels, stock_concentration_um=None):
    """Export unconstrained linear fits for review, separately for each medium/run.

    Fits do not assert linearity or quantify encapsulation. Review the range,
    residuals and blank variability before using concentration=(signal-intercept)/slope.
    """
    fits = []
    for (repetition, medium, timepoint_min), group in summary.groupby(["repetition", "medium", "timepoint_min"]):
        x = group.concentration_relative.to_numpy()
        for channel in channels:
            y = group[f"{channel}_blank_corrected"].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            residual = y - (slope * x + intercept)
            total = np.sum((y - y.mean()) ** 2)
            fits.append(dict(repetition=repetition, medium=medium, timepoint_min=timepoint_min, channel=channel,
                slope_per_relative=slope, intercept_blank_corrected=intercept,
                slope_per_um=None if stock_concentration_um is None else slope / stock_concentration_um,
                r_squared=None if total == 0 else 1 - np.sum(residual ** 2) / total,
                rmse=float(np.sqrt(np.mean(residual ** 2))),
                min_concentration_relative=float(x.min()), max_concentration_relative=float(x.max()),
                status="review_linearity_before_use" if slope > 0 else "nonpositive_slope_do_not_invert"))
    return pd.DataFrame(fits)


def simulate_fluorescence_readout(wells, channels):
    """Fake Cytation reads for simulate mode: intensity rises with relative dye concentration."""
    rng = np.random.default_rng(0)
    data = {"well_position": wells.well_position.to_numpy()}
    for channel in channels:
        data[channel] = 500 + wells.concentration_relative.to_numpy() * 5000 + rng.normal(0, 20, len(wells))
    return pd.DataFrame(data)


_MEDIUM_COLORS = {"water": "tab:blue", "surfactant": "tab:orange", "solvent": "tab:green"}


def plot_calibration(wells, summary, channels, output_dir):
    """Plot raw per-well points with a dotted mean trend (no error bars), one panel per channel/timepoint with all media overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x_column = "concentration_um" if "concentration_um" in summary else "concentration_relative"
    for timepoint_min, summary_time in summary.groupby("timepoint_min"):
        well_time = wells[wells.timepoint_min == timepoint_min]
        for channel in channels:
            fig, ax = plt.subplots(figsize=(6, 4))
            for medium, summary_group in summary_time.groupby("medium"):
                summary_group = summary_group.sort_values(x_column)
                well_group = well_time[well_time.medium == medium]
                color = _MEDIUM_COLORS.get(medium)
                ax.scatter(well_group[x_column], well_group[channel], alpha=0.6, s=30,
                           color=color, label=f"{medium} (wells)")
                ax.plot(summary_group[x_column], summary_group[f"{channel}_mean"], linestyle=":",
                        color=color, label=f"{medium} (mean)")
            ax.set(title=f"{channel}: t={timepoint_min} min", xlabel="Dye concentration (uM)" if x_column == "concentration_um" else "Relative dye concentration", ylabel="Fluorescence intensity")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(Path(output_dir) / f"intensity_vs_concentration_{channel}_t{timepoint_min}.png", dpi=180)
            plt.close(fig)


def plot_kinetics(wells, channels, output_dir):
    """Plot fluorescence vs. elapsed time, one panel per medium/channel, colored by concentration level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x_column = "concentration_um" if "concentration_um" in wells else "concentration_relative"
    levels = sorted(wells.concentration_relative.unique())
    colors = plt.colormaps["viridis"](np.linspace(0, 1, len(levels)))
    for medium, well_medium in wells.groupby("medium"):
        for channel in channels:
            fig, ax = plt.subplots(figsize=(6, 4))
            for level, color in zip(levels, colors):
                well_group = well_medium[well_medium.concentration_relative == level]
                if well_group.empty:
                    continue
                label = f"{well_group[x_column].iloc[0]:.3g}" if x_column == "concentration_um" else f"{level:.3g}"
                ax.scatter(well_group.timepoint_min, well_group[channel], alpha=0.6, s=30, color=color)
                means = well_group.groupby("timepoint_min")[channel].mean().sort_index()
                ax.plot(means.index, means.to_numpy(), linestyle=":", color=color, label=label)
            ax.set(title=f"{channel}: {medium} over time", xlabel="Elapsed time (min)", ylabel="Fluorescence intensity")
            ax.grid(alpha=0.25)
            ax.legend(title="Dye concentration (uM)" if x_column == "concentration_um" else "Relative dye concentration")
            fig.tight_layout()
            fig.savefig(Path(output_dir) / f"intensity_vs_time_{medium}_{channel}.png", dpi=180)
            plt.close(fig)


def dispense_vial_to_vial(lash, source, destination, volume, liquid):
    """Stage movable vials when a closed cap requires clamp exchanges."""
    robot = lash.nr_robot
    if not any(robot.get_vial_info(vial, "cap_type") == "closed"
               for vial in (source, destination)):
        return robot.dispense_from_vial_into_vial(source, destination, volume, liquid=liquid)
    if volume <= 0:
        return robot.dispense_from_vial_into_vial(source, destination, volume, liquid=liquid)

    staged = []
    for vial, slot in ((source, 0), (destination, 1)):
        if not robot.is_vial_movable(vial):
            continue  # Large solvent sources are aspirated in place.
        index = robot.get_vial_info(vial, "vial_index")
        occupant = robot.get_vial_in_location("main_8mL_rack", slot)
        if occupant is not None and occupant != index:
            raise RuntimeError(f"Vial-transfer staging main_8mL_rack[{slot}] is occupied")
        staged.append((vial, index, slot,
                       robot.get_vial_info(vial, "home_location"),
                       robot.get_vial_info(vial, "home_location_index")))

    robot.remove_pipet()
    try:
        for vial, index, slot, _, _ in staged:
            # Clamp exchanges return the previous vial home with a tip attached.
            robot.VIAL_DF.at[index, "home_location"] = "main_8mL_rack"
            robot.VIAL_DF.at[index, "home_location_index"] = slot
            robot.move_vial_to_location(vial, "main_8mL_rack", slot)
        result = robot.dispense_from_vial_into_vial(source, destination, volume, liquid=liquid)
        robot.remove_pipet()
    finally:
        # Restore metadata even on failure; do not attempt recovery movements.
        for _, index, _, home, home_index in staged:
            robot.VIAL_DF.at[index, "home_location"] = home
            robot.VIAL_DF.at[index, "home_location_index"] = home_index
        robot.save_robot_status()
    for vial, _, _, _, _ in staged:
        robot.return_vial_home(vial)
    return result


def plan_solvent_sources(source_volumes, volume, liquid, reserve):
    """Use sources in order with a minimum 0.2 mL contribution per source."""
    minimum = 0.2
    if not math.isfinite(volume) or volume < minimum - 1e-9:
        raise ValueError("Solvent additions must be finite and at least 0.2 mL")
    available = []
    for source, current in source_volumes.items():
        if not math.isfinite(current) or current < 0:
            raise ValueError(f"Invalid recorded solvent volume for {source}: {current}")
        usable = math.floor(max(0.0, current - reserve) * 1000 + 1e-6) / 1000
        available.append((source, usable if usable >= minimum - 1e-9 else 0.0))
    if sum(amount for _, amount in available) + 1e-9 < volume:
        raise ValueError(f"Insufficient bulk {liquid}: need {volume:.3f} mL, "
                         f"have {sum(amount for _, amount in available):.3f} mL usable")
    transfers = []
    remaining = volume
    for source, amount in available:
        take = min(amount, remaining)
        if 1e-9 < remaining - take < minimum - 1e-9:
            take = remaining - minimum
        if take >= minimum - 1e-9:
            transfers.append((source, take))
            remaining -= take
        if remaining <= 1e-9:
            break
    if remaining > 1e-9:
        raise ValueError(f"Cannot allocate {volume:.3f} mL of {liquid} with at least "
                         "0.2 mL per source while retaining source reserves")
    return transfers


def dispense_solvent(lash, sources, destination, volume, liquid, reserve):
    """Transfer from one or more sources without drawing below their reserves."""
    source_volumes = {source: lash.nr_robot.get_vial_info(source, "vial_volume")
                      for source in sources}
    for source, current in source_volumes.items():
        if reserve < current < reserve + 0.2:
            lash.logger.info(f"Leaving sub-200 uL usable remainder in solvent source {source}")
    for source, amount in plan_solvent_sources(source_volumes, volume, liquid, reserve):
        lash.logger.info(f"Solvent transfer: {source} -> {destination}, {amount:.3f} mL")
        dispense_vial_to_vial(lash, source, destination, amount, liquid)


def check_liquid_supply(config, plan, recipes, inventory, working_fill, surfactant_fill):
    """Check the full run before refills or preparation, retaining 0.1 mL per vial.

    Uses recorded inventory volumes; physical loading must match those records.
    Solvent allocation uses the same source-pooling plan as dispensing.
    """
    c = config
    pair_id, _, solvent = selected_pair(c)
    stock = configured_vial(c, f"{pair_id}_stock")
    working = configured_vial(c, f"{solvent}_working")
    pending = recipes[recipes.vial.map(lambda vial: inventory.loc[vial, "vial_volume"] <= 1.0)]
    required = {stock: float(pending.stock_ml.sum())}
    available = {}
    for recipe in recipes.itertuples():
        available[recipe.vial] = (c["SUBSTOCK_VOLUME_ML"] if recipe.vial in set(pending.vial)
                                  else inventory.loc[recipe.vial, "vial_volume"])
    available[working] = inventory.loc[working, "vial_volume"] + working_fill
    if not c["PREPARE_ONLY"]:
        for source, amount in plan.groupby("dye_source").dye_volume_ul.sum().items():
            required[source] = required.get(source, 0.0) + amount / 1000
        for medium, amount in plan.groupby("medium").medium_volume_ul.sum().items():
            source = working if medium == "solvent" else medium
            required[source] = required.get(source, 0.0) + amount / 1000
        if c["TEST_WATER"]:
            available["water"] = max(8.0, inventory.loc["water", "vial_volume"])
        if c["TEST_SURFACTANT"]:
            available["surfactant"] = inventory.loc["surfactant", "vial_volume"] + (
                surfactant_fill if surfactant_fill > 0.1 else 0.0)
    for source, demand in required.items():
        volume = available.get(source, inventory.loc[source, "vial_volume"])
        if not math.isfinite(volume) or volume < 0:
            raise ValueError(f"Invalid recorded liquid volume for {source}: {volume}")
        if demand > 0 and demand + 0.1 > volume + 1e-9:
            raise ValueError(f"Insufficient liquid in {source}: need {demand:.3f} mL "
                             f"plus 0.1 mL reserve, have {volume:.3f} mL after startup refills")
    remaining = {source: inventory.loc[source, "vial_volume"]
                 for source in c["SOLVENT_SOURCE_VIALS"]}
    for addition in [working_fill, *pending.solvent_ml]:
        if addition <= 0:
            continue
        for source, amount in plan_solvent_sources(
                remaining, addition, solvent, c["SOLVENT_SOURCE_RESERVE_ML"]):
            remaining[source] -= amount


def execute(config=None, show_gui=True):
    """Review settings normally, or use complete config with show_gui=False."""
    from master_usdl_coordinator import Lash_E, flatten_cytation_data
    lash = Lash_E(workflow_globals=globals(), workflow_name="fluorescence_calibration_workflow",
                  config=config, show_gui=show_gui)
    if not lash._workflow_should_continue:
        return None

    from workflows.interrupt_cleanup import cleanup_after_keyboard_interrupt
    try:
        lash.nr_robot.home_robot_components()
        c = lash.workflow_config
        plan, recipes = build_plan(c)
        from workflows import two_run_plate
        half = None
        if c["TWO_RUNS_PER_PLATE"] and not c["PREPARE_ONLY"]:
            capacity, columns = _PLATES[c["WELLPLATE_TYPE"]]
            if c["REPETITIONS"] != 1 or len(plan) > capacity // 2:
                raise ValueError("TWO_RUNS_PER_PLATE requires one repetition fitting in half a plate")
            half = two_run_plate.next_half(lash)
            plan["well_index"] += half * (capacity // 2)
            plan["well_position"] = plan.well_index.map(
                lambda index: f"{chr(65 + index // columns)}{index % columns + 1}")
            plan["plate_half"] = half + 1
        pair_id, dye, solvent = selected_pair(c)
        working_vial = configured_vial(c, f"{solvent}_working")
        stock_vial = configured_vial(c, f"{pair_id}_stock")
        protocol, channels = _PROTOCOLS[dye]
        protocol = c["PROTOCOL_FILE"] or protocol
        channels = c["RAW_CHANNELS"] or channels
        if not c["PREPARE_ONLY"]:
            if not protocol or not channels:
                raise ValueError("Set PROTOCOL_FILE/RAW_CHANNELS (or a configured dye protocol and channels) before execution")
            # Protocol files live on the Cytation PC; simulate mode never opens them, so skip the check.
            if not c["SIMULATE"] and not Path(protocol).is_file():
                raise ValueError(f"Set an existing, plate-matched Cytation protocol: {protocol}")
            if c["SHAKE_BEFORE_MEASUREMENT"] and not c["SIMULATE"] and not Path(c["SHAKE_PROTOCOL_FILE"]).is_file():
                raise ValueError(f"Set an existing, plate-matched shake protocol: {c['SHAKE_PROTOCOL_FILE']}")
        if not c["SIMULATE"]:
            slack_agent.send_slack_message(
                f"Fluorescence calibration workflow started: {pair_id} in {', '.join(plan.medium.unique())}")
        inventory = pd.read_csv(c["INPUT_VIAL_STATUS_FILE"]).set_index("vial_name")
        solvent_sources = c["SOLVENT_SOURCE_VIALS"]
        reserve = c["SOLVENT_SOURCE_RESERVE_ML"]
        required_vials = {*solvent_sources, working_vial, stock_vial, *recipes.vial}
        generated_roles = [working_vial, stock_vial, *recipes.vial]
        if len(generated_roles) != len(set(generated_roles)) or set(generated_roles) & {"water", "surfactant", "surfactant_refill"}:
            raise ValueError("Stock, substock and working-solvent roles must use distinct vials")
        if c["TEST_WATER"]:
            required_vials.add("water")
        if c["TEST_SURFACTANT"]:
            required_vials.update(("surfactant", "surfactant_refill"))
        missing_vials = sorted(required_vials - set(inventory.index))
        if missing_vials:
            raise ValueError(f"Missing vials for {pair_id}: {', '.join(missing_vials)}")
        if not inventory.index.is_unique:
            raise ValueError("Vial names must be unique in the selected inventory")
        if set(solvent_sources) & {working_vial, stock_vial, *recipes.vial,
                                  "water", "surfactant", "surfactant_refill"}:
            raise ValueError("SOLVENT_SOURCE_VIALS must contain only separate pure-solvent containers")
        source_volumes = pd.to_numeric(inventory.loc[solvent_sources, "vial_volume"], errors="raise")
        if not np.isfinite(source_volumes).all() or (source_volumes < 0).any():
            raise ValueError("Solvent source volumes must be finite and nonnegative")
        # Stage one 8 mL source at the clamp; large vials remain in place.
        for location, index in (("location", "location_index"), ("home_location", "home_location_index")):
            if ((inventory[location] == "clamp") & (inventory[index] == 0)).any():
                raise ValueError("Keep clamp[0] free for plate-dispensing staging")
        if not c["PREPARE_ONLY"]:
            if lash.nr_track.CURRENT_WP_TYPE != c["WELLPLATE_TYPE"]:
                raise ValueError("Track plate type must match configured WELLPLATE_TYPE")
            plate_config = lash.nr_robot.WELLPLATES[c["WELLPLATE_TYPE"]]
            maximum_dispensed_ul = (c["TOTAL_VOLUME_UL"] +
                                    (c["DYE_VOLUME_UL"] if c["VOLATILE_SOLVENT"] else 0)) * c["VOLUME_SCALE_BY_PLATE"][c["WELLPLATE_TYPE"]]
            if maximum_dispensed_ul / 1000 > plate_config["max_volume_per_well"]:
                raise ValueError("Initial dispensed volume exceeds plate capacity")
            if lash.nr_track.NUM_SOURCE < (0 if half == 1 else plan.plate.nunique()):
                raise ValueError("Load enough plates for the complete experiment")
        working = inventory.loc[working_vial]
        if working.vial_type != "8_mL" or working.cap_type != "closed":
            raise ValueError(f"{working_vial} must be a closed-cap 8 mL vial")
        working_required = (
            plan.loc[plan.dye_source == working_vial, "dye_volume_ul"].sum()
            + plan.loc[plan.medium == "solvent", "medium_volume_ul"].sum()) / 1000
        # Fill enough for this plan, with reserve, without exceeding the 8 mL vial.
        if working_required + 0.1 > 8.0:
            raise ValueError("Insufficient working solvent capacity: scaled plan needs more than one 8 mL vial fill")
        pending_solvent = sum(recipe.solvent_ml for recipe in recipes.itertuples()
                              if inventory.loc[recipe.vial, "vial_volume"] <= 1.0)
        usable_amounts = [math.floor(max(0.0, v - reserve) * 1000 + 1e-6) / 1000
                          for v in source_volumes]
        usable_solvent = sum(v for v in usable_amounts if v >= 0.2 - 1e-9)
        working_target = max(6.0, working_required + 0.1)
        validation_enabled = c["VALIDATE_LIQUIDS"] or c["VALIDATION_ONLY"]
        working_minimum = working_target if validation_enabled else working_required + 0.1
        working_fill = (0.0 if working.vial_volume >= working_minimum
                        else min(max(0.0, working_target - working.vial_volume),
                                 max(0.0, usable_solvent - pending_solvent)))
        if working_required + 0.1 > working.vial_volume + working_fill + 1e-9:
            raise ValueError("Insufficient working solvent for plate dispensing (including 0.1 mL reserve)")
        if usable_solvent + 1e-9 < pending_solvent + working_fill:
            raise ValueError(
                f"Insufficient bulk solvent: need {pending_solvent + working_fill:.3f} mL "
                f"for substocks and working-vial refill, have {usable_solvent:.3f} mL usable "
                f"across {', '.join(solvent_sources)} (reserve {reserve:.3f} mL per container)")
        surfactant_fill = 0.0
        if c["TEST_SURFACTANT"]:
            # Match the surfactant workflow's 7.5 mL working-vial target.
            surfactant_fill = max(0.0, 7.5 - lash.nr_robot.get_vial_info("surfactant", "vial_volume"))
            if surfactant_fill > 0.1 and lash.nr_robot.get_vial_info("surfactant_refill", "vial_volume") < surfactant_fill:
                raise ValueError("Insufficient surfactant_refill volume to top up surfactant to 7.5 mL")
        check_liquid_supply(c, plan, recipes, inventory, working_fill, surfactant_fill)
        if c["VALIDATE_LIQUIDS"] or c["VALIDATION_ONLY"]:
            from pipetting_data.embedded_liquid_validations import run_liquid_validations
            validation_sources = []
            pending = recipes[recipes.vial.map(lambda vial: inventory.loc[vial, "vial_volume"] <= 1.0)]
            preparation_volumes = [v * 1000 for v in pending.stock_ml if v > 0]
            remaining = source_volumes.to_dict()
            for addition in [working_fill, *pending.solvent_ml]:
                if addition <= 0:
                    continue
                for source, amount in plan_solvent_sources(remaining, addition, solvent, reserve):
                    preparation_volumes.append(amount * 1000)
                    remaining[source] -= amount
            vial_transfers = [(working_vial, solvent, preparation_volumes)]
            if not c["PREPARE_ONLY"]:
                solvent_volumes = plan.loc[plan.medium == "solvent", "medium_volume_ul"].tolist()
                solvent_volumes += plan["dye_volume_ul"].tolist()
                validation_sources.append((working_vial, solvent, solvent_volumes))
                if c["TEST_WATER"]:
                    validation_sources.append(("water", "water", plan.loc[plan.medium == "water", "medium_volume_ul"]))
                if c["TEST_SURFACTANT"]:
                    validation_sources.append(("surfactant", c["SURFACTANT_LIQUID"],
                                               plan.loc[plan.medium == "surfactant", "medium_volume_ul"]))
        if c["TEST_WATER"]:
            water_fill = 8.0 - lash.nr_robot.get_vial_info("water", "vial_volume")
            if water_fill > 0.1:
                lash.logger.info(f"Refilling water to 8 mL from reservoir 1 (adding {water_fill:.3f} mL)")
                lash.nr_robot.remove_pipet()
                lash.nr_robot.dispense_into_vial_from_reservoir(1, "water", water_fill)
        if surfactant_fill > 0.1:
            lash.logger.info(f"Refilling surfactant to 7.5 mL (adding {surfactant_fill:.3f} mL)")
            dispense_vial_to_vial(lash, "surfactant_refill", "surfactant", surfactant_fill,
                                 liquid=c["SURFACTANT_LIQUID"])
        if working_fill > 0:
            dispense_solvent(lash, solvent_sources, working_vial, working_fill, solvent, reserve)
        if validation_enabled:
            validation_results = run_liquid_validations(
                lash, validation_sources, Path("output") / "fluorescence_calibration",
                vial_transfers=vial_transfers)
            if c["VALIDATION_ONLY"]:
                return {"validation_results": validation_results}
            inventory = lash.nr_robot.VIAL_DF.copy().set_index("vial_name", drop=False)
            current = inventory.loc[working_vial, "vial_volume"]
            additional_fill = 0.0
            if not c["PREPARE_ONLY"] and current < working_required + 0.1:
                additional_fill = max(0.2, working_required + 0.1 - current)
            check_liquid_supply(c, plan, recipes, inventory, additional_fill, 0.0)
            if c["TEST_WATER"] and not c["PREPARE_ONLY"]:
                demand = plan.loc[plan.medium == "water", "medium_volume_ul"].sum() / 1000
                if inventory.loc["water", "vial_volume"] < demand + 0.1:
                    raise ValueError("Insufficient water after validation for plate dispensing")
            if additional_fill > 0:
                dispense_solvent(lash, solvent_sources, working_vial, additional_fill, solvent, reserve)
        output = None
        if not lash.simulate or _SAVE_FILES_ON_SIMULATE:
            suffix = "_simulate" if lash.simulate else ""
            name_suffix = "_" + c["DYE"] + "_" + c["DYE_SOLVENT"]
            output = Path("output") / "fluorescence_calibration" / (
                "fluorescence_calibration_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + suffix + name_suffix)
            output.mkdir(parents=True)
            plan.to_csv(output / "well_plan.csv", index=False)
            recipes.to_csv(output / "substock_recipes.csv", index=False)
            (output / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        else:
            lash.logger.info("Simulation file output disabled; results remain in memory.")
        prepared, measurements = set(), []
        for plate, wells in plan.groupby("plate", sort=True):
            batch = int(wells.substock_batch.iloc[0])
            if batch not in prepared:
                for recipe in recipes[recipes.batch == batch].itertuples():
                    # Loaded destinations are either empty or prepared; volume is
                    # the reuse marker, not a check of concentration or recipe.
                    if inventory.loc[recipe.vial, "vial_volume"] > 1.0:
                        lash.logger.info(f"Skipping already-prepared substock {recipe.vial}")
                        continue
                    dispense_solvent(lash, solvent_sources, recipe.vial, recipe.solvent_ml, solvent, reserve)
                    dispense_vial_to_vial(lash, stock_vial, recipe.vial, recipe.stock_ml, liquid=solvent)
                    lash.nr_robot.vortex_vial(recipe.vial, c["VORTEX_SECONDS"])
                    lash.nr_robot.return_vial_home(recipe.vial)
                prepared.add(batch)
            if c["PREPARE_ONLY"]:
                continue
            if half is None:
                lash.grab_new_wellplate()
            else:
                two_run_plate.begin(lash, half, str(output) if output is not None else None)
            for component in c["DISPENSE_ORDER"]:
                source_column = "medium" if component == "medium" else "dye_source"
                volume_column = f"{component}_volume_ul"
                for source, subset in wells.groupby(source_column):
                    vial_source = working_vial if component == "medium" and source == "solvent" else source
                    liquid = (solvent if component == "dye" or source == "solvent"
                              else c["SURFACTANT_LIQUID"] if source == "surfactant" else "water")
                    volumes = pd.DataFrame({vial_source: subset.set_index("well_index")[volume_column] / 1000})
                    if inventory.loc[vial_source, "vial_type"] == "8_mL":
                        lash.nr_robot.remove_pipet()
                        if lash.nr_robot.get_vial_in_location("clamp", 0) is not None:
                            raise RuntimeError("Plate-dispensing staging clamp[0] is occupied")
                        lash.nr_robot.move_vial_to_location(vial_source, "clamp", 0)
                    # Serial dispensing removes its tip and returns the source home.
                    # Large vials are aspirated in place, with no staging move.
                    lash.nr_robot.dispense_from_vials_into_wellplate(
                        volumes, liquid=liquid, strategy="serial", well_plate_type=c["WELLPLATE_TYPE"])
            # Schedule starts after dispensing; optional shaking adds to actual read time.
            t0 = time.monotonic()
            for timepoint_min in c["MEASUREMENT_SCHEDULE_MIN"]:
                target = t0 + timepoint_min * 60
                wait = target - time.monotonic()
                if wait > 0 and not lash.simulate:
                    time.sleep(wait)
                for read in range(1, c["MEASUREMENT_REPLICATES"] + 1):
                    read_options = {}
                    if c["SHAKE_BEFORE_MEASUREMENT"] and read == 1:
                        read_options["preparation_protocol"] = c["SHAKE_PROTOCOL_FILE"]
                    raw = lash.measure_wellplate(protocol, wells.well_index.tolist(), plate_type=c["WELLPLATE_TYPE"], **read_options)
                    if raw is None:
                        if not lash.simulate:
                            raise RuntimeError(f"No fluorescence data for plate {plate}, t={timepoint_min}min, read {read}")
                        data = simulate_fluorescence_readout(wells, channels)
                    else:
                        if output is not None:
                            raw.to_csv(output / f"raw_plate_{plate}_t{timepoint_min}_rep{read}.csv")
                        data = flatten_cytation_data(raw, "fluorescence")
                        if not set(["well_position", *channels]).issubset(data.columns):
                            raise ValueError(f"Unexpected Cytation channels: {list(data.columns)}")
                        data["well_position"] = data.well_position.map(lambda p: f"{str(p)[0].upper()}{int(str(p)[1:])}")
                    merged = wells.merge(data[["well_position", *channels]], on="well_position",
                                         how="left", validate="one_to_one")
                    merged[channels] = merged[channels].apply(pd.to_numeric, errors="raise")
                    if not np.isfinite(merged[channels].to_numpy()).all():
                        raise ValueError("Missing, saturated or non-finite fluorescence readings")
                    merged["timepoint_min"] = timepoint_min
                    merged["elapsed_actual_min"] = (time.monotonic() - t0) / 60
                    merged["measurement_replicate"] = read
                    measurements.append(merged)
                    if output is not None:
                        pd.concat(measurements).to_csv(output / "fluorescence_results.csv", index=False)
                if not c["SIMULATE"]:
                    slack_agent.send_slack_message(f"Fluorescence calibration: plate {plate}, t={timepoint_min} min measured")
            if half is None:
                lash.discard_used_wellplate()
            else:
                two_run_plate.finish(lash, half)
        if c["PREPARE_ONLY"]:
            lash.nr_robot.remove_pipet()
            lash.nr_robot.save_robot_status()
            lash.logger.info(f"Fluorescence solution preparation complete; plans saved to {output}")
            if not lash.simulate:
                slack_agent.send_slack_message(f"Fluorescence solution preparation complete: {pair_id}")
            return output
        combined = pd.concat(measurements)
        summary = summarize(combined, channels)
        effective_stock = c["STOCK_CONCENTRATION_UM"] or _STOCK_CONCENTRATIONS_UM[dye]
        summary["concentration_um"] = summary["concentration_relative"] * effective_stock
        fits = fit_calibration(summary, channels, effective_stock)
        wells = per_well_means(combined, channels)
        wells["concentration_um"] = wells["concentration_relative"] * effective_stock
        if output is not None:
            summary.to_csv(output / "calibration_summary.csv", index=False)
            fits.to_csv(output / "calibration_fits.csv", index=False)
            plot_calibration(wells, summary, channels, output)
            plot_kinetics(wells, channels, output)
            lash.logger.info(f"Calibration measurements saved to {output}")
        else:
            lash.logger.info("Fluorescence calibration complete; no experiment files saved.")
        if not c["SIMULATE"]:
            slack_agent.send_slack_message(f"Fluorescence calibration workflow complete! Results saved to {output}")
        return output
    except KeyboardInterrupt:
        lash.logger.warning("Workflow interrupted by keyboard; attempting cleanup")
        cleanup_after_keyboard_interrupt(lash)
        raise


if __name__ == "__main__":
    execute()

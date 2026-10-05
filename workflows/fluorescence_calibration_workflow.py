"""Constant-solvent fluorescence standards in water, surfactant and solvent.

Run from the repository root: python -m workflows.fluorescence_calibration_workflow
Defaults generate a hardware-free plan; edit the generated workflow config to run.
Concentrations are relative to the parent dye stock unless its concentration is set.
"""

import sys
sys.path.append("../utoronto_demo")

from datetime import datetime
from pathlib import Path
import logging
import math
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
INPUT_VIAL_STATUS_FILE = "status/fluorescence_calibration_vials.csv"
DYE = "coumarin-6"
DYE_SOLVENT = "DMSO"
STOCK_CONCENTRATION_UM = None
DILUTION_FACTORS = [0.0, 0.1, 0.25, 0.5, 1.0]
DYE_VOLUME_UL = 5.0
TOTAL_VOLUME_UL = 200.0
SUBSTOCK_VOLUME_ML = 6.0
REPLICATES = 3
REPETITIONS = 1
FRESH_SUBSTOCKS = False
MEASUREMENT_REPLICATES = 1
# Elapsed minutes (since the plate is ready) at which to read it; must start at 0.
MEASUREMENT_SCHEDULE_MIN = [0, 10, 30, 60, 120, 240, 360]
# Flat floor, not derived from MEASUREMENT_REPLICATES; widen by hand if needed.
MIN_SCHEDULE_GAP_MIN = 5
RANDOMIZED_ORDER = False
RANDOMIZATION_SEED = 42
WELLPLATE_TYPE = "96 WELL PLATE"
DISPENSE_ORDER = ["medium", "dye"]
# Which base media to prepare wells for, e.g. skip solvent-only wells for a volatile DYE_SOLVENT like acetone.
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

# Names of the constants above; used to snapshot them into a plain config dict.
_CONFIG_KEYS = [
    "SIMULATE", "INPUT_VIAL_STATUS_FILE", "DYE", "DYE_SOLVENT",
    "STOCK_CONCENTRATION_UM", "DILUTION_FACTORS", "DYE_VOLUME_UL",
    "TOTAL_VOLUME_UL", "SUBSTOCK_VOLUME_ML", "REPLICATES", "REPETITIONS",
    "FRESH_SUBSTOCKS", "MEASUREMENT_REPLICATES", "MEASUREMENT_SCHEDULE_MIN",
    "MIN_SCHEDULE_GAP_MIN", "RANDOMIZED_ORDER",
    "RANDOMIZATION_SEED", "WELLPLATE_TYPE", "DISPENSE_ORDER",
    "TEST_WATER", "TEST_SURFACTANT", "TEST_SOLVENT",
    "SURFACTANT_NAME", "SURFACTANT_CONCENTRATION_MM", "SURFACTANT_CMC_MM",
    "SURFACTANT_LIQUID", "VORTEX_SECONDS", "PROTOCOL_FILE", "RAW_CHANNELS",
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
_PLATES = {"96 WELL PLATE": (96, 12), "48 WELL PLATE": (48, 8)}
_STOCK_CONCENTRATIONS_UM = {"pyrene": 48.6, "nile-red": 50.0, "coumarin-6": 3.0}


def build_plan(config):
    """Return well map and direct-from-parent substock recipes, without hardware."""
    c = config
    if c["DYE"].lower() not in _PROTOCOLS:
        raise ValueError("DYE must be pyrene, coumarin-6 or nile-red")
    if c["WELLPLATE_TYPE"] not in _PLATES:
        raise ValueError("Only configured 48/96-well plates are supported; 24 needs robot geometry")
    for key in ("REPLICATES", "REPETITIONS", "MEASUREMENT_REPLICATES"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("DYE_VOLUME_UL", "TOTAL_VOLUME_UL", "SUBSTOCK_VOLUME_ML", "VORTEX_SECONDS"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
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
    if not 0 < fraction < 1:
        raise ValueError("Dye addition must be smaller than total well volume")
    for key in ("STOCK_CONCENTRATION_UM", "SURFACTANT_CONCENTRATION_MM", "SURFACTANT_CMC_MM"):
        if c[key] is not None and (not math.isfinite(c[key]) or c[key] <= 0):
            raise ValueError(f"{key} must be positive or null")
    surf = c["SURFACTANT_CONCENTRATION_MM"]
    cmc = c["SURFACTANT_CMC_MM"]
    if surf is not None and cmc is not None and surf * (1 - fraction) <= cmc:
        raise ValueError("Final surfactant concentration must remain above the configured CMC")
    stock_concentration = c["STOCK_CONCENTRATION_UM"]
    if stock_concentration is None:
        stock_concentration = _STOCK_CONCENTRATIONS_UM[c["DYE"].lower()]
    capacity, columns = _PLATES[c["WELLPLATE_TYPE"]]
    rng = np.random.default_rng(c["RANDOMIZATION_SEED"])
    media = [medium for medium, enabled in (
        ("water", c["TEST_WATER"]), ("surfactant", c["TEST_SURFACTANT"]), ("solvent", c["TEST_SOLVENT"])
    ) if enabled]
    if not media:
        raise ValueError("At least one of TEST_WATER, TEST_SURFACTANT, TEST_SOLVENT must be true")
    rows, recipes = [], []
    sets = c["REPETITIONS"] if c["FRESH_SUBSTOCKS"] else 1
    for batch in range(1, sets + 1):
        for level, factor in enumerate(factors):
            if 0 < factor < 1:
                recipes.append(dict(batch=batch, vial=f"dye_b{batch}_s{level}",
                                    stock_ml=c["SUBSTOCK_VOLUME_ML"] * factor,
                                    solvent_ml=c["SUBSTOCK_VOLUME_ML"] * (1 - factor)))
    plate_offset = 0
    for repetition in range(1, c["REPETITIONS"] + 1):
        batch = repetition if c["FRESH_SUBSTOCKS"] else 1
        block = []
        for medium in media:
            for level, factor in enumerate(factors):
                source = "solvent" if factor == 0 else (
                    "dye_stock" if factor == 1 else f"dye_b{batch}_s{level}")
                for replicate in range(1, c["REPLICATES"] + 1):
                    block.append(dict(repetition=repetition, substock_batch=batch,
                        replicate=replicate, medium=medium, dye_source=source,
                        dye=c["DYE"], dye_solvent=c["DYE_SOLVENT"],
                        dilution_factor=factor, concentration_relative=factor * fraction,
                        concentration_um=stock_concentration * factor * fraction,
                        dye_volume_ul=c["DYE_VOLUME_UL"],
                        medium_volume_ul=c["TOTAL_VOLUME_UL"] - c["DYE_VOLUME_UL"],
                        solvent_fraction=1.0 if medium == "solvent" else fraction,
                        surfactant_final_mm=surf * (1 - fraction) if medium == "surfactant" and surf is not None else None))
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
        required = (plan.dye_source == recipe["vial"]).sum() * c["DYE_VOLUME_UL"] / 1000
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


def execute(config=None):
    """Save a plan; optionally prepare substocks, dispense plates and read fluorescence."""
    if config is None:
        from workflow_config_manager import ConfigManager
        ConfigManager.setup_and_reload_config("fluorescence_calibration_workflow", globals())
        config = {key: globals()[key] for key in _CONFIG_KEYS}
    c = config
    plan, recipes = build_plan(c)
    protocol, channels = _PROTOCOLS[c["DYE"].lower()]
    protocol = c["PROTOCOL_FILE"] or protocol
    channels = c["RAW_CHANNELS"] or channels
    suffix = "_simulate" if c["SIMULATE"] else ""
    output = Path("output") / ("fluorescence_calibration_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + suffix)
    output.mkdir(parents=True)
    plan.to_csv(output / "well_plan.csv", index=False)
    recipes.to_csv(output / "substock_recipes.csv", index=False)
    (output / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    if not protocol or not channels:
        raise ValueError("Set PROTOCOL_FILE/RAW_CHANNELS (or a configured dye protocol and channels) before execution")
    if c["WELLPLATE_TYPE"] != "96 WELL PLATE" and not c["PROTOCOL_FILE"]:
        raise ValueError("Set a fluorescence protocol matching the selected plate format")
    # Protocol files live on the Cytation PC; simulate mode never opens them, so skip the check.
    if not c["SIMULATE"] and not Path(protocol).is_file():
        raise ValueError(f"Set an existing, plate-matched Cytation protocol: {protocol}")
    # Construct Lash_E (and its status-review GUI) before reading inventory, so
    # volumes/locations edited in the GUI are what gets validated below.
    from master_usdl_coordinator import Lash_E, flatten_cytation_data
    lash = Lash_E(c["INPUT_VIAL_STATUS_FILE"], simulate=c["SIMULATE"], show_gui=True,
                  workflow_globals=globals(), workflow_name="fluorescence_calibration_workflow")
    if not c["SIMULATE"]:
        slack_agent.send_slack_message(
            f"Fluorescence calibration workflow started: {c['DYE']} in {', '.join(plan.medium.unique())}")
    inventory = pd.read_csv(c["INPUT_VIAL_STATUS_FILE"]).set_index("vial_name")
    # Stage one 8 mL source at the clamp; large vials remain in place.
    for location, index in (("location", "location_index"), ("home_location", "home_location_index")):
        if ((inventory[location] == "clamp") & (inventory[index] == 0)).any():
            raise ValueError("Keep clamp[0] free for plate-dispensing staging")
    if lash.nr_track.CURRENT_WP_TYPE != c["WELLPLATE_TYPE"]:
        raise ValueError("Track plate type must match configured WELLPLATE_TYPE")
    plate_config = lash.nr_robot.WELLPLATES[c["WELLPLATE_TYPE"]]
    if c["TOTAL_VOLUME_UL"] / 1000 > plate_config["max_volume_per_well"]:
        raise ValueError("Total volume exceeds plate capacity")
    if lash.nr_track.NUM_SOURCE < plan.plate.nunique():
        raise ValueError("Load enough plates for the complete experiment")
    prepared, measurements = set(), []
    for plate, wells in plan.groupby("plate", sort=True):
        batch = int(wells.substock_batch.iloc[0])
        if batch not in prepared:
            for recipe in recipes[recipes.batch == batch].itertuples():
                target_ml = recipe.solvent_ml + recipe.stock_ml
                # Resuming an interrupted run: a substock already filled to its
                # target volume was prepared before the crash, so skip it.
                if inventory.loc[recipe.vial, "vial_volume"] + 1.0 >= target_ml:
                    lash.logger.info(f"Skipping already-prepared substock {recipe.vial}")
                    continue
                for source, volume in (("solvent", recipe.solvent_ml), ("dye_stock", recipe.stock_ml)):
                    lash.nr_robot.dispense_from_vial_into_vial(source, recipe.vial, volume,
                                                             liquid=c["DYE_SOLVENT"])
                lash.nr_robot.vortex_vial(recipe.vial, c["VORTEX_SECONDS"])
                lash.nr_robot.return_vial_home(recipe.vial)
            prepared.add(batch)
        lash.grab_new_wellplate()
        for component in c["DISPENSE_ORDER"]:
            source_column = "medium" if component == "medium" else "dye_source"
            volume_column = f"{component}_volume_ul"
            for source, subset in wells.groupby(source_column):
                liquid = (c["DYE_SOLVENT"] if component == "dye" or source == "solvent"
                          else c["SURFACTANT_LIQUID"] if source == "surfactant" else "water")
                volumes = pd.DataFrame({source: subset.set_index("well_index")[volume_column] / 1000})
                if inventory.loc[source, "vial_type"] == "8_mL":
                    lash.nr_robot.remove_pipet()
                    if lash.nr_robot.get_vial_in_location("clamp", 0) is not None:
                        raise RuntimeError("Plate-dispensing staging clamp[0] is occupied")
                    lash.nr_robot.move_vial_to_location(source, "clamp", 0)
                # Serial dispensing removes its tip and returns the source home.
                # Large vials are aspirated in place, with no staging move.
                lash.nr_robot.dispense_from_vials_into_wellplate(
                    volumes, liquid=liquid, strategy="serial", well_plate_type=c["WELLPLATE_TYPE"])
        # Shake/wait removed; measure as-prepared, then again per MEASUREMENT_SCHEDULE_MIN.
        t0 = time.monotonic()
        for timepoint_min in c["MEASUREMENT_SCHEDULE_MIN"]:
            target = t0 + timepoint_min * 60
            wait = target - time.monotonic()
            if wait > 0 and not lash.simulate:
                time.sleep(wait)
            for read in range(1, c["MEASUREMENT_REPLICATES"] + 1):
                raw = lash.measure_wellplate(protocol, wells.well_index.tolist(), plate_type=c["WELLPLATE_TYPE"])
                if raw is None:
                    if not lash.simulate:
                        raise RuntimeError(f"No fluorescence data for plate {plate}, t={timepoint_min}min, read {read}")
                    data = simulate_fluorescence_readout(wells, channels)
                else:
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
                pd.concat(measurements).to_csv(output / "fluorescence_results.csv", index=False)
            if not c["SIMULATE"]:
                slack_agent.send_slack_message(f"Fluorescence calibration: plate {plate}, t={timepoint_min} min measured")
        lash.discard_used_wellplate()
    combined = pd.concat(measurements)
    summary = summarize(combined, channels)
    effective_stock = c["STOCK_CONCENTRATION_UM"] or _STOCK_CONCENTRATIONS_UM[c["DYE"].lower()]
    summary["concentration_um"] = summary["concentration_relative"] * effective_stock
    summary.to_csv(output / "calibration_summary.csv", index=False)
    fit_calibration(summary, channels, effective_stock).to_csv(
        output / "calibration_fits.csv", index=False)
    wells = per_well_means(combined, channels)
    wells["concentration_um"] = wells["concentration_relative"] * effective_stock
    plot_calibration(wells, summary, channels, output)
    plot_kinetics(wells, channels, output)
    lash.logger.info(f"Calibration measurements saved to {output}")
    if not c["SIMULATE"]:
        slack_agent.send_slack_message(f"Fluorescence calibration workflow complete! Results saved to {output}")
    return output


if __name__ == "__main__":
    execute()

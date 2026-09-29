"""Constant-solvent fluorescence standards in water, surfactant and solvent.

Run from the repository root: python -m workflows.fluorescence_calibration_workflow
Defaults generate a hardware-free plan; edit the generated workflow config to run.
Concentrations are relative to the parent dye stock unless its concentration is set.
"""

from datetime import datetime
from pathlib import Path
import math

import numpy as np
import pandas as pd
import yaml


DEFAULTS = {
    "SIMULATE": True,
    "INPUT_VIAL_STATUS_FILE": "status/fluorescence_calibration_vials.csv",
    "DYE": "pyrene",
    "DYE_SOLVENT": "DMSO",
    "STOCK_CONCENTRATION_UM": None,
    "DILUTION_FACTORS": [0.0, 0.1, 0.25, 0.5, 1.0],
    "DYE_VOLUME_UL": 5.0,
    "TOTAL_VOLUME_UL": 200.0,
    "SUBSTOCK_VOLUME_ML": 6.0,
    "REPLICATES": 3,
    "REPETITIONS": 1,
    "FRESH_SUBSTOCKS": False,
    "MEASUREMENT_REPLICATES": 1,
    "RANDOMIZED_ORDER": False,
    "RANDOMIZATION_SEED": 42,
    "WELLPLATE_TYPE": "96 WELL PLATE",
    "DISPENSE_ORDER": ["medium", "dye"],
    "SURFACTANT_NAME": "SDS",
    "SURFACTANT_CONCENTRATION_MM": None,
    "SURFACTANT_CMC_MM": None,
    "SURFACTANT_LIQUID": "water",
    "VORTEX_SECONDS": 5,
    "PROTOCOL_FILE": None,
    "RAW_CHANNELS": None,
    "SHAKE_PROTOCOL_FILE": r"C:\Protocols\shake_5_wait_5.prt",
}

# Pyrene settings match surfactant_grid_ailsa, pending Cytation verification of
# the proposed 355/373/383 settings. Other dyes remain placeholders.
PROTOCOLS = {
    "pyrene": (r"C:\Protocols\CMC_Fluorescence_96.prt", ["334_373", "334_384"]),
    "coumarin-6": (r"C:\Protocols\Coumarin_96.prt", ["485_530"]),
    "nile-red": (None, ["550_648"]),
}
PLATES = {"96 WELL PLATE": (96, 12), "48 WELL PLATE": (48, 8)}
STOCK_CONCENTRATIONS_UM = {"pyrene": 48.6, "nile-red": 50.0, "coumarin-6": 3.0}


def build_plan(config):
    """Return well map and direct-from-parent substock recipes, without hardware."""
    c = config
    if c["DYE"].lower() not in PROTOCOLS:
        raise ValueError("DYE must be pyrene, coumarin-6 or nile-red")
    if c["WELLPLATE_TYPE"] not in PLATES:
        raise ValueError("Only configured 48/96-well plates are supported; 24 needs robot geometry")
    for key in ("REPLICATES", "REPETITIONS", "MEASUREMENT_REPLICATES"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("DYE_VOLUME_UL", "TOTAL_VOLUME_UL", "SUBSTOCK_VOLUME_ML", "VORTEX_SECONDS"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
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
        stock_concentration = STOCK_CONCENTRATIONS_UM[c["DYE"].lower()]
    capacity, columns = PLATES[c["WELLPLATE_TYPE"]]
    rng = np.random.default_rng(c["RANDOMIZATION_SEED"])
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
        for medium in ("water", "surfactant", "solvent"):
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


def summarize(results, channels):
    """Average repeat reads per well, then summarize independent preparation wells."""
    keys = ["repetition", "medium", "concentration_relative"]
    wells = results.groupby(keys + ["plate", "well_index"], as_index=False)[channels].mean()
    summary = wells.groupby(keys)[channels].agg(["mean", "std", "count"])
    summary.columns = [f"{channel}_{stat}" for channel, stat in summary.columns]
    summary = summary.reset_index()
    for channel in channels:
        blanks = summary[summary.concentration_relative == 0].set_index(
            ["repetition", "medium"])[f"{channel}_mean"]
        summary[f"{channel}_blank_corrected"] = [
            row[f"{channel}_mean"] - blanks.loc[(row.repetition, row.medium)]
            for _, row in summary.iterrows()]
    return summary


def fit_calibration(summary, channels, stock_concentration_um=None):
    """Export unconstrained linear fits for review, separately for each medium/run.

    Fits do not assert linearity or quantify encapsulation. Review the range,
    residuals and blank variability before using concentration=(signal-intercept)/slope.
    """
    fits = []
    for (repetition, medium), group in summary.groupby(["repetition", "medium"]):
        x = group.concentration_relative.to_numpy()
        for channel in channels:
            y = group[f"{channel}_blank_corrected"].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            residual = y - (slope * x + intercept)
            total = np.sum((y - y.mean()) ** 2)
            fits.append(dict(repetition=repetition, medium=medium, channel=channel,
                slope_per_relative=slope, intercept_blank_corrected=intercept,
                slope_per_um=None if stock_concentration_um is None else slope / stock_concentration_um,
                r_squared=None if total == 0 else 1 - np.sum(residual ** 2) / total,
                rmse=float(np.sqrt(np.mean(residual ** 2))),
                min_concentration_relative=float(x.min()), max_concentration_relative=float(x.max()),
                status="review_linearity_before_use" if slope > 0 else "nonpositive_slope_do_not_invert"))
    return pd.DataFrame(fits)


def plot_calibration(summary, channels, output_dir):
    """Save intensity-versus-concentration plots, one panel per medium/channel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x_column = "concentration_um" if "concentration_um" in summary else "concentration_relative"
    for medium, group in summary.groupby("medium"):
        for channel in channels:
            fig, ax = plt.subplots(figsize=(6, 4))
            y = f"{channel}_mean"
            error = f"{channel}_std"
            ax.errorbar(group[x_column], group[y], yerr=group[error], marker="o", linestyle="-", capsize=3)
            ax.set(title=f"{channel}: {medium}", xlabel="Dye concentration (uM)" if x_column == "concentration_um" else "Relative dye concentration", ylabel="Fluorescence intensity")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(Path(output_dir) / f"intensity_vs_concentration_{medium}_{channel}.png", dpi=180)
            plt.close(fig)


def execute(config=None):
    """Save a plan; optionally prepare substocks, dispense plates and read fluorescence."""
    if config is None:
        from workflow_config_manager import ConfigManager
        config = ConfigManager.get_or_create_config("fluorescence_calibration_workflow", DEFAULTS)
    c = {**DEFAULTS, **config}
    plan, recipes = build_plan(c)
    protocol, channels = PROTOCOLS[c["DYE"].lower()]
    protocol = c["PROTOCOL_FILE"] or protocol
    channels = c["RAW_CHANNELS"] or channels
    output = Path("output") / ("fluorescence_calibration_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True)
    plan.to_csv(output / "well_plan.csv", index=False)
    recipes.to_csv(output / "substock_recipes.csv", index=False)
    (output / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    if c["SIMULATE"]:
        print(f"Plan only: {len(plan)} wells on {plan.plate.nunique()} plates. Saved to {output}")
        return output
    if not protocol or not channels:
        raise ValueError("Set PROTOCOL_FILE/RAW_CHANNELS (or a configured dye protocol and channels) before execution")
    if c["WELLPLATE_TYPE"] != "96 WELL PLATE" and not c["PROTOCOL_FILE"]:
        raise ValueError("Set a fluorescence protocol matching the selected plate format")
    for path in (protocol, c["SHAKE_PROTOCOL_FILE"]):
        if not path or not Path(path).is_file():
            raise ValueError(f"Set an existing, plate-matched Cytation protocol: {path}")
    inventory = pd.read_csv(c["INPUT_VIAL_STATUS_FILE"]).set_index("vial_name")
    # Stage one 8 mL source at the clamp; large vials remain in place.
    for location, index in (("location", "location_index"), ("home_location", "home_location_index")):
        if ((inventory[location] == "clamp") & (inventory[index] == 0)).any():
            raise ValueError("Keep clamp[0] free for plate-dispensing staging")
    requirements = {"dye_stock": recipes.stock_ml.sum(), "solvent": recipes.solvent_ml.sum()}
    for row in plan.itertuples():
        for source, volume in ((row.medium, row.medium_volume_ul), (row.dye_source, row.dye_volume_ul)):
            requirements[source] = requirements.get(source, 0) + volume / 1000
    for source, required in requirements.items():
        if source not in inventory.index:
            raise ValueError(f"Missing vial {source} in input status")
        available = inventory.loc[source, "vial_volume"]
        if source in set(recipes.vial):
            if available != 0:
                raise ValueError(f"Substock destination {source} must start empty")
            if c["SUBSTOCK_VOLUME_ML"] > 7:
                raise ValueError("Substocks must fit the specified 8 mL destination vials")
        elif not math.isfinite(available) or available < required + 0.1:
            raise ValueError(f"{source} needs at least {required + 0.1:.3f} mL including reserve")
    from master_usdl_coordinator import Lash_E, flatten_cytation_data
    lash = Lash_E(c["INPUT_VIAL_STATUS_FILE"], simulate=False)
    lash.nr_robot.check_input_file()
    lash.nr_track.check_input_file()
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
        lash.measure_wellplate(c["SHAKE_PROTOCOL_FILE"], wells.well_index.tolist(),
                              plate_type=c["WELLPLATE_TYPE"])
        for read in range(1, c["MEASUREMENT_REPLICATES"] + 1):
            raw = lash.measure_wellplate(protocol, wells.well_index.tolist(), plate_type=c["WELLPLATE_TYPE"])
            if raw is None:
                raise RuntimeError(f"No fluorescence data for plate {plate}, read {read}")
            raw.to_csv(output / f"raw_plate_{plate}_read_{read}.csv")
            data = flatten_cytation_data(raw, "fluorescence")
            if not set(["well_position", *channels]).issubset(data.columns):
                raise ValueError(f"Unexpected Cytation channels: {list(data.columns)}")
            data["well_position"] = data.well_position.map(lambda p: f"{str(p)[0].upper()}{int(str(p)[1:])}")
            merged = wells.merge(data[["well_position", *channels]], on="well_position",
                                 how="left", validate="one_to_one")
            merged[channels] = merged[channels].apply(pd.to_numeric, errors="raise")
            if not np.isfinite(merged[channels].to_numpy()).all():
                raise ValueError("Missing, saturated or non-finite fluorescence readings")
            merged["measurement_replicate"] = read
            measurements.append(merged)
            pd.concat(measurements).to_csv(output / "fluorescence_results.csv", index=False)
        lash.discard_used_wellplate()
    summary = summarize(pd.concat(measurements), channels)
    effective_stock = c["STOCK_CONCENTRATION_UM"] or STOCK_CONCENTRATIONS_UM[c["DYE"].lower()]
    summary["concentration_um"] = summary["concentration_relative"] * effective_stock
    summary.to_csv(output / "calibration_summary.csv", index=False)
    fit_calibration(summary, channels, effective_stock).to_csv(
        output / "calibration_fits.csv", index=False)
    plot_calibration(summary, channels, output)
    print(f"Calibration measurements saved to {output}")
    return output


if __name__ == "__main__":
    execute()

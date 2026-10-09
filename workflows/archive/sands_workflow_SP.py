"""
Oil Sands Extraction Workflow
==============================

Goal:
    Optimize aqueous extraction efficiency of oil from sand/clay material
    using Bayesian optimization.

Procedure (one experiment):
    1. Add organic mix (solvent + pyrene + oil) to a sand/clay vial.
    2. Heat vial to evaporate the organic solvent, leaving oil+pyrene on sand.
    3. Cool the vial.
    4. Add aqueous extraction solution and mix.
    5. Aspirate aqueous phase into a wellplate well.
    6. Measure UV-Vis absorbance AND fluorescence (Cytation 5).
    7. Auto-dilute if signal is saturated.
    8. Report result to the Bayesian optimizer.
    9. Repeat with new optimizer-suggested conditions.

Optimization parameters (flexible - configure in SEARCH_SPACE_PARAMS below):
    - Aqueous solution volume (mL)
    - [Add more parameters as the experiment design matures]

Optimization objective:
    - Maximize extraction efficiency as measured by fluorescence (pyrene signal)
      and/or UV-Vis absorbance.
"""

import sys
sys.path.append("../utoronto_demo")

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from master_usdl_coordinator import Lash_E

# ================================================================================
# CONFIGURATION
# ================================================================================

SIMULATE = True

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/oilsands_vials.csv"

# Cytation 5 protocol files (update paths when protocols are finalized)
ABSORBANCE_PROTOCOL_FILE = r"C:\Protocols\oilsands_absorbance.prt"
FLUORESCENCE_PROTOCOL_FILE = r"C:\Protocols\oilsands_fluorescence.prt"

# Heater settings
EVAPORATION_TEMP_C = 80          # Temperature to evaporate organic solvent
EVAPORATION_TIME_S = 600         # Seconds to hold at temperature (10 min)
COOLING_TIME_S = 120             # Seconds to allow cooling before adding aqueous

# Wellplate settings
WELLPLATE_DISPENSE_VOLUME_ML = 0.200   # Volume dispensed per well
REPLICATES = 3                         # Wells per condition

# Auto-dilution settings
ABS_SATURATION_THRESHOLD = 3.5   # Absorbance above this triggers dilution
FLUOR_SATURATION_THRESHOLD = 5e5 # Fluorescence counts above this triggers dilution
MAX_DILUTION_FACTOR = 10         # Maximum auto-dilution factor

# Output directory
OUTPUT_DIR = Path("output/oilsands_extraction")

# Bayesian optimization loop
BO_ROUNDS = 5            # Total optimization rounds after initial batch
INITIAL_BATCH_SIZE = 3   # Number of initial random experiments before BO starts

# ================================================================================
# BAYESIAN OPTIMIZATION SEARCH SPACE
# ================================================================================
# Edit this dict to add or remove optimization parameters.
# Each key is the parameter name; values define the discrete search values (mL).
# BayBe will suggest combinations from this space.

SEARCH_SPACE_PARAMS = {
    "aqueous_volume_mL": list(np.round(np.arange(0.5, 3.5, 0.5), 2)),  # 0.5 to 3.0 mL
    # Add more parameters here as needed, e.g.:
    # "agitation_time_s": [30, 60, 120, 300],
    # "temperature_C": [25, 40, 60],
}

# ================================================================================
# HELPER: Initialize BayBe campaign
# ================================================================================

def _init_baybe_campaign(search_space_params: dict):
    """
    Build and return a BayBe Campaign targeting maximization of extraction_yield.

    Args:
        search_space_params: dict of {param_name: list_of_values}

    Returns:
        baybe.Campaign
    """
    from baybe import Campaign
    from baybe.targets import NumericalTarget, TargetMode
    from baybe.objectives import SingleTargetObjective
    from baybe.parameters import NumericalDiscreteParameter
    from baybe.searchspace import SearchSpace

    target = NumericalTarget(
        name="extraction_yield",
        mode=TargetMode.MAX,
    )
    objective = SingleTargetObjective(target)

    parameters = [
        NumericalDiscreteParameter(name=name, values=values)
        for name, values in search_space_params.items()
    ]
    searchspace = SearchSpace.from_product(parameters=parameters)
    return Campaign(searchspace, objective)


# ================================================================================
# HELPER: Single extraction experiment
# ================================================================================

def run_single_extraction(
    lash_e,
    params: dict,
    sand_vial: str,
    organic_mix_vial: str,
    aqueous_stock_vial: str,
    water_vial: str,
    extraction_vial: str,
    well_start_index: int,
    replicates: int = REPLICATES,
    wellplate_dispense_volume_ml: float = WELLPLATE_DISPENSE_VOLUME_ML,
) -> dict:
    """
    Run one extraction experiment with the given parameter set.

    Steps:
        1. Deposit organic mix onto sand vial.
        2. Heat to evaporate solvent, then cool.
        3. Add aqueous solution and mix.
        4. Aspirate aqueous phase to wellplate.
        5. Measure absorbance and fluorescence, auto-diluting if saturated.

    Args:
        lash_e: Initialized Lash_E coordinator.
        params: Dict of optimization parameters (must include aqueous_volume_mL).
        sand_vial: Name of the sand/clay source vial.
        organic_mix_vial: Name of vial containing the organic mixture.
        aqueous_stock_vial: Name of the aqueous extraction stock vial.
        water_vial: Name of the water vial (for dilution).
        extraction_vial: Name of the working extraction vial.
        well_start_index: First wellplate well index to use.
        replicates: Number of replicate wells.
        wellplate_dispense_volume_ml: Volume dispensed per well.

    Returns:
        dict with keys: 'abs_mean', 'fluor_mean', 'wells_used', 'dilution_factor'
    """
    aqueous_volume_ml = params["aqueous_volume_mL"]

    lash_e.logger.info(f"Starting extraction: params={params}, well_start={well_start_index}")

    # ------------------------------------------------------------------
    # STEP 1: Add organic mix to the sand vial
    # ------------------------------------------------------------------
    lash_e.logger.info("Step 1: Depositing organic mix onto sand vial")
    lash_e.nr_robot.move_vial_to_location(
        vial_name=organic_mix_vial, location="clamp", location_index=0
    )
    lash_e.nr_robot.aspirate_from_vial(organic_mix_vial, 0.5, liquid="ethanol")
    lash_e.nr_robot.move_vial_to_location(
        vial_name=sand_vial, location="clamp", location_index=0
    )
    lash_e.nr_robot.dispense_into_vial(sand_vial, 0.5)
    lash_e.nr_robot.remove_pipet()

    # ------------------------------------------------------------------
    # STEP 2: Heat to evaporate organic solvent, then cool
    # ------------------------------------------------------------------
    lash_e.logger.info(
        f"Step 2: Heating sand vial to {EVAPORATION_TEMP_C}C for {EVAPORATION_TIME_S}s"
    )
    lash_e.nr_robot.move_vial_to_location(
        vial_name=sand_vial, location="heater", location_index=0
    )
    lash_e.temp_controller.set_temp(EVAPORATION_TEMP_C)
    lash_e.temp_controller.turn_on_stirring()
    time.sleep(EVAPORATION_TIME_S)
    lash_e.temp_controller.turn_off_stirring()
    lash_e.logger.info(f"Cooling for {COOLING_TIME_S}s")
    time.sleep(COOLING_TIME_S)
    lash_e.nr_robot.move_vial_to_location(
        vial_name=sand_vial, location="clamp", location_index=0
    )

    # ------------------------------------------------------------------
    # STEP 3: Add aqueous solution and mix
    # ------------------------------------------------------------------
    lash_e.logger.info(
        f"Step 3: Adding {aqueous_volume_ml} mL aqueous solution to sand vial"
    )
    lash_e.nr_robot.dispense_from_vial_into_vial(
        source_vial_name=aqueous_stock_vial,
        dest_vial_name=sand_vial,
        volume=aqueous_volume_ml,
    )
    lash_e.nr_robot.vortex_vial(vial_name=sand_vial, vortex_time=30)

    # ------------------------------------------------------------------
    # STEP 4: Transfer aqueous phase to extraction vial
    # ------------------------------------------------------------------
    lash_e.logger.info("Step 4: Transferring aqueous phase to extraction vial")
    # Transfer total volume needed for all replicates + buffer
    transfer_volume = wellplate_dispense_volume_ml * replicates + 0.1
    lash_e.nr_robot.dispense_from_vial_into_vial(
        source_vial_name=sand_vial,
        dest_vial_name=extraction_vial,
        volume=transfer_volume,
    )

    # ------------------------------------------------------------------
    # STEP 5: Dispense to wellplate and measure
    # ------------------------------------------------------------------
    wells = list(range(well_start_index, well_start_index + replicates))
    dilution_factor = 1

    lash_e.logger.info(f"Step 5: Dispensing to wells {wells}")
    for well in wells:
        lash_e.nr_robot.aspirate_from_vial(extraction_vial, wellplate_dispense_volume_ml, liquid="water")
        lash_e.nr_robot.dispense_into_wellplate([well], [wellplate_dispense_volume_ml], liquid="water")
        lash_e.nr_robot.remove_pipet()

    # Measure absorbance
    lash_e.logger.info("Measuring UV-Vis absorbance")
    abs_data = lash_e.nr_track.read_wellplate(ABSORBANCE_PROTOCOL_FILE)

    # Measure fluorescence
    lash_e.logger.info("Measuring fluorescence")
    fluor_data = lash_e.nr_track.read_wellplate(FLUORESCENCE_PROTOCOL_FILE)

    # ------------------------------------------------------------------
    # Auto-dilution if signal is saturated
    # ------------------------------------------------------------------
    # [STUB] Parse abs_data and fluor_data for well values.
    # The exact keys depend on the Cytation protocol output format.
    # Replace the lines below once protocol output format is confirmed.
    abs_values = np.array([1.0] * replicates)   # TODO: extract from abs_data
    fluor_values = np.array([1e4] * replicates)  # TODO: extract from fluor_data

    abs_mean = float(np.mean(abs_values))
    fluor_mean = float(np.mean(fluor_values))

    if abs_mean > ABS_SATURATION_THRESHOLD or fluor_mean > FLUOR_SATURATION_THRESHOLD:
        dilution_factor = MAX_DILUTION_FACTOR
        lash_e.logger.info(
            f"Signal saturated (abs={abs_mean:.3f}, fluor={fluor_mean:.1f}) - "
            f"diluting x{dilution_factor} and re-measuring"
        )
        diluted_wells = list(range(well_start_index + replicates, well_start_index + 2 * replicates))
        diluted_volume = wellplate_dispense_volume_ml / dilution_factor
        water_volume = wellplate_dispense_volume_ml - diluted_volume

        for well in diluted_wells:
            lash_e.nr_robot.aspirate_from_vial(extraction_vial, diluted_volume, liquid="water")
            lash_e.nr_robot.dispense_into_wellplate([well], [diluted_volume], liquid="water")
            lash_e.nr_robot.remove_pipet()
            lash_e.nr_robot.aspirate_from_vial(water_vial, water_volume, liquid="water")
            lash_e.nr_robot.dispense_into_wellplate([well], [water_volume], liquid="water")
            lash_e.nr_robot.remove_pipet()

        abs_data_diluted = lash_e.nr_track.read_wellplate(ABSORBANCE_PROTOCOL_FILE)
        fluor_data_diluted = lash_e.nr_track.read_wellplate(FLUORESCENCE_PROTOCOL_FILE)

        # TODO: extract diluted values from abs_data_diluted / fluor_data_diluted
        abs_mean = abs_mean / dilution_factor
        fluor_mean = fluor_mean / dilution_factor

    lash_e.logger.info(
        f"Result: abs_mean={abs_mean:.4f}, fluor_mean={fluor_mean:.1f}, "
        f"dilution_factor={dilution_factor}"
    )

    return {
        "abs_mean": abs_mean,
        "fluor_mean": fluor_mean,
        "wells_used": wells,
        "dilution_factor": dilution_factor,
    }


# ================================================================================
# MAIN WORKFLOW: Bayesian optimization loop
# ================================================================================

def oilsands_extraction_workflow(
    lash_e,
    bo_rounds: int = BO_ROUNDS,
    initial_batch_size: int = INITIAL_BATCH_SIZE,
    replicates: int = REPLICATES,
    search_space_params: dict = None,
):
    """
    Closed-loop Bayesian optimization of oil extraction from sand/clay.

    Args:
        lash_e: Initialized Lash_E coordinator.
        bo_rounds: Number of Bayesian optimization rounds after the initial batch.
        initial_batch_size: Random experiments before BO starts.
        replicates: Wellplate replicates per condition.
        search_space_params: Override the module-level SEARCH_SPACE_PARAMS dict.
    """
    if search_space_params is None:
        search_space_params = SEARCH_SPACE_PARAMS

    # Vial names (must match oilsands_vials.csv)
    sand_vial = "sand_vial"
    organic_mix_vial = "organic_mix_vial"
    aqueous_stock_vial = "aqueous_stock"
    water_vial = "water_vial"
    extraction_vial = "extraction_vial"

    # Output setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"

    lash_e.logger.info(f"Oil sands extraction workflow started. Output: {output_dir}")

    # Initialize BayBe campaign
    campaign = _init_baybe_campaign(search_space_params)

    # Grab a fresh wellplate
    lash_e.grab_new_wellplate()

    # Track all results
    all_results = []
    well_cursor = 0
    total_rounds = 1 + bo_rounds  # initial batch counts as round 1

    for round_num in range(1, total_rounds + 1):
        batch_size = initial_batch_size if round_num == 1 else 1
        lash_e.logger.info(f"Round {round_num}/{total_rounds}: requesting {batch_size} suggestion(s)")

        suggestions = campaign.recommend(batch_size=batch_size)

        for _, row in suggestions.iterrows():
            params = row.to_dict()
            lash_e.logger.info(f"  Condition: {params}")

            result = run_single_extraction(
                lash_e=lash_e,
                params=params,
                sand_vial=sand_vial,
                organic_mix_vial=organic_mix_vial,
                aqueous_stock_vial=aqueous_stock_vial,
                water_vial=water_vial,
                extraction_vial=extraction_vial,
                well_start_index=well_cursor,
                replicates=replicates,
            )

            # Use fluorescence as extraction_yield (primary BO objective)
            extraction_yield = result["fluor_mean"] * result["dilution_factor"]

            record = {**params, "extraction_yield": extraction_yield, **result, "round": round_num}
            all_results.append(record)
            well_cursor += replicates * (2 if result["dilution_factor"] > 1 else 1)

        # Add measurements to BayBe campaign after each round
        measurements = pd.DataFrame(all_results)[list(search_space_params.keys()) + ["extraction_yield"]]
        # BayBe expects only new measurements since the last add; pass the full frame
        # (BayBe deduplicates internally via index tracking)
        campaign.add_measurements(measurements)

        # Save incremental results
        pd.DataFrame(all_results).to_csv(results_path, index=False)
        lash_e.logger.info(f"Round {round_num} complete. Results saved to {results_path}")

    lash_e.logger.info("Oil sands extraction workflow complete.")
    return pd.DataFrame(all_results)


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    lash_e = Lash_E(
        INPUT_VIAL_STATUS_FILE,
        initialize_t8=True,
        initialize_p2=False,
        simulate=SIMULATE,
    )

    oilsands_extraction_workflow(lash_e)

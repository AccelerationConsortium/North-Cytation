"""
Heating Mantle Position Test

Tests pipetting at each heating mantle position (2, 5, 8, 4).
Step 1: Move all vials to their heater positions.
Step 2: Grab one small tip, then aspirate/dispense 10 times per vial across
        all positions without changing the tip.
Step 3: Remove the tip.
Step 4: Return all vials to the 8mL rack.

Vials  : V0, V1, V2, V3 (open top, ~2 mL each)
Heater positions: 2, 5, 8, 4
"""

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

# ── Configuration ─────────────────────────────────────────────────────────────
VIAL_STATUS_FILE = "status/heating_mantle_vials.csv"
SIMULATE = False

# Each tuple: (vial_name, heater_position_index)
VIAL_HEATER_MAP = [
    #("V0", 2),
    #("V1", 5),
    #("V2", 8),
    ("V3", 11),
]

MIX_VOLUME_ML = 0.150   # volume per aspirate/dispense cycle (mL)
MIX_REPEATS   = 10   # number of aspirate-dispense cycles per vial
# ──────────────────────────────────────────────────────────────────────────────


def heating_mantle_mixing_test(simulate: bool = SIMULATE):
    lash_e = Lash_E(
        vial_file=VIAL_STATUS_FILE,
        initialize_biotek=False,
        simulate=simulate,
    )

    # Home robot before starting workflow
    lash_e.logger.info("Homing robot components.")
    lash_e.nr_robot.home_robot_components()

    try:
        # Step 1: Move all vials to their heater positions
        lash_e.logger.info("Step 1: Moving all vials to heater positions.")
        for vial_name, heater_pos in VIAL_HEATER_MAP:
            lash_e.logger.info("Moving %s to heater position %d", vial_name, heater_pos)
            lash_e.nr_robot.move_vial_to_location(vial_name, "heater", heater_pos)

        # Step 2: Grab one small tip, then pipette each vial 10 times
        lash_e.logger.info("Step 2: Grabbing small tip and pipetting all positions.")
        lash_e.nr_robot.get_pipet("small_tip")

        for vial_name, heater_pos in VIAL_HEATER_MAP:
            lash_e.logger.info(
                "Pipetting %s: %d cycles x %.0fuL at heater position %d",
                vial_name, MIX_REPEATS, MIX_VOLUME_ML * 1000, heater_pos,
            )
            for _ in range(MIX_REPEATS):
                lash_e.nr_robot.aspirate_from_vial(vial_name, MIX_VOLUME_ML, specified_tip="small_tip")
                lash_e.nr_robot.dispense_into_vial(vial_name, MIX_VOLUME_ML)

        # Step 3: Remove the tip
        lash_e.logger.info("Step 3: Removing tip.")
        lash_e.nr_robot.remove_pipet()

        # Step 4: Return all vials back to their original 8mL rack positions
        lash_e.logger.info("Step 4: Returning all vials to 8mL rack.")
        for vial_name, _ in VIAL_HEATER_MAP:
            lash_e.nr_robot.return_vial_home(vial_name)

        lash_e.logger.info("Heating mantle position test complete.")

    except Exception:
        lash_e.logger.exception("Workflow interrupted - running cleanup.")
        if lash_e.nr_robot.HELD_PIPET_TYPE is not None:
            lash_e.logger.info("Cleanup: removing pipet tip.")
            lash_e.nr_robot.remove_pipet()
        lash_e.logger.info("Cleanup: returning all vials to home positions.")
        for vial_name, _ in VIAL_HEATER_MAP:
            lash_e.nr_robot.return_vial_home(vial_name)
        raise


heating_mantle_mixing_test(simulate=SIMULATE)

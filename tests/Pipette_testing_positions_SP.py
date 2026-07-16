"""Safe pipette position tester for water vial.

Mirrors the safe-position placement context used by the surfactant
multidimensional workflow (surfactant_multidimensional_workflow.py).

Goal: physically verify the robot can reach the tip depth that corresponds
to a vial containing SIMULATED_VIAL_VOLUME_ML of liquid, at each safe
pipetting position -- without pumping any liquid.

Sequence for each test position:
1. Move water vial to the test position.
2. Get the tip specified by TIP_TYPE ("large_tip" or "small_tip").
3. Temporarily set the tracked vial volume to SIMULATED_VIAL_VOLUME_ML so
   get_aspirate_height computes the correct tip depth for that fill level.
4. Move the robot arm to the vial's XY coordinate.
5. Lower the tip to the computed aspirate height (no pump commands).
6. Retract to safe height.
7. Restore the original tracked vial volume.
8. Remove the pipet tip.
9. Return water vial to its home location.

Configuration:
- Edit SIMULATE, VIAL_FILE, TIP_TYPE, SIMULATED_VIAL_VOLUME_ML, and TEST_POSITIONS at the top.
- Set HOME_BETWEEN_POSITIONS = True to run full robot homing between each cycle.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from master_usdl_coordinator import Lash_E

# ============================================================
# CONFIGURATION  -- edit these values as needed
# ============================================================

SIMULATE = False
VIAL_FILE = str(REPO_ROOT / "status" / "surfactant_multidim_vials.csv")

# Vial to test with
WATER_VIAL = "water"

# Tip type to use: "large_tip" or "small_tip"
TIP_TYPE = "small_tip"

# The robot will descend to the height that corresponds to this volume
# remaining in the vial. Does NOT pump any liquid.
SIMULATED_VIAL_VOLUME_ML = 1.0

# Safe pipetting positions used by the surfactant workflow.
# Each entry is (location_name, location_index).
# Mirrors the safe_positions list in position_surfactant_vials_by_concentration.
TEST_POSITIONS = [
    ("main_8mL_rack", 47),
    ("main_8mL_rack", 46),
    ("main_8mL_rack", 45),
    ("main_8mL_rack", 44),
    ("clamp",          0),
    ("main_8mL_rack", 43),
]

# Set True to run full robot homing between each position cycle
HOME_BETWEEN_POSITIONS = False

# ============================================================


def describe_position(location, index):
    if location == "clamp":
        return "clamp[0]"
    return f"{location}[{index}]"


def run_pipette_position_test():
    lash_e = Lash_E(
        vial_file=VIAL_FILE,
        simulate=SIMULATE,
        initialize_biotek=False,
    )

    robot = lash_e.nr_robot
    logger = lash_e.logger
    logger.info("Starting safe pipette position test")
    logger.info(f"Simulation mode: {SIMULATE}")
    logger.info(f"Vial: {WATER_VIAL} | Simulated fill level: {SIMULATED_VIAL_VOLUME_ML * 1000:.0f}uL | Tip: {TIP_TYPE}")
    logger.info(f"Positions to test: {len(TEST_POSITIONS)}")

    for idx, (location, location_index) in enumerate(TEST_POSITIONS):
        pos_label = describe_position(location, location_index)
        logger.info("")
        logger.info(f"--- Position {idx + 1}/{len(TEST_POSITIONS)}: {pos_label} ---")

        # Step 1: Move water vial to the test position
        logger.info(f"Step 1: Moving {WATER_VIAL} -> {pos_label}")
        robot.move_vial_to_location(WATER_VIAL, location, location_index)

        # Step 2: Get the configured tip type
        logger.info(f"Step 2: Getting {TIP_TYPE}")
        robot.get_pipet(TIP_TYPE)

        # Step 3: Temporarily override tracked vial volume so get_aspirate_height
        # computes the depth corresponding to SIMULATED_VIAL_VOLUME_ML of liquid.
        vial_index = robot.normalize_vial_index(WATER_VIAL)
        original_volume = robot.get_vial_info(vial_index, 'vial_volume')
        robot.VIAL_DF.at[vial_index, 'vial_volume'] = SIMULATED_VIAL_VOLUME_ML
        logger.info(
            f"Step 3: Temporarily set vial volume to {SIMULATED_VIAL_VOLUME_ML * 1000:.0f}uL "
            f"(was {float(original_volume) * 1000:.0f}uL)"
        )

        # Step 4 & 5: Compute aspirate height and move to that position -- no pumping.
        vial_location = robot.get_location(True, location, location_index)
        asp_height = robot.get_aspirate_height(vial_index, 0.0, track_height=True)
        asp_height = robot.adjust_height_based_on_pipet_held(asp_height)
        logger.info(f"Step 4: Moving to XY of {pos_label} then lowering to aspirate height ({asp_height})")
        robot.c9.goto_xy_safe(vial_location, vel=robot.get_speed('standard_xy'))
        robot.c9.move_z(asp_height)

        # Step 6: Retract to safe height
        safe_height = robot.get_safe_height()
        logger.info(f"Step 5: Retracting to safe height ({safe_height})")
        robot.c9.move_z(safe_height, vel=robot.get_speed('retract'))

        # Step 7: Restore original tracked vial volume
        robot.VIAL_DF.at[vial_index, 'vial_volume'] = original_volume
        logger.info(f"Step 6: Restored vial volume to {float(original_volume) * 1000:.0f}uL")

        # Step 8: Remove pipet tip
        logger.info("Step 7: Removing pipet tip")
        robot.remove_pipet()

        # Step 9: Return water vial home
        logger.info(f"Step 8: Returning {WATER_VIAL} to home")
        robot.return_vial_home(WATER_VIAL)

        # Optional homing between positions
        if HOME_BETWEEN_POSITIONS and idx < len(TEST_POSITIONS) - 1:
            logger.info("Running full robot homing before next position")
            robot.home_robot_components()

    logger.info("")
    logger.info("Safe pipette position test complete")


if __name__ == "__main__":
    run_pipette_position_test()

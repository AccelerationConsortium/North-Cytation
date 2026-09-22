"""
Test workflow: moves vial_a to each "Safe" pipetting position (the _SAFE_SMALL_TIP
map defined in North_Safe.py's aspirate_from_vial) and pipets an alternating volume
from vial_a into stationary vial_b at each stop, then returns vial_a home.

vial_a: open cap (capped=True, cap_type=open) - movable by the gripper, but never
        needs uncapping since cap_type=open makes it pipetable while capped.
vial_b: closed cap (capped=True, cap_type=closed) - stays in main_8mL_rack index 1;
        the pipetting routine automatically moves it to the clamp, uncaps it,
        dispenses, recaps, and returns it home each time.

Author: North Robotics Team
"""
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
from robot_state.Locator import vial_clamp_cap
import time

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/safe_position_test_vials.csv"
SIMULATE = False

VIAL_A = "vial_a"
VIAL_B = "vial_b"

# Mirrors the _SAFE_SMALL_TIP map in North_Safe.py's aspirate_from_vial
SAFE_POSITIONS = [("main_8mL_rack", i) for i in range(43, 48)] + [
    ("clamp", 0),
    ("photoreactor_array", 0),
    ("heater", 2),
]

VOLUMES_ML = [0.05, 0.25]

# Half decap: on every dispense into vial_b, after the normal automatic uncap +
# dispense, we move to vial_clamp_cap ourselves and spin the cap backward by
# HALF_DECAP_REVS (a raw c9.uncap() call) - a "thread-finding" half-decap that
# reduces the risk of cross-threading - immediately before calling the normal
# recap_clamp_vial(), which re-arrives at vial_clamp_cap (a no-op move since
# we're already there) and does the real c9.cap() torque-tightening call.
# This only uses existing public methods of North_Safe/NorthC9 - no changes to
# North_Safe.py or any other script are required.
# Set RUN_HALF_DECAP_STEP = False to skip the extra rotation step entirely.
RUN_HALF_DECAP_STEP = False
HALF_DECAP_REVS = 0.75

# Set to False to run straight through without stopping for Enter key presses.
ENABLE_PAUSES = False


def pause(message: str):
    """Pause for Enter key press, but only if ENABLE_PAUSES is True."""
    if ENABLE_PAUSES:
        input(message)


def safe_position_pipetting_test(simulate: bool = SIMULATE):
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, initialize_biotek=False, simulate=simulate, show_gui=True)

    for i, (location, location_index) in enumerate(SAFE_POSITIONS):
        volume = VOLUMES_ML[i % 2]
        lash_e.logger.info(
            f"Step {i + 1}/{len(SAFE_POSITIONS)}: moving {VIAL_A} to {location}[{location_index}], "
            f"pipetting {volume} mL into {VIAL_B}"
        )

        lash_e.nr_robot.move_vial_to_location(VIAL_A, location, location_index)
        pause(f"Moved {VIAL_A} to {location}[{location_index}]. Press Enter to pipet...")

        lash_e.nr_robot.dispense_from_vial_into_vial(VIAL_A, VIAL_B, volume, return_vial_home=False)
        pause(f"Pipetted {volume} mL into {VIAL_B}. Press Enter to continue...")

        if RUN_HALF_DECAP_STEP:
            lash_e.logger.info(
                f"Half decap: moving to vial_clamp_cap and rotating cap {HALF_DECAP_REVS} revs before capping {VIAL_B}"
            )
            lash_e.nr_robot.c9.close_clamp()
            lash_e.nr_robot.goto_location_if_not_there(vial_clamp_cap)
            if not simulate:
                time.sleep(0.5)
            lash_e.nr_robot.c9.uncap(revs=HALF_DECAP_REVS)
            pause("Half decap complete. Press Enter to cap...")

        lash_e.nr_robot.recap_clamp_vial()
        pause(f"Capped {VIAL_B}. Press Enter to return vials home...")

        lash_e.nr_robot.return_vial_home(VIAL_B)
        lash_e.nr_robot.return_vial_home(VIAL_A)
        pause(f"Step {i + 1}/{len(SAFE_POSITIONS)} complete. Press Enter for next step...")

    pause("Pausing...")

    lash_e.logger.info("Safe position pipetting test complete.")


if __name__ == "__main__":
    safe_position_pipetting_test()

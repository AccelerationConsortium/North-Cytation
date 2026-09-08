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

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/safe_position_test_vials.csv"
SIMULATE = True

VIAL_A = "vial_a"
VIAL_B = "vial_b"

# Mirrors the _SAFE_SMALL_TIP map in North_Safe.py's aspirate_from_vial
SAFE_POSITIONS = [("main_8mL_rack", i) for i in range(43, 48)] + [
    ("clamp", 0),
    ("photoreactor_array", 0),
    ("heater", 2),
]

VOLUMES_ML = [0.05, 0.25]


def safe_position_pipetting_test(simulate: bool = SIMULATE):
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, initialize_biotek=False, simulate=simulate, show_gui=False)

    for i, (location, location_index) in enumerate(SAFE_POSITIONS):
        volume = VOLUMES_ML[i % 2]
        lash_e.logger.info(
            f"Step {i + 1}/{len(SAFE_POSITIONS)}: moving {VIAL_A} to {location}[{location_index}], "
            f"pipetting {volume} mL into {VIAL_B}"
        )

        lash_e.nr_robot.move_vial_to_location(VIAL_A, location, location_index)
        lash_e.nr_robot.dispense_from_vial_into_vial(VIAL_A, VIAL_B, volume)
        lash_e.nr_robot.return_vial_home(VIAL_A)

    lash_e.logger.info("Safe position pipetting test complete.")


if __name__ == "__main__":
    safe_position_pipetting_test()

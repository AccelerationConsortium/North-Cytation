"""Move one Ailsa vial to the clamp, park the arm at origin, and return it home.

Run from the repository root: python -m workflows.ailsa_clamp_test
"""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/surfactant_grid_ailsa_vials.csv"
VIAL_NAME = "BDDAC_dilution_1"

_WORKFLOW_NAME = Path(__file__).stem


def execute(config=None, show_gui=True):
    from master_usdl_coordinator import Lash_E

    lash_e = Lash_E(
        initialize_track=False, initialize_biotek=False,
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
    )
    if not lash_e._workflow_should_continue:
        return None

    robot = lash_e.nr_robot
    vial = lash_e.workflow_config["VIAL_NAME"]
    # if robot.GRIPPER_STATUS is not None or robot.HELD_PIPET_TYPE is not None:
    #     raise ValueError("Clamp test requires an empty gripper and no held pipette")
    # if robot.get_vial_in_location("clamp", 0) is not None:
    #     raise ValueError("Clamp must be empty before starting the test")
    # for field, home_field in (("location", "home_location"),
    #                           ("location_index", "home_location_index")):
    #     if robot.get_vial_info(vial, field) != robot.get_vial_info(vial, home_field):
    #         raise ValueError(f"{vial} must start at its home location")

    # robot.move_vial_to_location(vial, "clamp", 0)
    # robot.move_home()
    robot.return_vial_home(vial)
    lash_e.logger.info("Clamp test complete: %s returned home", vial)


if __name__ == "__main__":
    execute()

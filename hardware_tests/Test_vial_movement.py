"""Move a water vial through the 8 mL rack.

Run from the repository root: python hardware_tests/Test_vial_movement.py
The startup GUI defaults to simulation. This movement-only test visits each
position in the 8 mL rack (except 42), then returns the vial to its recorded
home. Use the dedicated one-vial inventory and ensure every other rack
position is physically empty before a live run. The test does not uncap or
recap the vial.
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from master_usdl_coordinator import Lash_E

SIMULATE = False
INPUT_VIAL_STATUS_FILE = str(_REPO_ROOT / "status" / "test_vial_movement_vials.csv")

_WORKFLOW_NAME = Path(__file__).stem
_CONFIG_KEYS = ["SIMULATE", "INPUT_VIAL_STATUS_FILE"]
_VIAL_NAME = "water"
_RACK = "main_8mL_rack"
_RACK_SIZE = 48
_EXCLUDED_POSITIONS = {42}


def _validate_and_get_route(robot):
    rack = robot.VIAL_POSITIONS[_RACK]
    if not rack["rack_present"] or rack["rack_size"] != _RACK_SIZE:
        raise ValueError("This test requires the 48-position main_8mL_rack.")
    if len(robot.VIAL_DF) != 1:
        raise ValueError("The test vial inventory must contain only the water vial.")
    if robot.GRIPPER_STATUS is not None or robot.HELD_PIPET_TYPE is not None:
        raise ValueError("Start with an empty gripper and no pipette tip.")
    if robot.get_vial_in_location("clamp", 0) is not None:
        raise ValueError("The clamp must be empty before starting.")

    vial_index = robot.normalize_vial_index(_VIAL_NAME)
    home = robot.get_vial_info(vial_index, "home_location")
    home_index = robot.get_vial_info(vial_index, "home_location_index")
    if (not robot.is_vial_movable(_VIAL_NAME)
            or home != _RACK
            or robot.get_vial_info(vial_index, "location") != home
            or robot.get_vial_info(vial_index, "location_index") != home_index):
        raise ValueError("The water vial must be capped at its recorded rack home.")

    route = [
        position for position in range(_RACK_SIZE)
        if position != home_index and position not in _EXCLUDED_POSITIONS
    ]
    return home_index, route


def execute(config=None, show_gui=True):
    lash = Lash_E(
        initialize_track=False, initialize_biotek=False,
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
    )
    if not lash._workflow_should_continue:
        return None

    robot = lash.nr_robot
    home_index, route = _validate_and_get_route(robot)
    lash.logger.info(
        "Water-vial movement test: visiting %s empty rack positions; simulate=%s",
        len(route), lash.simulate,
    )
    lash.logger.warning(
        "This test inventory contains only the water vial. Confirm every other "
        "rack position is physically empty before a live run."
    )

    current_position = home_index
    try:
        for destination in route:
            lash.logger.info(
                "Moving %s in %s: main_8mL_rack[%s] -> main_8mL_rack[%s]",
                _VIAL_NAME, _RACK, current_position, destination,
            )
            robot.move_vial_to_location(_VIAL_NAME, _RACK, destination)
            current_position = destination
        robot.move_vial_to_location(_VIAL_NAME, _RACK, home_index)
        lash.logger.info(
            "Movement test completed; %s returned to main_8mL_rack[%s].",
            _VIAL_NAME, home_index,
        )
        return len(route)
    except (Exception, KeyboardInterrupt):
        lash.logger.exception(
            "Movement test stopped at main_8mL_rack[%s]; no recovery movement attempted.",
            current_position,
        )
        raise


if __name__ == "__main__":
    execute()
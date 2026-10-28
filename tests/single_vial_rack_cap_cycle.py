"""Cycle one closed-cap 8 mL vial through the rack and auxiliary stations.

Run from the repository root: python tests/single_vial_rack_cap_cycle.py
Start with only one capped vial at main_8mL_rack[0], all other visited
positions empty, an empty gripper and no tip. Reconcile robot_status.yaml first.
The dedicated inventory describes an empty vial; no liquid is transferred.

At each slot: move vial to clamp -> uncap -> move arm home holding cap ->
recap -> move vial to next position -> move arm home. Visit rack slots 0-47
except 42, then photoreactor_array[0], clamp_2[0], and every configured heater
slot (currently 0-11), then return to rack slot 0. Only the main clamp is used
for uncapping/recapping; no heating or illumination is enabled.
MAX_CYCLES counts full sweeps (currently 61 cap cycles); 0 repeats until Ctrl+C.
Defaults to one simulated sweep with startup GUI review. No automatic recovery
motion follows an error or interruption; the saved state records progress.
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from master_usdl_coordinator import Lash_E

SIMULATE = True
INPUT_VIAL_STATUS_FILE = str(_REPO_ROOT / "status" / "single_vial_rack_cap_cycle_vials.csv")
MAX_CYCLES = 1

_WORKFLOW_NAME = Path(__file__).stem
_CONFIG_KEYS = ["SIMULATE", "INPUT_VIAL_STATUS_FILE", "MAX_CYCLES"]
_RACK = "main_8mL_rack"
_POSITIONS = tuple(position for position in range(48) if position != 42)


def _build_route(robot):
    route = [(_RACK, position) for position in _POSITIONS]
    for location in ("photoreactor_array", "clamp_2", "heater"):
        station = robot.VIAL_POSITIONS.get(location, {})
        size = station.get("rack_size")
        if (not station.get("rack_present") or not station.get("vials_movable")
                or not station.get("vial_positions_in_Locator")
                or type(size) is not int or size < 1):
            raise ValueError(f"{location} must be present with configured movable vial positions.")
        # The photoreactor and secondary clamp each have a single test position.
        route.extend((location, position) for position in range(size if location == "heater" else 1))
    return tuple(route)


def _validate_vial(robot):
    if len(robot.VIAL_DF) != 1:
        raise ValueError("Inventory must contain exactly one vial; all other visited positions must be empty.")
    vial = robot.VIAL_DF.iloc[0]
    if (vial["location"] != _RACK or vial["location_index"] != 0
            or vial["home_location"] != _RACK or vial["home_location_index"] != 0
            or vial["vial_type"] != "8_mL" or vial["cap_type"] != "closed"
            or not vial["capped"]):
        raise ValueError("The single vial must start capped with a closed cap at main_8mL_rack[0], its home.")
    rack = robot.VIAL_POSITIONS[_RACK]
    if not rack["rack_present"] or rack["rack_size"] != 48:
        raise ValueError("This test requires the 48-position main_8mL_rack to be present.")
    if robot.GRIPPER_STATUS is not None or robot.HELD_PIPET_TYPE is not None:
        raise ValueError("Start with an empty gripper and no pipette tip.")
    if robot.get_vial_in_location("clamp", 0) is not None:
        raise ValueError("Start with an empty clamp.")
    return vial["vial_name"]


def execute(config=None, show_gui=True):
    lash = Lash_E(
        initialize_track=False, initialize_biotek=False,
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
    )
    if not lash._workflow_should_continue:
        return None
    robot = lash.nr_robot
    limit = lash.workflow_config["MAX_CYCLES"]
    if type(limit) is not int or limit < 0:
        raise ValueError("MAX_CYCLES must be a nonnegative integer (0 means forever).")
    vial = _validate_vial(robot)
    route = _build_route(robot)
    completed = 0
    phase = "move_home"
    try:
        robot.move_home()
        while limit == 0 or completed < limit:
            for source, destination in zip(route, route[1:] + route[:1]):
                lash.logger.info(
                    "Sweep %s: %s[%s] -> clamp -> uncap -> arm home -> recap -> %s[%s] -> arm home",
                    completed + 1, *source, *destination,
                )
                phase = f"slot_{source}:move_to_clamp"
                robot.move_vial_to_location(vial, "clamp", 0)
                phase = f"slot_{source}:uncap"
                robot.uncap_clamp_vial()
                phase = f"slot_{source}:home_with_cap"
                robot.move_home()
                phase = f"slot_{source}:recap"
                robot.recap_clamp_vial()
                phase = f"slot_{source}:move_to_slot_{destination}"
                robot.move_vial_to_location(vial, *destination)
                phase = f"slot_{destination}:move_home"
                robot.move_home()
            completed += 1
        lash.logger.info("Completed %s sweeps; the capped vial is back at rack position 0.", completed)
        return completed
    except (Exception, KeyboardInterrupt):
        lash.logger.exception("Single-vial rack test stopped at %s; no automatic recovery motion attempted.", phase)
        raise


if __name__ == "__main__":
    execute()

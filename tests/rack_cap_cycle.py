"""Uncap and recap the 8 mL rack vials listed in the selected inventory.

Run: python tests/rack_cap_cycle.py
The startup GUI selects simulation/live mode and MAX_CYCLES. One cycle is
a sweep of the listed vials in rack-position order. MAX_CYCLES=0
repeats until Ctrl+C. Defaults to one simulated sweep, with no pipetting.
The supplied vial file has closed-cap 8 mL vials containing 5 mL each.
Keep the clamp empty at startup.
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from master_usdl_coordinator import Lash_E
from workflows.interrupt_cleanup import cleanup_after_keyboard_interrupt

SIMULATE = True
INPUT_VIAL_STATUS_FILE = str(_REPO_ROOT / "status" / "rack_cap_cycle_vials.csv")
MAX_CYCLES = 1

_WORKFLOW_NAME = Path(__file__).stem
_CONFIG_KEYS = ["SIMULATE", "INPUT_VIAL_STATUS_FILE", "MAX_CYCLES"]
_RACK_POSITIONS = tuple(range(31))


def _validate_vials(robot):
    """Validate listed vials; unlisted rack positions may be empty."""
    vials = {}
    for _, row in robot.VIAL_DF.iterrows():
        position = row["home_location_index"]
        if (row["home_location"] != "main_8mL_rack"
                or position not in _RACK_POSITIONS or position in vials):
            raise ValueError("Vials must have unique homes in main_8mL_rack positions 0-30 inclusive.")
        if (row["location"] != "main_8mL_rack" or row["location_index"] != position
                or row["vial_type"] != "8_mL" or row["cap_type"] != "closed" or not row["capped"]):
            raise ValueError(f"Vial {row['vial_name']} must start at home as a capped, closed-cap 8 mL vial.")
        vials[position] = row["vial_name"]
    if not vials:
        raise ValueError("No vials listed in the selected rack inventory.")
    if robot.get_vial_in_location("clamp", 0) is not None:
        raise ValueError("The clamp must be empty before starting.")
    if robot.HELD_PIPET_TYPE is not None:
        raise ValueError("Remove the held pipette tip before starting this movement-only test.")
    return vials


def execute(config=None, show_gui=True):
    """Visit each accessible rack vial and repeat for the configured sweeps."""
    lash_e = Lash_E(
        initialize_biotek=False,
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
    )
    if not lash_e._workflow_should_continue:
        return None
    robot = lash_e.nr_robot
    limit = lash_e.workflow_config["MAX_CYCLES"]
    if type(limit) is not int or limit < 0:
        raise ValueError("MAX_CYCLES must be a nonnegative integer (0 means forever).")
    vials = _validate_vials(robot)
    robot.home_robot_components()
    completed = 0
    try:
        while limit == 0 or completed < limit:
            for position in sorted(vials):
                name = vials[position]
                lash_e.logger.info(
                    "Sweep %s: rack position %s (%s) -> clamp -> uncap -> recap -> home",
                    completed + 1, position, name,
                )
                robot.move_vial_to_location(name, "clamp", 0)
                robot.uncap_clamp_vial()
                robot.move_home()
                robot.recap_clamp_vial()
                robot.return_vial_home(name)
            completed += 1
        robot.move_home()
        lash_e.logger.info("Completed %s rack cap sweeps (%s vials per sweep).", completed, len(vials))
        return completed
    except (Exception, KeyboardInterrupt):
        lash_e.logger.exception("Rack cap test stopped after %s complete sweeps; attempting cleanup", completed)
        cleanup_after_keyboard_interrupt(lash_e)
        raise


if __name__ == "__main__":
    execute()

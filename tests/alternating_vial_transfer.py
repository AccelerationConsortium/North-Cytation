"""Repeatedly stage two capped vials, transfer liquid, and return them home.

Run from the repository root: python tests/alternating_vial_transfer.py
The startup GUI selects simulation/live mode and transfer settings. Defaults
use simulation, 100 uL of water per cycle, and repeat until Ctrl+C. Set
MAX_CYCLES to a positive integer for a finite run (0 means repeat forever).

Load two supported 8 mL vials, each containing 5 mL, with closed caps at
main_8mL_rack positions 9 and 10. Keep staging positions 0 and 1 empty.
Each cycle stages vial_a at 0 and vial_b at 1, transfers once, then returns
both home. Odd cycles transfer A -> B; even cycles transfer B -> A.
Closed caps are handled automatically by the robot's pipetting routine.
"""

import math
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from master_usdl_coordinator import Lash_E
from workflows.interrupt_cleanup import cleanup_after_keyboard_interrupt

SIMULATE = True
INPUT_VIAL_STATUS_FILE = str(_REPO_ROOT / "status" / "alternating_vial_transfer_vials.csv")
TRANSFER_VOLUME_ML = 0.1
LIQUID = "water"
MAX_CYCLES = 4

_WORKFLOW_NAME = Path(__file__).stem
_CONFIG_KEYS = [
    "SIMULATE", "INPUT_VIAL_STATUS_FILE", "TRANSFER_VOLUME_ML", "LIQUID", "MAX_CYCLES",
]
_VIAL_NAMES = ("vial_a", "vial_b")


def execute(config=None, show_gui=True):
    """Run with reviewed settings; Ctrl+C uses the shared workflow cleanup."""
    lash_e = Lash_E(
        initialize_biotek=False,
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
    )
    if not lash_e._workflow_should_continue:
        return None

    robot = lash_e.nr_robot
    c = lash_e.workflow_config
    completed = 0
    original_homes = []
    try:
        volume = c["TRANSFER_VOLUME_ML"]
        if isinstance(volume, bool) or not isinstance(volume, (int, float)) or not math.isfinite(volume) or not 0 < volume <= 3:
            raise ValueError("TRANSFER_VOLUME_ML must be positive and at most 3 mL (8 mL vial capacity).")
        limit = c["MAX_CYCLES"]
        if type(limit) is not int or limit < 0:
            raise ValueError("MAX_CYCLES must be a nonnegative integer (0 means forever).")

        for name in _VIAL_NAMES:
            original_homes.append((
                robot.get_vial_info(name, "vial_index"),
                robot.get_vial_info(name, "home_location"),
                robot.get_vial_info(name, "home_location_index"),
            ))

        lash_e.logger.info("Alternating vial transfers started; press Ctrl+C to stop.")
        while limit == 0 or completed < limit:
            source, destination = _VIAL_NAMES if completed % 2 == 0 else _VIAL_NAMES[::-1]
            lash_e.logger.info(
                "Cycle %s: %s -> %s, %.3f mL", completed + 1, source, destination, volume,
            )
            for position, (index, _, _) in enumerate(original_homes):
                occupant = robot.get_vial_in_location("main_8mL_rack", position)
                if occupant is not None and occupant != index:
                    raise RuntimeError(f"Vial-transfer staging main_8mL_rack[{position}] is occupied")
            robot.remove_pipet()
            # Match the fluorescence/degradation staging pattern: automatic
            # clamp exchanges use HOME, not the vial's last rack position.
            for position, (index, _, _) in enumerate(original_homes):
                robot.VIAL_DF.at[index, "home_location"] = "main_8mL_rack"
                robot.VIAL_DF.at[index, "home_location_index"] = position
            for position, name in enumerate(_VIAL_NAMES):
                robot.move_vial_to_location(name, "main_8mL_rack", position)
            robot.dispense_from_vial_into_vial(
                source_vial_name=source,
                dest_vial_name=destination,
                volume=volume,
                liquid=c["LIQUID"],
                remove_tip=True,
                return_vial_home=False,
            )
            # Clear the clamp (recapping the destination) before restoring the
            # real homes. No tip may remain for movements to positions 9/10.
            robot.remove_pipet()
            for name in (destination, source):
                robot.return_vial_home(name)
            for index, home, position in original_homes:
                robot.VIAL_DF.at[index, "home_location"] = home
                robot.VIAL_DF.at[index, "home_location_index"] = position
            for name in _VIAL_NAMES:
                robot.return_vial_home(name)
            completed += 1

        robot.move_home()
        return completed
    except KeyboardInterrupt:
        lash_e.logger.warning(
            "Interrupted after %s completed cycles; attempting workflow cleanup", completed,
        )
        cleanup_after_keyboard_interrupt(lash_e)
        raise
    except Exception:
        lash_e.logger.exception("Vial transfer failed; attempting workflow cleanup")
        cleanup_after_keyboard_interrupt(lash_e)
        raise
    finally:
        # Keep safe homes active THROUGH interrupt cleanup, since returning
        # leftover liquid can itself trigger a clamp exchange with a tip held.
        for index, home, position in original_homes:
            robot.VIAL_DF.at[index, "home_location"] = home
            robot.VIAL_DF.at[index, "home_location_index"] = position
        if original_homes:
            robot.save_robot_status()


if __name__ == "__main__":
    execute()

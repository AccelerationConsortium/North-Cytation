r"""Supervised replay of TTAB_dilution_1's 2026-11-04 vortex/return sequence.

Run from the repository root:
    python hardware_tests\vortex_return_replay.py

Defaults to simulation and uses normal startup GUI review. Before selecting live
mode, recover the controller, reconcile recorded and physical vial/gripper state,
and ensure the clamp is empty and the test vial is at its recorded rack home.
Lash_E initialization itself can move hardware. Use an expendable vial.

No water or surfactant is added. Vortexing uses the original duration and speed.
RETURN_HOME=True drops the vial directly at its recorded home after vortexing.
False preserves clamp release followed immediately by a separate return pickup.
Uses current production release heights, pickup thresholds, and gripper commands.
This can expose an existing missed pickup; software state is not grip feedback.
No next vial, final home movement, or error recovery movement is included.
"""

import sys
sys.path.append("../utoronto_demo")

from pathlib import Path

from master_usdl_coordinator import Lash_E


SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/surfactant_multidim_vials.csv"
TARGET_VIAL = "TTAB_dilution_1"
VORTEX_TIME = 8
VORTEX_SPEED = 80
RETURN_HOME = False

_WORKFLOW_NAME = Path(__file__).stem
_CONFIG_KEYS = [
    "SIMULATE", "INPUT_VIAL_STATUS_FILE", "TARGET_VIAL",
    "VORTEX_TIME", "VORTEX_SPEED", "RETURN_HOME",
]
_CONFIG_DOCS = {
    "SIMULATE": {"description": "Run without controlling hardware.", "unit": None, "affects": []},
    "INPUT_VIAL_STATUS_FILE": {
        "description": "CSV of reconciled physical vial locations.",
        "unit": None, "affects": ["vials"], "shares": "vial_layout",
    },
    "TARGET_VIAL": {
        "description": "Expendable vial starting at its recorded main-rack home.",
        "unit": None, "affects": ["vials"],
    },
    "VORTEX_TIME": {
        "description": "Vortex duration; incident used 8 seconds.",
        "unit": "s", "affects": ["duration"],
    },
    "VORTEX_SPEED": {
        "description": "Controller vortex speed; incident used 80.",
        "unit": None, "affects": ["duration"],
    },
    "RETURN_HOME": {
        "description": "Return directly home after vortexing when true; false releases at clamp then regrips to return home.",
        "unit": None, "affects": ["duration", "vials"],
    },
}
_PRODUCT_VIALS = []


def execute(config=None, show_gui=True):
    lash_e = Lash_E(
        initialize_track=False, initialize_biotek=False,
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
    )
    if not lash_e._workflow_should_continue:
        return

    robot = lash_e.nr_robot
    logger = lash_e.logger
    c = lash_e.workflow_config
    phase = "preflight"
    try:
        if c["INPUT_VIAL_STATUS_FILE"] is None:
            raise ValueError("Vortex replay requires a vial CSV.")
        if type(c["RETURN_HOME"]) is not bool:
            raise ValueError("RETURN_HOME must be a boolean.")
        vial = c["TARGET_VIAL"]
        if not isinstance(vial, str) or not vial.strip():
            raise ValueError("TARGET_VIAL must be a nonempty vial name.")
        for key in ("VORTEX_TIME", "VORTEX_SPEED"):
            if type(c[key]) not in (int, float) or not 0 < c[key] < float("inf"):
                raise ValueError(f"{key} must be finite and positive.")
        if robot.GRIPPER_STATUS is not None or robot.HELD_PIPET_TYPE is not None:
            raise ValueError("Replay requires an empty recorded gripper and no pipet tip.")
        if robot.get_vial_in_location("clamp", 0) is not None:
            raise ValueError("Replay requires an empty recorded and physical clamp.")
        if not robot.is_vial_movable(vial):
            raise ValueError("Replay requires a movable 8 mL vial.")
        home = robot.get_vial_info(vial, "home_location")
        home_index = robot.get_vial_info(vial, "home_location_index")
        if (home != "main_8mL_rack"
                or robot.get_vial_info(vial, "location") != home
                or robot.get_vial_info(vial, "location_index") != home_index):
            raise ValueError("Replay vial must start at its recorded main-rack home.")
        occupants = robot.VIAL_DF.loc[
            (robot.VIAL_DF["location"] == home)
            & (robot.VIAL_DF["location_index"] == home_index)
        ]
        if len(occupants) != 1:
            raise ValueError("Replay vial home must have exactly one recorded occupant.")

        logger.info(
            "Vortex replay: vial=%s home=%s[%s] duration=%s speed=%s return_home=%s simulate=%s",
            vial, home, home_index, c["VORTEX_TIME"], c["VORTEX_SPEED"],
            c["RETURN_HOME"], lash_e.simulate,
        )
        logger.warning(
            "No liquids added. Pickup/release behavior unchanged. "
            "Recorded occupancy does not verify physical occupancy or open jaws."
        )
        phase = "place_at_clamp"
        logger.info("STEP: place target vial at clamp")
        robot.move_vial_to_location(vial, "clamp", 0)
        phase = "pre_vortex_home"
        logger.info("STEP: move arm/carousel home, matching the post-water approach")
        robot.move_home()
        phase = "vortex_and_return"
        if c["RETURN_HOME"]:
            logger.info("STEP: vortex, then release directly at recorded vial home")
        else:
            logger.info("STEP: vortex, release at clamp, then immediately return vial home")
        robot.vortex_vial(
            vial, vortex_time=c["VORTEX_TIME"], vortex_speed=c["VORTEX_SPEED"],
            return_location=(home, home_index) if c["RETURN_HOME"] else None,
        )
        if not c["RETURN_HOME"]:
            robot.return_vial_home(vial)
        logger.warning(
            "Replay command sequence ended; no further diagnostic motion. "
            "Visually verify the vial is back in its rack and the clamp is empty."
        )
    except (Exception, KeyboardInterrupt):
        logger.exception("Vortex replay stopped at %s; no recovery motion attempted", phase)
        raise


if __name__ == "__main__":
    execute()

r"""Supervised rack pickup/release diagnostic; run from the repository root.

Run: python tests\rack_vial_roundtrip.py
Defaults to simulation. Review SIMULATE and the vial inventory in startup GUI
before any live run. The CSV determines occupancy; this is not a presence sensor.
The robot must be recovered from any previous fault before starting.
"""

import sys
sys.path.append("../utoronto_demo")

from pathlib import Path

from master_usdl_coordinator import Lash_E

SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/fluorescence_calibration_vials.csv"
_CONFIG_KEYS = ["SIMULATE", "INPUT_VIAL_STATUS_FILE"]
_WORKFLOW_NAME = Path(__file__).stem


def execute(config=None, show_gui=True):
    lash = Lash_E(
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
        initialize_track=False, initialize_biotek=False,
    )
    if not lash._workflow_should_continue:
        return

    robot = lash.nr_robot
    logger = lash.logger
    try:
        vials = robot.VIAL_DF.loc[
            robot.VIAL_DF["location"] == "main_8mL_rack"
        ].copy().sort_values("location_index")
        if vials.empty:
            raise ValueError("No occupied main_8mL_rack positions in the confirmed inventory")
        if not vials["location_index"].is_unique or not vials["vial_name"].is_unique:
            raise ValueError("Rack positions and vial names must be unique")
        rack_size = robot.VIAL_POSITIONS["main_8mL_rack"]["rack_size"]
        for vial in vials.itertuples():
            if not 0 <= vial.location_index < rack_size or int(vial.location_index) != vial.location_index:
                raise ValueError(f"Invalid rack index for {vial.vial_name}: {vial.location_index}")
            if vial.home_location != "main_8mL_rack" or vial.home_location_index != vial.location_index:
                raise ValueError(f"{vial.vial_name} must start at its recorded rack home")
            if not robot.is_vial_movable(vial.vial_name):
                raise ValueError(f"{vial.vial_name} is not a movable capped 8 mL vial")
        if robot.GRIPPER_STATUS is not None or robot.HELD_PIPET_TYPE is not None:
            raise ValueError("Rack test requires an empty gripper and no pipet tip")

        logger.info("Rack roundtrip test: %s occupied positions, simulate=%s",
                    len(vials), lash.simulate)
        logger.warning("Supervise vial release: gripper commands do not confirm physical release")
        robot.move_home()
        robot.home_robot_components()
        for vial in vials.itertuples():
            logger.info("Picking up %s from main_8mL_rack index %s",
                        vial.vial_name, vial.location_index)
            robot.grab_vial(vial.vial_name)
            robot.move_home()
            logger.info("Returning held %s to main_8mL_rack index %s",
                        vial.vial_name, vial.home_location_index)
            # Already holding the vial: return_vial_home would try to grab again.
            robot.drop_off_vial(vial.vial_name, vial.home_location, int(vial.home_location_index))
            robot.move_home()
        logger.info("Rack roundtrip command sequence completed for %s vials", len(vials))
    except (Exception, KeyboardInterrupt):
        logger.exception("Rack test stopped; no automatic recovery motion attempted. Check physical state.")
        raise


if __name__ == "__main__":
    execute()

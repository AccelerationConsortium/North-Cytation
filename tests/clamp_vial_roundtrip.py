r"""Supervised cap/return diagnostic, without liquid transfers or tip removal.

Run from the repository root: python tests\clamp_vial_roundtrip.py
Defaults to simulation with startup GUI review. Recover the controller and
reconcile the physical inventory first: normal Lash_E initialization can move
hardware. Observe the final release; a returned command is not release feedback.
Shared production drop-off sends normal open, output 0 OFF, then lift/wiggle.
Both release commands use the configured settling wait.
Repeat the complete sequence REPETITIONS times, then move to normal home.
No recovery or home movement is attempted after a failure.
"""

import sys
sys.path.append("../utoronto_demo")

import inspect
from unittest.mock import patch

from master_usdl_coordinator import Lash_E

SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/fluorescence_calibration_vials.csv"
VIAL_NAME = "DMSO_rf_1"
REPETITIONS = 3
_CONFIG_KEYS = [
    "SIMULATE", "INPUT_VIAL_STATUS_FILE", "VIAL_NAME",
    "REPETITIONS",
]


def execute(config=None, show_gui=True):
    lash = Lash_E(
        workflow_globals=globals(), workflow_name="clamp_vial_roundtrip",
        config=config, show_gui=show_gui,
        initialize_track=False, initialize_biotek=False,
    )
    if not lash._workflow_should_continue:
        return

    robot = lash.nr_robot
    logger = lash.logger
    vial = lash.workflow_config["VIAL_NAME"]
    phase = "preflight"
    try:
        c = lash.workflow_config
        if type(c["REPETITIONS"]) is not int or c["REPETITIONS"] < 1:
            raise ValueError("REPETITIONS must be a positive integer")
        if robot.GRIPPER_STATUS is not None or robot.HELD_PIPET_TYPE is not None:
            raise ValueError("Clamp test requires an empty gripper and no pipet tip")
        if robot.get_vial_in_location("clamp", 0) is not None:
            raise ValueError("Clamp test requires an empty clamp")
        if not robot.is_vial_movable(vial) or robot.get_vial_info(vial, "cap_type") != "closed":
            raise ValueError("Clamp test requires a movable capped vial with a closed cap")
        home = robot.get_vial_info(vial, "home_location")
        home_index = robot.get_vial_info(vial, "home_location_index")
        if (home != "main_8mL_rack"
                or robot.get_vial_info(vial, "location") != home
                or robot.get_vial_info(vial, "location_index") != home_index):
            raise ValueError("Test vial must start at its recorded main-rack home")
        index = robot.normalize_vial_index(vial)
        occupants = robot.VIAL_DF.loc[
            (robot.VIAL_DF["location"] == home)
            & (robot.VIAL_DF["location_index"] == home_index)
        ]
        if len(occupants) != 1 or robot.get_vial_in_location(home, home_index) != index:
            raise ValueError("Test vial home must have exactly one recorded occupant")

        logger.info("Clamp diagnostic: vial=%s home=%s[%s] simulate=%s",
                    vial, home, home_index, lash.simulate)
        logger.info("Robot implementation: %s", inspect.getfile(type(robot)))
        logger.info("Driver implementation: %s", inspect.getfile(type(robot.c9)))
        logger.info("Gripper timing: %s", robot.ROBOT_HARDWARE["gripper_timing"])
        if lash.simulate:
            logger.warning("Simulation cannot verify gripper packets or physical release")

        original_send_packet = robot.c9.send_packet

        def traced_send_packet(command, *args, **kwargs):
            if command != "GRPR":
                return original_send_packet(command, *args, **kwargs)
            logger.info("GRIPPER TRACE phase=%s sending GRPR args=%r kwargs=%r",
                        phase, args, kwargs)
            try:
                response = original_send_packet(command, *args, **kwargs)
            except BaseException:
                logger.exception("GRIPPER TRACE phase=%s GRPR did not return successfully", phase)
                raise
            logger.info("GRIPPER TRACE phase=%s GRPR returned response=%r", phase, response)
            return response

        # Trace the real driver call unchanged; restore it even on interruption.
        with patch.object(robot.c9, "send_packet", new=traced_send_packet):
            for repetition in range(1, c["REPETITIONS"] + 1):
                logger.info("Clamp diagnostic repetition %s/%s", repetition, c["REPETITIONS"])
                phase = f"repetition_{repetition}:move_to_clamp"
                logger.info("STEP: move %s to clamp", vial)
                robot.move_vial_to_location(vial, "clamp", 0)
                phase = f"repetition_{repetition}:uncap"
                logger.info("STEP: uncap %s", vial)
                robot.uncap_clamp_vial()
                phase = f"repetition_{repetition}:recap"
                logger.info("STEP: recap %s", vial)
                robot.recap_clamp_vial()
                if not lash.simulate:
                    logger.info("Pose after recap: %s", robot.c9.get_robot_positions())
                phase = f"repetition_{repetition}:return_home"
                logger.info("STEP: return %s to %s[%s]", vial, home, home_index)
                robot.return_vial_home(vial)
            phase = "move_home"
            logger.info("STEP: move to normal home after all repetitions")
            robot.move_home()
        logger.warning(
            "Command sequence ended. No further diagnostic motion. "
            "Visually confirm vial release; software state is not physical confirmation."
        )
    except (Exception, KeyboardInterrupt):
        logger.exception("Clamp diagnostic stopped at %s; no recovery motion attempted", phase)
        raise


if __name__ == "__main__":
    execute()

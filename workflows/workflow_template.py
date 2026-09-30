"""
Template for creating new workflows in the North Robotics automation system.
Copy this file and modify it to create your own workflow.

Author: North Robotics Team
Date: {DATE}
"""
import sys
sys.path.append("../utoronto_demo")  # Always first, before any local imports

import logging
import time

from master_usdl_coordinator import Lash_E
from pipetting_data.pipetting_parameters import PipettingParameters

logger = logging.getLogger(__name__)

# Workflow config constants, auto-detected by ConfigManager and persisted to
# workflow_configs/your_workflow_name.yaml (module-level UPPERCASE constants +
# workflow_globals=globals() at Lash_E init). Keep these as module globals, not
# a dict or function-local variables - ConfigManager only picks up the former.
#
# SIMULATE must never gate the automation calls below (no "if SIMULATE: skip").
# Lash_E/North_Robot/North_Track already simulate hardware internally and log
# "SIMULATION MODE: Would pause for error: ..." instead of raising, so running
# the full workflow body with SIMULATE=True is how bugs get caught. Only use
# SIMULATE to skip file/folder creation or substitute synthetic instrument data.
SIMULATE = True  # [REPLACE] Set to False only when running on real hardware
INPUT_VIAL_STATUS_FILE = "status/your_vials.csv"  # [REPLACE] Create your own CSV
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Your_Protocol.prt"  # [REPLACE] Your Cytation protocol
PARAM1 = 0.5  # [REPLACE] Example: volume in mL
PARAM2 = 6    # [REPLACE] Example: number of replicates/wells
TARGET_TEMPERATURE = 25.0  # Celsius
REACTION_TIME = 30         # seconds
VORTEX_TIME = 5            # seconds

_CONFIG_KEYS = [
    "SIMULATE", "INPUT_VIAL_STATUS_FILE", "MEASUREMENT_PROTOCOL_FILE",
    "PARAM1", "PARAM2", "TARGET_TEMPERATURE", "REACTION_TIME", "VORTEX_TIME",
]


def execute(config=None):
    """
    [REPLACE] Brief description of what this workflow does.

    Workflow Steps:
        1. [REPLACE] Initialize system
        2. [REPLACE] Describe each major step
        3. [REPLACE] ...
        4. [REPLACE] Final measurements and cleanup
    """
    if config is None:
        from workflow_config_manager import ConfigManager
        ConfigManager.setup_and_reload_config("your_workflow_name", globals())
        config = {key: globals()[key] for key in _CONFIG_KEYS}
    c = config

    logger.info("Starting your_workflow_name workflow")
    logger.info(f"Parameters: param1={c['PARAM1']}, param2={c['PARAM2']}, simulate={c['SIMULATE']}")

    # Initialize the system - adjust initialization flags as needed.
    # simulate=c["SIMULATE"] runs the full workflow body safely: hardware moves
    # are stubbed out and errors are logged instead of raised, so the ONLY
    # difference from a real run should be "no physical hardware moved".
    lash_e = Lash_E(
        c["INPUT_VIAL_STATUS_FILE"],
        initialize_t8=True,      # Temperature controller
        initialize_p2=True,      # Photoreactor
        simulate=c["SIMULATE"],
        workflow_globals=globals(), workflow_name="your_workflow_name",
    )

    # === SAFETY CHECKS ===
    # Always validate input files before starting
    lash_e.nr_robot.check_input_file()
    lash_e.nr_track.check_input_file()

    try:
        # === STEP 1: SETUP ===
        logger.info("Step 1: System setup")

        # Set temperature if needed
        lash_e.temp_controller.set_temp(c["TARGET_TEMPERATURE"])

        # Get a new wellplate
        lash_e.grab_new_wellplate()

        # === STEP 2: SAMPLE PREPARATION ===
        logger.info("Step 2: Sample preparation")

        # [REPLACE] Add your sample preparation steps here
        # Examples:
        # lash_e.mass_dispense_into_vial('source_vial', mass_mg=20, return_home=False)
        # lash_e.nr_robot.dispense_into_vial_from_reservoir(
        #     reservoir_index=0,
        #     vial_index='source_vial',
        #     volume=5.0
        # )
        # lash_e.nr_robot.vortex_vial(vial_name='source_vial', vortex_time=c["VORTEX_TIME"])

        # === STEP 3: LIQUID HANDLING ===
        logger.info("Step 3: Liquid handling operations")

        # [REPLACE] Add your liquid handling steps here
        # Examples:
        # lash_e.nr_robot.dispense_from_vial_into_vial(
        #     source_vial_name='source_vial_a',
        #     dest_vial_name='target_vial',
        #     volume=c["PARAM1"]
        # )

        # === STEP 4: REACTIONS (if applicable) ===
        logger.info("Step 4: Reaction processing")

        # [REPLACE] Add reaction steps if needed
        # Examples:
        # REACTOR_NUM = 1
        # lash_e.nr_robot.move_vial_to_location('target_vial', 'photoreactor_array', 0)
        # lash_e.photoreactor.turn_on_reactor_led(reactor_num=REACTOR_NUM, intensity=100)
        # lash_e.photoreactor.stir_reactor(reactor_num=REACTOR_NUM, rpm=600)
        # time.sleep(c["REACTION_TIME"])
        # lash_e.photoreactor.turn_off_reactor_led(reactor_num=REACTOR_NUM)
        # lash_e.photoreactor.turn_off_stirring(reactor_num=REACTOR_NUM)

        # === STEP 5: WELLPLATE PREPARATION ===
        logger.info("Step 5: Wellplate preparation")

        # [REPLACE] Add your wellplate dispensing logic here
        well_indices = list(range(c["PARAM2"]))  # Use PARAM2 as number of wells
        # dispense_volume = c["PARAM1"] / c["PARAM2"]   # Example calculation

        # Example dispensing:
        # lash_e.nr_robot.aspirate_from_vial('target_vial', total_volume)
        # lash_e.nr_robot.dispense_into_wellplate(
        #     dest_wp_num_array=well_indices,
        #     amount_mL_array=[dispense_volume] * c["PARAM2"]
        # )
        # lash_e.nr_robot.remove_pipet()

        # === STEP 6: MEASUREMENTS ===
        logger.info("Step 6: Measurements")

        # Always call measure_wellplate - Lash_E already returns None instead of
        # a real reading in simulate mode, so this does not need to be skipped.
        data = lash_e.measure_wellplate(c["MEASUREMENT_PROTOCOL_FILE"], well_indices)
        if data is None and lash_e.simulate:
            # [REPLACE] Substitute synthetic data here if downstream analysis
            # needs exercising in simulate mode (see simulate_fluorescence_readout()
            # in workflows/fluorescence_calibration_workflow.py for an example).
            logger.info("Simulate mode: no real measurement data returned")
        elif data is None:
            raise RuntimeError("No measurement data returned")
        else:
            logger.info(f"Measurement data collected for {len(well_indices)} wells")

        # === STEP 7: CLEANUP ===
        logger.info("Step 7: Cleanup")

        # Return vials to home positions
        # [REPLACE] Add your vials here
        # lash_e.nr_robot.return_vial_home('source_vial')
        # lash_e.nr_robot.return_vial_home('target_vial')

        # Turn off equipment
        lash_e.temp_controller.turn_off_heating()
        lash_e.temp_controller.turn_off_stirring()

        # Discard wellplate
        lash_e.discard_used_wellplate()

        logger.info("Workflow completed successfully")
        return data

    except Exception as e:
        logger.error(f"Workflow failed with error: {e}")
        # Emergency cleanup
        try:
            lash_e.temp_controller.turn_off_heating()
            lash_e.temp_controller.turn_off_stirring()
            lash_e.photoreactor.emergency_stop()
            lash_e.nr_robot.move_home()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    execute()
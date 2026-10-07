"""
Template for creating new workflows in the North Robotics automation system.
Copy this file, replace the vial/protocol placeholders, and add experiment steps.
The script filename determines its workflow_configs/<name>.yaml configuration.
Lash_E creates the config if missing and owns the normal startup GUI review.
Human/operator: execute() uses the settings confirmed in that GUI.
Automation: execute(config=complete_config, show_gui=False) uses exactly that
configuration, without opening the GUI or reading/writing workflow YAML.
Supplying config does not mean simulation: its SIMULATE value selects the mode.
execute(show_gui=False) uses saved YAML without operator review.
The example enables temperature control but not powder dispensing. If adding
photoreactor steps, add supported per-reactor shutdown calls to cleanup too.
Simulation runs every automation step, but may return no instrument data.
Live runs send best-effort Slack lifecycle updates; simulation never sends Slack.
"""
import sys
sys.path.append("../utoronto_demo")  # Always first, before any local imports

from pathlib import Path
from copy import deepcopy

from master_usdl_coordinator import Lash_E

_WORKFLOW_NAME = Path(__file__).stem

# Workflow config constants, auto-detected by ConfigManager and persisted to
# workflow_configs/<script_name>.yaml (module-level UPPERCASE constants +
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


def _initialize_workflow(config=None, show_gui=True):
    """Return the coordinator and one authoritative experiment-config snapshot.

    With no supplied config, preload saved YAML to select the correct startup
    vial file and simulation mode. This is NOT the final experiment config.
    Lash_E creates missing YAML, reloads it, shows the GUI when requested, and
    reloads operator edits. Only AFTER it returns do we snapshot the confirmed
    globals. Build plans and perform calculations using that returned snapshot.

    A supplied config is a complete mapping containing every _CONFIG_KEYS key,
    not a partial override. Use it with show_gui=False for scheduler/robot calls.
    Copy it so this run cannot mutate the caller's configuration. Do not pass
    workflow globals to Lash_E in this mode: that would let saved YAML override
    the supplied values. No workflow YAML is loaded or saved on this path.

    Keep experiment validation after review in execute(), allowing the human
    to correct settings before they are used. Cancellation is checked there.
    """
    supplied_config = config is not None
    if supplied_config and show_gui:
        raise ValueError("Explicit config requires show_gui=False; otherwise use execute() for GUI review.")
    if not supplied_config:
        from workflow_config_manager import ConfigManager
        ConfigManager.setup_and_reload_config(_WORKFLOW_NAME, globals())
        config = {key: globals()[key] for key in _CONFIG_KEYS}
    launch_config = deepcopy({key: config[key] for key in _CONFIG_KEYS})
    if supplied_config:
        if not isinstance(launch_config["SIMULATE"], bool):
            raise ValueError("SIMULATE must be a boolean.")
        if not Path(launch_config["INPUT_VIAL_STATUS_FILE"]).is_file():
            raise FileNotFoundError(launch_config["INPUT_VIAL_STATUS_FILE"])

    # Initialize the system - adjust initialization flags as needed.
    # Simulated controllers execute the workflow without physical movement;
    # instrument data and error behavior still differ from a live run.
    lash_e = Lash_E(
        launch_config["INPUT_VIAL_STATUS_FILE"],
        initialize_t8=True,      # Temperature controller
        initialize_p2=False,
        simulate=launch_config["SIMULATE"],
        workflow_globals=None if supplied_config else globals(),
        workflow_name=None if supplied_config else _WORKFLOW_NAME,
        show_gui=show_gui,
    )
    confirmed_config = launch_config if supplied_config else deepcopy(
        {key: globals()[key] for key in _CONFIG_KEYS}
    )
    return lash_e, confirmed_config


def _send_workflow_slack(lash_e, message):
    """Send live-run updates without letting Slack failures fail the experiment."""
    if lash_e.simulate:
        return
    try:
        import slack_agent
        if not slack_agent.safe_send_slack_message(message):
            lash_e.logger.warning("Slack notification was not delivered (non-fatal)")
    except Exception as error:
        lash_e.logger.warning(f"Slack notification failed (non-fatal): {error}")


def execute(config=None, show_gui=True):
    """[REPLACE] Describe the experiment performed using confirmed parameters.

    execute(): normal operator GUI; confirmed GUI/YAML values win.
    execute(config=complete_config, show_gui=False): automated run; supplied
    values win, with no GUI or YAML override. Include every _CONFIG_KEYS key.
    Neither mode changes the scientific workflow body or skips automation steps.
    """
    lash_e, c = _initialize_workflow(config=config, show_gui=show_gui)
    if not lash_e._workflow_should_continue:
        return None
    logger = lash_e.logger

    try:
        if not isinstance(c["SIMULATE"], bool):
            raise ValueError("SIMULATE must be a boolean.")
        if not Path(c["INPUT_VIAL_STATUS_FILE"]).is_file():
            raise FileNotFoundError(c["INPUT_VIAL_STATUS_FILE"])
        if Path(c["INPUT_VIAL_STATUS_FILE"]).resolve() != Path(lash_e.nr_robot.VIAL_FILE).resolve():
            raise ValueError("Vial file changed during review. Restart with the new vial file before running.")
        if c["SIMULATE"] != lash_e.simulate:
            raise ValueError("Workflow and controller simulation modes do not match.")
        if not isinstance(c["PARAM2"], int) or isinstance(c["PARAM2"], bool) or not 1 <= c["PARAM2"] <= 96:
            raise ValueError("PARAM2 must be an integer between 1 and 96 for this 96-well example.")
        if not lash_e.simulate and not Path(c["MEASUREMENT_PROTOCOL_FILE"]).is_file():
            raise FileNotFoundError(c["MEASUREMENT_PROTOCOL_FILE"])
        logger.info(f"Starting {_WORKFLOW_NAME} workflow")
        logger.info(f"Parameters: param1={c['PARAM1']}, param2={c['PARAM2']}, simulate={lash_e.simulate}")
        _send_workflow_slack(
            lash_e,
            f"{_WORKFLOW_NAME} started | wells={c['PARAM2']} | temperature={c['TARGET_TEMPERATURE']}C",
        )
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
        _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} completed successfully")
        return data

    except (Exception, KeyboardInterrupt) as e:
        logger.exception(f"Workflow failed with error: {e}")
        for label, action in (
            ("turn_off_heating", lash_e.temp_controller.turn_off_heating),
            ("turn_off_stirring", lash_e.temp_controller.turn_off_stirring),
            ("move_home", lash_e.nr_robot.move_home),
        ):
            try:
                action()
            except Exception:
                logger.exception(f"Cleanup failed: {label}")
        outcome = "interrupted by operator" if isinstance(e, KeyboardInterrupt) else f"failed: {e}"
        _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} {outcome}; cleanup attempted, review experiment log")
        raise


if __name__ == "__main__":
    execute()
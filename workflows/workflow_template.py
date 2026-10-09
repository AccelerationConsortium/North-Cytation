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
Tasks without vial operations may set INPUT_VIAL_STATUS_FILE=None explicitly;
this disables vial tracking, not shared robot/track state or config review.
Live runs send best-effort Slack lifecycle updates; simulation never sends Slack.
"""
import sys
sys.path.append("../utoronto_demo")  # Always first, before any local imports

from pathlib import Path

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


def validate_experiment(config, lash_e):
    """Validate confirmed inputs before any experiment steps.

    Replace the example well-count/protocol constraints with this experiment's
    requirements. ConfigManager owns loading and required-key validation;
    this function does not load, merge or change configuration.
    """
    if not isinstance(config["SIMULATE"], bool):
        raise ValueError("SIMULATE must be a boolean.")
    vial_file = config["INPUT_VIAL_STATUS_FILE"]
    if vial_file is not None and not Path(vial_file).is_file():
        raise FileNotFoundError(vial_file)
    actual_vial_file = lash_e.nr_robot.VIAL_FILE
    if (vial_file is None) != (actual_vial_file is None) or (
        vial_file is not None and Path(vial_file).resolve() != Path(actual_vial_file).resolve()
    ):
        raise ValueError("Vial file changed during review. Restart with the new vial file before running.")
    if config["SIMULATE"] != lash_e.simulate:
        raise ValueError("Workflow and controller simulation modes do not match.")

    well_count = config["PARAM2"]
    if type(well_count) is not int or not 1 <= well_count <= 96:
        raise ValueError("PARAM2 must be an integer between 1 and 96 for this 96-well example.")
    protocol_file = config["MEASUREMENT_PROTOCOL_FILE"]
    if not lash_e.simulate and not Path(protocol_file).is_file():
        raise FileNotFoundError(protocol_file)


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
    ConfigManager handles selection through Lash_E; do not preload, merge or
    snapshot config here. Use lash_e.workflow_config after startup returns.
    Neither mode changes the scientific workflow body or skips automation steps.
    """
    lash_e = Lash_E(
        initialize_t8=True, initialize_p2=False,
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
    )
    if not lash_e._workflow_should_continue:
        return None
    logger = lash_e.logger
    c = lash_e.workflow_config

    try:
        validate_experiment(c, lash_e)
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
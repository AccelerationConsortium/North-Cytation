"""
Amine Protonation Workflow

Concept:
    Protonate amines (dissolved in ethanol) by adding HCl to a stirring solution
    in the photoreactor, then add a metal salt, vortex, and finally deposit a
    droplet onto a slide.

Per-experiment inputs (workflow function args):
    - reactor_vial:         Vial that holds the reaction. Assumed to be
                            pre-loaded with 5 mL of ethanol prior to workflow
                            start. Uses an open cap for now so it can be
                            pipetted into while sitting in the photoreactor.
    - amine_vial + volume:  Source vial and volume of amine-in-ethanol stock.
    - metal_salt_vial + volume: Source vial and volume of metal salt stock.

Fixed constants (not per-experiment parameters):
    - HCl vial identity and HCl addition volume (5 uL of 6 M HCl).
    - Photoreactor slot used for stirring.
    - Stirring RPM.

Notes:
    - Slide dispensing is left as a stub (locations not yet defined).
    - No measurement or data saving is performed here; Slack messages are used
      to communicate progress.
    - Run home_robot_components() once before the first experiment, and call
      move_home() after the last one. Multiple experiments can share a single
      Lash_E object.
"""

import sys
sys.path.append("../utoronto_demo")

try:
    import slack_agent
    _SLACK_AVAILABLE = True
except Exception:
    _SLACK_AVAILABLE = False

from master_usdl_coordinator import Lash_E


# ================================================================================
# FIXED CONFIGURATION (not workflow parameters)
# ================================================================================

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/amine_protonation_vials.csv"

SIMULATE = True  # Set to False for hardware execution

# HCl addition is fixed: 5 uL of 6 M HCl into the stirring reactor vial
HCL_VIAL = "hcl_stock"
HCL_VOLUME_ML = 0.005
HCL_LIQUID = "6M_HCl"  # calibrated liquid class in master_pipetting_measurements.csv

# Photoreactor slot used purely for its stir plate
PHOTOREACTOR_NUM = 0
STIR_RPM = 600

# Vortex duration after metal salt addition
VORTEX_TIME_S = 10

# Slide heater temperature — turned on once before all experiments,
# off once after all experiments, to drive off ethanol from deposited droplets
SLIDE_HEATER_TEMP_C = 60


# ================================================================================
# WORKFLOW
# ================================================================================

def amine_protonation_workflow(
    lash_e,
    reactor_vial: str,
    amine_vial: str,
    amine_volume_mL: float,
    metal_salt_vial: str,
    metal_salt_volume_mL: float,
    vortex_time_s: int = VORTEX_TIME_S,
):
    """
    Run one amine-protonation experiment on a single reactor vial.

    Reactor vial is assumed to be pre-loaded with 5 mL of ethanol prior to
    workflow start.

    Steps:
        1. Add amine solution to the reactor vial (no stirring yet).
        2. Move reactor vial into the photoreactor and start stirring.
        3. Add fixed HCl volume (5 uL of 6 M HCl) while stirring.
        4. Stop stirring and return reactor vial home.
        5. Add metal salt to the reactor vial.
        6. Vortex the reactor vial.
        7. Dispense droplet onto slide (stub - not implemented).
    """
    lash_e.logger.info(
        f"Amine protonation start: reactor={reactor_vial}, amine={amine_vial} "
        f"({amine_volume_mL:.3f} mL), metal={metal_salt_vial} ({metal_salt_volume_mL:.3f} mL)"
    )

    if not lash_e.simulate and _SLACK_AVAILABLE:
        slack_agent.send_slack_message(
            f"Amine protonation started\n"
            f"Reactor vial: {reactor_vial}\n"
            f"Amine: {amine_vial} ({amine_volume_mL:.3f} mL)\n"
            f"Metal salt: {metal_salt_vial} ({metal_salt_volume_mL:.3f} mL)\n"
            f"HCl: {HCL_VIAL} ({HCL_VOLUME_ML*1000:.1f} uL)"
        )

    # ---- 1. Add amine solution to the reactor vial ----
    lash_e.logger.info(f"Dispensing {amine_volume_mL:.3f} mL amine from {amine_vial} -> {reactor_vial}")
    lash_e.nr_robot.dispense_from_vial_into_vial(
        amine_vial, reactor_vial, amine_volume_mL, liquid="ethanol"
    )

    # ---- 2. Move reactor vial into photoreactor and start stirring ----
    lash_e.logger.info(f"Moving {reactor_vial} to photoreactor slot {PHOTOREACTOR_NUM}")
    lash_e.nr_robot.move_vial_to_location(
        reactor_vial, location="photoreactor_array", location_index=PHOTOREACTOR_NUM
    )
    lash_e.logger.info(f"Starting stirring at {STIR_RPM} RPM")
    lash_e.photoreactor.turn_on_reactor_fan(reactor_num=PHOTOREACTOR_NUM, rpm=STIR_RPM)

    # ---- 3. Add fixed HCl volume while stirring ----
    # Reactor vial must have cap_type='open' so it can be pipetted into
    # without being lifted out of the photoreactor.
    lash_e.logger.info(
        f"Adding {HCL_VOLUME_ML*1000:.1f} uL HCl from {HCL_VIAL} -> {reactor_vial} (stirring on)"
    )
    lash_e.nr_robot.dispense_from_vial_into_vial(
        HCL_VIAL, reactor_vial, HCL_VOLUME_ML, liquid=HCL_LIQUID
    )

    # ---- 4. Stop stirring and return reactor vial home ----
    lash_e.logger.info("Stopping stirring")
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=PHOTOREACTOR_NUM)
    lash_e.nr_robot.return_vial_home(reactor_vial)

    # ---- 5. Add metal salt to reactor vial ----
    lash_e.logger.info(f"Dispensing {metal_salt_volume_mL:.3f} mL metal salt from {metal_salt_vial} -> {reactor_vial}")
    lash_e.nr_robot.dispense_from_vial_into_vial(
        metal_salt_vial, reactor_vial, metal_salt_volume_mL, liquid="ethanol"
    )

    # ---- 6. Vortex reactor vial ----
    lash_e.logger.info(f"Vortexing {reactor_vial} for {vortex_time_s} s")
    lash_e.nr_robot.vortex_vial(vial_name=reactor_vial, vortex_time=vortex_time_s)

    # ---- 7. Dispense droplet onto slide (stub) ----
    dispense_droplet_onto_slide(lash_e, reactor_vial)

    lash_e.logger.info(f"Amine protonation complete for reactor vial {reactor_vial}")
    if not lash_e.simulate and _SLACK_AVAILABLE:
        slack_agent.send_slack_message(
            f"Amine protonation complete for reactor vial {reactor_vial}"
        )


def dispense_droplet_onto_slide(lash_e, reactor_vial: str):
    """
    Placeholder: deposit a droplet from the reactor vial onto a slide.

    Slide locations are not yet defined - fill in aspirate/dispense logic here
    once the slide holder position and droplet volume are known.
    """
    lash_e.logger.info(
        f"[TODO] Droplet dispense from {reactor_vial} onto slide - not implemented yet"
    )


# ================================================================================
# WORKFLOW EXECUTION
# ================================================================================

if __name__ == "__main__":
    lash_e = Lash_E(
        INPUT_VIAL_STATUS_FILE,
        simulate=SIMULATE,
        workflow_globals=globals(),
        workflow_name="amine_protonation_workflow",
    )

    # Home robot once before the first experiment
    lash_e.nr_robot.home_robot_components()

    # Turn on slide heater before any experiments so it is up to temperature
    lash_e.logger.info(f"Setting slide heater to {SLIDE_HEATER_TEMP_C} C")
    lash_e.temp_controller.set_temp(SLIDE_HEATER_TEMP_C)

    try:
        # Example: one experiment. Add more calls here for additional experiments
        # sharing the same Lash_E object.
        amine_protonation_workflow(
            lash_e,
            reactor_vial="reactor_vial_1",
            amine_vial="amine_1",
            amine_volume_mL=0.500,
            metal_salt_vial="metal_salt_1",
            metal_salt_volume_mL=0.500,
        )
    finally:
        # Turn off slide heater and return robot home after all experiments
        lash_e.logger.info("Turning off slide heater")
        lash_e.temp_controller.turn_off_heating()
        lash_e.nr_robot.move_home()

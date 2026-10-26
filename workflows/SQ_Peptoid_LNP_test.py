"""Combine two phases, mix at one height, aliquot three wells, and read the plate.

Run from the repository root with:
    python -m workflows.SQ_Peptoid_LNP_test

The normal startup GUI reviews vial positions, volumes, and workflow settings.
Simulation executes every automation step without running the reader protocol.
"""
import sys
sys.path.append("../utoronto_demo")

from datetime import datetime
import math
from pathlib import Path

from master_usdl_coordinator import Lash_E
from pipetting_data.pipetting_parameters import PipettingParameters
from workflows.workflow_template import _send_workflow_slack

_WORKFLOW_NAME = Path(__file__).stem

SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/SQ_Peptoid_LNP_test_vials.csv"
ORGANIC_VIAL = "organic_phase"
AQUEOUS_VIAL = "aqeuous_phase"
SAMPLE_VIAL = "sample"
PHASE_VOLUME_UL = 600.0
MIX_VOLUME_UL = 400.0
MIX_CYCLES = 3
ALIQUOT_VOLUME_UL = 150.0
WELL_INDICES = [0, 1, 2]
WELLPLATE_TYPE = "96 WELL PLATE"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\SQ_Peroxide.prt"
OUTPUT_DIRECTORY = r"C:\Users\Imaging Controller\Desktop\SQ\peptoid_LNP"

_CONFIG_KEYS = [
    "SIMULATE", "INPUT_VIAL_STATUS_FILE", "ORGANIC_VIAL", "AQUEOUS_VIAL",
    "SAMPLE_VIAL", "PHASE_VOLUME_UL", "MIX_VOLUME_UL", "MIX_CYCLES",
    "ALIQUOT_VOLUME_UL", "WELL_INDICES", "WELLPLATE_TYPE", "MEASUREMENT_PROTOCOL_FILE",
]


def validate_experiment(c, lash_e):
    """Check the reviewed experiment before dispensing either phase."""
    robot, track = lash_e.nr_robot, lash_e.nr_track
    if type(c["SIMULATE"]) is not bool or c["SIMULATE"] != lash_e.simulate:
        raise ValueError("Workflow and controller simulation modes must agree.")
    if c["INPUT_VIAL_STATUS_FILE"] is None or robot.VIAL_FILE is None:
        raise ValueError("This workflow requires its three-vial inventory.")
    if Path(c["INPUT_VIAL_STATUS_FILE"]).resolve() != Path(robot.VIAL_FILE).resolve():
        raise ValueError("Vial file changed during startup review; restart with that file.")
    names = [c["ORGANIC_VIAL"], c["AQUEOUS_VIAL"], c["SAMPLE_VIAL"]]
    if len(set(names)) != 3:
        raise ValueError("Organic, aqueous, and sample must be different vials.")
    positions = [(robot.get_vial_info(name, "location"),
                  robot.get_vial_info(name, "location_index")) for name in names]
    if len(set(positions)) != 3:
        raise ValueError("Each vial must occupy a different position.")
    for key in ("PHASE_VOLUME_UL", "MIX_VOLUME_UL", "ALIQUOT_VOLUME_UL"):
        value = c[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if type(c["MIX_CYCLES"]) is not int or c["MIX_CYCLES"] < 1:
        raise ValueError("MIX_CYCLES must be a positive integer.")
    wells = c["WELL_INDICES"]
    plate = robot.WELLPLATES[c["WELLPLATE_TYPE"]]
    if (not wells or any(type(well) is not int or not 0 <= well < plate["num_wells"] for well in wells)
            or len(set(wells)) != len(wells)):
        raise ValueError("WELL_INDICES must contain distinct valid well indices.")
    if c["ALIQUOT_VOLUME_UL"] / 1000 > plate["max_volume_per_well"]:
        raise ValueError("Aliquot volume exceeds the configured well capacity.")
    phase_ml = c["PHASE_VOLUME_UL"] / 1000
    for source in names[:2]:
        if robot.get_vial_info(source, "vial_volume") < phase_ml:
            raise ValueError(f"Insufficient starting liquid in {source}.")
    available_ul = robot.get_vial_info(c["SAMPLE_VIAL"], "vial_volume") * 1000 + 2 * c["PHASE_VOLUME_UL"]
    if c["MIX_VOLUME_UL"] > available_ul or len(wells) * c["ALIQUOT_VOLUME_UL"] > available_ul:
        raise ValueError("Sample volume is insufficient for mixing or the requested aliquots.")
    if max(c["MIX_VOLUME_UL"], c["ALIQUOT_VOLUME_UL"]) / 1000 >= robot.PIPET_TIPS["large_tip"]["volume"]:
        raise ValueError("Mixing and aliquot aspirations must fit a large tip.")
    if track.CURRENT_WP_TYPE != c["WELLPLATE_TYPE"]:
        raise ValueError("Source-stack plate type must match WELLPLATE_TYPE.")
    if track.NUM_SOURCE < 1 or track.ACTIVE_WELLPLATE_POSITION is not None:
        raise ValueError("A source plate and an empty track/pipetting area are required.")
    if not lash_e.simulate and not Path(c["MEASUREMENT_PROTOCOL_FILE"]).is_file():
        raise FileNotFoundError(c["MEASUREMENT_PROTOCOL_FILE"])


def execute(config=None, show_gui=True):
    """Execute the six requested steps using GUI-confirmed or supplied settings."""
    lash_e = Lash_E(workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
                    config=config, show_gui=show_gui)
    if not lash_e._workflow_should_continue:
        return None
    c = lash_e.workflow_config
    validate_experiment(c, lash_e)
    robot = lash_e.nr_robot
    phase_ml = c["PHASE_VOLUME_UL"] / 1000
    mix_ml = c["MIX_VOLUME_UL"] / 1000
    aliquot_ml = c["ALIQUOT_VOLUME_UL"] / 1000
    transfer_parameters = PipettingParameters()
    # The existing aspiration method performs N-1 complete cycles, then a final
    # aspiration. The following dispense completes cycle N at the same height.
    mix_parameters = PipettingParameters(asp_disp_cycles=c["MIX_CYCLES"] - 1)

    _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} started")
    try:
        robot.home_robot_components()
        robot.dispense_from_vial_into_vial(c["ORGANIC_VIAL"], c["SAMPLE_VIAL"], phase_ml,
            parameters=transfer_parameters, specified_tip="large_tip", compensate_overvolume=False)
        robot.dispense_from_vial_into_vial(c["AQUEOUS_VIAL"], c["SAMPLE_VIAL"], phase_ml,
            parameters=transfer_parameters, specified_tip="large_tip", compensate_overvolume=False,
            remove_tip=False, return_vial_home=False)
        # Keep the aqueous-transfer tip; no disposal or vial-return step intervenes.
        # Position once for mixing, then the built-in cycles move only the pump.
        robot.aspirate_from_vial(c["SAMPLE_VIAL"], mix_ml, parameters=mix_parameters,
            specified_tip="large_tip", track_height=False, move_up=False)
        robot.dispense_into_vial(c["SAMPLE_VIAL"], mix_ml, parameters=mix_parameters,
            initial_move=False, record_measurement=False)
        robot.remove_pipet()

        lash_e.grab_new_wellplate()
        for well in c["WELL_INDICES"]:
            robot.aspirate_from_vial(c["SAMPLE_VIAL"], aliquot_ml,
                parameters=transfer_parameters, specified_tip="large_tip")
            robot.dispense_into_wellplate([well], [aliquot_ml],
                parameters=transfer_parameters, well_plate_type=c["WELLPLATE_TYPE"])
        robot.remove_pipet()
        data = lash_e.measure_wellplate(c["MEASUREMENT_PROTOCOL_FILE"], c["WELL_INDICES"],
            plate_type=c["WELLPLATE_TYPE"])
        if data is None:
            if not lash_e.simulate:
                raise RuntimeError("Plate reader returned no measurements.")
            lash_e.logger.info("Simulation completed all steps; no real reader data returned.")
        elif not lash_e.simulate:
            output = Path(c["OUTPUT_DIRECTORY"])
            output.mkdir(parents=True, exist_ok=True)
            filename = f"{_WORKFLOW_NAME}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
            data.to_csv(output / filename)
            lash_e.logger.info("Reader measurements saved to %s", output / filename)
    except (Exception, KeyboardInterrupt) as error:
        lash_e.logger.exception("%s failed or was interrupted", _WORKFLOW_NAME)
        if isinstance(error, KeyboardInterrupt):
            from workflows.interrupt_cleanup import cleanup_after_keyboard_interrupt
            cleanup_after_keyboard_interrupt(lash_e)
        _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} interrupted" if isinstance(error, KeyboardInterrupt)
            else f"{_WORKFLOW_NAME} failed: {error}")
        raise
    _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} completed")
    return data


if __name__ == "__main__":
    execute()

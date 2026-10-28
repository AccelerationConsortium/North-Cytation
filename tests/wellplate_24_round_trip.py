"""One plate: source stack -> active area -> Cytation -> active area -> waste.

Run from the repository root: python tests/wellplate_24_round_trip.py
Defaults to simulation with the usual startup state/config review. For a live
run, verify the actual plate's grip heights and Cytation plate definition first.
Start with one lidless 24-well plate in source, empty waste and active areas,
and track plate type '24 WELL PLATE'. One plate avoids using provisional stack
spacing. No pipetting grid or measurement protocol is needed.
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from master_usdl_coordinator import Lash_E

SIMULATE = True
INPUT_VIAL_STATUS_FILE = None

_WORKFLOW_NAME = Path(__file__).stem
_CONFIG_KEYS = ["SIMULATE", "INPUT_VIAL_STATUS_FILE"]
_PLATE_TYPE = "24 WELL PLATE"


def execute(config=None, show_gui=True):
    lash = Lash_E(
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
    )
    if not lash._workflow_should_continue:
        return None
    track = lash.nr_track
    plate_id = track.inventory.plate_at("source_stack", _PLATE_TYPE)
    if track.NUM_SOURCE != 1 or track.NUM_WASTE != 0:
        raise ValueError("Start with exactly one source plate and an empty waste stack.")
    if any(track.inventory.contents(p) for p in ("pipetting_area", "cytation_tray", "gripper")):
        raise ValueError("Start with no active plate; pipetting area, Cytation and track gripper must be empty.")

    phase = "grab_new_wellplate"
    try:
        lash.grab_new_wellplate()
        if track.inventory.top("pipetting_area") != plate_id or track.NUM_SOURCE != 0:
            raise RuntimeError("Source plate did not reach the active pipetting area.")
        phase = "measure_wellplate (null protocol)"
        data = lash.measure_wellplate(None, plate_type=_PLATE_TYPE)
        if track.inventory.top("pipetting_area") != plate_id:
            raise RuntimeError("Plate did not return from Cytation to the active pipetting area.")
        phase = "discard_used_wellplate"
        lash.discard_used_wellplate()
        if track.inventory.contents("waste_stack") != [plate_id] or track.NUM_WASTE != 1:
            raise RuntimeError("Plate did not reach the waste stack.")
        lash.logger.info("24-well round trip complete; plate is in waste. No read was run.")
        return data  # None is the expected result for a null measurement.
    except (Exception, KeyboardInterrupt):
        lash.logger.exception("24-well test stopped during %s; inspect plate position before recovery.", phase)
        raise


if __name__ == "__main__":
    execute()

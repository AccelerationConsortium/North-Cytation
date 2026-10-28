"""For each source plate: grab, null-measure, discard, in physical top-first order.

Run from repo root: python tests/mixed_wellplate_round_trip.py
Default is simulation with normal startup review. Record the plates actually
at source in Track Status; any number and combination is accepted. Existing
waste is retained. Empty source is a successful no-op. The example at
tests/fixtures/mixed_wellplate_track_status.yaml has 96 on 24, but is optional.
Set SIMULATE=False in workflow config only for an intended physical run.
No pipetting or optical protocol is used; ordinary motion requirements apply
to each command (for example, the active area must have room for a pickup).
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from master_usdl_coordinator import Lash_E

SIMULATE = True
INPUT_VIAL_STATUS_FILE = None
_WORKFLOW_NAME = Path(__file__).stem
_CONFIG_KEYS = ["SIMULATE", "INPUT_VIAL_STATUS_FILE"]


def execute(config=None, show_gui=True):
    lash = Lash_E(workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
                  config=config, show_gui=show_gui)
    if not lash._workflow_should_continue:
        return None
    inventory = lash.nr_track.inventory
    source = inventory.contents("source_stack")
    initial_waste = inventory.contents("waste_stack")
    if not source:
        lash.logger.info("No plates at source; nothing to move.")
        return []
    completed = []
    phase = "pickup"
    expected_id = None
    try:
        for expected_id in reversed(source):
            plate_type = inventory.plate_type(expected_id)
            lash.logger.info("Mixed-stack test: %s (%s)", expected_id, plate_type)
            phase = "pickup from source"
            plate_id = lash.grab_new_wellplate()
            if plate_id != expected_id:
                raise RuntimeError("Pickup selected the wrong source plate")
            # Type and lid state are inferred from the physical plate's inventory.
            phase = "null measurement and return"
            result = lash.measure_wellplate(None)
            if result is not None or inventory.top("pipetting_area") != plate_id:
                raise RuntimeError("Null measurement did not return the expected plate")
            phase = "discard to waste"
            if lash.discard_used_wellplate() != plate_id:
                raise RuntimeError("Discard selected a different plate")
            completed.append(plate_id)
        if inventory.contents("source_stack") or inventory.contents("waste_stack") != initial_waste + completed:
            raise RuntimeError("Unexpected final source/waste order")
        lash.logger.info("Source-stack test complete: %s plates; waste bottom-to-top %s",
                         len(completed), inventory.contents("waste_stack"))
        return completed
    except (Exception, KeyboardInterrupt):
        lash.logger.exception("Source-stack test stopped during %s for %s. Current source (bottom to top): %s; held: %s. No automatic recovery.",
                              phase, expected_id, inventory.contents("source_stack"), inventory.positions["gripper"])
        raise


if __name__ == "__main__":
    execute()

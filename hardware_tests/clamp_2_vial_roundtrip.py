"""Move one multidimensional-workflow vial through clamp_2 and return it home.

Run from the repository root: python hardware_tests/clamp_2_vial_roundtrip.py
Defaults to simulation. A missed pickup at clamp_2 has previously caused a
collision, so reconcile physical state and use an expendable vial for live runs.
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from master_usdl_coordinator import Lash_E


SIMULATE = True
INPUT_VIAL_STATUS_FILE = str(_REPO_ROOT / "status" / "surfactant_multidim_vials.csv")
VIAL_NAME = "BDDAC_dilution_1"

_WORKFLOW_NAME = Path(__file__).stem
_CONFIG_KEYS = ["SIMULATE", "INPUT_VIAL_STATUS_FILE", "VIAL_NAME"]
_CONFIG_DOCS = {
    "SIMULATE": {"description": "Run without controlling robot hardware.", "unit": None, "affects": []},
    "INPUT_VIAL_STATUS_FILE": {"description": "CSV defining multidimensional workflow vial locations.", "unit": None, "affects": ["vials"], "shares": "vial_layout"},
    "VIAL_NAME": {"description": "Vial to move through clamp_2 and return to its recorded home.", "unit": None, "affects": ["vials"]},
}
_PRODUCT_VIALS = []


def execute(config=None, show_gui=True):
    lash = Lash_E(
        initialize_track=False, initialize_biotek=False,
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
    )
    if not lash._workflow_should_continue:
        return None

    robot = lash.nr_robot
    vial = lash.workflow_config["VIAL_NAME"]
    if robot.GRIPPER_STATUS is not None or robot.HELD_PIPET_TYPE is not None:
        raise ValueError("Start with an empty gripper and no pipette tip.")
    if robot.get_vial_in_location("clamp_2", 0) is not None:
        raise ValueError("clamp_2[0] must be empty before starting.")
    if not robot.is_vial_movable(vial):
        raise ValueError(f"Vial {vial} is not movable in its recorded state.")

    for i in range(5):
        lash.logger.info("Iteration %d: Moving %s to clamp_2[0]", i, vial)
        robot.move_vial_to_location(vial, "heater", 2)
        lash.logger.info("Iteration %d: Parking robot at home", i)
        robot.move_home()
        lash.logger.info("Iteration %d: Returning %s to its recorded vial home", i, vial)
        robot.return_vial_home(vial)
        robot.move_home()


if __name__ == "__main__":
    execute()
"""Scheduler-compatible copy of tests/test_vortex.py; the original is untouched.

Run this script normally for the existing startup GUI. The scheduler instead
calls execute(config=complete_config, show_gui=False). Lash_E owns YAML/GUI
review in the normal path; the experiment uses confirmed values afterward.
This task needs a vial CSV because vortex_vial and return_vial_home use vial
identity and location. Vial-less tasks may explicitly configure
INPUT_VIAL_STATUS_FILE=None, but must not call vial-dependent methods.
"""

import sys
sys.path.append("../utoronto_demo")
from pathlib import Path
from master_usdl_coordinator import Lash_E


SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/fluorescence_calibration_vials.csv"
TARGET_VIAL = "dye_b1_s1"
VORTEX_TIME = 3

_WORKFLOW_NAME = Path(__file__).stem
_CONFIG_KEYS = ["SIMULATE", "INPUT_VIAL_STATUS_FILE", "TARGET_VIAL", "VORTEX_TIME"]


def validate_experiment(config):
    if config["INPUT_VIAL_STATUS_FILE"] is None:
        raise ValueError("Vortex requires a vial CSV; no-vial tracking is not valid for this task.")
    if not isinstance(config["TARGET_VIAL"], str) or not config["TARGET_VIAL"].strip():
        raise ValueError("TARGET_VIAL must be a vial name.")
    if type(config["VORTEX_TIME"]) not in (int, float) or not 0 < config["VORTEX_TIME"] < float("inf"):
        raise ValueError("VORTEX_TIME must be finite and positive.")

def execute(config=None, show_gui=True):
    """Vortex, return the vial home, then home the robot using reviewed settings."""
    lash_e = Lash_E(
        initialize_track=False, initialize_biotek=False,
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
    )
    
    if not lash_e._workflow_should_continue:
        return None
    
    c = lash_e.workflow_config
    validate_experiment(c)
    
    lash_e.logger.info(f"Vortex task: {c['TARGET_VIAL']} for {c['VORTEX_TIME']} seconds")
    try:
        lash_e.nr_robot.vortex_vial(c["TARGET_VIAL"], vortex_time=c["VORTEX_TIME"])
        lash_e.nr_robot.return_vial_home(c["TARGET_VIAL"])
        lash_e.nr_robot.move_home()
    except (Exception, KeyboardInterrupt):
        lash_e.logger.exception("Vortex task failed")
        raise
    lash_e.logger.info("Vortex task completed")


if __name__ == "__main__":
    execute()
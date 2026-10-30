"""Recover the robot in an isolated live process using a reviewed vial file."""

import argparse
import builtins
import csv
import json
import logging
import os
from pathlib import Path
import traceback
from unittest.mock import patch

import yaml

from scheduler.live_runner import Capture, fingerprint, scheduler_live_lock
from scheduler.simulation_runner import Diagnostics
from scheduler.simulation_state import SESSION_ENV


def run_reset(folder, factory=None):
    folder = Path(folder)
    result = {"completed": False, "exception": None, "records": []}
    diagnostics = Diagnostics()
    logger = logging.getLogger("my_logger")
    capture = Capture(diagnostics)
    logger.addFilter(capture)
    try:
        if os.environ.get(SESSION_ENV):
            raise RuntimeError("Robot reset cannot run with simulation state.")
        job = json.loads((folder / "input.json").read_text(encoding="utf-8"))
        with scheduler_live_lock():
            if fingerprint(job["input_hashes"]) != job["input_hashes"]:
                raise ValueError("Reset inputs changed after confirmation. Review and try again.")
            state = yaml.safe_load(Path(job["robot_state"]).read_text(encoding="utf-8"))
            references = [state.get(key) for key in ("gripper_vial_index", "pipet_fluid_vial_index")]
            references = [str(value) for value in references if value not in (None, "None")]
            if references:
                if job["vial_file"] is None:
                    raise ValueError("Recorded held vial/liquid requires a row with the correct vial file.")
                with Path(job["vial_file"]).open(newline="", encoding="utf-8-sig") as stream:
                    indices = {row["vial_index"] for row in csv.DictReader(stream)}
                if not set(references).issubset(indices):
                    raise ValueError("Selected vial file does not contain the recorded held vial/liquid indices.")
            if factory is None:
                from master_usdl_coordinator import Lash_E
                factory = Lash_E

            def no_prompt(message=""):
                raise RuntimeError(f"Reset stopped for operator review: {message}")

            with patch.object(builtins, "input", no_prompt):
                # North_Robot.__init__ loads the live state and calls
                # reset_after_initialization itself. Do not reset twice.
                lash = factory(vial_file=job["vial_file"], simulate=False,
                               initialize_track=False, initialize_biotek=False,
                               show_gui=False)
                robot = lash.nr_robot
                for key in ("GRIPPER_STATUS", "GRIPPER_VIAL_INDEX", "HELD_PIPET_TYPE", "PIPET_FLUID_VIAL_INDEX"):
                    if getattr(robot, key) not in (None, "None"):
                        raise RuntimeError(f"Reset incomplete: {key}={getattr(robot, key)}")
                if robot.PIPET_FLUID_VOLUME not in (None, 0):
                    raise RuntimeError("Reset incomplete: liquid remains in the pipette.")
                robot.save_robot_status()
                if any(record["level"] in {"ERROR", "CRITICAL"} for record in diagnostics.records):
                    raise RuntimeError("Reset logged errors; review the console log.")
                result["completed"] = True
    except (Exception, KeyboardInterrupt, SystemExit):
        result["exception"] = traceback.format_exc()
        print(result["exception"], flush=True)
    finally:
        result["records"] = diagnostics.records
        logger.removeFilter(capture)
        temporary = folder / "result.tmp"
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(folder / "result.json")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-folder", required=True)
    args = parser.parse_args()
    result = run_reset(args.job_folder)
    if not result["completed"]:
        raise RuntimeError(result["exception"] or "Robot reset failed")


if __name__ == "__main__":
    main()

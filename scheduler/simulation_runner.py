"""One isolated simulation job, launched by the GUI as a Python subprocess."""

import argparse
import builtins
import csv
import importlib
import json
import logging
from decimal import Decimal
from pathlib import Path
import traceback
from unittest.mock import patch

import yaml

from scheduler import simulation_state as state


class Diagnostics(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append({"level": record.levelname, "message": record.getMessage()})


def handoff_issues(root):
    robot = yaml.safe_load((root / "robot_status.yaml").read_text())
    track = yaml.safe_load((root / "track_status.yaml").read_text())
    issues = []
    for key in ("gripper_status", "gripper_vial_index", "held_pipet_type", "pipet_fluid_vial_index"):
        value = robot[key]
        if value is not None and value != "None":
            issues.append(f"Robot state still contains {key}={value}")
    if robot["pipet_fluid_volume"] is not None and float(robot["pipet_fluid_volume"]) != 0:
        issues.append("Liquid remains in the pipet")
    position = track["active_wellplate_position"]
    if position is not None and position != "None":
        issues.append(f"Wellplate remains at {position}")
    occupied = {}
    for path in sorted((root / "vials").glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for record in csv.DictReader(stream):
                key = (record["location"], Decimal(record["location_index"]))
                if key in occupied:
                    issues.append(f"Current vial-position conflict at {key}: {occupied[key]} and {path.name}")
                occupied[key] = path.name
    return issues


def run_job(root, job_id):
    root = Path(root).resolve()
    job = root / "jobs" / job_id
    specification = json.loads((job / "input.json").read_text(encoding="utf-8"))
    name = specification["workflow"]
    if Path(name).name != name or name.startswith("_") or not (state.REPO_ROOT / "workflows" / f"{name}.py").is_file():
        raise ValueError("Invalid selected workflow")
    diagnostics = Diagnostics()
    root_logger = logging.getLogger()
    root_logger.addHandler(diagnostics)
    state._diagnostic_handler = diagnostics
    result = {"workflow": name, "completed": False, "handoff_ok": False,
              "records": [], "exception": None, "log_file": None, "tips_used": {}, "plates_used": None}

    def unexpected_input(prompt=""):
        raise RuntimeError(f"Simulation requires operator input: {prompt}")

    try:
        with state.workflow_state(root, job_id), patch.object(builtins, "input", unexpected_input):
            initial_robot = yaml.safe_load((root / "robot_status.yaml").read_text())
            initial_tips = initial_robot["pipets_used"]
            initial_plates = yaml.safe_load((root / "track_status.yaml").read_text())["num_in_source"]
            if specification.get("allow_vial_conflicts", False):
                for issue in handoff_issues(root):
                    if issue.startswith("Current vial-position conflict"):
                        result["records"].append({"level": "WARNING", "message": f"Simulation test override: {issue}"})
            module = importlib.import_module(f"workflows.{name}")
            config = state.simulated_config(root, specification["config"])
            module.execute(config=config, show_gui=False)
            if len(state._controllers) != 1:
                raise RuntimeError("Workflow did not initialize exactly one simulation coordinator")
            coordinator = state._controllers[0]
            if not coordinator._workflow_should_continue:
                raise RuntimeError("Workflow was cancelled before execution")
            result["log_file"] = str(state.REPO_ROOT / "logs" / coordinator.log_filename)
        result["completed"] = True
        issues = handoff_issues(root)
        blocking_issues = []
        for issue in issues:
            overridden = specification.get("allow_vial_conflicts", False) and issue.startswith("Current vial-position conflict")
            if not overridden:
                blocking_issues.append(issue)
            result["records"].append({
                "level": "WARNING" if overridden else "ERROR",
                "message": f"Simulation test override: {issue}" if overridden else f"Next workflow blocked: {issue}",
            })
        result["handoff_ok"] = not blocking_issues
        final_tips = yaml.safe_load((root / "robot_status.yaml").read_text())["pipets_used"]
        result["tips_used"] = {rack: final_tips[rack] - initial_tips[rack] for rack in final_tips}
        final_plates = yaml.safe_load((root / "track_status.yaml").read_text())["num_in_source"]
        result["plates_used"] = initial_plates - final_plates
    except (Exception, KeyboardInterrupt, SystemExit):
        result["exception"] = traceback.format_exc()
        result["records"].append({"level": "ERROR", "message": result["exception"]})
    finally:
        result["records"] = diagnostics.records + result["records"]
        root_logger.removeHandler(diagnostics)
        logging.getLogger("my_logger").removeHandler(diagnostics)
        state._diagnostic_handler = None
        temp = job / "result.tmp"
        temp.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temp.replace(job / "result.json")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    result = run_job(args.session, args.job)
    raise SystemExit(0 if result["completed"] and result["handoff_ok"] else 1)


if __name__ == "__main__":
    main()
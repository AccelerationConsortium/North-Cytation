"""Routine dispense records in the existing calibration measurement format.

Recording never changes calibration or retries a physical dispense. Simulation
is rejected before reading properties, serializing measurements, or writing files.
"""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import math
import uuid

import pandas as pd
import yaml

from pipetting_data.liquid_registry import get_liquid
from pipetting_data.validation_decisions import append_csv


MASTER_MEASUREMENTS = Path(__file__).resolve().parent / "master_pipetting_measurements.csv"


def recording_density(robot, liquid, enabled=True):
    if (robot.simulate or not enabled or liquid is None
            or getattr(robot, "dispense_recording", "off") != "auto"):
        return None
    try:
        return get_liquid(liquid)["density"]
    except (ValueError, OSError, yaml.YAMLError) as exc:
        robot.logger.warning("Skipping dispense recording: %s", exc)
        return None


def analyze_baselines(readings):
    """Use the existing stable-baseline means and variability calculation.

    Unstable/missing baselines have no substitute measurement. Keep the reason
    inside the existing stability_info field rather than extending the CSV.
    """
    info = {}
    baselines = []
    for prefix in ("pre", "post"):
        values = [r["mass_g"] for r in readings
                  if r["phase"] == "baseline_" + prefix
                  and math.isfinite(r["mass_g"])]
        stable = [r["mass_g"] for r in readings
                  if r["phase"] == "baseline_" + prefix
                  and r["steady_status"] and math.isfinite(r["mass_g"])]
        info[prefix + "_total_count"] = len(values)
        info[prefix + "_stable_count"] = len(stable)
        info[prefix + "_baseline_std"] = float(pd.Series(values, dtype=float).std())
        baselines.append(sum(stable) / len(stable) if stable else None)
    if len(readings) < 4 or any(value is None for value in baselines):
        info["failure_reason"] = "Insufficient readings or missing stable pre/post baseline"
        return None, info, None, None
    pre, post = baselines
    return post - pre, info, pre, post


def read_steady_mass(robot):
    """Optional weighing must never stop liquid handling."""
    if robot.simulate:
        return None
    try:
        mass = float(robot.c9.read_steady_scale())
        if math.isfinite(mass):
            return mass
    except Exception as exc:
        robot.logger.warning("Dispense weighing unavailable: %s", exc)
    return None


def record_dispense(robot, *, liquid, density, target, mass, stability,
                    parameters, destination, reservoir_index=None):
    if robot.simulate or density is None or liquid is None:
        return
    try:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        valid = mass is not None and math.isfinite(mass)
        info = dict(stability or {})
        if reservoir_index is None:
            info["tip_type"] = getattr(robot, "HELD_PIPET_TYPE", None)
        if not valid:
            info.setdefault("failure_reason", "Balance measurement unavailable")
        saved_info = {k: None if isinstance(v, float) and not math.isfinite(v) else v
                      for k, v in info.items()}
        source = getattr(robot, "_dispense_aspiration_source",
                         getattr(robot, "PIPET_FLUID_VIAL_INDEX", None))
        source_name = (robot.get_vial_info(source, "vial_name")
                       if reservoir_index is None and source is not None else None)
        settings = asdict(parameters)
        aspiration = getattr(robot, "_dispense_aspiration_settings", None)
        if reservoir_index is None:
            for name in ("aspirate_speed", "retract_speed", "aspirate_wait_time",
                         "post_retract_wait_time", "overaspirate_vol",
                         "pre_asp_air_vol", "post_asp_air_vol", "asp_disp_cycles"):
                settings[name] = aspiration[name] if aspiration else None
        row = dict(timestamp=now, measured_at=now,
                   session_id="routine_" + uuid.uuid4().hex,
                   validation_type="vial_to_vial" if reservoir_index is None else "reservoir_dispensing",
                   liquid_type=liquid, density_used=density,
                   target_volume_ml=target,
                   mass_difference_g=mass if valid else None,
                   measured_volume_ml=mass / density if valid else None,
                   accuracy_pct=(mass / density - target) / target * 100 if valid else None,
                   source_vial=source_name,
                   destination_vial=robot.get_vial_info(destination, "vial_name"),
                   simulate=False, retry_count=0, measurement_acceptable=valid,
                   stability_info=json.dumps(saved_info, allow_nan=False),
                   output_folder_path=getattr(robot, "log_filename", None),
                   **settings)
        if reservoir_index is not None:
            # Existing reservoir schema; no new classification columns.
            row["reservoir_index"] = reservoir_index
        from pipetting_data.embedded_calibration_validation import _get_latest_environmental_data
        environment = _get_latest_environmental_data()
        if environment:
            row.update(environment)
        append_csv(pd.DataFrame([row]), MASTER_MEASUREMENTS)
    except Exception as exc:
        robot.logger.warning("Could not save dispense measurement: %s", exc)

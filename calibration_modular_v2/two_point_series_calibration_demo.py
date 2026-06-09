"""Two-point overaspirate calibration demo across a series of volumes.

This script demonstrates the v2 two-point system in a compartmentalized form,
without running full screening/optimizer flows. It uses HardwareCalibrationProtocol
for measurements and applies the same Point 2 spread logic used in v2 experiment:

    spread_ul = max(abs(shortfall_ul) + tolerance_buffer_ul, 2.0)

Baseline parameters are extracted from prior optimization trial_results.csv files
using the same SDL composite scoring logic as v2 (including precision).

Volumes tested (uL): 25, 50, 75, 100, 150
Replicates per point: 3

Usage:
  python calibration_modular_v2/two_point_series_calibration_demo.py
  python calibration_modular_v2/two_point_series_calibration_demo.py --simulate
"""

import argparse
import csv
import logging
import statistics
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration_modular_v2.calibration_protocol_northrobot import HardwareCalibrationProtocol, LIQUIDS

# ────────────────────────────────────────────────────────────────────────────
# BASELINE PARAMETERS SOURCE: Prior trial_results.csv files
# Provide the path to trial_results.csv for each liquid, or leave as None to use defaults
TRIAL_RESULTS_BY_LIQUID: Dict[str, Optional[str]] = {
     "glycerol": "calibration_modular_v2/output/run_1780513054_glycerol/trial_results.csv",
     #"agar_water_4%": "calibration_modular_v2/output/run_1779813169_agar_water_4%/trial_results.csv",
    #"DMSO": "calibration_modular_v2/output/run_1779912579_DMSO/trial_results.csv",
    "water": "calibration_modular_v2/output/run_1779739005_water/trial_results.csv",
    "ethanol": "calibration_modular_v2/output/run_1780412080_ethanol/trial_results.csv",
    #"PVA_DMSO": "calibration_modular_v2/output/run_1779906029_PVA_DMSO/trial_results.csv",
}
# ────────────────────────────────────────────────────────────────────────────

SIMULATE = True
REPLICATES = 3
VOLUME_SERIES_UL = [70, 100.0, 150.0]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
CONFIG_FILE = Path(__file__).resolve().parent / "experiment_config.yaml"
HARDWARE_CONFIG_FILE = Path(__file__).resolve().parent / "north_robot_hardware.yaml"

logger = logging.getLogger("two_point_series_calibration_demo")


def _load_trial_results(csv_path: str) -> Optional[pd.DataFrame]:
    """Load trial_results.csv and parse relevant columns."""
    try:
        df = pd.read_csv(csv_path)
        for col in ["deviation_pct", "precision_cv_pct", "duration_mean_s", "measurement_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Filter out single-measurement trials (can't calculate valid precision)
        df = df[df["measurement_count"] >= 2]
        
        if df.empty:
            logger.warning(f"No trials with >=2 measurements found in {csv_path}")
            return None
        
        logger.info(f"Loaded {len(df)} valid trials (>=2 measurements) from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load trial results from {csv_path}: {e}")
        return None


def _desirability(metric: float, tolerance: float, s: float = 2.0) -> float:
    """Soft desirability: 1.0=perfect, 0.5=at tolerance, >0 beyond tolerance."""
    return 1.0 / (1.0 + (metric / tolerance) ** s)


def _extract_best_trial_parameters(df: pd.DataFrame, tolerance_pct: float = 3.0) -> Optional[Dict[str, float]]:
    """Find best trial using desirability scoring and extract its parameters.
    
    Desirability scoring: d=1/(1+(metric/tolerance)^2), higher=better.
    Time normalized within population (no fixed reference available).
    Weights: accuracy=0.4, precision=0.5, time=0.1.
    """
    if df is None or df.empty:
        return None

    df = df.copy()
    t_min = df["duration_mean_s"].min()
    t_max = df["duration_mean_s"].max()
    t_range = max(t_max - t_min, 1.0)

    df["d_acc"]  = df["deviation_pct"].apply(lambda x: _desirability(x, tolerance_pct))
    df["d_prec"] = df["precision_cv_pct"].apply(lambda x: _desirability(x, tolerance_pct))
    df["d_time"] = (t_max - df["duration_mean_s"]) / t_range
    df["composite_score"] = 0.4 * df["d_acc"] + 0.5 * df["d_prec"] + 0.1 * df["d_time"]

    # Find best trial (highest desirability)
    best_row = df.loc[df["composite_score"].idxmax()]
    logger.info(
        f"Best trial: desirability={best_row['composite_score']:.4f} | "
        f"accuracy={best_row['deviation_pct']:.2f}% | "
        f"precision={best_row['precision_cv_pct']:.2f}% | "
        f"time={best_row['duration_mean_s']:.2f}s"
    )
    
    # Extract all parameter columns (those starting with "param_" or matching known param names)
    param_cols = [
        "aspirate_speed", "dispense_speed", "aspirate_wait_time", "dispense_wait_time",
        "pre_asp_air_vol", "post_asp_air_vol", "blowout_vol", "asp_disp_cycles",
        "post_retract_wait_time", "retract_speed",
    ]
    
    params = {}
    for col in param_cols:
        # Try direct name, hardware_parameters_ prefix, and param_ prefix
        if col in best_row:
            params[col] = float(best_row[col])
        elif f"hardware_parameters_{col}" in best_row:
            params[col] = float(best_row[f"hardware_parameters_{col}"])
        elif f"param_{col}" in best_row:
            params[col] = float(best_row[f"param_{col}"])

    # Handle overaspirate separately — column name varies by run
    for ov_col in ("overaspirate_vol", "calibration_overaspirate_vol", "param_overaspirate_vol"):
        if ov_col in best_row:
            params["overaspirate_vol"] = float(best_row[ov_col])
            break
    
    if not params:
        logger.warning("No parameter columns found in trial results")
        return None
    
    logger.info(f"Extracted {len(params)} parameters from best trial")
    for k, v in sorted(params.items()):
        logger.info(f"  {k}: {v}")
    return params


def _get_tolerance_from_run_dir(run_dir: Path) -> float:
    """Read volume target from experiment_config_used.yaml and return matching tolerance %."""
    cfg_path = run_dir / "experiment_config_used.yaml"
    if not cfg_path.exists():
        logger.warning(f"No experiment_config_used.yaml found in {run_dir}, using default tolerance 3.0%")
        return 3.0
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    vol_ml = cfg["experiment"]["volume_targets_ml"][0]
    vol_ul = vol_ml * 1000
    for vr in cfg["tolerances"]["volume_ranges"]:
        if vr["volume_min_ul"] <= vol_ul <= vr["volume_max_ul"]:
            tol = float(vr["tolerance_pct"])
            logger.info(f"Loaded tolerance {tol}% for volume {vol_ul:.0f}uL from {cfg_path.name}")
            return tol
    return 3.0


def _get_baseline_params_for_liquid(liquid_name: str) -> Optional[Dict[str, float]]:
    """Load baseline parameters from trial_results.csv for a liquid, or return None to use defaults."""
    csv_path = TRIAL_RESULTS_BY_LIQUID.get(liquid_name)
    if csv_path is None:
        logger.debug(f"No trial_results.csv configured for '{liquid_name}', will use defaults")
        return None
    
    # Handle relative paths
    if not Path(csv_path).is_absolute():
        csv_path = Path(__file__).resolve().parent.parent / csv_path
    else:
        csv_path = Path(csv_path)
    
    if not csv_path.exists():
        logger.warning(f"Trial results file not found: {csv_path}")
        return None
    
    run_dir = csv_path.parent
    tolerance_pct = _get_tolerance_from_run_dir(run_dir)
    
    df = _load_trial_results(str(csv_path))
    if df is None:
        return None
    
    params = _extract_best_trial_parameters(df, tolerance_pct=tolerance_pct)
    return params

LIQUID_SERIES: List[Dict[str, str]] = [
    {"label": "glycerol_dye", "liquid_name": "glycerol", "vial_name": "glycerol_dye"},
    {"label": "water_dye",    "liquid_name": "water",    "vial_name": "water_dye"},
    {"label": "ethanol_dye",  "liquid_name": "ethanol",  "vial_name": "ethanol_dye"},
    {"label": "glycerol",     "liquid_name": "glycerol", "vial_name": "glycerol"},
    {"label": "water",        "liquid_name": "water",    "vial_name": "water"},
    {"label": "ethanol",      "liquid_name": "ethanol",  "vial_name": "ethanol"},
]

# Default baseline parameters (used if trial_results.csv not available)
DEFAULT_BASELINE_PARAMS: Dict[str, Dict[str, float]] = {
    "glycerol": {
        "aspirate_speed": 10,
        "dispense_speed": 10,
        "aspirate_wait_time": 0.0,
        "dispense_wait_time": 1.5,
        "pre_asp_air_vol": 0.0,
        "post_asp_air_vol": 0.0,
        "blowout_vol": 0.0,
        "asp_disp_cycles": 0,
        "overaspirate_vol": 0.004,
    },
    "agar_water_4%": {
        "aspirate_speed": 10,
        "dispense_speed": 10,
        "aspirate_wait_time": 0.0,
        "dispense_wait_time": 1.5,
        "pre_asp_air_vol": 0.0,
        "post_asp_air_vol": 0.0,
        "blowout_vol": 0.0,
        "asp_disp_cycles": 0,
        "overaspirate_vol": 0.004,
    },
    "PVA_DMSO": {
        "aspirate_speed": 10,
        "dispense_speed": 10,
        "aspirate_wait_time": 0.0,
        "dispense_wait_time": 1.5,
        "pre_asp_air_vol": 0.0,
        "post_asp_air_vol": 0.0,
        "blowout_vol": 0.0,
        "asp_disp_cycles": 0,
        "overaspirate_vol": 0.004,
    },
    "DMSO": {
        "aspirate_speed": 10,
        "dispense_speed": 10,
        "aspirate_wait_time": 0.0,
        "dispense_wait_time": 1.5,
        "pre_asp_air_vol": 0.0,
        "post_asp_air_vol": 0.0,
        "blowout_vol": 0.0,
        "asp_disp_cycles": 0,
        "overaspirate_vol": 0.004,
    },
    "water": {
        "aspirate_speed": 10,
        "dispense_speed": 10,
        "aspirate_wait_time": 0.0,
        "dispense_wait_time": 1.5,
        "pre_asp_air_vol": 0.0,
        "post_asp_air_vol": 0.0,
        "blowout_vol": 0.0,
        "asp_disp_cycles": 0,
        "overaspirate_vol": 0.004,
    },
    "ethanol": {
        "aspirate_speed": 10,
        "dispense_speed": 10,
        "aspirate_wait_time": 0.0,
        "dispense_wait_time": 1.5,
        "pre_asp_air_vol": 0.0,
        "post_asp_air_vol": 0.0,
        "blowout_vol": 0.0,
        "asp_disp_cycles": 0,
        "overaspirate_vol": 0.004,
    },
}


def _create_protocol_config(liquid_name: str, simulate: bool, vial_name: str, volumes_ml: List[float], show_gui: bool) -> Dict[str, Any]:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["experiment"]["liquid"] = liquid_name
    cfg["experiment"]["volume_targets_ml"] = volumes_ml
    cfg["experiment"]["simulate"] = simulate
    cfg["experiment"]["show_gui"] = show_gui

    with open(HARDWARE_CONFIG_FILE, "r", encoding="utf-8") as f:
        hw_cfg = yaml.safe_load(f)
    hw_cfg["vials"]["source_vial"] = vial_name
    hw_cfg["vials"]["measurement_vial"] = vial_name
    with open(HARDWARE_CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(hw_cfg, f, default_flow_style=False, sort_keys=False)

    return cfg


def _get_tolerance_pct(target_ul: float) -> float:
    # Mirrors experiment_config.yaml volume_ranges for tolerances.
    if 200.0 <= target_ul <= 1000.0:
        return 1.0
    if 60.0 <= target_ul <= 199.0:
        return 2.0
    if 20.0 <= target_ul <= 59.0:
        return 3.0
    if 2.5 <= target_ul <= 19.0:
        return 10.0
    return 10.0


def _build_measure_params(base: Dict[str, float], overaspirate_ml: float) -> Dict[str, Any]:
    return {
        "parameters": {
            "aspirate_speed": base["aspirate_speed"],
            "dispense_speed": base["dispense_speed"],
            "aspirate_wait_time": base["aspirate_wait_time"],
            "dispense_wait_time": base["dispense_wait_time"],
            "pre_asp_air_vol": base["pre_asp_air_vol"],
            "post_asp_air_vol": base["post_asp_air_vol"],
            "blowout_vol": base["blowout_vol"],
            "asp_disp_cycles": int(base["asp_disp_cycles"]),
            "post_retract_wait_time": base.get("post_retract_wait_time", 0.0),
            "retract_speed": base.get("retract_speed", 5.0),
        },
        "overaspirate_vol": float(overaspirate_ml),
    }


def _run_point(protocol: HardwareCalibrationProtocol, state: Dict[str, Any], volume_ml: float, base_params: Dict[str, float], overasp_ml: float) -> List[Dict[str, Any]]:
    params = _build_measure_params(base_params, overasp_ml)
    return protocol.measure(state, volume_ml, params, replicates=REPLICATES)


def _mean_volume_ml(measurements: List[Dict[str, Any]]) -> float:
    return mean([m["volume"] for m in measurements])


def _compute_optimal_overaspirate(point1_ov_ml: float, point1_measured_ml: float, point2_ov_ml: float, point2_measured_ml: float, target_ml: float) -> Tuple[float, float]:
    ov_diff = point2_ov_ml - point1_ov_ml
    if abs(ov_diff) < 1e-9:
        return point1_ov_ml, 0.0

    # Order points low->high by overaspirate (mirrors constraint_calibration.py)
    if point1_ov_ml <= point2_ov_ml:
        low_ov, low_vol = point1_ov_ml, point1_measured_ml
        high_ov, high_vol = point2_ov_ml, point2_measured_ml
    else:
        low_ov, low_vol = point2_ov_ml, point2_measured_ml
        high_ov, high_vol = point1_ov_ml, point1_measured_ml

    slope = (high_vol - low_vol) / (high_ov - low_ov)

    # Physics-based slope bounds (mirrors constraint_calibration.py: 0.5-1.5 uL/uL)
    MIN_SLOPE, MAX_SLOPE = 0.5, 1.5
    if slope < 0.1:
        # Degenerate — fall back to P1 as best guess
        logger.warning("Two-point slope degenerate (%.3f uL/uL), returning P1 overaspirate", slope)
        return point1_ov_ml, slope
    if not (MIN_SLOPE <= slope <= MAX_SLOPE):
        original_slope = slope
        slope = max(MIN_SLOPE, min(MAX_SLOPE, slope))
        logger.warning("Slope %.3f uL/uL outside physical bounds [%.1f, %.1f], clamped to %.3f", original_slope, MIN_SLOPE, MAX_SLOPE, slope)

    # Interpolate from low point (mirrors v2 reference point choice)
    optimal = low_ov + (target_ml - low_vol) / slope
    optimal = max(-0.010, optimal)
    return optimal, slope


def run_two_point_series_demo(simulate: bool = False) -> None:
    global SIMULATE
    if simulate:
        SIMULATE = simulate

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("two_point_series_calibration_demo")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = OUTPUT_DIR / f"two_point_series_demo_details_{timestamp}.csv"
    summary_path = OUTPUT_DIR / f"two_point_series_demo_summary_{timestamp}.csv"

    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    detail_fields = [
        "label", "liquid_name", "vial_name", "target_volume_uL", "point", "replicate",
        "overaspirate_uL", "measured_volume_uL", "elapsed_s", "density_g_mL", "timestamp",
    ]
    summary_fields = [
        "label", "liquid_name", "vial_name", "target_volume_uL", "replicates_per_point",
        "point1_overaspirate_uL", "point1_mean_uL", "point1_shortfall_uL",
        "tolerance_pct", "tolerance_buffer_uL", "spread_uL", "point2_direction",
        "point2_overaspirate_uL", "point2_mean_uL", "slope_mL_per_mL",
        "optimal_overaspirate_uL", "point3_mean_uL", "point3_deviation_pct", "delta_equation",
    ]

    # Write headers immediately so partial data is recoverable on crash
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=detail_fields).writeheader()
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=summary_fields).writeheader()

    protocol = HardwareCalibrationProtocol()

    logger.info("Starting two-point series demo | simulate=%s | replicates=%d", SIMULATE, REPLICATES)
    logger.info("Volumes (uL): %s", ", ".join([str(int(v)) for v in VOLUME_SERIES_UL]))

    first_run = True

    for liquid_entry in LIQUID_SERIES:
        label = liquid_entry["label"]
        liquid_name = liquid_entry["liquid_name"]
        vial_name = liquid_entry["vial_name"]

        if liquid_name not in LIQUIDS:
            raise ValueError(f"Liquid '{liquid_name}' not found in LIQUIDS")

        logger.info("--- Liquid: %s (%s) ---", label, liquid_name)

        # Try to load baseline from trial_results.csv; fall back to defaults
        base = _get_baseline_params_for_liquid(liquid_name)
        if base is None:
            logger.info(f"Using default baseline parameters for '{liquid_name}'")
            if liquid_name not in DEFAULT_BASELINE_PARAMS:
                raise ValueError(f"Missing default baseline params for liquid '{liquid_name}'")
            base = DEFAULT_BASELINE_PARAMS[liquid_name]
        else:
            logger.info(f"Loaded baseline parameters from trial results for '{liquid_name}'")


        show_gui = bool(first_run and (not SIMULATE))
        first_run = False

        cfg = _create_protocol_config(
            liquid_name=liquid_name,
            simulate=SIMULATE,
            vial_name=vial_name,
            volumes_ml=[v / 1000.0 for v in VOLUME_SERIES_UL],
            show_gui=show_gui,
        )

        state = protocol.initialize(cfg)

        try:
            for volume_ul in VOLUME_SERIES_UL:
                volume_ml = volume_ul / 1000.0

                # Point 1
                point1_ov_ml = float(base["overaspirate_vol"])
                point1_measurements = _run_point(protocol, state, volume_ml, base, point1_ov_ml)
                point1_mean_ml = _mean_volume_ml(point1_measurements)

                target_ul = volume_ul
                point1_mean_ul = point1_mean_ml * 1000.0
                shortfall_ul = target_ul - point1_mean_ul

                tolerance_pct = _get_tolerance_pct(target_ul)
                tolerance_buffer_ul = target_ul * tolerance_pct / 100.0
                spread_ul = max(abs(shortfall_ul) + tolerance_buffer_ul, 2.0)

                if shortfall_ul > 0:
                    point2_ov_ml = point1_ov_ml + (spread_ul / 1000.0)
                    point2_direction = "increased"
                else:
                    point2_ov_ml = point1_ov_ml - (spread_ul / 1000.0)
                    point2_direction = "decreased"
                point2_ov_ml = max(-0.010, point2_ov_ml)

                # Point 2
                point2_measurements = _run_point(protocol, state, volume_ml, base, point2_ov_ml)
                point2_mean_ml = _mean_volume_ml(point2_measurements)

                optimal_ov_ml, slope = _compute_optimal_overaspirate(
                    point1_ov_ml,
                    point1_mean_ml,
                    point2_ov_ml,
                    point2_mean_ml,
                    volume_ml,
                )

                # Point 3: validate at optimal overaspirate
                point3_measurements = _run_point(protocol, state, volume_ml, base, optimal_ov_ml)
                point3_mean_ml = _mean_volume_ml(point3_measurements)
                point3_mean_ul = point3_mean_ml * 1000.0
                point3_deviation_pct = abs(point3_mean_ul - target_ul) / target_ul * 100.0

                logger.info(
                    "Liquid=%s Volume=%.1fuL | P1=%.2fuL@%.2fuL ov | P2=%.2fuL@%.2fuL ov | opt_ov=%.2fuL | P3=%.2fuL (dev=%.2f%%)",
                    liquid_name,
                    volume_ul,
                    point1_mean_ul,
                    point1_ov_ml * 1000.0,
                    point2_mean_ml * 1000.0,
                    point2_ov_ml * 1000.0,
                    optimal_ov_ml * 1000.0,
                    point3_mean_ul,
                    point3_deviation_pct,
                )

                for point_name, overasp_ml, measurements in [
                    ("point_1", point1_ov_ml, point1_measurements),
                    ("point_2", point2_ov_ml, point2_measurements),
                    ("point_3_validation", optimal_ov_ml, point3_measurements),
                ]:
                    for m in measurements:
                        detail_rows.append(
                            {
                                "label": label,
                                "liquid_name": liquid_name,
                                "vial_name": vial_name,
                                "target_volume_uL": target_ul,
                                "point": point_name,
                                "replicate": m["replicate"],
                                "overaspirate_uL": overasp_ml * 1000.0,
                                "measured_volume_uL": m["volume"] * 1000.0,
                                "elapsed_s": m["elapsed_s"],
                                "density_g_mL": LIQUIDS[liquid_name]["density"],
                                "timestamp": m.get("start_time", datetime.now().isoformat()),
                            }
                        )

                summary_rows.append(
                    {
                        "label": label,
                        "liquid_name": liquid_name,
                        "vial_name": vial_name,
                        "target_volume_uL": target_ul,
                        "replicates_per_point": REPLICATES,
                        "point1_overaspirate_uL": point1_ov_ml * 1000.0,
                        "point1_mean_uL": point1_mean_ul,
                        "point1_shortfall_uL": shortfall_ul,
                        "tolerance_pct": tolerance_pct,
                        "tolerance_buffer_uL": tolerance_buffer_ul,
                        "spread_uL": spread_ul,
                        "point2_direction": point2_direction,
                        "point2_overaspirate_uL": point2_ov_ml * 1000.0,
                        "point2_mean_uL": point2_mean_ml * 1000.0,
                        "slope_mL_per_mL": slope,
                        "optimal_overaspirate_uL": optimal_ov_ml * 1000.0,
                        "point3_mean_uL": point3_mean_ul,
                        "point3_deviation_pct": point3_deviation_pct,
                        "delta_equation": "spread_ul=max(abs(shortfall_ul)+tolerance_buffer_ul,2.0)",
                    }
                )

                # Flush completed rows to disk after every volume
                with open(detail_path, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=detail_fields).writerows(detail_rows[-REPLICATES * 3:])
                with open(summary_path, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=summary_fields).writerows(summary_rows[-1:])
                logger.info("Flushed data for %s %.0fuL", liquid_name, volume_ul)

        finally:
            protocol.wrapup(state)

    logger.info("Saved detail CSV: %s", detail_path)
    logger.info("Saved summary CSV: %s", summary_path)
    logger.info("Two-point series demo complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v2-style two-point series calibration demo")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
    args = parser.parse_args()

    run_two_point_series_demo(simulate=args.simulate)

"""Two-point overaspirate calibration demo across a series of volumes.

This script demonstrates the v2 two-point system in a compartmentalized form,
without running full screening/optimizer flows. It uses HardwareCalibrationProtocol
for measurements and applies the same Point 2 spread logic used in v2 experiment:

    spread_ul = max(abs(shortfall_ul) + tolerance_buffer_ul, 2.0)

Volumes tested (uL): 25, 50, 75, 100, 150
Replicates per point: 3

Usage:
  python calibration_modular_v2/two_point_series_calibration_demo.py
  python calibration_modular_v2/two_point_series_calibration_demo.py --simulate
"""

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration_modular_v2.calibration_protocol_northrobot import HardwareCalibrationProtocol, LIQUIDS

SIMULATE = False
REPLICATES = 3
VOLUME_SERIES_UL = [25.0, 50.0, 75.0, 100.0, 150.0]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
CONFIG_FILE = Path(__file__).resolve().parent / "experiment_config.yaml"
HARDWARE_CONFIG_FILE = Path(__file__).resolve().parent / "north_robot_hardware.yaml"

LIQUID_SERIES: List[Dict[str, str]] = [
    {"label": "glycerol", "liquid_name": "glycerol", "vial_name": "glycerol"},
    {"label": "alginate_4pct", "liquid_name": "agar_water_4%", "vial_name": "agar_water_4%"},
    {"label": "PVA_dmso", "liquid_name": "PVA_DMSO", "vial_name": "PVA_DMSO"},
    {"label": "dmso", "liquid_name": "DMSO", "vial_name": "DMSO"},
    {"label": "water", "liquid_name": "water", "vial_name": "water"},
    {"label": "ethanol", "liquid_name": "ethanol", "vial_name": "ethanol"},
]

# Placeholder "existing best conditions". Keep as a single source and replace later.
# Values are in protocol units / mL.
BASELINE_PARAMS_BY_LIQUID: Dict[str, Dict[str, float]] = {
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


def _create_protocol_config(liquid_name: str, simulate: bool, vial_name: str, volume_ml: float, show_gui: bool) -> Dict[str, Any]:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["experiment"]["liquid"] = liquid_name
    cfg["experiment"]["volume_targets_ml"] = [volume_ml]
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

    slope = (point2_measured_ml - point1_measured_ml) / ov_diff
    if abs(slope) < 1e-9:
        return point1_ov_ml, slope

    optimal = point1_ov_ml + (target_ml - point1_measured_ml) / slope
    # Keep v2 demo conservative and non-explosive; allow same v2 lower bound.
    optimal = max(-0.010, optimal)
    return optimal, slope


def run_two_point_series_demo(simulate: bool = False, show_first_gui: bool = False) -> None:
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
        if liquid_name not in BASELINE_PARAMS_BY_LIQUID:
            raise ValueError(f"Missing baseline params for liquid '{liquid_name}'")

        logger.info("--- Liquid: %s (%s) ---", label, liquid_name)

        base = BASELINE_PARAMS_BY_LIQUID[liquid_name]

        for volume_ul in VOLUME_SERIES_UL:
            volume_ml = volume_ul / 1000.0
            show_gui = bool(show_first_gui and first_run and (not SIMULATE))
            first_run = False

            cfg = _create_protocol_config(
                liquid_name=liquid_name,
                simulate=SIMULATE,
                vial_name=vial_name,
                volume_ml=volume_ml,
                show_gui=show_gui,
            )

            state = protocol.initialize(cfg)

            try:
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

                logger.info(
                    "Liquid=%s Volume=%.1fuL | P1=%.2fuL@%.2fuL ov | P2=%.2fuL@%.2fuL ov | spread=%.2fuL | opt_ov=%.2fuL",
                    liquid_name,
                    volume_ul,
                    point1_mean_ul,
                    point1_ov_ml * 1000.0,
                    point2_mean_ml * 1000.0,
                    point2_ov_ml * 1000.0,
                    spread_ul,
                    optimal_ov_ml * 1000.0,
                )

                for point_name, overasp_ml, measurements in [
                    ("point_1", point1_ov_ml, point1_measurements),
                    ("point_2", point2_ov_ml, point2_measurements),
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
                        "delta_equation": "spread_ul=max(abs(shortfall_ul)+tolerance_buffer_ul,2.0)",
                    }
                )
            finally:
                protocol.wrapup(state)

    detail_fields = [
        "label",
        "liquid_name",
        "vial_name",
        "target_volume_uL",
        "point",
        "replicate",
        "overaspirate_uL",
        "measured_volume_uL",
        "elapsed_s",
        "density_g_mL",
        "timestamp",
    ]

    summary_fields = [
        "label",
        "liquid_name",
        "vial_name",
        "target_volume_uL",
        "replicates_per_point",
        "point1_overaspirate_uL",
        "point1_mean_uL",
        "point1_shortfall_uL",
        "tolerance_pct",
        "tolerance_buffer_uL",
        "spread_uL",
        "point2_direction",
        "point2_overaspirate_uL",
        "point2_mean_uL",
        "slope_mL_per_mL",
        "optimal_overaspirate_uL",
        "delta_equation",
    ]

    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    logger.info("Saved detail CSV: %s", detail_path)
    logger.info("Saved summary CSV: %s", summary_path)
    logger.info("Two-point series demo complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v2-style two-point series calibration demo")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
    parser.add_argument(
        "--show-first-gui",
        action="store_true",
        help="Show status/config GUI for the first liquid-volume run (disabled automatically in --simulate mode)",
    )
    args = parser.parse_args()

    run_two_point_series_demo(simulate=args.simulate, show_first_gui=args.show_first_gui)

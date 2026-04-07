"""
Baseline workflow for glycerol Sobol benchmark execution.

Focus:
- Always use status GUI before hardware actions (optional toggle)
- Optionally set pipette usage counters (for partially used tip racks)
- Execute parameterized aspirate/dispense tests from Sobol CSV files
- Dispense into a clamped vial on scale (no plate handling)
"""

import os
import sys
from datetime import datetime
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from master_usdl_coordinator import Lash_E
from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy
from pipetting_data.pipetting_parameters import PipettingParameters


# ===== WORKFLOW CONFIG (auto-exported to workflow_configs on first run) =====
SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/calibration_vials.csv"
SHOW_GUI = True

# Liquid/validation setup
LIQUID_TYPE = "glycerol"
SOURCE_VIAL = "liquid_source"
DESTINATION_VIAL = "measurement_vial"
REPLICATES_PER_ROW = 1
SWITCH_PIPET_BETWEEN_REPLICATES = False

# Sobol benchmark files
SOBOL_200UL_CSV = "workflows/Glycerol_benchmark/Glycerin_Sobol_Parameters_200uL.csv"
SOBOL_1000UL_CSV = "workflows/Glycerol_benchmark/Glycerin_Sobol_Parameters_1000uL.csv"

# Simulation-first safety controls
# Set MAX_ROWS_PER_FILE=None to run the full CSV (5120 rows/file).
MAX_ROWS_PER_FILE = 20 if SIMULATE else None

# Tip inventory baseline
# If True, the script sets robot counters so the next picked tip aligns with
# physical rack state. Example: 50 means 50 tips already consumed.
SET_TIP_BASELINE = True
START_SMALL_TIPS_USED = 50
START_LARGE_TIPS_USED = 50


def _load_sobol_dataframe(csv_path, max_rows=None):
    """Load Sobol CSV, drop unnamed index column, and enforce required schema."""
    df = pd.read_csv(csv_path)
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed") or str(c) == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_cols = [
        "aspirate_speed",
        "dispense_speed",
        "aspirate_wait_time",
        "dispense_wait_time",
        "pre_asp_air_vol",
        "blowout_vol",
        "overaspirate_vol",
        "vol",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    if max_rows is not None:
        df = df.head(int(max_rows)).copy()

    return df


def _row_to_parameters(row):
    """
    Convert Sobol row to PipettingParameters.

    CSV units for pre_asp_air_vol, blowout_vol, overaspirate_vol, vol are in uL.
    The robot API expects mL, so convert uL -> mL.
    """
    params = PipettingParameters(
        aspirate_speed=int(row["aspirate_speed"]),
        dispense_speed=int(row["dispense_speed"]),
        aspirate_wait_time=float(row["aspirate_wait_time"]),
        dispense_wait_time=float(row["dispense_wait_time"]),
        pre_asp_air_vol=float(row["pre_asp_air_vol"]) / 1000.0,
        blowout_vol=float(row["blowout_vol"]) / 1000.0,
        overaspirate_vol=float(row["overaspirate_vol"]) / 1000.0,
    )
    volume_ml = float(row["vol"]) / 1000.0
    return params, volume_ml


def _reset_simulation_vial_volumes(robot, source_vial_name, destination_vial_name):
    """
    Keep simulation rows independent by resetting source/destination vial volumes.
    This prevents artificial source depletion and destination overflow in long runs.
    """
    try:
        source_idx = robot.normalize_vial_index(source_vial_name)
        dest_idx = robot.normalize_vial_index(destination_vial_name)
        robot.VIAL_DF.at[source_idx, "vial_volume"] = 7.5
        robot.VIAL_DF.at[dest_idx, "vial_volume"] = 0.0
    except Exception as exc:
        print(f"Simulation vial reset warning: {exc}")


def _set_tip_baseline(robot, total_small_used, total_large_used):
    """Set pipette usage counters across all racks of a tip type."""
    rack_groups = {"small_tip": [], "large_tip": []}
    for rack_name, rack_cfg in robot.PIPET_RACKS.items():
        tip_type = rack_cfg.get("tip_type")
        if tip_type in rack_groups:
            rack_groups[tip_type].append(rack_name)

    for tip_type in rack_groups:
        rack_groups[tip_type].sort()

    desired = {
        "small_tip": int(total_small_used),
        "large_tip": int(total_large_used),
    }

    for tip_type, total_used in desired.items():
        remaining = max(0, total_used)
        for rack_name in rack_groups[tip_type]:
            max_tips = int(robot.PIPET_RACKS[rack_name].get("num_tips", 48))
            rack_used = min(remaining, max_tips)
            robot.PIPETS_USED[rack_name] = rack_used
            remaining -= rack_used

    robot.save_robot_status()


def _print_tip_status(robot, title):
    print(f"\n{title}")
    for rack_name in sorted(robot.PIPET_RACKS.keys()):
        rack_cfg = robot.PIPET_RACKS[rack_name]
        tip_type = rack_cfg.get("tip_type")
        used = int(robot.PIPETS_USED.get(rack_name, 0))
        capacity = int(rack_cfg.get("num_tips", 48))
        print(f"  {rack_name}: {tip_type}, used={used}/{capacity}, remaining={capacity - used}")


def run_baseline():
    print("Starting glycerol Sobol benchmark baseline workflow")
    print(f"Simulation mode: {SIMULATE}")
    print(f"Show GUI: {SHOW_GUI}")

    lash_e = Lash_E(
        INPUT_VIAL_STATUS_FILE,
        initialize_track=False,
        initialize_biotek=False,
        initialize_t8=False,
        initialize_p2=False,
        simulate=SIMULATE,
        workflow_globals=globals(),
        workflow_name="glycerol_dispense_baseline",
        show_gui=SHOW_GUI,
    )

    if not hasattr(lash_e, "nr_robot") or lash_e.nr_robot is None:
        print("Workflow stopped before robot initialization (likely canceled from GUI).")
        return None

    lash_e.nr_robot.home_robot_components()
    _print_tip_status(lash_e.nr_robot, "Tip counters before baseline setup:")

    if SET_TIP_BASELINE:
        _set_tip_baseline(
            lash_e.nr_robot,
            total_small_used=START_SMALL_TIPS_USED,
            total_large_used=START_LARGE_TIPS_USED,
        )
        _print_tip_status(lash_e.nr_robot, "Tip counters after baseline setup:")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join("output", f"glycerol_sobol_benchmark_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    print("\nLoading Sobol CSV files...")
    df_200 = _load_sobol_dataframe(SOBOL_200UL_CSV, max_rows=MAX_ROWS_PER_FILE)
    df_1000 = _load_sobol_dataframe(SOBOL_1000UL_CSV, max_rows=MAX_ROWS_PER_FILE)
    print(f"  200uL file rows loaded: {len(df_200)}")
    print(f"  1000uL file rows loaded: {len(df_1000)}")

    campaigns = [
        ("200uL", df_200),
        ("1000uL", df_1000),
    ]

    campaign_results = {}

    for campaign_name, campaign_df in campaigns:
        print(f"\n=== Running campaign: {campaign_name} ===")
        campaign_folder = os.path.join(output_folder, campaign_name)
        os.makedirs(campaign_folder, exist_ok=True)

        per_row_summaries = []
        for row_idx, row in campaign_df.iterrows():
            params, volume_ml = _row_to_parameters(row)
            volume_ul = volume_ml * 1000.0
            print(f"[{campaign_name}] Row {row_idx + 1}/{len(campaign_df)} | vol={volume_ul:.2f}uL")

            if SIMULATE:
                _reset_simulation_vial_volumes(lash_e.nr_robot, SOURCE_VIAL, DESTINATION_VIAL)

            row_output_folder = os.path.join(campaign_folder, f"row_{row_idx + 1:05d}")
            os.makedirs(row_output_folder, exist_ok=True)
            validation_output_folder = None if SIMULATE else row_output_folder

            try:
                row_result = validate_pipetting_accuracy(
                    lash_e=lash_e,
                    source_vial=SOURCE_VIAL,
                    destination_vial=DESTINATION_VIAL,
                    liquid_type=LIQUID_TYPE,
                    volumes_ml=[volume_ml],
                    replicates=REPLICATES_PER_ROW,
                    output_folder=validation_output_folder,
                    switch_pipet=SWITCH_PIPET_BETWEEN_REPLICATES,
                    save_raw_data=not SIMULATE,
                    compensate_overvolume=False,
                    condition_tip_enabled=False,
                    parameters=params,
                    adaptive_correction=False,
                )
            except Exception as exc:
                print(f"[{campaign_name}] Row {row_idx + 1} failed: {exc}")
                if not SIMULATE:
                    raise
                row_result = {}

            per_row_summaries.append({
                "row_index": int(row_idx),
                "volume_ul": float(volume_ul),
                "r_squared": row_result.get("r_squared"),
                "mean_accuracy_pct": row_result.get("mean_accuracy_pct"),
                "status": "ok" if row_result else "failed",
            })

            if lash_e.nr_robot.HELD_PIPET_TYPE is not None:
                lash_e.nr_robot.remove_pipet()

        summary_df = pd.DataFrame(per_row_summaries)
        summary_path = os.path.join(campaign_folder, "campaign_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        campaign_results[campaign_name] = {
            "rows_processed": len(campaign_df),
            "summary_path": summary_path,
        }
        print(f"[{campaign_name}] complete. Summary: {summary_path}")

    lash_e.nr_robot.move_home()

    try:
        lash_e.nr_robot.return_vial_home(SOURCE_VIAL)
        lash_e.nr_robot.return_vial_home(DESTINATION_VIAL)
    except Exception as exc:
        print(f"Vial return warning: {exc}")

    print("\nBaseline workflow complete")
    print(f"Output folder: {output_folder}")
    for campaign_name, summary in campaign_results.items():
        print(f"  {campaign_name}: rows={summary['rows_processed']}, summary={summary['summary_path']}")

    return {
        "output_folder": output_folder,
        "results": campaign_results,
        "workflow_complete": True,
    }


if __name__ == "__main__":
    run_baseline()

"""
Redo workflow for glycerol Sobol benchmark — runs specific row indices only.

Use this when you want to:
- Fill gaps (indices never measured)
- Re-run specific conditions for any reason

Results are appended to the same incremental_results.csv as the main campaign,
with run_type = "original" (for gap fills) or "redo" (for deliberate re-runs).
"""

import os
import sys
import time
import glob
import shutil
import signal
from datetime import datetime, timedelta
import pandas as pd
import yaml

sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
from pipetting_data.pipetting_parameters import PipettingParameters
import slack_agent


# ===== CONFIG =====
SIMULATE = False

# Which campaign to run
CAMPAIGN_TYPE = "1000uL" #select "200uL" or "1000uL" 

# Specific Sobol row indices to run (0-based, matching row_index in the results CSV)
#ROW_INDICES = [938, 939, 940, 941, 942, 943, 944, 945, 946,
#              2442, 2447, 2455, 2456, 2457, 2462,
#              5032, 5044]  # 200uL gaps "original"

# This was the remaining glycerol samples we didn't do yet. Accidentally stopped it prematurely and started the redos
#ROW_INDICES = [5109,5110,5111,5112,5113,5114,5115,5116,5117,5118,5119] #5110-5119

#ROW_INDICES = [80, 81, 82, 83]  # 1000uL gaps "original"

# 1000uL near-zero measurements to redo (|measured_volume_ul| < 20uL, i.e. essentially nothing dispensed).
# Criterion: abs(measured_volume_ul) < 20. Total: 108 rows.
# Two large clusters (scale/automation failure on those sessions):
#   Jul 23 2026: rows 4314-4359 (46 rows)
#   Jul 24 2026: rows 4462-4481 (20 rows)
# Plus scattered individual failures throughout the campaign, and row 3175 (extreme -10254uL outlier).
# Scattered failures Apr 30 - Jun 8 (41 rows)
# 1000uL near-zero redos — set CAMPAIGN_TYPE = "1000uL" and RUN_TYPE = "redo"

#ROW_INDICES = [#101, 105, 129, 218, 235, 251, 312, 324, 342,
              #385, 392, 400, 423, 424, 456,
              #485, 497, 502, 594,
              #633, 714, 722, 732, 744, 775,
              #843, 853, 857,
              #1019, 1190, 1552, 
              #1674, 1731, 1855, 2041,
#     # Scattered failures Jul 14 - Aug 5 (6 rows)
              #3175, 3836, 3884, 4028, 4205, 4649, 4963,
#     # Jul 23 cluster - scale failure (46 rows)
              #4314, 4315, 4316, 4317, 4318, 4319, 4320, 4321, 4322, 4323,
              #4324, 4325, 4326, 4327, 4328, 4329, 4330, 4331, 4332, 4333,
              #4334, 4335, 4336, 
              #4337, 4338, 4339, 4340, 4341, 4342, 4343,
              #4344, 4345, 4346, 4347, 4348, 4349, 4350, 4351, 4352, 4353,
              #4354, 4355, 4356, 4357, 4358, 4359,
#     # Jul 24 cluster - scale failure (20 rows)
              #4462, 4463, 4464, 4465, 4466, 4467, 4468, 4469, 4470, 4471,
              #4472, 4473, 4474, 4475, 4476, 4477, 4478, 4479, 4480, 4481,]  


ROW_INDICES = [300]  # 1000uL redo: code-hang outlier (elapsed=1276s, status=failed)

# 200uL near-zero measurements to redo (|measured_volume_ul| < 10uL).
# Only one small cluster found (Jul 24, scale taring issue ending a session - 5 rows).
# All other 124 near-zero rows are isolated scatter across the campaign (expected noise, not flagged).
# ROW_INDICES = [4538, 4539, 4540, 4541, 4542]  # 200uL Jul 24 cluster — set CAMPAIGN_TYPE = "200uL" and RUN_TYPE = "redo"

#Other sets to consider redoes:

# Label for these runs - "original" if filling genuine gaps, "redo" if re-running existing data
RUN_TYPE = "redo"

INPUT_VIAL_STATUS_FILE = "status/calibration_vials.csv"
CAMPAIGN_OUTPUT_FOLDER = "output/glycerol_sobol_campaign"

LIQUID_TYPE = "glycerol"
VIAL_LOW_THRESHOLD = 2.0  # mL

SOBOL_200UL_CSV = "inputs/Glycerin_Sobol_Parameters_200uL.csv"
SOBOL_1000UL_CSV = "inputs/Glycerin_Sobol_Parameters_1000uL.csv"

GLYCEROL_OPENED_DATE = "2026-03-24"
GLYCEROL_DENSITY = 1.26  # g/mL

RETRACT_SPEED = 2
POST_ASP_WAIT_TIME = 5
POST_ASP_AIR_VOL = 0

ADJUST_VOLUME = True
UPDATE_EVERY_NUM_PIPS = 12

MQTT_LOG_FILE = "C:\\Users\\Imaging Controller\\Desktop\\m5stack\\mqtt_log.csv"

MAX_ROWS_FOR_SIMULATION = 5

# Global interrupt state
_interrupt_state = {
    "lash_e": None, 
    "current_vial_name": None,
    "interrupted": False,
}


def _interrupt_handler(signum, frame):
    _interrupt_state["interrupted"] = True
    print("\n\nINTERRUPT DETECTED - Shutting down gracefully...")

    lash_e = _interrupt_state.get("lash_e")
    current_vial_name = _interrupt_state.get("current_vial_name")

    if lash_e and current_vial_name:
        try:
            lash_e.nr_robot.remove_pipet()
            lash_e.nr_robot.recap_clamp_vial()
            lash_e.nr_robot.return_vial_home(current_vial_name)
            lash_e.nr_robot.move_home()
            print("Cleanup complete - vial capped and home")
        except Exception as e:
            print(f"Error during interrupt cleanup: {e}")

    sys.exit(0)


signal.signal(signal.SIGINT, _interrupt_handler)


def _load_sobol_rows_by_index(csv_path, indices):
    """Load specific rows from a Sobol CSV by their 0-based integer index."""
    df = pd.read_csv(csv_path)
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed") or str(c) == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_cols = [
        "aspirate_speed", "dispense_speed", "aspirate_wait_time",
        "dispense_wait_time", "pre_asp_air_vol_uL", "blowout_vol_uL",
        "overaspirate_vol_uL", "vol_uL",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    out_of_range = [i for i in indices if i >= len(df) or i < 0]
    if out_of_range:
        raise ValueError(f"Row indices out of range for {csv_path} ({len(df)} rows): {out_of_range}")

    selected = df.iloc[indices].copy()
    selected.index = list(range(len(selected)))  # Reset index for clean iteration
    return selected, indices  # Return both df and the original indices list


def _row_to_parameters(row):
    """Convert Sobol row to PipettingParameters. Identical to main workflow."""
    params = PipettingParameters(
        aspirate_speed=int(row["aspirate_speed"]),
        dispense_speed=int(row["dispense_speed"]),
        aspirate_wait_time=float(row["aspirate_wait_time"]),
        dispense_wait_time=float(row["dispense_wait_time"]),
        pre_asp_air_vol=float(row["pre_asp_air_vol_uL"]) / 1000.0,
        blowout_vol=float(row["blowout_vol_uL"]) / 1000.0,
        overaspirate_vol=float(row["overaspirate_vol_uL"]) / 1000.0,
        retract_speed=RETRACT_SPEED,
        post_retract_wait_time=POST_ASP_WAIT_TIME,
        post_asp_air_vol=POST_ASP_AIR_VOL,
    )
    volume_ml = float(row["vol_uL"]) / 1000.0

    critical_params = ["aspirate_speed", "dispense_speed", "overaspirate_vol"]
    for param in critical_params:
        value = getattr(params, param)
        if value is None:
            raise ValueError(f"CRITICAL: {param} is None - this would trigger silent defaults!")

    return params, volume_ml


def _check_and_swap_vials(lash_e, current_vial_number, current_vial_name):
    """Check if current vial is low and swap to next vial if needed."""
    try:
        current_volume = lash_e.nr_robot.get_vial_info(current_vial_name, "vial_volume")
        if current_volume is not None and current_volume <= VIAL_LOW_THRESHOLD:
            print(f"\nVIAL SWAP: {current_vial_name} low at {current_volume:.2f}mL (<= {VIAL_LOW_THRESHOLD}mL)")
            slack_agent.send_slack_message("GLYCEROL VIAL NEEDS TO BE REPLACED!")
            input("Waiting...")
            lash_e.nr_robot.return_vial_home(current_vial_name)
            new_vial_number = current_vial_number + 1
            new_vial_name = f"vial_{new_vial_number}"
            lash_e.nr_robot.move_vial_to_location(new_vial_name, "clamp", 0)
            if not lash_e.nr_robot.is_vial_pipetable(new_vial_name):
                lash_e.nr_robot.uncap_clamp_vial()
            print(f"SWAP complete: {current_vial_name} -> {new_vial_name}")
            return new_vial_number, new_vial_name
        else:
            return current_vial_number, current_vial_name
    except Exception as e:
        print(f"Warning: Vial swap check failed: {e}")
        return current_vial_number, current_vial_name


def _get_latest_environmental_data():
    """Get most recent environmental data from MQTT log."""
    try:
        if not os.path.exists(MQTT_LOG_FILE):
            print(f"Environmental data file not found: {MQTT_LOG_FILE}")
            return None
        df = pd.read_csv(MQTT_LOG_FILE)
        if len(df) == 0:
            print("Environmental data file is empty")
            return None
        latest = df.iloc[-1]
        timestamp = pd.to_datetime(latest["Timestamp"]).to_pydatetime()
        return {
            "temp_c": float(latest["sht_temp_c"]) if pd.notna(latest["sht_temp_c"]) else None,
            "humidity_pct": float(latest["sht_rh"]) if pd.notna(latest["sht_rh"]) else None,
            "pressure_pa": float(latest["bmp_pa"]) if pd.notna(latest["bmp_pa"]) else None,
            "Timestamp": timestamp,
        }
    except Exception as e:
        print(f"Could not read environmental data: {e}")
        return None


def _check_environmental_data_freshness():
    """Check if environmental data is fresh. Auto-passes in simulation."""
    if SIMULATE:
        print("Simulation mode: Skipping environmental data check")
        return True
    env_data = _get_latest_environmental_data()
    if env_data is None:
        return False
    last_time = env_data["Timestamp"]
    time_diff = datetime.now() - last_time
    if time_diff > timedelta(hours=1):
        print(f"WARNING: Last environmental data is {time_diff} old (last: {last_time})")
        response = input("Environmental data is stale. Continue anyway? (y/n): ").strip().lower()
        if response not in ["y", "yes"]:
            print("Stopping workflow due to stale environmental data.")
            return False
    else:
        print(f"Environmental data is fresh (last reading: {last_time})")
    return True


def _copy_latest_mass_data(campaign_folder, row_num):
    """Copy most recent mass data files to organized campaign folder."""
    mass_data_dir = os.path.join(campaign_folder, "mass_time_data")
    os.makedirs(mass_data_dir, exist_ok=True)

    mass_files = glob.glob("output/mass_measurements/*/mass_data_vial_*.csv")
    plot_files = glob.glob("output/mass_measurements/*/mass_plot_vial_*.png")

    if mass_files:
        latest_mass_file = max(mass_files, key=os.path.getmtime)
        latest_plot_file = max(plot_files, key=os.path.getmtime) if plot_files else None

        mass_filename = f"mass_data_row_{row_num:04d}_redo.csv"
        plot_filename = f"mass_plot_row_{row_num:04d}_redo.png"

        shutil.copy2(latest_mass_file, os.path.join(mass_data_dir, mass_filename))
        if latest_plot_file:
            shutil.copy2(latest_plot_file, os.path.join(mass_data_dir, plot_filename))

        print(f"    Copied mass data to: {mass_filename}")
        return mass_filename
    else:
        print("    Warning: No mass data files found to copy")
        return None


def _save_row_result_immediately(campaign_folder, row_idx, volume_ul, row_result, params, run_type):
    """Append measurement result to incremental_results.csv with run_type label."""
    try:
        incremental_csv_path = os.path.join(campaign_folder, "incremental_results.csv")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        opened_date = datetime.strptime(GLYCEROL_OPENED_DATE, "%Y-%m-%d")
        days_since_opened = (datetime.now() - opened_date).days

        row_data = {
            "row_index": int(row_idx),
            "timestamp": timestamp,
            "volume_ul_target": float(volume_ul),
            "aspirate_speed": params.aspirate_speed,
            "dispense_speed": params.dispense_speed,
            "aspirate_wait_time": params.aspirate_wait_time,
            "dispense_wait_time": params.dispense_wait_time,
            "pre_asp_air_vol_uL": params.pre_asp_air_vol * 1000,
            "post_asp_air_vol_uL": params.post_asp_air_vol * 1000,
            "overaspirate_vol_uL": params.overaspirate_vol * 1000,
            "blowout_vol_uL": params.blowout_vol * 1000,
            "retract_speed": params.retract_speed,
            "post_retract_wait_time": params.post_retract_wait_time,
            "measured_volume_ml": row_result.get("measured_volume_ml"),
            "measured_volume_ul": row_result.get("measured_volume_ml") * 1000 if row_result.get("measured_volume_ml") is not None else None,
            "accuracy_pct": row_result.get("accuracy_pct"),
            "elapsed_s": row_result.get("elapsed_s"),
            "measured_mass_g": row_result.get("measured_mass_g"),
            "temp_c": row_result.get("temp_c"),
            "humidity_pct": row_result.get("humidity_pct"),
            "pressure_pa": row_result.get("pressure_pa"),
            "pre_baseline_std": row_result.get("pre_baseline_std"),
            "post_baseline_std": row_result.get("post_baseline_std"),
            "pre_stable_pct": row_result.get("pre_stable_pct"),
            "post_stable_pct": row_result.get("post_stable_pct"),
            "glycerol_opened_date": GLYCEROL_OPENED_DATE,
            "days_bottle_opened": days_since_opened,
            "mass_data_file": row_result.get("mass_data_file"),
            "status": "ok" if row_result else "failed",
            "run_type": run_type,
        }

        pd.DataFrame([row_data]).to_csv(incremental_csv_path, mode="a", header=False, index=False)
        print(f"    Row {row_idx} saved ({run_type})")

    except Exception as e:
        print(f"    Could not save row {row_idx} immediately: {e}")


def run_redo():
    print(f"Starting glycerol redo workflow")
    print(f"Campaign: {CAMPAIGN_TYPE} | run_type: {RUN_TYPE} | Indices: {ROW_INDICES}")
    print(f"Simulation mode: {SIMULATE}")

    if not ROW_INDICES:
        print("ROW_INDICES is empty — nothing to run.")
        return

    if not _check_environmental_data_freshness():
        print("Stopping workflow due to environmental data issues.")
        return

    # Resolve paths
    base_folder = CAMPAIGN_OUTPUT_FOLDER
    if SIMULATE:
        base_folder = CAMPAIGN_OUTPUT_FOLDER + "_simulate"

    campaign_folder = os.path.join(base_folder, CAMPAIGN_TYPE)
    os.makedirs(campaign_folder, exist_ok=True)

    incremental_csv_path = os.path.join(campaign_folder, "incremental_results.csv")
    if not os.path.exists(incremental_csv_path):
        raise FileNotFoundError(
            f"incremental_results.csv not found at {incremental_csv_path}. "
            "Run the main campaign workflow first."
        )

    csv_path = SOBOL_200UL_CSV if CAMPAIGN_TYPE == "200uL" else SOBOL_1000UL_CSV

    # Load only the requested rows from the Sobol CSV
    indices_to_run = ROW_INDICES if not SIMULATE else ROW_INDICES[:MAX_ROWS_FOR_SIMULATION]
    campaign_df, original_indices = _load_sobol_rows_by_index(csv_path, indices_to_run)

    print(f"Loaded {len(campaign_df)} rows to process")

    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False)

    if not hasattr(lash_e, "nr_robot") or lash_e.nr_robot is None:
        print("Workflow stopped before robot initialization.")
        return

    lash_e.nr_robot.home_robot_components()

    current_vial_number = 0
    current_vial_name = f"vial_{current_vial_number}"
    _interrupt_state["lash_e"] = lash_e
    _interrupt_state["current_vial_name"] = current_vial_name

    lash_e.nr_robot.move_vial_to_location(current_vial_name, "clamp", 0)
    if not lash_e.nr_robot.is_vial_pipetable(current_vial_name):
        lash_e.nr_robot.uncap_clamp_vial()

    # Notify Slack
    if not SIMULATE:
        try:
            slack_agent.send_slack_message(
                f"Starting Glycerol Redo: {CAMPAIGN_TYPE}\n"
                f"run_type: {RUN_TYPE}\n"
                f"Indices: {indices_to_run}\n"
                f"Count: {len(campaign_df)}"
            )
        except Exception as e:
            print(f"Could not send Slack startup notification: {e}")

    total = len(campaign_df)
    for pos, (_, row) in enumerate(campaign_df.iterrows()):
        if _interrupt_state["interrupted"]:
            print("Interrupt flag detected — stopping.")
            break

        row_idx = original_indices[pos]  # The actual Sobol row number
        params, volume_ml = _row_to_parameters(row)
        volume_ul = volume_ml * 1000.0

        print(f"\n[{CAMPAIGN_TYPE}] {pos + 1}/{total} | row_index={row_idx} | vol={volume_ul:.2f}uL | run_type={RUN_TYPE}")
        print(f"    aspirate_speed={params.aspirate_speed}, dispense_speed={params.dispense_speed}")
        print(f"    aspirate_wait_time={params.aspirate_wait_time:.3f}s, dispense_wait_time={params.dispense_wait_time:.3f}s")
        print(f"    pre_asp_air_vol={params.pre_asp_air_vol*1000:.1f}uL, overaspirate_vol={params.overaspirate_vol*1000:.1f}uL, blowout_vol={params.blowout_vol*1000:.1f}uL")

        if (pos + 1) % UPDATE_EVERY_NUM_PIPS == 0 and not SIMULATE:
            try:
                slack_agent.send_slack_message(
                    f"Glycerol Redo {CAMPAIGN_TYPE} progress: {pos + 1}/{total}"
                )
            except Exception as e:
                print(f"Could not send Slack progress notification: {e}")

        current_vial_number, current_vial_name = _check_and_swap_vials(lash_e, current_vial_number, current_vial_name)
        _interrupt_state["current_vial_name"] = current_vial_name

        try:
            source_volume_before = None
            before_mass_g = None
            if ADJUST_VOLUME and not SIMULATE:
                try:
                    source_volume_before = lash_e.nr_robot.get_vial_info(current_vial_name, "vial_volume")
                    before_mass_g = lash_e.nr_robot.c9.read_steady_scale()
                    print(f"    Before: {current_vial_name}={source_volume_before:.3f}mL, mass={before_mass_g:.6f}g")
                except Exception as e:
                    print(f"    Warning: Could not get initial state: {e}")

            start_time = time.perf_counter()

            lash_e.nr_robot.aspirate_from_vial(current_vial_name, volume_ml, parameters=params, liquid=None)
            dispense_result = lash_e.nr_robot.dispense_into_vial(
                current_vial_name, volume_ml, parameters=params, liquid=None,
                measure_weight=True, continuous_mass_monitoring=True, save_mass_data=True
            )
            measured_mass_g, stability_info = dispense_result
            elapsed_s = time.perf_counter() - start_time

            if ADJUST_VOLUME and not SIMULATE and before_mass_g is not None and source_volume_before is not None:
                try:
                    after_mass_g = lash_e.nr_robot.c9.read_steady_scale()
                    actual_volume_consumed_ml = (before_mass_g - after_mass_g) / GLYCEROL_DENSITY
                    source_vial_index = lash_e.nr_robot.get_vial_index_from_name(current_vial_name)
                    corrected_source_volume = source_volume_before - actual_volume_consumed_ml
                    if source_vial_index is not None:
                        lash_e.nr_robot.VIAL_DF.at[source_vial_index, "vial_volume"] = corrected_source_volume
                    lash_e.nr_robot.save_robot_status()
                    print(f"    Corrected {current_vial_name}: {corrected_source_volume:.3f}mL")
                except Exception as e:
                    print(f"    Warning: Could not correct volume tracking: {e}")

            measured_volume_ml = measured_mass_g / GLYCEROL_DENSITY if not SIMULATE else volume_ml * 0.9
            accuracy_pct = (measured_volume_ml / volume_ml) * 100.0

            print(f"    Target: {volume_ul:.1f}uL | Measured: {measured_volume_ml*1000:.1f}uL | Accuracy: {accuracy_pct:.1f}% | Time: {elapsed_s:.2f}s")

            env_data = _get_latest_environmental_data() or {"temp_c": None, "humidity_pct": None, "pressure_pa": None}

            row_result = {
                "measured_volume_ml": measured_volume_ml,
                "accuracy_pct": accuracy_pct,
                "elapsed_s": elapsed_s,
                "measured_mass_g": measured_mass_g if not SIMULATE else 0.0,
                "temp_c": env_data["temp_c"],
                "humidity_pct": env_data["humidity_pct"],
                "pressure_pa": env_data["pressure_pa"],
                "pre_baseline_std": stability_info.get("pre_baseline_std"),
                "post_baseline_std": stability_info.get("post_baseline_std"),
                "pre_stable_pct": (stability_info.get("pre_stable_count", 0) / max(stability_info.get("pre_total_count", 1), 1)) * 100,
                "post_stable_pct": (stability_info.get("post_stable_count", 0) / max(stability_info.get("post_total_count", 1), 1)) * 100,
            }

            if not SIMULATE:
                row_result["mass_data_file"] = _copy_latest_mass_data(campaign_folder, row_idx)
            else:
                row_result["mass_data_file"] = None

            _save_row_result_immediately(campaign_folder, row_idx, volume_ul, row_result, params, RUN_TYPE)

        except Exception as exc:
            print(f"Row {row_idx} failed: {exc}")
            if not SIMULATE:
                raise

        lash_e.nr_robot.remove_pipet()

    # Cleanup
    lash_e.nr_robot.move_home()
    try:
        lash_e.nr_robot.return_vial_home(current_vial_name)
    except Exception as exc:
        print(f"Vial return warning: {exc}")

    print(f"\nRedo workflow complete. Processed {min(total, len(indices_to_run))} rows.")
    print(f"Results appended to: {incremental_csv_path}")

    if not SIMULATE:
        try:
            slack_agent.send_slack_message(
                f"Glycerol Redo COMPLETE: {CAMPAIGN_TYPE}\n"
                f"run_type: {RUN_TYPE}\n"
                f"Rows processed: {len(indices_to_run)}"
            )
        except Exception as e:
            print(f"Could not send Slack completion notification: {e}")


if __name__ == "__main__":
    run_redo()

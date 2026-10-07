"""
Baseline workflow for glycerol Sobol benchmark execution.

Focus:
- Always use status GUI before hardware actions (optional toggle)
- Optionally set pipette usage counters (for partially used tip racks)
- Execute parameterized aspirate/dispense tests from Sobol CSV files
- Dispense into a clamped vial on scale (no plate handling)
"""

#
# no clean up when workflow ends: 
#remove of tip
#move vial back to rack position
#record left over residual volume inside the vial
#close the vial cap
# when we resume the workflow, GUI for the vial rack is not appearing
# close the 8ml vial cap


import os
import sys
import time
import glob
import shutil
import signal
from datetime import datetime, timedelta
import pandas as pd
from sympy import false
import yaml
from pathlib import Path

sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
from pipetting_data.pipetting_parameters import PipettingParameters
import slack_agent

#Step 1: Refill tips
#Step 2: Make sure machine is on and there's air pressure
#Step 3: Setup vials with glycerol
#Step 4: Update experiment number
#Step 5: Run workflow on simulate=True, check log to make sure no errors
#Step 6: Run workflow on simulate=False


# ===== WORKFLOW CONFIG (auto-exported to workflow_configs on first run) =====
SIMULATE = False  
# EXPERIMENT_NUMBER removed - now auto-detects progress and resumes

TIPS_PER_BATCH = 96  # Fallback only - actual batch size is read from robot_status.yaml at runtime
TOTAL_TIPS_PER_CAMPAIGN = 5120  # Target completion per campaign (200uL and 1000uL), SP changed from 96 to 500 to see if it contiues
INPUT_VIAL_STATUS_FILE = "status/calibration_vials.csv"

# Fixed output folder - no more timestamps or experiment numbers
CAMPAIGN_OUTPUT_FOLDER = "output/glycerol_sobol_campaign"

# Liquid setup
LIQUID_TYPE = "glycerol"
VIAL_LOW_THRESHOLD = 2.0  # mL - switch vials when volume drops below this

# Sobol benchmark files
SOBOL_200UL_CSV = "inputs/Glycerin_Sobol_Parameters_200uL.csv"
SOBOL_1000UL_CSV = "inputs/Glycerin_Sobol_Parameters_1000uL.csv"

# Glycerol bottle tracking
GLYCEROL_OPENED_DATE = "2026-03-24"  # ISO format for easy parsing

# Fixed pipetting parameters (not varied in Sobol)
RETRACT_SPEED = 2
POST_ASP_WAIT_TIME = 5
POST_ASP_AIR_VOL = 0

# Volume adjustment tracking
ADJUST_VOLUME = True  # Enable volume correction for real hardware
GLYCEROL_DENSITY = 1.26  # g/mL

UPDATE_EVERY_NUM_PIPS = 12

# Environmental data tracking
MQTT_LOG_FILE = "C:\\Users\\Imaging Controller\\Desktop\\m5stack\\mqtt_log.csv"

# Global state for graceful interrupt handling
_interrupt_state = {
    "lash_e": None,
    "current_vial_name": None,
    "interrupted": False
}

def _interrupt_handler(signum, frame):
    """Handle interrupt signal by capping vial and returning home."""
    _interrupt_state["interrupted"] = True
    print("\n\n⚠️  INTERRUPT DETECTED - Shutting down gracefully...")
    
    lash_e = _interrupt_state.get("lash_e")
    current_vial_name = _interrupt_state.get("current_vial_name")
    
    if lash_e and current_vial_name:
        try:
            lash_e.nr_robot.remove_pipet()

            print(f"🔒 Recapping clamp vial ({current_vial_name})...")
            lash_e.nr_robot.recap_clamp_vial()
            
            print(f"🏠 Returning {current_vial_name} to home position...")
            lash_e.nr_robot.return_vial_home(current_vial_name)
            
            print(f"🏠 Moving robot to home...")
            lash_e.nr_robot.move_home()
            
            print("✓ Cleanup complete - vial capped and home")
        except Exception as e:
            print(f"❌ Error during interrupt cleanup: {e}")
    
    print("Exiting workflow...")
    sys.exit(0)

# Register interrupt handler
signal.signal(signal.SIGINT, _interrupt_handler)
# Simulation-first safety controls
# Simulation mode processes fewer rows for faster testing
MAX_ROWS_FOR_SIMULATION = 20

def _get_remaining_tips(campaign_type):
    """Read robot_status.yaml and pipet_racks.yaml to compute remaining tips for this campaign.
    200uL uses small_tip racks, 1000uL uses large_tip racks.
    Returns total remaining tips across both racks of the relevant type.
    """
    tip_type = "small_tip" if campaign_type == "200uL" else "large_tip"

    status_path = os.path.join("..", "utoronto_demo", "robot_state", "robot_status.yaml")
    racks_path = os.path.join("..", "utoronto_demo", "robot_state", "pipet_racks.yaml")

    # Resolve paths relative to workflow file location
    workflow_dir = os.path.dirname(os.path.abspath(__file__))
    status_path = os.path.join(workflow_dir, "..", "robot_state", "robot_status.yaml")
    racks_path = os.path.join(workflow_dir, "..", "robot_state", "pipet_racks.yaml")

    with open(status_path, "r") as f:
        status = yaml.safe_load(f)
    with open(racks_path, "r") as f:
        racks = yaml.safe_load(f)

    pipets_used = status["pipets_used"]
    total_remaining = 0
    for rack_name, rack_cfg in racks.items():
        if rack_cfg.get("tip_type") == tip_type:
            used = pipets_used.get(rack_name, 0)
            capacity = rack_cfg["num_tips"]
            remaining = max(0, capacity - used)
            total_remaining += remaining
            print(f"  {rack_name}: {used}/{capacity} used, {remaining} remaining")

    print(f"Total remaining {tip_type} tips for {campaign_type} campaign: {total_remaining}")
    return total_remaining

def _check_campaign_progress(campaign_folder):
    """Check how many rows have been completed in a campaign folder."""
    incremental_csv_path = os.path.join(campaign_folder, "incremental_results.csv")
    
    if not os.path.exists(incremental_csv_path):
        return 0  # No progress yet - this is a legitimate case
    
    # CRITICAL: No silent fallbacks - if CSV exists but can't be read, fail loudly
    df = pd.read_csv(incremental_csv_path)
    
    # CRITICAL: Status column MUST exist if CSV exists - no silent fallbacks
    if 'status' not in df.columns:
        raise ValueError(f"CRITICAL: {incremental_csv_path} missing 'status' column - data may be corrupted")
    
    # Count successful rows (status == "ok")
    completed_rows = len(df[df['status'] == 'ok'])
    return completed_rows

def _determine_next_campaign():
    """Determine which campaign to run next based on progress."""
    # Use separate folders for simulation vs real mode
    base_folder = CAMPAIGN_OUTPUT_FOLDER
    if SIMULATE:
        base_folder = CAMPAIGN_OUTPUT_FOLDER + "_simulate"
    
    # Ensure campaign folders exist
    campaign_200_folder = os.path.join(base_folder, "200uL")
    campaign_1000_folder = os.path.join(base_folder, "1000uL")
    
    os.makedirs(campaign_200_folder, exist_ok=True)
    os.makedirs(campaign_1000_folder, exist_ok=True)
    
    # Check progress in both campaigns
    progress_200 = _check_campaign_progress(campaign_200_folder)
    progress_1000 = _check_campaign_progress(campaign_1000_folder)
    
    print(f"Progress check: 200uL={progress_200}/{TOTAL_TIPS_PER_CAMPAIGN}, 1000uL={progress_1000}/{TOTAL_TIPS_PER_CAMPAIGN}")
    
    # Determine which is further behind
    if progress_200 >= TOTAL_TIPS_PER_CAMPAIGN and progress_1000 >= TOTAL_TIPS_PER_CAMPAIGN:
        return None, "Both campaigns completed!"
    elif progress_200 >= TOTAL_TIPS_PER_CAMPAIGN:
        return "1000uL", f"200uL complete, continuing 1000uL from row {progress_1000}"
    elif progress_1000 >= TOTAL_TIPS_PER_CAMPAIGN:
        return "200uL", f"1000uL complete, continuing 200uL from row {progress_200}"
    elif progress_200 <= progress_1000:
        return "200uL", f"200uL further behind ({progress_200} vs {progress_1000}), starting from row {progress_200}"
    else:
        return "1000uL", f"1000uL further behind ({progress_1000} vs {progress_200}), starting from row {progress_1000}"

def _load_sobol_dataframe(csv_path, start_row, batch_size):
    """Load Sobol CSV with row slicing based on progress."""
    df = pd.read_csv(csv_path)
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed") or str(c) == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_cols = [
        "aspirate_speed",
        "dispense_speed",
        "aspirate_wait_time",
        "dispense_wait_time",
        "pre_asp_air_vol_uL",
        "blowout_vol_uL",
        "overaspirate_vol_uL",
        "vol_uL",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    # Calculate end row, but don't exceed available data
    total_rows = len(df)
    end_row = min(start_row + batch_size, total_rows)
    actual_batch_size = end_row - start_row
    
    # CRITICAL: If past end of data, fail loudly instead of silent fallback
    if start_row >= total_rows:
        raise ValueError(f"CRITICAL: Requested start_row {start_row} >= total_rows {total_rows} in {csv_path}. Progress calculation error!")
    
    # Slice based on progress
    df = df.iloc[start_row:end_row].copy()
    
    if SIMULATE:
        # In simulation, take first MAX_ROWS_FOR_SIMULATION from the slice
        df = df.head(MAX_ROWS_FOR_SIMULATION).copy()
        print(f"  SIMULATION: Using rows {start_row}-{start_row+min(MAX_ROWS_FOR_SIMULATION-1, actual_batch_size-1)} (max {MAX_ROWS_FOR_SIMULATION} for sim)")
    else:
        # In real mode, use the calculated slice
        print(f"  REAL MODE: Using rows {start_row}-{end_row-1} ({actual_batch_size} rows)")

    return df

def _row_to_parameters(row):
    """
    Convert Sobol row to PipettingParameters.

    CSV units for pre_asp_air_vol_uL, blowout_vol_uL, overaspirate_vol_uL, vol_uL are in uL.
    The robot API expects mL, so convert uL -> mL.
    """
    params = PipettingParameters(
        aspirate_speed=int(row["aspirate_speed"]),
        dispense_speed=int(row["dispense_speed"]),
        aspirate_wait_time=float(row["aspirate_wait_time"]),
        dispense_wait_time=float(row["dispense_wait_time"]),
        pre_asp_air_vol=float(row["pre_asp_air_vol_uL"]) / 1000.0,
        blowout_vol=float(row["blowout_vol_uL"]) / 1000.0,
        overaspirate_vol=float(row["overaspirate_vol_uL"]) / 1000.0,
        retract_speed=RETRACT_SPEED,  # Fixed per workflow design
        post_retract_wait_time=POST_ASP_WAIT_TIME,  # Fixed per workflow design  
        post_asp_air_vol=POST_ASP_AIR_VOL,  # Fixed per workflow design
        # blowout_speed defaults to None (uses aspirate_speed) - intentional
        # asp_disp_cycles defaults to 0 - intentional for single aspirate/dispense
    )
    volume_ml = float(row["vol_uL"]) / 1000.0
    
    # CRITICAL: Verify no None values that could trigger silent defaults
    critical_params = ['aspirate_speed', 'dispense_speed', 'overaspirate_vol']
    for param in critical_params:
        value = getattr(params, param)
        if value is None:
            raise ValueError(f"CRITICAL: {param} is None - this would trigger silent defaults!")
            
    return params, volume_ml

def _check_and_swap_vials(lash_e, current_vial_number, current_vial_name):
    """Check if current vial is low and swap to next vial if needed."""
    try:
        current_volume = lash_e.nr_robot.get_vial_info(current_vial_name, 'vial_volume')
        
        if current_volume is not None and current_volume <= VIAL_LOW_THRESHOLD:
            print(f"\n🔄 VIAL SWAP: {current_vial_name} low at {current_volume:.2f}mL (≤ {VIAL_LOW_THRESHOLD}mL)")

            slack_agent.send_slack_message("GLYCEROL VIAL NEEDS TO BE REPLACED!")
            input("Waiting...")
            
            # Return old vial home
            lash_e.nr_robot.return_vial_home(current_vial_name)
            
            # Switch to next vial
            new_vial_number = current_vial_number + 1
            new_vial_name = f"vial_{new_vial_number}"
            
            # Move new vial to clamp position
            lash_e.nr_robot.move_vial_to_location(new_vial_name, "clamp", 0)
            
            if not lash_e.nr_robot.is_vial_pipetable(new_vial_name):
                lash_e.nr_robot.uncap_clamp_vial()
            
            print(f"SWAP complete: {current_vial_name} → {new_vial_name}")
            return new_vial_number, new_vial_name
        else:
            # No swap needed
            return current_vial_number, current_vial_name
            
    except Exception as e:
        print(f"Warning: Vial swap check failed: {e}")
        return current_vial_number, current_vial_name

def _create_incremental_csv_headers(csv_path):
    """Create CSV file with headers for incremental saving."""
    headers = [
        "row_index", "timestamp", "volume_ul_target", 
        "aspirate_speed", "dispense_speed", "aspirate_wait_time", "dispense_wait_time",
        "pre_asp_air_vol_uL", "post_asp_air_vol_uL", "overaspirate_vol_uL", "blowout_vol_uL",
        "retract_speed", "post_retract_wait_time",
        "measured_volume_ml", "measured_volume_ul", "accuracy_pct", "elapsed_s",
        "measured_mass_g", "temp_c", "humidity_pct", "pressure_pa",
        "pre_baseline_std", "post_baseline_std", "pre_stable_pct", "post_stable_pct",
        "glycerol_opened_date", "days_bottle_opened",
        "mass_data_file", "status"
    ]
    
    # Create CSV with headers
    pd.DataFrame(columns=headers).to_csv(csv_path, index=False)
    print(f"    📄 Created incremental results file: {os.path.basename(csv_path)}")

def _save_row_result_immediately(campaign_folder, row_idx, row, volume_ul, row_result, params):
    """Save measurement result immediately to prevent data loss."""
    try:
        incremental_csv_path = os.path.join(campaign_folder, "incremental_results.csv")
        
        # Prepare row data with all parameters and results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate days since glycerol bottle was opened
        opened_date = datetime.strptime(GLYCEROL_OPENED_DATE, "%Y-%m-%d")
        current_date = datetime.now()
        days_since_opened = round((current_date - opened_date).days)
        
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
            "status": "ok" if row_result else "failed"
        }
        
        # Append to CSV
        pd.DataFrame([row_data]).to_csv(incremental_csv_path, mode='a', header=False, index=False)
        print(f"    💾 Row {row_idx + 1} data saved immediately")
        
    except Exception as e:
        print(f"    ⚠️ Could not save row {row_idx + 1} immediately: {e}")
        # Don't raise - measurement succeeded, saving is supplementary

def _copy_latest_mass_data(campaign_folder, row_num):
    """Find and copy the most recent mass data files to organized campaign folder."""
    # Create mass_time_data subfolder
    mass_data_dir = os.path.join(campaign_folder, "mass_time_data")
    os.makedirs(mass_data_dir, exist_ok=True)
    
    # Find the most recent mass data files (CSV and PNG)
    mass_pattern = "output/mass_measurements/*/mass_data_vial_*.csv"
    plot_pattern = "output/mass_measurements/*/mass_plot_vial_*.png"
    
    mass_files = glob.glob(mass_pattern)
    plot_files = glob.glob(plot_pattern)
    
    if mass_files:
        # Get most recent file
        latest_mass_file = max(mass_files, key=os.path.getmtime)
        latest_plot_file = max(plot_files, key=os.path.getmtime) if plot_files else None
        
        # Copy with organized naming
        mass_filename = f"mass_data_row_{row_num:03d}.csv"
        plot_filename = f"mass_plot_row_{row_num:03d}.png"
        
        mass_dest = os.path.join(mass_data_dir, mass_filename)
        plot_dest = os.path.join(mass_data_dir, plot_filename)
        
        shutil.copy2(latest_mass_file, mass_dest)
        if latest_plot_file:
            shutil.copy2(latest_plot_file, plot_dest)
        
        print(f"    Copied mass data to: {mass_filename}")
        return mass_filename
    else:
        print(f"    Warning: No mass data files found to copy")
        return None

def _get_latest_environmental_data():
    """
    Get the most recent environmental data from MQTT log file.
    Returns dict with temp_c, humidity_pct, pressure_pa, and Timestamp.
    Returns None if file cannot be read.
    """
    try:
        if not os.path.exists(MQTT_LOG_FILE):
            print(f"❌ Environmental data file not found: {MQTT_LOG_FILE}")
            return None
            
        df = pd.read_csv(MQTT_LOG_FILE)
        if len(df) == 0:
            print("❌ Environmental data file is empty")
            return None
            
        latest = df.iloc[-1]  # Most recent row
        timestamp = pd.to_datetime(latest["Timestamp"]).to_pydatetime()
        
        return {
            "temp_c": float(latest["sht_temp_c"]) if pd.notna(latest["sht_temp_c"]) else None,
            "humidity_pct": float(latest["sht_rh"]) if pd.notna(latest["sht_rh"]) else None,
            "pressure_pa": float(latest["bmp_pa"]) if pd.notna(latest["bmp_pa"]) else None,
            "Timestamp": timestamp
        }
    except Exception as e:
        print(f"❌ Could not read environmental data: {e}")
        return None

def _check_environmental_data_freshness():
    """
    Check if environmental data has been logged within the last hour.
    In simulation mode, automatically returns True.
    Returns True if data is fresh or user chooses to continue, False otherwise.
    """
    if SIMULATE:
        print("✓ Simulation mode: Skipping environmental data check")
        return True
        
    env_data = _get_latest_environmental_data()
    if env_data is None:
        return False
        
    last_time = env_data['Timestamp']
    current_time = datetime.now()
    time_diff = current_time - last_time
    
    if time_diff > timedelta(hours=1):
        print(f"⚠️  WARNING: Last environmental data is {time_diff} old")
        print(f"   Last reading: {last_time}")
        print(f"   Current time: {current_time}")
        
        response = input("Environmental data is stale. Continue anyway? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Stopping workflow due to stale environmental data.")
            return False
    else:
        print(f"✓ Environmental data is fresh (last reading: {last_time})")
        
    return True

def run_baseline():
    print("Starting glycerol Sobol benchmark baseline workflow")
    print(f"Simulation mode: {SIMULATE}")
    
    # Check environmental data freshness
    if not _check_environmental_data_freshness():
        print("Stopping workflow due to environmental data issues.")
        return None

    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE,simulate=SIMULATE,initialize_biotek=False)

    if not hasattr(lash_e, "nr_robot") or lash_e.nr_robot is None:
        print("Workflow stopped before robot initialization (likely canceled from GUI).")
        return None

    lash_e.nr_robot.home_robot_components()
    
    # Initialize vial tracking
    current_vial_number = 0
    current_vial_name = f"vial_{current_vial_number}"
    
    # Store in global interrupt handler state
    _interrupt_state["lash_e"] = lash_e
    _interrupt_state["current_vial_name"] = current_vial_name
    
    # Move initial vial to clamp position once
    lash_e.nr_robot.move_vial_to_location(current_vial_name, "clamp", 0)
    print(f"Moved {current_vial_name} to clamp position")

    if not lash_e.nr_robot.is_vial_pipetable(current_vial_name):
        lash_e.nr_robot.uncap_clamp_vial()

    # Use fixed campaign output folder with simulate suffix if needed
    base_campaign_folder = CAMPAIGN_OUTPUT_FOLDER
    if SIMULATE:
        base_campaign_folder = CAMPAIGN_OUTPUT_FOLDER + "_simulate"

    os.makedirs(base_campaign_folder, exist_ok=True)

    chunks_run = []

    while True:
        # Determine which campaign chunk to run next
        next_campaign, reason = _determine_next_campaign()
        if next_campaign is None:
            print(reason)
            break

        print(f"Campaign selection: {reason}")

        campaign_folder = os.path.join(base_campaign_folder, next_campaign)
        os.makedirs(campaign_folder, exist_ok=True)

        print(f"\nOutput folder: {campaign_folder}")

        # Determine starting row and load appropriate CSV
        start_row = _check_campaign_progress(campaign_folder)

        if next_campaign == "200uL":
            csv_path = SOBOL_200UL_CSV
        elif next_campaign == "1000uL":
            csv_path = SOBOL_1000UL_CSV
        else:
            raise ValueError(f"Unknown campaign: {next_campaign}")

        print(f"\nLoading {next_campaign} Sobol CSV starting from row {start_row}...")
        if SIMULATE:
            batch_size = TIPS_PER_BATCH
        else:
            batch_size = _get_remaining_tips(next_campaign)
            if batch_size == 0:
                print(f"No remaining tips for {next_campaign} campaign. Stopping.")
                break
        campaign_df = _load_sobol_dataframe(csv_path, start_row, batch_size)

        if len(campaign_df) == 0:
            print(f"No more rows to process in {next_campaign} campaign!")
            break

        print(f"  Loaded {len(campaign_df)} rows for processing")

        # Create incremental results CSV with headers if it doesn't exist
        incremental_csv_path = os.path.join(campaign_folder, "incremental_results.csv")
        if not os.path.exists(incremental_csv_path):
            _create_incremental_csv_headers(incremental_csv_path)

        total_tips_to_process = len(campaign_df)
        total_tips_processed = 0  # Counter for this chunk

        # Send startup slack notification
        if not SIMULATE:
            try:
                startup_message = f"🚀 Starting Glycerol Campaign: {next_campaign}!\n"
                startup_message += f"📊 Tips to process this chunk: {total_tips_to_process}\n"
                startup_message += f"🏁 Starting from row: {start_row}\n"
                startup_message += f"⚗️ Mode: {'Simulation' if SIMULATE else 'Hardware'}\n"
                startup_message += f"📁 Output: {next_campaign} campaign"
                slack_agent.send_slack_message(startup_message)
                print(f"📱 Slack startup notification sent for {next_campaign} campaign")
            except Exception as slack_error:
                print(f"⚠ Could not send Slack startup notification: {slack_error}")
        else:
            print("ℹ️ Simulation mode: Slack notifications disabled")

        print(f"\n=== Running {next_campaign} campaign ===\n")

        per_row_summaries = []
        for row_idx, row in campaign_df.iterrows():
            if _interrupt_state["interrupted"]:
                print("Interrupt flag detected — stopping row loop.")
                break
            total_tips_processed += 1
            params, volume_ml = _row_to_parameters(row)
            volume_ul = volume_ml * 1000.0
            actual_row_num = start_row + (row_idx - campaign_df.index[0])  # Calculate actual row number
            print(f"[{next_campaign}] Processing actual row {actual_row_num + 1} | vol={volume_ul:.2f}uL | Batch progress: {total_tips_processed}/{total_tips_to_process}")

            # LOG ALL PARAMETERS TO VERIFY NO SILENT OVERRIDES
            print(f"    CSV Parameters:")
            print(f"      aspirate_speed={params.aspirate_speed}, dispense_speed={params.dispense_speed}")
            print(f"      aspirate_wait_time={params.aspirate_wait_time:.3f}s, dispense_wait_time={params.dispense_wait_time:.3f}s")  
            print(f"      pre_asp_air_vol={params.pre_asp_air_vol*1000:.1f}uL, post_asp_air_vol={params.post_asp_air_vol*1000:.1f}uL")
            print(f"      overaspirate_vol={params.overaspirate_vol*1000:.1f}uL, blowout_vol={params.blowout_vol*1000:.1f}uL")
            print(f"      retract_speed={params.retract_speed}, post_retract_wait_time={params.post_retract_wait_time:.1f}s")

            # Send slack progress update every UPDATE_EVERY_NUM_PIPS tips
            if total_tips_processed % UPDATE_EVERY_NUM_PIPS == 0:
                if not SIMULATE:
                    progress_pct = (total_tips_processed / total_tips_to_process) * 100
                    try:
                        slack_message = f"🧪 Glycerol {next_campaign} Campaign Progress Update\n"
                        slack_message += f"📊 Tips completed this chunk: {total_tips_processed}/{total_tips_to_process} ({progress_pct:.1f}%)\n"
                        slack_message += f"🏁 Processing from row: {start_row}\n"
                        slack_message += f"⚗️ Mode: {'Simulation' if SIMULATE else 'Hardware'}"
                        slack_agent.send_slack_message(slack_message)
                        print(f"    📱 Slack progress notification sent ({total_tips_processed} tips completed)")
                    except Exception as slack_error:
                        print(f"    ⚠ Could not send Slack progress notification: {slack_error}")
                else:
                    print(f"    ℹ️ Simulation mode: Would send progress update ({total_tips_processed} tips completed)")

            # Check if vial needs swapping before pipetting
            current_vial_number, current_vial_name = _check_and_swap_vials(lash_e, current_vial_number, current_vial_name)
            _interrupt_state["current_vial_name"] = current_vial_name  # Update global state

            try:
                # Simple aspirate/dispense with timing

                # Volume adjustment tracking - get initial state before pipetting
                source_volume_before = None
                before_mass_g = None
                if ADJUST_VOLUME and not SIMULATE:
                    try:
                        # Get initial source volume from robot tracking
                        source_volume_before = lash_e.nr_robot.get_vial_info(current_vial_name, 'vial_volume')

                        # Read mass (vial already at clamp)
                        before_mass_g = lash_e.nr_robot.c9.read_steady_scale()
                        print(f"    Before: {current_vial_name}={source_volume_before:.3f}mL, mass={before_mass_g:.6f}g")
                    except Exception as e:
                        print(f"    Warning: Could not get initial state for volume adjustment: {e}")

                start_time = time.perf_counter()

                # Aspirate from vial - CRITICAL: Pass liquid=None to prevent calibrated parameter override
                lash_e.nr_robot.aspirate_from_vial(current_vial_name, volume_ml, parameters=params, liquid=None)

                # Dispense into same vial and measure weight
                dispense_result = lash_e.nr_robot.dispense_into_vial(
                    current_vial_name, volume_ml, parameters=params, liquid=None, measure_weight=True,continuous_mass_monitoring=True,save_mass_data=True
                )

                # Extract mass and stability info
                measured_mass_g, stability_info = dispense_result

                end_time = time.perf_counter()
                elapsed_s = end_time - start_time

                # Volume adjustment tracking - correct robot tracking with actual consumption
                if ADJUST_VOLUME and not SIMULATE and before_mass_g is not None and source_volume_before is not None:
                    try:
                        # Read mass (vial still at clamp)
                        after_mass_g = lash_e.nr_robot.c9.read_steady_scale()
                        actual_mass_consumed_g = before_mass_g - after_mass_g
                        actual_volume_consumed_ml = actual_mass_consumed_g / GLYCEROL_DENSITY

                        # Get source vial index for direct VIAL_DF access
                        source_vial_index = lash_e.nr_robot.get_vial_index_from_name(current_vial_name)

                        # Calculate corrected source volume based on actual consumption
                        corrected_source_volume = source_volume_before - actual_volume_consumed_ml

                        # Manually override robot's source volume tracking with actual measurement
                        if source_vial_index is not None:
                            lash_e.nr_robot.VIAL_DF.at[source_vial_index, 'vial_volume'] = corrected_source_volume
                            print(f"    Corrected {current_vial_name}: {corrected_source_volume:.3f}mL (actual consumed: {actual_volume_consumed_ml*1000:.1f}uL vs nominal {volume_ml*1000:.1f}uL)")

                        # Save the corrected volume
                        lash_e.nr_robot.save_robot_status()

                        # Warning if source volume is getting low
                        if corrected_source_volume < 0.5:
                            print(f"    ⚠️  WARNING: {current_vial_name} volume low: {corrected_source_volume:.3f}mL remaining")

                    except Exception as e:
                        print(f"    Warning: Could not correct volume tracking: {e}")

                # Convert mass to volume (glycerol density = 1.26 g/mL)
                measured_volume_ml = measured_mass_g / GLYCEROL_DENSITY if not SIMULATE else volume_ml * 0.9
                accuracy_pct = (measured_volume_ml / volume_ml) * 100.0

                # LOG MEASUREMENT RESULTS IMMEDIATELY
                print(f"    📏 MEASUREMENT RESULTS:")
                print(f"      Target volume: {volume_ml*1000:.1f}uL")
                print(f"      Measured mass: {measured_mass_g:.6f}g")
                print(f"      Measured volume: {measured_volume_ml*1000:.1f}uL")
                print(f"      Accuracy: {accuracy_pct:.1f}% ({measured_volume_ml/volume_ml:.3f}x)")
                print(f"      Elapsed time: {elapsed_s:.2f}s")

                # Get environmental data
                env_data = _get_latest_environmental_data()
                if env_data is None:
                    env_data = {"temp_c": None, "humidity_pct": None, "pressure_pa": None}
                else:
                    env_parts = []
                    if env_data["temp_c"] is not None:
                        env_parts.append(f"{env_data['temp_c']:.1f}°C")
                    if env_data["humidity_pct"] is not None:
                        env_parts.append(f"{env_data['humidity_pct']:.1f}% RH")
                    if env_data["pressure_pa"] is not None:
                        env_parts.append(f"{env_data['pressure_pa']:.0f}Pa")
                    if env_parts:
                        print(f"      Environment: {' | '.join(env_parts)}")

                row_result = {
                    "measured_volume_ml": measured_volume_ml,
                    "accuracy_pct": accuracy_pct,
                    "elapsed_s": elapsed_s,
                    "measured_mass_g": measured_mass_g if not SIMULATE else 0.0,
                    "temp_c": env_data["temp_c"],
                    "humidity_pct": env_data["humidity_pct"],
                    "pressure_pa": env_data["pressure_pa"]
                }

                # Add stability metrics
                row_result.update({
                    "pre_baseline_std": stability_info.get("pre_baseline_std"),
                    "post_baseline_std": stability_info.get("post_baseline_std"),
                    "pre_stable_pct": (stability_info.get("pre_stable_count", 0) / max(stability_info.get("pre_total_count", 1), 1)) * 100,
                    "post_stable_pct": (stability_info.get("post_stable_count", 0) / max(stability_info.get("post_total_count", 1), 1)) * 100,
                })

                # Copy mass data files to organized location
                if not SIMULATE:
                    mass_filename = _copy_latest_mass_data(campaign_folder, actual_row_num + 1)
                    row_result["mass_data_file"] = mass_filename
                else:
                    row_result["mass_data_file"] = None

                # SAVE RESULTS IMMEDIATELY - Critical for data safety!
                _save_row_result_immediately(campaign_folder, actual_row_num, row, volume_ul, row_result, params)

            except Exception as exc:
                print(f"[{next_campaign}] Row {actual_row_num + 1} failed: {exc}")
                if not SIMULATE:
                    raise
                row_result = {}

            per_row_summaries.append({
                "row_index": int(actual_row_num),
                "volume_ul": float(volume_ul),
                "measured_volume_ml": row_result.get("measured_volume_ml"),
                "accuracy_pct": row_result.get("accuracy_pct"),
                "elapsed_s": row_result.get("elapsed_s"),
                "temp_c": row_result.get("temp_c"),
                "humidity_pct": row_result.get("humidity_pct"),
                "pressure_pa": row_result.get("pressure_pa"),
                "pre_baseline_std": row_result.get("pre_baseline_std"),
                "post_baseline_std": row_result.get("post_baseline_std"),
                "pre_stable_pct": row_result.get("pre_stable_pct"),
                "post_stable_pct": row_result.get("post_stable_pct"),
                "mass_data_file": row_result.get("mass_data_file"),
                "status": "ok" if row_result else "failed",
            })

            lash_e.nr_robot.remove_pipet()

        summary_df = pd.DataFrame(per_row_summaries)
        summary_path = os.path.join(campaign_folder, "campaign_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"[{next_campaign}] chunk complete. Processed {len(campaign_df)} rows.")
        print(f"Summary: {summary_path}")

        if _interrupt_state["interrupted"]:
            print("Interrupt flag set — exiting campaign loop after chunk cleanup.")
            break

        chunks_run.append({
            "campaign": next_campaign,
            "rows_processed": len(campaign_df),
            "campaign_folder": campaign_folder,
        })

        # Send per-chunk completion slack
        if not SIMULATE:
            try:
                completion_message = f"✅ Glycerol {next_campaign} Chunk COMPLETED!\n"
                completion_message += f"📊 Tips processed this chunk: {total_tips_processed}\n"
                completion_message += f"🏁 Started from row: {start_row}\n"
                completion_message += f"📁 Campaign folder: {next_campaign}\n"
                completion_message += f"⚗️ Mode: {'Simulation' if SIMULATE else 'Hardware'}"
                slack_agent.send_slack_message(completion_message)
                print(f"📱 Slack chunk completion notification sent")
            except Exception as slack_error:
                print(f"⚠ Could not send Slack completion notification: {slack_error}")
        else:
            print("ℹ️ Simulation mode: Slack completion notification disabled")

    # Final cleanup (runs once after all chunks)
    lash_e.nr_robot.move_home()

    # Return active vial home
    try:
        lash_e.nr_robot.return_vial_home(current_vial_name)
        print(f"Returned {current_vial_name} to home position")
    except Exception as exc:
        print(f"Vial return warning: {exc}")

    print("\nAll chunks complete")
    print(f"Chunks run: {len(chunks_run)}")

    # Final progress summary
    final_progress_200 = _check_campaign_progress(os.path.join(base_campaign_folder, "200uL"))
    final_progress_1000 = _check_campaign_progress(os.path.join(base_campaign_folder, "1000uL"))

    remaining_message = f"\n📊 Campaign Progress Summary:\n"
    remaining_message += f"  200uL: {final_progress_200}/{TOTAL_TIPS_PER_CAMPAIGN} complete\n"
    remaining_message += f"  1000uL: {final_progress_1000}/{TOTAL_TIPS_PER_CAMPAIGN} complete\n"

    if final_progress_200 < TOTAL_TIPS_PER_CAMPAIGN or final_progress_1000 < TOTAL_TIPS_PER_CAMPAIGN:
        remaining_message += "\n🔄 More work remains. Run the workflow again to continue."
    else:
        remaining_message += "\n✅ Both campaigns completed!"

    print(remaining_message)

    # Send final overall slack message
    if not SIMULATE:
        try:
            overall_message = f"🏁 Glycerol Workflow Run FINISHED\n"
            overall_message += f"📊 Chunks run: {len(chunks_run)}\n"
            for chunk in chunks_run:
                overall_message += f"  {chunk['campaign']}: {chunk['rows_processed']} rows\n"
            overall_message += f"📈 Overall: 200uL={final_progress_200}/{TOTAL_TIPS_PER_CAMPAIGN}, 1000uL={final_progress_1000}/{TOTAL_TIPS_PER_CAMPAIGN}"
            if final_progress_200 < TOTAL_TIPS_PER_CAMPAIGN or final_progress_1000 < TOTAL_TIPS_PER_CAMPAIGN:
                overall_message += "\n🔄 Run again to continue remaining work."
            else:
                overall_message += "\n🎉 All campaigns completed!"
            slack_agent.send_slack_message(overall_message)
            print(f"📱 Slack final notification sent")
        except Exception as slack_error:
            print(f"⚠ Could not send Slack final notification: {slack_error}")
    else:
        print("ℹ️ Simulation mode: Slack final notification disabled")

    return {
        "chunks_run": chunks_run,
        "workflow_complete": True,
    }


if __name__ == "__main__":
    run_baseline()

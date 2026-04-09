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
import time
import glob
import shutil
from datetime import datetime, timedelta
import pandas as pd
import yaml
from pathlib import Path

sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
from pipetting_data.pipetting_parameters import PipettingParameters


# ===== WORKFLOW CONFIG (auto-exported to workflow_configs on first run) =====
SIMULATE = True
EXPERIMENT_NUMBER = 1  # Manual configuration - which experiment batch to run
TIPS_PER_EXPERIMENT = 96  # Tips available per experiment
INPUT_VIAL_STATUS_FILE = "status/calibration_vials.csv"
EXPERIMENT_STATE_FILE = "inputs/glycerol_experiment_state.yaml"

# Liquid setup
LIQUID_TYPE = "glycerol"
VIAL_LOW_THRESHOLD = 2.0  # mL - switch vials when volume drops below this

# Sobol benchmark files
SOBOL_200UL_CSV = "inputs/Glycerin_Sobol_Parameters_200uL.csv"
SOBOL_1000UL_CSV = "inputs/Glycerin_Sobol_Parameters_1000uL.csv"

# Fixed pipetting parameters (not varied in Sobol)
RETRACT_SPEED = 2
POST_ASP_WAIT_TIME = 5
POST_ASP_AIR_VOL = 0

# Volume adjustment tracking
ADJUST_VOLUME = True  # Enable volume correction for real hardware
GLYCEROL_DENSITY = 1.26  # g/mL

# Environmental data tracking
MQTT_LOG_FILE = "C:\\Users\\Imaging Controller\\Desktop\\m5stack\\mqtt_log.csv"

# Simulation-first safety controls
# Set MAX_ROWS_PER_FILE=None to run the full CSV (5120 rows/file).
MAX_ROWS_PER_FILE = 20 if SIMULATE else None

def _load_experiment_state():
    """Load experiment state from inputs directory."""
    if os.path.exists(EXPERIMENT_STATE_FILE):
        with open(EXPERIMENT_STATE_FILE, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Initialize first time
        return {"current_experiment": 1, "last_updated": datetime.now().isoformat()}

def _save_experiment_state(experiment_num):
    """Save experiment state to inputs directory."""
    state = {
        "current_experiment": experiment_num,
        "last_updated": datetime.now().isoformat()
    }
    with open(EXPERIMENT_STATE_FILE, 'w') as f:
        yaml.dump(state, f)
    print(f"Updated experiment state to: {experiment_num}")

def _validate_experiment_numbers():
    """Check that manual config matches hardware state."""
    hw_state = _load_experiment_state()
    hw_experiment = hw_state["current_experiment"]
    
    print(f"Manual config: EXPERIMENT_NUMBER = {EXPERIMENT_NUMBER}")
    print(f"Hardware state: current_experiment = {hw_experiment}")
    
    if EXPERIMENT_NUMBER != hw_experiment:
        raise ValueError(
            f"Experiment number mismatch!\n"
            f"  Manual config (EXPERIMENT_NUMBER): {EXPERIMENT_NUMBER}\n"
            f"  Hardware state: {hw_experiment}\n"
            f"  Please update EXPERIMENT_NUMBER to {hw_experiment} or adjust hardware state file."
        )
    print("✓ Experiment numbers match")

def _load_sobol_dataframe(csv_path, experiment_num, tips_per_experiment):
    """Load Sobol CSV with proper row slicing for experiment batch."""
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

    # Calculate row slice for this experiment
    start_row = (experiment_num - 1) * tips_per_experiment
    end_row = start_row + tips_per_experiment
    
    # Slice by experiment number for both modes
    df = df.iloc[start_row:end_row].copy()
    
    if SIMULATE:
        # In simulation, take first 20 from the experiment slice
        df = df.head(20).copy()
        print(f"  SIMULATION: Using rows {start_row}-{start_row+19} from experiment {experiment_num} slice")
    else:
        # In real mode, use full experiment slice
        print(f"  REAL MODE: Using rows {start_row}-{end_row-1} for experiment {experiment_num}")
        
        if len(df) != tips_per_experiment:
            raise ValueError(f"Experiment {experiment_num} only has {len(df)} rows, expected {tips_per_experiment}")

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
        retract_speed=RETRACT_SPEED,
        post_asp_wait_time=POST_ASP_WAIT_TIME,
        post_asp_air_vol=POST_ASP_AIR_VOL,
    )
    volume_ml = float(row["vol"]) / 1000.0
    return params, volume_ml

def _check_and_swap_vials(lash_e, current_vial_number, current_vial_name):
    """Check if current vial is low and swap to next vial if needed."""
    try:
        current_volume = lash_e.nr_robot.get_vial_info(current_vial_name, 'vial_volume')
        
        if current_volume is not None and current_volume <= VIAL_LOW_THRESHOLD:
            print(f"\n🔄 VIAL SWAP: {current_vial_name} low at {current_volume:.2f}mL (≤ {VIAL_LOW_THRESHOLD}mL)")
            
            # Return old vial home
            lash_e.nr_robot.return_vial_home(current_vial_name)
            
            # Switch to next vial
            new_vial_number = current_vial_number + 1
            new_vial_name = f"vial_{new_vial_number}"
            
            # Move new vial to clamp position
            lash_e.nr_robot.move_vial_to_location(new_vial_name, "clamp", 0)
            
            print(f"SWAP complete: {current_vial_name} → {new_vial_name}")
            return new_vial_number, new_vial_name
        else:
            # No swap needed
            return current_vial_number, current_vial_name
            
    except Exception as e:
        print(f"Warning: Vial swap check failed: {e}")
        return current_vial_number, current_vial_name

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
    
    # Validate experiment configuration
    _validate_experiment_numbers()
    
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
    
    # Move initial vial to clamp position once
    lash_e.nr_robot.move_vial_to_location(current_vial_name, "clamp", 0)
    print(f"Moved {current_vial_name} to clamp position")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join("output", f"glycerol_sobol_exp{EXPERIMENT_NUMBER}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    print(f"\nLoading Sobol CSV files for experiment {EXPERIMENT_NUMBER}...")
    df_200 = _load_sobol_dataframe(SOBOL_200UL_CSV, EXPERIMENT_NUMBER, TIPS_PER_EXPERIMENT)
    df_1000 = _load_sobol_dataframe(SOBOL_1000UL_CSV, EXPERIMENT_NUMBER, TIPS_PER_EXPERIMENT)
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
            
            # Check if vial needs swapping before pipetting
            current_vial_number, current_vial_name = _check_and_swap_vials(lash_e, current_vial_number, current_vial_name)

            try:
                # Simple aspirate/dispense with timing
                
                
                # Volume adjustment tracking - get initial state before pipetting
                source_volume_before = None
                before_mass_g = 0.0
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
                
                # Aspirate from vial
                lash_e.nr_robot.aspirate_from_vial(current_vial_name, volume_ml, parameters=params)
                
                # Dispense into same vial and measure weight
                dispense_result = lash_e.nr_robot.dispense_into_vial(
                    current_vial_name, volume_ml, parameters=params, measure_weight=True,continuous_mass_monitoring=True,save_mass_data=True
                )
                
                # Extract mass and stability info
                measured_mass_g, stability_info = dispense_result
                
                end_time = time.perf_counter()
                elapsed_s = end_time - start_time
                
                # Volume adjustment tracking - correct robot tracking with actual consumption
                if ADJUST_VOLUME and not SIMULATE and before_mass_g > 0 and source_volume_before is not None:
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
                
                # Get environmental data
                env_data = _get_latest_environmental_data()
                
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
                    mass_filename = _copy_latest_mass_data(campaign_folder, row_idx + 1)
                    row_result["mass_data_file"] = mass_filename
                else:
                    row_result["mass_data_file"] = None
                
            except Exception as exc:
                print(f"[{campaign_name}] Row {row_idx + 1} failed: {exc}")
                if not SIMULATE:
                    raise
                row_result = {}

            per_row_summaries.append({
                "row_index": int(row_idx),
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
        campaign_results[campaign_name] = {
            "rows_processed": len(campaign_df),
            "summary_path": summary_path,
        }
        print(f"[{campaign_name}] complete. Summary: {summary_path}")

    lash_e.nr_robot.move_home()

    # Return active vial home
    try:
        lash_e.nr_robot.return_vial_home(current_vial_name)
        print(f"Returned {current_vial_name} to home position")
    except Exception as exc:
        print(f"Vial return warning: {exc}")

    print("\nBaseline workflow complete")
    print(f"Output folder: {output_folder}")
    for campaign_name, summary in campaign_results.items():
        print(f"  {campaign_name}: rows={summary['rows_processed']}, summary={summary['summary_path']}")
    
    # Increment experiment counter ONLY after successful real hardware run
    if not SIMULATE:
        next_experiment = EXPERIMENT_NUMBER + 1
        _save_experiment_state(next_experiment)
        print(f"Hardware state updated to experiment {next_experiment} for next run")
    else:
        print("Simulation mode - experiment counter not incremented")

    return {
        "output_folder": output_folder,
        "results": campaign_results,
        "workflow_complete": True,
    }


if __name__ == "__main__":
    run_baseline()

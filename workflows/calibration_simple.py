# calibration_simple.py
"""
Simplified calibration runner that:
1. Reads conditions from inputs/custom_4_conditions.csv
2. Runs each condition 24 times (replicates)
3. Saves raw data only (no optimization)
4. Uses same base functions as calibration_sdl_short.py
"""

from calibration_sdl_base import *
import sys
sys.path.append("../utoronto_demo")
import os
import logging
from master_usdl_coordinator import Lash_E
import slack_agent
from datetime import datetime
import pandas as pd

# --- Experiment Config ---
LIQUID = "glycerol"  # Change this as needed
SIMULATE = False  # Change this for real experiments
REPLICATES = 24  # Number of replicates per condition
VOLUME = 0.05  # Single volume to test
INPUT_CONDITIONS_FILE = "inputs/custom_4_conditions.csv"  # Input conditions CSV

# Output folder configuration
OUTPUT_FOLDER_REAL = r"C:\Users\Imaging Controller\Desktop\Calibration_SDL_Output\simple_calibration"  # For real experiments
OUTPUT_FOLDER_SIM = "output"  # For simulation (relative to current directory)

# Derived parameters
DENSITY_LIQUID = LIQUIDS[LIQUID]["density"]
NEW_PIPET_EACH_TIME_SET = LIQUIDS[LIQUID]["refill_pipets"]
EXPECTED_MASS = VOLUME * DENSITY_LIQUID
EXPECTED_TIME = VOLUME * 10.146 + 9.5813

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/calibration_vials_short.csv"

# Initialize state
state = {
    "measurement_vial_index": 0,
    "measurement_vial_name": "measurement_vial_0",
    "waste_vial_index": 0
}

# --- Initialize Lash_E ---
print(f"🧪 Starting simple calibration experiment with {LIQUID}")
print(f"📊 Running {REPLICATES} replicates per condition")
print(f"💧 Testing volume: {VOLUME} mL")

lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, initialize_biotek=False)
lash_e.nr_robot.check_input_file()
lash_e.nr_robot.move_vial_to_location("measurement_vial_0", "clamp", 0)

lash_e.logger.info(f"Liquid: {LIQUIDS[LIQUID]}")
if not SIMULATE:
    slack_agent.send_slack_message(f"Starting simple calibration experiment with {LIQUID}")

# --- Setup Output Paths ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if not SIMULATE:
    output_dir = os.path.join(OUTPUT_FOLDER_REAL, f"{timestamp}_{LIQUID}_simple")
    os.makedirs(output_dir, exist_ok=True)
    raw_data_path = os.path.join(output_dir, "raw_data.csv")
    summary_path = os.path.join(output_dir, "summary.csv")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Raw data will be saved to: {raw_data_path}")
    print(f"📄 Summary will be saved to: {summary_path}")
else:
    print("🎮 SIMULATION MODE - No data will be saved to files")
    output_dir = None
    raw_data_path = None
    summary_path = None

# --- Load Input Conditions ---
if not os.path.exists(INPUT_CONDITIONS_FILE):
    raise FileNotFoundError(f"Input conditions file not found: {INPUT_CONDITIONS_FILE}")

conditions_df = pd.read_csv(INPUT_CONDITIONS_FILE)
print(f"📋 Loaded {len(conditions_df)} conditions from {INPUT_CONDITIONS_FILE}")
print("Conditions to test:")
print(conditions_df)

# Convert column names to lowercase with underscores (to match expected parameter names)
conditions_df.columns = [col.lower().replace(' ', '_') for col in conditions_df.columns]

# --- Helper Functions ---
def check_if_measurement_vial_full():
    """Check if measurement vial is full and switch to new one if needed"""
    global state
    current_vial = state["measurement_vial_name"]
    vol = lash_e.nr_robot.get_vial_info(current_vial, "vial_volume")
    if vol > 7.0:
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.return_vial_home(current_vial)
        state["measurement_vial_index"] += 1
        new_vial_name = f"measurement_vial_{state['measurement_vial_index']}"
        state["measurement_vial_name"] = new_vial_name
        lash_e.logger.info(f"[info] Switching to new measurement vial: {new_vial_name}")
        lash_e.nr_robot.move_vial_to_location(new_vial_name, "clamp", 0)

# --- Main Experiment Loop ---
all_raw_data = []
all_summary_data = []

for condition_idx, condition_row in conditions_df.iterrows():
    print(f"\n🔬 Running Condition {condition_idx + 1}/{len(conditions_df)}")
    
    # Convert condition to parameter dictionary
    params = condition_row.to_dict()
    # Convert numpy types to regular Python types and ensure speeds are integers
    for k, v in params.items():
        if k in ['aspirate_speed', 'dispense_speed']:
            params[k] = int(float(v))  # Convert speeds to integers
        elif isinstance(v, (int, float)):
            params[k] = float(v)  # Other numeric values stay as floats
    print(f"Parameters: {params}")
    print(f"🔍 DEBUG: Parameter keys: {list(params.keys())}")
    print(f"🔍 DEBUG: Expected keys: ['aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 'dispense_wait_time', 'retract_speed', 'pre_asp_air_vol', 'post_asp_air_vol', 'overaspirate_vol']")
    
    # Check if measurement vial needs to be changed
    check_if_measurement_vial_full()
    
    # Run the experiment with replicates
    source_vial = "liquid_source"  # Use the same source vial name as SDL
    dest_vial = state["measurement_vial_name"]
    
    try:
        print(f"   Running {REPLICATES} replicates...")
        
        # Use the same pipet_and_measure function from calibration_sdl_base
        

        result = pipet_and_measure(lash_e, source_vial, dest_vial, VOLUME, params, EXPECTED_MASS, EXPECTED_TIME, REPLICATES, SIMULATE, raw_data_path, all_raw_data, LIQUID, NEW_PIPET_EACH_TIME_SET)
        
        # Create summary entry
        summary_entry = {
            "condition_id": condition_idx + 1,
            "volume": VOLUME,
            "liquid": LIQUID,
            "replicates": REPLICATES,
            "deviation": result["deviation"], 
            "variability": result["variability"],
            "time": result["time"],
            "timestamp": datetime.now().isoformat(),
            **params  # Include all parameters
        }
        
        all_summary_data.append(summary_entry)
        
        print(f"   ✅ Condition {condition_idx + 1} completed:")
        print(f"      Deviation: {result['deviation']:.2f}%")
        print(f"      Variability: {result['variability']:.2f}%") 
        print(f"      Time: {result['time']:.2f}s")
        
        # Save summary data incrementally (only if not simulating)
        if not SIMULATE and summary_path:
            pd.DataFrame(all_summary_data).to_csv(summary_path, index=False)
        
    except Exception as e:
        error_msg = f"Error in condition {condition_idx + 1}: {str(e)}"
        print(f"   ❌ {error_msg}")
        lash_e.logger.error(error_msg)
        
        # Clean up hardware state after error
        try:
            lash_e.nr_robot.remove_pipet()
        except:
            pass  # Ignore cleanup errors
        
        # Add error entry to summary
        error_entry = {
            "condition_id": condition_idx + 1,
            "volume": VOLUME,
            "liquid": LIQUID,
            "replicates": REPLICATES,
            "deviation": None,
            "variability": None, 
            "time": None,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            **params
        }
        all_summary_data.append(error_entry)
        if not SIMULATE and summary_path:
            pd.DataFrame(all_summary_data).to_csv(summary_path, index=False)
        continue

# --- Final Cleanup and Summary ---
print(f"\n🎉 Experiment completed!")
print(f"📊 Tested {len(conditions_df)} conditions with {REPLICATES} replicates each")

if not SIMULATE:
    print(f"📄 Raw data saved to: {raw_data_path}")
    print(f"📄 Summary saved to: {summary_path}")
else:
    print("🎮 SIMULATION - No files were saved")

if all_raw_data:
    print(f"📈 Total measurements: {len(all_raw_data)}")
    
    # Calculate overall statistics
    if all_summary_data:
        valid_results = [r for r in all_summary_data if r.get('deviation') is not None]
        if valid_results:
            avg_deviation = sum(r['deviation'] for r in valid_results) / len(valid_results)
            avg_variability = sum(r['variability'] for r in valid_results) / len(valid_results)
            avg_time = sum(r['time'] for r in valid_results) / len(valid_results)
            
            print(f"📈 Overall averages:")
            print(f"   Deviation: {avg_deviation:.2f}%")
            print(f"   Variability: {avg_variability:.2f}%")
            print(f"   Time per replicate: {avg_time:.2f}s")

# Clean up
lash_e.nr_robot.remove_pipet()
current_vial = state["measurement_vial_name"]
lash_e.nr_robot.return_vial_home(current_vial)

if not SIMULATE:
    slack_agent.send_slack_message(f"Simple calibration experiment completed with {LIQUID}. Results saved to {output_dir}")

print("✅ Simple calibration experiment finished!")
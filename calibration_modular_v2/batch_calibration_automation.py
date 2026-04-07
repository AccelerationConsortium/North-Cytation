#!/usr/bin/env python3
"""
Water 200µL Calibration Script

Simplified batch calibration for water at 200 µL only.
Automates calibration by:
1. Modifying config YAML files (liquid type, volumes)
2. Swapping vial names in CSV (water -> liquid_source_0, old -> finished)
3. Running calibration and validation in sequence

Does NOT modify the core calibration programs.
"""

import os
import sys
import yaml
import pandas as pd
import subprocess
from pathlib import Path
import glob
from datetime import datetime
import shutil

# Configuration - Edit this list for your liquids to calibrate
LIQUIDS_TO_CALIBRATE = [
    {
        'liquid_name': 'water',
        'target_vial': 'water',
        'volume_targets_ml': [0.200],
        'validation_volumes_ml': [0.200],
    },
]

# File paths - assumes running from main utoronto_demo directory (like other workflows)
CONFIG_FILE = "calibration_modular_v2/experiment_config.yaml"
VIALS_CSV = "status/calibration_vials_short.csv"
CALIBRATION_SCRIPT = "calibration_modular_v2/run_calibration.py"
VALIDATION_SCRIPT = "calibration_modular_v2/run_validation.py"
OUTPUT_DIR = "calibration_modular_v2/output"  # Base path where run folders are created

class BatchCalibrationAutomator:
    def __init__(self):
        self.original_config = None
        self.original_vials = None
        self.backup_created = False
        
    def create_backups(self):
        """Create single backup files in calibration_modular_v2 folder (overwrite previous ones to prevent accumulation)."""
        
        # Get the directory where this script is located (calibration_modular_v2)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Backup config file - store in calibration_modular_v2 folder
        config_backup = os.path.join(script_dir, "experiment_config_backup.yaml")
        shutil.copy2(CONFIG_FILE, config_backup)
        
        # Backup vials CSV - store in calibration_modular_v2 folder
        vials_backup = os.path.join(script_dir, "calibration_vials_short_backup.csv")
        shutil.copy2(VIALS_CSV, vials_backup)
        
        print(f"Created backups in {script_dir}:")
        print(f"  - experiment_config_backup.yaml")
        print(f"  - calibration_vials_short_backup.csv")
        self.backup_created = True
        
    def load_original_files(self):
        """Load original file contents for restoration."""
        with open(CONFIG_FILE, 'r') as f:
            self.original_config = yaml.safe_load(f)
        
        self.original_vials = pd.read_csv(VIALS_CSV)
        
    def modify_config_file(self, liquid_config):
        """Modify experiment_config.yaml for current liquid."""
        print(f"  Modifying config for {liquid_config['liquid_name']}...")
        
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update liquid type and volumes
        config['experiment']['liquid'] = liquid_config['liquid_name']
        config['experiment']['volume_targets_ml'] = liquid_config['volume_targets_ml']
        config['validation']['volumes_ml'] = liquid_config['validation_volumes_ml']
        
        # Update hardware parameters if specified
        if 'hardware_parameters' in liquid_config:
            print(f"    Applying hardware parameters: {list(liquid_config['hardware_parameters'].keys())}")
            if 'hardware_parameters' not in config:
                config['hardware_parameters'] = {}
            
            for param_name, param_config in liquid_config['hardware_parameters'].items():
                config['hardware_parameters'][param_name] = param_config
                print(f"      {param_name}: bounds={param_config.get('bounds')}, type={param_config.get('type')}")

        # Update fixed parameters if specified (merges with YAML defaults)
        if 'fixed_parameters' in liquid_config:
            print(f"    Applying fixed parameters: {liquid_config['fixed_parameters']}")
            if 'fixed_parameters' not in config['experiment']:
                config['experiment']['fixed_parameters'] = {}
            config['experiment']['fixed_parameters'].update(liquid_config['fixed_parameters'])
        
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
    def modify_vials_csv(self, target_vial):
        """Swap vial names: target -> liquid_source_0, old liquid_source_0 -> finished."""
        print(f"  Swapping vial names: {target_vial} -> liquid_source_0...")
        
        df = pd.read_csv(VIALS_CSV)
        
        # Find current liquid_source_0 and mark as finished
        current_source_idx = df[df['vial_name'] == 'liquid_source_0'].index
        if len(current_source_idx) > 0:
            df.loc[current_source_idx[0], 'vial_name'] = 'finished'
            print(f"    Marked old liquid_source_0 as finished")
        
        # Find target vial and rename to liquid_source_0
        target_idx = df[df['vial_name'] == target_vial].index
        if len(target_idx) > 0:
            df.loc[target_idx[0], 'vial_name'] = 'liquid_source_0'
            print(f"    Renamed {target_vial} to liquid_source_0")
        else:
            print(f"    WARNING: Target vial {target_vial} not found in CSV!")
            
        # Save modified CSV
        df.to_csv(VIALS_CSV, index=False)
        
    def get_latest_output_folder(self):
        """Find the most recent output folder."""
        print(f"    Looking for output folders in: {OUTPUT_DIR}")
        output_folders = glob.glob(os.path.join(OUTPUT_DIR, "*"))
        print(f"    Found folders: {output_folders}")
        
        if not output_folders:
            return None
            
        # Sort by modification time, get newest
        latest_folder = max(output_folders, key=os.path.getmtime)
        print(f"    Latest folder: {latest_folder}")
        return latest_folder
        
    def update_validation_config(self):
        """Update validation config to point to latest optimal_conditions.csv."""
        latest_folder = self.get_latest_output_folder()
        if not latest_folder:
            print("    WARNING: No output folder found!")
            return False
            
        # Convert full path to relative path (remove calibration_modular_v2/ prefix)
        # latest_folder is like: "calibration_modular_v2/output/run_1770420779"
        # We want: "output/run_1770420779/optimal_conditions.csv"
        relative_folder = latest_folder.replace("calibration_modular_v2/", "").replace("calibration_modular_v2\\", "")
        # Find the optimal_conditions_*.csv file (name includes liquid)
        import glob as _glob
        oc_matches = _glob.glob(os.path.join(latest_folder, "optimal_conditions_*.csv"))
        oc_filename = os.path.basename(oc_matches[0]) if oc_matches else "optimal_conditions.csv"
        optimal_conditions_path = os.path.join(relative_folder, oc_filename).replace("\\", "/")
        
        print(f"    Checking for {oc_filename} at: {latest_folder}/{oc_filename}")
        print(f"    Will write to config as: {optimal_conditions_path}")
        
        # Check if the file actually exists (using full path)
        full_optimal_conditions_path = os.path.join(latest_folder, oc_filename)
        if not os.path.exists(full_optimal_conditions_path):
            print(f"    WARNING: {oc_filename} not found at {full_optimal_conditions_path}")
            # Try to list what files ARE in the directory
            try:
                actual_files = os.listdir(latest_folder)
                print(f"    Files actually in {latest_folder}: {actual_files}")
            except:
                print(f"    Cannot list directory contents")
            print("    Calibration may have failed - skipping validation")
            return False
        
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
            
        config['validation']['optimal_conditions_file'] = optimal_conditions_path
        
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        print(f"    Updated optimal_conditions_file: {optimal_conditions_path}")
        return True
        
    def run_script(self, script_name):
        """Run a Python script and return success status."""
        print(f"  Running {script_name}...")
        print(f"    (Output will appear below - press Ctrl+C if hanging)")
        try:
            # Run with visible output and extended timeout for long calibrations
            result = subprocess.run([sys.executable, script_name], 
                                  timeout=10000,  # 2 hour timeout (calibrations can take over an hour)
                                  check=True)
            print(f"    {script_name} completed successfully")
            return True
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT: {script_name} took longer than 2 hours")
            return False
        except subprocess.CalledProcessError as e:
            print(f"    ERROR: {script_name} failed!")
            print(f"    Return code: {e.returncode}")
            return False
        except KeyboardInterrupt:
            print(f"    INTERRUPTED: {script_name} was stopped by user")
            raise  # Re-raise to stop the batch process
            
    def restore_files(self):
        """Restore original file contents."""
        print("Restoring original files...")
        
        if self.original_config:
            with open(CONFIG_FILE, 'w') as f:
                yaml.dump(self.original_config, f, default_flow_style=False, sort_keys=False)
                
        if self.original_vials is not None:
            self.original_vials.to_csv(VIALS_CSV, index=False)
            
        print("Files restored to original state")
        
    def show_initial_gui(self):
        """Show the GUI once at the start for system setup/verification."""
        print("Initializing GUI for system setup...")
        try:
            # Add parent directory to path for imports  
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from master_usdl_coordinator import Lash_E
            
            # Initialize Lash_E with GUI enabled for setup
            vial_file = VIALS_CSV
            lash_e = Lash_E(vial_file, simulate=False, show_gui=True, initialize_biotek=False)
            
            # Give user time to interact with GUI
            print("GUI launched for system setup and verification.")
            print("Please use the GUI to:")
            print("  - Verify vial positions")  
            print("  - Check robot status")
            print("  - Perform any manual setup needed")
            input("\nPress Enter when ready to proceed with batch calibration...")
            
            # Clean shutdown
            try:
                lash_e.nr_robot.move_home()
            except:
                pass
                
            print("GUI setup complete. Starting batch calibration...")
            return True
            
        except Exception as e:
            print(f"Warning: Could not initialize GUI ({e})")
            print("Proceeding without GUI setup...")
            return False
    
    def run_batch_calibration(self):
        """Main water calibration at 200µL."""
        print("="*60)
        print("WATER 200µL CALIBRATION")
        print("="*60)
        print("Calibrating water at 200 µL")
        print()
        
        try:
            # Show GUI once for system setup before calibration starts
            self.show_initial_gui()
            
            # Setup
            self.create_backups()
            self.load_original_files()
            
            # Process water calibration
            liquid_config = LIQUIDS_TO_CALIBRATE[0]  # Only water
            print(f"Processing {liquid_config['liquid_name']} at {liquid_config['volume_targets_ml'][0]*1000:.0f} µL...")
            
            # Step 0: Restore original vials state before calibration
            self.original_vials.to_csv(VIALS_CSV, index=False)
            
            # Step 1: Modify config file
            self.modify_config_file(liquid_config)
            
            # Step 2: Swap vial names
            self.modify_vials_csv(liquid_config['target_vial'])
            
            # Step 3: Run calibration
            if not self.run_script(CALIBRATION_SCRIPT):
                print(f"Calibration failed for {liquid_config['liquid_name']}")
                return
                
            # Step 4: Update validation config (skip if optimal conditions not found)
            if not self.update_validation_config():
                print(f"Optimal conditions not found for {liquid_config['liquid_name']}, skipping validation...")
                return
            
            # Step 5: Run validation
            if not self.run_script(VALIDATION_SCRIPT):
                print(f"Validation failed for {liquid_config['liquid_name']}")
                
            print(f"Water 200µL calibration cycle complete")
                
        except KeyboardInterrupt:
            print("\nWater calibration interrupted by user")
        except Exception as e:
            print(f"ERROR: Water calibration failed: {e}")
        finally:
            # Always restore original files
            self.restore_files()
            
        print("\n" + "="*60)
        print("WATER CALIBRATION COMPLETE")
        print("="*60)

if __name__ == "__main__":
    automator = BatchCalibrationAutomator()
    automator.run_batch_calibration()
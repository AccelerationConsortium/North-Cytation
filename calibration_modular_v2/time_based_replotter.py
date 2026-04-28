"""
Time-based replotter for v2 calibration data.

This program reads existing calibration output data and creates new plots
with time as the x-axis instead of measurement number. Saves plots to 
the same output --> plots folder.

CONFIGURATION - Edit these fields:
"""

# ===== EDIT THESE FIELDS =====
RUN_FOLDER_TO_PROCESS = "run_1769124858_agar_newtips"  # Specific run folder name, or None to process all
PROCESS_ALL_RUNS = False  # Set to True to process all runs, False to use specific folder above
OUTPUT_BASE_DIR = "calibration_modular_v2/output"  # Base directory containing run folders
# =============================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeBasedReplotter:
    """Generate time-based plots from calibration data."""
    
    def __init__(self, output_base_dir="output"):
        """Initialize with base output directory."""
        self.output_base_dir = Path(output_base_dir)
        
        # Configure matplotlib for better plots
        plt.style.use('default')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        
    def process_run_folder(self, run_folder_name):
        """Process a specific run folder and generate time-based plots."""
        run_dir = self.output_base_dir / run_folder_name
        
        if not run_dir.exists():
            logger.error(f"Run folder not found: {run_dir}")
            return False
            
        plots_dir = run_dir / "plots"
        if not plots_dir.exists():
            logger.warning(f"No plots directory in {run_folder_name}, creating one...")
            plots_dir.mkdir(exist_ok=True)
        
        # Check for raw measurements data
        raw_measurements_csv = run_dir / "raw_measurements.csv"
        if not raw_measurements_csv.exists():
            logger.error(f"No raw_measurements.csv found in {run_folder_name}")
            return False
            
        try:
            # Load the data
            logger.info(f"Loading data from {run_folder_name}...")
            df = pd.read_csv(raw_measurements_csv)
            
            # Generate the time vs accuracy tradeoff plot
            self._create_time_vs_accuracy_plot(df, plots_dir)
            
            logger.info(f"[SUCCESS] Time vs accuracy plot saved to: {plots_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {run_folder_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def _create_time_vs_accuracy_plot(self, df, plots_dir):
        """Create plot showing time per measurement vs accuracy - the tradeoff you want to see!"""
        if 'duration_s' not in df.columns:
            logger.warning("No duration_s column found, skipping time vs accuracy plot")
            return
            
        df = df.copy()
        
        # Calculate deviation from target for each measurement
        if 'volume_measured_ml' in df.columns and 'volume_target_ml' in df.columns:
            df['deviation_pct'] = abs((df['volume_measured_ml'] - df['volume_target_ml']) / df['volume_target_ml']) * 100
        else:
            logger.warning("Missing volume columns for accuracy calculation")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Group by target volume for color coding
        if 'volume_target_ml' in df.columns:
            volumes = sorted(df['volume_target_ml'].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(volumes)))
            
            for i, vol in enumerate(volumes):
                vol_data = df[df['volume_target_ml'] == vol]
                target_ul = vol * 1000
                
                # Scatter plot: X = time per measurement, Y = deviation from target
                plt.scatter(
                    vol_data['duration_s'],
                    vol_data['deviation_pct'],
                    color=colors[i],
                    alpha=0.7,
                    label=f'{target_ul:.0f}uL target',
                    s=60,
                    edgecolors='black',
                    linewidth=0.5
                )
        else:
            plt.scatter(df['duration_s'], df['deviation_pct'], alpha=0.7, s=60)
        
        # Smart axis limits - focus on main data cluster 
        time_5th = df['duration_s'].quantile(0.05)
        time_95th = df['duration_s'].quantile(0.95)
        accuracy_95th = df['deviation_pct'].quantile(0.95)
        
        # Set limits to show 95% of data clearly, with some padding
        plt.xlim(time_5th * 0.9, time_95th * 1.1)
        plt.ylim(0, accuracy_95th * 1.2)
            
        plt.xlabel('Time per Measurement (seconds)')
        plt.ylabel('Deviation from Target (%)')
        plt.title('Time vs Accuracy Tradeoff: Does More Time = Better Accuracy?')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = plots_dir / "time_vs_accuracy_tradeoff.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[OK] Time vs accuracy tradeoff plot saved: {save_path.name}")
        
        # Also create measured volume version (like your original plot but with time on x-axis)
        plt.figure(figsize=(12, 8))
        
        if 'volume_target_ml' in df.columns:
            volumes = sorted(df['volume_target_ml'].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(volumes)))
            
            for i, vol in enumerate(volumes):
                vol_data = df[df['volume_target_ml'] == vol]
                target_ul = vol * 1000
                
                # Scatter plot: X = time per measurement, Y = measured volume
                plt.scatter(
                    vol_data['duration_s'],
                    vol_data['volume_measured_ml'] * 1000,
                    color=colors[i],
                    alpha=0.7,
                    label=f'{target_ul:.0f}uL target',
                    s=60
                )
                
                # Target line (horizontal)
                plt.axhline(
                    y=target_ul,
                    color=colors[i],
                    linestyle='--',
                    alpha=0.8,
                    linewidth=2
                )
        
        # Smart x-axis limit for time 
        plt.xlim(time_5th * 0.9, time_95th * 1.1)
        
        plt.xlabel('Time per Measurement (seconds)')
        plt.ylabel('Measured Volume (uL)')
        plt.title('Measured Volume vs Time per Measurement')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = plots_dir / "measured_volume_vs_time_per_measurement.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[OK] Measured volume vs time plot saved: {save_path.name}")
    
    def process_all_runs(self):
        """Process all run folders that have a plots directory."""
        if not self.output_base_dir.exists():
            logger.error(f"Output directory not found: {self.output_base_dir}")
            return 0
            
        run_folders = [d for d in self.output_base_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('run_')]
        
        # Filter to runs that have raw_measurements.csv
        valid_runs = []
        for run_dir in run_folders:
            if (run_dir / "raw_measurements.csv").exists():
                valid_runs.append(run_dir.name)
        
        logger.info(f"Found {len(valid_runs)} runs with measurement data")
        
        success_count = 0
        for run_name in valid_runs:
            logger.info(f"Processing {run_name}...")
            if self.process_run_folder(run_name):
                success_count += 1
                
        logger.info(f"Successfully processed {success_count}/{len(valid_runs)} runs")
        return success_count


def main():
    """Main function using configuration fields instead of command line."""
    
    replotter = TimeBasedReplotter(OUTPUT_BASE_DIR)
    
    if PROCESS_ALL_RUNS:
        print("Processing all run folders...")
        success_count = replotter.process_all_runs()
        print(f"\nCompleted processing {success_count} runs")
    elif RUN_FOLDER_TO_PROCESS:
        print(f"Processing specific run: {RUN_FOLDER_TO_PROCESS}")
        success = replotter.process_run_folder(RUN_FOLDER_TO_PROCESS)
        if success:
            print(f"\nSuccessfully processed {RUN_FOLDER_TO_PROCESS}")
        else:
            print(f"\nFailed to process {RUN_FOLDER_TO_PROCESS}")
    else:
        print("Please edit RUN_FOLDER_TO_PROCESS or set PROCESS_ALL_RUNS = True at the top of this file")


if __name__ == "__main__":
    main()
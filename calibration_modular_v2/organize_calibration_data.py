#!/usr/bin/env python3
"""
Organize Calibration Data by Completion Status and Type
======================================================

This script organizes calibration runs into folders:
- incomplete/ - Runs missing plots folder
- simulation/ - Runs that completed quickly (< 20 min)
- hardware/ - Runs that took real time (>= 20 min)

Usage: python organize_calibration_data.py
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

def get_run_duration(run_path):
    """Get the duration of a calibration run from summary file."""
    summary_path = run_path / "experiment_summary.json"
    
    if not summary_path.exists():
        return None
    
    try:
        with open(summary_path) as f:
            summary = json.load(f)
            return summary.get('total_duration_s', None)
    except:
        return None

def is_run_complete(run_path):
    """Check if a run is complete by looking for plots folder."""
    plots_path = run_path / "plots"
    return plots_path.exists()

def organize_calibration_runs():
    """Organize calibration runs into appropriate folders."""
    output_dir = Path("output")
    
    if not output_dir.exists():
        print("❌ No output directory found")
        return
    
    # Create organization folders
    incomplete_dir = output_dir / "incomplete"
    simulation_dir = output_dir / "simulation"
    hardware_dir = output_dir / "hardware"
    
    incomplete_dir.mkdir(exist_ok=True)
    simulation_dir.mkdir(exist_ok=True)
    hardware_dir.mkdir(exist_ok=True)
    
    # Find all run folders
    run_folders = [f for f in output_dir.iterdir() if f.is_dir() and f.name.startswith('run_')]
    
    if not run_folders:
        print("❌ No run folders found")
        return
    
    print(f"🔍 Found {len(run_folders)} calibration runs to organize")
    print("=" * 60)
    
    moved_counts = {
        'incomplete': 0,
        'simulation': 0,
        'hardware': 0,
        'errors': 0
    }
    
    for run_folder in sorted(run_folders):
        run_name = run_folder.name
        
        # Extract timestamp for display
        try:
            timestamp_str = run_name.split('run_')[1]
            timestamp = datetime.fromtimestamp(int(timestamp_str))
            time_display = timestamp.strftime("%Y-%m-%d %H:%M")
        except:
            time_display = "Unknown time"
        
        try:
            # Check if complete first
            if not is_run_complete(run_folder):
                dest = incomplete_dir / run_name
                shutil.move(str(run_folder), str(dest))
                print(f"📁 INCOMPLETE: {run_name} ({time_display}) → incomplete/")
                moved_counts['incomplete'] += 1
                continue
            
            # Get duration to classify simulation vs hardware
            duration_s = get_run_duration(run_folder)
            
            if duration_s is None:
                dest = incomplete_dir / run_name
                shutil.move(str(run_folder), str(dest))
                print(f"❓ NO DURATION: {run_name} ({time_display}) → incomplete/")
                moved_counts['incomplete'] += 1
                continue
            
            duration_min = duration_s / 60
            
            if duration_min < 20:
                dest = simulation_dir / run_name
                shutil.move(str(run_folder), str(dest))
                print(f"🖥️  SIMULATION: {run_name} ({time_display}) - {duration_min:.1f} min → simulation/")
                moved_counts['simulation'] += 1
            else:
                dest = hardware_dir / run_name
                shutil.move(str(run_folder), str(dest))
                print(f"🔧 HARDWARE: {run_name} ({time_display}) - {duration_min:.1f} min → hardware/")
                moved_counts['hardware'] += 1
                
        except Exception as e:
            print(f"❌ ERROR: {run_name} - {e}")
            moved_counts['errors'] += 1
    
    # Summary
    print("=" * 60)
    print("ORGANIZATION COMPLETE:")
    print(f"📁 Incomplete runs: {moved_counts['incomplete']} → output/incomplete/")
    print(f"🖥️  Simulation runs: {moved_counts['simulation']} → output/simulation/")
    print(f"🔧 Hardware runs: {moved_counts['hardware']} → output/hardware/")
    if moved_counts['errors'] > 0:
        print(f"❌ Errors: {moved_counts['errors']}")
    
    print(f"\\n✅ Your real hardware data is now in: output/hardware/")

if __name__ == "__main__":
    organize_calibration_runs()
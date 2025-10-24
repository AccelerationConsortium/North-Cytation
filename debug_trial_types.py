#!/usr/bin/env python3
"""
DIAGNOSTIC TOOL: Trial Type Labeling Analysis
This script analyzes how trial types are assigned and helps debug the labeling system.
"""
import csv
import os
from collections import defaultdict, Counter

def analyze_raw_data_file(file_path):
    """Analyze a raw data CSV file to understand trial type distribution."""
    print(f"\n🔍 ANALYZING: {os.path.basename(file_path)}")
    print("=" * 60)
    
    try:
        data = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            for row in reader:
                data.append(row)
        
        # Basic info
        print(f"📊 Total measurements: {len(data)}")
        print(f"📋 Columns: {list(columns)}")
        
        # Check if trial_type column exists
        if 'trial_type' not in columns:
            print("❌ ERROR: No 'trial_type' column found!")
            return None
        
        # Collect data
        volumes = []
        trial_types = []
        volume_trial_map = defaultdict(list)
        
        for row in data:
            try:
                volume = float(row['volume'])
                trial_type = row['trial_type']
                volumes.append(volume)
                trial_types.append(trial_type)
                volume_trial_map[volume].append(trial_type)
            except (ValueError, KeyError):
                continue
        
        # Volume distribution
        print(f"\n📏 VOLUME DISTRIBUTION:")
        volume_counts = Counter(volumes)
        for volume in sorted(volume_counts.keys()):
            count = volume_counts[volume]
            volume_ul = volume * 1000
            print(f"   {volume_ul:>6.0f}µL: {count:>3} measurements")
        
        # Trial type distribution
        print(f"\n🏷️  TRIAL TYPE DISTRIBUTION:")
        type_counts = Counter(trial_types)
        for trial_type in sorted(type_counts.keys()):
            count = type_counts[trial_type]
            percentage = (count / len(data)) * 100
            print(f"   {trial_type:>15}: {count:>3} ({percentage:>5.1f}%)")
        
        # Cross-tabulation: Volume vs Trial Type
        print(f"\n📊 VOLUME vs TRIAL TYPE BREAKDOWN:")
        for volume in sorted(volume_trial_map.keys()):
            volume_ul = volume * 1000
            vol_trial_counts = Counter(volume_trial_map[volume])
            print(f"   {volume_ul:>6.0f}µL:", end="")
            for trial_type in sorted(vol_trial_counts.keys()):
                count = vol_trial_counts[trial_type]
                print(f" {trial_type}({count})", end="")
            print()
        
        # Look for patterns that seem wrong
        print(f"\n🔍 POTENTIAL ISSUES:")
        
        # Check for volumes with only SCREENING (suspicious for precision tests)
        for volume, vol_trials in volume_trial_map.items():
            volume_ul = volume * 1000
            unique_types = list(set(vol_trials))
            
            if len(unique_types) == 1 and unique_types[0] == 'SCREENING' and len(vol_trials) > 3:
                print(f"   ⚠️  {volume_ul:.0f}µL has {len(vol_trials)} measurements, all SCREENING (suspicious)")
            elif 'PRECISION' not in unique_types and len(vol_trials) > 5:
                print(f"   ⚠️  {volume_ul:.0f}µL has {len(vol_trials)} measurements, no PRECISION trials")
            elif len(unique_types) > 3:
                print(f"   ℹ️  {volume_ul:.0f}µL has multiple trial types: {unique_types}")
        
        # Check for missing precision tests
        precision_count = trial_types.count('PRECISION')
        if precision_count == 0:
            print(f"   ❌ NO PRECISION TRIALS FOUND AT ALL!")
        elif precision_count < 10:
            print(f"   ⚠️  Very few precision trials found: {precision_count}")
        else:
            print(f"   ✅ Found {precision_count} precision trials")
        
        return data
        
    except Exception as e:
        print(f"❌ ERROR reading file: {e}")
        return None

def main():
    print("🧪 TRIAL TYPE DIAGNOSTIC TOOL")
    print("=" * 60)
    print("This tool analyzes raw measurement data to understand trial type labeling.")
    
    # Look for recent raw data files
    base_path = r"C:\Users\Imaging Controller\Desktop\Calibration_SDL_Output\New_Method"
    
    raw_files = []
    
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                raw_file_path = os.path.join(item_path, "raw_replicate_data.csv")
                if os.path.exists(raw_file_path):
                    raw_files.append(raw_file_path)
    
    if not raw_files:
        print("❌ No raw data files found!")
        print(f"   Searched in: {base_path}")
        return
    
    # Sort by modification time (newest first)
    raw_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"📁 Found {len(raw_files)} raw data files:")
    for i, file_path in enumerate(raw_files[:3]):  # Show first 3
        mod_time = os.path.getmtime(file_path)
        import datetime
        mod_str = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
        print(f"   {i+1}. {os.path.basename(os.path.dirname(file_path))} ({mod_str})")
    
    # Analyze the most recent file
    if raw_files:
        print(f"\n🔍 ANALYZING MOST RECENT FILE:")
        df = analyze_raw_data_file(raw_files[0])
        
        if df is not None:
            print(f"\n💡 EXPECTED PATTERN:")
            print(f"   • 1st volume: SCREENING trials (exploration)")
            print(f"   • 1st volume: OPTIMIZATION trials (refinement)")
            print(f"   • All volumes: PRECISION trials (validation)")
            print(f"   • Some volumes: OVERVOLUME_ASSAY trials (calibration)")
            
            print(f"\n🔧 WHAT TO CHECK:")
            print(f"   1. Are there any PRECISION trials at all?")
            print(f"   2. Do later volumes have only SCREENING (should be PRECISION)?")
            print(f"   3. Are trial types consistent with the workflow?")

if __name__ == "__main__":
    main()
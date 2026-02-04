#!/usr/bin/env python3
"""
Analyze mass measurement data to determine appropriate standard deviation thresholds.
Uses only built-in Python modules.
"""

import csv
import math
import os
from pathlib import Path

def calculate_std(values):
    """Calculate standard deviation manually."""
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

def analyze_mass_data(data_folder):
    """Analyze all CSV files in the folder and calculate baseline standard deviations."""
    
    data_folder = Path(data_folder)
    results = []
    
    # Get all CSV files
    csv_files = list(data_folder.glob("mass_data_*.csv"))
    csv_files.sort()
    
    print(f"Found {len(csv_files)} CSV files to analyze")
    
    for csv_file in csv_files:
        try:
            # Extract timestamp from filename for identification
            timestamp = csv_file.stem.split('_')[-2:]  # Get last two parts: date and time
            file_id = f"{timestamp[0]}_{timestamp[1]}"
            
            # Read the data
            pre_baseline_masses = []
            post_baseline_masses = []
            pre_stable_count = 0
            post_stable_count = 0
            pre_total_count = 0
            post_total_count = 0
            
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mass = float(row['mass_g'])
                    phase = row['phase']
                    steady = row['steady_status'].lower() == 'true'
                    
                    if phase == 'baseline_pre':
                        pre_baseline_masses.append(mass)
                        pre_total_count += 1
                        if steady:
                            pre_stable_count += 1
                    elif phase == 'baseline_post':
                        post_baseline_masses.append(mass)
                        post_total_count += 1
                        if steady:
                            post_stable_count += 1
            
            if len(pre_baseline_masses) == 0 or len(post_baseline_masses) == 0:
                print(f"WARNING: Missing baseline data in {csv_file.name}")
                continue
            
            # Calculate standard deviations
            pre_std = calculate_std(pre_baseline_masses)
            post_std = calculate_std(post_baseline_masses)
            
            # Calculate stability percentages
            pre_stable_pct = (pre_stable_count / pre_total_count) * 100 if pre_total_count > 0 else 0
            post_stable_pct = (post_stable_count / post_total_count) * 100 if post_total_count > 0 else 0
            
            results.append({
                'file_id': file_id,
                'filename': csv_file.name,
                'pre_std': pre_std,
                'post_std': post_std,
                'max_std': max(pre_std, post_std),
                'pre_stable_count': pre_stable_count,
                'pre_total_count': pre_total_count,
                'post_stable_count': post_stable_count,
                'post_total_count': post_total_count,
                'pre_stable_pct': pre_stable_pct,
                'post_stable_pct': post_stable_pct,
                'min_stable_pct': min(pre_stable_pct, post_stable_pct)
            })
            
        except Exception as e:
            print(f"ERROR processing {csv_file.name}: {e}")
    
    return results

def calculate_percentile(values, percentile):
    """Calculate percentile manually."""
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * percentile / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)

def main():
    data_folder = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\mass_measurements\2026-01-21"
    
    # Analyze the data
    results = analyze_mass_data(data_folder)
    
    if len(results) == 0:
        print("No data to analyze!")
        return
    
    print(f"\n=== ANALYSIS OF {len(results)} MASS MEASUREMENTS ===\n")
    
    # Flagged bad measurements (from user)
    bad_measurements = ['20260121_150248', '20260121_150552', '20260121_150823']
    
    # Separate good and bad measurements
    good_data = [r for r in results if r['file_id'] not in bad_measurements]
    bad_data = [r for r in results if r['file_id'] in bad_measurements]
    
    print(f"Good measurements: {len(good_data)}")
    print(f"Bad measurements (flagged): {len(bad_data)}")
    
    # Helper functions for statistics
    def get_values(data, key):
        return [r[key] for r in data]
    
    def calculate_stats(values):
        if not values:
            return 0, 0, 0
        return sum(values)/len(values), calculate_percentile(values, 50), max(values)
    
    # Statistics for standard deviations
    print(f"\n=== STANDARD DEVIATION ANALYSIS ===")
    
    all_pre_std = get_values(results, 'pre_std')
    all_post_std = get_values(results, 'post_std')
    all_max_std = get_values(results, 'max_std')
    
    print(f"\nALL MEASUREMENTS:")
    mean, median, maximum = calculate_stats(all_pre_std)
    print(f"Pre-baseline std:  mean={mean:.6f}, median={median:.6f}, max={maximum:.6f}")
    mean, median, maximum = calculate_stats(all_post_std)
    print(f"Post-baseline std: mean={mean:.6f}, median={median:.6f}, max={maximum:.6f}")
    mean, median, maximum = calculate_stats(all_max_std)
    print(f"Max std (either):  mean={mean:.6f}, median={median:.6f}, max={maximum:.6f}")
    
    good_pre_std = get_values(good_data, 'pre_std')
    good_post_std = get_values(good_data, 'post_std')
    good_max_std = get_values(good_data, 'max_std')
    
    print(f"\nGOOD MEASUREMENTS:")
    mean, median, maximum = calculate_stats(good_pre_std)
    print(f"Pre-baseline std:  mean={mean:.6f}, median={median:.6f}, max={maximum:.6f}")
    mean, median, maximum = calculate_stats(good_post_std)
    print(f"Post-baseline std: mean={mean:.6f}, median={median:.6f}, max={maximum:.6f}")
    mean, median, maximum = calculate_stats(good_max_std)
    print(f"Max std (either):  mean={mean:.6f}, median={median:.6f}, max={maximum:.6f}")
    
    bad_pre_std = get_values(bad_data, 'pre_std')
    bad_post_std = get_values(bad_data, 'post_std')
    bad_max_std = get_values(bad_data, 'max_std')
    
    print(f"\nBAD MEASUREMENTS:")
    mean, median, maximum = calculate_stats(bad_pre_std)
    print(f"Pre-baseline std:  mean={mean:.6f}, median={median:.6f}, max={maximum:.6f}")
    mean, median, maximum = calculate_stats(bad_post_std)
    print(f"Post-baseline std: mean={mean:.6f}, median={median:.6f}, max={maximum:.6f}")
    mean, median, maximum = calculate_stats(bad_max_std)
    print(f"Max std (either):  mean={mean:.6f}, median={median:.6f}, max={maximum:.6f}")
    
    # Percentiles for thresholding
    print(f"\n=== PERCENTILES FOR MAX STD (good measurements only) ===")
    percentiles = [50, 75, 80, 85, 90, 95, 99]
    for p in percentiles:
        value = calculate_percentile(good_max_std, p)
        print(f"{p}th percentile: {value:.6f}g")
    
    print(f"\n=== STABILITY PERCENTAGE ANALYSIS ===")
    good_min_stable = get_values(good_data, 'min_stable_pct')
    bad_min_stable = get_values(bad_data, 'min_stable_pct')
    
    print(f"\nGOOD MEASUREMENTS:")
    mean, median, minimum = sum(good_min_stable)/len(good_min_stable), calculate_percentile(good_min_stable, 50), min(good_min_stable) if good_min_stable else 0
    print(f"Min stable %: mean={mean:.1f}%, median={median:.1f}%, min={minimum:.1f}%")
    
    print(f"\nBAD MEASUREMENTS:")  
    mean, median, minimum = (sum(bad_min_stable)/len(bad_min_stable) if bad_min_stable else 0), (calculate_percentile(bad_min_stable, 50) if bad_min_stable else 0), (min(bad_min_stable) if bad_min_stable else 0)
    print(f"Min stable %: mean={mean:.1f}%, median={median:.1f}%, min={minimum:.1f}%")
    
    # Detailed look at flagged bad measurements
    print(f"\n=== DETAILED ANALYSIS OF FLAGGED BAD MEASUREMENTS ===")
    for row in bad_data:
        print(f"\n{row['filename']}:")
        print(f"  Pre:  {row['pre_stable_count']}/{row['pre_total_count']} stable ({row['pre_stable_pct']:.1f}%), std={row['pre_std']:.6f}g")
        print(f"  Post: {row['post_stable_count']}/{row['post_total_count']} stable ({row['post_stable_pct']:.1f}%), std={row['post_std']:.6f}g")
    
    # Suggest thresholds
    print(f"\n=== SUGGESTED THRESHOLDS ===")
    
    # Conservative: 95th percentile of good measurements
    conservative_threshold = calculate_percentile(good_max_std, 95)
    print(f"Conservative (95th percentile of good): {conservative_threshold:.6f}g")
    
    # Aggressive: catch all bad measurements
    min_bad_std = min(bad_max_std) if bad_max_std else 0
    print(f"Aggressive (minimum bad measurement): {min_bad_std:.6f}g")
    
    # Balanced: between 90th percentile good and minimum bad
    balanced_threshold = calculate_percentile(good_max_std, 90)
    print(f"Balanced (90th percentile of good): {balanced_threshold:.6f}g")
    
    # Test thresholds
    print(f"\n=== THRESHOLD TESTING ===")
    thresholds_to_test = [conservative_threshold, balanced_threshold, min_bad_std]
    threshold_names = ['Conservative', 'Balanced', 'Aggressive']
    
    for name, thresh in zip(threshold_names, thresholds_to_test):
        good_flagged = sum(1 for r in good_data if r['max_std'] > thresh)
        bad_caught = sum(1 for r in bad_data if r['max_std'] > thresh)
        print(f"{name} threshold ({thresh:.6f}g): catches {bad_caught}/{len(bad_data)} bad, flags {good_flagged}/{len(good_data)} good")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Analyze mass data files to find instances where absolute mass is 0 or below.
This indicates potential robot/scale issues during the glycerol workflow.
"""

import csv
import glob
import os

def analyze_mass_data(data_directory):
    """
    Analyze all mass data CSV files in directory to find problematic measurements.
    
    Args:
        data_directory (str): Path to directory containing mass_data_row_*.csv files
        
    Returns:
        dict: Summary of problematic files and data points
    """
    # Find all mass data CSV files
    pattern = os.path.join(data_directory, "mass_data_row_*.csv")
    mass_files = glob.glob(pattern)
    mass_files.sort()  # Sort for consistent processing order
    
    print(f"Found {len(mass_files)} mass data files to analyze")
    
    problematic_files = {}
    total_problematic_points = 0
    
    for file_path in mass_files:
        try:
            # Extract row number from filename
            filename = os.path.basename(file_path)
            row_num = filename.replace("mass_data_row_", "").replace(".csv", "")
            
            # Read the CSV file
            data_points = []
            problematic_points = []
            
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data_points.append(row)
                    mass_g = float(row['mass_g'])
                    if mass_g <= 0.0:
                        problematic_points.append(row)
            
            if len(problematic_points) > 0:
                # Calculate mass statistics
                masses = [float(row['mass_g']) for row in data_points]
                min_mass = min(masses)
                max_mass = max(masses)
                
                # Get unique phases affected
                phases_affected = list(set([point['phase'] for point in problematic_points]))
                
                problematic_files[row_num] = {
                    'file': filename,
                    'num_bad_points': len(problematic_points),
                    'total_points': len(data_points),
                    'bad_percentage': (len(problematic_points) / len(data_points)) * 100,
                    'min_mass': min_mass,
                    'max_mass': max_mass,
                    'phases_affected': phases_affected,
                    'sample_bad_points': problematic_points[:3]  # First 3 bad points
                }
                total_problematic_points += len(problematic_points)
                
                # Show detailed info for severely affected files
                bad_percentage = (len(problematic_points) / len(data_points)) * 100
                if bad_percentage > 10:  # More than 10% bad data
                    print(f"\n🔴 SEVERE ISSUE - Row {row_num}: {bad_percentage:.1f}% bad data ({len(problematic_points)}/{len(data_points)} points)")
                    print(f"   Mass range: {min_mass:.4f}g to {max_mass:.4f}g")
                    print(f"   Phases affected: {', '.join(phases_affected)}")
                elif bad_percentage > 1:  # More than 1% bad data  
                    print(f"\n⚠️  MODERATE ISSUE - Row {row_num}: {bad_percentage:.1f}% bad data ({len(problematic_points)}/{len(data_points)} points)")
                else:
                    print(f"\n⚪ MINOR ISSUE - Row {row_num}: {bad_percentage:.1f}% bad data ({len(problematic_points)}/{len(data_points)} points)")
                    
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
    
    # Summary report
    print(f"\n{'='*60}")
    print(f"MASS DATA ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total files analyzed: {len(mass_files)}")
    print(f"Files with issues: {len(problematic_files)}")
    print(f"Total problematic data points: {total_problematic_points}")
    
    if problematic_files:
        print(f"\nPROBLEMATIC ROWS (mass ≤ 0g):")
        for row_num in sorted(problematic_files.keys(), key=int):
            data = problematic_files[row_num]
            print(f"  Row {row_num}: {data['num_bad_points']} bad points ({data['bad_percentage']:.1f}%) - Mass range: {data['min_mass']:.4f}g to {data['max_mass']:.4f}g")
    else:
        print("\n✅ No problematic mass readings found (all mass values > 0)")
    
    return problematic_files

def analyze_specific_row(data_directory, row_number):
    """
    Detailed analysis of a specific row's mass data.
    
    Args:
        data_directory (str): Path to directory containing mass data files
        row_number (str or int): Row number to analyze in detail
    """
    file_path = os.path.join(data_directory, f"mass_data_row_{str(row_number).zfill(3)}.csv")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Read CSV file
    data_points = []
    problematic_points = []
    
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_points.append(row)
            mass_g = float(row['mass_g'])
            if mass_g <= 0.0:
                problematic_points.append(row)
    
    print(f"\n📊 DETAILED ANALYSIS - Row {row_number}")
    print(f"File: {os.path.basename(file_path)}")
    print(f"Total data points: {len(data_points)}")
    print(f"Problematic points (mass ≤ 0): {len(problematic_points)}")
    
    if len(problematic_points) > 0:
        print(f"Bad data percentage: {(len(problematic_points)/len(data_points))*100:.2f}%")
        
        # Calculate mass statistics
        masses = [float(row['mass_g']) for row in data_points]
        min_mass = min(masses)
        max_mass = max(masses)
        mean_mass = sum(masses) / len(masses)
        
        print(f"\nMass statistics:")
        print(f"  Min mass: {min_mass:.6f}g")
        print(f"  Max mass: {max_mass:.6f}g")
        print(f"  Mean mass: {mean_mass:.6f}g")
        
        # Count phases affected
        phase_counts = {}
        for point in problematic_points:
            phase = point['phase']
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        print(f"\nPhases affected by bad readings:")
        for phase, count in phase_counts.items():
            print(f"  {phase}: {count} bad points")
        
        print(f"\nFirst 5 problematic data points:")
        for i, point in enumerate(problematic_points[:5]):
            print(f"  {i+1}. Time: {point['time_relative']}s, Mass: {point['mass_g']}g, Phase: {point['phase']}")
        
        # Check if there are consecutive bad readings - simplified version
        if len(problematic_points) > 2:
            print(f"\n⚠️  Found {len(problematic_points)} bad readings - check for consecutive patterns")
    else:
        print("✅ No problematic mass readings found in this row")

if __name__ == "__main__":
    # Analyze the mass data directory
    data_dir = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign\1000uL\mass_time_data"
    
    print("🔍 ANALYZING GLYCEROL MASS DATA FOR ROBOT ISSUES")
    print(f"Directory: {data_dir}")
    
    # Run the analysis
    problematic_files = analyze_mass_data(data_dir)
    
    # If any issues found, analyze the most problematic ones in detail
    if problematic_files:
        print(f"\n🔬 DETAILED ANALYSIS OF WORST CASES:")
        
        # Sort by percentage of bad data and analyze top 3 worst cases
        sorted_problems = sorted(problematic_files.items(), 
                               key=lambda x: x[1]['bad_percentage'], 
                               reverse=True)
        
        for i, (row_num, data) in enumerate(sorted_problems[:3]):
            analyze_specific_row(data_dir, row_num)
            if i < 2:  # Add separator between analyses
                print(f"\n{'-'*40}")
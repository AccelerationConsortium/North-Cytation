#!/usr/bin/env python3
"""
Generate a summary report of mass data issues for the glycerol campaign
"""

import csv
import glob
import os

def create_summary_report(data_directory, output_file="mass_issues_summary.txt"):
    """
    Create a concise summary report of problematic mass measurements
    """
    pattern = os.path.join(data_directory, "mass_data_row_*.csv")
    mass_files = glob.glob(pattern)
    mass_files.sort()
    
    problematic_rows = []
    total_problematic_points = 0
    
    for file_path in mass_files:
        try:
            filename = os.path.basename(file_path)
            row_num = filename.replace("mass_data_row_", "").replace(".csv", "")
            
            # Count problematic points
            total_points = 0
            bad_points = 0
            
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    total_points += 1
                    mass_g = float(row['mass_g'])
                    if mass_g <= 0.0:
                        bad_points += 1
            
            if bad_points > 0:
                bad_percentage = (bad_points / total_points) * 100
                problematic_rows.append({
                    'row': int(row_num),
                    'bad_points': bad_points,
                    'total_points': total_points,
                    'percentage': bad_percentage
                })
                total_problematic_points += bad_points
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Write summary report
    with open(output_file, 'w') as f:
        f.write("GLYCEROL CAMPAIGN MASS DATA ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total files analyzed: {len(mass_files)}\n")
        f.write(f"Files with issues: {len(problematic_rows)}\n")
        f.write(f"Total problematic data points: {total_problematic_points}\n\n")
        
        if problematic_rows:
            f.write("ROWS WITH MASS <= 0g ISSUES:\n")
            f.write("-" * 40 + "\n")
            for row_data in sorted(problematic_rows, key=lambda x: x['row']):
                f.write(f"Row {row_data['row']:03d}: {row_data['bad_points']:4d} bad points ")
                f.write(f"({row_data['percentage']:5.1f}%) of {row_data['total_points']}\n")
        else:
            f.write("✅ No problematic mass readings found\n")
    
    print(f"Summary report saved to: {output_file}")
    print(f"\nQuick Summary:")
    print(f"- {len(problematic_rows)} rows out of {len(mass_files)} have mass <= 0g issues")
    print(f"- {total_problematic_points} total problematic data points")
    
    if problematic_rows:
        severe_issues = [r for r in problematic_rows if r['percentage'] > 90]
        print(f"- {len(severe_issues)} rows have >90% bad data (severe robot issues)")
        
        # Show range of affected rows
        row_numbers = [r['row'] for r in problematic_rows]
        print(f"- Affected rows: {min(row_numbers)} to {max(row_numbers)}")

if __name__ == "__main__":
    data_dir = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign\1000uL\mass_time_data"
    create_summary_report(data_dir)
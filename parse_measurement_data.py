#!/usr/bin/env python3
"""
Temporary script to parse measurement JSON files and create a consolidated CSV.
"""
import json
import pandas as pd
import re
import os

def parse_dataframe_string(data_string):
    """Parse a pandas DataFrame string representation back into a DataFrame."""
    lines = data_string.strip().split('\n')
    
    # Find the header line (contains column names)
    header_line = None
    data_start = None
    
    for i, line in enumerate(lines):
        # Look for lines that contain numeric data or well positions
        if re.search(r'[A-H]\d+', line):
            data_start = i
            # Header is usually 1-2 lines before data starts
            for j in range(max(0, i-3), i):
                if any(col in lines[j] for col in ['600', '334_373', '334_384']):
                    header_line = j
                    break
            break
    
    if data_start is None:
        print("Could not find data in string:")
        print(data_string[:200] + "...")
        return None
        
    # Extract column names
    if header_line is not None:
        header_text = lines[header_line]
        if '334_373' in header_text and '334_384' in header_text:
            columns = ['334_373', '334_384']
        elif '600' in header_text:
            columns = ['600']
        else:
            columns = ['value']
    else:
        # Guess from data
        sample_line = lines[data_start]
        data_parts = sample_line.split()
        if len(data_parts) >= 3:  # well + 2 values
            columns = ['334_373', '334_384'] 
        else:
            columns = ['600']
    
    # Parse data lines
    data = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('..') or line.startswith('[') or 'rows x' in line:
            continue
            
        # Split the line and extract well + values
        parts = line.split()
        if len(parts) >= 2:
            well = parts[0]
            values = []
            for j in range(1, len(parts)):
                try:
                    val = float(parts[j])
                    values.append(val)
                except ValueError:
                    continue
            
            if values:
                row = {'well': well}
                for k, col in enumerate(columns):
                    if k < len(values):
                        row[col] = values[k]
                data.append(row)
    
    if data:
        df = pd.DataFrame(data)
        return df
    else:
        return None

def main():
    # File paths
    turbidity_file = r"c:\Users\Imaging Controller\Desktop\utoronto_demo\output\surfactant_grid_SDS_TTAB_20260203_190027\measurement_backups\raw_measurement_plate1_wells0-86_20260203_190301_366.json"
    fluorescence_file = r"c:\Users\Imaging Controller\Desktop\utoronto_demo\output\surfactant_grid_SDS_TTAB_20260203_190027\measurement_backups\raw_measurement_plate1_wells0-86_20260203_193522_343.json"
    output_file = r"c:\Users\Imaging Controller\Desktop\utoronto_demo\consolidated_measurements.csv"
    
    print("Parsing measurement data...")
    
    # Load turbidity data
    turbidity_df = None
    if os.path.exists(turbidity_file):
        with open(turbidity_file, 'r') as f:
            turb_data = json.load(f)
            turb_string = turb_data['measurement_entry']['data']
            print("Turbidity data string (first 200 chars):")
            print(turb_string[:200] + "...")
            print()
            turbidity_df = parse_dataframe_string(turb_string)
            if turbidity_df is not None:
                print(f"Parsed turbidity data: {len(turbidity_df)} rows")
                print(turbidity_df.head())
                print()
            else:
                print("Failed to parse turbidity data")
    else:
        print(f"Turbidity file not found: {turbidity_file}")
    
    # Load fluorescence data  
    fluorescence_df = None
    if os.path.exists(fluorescence_file):
        with open(fluorescence_file, 'r') as f:
            fluor_data = json.load(f)
            fluor_string = fluor_data['measurement_entry']['data']
            print("Fluorescence data string (first 200 chars):")
            print(fluor_string[:200] + "...")
            print()
            fluorescence_df = parse_dataframe_string(fluor_string)
            if fluorescence_df is not None:
                print(f"Parsed fluorescence data: {len(fluorescence_df)} rows")
                print(fluorescence_df.head())
                print()
            else:
                print("Failed to parse fluorescence data")
    else:
        print(f"Fluorescence file not found: {fluorescence_file}")
    
    # Combine the data
    if turbidity_df is not None and fluorescence_df is not None:
        print("Merging turbidity and fluorescence data...")
        combined_df = pd.merge(turbidity_df, fluorescence_df, on='well', how='outer')
        
        # Rename columns for clarity
        column_mapping = {
            '600': 'turbidity_600',
            '334_373': 'fluorescence_334_373', 
            '334_384': 'fluorescence_334_384'
        }
        combined_df = combined_df.rename(columns=column_mapping)
        
        # Reorder columns
        ordered_columns = ['well', 'turbidity_600', 'fluorescence_334_373', 'fluorescence_334_384']
        available_columns = [col for col in ordered_columns if col in combined_df.columns]
        combined_df = combined_df[available_columns]
        
        print(f"Combined data: {len(combined_df)} rows")
        print(combined_df.head(10))
        print()
        
        # Save to CSV
        combined_df.to_csv(output_file, index=False)
        print(f"Saved consolidated data to: {output_file}")
        
        # Summary statistics
        print("\n=== DATA SUMMARY ===")
        print(f"Total wells: {len(combined_df)}")
        if 'turbidity_600' in combined_df.columns:
            print(f"Turbidity range: {combined_df['turbidity_600'].min():.4f} - {combined_df['turbidity_600'].max():.4f}")
        if 'fluorescence_334_373' in combined_df.columns:
            print(f"Fluorescence 334_373 range: {combined_df['fluorescence_334_373'].min():.0f} - {combined_df['fluorescence_334_373'].max():.0f}")
        if 'fluorescence_334_384' in combined_df.columns:
            print(f"Fluorescence 334_384 range: {combined_df['fluorescence_334_384'].min():.0f} - {combined_df['fluorescence_334_384'].max():.0f}")
        
    else:
        print("Could not combine data - missing turbidity or fluorescence data")
        
        # Save individual DataFrames if available
        if turbidity_df is not None:
            turb_file = r"c:\Users\Imaging Controller\Desktop\utoronto_demo\turbidity_only.csv"
            turbidity_df.to_csv(turb_file, index=False)
            print(f"Saved turbidity data to: {turb_file}")
            
        if fluorescence_df is not None:
            fluor_file = r"c:\Users\Imaging Controller\Desktop\utoronto_demo\fluorescence_only.csv"
            fluorescence_df.to_csv(fluor_file, index=False)
            print(f"Saved fluorescence data to: {fluor_file}")

if __name__ == "__main__":
    main()
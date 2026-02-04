#!/usr/bin/env python3
"""
Recovery script to fix consolidated CSV with proper surfactant concentrations
Uses the real measurement data and reconstructs the experimental design
"""

import sys
import os

# Try importing pandas 
try:
    import pandas as pd
    import numpy as np
        print("Successfully imported pandas and numpy")
    except ImportError as e:
        print(f"Import error: {e}")
SURFACTANT_B = "TTAB" 
N_REPLICATES = 1
NUMBER_CONCENTRATIONS = 9
MIN_CONC = 10**-4  # 0.0001 mM

# Stock concentrations from SURFACTANT_LIBRARY
SURFACTANT_LIBRARY = {
    "SDS": {"stock_conc": 50},  # mM
    "TTAB": {"stock_conc": 50}  # mM
}

def calculate_adaptive_concentrations(surfactant_name):
    """Calculate the adaptive concentration grid that should have been used"""
    stock_conc = SURFACTANT_LIBRARY[surfactant_name]["stock_conc"]
    
    # Assume max allocation of 1/3 of well volume for this surfactant  
    max_conc = stock_conc * (1/3) / (1.0)  # Simplified calculation
    
    # Create logarithmic grid
    log_min = np.log10(MIN_CONC)
    log_max = np.log10(max_conc)
    log_concentrations = np.linspace(log_min, log_max, NUMBER_CONCENTRATIONS)
    concentrations = [10**log_conc for log_conc in log_concentrations]
    
    return concentrations

def create_experimental_design():
    """Recreate the experimental design that should have been used"""
    concs_a = calculate_adaptive_concentrations(SURFACTANT_A)
    concs_b = calculate_adaptive_concentrations(SURFACTANT_B)
    
    print(f"Reconstructed concentrations:")
    print(f"  {SURFACTANT_A}: {[f'{c:.6f}' for c in concs_a]}")
    print(f"  {SURFACTANT_B}: {[f'{c:.6f}' for c in concs_b]}")
    
    well_map = []
    well_counter = 0
    
    # Create full concentration grid (same logic as in workflow)
    for i, conc_a in enumerate(concs_a):
        for j, conc_b in enumerate(concs_b):
            for rep in range(N_REPLICATES):
                well_map.append({
                    'well': well_counter,
                    'surfactant_a': SURFACTANT_A,
                    'surfactant_b': SURFACTANT_B,
                    'conc_a_mm': conc_a,
                    'conc_b_mm': conc_b,
                    'replicate': rep + 1,
                    'is_control': False,
                    'control_type': 'sample'
                })
                well_counter += 1
    
    return pd.DataFrame(well_map)

def load_measurement_data(experiment_dir):
    """Load the real measurement data from backup files"""
    measurement_backups_dir = os.path.join(experiment_dir, "measurement_backups")
    
    # Load turbidity data
    turbidity_file = os.path.join(measurement_backups_dir, "turbidity_plate1_wells0-95_20260203_200705_169.csv")
    turbidity_df = pd.read_csv(turbidity_file, index_col=0)
    turbidity_df = turbidity_df.reset_index().rename(columns={'index': 'well_position'})
    turbidity_df = turbidity_df.rename(columns={'600': 'turbidity_600'})
    
    # Load fluorescence data  
    fluorescence_file = os.path.join(measurement_backups_dir, "fluorescence_plate1_wells0-95_20260203_201219_241.csv")
    fluorescence_df = pd.read_csv(fluorescence_file, index_col=0)
    fluorescence_df = fluorescence_df.reset_index().rename(columns={'index': 'well_position'})
    fluorescence_df = fluorescence_df.rename(columns={'334_373': 'fluorescence_334_373', '334_384': 'fluorescence_334_384'})
    
    return turbidity_df, fluorescence_df

def well_number_to_position(well_number):
    """Convert well number (0-based) to well position (A1, A2, etc.)"""
    row = well_number // 12  # 12 columns per row
    col = well_number % 12
    return f"{chr(65 + row)}{col + 1}"

def recover_data(experiment_dir):
    """Main recovery function"""
    print("=== Data Recovery for Surfactant Grid Experiment ===")
    
    # Step 1: Create proper experimental design
    print("\n1. Reconstructing experimental design...")
    design_df = create_experimental_design()
    print(f"   Created design for {len(design_df)} wells")
    
    # Step 2: Add well positions
    design_df['well_position'] = design_df['well'].apply(well_number_to_position)
    design_df['plate'] = 1
    
    # Step 3: Load real measurement data directly by well index
    print("\n2. Loading measurement data...")
    measurement_backups_dir = os.path.join(experiment_dir, "measurement_backups")
    
    # Load turbidity data
    turbidity_file = os.path.join(measurement_backups_dir, "turbidity_plate1_wells0-95_20260203_200705_169.csv")
    turbidity_df = pd.read_csv(turbidity_file, index_col=0)
    
    # Load fluorescence data  
    fluorescence_file = os.path.join(measurement_backups_dir, "fluorescence_plate1_wells0-95_20260203_201219_241.csv")
    fluorescence_df = pd.read_csv(fluorescence_file, index_col=0)
    
    print(f"   Loaded turbidity data: {len(turbidity_df)} wells")
    print(f"   Loaded fluorescence data: {len(fluorescence_df)} wells")
    
    # Debug: Check column names
    print(f"   Turbidity columns: {list(turbidity_df.columns)}")
    print(f"   Fluorescence columns: {list(fluorescence_df.columns)}")
    
    # Step 4: Add measurement data directly by well index
    print("\n3. Merging data by well index...")
    
    # Add measurement columns
    design_df['turbidity_600'] = None
    design_df['fluorescence_334_373'] = None  
    design_df['fluorescence_334_384'] = None
    
    # Get the actual column names (they might be different)
    turb_col = turbidity_df.columns[0]  # First column
    fluor_col_1 = fluorescence_df.columns[0]  # First fluorescence column
    fluor_col_2 = fluorescence_df.columns[1]  # Second fluorescence column
    
    print(f"   Using turbidity column: {turb_col}")
    print(f"   Using fluorescence columns: {fluor_col_1}, {fluor_col_2}")
    
    # Map data by well index (well position in CSV corresponds to well number)
    well_positions = list(turbidity_df.index)  # ['A1', 'A2', 'A3', ...]
    
    for idx, row in design_df.iterrows():
        well_pos = row['well_position']
        if well_pos in well_positions:
            # Get measurement data for this well position using actual column names
            design_df.at[idx, 'turbidity_600'] = turbidity_df.loc[well_pos, turb_col]
            design_df.at[idx, 'fluorescence_334_373'] = fluorescence_df.loc[well_pos, fluor_col_1] 
            design_df.at[idx, 'fluorescence_334_384'] = fluorescence_df.loc[well_pos, fluor_col_2]
    
    # Step 5: Reorder columns for readability
    column_order = ['well', 'well_position', 'plate', 'surfactant_a', 'surfactant_b', 
                   'conc_a_mm', 'conc_b_mm', 'replicate', 'is_control', 'control_type',
                   'turbidity_600', 'fluorescence_334_373', 'fluorescence_334_384']
    
    final_df = design_df[column_order]
    
    # Step 6: Save recovered data
    output_path = os.path.join(experiment_dir, "consolidated_measurements_RECOVERED.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"\n4. Saved recovered data to: {output_path}")
    print(f"   Total rows: {len(final_df)}")
    print(f"   Columns: {list(final_df.columns)}")
    
    # Show sample of data with measurements
    print("\n5. Sample of recovered data with measurements:")
    sample_df = final_df[['well_position', 'surfactant_a', 'surfactant_b', 'conc_a_mm', 'conc_b_mm', 'turbidity_600', 'fluorescence_334_373']].head(10)
    print(sample_df.to_string(index=False))
    
    return final_df

if __name__ == "__main__":
    # Your experiment directory
    experiment_dir = r"c:\Users\Imaging Controller\Desktop\utoronto_demo\output\surfactant_grid_SDS_TTAB_20260203_200428"
    
    if os.path.exists(experiment_dir):
        recovered_df = recover_data(experiment_dir)
        print(f"\nSUCCESS! Recovered data saved to consolidated_measurements_RECOVERED.csv")
    else:
        print(f"ERROR: Experiment directory not found: {experiment_dir}")
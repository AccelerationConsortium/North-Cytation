"""
Recover DTAB+SDS and TTAB+SDS experiments from raw Cytation data.

The existing results files have the correct well→concentration mapping,
but wrong turbidity values and surfactant labels. This script:
1. Uses the concentration mapping from existing results
2. Replaces turbidity with real Cytation data
3. Fixes surfactant labels
4. Removes extra rows from other experiments
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def well_name_to_index(well_name):
    """Convert well name (e.g. 'A1') to 0-based index."""
    row = ord(well_name[0]) - ord('A')  # A=0, B=1, etc.
    col = int(well_name[1:]) - 1        # 1=0, 2=1, etc.
    return row * 12 + col

def recover_experiment(experiment_name, correct_surfactant_a, correct_surfactant_b, 
                      raw_data_file, corrupted_results_file, output_folder):
    """
    Recover a single experiment by combining correct mapping with real Cytation data.
    
    Args:
        experiment_name: Name for output files (e.g. "DTAB_SDS")
        correct_surfactant_a: Correct surfactant A name (e.g. "DTAB")
        correct_surfactant_b: Correct surfactant B name (e.g. "SDS") 
        raw_data_file: Path to raw Cytation CSV file
        corrupted_results_file: Path to existing results file with correct concentrations
        output_folder: Where to save recovered results
    """
    
    print(f"\n=== Recovering {experiment_name} ===")
    
    # 1. Load raw Cytation data
    print(f"Loading raw Cytation data from: {raw_data_file}")
    raw_df = pd.read_csv(raw_data_file, index_col=0)
    print(f"Raw data wells: {list(raw_df.index)}")
    
    # 2. Load corrupted results file (has correct concentrations)
    print(f"Loading concentration mapping from: {corrupted_results_file}")
    corrupted_df = pd.read_csv(corrupted_results_file)
    print(f"Corrupted file has {len(corrupted_df)} rows")
    
    # 3. Filter to only the relevant surfactant combination
    print(f"Filtering for {correct_surfactant_a} + {correct_surfactant_b}...")
    
    # Find rows that should be this experiment (even if labels are wrong)
    # We'll use the well range to identify the correct rows
    if correct_surfactant_a == "DTAB":
        # DTAB+SDS should be wells 24-59 (but only real data up to 35)
        relevant_rows = corrupted_df[(corrupted_df['well'] >= 24) & (corrupted_df['well'] <= 59)].copy()
    elif correct_surfactant_a == "TTAB":  
        # TTAB+SDS should be wells 60-95
        relevant_rows = corrupted_df[(corrupted_df['well'] >= 60) & (corrupted_df['well'] <= 95)].copy()
    else:
        raise ValueError(f"Unknown surfactant A: {correct_surfactant_a}")
    
    print(f"Found {len(relevant_rows)} relevant wells")
    
    # 4. Fix the surfactant labels
    relevant_rows['surfactant_a'] = correct_surfactant_a
    relevant_rows['surfactant_b'] = correct_surfactant_b
    
    # 5. Replace turbidity with real Cytation data
    print("Mapping real turbidity data...")
    recovered_rows = []
    
    for _, row in relevant_rows.iterrows():
        well_idx = int(row['well'])
        
        # Convert absolute well index to wellplate position
        # Assuming 96-well plate: well 0 = A1, well 1 = A2, ..., well 12 = B1, etc.
        plate_row = well_idx // 12
        plate_col = (well_idx % 12) + 1
        well_name = f"{chr(ord('A') + plate_row)}{plate_col}"
        
        # Look up real turbidity data
        if well_name in raw_df.index:
            real_turbidity = raw_df.loc[well_name, 'rep1_CMC_Absorbance_96']
            row['turbidity'] = real_turbidity
            recovered_rows.append(row)
            print(f"  Well {well_idx} ({well_name}): {correct_surfactant_a} {row['conc_a_mm']:.3f} + {correct_surfactant_b} {row['conc_b_mm']:.3f} = {real_turbidity:.4f}")
        else:
            print(f"  Well {well_idx} ({well_name}): NO REAL DATA - skipping")
    
    # 6. Create recovered DataFrame
    if not recovered_rows:
        print(f"ERROR: No valid data recovered for {experiment_name}")
        return None
        
    recovered_df = pd.DataFrame(recovered_rows)
    print(f"Successfully recovered {len(recovered_df)} wells")
    
    # 7. Save recovered results
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"results_{experiment_name}_RECOVERED.csv")
    recovered_df.to_csv(output_file, index=False)
    print(f"Saved recovered results to: {output_file}")
    
    # 8. Create heatmap
    create_heatmap(recovered_df, correct_surfactant_a, correct_surfactant_b, output_folder)
    
    return recovered_df

def create_heatmap(results_df, surfactant_a, surfactant_b, output_folder):
    """Create turbidity heatmap from recovered data."""
    
    print(f"Creating heatmap for {surfactant_a} + {surfactant_b}...")
    
    # Get unique concentrations
    concs_a = sorted(results_df['conc_a_mm'].unique())
    concs_b = sorted(results_df['conc_b_mm'].unique())
    
    print(f"Concentration grid: {len(concs_a)} x {len(concs_b)}")
    
    # Create turbidity matrix
    turbidity_matrix = np.full((len(concs_a), len(concs_b)), np.nan)
    
    for _, row in results_df.iterrows():
        i = concs_a.index(row['conc_a_mm'])
        j = concs_b.index(row['conc_b_mm'])
        turbidity_matrix[i, j] = row['turbidity']
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(turbidity_matrix, cmap='viridis', aspect='auto', origin='lower')
    
    # Labels and title
    plt.title(f'Turbidity Heatmap: {surfactant_a} + {surfactant_b}', fontsize=14, fontweight='bold')
    plt.xlabel(f'{surfactant_b} Concentration (mM)', fontsize=12)
    plt.ylabel(f'{surfactant_a} Concentration (mM)', fontsize=12)
    
    # Tick labels
    plt.xticks(range(len(concs_b)), [f'{c:.3f}' for c in concs_b], rotation=45)
    plt.yticks(range(len(concs_a)), [f'{c:.3f}' for c in concs_a])
    
    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Turbidity (OD600)', fontsize=12)
    
    # Save heatmap
    heatmap_file = os.path.join(output_folder, f'turbidity_heatmap_{surfactant_a}_{surfactant_b}_RECOVERED.png')
    plt.tight_layout()
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {heatmap_file}")
    plt.close()

def main():
    """Recover both DTAB+SDS and TTAB+SDS experiments."""
    
    # Paths
    base_path = os.path.dirname(__file__)
    experiments_path = "../comprehensive_surfactant_screening_20251125_163804"
    
    # Output folder for recovered data
    output_folder = os.path.join(base_path, "recovered_experiments")
    
    # Recover DTAB+SDS (Experiment 2)
    recover_experiment(
        experiment_name="DTAB_SDS",
        correct_surfactant_a="DTAB", 
        correct_surfactant_b="SDS",
        raw_data_file=os.path.join(base_path, "raw_cytation_wells24-59_20251125_173411_693.csv"),
        corrupted_results_file=os.path.join(base_path, experiments_path, "02_SDS_DTAB", "results_DTAB_SDS.csv"),
        output_folder=output_folder
    )
    
    # Recover TTAB+SDS (Experiment 3)  
    recover_experiment(
        experiment_name="TTAB_SDS",
        correct_surfactant_a="TTAB",
        correct_surfactant_b="SDS", 
        raw_data_file=os.path.join(base_path, "raw_cytation_wells60-95_20251125_181000_607.csv"),
        corrupted_results_file=os.path.join(base_path, experiments_path, "03_SDS_TTAB", "results_TTAB_SDS.csv"),
        output_folder=output_folder
    )
    
    print("\n=== Recovery Complete ===")
    print(f"Recovered data saved to: {output_folder}")

if __name__ == "__main__":
    main()
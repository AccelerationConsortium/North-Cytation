#!/usr/bin/env python3
"""
DMSO Turbidity Comparison Analyzer
Compares turbidity measurements across different datasets to determine DMSO effects.

Analyzes:
1. Initial turbidity across datasets
2. Post-shake comparisons between conditions  
3. 5-minute timepoint comparisons
4. Statistical analysis of DMSO impact

Datasets compared:
- prep_turbidity: Raw turbidity measurements
- prep_turbidity_dmso_ratio: Turbidity with DMSO ratio
- prep_dmso_turbidity_ratio: DMSO-turbidity ratio
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

def load_well_recipes(base_path):
    """Load experiment plan to understand what's in each well."""
    recipe_file = Path(base_path) / "experiment_plan_well_recipes.csv"
    well_labels = {}
    
    if recipe_file.exists():
        df = pd.read_csv(recipe_file)
        for idx, row in df.iterrows():
            well_pos = f"{chr(65 + idx // 12)}{(idx % 12) + 1}"  
            
            if row['well_type'] == 'control':
                label = f"{row['control_type']}"
            else:
                sds_conc = row.get('surf_A_conc_mm', 0) if pd.notna(row.get('surf_A_conc_mm')) else 0
                ttab_conc = row.get('surf_B_conc_mm', 0) if pd.notna(row.get('surf_B_conc_mm')) else 0
                label = f"SDS:{sds_conc:.1f}, TTAB:{ttab_conc:.1f}"
            
            well_labels[well_pos] = label
            
    return well_labels

def find_turbidity_files(folder_path, timepoint):
    """Find turbidity files for a specific timepoint."""
    target_patterns = {
        'prep': ['prep_', 'initial_'],
        'post_shake': ['post_shake'],
        't5min': ['t5min']
    }
    
    if timepoint not in target_patterns:
        return None
    
    for file in Path(folder_path).glob('*.csv'):
        filename = file.name.lower()
        
        # Skip experiment results
        if 'experiment_results' in filename:
            continue
            
        # Must contain 'turb' for turbidity and NOT 'fluor' for fluorescence
        if 'turb' not in filename or 'fluor' in filename:
            continue
        
        # Check for timepoint patterns
        for pattern in target_patterns[timepoint]:
            if pattern in filename:
                # For prep, make sure it's not post_shake
                if timepoint == 'prep' and 'post_shake' in filename:
                    continue
                return str(file)
    
    return None

def load_turbidity_data(folder_path, timepoint):
    """Load turbidity data for a specific timepoint from a dataset folder."""
    file_path = find_turbidity_files(folder_path, timepoint)
    
    if not file_path:
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # Find turbidity column
        turb_column = None
        for col in df.columns:
            if 'turbidity' in col.lower():
                turb_column = col
                break
        
        if turb_column:
            data = df.set_index('well_position')[turb_column]
            print(f"    ✓ {timepoint}: {Path(file_path).name} ({len(data)} wells)")
            return data
        else:
            print(f"    ✗ {timepoint}: No turbidity column in {Path(file_path).name}")
            return None
            
    except Exception as e:
        print(f"    ✗ {timepoint}: Error loading {Path(file_path).name}: {e}")
        return None

def create_dataset_comparison(base_path, timepoints=['prep', 'post_shake', 't5min']):
    """Compare turbidity data across different datasets."""
    
    print("=== DMSO TURBIDITY COMPARISON ANALYSIS ===")
    
    # Define datasets to compare
    datasets = {
        'Raw Turbidity': 'prep_turbidity',
        'Turbidity + DMSO Ratio': 'prep_turbidity_dmso_ratio', 
        'DMSO + Turbidity Ratio': 'prep_dmso_turbidity_ratio'
    }
    
    # Load well recipes
    well_recipes = load_well_recipes(base_path)
    print(f"Loaded recipes for {len(well_recipes)} wells")
    
    # Collect data for all datasets and timepoints
    all_data = {}
    
    for dataset_name, folder_name in datasets.items():
        folder_path = Path(base_path) / folder_name
        
        if not folder_path.exists():
            print(f"✗ {dataset_name}: Folder not found - {folder_name}")
            continue
            
        print(f"\n{dataset_name} ({folder_name}):")
        
        dataset_data = {}
        for timepoint in timepoints:
            data = load_turbidity_data(folder_path, timepoint)
            if data is not None:
                dataset_data[timepoint] = data
        
        if dataset_data:
            all_data[dataset_name] = dataset_data
            print(f"  → Loaded {len(dataset_data)} timepoints")
        else:
            print(f"  → No valid turbidity data found")
    
    if not all_data:
        print("No valid data found for comparison")
        return
    
    # Summary of what was successfully loaded
    print("\n=== SUMMARY: Successfully Loaded Turbidity Data ===")
    
    for dataset_name, dataset_data in all_data.items():
        timepoints_loaded = list(dataset_data.keys())
        print(f"{dataset_name}: {timepoints_loaded}")
    
    print("\n=== Available for plotting ===")
    # Find what timepoints we can actually plot across datasets
    all_timepoints = set()
    for dataset_data in all_data.values():
        all_timepoints.update(dataset_data.keys())
    
    for tp in sorted(all_timepoints):
        datasets_with_tp = [name for name, data in all_data.items() if tp in data]
        print(f"{tp}: {len(datasets_with_tp)} datasets - {datasets_with_tp}")
    
    # Now create the actual plots
    print("\n=== Creating well tracking plots ===")
    create_well_tracking_plots(all_data, well_recipes, base_path)

def normalize_timepoint_name(timepoint):
    """Normalize timepoint names - treat 'prep' and 'initial' as equivalent."""
    if timepoint in ['prep', 'initial']:
        return 'prep'  # Standardize to 'prep'
    return timepoint

def get_normalized_timepoints(dataset_data):
    """Get normalized timepoint names for a dataset."""
    normalized = {}
    for tp, data in dataset_data.items():
        norm_tp = normalize_timepoint_name(tp)
        normalized[norm_tp] = data
    return normalized

def create_well_tracking_plots(all_data, well_recipes, base_path):
    """Create plots tracking each well across different datasets for each timepoint."""
    
    # Normalize timepoints across all datasets
    normalized_data = {}
    for dataset_name, dataset_data in all_data.items():
        normalized_data[dataset_name] = get_normalized_timepoints(dataset_data)
    
    # Find all timepoints that exist in any dataset (not just common ones)
    all_timepoints = set()
    for dataset_name, dataset_data in normalized_data.items():
        timepoints_in_dataset = list(dataset_data.keys())
        print(f"Dataset '{dataset_name}' has timepoints: {timepoints_in_dataset}")
        all_timepoints.update(dataset_data.keys())
    
    all_timepoints = sorted(list(all_timepoints))
    print(f"\nAll available timepoints across datasets: {all_timepoints}")
    
    if not all_timepoints:
        print("No timepoints found in any dataset")
        return
    
    datasets = list(normalized_data.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple'][:len(datasets)]
    
    # Create separate plot for each timepoint that exists in at least one dataset
    for timepoint in all_timepoints:
        print(f"\nCreating well tracking plot for {timepoint}")
        
        # Find datasets that have this timepoint
        datasets_with_timepoint = []
        for dataset_name in datasets:
            if timepoint in normalized_data[dataset_name]:
                datasets_with_timepoint.append(dataset_name)
        
        print(f"  Timepoint '{timepoint}' found in datasets: {datasets_with_timepoint}")
        
        if len(datasets_with_timepoint) < 1:
            print(f"  No datasets found for {timepoint}")
            continue
        
        # Find wells that exist in at least some datasets at this timepoint
        all_wells = set()
        for dataset_name in datasets_with_timepoint:
            if timepoint in normalized_data[dataset_name]:
                all_wells.update(normalized_data[dataset_name][timepoint].index)
        
        all_wells = sorted(list(all_wells))
        print(f"  Found {len(all_wells)} wells total for {timepoint}")
        
        if len(all_wells) < 1:
            print(f"  No wells found for {timepoint}")
            continue
        
        # Create subplots for each well (limit to manageable number)
        max_wells = 35  # Limit for readability
        wells_to_plot = all_wells[:max_wells]
        
        n_cols = 6
        n_rows = int(np.ceil(len(wells_to_plot) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows))
        fig.suptitle(f'Well-by-Well Tracking Across Datasets ({timepoint.replace("_", " ").title()})', fontsize=16)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for well_idx, well in enumerate(wells_to_plot):
            ax = axes[well_idx]
            
            # Get values for this well - only from datasets that have this timepoint
            datasets_for_plot = []
            values = []
            colors_for_plot = []
            
            for i, dataset_name in enumerate(datasets):
                if (timepoint in normalized_data[dataset_name] and 
                    well in normalized_data[dataset_name][timepoint].index):
                    datasets_for_plot.append(dataset_name)
                    values.append(normalized_data[dataset_name][timepoint].loc[well])
                    colors_for_plot.append(colors[i])
            
            if not values:  # Skip if no data for this well
                ax.set_visible(False)
                continue
            
            # Create x positions for only the datasets with data
            x_pos = np.arange(len(datasets_for_plot))
            
            # Plot bar chart for this well
            bars = ax.bar(x_pos, values, color=colors_for_plot, alpha=0.7)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=6)
            
            # Customize subplot
            ax.set_title(f'{well}', fontsize=10, weight='bold')
            if well in well_recipes:
                recipe = well_recipes[well]
                if len(recipe) > 25:
                    recipe = recipe[:22] + "..."
                ax.set_title(f'{well}\\n{recipe}', fontsize=8)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                               for name in datasets_for_plot], rotation=45, ha='right', fontsize=7)
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(axis='y', alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(wells_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save plot
        timepoint_name = timepoint.replace("_", "-")
        output_path = Path(base_path) / f"well_tracking_{timepoint_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved well tracking plot: {output_path}")
        plt.close()
    
    print(f"\nCompleted well tracking plots for {len(common_timepoints)} timepoints")

def create_well_difference_analysis(all_data, well_recipes, base_path):
    """Analyze differences between datasets for each well."""
    
    # Normalize timepoints across all datasets
    normalized_data = {}
    for dataset_name, dataset_data in all_data.items():
        normalized_data[dataset_name] = get_normalized_timepoints(dataset_data)
    
    # Find timepoints common to all datasets (after normalization)
    all_timepoints = set()
    for dataset_data in normalized_data.values():
        all_timepoints.update(dataset_data.keys())
    
    common_timepoints = all_timepoints
    for dataset_data in normalized_data.values():
        common_timepoints = common_timepoints.intersection(set(dataset_data.keys()))
    
    common_timepoints = sorted(list(common_timepoints))
    print(f"\nNormalized common timepoints for difference analysis: {common_timepoints}")
    
    if not common_timepoints:
        print("No common timepoints found across all datasets")
        return
    
    # Use first common timepoint
    timepoint = common_timepoints[0]
    datasets = list(normalized_data.keys())
    
    if len(datasets) < 2:
        print("Need at least 2 datasets for difference analysis")
        return
    
    # Use first dataset as reference
    reference_dataset = datasets[0]
    print(f"Using '{reference_dataset}' as reference for difference analysis")
    
    # Find wells common to all datasets for this timepoint
    common_wells = set(normalized_data[reference_dataset][timepoint].index)
    for dataset_name in datasets:
        if timepoint in normalized_data[dataset_name]:
            common_wells = common_wells.intersection(set(normalized_data[dataset_name][timepoint].index))
    
    common_wells = sorted(list(common_wells))
    print(f"Found {len(common_wells)} common wells for {timepoint}")
    
    if len(common_wells) < 1:
        print("No common wells found")
        return
    
    # Create difference plots
    n_comparisons = len(datasets) - 1
    fig, axes = plt.subplots(n_comparisons, 1, figsize=(15, 5*n_comparisons))
    
    if n_comparisons == 1:
        axes = [axes]
    
    comparison_idx = 0
    for comparison_dataset in datasets[1:]:
        # Check that both datasets have this timepoint
        if timepoint not in normalized_data[reference_dataset] or timepoint not in normalized_data[comparison_dataset]:
            print(f"Skipping {comparison_dataset} - missing timepoint {timepoint}")
            continue
        
        ax = axes[comparison_idx]
        
        # Calculate differences for each well
        well_positions = []
        differences = []
        ratios = []
        
        for well in common_wells:
            if well in normalized_data[reference_dataset][timepoint] and well in normalized_data[comparison_dataset][timepoint]:
                ref_val = normalized_data[reference_dataset][timepoint][well]
                comp_val = normalized_data[comparison_dataset][timepoint][well]
                
                diff = comp_val - ref_val
                ratio = comp_val / ref_val if ref_val != 0 else np.nan
                
                well_positions.append(well)
                differences.append(diff)
                ratios.append(ratio)
        
        if not well_positions:
            print(f"No valid wells found for comparison: {comparison_dataset}")
            comparison_idx += 1
            continue
        
        # Create bar plot of differences
        x_pos = np.arange(len(well_positions))
        bars = ax.bar(x_pos, differences, 
                     color=['red' if d < 0 else 'blue' for d in differences],
                     alpha=0.7)
        
        # Add horizontal line at zero
        ax.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels (show ratios)
        for i, (bar, ratio) in enumerate(zip(bars, ratios)):
            if not np.isnan(ratio):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., 
                       height + 0.01*max(abs(d) for d in differences),
                       f'{ratio:.2f}x', ha='center', va='bottom', fontsize=7,
                       rotation=90)
        
        # Customize plot
        ax.set_title(f'Difference: {comparison_dataset} - {reference_dataset} ({timepoint})', 
                    fontsize=14)
        ax.set_xlabel('Wells', fontsize=12)
        ax.set_ylabel('Turbidity Difference', fontsize=12)
        
        # Set x-axis labels
        ax.set_xticks(x_pos[::max(1, len(x_pos)//20)])  # Show every nth label
        ax.set_xticklabels([well_positions[i] for i in range(0, len(well_positions), 
                           max(1, len(x_pos)//20))], rotation=90, fontsize=8)
        
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        mean_ratio = np.nanmean(ratios)
        
        stats_text = f'Mean diff: {mean_diff:.4f} ± {std_diff:.4f}\\nMean ratio: {mean_ratio:.3f}x'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
        
        print(f"  {comparison_dataset} vs {reference_dataset}:")
        print(f"    Mean difference: {mean_diff:.4f} ± {std_diff:.4f}")
        print(f"    Mean ratio: {mean_ratio:.3f}x")
        
        comparison_idx += 1
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(base_path) / f"well_differences_{timepoint}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved well differences plot: {output_path}")

def main():
    """Main function to run DMSO turbidity comparison."""
    
    # Base directory
    base_path = r"C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo\\output\\surfactant_grid_SDS_TTAB_20260226_180227_kinetics_thursday_overnight"
    
    if not Path(base_path).exists():
        print(f"Error: Directory not found: {base_path}")
        return
    
    # Run comparison analysis
    create_dataset_comparison(base_path)
    
    print(f"\\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved in: {base_path}")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
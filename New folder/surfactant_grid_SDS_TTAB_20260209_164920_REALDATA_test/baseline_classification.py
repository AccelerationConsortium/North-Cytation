"""
Baseline Classification for Surfactant Grid Data

This program analyzes surfactant grid experiment data to classify wells as:
- Baseline: No significant surfactant interaction, similar to control behavior
- Non-baseline: Significant interaction effects (micelle formation, synergistic effects, etc.)

Classification uses multiple criteria:
1. Turbidity thresholds (high turbidity = interaction)
2. Fluorescence ratio deviations from controls
3. Concentration-based analysis (low concentrations more likely baseline)
4. Statistical outlier detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_and_analyze_data(csv_file_path):
    """Load data and separate experiments from controls."""
    df = pd.read_csv(csv_file_path)
    
    experiment_data = df[df['well_type'] == 'experiment'].copy()
    control_data = df[df['well_type'] == 'control'].copy()
    
    print(f"Loaded {len(experiment_data)} experiment wells and {len(control_data)} controls")
    
    return experiment_data, control_data

def analyze_control_baselines(control_data):
    """Analyze control wells to establish baseline reference values."""
    baselines = {}
    
    for _, control in control_data.iterrows():
        control_type = control['control_type']
        baselines[control_type] = {
            'turbidity_600': control['turbidity_600'],
            'ratio': control['ratio'],
            'fluorescence_334_373': control['fluorescence_334_373'],
            'fluorescence_334_384': control['fluorescence_334_384']
        }
    
    print("\\nControl Reference Values:")
    for control_type, values in baselines.items():
        print(f"  {control_type}:")
        print(f"    Turbidity: {values['turbidity_600']:.4f}")
        print(f"    Ratio: {values['ratio']:.4f}")
    
    return baselines

def classify_baseline_wells(experiment_data, control_baselines):
    """
    Classify wells as baseline or non-baseline using multiple criteria.
    
    Criteria:
    1. Turbidity threshold: High turbidity suggests interaction
    2. Ratio deviation: Significant deviation from water blank suggests interaction
    3. Concentration threshold: Very low concentrations likely baseline
    4. Statistical outliers: Wells that are statistical outliers in turbidity
    """
    
    # Get reference values from controls
    water_turbidity = control_baselines.get('water_blank', {}).get('turbidity_600', 0.038)
    water_ratio = control_baselines.get('water_blank', {}).get('ratio', 0.88)
    
    # Calculate turbidity statistics for outlier detection
    turb_mean = experiment_data['turbidity_600'].mean()
    turb_std = experiment_data['turbidity_600'].std()
    turb_threshold_statistical = turb_mean + 2 * turb_std  # 2-sigma threshold
    
    # Define thresholds
    thresholds = {
        'turbidity_absolute': 0.1,  # Absolute turbidity threshold
        'turbidity_fold_change': 2.0,  # Fold change vs water blank
        'turbidity_statistical': turb_threshold_statistical,  # Statistical outlier
        'ratio_deviation': 0.05,  # Deviation from water blank ratio
        'concentration_threshold': 0.001  # mM - very low concentration cutoff
    }
    
    print(f"\\nClassification Thresholds:")
    print(f"  Turbidity absolute: >{thresholds['turbidity_absolute']:.3f}")
    print(f"  Turbidity fold change: >{thresholds['turbidity_fold_change']:.1f}x water blank ({water_turbidity:.4f})")
    print(f"  Turbidity statistical: >{thresholds['turbidity_statistical']:.4f} (mean + 2σ)")
    print(f"  Ratio deviation: ±{thresholds['ratio_deviation']:.3f} from water blank ({water_ratio:.4f})")
    print(f"  Low concentration: <{thresholds['concentration_threshold']:.4f} mM for both surfactants")
    
    # Create classification DataFrame
    classification_df = experiment_data.copy()
    
    # Apply criteria
    classification_df['high_turbidity_absolute'] = classification_df['turbidity_600'] > thresholds['turbidity_absolute']
    classification_df['high_turbidity_fold'] = classification_df['turbidity_600'] > (water_turbidity * thresholds['turbidity_fold_change'])
    classification_df['high_turbidity_statistical'] = classification_df['turbidity_600'] > thresholds['turbidity_statistical']
    classification_df['ratio_deviation'] = abs(classification_df['ratio'] - water_ratio) > thresholds['ratio_deviation']
    classification_df['very_low_concentrations'] = (
        (classification_df['surf_A_conc_mm'] < thresholds['concentration_threshold']) & 
        (classification_df['surf_B_conc_mm'] < thresholds['concentration_threshold'])
    )
    
    # Classification logic: Non-baseline if ANY interaction indicator is True AND not in very low concentration range
    classification_df['interaction_indicator'] = (
        classification_df['high_turbidity_absolute'] | 
        classification_df['high_turbidity_fold'] | 
        classification_df['high_turbidity_statistical'] | 
        classification_df['ratio_deviation']
    )
    
    classification_df['is_baseline'] = (
        ~classification_df['interaction_indicator'] | 
        classification_df['very_low_concentrations']
    )
    
    classification_df['classification'] = classification_df['is_baseline'].map({True: 'baseline', False: 'non-baseline'})
    
    return classification_df, thresholds

def generate_classification_summary(classification_df, thresholds):
    """Generate summary statistics for the classification."""
    
    total_wells = len(classification_df)
    baseline_wells = classification_df['is_baseline'].sum()
    non_baseline_wells = total_wells - baseline_wells
    
    print(f"\\nClassification Summary:")
    print(f"  Total wells: {total_wells}")
    print(f"  Baseline wells: {baseline_wells} ({baseline_wells/total_wells*100:.1f}%)")
    print(f"  Non-baseline wells: {non_baseline_wells} ({non_baseline_wells/total_wells*100:.1f}%)")
    
    print(f"\\nCriteria Breakdown:")
    print(f"  High turbidity (absolute): {classification_df['high_turbidity_absolute'].sum()} wells")
    print(f"  High turbidity (fold change): {classification_df['high_turbidity_fold'].sum()} wells")
    print(f"  High turbidity (statistical): {classification_df['high_turbidity_statistical'].sum()} wells")
    print(f"  Significant ratio deviation: {classification_df['ratio_deviation'].sum()} wells")
    print(f"  Very low concentrations: {classification_df['very_low_concentrations'].sum()} wells")
    
    # Show turbidity and ratio ranges for each class
    baseline_data = classification_df[classification_df['is_baseline']]
    non_baseline_data = classification_df[~classification_df['is_baseline']]
    
    print(f"\\nBaseline Wells Characteristics:")
    print(f"  Turbidity range: {baseline_data['turbidity_600'].min():.4f} - {baseline_data['turbidity_600'].max():.4f}")
    print(f"  Ratio range: {baseline_data['ratio'].min():.4f} - {baseline_data['ratio'].max():.4f}")
    print(f"  Concentration ranges:")
    print(f"    SDS: {baseline_data['surf_A_conc_mm'].min():.6f} - {baseline_data['surf_A_conc_mm'].max():.4f} mM")
    print(f"    TTAB: {baseline_data['surf_B_conc_mm'].min():.6f} - {baseline_data['surf_B_conc_mm'].max():.4f} mM")
    
    print(f"\\nNon-baseline Wells Characteristics:")
    if len(non_baseline_data) > 0:
        print(f"  Turbidity range: {non_baseline_data['turbidity_600'].min():.4f} - {non_baseline_data['turbidity_600'].max():.4f}")
        print(f"  Ratio range: {non_baseline_data['ratio'].min():.4f} - {non_baseline_data['ratio'].max():.4f}")
        print(f"  Concentration ranges:")
        print(f"    SDS: {non_baseline_data['surf_A_conc_mm'].min():.6f} - {non_baseline_data['surf_A_conc_mm'].max():.4f} mM")
        print(f"    TTAB: {non_baseline_data['surf_B_conc_mm'].min():.6f} - {non_baseline_data['surf_B_conc_mm'].max():.4f} mM")
    else:
        print("  No non-baseline wells found")
    
    return {
        'total_wells': total_wells,
        'baseline_wells': baseline_wells,
        'non_baseline_wells': non_baseline_wells,
        'baseline_percentage': baseline_wells/total_wells*100,
        'baseline_data': baseline_data,
        'non_baseline_data': non_baseline_data
    }

def create_classification_visualizations(classification_df, output_dir, summary_stats):
    """Create visualizations showing the baseline classification."""
    
    # Set up plotting
    plt.style.use('default')
    
    # Get concentration values for axes
    surf_A_concs = sorted(classification_df['surf_A_conc_mm'].unique())
    surf_B_concs = sorted(classification_df['surf_B_conc_mm'].unique())
    
    # Create pivot tables for all measurements and criteria
    classification_grid = classification_df.pivot_table(
        index='surf_B_conc_mm', columns='surf_A_conc_mm', 
        values='is_baseline', aggfunc='mean'
    ).reindex(sorted(surf_B_concs, reverse=True))
    classification_grid = classification_grid.reindex(sorted(surf_A_concs), axis=1)
    
    turbidity_grid = classification_df.pivot_table(
        index='surf_B_conc_mm', columns='surf_A_conc_mm', 
        values='turbidity_600', aggfunc='mean'
    ).reindex(sorted(surf_B_concs, reverse=True))
    turbidity_grid = turbidity_grid.reindex(sorted(surf_A_concs), axis=1)
    
    ratio_grid = classification_df.pivot_table(
        index='surf_B_conc_mm', columns='surf_A_conc_mm', 
        values='ratio', aggfunc='mean'
    ).reindex(sorted(surf_B_concs, reverse=True))
    ratio_grid = ratio_grid.reindex(sorted(surf_A_concs), axis=1)
    
    # Create criteria grids to show which criterion triggered classification
    turbidity_criteria_grid = classification_df.pivot_table(
        index='surf_B_conc_mm', columns='surf_A_conc_mm', 
        values='high_turbidity_absolute', aggfunc='mean'
    ).reindex(sorted(surf_B_concs, reverse=True))
    turbidity_criteria_grid = turbidity_criteria_grid.reindex(sorted(surf_A_concs), axis=1)
    
    ratio_criteria_grid = classification_df.pivot_table(
        index='surf_B_conc_mm', columns='surf_A_conc_mm', 
        values='ratio_deviation', aggfunc='mean'
    ).reindex(sorted(surf_B_concs, reverse=True))
    ratio_criteria_grid = ratio_criteria_grid.reindex(sorted(surf_A_concs), axis=1)
    
    # Format labels
    x_labels = [f'{x:.4f}' if x < 1 else f'{x:.1f}' for x in surf_A_concs]
    y_labels = [f'{y:.4f}' if y < 1 else f'{y:.1f}' for y in sorted(surf_B_concs, reverse=True)]  # Match the reindexed order
    
    # 1. Main analysis plot - 2x3 layout showing both measurements and criteria
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    
    # Row 1: Raw measurements
    # Turbidity data
    sns.heatmap(turbidity_grid, ax=axes[0,0], cmap='viridis', annot=True, fmt='.3f',
                cbar_kws={'label': 'Turbidity (600 nm)'},
                xticklabels=x_labels, yticklabels=y_labels)
    axes[0,0].set_title('Turbidity Measurements', fontweight='bold', fontsize=14)
    axes[0,0].set_xlabel('SDS Concentration (mM)')
    axes[0,0].set_ylabel('TTAB Concentration (mM)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Ratio data  
    sns.heatmap(ratio_grid, ax=axes[0,1], cmap='plasma', annot=True, fmt='.3f',
                cbar_kws={'label': 'Fluorescence Ratio'},
                xticklabels=x_labels, yticklabels=y_labels)
    axes[0,1].set_title('Fluorescence Ratio Measurements', fontweight='bold', fontsize=14)
    axes[0,1].set_xlabel('SDS Concentration (mM)')
    axes[0,1].set_ylabel('TTAB Concentration (mM)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Final classification
    classification_display = 1 - classification_grid  # Invert for display (1=non-baseline=red)
    sns.heatmap(classification_display, ax=axes[0,2], cmap='RdBu_r', vmin=0, vmax=1,
                annot=True, fmt='.0f', cbar_kws={'label': 'Non-baseline (1) vs Baseline (0)'},
                xticklabels=x_labels, yticklabels=y_labels)
    axes[0,2].set_title('Final Classification\\n(Red=Non-baseline, Blue=Baseline)', fontweight='bold', fontsize=14)
    axes[0,2].set_xlabel('SDS Concentration (mM)')
    axes[0,2].set_ylabel('TTAB Concentration (mM)')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # Row 2: Criteria contribution
    # Turbidity criteria (which wells triggered turbidity thresholds)
    sns.heatmap(turbidity_criteria_grid, ax=axes[1,0], cmap='Reds', vmin=0, vmax=1,
                annot=True, fmt='.0f', cbar_kws={'label': 'Turbidity Threshold Triggered (1=Yes, 0=No)'},
                xticklabels=x_labels, yticklabels=y_labels)
    axes[1,0].set_title('Wells Triggering Turbidity Criteria\\n(High Turbidity = Non-baseline)', fontweight='bold', fontsize=14)
    axes[1,0].set_xlabel('SDS Concentration (mM)')
    axes[1,0].set_ylabel('TTAB Concentration (mM)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Ratio criteria (which wells triggered ratio deviation)
    sns.heatmap(ratio_criteria_grid, ax=axes[1,1], cmap='Purples', vmin=0, vmax=1,
                annot=True, fmt='.0f', cbar_kws={'label': 'Ratio Deviation Triggered (1=Yes, 0=No)'},
                xticklabels=x_labels, yticklabels=y_labels)
    axes[1,1].set_title('Wells Triggering Ratio Criteria\\n(Ratio Deviation = Non-baseline)', fontweight='bold', fontsize=14)
    axes[1,1].set_xlabel('SDS Concentration (mM)')
    axes[1,1].set_ylabel('TTAB Concentration (mM)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Combined criteria - show which criterion was the deciding factor
    # Create a combined criteria grid: 0=baseline, 1=turbidity only, 2=ratio only, 3=both
    combined_criteria = np.zeros_like(turbidity_criteria_grid.values)
    turb_vals = turbidity_criteria_grid.fillna(0).values
    ratio_vals = ratio_criteria_grid.fillna(0).values
    
    combined_criteria[turb_vals == 1] += 1  # Turbidity criterion
    combined_criteria[ratio_vals == 1] += 2  # Ratio criterion
    
    sns.heatmap(combined_criteria, ax=axes[1,2], cmap='viridis', vmin=0, vmax=3,
                annot=True, fmt='.0f', 
                cbar_kws={'label': '0=Baseline, 1=Turbidity, 2=Ratio, 3=Both'},
                xticklabels=x_labels, yticklabels=y_labels)
    axes[1,2].set_title('Criteria Contribution\\n(Which measurement caused non-baseline classification)', fontweight='bold', fontsize=14)
    axes[1,2].set_xlabel('SDS Concentration (mM)')
    axes[1,2].set_ylabel('TTAB Concentration (mM)')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the main analysis plot
    analysis_path = os.path.join(output_dir, 'baseline_classification_detailed_analysis.png')
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\\nDetailed classification analysis saved: {os.path.basename(analysis_path)}")
    
    # 2. Scatter plots showing both measurements
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get data by classification
    baseline_data = classification_df[classification_df['is_baseline']]
    non_baseline_data = classification_df[~classification_df['is_baseline']]
    
    # Scatter plot: Turbidity vs Ratio colored by classification
    ax1.scatter(baseline_data['turbidity_600'], baseline_data['ratio'], 
               c='blue', alpha=0.6, label=f'Baseline (n={len(baseline_data)})', s=50)
    ax1.scatter(non_baseline_data['turbidity_600'], non_baseline_data['ratio'], 
               c='red', alpha=0.6, label=f'Non-baseline (n={len(non_baseline_data)})', s=50)
    ax1.set_xlabel('Turbidity (600 nm)')
    ax1.set_ylabel('Fluorescence Ratio')
    ax1.set_title('Wells by Turbidity vs Ratio\\n(Both measurements)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Show wells that triggered each criterion separately
    turb_only = classification_df[(classification_df['high_turbidity_absolute']) & (~classification_df['ratio_deviation'])]
    ratio_only = classification_df[(~classification_df['high_turbidity_absolute']) & (classification_df['ratio_deviation'])]
    both_criteria = classification_df[(classification_df['high_turbidity_absolute']) & (classification_df['ratio_deviation'])]
    neither = classification_df[(~classification_df['high_turbidity_absolute']) & (~classification_df['ratio_deviation'])]
    
    ax2.scatter(neither['turbidity_600'], neither['ratio'], c='lightblue', alpha=0.6, 
               label=f'Baseline (n={len(neither)})', s=50)
    ax2.scatter(turb_only['turbidity_600'], turb_only['ratio'], c='orange', alpha=0.7, 
               label=f'Turbidity only (n={len(turb_only)})', s=50)
    ax2.scatter(ratio_only['turbidity_600'], ratio_only['ratio'], c='purple', alpha=0.7, 
               label=f'Ratio only (n={len(ratio_only)})', s=50)
    ax2.scatter(both_criteria['turbidity_600'], both_criteria['ratio'], c='red', alpha=0.8, 
               label=f'Both criteria (n={len(both_criteria)})', s=50)
    ax2.set_xlabel('Turbidity (600 nm)')
    ax2.set_ylabel('Fluorescence Ratio')
    ax2.set_title('Wells by Criterion Triggered\\n(Which measurement caused classification)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Distribution comparisons
    baseline_turb = baseline_data['turbidity_600']
    non_baseline_turb = non_baseline_data['turbidity_600']
    
    ax3.hist(baseline_turb, bins=20, alpha=0.7, color='blue', label='Baseline', density=True)
    ax3.hist(non_baseline_turb, bins=20, alpha=0.7, color='red', label='Non-baseline', density=True)
    ax3.set_xlabel('Turbidity (600 nm)')
    ax3.set_ylabel('Density')
    ax3.set_title('Turbidity Distribution by Classification')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    baseline_ratio = baseline_data['ratio']
    non_baseline_ratio = non_baseline_data['ratio']
    
    ax4.hist(baseline_ratio, bins=20, alpha=0.7, color='blue', label='Baseline', density=True)
    ax4.hist(non_baseline_ratio, bins=20, alpha=0.7, color='red', label='Non-baseline', density=True)
    ax4.set_xlabel('Fluorescence Ratio')
    ax4.set_ylabel('Density')
    ax4.set_title('Fluorescence Ratio Distribution by Classification')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save distribution plot
    dist_path = os.path.join(output_dir, 'baseline_measurements_comparison.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Measurements comparison saved: {os.path.basename(dist_path)}")
    
    # Print summary of criteria contribution
    print(f"\\nCriteria Contribution Summary:")
    print(f"  Wells triggering turbidity criteria only: {len(turb_only)}")
    print(f"  Wells triggering ratio criteria only: {len(ratio_only)}")
    print(f"  Wells triggering both criteria: {len(both_criteria)}")
    print(f"  Wells triggering neither (baseline): {len(neither)}")

def save_classification_results(classification_df, output_dir, thresholds, summary_stats):
    """Save classification results to CSV files."""
    
    # Save full classification results
    results_path = os.path.join(output_dir, 'baseline_classification_results.csv')
    classification_df.to_csv(results_path, index=False)
    print(f"\\nClassification results saved: {os.path.basename(results_path)}")
    
    # Save just baseline wells
    baseline_wells_path = os.path.join(output_dir, 'baseline_wells_only.csv')
    baseline_wells = classification_df[classification_df['is_baseline']]
    baseline_wells.to_csv(baseline_wells_path, index=False)
    print(f"Baseline wells saved: {os.path.basename(baseline_wells_path)}")
    
    # Save just non-baseline wells
    non_baseline_wells_path = os.path.join(output_dir, 'non_baseline_wells_only.csv')
    non_baseline_wells = classification_df[~classification_df['is_baseline']]
    non_baseline_wells.to_csv(non_baseline_wells_path, index=False)
    print(f"Non-baseline wells saved: {os.path.basename(non_baseline_wells_path)}")
    
    # Save summary report
    summary_path = os.path.join(output_dir, 'baseline_classification_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Baseline Classification Summary Report\\n")
        f.write("="*50 + "\\n\\n")
        
        f.write("Classification Thresholds Used:\\n")
        for key, value in thresholds.items():
            f.write(f"  {key}: {value}\\n")
        f.write("\\n")
        
        f.write("Results Summary:\\n")
        f.write(f"  Total wells analyzed: {summary_stats['total_wells']}\\n")
        f.write(f"  Baseline wells: {summary_stats['baseline_wells']} ({summary_stats['baseline_percentage']:.1f}%)\\n")
        f.write(f"  Non-baseline wells: {summary_stats['non_baseline_wells']} ({100-summary_stats['baseline_percentage']:.1f}%)\\n")
        f.write("\\n")
        
        f.write("Baseline Wells Characteristics:\\n")
        baseline_data = summary_stats['baseline_data']
        f.write(f"  Turbidity range: {baseline_data['turbidity_600'].min():.4f} - {baseline_data['turbidity_600'].max():.4f}\\n")
        f.write(f"  Ratio range: {baseline_data['ratio'].min():.4f} - {baseline_data['ratio'].max():.4f}\\n")
        
        f.write("\\nNon-baseline Wells Characteristics:\\n")
        non_baseline_data = summary_stats['non_baseline_data']
        if len(non_baseline_data) > 0:
            f.write(f"  Turbidity range: {non_baseline_data['turbidity_600'].min():.4f} - {non_baseline_data['turbidity_600'].max():.4f}\\n")
            f.write(f"  Ratio range: {non_baseline_data['ratio'].min():.4f} - {non_baseline_data['ratio'].max():.4f}\\n")
        else:
            f.write("  No non-baseline wells found\\n")
    
    print(f"Summary report saved: {os.path.basename(summary_path)}")

def main():
    """Main function to run baseline classification analysis."""
    
    # Find the CSV file in current directory
    current_dir = Path(__file__).parent
    csv_files = list(current_dir.glob("complete_experiment_results.csv"))
    
    if not csv_files:
        print("Error: No 'complete_experiment_results.csv' found in current directory.")
        return
    
    csv_file_path = csv_files[0]
    print(f"Analyzing data from: {csv_file_path.name}")
    print("="*60)
    
    # Load and analyze data
    experiment_data, control_data = load_and_analyze_data(csv_file_path)
    
    # Analyze control baselines
    control_baselines = analyze_control_baselines(control_data)
    
    # Classify wells
    classification_df, thresholds = classify_baseline_wells(experiment_data, control_baselines)
    
    # Generate summary
    summary_stats = generate_classification_summary(classification_df, thresholds)
    
    # Create output directory
    output_dir = current_dir / "baseline_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Create visualizations
    create_classification_visualizations(classification_df, output_dir, summary_stats)
    
    # Save results
    save_classification_results(classification_df, output_dir, thresholds, summary_stats)
    
    print(f"\\n" + "="*60)
    print("BASELINE CLASSIFICATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"Classification: {summary_stats['baseline_wells']} baseline, {summary_stats['non_baseline_wells']} non-baseline wells")

if __name__ == "__main__":
    main()
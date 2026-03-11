"""
Minimum Concentration Threshold Analysis

This program analyzes baseline classification results to determine optimal minimum 
concentrations for future surfactant experiments, excluding baseline regions.

Strategies:
1. Conservative: Lowest concentrations where ANY non-baseline behavior appears
2. Moderate: Concentrations where 50% of wells become non-baseline  
3. Aggressive: Concentrations where majority (75%+) of wells are non-baseline

Analysis considers:
- Individual surfactant effects (SDS-only, TTAB-only)
- Combined/synergistic effects
- Concentration boundaries along grid axes
- Statistical transition points
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_classification_data(classification_file_path):
    """Load the baseline classification results."""
    if not os.path.exists(classification_file_path):
        print(f"Error: Classification file not found: {classification_file_path}")
        print("Please run baseline_classification.py first!")
        return None
    
    df = pd.read_csv(classification_file_path)
    print(f"Loaded {len(df)} classified wells")
    
    baseline_count = df['is_baseline'].sum()
    non_baseline_count = len(df) - baseline_count
    print(f"  Baseline: {baseline_count} wells ({baseline_count/len(df)*100:.1f}%)")
    print(f"  Non-baseline: {non_baseline_count} wells ({non_baseline_count/len(df)*100:.1f}%)")
    
    return df

def analyze_concentration_boundaries(classification_df):
    """
    Analyze where baseline→non-baseline transitions occur in concentration space.
    """
    
    # Get unique concentrations
    surf_A_concs = sorted(classification_df['surf_A_conc_mm'].unique())
    surf_B_concs = sorted(classification_df['surf_B_conc_mm'].unique())
    
    print(f"\\nConcentration Analysis:")
    print(f"SDS concentrations: {len(surf_A_concs)} levels from {min(surf_A_concs):.6f} to {max(surf_A_concs):.1f} mM")
    print(f"TTAB concentrations: {len(surf_B_concs)} levels from {min(surf_B_concs):.6f} to {max(surf_B_concs):.1f} mM")
    
    results = {}
    
    # 1. Analyze SDS-axis transitions (varying SDS at each fixed TTAB level)
    print(f"\\n1. SDS Axis Analysis (transitions along SDS concentration):")
    sds_transitions = {}
    
    for ttab_conc in surf_B_concs:
        # Get wells at this TTAB level, sorted by SDS concentration
        ttab_slice = classification_df[classification_df['surf_B_conc_mm'] == ttab_conc].copy()
        ttab_slice = ttab_slice.sort_values('surf_A_conc_mm')
        
        if len(ttab_slice) == 0:
            continue
            
        # Find first non-baseline well
        first_non_baseline = ttab_slice[~ttab_slice['is_baseline']]
        if len(first_non_baseline) > 0:
            min_sds_for_effect = first_non_baseline['surf_A_conc_mm'].min()
            sds_transitions[ttab_conc] = min_sds_for_effect
            print(f"  TTAB={ttab_conc:.6f} mM: First non-baseline at SDS={min_sds_for_effect:.6f} mM")
        else:
            sds_transitions[ttab_conc] = None
            print(f"  TTAB={ttab_conc:.6f} mM: No non-baseline wells found")
    
    # 2. Analyze TTAB-axis transitions (varying TTAB at each fixed SDS level)
    print(f"\\n2. TTAB Axis Analysis (transitions along TTAB concentration):")
    ttab_transitions = {}
    
    for sds_conc in surf_A_concs:
        # Get wells at this SDS level, sorted by TTAB concentration  
        sds_slice = classification_df[classification_df['surf_A_conc_mm'] == sds_conc].copy()
        sds_slice = sds_slice.sort_values('surf_B_conc_mm')
        
        if len(sds_slice) == 0:
            continue
            
        # Find first non-baseline well
        first_non_baseline = sds_slice[~sds_slice['is_baseline']]
        if len(first_non_baseline) > 0:
            min_ttab_for_effect = first_non_baseline['surf_B_conc_mm'].min()
            ttab_transitions[sds_conc] = min_ttab_for_effect
            print(f"  SDS={sds_conc:.6f} mM: First non-baseline at TTAB={min_ttab_for_effect:.6f} mM")
        else:
            ttab_transitions[sds_conc] = None
            print(f"  SDS={sds_conc:.6f} mM: No non-baseline wells found")
    
    results['sds_transitions'] = sds_transitions
    results['ttab_transitions'] = ttab_transitions
    
    return results

def calculate_statistical_thresholds(classification_df):
    """
    Calculate statistical thresholds where >50% of wells become non-baseline.
    """
    
    surf_A_concs = sorted(classification_df['surf_A_conc_mm'].unique())
    surf_B_concs = sorted(classification_df['surf_B_conc_mm'].unique())
    
    # Find concentrations where >50% wells are non-baseline
    sds_threshold = find_statistical_threshold(classification_df, 'surf_A_conc_mm', 0.5)
    ttab_threshold = find_statistical_threshold(classification_df, 'surf_B_conc_mm', 0.5)
    
    strategy = {
        'sds_min': sds_threshold,
        'ttab_min': ttab_threshold,
        'description': 'Concentrations where >50% wells are non-baseline',
        'threshold_fraction': 0.5
    }
    
    print(f"\\nStatistical Threshold Analysis (>50% non-baseline):")
    print(f"  Minimum SDS: {sds_threshold:.6f} mM")
    print(f"  Minimum TTAB: {ttab_threshold:.6f} mM")
    
    # Calculate how many wells would be excluded
    excluded_wells = classification_df[
        (classification_df['surf_A_conc_mm'] < sds_threshold) | 
        (classification_df['surf_B_conc_mm'] < ttab_threshold)
    ]
    
    remaining_wells = classification_df[
        (classification_df['surf_A_conc_mm'] >= sds_threshold) & 
        (classification_df['surf_B_conc_mm'] >= ttab_threshold)
    ]
    
    print(f"\\nImpact Analysis:")
    print(f"  Wells excluded: {len(excluded_wells)} ({len(excluded_wells)/len(classification_df)*100:.1f}%)")
    print(f"  Wells remaining: {len(remaining_wells)} ({len(remaining_wells)/len(classification_df)*100:.1f}%)")
    
    if len(remaining_wells) > 0:
        remaining_baseline = remaining_wells['is_baseline'].sum()
        remaining_non_baseline = len(remaining_wells) - remaining_baseline
        print(f"  Remaining baseline: {remaining_baseline} ({remaining_baseline/len(remaining_wells)*100:.1f}%)")
        print(f"  Remaining non-baseline: {remaining_non_baseline} ({remaining_non_baseline/len(remaining_wells)*100:.1f}%)")
    
    return strategy



def largest_rectangle_in_histogram(heights):
    """
    Find the area of the largest rectangle in a histogram.
    Returns (area, left_idx, right_idx, height)
    """
    stack = []
    max_area = 0
    best_rect = (0, 0, 0, 0)  # (area, left, right, height)
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            area = height * width
            if area > max_area:
                max_area = area
                left = 0 if not stack else stack[-1] + 1
                best_rect = (area, left, i - 1, height)
        stack.append(i)
    
    while stack:
        height = heights[stack.pop()]
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        area = height * width
        if area > max_area:
            max_area = area
            left = 0 if not stack else stack[-1] + 1
            best_rect = (area, left, len(heights) - 1, height)
    
    return best_rect

def find_largest_baseline_rectangle(classification_df):
    """
    Find the largest rectangle that contains only baseline wells.
    """
    
    print("\\nCalculating largest baseline rectangle...")
    
    # Get sorted unique concentrations
    surf_A_concs = sorted(classification_df['surf_A_conc_mm'].unique())
    surf_B_concs = sorted(classification_df['surf_B_conc_mm'].unique())
    
    print(f"Grid dimensions: {len(surf_A_concs)} x {len(surf_B_concs)}")
    
    # Create binary matrix (1 = baseline, 0 = non-baseline)
    # Note: surf_B (TTAB) will be rows (y-axis), surf_A (SDS) will be columns (x-axis)
    matrix = np.zeros((len(surf_B_concs), len(surf_A_concs)), dtype=int)
    
    for i, ttab in enumerate(surf_B_concs):
        for j, sds in enumerate(surf_A_concs):
            well = classification_df[
                (classification_df['surf_A_conc_mm'] == sds) & 
                (classification_df['surf_B_conc_mm'] == ttab)
            ]
            if len(well) > 0:
                matrix[i, j] = 1 if well.iloc[0]['is_baseline'] else 0
    
    print("Baseline matrix (1=baseline, 0=non-baseline):")
    print("Rows = TTAB (low to high), Cols = SDS (low to high)")
    for i, ttab in enumerate(surf_B_concs):
        row_str = f"TTAB {ttab:7.4f}: "
        for j in range(len(surf_A_concs)):
            row_str += f"{matrix[i, j]} "
        print(row_str)
    
    # Find largest rectangle using histogram method
    max_area = 0
    best_rectangle = None
    
    # For each row, calculate heights of consecutive 1s ending at that row
    heights = np.zeros(len(surf_A_concs), dtype=int)
    
    for row in range(len(surf_B_concs)):
        # Update heights for this row
        for col in range(len(surf_A_concs)):
            if matrix[row, col] == 1:
                heights[col] += 1
            else:
                heights[col] = 0
        
        # Find largest rectangle in this histogram
        area, left_col, right_col, height = largest_rectangle_in_histogram(heights)
        
        if area > max_area:
            max_area = area
            bottom_row = row - height + 1
            top_row = row
            best_rectangle = {
                'area': area,
                'bottom_row': bottom_row,
                'top_row': top_row,
                'left_col': left_col,
                'right_col': right_col,
                'sds_min': surf_A_concs[left_col],
                'sds_max': surf_A_concs[right_col],
                'ttab_min': surf_B_concs[bottom_row],
                'ttab_max': surf_B_concs[top_row]
            }
    
    if best_rectangle is None:
        print("No baseline rectangle found!")
        return None
    
    print(f"\\nLargest baseline rectangle found:")
    print(f"  Area: {best_rectangle['area']} wells")
    print(f"  SDS range: {best_rectangle['sds_min']:.6f} - {best_rectangle['sds_max']:.6f} mM")
    print(f"  TTAB range: {best_rectangle['ttab_min']:.6f} - {best_rectangle['ttab_max']:.6f} mM")
    print(f"  Grid position: rows {best_rectangle['bottom_row']}-{best_rectangle['top_row']}, cols {best_rectangle['left_col']}-{best_rectangle['right_col']}")
    
    # The thresholds are the upper bounds of the baseline rectangle
    strategy = {
        'description': 'Largest rectangle containing only baseline wells',
        'sds_min': best_rectangle['sds_max'],  # Upper bound becomes minimum for next experiments
        'ttab_min': best_rectangle['ttab_max'],  # Upper bound becomes minimum for next experiments
        'rectangle_info': best_rectangle
    }
    
    return strategy

def create_threshold_visualization(classification_df, strategy, output_dir):
    """
    Create visualization showing the largest baseline rectangle strategy.
    """
    
    # Get concentration grids
    surf_A_concs = sorted(classification_df['surf_A_conc_mm'].unique())
    surf_B_concs = sorted(classification_df['surf_B_conc_mm'].unique())
    
    # Create classification grid
    classification_grid = classification_df.pivot_table(
        index='surf_B_conc_mm', columns='surf_A_conc_mm', 
        values='is_baseline', aggfunc='mean'
    ).reindex(sorted(surf_B_concs, reverse=True))
    classification_grid = classification_grid.reindex(sorted(surf_A_concs), axis=1)
    
    # Invert for display (1=non-baseline=red)
    classification_display = 1 - classification_grid
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Classification with baseline rectangle
    sns.heatmap(classification_display, ax=ax1, cmap='RdBu_r', vmin=0, vmax=1,
               annot=True, fmt='.0f', cbar_kws={'label': 'Non-baseline (1) vs Baseline (0)'},
               xticklabels=[f'{x:.4f}' if x < 1 else f'{x:.1f}' for x in surf_A_concs],
               yticklabels=[f'{y:.4f}' if y < 1 else f'{y:.1f}' for y in sorted(surf_B_concs, reverse=True)])
    
    # Draw baseline rectangle if available
    if 'rectangle_info' in strategy:
        rect_info = strategy['rectangle_info']
        
        # Convert to display coordinates (reversed y-axis)
        left_col = rect_info['left_col']
        right_col = rect_info['right_col'] + 1  # +1 for rectangle width
        bottom_row_display = len(surf_B_concs) - 1 - rect_info['top_row']
        top_row_display = len(surf_B_concs) - rect_info['bottom_row'] # +1 for rectangle height
        
        # Draw rectangle outline
        from matplotlib.patches import Rectangle
        rect_patch = Rectangle((left_col, bottom_row_display), 
                              right_col - left_col, 
                              top_row_display - bottom_row_display,
                              linewidth=3, edgecolor='green', facecolor='none', 
                              linestyle='-', label='Largest baseline rectangle')
        ax1.add_patch(rect_patch)
        
        # Draw threshold lines (upper bounds of rectangle)
        sds_threshold = strategy['sds_min']
        ttab_threshold = strategy['ttab_min']
        
        # Find grid positions for thresholds
        try:
            sds_idx = surf_A_concs.index(sds_threshold) + 0.5
            ttab_idx = sorted(surf_B_concs, reverse=True).index(ttab_threshold) + 0.5
        except ValueError:
            sds_idx = min(range(len(surf_A_concs)), key=lambda x: abs(surf_A_concs[x] - sds_threshold)) + 0.5
            ttab_idx = min(range(len(surf_B_concs)), key=lambda x: abs(sorted(surf_B_concs, reverse=True)[x] - ttab_threshold)) + 0.5
        
        # Draw threshold lines
        ax1.axvline(x=sds_idx, color='red', linewidth=2, linestyle='--', label=f'New SDS min: {sds_threshold:.4f}')
        ax1.axhline(y=ttab_idx, color='red', linewidth=2, linestyle='--', label=f'New TTAB min: {ttab_threshold:.4f}')
    
    ax1.set_xlabel('SDS Concentration (mM)')
    ax1.set_ylabel('TTAB Concentration (mM)')
    ax1.set_title('Baseline Classification with Largest Baseline Rectangle\\n(Green box = all baseline, Red lines = new thresholds)', fontweight='bold', fontsize=14)
    ax1.legend()
    
    # Plot 2: Excluded vs Included wells
    excluded_wells = classification_df[
        (classification_df['surf_A_conc_mm'] <= strategy['sds_min']) & 
        (classification_df['surf_B_conc_mm'] <= strategy['ttab_min'])
    ]
    
    included_wells = classification_df[
        (classification_df['surf_A_conc_mm'] > strategy['sds_min']) | 
        (classification_df['surf_B_conc_mm'] > strategy['ttab_min'])
    ]
    
    # Plot excluded wells
    ax2.scatter(excluded_wells['surf_A_conc_mm'], excluded_wells['surf_B_conc_mm'], 
               c='lightgray', s=100, alpha=0.7, label=f'Excluded (n={len(excluded_wells)})')
    
    # Plot included wells by type
    included_baseline = included_wells[included_wells['is_baseline'] == True]
    included_non_baseline = included_wells[included_wells['is_baseline'] == False]
    
    ax2.scatter(included_baseline['surf_A_conc_mm'], included_baseline['surf_B_conc_mm'],
               c='blue', s=100, alpha=0.7, label=f'Included Baseline (n={len(included_baseline)})')
    ax2.scatter(included_non_baseline['surf_A_conc_mm'], included_non_baseline['surf_B_conc_mm'],
               c='red', s=100, alpha=0.7, label=f'Included Non-baseline (n={len(included_non_baseline)})')
    
    # Draw threshold lines
    ax2.axvline(x=strategy['sds_min'], color='green', linewidth=2, linestyle='--')
    ax2.axhline(y=strategy['ttab_min'], color='green', linewidth=2, linestyle='--')
    
    ax2.set_xlabel('SDS Concentration (mM)')
    ax2.set_ylabel('TTAB Concentration (mM)')
    ax2.set_title('Experimental Wells: Excluded vs Included\\n(Baseline Rectangle Strategy)', fontweight='bold', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout(pad=2.0)
    
    # Save visualization
    vis_path = os.path.join(output_dir, 'baseline_rectangle_strategy.png')
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\\nBaseline rectangle visualization saved: {os.path.basename(vis_path)}")
    
    ax1.set_title('Statistical Threshold Strategy\\n(>50% Wells Non-baseline)', fontweight='bold', fontsize=14)
    ax1.set_xlabel('SDS Concentration (mM)')
    ax1.set_ylabel('TTAB Concentration (mM)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add threshold text
    ax1.text(0.02, 0.98, f'Min SDS: {sds_threshold:.4f} mM\\nMin TTAB: {ttab_threshold:.4f} mM', 
           transform=ax1.transAxes, verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Show what gets excluded vs included
    excluded_wells = classification_df[
        (classification_df['surf_A_conc_mm'] < sds_threshold) | 
        (classification_df['surf_B_conc_mm'] < ttab_threshold)
    ]
    
    included_wells = classification_df[
        (classification_df['surf_A_conc_mm'] >= sds_threshold) & 
        (classification_df['surf_B_conc_mm'] >= ttab_threshold)
    ]
    
    # Scatter plot of excluded vs included wells
    ax2.scatter(excluded_wells['surf_A_conc_mm'], excluded_wells['surf_B_conc_mm'], 
               c='lightgray', alpha=0.6, label=f'Excluded (n={len(excluded_wells)})', s=50)
    ax2.scatter(included_wells['surf_A_conc_mm'], included_wells['surf_B_conc_mm'], 
               c=included_wells['is_baseline'].map({True: 'blue', False: 'red'}), 
               alpha=0.7, s=50)
    
    # Add custom legend for included wells
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=8, label=f'Excluded (n={len(excluded_wells)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label=f'Included Baseline (n={included_wells["is_baseline"].sum()})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label=f'Included Non-baseline (n={(~included_wells["is_baseline"]).sum()})')
    ]
    ax2.legend(handles=legend_elements)
    
    # Draw threshold lines
    ax2.axvline(x=sds_threshold, color='green', linewidth=2, linestyle='--')
    ax2.axhline(y=ttab_threshold, color='green', linewidth=2, linestyle='--')
    
    ax2.set_xlabel('SDS Concentration (mM)')
    ax2.set_ylabel('TTAB Concentration (mM)')
    ax2.set_title('Experimental Wells: Excluded vs Included\\n(Statistical Threshold Strategy)', fontweight='bold', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    
    # Save visualization
    vis_path = os.path.join(output_dir, 'statistical_threshold_strategy.png')
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\\nStatistical threshold visualization saved: {os.path.basename(vis_path)}")

def save_threshold_recommendations(strategy, boundary_analysis, output_dir):
    """
    Save statistical threshold recommendations to files for future experiments.
    """
    
    # Save detailed analysis
    analysis_path = os.path.join(output_dir, 'statistical_threshold_analysis.txt')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write("Baseline Rectangle Threshold Analysis Report\\n")
        f.write("="*50 + "\\n\\n")
        
        f.write("STRATEGY: Largest Baseline Rectangle\\n")
        f.write("-"*50 + "\\n")
        f.write(f"Description: {strategy['description']}\\n")
        f.write(f"Minimum SDS: {strategy['sds_min']:.6f} mM\\n")
        f.write(f"Minimum TTAB: {strategy['ttab_min']:.6f} mM\\n\\n")
        
        f.write("BOUNDARY ANALYSIS (for reference):\\n")
        f.write("-"*30 + "\\n")
        f.write("SDS transitions (first non-baseline at each TTAB level):\\n")
        for ttab, sds in boundary_analysis['sds_transitions'].items():
            if sds is not None:
                f.write(f"  TTAB {ttab:.6f} mM -> SDS {sds:.6f} mM\n")
        
        f.write("\nTTAB transitions (first non-baseline at each SDS level):\n")
        for sds, ttab in boundary_analysis['ttab_transitions'].items():
            if ttab is not None:
                f.write(f"  SDS {sds:.6f} mM -> TTAB {ttab:.6f} mM\n")
    
    print(f"Detailed analysis saved: {os.path.basename(analysis_path)}")
    
    # Save machine-readable recommendations
    recommendations_path = os.path.join(output_dir, 'baseline_rectangle_concentrations.csv')
    
    recommendations_data = [{
        'strategy': 'baseline_rectangle',
        'description': strategy['description'],
        'sds_min_mm': strategy['sds_min'],
        'ttab_min_mm': strategy['ttab_min']
    }]
    
    if 'rectangle_info' in strategy:
        recommendations_data[0].update({
            'rectangle_area': strategy['rectangle_info']['area'],
            'rectangle_sds_max': strategy['rectangle_info']['sds_max'],
            'rectangle_ttab_max': strategy['rectangle_info']['ttab_max']
        })
    
    recommendations_df = pd.DataFrame(recommendations_data)
    recommendations_df.to_csv(recommendations_path, index=False)
    print(f"Machine-readable recommendations saved: {os.path.basename(recommendations_path)}")
    
    # Print summary
    print(f"\\n" + "="*60)
    print("STATISTICAL CONCENTRATION THRESHOLD RECOMMENDATION")
    print("="*60)
    print(f"Strategy: {strategy['description']}")
    print(f"-> Minimum SDS: {strategy['sds_min']:.6f} mM")
    print(f"-> Minimum TTAB: {strategy['ttab_min']:.6f} mM")
    
    print(f"\\n" + "="*60)
    print("USAGE FOR FUTURE EXPERIMENTS:")
    print("="*60)
    print("Use these minimum concentrations to:")
    print("• Focus on regions with clear surfactant interactions")
    print("• Skip baseline regions entirely") 
    print("• Reduce total experimental wells needed")
    print("• Improve signal-to-noise ratio in measurements")

def main():
    """Main function to run statistical concentration threshold analysis."""
    
    # Look for classification results
    current_dir = Path(__file__).parent
    classification_file = current_dir / "baseline_analysis" / "baseline_classification_results.csv"
    
    if not classification_file.exists():
        print("Error: No baseline classification results found!")
        print("Please run baseline_classification.py first.")
        return
    
    print("Baseline Rectangle Analysis")
    print("="*50)
    
    # Load classification data
    classification_df = load_classification_data(classification_file)
    if classification_df is None:
        return
    
    # Analyze concentration boundaries (for reference)
    boundary_analysis = analyze_concentration_boundaries(classification_df)
    
    # Calculate largest baseline rectangle strategy
    strategy = find_largest_baseline_rectangle(classification_df)
    if strategy is None:
        return
    
    # Create output directory
    output_dir = current_dir / "concentration_thresholds"
    output_dir.mkdir(exist_ok=True)
    
    # Create visualizations
    create_threshold_visualization(classification_df, strategy, output_dir)
    
    # Save recommendations
    save_threshold_recommendations(strategy, boundary_analysis, output_dir)
    
    print(f"\\nAnalysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
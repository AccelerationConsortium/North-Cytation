# -*- coding: utf-8 -*-
"""
Surfactant Grid Turbidity + Fluorescence Screening Workflow - Adaptive Concentrations
Systematic dilution grid of two surfactants with adaptive concentration ranges based on stock concentrations.

UPDATED CONCENTRATION APPROACH:
- Uses adaptive concentration ranges: min_conc = 10^-4 mM, max_conc = stock_conc * (allocated_volume / well_volume)
- Logarithmic spacing with fixed number of concentrations (9 by default)
- Each surfactant gets its own optimized concentration range based on volume constraints

VALIDATION MODES:
- Set VALIDATE_LIQUIDS=True to run validation alongside full experiment
- Set VALIDATION_ONLY=True to run only pipetting validation and skip experiment (great for testing)
- Both modes save validation results to experiment_name/calibration_validation/

DATA PROTECTION FEATURES:
- Raw Cytation data is immediately backed up to output/cytation_raw_backups/ (preserves complete original data)
- Processed measurement data is backed up to output/measurement_backups/ after each interval
- If processing fails, use recover_raw_cytation_data() and recover_from_measurement_backups() functions

RECOVERY USAGE:
  # Recover original Cytation data if processing failed:
  raw_data_list = recover_raw_cytation_data()
  
  # Recover processed measurements from crashed workflow:  
  measurements = recover_from_measurement_backups()
"""

# ================================================================================
# IMPORTS AND DEPENDENCIES
# ================================================================================

import sys
sys.path.append("../utoronto_demo")
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from master_usdl_coordinator import Lash_E, flatten_cytation_data
import slack_agent

# ================================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# ================================================================================

# Surfactant library with stock concentrations (from cmc_exp_new.py)
SURFACTANT_LIBRARY = {
    "SDS": {
        "full_name": "Sodium Dodecyl Sulfate",
        "category": "anionic",
        "stock_conc": 50,  # mM
    },
    "NaDC": {
        "full_name": "Sodium Docusate", 
        "category": "anionic",
        "stock_conc": 25,  # mM
    },
    "NaC": {
        "full_name": "Sodium Cholate",
        "category": "anionic", 
        "stock_conc": 50,  # mM
    },
    "CTAB": {
        "full_name": "Hexadecyltrimethylammonium Bromide",
        "category": "cationic",
        "stock_conc": 5,  # mM
    },
    "DTAB": {
        "full_name": "Dodecyltrimethylammonium Bromide",
        "category": "cationic",
        "stock_conc": 50,  # mM
    },
    "TTAB": {
        "full_name": "Tetradecyltrimethylammonium Bromide", 
        "category": "cationic",
        "stock_conc": 50,  # mM
    }
}

SURFACTANT_A = "SDS"
SURFACTANT_B = "TTAB"

# WORKFLOW CONSTANTS
SIMULATE = True # Set to False for actual hardware execution
VALIDATE_LIQUIDS = False # Set to False to skip pipetting validation during initialization
CREATE_WELLPLATE = True  # Set to True to create wellplate, False to skip to measurements only
VALIDATION_ONLY = False  # Set to True to run only pipetting validation and skip experiment (great for testing)

# WORKFLOW OPTIONS
RUN_2_STAGE_WORKFLOW = False  # Set to True to run 2-stage adaptive workflow
RUN_SINGLE_STAGE = False       # Set to True to run single-stage workflow
RUN_ITERATIVE_WORKFLOW = True #Using triangles

# Adaptive grid parameters - concentration ranges adapt to stock concentrations
MIN_CONC = 10**-4  # 0.0001 mM minimum concentration for all surfactants
NUMBER_CONCENTRATIONS = 9  # Number of concentration steps in logarithmic grid
N_REPLICATES = 1
WELL_VOLUME_UL = 200  # uL per well
PYRENE_VOLUME_UL = 5  # uL pyrene_DMSO to add per well
ITERATIVE_MEASUREMENT_TOTAL= 192 #The number of measurements done

# Buffer addition settings
ADD_BUFFER = False  # Set to False to skip buffer addition
BUFFER_VOLUME_UL = 20  # uL buffer to add per well
BUFFER_OPTIONS = ['MES', 'HEPES', 'CAPS']  # Available buffers
SELECTED_BUFFER = 'HEPES'  # Choose from BUFFER_OPTIONS

# Volume calculation with buffer compensation
# Always reserve space for buffer and pyrene to maintain consistent concentration ranges
# This ensures max concentrations are always the same regardless of ADD_BUFFER setting
EFFECTIVE_SURFACTANT_VOLUME = (WELL_VOLUME_UL - BUFFER_VOLUME_UL ) / 2  # Always reserve space
#lash_e.logger.info(f"Effective surfactant volume: {EFFECTIVE_SURFACTANT_VOLUME} uL per surfactant (reserves space for buffer+pyrene)")

# CRITICAL: Concentration correction factor for buffer dilution
# When buffer is added, stock concentrations must be higher to compensate for dilution
# This ensures final concentrations match intended values
CONCENTRATION_CORRECTION_FACTOR = WELL_VOLUME_UL / (2 * EFFECTIVE_SURFACTANT_VOLUME)
#lash_e.logger.info(f"Concentration correction factor: {CONCENTRATION_CORRECTION_FACTOR:.3f} (buffer={ADD_BUFFER}, buffer_vol={BUFFER_VOLUME_UL if ADD_BUFFER else 0}uL)")
MAX_WELLS = 96 #Wellplate size

# Constants
FINAL_SUBSTOCK_VOLUME = 6  # mL final volume for each dilution
MINIMUM_PIPETTE_VOLUME = 0.2  # mL (200 uL) - minimum volume for accurate pipetting
MEASUREMENT_INTERVAL = 96    # Measure every N wells to prevent evaporation

# Pipetting volume limits (consistent across all functions)
MIN_WELL_PIPETTE_VOLUME_UL = 10.0  # Minimum volume for well dispensing
MAX_SURFACTANT_VOLUME_UL = 90.0   # Maximum surfactant volume per well

# Measurement protocol files for Cytation TODO: Add shaking for 10 minutes... 
TURBIDITY_PROTOCOL_FILE = r"C:\Protocols\CMC_Absorbance_96_shake.prt"
FLUORESCENCE_PROTOCOL_FILE = r"C:\Protocols\CMC_Fluorescence_96_shake.prt"



# File paths
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/surfactant_grid_vials_expanded.csv"

# ================================================================================
# HEATMAP VISUALIZATION FUNCTIONS
# ================================================================================

def generate_surfactant_grid_heatmaps(csv_file_path, output_dir, logger, surfactant_a_name, surfactant_b_name):
    """
    Generate heatmap visualizations for surfactant grid experiment results.
    Creates both individual and combined plots with controls displayed below main grids.
    
    Args:
        csv_file_path (str): Path to the complete_experiment_results.csv file
        output_dir (str): Directory where heatmaps should be saved  
        logger: Logger instance for reporting progress
        surfactant_a_name (str): Name of surfactant A (x-axis)
        surfactant_b_name (str): Name of surfactant B (y-axis)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        
        logger.info(f"    Creating heatmaps from: {os.path.basename(csv_file_path)}")
        
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Filter data
        experiment_data = df[df['well_type'] == 'experiment'].copy()
        control_data = df[df['well_type'] == 'control'].copy()
        
        if experiment_data.empty:
            logger.warning("    No experiment data found - skipping heatmap generation")
            return
            
        # Process control data - handle NaN values
        control_processed = []
        for _, row in control_data.iterrows():
            control_info = {
                'control_type': row['control_type'],
                'surf_A_conc_mm': row['surf_A_conc_mm'] if not pd.isna(row['surf_A_conc_mm']) else 0,
                'surf_B_conc_mm': row['surf_B_conc_mm'] if not pd.isna(row['surf_B_conc_mm']) else 0,
                'turbidity_600': row['turbidity_600'],
                'ratio': row['ratio']
            }
            control_processed.append(control_info)
        
        if not control_processed:
            logger.warning("    No control data found - creating experiment-only heatmaps")
            control_df = pd.DataFrame()
        else:
            control_df = pd.DataFrame(control_processed)
        
        # Get concentration ranges
        surf_A_concs = sorted(experiment_data['surf_A_conc_mm'].unique())
        surf_B_concs = sorted(experiment_data['surf_B_conc_mm'].unique())
        
        logger.info(f"    Grid size: {len(surf_A_concs)} x {len(surf_B_concs)}")
        
        # Create pivot tables for heatmap data (high concentrations top-right)
        turbidity_grid = experiment_data.pivot_table(
            index='surf_B_conc_mm', columns='surf_A_conc_mm', values='turbidity_600', aggfunc='mean'
        ).reindex(sorted(experiment_data['surf_B_conc_mm'].unique(), reverse=True))
        turbidity_grid = turbidity_grid.reindex(sorted(experiment_data['surf_A_conc_mm'].unique()), axis=1)
        
        ratio_grid = experiment_data.pivot_table(
            index='surf_B_conc_mm', columns='surf_A_conc_mm', values='ratio', aggfunc='mean'
        ).reindex(sorted(experiment_data['surf_B_conc_mm'].unique(), reverse=True))
        ratio_grid = ratio_grid.reindex(sorted(experiment_data['surf_A_conc_mm'].unique()), axis=1)
        
        # Determine color scale ranges
        all_turbidity = list(experiment_data['turbidity_600'])
        all_ratio = list(experiment_data['ratio'])
        if not control_df.empty:
            all_turbidity.extend(list(control_df['turbidity_600']))
            all_ratio.extend(list(control_df['ratio']))
        
        turbidity_vmin, turbidity_vmax = min(all_turbidity), max(all_turbidity)
        ratio_vmin, ratio_vmax = min(all_ratio), max(all_ratio)
        
        # Set up plotting
        plt.style.use('default')
        
        # Format concentration labels for readability
        def format_conc_label(conc):
            if conc < 0.001:
                return f'{conc:.6f}'
            elif conc < 1:
                return f'{conc:.4f}'
            else:
                return f'{conc:.1f}'
        
        # Create combined plot if controls exist
        if not control_df.empty:
            # Create control grids (3 controls: water, surfactant A only, surfactant B only)
            control_labels = ['Water\\n(0,0)', f'{surfactant_a_name} Only\\n(50,0)', f'{surfactant_b_name} Only\\n(0,50)']
            
            turbidity_controls = np.array([[
                control_df[control_df['control_type'] == 'water_blank']['turbidity_600'].iloc[0] if 'water_blank' in control_df['control_type'].values else 0,
                control_df[control_df['control_type'] == 'surfactant_A_stock']['turbidity_600'].iloc[0] if 'surfactant_A_stock' in control_df['control_type'].values else 0,
                control_df[control_df['control_type'] == 'surfactant_B_stock']['turbidity_600'].iloc[0] if 'surfactant_B_stock' in control_df['control_type'].values else 0
            ]])
            
            ratio_controls = np.array([[
                control_df[control_df['control_type'] == 'water_blank']['ratio'].iloc[0] if 'water_blank' in control_df['control_type'].values else 0,
                control_df[control_df['control_type'] == 'surfactant_A_stock']['ratio'].iloc[0] if 'surfactant_A_stock' in control_df['control_type'].values else 0,
                control_df[control_df['control_type'] == 'surfactant_B_stock']['ratio'].iloc[0] if 'surfactant_B_stock' in control_df['control_type'].values else 0
            ]])
            
            # Combined plot with controls
            fig = plt.figure(figsize=(20, 14))
            gs = fig.add_gridspec(2, 2, height_ratios=[15, 1], hspace=0.15, wspace=0.25)
            
            # Main turbidity plot
            ax1_main = fig.add_subplot(gs[0, 0])
            sns.heatmap(turbidity_grid, ax=ax1_main, cmap='viridis', annot=True, fmt='.3f',
                       cbar_kws={'label': 'Turbidity (600 nm)'}, square=True,
                       xticklabels=[format_conc_label(x) for x in turbidity_grid.columns],
                       yticklabels=[format_conc_label(y) for y in turbidity_grid.index],
                       vmin=turbidity_vmin, vmax=turbidity_vmax)
            ax1_main.set_title(f'Turbidity vs Surfactant Concentrations\\n({surfactant_a_name} vs {surfactant_b_name})', fontsize=16, fontweight='bold')
            ax1_main.set_xlabel(f'{surfactant_a_name} Concentration (mM)', fontsize=12)
            ax1_main.set_ylabel(f'{surfactant_b_name} Concentration (mM)', fontsize=12)
            ax1_main.tick_params(axis='x', rotation=45)
            
            # Turbidity controls
            ax1_ctrl = fig.add_subplot(gs[1, 0])
            sns.heatmap(turbidity_controls, ax=ax1_ctrl, cmap='viridis', annot=True, fmt='.3f',
                       cbar=False, xticklabels=control_labels, yticklabels=['Controls'],
                       vmin=turbidity_vmin, vmax=turbidity_vmax)
            ax1_ctrl.set_title('Control Samples', fontsize=11)
            ax1_ctrl.tick_params(axis='x', rotation=0, labelsize=10)
            
            # Main ratio plot
            ax2_main = fig.add_subplot(gs[0, 1])
            sns.heatmap(ratio_grid, ax=ax2_main, cmap='plasma', annot=True, fmt='.3f',
                       cbar_kws={'label': 'Fluorescence Ratio (334/373 : 334/384)'}, square=True,
                       xticklabels=[format_conc_label(x) for x in ratio_grid.columns],
                       yticklabels=[format_conc_label(y) for y in ratio_grid.index],
                       vmin=ratio_vmin, vmax=ratio_vmax)
            ax2_main.set_title(f'Fluorescence Ratio vs Surfactant Concentrations\\n({surfactant_a_name} vs {surfactant_b_name})', fontsize=16, fontweight='bold')
            ax2_main.set_xlabel(f'{surfactant_a_name} Concentration (mM)', fontsize=12)
            ax2_main.set_ylabel(f'{surfactant_b_name} Concentration (mM)', fontsize=12)
            ax2_main.tick_params(axis='x', rotation=45)
            
            # Ratio controls
            ax2_ctrl = fig.add_subplot(gs[1, 1])
            sns.heatmap(ratio_controls, ax=ax2_ctrl, cmap='plasma', annot=True, fmt='.3f',
                       cbar=False, xticklabels=control_labels, yticklabels=['Controls'],
                       vmin=ratio_vmin, vmax=ratio_vmax)
            ax2_ctrl.set_title('Control Samples', fontsize=11)
            ax2_ctrl.tick_params(axis='x', rotation=0, labelsize=10)
            
            # Save combined plot
            combined_path = os.path.join(output_dir, 'surfactant_grid_heatmaps_with_controls.png')
            plt.savefig(combined_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"    Combined heatmap saved: {os.path.basename(combined_path)}")
        
        # Individual plots (always create these)
        # Turbidity only
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(turbidity_grid, ax=ax, cmap='viridis', annot=True, fmt='.3f',
                   cbar_kws={'label': 'Turbidity (600 nm)'}, square=True,
                   xticklabels=[format_conc_label(x) for x in turbidity_grid.columns],
                   yticklabels=[format_conc_label(y) for y in turbidity_grid.index])
        ax.set_title(f'Turbidity vs Surfactant Concentrations ({surfactant_a_name} vs {surfactant_b_name})', fontsize=18, fontweight='bold')
        ax.set_xlabel(f'{surfactant_a_name} Concentration (mM)', fontsize=14)
        ax.set_ylabel(f'{surfactant_b_name} Concentration (mM)', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        
        turb_path = os.path.join(output_dir, 'surfactant_grid_turbidity.png')
        plt.savefig(turb_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"    Turbidity heatmap saved: {os.path.basename(turb_path)}")
        
        # Ratio only
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(ratio_grid, ax=ax, cmap='plasma', annot=True, fmt='.3f',
                   cbar_kws={'label': 'Fluorescence Ratio (334/373 : 334/384)'}, square=True,
                   xticklabels=[format_conc_label(x) for x in ratio_grid.columns],
                   yticklabels=[format_conc_label(y) for y in ratio_grid.index])
        ax.set_title(f'Fluorescence Ratio vs Surfactant Concentrations ({surfactant_a_name} vs {surfactant_b_name})', fontsize=18, fontweight='bold')
        ax.set_xlabel(f'{surfactant_a_name} Concentration (mM)', fontsize=14)
        ax.set_ylabel(f'{surfactant_b_name} Concentration (mM)', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        
        ratio_path = os.path.join(output_dir, 'surfactant_grid_ratio.png')
        plt.savefig(ratio_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"    Ratio heatmap saved: {os.path.basename(ratio_path)}")
        
        # Summary
        logger.info(f"    Heatmap generation complete - {len(surf_A_concs)}x{len(surf_B_concs)} grid visualized")
        
    except ImportError as e:
        logger.warning(f"    Missing visualization libraries: {e}")
        logger.warning("    Install matplotlib and seaborn: pip install matplotlib seaborn")
    except Exception as e:
        logger.error(f"    Heatmap generation failed: {e}")
        raise

def calculate_adaptive_concentration_bounds(experiment_df, surfactant_a_name, surfactant_b_name, output_dir, logger):
    """
    Calculate new minimum concentration bounds using baseline rectangle method.
    
    Args:
        experiment_df: DataFrame with experimental results (surf_A_conc_mm, surf_B_conc_mm, turbidity_600, ratio columns)
        surfactant_a_name: Name of surfactant A
        surfactant_b_name: Name of surfactant B  
        output_dir: Directory to save threshold analysis
        logger: Logger instance
        
    Returns:
        dict: {surfactant_a_name: min_conc, surfactant_b_name: min_conc}
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        
        logger.info("    Calculating adaptive concentration bounds using baseline rectangle method...")
        
        # Step 1: Classify baseline vs non-baseline wells
        df = experiment_df.copy()
        
        # Classification criteria (from baseline_classification.py)
        turbidity_threshold = 0.15  # Above this = likely interaction
        ratio_baseline = df['ratio'].median()  # Use median as baseline reference
        ratio_std = df['ratio'].std()
        ratio_threshold = 2.0 * ratio_std  # 2 standard deviations from baseline
        
        logger.info(f"      Classification thresholds: turbidity > {turbidity_threshold:.3f}, ratio deviation > {ratio_threshold:.3f}")
        
        # Classify each well
        df['is_baseline'] = (
            (df['turbidity_600'] <= turbidity_threshold) & 
            (np.abs(df['ratio'] - ratio_baseline) <= ratio_threshold)
        )
        
        baseline_count = df['is_baseline'].sum()
        non_baseline_count = len(df) - baseline_count
        logger.info(f"      Classification results: {baseline_count} baseline, {non_baseline_count} non-baseline wells")
        
        # Step 2: Find largest baseline rectangle
        def largest_rectangle_in_histogram(heights):
            """Find the area of the largest rectangle in a histogram."""
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
        
        # Get sorted unique concentrations
        surf_A_concs = sorted(df['surf_A_conc_mm'].unique())
        surf_B_concs = sorted(df['surf_B_conc_mm'].unique())
        
        logger.info(f"      Grid dimensions: {len(surf_A_concs)} x {len(surf_B_concs)}")
        
        # Create binary matrix (1 = baseline, 0 = non-baseline)
        matrix = np.zeros((len(surf_B_concs), len(surf_A_concs)), dtype=int)
        
        for i, surf_B_conc in enumerate(surf_B_concs):
            for j, surf_A_conc in enumerate(surf_A_concs):
                well = df[
                    (df['surf_A_conc_mm'] == surf_A_conc) & 
                    (df['surf_B_conc_mm'] == surf_B_conc)
                ]
                if len(well) > 0:
                    matrix[i, j] = 1 if well.iloc[0]['is_baseline'] else 0
        
        # Find largest rectangle using histogram method
        max_area = 0
        best_rectangle = None
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
                    'surf_A_min': surf_A_concs[left_col],
                    'surf_A_max': surf_A_concs[right_col],
                    'surf_B_min': surf_B_concs[bottom_row],
                    'surf_B_max': surf_B_concs[top_row]
                }
        
        if best_rectangle is None:
            logger.warning("      No baseline rectangle found - using original bounds")
            return {
                'surf_a_min': min(surf_A_concs), 
                'surf_a_max': max(surf_A_concs),
                'surf_b_min': min(surf_B_concs),
                'surf_b_max': max(surf_B_concs)
            }
        
        logger.info(f"      Largest baseline rectangle: {best_rectangle['area']} wells")
        logger.info(f"      {surfactant_a_name} range: {best_rectangle['surf_A_min']:.6f} - {best_rectangle['surf_A_max']:.6f} mM")
        logger.info(f"      {surfactant_b_name} range: {best_rectangle['surf_B_min']:.6f} - {best_rectangle['surf_B_max']:.6f} mM")
        
        # Step 3: Create and save visualization
        threshold_dir = Path(output_dir) / "concentration_thresholds"
        threshold_dir.mkdir(exist_ok=True)
        
        # Create classification grid for visualization
        classification_grid = df.pivot_table(
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
        
        # Draw baseline rectangle
        left_col = best_rectangle['left_col']
        right_col = best_rectangle['right_col'] + 1
        bottom_row_display = len(surf_B_concs) - 1 - best_rectangle['top_row']
        top_row_display = len(surf_B_concs) - best_rectangle['bottom_row']
        
        from matplotlib.patches import Rectangle
        rect_patch = Rectangle((left_col, bottom_row_display), 
                              right_col - left_col, 
                              top_row_display - bottom_row_display,
                              linewidth=3, edgecolor='green', facecolor='none', 
                              linestyle='-', label='Largest baseline rectangle')
        ax1.add_patch(rect_patch)
        
        # Draw new threshold lines
        new_surf_A_min = best_rectangle['surf_A_max']
        new_surf_B_min = best_rectangle['surf_B_max']
        
        try:
            surf_A_idx = surf_A_concs.index(new_surf_A_min) + 0.5
            surf_B_idx = sorted(surf_B_concs, reverse=True).index(new_surf_B_min) + 0.5
        except ValueError:
            surf_A_idx = min(range(len(surf_A_concs)), key=lambda x: abs(surf_A_concs[x] - new_surf_A_min)) + 0.5
            surf_B_idx = min(range(len(surf_B_concs)), key=lambda x: abs(sorted(surf_B_concs, reverse=True)[x] - new_surf_B_min)) + 0.5
        
        ax1.axvline(x=surf_A_idx, color='red', linewidth=2, linestyle='--', label=f'New {surfactant_a_name} min: {new_surf_A_min:.4f}')
        ax1.axhline(y=surf_B_idx, color='red', linewidth=2, linestyle='--', label=f'New {surfactant_b_name} min: {new_surf_B_min:.4f}')
        
        ax1.set_xlabel(f'{surfactant_a_name} Concentration (mM)')
        ax1.set_ylabel(f'{surfactant_b_name} Concentration (mM)')
        ax1.set_title(f'Baseline Classification with Largest Baseline Rectangle\\n(Green box = all baseline, Red lines = new thresholds)', fontweight='bold', fontsize=14)
        ax1.legend()
        
        # Plot 2: Excluded vs Included wells
        excluded_wells = df[
            (df['surf_A_conc_mm'] <= new_surf_A_min) & 
            (df['surf_B_conc_mm'] <= new_surf_B_min)
        ]
        
        included_wells = df[
            (df['surf_A_conc_mm'] > new_surf_A_min) | 
            (df['surf_B_conc_mm'] > new_surf_B_min)
        ]
        
        ax2.scatter(excluded_wells['surf_A_conc_mm'], excluded_wells['surf_B_conc_mm'], 
                   c='lightgray', s=100, alpha=0.7, label=f'Excluded (n={len(excluded_wells)})')
        
        included_baseline = included_wells[included_wells['is_baseline'] == True]
        included_non_baseline = included_wells[included_wells['is_baseline'] == False]
        
        ax2.scatter(included_baseline['surf_A_conc_mm'], included_baseline['surf_B_conc_mm'],
                   c='blue', s=100, alpha=0.7, label=f'Included Baseline (n={len(included_baseline)})')
        ax2.scatter(included_non_baseline['surf_A_conc_mm'], included_non_baseline['surf_B_conc_mm'],
                   c='red', s=100, alpha=0.7, label=f'Included Non-baseline (n={len(included_non_baseline)})')
        
        ax2.axvline(x=new_surf_A_min, color='green', linewidth=2, linestyle='--')
        ax2.axhline(y=new_surf_B_min, color='green', linewidth=2, linestyle='--')
        
        ax2.set_xlabel(f'{surfactant_a_name} Concentration (mM)')
        ax2.set_ylabel(f'{surfactant_b_name} Concentration (mM)')
        ax2.set_title(f'Wells: Excluded vs Included\\n(Baseline Rectangle Strategy)', fontweight='bold', fontsize=14)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout(pad=2.0)
        
        # Save visualization
        vis_path = threshold_dir / 'adaptive_concentration_bounds.png'
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"      Threshold visualization saved: {vis_path.name}")
        
        # Return new bounds
        new_bounds = {
            'surf_a_min': new_surf_A_min,
            'surf_a_max': max(surf_A_concs),  # Keep original max
            'surf_b_min': new_surf_B_min,
            'surf_b_max': max(surf_B_concs)   # Keep original max
        }
        
        logger.info(f"      New concentration bounds: {new_bounds}")
        return new_bounds
        
    except Exception as e:
        logger.error(f"    Error calculating adaptive bounds: {str(e)}")
        return {
            'surf_a_min': 0.001, 
            'surf_a_max': 1.0,
            'surf_b_min': 0.001,
            'surf_b_max': 1.0
        }  # Fallback bounds

# ================================================================================
# DATA BACKUP AND RECOVERY FUNCTIONS
# ================================================================================
    """
    Immediately backup raw measurement data to prevent data loss if processing crashes.
    
    Args:
        lash_e: Lash_E coordinator for logging
        measurement_entry: Dictionary containing measurement data
        plate_number: Current plate number
        wells_measured: List of wells that were measured
        experiment_name: Name of current experiment for folder organization
    """
    try:
        # Create backup directory within experiment folder
        if experiment_name:
            backup_dir = os.path.join("output", experiment_name, "measurement_backups")
        else:
            backup_dir = os.path.join("output", "measurement_backups")  # Fallback
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create unique backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        backup_filename = f"raw_measurement_plate{plate_number}_wells{wells_measured[0]}-{wells_measured[-1]}_{timestamp}.json"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Skip file saving during simulation
        if hasattr(lash_e, 'simulate') and lash_e.simulate:
            lash_e.logger.info(f"    [SIMULATED] Would save backup: {backup_filename}")
            return
        
        # Save raw measurement data
        backup_data = {
            'measurement_entry': measurement_entry,
            'plate_number': plate_number,
            'wells_measured': wells_measured,
            'backup_timestamp': timestamp,
            'workflow': 'surfactant_grid_adaptive_concentrations'
        }
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)  # default=str handles datetime objects
        
        lash_e.logger.info(f"OK Backed up measurement data to: {backup_path}")
        
    except Exception as e:
        lash_e.logger.info(f"WARNING: Failed to backup measurement data: {e}")
        # Don't crash the workflow if backup fails

def condition_tip(lash_e, vial_name, conditioning_volume_ul=100, liquid_type='water'):
    """Condition a pipette tip by aspirating and dispensing into source vial multiple times
    
    Args:
        lash_e: Lash_E robot controller
        vial_name: Name of vial to condition tip with
        conditioning_volume_ul: Total volume for conditioning (default 100 uL)
        liquid_type: Type of liquid for pipetting parameters ('water', 'DMSO', etc.)
    """
    try:
        # Calculate volume per conditioning cycle (5 cycles total)
        cycles = 5
        volume_per_cycle_ul = conditioning_volume_ul
        volume_per_cycle_ml = volume_per_cycle_ul / 1000
        
        lash_e.logger.info(f"    Conditioning tip with {vial_name}: {cycles} cycles of {volume_per_cycle_ul:.1f}uL")
        
        for cycle in range(cycles):
            # Aspirate from vial 
            lash_e.nr_robot.aspirate_from_vial(vial_name, volume_per_cycle_ml, liquid=liquid_type)
            # Dispense back into same vial
            lash_e.nr_robot.dispense_into_vial(vial_name, volume_per_cycle_ml, liquid=liquid_type)
        
        lash_e.logger.info(f"    Tip conditioning complete for {vial_name}")
        
    except Exception as e:
        lash_e.logger.info(f"    Warning: Could not condition tip with {vial_name}: {e}")

def get_pipette_usage_breakdown(lash_e):
    """Get detailed pipette usage breakdown by tip type from North_Robot.PIPETS_USED property."""
    try:
        # Access the actual PIPETS_USED property from North_Robot
        pipette_usage = lash_e.nr_robot.PIPETS_USED
        
        large_tips = (pipette_usage.get('large_tip_rack_1', 0) + 
                     pipette_usage.get('large_tip_rack_2', 0))
        small_tips = (pipette_usage.get('small_tip_rack_1', 0) + 
                     pipette_usage.get('small_tip_rack_2', 0))
        
        return {
            'large_tips': large_tips,
            'small_tips': small_tips,
            'total': large_tips + small_tips
        }
    except Exception as e:
        lash_e.logger.info(f"  Warning: Could not read pipette status: {e}")
        # Fallback to manual count if something fails
        manual_count = getattr(lash_e, 'pipette_count', 0)
        return {
            'large_tips': manual_count,
            'small_tips': 0,
            'total': manual_count
        }

def fill_water_vial(lash_e, vial_name):
    """
    Fill a water vial to maximum capacity (8mL) by moving it to reservoir,
    calculating needed volume, dispensing from reservoir, and returning home.
    
    Args:
        lash_e: The Lash_E coordinator instance
        vial_name (str): Name of water vial to fill ('water' or 'water_2')
    """
    # Get current volume and vial capacity
    current_volume_ml = lash_e.nr_robot.get_vial_info(vial_name, 'vial_volume')
    max_volume_ml = 8
    
    # Calculate volume needed to fill to max capacity
    fill_volume_ml = max_volume_ml - current_volume_ml
    
    if fill_volume_ml <= 0.1:  # Already nearly full (within 100uL)
        lash_e.logger.info(f"    Water vial '{vial_name}' already full ({current_volume_ml:.2f}mL), skipping fill")
        return
        
    lash_e.logger.info(f"    Filling water vial '{vial_name}': {current_volume_ml:.2f}mL -> {max_volume_ml:.2f}mL (adding {fill_volume_ml:.2f}mL)")
    
    # Fill vial from reservoir
    lash_e.nr_robot.dispense_into_vial_from_reservoir(1, vial_name, fill_volume_ml)
    
    
    lash_e.logger.info(f"    Water vial '{vial_name}' filled successfully to {max_volume_ml:.2f}mL")

def refill_surfactant_vial(lash_e, vial_name, liquid='SDS'):
    """
    Refill a surfactant vial to maximum capacity (8mL) by moving it to reservoir,
    calculating needed volume, dispensing from reservoir, and returning home.
    
    Args:
        lash_e: The Lash_E coordinator instance
        vial_name (str): Name of surfactant vial to fill (e.g., 'surfactant_A_stock')
    """
    # Get current volume and vial capacity
    current_volume_ml = lash_e.nr_robot.get_vial_info(vial_name, 'vial_volume')
    max_volume_ml = 8
    
    # Calculate volume needed to fill to max capacity
    fill_volume_ml = max_volume_ml - current_volume_ml
    
    if fill_volume_ml <= 2.0:  # Already nearly full (within 2mL)
        lash_e.logger.info(f"    Surfactant vial '{vial_name}' already full ({current_volume_ml:.2f}mL), skipping fill")
        return
        
    lash_e.logger.info(f"    Refilling surfactant vial '{vial_name}': {current_volume_ml:.2f}mL -> {max_volume_ml:.2f}mL (adding {fill_volume_ml:.2f}mL)")
    
    # Fill vial from reservoir
    # Parse surfactant name (everything before first underscore) and add _refill
    surfactant_base_name = vial_name.split('_')[0]  # e.g., "SDS" from "SDS_stock"
    source_refill_vial = f"{surfactant_base_name}_refill"
    lash_e.nr_robot.dispense_from_vial_into_vial(source_vial_name=source_refill_vial,dest_vial_name=vial_name, volume=fill_volume_ml, liquid=liquid)
    
    lash_e.logger.info(f"    Surfactant vial '{vial_name}' refilled successfully to {max_volume_ml:.2f}mL")

# ================================================================================
# SECTION 1: SMART SUBSTOCK MANAGEMENT
# ================================================================================

class SurfactantSubstockTracker:
    """Track surfactant substock vial contents and find optimal dilution strategies."""
    
    def __init__(self):
        self.substocks = {}  # vial_name: {'surfactant_name': str, 'concentration_mm': float, 'volume_ml': float}
        self.next_available_substock = 0
        self.min_pipette_volume_ul = MIN_WELL_PIPETTE_VOLUME_UL  # Consistent with other functions
        self.max_surfactant_volume_ul = MAX_SURFACTANT_VOLUME_UL  # Consistent with other functions
    
    def find_best_solution_for_concentration(self, surfactant_name, target_conc_mm, lash_e=None):
        """
        Find best available solution (stock or substock) for achieving target concentration.
        Returns solution dict or None if no suitable solution exists within pipetting limits.
        """
        options = []
        max_volume_ml = self.max_surfactant_volume_ul / 1000
        
        # Check stock solution first
        if surfactant_name in SURFACTANT_LIBRARY:
            stock_conc = SURFACTANT_LIBRARY[surfactant_name]["stock_conc"]
            # Calculate volume needed for target final concentration in 200 uL well
            # Final_conc = (stock_conc * vol_needed) / 200 uL
            # Therefore: vol_needed = (target_conc * 200) / stock_conc
            vol_needed_ul = (target_conc_mm * WELL_VOLUME_UL) / stock_conc
            vol_needed_ml = vol_needed_ul / 1000
            
            if vol_needed_ul >= self.min_pipette_volume_ul and vol_needed_ul <= self.max_surfactant_volume_ul:
                options.append({
                    'vial_name': f"{surfactant_name}_stock",
                    'concentration_mm': stock_conc,
                    'volume_needed_ml': vol_needed_ml,
                    'volume_needed_ul': vol_needed_ul,
                    'is_stock': True
                })
        
        # Check existing substocks
        for vial_name, contents in self.substocks.items():
            if (contents['surfactant_name'] == surfactant_name and 
                contents['volume_ml'] > 0):
                
                # Calculate volume needed for target final concentration in 200 uL well
                vol_needed_ul = (target_conc_mm * WELL_VOLUME_UL) / contents['concentration_mm']
                vol_needed_ml = vol_needed_ul / 1000
                
                if (vol_needed_ul >= self.min_pipette_volume_ul and 
                    vol_needed_ul <= self.max_surfactant_volume_ul and
                    contents['volume_ml'] >= vol_needed_ml):
                    
                    options.append({
                        'vial_name': vial_name,
                        'concentration_mm': contents['concentration_mm'],
                        'volume_needed_ml': vol_needed_ml,
                        'volume_needed_ul': vol_needed_ul,
                        'is_stock': False
                    })
        
        if not options:
            return None
        
        # Use conservation-aware ranking to select best option
        return self.rank_options_with_conservation(options, lash_e)
    
    def rank_options_with_conservation(self, options, lash_e=None, conservation_threshold=4.0):
        """Rank options balancing pipetting volume preference with vial conservation."""
        scored_options = []
        
        for option in options:
            vial_name = option['vial_name']
            volume_needed_ml = option['volume_needed_ml']
            pipette_volume_ul = option['volume_needed_ul']
            
            # Get current vial volume (with fallback for simulation/errors)
            try:
                if lash_e is not None:
                    current_vial_volume = lash_e.nr_robot.get_vial_info(vial_name, 'vial_volume')
                else:
                    current_vial_volume = 6.0  # Conservative default
            except:
                current_vial_volume = 6.0  # Default for simulation/errors
            
            # Conservation penalty: penalize high usage from low vials
            if current_vial_volume < conservation_threshold:
                conservation_penalty = (volume_needed_ml / current_vial_volume) * 50  # Scale factor
            else:
                conservation_penalty = 0  # No penalty for vials with plenty left
            
            # Pipetting preference: higher volumes generally preferred  
            pipetting_score = pipette_volume_ul
            
            # Combined score: preference minus penalty
            total_score = pipetting_score - conservation_penalty
            
            scored_options.append((option, total_score))
        
        # Return option with highest score
        return max(scored_options, key=lambda x: x[1])[0]
    
    def calculate_optimal_substock_concentration(self, surfactant_name, target_conc_mm):
        """
        Calculate optimal substock concentration for a target.
        Aims for 15-20 uL pipetting volumes for reusability.
        """
        max_volume_ml = self.max_surfactant_volume_ul / 1000
        
        # Target 25 uL volumes for efficient, reusable solutions (configurable)
        target_volume_ul = 25  # Increased from 15 to push volumes higher
        # Calculate concentration needed to achieve target final concentration with target volume
        # target_conc_mm = (optimal_conc * target_volume_ul) / 200
        # Therefore: optimal_conc = (target_conc_mm * 200) / target_volume_ul
        optimal_conc = (target_conc_mm * WELL_VOLUME_UL) / target_volume_ul
        
        # Ensure we don't go below minimum pipette volume
        min_volume_ul = self.min_pipette_volume_ul
        max_conc_absolute = (target_conc_mm * WELL_VOLUME_UL) / min_volume_ul
        
        # CRITICAL: Substock cannot be more concentrated than the stock!
        stock_conc = SURFACTANT_LIBRARY[surfactant_name]['stock_conc']
        max_substock_conc = stock_conc * (EFFECTIVE_SURFACTANT_VOLUME / WELL_VOLUME_UL)  # 90/200 = 0.45
        
        # Use the most restrictive constraint
        final_conc = min(optimal_conc, max_conc_absolute, max_substock_conc)
        
        # Round to nice numbers - more flexible options
        def round_to_nice_concentration(value):
            if value <= 0:
                return 0
            log_val = np.log10(value)
            magnitude = 10 ** np.floor(log_val)
            normalized = value / magnitude
            
            # More granular rounding options to fill concentration gaps
            if normalized <= 1.2:
                nice_normalized = 1.0
            elif normalized <= 1.8:
                nice_normalized = 1.5
            elif normalized <= 2.5:
                nice_normalized = 2.0
            elif normalized <= 3.5:
                nice_normalized = 3.0
            elif normalized <= 4.5:
                nice_normalized = 4.0
            elif normalized <= 6.0:
                nice_normalized = 5.0
            elif normalized <= 8.0:
                nice_normalized = 7.0
            else:
                nice_normalized = 10.0
                
            return nice_normalized * magnitude
        
        # Use 80% for safety margin
        safe_conc = final_conc * 0.8
        return round_to_nice_concentration(safe_conc)
    
    def add_substock(self, surfactant_name, concentration_mm, volume_ml=6.0):
        """Add a new substock to tracking."""
        vial_name = f"{surfactant_name}_dilution_{self.next_available_substock}"
        self.substocks[vial_name] = {
            'surfactant_name': surfactant_name,
            'concentration_mm': concentration_mm,
            'volume_ml': volume_ml
        }
        self.next_available_substock += 1
        return vial_name

def calculate_systematic_dilution_series(surfactant_name, target_concentrations_mm, num_substocks=6, target_volume_ul=25):
    """
    Calculate a systematic dilution series with even decreases to cover the concentration range.
    Can be optimized for target pipetting volumes.
    """
    import numpy as np
    
    # Get stock concentration
    stock_conc = SURFACTANT_LIBRARY[surfactant_name]['stock_conc']
    
    # Find the range we need to cover
    min_target = min(target_concentrations_mm)
    max_target = max(target_concentrations_mm)
    
    series_max = stock_conc * 0.20
   
    series_min = min_target * 10
    
    # Create evenly spaced points on log scale
    log_points = np.linspace(np.log10(series_max), np.log10(series_min), num_substocks)
    concentrations = 10 ** log_points
    
    # Round each to nice numbers
    def round_to_nice_concentration(value):
        if value <= 0:
            return 0
        log_val = np.log10(value)
        magnitude = 10 ** np.floor(log_val)
        normalized = value / magnitude
        
        # More granular rounding options
        if normalized <= 1.2:
            nice_normalized = 1.0
        elif normalized <= 1.8:
            nice_normalized = 1.5
        elif normalized <= 2.5:
            nice_normalized = 2.0
        elif normalized <= 3.5:
            nice_normalized = 3.0
        elif normalized <= 4.5:
            nice_normalized = 4.0
        elif normalized <= 6.0:
            nice_normalized = 5.0
        elif normalized <= 8.0:
            nice_normalized = 7.0
        else:
            nice_normalized = 10.0
            
        return nice_normalized * magnitude
    
    # Round and remove duplicates
    rounded_concentrations = []
    for conc in concentrations:
        rounded = round_to_nice_concentration(conc)
        if rounded not in rounded_concentrations and rounded < stock_conc:
            rounded_concentrations.append(rounded)
    
    return sorted(rounded_concentrations, reverse=True)

def calculate_smart_dilution_plan(lash_e, surfactant_name, target_concentrations_mm):
    """
    Calculate optimal dilution strategy for a surfactant across all target concentrations.
    Uses systematic dilution series for better coverage.
    """
    tracker = SurfactantSubstockTracker()
    plan = {'substocks_needed': [], 'concentration_map': {}}
    
    lash_e.logger.info(f"\n=== Analyzing dilution strategy for {surfactant_name} ===")
    lash_e.logger.info(f"Target concentrations: {[f'{c:.2e}' for c in target_concentrations_mm]} mM")
    
    # Pre-calculate systematic dilution series optimized for 30-40 ╬╝L volumes
    systematic_series = calculate_systematic_dilution_series(surfactant_name, target_concentrations_mm, target_volume_ul=35)
    lash_e.logger.info(f"Systematic dilution series: {[f'{c:.2g}' for c in systematic_series]} mM")
    
    # Add systematic substocks to tracker
    for conc in systematic_series:
        vial_name = tracker.add_substock(surfactant_name, conc)
        plan['substocks_needed'].append({
            'vial_name': vial_name,
            'concentration_mm': conc,
            'needed_for': []  # Will be filled in below
        })
    
    # Now map each target to best available solution
    for target_conc in target_concentrations_mm:
        solution = tracker.find_best_solution_for_concentration(surfactant_name, target_conc, lash_e)
        
        if solution:
            plan['concentration_map'][target_conc] = solution
            lash_e.logger.info(f"  {target_conc:.2e} mM: Use {solution['vial_name']} ({solution['volume_needed_ul']:.1f} uL)")
            
            # Add to needed_for list for the substock
            for substock in plan['substocks_needed']:
                if substock['vial_name'] == solution['vial_name']:
                    substock['needed_for'].append(target_conc)
                    break
        else:
            lash_e.logger.warning(f"  {target_conc:.2e} mM: *** CANNOT ACHIEVE with systematic series ***")
    
    return plan, tracker

def calculate_dilution_recipes(lash_e, plan_a, plan_b, surfactant_a_name, surfactant_b_name):
    """
    Calculate the exact dilution recipes for each substock that needs to be created.
    Uses serial dilutions when direct dilution would require volumes < 200 uL.
    """
    MIN_SUBSTOCK_VOLUME_UL = 200  # Minimum volume for accurate substock preparation
    FINAL_SUBSTOCK_VOLUME_ML = 6.0
    
    recipes = []
    
    lash_e.logger.info(f"\n=== SUBSTOCK DILUTION RECIPES ===")
    lash_e.logger.info("For each substock showing: source + volume -> target_concentration")
    lash_e.logger.info(f"(Using minimum {MIN_SUBSTOCK_VOLUME_UL}uL pipetting volumes for accuracy)")
    
    # Process substocks for both surfactants
    for plan, surfactant_name in [(plan_a, surfactant_a_name), (plan_b, surfactant_b_name)]:
        if plan['substocks_needed']:
            lash_e.logger.info(f"\n{surfactant_name} substocks:")
            
            # Get stock concentration
            stock_conc = SURFACTANT_LIBRARY[surfactant_name]["stock_conc"]
            
            # Sort substocks by concentration (highest first for cascade approach)
            substocks = sorted(plan['substocks_needed'], 
                             key=lambda x: x['concentration_mm'], reverse=True)
            
            created_substocks = {}  # Track what we've created: {conc: vial_name}
            created_substocks[stock_conc] = f"{surfactant_name}_stock"  # Stock is available
            
            for substock in substocks:
                target_conc = substock['concentration_mm']
                vial_name = substock['vial_name']
                
                # Find best source (stock or existing substock)
                best_source = None
                min_volume_needed = float('inf')
                
                for source_conc, source_name in created_substocks.items():
                    if source_conc > target_conc:  # Can only dilute down
                        # Simple dilution: no correction factor for substock creation
                        dilution_factor = source_conc / target_conc
                        source_volume_ml = FINAL_SUBSTOCK_VOLUME_ML / dilution_factor
                        source_volume_ul = source_volume_ml * 1000
                        
                        if source_volume_ul >= MIN_SUBSTOCK_VOLUME_UL:
                            if source_volume_ul < min_volume_needed:
                                min_volume_needed = source_volume_ul
                                best_source = {
                                    'name': source_name,
                                    'conc': source_conc,
                                    'volume_ul': source_volume_ul,
                                    'volume_ml': source_volume_ml
                                }
                
                if best_source:
                    # Simple dilution: no correction factor for substock creation
                    dilution_factor = best_source['conc'] / target_conc
                    
                    source_volume_ml = FINAL_SUBSTOCK_VOLUME_ML / dilution_factor
                    source_volume_ul = source_volume_ml * 1000
                    water_volume_ml = FINAL_SUBSTOCK_VOLUME_ML - source_volume_ml
                    water_volume_ul = water_volume_ml * 1000
                    
                    recipes.append({
                        'Vial_Name': vial_name,
                        'Surfactant': surfactant_name,
                        'Target_Conc_mM': target_conc,
                        'Source_Vial': best_source['name'],
                        'Source_Conc_mM': best_source['conc'],
                        'Source_Volume_mL': source_volume_ml,
                        'Source_Volume_uL': source_volume_ul,
                        'Water_Volume_mL': water_volume_ml,
                        'Water_Volume_uL': water_volume_ul,
                        'Final_Volume_mL': FINAL_SUBSTOCK_VOLUME_ML,
                        'Dilution_Factor': dilution_factor
                    })
                    
                    # Add to available sources
                    created_substocks[target_conc] = vial_name
                    
                    source_type = "stock" if best_source['name'].endswith('_stock') else "dilution"
                    lash_e.logger.info(f"  {vial_name}: {source_volume_ul:.0f}uL {best_source['name']} + {water_volume_ul:.0f}uL water -> {target_conc:.2e} mM")
                    lash_e.logger.info(f"    (Dilution factor: {dilution_factor:.1f}x from {best_source['conc']:.2e} mM {source_type})")
                
                else:
                    # Cannot make with current minimum volume - needs intermediate
                    lash_e.logger.info(f"  {vial_name}: *** NEEDS INTERMEDIATE DILUTION - too dilute for {MIN_SUBSTOCK_VOLUME_UL}uL minimum ***")
                    
                    # Suggest intermediate concentration
                    max_possible_dilution = FINAL_SUBSTOCK_VOLUME_ML / (MIN_SUBSTOCK_VOLUME_UL / 1000)
                    intermediate_conc = stock_conc / max_possible_dilution
                    lash_e.logger.info(f"    Suggestion: Create intermediate at ~{intermediate_conc:.2e} mM first")
    
    if not recipes:
        lash_e.logger.info("No substocks needed - using only stock solutions!")
    
    return recipes

def create_substocks_from_recipes(lash_e, recipes):
    """
    Create physical substocks according to the calculated recipes.
    Checks for existing substocks (volume > 0) before creating new ones.
    """
    logger = lash_e.logger
    logger.info(f"Checking for existing substocks and creating {len(recipes)} new ones as needed")
    
    # First, check for existing substocks
    existing_substocks = {}
    try:
        vial_file_path = "status/surfactant_grid_vials_expanded.csv"
        if os.path.exists(vial_file_path):
            import pandas as pd
            df = pd.read_csv(vial_file_path)
            
            for _, row in df.iterrows():
                vial_name = row['vial_name']
                volume = float(row['vial_volume']) if pd.notna(row['vial_volume']) else 0.0
                
                if volume > 0:
                    existing_substocks[vial_name] = volume
                    logger.info(f"Found existing substock: {vial_name} ({volume:.1f} mL)")
    except Exception as e:
        logger.warning(f"Could not check existing substocks: {e}")
    
    created_substocks = []
    skipped_count = 0
    
    for recipe in recipes:
        vial_name = recipe['Vial_Name']
        surfactant = recipe['Surfactant']
        target_conc = recipe['Target_Conc_mM']
        source_vial = recipe['Source_Vial']
        source_volume_ml = recipe['Source_Volume_mL'] 
        water_volume_ml = recipe['Water_Volume_mL']
        
        # Check if this substock already exists with sufficient volume
        if vial_name in existing_substocks and existing_substocks[vial_name] > 0:
            logger.info(f"Skipping {vial_name}: already exists with {existing_substocks[vial_name]:.1f} mL")
            created_substocks.append({
                'vial_name': vial_name,
                'concentration_mm': target_conc,
                'volume_ml': existing_substocks[vial_name],
                'created': False,  # Not newly created
                'existed': True
            })
            skipped_count += 1
            continue
        
        logger.info(f"Creating {vial_name}: {target_conc:.2e} mM")
        logger.info(f"  Recipe: {source_volume_ml*1000:.0f}uL {source_vial} + {water_volume_ml*1000:.0f}uL water")
        
        # Always call robot functions - Lash_E handles simulation internally
        try:
            # Add source solution
            lash_e.nr_robot.dispense_from_vial_into_vial(
                source_vial_name=source_vial, 
                dest_vial_name=vial_name, 
                volume=source_volume_ml,
                liquid='water'
            )

            # Add water first if needed
            if water_volume_ml > 0:
                lash_e.nr_robot.dispense_into_vial_from_reservoir(
                    reservoir_index=1, vial_index=vial_name, 
                    volume=water_volume_ml, return_home=False
                )
            

            
            # Vortex to mix
            lash_e.nr_robot.vortex_vial(vial_name=vial_name, vortex_time=8, vortex_speed=80)
            lash_e.nr_robot.return_vial_home(vial_name=vial_name)
            

            created_substocks.append({
                'vial_name': vial_name,
                'concentration_mm': target_conc,
                'volume_ml': recipe['Final_Volume_mL'],
                'created': True,
                'existed': False
            })
            
            logger.info(f"  + Successfully created {vial_name}")
            
        except Exception as e:
            logger.error(f"  - Failed to create {vial_name}: {str(e)}")
            created_substocks.append({
                'vial_name': vial_name,
                'concentration_mm': target_conc,
                'volume_ml': 0,
                'created': False,
                'existed': False,
                'error': str(e)
            })
    
    newly_created = len([s for s in created_substocks if s['created'] and not s.get('existed', False)])
    total_available = len([s for s in created_substocks if (s['created'] or s.get('existed', False))])
    
    logger.info(f"Substock summary: {newly_created} newly created, {skipped_count} already existed, {total_available} total available")
    return created_substocks


# ================================================================================
# SECTION 2: CONCENTRATION AND GRID CALCULATION FUNCTIONS
# ================================================================================

def calculate_grid_concentrations(lash_e, surfactant_name=None, min_conc=None, max_conc=None, number_concentrations=9):
    """
    Calculate adaptive concentration grid points for surfactants.
    Each surfactant gets its own optimized range: min_conc to max_conc.
    
    Args:
        lash_e: Lash_E coordinator for logging
        surfactant_name: Name of surfactant (if None, uses generic range)
        min_conc: Custom minimum concentration (if None, uses MIN_CONC)
        max_conc: Custom maximum concentration (if None, calculates from stock)
        
    Returns:
        numpy.array: Concentration values in mM
    """
    # Use custom min_conc or default
    effective_min_conc = min_conc if min_conc is not None else MIN_CONC
    
    # Calculate max_conc if not provided
    if max_conc is None:
        if surfactant_name and surfactant_name in SURFACTANT_LIBRARY:
            stock_conc = SURFACTANT_LIBRARY[surfactant_name]["stock_conc"]
            # Maximum concentration is limited by volume allocation in the well
            # Max achievable: stock_conc * (allocated_volume / total_well_volume)
            effective_max_conc = stock_conc * (EFFECTIVE_SURFACTANT_VOLUME / WELL_VOLUME_UL)
        else:
            # Generic range if no surfactant specified
            effective_max_conc = 25  # Generic max for calculations
    else:
        effective_max_conc = max_conc
    
    # Create logarithmic spacing from effective_min_conc to effective_max_conc
    # np.logspace(start, stop, num) where start and stop are log10 values
    log_min = np.log10(effective_min_conc)
    log_max = np.log10(effective_max_conc)
    
    concentrations = np.logspace(log_min, log_max, number_concentrations)
    
    lash_e.logger.info(f"Adaptive grid for {surfactant_name or 'generic'}: {effective_min_conc:.1e} to {effective_max_conc:.1f} mM ({number_concentrations} steps)")
    lash_e.logger.info(f"  Concentrations: {[f'{c:.3e}' for c in concentrations]}")
    
    return concentrations

def calculate_dual_surfactant_grids(lash_e, surfactant_a_name, surfactant_b_name, number_concentrations=9):
    """
    Calculate optimized concentration grids for both surfactants.
    
    Args:
        lash_e: Lash_E coordinator for logging
        surfactant_a_name: Name of first surfactant
        surfactant_b_name: Name of second surfactant
        
    Returns:
        tuple: (concentrations_a, concentrations_b)
    """
    concs_a = calculate_grid_concentrations(lash_e, surfactant_a_name, number_concentrations=number_concentrations)
    concs_b = calculate_grid_concentrations(lash_e, surfactant_b_name, number_concentrations=number_concentrations)
    
    lash_e.logger.info(f"\nAdaptive concentration grid summary:")
    lash_e.logger.info(f"  {surfactant_a_name}: {len(concs_a)} concentrations from {concs_a[0]:.3e} to {concs_a[-1]:.3e} mM")
    lash_e.logger.info(f"  {surfactant_b_name}: {len(concs_b)} concentrations from {concs_b[0]:.3e} to {concs_b[-1]:.3e} mM")
    lash_e.logger.info(f"  Total combinations: {len(concs_a)} x {len(concs_b)} = {len(concs_a) * len(concs_b)} wells (x {N_REPLICATES} replicates = {len(concs_a) * len(concs_b) * N_REPLICATES} total wells)")
    
    return concs_a, concs_b

def rank_options_with_conservation_external(options, lash_e, conservation_threshold=4.0):
    """External ranking function for create_plan_from_existing_stocks with different data structure."""
    scored_options = []
    
    for option in options:
        vial_name = option['stock']['vial_name']
        volume_needed_ml = option['volume_needed_ul'] / 1000  # Convert to mL
        pipette_volume_ul = option['volume_needed_ul']
        
        # Get current vial volume (with fallback for simulation/errors)
        current_vial_volume = lash_e.nr_robot.get_vial_info(vial_name, 'vial_volume')
        
        # Conservation penalty: penalize high usage from low vials
        if current_vial_volume < conservation_threshold:
            conservation_penalty = (volume_needed_ml / current_vial_volume) * 50  # Scale factor
        else:
            conservation_penalty = 0  # No penalty for vials with plenty left
        
        # Pipetting preference: higher volumes generally preferred  
        pipetting_score = pipette_volume_ul
        
        # Combined score: preference minus penalty
        total_score = pipetting_score - conservation_penalty
        
        scored_options.append((option, total_score))
    
    # Return option with highest score
    return max(scored_options, key=lambda x: x[1])[0]

def create_plan_from_existing_stocks(existing_stock_solutions, surfactant_name, target_concentrations):
    """
    Create a dilution plan using existing stock solutions with proper dilution calculations.
    
    Args:
        existing_stock_solutions: List of stock solutions from previous stage
        surfactant_name: Name of surfactant (e.g., 'SDS', 'TTAB') 
        target_concentrations: List of concentrations needed for this stage
        
    Returns:
        plan: Dictionary with concentration_map and substocks_needed (like calculate_smart_dilution_plan output)
    """
    # Filter existing stocks for this surfactant and sort by concentration (highest first)
    surfactant_stocks = [stock for stock in existing_stock_solutions 
                        if stock['surfactant'] == surfactant_name]
    
    # Add the stock solution itself (always available)
    stock_vial_name = f"{surfactant_name}_stock"
    
    print(f"DEBUG: Looking for surfactant: {surfactant_name}")
    print(f"DEBUG: SURFACTANT_LIBRARY keys: {list(SURFACTANT_LIBRARY.keys())}")
    
    surfactant_info = SURFACTANT_LIBRARY[surfactant_name]
    print(f"DEBUG: Surfactant info: {surfactant_info}")
    
    # Look for stock concentration in the surfactant info
    if isinstance(surfactant_info, dict) and 'stock_conc' in surfactant_info:
        stock_conc = surfactant_info['stock_conc']
    elif isinstance(surfactant_info, (int, float)):
        stock_conc = surfactant_info  # Direct concentration value
    else:
        # Default fallback - assume 50 mM for common surfactants
        stock_conc = 50.0
        print(f"DEBUG: Using default stock concentration 50 mM for {surfactant_name}")
    
    surfactant_stocks.append({
        'vial_name': stock_vial_name,
        'target_concentration_mm': stock_conc,
        'surfactant': surfactant_name
    })
    print(f"DEBUG: Added stock solution {stock_vial_name} = {stock_conc} mM")

    
    # Sort by concentration (highest first)
    surfactant_stocks.sort(key=lambda x: x['target_concentration_mm'], reverse=True)
    
    print(f"DEBUG: Available stocks for {surfactant_name}:")
    for stock in surfactant_stocks:
        print(f"  {stock['vial_name']}: {stock['target_concentration_mm']} mM")
    
    # Create concentration map: {target_conc: {vial_name, concentration_mm, volume_needed_ul}}
    concentration_map = {}
    substocks_needed = []
    
    for target_conc in target_concentrations:
        # Find best solution based on pipettable volumes (like the smart dilution planner)
        min_pipette_volume_ul = MIN_WELL_PIPETTE_VOLUME_UL  # Consistent across all functions
        max_surfactant_volume_ul = MAX_SURFACTANT_VOLUME_UL  # Consistent across all functions
        
        options = []
        
        # Evaluate all available stocks for pipettability
        for stock in surfactant_stocks:
            source_conc = stock['target_concentration_mm']
            if source_conc > target_conc:  # Can only dilute down
                # Calculate volume needed: C1*V1 = C2*V2, so V1 = (C2*V2)/C1
                volume_needed_ul = (target_conc * WELL_VOLUME_UL) / source_conc
                
                # Check if volume is pipettable
                if (volume_needed_ul >= min_pipette_volume_ul and 
                    volume_needed_ul <= max_surfactant_volume_ul):
                    options.append({
                        'stock': stock,
                        'volume_needed_ul': volume_needed_ul,
                        'concentration_mm': source_conc
                    })
        
        if options:
            # Choose option using conservation-aware ranking
            best_option = rank_options_with_conservation_external(options, lash_e)
            
            concentration_map[target_conc] = {
                'vial_name': best_option['stock']['vial_name'],
                'concentration_mm': best_option['concentration_mm'], 
                'volume_needed_ul': best_option['volume_needed_ul']
            }
            
            # Add to substocks_needed (for compatibility)
            if not any(s['vial_name'] == best_option['stock']['vial_name'] for s in substocks_needed):
                substocks_needed.append({
                    'vial_name': best_option['stock']['vial_name'],
                    'concentration_mm': best_option['concentration_mm'],
                    'needed_for': [target_conc]
                })
                
            print(f"DEBUG: {target_conc:.3e} mM -> {best_option['stock']['vial_name']} ({best_option['volume_needed_ul']:.1f} uL)")
        else:
            # No pipettable solution found
            raise ValueError(f"No pipettable stock solution found for {target_conc:.3e} mM {surfactant_name} (volumes would be too small or too large)")
    
    return {
        'concentration_map': concentration_map,
        'substocks_needed': substocks_needed
    }

def get_achievable_concentrations(surfactant_name, target_concentrations):
    """
    Get which concentrations are achievable for a surfactant.
    
    Args:
        surfactant_name: Name of surfactant
        target_concentrations: List of target concentrations
        
    Returns:
        list: Achievable concentrations (None for non-achievable)
    """
    stock_conc = SURFACTANT_LIBRARY[surfactant_name]["stock_conc"]
    
    achievable = []
    for target in target_concentrations:
        if target is None:
            achievable.append(None)
            continue
        
        # Apply concentration correction factor for buffer dilution
        required_stock_conc = target * CONCENTRATION_CORRECTION_FACTOR
        
        # Check if this concentration is achievable given stock concentration and dilution limits  
        dilution_factor = stock_conc / required_stock_conc
        if dilution_factor >= 1.0:  # Only check that we're not trying to concentrate (dilution_factor >= 1)
            achievable.append(target)
        else:
            achievable.append(None)
    
    return achievable
    
def create_control_wells(surfactant_a_name, surfactant_b_name, position_prefix="start"):
    """
    Create quality control wells for start/end of experiment.
    Returns list of control well requirements.
    """
    controls = []
    
    # Control 1: Surfactant A stock only (200 uL)
    controls.append({
        'control_type': f'{position_prefix}_control_surfactant_A',
        'description': f'{surfactant_a_name} stock (200 uL)',
        'dilution_a_vial': f'{surfactant_a_name}_stock',
        'dilution_b_vial': None,
        'volume_a_ul': 200,
        'volume_b_ul': 0,
        'replicate': 1,
        'is_control': True
    })
    
    # Control 2: Surfactant B stock only (200 uL)  
    controls.append({
        'control_type': f'{position_prefix}_control_surfactant_B',
        'description': f'{surfactant_b_name} stock (200 uL)',
        'dilution_a_vial': None,
        'dilution_b_vial': f'{surfactant_b_name}_stock',
        'volume_a_ul': 0,
        'volume_b_ul': 200,
        'replicate': 1,
        'is_control': True
    })
    
    # Control 3: Buffer only (if using buffer)
    if ADD_BUFFER:
        controls.append({
            'control_type': f'{position_prefix}_control_buffer',
            'description': f'{SELECTED_BUFFER} buffer (200 uL)',
            'dilution_a_vial': None,
            'dilution_b_vial': None,
            'volume_a_ul': 0,
            'volume_b_ul': 0,
            'buffer_only': True,
            'replicate': 1,
            'is_control': True
        })
    
    # Control 4: Water only (200 uL)
    controls.append({
        'control_type': f'{position_prefix}_control_water',
        'description': 'Water blank (200 uL)',
        'dilution_a_vial': None,
        'dilution_b_vial': None,
        'volume_a_ul': 0,
        'volume_b_ul': 0,
        'water_only': True,
        'replicate': 1,
        'is_control': True
    })
    
    return controls
    
def measure_wellplate_turbidity(lash_e, wells_in_batch, wellplate_data, batch_recipes=None):
    """Measure turbidity for a batch of wells and save data."""
    from datetime import datetime
    import os
    
    lash_e.logger.info(f"  Measuring turbidity for wells {wells_in_batch[0]}-{wells_in_batch[-1]} (with buffer present)")
    turbidity_data = measure_turbidity(lash_e, wells_in_batch, batch_recipes)
    
    # Save raw turbidity data to CSV immediately (skip in simulation)
    turbidity_filename = None
    if turbidity_data is not None and not lash_e.simulate:
        experiment_name = getattr(lash_e, 'current_experiment_name', 'unknown_experiment')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        turbidity_filename = f"turbidity_plate{wellplate_data['current_plate']}_wells{wells_in_batch[0]}-{wells_in_batch[-1]}_{timestamp}.csv"
        turbidity_path = os.path.join("output", experiment_name, "measurement_backups", turbidity_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(turbidity_path), exist_ok=True)
        
        # Save actual DataFrame to CSV
        turbidity_data.to_csv(turbidity_path, index=True)
        lash_e.logger.info(f"    Saved raw turbidity data: {turbidity_filename}")
    
    # Store turbidity measurement data (keep for compatibility)
    turbidity_entry = {
        'plate_number': wellplate_data['current_plate'],
        'wells_measured': wells_in_batch,
        'measurement_type': 'turbidity_batch',
        'data_file': turbidity_filename,
        'timestamp': datetime.now().isoformat()
    }
    wellplate_data['measurements'].append(turbidity_entry)
    
    return turbidity_entry, turbidity_data

def measure_wellplate_fluorescence(lash_e, wells_in_batch, wellplate_data, batch_recipes=None):
    """Measure fluorescence for a batch of wells and save data."""
    from datetime import datetime
    import os
    
    lash_e.logger.info(f"  Measuring fluorescence for wells {wells_in_batch[0]}-{wells_in_batch[-1]}")
    fluorescence_data = measure_fluorescence(lash_e, wells_in_batch, batch_recipes)
    
    # Save raw fluorescence data to CSV immediately (skip in simulation)
    fluorescence_filename = None
    if fluorescence_data is not None and not lash_e.simulate:
        experiment_name = getattr(lash_e, 'current_experiment_name', 'unknown_experiment')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fluorescence_filename = f"fluorescence_plate{wellplate_data['current_plate']}_wells{wells_in_batch[0]}-{wells_in_batch[-1]}_{timestamp}.csv"
        fluorescence_path = os.path.join("output", experiment_name, "measurement_backups", fluorescence_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(fluorescence_path), exist_ok=True)
        
        # Save actual DataFrame to CSV
        fluorescence_data.to_csv(fluorescence_path, index=True)
        lash_e.logger.info(f"    Saved raw fluorescence data: {fluorescence_filename}")
    
    # Store fluorescence measurement data (keep for compatibility)
    fluorescence_entry = {
        'plate_number': wellplate_data['current_plate'],
        'wells_measured': wells_in_batch,
        'measurement_type': 'fluorescence_batch', 
        'data_file': fluorescence_filename,
        'timestamp': datetime.now().isoformat()
    }
    wellplate_data['measurements'].append(fluorescence_entry)
    
    return fluorescence_entry, fluorescence_data

def dispense_component_to_wellplate(lash_e, batch_df, vial_name, liquid_type, volume_column):
    """
    Unified dispensing method for any liquid component.
    
    Args:
        lash_e: Robot coordinator
        batch_df: DataFrame with well recipes for this batch
        vial_name: Name of source vial (e.g., 'water', 'SDS_1.0mM', 'pyrene_DMSO')
        liquid_type: Type of liquid for pipetting parameters ('water', 'DMSO')
        volume_column: Column name for volume (e.g., 'surf_A_volume_ul', 'water_volume_ul')
    """
    logger = lash_e.logger
    
    # Get component name for logging
    component_name = volume_column.replace('_volume_ul', '').replace('_', ' ').title()
    
    # Filter to wells that need this specific component/substock
    if volume_column == 'surf_A_volume_ul':
        # For surfactant A, also check that this is the correct substock
        wells_needing_component = batch_df[
            (batch_df[volume_column] > 0) & 
            (batch_df['substock_A_name'] == vial_name)
        ].copy()
    elif volume_column == 'surf_B_volume_ul':
        # For surfactant B, also check that this is the correct substock  
        wells_needing_component = batch_df[
            (batch_df[volume_column] > 0) & 
            (batch_df['substock_B_name'] == vial_name)
        ].copy()
    else:
        # For other components (water, buffer), use original logic
        wells_needing_component = batch_df[batch_df[volume_column] > 0].copy()
    
    if len(wells_needing_component) == 0:
        logger.info(f"  {component_name}: No wells need this component from {vial_name}")
        return

    # Sort wells by volume (largest first) to minimize tip changes
    wells_needing_component = wells_needing_component.sort_values(volume_column, ascending=False)
    
    logger.info(f"  {component_name}: Dispensing from {vial_name} to {len(wells_needing_component)} wells (sorted by volume)")
    
    # Dispense to each well individually
    for _, row in wells_needing_component.iterrows():
        well_idx = row['wellplate_index']
        volume_ul = row[volume_column]
        volume_ml = volume_ul / 1000
        
        logger.info(f"    Well {well_idx}: {volume_ul:.1f}uL from {vial_name}")
        
        # Robot actions (Lash_E handles simulation internally)
        lash_e.nr_robot.aspirate_from_vial(vial_name, volume_ml, liquid=liquid_type)
        lash_e.nr_robot.dispense_into_wellplate(
            dest_wp_num_array=[well_idx], 
            amount_mL_array=[volume_ml],
            liquid=liquid_type
        )
           
    logger.info(f"    {component_name}: Dispensing complete")

def position_surfactant_vials_by_concentration(lash_e, vial_names, batch_df, vial_type):
    """
    Cute little method to sort surfactant vials by concentration and move to safe positions.
    Prevents contamination by going concentrated ΓåÆ dilute, starting at clamp.
    
    Args:
        lash_e: Robot coordinator
        vial_names: List of vial names (e.g., ['SDS_1.0mM', 'SDS_5.0mM'])
        batch_df: DataFrame with concentration info
        vial_type: 'A' or 'B' for logging and column selection
        
    Returns:
        list: Vial names sorted by concentration (concentrated first)
    """
    if len(vial_names) == 0:
        return []
        
    logger = lash_e.logger
    safe_positions = [47, 46, 45, 44, 'clamp', 43, 36]  # Safe spots in order: clamp -> rack positions
    
    # Get concentration for each vial from the DataFrame
    vial_concentrations = []
    conc_column = f'substock_{vial_type}_conc_mm'
    name_column = f'substock_{vial_type}_name'
    
    for vial_name in vial_names:
        # Find the concentration for this vial
        vial_rows = batch_df[batch_df[name_column] == vial_name]
        if len(vial_rows) > 0:
            concentration = vial_rows.iloc[0][conc_column]
            vial_concentrations.append((vial_name, concentration))
        else:
            # Fallback - assume concentration from vial name if possible
            try:
                if '_' in vial_name and 'mM' in vial_name:
                    conc_str = vial_name.split('_')[-1].replace('mM', '')
                    concentration = float(conc_str)
                    vial_concentrations.append((vial_name, concentration))
                else:
                    # Can't determine concentration, put at end
                    vial_concentrations.append((vial_name, float('inf')))
            except:
                vial_concentrations.append((vial_name, float('inf')))
    
    # Sort by concentration (concentrated first)
    vial_concentrations.sort(key=lambda x: x[1], reverse=True)
    sorted_vials = [vial for vial, conc in vial_concentrations]
    
    logger.info(f"  Positioning surfactant {vial_type} vials by concentration (concentrated -> dilute):")
    
    # Move vials to safe positions in concentration order (concentrated -> dilute for positioning)
    for i, vial_name in enumerate(sorted_vials):
        if i < len(safe_positions):
            position = safe_positions[i]
            concentration = vial_concentrations[i][1]
            if position == 'clamp':
                logger.info(f"    {vial_name} ({concentration:.2f}mM) -> clamp")
                lash_e.nr_robot.move_vial_to_location(vial_name, 'clamp', 0)
            else:
                logger.info(f"    {vial_name} ({concentration:.2f}mM) -> main_8mL_rack[{position}]")
                lash_e.nr_robot.move_vial_to_location(vial_name, 'main_8mL_rack', position)
        else:
            logger.warning(f"    {vial_name}: No safe position available (too many vials)")
    
    # Return in reverse order for pipetting (dilute -> concentrated to prevent contamination)
    logger.info(f"  Dispensing order will be: dilute -> concentrated")
    return sorted_vials[::-1]  # Reverse the list for dispensing

def return_surfactant_vials_home(lash_e, vial_names, vial_type):
    """Return surfactant vials to home positions after dispensing."""
    if len(vial_names) == 0:
        return
        
    logger = lash_e.logger
    logger.info(f"  Returning surfactant {vial_type} vials to home positions:")
    
    for vial_name in vial_names:
        logger.info(f"    {vial_name} -> home")
        lash_e.nr_robot.return_vial_home(vial_name)

def return_water_vial_home(lash_e, vial_name):
    """Return water vial to home position after dispensing."""
    logger = lash_e.logger
    logger.info(f"  Returning {vial_name} -> home")
    lash_e.nr_robot.return_vial_home(vial_name)

def measure_turbidity(lash_e, well_indices, batch_recipes=None):
    """Measure turbidity using Cytation plate reader with predefined protocol."""
    lash_e.logger.info(f"Measuring turbidity in wells {well_indices} using protocol {TURBIDITY_PROTOCOL_FILE}...")
    
    if not lash_e.simulate:
        # Use the predefined turbidity protocol
        turbidity_data = lash_e.measure_wellplate(
            protocol_file_path=TURBIDITY_PROTOCOL_FILE,
            wells_to_measure=well_indices,
            plate_type="96 WELL PLATE"
        )
        
        # DEBUG: Show raw data structure
        lash_e.logger.info(f"TURBIDITY RAW DEBUG: type = {type(turbidity_data)}")
        if hasattr(turbidity_data, 'shape'):
            lash_e.logger.info(f"TURBIDITY RAW DEBUG: shape = {turbidity_data.shape}")
            lash_e.logger.info(f"TURBIDITY RAW DEBUG: columns = {list(turbidity_data.columns)}")
            lash_e.logger.info(f"TURBIDITY RAW DEBUG: first 3 rows:\n{turbidity_data.head(3)}")
        
        # Process Cytation format using utility function
        turbidity_data = flatten_cytation_data(turbidity_data, 'turbidity')
        if turbidity_data is not None:
            lash_e.logger.info(f"TURBIDITY PROCESSING: Final columns = {list(turbidity_data.columns)}")
            lash_e.logger.info(f"TURBIDITY PROCESSING: Final shape = {turbidity_data.shape}")
            lash_e.logger.info(f"TURBIDITY PROCESSING: First few processed rows:\n{turbidity_data.head(3)}")
        
        lash_e.logger.info(f"Successfully measured turbidity for {len(well_indices)} wells")
        return turbidity_data

    else:
        lash_e.logger.info("Simulation mode - generating realistic turbidity data")
        
        # Use passed batch recipe data for simulation
        simulated_data = []
        
        for well_idx in well_indices:
            # Try to find recipe for this well
            if batch_recipes is not None:
                well_recipe = batch_recipes[batch_recipes['wellplate_index'] == well_idx]
                if len(well_recipe) > 0:
                    row = well_recipe.iloc[0]
                    
                    # Experiment wells: use sophisticated simulation
                    if (row['well_type'] == 'experiment' and 
                        pd.notna(row['surf_A_conc_mm']) and pd.notna(row['surf_B_conc_mm'])):
                        surf_a_conc = row['surf_A_conc_mm']
                        surf_b_conc = row['surf_B_conc_mm']
                        sim_result = simulate_surfactant_measurements(surf_a_conc, surf_b_conc, add_noise=True)
                        simulated_data.append(sim_result['turbidity_600'])
                        
                    # Control wells: simple baseline values
                    elif 'water' in str(row['control_type']).lower():
                        simulated_data.append(0.02)  # Water blank
                    else:
                        simulated_data.append(0.35)  # Other controls
                else:
                    # Well not found in recipe - use simple pattern
                    simulated_data.append(0.25 + 0.1 * (well_idx % 5) / 5.0)
            else:
                # No recipe available - use simple pattern
                simulated_data.append(0.25 + 0.1 * (well_idx % 5) / 5.0)
        
        lash_e.logger.info(f"Generated realistic turbidity data for {len(well_indices)} wells (range: {min(simulated_data):.3f}-{max(simulated_data):.3f})")
        return {'turbidity': simulated_data}

def measure_fluorescence(lash_e, well_indices, batch_recipes=None):
    """Measure fluorescence using Cytation plate reader with predefined protocol."""
    lash_e.logger.info(f"Measuring fluorescence in wells {well_indices} using protocol {FLUORESCENCE_PROTOCOL_FILE}...")
    
    if not lash_e.simulate:
        # Use the predefined fluorescence protocol
        fluorescence_data = lash_e.measure_wellplate(
            protocol_file_path=FLUORESCENCE_PROTOCOL_FILE,
            wells_to_measure=well_indices,
            plate_type="96 WELL PLATE"
        )
        
        # DEBUG: Show raw data structure
        lash_e.logger.info(f"FLUORESCENCE RAW DEBUG: type = {type(fluorescence_data)}")
        if hasattr(fluorescence_data, 'shape'):
            lash_e.logger.info(f"FLUORESCENCE RAW DEBUG: shape = {fluorescence_data.shape}")
            lash_e.logger.info(f"FLUORESCENCE RAW DEBUG: columns = {list(fluorescence_data.columns)}")
            lash_e.logger.info(f"FLUORESCENCE RAW DEBUG: first 3 rows:\n{fluorescence_data.head(3)}")
        
        # Process Cytation format using utility function
        fluorescence_data = flatten_cytation_data(fluorescence_data, 'fluorescence')
        if fluorescence_data is not None:
            lash_e.logger.info(f"FLUORESCENCE PROCESSING: Final columns = {list(fluorescence_data.columns)}")
            lash_e.logger.info(f"FLUORESCENCE PROCESSING: Final shape = {fluorescence_data.shape}")
            lash_e.logger.info(f"FLUORESCENCE PROCESSING: First few processed rows:\n{fluorescence_data.head(3)}")
        
        lash_e.logger.info(f"Successfully measured fluorescence for {len(well_indices)} wells")
        return fluorescence_data

    else:
        lash_e.logger.info("Simulation mode - generating realistic fluorescence data")
        
        # Use passed batch recipe data for simulation
        f373_data = []
        f384_data = []
        
        for well_idx in well_indices:
            # Try to find recipe for this well
            if batch_recipes is not None:
                well_recipe = batch_recipes[batch_recipes['wellplate_index'] == well_idx]
                if len(well_recipe) > 0:
                    row = well_recipe.iloc[0]
                    
                    # Experiment wells: use sophisticated simulation
                    if (row['well_type'] == 'experiment' and 
                        pd.notna(row['surf_A_conc_mm']) and pd.notna(row['surf_B_conc_mm'])):
                        surf_a_conc = row['surf_A_conc_mm']
                        surf_b_conc = row['surf_B_conc_mm']
                        sim_result = simulate_surfactant_measurements(surf_a_conc, surf_b_conc, add_noise=True)
                        f373_data.append(sim_result['fluorescence_334_373'])
                        f384_data.append(sim_result['fluorescence_334_384'])
                        
                    # Control wells: simple baseline values
                    elif 'water' in str(row['control_type']).lower():
                        f373_data.append(5.0)   # Water blank
                        f384_data.append(8.0)
                    else:
                        f373_data.append(70.0)  # Other controls
                        f384_data.append(90.0)
                else:
                    # Well not found in recipe - use simple pattern
                    f373_data.append(70.0 + 20.0 * (well_idx % 7) / 7.0)
                    f384_data.append(90.0 + 15.0 * (well_idx % 7) / 7.0)
            else:
                # No recipe available - use simple pattern
                f373_data.append(70.0 + 20.0 * (well_idx % 7) / 7.0)
                f384_data.append(90.0 + 15.0 * (well_idx % 7) / 7.0)
        
        lash_e.logger.info(f"Generated realistic fluorescence data for {len(well_indices)} wells")
        lash_e.logger.info(f"  F373 range: {min(f373_data):.1f}-{max(f373_data):.1f}")
        lash_e.logger.info(f"  F384 range: {min(f384_data):.1f}-{max(f384_data):.1f}")
        
        return {
            '334_373': f373_data,
            '334_384': f384_data
        }

# ================================================================================
# SECTION 6: FULL WORKFLOW EXECUTION WITH LASH_E INTEGRATION
# ================================================================================



def create_experiment_folder_structure(experiment_name):
    """
    Create organized folder structure for experiment data.
    
    Folder organization:
    output/{experiment_name}/
    Γö£ΓöÇΓöÇ calibration_validation/     # Pipetting validation results
    Γö£ΓöÇΓöÇ measurement_backups/        # Raw measurement data backups  
    Γö£ΓöÇΓöÇ substocks/                  # Substock preparation logs
    Γö£ΓöÇΓöÇ analysis/                   # Data analysis outputs
    ΓööΓöÇΓöÇ logs/                       # Workflow execution logs
    
    Args:
        experiment_name: Name of the experiment (e.g., surfactant_grid_SDS_DTAB_20240203_143022)
        
    Returns:
        dict: Paths to all created subdirectories
    """
    base_folder = os.path.join("output", experiment_name)
    
    subfolders = {
        'base': base_folder,
        'validation': os.path.join(base_folder, "calibration_validation"),
        'measurement_backups': os.path.join(base_folder, "measurement_backups"), 
        'substocks': os.path.join(base_folder, "substocks"),
        'analysis': os.path.join(base_folder, "analysis"),
        'logs': os.path.join(base_folder, "logs")
    }
    
    # Create all subdirectories
    for folder_path in subfolders.values():
        os.makedirs(folder_path, exist_ok=True)
    
    return subfolders

def create_well_recipe_from_control(control, well_index, surfactant_a_name, surfactant_b_name):
    """Convert control well specification to well recipe DataFrame row."""
    # Start with base recipe structure using None for not-applicable values
    recipe = {
        'wellplate_index': well_index,
        'well_type': 'control',
        'control_type': control['control_type'],
        'surf_A': surfactant_a_name,
        'surf_B': surfactant_b_name,
        'surf_A_conc_mm': None,  # Will be set if applicable
        'surf_B_conc_mm': None,  # Will be set if applicable  
        'substock_A_name': None,
        'substock_A_conc_mm': None,
        'surf_A_volume_ul': 0.0,
        'substock_B_name': None, 
        'substock_B_conc_mm': None,
        'surf_B_volume_ul': 0.0,
        'water_volume_ul': 0.0,
        'buffer_volume_ul': BUFFER_VOLUME_UL if ADD_BUFFER else 0.0,
        'buffer_used': SELECTED_BUFFER if ADD_BUFFER else None,
        'pyrene_volume_ul': PYRENE_VOLUME_UL,  # Add pyrene to all wells
        'replicate': control['replicate']
    }
    
    # Handle different control types
    if control.get('water_only', False):
        # Water-only control: 200 ┬╡L water
        recipe['water_volume_ul'] = 200.0
        recipe['control_type'] = 'water_blank'
        
    elif control.get('buffer_only', False):
        # Buffer-only control: 200 ┬╡L buffer  
        recipe['buffer_volume_ul'] = 200.0
        recipe['water_volume_ul'] = 0.0
        recipe['control_type'] = 'buffer_blank'
        
    elif control['volume_a_ul'] > 0:
        # Surfactant A control: 200 ┬╡L surfactant A stock
        recipe['surf_A_conc_mm'] = SURFACTANT_LIBRARY[surfactant_a_name]['stock_conc']
        recipe['substock_A_name'] = control['dilution_a_vial']
        recipe['substock_A_conc_mm'] = SURFACTANT_LIBRARY[surfactant_a_name]['stock_conc'] 
        recipe['surf_A_volume_ul'] = control['volume_a_ul']
        recipe['water_volume_ul'] = 0.0
        recipe['control_type'] = 'surfactant_A_stock'
        # surf_B stays None (not applicable)
        
    elif control['volume_b_ul'] > 0:
        # Surfactant B control: 200 ┬╡L surfactant B stock
        recipe['surf_B_conc_mm'] = SURFACTANT_LIBRARY[surfactant_b_name]['stock_conc']
        recipe['substock_B_name'] = control['dilution_b_vial']
        recipe['substock_B_conc_mm'] = SURFACTANT_LIBRARY[surfactant_b_name]['stock_conc']
        recipe['surf_B_volume_ul'] = control['volume_b_ul']
        recipe['water_volume_ul'] = 0.0
        recipe['control_type'] = 'surfactant_B_stock'
        # surf_A stays None (not applicable)
    
    return recipe

def create_well_recipe_from_concentrations(conc_a, conc_b, plan_a, plan_b, well_index, surfactant_a_name, surfactant_b_name, replicate):
    """Convert target concentrations to well recipe DataFrame row using dilution plans."""
    
    # Get solutions from plans (this uses the working calculation logic)
    solution_a = plan_a['concentration_map'].get(conc_a)
    solution_b = plan_b['concentration_map'].get(conc_b)
    
    if not solution_a or not solution_b:
        raise ValueError(f"No solution found for concentrations {conc_a:.2e} + {conc_b:.2e} mM")
    
    # Calculate remaining water volume
    total_surfactant_volume = solution_a['volume_needed_ul'] + solution_b['volume_needed_ul']
    target_volume_before_buffer = WELL_VOLUME_UL - (BUFFER_VOLUME_UL if ADD_BUFFER else 0)
    water_volume = target_volume_before_buffer - total_surfactant_volume
    
    recipe = {
        'wellplate_index': well_index,
        'well_type': 'experiment',
        'control_type': 'experiment',
        'surf_A': surfactant_a_name,
        'surf_B': surfactant_b_name,
        'surf_A_conc_mm': conc_a,
        'surf_B_conc_mm': conc_b,
        'substock_A_name': solution_a['vial_name'],
        'substock_A_conc_mm': solution_a['concentration_mm'],
        'surf_A_volume_ul': solution_a['volume_needed_ul'],
        'substock_B_name': solution_b['vial_name'],
        'substock_B_conc_mm': solution_b['concentration_mm'],
        'surf_B_volume_ul': solution_b['volume_needed_ul'],
        'water_volume_ul': max(0, water_volume),  # Ensure non-negative
        'buffer_volume_ul': BUFFER_VOLUME_UL if ADD_BUFFER else 0.0,
        'buffer_used': SELECTED_BUFFER if ADD_BUFFER else None,
        'pyrene_volume_ul': PYRENE_VOLUME_UL,  # Add pyrene to all wells
        'replicate': replicate
    }
    
    return recipe

def create_complete_experiment_plan(lash_e, surfactant_a_name, surfactant_b_name, experiment_name,
                                  surf_a_min=None, surf_a_max=None, surf_b_min=None, surf_b_max=None, 
                                  existing_stock_solutions=None, number_concentrations=9):
    """
    Create complete experiment plan with simplified, clear data structure.
    Returns: experiment_plan dict with surfactants, stock_solutions_needed, and well_recipes_df
    """
    lash_e.logger.info("Step 2: Creating complete experiment plan...")
    
    # Step 1: Calculate concentration grids (with optional custom bounds)
    lash_e.logger.info("  Calculating adaptive concentration grids...")
    
    if any([surf_a_min, surf_a_max, surf_b_min, surf_b_max]):
        # Custom bounds provided - use them
        lash_e.logger.info("  Using custom concentration bounds...")
        concs_a = calculate_grid_concentrations(lash_e, surfactant_a_name, 
                                              min_conc=surf_a_min, max_conc=surf_a_max, 
                                              number_concentrations=number_concentrations)
        concs_b = calculate_grid_concentrations(lash_e, surfactant_b_name, 
                                              min_conc=surf_b_min, max_conc=surf_b_max, 
                                              number_concentrations=number_concentrations)
    else:
        # Use default bounds
        concs_a, concs_b = calculate_dual_surfactant_grids(lash_e, surfactant_a_name, surfactant_b_name, 
                                                          number_concentrations=number_concentrations)
    
    # Step 2: Check achievability and create smart dilution plans (keep existing working logic)
    achievable_a = get_achievable_concentrations(surfactant_a_name, concs_a)
    achievable_b = get_achievable_concentrations(surfactant_b_name, concs_b)
    achievable_concs_a = [c for c in achievable_a if c is not None]
    achievable_concs_b = [c for c in achievable_b if c is not None]
    
    lash_e.logger.info(f"  {surfactant_a_name}: {len(achievable_concs_a)}/{len(concs_a)} concentrations achievable")
    lash_e.logger.info(f"  {surfactant_b_name}: {len(achievable_concs_b)}/{len(concs_b)} concentrations achievable")
    
    # Step 3: Calculate smart dilution plans or use existing stock solutions
    if existing_stock_solutions:
        lash_e.logger.info("  Using existing stock solutions from previous stage...")
        # Convert existing stock solutions to plan format
        plan_a = create_plan_from_existing_stocks(existing_stock_solutions, surfactant_a_name, achievable_concs_a)
        plan_b = create_plan_from_existing_stocks(existing_stock_solutions, surfactant_b_name, achievable_concs_b)
        tracker_a = None
        tracker_b = None
    else:
        lash_e.logger.info("  Calculating optimal dilution strategies...")
        plan_a, tracker_a = calculate_smart_dilution_plan(lash_e, surfactant_a_name, achievable_concs_a)
        plan_b, tracker_b = calculate_smart_dilution_plan(lash_e, surfactant_b_name, achievable_concs_b)
    
    # Step 4: Create stock solutions list with dilution recipes
    stock_solutions_needed = []
    
    # Calculate dilution recipes to get the "how to make" details
    dilution_recipes = calculate_dilution_recipes(lash_e, plan_a, plan_b, surfactant_a_name, surfactant_b_name)
    
    # Create lookup for recipe details
    recipe_lookup = {recipe['Vial_Name']: recipe for recipe in dilution_recipes}
    
    # Add substocks for surfactant A
    for substock in plan_a['substocks_needed']:
        recipe_details = recipe_lookup.get(substock['vial_name'], {})
        stock_solutions_needed.append({
            'vial_name': substock['vial_name'],
            'surfactant': surfactant_a_name,
            'target_concentration_mm': substock['concentration_mm'],
            'needed_for_concentrations': ', '.join([f"{c:.2e}" for c in substock['needed_for']]),  # Remove brackets
            'source_vial': recipe_details.get('Source_Vial', 'Unknown'),
            'source_concentration_mm': recipe_details.get('Source_Conc_mM', 'Unknown'),
            'source_volume_ml': recipe_details.get('Source_Volume_mL', 'Unknown'),
            'water_volume_ml': recipe_details.get('Water_Volume_mL', 'Unknown'),
            'final_volume_ml': recipe_details.get('Final_Volume_mL', 6.0),
            'dilution_factor': recipe_details.get('Dilution_Factor', 'Unknown')
        })
    
    # Add substocks for surfactant B  
    for substock in plan_b['substocks_needed']:
        recipe_details = recipe_lookup.get(substock['vial_name'], {})
        stock_solutions_needed.append({
            'vial_name': substock['vial_name'],
            'surfactant': surfactant_b_name,
            'target_concentration_mm': substock['concentration_mm'],
            'needed_for_concentrations': ', '.join([f"{c:.2e}" for c in substock['needed_for']]),  # Remove brackets
            'source_vial': recipe_details.get('Source_Vial', 'Unknown'),
            'source_concentration_mm': recipe_details.get('Source_Conc_mM', 'Unknown'),
            'source_volume_ml': recipe_details.get('Source_Volume_mL', 'Unknown'),
            'water_volume_ml': recipe_details.get('Water_Volume_mL', 'Unknown'),
            'final_volume_ml': recipe_details.get('Final_Volume_mL', 6.0),
            'dilution_factor': recipe_details.get('Dilution_Factor', 'Unknown')
        })
    
    # Step 5: Create complete well-by-well recipes DataFrame
    lash_e.logger.info("  Creating complete well recipes...")
    well_recipes = []
    well_index = 0
    
    # Add start control wells
    start_controls = create_control_wells(surfactant_a_name, surfactant_b_name, "start")
    for control in start_controls:
        well_recipe = create_well_recipe_from_control(control, well_index, surfactant_a_name, surfactant_b_name)
        well_recipes.append(well_recipe)
        well_index += 1
    
    # Add grid experiment wells
    for conc_a in achievable_concs_a:
        for conc_b in achievable_concs_b:
            for replicate in range(N_REPLICATES):
                well_recipe = create_well_recipe_from_concentrations(
                    conc_a, conc_b, plan_a, plan_b, well_index, 
                    surfactant_a_name, surfactant_b_name, replicate + 1
                )
                well_recipes.append(well_recipe)
                well_index += 1
    
    # Add end control wells
    end_controls = create_control_wells(surfactant_a_name, surfactant_b_name, "end")
    for control in end_controls:
        well_recipe = create_well_recipe_from_control(control, well_index, surfactant_a_name, surfactant_b_name)
        well_recipes.append(well_recipe)
        well_index += 1
    
    # Convert to DataFrame
    import pandas as pd
    well_recipes_df = pd.DataFrame(well_recipes)
    
    # Step 6: Create experiment plan structure
    experiment_plan = {
        'surfactants': {
            'A': surfactant_a_name,
            'B': surfactant_b_name
        },
        'stock_solutions_needed': stock_solutions_needed,
        'well_recipes_df': well_recipes_df
    }
    
    # Step 7: Export to experiment folder (always save planning CSVs, even in simulation)
    experiment_output_folder = os.path.join("output", experiment_name)
    os.makedirs(experiment_output_folder, exist_ok=True)
    
    # Save well recipes to CSV - directly in experiment folder  
    recipes_csv_path = os.path.join(experiment_output_folder, "experiment_plan_well_recipes.csv")
    well_recipes_df.to_csv(recipes_csv_path, index=False)
    lash_e.logger.info(f"  Saved experiment plan: {recipes_csv_path}")
    
    # Save stock solutions to CSV - directly in experiment folder
    stocks_csv_path = os.path.join(experiment_output_folder, "experiment_plan_stock_solutions.csv")
    stocks_df = pd.DataFrame(stock_solutions_needed)
    stocks_df.to_csv(stocks_csv_path, index=False)
    lash_e.logger.info(f"  Saved stock solutions plan: {stocks_csv_path}")
    
    lash_e.logger.info(f"+ Planning complete: {len(well_recipes)} total wells ({len(achievable_concs_a)} x {len(achievable_concs_b)} grid + {len(start_controls) + len(end_controls)} controls)")
    
    return experiment_plan

def setup_experiment_environment(lash_e, surfactant_a_name, surfactant_b_name, simulate):
    """Initialize experiment environment: create folders, set name, log header."""
    # Create experiment name with timestamp
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"surfactant_grid_{surfactant_a_name}_{surfactant_b_name}_{experiment_timestamp}"
    lash_e.current_experiment_name = experiment_name  # Store for access by other functions
    
    # Create organized experiment folder structure
    experiment_folders = create_experiment_folder_structure(experiment_name)
    experiment_output_folder = experiment_folders['base']
    
    # Log experiment details
    lash_e.logger.info(f"Experiment: {experiment_name}")
    lash_e.logger.info(f"Output folder: {experiment_output_folder}")
    lash_e.logger.info(f"Organized subfolders: validation, measurement_backups, substocks, analysis, logs")
    
    # Log workflow header
    lash_e.logger.info("="*80)
    lash_e.logger.info("ADAPTIVE SURFACTANT GRID SCREENING WORKFLOW")
    lash_e.logger.info("="*80)
    lash_e.logger.info(f"Surfactants: {surfactant_a_name} + {surfactant_b_name}")
    lash_e.logger.info(f"Mode: {'SIMULATION' if simulate else 'HARDWARE'}")
    lash_e.logger.info("")
    
    return experiment_output_folder, experiment_name

def execute_dispensing_and_measurements(lash_e, well_recipes_df):
    """
    Execute dispensing and measurement phases for arbitrary well recipe data.
    
    Args:
        lash_e: Lash_E coordinator instance
        well_recipes_df: DataFrame with well recipes containing volume columns
        
    Returns:
        DataFrame: Updated well_recipes_df with measurement data added
    """
    lash_e.logger.info("Executing dispensing and measurements...")
    
    # Initialize wellplate tracking for measurements
    wellplate_data = {
        'current_plate': 1,
        'wells_used': 0,
        'last_measured_well': -1,
        'measurements': []
    }
    
    # Initialize measurement columns in the DataFrame
    well_recipes_df['turbidity_600'] = None
    well_recipes_df['fluorescence_334_373'] = None 
    well_recipes_df['fluorescence_334_384'] = None
    well_recipes_df['ratio'] = None
    
    # Process in batches to match measurement intervals
    total_wells = len(well_recipes_df)
    lash_e.logger.info(f"Total wells to process: {total_wells}")
    
    for batch_start in range(0, total_wells, MEASUREMENT_INTERVAL):
        batch_end = min(batch_start + MEASUREMENT_INTERVAL, total_wells)
        batch_df = well_recipes_df.iloc[batch_start:batch_end]
        
        lash_e.logger.info(f"\nProcessing batch {batch_start//MEASUREMENT_INTERVAL + 1}: wells {batch_start}-{batch_end-1}")
        
        # DISPENSING PHASE: Use unified dispensing for each component with proper vial positioning
        # Get unique vial names needed for this batch
        surf_a_vials = batch_df[batch_df['surf_A_volume_ul'] > 0]['substock_A_name'].dropna().unique()
        surf_b_vials = batch_df[batch_df['surf_B_volume_ul'] > 0]['substock_B_name'].dropna().unique()
        
        if CREATE_WELLPLATE:
            # Position surfactant A vials by concentration (concentrated ΓåÆ dilute)
            if len(surf_a_vials) > 0:
                sorted_surf_a_vials = position_surfactant_vials_by_concentration(lash_e, surf_a_vials, batch_df, 'A')
                
                # Dispense surfactant A substocks in concentration order
                for surf_a_vial in sorted_surf_a_vials:
                    dispense_component_to_wellplate(lash_e, batch_df, surf_a_vial, 'water', 'surf_A_volume_ul')
                
                # Return surfactant A vials to home
                lash_e.nr_robot.remove_pipet()
                return_surfactant_vials_home(lash_e, sorted_surf_a_vials, 'A')
            
            # Dispense water - split between two water vials to prevent running out
            water_wells = batch_df[batch_df['water_volume_ul'] > 0]
            if len(water_wells) > 0:
                # Split water dispensing in half
                mid_point = len(water_wells) // 2
                water_batch_1 = water_wells.iloc[:mid_point]
                water_batch_2 = water_wells.iloc[mid_point:]

                lash_e.nr_robot.move_vial_to_location('water', 'main_8mL_rack', 44)
                lash_e.nr_robot.move_vial_to_location('water_2', 'main_8mL_rack', 45)

                # Dispense first half with water vial
                if len(water_batch_1) > 0:
                    dispense_component_to_wellplate(lash_e, water_batch_1, 'water', 'water', 'water_volume_ul')
                    
                # Dispense second half with water_2 vial
                if len(water_batch_2) > 0:
                    dispense_component_to_wellplate(lash_e, water_batch_2, 'water_2', 'water', 'water_volume_ul')

                lash_e.logger.info("    Water dispensing complete")

                lash_e.nr_robot.remove_pipet()
                return_water_vial_home(lash_e, 'water')
                return_water_vial_home(lash_e, 'water_2')
            
            # Dispense buffer with safe positioning
            if ADD_BUFFER:
                lash_e.logger.info(f"  Positioning {SELECTED_BUFFER} buffer at clamp (safe position)")
                lash_e.nr_robot.move_vial_to_location(SELECTED_BUFFER, 'clamp', 0)
                dispense_component_to_wellplate(lash_e, batch_df, SELECTED_BUFFER, 'water', 'buffer_volume_ul')
                lash_e.nr_robot.remove_pipet()
                lash_e.nr_robot.return_vial_home(SELECTED_BUFFER)
            
            # Position surfactant B vials by concentration (concentrated ΓåÆ dilute)
            if len(surf_b_vials) > 0:
                sorted_surf_b_vials = position_surfactant_vials_by_concentration(lash_e, surf_b_vials, batch_df, 'B')
                
                # Dispense surfactant B substocks in concentration order
                for surf_b_vial in sorted_surf_b_vials:
                    dispense_component_to_wellplate(lash_e, batch_df, surf_b_vial, 'water', 'surf_B_volume_ul')
                
                # Return surfactant B vials to home
                lash_e.nr_robot.remove_pipet()
                return_surfactant_vials_home(lash_e, sorted_surf_b_vials, 'B')
        
        # MEASUREMENT PHASE: Turbidity first
        wells_in_batch = batch_df['wellplate_index'].tolist()
        
        lash_e.logger.info("  Measuring turbidity...")
        turbidity_entry, turbidity_data = measure_wellplate_turbidity(lash_e, wells_in_batch, wellplate_data, batch_df)
        
        # DEBUG: Show what turbidity data we got
        lash_e.logger.info(f"  TURBIDITY DEBUG: wells measured = {wells_in_batch}")
        lash_e.logger.info(f"  TURBIDITY DEBUG: backup entry = {turbidity_entry}")
        lash_e.logger.info(f"  TURBIDITY DEBUG: data type = {type(turbidity_data)}")
        if hasattr(turbidity_data, 'columns'):
            lash_e.logger.info(f"  TURBIDITY DEBUG: columns = {list(turbidity_data.columns)}")
            lash_e.logger.info(f"  TURBIDITY DEBUG: shape = {turbidity_data.shape}")
            lash_e.logger.info(f"  TURBIDITY DEBUG: first few rows:\n{turbidity_data.head()}")
        else:
            lash_e.logger.info(f"  TURBIDITY DEBUG: raw data = {turbidity_data}")
        
        # Add turbidity data to DataFrame by well position mapping
        if turbidity_data is not None:
            if not lash_e.simulate:
                # Hardware mode - turbidity_data is a DataFrame with well_position column
                if hasattr(turbidity_data, 'values') and len(turbidity_data) > 0 and 'well_position' in turbidity_data.columns:
                    # Convert well indices to well positions for lookup
                    def well_index_to_position(well_idx):
                        """Convert well index (0-95) to well position (A1-H12)"""
                        row = well_idx // 12  # 12 columns per row
                        col = well_idx % 12
                        return f"{chr(65 + row)}{col + 1}"
                    
                    # Get turbidity column (should be 'turbidity_600' after renaming)
                    turbidity_col = 'turbidity_600' if 'turbidity_600' in turbidity_data.columns else turbidity_data.columns[-1]
                    lash_e.logger.info(f"  Using turbidity column: {turbidity_col}")
                    
                    # Map each well by position
                    for well_idx in wells_in_batch:
                        well_position = well_index_to_position(well_idx)
                        matching_rows = turbidity_data[turbidity_data['well_position'] == well_position]
                        
                        if len(matching_rows) > 0:
                            turbidity_value = matching_rows.iloc[0][turbidity_col]
                            well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'turbidity_600'] = turbidity_value
                            lash_e.logger.info(f"  Mapped {well_position} (idx {well_idx}) -> turbidity {turbidity_value}")
                        else:
                            lash_e.logger.warning(f"  No turbidity data found for well {well_position} (idx {well_idx})")
                else:
                    lash_e.logger.warning("  Turbidity data missing or has unexpected format")
            else:
                # Simulation mode - realistic concentration-based values
                for i, well_idx in enumerate(wells_in_batch):
                    # Get concentrations for this well
                    well_row = well_recipes_df[well_recipes_df['wellplate_index'] == well_idx]
                    if len(well_row) > 0 and well_row.iloc[0]['well_type'] == 'experiment':
                        surf_a_conc = well_row.iloc[0]['surf_A_conc_mm']
                        surf_b_conc = well_row.iloc[0]['surf_B_conc_mm']
                        sim_data = simulate_surfactant_measurements(surf_a_conc, surf_b_conc)
                        turbidity_value = sim_data['turbidity_600']
                    else:
                        # Control wells get baseline values
                        turbidity_value = 0.2 + 0.1 * np.random.random()  # Small variation for controls
                    
                    well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'turbidity_600'] = turbidity_value
        
        # Add DMSO with safe positioning then measure fluorescence
        if CREATE_WELLPLATE:
            lash_e.logger.info("  Positioning pyrene_DMSO at clamp (safe position)")
            lash_e.nr_robot.move_vial_to_location('pyrene_DMSO', 'clamp', 0)
            dispense_component_to_wellplate(lash_e, batch_df, 'pyrene_DMSO', 'DMSO', 'pyrene_volume_ul')
            lash_e.nr_robot.remove_pipet()
            lash_e.nr_robot.return_vial_home('pyrene_DMSO')
        
        lash_e.logger.info("  Measuring fluorescence...")
        fluorescence_entry, fluorescence_data = measure_wellplate_fluorescence(lash_e, wells_in_batch, wellplate_data, batch_df)
        
        # DEBUG: Show what fluorescence data we got
        lash_e.logger.info(f"  FLUORESCENCE DEBUG: wells measured = {wells_in_batch}")
        lash_e.logger.info(f"  FLUORESCENCE DEBUG: backup entry = {fluorescence_entry}")
        lash_e.logger.info(f"  FLUORESCENCE DEBUG: data type = {type(fluorescence_data)}")
        if hasattr(fluorescence_data, 'columns'):
            lash_e.logger.info(f"  FLUORESCENCE DEBUG: columns = {list(fluorescence_data.columns)}")
            lash_e.logger.info(f"  FLUORESCENCE DEBUG: shape = {fluorescence_data.shape}")
            lash_e.logger.info(f"  FLUORESCENCE DEBUG: first few rows:\n{fluorescence_data.head()}")
        else:
            lash_e.logger.info(f"  FLUORESCENCE DEBUG: raw data = {fluorescence_data}")
        
        # Add fluorescence data to DataFrame by well position mapping
        if fluorescence_data is not None:
            if not lash_e.simulate:
                # Hardware mode - fluorescence_data is a DataFrame with well_position column
                if hasattr(fluorescence_data, 'values') and len(fluorescence_data) > 0 and 'well_position' in fluorescence_data.columns:
                    # Convert well indices to well positions for lookup
                    def well_index_to_position(well_idx):
                        """Convert well index (0-95) to well position (A1-H12)"""
                        row = well_idx // 12  # 12 columns per row
                        col = well_idx % 12
                        return f"{chr(65 + row)}{col + 1}"
                    
                    # Check for fluorescence columns
                    has_373 = '334_373' in fluorescence_data.columns
                    has_384 = '334_384' in fluorescence_data.columns
                    lash_e.logger.info(f"  Fluorescence columns available: 334_373={has_373}, 334_384={has_384}")
                    
                    if has_373 and has_384:
                        # Map each well by position
                        for well_idx in wells_in_batch:
                            well_position = well_index_to_position(well_idx)
                            matching_rows = fluorescence_data[fluorescence_data['well_position'] == well_position]
                            
                            if len(matching_rows) > 0:
                                val_373 = matching_rows.iloc[0]['334_373']
                                val_384 = matching_rows.iloc[0]['334_384']
                                ratio = val_373 / val_384 if val_384 != 0 else None
                                
                                well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'fluorescence_334_373'] = val_373
                                well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'fluorescence_334_384'] = val_384
                                well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'ratio'] = ratio
                                
                                lash_e.logger.info(f"  Mapped {well_position} (idx {well_idx}) -> F373={val_373}, F384={val_384}, ratio={ratio:.3f}")
                            else:
                                lash_e.logger.warning(f"  No fluorescence data found for well {well_position} (idx {well_idx})")
                    else:
                        lash_e.logger.warning(f"  Missing expected fluorescence columns. Available: {list(fluorescence_data.columns)}")
                else:
                    lash_e.logger.warning("  Fluorescence data missing or has unexpected format")
            else:
                # Simulation mode - realistic concentration-based values  
                for i, well_idx in enumerate(wells_in_batch):
                    # Get concentrations for this well
                    well_row = well_recipes_df[well_recipes_df['wellplate_index'] == well_idx]
                    if len(well_row) > 0 and well_row.iloc[0]['well_type'] == 'experiment':
                        surf_a_conc = well_row.iloc[0]['surf_A_conc_mm']
                        surf_b_conc = well_row.iloc[0]['surf_B_conc_mm']
                        sim_data = simulate_surfactant_measurements(surf_a_conc, surf_b_conc)
                        val_373 = sim_data['fluorescence_334_373']
                        val_384 = sim_data['fluorescence_334_384'] 
                        ratio = sim_data['ratio']
                    else:
                        # Control wells get baseline values
                        val_373 = 75.0 + 10.0 * np.random.random()
                        val_384 = 95.0 + 10.0 * np.random.random() 
                        ratio = val_373 / val_384
                    
                    well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'fluorescence_334_373'] = val_373
                    well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'fluorescence_334_384'] = val_384
                    well_recipes_df.loc[well_recipes_df['wellplate_index'] == well_idx, 'ratio'] = ratio

    return well_recipes_df

def simulate_surfactant_measurements(surf_a_conc, surf_b_conc, add_noise=True):
    """
    Generate realistic simulation data for turbidity and ratio based on surfactant concentrations.
    
    Creates 2D boundary patterns that transition between baseline and elevated states:
    - Ratio: Diagonal boundary from bottom-left to top-right
    - Turbidity: Circular boundary with different center
    
    Args:
        surf_a_conc: Surfactant A concentration in mM
        surf_b_conc: Surfactant B concentration in mM 
        add_noise: Whether to add realistic experimental noise
        
    Returns:
        dict: {'turbidity_600': float, 'fluorescence_334_373': float, 
               'fluorescence_334_384': float, 'ratio': float}
    """
    import numpy as np
    
    # Work in log space for realistic concentration effects
    log_a = np.log10(surf_a_conc)
    log_b = np.log10(surf_b_conc)
    
    # RATIO SIMULATION: Diagonal boundary (sigmoid transition)
    # Boundary runs from (-6, -4) to (-3, -1) in log space
    diagonal_distance = (log_a + log_b + 5.0) / np.sqrt(2)  # Distance from diagonal line
    ratio_transition = 1.0 / (1.0 + np.exp(-8.0 * diagonal_distance))  # Sharp sigmoid
    ratio_baseline = 0.6   # Low ratio state
    ratio_elevated = 1.4   # High ratio state 
    simulated_ratio = ratio_baseline + (ratio_elevated - ratio_baseline) * ratio_transition
    
    # TURBIDITY SIMULATION: Circular boundary (different center)
    # Circle centered at (-4.5, -2.5) in log space with radius 1.2
    center_a, center_b = -4.5, -2.5
    radius_distance = np.sqrt((log_a - center_a)**2 + (log_b - center_b)**2)
    turbidity_transition = 1.0 / (1.0 + np.exp(-5.0 * (radius_distance - 1.2)))
    turbidity_baseline = 0.15  # Low turbidity state
    turbidity_elevated = 0.85  # High turbidity state
    simulated_turbidity = turbidity_baseline + (turbidity_elevated - turbidity_baseline) * turbidity_transition
    
    # FLUORESCENCE: Derive from ratio (realistic relationship)
    # F384 stays relatively constant, F373 varies with ratio
    f384_base = 95.0 + 10.0 * np.sin(log_a + log_b)  # Slight spatial variation
    f373_base = simulated_ratio * f384_base  # F373/F384 = ratio
    
    # Add realistic experimental noise if requested
    if add_noise:
        noise_scale = 0.01  # 1% coefficient of variation (reduced from 5%)
        ratio_noise = 1.0 + np.random.normal(0, noise_scale)
        turbidity_noise = 1.0 + np.random.normal(0, noise_scale)
        fluorescence_noise = 1.0 + np.random.normal(0, noise_scale * 0.5)  # Lower noise for fluorescence
        
        simulated_ratio *= ratio_noise
        simulated_turbidity *= turbidity_noise
        f373_base *= fluorescence_noise
        f384_base *= fluorescence_noise
        
        # Recalculate ratio from potentially noisy fluorescence
        simulated_ratio = f373_base / f384_base if f384_base > 0 else simulated_ratio
    
    # Ensure physically reasonable bounds
    simulated_ratio = max(0.1, min(3.0, simulated_ratio))
    simulated_turbidity = max(0.01, min(1.5, simulated_turbidity))
    f373_base = max(10.0, min(300.0, f373_base))
    f384_base = max(10.0, min(300.0, f384_base))
    
    return {
        'turbidity_600': round(simulated_turbidity, 4),
        'fluorescence_334_373': round(f373_base, 2),
        'fluorescence_334_384': round(f384_base, 2),
        'ratio': round(simulated_ratio, 4)
    }

def validate_pipetting_system(lash_e, experiment_output_folder):
    """Perform comprehensive pipetting validation tests for all liquid types."""
    if VALIDATION_ONLY:
        lash_e.logger.info("  VALIDATION-ONLY MODE: Running comprehensive pipetting validation...")
    else:
        lash_e.logger.info("  Validating pipetting capability using embedded validation...")
    
    # Use the already-created experiment output folder
    lash_e.logger.info(f"  Validation data will be saved to: {experiment_output_folder}/calibration_validation/")
    
    try:
        # Import embedded validation functions
        from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy
        
        # Define test volumes for different liquid types
        dmso_test_volume = [0.005]  # 5 uL in mL
        
        validation_results = {}
        
        # Test 1: Water validation
        lash_e.logger.info("    Validating water pipetting (10-900 uL)...")
        # Split into two separate tests as requested
        
        # Test 1a: Small water volumes with conditioning
        small_volumes = [0.02,0.05,0.1]
        lash_e.logger.info("      Testing small water volumes (10-100 uL) with conditioning...")
        
        small_water_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial='water',
            destination_vial='water',  
            liquid_type='water',
            volumes_ml=small_volumes,
            replicates=3,
            output_folder=experiment_output_folder,
            switch_pipet=False,
            save_raw_data=not (hasattr(lash_e, 'simulate') and lash_e.simulate),
            condition_tip_enabled=True,
            conditioning_volume_ul=150
        )
        validation_results['water_small'] = small_water_results
        lash_e.logger.info(f"        Small water: R^2={small_water_results['r_squared']:.3f}, Accuracy={small_water_results['mean_accuracy_pct']:.1f}%")
        
        # Test 1b: Large water volumes with conditioning
        large_volumes = [0.2, 0.5, 0.9]
        lash_e.logger.info("      Testing large water volumes (200-900 uL) with conditioning...")
        
        large_water_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial='water',
            destination_vial='water',
            liquid_type='water',
            volumes_ml=large_volumes,
            replicates=3,
            output_folder=experiment_output_folder,
            switch_pipet=False,
            save_raw_data=not (hasattr(lash_e, 'simulate') and lash_e.simulate),
            condition_tip_enabled=True,
            conditioning_volume_ul=900
        )
        validation_results['water_large'] = large_water_results
        lash_e.logger.info(f"        Large water: R^2={large_water_results['r_squared']:.3f}, Accuracy={large_water_results['mean_accuracy_pct']:.1f}%")
        
        # Test 2: DMSO validation  
        lash_e.logger.info("    Validating DMSO pipetting (5 uL) with conditioning...")
        
        dmso_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial='pyrene_DMSO',
            destination_vial='pyrene_DMSO',
            liquid_type='DMSO',
            volumes_ml=dmso_test_volume,
            replicates=5,
            output_folder=experiment_output_folder,
            switch_pipet=False,
            save_raw_data=not (hasattr(lash_e, 'simulate') and lash_e.simulate),
            condition_tip_enabled=True,
            conditioning_volume_ul=25
        )
        validation_results['dmso'] = dmso_results
        lash_e.logger.info(f"      DMSO validation: R^2={dmso_results['r_squared']:.3f}, Accuracy={dmso_results['mean_accuracy_pct']:.1f}%")
        
        # Test 3a: Surfactant A stock validation - Small volumes (small tips)
        surfactant_a_stock = f"{SURFACTANT_A}_stock"
        lash_e.logger.info(f"    Validating {surfactant_a_stock} pipetting - Small volumes (10-100 uL) with conditioning...")
        
        surf_a_small_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial=surfactant_a_stock,
            destination_vial=surfactant_a_stock,
            liquid_type='water',  # Aqueous surfactant solution
            volumes_ml=small_volumes,  # Small volumes: 0.01, 0.05, 0.1 mL
            replicates=3,
            output_folder=experiment_output_folder,
            switch_pipet=False,
            save_raw_data=not (hasattr(lash_e, 'simulate') and lash_e.simulate),
            condition_tip_enabled=True,
            conditioning_volume_ul=100
        )
        validation_results['surfactant_a_small'] = surf_a_small_results
        lash_e.logger.info(f"        Small {surfactant_a_stock}: R^2={surf_a_small_results['r_squared']:.3f}, Accuracy={surf_a_small_results['mean_accuracy_pct']:.1f}%")
        
        # Test 3b: Surfactant A stock validation - Large volumes (large tips)
        lash_e.logger.info(f"    Validating {surfactant_a_stock} pipetting - Large volumes (200-900 uL) with conditioning...")
        
        surf_a_large_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial=surfactant_a_stock,
            destination_vial=surfactant_a_stock,
            liquid_type='water',  # Aqueous surfactant solution
            volumes_ml=large_volumes,  # Large volumes: 0.2, 0.5, 0.9 mL
            replicates=3,
            output_folder=experiment_output_folder,
            switch_pipet=False,
            save_raw_data=not (hasattr(lash_e, 'simulate') and lash_e.simulate),
            condition_tip_enabled=True,
            conditioning_volume_ul=800
        )
        validation_results['surfactant_a_large'] = surf_a_large_results
        lash_e.logger.info(f"        Large {surfactant_a_stock}: R^2={surf_a_large_results['r_squared']:.3f}, Accuracy={surf_a_large_results['mean_accuracy_pct']:.1f}%")
        
        # Test 4a: Surfactant B stock validation - Small volumes (small tips)
        surfactant_b_stock = f"{SURFACTANT_B}_stock"
        lash_e.logger.info(f"    Validating {surfactant_b_stock} pipetting - Small volumes (10-100 uL) with conditioning...")
        
        surf_b_small_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial=surfactant_b_stock,
            destination_vial=surfactant_b_stock,
            liquid_type='water',  # Aqueous surfactant solution
            volumes_ml=small_volumes,  # Small volumes: 0.01, 0.05, 0.1 mL
            replicates=1,
            output_folder=experiment_output_folder,
            switch_pipet=False,
            save_raw_data=not (hasattr(lash_e, 'simulate') and lash_e.simulate),
            condition_tip_enabled=True,
            conditioning_volume_ul=150
        )
        validation_results['surfactant_b_small'] = surf_b_small_results
        lash_e.logger.info(f"        Small {surfactant_b_stock}: R^2={surf_b_small_results['r_squared']:.3f}, Accuracy={surf_b_small_results['mean_accuracy_pct']:.1f}%")
        
        # Test 4b: Surfactant B stock validation - Large volumes (large tips)
        lash_e.logger.info(f"    Validating {surfactant_b_stock} pipetting - Large volumes (200-900 uL) with conditioning...")
        
        surf_b_large_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial=surfactant_b_stock,
            destination_vial=surfactant_b_stock,
            liquid_type='water',  # Aqueous surfactant solution
            volumes_ml=large_volumes,  # Large volumes: 0.2, 0.5, 0.9 mL
            replicates=1,
            output_folder=experiment_output_folder,
            switch_pipet=False,
            save_raw_data=not (hasattr(lash_e, 'simulate') and lash_e.simulate),
            condition_tip_enabled=True,
            conditioning_volume_ul=800
        )
        validation_results['surfactant_b_large'] = surf_b_large_results
        lash_e.logger.info(f"        Large {surfactant_b_stock}: R^2={surf_b_large_results['r_squared']:.3f}, Accuracy={surf_b_large_results['mean_accuracy_pct']:.1f}%")
        
        # Overall validation summary
        all_r_squared = [r['r_squared'] for r in validation_results.values()]
        all_accuracy = [r['mean_accuracy_pct'] for r in validation_results.values()]
        avg_r_squared = sum(all_r_squared) / len(all_r_squared)
        avg_accuracy = sum(all_accuracy) / len(all_accuracy)
        
        lash_e.logger.info(f"    All pipetting validations COMPLETE:")
        lash_e.logger.info(f"      Average R^2: {avg_r_squared:.3f}")
        lash_e.logger.info(f"      Average accuracy: {avg_accuracy:.1f}%")
        lash_e.logger.info(f"      Results saved to: {experiment_output_folder}/calibration_validation/")
        lash_e.logger.info("")
        lash_e.logger.info("="*60)
        lash_e.logger.info("PIPETTING VALIDATION COMPLETE - REVIEW RESULTS")
        lash_e.logger.info("="*60)
        lash_e.logger.info("")
        
        # Early exit for validation-only mode
        if VALIDATION_ONLY:
            lash_e.logger.info("="*60)
            lash_e.logger.info("VALIDATION-ONLY MODE: Exiting after validation completion")
            lash_e.logger.info("="*60)
            return {
                'validation_only': True,
                'validation_results': validation_results,
                'workflow_complete': True
            }
        
        return validation_results
        
    except ImportError as e:
        lash_e.logger.info(f"    Could not import embedded validation: {e}")
        lash_e.logger.info("    Skipping validation (validation system not available)...")
        return None
    except Exception as e:
        lash_e.logger.info(f"    Pipetting validation FAILED: {e}")
        lash_e.logger.info("    Continuing with workflow (validation failure non-critical)...")     
        return None

def execute_adaptive_surfactant_screening(surfactant_a_name="SDS", surfactant_b_name="DTAB", 
                                         surf_a_min=None, surf_a_max=None,
                                         surf_b_min=None, surf_b_max=None, 
                                         lash_e=None, existing_stock_solutions=None, 
                                         number_concentrations=9, experiment_output_folder=None, simulate=True):
    """
    Execute the complete surfactant grid screening workflow using adaptive concentrations.
    
    Args:
        surfactant_a_name: Name of first surfactant (cationic)
        surfactant_b_name: Name of second surfactant (anionic) 
        surf_a_min, surf_a_max: Custom concentration bounds for surfactant A (None = use defaults)
        surf_b_min, surf_b_max: Custom concentration bounds for surfactant B (None = use defaults)
        lash_e: Existing Lash_E instance (if None, creates new one)
        simulate: Run in simulation mode
        
    Returns:
        dict: Results including well_map, measurements, and concentrations used
    """
    # Using existing lash_e instance (always provided from main)
    lash_e.logger.info("Using existing Lash_E instance for workflow stage...")
    
    # Refill water vials to reset volume tracking between stages
    lash_e.logger.info("  Refilling water vials to reset volume tracking...")
    fill_water_vial(lash_e, "water")
    fill_water_vial(lash_e, "water_2")
    
    # Use provided experiment folder or create new one if not provided
    if experiment_output_folder is None:
        experiment_output_folder, experiment_name = setup_experiment_environment(lash_e, surfactant_a_name, surfactant_b_name, simulate)
    else:
        # Extract experiment name from folder path
        experiment_name = os.path.basename(experiment_output_folder)
        lash_e.logger.info(f"Using existing experiment folder: {experiment_output_folder}")
       
    # STEP 2: Create complete experiment plan with simplified structure
    experiment_plan = create_complete_experiment_plan(lash_e, surfactant_a_name, surfactant_b_name, experiment_name,
                                                     surf_a_min, surf_a_max, surf_b_min, surf_b_max, 
                                                     existing_stock_solutions, number_concentrations)
    
    # Extract the well recipes DataFrame
    well_recipes_df = experiment_plan['well_recipes_df']
    stock_solutions_needed = experiment_plan['stock_solutions_needed']
    
    lash_e.logger.info(f"+ Experiment plan created: {len(well_recipes_df)} wells planned")
    lash_e.logger.info(f"+ Stock solutions needed: {len(stock_solutions_needed)}")   
    
    # STEP 3: Create physical substocks BEFORE dispensing
    lash_e.logger.info("Step 4: Creating physical substocks from calculated recipes...")
    
    # Get dilution recipes from the experiment plan
    dilution_recipes = []
    for stock in stock_solutions_needed:
        if stock['source_vial'] != 'Unknown':  # Only include valid recipes
            dilution_recipes.append({
                'Vial_Name': stock['vial_name'],
                'Surfactant': stock['surfactant'],
                'Target_Conc_mM': stock['target_concentration_mm'],
                'Source_Vial': stock['source_vial'],
                'Source_Conc_mM': stock['source_concentration_mm'],
                'Source_Volume_mL': stock['source_volume_ml'],
                'Water_Volume_mL': stock['water_volume_ml'],
                'Final_Volume_mL': stock['final_volume_ml']
            })
    
    # Sort by concentration (highest first) for correct creation order
    dilution_recipes.sort(key=lambda x: x['Target_Conc_mM'], reverse=True)
    
    # Create substocks in the correct order (highest concentration first)
    if dilution_recipes:
        created_substocks = create_substocks_from_recipes(lash_e, dilution_recipes)
        newly_created = [s for s in created_substocks if s['created']]
        already_existing = [s for s in created_substocks if not s['created'] and not s.get('error')]
        actually_failed = [s for s in created_substocks if s.get('error')]
        
        lash_e.logger.info(f"+ Substock creation complete: {len(newly_created)} newly created, {len(already_existing)} already existed, {len(actually_failed)} failed")
        
        if already_existing:
            lash_e.logger.info(f"Substocks already existed (OK):")
            for substock in already_existing:
                lash_e.logger.info(f"  {substock['vial_name']}: already available")
        
        if actually_failed:
            lash_e.logger.info(f"Failed substocks (ERRORS):")
            for substock in actually_failed:
                error_msg = substock.get('error', 'Unknown error')
                lash_e.logger.info(f"  {substock['vial_name']}: {error_msg}")
    else:
        lash_e.logger.info("+ No substocks needed (using only stock solutions)")
    
    # STEP 5: Execute dispensing and measurements in clear phases
    lash_e.logger.info("Step 5: Executing dispensing and measurements...")
    well_recipes_df = execute_dispensing_and_measurements(lash_e, well_recipes_df)
    # STEP 6: Save results to experiment folder
    lash_e.logger.info("Step 6: Saving results...")
    output_folder = experiment_output_folder
    
    # Save the complete well recipes with measurements
    final_results_path = os.path.join(output_folder, "complete_experiment_results.csv")
    well_recipes_df.to_csv(final_results_path, index=False)
    lash_e.logger.info(f"  Complete results saved to: {final_results_path}")
    
    # Generate heatmap visualizations
    try:
        lash_e.logger.info("  Generating heatmap visualizations...")
        
        # Create heatmap subfolder
        heatmap_folder = os.path.join(output_folder, "heatmap")
        os.makedirs(heatmap_folder, exist_ok=True)
        
        # Generate heatmaps using the visualization function
        generate_surfactant_grid_heatmaps(final_results_path, heatmap_folder, lash_e.logger, surfactant_a_name, surfactant_b_name)
        
        lash_e.logger.info(f"  Heatmaps saved to: {heatmap_folder}")
        
        # Calculate adaptive concentration bounds using baseline rectangle method
        lash_e.logger.info("  Calculating adaptive concentration bounds...")
        
        # Load experimental data for bounds calculation
        experiment_results_df = pd.read_csv(final_results_path)
        experiment_data_for_bounds = experiment_results_df[experiment_results_df['well_type'] == 'experiment'].copy()
        
        new_bounds = calculate_adaptive_concentration_bounds(
            experiment_data_for_bounds, 
            surfactant_a_name, 
            surfactant_b_name, 
            output_folder, 
            lash_e.logger
        )
        
        lash_e.logger.info(f"  Adaptive bounds calculated: {new_bounds}")
        
    except Exception as e:
        lash_e.logger.warning(f"  Failed to generate heatmaps: {str(e)}")
        lash_e.logger.warning("  Continuing without heatmap generation...")
    
    lash_e.logger.info(f"+ Results saved to: {output_folder}")
    
    # # Get actual pipette usage breakdown 
    pipette_breakdown = get_pipette_usage_breakdown(lash_e)
    lash_e.logger.info(f"+ Pipette tips used: {pipette_breakdown['large_tips']} large, {pipette_breakdown['small_tips']} small (total: {pipette_breakdown['total']}) ({'simulated' if simulate else 'actual'})")
    
    # Summary statistics
    experiment_wells = well_recipes_df[well_recipes_df['well_type'] == 'experiment']
    control_wells = well_recipes_df[well_recipes_df['well_type'] == 'control']
    measured_wells = well_recipes_df[well_recipes_df['turbidity_600'].notna()]
    
    lash_e.logger.info("\n" + "="*60)
    lash_e.logger.info("EXPERIMENT COMPLETE - SUMMARY")
    lash_e.logger.info("="*60)
    lash_e.logger.info(f"Surfactants: {surfactant_a_name} + {surfactant_b_name}")
    lash_e.logger.info(f"Wells: {len(experiment_wells)} experiment + {len(control_wells)} control = {len(well_recipes_df)} total")
    lash_e.logger.info(f"Measurements: {len(measured_wells)} wells measured")
    lash_e.logger.info(f"Mode: {'SIMULATION' if simulate else 'HARDWARE'}")
    lash_e.logger.info(f"Results: {final_results_path}")
    lash_e.logger.info("="*60)
       
    # Display final DataFrame for verification
    if not well_recipes_df.empty:
        lash_e.logger.info("DataFrame sample with measurements:")
        sample_cols = ['wellplate_index', 'surf_A_conc_mm', 'surf_B_conc_mm', 'turbidity_600', 'fluorescence_334_373', 'fluorescence_334_384', 'ratio']
        existing_cols = [col for col in sample_cols if col in well_recipes_df.columns]
        sample_df = well_recipes_df[existing_cols].head(10)
        lash_e.logger.info(f"\n{sample_df.to_string()}")
    
    # Send Slack completion message (hardware mode only)
    if not simulate:
        stage_info = f"Adaptive screening complete: {surfactant_a_name}+{surfactant_b_name}, {len(well_recipes_df)} wells, {len(measured_wells)} measured"
        slack_agent.send_slack_message(stage_info)
    
    # Return clean results with just the essential data
    return {
        'surfactant_a': surfactant_a_name,
        'surfactant_b': surfactant_b_name,
        'well_recipes_df': well_recipes_df,  # Complete DataFrame with measurements
        'experiment_plan': experiment_plan,
        'total_wells': len(well_recipes_df),
        'measured_wells': len(measured_wells),
        'output_folder': output_folder,
        'pipette_breakdown': pipette_breakdown,
        'simulation': simulate,
        'workflow_complete': True
    }

def execute_iterative_workflow(surfactant_a_name="SDS", surfactant_b_name="DTAB", 
                              number_concentrations=5, target_measurements=96, 
                              gradient_suggestions_per_iteration=12, lash_e=None, 
                              experiment_output_folder=None, simulate=True):
    """
    Execute iterative surfactant screening with configurable number of concentrations.
    
    This method allows for rapid initial screening with fewer concentrations (e.g., 5)
    before proceeding to iterative gradient-based refinement until target measurements reached.
    
    Args:
        surfactant_a_name: Name of first surfactant (e.g., 'SDS')
        surfactant_b_name: Name of second surfactant (e.g., 'DTAB') 
        number_concentrations: Number of concentration steps (default 5 for rapid screening)
        target_measurements: Total measurements to reach (default 200)
        gradient_suggestions_per_iteration: Number of gradient suggestions per iteration (default 12)
        simulate: Run in simulation mode
        
    Returns:
        dict: Results from iterative screening with experiment plan and measurements
    """
    print("="*80)
    print(f"ITERATIVE SURFACTANT SCREENING WORKFLOW")
    print(f"Surfactants: {surfactant_a_name} + {surfactant_b_name}")
    print(f"Concentrations: {number_concentrations} steps (rapid screening)")
    print(f"Target measurements: {target_measurements}")
    print(f"Mode: {'SIMULATION' if simulate else 'HARDWARE'}")
    print("="*80)

    lash_e.grab_new_wellplate()

    # Execute initial adaptive screening with configurable number of concentrations
    results = execute_adaptive_surfactant_screening(
        surfactant_a_name=surfactant_a_name,
        surfactant_b_name=surfactant_b_name, 
        surf_a_min=None,  # Use default bounds
        surf_a_max=None,
        surf_b_min=None,
        surf_b_max=None,
        lash_e=lash_e,
        existing_stock_solutions=None,  # Fresh start, no existing solutions
        number_concentrations=number_concentrations,  # Configurable parameter
        experiment_output_folder=experiment_output_folder,
        simulate=simulate
    )

    # Track current measurements and wellplate capacity
    current_measurements = len(results['well_recipes_df'])
    current_wellplate_wells = current_measurements
    iteration = 1
    
    print(f"Initial screening complete: {current_measurements} measurements")
    print(f"Starting iterative gradient exploration...")
    
    # Iterative gradient exploration loop
    while current_measurements < target_measurements:
        print(f"\n--- Iteration {iteration} ---")
        print(f"Current measurements: {current_measurements}/{target_measurements}")
        print(f"Current wellplate usage: {current_wellplate_wells}/96 wells")
        
        fill_water_vial(lash_e, "water")  # Ensure water is refilled for pipetting
        fill_water_vial(lash_e, "water_2")  # Refill second water vial as well

        # Refill both surfactant stock vials
        refill_surfactant_vial(lash_e, f"{surfactant_a_name}_stock", liquid=surfactant_a_name)
        refill_surfactant_vial(lash_e, f"{surfactant_b_name}_stock", liquid=surfactant_b_name)

        # Calculate how many wells we can use in current wellplate
        wells_remaining_in_plate = 96 - current_wellplate_wells
        max_measurements_this_iteration = min(
            target_measurements - current_measurements,
            wells_remaining_in_plate
        )
        
        # Always get full gradient suggestions, regardless of plate constraints
        gradient_suggestions_requested = gradient_suggestions_per_iteration
        
        print(f"Requesting {gradient_suggestions_requested} gradient suggestions")
        print(f"Can process maximum {max_measurements_this_iteration} measurements this iteration ({wells_remaining_in_plate} wells remaining in plate)")
            
        # Get gradient suggestions (always request full amount)
        gradient_results = find_high_gradient_areas(
            results, lash_e, 
            n_suggestions=gradient_suggestions_requested,
            starting_well_index=current_wellplate_wells  # Start from current position in wellplate
        )
        
        # Limit the actual measurements to what we can process
        if len(gradient_results) > max_measurements_this_iteration:
            print(f"Limiting {len(gradient_results)} suggestions to {max_measurements_this_iteration} due to plate constraints")
            gradient_results = gradient_results.head(max_measurements_this_iteration)
        
        # Merge new results with existing data
        print(f"Merging {len(gradient_results)} new measurements with existing data...")
        merged_df = pd.concat([results['well_recipes_df'], gradient_results], ignore_index=True)
        results['well_recipes_df'] = merged_df
        
        # Update tracking
        current_measurements = len(merged_df)
        current_wellplate_wells += len(gradient_results)
        
        # Switch wellplate only if current one is full
        if current_wellplate_wells >= 96 and current_measurements < target_measurements:
            print("Current wellplate full, switching to new wellplate...")
            lash_e.discard_used_wellplate()
            lash_e.grab_new_wellplate()
            current_wellplate_wells = 0
        
        # Save updated results
        updated_results_path = os.path.join(results['output_folder'], "iterative_experiment_results.csv")
        merged_df.to_csv(updated_results_path, index=False)
        print(f"Updated results saved: {updated_results_path}")
        
        iteration += 1
        
        # Safety check to prevent infinite loops
        if iteration > 20:
            print("Maximum iterations reached (20), stopping...")
            break

    # Final cleanup
    lash_e.discard_used_wellplate()
    
    # Update final results
    results['total_wells'] = len(results['well_recipes_df'])
    results['measured_wells'] = len(results['well_recipes_df'][results['well_recipes_df']['turbidity_600'].notna()])

    print(f"\nIterative workflow complete!")
    print(f"Final results: {results['total_wells']} wells, {results['measured_wells']} measured")
    print(f"Iterations completed: {iteration-1}")
    print(f"Output folder: {results['output_folder']}")
        
    return results

def find_high_gradient_areas(existing_results, lash_e, n_suggestions=12, starting_well_index=None):
    """
    Find high-gradient areas from initial screening and perform targeted exploration.
    
    Uses existing substocks from initial screening - NO new substock creation.
    Continues well indexing from current wellplate position with 96-well limit.
    
    Args:
        existing_results: Results dict from execute_iterative_workflow() 
        lash_e: Lash_E coordinator instance (reused from initial screening)
        n_suggestions: Number of concentration pairs to explore (default 12)
        starting_well_index: Starting well position (None = continue from last used well)
        
    Returns:
        DataFrame: New measurement data from targeted exploration
    """
    logger = lash_e.logger
    logger.info(f"Starting high-gradient area exploration with {n_suggestions} suggestions...")
    
    # Extract data from initial screening
    initial_data_df = existing_results['well_recipes_df']
    available_substocks = existing_results['experiment_plan']['stock_solutions_needed']
    surfactant_a_name = existing_results['surfactant_a']
    surfactant_b_name = existing_results['surfactant_b']
    
    logger.info(f"Initial data: {len(initial_data_df)} wells measured")
    logger.info(f"Available substocks: {len(available_substocks)} solutions")
    
    # STEP 1: Analyze gradients and get suggested concentrations (placeholder for your method)
    logger.info("Analyzing gradients to identify high-interest regions...")
    suggested_concentrations = get_suggested_concentrations(
        initial_data_df, 
        surfactant_a_name, 
        surfactant_b_name, 
        n_suggestions=n_suggestions,
        output_dir=existing_results['output_folder']  # Pass output directory for visualizations
    )
    
    logger.info(f"Generated {len(suggested_concentrations)} concentration suggestions")
    
    # STEP 2: Use existing substocks to match suggested concentrations
    logger.info("Matching suggestions to existing substocks...")
    
    # Separate concentrations for each surfactant
    target_concs_a = [pair[0] for pair in suggested_concentrations]
    target_concs_b = [pair[1] for pair in suggested_concentrations]
    
    # Use smart matching with existing substocks (NO new dilutions)
    plan_a = create_plan_from_existing_stocks(available_substocks, surfactant_a_name, target_concs_a)
    plan_b = create_plan_from_existing_stocks(available_substocks, surfactant_b_name, target_concs_b)
    
    # STEP 3: Create well recipes for suggestions
    logger.info("Creating well recipes for targeted exploration...")
    
    # Determine starting well position (continue from where previous operations left off)
    if starting_well_index is None:
        # Get the highest well index from existing results + 1
        last_well_used = existing_results['well_recipes_df']['wellplate_index'].max()
        starting_well_index = last_well_used + 1 if pd.notna(last_well_used) else 0
    
    # CRITICAL FIX: Ensure we don't exceed 96-well plate limits
    wells_remaining_in_plate = 96 - starting_well_index
    if wells_remaining_in_plate <= 0:
        raise ValueError(f"CRITICAL: Starting well index {starting_well_index} exceeds 96-well plate capacity! Current wellplate is full.")
    
    # Limit suggestions to available wells in current plate
    max_suggestions_possible = min(n_suggestions, wells_remaining_in_plate)
    if max_suggestions_possible < n_suggestions:
        logger.info(f"Limiting suggestions from {n_suggestions} to {max_suggestions_possible} due to wellplate capacity")
    
    logger.info(f"Continuing well indexing from position {starting_well_index} (max {max_suggestions_possible} suggestions possible)")
    
    well_recipes = []
    
    # Only process the number of suggestions we can actually fit
    suggestions_to_process = suggested_concentrations[:max_suggestions_possible]
    
    for i, (conc_a, conc_b) in enumerate(suggestions_to_process):
        well_index = starting_well_index + i  # Continue from last used well position
        
        # Double-check well index is valid
        if well_index >= 96:
            logger.warning(f"Skipping suggestion {i+1}: well index {well_index} exceeds 96-well plate capacity")
            break
            
        replicate = 1
        
        try:
            recipe = create_well_recipe_from_concentrations(
                conc_a, conc_b, plan_a, plan_b, 
                well_index, surfactant_a_name, surfactant_b_name, replicate
            )
            well_recipes.append(recipe)
        except ValueError as e:
            raise ValueError(f"CRITICAL: Cannot create recipe for {conc_a:.3e} + {conc_b:.3e} mM: {e}. Algorithm suggestions are not achievable with current stocks!")
    
    # Convert to DataFrame
    suggestions_df = pd.DataFrame(well_recipes)
    logger.info(f"Created {len(suggestions_df)} well recipes for dispensing")
    
    # Execute dispensing and measurements using existing infrastructure
    logger.info("Executing targeted dispensing and measurements...")
    measured_suggestions_df = execute_dispensing_and_measurements(lash_e, suggestions_df)
    
    logger.info(f"High-gradient exploration complete: {len(measured_suggestions_df)} wells measured")
    return measured_suggestions_df

def get_suggested_concentrations(experiment_data_df, surfactant_a_name, surfactant_b_name, n_suggestions=12, output_dir=None):
    """
    Use Delaunay triangle refinement to suggest high-variation concentration pairs for exploration.
    
    Uses DelaunayTriangleRecommender to find boundaries in ratio measurements
    by analyzing output variation among triangle vertices in the concentration space.
    
    Args:
        experiment_data_df: DataFrame with measured experimental data including:
                           - surf_A_conc_mm, surf_B_conc_mm (concentrations)
                           - turbidity_600, ratio (measurements)
        surfactant_a_name: Name of first surfactant (for logging)
        surfactant_b_name: Name of second surfactant (for logging)  
        n_suggestions: Number of concentration pairs to suggest
        
    Returns:
        list of tuples: [(conc_a_1, conc_b_1), (conc_a_2, conc_b_2), ...] 
                       concentration pairs in mM for high-variation exploration
    """
    
    print(f"Analyzing triangle boundaries in {surfactant_a_name}+{surfactant_b_name} data...")
    print(f"Input data: {len(experiment_data_df)} experimental wells")
    
    # Import the Delaunay triangle recommender
    try:
        import sys
        import os
        recommender_path = os.path.join(os.path.dirname(__file__), '..', 'recommenders')
        if recommender_path not in sys.path:
            sys.path.append(recommender_path)
        from recommenders.delaunay_triangle_recommender import DelaunayTriangleRecommender
    except ImportError as e:
        raise ImportError(f"CRITICAL: DelaunayTriangleRecommender import failed: {e}. Adaptive algorithm cannot run without this!")
    
    # Initialize the triangle recommender (focus on ratio AND turbidity boundaries)
    recommender = DelaunayTriangleRecommender(
        input_columns=['surf_A_conc_mm', 'surf_B_conc_mm'],  # 2D concentration space (X, Y)
        output_columns=['ratio', 'turbidity_600'],           # Focus on BOTH ratio and turbidity boundaries 
        log_transform_inputs=True,    # Work in log concentration space
        normalization_method='log_zscore'  # Log + z-score normalization
    )
    
    # Get recommendations from triangle analysis
    print(f"Running Delaunay triangle analysis for {n_suggestions} boundary suggestions...")
    print(f"DEBUG: Using output columns: {recommender.output_columns}")
    print(f"DEBUG: Input data shape: {experiment_data_df.shape}")
    print(f"DEBUG: Available columns: {list(experiment_data_df.columns)}")
    
    recommendations_df = recommender.get_recommendations(
        experiment_data_df, 
        n_points=n_suggestions,
        min_spacing_factor=0.5,  # Minimum spacing between triangle centroids
        tol_factor=0.1,         # Tolerance for duplicate detection
        triangle_score_method='max',  # Use max distance for sensitivity
        output_dir=output_dir,  # Save triangle visualizations to output folder
        create_visualization=True  # Create and save triangle analysis plots
    )
    
    print(f"DEBUG: Triangle recommender returned {len(recommendations_df)} suggestions")
    if len(recommendations_df) > 0:
        print(f"DEBUG: First few suggestions:")
        print(recommendations_df[['surf_A_conc_mm', 'surf_B_conc_mm']].head())
    
    if len(recommendations_df) == 0:
        raise RuntimeError(f"CRITICAL: Triangle analysis returned 0 recommendations! Algorithm is broken. Input data shape: {experiment_data_df.shape}, columns: {list(experiment_data_df.columns)}, output columns: {recommender.output_columns}")
    
    # Extract concentration pairs from recommendations
    concentration_pairs = []
    for _, row in recommendations_df.iterrows():
        conc_a = row['surf_A_conc_mm']
        conc_b = row['surf_B_conc_mm']
        concentration_pairs.append((conc_a, conc_b))
    
    print(f"Triangle analysis complete: {len(concentration_pairs)} concentration pairs identified")
    print(f"Triangle score range: {recommendations_df['triangle_score'].min():.4f} - {recommendations_df['triangle_score'].max():.4f}")
    
    # Show sample of recommendations for verification
    print("Sample recommendations:")
    for i, (conc_a, conc_b) in enumerate(concentration_pairs[:5]):
        score = recommendations_df.iloc[i]['triangle_score']
        print(f"  {i+1}: {surfactant_a_name}={conc_a:.3e} mM, {surfactant_b_name}={conc_b:.3e} mM (score: {score:.4f})")
    
    return concentration_pairs

def execute_2_stage_workflow(lash_e, surfactant_a_name="SDS", surfactant_b_name="DTAB", 
                            experiment_output_folder=None, simulate=True):
    """
    Execute complete 2-stage adaptive workflow:
    1. Broad exploration with default concentration ranges
    2. Calculate optimal bounds from baseline classification  
    3. Focused exploration with adaptive bounds
    
    Args:
        lash_e: Pre-initialized and validated Lash_E instance
        surfactant_a_name: Name of first surfactant
        surfactant_b_name: Name of second surfactant
        simulate: Run in simulation mode
        
    Returns:
        dict: Combined results from both stages
    """
    stage_1_results = {}
    stage_2_results = {}
    
    try:
        # Use the pre-initialized and validated Lash_E instance
        print("Using validated Lash_E for 2-stage adaptive workflow...")
        
        # STAGE 1: Broad exploration with default bounds
        print("="*80)
        print("STARTING 2-STAGE ADAPTIVE WORKFLOW")
        print("="*80)
        print("Stage 1: Broad concentration exploration...")

        lash_e.grab_new_wellplate()  # Get fresh wellplate for stage 1
        
        stage_1_results = execute_adaptive_surfactant_screening(
            surfactant_a_name=surfactant_a_name, 
            surfactant_b_name=surfactant_b_name,
            lash_e=lash_e,
            experiment_output_folder=experiment_output_folder,
            simulate=simulate
        )
        
        lash_e.discard_used_wellplate()  # Discard wellplate after stage 1

        if not stage_1_results['workflow_complete']:
            raise Exception("Stage 1 workflow failed to complete")
        
        print("Stage 1 complete! Analyzing results...")
        
        # STAGE 2: Calculate adaptive bounds and focused exploration
        print("Calculating adaptive concentration bounds from stage 1 data...")
        
        # Hardware mode: always use actual stage 1 results
        stage_1_df = stage_1_results['well_recipes_df']
        experiment_data = stage_1_df[stage_1_df['well_type'] == 'experiment'].copy()
        
        # Calculate new bounds using the rectangle method
        class SimpleLogger:
            def info(self, msg): print(msg)
            def error(self, msg): print(f"ERROR: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
        
        new_bounds = calculate_adaptive_concentration_bounds(
            experiment_data, 
            surfactant_a_name, 
            surfactant_b_name,
            stage_1_results['output_folder'],
            SimpleLogger()  # Simple logger wrapper for print
        )
        
        print(f"New adaptive bounds calculated:")
        print(f"  {surfactant_a_name}: {new_bounds['surf_a_min']:.3e} to {new_bounds['surf_a_max']:.3e} mM")
        print(f"  {surfactant_b_name}: {new_bounds['surf_b_min']:.3e} to {new_bounds['surf_b_max']:.3e} mM")
        
        # Stage 2: Focused exploration with adaptive bounds (reuse same lash_e!)
        print("\nStage 2: Focused exploration with adaptive bounds...")
        
        lash_e.grab_new_wellplate() 

        stage_2_results = execute_adaptive_surfactant_screening(
            surfactant_a_name=surfactant_a_name,
            surfactant_b_name=surfactant_b_name,
            surf_a_min=new_bounds['surf_a_min'],
            surf_a_max=new_bounds['surf_a_max'],
            surf_b_min=new_bounds['surf_b_min'], 
            surf_b_max=new_bounds['surf_b_max'],
            lash_e=lash_e,
            existing_stock_solutions=stage_1_results['experiment_plan']['stock_solutions_needed'],
            experiment_output_folder=experiment_output_folder,
            simulate=simulate
        )

        lash_e.discard_used_wellplate()
        
        if not stage_2_results['workflow_complete']:
            raise Exception("Stage 2 workflow failed to complete")
        
        print("Stage 2 complete!")
        
        # Combine results
        print("\\n" + "="*80)
        print("2-STAGE WORKFLOW COMPLETE!")
        print("="*80)
        print(f"Stage 1: {len(stage_1_results['well_recipes_df'])} wells")
        print(f"Stage 2: {len(stage_2_results['well_recipes_df'])} wells") 
        print(f"Total: {len(stage_1_results['well_recipes_df']) + len(stage_2_results['well_recipes_df'])} wells across both stages")
        
        return {
            'surfactant_a': surfactant_a_name,
            'surfactant_b': surfactant_b_name,
            'stage_1_results': stage_1_results,
            'stage_2_results': stage_2_results,
            'adaptive_bounds': new_bounds,
            'total_wells': len(stage_1_results['well_recipes_df']) + len(stage_2_results['well_recipes_df']),
            'simulation': simulate,
            'workflow_complete': True
        }
        
    except Exception as e:
        print(f"ERROR in 2-stage workflow: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'stage_1_results': stage_1_results,
            'stage_2_results': stage_2_results, 
            'error': str(e),
            'workflow_complete': False
        }

if __name__ == "__main__":
    """
    Run the adaptive surfactant grid screening workflow.
    """
    
    # Initialize Lash_E and validate once
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
    print("Validating robot and track status...")
    lash_e.nr_robot.check_input_file()
    lash_e.nr_track.check_input_file()

    fill_water_vial(lash_e, "water")
    fill_water_vial(lash_e, "water_2")

    # Create unified experiment folder for all workflows
    experiment_output_folder, experiment_name = setup_experiment_environment(
        lash_e, SURFACTANT_A, SURFACTANT_B, SIMULATE
    )
    print(f"Unified experiment folder created: {experiment_output_folder}")

    if VALIDATE_LIQUIDS:
        validate_pipetting_system(lash_e, experiment_output_folder)

    if RUN_2_STAGE_WORKFLOW:
        print("Starting 2-stage adaptive surfactant grid screening...")
        if not SIMULATE:
            slack_agent.send_slack_message("Starting 2-stage adaptive surfactant grid screening workflow...")

        results = execute_2_stage_workflow(
            lash_e=lash_e,
            surfactant_a_name=SURFACTANT_A, 
            surfactant_b_name=SURFACTANT_B, 
            experiment_output_folder=experiment_output_folder,
            simulate=SIMULATE
        )
        
        if results and results['workflow_complete']:
            print("="*80)
            print("2-STAGE WORKFLOW COMPLETE!")
            print("="*80)
            print(f"+ Surfactants: {results['surfactant_a']} + {results['surfactant_b']}")
            print(f"+ Stage 1 wells: {len(results['stage_1_results']['well_recipes_df'])}")
            print(f"+ Stage 2 wells: {len(results['stage_2_results']['well_recipes_df'])}")
            print(f"+ Total wells: {results['total_wells']}")
            print(f"+ Mode: {'SIMULATION' if SIMULATE else 'HARDWARE'}")
        else:
            print("2-stage workflow failed!")
            
    elif RUN_SINGLE_STAGE:
        # SINGLE STAGE WORKFLOW 
        print("Starting single-stage adaptive surfactant grid screening...")
        if not SIMULATE:
            slack_agent.send_slack_message("Starting adaptive surfactant grid screening workflow...")

        results = execute_adaptive_surfactant_screening(
            surfactant_a_name=SURFACTANT_A, 
            surfactant_b_name=SURFACTANT_B,
            lash_e=lash_e,
            experiment_output_folder=experiment_output_folder,
            simulate=SIMULATE
        )
        
        if results and results['workflow_complete']:
            print("="*80)
            print("SINGLE-STAGE WORKFLOW COMPLETE!")
            print("="*80)
            print(f"+ Surfactants: {results['surfactant_a']} + {results['surfactant_b']}")
            print(f"+ Wells: {results['total_wells']}")
            print(f"+ Mode: {'SIMULATION' if SIMULATE else 'HARDWARE'}")
        else:
            print("Single-stage workflow failed!")
    
    elif RUN_ITERATIVE_WORKFLOW:
        # ITERATIVE WORKFLOW WITH GRADIENT EXPLORATION
        print("Starting iterative gradient exploration workflow...")
        if not SIMULATE:
            slack_agent.send_slack_message("Starting iterative gradient exploration workflow...")

        results = execute_iterative_workflow(
            surfactant_a_name=SURFACTANT_A, 
            surfactant_b_name=SURFACTANT_B,
            number_concentrations=5,  # Start with rapid 5x5 screening
            target_measurements=ITERATIVE_MEASUREMENT_TOTAL,   # Fill one wellplate with iterative exploration
            gradient_suggestions_per_iteration=12,
            lash_e=lash_e,
            experiment_output_folder=experiment_output_folder,
            simulate=SIMULATE
        )
        
        if results and results['workflow_complete']:
            print("="*80)
            print("ITERATIVE WORKFLOW COMPLETE!")
            print("="*80)
            print(f"+ Surfactants: {results['surfactant_a']} + {results['surfactant_b']}")
            print(f"+ Total wells: {results['total_wells']}")
            print(f"+ Measured wells: {results['measured_wells']}")
            print(f"+ Mode: {'SIMULATION' if SIMULATE else 'HARDWARE'}")

            print(lash_e.nr_robot.VIAL_DF)  # Debug: Show final vial status after workflow
        else:
            print("Iterative workflow failed!")
    
    else:
        print("No workflow selected. Set RUN_2_STAGE_WORKFLOW=True, RUN_SINGLE_STAGE=True, or RUN_ITERATIVE_WORKFLOW=True")


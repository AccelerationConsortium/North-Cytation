#!/usr/bin/env python3
"""
Time Point Data Visualizer
1. Compares Prep vs Post Shake vs 5 minutes for surfactant kinetics experiments
2. Creates time series plots showing trends over full experiment duration  
3. Generates coefficient of variation analysis for both initial timepoints and full timeseries
4. CMC Analysis: Extracts CMC wells, fits sigmoid curves, determines CMC values
   - CMC vs time plots showing how critical micelle concentration changes
   - Concentration vs ratio plots for CMC determination at each timepoint
   - Supports both SDS and TTAB single-surfactant CMC measurements
Creates individual subplot charts for each well to see differences clearly.
Handles both turbidity and fluorescence data automatically.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

def find_time_point_files(folder_path, measurement_type='turb'):
    """Find the relevant CSV files for the three time points."""
    files = {
        'prep': None,
        'post_shake': None,
        't5min': None
    }
    
    for file in Path(folder_path).glob('*.csv'):
        filename = file.name.lower()
        
        # Skip experiment results summary
        if 'experiment_results' in filename:
            continue
            
        # Only consider files with the specified measurement type
        if measurement_type not in filename:
            continue
            
        # Find prep files (also check for "initial")
        if ('prep_' in filename or 'initial_' in filename) and 'post_shake' not in filename:
            files['prep'] = str(file)
        
        # Find post shake files  
        elif 'post_shake' in filename:
            files['post_shake'] = str(file)
            
        # Find 5 minute timepoint files
        elif 't5min' in filename:
            files['t5min'] = str(file)
    
    return files

def load_well_recipes(base_path):
    """Load experiment plan to understand what's in each well."""
    recipe_file = Path(base_path) / "experiment_plan_well_recipes.csv"
    if recipe_file.exists():
        df = pd.read_csv(recipe_file)
        # Create well labels combining type and concentrations
        well_labels = {}
        for idx, row in df.iterrows():
            well_pos = f"{chr(65 + idx // 12)}{(idx % 12) + 1}"  # Convert index to well position
            
            if row['well_type'] == 'control':
                label = f"{row['control_type']}"
            else:
                sds_conc = row.get('surf_A_conc_mm', 0) if pd.notna(row.get('surf_A_conc_mm')) else 0
                ttab_conc = row.get('surf_B_conc_mm', 0) if pd.notna(row.get('surf_B_conc_mm')) else 0
                label = f"SDS:{sds_conc:.1f}, TTAB:{ttab_conc:.1f}"
            
            well_labels[well_pos] = label
        return well_labels
    return {}

def sigmoid_curve(x, bottom, top, slope, inflection):
    """Sigmoid curve for CMC fitting."""
    return bottom + (top - bottom) / (1 + np.exp(-slope * (x - inflection)))

def find_cmc_from_curve(concentrations, ratios, surfactant_name):
    """Fit sigmoid curve and find CMC (inflection point)."""
    if len(concentrations) < 4 or len(ratios) < 4:
        return np.nan, None
    
    try:
        # Use log scale for better fitting
        log_conc = np.log10(concentrations)
        
        # Initial parameter guesses
        bottom_guess = min(ratios)
        top_guess = max(ratios) 
        slope_guess = -2.0  # Negative for I1/I3 decrease with micelle formation
        inflection_guess = np.median(log_conc)
        
        # Fit the curve
        popt, _ = curve_fit(sigmoid_curve, log_conc, ratios, 
                           p0=[bottom_guess, top_guess, slope_guess, inflection_guess],
                           maxfev=2000)
        
        # CMC is at the inflection point (convert back from log)
        cmc_value = 10**popt[3]
        return cmc_value, popt
        
    except Exception as e:
        print(f"    Warning: CMC fitting failed for {surfactant_name}: {e}")
        return np.nan, None

def extract_cmc_wells(well_recipes):
    """Extract CMC control wells with their concentrations."""
    cmc_wells = {'SDS': {}, 'TTAB': {}}
    
    for well_pos, recipe in well_recipes.items():
        if 'cmc_SDS' in recipe:
            # Extract concentration from recipe - look in the experiment plan
            cmc_wells['SDS'][well_pos] = recipe
        elif 'cmc_TTAB' in recipe:
            cmc_wells['TTAB'][well_pos] = recipe
            
    return cmc_wells

def load_cmc_concentrations(base_path):
    """Load actual CMC concentrations from experiment plan."""
    recipe_file = Path(base_path) / "experiment_plan_well_recipes.csv"
    cmc_data = {'SDS': {}, 'TTAB': {}}
    
    if recipe_file.exists():
        df = pd.read_csv(recipe_file)
        
        for idx, row in df.iterrows():
            well_pos = f"{chr(65 + idx // 12)}{(idx % 12) + 1}"
            
            if row['well_type'] == 'control':
                if 'cmc_SDS' in row['control_type']:
                    conc = row.get('surf_A_conc_mm', 0) if pd.notna(row.get('surf_A_conc_mm')) else 0
                    if conc > 0:
                        cmc_data['SDS'][well_pos] = conc
                        
                elif 'cmc_TTAB' in row['control_type']:
                    conc = row.get('surf_B_conc_mm', 0) if pd.notna(row.get('surf_B_conc_mm')) else 0 
                    if conc > 0:
                        cmc_data['TTAB'][well_pos] = conc
    
    return cmc_data

def create_cmc_analysis(folder_path, folder_name, well_recipes, base_path):
    """Create CMC analysis with ratio vs concentration plots and CMC determination."""
    
    print(f"\n  === CMC ANALYSIS ===")
    
    # Load CMC concentration data
    cmc_concentrations = load_cmc_concentrations(base_path)
    
    if not any(cmc_concentrations.values()):
        print("  No CMC wells found in experiment plan - skipping CMC analysis")
        print("  (CMC analysis requires cmc_SDS and cmc_TTAB control wells)")
        return 0
    
    # Check if we have enough CMC wells for meaningful analysis
    sds_wells = len(cmc_concentrations.get('SDS', {}))
    ttab_wells = len(cmc_concentrations.get('TTAB', {}))
    
    if sds_wells < 4 and ttab_wells < 4:
        print(f"  Insufficient CMC wells for analysis: SDS={sds_wells}, TTAB={ttab_wells}")
        print("  Need at least 4 concentration points for curve fitting - skipping CMC analysis")
        return 0
    
    print(f"  Found CMC wells: SDS={sds_wells}, TTAB={ttab_wells}")
    
    # Check for fluorescence data (CMC analysis typically uses fluorescence)
    measurement_types = ['fluor']
    if not any('fluor' in f.name.lower() for f in Path(folder_path).glob('*.csv')):
        print("  No fluorescence data found - CMC analysis requires fluorescence")
        print("  (Turbidity measurements don't typically include CMC controls)")
        return 0
    
    results = {}   
    for measurement_type in measurement_types:
        # 1. Initial timepoints analysis (prep/post-shake/5min)
        timepoint_files = find_time_point_files(folder_path, measurement_type)
        target_timepoints = ['prep', 'post_shake', 't5min']
        available_timepoints = [tp for tp in target_timepoints if timepoint_files.get(tp) is not None]
        
        if len(available_timepoints) >= 2:
            create_cmc_timepoint_analysis(folder_path, folder_name, cmc_concentrations, 
                                        timepoint_files, available_timepoints, measurement_type)
        
        # 2. Time series analysis 
        timeseries_files = find_timeseries_files(folder_path, measurement_type)
        if len(timeseries_files) >= 3:
            create_cmc_timeseries_analysis(folder_path, folder_name, cmc_concentrations,
                                         timeseries_files, measurement_type)
        
        results[measurement_type] = 1
    
    return len(results)

def create_cmc_timepoint_analysis(folder_path, folder_name, cmc_concentrations, files, timepoints, measurement_type):
    """CMC analysis for initial timepoints (prep/post-shake/5min)."""
    
    print(f"  CMC timepoint analysis for: {timepoints}")
    
    # Load data for each timepoint
    timepoint_data = {}
    for tp in timepoints:
        if files[tp]:
            df = pd.read_csv(files[tp])
            if 'fluorescence_334_373' in df.columns and 'fluorescence_334_384' in df.columns:
                df['fluorescence_ratio'] = df['fluorescence_334_373'] / df['fluorescence_334_384']
                timepoint_data[tp] = df.set_index('well_position')['fluorescence_ratio']
    
    if not timepoint_data:
        return
    
    # Create CMC plots for each surfactant
    for surfactant in ['SDS', 'TTAB']:
        cmc_wells = cmc_concentrations[surfactant]
        if len(cmc_wells) < 4:
            continue
            
        fig, axes = plt.subplots(1, len(timepoints), figsize=(5*len(timepoints), 4))
        if len(timepoints) == 1:
            axes = [axes]
        
        cmc_values = []
        
        for i, tp in enumerate(timepoints):
            ax = axes[i]
            
            # Get concentration and ratio data for this timepoint
            concentrations = []
            ratios = []
            
            for well_pos, conc in cmc_wells.items():
                if well_pos in timepoint_data[tp]:
                    concentrations.append(conc)
                    ratios.append(timepoint_data[tp][well_pos])
            
            if len(concentrations) < 4:
                continue
            
            # Sort by concentration
            sorted_data = sorted(zip(concentrations, ratios))
            concentrations = [x[0] for x in sorted_data]
            ratios = [x[1] for x in sorted_data]
            
            # Plot data points
            ax.semilogx(concentrations, ratios, 'o-', markersize=6, linewidth=2, 
                       label=f'{surfactant} {tp}')
            
            # Fit curve and find CMC
            cmc_value, curve_params = find_cmc_from_curve(concentrations, ratios, f'{surfactant}_{tp}')
            cmc_values.append(cmc_value)
            
            if not np.isnan(cmc_value) and curve_params is not None:
                # Plot fitted curve
                conc_smooth = np.logspace(np.log10(min(concentrations)), 
                                        np.log10(max(concentrations)), 100)
                log_smooth = np.log10(conc_smooth)
                ratio_smooth = sigmoid_curve(log_smooth, *curve_params)
                ax.semilogx(conc_smooth, ratio_smooth, '--', alpha=0.7)
                
                # Mark CMC
                ax.axvline(cmc_value, color='red', linestyle=':', alpha=0.8)
                ax.text(cmc_value, max(ratios)*0.95, f'CMC: {cmc_value:.2f}mM', 
                       rotation=90, ha='right', va='top', fontsize=8)
                
                print(f"    {surfactant} {tp}: CMC = {cmc_value:.2f} mM")
            
            ax.set_xlabel('Concentration (mM)', fontsize=10)
            ax.set_ylabel('Fluorescence Ratio (I1/I3)', fontsize=10)
            ax.set_title(f'{surfactant} CMC - {tp}', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle(f'{surfactant} CMC Analysis: {folder_name.replace("_", " ").title()}\\nTimepoint Comparison', 
                     fontsize=14)
        plt.tight_layout()
        
        # Save plot
        output_path = Path(folder_path).parent / f"cmc_timepoints_{folder_name}_{surfactant.lower()}_{measurement_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path}")

def create_cmc_timeseries_analysis(folder_path, folder_name, cmc_concentrations, files, measurement_type):
    """CMC analysis over full time series."""
    
    print(f"  CMC timeseries analysis: {len(files)} timepoints")
    
    time_points = sorted(files.keys())
    
    # Load data for each timepoint 
    timeseries_data = {}
    for time_min in time_points:
        df = pd.read_csv(files[time_min])
        if 'fluorescence_334_373' in df.columns and 'fluorescence_334_384' in df.columns:
            df['fluorescence_ratio'] = df['fluorescence_334_373'] / df['fluorescence_334_384']
            timeseries_data[time_min] = df.set_index('well_position')['fluorescence_ratio']
    
    if not timeseries_data:
        return
    
    # Analyze each surfactant
    for surfactant in ['SDS', 'TTAB']:
        cmc_wells = cmc_concentrations[surfactant]
        if len(cmc_wells) < 4:
            continue
        
        # Calculate CMC at each timepoint
        cmc_over_time = []
        times = []
        
        for time_min in time_points:
            if time_min not in timeseries_data:
                continue
                
            # Get concentration and ratio data
            concentrations = []
            ratios = []
            
            for well_pos, conc in cmc_wells.items():
                if well_pos in timeseries_data[time_min]:
                    concentrations.append(conc)
                    ratios.append(timeseries_data[time_min][well_pos])
            
            if len(concentrations) >= 4:
                # Sort by concentration
                sorted_data = sorted(zip(concentrations, ratios))
                concentrations = [x[0] for x in sorted_data]
                ratios = [x[1] for x in sorted_data]
                
                cmc_value, _ = find_cmc_from_curve(concentrations, ratios, f'{surfactant}_{time_min}min')
                if not np.isnan(cmc_value):
                    cmc_over_time.append(cmc_value)
                    times.append(time_min)
        
        if len(cmc_over_time) < 2:
            continue
            
        # Create CMC vs time plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Top plot: CMC over time
        ax1.plot(times, cmc_over_time, 'o-', linewidth=2, markersize=6, label=f'{surfactant} CMC')
        ax1.set_xlabel('Time (minutes)', fontsize=12)
        ax1.set_ylabel('CMC (mM)', fontsize=12)
        ax1.set_title(f'{surfactant} CMC vs Time: {folder_name.replace("_", " ").title()}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Bottom plot: Sample concentration curves at different times
        selected_times = [times[0], times[len(times)//2], times[-1]] if len(times) >= 3 else times
        
        for i, time_min in enumerate(selected_times):
            concentrations = []
            ratios = []
            
            for well_pos, conc in cmc_wells.items():
                if well_pos in timeseries_data[time_min]:
                    concentrations.append(conc)
                    ratios.append(timeseries_data[time_min][well_pos])
            
            if len(concentrations) >= 4:
                sorted_data = sorted(zip(concentrations, ratios))
                concentrations = [x[0] for x in sorted_data]
                ratios = [x[1] for x in sorted_data]
                
                ax2.semilogx(concentrations, ratios, 'o-', markersize=4, linewidth=1.5,
                           label=f'{time_min} min', alpha=0.8)
        
        ax2.set_xlabel('Concentration (mM)', fontsize=12)
        ax2.set_ylabel('Fluorescence Ratio (I1/I3)', fontsize=12)
        ax2.set_title(f'{surfactant} Concentration Curves at Different Times', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(folder_path).parent / f"cmc_timeseries_{folder_name}_{surfactant.lower()}_{measurement_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        print(f"    {surfactant} CMC range: {min(cmc_over_time):.2f} - {max(cmc_over_time):.2f} mM")

def find_timeseries_files(folder_path, measurement_type='turb'):
    """Find all timeseries CSV files for time trend analysis (ignore prep/post_shake)."""
    files = {}
    
    for file in Path(folder_path).glob('*.csv'):
        filename = file.name.lower()
        
        # Skip experiment results summary
        if 'experiment_results' in filename:
            continue
            
        # Skip prep and post_shake files - only want timeseries
        if 'prep_' in filename or 'initial_' in filename or 'post_shake' in filename:
            continue
            
        # Only consider timeseries files with the specified measurement type
        if 'timeseries' not in filename or measurement_type not in filename:
            continue
            
        # Extract time point from filename (e.g., t5min, t10min)
        import re
        time_match = re.search(r't(\d+)min', filename)
        if time_match:
            time_minutes = int(time_match.group(1))
            files[time_minutes] = str(file)
    
    return files

def create_timeseries_analysis(folder_path, folder_name, well_recipes):
    """Create time series plots and CV analysis for each well over time (excluding prep/post_shake)."""
    
    # Check for both turbidity and fluorescence data
    measurement_types = []
    if any('turb' in f.name.lower() for f in Path(folder_path).glob('*.csv')):
        measurement_types.append('turb')
    if any('fluor' in f.name.lower() for f in Path(folder_path).glob('*.csv')):
        measurement_types.append('fluor')
    
    results = {}
    
    for measurement_type in measurement_types:
        files = find_timeseries_files(folder_path, measurement_type)
        
        if len(files) < 3:  # Need at least 3 time points for meaningful trend
            print(f"  Skipping {measurement_type} timeseries - need at least 3 time points, found {len(files)}")
            continue
            
        print(f"  {measurement_type} timeseries: Found {len(files)} timepoints: {sorted(files.keys())} minutes")
        
        # Load data for each timepoint
        data = {}
        measurement_column = None
        
        for time_minutes in sorted(files.keys()):
            df = pd.read_csv(files[time_minutes])
            
            # Handle different measurement types correctly
            if 'turb' in measurement_type:
                if 'turbidity_600' in df.columns:
                    measurement_column = 'turbidity_600'
                    data[time_minutes] = df.set_index('well_position')[measurement_column]
                else:
                    continue
                    
            elif 'fluor' in measurement_type:
                if 'fluorescence_334_373' in df.columns and 'fluorescence_334_384' in df.columns:
                    df['fluorescence_ratio'] = df['fluorescence_334_373'] / df['fluorescence_334_384']
                    measurement_column = 'fluorescence_ratio'
                    data[time_minutes] = df.set_index('well_position')[measurement_column]
                else:
                    continue
        
        if not data or not measurement_column:
            print(f"    No valid timeseries data found for {measurement_type}")
            continue
        
        # Get all wells
        all_wells = list(set().union(*[df.index for df in data.values()]))
        all_wells.sort()
        
        # Create time series plots
        n_wells = len(all_wells)
        n_cols = min(6, int(np.ceil(np.sqrt(n_wells))))
        n_rows = int(np.ceil(n_wells / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8*n_cols, 1.5*n_rows))
        fig.suptitle(f'{folder_name.replace("_", " ").title()} - {measurement_type.upper()}\nTime Series by Well', 
                     fontsize=16, y=0.98)
        
        # Flatten axes for easier indexing
        if n_wells == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Calculate CV for each well and plot time series
        well_cvs = []
        well_names = []
        time_points = sorted(data.keys())
        
        for well_idx, well in enumerate(all_wells):
            ax = axes[well_idx]
            
            # Get data for this well across all timepoints
            values = [data[tp].get(well, np.nan) for tp in time_points]
            values = [v for v in values if not np.isnan(v)]  # Remove NaN values
            
            if len(values) >= 3:  # Need at least 3 values for meaningful CV
                # Calculate CV
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = (std_val / mean_val * 100) if mean_val != 0 else 0
                well_cvs.append(cv)
                well_names.append(well)
                
                # Plot time series
                valid_times = [tp for tp, v in zip(time_points, [data[tp].get(well, np.nan) for tp in time_points]) if not np.isnan(v)]
                valid_values = [data[tp].get(well) for tp in valid_times]
                
                ax.plot(valid_times, valid_values, 'o-', linewidth=1, markersize=2, alpha=0.8)
                
                # Customize subplot
                ax.set_title(f'{well}', fontsize=8, weight='bold')
                if well in well_recipes:
                    recipe = well_recipes[well]
                    if len(recipe) > 20:
                        recipe = recipe[:17] + "..."
                    ax.set_title(f'{well}\n{recipe}', fontsize=6)
                
                ax.tick_params(axis='both', labelsize=5)
                ax.grid(alpha=0.3)
                
                # Set appropriate y-axis limits for fluorescence ratios
                if 'fluor' in measurement_type and valid_values:
                    min_val = min(valid_values)
                    max_val = max(valid_values)
                    y_min = max(0.6, min_val - 0.05)  # Don't go below 0.6
                    y_max = max_val + 0.05
                    ax.set_ylim(y_min, y_max)
        
        # Hide empty subplots
        for i in range(len(all_wells), len(axes)):
            axes[i].set_visible(False)
        
        # Add common labels
        if 'turb' in measurement_type:
            y_label = 'Turbidity Ratio (600nm)'
        elif 'fluor' in measurement_type:
            y_label = 'Fluorescence Ratio (I1/I3)'
        else:
            y_label = measurement_type.upper()
            
        fig.text(0.04, 0.5, y_label, va='center', rotation='vertical', fontsize=10)
        fig.text(0.5, 0.02, 'Time (minutes)', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.08, bottom=0.08, top=0.92)
        
        # Save the time series plot
        output_path = Path(folder_path).parent / f"timeseries_{folder_name}_{measurement_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved timeseries: {output_path}")
        
        # Create CV analysis for time series
        if well_cvs:
            fig_cv, ax_cv = plt.subplots(figsize=(12, 6))
            
            bars = ax_cv.bar(range(len(well_names)), well_cvs, 
                            color='lightsteelblue', alpha=0.8, edgecolor='navy')
            
            ax_cv.set_xlabel('Wells', fontsize=12)
            ax_cv.set_ylabel('Coefficient of Variation (%)', fontsize=12)
            
            # Create appropriate title
            if 'turb' in measurement_type:
                measurement_name = 'Turbidity Ratio'
            elif 'fluor' in measurement_type:
                measurement_name = 'Fluorescence Ratio (I1/I3)'
            else:
                measurement_name = measurement_type.upper()
                
            ax_cv.set_title(f'Time Series Variability: {folder_name.replace("_", " ").title()}\n{measurement_name} - Higher CV = More Variable Over Time', 
                           fontsize=14, pad=20)
            
            # Well labels
            display_labels = []
            for well in well_names:
                if well in well_recipes:
                    recipe = well_recipes[well]
                    if len(recipe) > 15:
                        recipe = recipe[:12] + "..."
                    display_labels.append(f"{well}\n{recipe}")
                else:
                    display_labels.append(well)
            
            ax_cv.set_xticks(range(len(well_names)))
            ax_cv.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=8)
            
            # Add CV values on top of bars
            for bar, cv in zip(bars, well_cvs):
                height = bar.get_height()
                ax_cv.text(bar.get_x() + bar.get_width()/2., height,
                          f'{cv:.1f}%', ha='center', va='bottom', fontsize=8)
            
            ax_cv.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save the CV plot
            output_path_cv = Path(folder_path).parent / f"timeseries_cv_{folder_name}_{measurement_type}.png"
            plt.savefig(output_path_cv, dpi=300, bbox_inches='tight')
            print(f"    Saved timeseries CV: {output_path_cv}")
            
            results[f"{measurement_type}_timeseries"] = len(well_cvs)
            results[f"{measurement_type}_cv"] = len(well_cvs)
    
    return len(results)

def create_cv_analysis(folder_path, folder_name, well_recipes):
    
    # Check for both turbidity and fluorescence data
    measurement_types = []
    if any('turb' in f.name.lower() for f in Path(folder_path).glob('*.csv')):
        measurement_types.append('turb')
    if any('fluor' in f.name.lower() for f in Path(folder_path).glob('*.csv')):
        measurement_types.append('fluor')
    
    cv_results = {}
    
    for measurement_type in measurement_types:
        files = find_time_point_files(folder_path, measurement_type)
        
        # Only use the first 3 timepoints for CV analysis
        target_timepoints = ['prep', 'post_shake', 't5min']
        available_timepoints = [tp for tp in target_timepoints if files.get(tp) is not None]
        
        if len(available_timepoints) < 2:
            continue
            
        # Load data for each available timepoint
        data = {}
        measurement_column = None
        
        for timepoint in available_timepoints:
            if files[timepoint]:
                df = pd.read_csv(files[timepoint])
                
                # Handle different measurement types correctly
                if 'turb' in measurement_type:
                    # For turbidity, use the turbidity_600 column (already a ratio)
                    if 'turbidity_600' in df.columns:
                        measurement_column = 'turbidity_600'
                        data[timepoint] = df.set_index('well_position')[measurement_column]
                    else:
                        continue
                        
                elif 'fluor' in measurement_type:
                    # For fluorescence, calculate I1/I3 ratio from the two wavelengths
                    if 'fluorescence_334_373' in df.columns and 'fluorescence_334_384' in df.columns:
                        df['fluorescence_ratio'] = df['fluorescence_334_373'] / df['fluorescence_334_384']
                        measurement_column = 'fluorescence_ratio'
                        data[timepoint] = df.set_index('well_position')[measurement_column]
                    else:
                        continue
        
        if not data or not measurement_column:
            continue
            
        # Get all wells and calculate CV for each
        all_wells = list(set().union(*[df.index for df in data.values()]))
        all_wells.sort()
        
        well_cvs = []
        well_names = []
        
        for well in all_wells:
            # Get values for this well across all timepoints
            values = [data[tp].get(well, np.nan) for tp in available_timepoints]
            values = [v for v in values if not np.isnan(v)]  # Remove NaN values
            
            if len(values) >= 2:  # Need at least 2 values for CV
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = (std_val / mean_val * 100) if mean_val != 0 else 0
                well_cvs.append(cv)
                well_names.append(well)
        
        if well_cvs:
            # Create CV bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            bars = ax.bar(range(len(well_names)), well_cvs, 
                         color='lightsteelblue', alpha=0.8, edgecolor='navy')
            
            ax.set_xlabel('Wells', fontsize=12)
            ax.set_ylabel('Coefficient of Variation (%)', fontsize=12)
            
            # Create appropriate title based on measurement type
            if 'turb' in measurement_type:
                measurement_name = 'Turbidity Ratio'
            elif 'fluor' in measurement_type:
                measurement_name = 'Fluorescence Ratio (I1/I3)'
            else:
                measurement_name = measurement_type.upper()
                
            ax.set_title(f'Time Point Variability: {folder_name.replace("_", " ").title()}\n{measurement_name} - Higher CV = More Variable', 
                         fontsize=14, pad=20)
            
            # Well labels with recipes
            display_labels = []
            for well in well_names:
                if well in well_recipes:
                    recipe = well_recipes[well]
                    if len(recipe) > 15:
                        recipe = recipe[:12] + "..."
                    display_labels.append(f"{well}\n{recipe}")
                else:
                    display_labels.append(well)
            
            ax.set_xticks(range(len(well_names)))
            ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=8)
            
            # Add CV values on top of bars
            for bar, cv in zip(bars, well_cvs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{cv:.1f}%', ha='center', va='bottom', fontsize=8)
            
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            output_path = Path(folder_path).parent / f"cv_analysis_{folder_name}_{measurement_type}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"    Saved CV analysis: {output_path}")
            
            cv_results[measurement_type] = len(well_cvs)
    
    return len(cv_results)
def create_comparison_chart(folder_path, folder_name, well_recipes):
    """Create individual subplot comparison charts for each well."""
    
    # Check for both turbidity and fluorescence data
    measurement_types = []
    if any('turb' in f.name.lower() for f in Path(folder_path).glob('*.csv')):
        measurement_types.append('turb')
    if any('fluor' in f.name.lower() for f in Path(folder_path).glob('*.csv')):
        measurement_types.append('fluor')
    
    print(f"\n{folder_name}: Found measurement types: {measurement_types}")
    
    for measurement_type in measurement_types:
        files = find_time_point_files(folder_path, measurement_type)
        
        # Check what files we found
        available_timepoints = [k for k, v in files.items() if v is not None]
        print(f"  {measurement_type}: Found data for {available_timepoints}")
        
        if len(available_timepoints) < 2:
            print(f"  Skipping {measurement_type} - need at least 2 time points")
            continue
        
        # Load data for each available timepoint
        data = {}
        measurement_column = None
        
        for timepoint in available_timepoints:
            if files[timepoint]:
                df = pd.read_csv(files[timepoint])
                print(f"    {timepoint}: {len(df)} wells, columns: {list(df.columns)}")
                
                # Handle different measurement types correctly
                if 'turb' in measurement_type:
                    # For turbidity, use the turbidity_600 column (already a ratio)
                    if 'turbidity_600' in df.columns:
                        measurement_column = 'turbidity_600'
                        data[timepoint] = df.set_index('well_position')[measurement_column]
                        print(f"    Using turbidity column: {measurement_column}")
                    else:
                        print(f"    Warning: No turbidity_600 column found in {files[timepoint]}")
                        continue
                        
                elif 'fluor' in measurement_type:
                    # For fluorescence, calculate I1/I3 ratio from the two wavelengths
                    if 'fluorescence_334_373' in df.columns and 'fluorescence_334_384' in df.columns:
                        df['fluorescence_ratio'] = df['fluorescence_334_373'] / df['fluorescence_334_384']
                        measurement_column = 'fluorescence_ratio'
                        data[timepoint] = df.set_index('well_position')[measurement_column]
                        print(f"    Calculated fluorescence ratio (I1/I3): 334_373/334_384")
                    else:
                        print(f"    Warning: Missing fluorescence columns in {files[timepoint]}")
                        continue
                
                # Show sample values to verify we have reasonable numbers
                sample_values = data[timepoint].head(3)
                print(f"    Sample values: {list(sample_values.values)}")
        
        if not data or not measurement_column:
            print(f"    No valid data found for {measurement_type}")
            continue
        
        # Get all wells that appear in any timepoint
        all_wells = list(set().union(*[df.index for df in data.values()]))
        all_wells.sort()
        
        # Create subplot grid - make it more compact
        n_wells = len(all_wells)
        n_cols = min(6, int(np.ceil(np.sqrt(n_wells))))
        n_rows = int(np.ceil(n_wells / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8*n_cols, 1.5*n_rows))
        fig.suptitle(f'{folder_name.replace("_", " ").title()} - {measurement_type.upper()}\nTime Point Comparison by Well', 
                     fontsize=16, y=0.98)
        
        # Flatten axes for easier indexing
        if n_wells == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        timepoint_labels = {
            'prep': 'Prep', 
            'post_shake': 'Post Shake', 
            't5min': '5 Minutes'
        }
        
        # Plot each well in its own subplot
        for well_idx, well in enumerate(all_wells):
            ax = axes[well_idx]
            
            # Get data for this well across all timepoints
            timepoints = list(data.keys())
            values = [data[tp].get(well, 0) for tp in timepoints]
            labels = [timepoint_labels.get(tp, tp) for tp in timepoints]
            
            # Create bar chart for this well
            bars = ax.bar(range(len(timepoints)), values, 
                         color=[colors[i] for i in range(len(timepoints))], 
                         alpha=0.8)
            
            # Customize subplot  
            ax.set_title(f'{well}', fontsize=8, weight='bold')
            if well in well_recipes:
                recipe = well_recipes[well]
                if len(recipe) > 20:
                    recipe = recipe[:17] + "..."
                ax.set_title(f'{well}\n{recipe}', fontsize=6)
            
            # Remove x-axis labels for clarity
            ax.set_xticks([])
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='y', labelsize=6)
            
            # Set appropriate y-axis limits for fluorescence ratios
            if 'fluor' in measurement_type:
                # Fluorescence I1/I3 ratios typically range 0.6-2.0
                min_val = min(values) if values else 0.6
                max_val = max(values) if values else 2.0
                y_min = max(0.6, min_val - 0.05)  # Don't go below 0.6
                y_max = max_val + 0.05
                ax.set_ylim(y_min, y_max)
            
            # Remove value labels - they're too cluttered
            # for bar, value in zip(bars, values):
            #     height = bar.get_height()
            #     ax.text(bar.get_x() + bar.get_width()/2., height,
            #            f'{value:.3f}', ha='center', va='bottom', fontsize=5)
        
        # Hide empty subplots
        for i in range(len(all_wells), len(axes)):
            axes[i].set_visible(False)
        
        # Add common y-label
        if 'turb' in measurement_type:
            y_label = 'Turbidity Ratio (600nm)'
        elif 'fluor' in measurement_type:
            y_label = 'Fluorescence Ratio (I1/I3)'
        else:
            y_label = measurement_column.replace("_", " ").title()
            
        fig.text(0.04, 0.5, y_label, va='center', rotation='vertical', fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.08, top=0.92)
        
        # Save the plot
        output_path = Path(folder_path).parent / f"well_comparison_{folder_name}_{measurement_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path}")
    
    return len(measurement_types)

def main():
    """Main function to process all subfolders."""
    
    # Base directory 
    base_path = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\surfactant_grid_SDS_TTAB_20260226_180227_kinetics_thursday_overnight"
    
    if not Path(base_path).exists():
        print(f"Error: Directory not found: {base_path}")
        return
    
    # Load well recipes for better labeling
    well_recipes = load_well_recipes(base_path)
    print(f"Loaded recipes for {len(well_recipes)} wells")
    
    # Focus on prep_turbidity subfolder
    subfolders = [
        "prep_turbidity"
    ]
    
    created_plots = []
    total_charts = 0
    total_cv_charts = 0
    
    for subfolder in subfolders:
        subfolder_path = Path(base_path) / subfolder
        
        if subfolder_path.exists():
            print(f"\nProcessing {subfolder}...")
            n_charts = create_comparison_chart(subfolder_path, subfolder, well_recipes)
            n_cv_charts = create_cv_analysis(subfolder_path, subfolder, well_recipes)
            n_timeseries_charts = create_timeseries_analysis(subfolder_path, subfolder, well_recipes)
            n_cmc_charts = create_cmc_analysis(subfolder_path, subfolder, well_recipes, base_path)
            
            if n_charts and n_charts > 0:
                created_plots.append(subfolder)
                total_charts += n_charts
                total_cv_charts += n_cv_charts
                total_cv_charts += n_timeseries_charts
                total_cv_charts += n_cmc_charts
        else:
            print(f"Subfolder not found: {subfolder}")
    
    print(f"\n=== COMPLETED ===")
    print(f"Processed {len(created_plots)} folders:")
    print(f"  - Created {total_charts} time-point comparison charts (prep/post-shake/5min)")
    print(f"  - Created {total_cv_charts} analysis charts (CV + timeseries + CMC)")
    for plot in created_plots:
        print(f"  - {plot}")
    print(f"\nCMC Analysis includes:")
    print(f"  - Ratio vs concentration plots with curve fitting")
    print(f"  - CMC determination for each timepoint")
    print(f"  - CMC evolution over time")
    
    if created_plots:
        print(f"\nPNG files saved in: {base_path}")
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()
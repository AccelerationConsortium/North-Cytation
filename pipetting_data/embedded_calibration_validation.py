# embedded_calibration_validation.py
"""
Lightweight Pipetting Validation for Integration with Other Workflows

This module provides validation functions for different pipetting scenarios:
1. Vial-to-vial pipetting validation
2. Reservoir-to-vial dispensing validation

Key Features:
- Embeddable in other workflows
- Uses existing lash_e instance
- Minimal configuration required
- Automatic density lookup and calculations
- Generates validation plots and statistics
- Results saved to organized output structure

Usage Examples:
    from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy, validate_reservoir_accuracy
    
    # Validate vial-to-vial pipetting
    results = validate_pipetting_accuracy(
        lash_e=my_lash_e_instance,
        source_vial="water_stock",
        destination_vial="measurement_vial", 
        liquid_type="water",
        volumes_ml=[0.01, 0.02, 0.05, 0.1],
        replicates=3
    )
    
    # Validate reservoir dispensing
    results = validate_reservoir_accuracy(
        lash_e=my_lash_e_instance,
        reservoir_index=1,
        target_vial="validation_vial",
        liquid_type="water", 
        volumes_ml=[0.5, 1.0, 2.0],
        replicates=3
    )
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

# Add path for imports if needed
if "../utoronto_demo" not in sys.path:
    sys.path.append("../utoronto_demo")

# Liquid density database - expandable as needed
LIQUID_DENSITIES = {
    "water": 1.00,
    "ethanol": 0.789,
    "toluene": 0.867,
    "2MeTHF": 0.852,
    "isopropanol": 0.789,
    "DMSO": 1.1,
    "acetone": 0.79,
    "glycerol": 1.26,
    "PEG_Water": 1.05,
    "4%_hyaluronic_acid_water": 1.01,
}

def validate_pipetting_accuracy(
    lash_e,
    source_vial: str,
    destination_vial: str,
    liquid_type: str,
    volumes_ml: List[float],
    replicates: int = 3,
    output_folder: str = "output",
    plot_title: Optional[str] = None,
    save_raw_data: bool = True,
    switch_pipet: bool = False,
    compensate_overvolume: bool = True,
    smooth_overvolume: bool = False
) -> Dict:
    """
    Validate pipetting accuracy for specified volumes and generate analysis.
    
    Args:
        lash_e: Lash_E coordinator instance (must be initialized)
        source_vial: Name of source vial (must exist in robot status)
        destination_vial: Name of destination vial (must be in clamp position)
        liquid_type: Type of liquid for density calculation (e.g., 'water', 'glycerol')
        volumes_ml: List of volumes to test in mL (e.g., [0.01, 0.02, 0.05])
        replicates: Number of replicates per volume (default: 3)
        output_folder: Base output directory (creates calibration_validation subfolder)
        plot_title: Optional custom title for plots
        save_raw_data: Whether to save raw measurement data (default: True)
        switch_pipet: Whether to change pipet tip between measurements (default: False)
        compensate_overvolume: Apply overvolume compensation based on measured accuracy (default: True)
        smooth_overvolume: Apply local smoothing to remove overvolume outliers (default: False)
        
    Returns:
        dict: Results summary containing accuracy metrics, file paths, and statistics
        
    Raises:
        ValueError: If liquid_type not in LIQUID_DENSITIES
        FileNotFoundError: If vials don't exist in robot status
    """
    
    # === VALIDATION ===
    if liquid_type not in LIQUID_DENSITIES:
        available = list(LIQUID_DENSITIES.keys())
        raise ValueError(f"Unknown liquid_type '{liquid_type}'. Available: {available}")
    
    density = LIQUID_DENSITIES[liquid_type]
    
    # Validate vials exist
    try:
        lash_e.nr_robot.get_vial_info(source_vial, 'location')
        lash_e.nr_robot.get_vial_info(destination_vial, 'location')
    except Exception as e:
        raise FileNotFoundError(f"Vial validation failed: {e}")
    
    # === SETUP OUTPUT DIRECTORY ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calib_folder = os.path.join(output_folder, "calibration_validation", f"validation_{timestamp}")
    os.makedirs(calib_folder, exist_ok=True)
    
    print(f"\\n=== Pipetting Validation ===")
    print(f"Source: {source_vial}")
    print(f"Destination: {destination_vial}")
    print(f"Liquid: {liquid_type} (density: {density} g/mL)")
    print(f"Volumes: {volumes_ml} mL")
    print(f"Replicates: {replicates}")
    print(f"Switch pipet: {switch_pipet}")
    print(f"Output: {calib_folder}")
    
    # === DATA COLLECTION ===
    all_results = []
    
    # Pre-position destination vial in clamp before starting pipetting
    lash_e.nr_robot.move_vial_to_location(destination_vial, "clamp", 0)

    for volume_ml in volumes_ml:
        print(f"\\nTesting volume: {volume_ml:.3f} mL ({replicates} replicates)")
        
        for rep in range(replicates):
            print(f"  Replicate {rep + 1}/{replicates}...")
            
            # Pipette using wizard (automatically optimizes parameters)
            mass_difference = lash_e.nr_robot.dispense_from_vial_into_vial(
                source_vial_name=source_vial,
                dest_vial_name=destination_vial,
                volume=volume_ml,
                liquid=liquid_type,
                remove_tip=switch_pipet,  # Use built-in pipet removal control
                use_safe_location=False,
                return_vial_home=False,   # Keep destination vial in clamp for mass measurement
                compensate_overvolume=compensate_overvolume,
                smooth_overvolume=smooth_overvolume
            )
                       

            measured_volume_ml = mass_difference / density  # mass(g) / density(g/mL) = volume(mL)
            
            # Store result
            result = {
                'target_volume_ml': volume_ml,
                'measured_volume_ml': measured_volume_ml,
                'mass_difference_g': mass_difference,
                'density_used': density,
                'liquid_type': liquid_type,
                'replicate': rep + 1,
                'timestamp': datetime.now().isoformat(),
                'source_vial': source_vial,
                'destination_vial': destination_vial,
                'switch_pipet': switch_pipet
            }
            all_results.append(result)
            
            print(f"    Target: {volume_ml*1000:.1f} µL → Measured: {measured_volume_ml*1000:.1f} µL (Error: {((measured_volume_ml-volume_ml)/volume_ml)*100:.1f}%)")
    
    # === DATA ANALYSIS ===
    df = pd.DataFrame(all_results)
    
    # Calculate statistics per volume
    stats_per_volume = []
    for volume in volumes_ml:
        subset = df[df['target_volume_ml'] == volume]
        
        measured_mean = subset['measured_volume_ml'].mean()
        measured_std = subset['measured_volume_ml'].std()
        
        # Handle single replicate case (std returns NaN for n=1)
        if len(subset) == 1:
            measured_std = 0.0  # No variation with single measurement
        
        # Calculate metrics
        accuracy_pct = ((measured_mean - volume) / volume) * 100
        precision_cv_pct = (measured_std / measured_mean) * 100 if measured_mean > 0 and measured_std > 0 else 0
        
        stats = {
            'target_volume_ml': volume,
            'target_volume_ul': volume * 1000,
            'measured_mean_ml': measured_mean,
            'measured_mean_ul': measured_mean * 1000,
            'measured_std_ml': measured_std,
            'measured_std_ul': measured_std * 1000,
            'accuracy_pct': accuracy_pct,
            'precision_cv_pct': precision_cv_pct,
            'n_replicates': len(subset)
        }
        stats_per_volume.append(stats)
    
    stats_df = pd.DataFrame(stats_per_volume)
    
    # === OVERALL STATISTICS ===
    # Linear regression (handle single volume case)
    from scipy import stats as scipy_stats
    
    if len(volumes_ml) == 1:
        # Single volume case - can't do regression, but can report accuracy
        slope, intercept, r_squared, p_value, std_err = 1.0, 0.0, 1.0, 0.0, 0.0
        print("Single volume validation - linear regression not applicable")
    else:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            df['target_volume_ml'], df['measured_volume_ml']
        )
        r_squared = r_value ** 2
    
    overall_stats = {
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'p_value': p_value,
        'std_error': std_err,
        'mean_accuracy_pct': stats_df['accuracy_pct'].mean(),
        'mean_precision_cv_pct': stats_df['precision_cv_pct'].mean(),
        'total_measurements': len(df)
    }
    
    # === GENERATE PLOTS ===
    plot_file = _create_validation_plots(df, stats_df, overall_stats, calib_folder, plot_title, liquid_type)
    
    # === SAVE DATA ===
    data_files = {}
    
    if save_raw_data:
        # Raw data
        raw_file = os.path.join(calib_folder, "raw_measurements.csv")
        df.to_csv(raw_file, index=False)
        data_files['raw_data'] = raw_file
        
        # Statistics summary
        stats_file = os.path.join(calib_folder, "volume_statistics.csv")
        stats_df.to_csv(stats_file, index=False)
        data_files['volume_stats'] = stats_file
        
        # Overall summary
        summary_file = os.path.join(calib_folder, "validation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        data_files['summary'] = summary_file
    
    # === PREPARE RETURN DATA ===
    results = {
        'success': True,
        'r_squared': r_squared,
        'mean_accuracy_pct': overall_stats['mean_accuracy_pct'],
        'mean_precision_cv_pct': overall_stats['mean_precision_cv_pct'],
        'linear_fit': {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared
        },
        'volume_statistics': stats_df.to_dict('records'),
        'files': {
            'plot': plot_file,
            **data_files
        },
        'output_folder': calib_folder,
        'metadata': {
            'liquid_type': liquid_type,
            'density_used': density,
            'volumes_tested': volumes_ml,
            'replicates': replicates,
            'total_measurements': len(df),
            'source_vial': source_vial,
            'destination_vial': destination_vial,
            'switch_pipet': switch_pipet,
            'timestamp': timestamp
        }
    }
    
    # === CLEANUP ===
    # Return destination vial to home position at the very end
    try:
        lash_e.nr_robot.remove_pipet()  # Ensure pipet is removed first
        lash_e.nr_robot.return_vial_home(destination_vial)  # Return by vial name
        print(f"Returned {destination_vial} to home position")
    except Exception as e:
        print(f"Warning: Could not return vial to home: {e}")
    
    # === PRINT SUMMARY ===
    print(f"\\n=== Validation Results ===")
    print(f"R² = {r_squared:.4f}")
    print(f"Mean accuracy: {overall_stats['mean_accuracy_pct']:.2f}%")
    print(f"Mean precision (CV): {overall_stats['mean_precision_cv_pct']:.2f}%")
    print(f"Linear fit: y = {slope:.4f}x + {intercept:.6f}")
    print(f"Results saved to: {calib_folder}")
    
    return results

def validate_reservoir_accuracy(
    lash_e,
    reservoir_index: int,
    target_vial: str,
    liquid_type: str,
    volumes_ml: List[float],
    replicates: int = 3,
    output_folder: str = "output",
    plot_title: Optional[str] = None,
    save_raw_data: bool = True
) -> Dict:
    """
    Validate reservoir dispensing accuracy for specified volumes and generate analysis.
    
    Args:
        lash_e: Lash_E coordinator instance (must be initialized)
        reservoir_index: Reservoir index (e.g., 1 for water reservoir)
        target_vial: Name of target vial (must be in clamp position)
        liquid_type: Type of liquid for density calculation (e.g., 'water', 'buffer')
        volumes_ml: List of volumes to test in mL (e.g., [0.5, 1.0, 2.0])
        replicates: Number of replicates per volume (default: 3)
        output_folder: Base output directory (creates reservoir_validation subfolder)
        plot_title: Optional custom title for plots
        save_raw_data: Whether to save raw measurement data (default: True)
        
    Returns:
        dict: Results summary containing accuracy metrics, file paths, and statistics
        
    Raises:
        ValueError: If liquid_type not in LIQUID_DENSITIES
        FileNotFoundError: If target vial doesn't exist in robot status
    """
    
    # === VALIDATION ===
    if liquid_type not in LIQUID_DENSITIES:
        available = list(LIQUID_DENSITIES.keys())
        raise ValueError(f"Unknown liquid_type '{liquid_type}'. Available: {available}")
    
    density = LIQUID_DENSITIES[liquid_type]
    
    # Validate target vial exists
    try:
        lash_e.nr_robot.get_vial_info(target_vial, 'location')
    except Exception as e:
        raise FileNotFoundError(f"Target vial validation failed: {e}")
    
    # === SETUP OUTPUT DIRECTORY ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calib_folder = os.path.join(output_folder, "reservoir_validation", f"validation_{timestamp}")
    os.makedirs(calib_folder, exist_ok=True)
    
    print(f"\\n=== Reservoir Validation ===")
    print(f"Reservoir index: {reservoir_index}")
    print(f"Target vial: {target_vial}")
    print(f"Liquid: {liquid_type} (density: {density} g/mL)")
    print(f"Volumes: {volumes_ml} mL")
    print(f"Replicates: {replicates}")
    print(f"Output: {calib_folder}")
    
    # === DATA COLLECTION ===
    all_results = []
    
    for volume_ml in volumes_ml:
        print(f"\\nTesting volume: {volume_ml:.3f} mL ({replicates} replicates)")
        
        for rep in range(replicates):
            print(f"  Replicate {rep + 1}/{replicates}...")
            
            # Get initial mass
            if not lash_e.simulate:
                lash_e.nr_robot.c9.zero_scale()  # Zero scale before measurement
                initial_mass = lash_e.nr_robot.c9.read_steady_scale()
            else:
                initial_mass = 0.0
            
            # Dispense from reservoir
            lash_e.nr_robot.dispense_into_vial_from_reservoir(
                reservoir_index=reservoir_index,
                vial_index=target_vial,
                volume=volume_ml,
                measure_weight=False  # We handle weight measurement ourselves
            )
            
            # Wait for stabilization (only in hardware mode)
            if not lash_e.simulate:
                time.sleep(0.5)
            
            # Get final mass
            if not lash_e.simulate:
                final_mass = lash_e.nr_robot.c9.read_steady_scale()
            else:
                final_mass = initial_mass + (volume_ml * density)  # Simulate perfect transfer
            
            # Calculate transferred volume
            mass_difference = final_mass - initial_mass
            measured_volume_ml = mass_difference / density  # mass(g) / density(g/mL) = volume(mL)
            
            # Store result
            result = {
                'target_volume_ml': volume_ml,
                'measured_volume_ml': measured_volume_ml,
                'mass_difference_g': mass_difference,
                'density_used': density,
                'liquid_type': liquid_type,
                'replicate': rep + 1,
                'initial_mass': initial_mass,
                'final_mass': final_mass,
                'timestamp': datetime.now().isoformat(),
                'reservoir_index': reservoir_index,
                'target_vial': target_vial
            }
            all_results.append(result)
            
            print(f"    Target: {volume_ml*1000:.1f} µL → Measured: {measured_volume_ml*1000:.1f} µL (Error: {((measured_volume_ml-volume_ml)/volume_ml)*100:.1f}%)")
    
    # === DATA ANALYSIS ===
    df = pd.DataFrame(all_results)
    
    # Calculate statistics per volume
    stats_per_volume = []
    for volume in volumes_ml:
        subset = df[df['target_volume_ml'] == volume]
        
        measured_mean = subset['measured_volume_ml'].mean()
        measured_std = subset['measured_volume_ml'].std()
        
        # Handle single replicate case (std returns NaN for n=1)
        if len(subset) == 1:
            measured_std = 0.0  # No variation with single measurement
        
        # Calculate metrics
        accuracy_pct = ((measured_mean - volume) / volume) * 100
        precision_cv_pct = (measured_std / measured_mean) * 100 if measured_mean > 0 and measured_std > 0 else 0
        
        stats = {
            'target_volume_ml': volume,
            'target_volume_ul': volume * 1000,
            'measured_mean_ml': measured_mean,
            'measured_mean_ul': measured_mean * 1000,
            'measured_std_ml': measured_std,
            'measured_std_ul': measured_std * 1000,
            'accuracy_pct': accuracy_pct,
            'precision_cv_pct': precision_cv_pct,
            'n_replicates': len(subset)
        }
        stats_per_volume.append(stats)
    
    stats_df = pd.DataFrame(stats_per_volume)
    
    # === OVERALL STATISTICS ===
    # Linear regression (handle single volume case)
    from scipy import stats as scipy_stats
    
    if len(volumes_ml) == 1:
        # Single volume case - can't do regression, but can report accuracy
        slope, intercept, r_squared, p_value, std_err = 1.0, 0.0, 1.0, 0.0, 0.0
        print("Single volume validation - linear regression not applicable")
    else:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            df['target_volume_ml'], df['measured_volume_ml']
        )
        r_squared = r_value ** 2
    
    overall_stats = {
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'p_value': p_value,
        'std_error': std_err,
        'mean_accuracy_pct': stats_df['accuracy_pct'].mean(),
        'mean_precision_cv_pct': stats_df['precision_cv_pct'].mean(),
        'total_measurements': len(df)
    }
    
    # === GENERATE PLOTS ===
    plot_file = _create_validation_plots(df, stats_df, overall_stats, calib_folder, plot_title, liquid_type)
    
    # === SAVE DATA ===
    data_files = {}
    
    if save_raw_data:
        # Raw data
        raw_file = os.path.join(calib_folder, "raw_measurements.csv")
        df.to_csv(raw_file, index=False)
        data_files['raw_data'] = raw_file
        
        # Statistics summary
        stats_file = os.path.join(calib_folder, "volume_statistics.csv")
        stats_df.to_csv(stats_file, index=False)
        data_files['volume_stats'] = stats_file
        
        # Overall summary
        summary_file = os.path.join(calib_folder, "validation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        data_files['summary'] = summary_file
    
    # === PREPARE RETURN DATA ===
    results = {
        'success': True,
        'r_squared': r_squared,
        'mean_accuracy_pct': overall_stats['mean_accuracy_pct'],
        'mean_precision_cv_pct': overall_stats['mean_precision_cv_pct'],
        'linear_fit': {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared
        },
        'volume_statistics': stats_df.to_dict('records'),
        'files': {
            'plot': plot_file,
            **data_files
        },
        'output_folder': calib_folder,
        'metadata': {
            'liquid_type': liquid_type,
            'density_used': density,
            'volumes_tested': volumes_ml,
            'replicates': replicates,
            'total_measurements': len(df),
            'reservoir_index': reservoir_index,
            'target_vial': target_vial,
            'timestamp': timestamp
        }
    }
    
    # === CLEANUP ===
    # Return target vial to home position at the very end
    try:
        clamp_vial_index = lash_e.nr_robot.get_vial_in_location('clamp', 0)
        if clamp_vial_index is not None:
            lash_e.nr_robot.return_vial_home(clamp_vial_index)
            print(f"Returned {target_vial} to home position")
    except Exception as e:
        print(f"Warning: Could not return vial to home: {e}")
    
    # === PRINT SUMMARY ===
    print(f"\\n=== Reservoir Validation Results ===")
    print(f"R² = {r_squared:.4f}")
    print(f"Mean accuracy: {overall_stats['mean_accuracy_pct']:.2f}%")
    print(f"Mean precision (CV): {overall_stats['mean_precision_cv_pct']:.2f}%")
    print(f"Linear fit: y = {slope:.4f}x + {intercept:.6f}")
    print(f"Results saved to: {calib_folder}")
    
    return results

def _create_validation_plots(df: pd.DataFrame, stats_df: pd.DataFrame, overall_stats: dict, 
                           output_folder: str, plot_title: Optional[str], liquid_type: str) -> str:
    """Create validation plots and save to file."""
    
    # Set up the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    if plot_title:
        fig.suptitle(f"{plot_title} - {liquid_type.title()} Validation", fontsize=16)
    else:
        fig.suptitle(f"Pipetting Validation - {liquid_type.title()}", fontsize=16)
    
    # Convert to µL for plotting
    df['target_ul'] = df['target_volume_ml'] * 1000
    df['measured_ul'] = df['measured_volume_ml'] * 1000
    
    # Plot 1: Target vs Measured with perfect line
    ax1.scatter(df['target_ul'], df['measured_ul'], alpha=0.7, s=50)
    
    # Add perfect line
    min_vol = min(df['target_ul'].min(), df['measured_ul'].min())
    max_vol = max(df['target_ul'].max(), df['measured_ul'].max())
    ax1.plot([min_vol, max_vol], [min_vol, max_vol], 'r--', label='Perfect accuracy', linewidth=2)
    
    # Add regression line
    slope, intercept = overall_stats['slope'], overall_stats['intercept']
    x_fit = np.array([min_vol, max_vol]) / 1000  # Convert back to mL for calculation
    y_fit = (slope * x_fit + intercept) * 1000  # Convert result to µL
    ax1.plot(x_fit * 1000, y_fit, 'b-', label=f'Linear fit (R² = {overall_stats["r_squared"]:.4f})', linewidth=2)
    
    ax1.set_xlabel('Target Volume (µL)')
    ax1.set_ylabel('Measured Volume (µL)')
    ax1.set_title('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy per volume
    ax2.bar(stats_df['target_volume_ul'], stats_df['accuracy_pct'])
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Target Volume (µL)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy by Volume')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Precision (CV) per volume
    ax3.bar(stats_df['target_volume_ul'], stats_df['precision_cv_pct'])
    ax3.set_xlabel('Target Volume (µL)')
    ax3.set_ylabel('Precision CV (%)')
    ax3.set_title('Precision by Volume')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    df['error_pct'] = ((df['measured_volume_ml'] - df['target_volume_ml']) / df['target_volume_ml']) * 100
    ax4.hist(df['error_pct'], bins=max(10, len(df)//3), alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Error (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_folder, "validation_plots.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file
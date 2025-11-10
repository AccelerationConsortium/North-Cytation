# calibration_validation.py
"""
Calibration Validation Workflow

Tests intelligent parameter optimization across specified volume ranges to validate 
calibration accuracy. Uses the integrated parameter system in North_Safe for 
automatic optimization based on liquid type and volume.

Workflow:
1. Specify liquid type to enable automatic parameter optimization
2. Test specified volumes with replicates using intelligent parameter system
3. Measure mass and convert to volume using liquid density  
4. Calculate accuracy and precision metrics
5. Generate validation report and graphs

The robot automatically optimizes parameters using: defaults → liquid-calibrated → user overrides
"""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add paths for imports
sys.path.append("../North-Cytation")

# Import base functionality
from calibration_sdl_base import (
    pipet_and_measure, LIQUIDS, set_vial_management, 
    normalize_parameters, empty_vial_if_needed, fill_liquid_if_needed
)
from master_usdl_coordinator import Lash_E

# Pipetting wizard integration now handled automatically by North_Safe parameter system

# --- CONFIGURATION ---
DEFAULT_LIQUID = "glycerol"
DEFAULT_SIMULATE = False
DEFAULT_VOLUMES = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15]  # mL
DEFAULT_REPLICATES = 3
DEFAULT_INPUT_VIAL_STATUS_FILE = "status/calibration_vials_short.csv"

# Vial management mode - set to match your calibration setup
# Options: "legacy" (no vial management), "single", "dual", etc.
VIAL_MANAGEMENT_MODE = "legacy"  # Change this to match calibration setup

def initialize_experiment(lash_e, liquid):
    """Initialize experiment with proper vial setup"""
    print(f"🔧 Initializing experiment for {liquid} validation...")
    
    # Set vial management mode (configurable at top of file)
    if VIAL_MANAGEMENT_MODE != "legacy":
        set_vial_management(mode=VIAL_MANAGEMENT_MODE)
        print(f"   🧪 Vial management: {VIAL_MANAGEMENT_MODE}")
    else:
        print(f"   🧪 Vial management: legacy (no vial management)")
    
    # Ensure measurement vial is in clamp position
    lash_e.nr_robot.move_vial_to_location("measurement_vial_0", "clamp", 0)
    
    print("✅ Experiment initialized")

def validate_volumes(lash_e, liquid, volumes, replicates, simulate):
    """
    Validate pipetting accuracy across specified volumes using intelligent parameter optimization
    
    Args:
        lash_e: Lash_E coordinator instance
        liquid: Liquid type for density calculations and automatic parameter optimization
        volumes: List of volumes to test (mL)
        replicates: Number of replicates per volume
        simulate: Simulation mode flag
    
    Returns:
        results_df: Summary results per volume
        raw_df: Raw measurement data
    """
    print(f"🧪 Starting validation for {len(volumes)} volumes with {replicates} replicates each...")
    
    # Get liquid properties
    if liquid not in LIQUIDS:
        raise ValueError(f"Unknown liquid '{liquid}' - must be one of: {list(LIQUIDS.keys())}")
    liquid_density = LIQUIDS[liquid]["density"]
    
    # Get liquid-specific pipet behavior (CRITICAL: matches calibration setup)
    new_pipet_each_time_set = LIQUIDS[liquid]["refill_pipets"]
    print(f"   🔧 Pipet behavior for {liquid}: {'new tip each time' if new_pipet_each_time_set else 'reuse tip'}")
    
    # Initialize results storage
    results = []
    raw_measurements = []
    
    # Setup file paths for raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "validation_runs" / f"validation_{liquid}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_validation_data.csv"
    
    # Vial configuration based on vial management mode
    if VIAL_MANAGEMENT_MODE == "legacy":
        # Legacy mode: separate source and destination vials
        source_vial = "liquid_source_0"      # Liquid reservoir
        dest_vial = "measurement_vial_0"     # Measurement vial 
        print(f"   🧪 Legacy mode: {source_vial} → {dest_vial}")
    else:
        # Single vial mode: same vial for source and destination
        source_vial = "measurement_vial_0"   # Same vial
        dest_vial = "measurement_vial_0"     # Same vial
        print(f"   🧪 Single vial mode: {source_vial} (same vial)")
    
    # Process each volume
    for i, volume in enumerate(volumes):
        print(f"\n📏 Testing volume {i+1}/{len(volumes)}: {volume*1000:.1f} μL")
        
        try:
            # Use intelligent parameter system - no manual wizard lookup needed!
            # The robot methods will automatically optimize parameters based on liquid and volume
            print(f"   Using intelligent parameter optimization for {liquid} at {volume*1000:.1f} μL")
            
            # Calculate expected mass
            expected_mass = volume * liquid_density
            
            # Perform measurements
            volume_measurements = []
            mass_measurements = []
            
            for rep in range(replicates):
                print(f"   Replicate {rep+1}/{replicates}...", end="")
                
                # Measure using pipet_and_measure function
                # We'll collect the raw data directly from the global raw_measurements list
                raw_count_before = len(raw_measurements)
                
                result = pipet_and_measure(
                    lash_e=lash_e,
                    source_vial=source_vial,
                    dest_vial=dest_vial,
                    volume=volume,
                    params=None,  # Let intelligent parameter system optimize automatically
                    expected_measurement=expected_mass,
                    expected_time=30.0,  # Placeholder
                    replicate_count=1,  # Single measurement
                    simulate=simulate,
                    raw_path=str(raw_path),
                    raw_measurements=raw_measurements,
                    liquid=liquid,  # CRITICAL: This enables automatic parameter optimization!
                    new_pipet_each_time=new_pipet_each_time_set,  # Use liquid-specific setting
                    trial_type="VALIDATION"
                )
                
                # Extract the measurement from the raw data
                latest_measurement = raw_measurements[-1]  # Last added measurement
                measured_mass = latest_measurement["mass"]
                measured_volume = latest_measurement["calculated_volume"]
                
                volume_measurements.append(measured_volume)
                mass_measurements.append(measured_mass)
                
                print(f" {measured_volume*1000:.2f} μL")
            
            # Calculate metrics for this volume
            target_volume = volume
            mean_volume = np.mean(volume_measurements)
            std_volume = np.std(volume_measurements)
            cv_percent = (std_volume / mean_volume * 100) if mean_volume > 0 else 0
            
            # Accuracy metrics
            accuracy_percent = (mean_volume - target_volume) / target_volume * 100
            absolute_accuracy = abs(accuracy_percent)
            
            # Individual errors
            individual_errors = [abs(v - target_volume) / target_volume * 100 for v in volume_measurements]
            mean_absolute_error = np.mean(individual_errors)
            
            # Store results
            volume_result = {
                'volume_target_ml': target_volume,
                'volume_target_ul': target_volume * 1000,
                'volume_measured_mean_ml': mean_volume,
                'volume_measured_mean_ul': mean_volume * 1000,
                'volume_measured_std_ml': std_volume,
                'volume_measured_std_ul': std_volume * 1000,
                'cv_percent': cv_percent,
                'accuracy_percent': accuracy_percent,
                'absolute_accuracy_percent': absolute_accuracy,
                'mean_absolute_error_percent': mean_absolute_error,
                'replicates': replicates,
                'liquid': liquid,
                'parameter_source': 'intelligent_optimization'  # Parameters automatically optimized
            }
            results.append(volume_result)
            
            print(f"   ✅ Mean: {mean_volume*1000:.2f} μL, "
                  f"Accuracy: {accuracy_percent:+.1f}%, "
                  f"CV: {cv_percent:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            # Add failed entry to maintain structure
            volume_result = {
                'volume_target_ml': volume,
                'volume_target_ul': volume * 1000,
                'volume_measured_mean_ml': np.nan,
                'volume_measured_mean_ul': np.nan,
                'volume_measured_std_ml': np.nan,
                'volume_measured_std_ul': np.nan,
                'cv_percent': np.nan,
                'accuracy_percent': np.nan,
                'absolute_accuracy_percent': np.nan,
                'mean_absolute_error_percent': np.nan,
                'replicates': replicates,
                'liquid': liquid,
                'error': str(e)
            }
            results.append(volume_result)
    
    # Convert to DataFrames
    results_df = pd.DataFrame(results)
    raw_df = pd.DataFrame(raw_measurements)
    
    # Save results
    results_df.to_csv(output_dir / "validation_summary.csv", index=False)
    raw_df.to_csv(output_dir / "raw_validation_data.csv", index=False)
    
    print(f"\n💾 Results saved to: {output_dir}")
    
    return results_df, raw_df, output_dir

def generate_validation_report(results_df, raw_df, output_dir, liquid):
    """Generate validation analysis and plots"""
    print(f"\n📊 Generating validation report...")
    
    # Filter out failed measurements for analysis
    valid_results = results_df.dropna(subset=['accuracy_percent'])
    
    if len(valid_results) == 0:
        print("❌ No valid results for analysis")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{liquid.capitalize()} Calibration Validation Report', fontsize=16)
    
    # Plot 1: Target vs Measured Volume (Scatter)
    ax1 = axes[0, 0]
    target_ul = valid_results['volume_target_ul']
    measured_ul = valid_results['volume_measured_mean_ul']
    measured_std_ul = valid_results['volume_measured_std_ul']
    
    # Scatter plot with error bars
    ax1.errorbar(target_ul, measured_ul, yerr=measured_std_ul, 
                 fmt='o', capsize=5, capthick=2, markersize=8, color='blue', alpha=0.7)
    
    # Perfect accuracy line
    min_vol, max_vol = target_ul.min(), target_ul.max()
    ax1.plot([min_vol, max_vol], [min_vol, max_vol], 
             'k--', alpha=0.8, linewidth=2, label='Perfect accuracy')
    
    # ±5% and ±10% accuracy bands
    ax1.fill_between([min_vol, max_vol], 
                     [min_vol * 0.95, max_vol * 0.95], 
                     [min_vol * 1.05, max_vol * 1.05], 
                     alpha=0.2, color='green', label='±5% accuracy')
    ax1.fill_between([min_vol, max_vol], 
                     [min_vol * 0.9, max_vol * 0.9], 
                     [min_vol * 1.1, max_vol * 1.1], 
                     alpha=0.1, color='orange', label='±10% accuracy')
    
    ax1.set_xlabel('Target Volume (μL)')
    ax1.set_ylabel('Measured Volume (μL)')
    ax1.set_title('Target vs Measured Volume')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Deviation Bar Chart
    ax2 = axes[0, 1]
    colors = ['green' if abs(x) <= 5 else 'orange' if abs(x) <= 10 else 'red' 
              for x in valid_results['accuracy_percent']]
    
    bars = ax2.bar(range(len(valid_results)), valid_results['accuracy_percent'], 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add horizontal reference lines
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.8, linewidth=1)
    ax2.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='±5% target')
    ax2.axhline(y=-5, color='green', linestyle='--', alpha=0.7)
    ax2.axhline(y=10, color='orange', linestyle=':', alpha=0.7, label='±10% limit')
    ax2.axhline(y=-10, color='orange', linestyle=':', alpha=0.7)
    
    # Customize x-axis labels
    ax2.set_xticks(range(len(valid_results)))
    ax2.set_xticklabels([f'{vol:.0f}' for vol in valid_results['volume_target_ul']], rotation=45)
    ax2.set_xlabel('Target Volume (μL)')
    ax2.set_ylabel('Deviation (%)')
    ax2.set_title('Accuracy Deviation by Volume')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Precision (CV) Bar Chart  
    ax3 = axes[1, 0]
    cv_colors = ['green' if x <= 5 else 'orange' if x <= 10 else 'red' 
                 for x in valid_results['cv_percent']]
    
    bars3 = ax3.bar(range(len(valid_results)), valid_results['cv_percent'], 
                    color=cv_colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add horizontal reference lines for precision
    ax3.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='5% CV target')
    ax3.axhline(y=10, color='orange', linestyle=':', alpha=0.7, label='10% CV limit')
    
    ax3.set_xticks(range(len(valid_results)))
    ax3.set_xticklabels([f'{vol:.0f}' for vol in valid_results['volume_target_ul']], rotation=45)
    ax3.set_xlabel('Target Volume (μL)')
    ax3.set_ylabel('Coefficient of Variation (%)')
    ax3.set_title('Precision (CV) by Volume')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Individual Measurements Scatter
    ax4 = axes[1, 1]
    if len(raw_df) > 0:
        raw_df['target_volume_ul'] = raw_df['volume'] * 1000
        raw_df['measured_volume_ul'] = raw_df['calculated_volume'] * 1000
        
        # Create scatter plot for each volume with slight x-offset for visibility
        for i, volume in enumerate(valid_results['volume_target_ul']):
            volume_ml = volume / 1000
            mask = np.isclose(raw_df['volume'], volume_ml, rtol=0.01)
            volume_measurements = raw_df[mask]['measured_volume_ul']
            
            if len(volume_measurements) > 0:
                # Add small random offset to x-position for visibility
                x_positions = np.full(len(volume_measurements), i) + np.random.normal(0, 0.1, len(volume_measurements))
                ax4.scatter(x_positions, volume_measurements, alpha=0.6, s=50, 
                           label=f'{volume:.0f} μL' if i < 5 else "")  # Limit legend entries
    
    ax4.set_xticks(range(len(valid_results)))
    ax4.set_xticklabels([f'{vol:.0f}' for vol in valid_results['volume_target_ul']], rotation=45)
    ax4.set_xlabel('Target Volume (μL)')
    ax4.set_ylabel('Measured Volume (μL)')
    ax4.set_title('Distribution of Individual Measurements')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"validation_report_{liquid}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate text summary
    summary_lines = [
        f"Calibration Validation Report - {liquid.capitalize()}",
        "=" * 50,
        f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Volumes Tested: {len(valid_results)}",
        f"Total Measurements: {len(raw_df)}",
        "",
        "Summary Statistics:",
        f"  Mean Absolute Accuracy Error: {valid_results['absolute_accuracy_percent'].mean():.2f}%",
        f"  Mean Precision (CV): {valid_results['cv_percent'].mean():.2f}%",
        f"  Best Accuracy: {valid_results['absolute_accuracy_percent'].min():.2f}% at {valid_results.loc[valid_results['absolute_accuracy_percent'].idxmin(), 'volume_target_ul']:.1f} μL",
        f"  Worst Accuracy: {valid_results['absolute_accuracy_percent'].max():.2f}% at {valid_results.loc[valid_results['absolute_accuracy_percent'].idxmax(), 'volume_target_ul']:.1f} μL",
        "",
        "Volume-by-Volume Results:",
    ]
    
    for _, row in valid_results.iterrows():
        summary_lines.append(
            f"  {row['volume_target_ul']:6.1f} μL: "
            f"measured {row['volume_measured_mean_ul']:6.2f}±{row['volume_measured_std_ul']:5.2f} μL "
            f"({row['accuracy_percent']:+5.1f}%, CV {row['cv_percent']:4.1f}%)"
        )
    
    # Performance categories
    summary_lines.extend([
        "",
        "Performance Assessment:",
    ])
    
    excellent = valid_results['absolute_accuracy_percent'] <= 5
    good = (valid_results['absolute_accuracy_percent'] > 5) & (valid_results['absolute_accuracy_percent'] <= 10)
    poor = valid_results['absolute_accuracy_percent'] > 10
    
    summary_lines.append(f"  Excellent (≤5% error): {excellent.sum()}/{len(valid_results)} volumes")
    summary_lines.append(f"  Good (5-10% error): {good.sum()}/{len(valid_results)} volumes")
    summary_lines.append(f"  Poor (>10% error): {poor.sum()}/{len(valid_results)} volumes")
    
    # Save summary
    summary_text = "\n".join(summary_lines)
    summary_path = output_dir / f"validation_summary_{liquid}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"📊 Validation report generated:")
    print(f"   Plot: {plot_path}")
    print(f"   Summary: {summary_path}")
    print("\n" + summary_text)

def run_validation(liquid=DEFAULT_LIQUID, simulate=DEFAULT_SIMULATE, 
                  volumes=None, replicates=DEFAULT_REPLICATES, 
                  input_vial_file=DEFAULT_INPUT_VIAL_STATUS_FILE):
    """
    Run calibration validation with intelligent parameter optimization
    
    Args:
        liquid: Liquid type (e.g., "glycerol", "water", "ethanol") - enables automatic optimization
        simulate: Run in simulation mode
        volumes: List of volumes to test in mL (defaults to DEFAULT_VOLUMES)
        replicates: Number of replicates per volume
        input_vial_file: Path to vial configuration file
    """
    if volumes is None:
        volumes = DEFAULT_VOLUMES
    
    print(f"🔬 Calibration Validation Workflow")
    print(f"   Liquid: {liquid}")
    print(f"   Simulation: {simulate}")
    print(f"   Volumes: {[v*1000 for v in volumes]} μL")
    print(f"   Replicates: {replicates}")
    print()
    
    try:
        # Initialize Lash_E coordinator
        lash_e = Lash_E(input_vial_file, simulate=simulate)
        lash_e.nr_robot.check_input_file()
        lash_e.nr_track.check_input_file()
        
        # Intelligent parameter system handles optimization automatically
        print(f"✅ Using intelligent parameter optimization for {liquid}")
        print(f"   Robot will automatically optimize parameters for each volume")
        if liquid is None:
            print(f"   Liquid=None: Will use pure system defaults (no calibration)")
        else:
            print(f"   Liquid={liquid}: Will use calibrated parameters if available, defaults otherwise")
        
        # Initialize experiment
        initialize_experiment(lash_e, liquid)
        
        # Run validation using intelligent parameter optimization
        results_df, raw_df, output_dir = validate_volumes(
            lash_e, liquid, volumes, replicates, simulate
        )
        
        # Generate report
        generate_validation_report(results_df, raw_df, output_dir, liquid)
        
        print(f"\n🎉 Validation complete! Check {output_dir} for results.")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function for command line execution"""
    # Default configuration
    run_validation()

if __name__ == "__main__":
    main()
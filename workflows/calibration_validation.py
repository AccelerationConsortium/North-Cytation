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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent figure display
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add paths for imports
sys.path.append("../utoronto_demo")

# Import slack functionality
import slack_agent

# Import base functionality
from calibration_sdl_base import (
    pipet_and_measure, LIQUIDS, set_vial_management, 
    normalize_parameters, empty_vial_if_needed, fill_liquid_if_needed
)
import calibration_sdl_base as base_module  # Need this for vial management
from master_usdl_coordinator import Lash_E

# Import pipetting wizard for direct parameter control
from pipetting_data.pipetting_wizard import PipettingWizard

# --- CONFIGURATION ---
DEFAULT_LIQUID = "DMSO"
DEFAULT_SIMULATE = False
#DEFAULT_VOLUMES = [0.008, 0.012, 0.02, 0.04, 0.075, 0.12]  # mL
DEFAULT_VOLUMES = [0.25, 0.45, 0.65, 0.85]  # mL
DEFAULT_REPLICATES = 5
DEFAULT_INPUT_VIAL_STATUS_FILE = "status/calibration_vials_overnight.csv"
COMPENSATE_OVERVOLUME = False  # NEW: Control overvolume compensation in wizard
# Vial management mode - set to match your calibration setup
# Options: "legacy" (no vial management), "single", "dual", etc.
VIAL_MANAGEMENT_MODE = "swap"  # Change this to match calibration setup


# --- ESSENTIAL VIAL MANAGEMENT FUNCTIONS (from simplified workflow) ---
def get_liquid_source_with_vial_management(lash_e, state, minimum_volume=2.0):
    """Get liquid source with vial management."""
    try:
        # Apply vial management first
        if base_module._VIAL_MANAGEMENT_MODE_OVERRIDE and base_module._VIAL_MANAGEMENT_MODE_OVERRIDE.lower() != "legacy":
            base_module.manage_vials(lash_e, state)
            
            # Get current source from config
            if base_module._VIAL_MANAGEMENT_CONFIG_OVERRIDE:
                cfg = {**base_module.VIAL_MANAGEMENT_DEFAULTS, **base_module._VIAL_MANAGEMENT_CONFIG_OVERRIDE}
            else:
                cfg = base_module.VIAL_MANAGEMENT_DEFAULTS
            
            current_source = cfg.get('source_vial', 'liquid_source_0')
            print(f"[vial-mgmt] Using current source vial: {current_source}")
            return current_source
    except Exception as e:
        print(f"[vial-mgmt] pre-pipetting management skipped: {e}")
    
    # Legacy mode with smart source switching
    return get_next_available_liquid_source(lash_e, state, minimum_volume)

def get_next_available_liquid_source(lash_e, state, minimum_volume=2.0):
    """Legacy mode: Find next available liquid source with enough volume."""
    # Initialize source index if not exists
    if "current_source_index" not in state:
        state["current_source_index"] = 0
    
    # Try current source first
    current_index = state["current_source_index"]
    current_source = f"liquid_source_{current_index}"
    
    try:
        current_vol = lash_e.nr_robot.get_vial_info(current_source, "vial_volume")
        if current_vol is not None and current_vol >= minimum_volume:
            return current_source
        else:
            print(f"[legacy] {current_source} volume ({current_vol:.1f}mL) below minimum ({minimum_volume:.1f}mL)")
    except Exception as e:
        print(f"[legacy] Could not check {current_source} volume: {e}")
    
    # Current source is low, try to find next available source
    for i in range(5):  # Check up to 5 sources
        next_index = current_index + 1 + i
        next_source = f"liquid_source_{next_index}"
        
        try:
            next_vol = lash_e.nr_robot.get_vial_info(next_source, "vial_volume")
            if next_vol is not None and next_vol >= minimum_volume:
                # Found good source, switch to it
                state["current_source_index"] = next_index
                print(f"[legacy] Switching to {next_source} (volume: {next_vol:.1f}mL)")
                return next_source
        except Exception as e:
            continue
    
    # No good sources found, stick with current
    print(f"[legacy] Warning: No liquid sources with ≥{minimum_volume:.1f}mL found, using {current_source}")
    return current_source

def update_vial_assignments(state):
    """Update source and dest vials based on current state."""
    # Get current source vial
    source_vial = get_liquid_source_with_vial_management(None, state)
    
    # Get current measurement vial
    dest_vial = state.get("measurement_vial_name", "measurement_vial_0")
    
    return source_vial, dest_vial

def check_if_measurement_vial_full(lash_e, state, max_volume=7.0):
    """Check and handle full measurement vial."""
    current_vial = state["measurement_vial_name"]
    
    try:
        vol = lash_e.nr_robot.get_vial_info(current_vial, "vial_volume")
        if vol is not None and vol > max_volume:
            print(f"[legacy] Measurement vial {current_vial} full ({vol:.1f}mL) - switching to next vial")
            
            # Remove pipet before vial operations
            lash_e.nr_robot.remove_pipet()
            
            # Return current vial home
            lash_e.nr_robot.return_vial_home(current_vial)
            
            # Ensure measurement_vial_index exists
            if "measurement_vial_index" not in state:
                print(f"[legacy] Warning: measurement_vial_index missing from state, initializing to 0")
                state["measurement_vial_index"] = 0
            
            # Switch to next measurement vial
            state["measurement_vial_index"] += 1
            new_vial_name = f"measurement_vial_{state['measurement_vial_index']}"
            state["measurement_vial_name"] = new_vial_name
            
            # Move new vial to clamp position
            lash_e.nr_robot.move_vial_to_location(new_vial_name, "clamp", 0)
            
            print(f"[legacy] Switched to new measurement vial: {new_vial_name}")
            return True
    except Exception as e:
        print(f"[legacy] Could not check measurement vial volume: {e}")
    
    return False



def initialize_experiment(lash_e, liquid, initial_measurement_vial="measurement_vial_0"):
    """Initialize experiment with proper vial setup"""
    print(f"🔧 Initializing experiment for {liquid} validation...")
    
    # Set vial management mode (configurable at top of file)
    if VIAL_MANAGEMENT_MODE != "legacy":
        set_vial_management(mode=VIAL_MANAGEMENT_MODE)
        print(f"   🧪 Vial management: {VIAL_MANAGEMENT_MODE}")
    else:
        print(f"   🧪 Vial management: legacy (custom vial switching)")
    
    # Ensure measurement vial is in clamp position (use specified vial, not hardcoded)
    lash_e.nr_robot.move_vial_to_location(initial_measurement_vial, "clamp", 0)
    
    print("✅ Experiment initialized")

def validate_volumes(lash_e, liquid, volumes, replicates, simulate, liquid_for_params=None, reset_vials=True):
    """
    Validate pipetting accuracy across specified volumes using intelligent parameter optimization
    
    Args:
        lash_e: Lash_E coordinator instance
        liquid: Liquid type for density calculations and pipet behavior
        volumes: List of volumes to test (mL)
        replicates: Number of replicates per volume
        simulate: Simulation mode flag
        liquid_for_params: Liquid type for parameter optimization (None = defaults, liquid = optimized)
    
    Returns:
        results_df: Summary results per volume
        raw_df: Raw measurement data
        output_dir: Output directory path
    """
    print(f"🧪 Starting validation for {len(volumes)} volumes with {replicates} replicates each...")
    
    # Determine which liquid to use for parameter optimization
    if liquid_for_params is None:
        param_liquid = None  # Use default parameters
    else:
        param_liquid = liquid_for_params  # Use liquid-specific optimization
    
    # Get liquid properties for density calculation (always use the actual liquid)
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
    param_type = "wizard" if liquid_for_params else "default"
    output_dir = Path("output") / "validation_runs" / f"validation_{param_type}_{liquid or 'default'}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_validation_data.csv"
    
    # Vial configuration based on vial management mode
    if VIAL_MANAGEMENT_MODE == "legacy":
        # Legacy mode: separate source and destination vials (static assignment)
        source_vial = "liquid_source_0"      # Liquid reservoir
        dest_vial = "measurement_vial_0"     # Measurement vial 
        print(f"   🧪 Legacy mode: {source_vial} → {dest_vial}")
    else:
        # Non-legacy mode: let vial management system handle vial selection
        # Start with defaults - swap mode will update these automatically as needed
        source_vial = "liquid_source_0"      # Will be managed by swap system
        dest_vial = "measurement_vial_0"     # Will be managed by swap system
        print(f"   🧪 {VIAL_MANAGEMENT_MODE.title()} mode: {source_vial} → {dest_vial} (managed by vial system)")
    
    # Initialize vial management state (only if requested or first time)
    if reset_vials or not hasattr(validate_volumes, '_persistent_state'):
        validate_volumes._persistent_state = {
            "measurement_vial_name": "measurement_vial_0",
            "measurement_vial_index": 0,
            "current_source_index": 0
        }
        print(f"   🔧 {'Resetting' if reset_vials else 'Initializing'} vial state: measurement_vial_0")
    else:
        print(f"   🔄 Using existing vial state: {validate_volumes._persistent_state['measurement_vial_name']}")
    
    state = validate_volumes._persistent_state
    
    # Process each volume
    for i, volume in enumerate(volumes):
        print(f"\n📏 Testing volume {i+1}/{len(volumes)}: {volume*1000:.1f} μL")
        
        try:
            # Use intelligent parameter system - param_liquid=None triggers defaults
            param_source = "wizard-optimized" if param_liquid else "default"
            print(f"   Using {param_source} parameters for {volume*1000:.1f} μL")
            
            # Calculate expected mass
            expected_mass = volume * liquid_density
            
            # Update vial assignments using vial management
            if VIAL_MANAGEMENT_MODE != "legacy":
                source_vial = get_liquid_source_with_vial_management(lash_e, state)
                dest_vial = state["measurement_vial_name"]
            else:
                source_vial = get_next_available_liquid_source(lash_e, state)
                dest_vial = state["measurement_vial_name"]
            
            print(f"   🧪 Using vials: {source_vial} → {dest_vial}")
            
            # Perform measurements
            volume_measurements = []
            mass_measurements = []
            
            for rep in range(replicates):
                print(f"   Replicate {rep+1}/{replicates}...", end="")
                
                # CRITICAL: Update vial assignments INSIDE the loop after each measurement
                # Vial management (swap mode) can change vials during pipet_and_measure calls
                if VIAL_MANAGEMENT_MODE != "legacy":
                    source_vial = get_liquid_source_with_vial_management(lash_e, state)
                    dest_vial = state["measurement_vial_name"]
                else:
                    source_vial = get_next_available_liquid_source(lash_e, state)
                    dest_vial = state["measurement_vial_name"]
                
                # Get parameters directly from wizard with explicit compensation control
                validation_params = None
                if param_liquid:  # Only use wizard if we have a liquid specified
                    try:
                        wizard = PipettingWizard()
                        wizard_params = wizard.get_pipetting_parameters(
                            param_liquid, 
                            volume, 
                            compensate_overvolume=COMPENSATE_OVERVOLUME  # Explicit control!
                        )
                        
                        if wizard_params:
                            # Convert wizard dict to format expected by calibration_sdl_base
                            validation_params = {
                                'aspirate_speed': wizard_params.get('aspirate_speed', 10),
                                'dispense_speed': wizard_params.get('dispense_speed', 10),
                                'aspirate_wait_time': wizard_params.get('aspirate_wait_time', 0.0),
                                'dispense_wait_time': wizard_params.get('dispense_wait_time', 0.0),
                                'retract_speed': wizard_params.get('retract_speed', 10),
                                'blowout_vol': wizard_params.get('blowout_vol', 0.0),
                                'post_asp_air_vol': wizard_params.get('post_asp_air_vol', 0.0),
                                'overaspirate_vol': wizard_params.get('overaspirate_vol', 0.0)
                            }
                            print(f"wizard→", end="")  # Indicate wizard was used
                        else:
                            print(f"defaults→", end="")  # Indicate fallback to defaults
                    except Exception as e:
                        print(f"error({e})→defaults→", end="")  # Indicate error fallback
                else:
                    print(f"no-liquid→defaults→", end="")  # Indicate no liquid specified
                
                # DEBUG: Show parameters for first replicate
                if rep == 0:
                    if validation_params:
                        print(f"\n   🔧 Parameters: asp={validation_params.get('aspirate_speed', 'N/A')}, "
                              f"disp={validation_params.get('dispense_speed', 'N/A')}, "
                              f"blow={validation_params.get('blowout_vol', 'N/A'):.3f}mL, "
                              f"over={validation_params.get('overaspirate_vol', 'N/A'):.4f}mL")
                    else:
                        print(f"\n   🔧 Parameters: Using robot defaults")
                
                result = pipet_and_measure(
                    lash_e=lash_e,
                    source_vial=source_vial,
                    dest_vial=dest_vial,
                    volume=volume,
                    params=validation_params,  # Pass explicit params (or None for defaults)
                    expected_measurement=expected_mass,
                    expected_time=30.0,  # Placeholder
                    replicate_count=1,  # Single measurement
                    simulate=simulate,
                    raw_path=str(raw_path),
                    raw_measurements=raw_measurements,
                    liquid=liquid,  # Always use real liquid for density calculation
                    new_pipet_each_time=new_pipet_each_time_set,  # Use liquid-specific setting
                    trial_type="VALIDATION",
                    liquid_for_params=param_liquid  # Keep for potential future use
                )
                
                # Check if measurement vial is getting full and switch if needed
                vial_switched = check_if_measurement_vial_full(lash_e, state)
                if vial_switched:
                    # Update dest_vial for next replicate
                    dest_vial = state["measurement_vial_name"]
                    print(f"   🧪 Switched to new measurement vial: {dest_vial}")
                
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
                'liquid': param_liquid or 'default',  # Use param_liquid for parameter tracking
                'parameter_source': param_source
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
                'liquid': param_liquid or 'default',  # Use param_liquid for parameter tracking
                'parameter_source': param_source,
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


def validate_volumes_wizard_only(lash_e, liquid, volumes, replicates, simulate):
    """
    Validate pipetting accuracy using only wizard-optimized parameters (original behavior)
    """
    print(f"🧪 Starting wizard-only validation for {len(volumes)} volumes with {replicates} replicates each...")
    
    # Get liquid properties
    if liquid not in LIQUIDS:
        raise ValueError(f"Unknown liquid '{liquid}' - must be one of: {list(LIQUIDS.keys())}")
    liquid_density = LIQUIDS[liquid]["density"]
    
    # Get liquid-specific pipet behavior
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
    
    # Vial configuration
    if VIAL_MANAGEMENT_MODE == "legacy":
        source_vial = "liquid_source_0"
        dest_vial = "measurement_vial_0"
        print(f"   🧪 Legacy mode: {source_vial} → {dest_vial}")
    else:
        source_vial = "measurement_vial_0"
        dest_vial = "measurement_vial_0"
        print(f"   🧪 Single vial mode: {source_vial} (same vial)")
    
    # Process each volume with wizard parameters only
    for i, volume in enumerate(volumes):
        print(f"\n📏 Testing volume {i+1}/{len(volumes)}: {volume*1000:.1f} μL")
        
def validate_volumes_wizard_only(lash_e, liquid, volumes, replicates, simulate):
    """
    Validate pipetting accuracy using only wizard-optimized parameters (original behavior)
    """
    return validate_volumes(lash_e, liquid, volumes, replicates, simulate)


def compare_validation_results(wizard_csv_path, default_csv_path, output_dir=None):
    """
    Compare two validation result CSV files and generate comparison report.
    
    Args:
        wizard_csv_path: Path to wizard validation results CSV
        default_csv_path: Path to default validation results CSV  
        output_dir: Directory to save comparison results (optional)
    
    Returns:
        comparison_df: DataFrame with side-by-side comparison
    """
    print(f"📊 Comparing validation results...")
    print(f"   Wizard results: {wizard_csv_path}")
    print(f"   Default results: {default_csv_path}")
    
    # Load both result files
    try:
        wizard_df = pd.read_csv(wizard_csv_path)
        default_df = pd.read_csv(default_csv_path)
    except Exception as e:
        print(f"❌ Error loading result files: {e}")
        return None
    
    # Merge on volume for comparison
    comparison_data = []
    
    for _, wizard_row in wizard_df.iterrows():
        volume = wizard_row['volume_target_ul']
        
        # Find matching default result
        default_row = default_df[default_df['volume_target_ul'] == volume]
        
        if len(default_row) > 0:
            default_row = default_row.iloc[0]
            
            # Calculate improvements
            accuracy_improvement = default_row['absolute_accuracy_percent'] - wizard_row['absolute_accuracy_percent']
            precision_improvement = default_row['cv_percent'] - wizard_row['cv_percent']
            
            comparison = {
                'volume_target_ul': volume,
                'wizard_accuracy_pct': wizard_row['accuracy_percent'],
                'default_accuracy_pct': default_row['accuracy_percent'],
                'wizard_absolute_accuracy_pct': wizard_row['absolute_accuracy_percent'],
                'default_absolute_accuracy_pct': default_row['absolute_accuracy_percent'],
                'accuracy_improvement_pct': accuracy_improvement,
                'wizard_cv_pct': wizard_row['cv_percent'],
                'default_cv_pct': default_row['cv_percent'],
                'precision_improvement_pct': precision_improvement,
                'wizard_better_accuracy': wizard_row['absolute_accuracy_percent'] < default_row['absolute_accuracy_percent'],
                'wizard_better_precision': wizard_row['cv_percent'] < default_row['cv_percent']
            }
            comparison_data.append(comparison)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path / "parameter_comparison.csv", index=False)
        
        # Generate simple comparison plot
        create_comparison_plot(comparison_df, output_path)
        print(f"   💾 Comparison saved to: {output_path}")
    
    # Print summary
    print_comparison_summary(comparison_df)
    
    return comparison_df


def compare_multiple_validation_results(all_results, output_dir=None):
    """
    Compare multiple validation results (N-way comparison) and generate simplified comparison.
    
    Args:
        all_results: Dict of {method_name: {'results_df': df, 'method': config}}
        output_dir: Directory to save comparison results (optional)
    
    Returns:
        comparison_df: DataFrame with simplified comparison data
    """
    print(f"📊 Comparing {len(all_results)} validation methods...")
    
    # Create simplified comparison DataFrame
    comparison_data = []
    
    # Get all volumes (assume all methods tested same volumes)
    first_method = next(iter(all_results.values()))
    volumes = first_method['results_df']['volume_target_ul'].tolist()
    
    for volume in volumes:
        volume_comparison = {'volume_target_ul': volume}
        
        # Extract data for each method at this volume
        for method_name, result_data in all_results.items():
            results_df = result_data['results_df']
            method_config = result_data['method']
            
            # Find row for this volume
            volume_row = results_df[results_df['volume_target_ul'] == volume]
            
            if len(volume_row) > 0:
                row = volume_row.iloc[0]
                volume_comparison[f'{method_name}_accuracy_pct'] = row.get('accuracy_percent', np.nan)
                volume_comparison[f'{method_name}_absolute_accuracy_pct'] = row.get('absolute_accuracy_percent', np.nan)
                volume_comparison[f'{method_name}_cv_pct'] = row.get('cv_percent', np.nan)
                volume_comparison[f'{method_name}_measured_ul'] = row.get('volume_measured_mean_ul', np.nan)
                volume_comparison[f'{method_name}_std_ul'] = row.get('volume_measured_std_ul', np.nan)
            else:
                # Missing data for this volume
                volume_comparison[f'{method_name}_accuracy_pct'] = np.nan
                volume_comparison[f'{method_name}_absolute_accuracy_pct'] = np.nan
                volume_comparison[f'{method_name}_cv_pct'] = np.nan
                volume_comparison[f'{method_name}_measured_ul'] = np.nan
                volume_comparison[f'{method_name}_std_ul'] = np.nan
        
        comparison_data.append(volume_comparison)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path / "multi_method_comparison.csv", index=False)
        
        # Generate comparison plot
        create_multi_method_comparison_plot(comparison_df, all_results, output_path)
        
        # Generate simple summary
        create_multi_method_summary(comparison_df, all_results, output_path)
        
        print(f"   💾 Multi-method comparison saved to: {output_path}")
    
    # Print summary
    print_multi_method_summary(comparison_df, all_results)
    
    return comparison_df


def create_multi_method_comparison_plot(comparison_df, all_results, output_dir):
    """Create comparison plots with multiple bars per volume"""
    if len(comparison_df) == 0:
        return
    
    methods = list(all_results.keys())
    volumes = comparison_df['volume_target_ul']
    volume_labels = [f"{v:.0f}" for v in volumes]
    x_positions = np.arange(len(volumes))
    
    n_methods = len(methods)
    bar_width = 0.8 / n_methods  # Distribute bars evenly
    colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown'][:n_methods]  # Support up to 6 methods
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    for i, method in enumerate(methods):
        accuracy_col = f'{method}_absolute_accuracy_pct'
        if accuracy_col in comparison_df.columns:
            offset = (i - (n_methods-1)/2) * bar_width
            ax1.bar(x_positions + offset, comparison_df[accuracy_col], 
                   width=bar_width, label=method.replace('_', ' ').title(), 
                   color=colors[i], alpha=0.7)
    
    ax1.set_xlabel('Volume (μL)')
    ax1.set_ylabel('Absolute Accuracy Error (%)')
    ax1.set_title('Accuracy Comparison (Lower is Better)')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(volume_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision comparison
    for i, method in enumerate(methods):
        cv_col = f'{method}_cv_pct'
        if cv_col in comparison_df.columns:
            offset = (i - (n_methods-1)/2) * bar_width
            ax2.bar(x_positions + offset, comparison_df[cv_col], 
                   width=bar_width, label=method.replace('_', ' ').title(), 
                   color=colors[i], alpha=0.7)
    
    ax2.set_xlabel('Volume (μL)')
    ax2.set_ylabel('Precision (CV %)')
    ax2.set_title('Precision Comparison (Lower is Better)')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(volume_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "multi_method_comparison.png", dpi=300, bbox_inches='tight')
    
    # Also create a clipped version (5% max)
    fig2, (ax1_clip, ax2_clip) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison (clipped)
    for i, method in enumerate(methods):
        accuracy_col = f'{method}_absolute_accuracy_pct'
        if accuracy_col in comparison_df.columns:
            offset = (i - (n_methods-1)/2) * bar_width
            ax1_clip.bar(x_positions + offset, comparison_df[accuracy_col], 
                        width=bar_width, label=method.replace('_', ' ').title(), 
                        color=colors[i], alpha=0.7)
    
    ax1_clip.set_xlabel('Volume (μL)')
    ax1_clip.set_ylabel('Absolute Accuracy Error (%)')
    ax1_clip.set_title('Accuracy Comparison (Clipped at 5%)')
    ax1_clip.set_xticks(x_positions)
    ax1_clip.set_xticklabels(volume_labels)
    ax1_clip.set_ylim(0, 5)
    ax1_clip.legend()
    ax1_clip.grid(True, alpha=0.3)
    
    # Precision comparison (clipped)
    for i, method in enumerate(methods):
        cv_col = f'{method}_cv_pct'
        if cv_col in comparison_df.columns:
            offset = (i - (n_methods-1)/2) * bar_width
            ax2_clip.bar(x_positions + offset, comparison_df[cv_col], 
                        width=bar_width, label=method.replace('_', ' ').title(), 
                        color=colors[i], alpha=0.7)
    
    ax2_clip.set_xlabel('Volume (μL)')
    ax2_clip.set_ylabel('Precision (CV %)')
    ax2_clip.set_title('Precision Comparison (Clipped at 5%)')
    ax2_clip.set_xticks(x_positions)
    ax2_clip.set_xticklabels(volume_labels)
    ax2_clip.set_ylim(0, 5)
    ax2_clip.legend()
    ax2_clip.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "multi_method_comparison_clipped_5pct.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close(fig2)
    
    print(f"   📈 Multi-method comparison plots saved")


def create_multi_method_summary(comparison_df, all_results, output_dir):
    """Create text summary of multi-method comparison"""
    methods = list(all_results.keys())
    
    summary_lines = [
        f"Multi-Method Validation Comparison",
        "=" * 40,
        f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Methods Compared: {len(methods)}",
        f"Volumes Tested: {len(comparison_df)}",
        "",
        "Methods:",
    ]
    
    for method_name, result_data in all_results.items():
        method_config = result_data['method']
        liquid_param = method_config['liquid_for_params']
        comp_ov = method_config['compensate_overvolume']
        summary_lines.append(f"  {method_name}: liquid_params={liquid_param}, compensate_overvolume={comp_ov}")
    
    summary_lines.extend([
        "",
        "Overall Performance (Mean Across All Volumes):",
    ])
    
    for method in methods:
        acc_col = f'{method}_absolute_accuracy_pct'
        cv_col = f'{method}_cv_pct'
        
        if acc_col in comparison_df.columns and cv_col in comparison_df.columns:
            mean_acc = comparison_df[acc_col].mean()
            mean_cv = comparison_df[cv_col].mean()
            summary_lines.append(f"  {method}: Accuracy {mean_acc:.2f}%, Precision {mean_cv:.2f}% CV")
    
    summary_text = "\n".join(summary_lines)
    
    with open(output_dir / "multi_method_summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary_text)


def print_multi_method_summary(comparison_df, all_results):
    """Print summary of multi-method comparison"""
    methods = list(all_results.keys())
    
    if len(comparison_df) == 0:
        print("   ⚠️  No data for comparison")
        return
    
    print(f"\n📋 MULTI-METHOD COMPARISON SUMMARY:")
    print(f"   Methods compared: {len(methods)}")
    print(f"   Volumes tested: {len(comparison_df)}")
    
    # Calculate overall performance for each method
    print(f"\n   Overall Performance (Mean):")
    best_accuracy_method = None
    best_accuracy_value = float('inf')
    best_precision_method = None
    best_precision_value = float('inf')
    
    for method in methods:
        acc_col = f'{method}_absolute_accuracy_pct'
        cv_col = f'{method}_cv_pct'
        
        if acc_col in comparison_df.columns and cv_col in comparison_df.columns:
            mean_acc = comparison_df[acc_col].mean()
            mean_cv = comparison_df[cv_col].mean()
            
            print(f"     {method}: Accuracy {mean_acc:.2f}%, Precision {mean_cv:.2f}% CV")
            
            if mean_acc < best_accuracy_value:
                best_accuracy_value = mean_acc
                best_accuracy_method = method
            
            if mean_cv < best_precision_value:
                best_precision_value = mean_cv
                best_precision_method = method
    
    if best_accuracy_method:
        print(f"\n   🏆 Best Accuracy: {best_accuracy_method} ({best_accuracy_value:.2f}%)")
    if best_precision_method:
        print(f"   🏆 Best Precision: {best_precision_method} ({best_precision_value:.2f}% CV)")


def compare_validation_results(wizard_csv_path, default_csv_path, output_dir=None):
    """
    Compare two validation result CSV files and generate comparison report.
    
    Args:
        wizard_csv_path: Path to wizard validation results CSV
        default_csv_path: Path to default validation results CSV  
        output_dir: Directory to save comparison results (optional)
    
    Returns:
        comparison_df: DataFrame with side-by-side comparison
    """
    print(f"📊 Comparing validation results...")
    print(f"   Wizard results: {wizard_csv_path}")
    print(f"   Default results: {default_csv_path}")
    
    # Load both result files
    try:
        wizard_df = pd.read_csv(wizard_csv_path)
        default_df = pd.read_csv(default_csv_path)
    except Exception as e:
        print(f"❌ Error loading result files: {e}")
        return None
    
    # Merge on volume for comparison
    comparison_data = []
    
    for _, wizard_row in wizard_df.iterrows():
        volume = wizard_row['volume_target_ul']
        
        # Find matching default result
        default_row = default_df[default_df['volume_target_ul'] == volume]
        
        if len(default_row) > 0:
            default_row = default_row.iloc[0]
            
            # Calculate improvements
            accuracy_improvement = default_row['absolute_accuracy_percent'] - wizard_row['absolute_accuracy_percent']
            precision_improvement = default_row['cv_percent'] - wizard_row['cv_percent']
            
            comparison = {
                'volume_target_ul': volume,
                'wizard_accuracy_pct': wizard_row['accuracy_percent'],
                'default_accuracy_pct': default_row['accuracy_percent'],
                'wizard_absolute_accuracy_pct': wizard_row['absolute_accuracy_percent'],
                'default_absolute_accuracy_pct': default_row['absolute_accuracy_percent'],
                'accuracy_improvement_pct': accuracy_improvement,
                'wizard_cv_pct': wizard_row['cv_percent'],
                'default_cv_pct': default_row['cv_percent'],
                'precision_improvement_pct': precision_improvement,
                'wizard_better_accuracy': wizard_row['absolute_accuracy_percent'] < default_row['absolute_accuracy_percent'],
                'wizard_better_precision': wizard_row['cv_percent'] < default_row['cv_percent']
            }
            comparison_data.append(comparison)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path / "parameter_comparison.csv", index=False)
        
        # Generate simple comparison plot
        create_comparison_plot(comparison_df, output_path)
        print(f"   💾 Comparison saved to: {output_path}")
    
    # Print summary
    print_comparison_summary(comparison_df)
    
    return comparison_df


def create_comparison_plot(comparison_df, output_dir):
    """Create comparison plots - both full scale and clipped at 5%"""
    if len(comparison_df) == 0:
        return
    
    volumes = comparison_df['volume_target_ul']
    volume_labels = [f"{v:.0f}" for v in volumes]  # Create clean labels
    x_positions = range(len(volumes))  # Use sequential positions for bars
    bar_width = 0.35
    
    # Create full-scale version
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison (full scale)
    ax1.bar([x - bar_width/2 for x in x_positions], comparison_df['wizard_absolute_accuracy_pct'], 
            width=bar_width, label='Wizard', color='green', alpha=0.7)
    ax1.bar([x + bar_width/2 for x in x_positions], comparison_df['default_absolute_accuracy_pct'], 
            width=bar_width, label='Default', color='red', alpha=0.7)
    ax1.set_xlabel('Volume (μL)')
    ax1.set_ylabel('Absolute Accuracy Error (%)')
    ax1.set_title('Accuracy Comparison')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(volume_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision comparison (full scale)
    ax2.bar([x - bar_width/2 for x in x_positions], comparison_df['wizard_cv_pct'], 
            width=bar_width, label='Wizard', color='green', alpha=0.7)
    ax2.bar([x + bar_width/2 for x in x_positions], comparison_df['default_cv_pct'], 
            width=bar_width, label='Default', color='red', alpha=0.7)
    ax2.set_xlabel('Volume (μL)')
    ax2.set_ylabel('Precision (CV %)')
    ax2.set_title('Precision Comparison')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(volume_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)  # Close to free memory
    
    # Create clipped version (5% max)
    fig2, (ax1_clip, ax2_clip) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison (clipped at 5%)
    ax1_clip.bar([x - bar_width/2 for x in x_positions], comparison_df['wizard_absolute_accuracy_pct'], 
                 width=bar_width, label='Wizard', color='green', alpha=0.7)
    ax1_clip.bar([x + bar_width/2 for x in x_positions], comparison_df['default_absolute_accuracy_pct'], 
                 width=bar_width, label='Default', color='red', alpha=0.7)
    ax1_clip.set_xlabel('Volume (μL)')
    ax1_clip.set_ylabel('Absolute Accuracy Error (%)')
    ax1_clip.set_title('Accuracy Comparison (Clipped at 5%)')
    ax1_clip.set_xticks(x_positions)
    ax1_clip.set_xticklabels(volume_labels)
    ax1_clip.set_ylim(0, 5)  # Clip at 5%
    ax1_clip.legend()
    ax1_clip.grid(True, alpha=0.3)
    
    # Precision comparison (clipped at 5%)
    ax2_clip.bar([x - bar_width/2 for x in x_positions], comparison_df['wizard_cv_pct'], 
                 width=bar_width, label='Wizard', color='green', alpha=0.7)
    ax2_clip.bar([x + bar_width/2 for x in x_positions], comparison_df['default_cv_pct'], 
                 width=bar_width, label='Default', color='red', alpha=0.7)
    ax2_clip.set_xlabel('Volume (μL)')
    ax2_clip.set_ylabel('Precision (CV %)')
    ax2_clip.set_title('Precision Comparison (Clipped at 5%)')
    ax2_clip.set_xticks(x_positions)
    ax2_clip.set_xticklabels(volume_labels)
    ax2_clip.set_ylim(0, 5)  # Clip at 5%
    ax2_clip.legend()
    ax2_clip.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_comparison_clipped_5pct.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)  # Close to free memory
    
    print(f"   📈 Comparison plots saved (full scale and clipped at 5%)")


def print_comparison_summary(comparison_df):
    """Print a summary of the comparison results"""
    if len(comparison_df) == 0:
        print("   ⚠️  No matching volumes found for comparison")
        return
        
    print(f"\n📋 COMPARISON SUMMARY:")
    print(f"   Volumes compared: {len(comparison_df)}")
    
    # Count wins
    wizard_accuracy_wins = sum(comparison_df['wizard_better_accuracy'])
    wizard_precision_wins = sum(comparison_df['wizard_better_precision'])
    total = len(comparison_df)
    
    print(f"   Wizard accuracy wins: {wizard_accuracy_wins}/{total} ({wizard_accuracy_wins/total*100:.0f}%)")
    print(f"   Wizard precision wins: {wizard_precision_wins}/{total} ({wizard_precision_wins/total*100:.0f}%)")
    
    # Average improvements
    avg_acc_improvement = comparison_df['accuracy_improvement_pct'].mean()
    avg_prec_improvement = comparison_df['precision_improvement_pct'].mean()
    
    print(f"   Avg accuracy improvement: {avg_acc_improvement:+.1f}% (positive = wizard better)")
    print(f"   Avg precision improvement: {avg_prec_improvement:+.1f}% (positive = wizard better)")


def run_comparison_validation(liquid=DEFAULT_LIQUID, simulate=DEFAULT_SIMULATE, 
                             volumes=None, replicates=DEFAULT_REPLICATES,
                             input_vial_file=DEFAULT_INPUT_VIAL_STATUS_FILE,
                             comparison_methods=None):
    """
    Run validation with multiple parameter methods and compare results.
    
    Args:
        liquid: Liquid type for wizard optimization
        simulate: Simulation mode
        volumes: List of volumes to test
        replicates: Number of replicates per volume
        input_vial_file: Vial configuration file
        comparison_methods: List of method configurations. If None, defaults to 3-way comparison:
            [
                {'name': 'wizard_comp_on', 'liquid_for_params': liquid, 'compensate_overvolume': True},
                {'name': 'wizard_comp_off', 'liquid_for_params': liquid, 'compensate_overvolume': False}, 
                {'name': 'default', 'liquid_for_params': None, 'compensate_overvolume': False}
            ]
    """
    if volumes is None:
        volumes = DEFAULT_VOLUMES.copy()
    
    # Default 3-way comparison if not specified: wizard with/without compensation vs default
    if comparison_methods is None:
        comparison_methods = [
            {'name': 'wizard_comp_on', 'liquid_for_params': liquid, 'compensate_overvolume': True},
            {'name': 'wizard_comp_off', 'liquid_for_params': liquid, 'compensate_overvolume': False},
            {'name': 'default', 'liquid_for_params': None, 'compensate_overvolume': False}
        ]
    
    print(f"🔬 MULTI-WAY VALIDATION COMPARISON: {len(comparison_methods)} Methods")
    print(f"   Liquid: {liquid}")
    print(f"   Volumes: {[f'{v*1000:.0f}μL' for v in volumes]}")
    print(f"   Replicates per volume: {replicates}")
    print(f"   Methods: {[m['name'] for m in comparison_methods]}")
    
    try:
        # Initialize robot
        lash_e = Lash_E(input_vial_file, simulate=simulate, initialize_biotek=False)
        
        # Run validation for each method
        all_results = {}
        all_output_dirs = {}
        
        for i, method in enumerate(comparison_methods):
            method_name = method['name']
            liquid_for_params = method['liquid_for_params']
            compensate_overvolume = method['compensate_overvolume']
            
            print(f"\n🧪 PHASE {i+1}: Testing {method_name}")
            print(f"   Liquid for params: {liquid_for_params}")
            print(f"   Compensate overvolume: {compensate_overvolume}")
            
            # Home robot at start of each validation method
            print(f"   🏠 Homing robot for clean start...")
            try:
                lash_e.nr_robot.move_home()  # Fast move to home position
                lash_e.nr_robot.home_robot_components()  # Full homing sequence
                print(f"   ✅ Robot homed successfully")
            except Exception as e:
                print(f"   ⚠️  Homing warning: {e}")
            
            # Set global compensation setting for this method
            global COMPENSATE_OVERVOLUME
            COMPENSATE_OVERVOLUME = compensate_overvolume
            
            # Initialize only on first run, continue with existing vials after that
            if i == 0:
                initialize_experiment(lash_e, liquid)
                reset_vials = True
            else:
                print(f"   🧪 Continuing with current vial configuration...")
                reset_vials = False
            
            # Run validation for this method
            results_df, raw_df, output_dir = validate_volumes(
                lash_e, liquid, volumes, replicates, simulate, 
                liquid_for_params=liquid_for_params, 
                reset_vials=reset_vials
            )
            
            # Store results
            all_results[method_name] = {
                'results_df': results_df,
                'raw_df': raw_df,
                'output_dir': output_dir,
                'method': method
            }
            all_output_dirs[method_name] = output_dir
            
            # Generate individual report
            generate_validation_report(results_df, raw_df, output_dir, f"{liquid}_{method_name}")
        
        # Phase N+1: Multi-way comparison
        print(f"\n📊 PHASE {len(comparison_methods)+1}: Comparing all methods...")
        
        # Create comparison output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_output = Path("output") / "validation_runs" / f"comparison_{len(comparison_methods)}way_{liquid}_{timestamp}"
        
        # Generate multi-way comparison
        comparison_df = compare_multiple_validation_results(all_results, comparison_output)
        
        # Cleanup: Remove pipet, return vials, and move to home
        print(f"\n🧹 Cleaning up after validation...")
        try:
            # Remove any held pipet
            if hasattr(lash_e.nr_robot, 'HELD_PIPET_TYPE') and lash_e.nr_robot.HELD_PIPET_TYPE is not None:
                print(f"   🗑️  Removing {lash_e.nr_robot.HELD_PIPET_TYPE} pipet...")
                lash_e.nr_robot.remove_pipet()
            
            # Return measurement vial from clamp to home position
            if hasattr(validate_volumes, '_persistent_state') and validate_volumes._persistent_state:
                current_measurement_vial = validate_volumes._persistent_state.get("measurement_vial_name", "measurement_vial_0")
                print(f"   � Returning {current_measurement_vial} from clamp to home...")
                lash_e.nr_robot.return_vial_home(current_measurement_vial)
            
            # Move robot to home position (not jnot home)
            print(f"   🏠 Moving robot to home position...")
            lash_e.nr_robot.move_home()
            
            print(f"   ✅ Cleanup complete")
            
        except Exception as e:
            print(f"   ⚠️  Cleanup warning: {e}")
        
        print(f"\n�🎉 {len(comparison_methods)}-way comparison validation complete!")
        for method_name, output_dir in all_output_dirs.items():
            print(f"   {method_name}: {output_dir}")
        print(f"   Comparison: {comparison_output}")
        
        # Send Slack notification of successful completion
        try:
            method_names = [m['name'] for m in comparison_methods]
            slack_msg = f"🧪 Validation Complete! {len(comparison_methods)}-way comparison for {liquid}\n"
            slack_msg += f"Methods: {', '.join(method_names)}\n"
            slack_msg += f"Volumes tested: {len(volumes)}, Replicates: {replicates}\n"
            slack_msg += f"Results saved to: {comparison_output.name}"
            
            if not simulate:  # Only send Slack in real mode
                slack_agent.send_slack_message(slack_msg)
                print(f"   📱 Slack notification sent")
            else:
                print(f"   📱 Slack notification skipped (simulation mode)")
        except Exception as e:
            print(f"   ⚠️  Slack notification failed: {e}")
        
        return all_results, comparison_df
        
    except Exception as e:
        print(f"❌ Multi-way validation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Attempt cleanup even on failure
        try:
            print(f"\n🧹 Attempting cleanup after failure...")
            if 'lash_e' in locals():
                if hasattr(lash_e.nr_robot, 'HELD_PIPET_TYPE') and lash_e.nr_robot.HELD_PIPET_TYPE is not None:
                    lash_e.nr_robot.remove_pipet()
                if hasattr(validate_volumes, '_persistent_state') and validate_volumes._persistent_state:
                    current_measurement_vial = validate_volumes._persistent_state.get("measurement_vial_name", "measurement_vial_0")
                    lash_e.nr_robot.return_vial_home(current_measurement_vial)
                lash_e.nr_robot.move_home()
                print(f"   ✅ Emergency cleanup complete")
        except Exception as cleanup_error:
            print(f"   ⚠️  Emergency cleanup failed: {cleanup_error}")
        
        return None, None, None
        
        results.append(wizard_results)
    
    # Convert to DataFrames
    results_df = pd.DataFrame(results)
    raw_df = pd.DataFrame(raw_measurements)
    
    # Save results
    results_df.to_csv(output_dir / "validation_summary.csv", index=False)
    raw_df.to_csv(output_dir / "raw_validation_data.csv", index=False)
    
    print(f"\n💾 Results saved to: {output_dir}")
    
    return results_df, raw_df, output_dir


def generate_comparison_report(results_df, raw_df, comparison_df, output_dir, liquid):
    """Generate comparison analysis and plots between wizard and default parameters"""
    print(f"\n📊 Generating parameter comparison report...")
    
    # Filter out failed measurements for analysis
    valid_results = results_df.dropna(subset=['accuracy_percent'])
    valid_comparisons = comparison_df.dropna()
    
    if len(valid_results) == 0:
        print("❌ No valid results for analysis")
        return
    
    # Create figure with subplots for comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{liquid.capitalize()} Parameter Comparison: Wizard vs Default', fontsize=16)
    
    # Plot 1: Accuracy Comparison (Bar Chart)
    ax1 = axes[0, 0]
    volumes = valid_comparisons['volume_target_ul']
    wizard_acc = valid_comparisons['wizard_absolute_accuracy_pct']
    default_acc = valid_comparisons['default_absolute_accuracy_pct']
    
    x = np.arange(len(volumes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, wizard_acc, width, label='Wizard-optimized', color='green', alpha=0.7)
    bars2 = ax1.bar(x + width/2, default_acc, width, label='Default parameters', color='red', alpha=0.7)
    
    ax1.set_xlabel('Volume (μL)')
    ax1.set_ylabel('Absolute Accuracy Error (%)')
    ax1.set_title('Accuracy Comparison (Lower is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{int(v)}' for v in volumes])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision Comparison (CV%)
    ax2 = axes[0, 1]
    wizard_cv = valid_comparisons['wizard_cv_pct']
    default_cv = valid_comparisons['default_cv_pct']
    
    bars3 = ax2.bar(x - width/2, wizard_cv, width, label='Wizard-optimized', color='green', alpha=0.7)
    bars4 = ax2.bar(x + width/2, default_cv, width, label='Default parameters', color='red', alpha=0.7)
    
    ax2.set_xlabel('Volume (μL)')
    ax2.set_ylabel('Precision (CV %)')
    ax2.set_title('Precision Comparison (Lower is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{int(v)}' for v in volumes])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Improvement Metrics
    ax3 = axes[0, 2]
    accuracy_improvements = valid_comparisons['accuracy_improvement_pct']
    precision_improvements = valid_comparisons['precision_improvement_pct']
    
    ax3.bar(x - width/2, accuracy_improvements, width, label='Accuracy improvement', color='blue', alpha=0.7)
    ax3.bar(x + width/2, precision_improvements, width, label='Precision improvement', color='purple', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    ax3.set_xlabel('Volume (μL)')
    ax3.set_ylabel('Improvement (%)')
    ax3.set_title('Parameter Optimization Benefits')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{int(v)}' for v in volumes])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Target vs Measured (Wizard)
    ax4 = axes[1, 0]
    wizard_data = valid_results[valid_results['parameter_source'] == 'wizard']
    target_ul = wizard_data['volume_target_ul']
    measured_ul = wizard_data['volume_measured_mean_ul']
    measured_std_ul = wizard_data['volume_measured_std_ul']
    
    ax4.errorbar(target_ul, measured_ul, yerr=measured_std_ul, 
                 fmt='o', capsize=5, capthick=2, markersize=8, color='green', alpha=0.7)
    min_vol, max_vol = target_ul.min(), target_ul.max()
    ax4.plot([min_vol, max_vol], [min_vol, max_vol], 'k--', alpha=0.8, linewidth=2)
    ax4.set_xlabel('Target Volume (μL)')
    ax4.set_ylabel('Measured Volume (μL)')
    ax4.set_title('Wizard Parameters: Target vs Measured')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Target vs Measured (Default)
    ax5 = axes[1, 1]
    default_data = valid_results[valid_results['parameter_source'] == 'default']
    target_ul_def = default_data['volume_target_ul']
    measured_ul_def = default_data['volume_measured_mean_ul']
    measured_std_ul_def = default_data['volume_measured_std_ul']
    
    ax5.errorbar(target_ul_def, measured_ul_def, yerr=measured_std_ul_def, 
                 fmt='o', capsize=5, capthick=2, markersize=8, color='red', alpha=0.7)
    min_vol_def, max_vol_def = target_ul_def.min(), target_ul_def.max()
    ax5.plot([min_vol_def, max_vol_def], [min_vol_def, max_vol_def], 'k--', alpha=0.8, linewidth=2)
    ax5.set_xlabel('Target Volume (μL)')
    ax5.set_ylabel('Measured Volume (μL)')
    ax5.set_title('Default Parameters: Target vs Measured')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary Statistics
    ax6 = axes[1, 2]
    
    # Calculate overall statistics
    wizard_mean_acc = wizard_acc.mean()
    default_mean_acc = default_acc.mean()
    wizard_mean_cv = wizard_cv.mean()
    default_mean_cv = default_cv.mean()
    
    categories = ['Accuracy Error\n(%)', 'Precision\n(CV %)']
    wizard_stats = [wizard_mean_acc, wizard_mean_cv]
    default_stats = [default_mean_acc, default_mean_cv]
    
    x_stats = np.arange(len(categories))
    ax6.bar(x_stats - width/2, wizard_stats, width, label='Wizard-optimized', color='green', alpha=0.7)
    ax6.bar(x_stats + width/2, default_stats, width, label='Default parameters', color='red', alpha=0.7)
    
    ax6.set_ylabel('Performance (%)')
    ax6.set_title('Overall Performance Summary')
    ax6.set_xticks(x_stats)
    ax6.set_xticklabels(categories)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (w_val, d_val) in enumerate(zip(wizard_stats, default_stats)):
        ax6.text(i - width/2, w_val + max(wizard_stats + default_stats) * 0.02, f'{w_val:.1f}', 
                ha='center', va='bottom', fontweight='bold', color='green')
        ax6.text(i + width/2, d_val + max(wizard_stats + default_stats) * 0.02, f'{d_val:.1f}', 
                ha='center', va='bottom', fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_comparison_report.png", dpi=300, bbox_inches='tight')
    print(f"   📈 Comparison plots saved to: parameter_comparison_report.png")
    
    # Print summary statistics
    print(f"\n📋 PARAMETER COMPARISON SUMMARY:")
    print(f"   Liquid: {liquid.upper()}")
    print(f"   Volumes tested: {len(valid_comparisons)}")
    
    # Count wins for wizard vs default
    wizard_accuracy_wins = sum(valid_comparisons['wizard_better_accuracy'])
    wizard_precision_wins = sum(valid_comparisons['wizard_better_precision'])
    total_tests = len(valid_comparisons)
    
    print(f"   Wizard wins (accuracy): {wizard_accuracy_wins}/{total_tests} ({wizard_accuracy_wins/total_tests*100:.0f}%)")
    print(f"   Wizard wins (precision): {wizard_precision_wins}/{total_tests} ({wizard_precision_wins/total_tests*100:.0f}%)")
    
    # Average improvements
    avg_acc_improvement = accuracy_improvements.mean()
    avg_prec_improvement = precision_improvements.mean()
    
    print(f"   Average accuracy improvement: {avg_acc_improvement:+.1f}% (wizard vs default)")
    print(f"   Average precision improvement: {avg_prec_improvement:+.1f}% (wizard vs default)")
    
    # Best and worst performing volumes
    best_acc_idx = accuracy_improvements.idxmax()
    worst_acc_idx = accuracy_improvements.idxmin()
    
    print(f"   Best accuracy improvement: {accuracy_improvements.iloc[best_acc_idx]:+.1f}% at {volumes.iloc[best_acc_idx]:.0f}μL")
    print(f"   Worst accuracy performance: {accuracy_improvements.iloc[worst_acc_idx]:+.1f}% at {volumes.iloc[worst_acc_idx]:.0f}μL")
    
    plt.show()


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
                  input_vial_file=DEFAULT_INPUT_VIAL_STATUS_FILE,
                  comparison_mode=False):
    """
    Run calibration validation with intelligent parameter optimization
    
    Args:
        liquid: Liquid type (e.g., "glycerol", "water", "ethanol") - enables automatic optimization
        simulate: Run in simulation mode
        volumes: List of volumes to test in mL (defaults to DEFAULT_VOLUMES)
        replicates: Number of replicates per volume
        input_vial_file: Path to vial configuration file
        comparison_mode: If True, compare wizard vs default parameters; if False, test only wizard
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
        lash_e = Lash_E(input_vial_file, simulate=simulate, initialize_biotek=False)
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
        
        # Choose validation mode
        if comparison_mode:
            print(f"🔬 Running COMPARISON mode: Use run_comparison_validation() instead")
            print(f"   Tip: Call run_comparison_validation() for automatic wizard vs default comparison")
            return None, None, None
        else:
            print(f"🔬 Running single validation with {'wizard-optimized' if liquid else 'default'} parameters")
            # Run single validation
            results_df, raw_df, output_dir = validate_volumes(
                lash_e, liquid, volumes, replicates, simulate, liquid_for_params=liquid
            )
            
            # Generate standard report
            generate_validation_report(results_df, raw_df, output_dir, liquid or 'default')
        
        print(f"\n🎉 Validation complete! Check {output_dir} for results.")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function for command line execution"""
    # Default configuration - comparison validation (wizard vs default)
    run_comparison_validation()
    
    # Uncomment the line below to run single validation instead:
    # run_validation()

if __name__ == "__main__":
    main()
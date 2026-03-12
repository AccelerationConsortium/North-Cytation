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
import uuid


# Import PipettingParameters for type hints
try:
    from pipetting_data.pipetting_parameters import PipettingParameters, ReservoirParameters
except ImportError:
    # Fallback for type hints if import fails
    PipettingParameters = None
    ReservoirParameters = None

def condition_tip(lash_e, vial_name, conditioning_volume_ul=100, liquid_type='water'):
    """Condition a pipette tip by aspirating and dispensing into source vial multiple times
    
    Args:
        lash_e: Lash_E robot controller
        vial_name: Name of vial to condition tip with
        conditioning_volume_ul: Total volume for conditioning (default 100 uL)
        liquid_type: Type of liquid for pipetting parameters (ignored - always uses dummy params)
    """
    try:
        # Calculate volume per conditioning cycle (3 cycles total)
        cycles = 3
        volume_per_cycle_ul = conditioning_volume_ul
        volume_per_cycle_ml = volume_per_cycle_ul / 1000
        
        # Use simple dummy parameters for fast conditioning (accuracy not needed)
        lash_e.logger.info(f"    Conditioning tip with {vial_name}: {cycles} cycles of {volume_per_cycle_ul:.1f}uL (fast dummy params)")
        
        # Simple dummy parameters - fast and reliable for conditioning only
        from pipetting_data.pipetting_parameters import PipettingParameters
        dummy_params = PipettingParameters(
            aspirate_speed=15,      # Fast aspirate
            dispense_speed=5,       # Fast dispense  
            dispense_wait_time=0.0, # No waiting
            blowout_vol=0.5         # Minimal blowout
        )
        
        for cycle in range(cycles):
            lash_e.nr_robot.aspirate_from_vial(vial_name, volume_per_cycle_ml, parameters=dummy_params)
            lash_e.nr_robot.dispense_into_vial(vial_name, volume_per_cycle_ml, parameters=dummy_params)
        
        lash_e.logger.info(f"    Tip conditioning complete for {vial_name}")
        
    except Exception as e:
        lash_e.logger.info(f"    Warning: Could not condition tip with {vial_name}: {e}")

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
    "SDS": 1.00,
    "6M_HCl": 1.10,
    "TFA": 1.49
}

# === MASTER DATABASE PATHS ===
MASTER_PIPETTING_MEASUREMENTS_DB = os.path.join("pipetting_data", "master_pipetting_measurements.csv")
MASTER_PIPETTING_SESSIONS_DB = os.path.join("pipetting_data", "master_pipetting_sessions.csv")
MASTER_RESERVOIR_MEASUREMENTS_DB = os.path.join("pipetting_data", "master_reservoir_measurements.csv")
MASTER_RESERVOIR_SESSIONS_DB = os.path.join("pipetting_data", "master_reservoir_sessions.csv")

# === PARAMETER CORRECTION LOGGING ===
PARAMETER_CORRECTIONS_LOG = os.path.join("pipetting_data", "parameter_corrections.csv")

def _calculate_volume_tolerance_ul(target_volume_ml: float) -> float:
    """Calculate volume-dependent accuracy tolerance in uL (from v2 calibration system).
    
    Args:
        target_volume_ml: Target volume in mL
        
    Returns:
        float: Accuracy tolerance in uL
    """
    volume_ul = target_volume_ml * 1000
    
    # V2 calibration tolerance ranges
    if 200 <= volume_ul <= 1000:
        tolerance_pct = 1.0
    elif 60 <= volume_ul <= 199:
        tolerance_pct = 2.0  
    elif 20 <= volume_ul <= 59:
        tolerance_pct = 3.0
    elif 2.5 <= volume_ul <= 19:
        tolerance_pct = 5.0
    else:  # 0-2.5uL
        tolerance_pct = 10.0
        
    return volume_ul * tolerance_pct / 100

def _log_parameter_correction(timestamp: str, liquid_type: str, target_volume_ml: float, 
                             old_overaspirate: float, new_overaspirate: float, 
                             validation_type: str, session_id: str, lash_e=None):
    """Log parameter corrections for tracking and analysis.
    
    Args:
        timestamp: ISO timestamp
        liquid_type: Type of liquid  
        target_volume_ml: Target volume in mL
        old_overaspirate: Original overaspirate value
        new_overaspirate: Corrected overaspirate value
        validation_type: 'pipetting' or 'reservoir'
        session_id: Session identifier
        lash_e: Lash_E instance for simulation check
    """
    # Skip logging in simulation mode
    if lash_e and hasattr(lash_e, 'simulate') and lash_e.simulate:
        print(f"    SIMULATION: Would log parameter correction: {old_overaspirate:.4f} → {new_overaspirate:.4f} mL")
        return
    
    try:
        correction_record = {
            'timestamp': timestamp,
            'session_id': session_id,
            'validation_type': validation_type,
            'liquid_type': liquid_type,
            'target_volume_ml': target_volume_ml,
            'target_volume_ul': target_volume_ml * 1000,
            'old_overaspirate_ml': old_overaspirate,
            'new_overaspirate_ml': new_overaspirate,
            'correction_ml': new_overaspirate - old_overaspirate,
            'correction_ul': (new_overaspirate - old_overaspirate) * 1000
        }
        
        df_new = pd.DataFrame([correction_record])
        
        if os.path.exists(PARAMETER_CORRECTIONS_LOG):
            df_new.to_csv(PARAMETER_CORRECTIONS_LOG, mode='a', header=False, index=False)
        else:
            os.makedirs(os.path.dirname(PARAMETER_CORRECTIONS_LOG), exist_ok=True)
            df_new.to_csv(PARAMETER_CORRECTIONS_LOG, index=False)
            
        print(f"    ✓ Logged parameter correction: {old_overaspirate:.4f} → {new_overaspirate:.4f} mL")
        
    except Exception as e:
        print(f"    ⚠ Warning: Could not log parameter correction: {e}")

def _generate_simulated_mass(target_volume_ml: float, density: float, 
                            current_overaspirate: float = 0.004, 
                            replicate_num: int = 1, 
                            validation_mode: bool = False) -> float:
    """Generate realistic simulated mass values for testing adaptive correction.
    
    CRITICAL SAFETY: This function should ONLY be called in simulation mode!
    
    Args:
        target_volume_ml: Target volume in mL
        density: Liquid density in g/mL
        current_overaspirate: Current overaspirate setting in mL
        replicate_num: Replicate number for variability
        validation_mode: If True, uses lower noise for validation stability
        
    Returns:
        float: Simulated mass difference in grams
        
    Raises:
        RuntimeError: If called outside simulation context
    """
    import random
    import os
    
    # CRITICAL SAFETY CHECK: Prevent fake data in real runs
    if os.getenv('ROBOT_SIMULATION_MODE') != 'TRUE':
        raise RuntimeError(
            "CRITICAL ERROR: _generate_simulated_mass called outside simulation! "
            "This would create fake data in real experiments. "
            "Set ROBOT_SIMULATION_MODE=TRUE environment variable for simulation."
        )
    
    # Base efficiency: overaspirate helps accuracy
    base_efficiency = 0.88 + (current_overaspirate * 15)  # 0.004 -> ~94% efficiency
    base_efficiency = min(base_efficiency, 1.03)  # Cap at 103% (more realistic)
    
    # Volume-dependent error (smaller volumes harder)
    volume_ul = target_volume_ml * 1000
    
    if validation_mode:
        # VALIDATION MODE: Reduced noise for stable validation results
        if volume_ul < 20:
            error_factor = random.uniform(-0.04, 0.03)  # ±4% error (reduced from ±15%)
        elif volume_ul < 100:
            error_factor = random.uniform(-0.03, 0.02)  # ±3% error (reduced from ±8%)
        else:
            error_factor = random.uniform(-0.02, 0.015)  # ±2% error (reduced from ±4%)
        
        # Reduced replicate noise for validation
        replicate_noise = random.uniform(-0.005, 0.005) * (replicate_num % 3) * 0.3
    else:
        # CALIBRATION MODE: Normal noise levels
        if volume_ul < 20:
            error_factor = random.uniform(-0.12, 0.08)  # ±12% error for small volumes
        elif volume_ul < 100:
            error_factor = random.uniform(-0.06, 0.04)  # ±6% error for medium volumes  
        else:
            error_factor = random.uniform(-0.03, 0.02)  # ±3% error for large volumes
        
        # Normal replicate variability
        replicate_noise = random.uniform(-0.015, 0.015) * (replicate_num % 3) * 0.5
    
    # Calculate simulated volume
    final_efficiency = base_efficiency + error_factor + replicate_noise
    simulated_volume_ml = target_volume_ml * final_efficiency
    simulated_volume_ml = max(0, simulated_volume_ml)  # Can't be negative
    
    # Convert to mass
    simulated_mass = simulated_volume_ml * density
    
    return simulated_mass


def _interpolate_optimal_overaspirate(point1_overaspirate: float, point1_measured: float,
                                     point2_overaspirate: float, point2_measured: float, 
                                     target_volume: float) -> float:
    """Interpolate optimal overaspirate using 2-point linear method from v2 calibration.
    
    Args:
        point1_overaspirate: First measurement overaspirate (mL)
        point1_measured: First measurement result (mL) 
        point2_overaspirate: Second measurement overaspirate (mL)
        point2_measured: Second measurement result (mL)
        target_volume: Target volume (mL)
        
    Returns:
        float: Optimal overaspirate value (mL)
    """
    # Calculate slope: change in volume per change in overaspirate
    volume_diff = point2_measured - point1_measured
    overaspirate_diff = point2_overaspirate - point1_overaspirate
    
    if abs(overaspirate_diff) < 1e-6:  # Avoid division by zero
        return point1_overaspirate  # Return original if no difference
    
    slope = volume_diff / overaspirate_diff
    
    # Linear interpolation: target = point1_measured + slope * (optimal_overaspirate - point1_overaspirate)
    # Solve for optimal_overaspirate
    volume_needed = target_volume - point1_measured
    overaspirate_adjustment = volume_needed / slope if abs(slope) > 1e-6 else 0
    optimal_overaspirate = point1_overaspirate + overaspirate_adjustment
    
    # Safety bounds: keep reasonable
    optimal_overaspirate = max(0.0, min(0.015, optimal_overaspirate))  # 0-15uL range
    
    return optimal_overaspirate

def _get_parameter_values(parameters, param_type="pipetting"):
    """Extract all parameter values with defaults filled in.
    
    Args:
        parameters: PipettingParameters or ReservoirParameters instance, or None
        param_type: "pipetting" or "reservoir"
        
    Returns:
        dict: All parameter values with defaults applied
    """
    if param_type == "pipetting":
        # Default PipettingParameters values
        defaults = {
            'aspirate_speed': 10,
            'dispense_speed': 10, 
            'retract_speed': 10,
            'blowout_speed': None,  # Uses aspirate_speed if None
            'aspirate_wait_time': 0.0,
            'dispense_wait_time': 0.0,
            'post_retract_wait_time': 0.0,
            'overaspirate_vol': 0.0,
            'pre_asp_air_vol': 0.0,
            'post_asp_air_vol': 0.0,
            'blowout_vol': 0.0,
            'asp_disp_cycles': 0
        }
    else:  # reservoir
        defaults = {
            'aspirate_speed': 11,
            'dispense_speed': 11,
            'aspirate_wait_time': 0.0,
            'dispense_wait_time': 0.0,
            'valve_switch_delay': 0.1,
            'overaspirate_vol': 0.0
        }
    
    if parameters is not None:
        # Update defaults with provided parameters
        for key, default_value in defaults.items():
            if hasattr(parameters, key):
                param_value = getattr(parameters, key)
                defaults[key] = param_value if param_value is not None else default_value
    
    # Handle blowout_speed special case for pipetting
    if param_type == "pipetting" and defaults['blowout_speed'] is None:
        defaults['blowout_speed'] = defaults['aspirate_speed']
    
    return defaults

def _append_to_master_measurements(measurements_data, validation_type="pipetting"):
    """Append individual measurements to appropriate master measurements database.
    
    Args:
        measurements_data: List of dict, each containing one measurement record
        validation_type: "pipetting" or "reservoir" to determine which database to use
    """
    try:
        df_new = pd.DataFrame(measurements_data)
        
        # Select appropriate database file
        db_file = MASTER_PIPETTING_MEASUREMENTS_DB if validation_type == "pipetting" else MASTER_RESERVOIR_MEASUREMENTS_DB
        
        # Check if master file exists
        if os.path.exists(db_file):
            # Append to existing file
            df_new.to_csv(db_file, mode='a', header=False, index=False)
        else:
            # Create new file with headers
            os.makedirs(os.path.dirname(db_file), exist_ok=True)
            df_new.to_csv(db_file, index=False)
        
        print(f"    ✓ Added {len(measurements_data)} measurements to {validation_type} database")
        
    except Exception as e:
        print(f"    ⚠ Warning: Could not update {validation_type} measurements database: {e}")

def _append_to_master_sessions(session_data, validation_type="pipetting"):
    """Append session summary to appropriate master sessions database.
    
    Args:
        session_data: Dict containing session summary record
        validation_type: "pipetting" or "reservoir" to determine which database to use
    """
    try:
        df_new = pd.DataFrame([session_data])
        
        # Select appropriate database file
        db_file = MASTER_PIPETTING_SESSIONS_DB if validation_type == "pipetting" else MASTER_RESERVOIR_SESSIONS_DB
        
        # Check if master file exists
        if os.path.exists(db_file):
            # Append to existing file
            df_new.to_csv(db_file, mode='a', header=False, index=False)
        else:
            # Create new file with headers  
            os.makedirs(os.path.dirname(db_file), exist_ok=True)
            df_new.to_csv(db_file, index=False)
        
        print(f"    ✓ Added session to {validation_type} database")
        
    except Exception as e:
        print(f"    ⚠ Warning: Could not update {validation_type} sessions database: {e}")

def _evaluate_measurement(stability_info: Dict, std_threshold: float = 0.001) -> bool:
    """Evaluate if a measurement is acceptable based on stability criteria.
    
    A baseline is considered stable if:
    - Stable readings percentage > 50%, OR
    - Standard deviation < threshold
    
    Both baselines must be stable for measurement to be trustworthy.
    
    Args:
        stability_info: Dictionary with stability metrics from dispense_into_vial
        std_threshold: Maximum acceptable standard deviation in grams (default: 0.001g = 1.0mg)
        
    Returns:
        bool: True if measurement is acceptable, False if should be retried
    """
    # Check pre-baseline stability
    pre_stable_pct = (stability_info['pre_stable_count'] / max(stability_info['pre_total_count'], 1)) * 100
    pre_stable = (pre_stable_pct > 50.0) or (stability_info['pre_baseline_std'] < std_threshold)
    
    # Check post-baseline stability  
    post_stable_pct = (stability_info['post_stable_count'] / max(stability_info['post_total_count'], 1)) * 100
    post_stable = (post_stable_pct > 50.0) or (stability_info['post_baseline_std'] < std_threshold)
    
    # Both baselines must be stable
    is_acceptable = pre_stable and post_stable
    
    print(f"    Quality check: pre={pre_stable_pct:.1f}% stable (std={stability_info['pre_baseline_std']:.6f}g), "
          f"post={post_stable_pct:.1f}% stable (std={stability_info['post_baseline_std']:.6f}g)")
    print(f"    Result: {'ACCEPTABLE' if is_acceptable else 'RETRY NEEDED'} (threshold: {std_threshold:.6f}g)")
    
    
    return is_acceptable

def calculate_quality_threshold(volume_ml: float) -> float:
    """
    Calculate appropriate quality standard deviation threshold based on volume.
    
    Args:
        volume_ml: Volume being tested in mL
        
    Returns:
        float: Quality threshold in grams
        
    Logic:
        - For volumes <= 0.005 mL (5 uL): threshold = 0.0005g (0.5 mg)
        - For volumes >= 0.5 mL (500 uL): threshold = 0.005g (5 mg)
        - For volumes in between: linear interpolation
    """
    if volume_ml <= 0.005:
        return 0.0005
    elif volume_ml >= 0.5:
        return 0.005
    else:
        # Linear interpolation between the two points
        # (0.005 mL, 0.0005g) and (0.5 mL, 0.005g)
        slope = (0.005 - 0.0005) / (0.5 - 0.005)
        threshold = 0.0005 + slope * (volume_ml - 0.005)
        return threshold

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
    smooth_overvolume: bool = False,
    quality_std_threshold: float = None,
    condition_tip_enabled: bool = False,
    conditioning_volume_ul: float = 100,
    parameters: Optional[PipettingParameters] = None,
    adaptive_correction: bool = False,
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
        quality_std_threshold: Standard deviation threshold for quality assessment (default: None = auto-calculate per volume)
        condition_tip_enabled: Whether to condition tip before validation (default: False)
        conditioning_volume_ul: Volume for tip conditioning in uL (default: 100)
        parameters: Optional custom pipetting parameters to override liquid-calibrated settings (default: None)
        adaptive_correction: Enable dynamic overaspirate correction during validation (default: False)
        
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
    
    # === SIMULATION SAFETY SETUP ===
    # Set simulation environment flag if in simulation mode
    if hasattr(lash_e, 'simulate') and lash_e.simulate:
        import os
        os.environ['ROBOT_SIMULATION_MODE'] = 'TRUE'
        print(f"    SIMULATION MODE: Environment safety flag set")
    else:
        # Ensure simulation flag is cleared for real runs
        import os
        if 'ROBOT_SIMULATION_MODE' in os.environ:
            del os.environ['ROBOT_SIMULATION_MODE']
    
    # Validate vials exist
    try:
        lash_e.nr_robot.get_vial_info(source_vial, 'location')
        lash_e.nr_robot.get_vial_info(destination_vial, 'location')
    except Exception as e:
        raise FileNotFoundError(f"Vial validation failed: {e}")
    
    # === SETUP OUTPUT DIRECTORY ===
    # Generate unique session ID
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Handle None output_folder for simulation mode
    if output_folder is not None:
        calib_folder = os.path.join(output_folder, "calibration_validation", f"validation_{liquid_type}_{source_vial}_{timestamp}")
        # Only create directories and prepare file operations in non-simulation mode
        if save_raw_data:
            os.makedirs(calib_folder, exist_ok=True)
    else:
        calib_folder = None
        save_raw_data = False  # Force disable file saving when output_folder is None
    
    # Store actual parameters used for validation session (not defaults)
    # Get the robot's actual optimized parameters that were used
    actual_params = lash_e.nr_robot._get_optimized_parameters(
        volume=volumes_ml[0] if volumes_ml else 0.1,  # Use first volume as representative
        liquid=liquid_type,
        user_parameters=parameters,
        compensate_overvolume=compensate_overvolume,
        smooth_overvolume=smooth_overvolume
    )
    param_values = _get_parameter_values(actual_params, "pipetting")
    
    print(f"\\n=== Vial-to-Vial Validation ===\\nSession ID: {session_id}")
    print(f"Source: {source_vial}")
    print(f"Destination: {destination_vial}")
    print(f"Liquid: {liquid_type} (density: {density} g/mL)")
    print(f"Volumes: {volumes_ml} mL")
    print(f"Replicates: {replicates}")
    print(f"Switch pipet: {switch_pipet}")
    print(f"Condition tip: {condition_tip_enabled}")
    if save_raw_data:
        print(f"Output: {calib_folder}")
    else:
        print("Output: Simulation mode - no files saved")
    
    # === TIP CONDITIONING (OPTIONAL) ===
    if condition_tip_enabled:
        print(f"\n--- Tip Conditioning ---")
        print(f"Moving {source_vial} to clamp position for conditioning...")
        
        try:

            # Move source vial to clamp position
            lash_e.nr_robot.move_vial_to_location(source_vial, 'clamp', 0)
            
            # Condition tip with specified volume
            condition_tip(lash_e, source_vial, conditioning_volume_ul, liquid_type)
            
        except Exception as e:
            print(f"Warning: Tip conditioning failed: {e}")
            print("Continuing with validation without conditioning...")
    
    # === DATA COLLECTION ===
    all_results = []
    
    # Pre-position destination vial in clamp before starting pipetting
    lash_e.nr_robot.move_vial_to_location(destination_vial, "clamp", 0)

    for volume_idx, volume_ml in enumerate(volumes_ml):
        print(f"\\nTesting volume: {volume_ml:.3f} mL ({replicates} replicates)")
        
        # Calculate appropriate quality threshold for this volume
        current_threshold = quality_std_threshold if quality_std_threshold is not None else calculate_quality_threshold(volume_ml)
        print(f"  Using quality threshold: {current_threshold:.6f}g ({current_threshold*1000:.3f}mg)")
        
        for rep in range(replicates):
            print(f"  Replicate {rep + 1}/{replicates}...")
            
            # Quality-controlled retry loop for calibration/validation
            max_retries = 3
            retry_count = 0
            measurement_acceptable = False
            
            while not measurement_acceptable and retry_count <= max_retries:
                if retry_count > 0:
                    print(f"    Retry attempt {retry_count}/{max_retries}")
                
                # Pipette using optimized parameters with continuous mass monitoring
                transfer_result = lash_e.nr_robot.dispense_from_vial_into_vial(
                    source_vial_name=source_vial,
                    dest_vial_name=destination_vial,
                    volume=volume_ml,
                    parameters=parameters,  # Pass through custom parameters if provided
                    liquid=liquid_type,
                    remove_tip=switch_pipet,  # Use built-in pipet removal control
                    use_safe_location=False,
                    return_vial_home=False,   # Keep destination vial in clamp for mass measurement
                    compensate_overvolume=compensate_overvolume,
                    smooth_overvolume=smooth_overvolume,
                    measure_weight=True,  # Enable mass measurement with continuous monitoring
                    continuous_mass_monitoring=True, 
                    save_mass_data=True
                    
                )
                
                # Handle new return format (mass, stability_info)
                if isinstance(transfer_result, tuple):
                    mass_difference, stability_info = transfer_result
                    # Evaluate measurement quality for calibration/validation
                    measurement_acceptable = _evaluate_measurement(stability_info, current_threshold)
                else:
                    # Backwards compatibility - old format returns just mass
                    mass_difference = transfer_result
                    stability_info = None
                    measurement_acceptable = True
                
                retry_count += 1
                
                if not measurement_acceptable and retry_count <= max_retries:
                    print(f"    Measurement quality insufficient, retrying...")
            
            if not measurement_acceptable:
                print(f"    WARNING: Measurement still poor quality after {max_retries} retries")

            # Handle simulation mode - generate realistic simulated mass
            # CRITICAL SAFETY: Multiple checks to prevent fake data in real runs
            if (hasattr(lash_e, 'simulate') and lash_e.simulate and 
                mass_difference == 0 and 
                getattr(lash_e, 'simulate', False) is True):  # Double-check simulate flag
                
                # Set environment variable for additional safety
                import os
                os.environ['ROBOT_SIMULATION_MODE'] = 'TRUE'
                
                current_overaspirate = parameters.overaspirate_vol if parameters else 0.004
                mass_difference = _generate_simulated_mass(
                    target_volume_ml=volume_ml, 
                    density=density,
                    current_overaspirate=current_overaspirate,
                    replicate_num=rep + 1,
                    validation_mode=True  # Use low-noise validation mode
                )
                print(f"    SIMULATION: Generated mass {mass_difference:.6f}g for validation")
            elif mass_difference == 0:
                # Real hardware returned zero - this is a problem!
                error_msg = (
                    f"Hardware returned zero mass for volume {volume_ml*1000:.1f}uL. "
                    f"Check scale connection, vial positioning, or liquid availability."
                )
                lash_e.nr_robot.pause_after_error(error_msg, send_slack=True)
                raise ValueError(error_msg)

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
                'switch_pipet': switch_pipet,
                'retry_count': retry_count,
                'stability_info': stability_info,
                'volume_index': volume_idx,  # Track which volume in the list
                'optimization_stage': 1  # Mark initial measurements as Stage 1
            }
            all_results.append(result)
            
            print(f"    Target: {volume_ml*1000:.1f} uL -> Measured: {measured_volume_ml*1000:.1f} uL (Error: {((measured_volume_ml-volume_ml)/volume_ml)*100:.1f}%)")
        
        # === ITERATIVE ADAPTIVE CORRECTION (3-STAGE PROCESS) ===
        if adaptive_correction:
            # Stage 1: Evaluate initial measurements 
            stage1_results = [r for r in all_results if r['target_volume_ml'] == volume_ml]
            stage1_volumes = [r['measured_volume_ml'] for r in stage1_results]
            stage1_mean = np.mean(stage1_volumes)
            
            # Calculate tolerance for this volume
            tolerance_ul = _calculate_volume_tolerance_ul(volume_ml)
            target_ul = volume_ml * 1000
            stage1_mean_ul = stage1_mean * 1000
            stage1_error_ul = abs(stage1_mean_ul - target_ul)
            
            print(f"    Stage 1 - Error: {stage1_error_ul:.1f}uL vs {tolerance_ul:.1f}uL tolerance")
            
            if stage1_error_ul > tolerance_ul:
                print(f"    Stage 1 outside tolerance - proceeding to iterative optimization")
                
                # Get initial overaspirate
                initial_overaspirate = parameters.overaspirate_vol if parameters else 0.004
                
                try:
                    from pipetting_data.pipetting_parameters import PipettingParameters
                    
                    # === STAGE 2: First Optimization ===
                    print(f"    Stage 2: Testing adjusted parameters...")
                    
                    # Calculate Stage 2 correction
                    if stage1_mean < volume_ml:  # Under-dispensing
                        stage2_overaspirate = initial_overaspirate + 0.005  # Increase by 5uL
                        print(f"      Under-dispensing: trying higher overaspirate {stage2_overaspirate:.4f} mL")
                    else:  # Over-dispensing
                        stage2_overaspirate = max(0.0, initial_overaspirate - 0.003)  # Decrease by 3uL  
                        print(f"      Over-dispensing: trying lower overaspirate {stage2_overaspirate:.4f} mL")
                    
                    # Create Stage 2 parameters
                    # Get the robot's current optimized parameters for this volume/liquid
                    base_params = lash_e.nr_robot._get_optimized_parameters(
                        volume=volume_ml, 
                        liquid=liquid_type, 
                        user_parameters=parameters,  # Include any user overrides
                        compensate_overvolume=compensate_overvolume,
                        smooth_overvolume=smooth_overvolume
                    )
                    
                    # Create Stage 2 parameters by copying all optimized params
                    stage2_params = PipettingParameters()
                    for attr in dir(base_params):
                        if not attr.startswith('_') and hasattr(stage2_params, attr):
                            setattr(stage2_params, attr, getattr(base_params, attr))
                    
                    # Apply ONLY the overaspirate adjustment
                    stage2_params.overaspirate_vol = stage2_overaspirate
                    
                    # Take Stage 2 measurements (3 replicates)
                    stage2_volumes = []
                    for rep in range(replicates):
                        print(f"      Stage 2 replicate {rep + 1}/{replicates}...")
                        
                        stage2_result = lash_e.nr_robot.dispense_from_vial_into_vial(
                            source_vial_name=source_vial,
                            dest_vial_name=destination_vial,
                            volume=volume_ml,
                            parameters=stage2_params,
                            liquid=liquid_type,
                            remove_tip=switch_pipet,
                            use_safe_location=False,
                            return_vial_home=False,
                            compensate_overvolume=compensate_overvolume,
                            smooth_overvolume=smooth_overvolume,
                            measure_weight=True,
                            continuous_mass_monitoring=True,
                            save_mass_data=True
                        )
                        
                        if isinstance(stage2_result, tuple):
                            stage2_mass, _ = stage2_result
                        else:
                            stage2_mass = stage2_result
                        
                        # Handle simulation mode with safety checks
                        if (hasattr(lash_e, 'simulate') and lash_e.simulate and 
                            stage2_mass == 0 and getattr(lash_e, 'simulate', False) is True):
                            
                            import os
                            os.environ['ROBOT_SIMULATION_MODE'] = 'TRUE'
                            
                            stage2_mass = _generate_simulated_mass(
                                target_volume_ml=volume_ml,
                                density=density,
                                current_overaspirate=stage2_overaspirate,
                                replicate_num=rep + 10,  # Different seed for stage 2
                                validation_mode=True  # Use low-noise validation mode
                            )
                            print(f"        SIMULATION: Generated Stage 2 mass {stage2_mass:.6f}g")
                        elif stage2_mass == 0:
                            error_msg = (
                                f"Stage 2 hardware returned zero mass for {volume_ml*1000:.1f}uL. "
                                f"Check scale/hardware during optimization."
                            )
                            lash_e.nr_robot.pause_after_error(error_msg, send_slack=True)
                            raise ValueError(error_msg)
                        
                        stage2_volume = stage2_mass / density
                        stage2_volumes.append(stage2_volume)
                        
                        # Store Stage 2 result
                        stage2_result_dict = {
                            'target_volume_ml': volume_ml,
                            'measured_volume_ml': stage2_volume,
                            'mass_difference_g': stage2_mass,
                            'density_used': density,
                            'liquid_type': liquid_type,
                            'replicate': rep + 1,
                            'timestamp': datetime.now().isoformat(),
                            'source_vial': source_vial,
                            'destination_vial': destination_vial,
                            'switch_pipet': switch_pipet,
                            'retry_count': 0,
                            'stability_info': None,
                            'volume_index': volume_idx,
                            'optimization_stage': 2
                        }
                        all_results.append(stage2_result_dict)
                        
                        print(f"        Target: {volume_ml*1000:.1f} uL -> Stage 2: {stage2_volume*1000:.1f} uL (Error: {((stage2_volume-volume_ml)/volume_ml)*100:.1f}%)")
                    
                    # Evaluate Stage 2
                    stage2_mean = np.mean(stage2_volumes)
                    stage2_error_ul = abs((stage2_mean * 1000) - target_ul)
                    print(f"    Stage 2 - Mean error: {stage2_error_ul:.1f}uL vs {tolerance_ul:.1f}uL tolerance")
                    
                    # === STAGE 3: Final Optimization ===
                    print(f"    Stage 3: Final parameter optimization...")
                    
                    # Interpolate optimal overaspirate from Stage 1 and Stage 2 results
                    optimal_overaspirate = _interpolate_optimal_overaspirate(
                        initial_overaspirate, stage1_mean,
                        stage2_overaspirate, stage2_mean,
                        volume_ml
                    )
                    
                    # Ensure reasonable bounds
                    optimal_overaspirate = max(0.0, min(0.020, optimal_overaspirate))  # Cap at 20uL
                    
                    print(f"      Optimized overaspirate: {optimal_overaspirate:.4f} mL")
                    
                    # Create Stage 3 parameters
                    # Get fresh optimized parameters for Stage 3
                    base_params = lash_e.nr_robot._get_optimized_parameters(
                        volume=volume_ml, 
                        liquid=liquid_type, 
                        user_parameters=parameters,
                        compensate_overvolume=compensate_overvolume,
                        smooth_overvolume=smooth_overvolume
                    )
                    
                    # Create Stage 3 parameters preserving all calibration
                    stage3_params = PipettingParameters()
                    for attr in dir(base_params):
                        if not attr.startswith('_') and hasattr(stage3_params, attr):
                            setattr(stage3_params, attr, getattr(base_params, attr))
                    
                    # Apply ONLY the optimized overaspirate
                    stage3_params.overaspirate_vol = optimal_overaspirate
                    
                    # Take Stage 3 measurements (3 replicates) 
                    stage3_volumes = []
                    for rep in range(replicates):
                        print(f"      Stage 3 replicate {rep + 1}/{replicates}...")
                        
                        stage3_result = lash_e.nr_robot.dispense_from_vial_into_vial(
                            source_vial_name=source_vial,
                            dest_vial_name=destination_vial,
                            volume=volume_ml,
                            parameters=stage3_params,
                            liquid=liquid_type,
                            remove_tip=switch_pipet,
                            use_safe_location=False,
                            return_vial_home=False,
                            compensate_overvolume=compensate_overvolume,
                            smooth_overvolume=smooth_overvolume,
                            measure_weight=True,
                            continuous_mass_monitoring=True,
                            save_mass_data=True
                        )
                        
                        if isinstance(stage3_result, tuple):
                            stage3_mass, _ = stage3_result
                        else:
                            stage3_mass = stage3_result
                        
                        # Handle simulation mode with safety checks
                        if (hasattr(lash_e, 'simulate') and lash_e.simulate and 
                            stage3_mass == 0 and getattr(lash_e, 'simulate', False) is True):
                            
                            import os
                            os.environ['ROBOT_SIMULATION_MODE'] = 'TRUE'
                            
                            stage3_mass = _generate_simulated_mass(
                                target_volume_ml=volume_ml,
                                density=density,
                                current_overaspirate=optimal_overaspirate,
                                replicate_num=rep + 20,  # Different seed for stage 3
                                validation_mode=True  # Use low-noise validation mode
                            )
                            print(f"        SIMULATION: Generated Stage 3 mass {stage3_mass:.6f}g")
                        elif stage3_mass == 0:
                            error_msg = (
                                f"Stage 3 hardware returned zero mass for {volume_ml*1000:.1f}uL. "
                                f"Check scale/hardware during final optimization."
                            )
                            lash_e.nr_robot.pause_after_error(error_msg, send_slack=True)
                            raise ValueError(error_msg)
                        
                        stage3_volume = stage3_mass / density
                        stage3_volumes.append(stage3_volume)
                        
                        # Store Stage 3 result
                        stage3_result_dict = {
                            'target_volume_ml': volume_ml,
                            'measured_volume_ml': stage3_volume,
                            'mass_difference_g': stage3_mass,
                            'density_used': density,
                            'liquid_type': liquid_type,
                            'replicate': rep + 1,
                            'timestamp': datetime.now().isoformat(),
                            'source_vial': source_vial,
                            'destination_vial': destination_vial,
                            'switch_pipet': switch_pipet,
                            'retry_count': 0,
                            'stability_info': None,
                            'volume_index': volume_idx,
                            'optimization_stage': 3
                        }
                        all_results.append(stage3_result_dict)
                        
                        print(f"        Target: {volume_ml*1000:.1f} uL -> Stage 3: {stage3_volume*1000:.1f} uL (Error: {((stage3_volume-volume_ml)/volume_ml)*100:.1f}%)")
                    
                    # Evaluate final Stage 3 performance
                    stage3_mean = np.mean(stage3_volumes)
                    stage3_error_ul = abs((stage3_mean * 1000) - target_ul)
                    print(f"    Stage 3 - Final error: {stage3_error_ul:.1f}uL vs {tolerance_ul:.1f}uL tolerance")
                    
                    # Log the parameter optimization
                    _log_parameter_correction(
                        timestamp, liquid_type, volume_ml,
                        initial_overaspirate, optimal_overaspirate,
                        "iterative_optimization", session_id, lash_e
                    )
                    
                    # Send Slack notification about optimization (skip in simulation)
                    if hasattr(lash_e, 'simulate') and lash_e.simulate:
                        correction_change_ul = (optimal_overaspirate - initial_overaspirate) * 1000
                        print(f"    SIMULATION: Would send Slack notification:")
                        print(f"    '3-Stage Optimization: {liquid_type} {volume_ml*1000:.1f}uL, overaspirate {initial_overaspirate:.4f}→{optimal_overaspirate:.4f}mL ({correction_change_ul:+.1f}uL)'")
                    else:
                        correction_change_ul = (optimal_overaspirate - initial_overaspirate) * 1000
                        slack_message = (
                            f"🎯 3-STAGE PARAMETER OPTIMIZATION\n"
                            f"Liquid: {liquid_type}\n"
                            f"Volume: {volume_ml*1000:.1f}uL\n"
                            f"Stage 1 Error: {stage1_error_ul:.1f}uL\n"
                            f"Stage 2 Error: {stage2_error_ul:.1f}uL\n" 
                            f"Stage 3 Error: {stage3_error_ul:.1f}uL\n"
                            f"Overaspirate: {initial_overaspirate:.4f} → {optimal_overaspirate:.4f} mL\n"
                            f"Session: {session_id}"
                        )
                        
                        try:
                            import slack_agent
                            slack_agent.send_slack_message(slack_message)
                            print(f"    📱 Slack optimization notification sent")
                        except Exception as slack_error:
                            print(f"    ⚠ Could not send Slack notification: {slack_error}")
                    
                    print(f"    ✅ 3-stage iterative optimization completed")
                    
                except Exception as e:
                    print(f"    Warning: Iterative optimization failed: {e}")
                    print(f"    Continuing with Stage 1 measurements...")
                    # Mark failed optimization attempts
                    for result in all_results:
                        if result['target_volume_ml'] == volume_ml and 'optimization_stage' not in result:
                            result['optimization_stage'] = 1
            else:
                print(f"    ✅ Stage 1 measurements within tolerance - no optimization needed")
    
    # === DATA ANALYSIS ===
    df = pd.DataFrame(all_results)
    
    # === STAGE-FILTERED ANALYSIS FOR ADAPTIVE CORRECTION ===
    if adaptive_correction:
        # For adaptive correction, use only the highest stage results for each volume
        final_results = []
        for volume in volumes_ml:
            volume_results = df[df['target_volume_ml'] == volume]
            
            if 'optimization_stage' in volume_results.columns:
                # Find the highest stage for this volume
                max_stage = volume_results['optimization_stage'].max()
                final_stage_results = volume_results[volume_results['optimization_stage'] == max_stage]
                final_results.extend(final_stage_results.to_dict('records'))
                
                stage_count = len(final_stage_results)
                print(f"    Volume {volume*1000:.1f}uL: Using Stage {max_stage} results ({stage_count} measurements)")
            else:
                # Fallback if no stage info
                final_results.extend(volume_results.to_dict('records'))
        
        df_analysis = pd.DataFrame(final_results)
        print(f"    Final validation analysis based on {len(df_analysis)} optimized measurements")
    else:
        # For non-adaptive correction, use all results
        df_analysis = df
        print(f"    Standard validation analysis based on {len(df_analysis)} measurements")
    
    # === ZERO-VOLUME DETECTION ===
    # Check if all measurements are significantly below target (< 10%)  
    df_analysis['volume_ratio'] = df_analysis['measured_volume_ml'] / df_analysis['target_volume_ml']
    all_measurements_low = (df_analysis['volume_ratio'] < 0.10).all()
    
    if all_measurements_low:
        mean_ratio = df_analysis['volume_ratio'].mean()
        min_ratio = df_analysis['volume_ratio'].min()
        lash_e.nr_robot.pause_after_error(
            f"VALIDATION FAILURE: All measurements below 10% of target. "
            f"Mean ratio: {mean_ratio:.1%}, Min ratio: {min_ratio:.1%}. "
            f"Check for pipetting hardware issues or empty source vial.",
            send_slack=True
        )
    
    # Calculate statistics per volume (using filtered results)
    stats_per_volume = []
    for volume in volumes_ml:
        subset = df_analysis[df_analysis['target_volume_ml'] == volume]
        
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
            df_analysis['target_volume_ml'], df_analysis['measured_volume_ml']
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
        'total_measurements': len(df_analysis)
    }
    
    # === GENERATE PLOTS ===
    try:
        plot_file = _create_validation_plots(
            df_analysis=df_analysis,  # Filtered data for calculations
            df_raw=df,               # Raw data for multi-stage visualization 
            stats_df=stats_df, 
            overall_stats=overall_stats, 
            output_folder=calib_folder, 
            plot_title=plot_title, 
            liquid_type=liquid_type,
            adaptive_correction=adaptive_correction  # Pass adaptive flag for plotting logic
        )
    except Exception as e:
        print(f"Warning: Could not create validation plots: {e}")
        plot_file = None
    
    # === SAVE DATA ===
    data_files = {}
    
    if save_raw_data:
        try:
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
        except Exception as e:
            print(f"Warning: Could not save some data files: {e}")
    
    # === PREPARE RETURN DATA ===
    results = {
        'success': True,
        'session_id': session_id,
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
            'timestamp': timestamp,
            'parameters_used': param_values
        }
    }
    
    # === UPDATE MASTER DATABASES ===
    print("\\n=== Updating Master Databases ===")
    
    # Prepare individual measurements data
    measurements_data = []
    for idx, row in df.iterrows():
        # Individual measurements don't have precision - that's a group property
        # Precision will be calculated separately for volume groups
        
        # Calculate accuracy for this measurement
        accuracy_pct = ((row['measured_volume_ml'] - row['target_volume_ml']) / row['target_volume_ml']) * 100
        
        measurement_record = {
            'timestamp': timestamp,
            'session_id': session_id,
            'validation_type': 'vial_to_vial',
            'liquid_type': liquid_type,
            'density_used': density,
            'target_volume_ml': row['target_volume_ml'],
            'measured_volume_ml': row['measured_volume_ml'],
            'accuracy_pct': accuracy_pct,
            'replicate_number': row['replicate'],
            'volume_index': row['volume_index'],
            'source_vial': source_vial,
            'destination_vial': destination_vial,
            **param_values  # Add all parameter columns
        }
        measurements_data.append(measurement_record)
    
    _append_to_master_measurements(measurements_data, "pipetting")
    
    # Prepare session summary data - separate records for each volume
    for idx, volume_stat in enumerate(stats_per_volume):
        session_record = {
            'timestamp': timestamp,
            'session_id': session_id,
            'validation_type': 'vial_to_vial',
            'liquid_type': liquid_type,
            'density_used': density,
            'target_volume_ml': volume_stat['target_volume_ml'],
            'num_replicates': volume_stat['n_replicates'],
            'accuracy_pct': volume_stat['accuracy_pct'],
            'precision_cv_pct': volume_stat['precision_cv_pct'],
            'measured_mean_ml': volume_stat['measured_mean_ml'],
            'measured_std_ml': volume_stat['measured_std_ml'],
            'r_squared': r_squared,  # Overall fit still applies to all volumes
            'linear_slope': slope,
            'linear_intercept': intercept,
            'output_folder_path': calib_folder,
            'source_vial': source_vial,
            'destination_vial': destination_vial,
            **param_values  # Add all parameter columns
        }
        
        _append_to_master_sessions(session_record, "pipetting")
    
    # === CLEANUP ===
    # Return destination vial to home position at the very end
    try:
        lash_e.nr_robot.remove_pipet()  # Ensure pipet is removed first
        lash_e.nr_robot.return_vial_home(destination_vial)  # Return by vial name
        print(f"Returned {destination_vial} to home position")
    except Exception as e:
        print(f"Warning: Could not return vial to home: {e}")
    
    # === PRINT SUMMARY ===
    print(f"\n=== Vial-to-Vial Validation Results (Session: {session_id}) ===")
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
    save_raw_data: bool = True,
    quality_std_threshold: float = 0.001,
    adaptive_correction: bool = False
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
        quality_std_threshold: Standard deviation threshold for quality control (default: 0.001g = 1mg)
        adaptive_correction: Enable dynamic overaspirate correction during validation (default: False)
        
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
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Validate target vial exists
    try:
        lash_e.nr_robot.get_vial_info(target_vial, 'location')
    except Exception as e:
        raise FileNotFoundError(f"Target vial validation failed: {e}")
    
    # === SETUP OUTPUT DIRECTORY ===
    # Generate unique session ID
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calib_folder = os.path.join(output_folder, "reservoir_validation", f"validation_{timestamp}")
    os.makedirs(calib_folder, exist_ok=True)
    
    print(f"\\n=== Reservoir Validation ===\\nSession ID: {session_id}")
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
            
            # Quality control retry loop
            max_retries = 3
            retry_count = 0
            measurement_acceptable = False
            
            while not measurement_acceptable and retry_count <= max_retries:
                # Dispense from reservoir with continuous monitoring
                transfer_result = lash_e.nr_robot.dispense_into_vial_from_reservoir(
                    reservoir_index=reservoir_index,
                    vial_index=target_vial,
                    volume=volume_ml,
                    measure_weight=True,
                    continuous_mass_monitoring=True,
                    save_mass_data=True
                )
                
                # Handle new return format with quality control
                if isinstance(transfer_result, tuple) and len(transfer_result) == 2:
                    mass_difference, stability_info = transfer_result
                    measurement_acceptable = _evaluate_measurement(stability_info, quality_std_threshold)
                else:
                    # Backwards compatibility - old format returns just mass
                    mass_difference = transfer_result
                    stability_info = None
                    measurement_acceptable = True
                
                retry_count += 1
                
                if not measurement_acceptable and retry_count <= max_retries:
                    print(f"    Measurement quality insufficient, retrying...")
            
            if not measurement_acceptable:
                print(f"    WARNING: Measurement still poor quality after {max_retries} retries")

            # Handle simulation mode - generate realistic simulated mass for reservoir
            if hasattr(lash_e, 'simulate') and lash_e.simulate and mass_difference == 0:
                import random
                # Reservoir dispensing typically more accurate than pipetting
                base_accuracy = random.uniform(0.95, 1.02)  # 95-102% accuracy
                volume_ul = volume_ml * 1000
                if volume_ul > 500:  # Large volumes more accurate
                    noise_factor = random.uniform(-0.01, 0.01)  # ±1% noise
                else:
                    noise_factor = random.uniform(-0.03, 0.03)  # ±3% noise
                
                simulated_volume = volume_ml * (base_accuracy + noise_factor)
                simulated_volume = max(0, simulated_volume)
                mass_difference = simulated_volume * density
                print(f"    SIMULATION: Generated reservoir mass {mass_difference:.6f}g for testing")

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
                'reservoir_index': reservoir_index,
                'target_vial': target_vial,
                'retry_count': retry_count,
                'stability_info': stability_info,
                'volume_index': volume_idx  # Track which volume in the list
            }
            all_results.append(result)
            
            print(f"    Target: {volume_ml*1000:.1f} uL -> Measured: {measured_volume_ml*1000:.1f} uL (Error: {((measured_volume_ml-volume_ml)/volume_ml)*100:.1f}%)")
        
        # === ADAPTIVE CORRECTION (OPTIONAL) ===
        if adaptive_correction:
            # Calculate mean measured volume for this target
            volume_results = [r for r in all_results if r['target_volume_ml'] == volume_ml]
            measured_volumes = [r['measured_volume_ml'] for r in volume_results]
            mean_measured = np.mean(measured_volumes)
            
            # Calculate tolerance for this volume
            tolerance_ul = _calculate_volume_tolerance_ul(volume_ml)
            target_ul = volume_ml * 1000
            mean_measured_ul = mean_measured * 1000
            error_ul = abs(mean_measured_ul - target_ul)
            
            print(f"    Tolerance check: {error_ul:.1f}uL error vs {tolerance_ul:.1f}uL tolerance")
            
            if error_ul > tolerance_ul:
                print(f"    Warning: Mean accuracy outside tolerance")
                print(f"    Note: Reservoir correction requires pump calibration - logging for manual review")
                
                # Log the correction need for reservoir (different approach needed)
                try:
                    _log_parameter_correction(
                        timestamp, liquid_type, volume_ml,
                        0.0,  # Reservoir doesn't use overaspirate
                        0.0,  # Would need pump flow rate correction
                        "reservoir", session_id, lash_e
                    )
                except Exception as e:
                    print(f"    Warning: Could not log reservoir correction: {e}")
            else:
                print(f"    Checkmark: Measurements within tolerance - no correction needed")
    
    # === DATA ANALYSIS ===
    df = pd.DataFrame(all_results)
    
    # === ZERO-VOLUME DETECTION ===
    # Check if all measurements are significantly below target (< 10%)
    df['volume_ratio'] = df['measured_volume_ml'] / df['target_volume_ml']
    all_measurements_low = (df['volume_ratio'] < 0.10).all()
    
    if all_measurements_low:
        mean_ratio = df['volume_ratio'].mean()
        min_ratio = df['volume_ratio'].min()
        lash_e.nr_robot.pause_after_error(
            f"RESERVOIR VALIDATION FAILURE: All measurements below 10% of target. "
            f"Mean ratio: {mean_ratio:.1%}, Min ratio: {min_ratio:.1%}. "
            f"Check for reservoir connections or pump issues.",
            send_slack=True
        )
    
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
    try:
        plot_file = _create_validation_plots(
            df_analysis=df,      # No filtering for reservoir validation
            df_raw=df,          # Same data for both
            stats_df=stats_df, 
            overall_stats=overall_stats, 
            output_folder=calib_folder, 
            plot_title=plot_title, 
            liquid_type=liquid_type,
            adaptive_correction=False  # Reservoir validation doesn't use adaptive correction
        )
    except Exception as e:
        print(f"Warning: Could not create validation plots: {e}")
        plot_file = None
    
    # === SAVE DATA ===
    data_files = {}
    
    if save_raw_data:
        try:
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
        except Exception as e:
            print(f"Warning: Could not save some data files: {e}")
    
    # === PREPARE RETURN DATA ===
    results = {
        'success': True,
        'session_id': session_id,
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
    
    # === UPDATE MASTER DATABASES ===
    print("\n=== Updating Master Databases ===")
    
    # Get reservoir parameter defaults (ReservoirParameters)
    reservoir_param_values = _get_parameter_values(None, "reservoir")  # Always use defaults for reservoir
    
    # Prepare individual measurements data
    measurements_data = []
    for idx, row in df.iterrows():
        # Individual measurements don't have precision - that's a group property
        # Precision will be calculated separately for volume groups
        
        # Calculate accuracy for this measurement
        accuracy_pct = ((row['measured_volume_ml'] - row['target_volume_ml']) / row['target_volume_ml']) * 100
        
        measurement_record = {
            'timestamp': timestamp,
            'session_id': session_id,
            'validation_type': 'reservoir_dispensing',
            'liquid_type': liquid_type,
            'density_used': density,
            'target_volume_ml': row['target_volume_ml'],
            'measured_volume_ml': row['measured_volume_ml'],
            'accuracy_pct': accuracy_pct,
            'replicate_number': row['replicate'],
            'volume_index': row['volume_index'],
            'source_vial': None,
            'destination_vial': target_vial,
            'reservoir_index': reservoir_index,
            **reservoir_param_values  # Add all reservoir parameter columns
        }
        measurements_data.append(measurement_record)
    
    _append_to_master_measurements(measurements_data, "reservoir")
    
    # Prepare session summary data - separate records for each volume
    for idx, volume_stat in enumerate(stats_per_volume):
        session_record = {
            'timestamp': timestamp,
            'session_id': session_id,
            'validation_type': 'reservoir_dispensing',
            'liquid_type': liquid_type,
            'density_used': density,
            'target_volume_ml': volume_stat['target_volume_ml'],
            'num_replicates': volume_stat['n_replicates'],
            'accuracy_pct': volume_stat['accuracy_pct'],
            'precision_cv_pct': volume_stat['precision_cv_pct'],
            'measured_mean_ml': volume_stat['measured_mean_ml'],
            'measured_std_ml': volume_stat['measured_std_ml'],
            'r_squared': r_squared,  # Overall fit still applies to all volumes
            'linear_slope': slope,
            'linear_intercept': intercept,
            'output_folder_path': calib_folder,
            'source_vial': None,
            'destination_vial': target_vial,
            'reservoir_index': reservoir_index,
            **reservoir_param_values  # Add all reservoir parameter columns
        }
        
        _append_to_master_sessions(session_record, "reservoir")
    
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
    print(f"\n=== Reservoir Validation Results (Session: {session_id}) ===")
    print(f"R² = {r_squared:.4f}")
    print(f"Mean accuracy: {overall_stats['mean_accuracy_pct']:.2f}%")
    print(f"Mean precision (CV): {overall_stats['mean_precision_cv_pct']:.2f}%")
    print(f"Linear fit: y = {slope:.4f}x + {intercept:.6f}")
    print(f"Results saved to: {calib_folder}")
    
    return results

def _create_validation_plots(df_analysis: pd.DataFrame, df_raw: pd.DataFrame, stats_df: pd.DataFrame, 
                           overall_stats: dict, output_folder: str, plot_title: Optional[str], 
                           liquid_type: str, adaptive_correction: bool = False) -> str:
    """Create validation plots and save to file with multi-stage visualization support."""
    
    # Use analysis data for calculations, raw data for visualization
    df = df_analysis.copy()  # For calculations and backwards compatibility
    
    # Convert to uL for plotting
    df['target_ul'] = df['target_volume_ml'] * 1000
    df['measured_ul'] = df['measured_volume_ml'] * 1000
    
    # Prepare raw data for multi-stage visualization if available
    has_stage_info = adaptive_correction and 'optimization_stage' in df_raw.columns
    if has_stage_info:
        try:
            df_raw_plot = df_raw.copy()
            df_raw_plot['target_ul'] = df_raw_plot['target_volume_ml'] * 1000
            df_raw_plot['measured_ul'] = df_raw_plot['measured_volume_ml'] * 1000
        except Exception as e:
            print(f"    Warning: Multi-stage plotting preparation failed, using standard plots: {e}")
            df_raw_plot = None
            has_stage_info = False
    else:
        df_raw_plot = None
    
    # Check if this is a single-volume calibration
    unique_volumes = df['target_volume_ml'].nunique()
    is_single_volume = (unique_volumes == 1)
    
    if is_single_volume:
        # Single volume case - use histogram-focused layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        if plot_title:
            fig.suptitle(f"{plot_title} - {liquid_type.title()} Single-Volume Validation", fontsize=16)
        else:
            fig.suptitle(f"Single-Volume Pipetting Validation - {liquid_type.title()}", fontsize=16)
        
        target_volume_ul = df['target_ul'].iloc[0]  # All targets are the same
        
        # Plot 1: Histogram of measured volumes with target reference
        ax1.hist(df['measured_ul'], bins=max(5, len(df)//2), alpha=0.7, edgecolor='black', color='skyblue')
        ax1.axvline(x=target_volume_ul, color='red', linestyle='--', linewidth=2, 
                   label=f'Target: {target_volume_ul:.1f}uL')
        ax1.axvline(x=df['measured_ul'].mean(), color='orange', linestyle='-', linewidth=2,
                   label=f'Mean: {df["measured_ul"].mean():.1f}uL')
        ax1.set_xlabel('Measured Volume (uL)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Measured Volumes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Individual measurements with target line
        measurement_indices = range(1, len(df) + 1)
        ax2.scatter(measurement_indices, df['measured_ul'], s=60, alpha=0.8, color='blue')
        ax2.axhline(y=target_volume_ul, color='red', linestyle='--', linewidth=2, 
                   label=f'Target: {target_volume_ul:.1f}uL')
        ax2.axhline(y=df['measured_ul'].mean(), color='orange', linestyle='-', linewidth=1,
                   label=f'Mean: {df["measured_ul"].mean():.1f}uL')
        ax2.set_xlabel('Measurement #')
        ax2.set_ylabel('Measured Volume (uL)')
        ax2.set_title('Individual Measurements')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy and precision summary
        accuracy_pct = stats_df['accuracy_pct'].iloc[0]
        precision_cv_pct = stats_df['precision_cv_pct'].iloc[0]
        
        metrics = ['Accuracy (%)', 'Precision CV (%)']
        values = [accuracy_pct, precision_cv_pct]
        colors = ['green' if accuracy_pct > -5 else 'orange' if accuracy_pct > -10 else 'red',
                 'green' if precision_cv_pct < 5 else 'orange' if precision_cv_pct < 10 else 'red']
        
        bars = ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Performance Summary')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Plot 4: Error from target
        df['error_ul'] = df['measured_ul'] - target_volume_ul
        ax4.hist(df['error_ul'], bins=max(5, len(df)//2), alpha=0.7, edgecolor='black', color='lightcoral')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect accuracy')
        ax4.axvline(x=df['error_ul'].mean(), color='orange', linestyle='-', linewidth=2,
                   label=f'Mean error: {df["error_ul"].mean():.1f}uL')
        ax4.set_xlabel('Error from Target (uL)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Error Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
    else:
        # Multi-volume case - use existing scatter plot approach
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        if plot_title:
            fig.suptitle(f"{plot_title} - {liquid_type.title()} Validation", fontsize=16)
        else:
            fig.suptitle(f"Pipetting Validation - {liquid_type.title()}", fontsize=16)
        
        # Plot 1: Target vs Measured with perfect line and multi-stage visualization
        if has_stage_info and df_raw_plot is not None:
            try:
                # Multi-stage visualization - show progression with different colors/markers
                stage_colors = {1: '#ffcccc', 2: '#ffaa44', 3: '#44aa44'}  # Light red, orange, green
                stage_markers = {1: 'o', 2: '^', 3: 's'}  # Circle, triangle, square
                stage_labels = {1: 'Stage 1 (Initial)', 2: 'Stage 2 (1st Optimization)', 3: 'Stage 3 (Final Optimization)'}
                
                # Plot all stages with different styling
                for stage in sorted(df_raw_plot['optimization_stage'].unique()):
                    stage_data = df_raw_plot[df_raw_plot['optimization_stage'] == stage]
                    ax1.scatter(stage_data['target_ul'], stage_data['measured_ul'], 
                              color=stage_colors.get(stage, '#888888'), 
                              marker=stage_markers.get(stage, 'o'),
                              alpha=0.6, s=45, 
                              label=stage_labels.get(stage, f'Stage {stage}'),
                              edgecolor='black', linewidth=0.5)
                
                # Overlay final results prominently for clarity
                ax1.scatter(df['target_ul'], df['measured_ul'], 
                          color='darkgreen', marker='s', s=80, alpha=0.9,
                          label='Final Results (used for validation)', 
                          edgecolor='black', linewidth=1.5, zorder=5)
            except Exception as e:
                print(f"    Warning: Multi-stage visualization failed, using standard plot: {e}")
                # Fallback to standard visualization
                ax1.scatter(df['target_ul'], df['measured_ul'], alpha=0.7, s=50, 
                          color='blue', label='Measurements')
        else:
            # Standard single-stage visualization
            ax1.scatter(df['target_ul'], df['measured_ul'], alpha=0.7, s=50, 
                      color='blue', label='Measurements')
        
        # Add perfect line
        min_vol = min(df['target_ul'].min(), df['measured_ul'].min())
        max_vol = max(df['target_ul'].max(), df['measured_ul'].max())
        ax1.plot([min_vol, max_vol], [min_vol, max_vol], 'r--', label='Perfect accuracy', linewidth=2)
        
        # Add regression line
        slope, intercept = overall_stats['slope'], overall_stats['intercept']
        x_fit = np.array([min_vol, max_vol]) / 1000  # Convert back to mL for calculation
        y_fit = (slope * x_fit + intercept) * 1000  # Convert result to uL
        ax1.plot(x_fit * 1000, y_fit, 'b-', label=f'Linear fit (R² = {overall_stats["r_squared"]:.4f})', linewidth=2)
        
        ax1.set_xlabel('Target Volume (uL)')
        ax1.set_ylabel('Measured Volume (uL)')
        if has_stage_info:
            ax1.set_title('Accuracy (Multi-Stage Optimization)')
        else:
            ax1.set_title('Accuracy')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left') if has_stage_info else ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy per volume
        ax2.bar(stats_df['target_volume_ul'], stats_df['accuracy_pct'])
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Target Volume (uL)')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy by Volume')
        ax2.grid(True, alpha=0.3)
        
        # Add multi-stage progression info if available
        if has_stage_info and df_raw_plot is not None:
            try:
                # Create stage progression summary
                stage_summary = []
                for volume in sorted(df['target_volume_ml'].unique()):
                    volume_stages = df_raw_plot[df_raw_plot['target_volume_ml'] == volume]['optimization_stage']
                    max_stage = volume_stages.max()
                    if max_stage > 1:
                        stage_summary.append(f"{volume*1000:.0f}uL: Stage {max_stage}")
                    else:
                        stage_summary.append(f"{volume*1000:.0f}uL: Stage {max_stage} ✓")
                
                # Add text box with stage progression
                if stage_summary:
                    progression_text = "Optimization Stages:\\n" + "\\n".join(stage_summary)
                    ax2.text(0.02, 0.98, progression_text, transform=ax2.transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', 
                            facecolor='wheat', alpha=0.8), fontsize=9)
            except Exception as e:
                print(f"    Warning: Stage progression text failed, continuing with standard plot: {e}")
                # Continue without the text box - plot still works
        
        # Plot 3: Precision (CV) per volume
        ax3.bar(stats_df['target_volume_ul'], stats_df['precision_cv_pct'])
        ax3.set_xlabel('Target Volume (uL)')
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
    
    # Save plot with error protection
    plot_file = os.path.join(output_folder, "validation_plots.png")
    try:
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"    Warning: Could not save plot to {plot_file}: {e}")
        plt.close()  # Still close the figure to free memory
        plot_file = None  # Return None if saving failed
    
    return plot_file
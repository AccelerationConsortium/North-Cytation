# calibration_sdl_base.py
import sys
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global variable to track last vial management call to prevent spam
_last_vial_mgmt_call = 0
_VIAL_MGMT_THROTTLE_SEC = 1.0  # Minimum time between calls
sys.path.append("../utoronto_demo")

from pipetting_data.pipetting_parameters import PipettingParameters

# Conditionally import analyzer only when needed
try:
    import analysis.calibration_analyzer as analyzer
    ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import analyzer module: {e}")
    print("Analysis functions will be disabled. This is fine for simulation mode.")
    analyzer = None
    ANALYZER_AVAILABLE = False

LIQUIDS = {
    "water": {"density": 1.00, "refill_pipets": False},
    "ethanol": {"density": 0.789, "refill_pipets": False},
    "toluene": {"density": 0.867, "refill_pipets": False},
    "2MeTHF": {"density": 0.852, "refill_pipets": False},
    "isopropanol": {"density": 0.789, "refill_pipets": False},
    "DMSO": {"density": 1.1, "refill_pipets": False},
    "acetone": {"density": 0.79, "refill_pipets": False},
    "glycerol": {"density": 1.26, "refill_pipets": True},
    "PEG_Water": {"density": 1.05, "refill_pipets": True},
    "4%_hyaluronic_acid_water": {"density": 1.01, "refill_pipets": True},
}

# --- Utility Functions ---
def normalize_parameters(params):
    """
    Normalize parameter names to handle different parameter naming conventions.
    
    Sometimes we get 'blowout_vol', sometimes 'pre_asp_air_vol'.
    This function ensures consistent parameter names for simulation.
    """
    # Handle None params by creating empty dict
    if params is None:
        params = {}
    
    normalized = params.copy()
    
    # Handle blowout_vol vs pre_asp_air_vol (only map if both are missing)
    if 'pre_asp_air_vol' in params and 'blowout_vol' not in params:
        normalized['blowout_vol'] = params['pre_asp_air_vol']
    elif 'blowout_vol' in params and 'pre_asp_air_vol' not in params:
        # Don't automatically add pre_asp_air_vol - let the calling code decide
        pass
    
    # Ensure all expected parameters exist with defaults
    defaults = {
        'aspirate_speed': 20,
        'dispense_speed': 20, 
        'aspirate_wait_time': 2.0,
        'dispense_wait_time': 2.0,
        'retract_speed': 8.0,
        'blowout_vol': 0.05,
        'post_asp_air_vol': 0.05,
        'overaspirate_vol': 0.01
    }
    
    for param, default_value in defaults.items():
        if param not in normalized:
            normalized[param] = default_value
    
    return normalized

def pipet_and_measure_simulated(volume, params, expected_mass, expected_time):
    time.sleep(0.2)
    
    # Normalize parameters to handle different naming conventions
    params = normalize_parameters(params)
    
    # ENHANCED SIMULATION BASED ON calibration_protocol_simulated.py
    # More realistic physics with improved overaspirate effectiveness
    
    # Extract parameters with fallbacks
    asp_speed = params.get("aspirate_speed", 15)
    dsp_speed = params.get("dispense_speed", 15)
    asp_wait = params.get("aspirate_wait_time", 5)
    dsp_wait = params.get("dispense_wait_time", 5)
    blowout_vol = params.get("blowout_vol", 0.06)
    post_asp_air_vol = params.get("post_asp_air_vol", 0.06)
    retract_speed = params.get("retract_speed", 10)
    overasp_vol = params.get("overaspirate_vol", 0.0)
    
    # 1. Systematic under-delivery bias (5%)
    underdelivery_bias = -0.05  # 5% systematic underdelivery
    
    # 2. Overaspirate compensation: IMPROVED 80% effectiveness (was 70%)
    overasp_absolute_compensation = overasp_vol * 0.8  # 80% effectiveness in absolute terms
    overasp_relative_compensation = overasp_absolute_compensation / volume if volume > 0 else 0
    
    # 3. Parameter effects - both penalties AND bonuses
    parameter_effect = 0.0
    
    # Speed effects: optimal speeds provide accuracy bonus
    if 20 <= asp_speed <= 25:  # Optimal aspirate speed range
        parameter_effect += 0.015  # Accuracy bonus for optimal speed
    elif asp_speed < 18 or asp_speed > 28:
        parameter_effect -= 0.02  # Poor aspirate speed penalty
        
    if 16 <= dsp_speed <= 20:  # Optimal dispense speed range  
        parameter_effect += 0.012  # Accuracy bonus for optimal speed
    elif dsp_speed < 14 or dsp_speed > 22:
        parameter_effect -= 0.015  # Poor dispense speed penalty
        
    # Wait time effects: optimal waits provide bonus
    if 4 <= asp_wait <= 6:  # Optimal aspirate wait range
        parameter_effect += 0.008  # Accuracy bonus
    elif asp_wait < 3:
        parameter_effect -= 0.01  # Too short penalty
        
    if 2.5 <= dsp_wait <= 4:  # Optimal dispense wait range
        parameter_effect += 0.006  # Accuracy bonus  
    elif dsp_wait < 2:
        parameter_effect -= 0.008  # Too short penalty
        
    # Volume parameter effects: optimal values provide bonuses
    if abs(blowout_vol - 0.06) < 0.02:  # Near optimal ~0.06
        parameter_effect += 0.008  # Accuracy bonus
    elif abs(blowout_vol - 0.06) > 0.04:  # Far from optimal
        parameter_effect -= 0.01  # Penalty
        
    if abs(post_asp_air_vol - 0.06) < 0.02:  # Near optimal ~0.06
        parameter_effect += 0.010  # Accuracy bonus
    elif abs(post_asp_air_vol - 0.06) > 0.03:  # Far from optimal
        parameter_effect -= 0.012  # Penalty
        
    if abs(retract_speed - 10) < 2:  # Near optimal ~10
        parameter_effect += 0.006  # Accuracy bonus
    elif abs(retract_speed - 10) > 4:  # Far from optimal
        parameter_effect -= 0.008  # Penalty
    
    # 4. Total error: bias + overaspirate compensation + parameter effects  
    total_error = underdelivery_bias + overasp_relative_compensation + parameter_effect
    
    # 5. Soft saturation: use tanh to prevent extreme values but preserve differences
    # Allow wider range (-50% to +50%) but compress extreme values
    if total_error > 0:
        final_error = 0.5 * np.tanh(total_error / 0.3)  # Positive errors compressed more gently
    else:
        final_error = 0.5 * np.tanh(total_error / 0.3)  # Negative errors same treatment
    
    # 6. Parameter-dependent replicate noise instead of fixed noise
    # Better parameters = more consistent results, worse parameters = more variable
    replicate_noise_std = 0.003  # Reduced base noise
    
    # Speed consistency: extreme speeds reduce replicate consistency  
    if asp_speed < 18 or asp_speed > 28:
        replicate_noise_std += 0.004
    if dsp_speed < 14 or dsp_speed > 22:
        replicate_noise_std += 0.003
        
    # Wait time consistency: too short = inconsistent results
    if asp_wait < 5:
        replicate_noise_std += 0.003
    if dsp_wait < 3:
        replicate_noise_std += 0.002
        
    # Volume parameter consistency
    if abs(blowout_vol - 0.06) > 0.04:  # Far from optimal ~0.06
        replicate_noise_std += 0.002
    if abs(post_asp_air_vol - 0.06) > 0.03:  # Far from optimal ~0.06
        replicate_noise_std += 0.003
    if abs(retract_speed - 10) > 4:  # Far from optimal ~10
        replicate_noise_std += 0.002
    
    # Apply parameter-dependent replicate noise
    replicate_noise = np.random.normal(0, replicate_noise_std)
    final_error_with_noise = final_error + replicate_noise
    
    # Generate simulated mass with enhanced error model
    simulated_mass = expected_mass * (1 + final_error_with_noise)
    simulated_mass = max(simulated_mass, 0)  # Can't have negative mass
    
    # ENHANCED Realistic time simulation with stronger parameter effects
    baseline = 12.0  # Base pipetting time in seconds
    
    # Wait times ALWAYS add to the time (no "optimal" value for time)
    wait_time_penalty = asp_wait + dsp_wait
    
    # ENHANCED Speed penalties: Higher numbers (like 35) are SLOWER, lower numbers (like 10) are FASTER
    # More realistic time differences to make speed optimization meaningful
    aspirate_time_penalty = (asp_speed - 10) * 0.8  # Increased from 0.3 to 0.8s per speed unit
    dispense_time_penalty = (dsp_speed - 10) * 0.6  # Increased from 0.3 to 0.6s per speed unit
    
    # Retract speed also affects time
    retract_time_penalty = (retract_speed - 8) * 0.4  # 0.4s per speed unit above 8
    
    # Calculate total time with enhanced parameter sensitivity
    total_time = baseline + wait_time_penalty + aspirate_time_penalty + dispense_time_penalty + retract_time_penalty
    
    # Parameter-dependent timing variability instead of fixed noise
    # Poor parameters = more inconsistent timing
    timing_noise_std = 0.3  # Reduced base timing noise
    
    # Extreme speeds create more timing variability
    if asp_speed < 15 or asp_speed > 30:
        timing_noise_std += 0.4
    if dsp_speed < 12 or dsp_speed > 25:
        timing_noise_std += 0.3
    if retract_speed < 6 or retract_speed > 15:
        timing_noise_std += 0.2
        
    # Very short wait times create timing inconsistency
    if asp_wait < 3:
        timing_noise_std += 0.3
    if dsp_wait < 2:
        timing_noise_std += 0.2
    
    # Apply parameter-dependent timing noise
    total_time += np.random.normal(0, timing_noise_std)
    time_score = max(total_time, 2.0)  # Minimum 2 seconds
    
    # Ensure no NaN values are returned
    time_score = np.nan_to_num(time_score, nan=50.0)
    
    return {"time": time_score, "simulated_mass": simulated_mass}

def empty_vial_if_needed(lash_e, vial_name, state):
    volume = lash_e.nr_robot.get_vial_info(vial_name, 'vial_volume')
    if volume > 7.0:
        lash_e.nr_robot.remove_pipet()
        disp_volume = volume / np.ceil(volume)
        for _ in range(int(np.ceil(volume)) - 1):
            waste_vial_name = f"waste_vial_{state['waste_vial_index']}"
            if lash_e.nr_robot.get_vial_info(waste_vial_name, 'vial_volume') + disp_volume > 18.0:
                state['waste_vial_index'] += 1
                waste_vial_name = f"waste_vial_{state['waste_vial_index']}"
            lash_e.nr_robot.dispense_from_vial_into_vial(vial_name, waste_vial_name, disp_volume)
        lash_e.nr_robot.remove_pipet()

def fill_liquid_if_needed(lash_e, vial_name, liquid_source_name):
    volume = lash_e.nr_robot.get_vial_info(liquid_source_name, 'vial_volume')
    if volume < 4.0:
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.return_vial_home(vial_name)
        lash_e.nr_robot.dispense_into_vial_from_reservoir(1, liquid_source_name, 8 - volume)
        lash_e.nr_robot.return_vial_home(liquid_source_name)
        lash_e.nr_robot.move_vial_to_location(vial_name, "clamp", 0)

def pipet_and_measure(lash_e, source_vial, dest_vial, volume, params, expected_measurement, expected_time, replicate_count, simulate, raw_path, raw_measurements, liquid, new_pipet_each_time, trial_type="UNKNOWN", liquid_for_params=None):
    # Determine which liquid to use for parameter optimization
    # liquid_for_params takes precedence - can be None to force defaults
    param_liquid = liquid_for_params
    
    # DEBUG: Print parameter resolution only when different from normal case
    if liquid_for_params != liquid:
        print(f"[PARAM DEBUG] liquid={liquid}, liquid_for_params={liquid_for_params}, param_liquid={param_liquid}")
    
    # Use intelligent parameter system if params=None (allows liquid=None for default comparison)
    if params is None:
        try:
            # Get optimized parameters from the robot's intelligent system
            # param_liquid=None will get defaults, param_liquid="water" will get optimized parameters
            optimized_params = lash_e.nr_robot._get_optimized_parameters(volume, param_liquid)
            if liquid_for_params != liquid:
                print(f"[PARAM DEBUG] Got optimized params from robot for param_liquid={param_liquid}")
            # Convert PipettingParameters object to dict for normalize_parameters
            params = {
                'aspirate_speed': optimized_params.aspirate_speed,
                'dispense_speed': optimized_params.dispense_speed,
                'aspirate_wait_time': optimized_params.aspirate_wait_time,
                'dispense_wait_time': optimized_params.dispense_wait_time,
                'retract_speed': optimized_params.retract_speed,
                'blowout_vol': optimized_params.blowout_vol,
                'post_asp_air_vol': optimized_params.post_asp_air_vol,
                'overaspirate_vol': optimized_params.overaspirate_vol
            }
        except Exception as e:
            print(f"Warning: Could not get optimized parameters for {param_liquid}: {e}")
            params = None  # Fall back to defaults
    
    # Normalize parameters to handle different naming conventions
    params = normalize_parameters(params)
    
    # DEBUG: Show final parameters only when investigating parameter issues
    if liquid_for_params != liquid:
        print(f"[PARAM DEBUG] Final parameters being used: {params}")
    # Removed duplicate print(params) - too verbose

    blowout_vol = params.get("blowout_vol", 0.0)  # Default blowout volume
    post_air = params.get("post_asp_air_vol", 0)
    over_volume = params.get("overaspirate_vol", 0)
    #over_volume = 0
    air_vol = post_air  # Only post_asp_air_vol contributes to air_vol now

    print(params)
    
    # Create PipettingParameters objects instead of kwargs dictionaries
    aspirate_params = PipettingParameters(
        aspirate_speed=params["aspirate_speed"],
        aspirate_wait_time=params["aspirate_wait_time"],
        retract_speed=params["retract_speed"],
        pre_asp_air_vol=0.0,  # Set to 0 since we're using blowout_vol now
        post_asp_air_vol=post_air,
        overaspirate_vol=over_volume,  # CRITICAL: Add overaspirate_vol for calibration

    )
    dispense_params = PipettingParameters(
        dispense_speed=params["dispense_speed"],
        dispense_wait_time=params["dispense_wait_time"],
        blowout_vol=blowout_vol,
        overaspirate_vol=over_volume,  # CRITICAL: Add overaspirate_vol for overdispense calculation
        pre_asp_air_vol=0.0,  # Include for overdispense calculation
        post_asp_air_vol=post_air,  # Include for overdispense calculation
    )  

    if simulate:
        # In simulation mode, generate simulated data directly
        simulated_result = pipet_and_measure_simulated(volume, params, expected_measurement, expected_time)
        
        # Get liquid density for volume calculation - FAIL if not found
        print(f"[DEBUG] About to look up liquid density for: '{liquid}' (type: {type(liquid).__name__})")
        print(f"[DEBUG] Available liquids: {list(LIQUIDS.keys())}")
        print(f"[DEBUG] liquid == None: {liquid is None}")
        print(f"[DEBUG] liquid == 'None': {liquid == 'None'}")
        
        if liquid not in LIQUIDS:
            print(f"[DEBUG] ERROR: liquid '{liquid}' not found in LIQUIDS dictionary!")
            raise ValueError(f"Unknown liquid '{liquid}' - must be one of: {list(LIQUIDS.keys())}")
        if "density" not in LIQUIDS[liquid]:
            print(f"[DEBUG] ERROR: no density key for liquid '{liquid}'!")
            raise ValueError(f"No density specified for liquid '{liquid}' in LIQUIDS dictionary")
        liquid_density = LIQUIDS[liquid]["density"]
        print(f"[DEBUG] Successfully got liquid_density: {liquid_density}")
        print(f"=== PIPET_AND_MEASURE DEBUG END ===\n")

        # Simple debug for simulation mode
        if simulate:
            print(f"[sim-debug] target_volume_ul={volume*1000:.2f} expected_mass={expected_measurement:.5f}g density={liquid_density} -> expected_vol_from_mass={(expected_measurement/liquid_density)*1000:.2f}µL")
        
        for replicate_idx in range(replicate_count):
            # Generate a slightly different mass for each replicate to simulate real variability
            
            # For robot calls: use param_liquid to control optimization, but disable user parameters
            # to prevent double optimization (once in pipet_and_measure, once in robot methods)
            lash_e.nr_robot.aspirate_from_vial(source_vial, volume, parameters=None, liquid=param_liquid)
            measurement = lash_e.nr_robot.dispense_into_vial(dest_vial, volume, parameters=None, measure_weight=True, liquid=param_liquid)
            if new_pipet_each_time:
                lash_e.nr_robot.remove_pipet()

            base_mass = simulated_result["simulated_mass"]
            replicate_mass = base_mass + np.random.normal(0, base_mass * 0.02)  # 2% replicate variation
            replicate_mass = max(replicate_mass, 0)  # Can't be negative
            
            # Calculate volume from mass and density
            print(f"[DEBUG] SIM: About to calculate volume: {replicate_mass} / {liquid_density}")
            print(f"[DEBUG] SIM: replicate_mass type: {type(replicate_mass).__name__}, liquid_density type: {type(liquid_density).__name__}")
            calculated_volume = replicate_mass / liquid_density
            print(f"[DEBUG] SIM: Calculated volume: {calculated_volume}")
            
            # Calculate deviation AFTER volume conversion - this is the correct place!
            target_volume = volume  # Target volume we were trying to pipette
            replicate_deviation = abs(calculated_volume - target_volume) / target_volume * 100

            if simulate and replicate_idx == 0:
                print(f"[sim-debug] first_replicate_mass={replicate_mass:.5f}g calc_volume_ul={calculated_volume*1000:.2f} deviation_pct={replicate_deviation:.2f}")
            
            # Use the time directly from simulation - no additional noise needed
            replicate_time = simulated_result["time"]
            
            replicate_start = datetime.now().isoformat()
            # Simulate end time by adding the replicate time (in seconds)
            import datetime as dt
            start_dt = dt.datetime.fromisoformat(replicate_start)
            end_dt = start_dt + dt.timedelta(seconds=replicate_time)
            replicate_end = end_dt.isoformat()
            
            # Filter out internal simulation data and trial_type from params before saving
            # trial_type will be set explicitly from the function parameter
            filtered_params = {k: v for k, v in params.items() if k not in ['simulated_mass', 'trial_type', 'variability']}
            
            raw_entry = {
                "volume": volume, 
                "replicate": replicate_idx, 
                "mass": replicate_mass,
                "calculated_volume": calculated_volume,
                "replicate_time": replicate_time,
                "start_time": replicate_start, 
                "end_time": replicate_end, 
                "liquid": liquid, 
                "trial_type": trial_type,
                **filtered_params
            }
            raw_measurements.append(raw_entry)
            # Save to CSV (same as real robot mode)
            pd.DataFrame([raw_entry]).to_csv(raw_path, mode='a', index=False, header=not os.path.exists(raw_path))
        
        # Calculate average deviation from all replicates (volume-based, not mass-based)
        all_deviations = []
        for entry in raw_measurements[-replicate_count:]:  # Get the replicates we just added
            calculated_vol = entry["calculated_volume"]
            target_vol = entry["volume"]
            dev = abs(calculated_vol - target_vol) / target_vol * 100
            all_deviations.append(dev)
        
        avg_deviation = np.mean(all_deviations)
        avg_time = np.mean([entry["replicate_time"] for entry in raw_measurements[-replicate_count:]])
        
        # Get measurement data from simulation
        recent_measurements = raw_measurements[-replicate_count:]
        all_masses = [entry["mass"] for entry in recent_measurements]
        all_volumes = [entry["calculated_volume"] for entry in recent_measurements]
        avg_mass = np.mean(all_masses)
        avg_volume = np.mean(all_volumes)
        
        return {
            "deviation": avg_deviation, 
            "time": avg_time,
            "measured_mass": avg_mass,
            "measured_volume": avg_volume,
            "all_masses": all_masses,
            "all_volumes": all_volumes
        }
    else:
        # Real robot mode
        measurements = []
        start = time.time()
        
        # Get liquid density for volume calculation - FAIL if not found
        if liquid not in LIQUIDS:
            raise ValueError(f"Unknown liquid '{liquid}' - must be one of: {list(LIQUIDS.keys())}")
        if "density" not in LIQUIDS[liquid]:
            raise ValueError(f"No density specified for liquid '{liquid}' in LIQUIDS dictionary")
        liquid_density = LIQUIDS[liquid]["density"]
        
        # Vial management hook - only use per-experiment overrides (throttled to prevent spam)
        try:
            # Use only the global overrides set by set_vial_management()
            if _VIAL_MANAGEMENT_MODE_OVERRIDE and _VIAL_MANAGEMENT_MODE_OVERRIDE.lower() != "legacy":
                global _last_vial_mgmt_call
                current_time = time.time()
                if current_time - _last_vial_mgmt_call > _VIAL_MGMT_THROTTLE_SEC:
                    # state may be managed higher level; create minimal state if absent
                    _state = {'measurement_vial_name': dest_vial}
                    manage_vials(lash_e, _state)
                    _last_vial_mgmt_call = current_time
        except Exception as _e:
            lash_e.logger.info(f"[vial-mgmt] skipped: {_e}")

        for replicate_idx in range(replicate_count):
            replicate_start_time = time.time()
            replicate_start = datetime.now().isoformat()
            
            # For robot calls: use param_liquid to control optimization, but disable user parameters
            # to prevent double optimization (once in pipet_and_measure, once in robot methods)
            lash_e.nr_robot.aspirate_from_vial(source_vial, volume, parameters=None, liquid=param_liquid)
            measurement = lash_e.nr_robot.dispense_into_vial(dest_vial, volume, parameters=None, measure_weight=True, liquid=param_liquid)
            
            if new_pipet_each_time:
                lash_e.nr_robot.remove_pipet()
                
            replicate_end_time = time.time()
            replicate_end = datetime.now().isoformat()
            replicate_time = replicate_end_time - replicate_start_time
            
            # Calculate volume from mass and density
            calculated_volume = measurement / liquid_density
            
            # Filter out internal simulation data and trial_type from params before saving
            # trial_type will be set explicitly from the function parameter
            filtered_params = {k: v for k, v in params.items() if k not in ['simulated_mass', 'trial_type', 'variability']}
            
            raw_entry = {
                "volume": volume, 
                "replicate": replicate_idx, 
                "mass": measurement,
                "calculated_volume": calculated_volume,
                "replicate_time": replicate_time,
                "start_time": replicate_start, 
                "end_time": replicate_end, 
                "liquid": liquid, 
                "trial_type": trial_type,
                **filtered_params
            }
            raw_measurements.append(raw_entry) 
            pd.DataFrame([raw_entry]).to_csv(raw_path, mode='a', index=False, header=not os.path.exists(raw_path))
            measurements.append(measurement)
        end = time.time()
        
        avg_measurement = np.mean(measurements)
        std_measurement = np.std(measurements) / avg_measurement * 100
        percent_errors = [abs((m - expected_measurement) / expected_measurement * 100) for m in measurements]
        deviation = np.mean(percent_errors)
        time_score = ((end - start) / replicate_count)
        
        # Calculate average measured volume for display
        avg_measured_volume = avg_measurement / liquid_density
        
        return {
            "deviation": deviation, 
            "time": time_score,
            "measured_mass": avg_measurement,
            "measured_volume": avg_measured_volume,
            "all_masses": measurements,
            "all_volumes": [m / liquid_density for m in measurements]
        }

def strip_tuples(d):
    """Convert any (x, None) → x in a flat dict."""
    return {k: (v if not (isinstance(v, tuple) and v[1] is None) else v[0]) for k, v in d.items()}

def save_analysis(results_df, raw_df, save_dir, include_shap=True, include_scatter=True, 
                  include_boxplots=False, include_pairplot=False, include_learning_curves=False, 
                  include_improvement=False, include_top_trials=False, optimal_conditions=None,
                  learning_curve_metrics=None):
    """
    Save analysis results with configurable plot generation.
    
    Args:
        results_df: DataFrame with optimization results
        raw_df: DataFrame with raw measurements
        save_dir: Directory to save files
        include_shap: Generate SHAP analysis (default: True)
        include_scatter: Generate time vs deviation scatter plot (default: True)
        include_boxplots: Generate parameter boxplots (default: False)
        include_pairplot: Generate parameter pairplot (default: False)
        include_learning_curves: Generate learning curves (default: False)
        include_improvement: Generate improvement summary (default: False)
        include_top_trials: Generate top trial histograms (default: False)
    """
    results_df.to_csv(os.path.join(save_dir, "experiment_summary.csv"), index=False)
    raw_df.to_csv(os.path.join(save_dir, "raw_replicate_data.csv"), index=False)

    # Only run analysis if analyzer is available (not in simulation mode)
    if not ANALYZER_AVAILABLE:
        print("Skipping advanced analysis - analyzer module not available (simulation mode)")
        return

    for metric in ['deviation', 'time', 'variability']:
        if metric in results_df.columns:
            results_df[metric] = pd.to_numeric(results_df[metric], errors='coerce')

    try:
        plots_generated = []
        
        if include_shap:
            analyzer.run_shap_analysis(results_df, save_dir)
            plots_generated.append("SHAP analysis")
            
        if include_scatter:
            analyzer.plot_time_vs_deviation(results_df, save_dir, optimal_conditions)
            analyzer.plot_measured_volume_over_time(raw_df, save_dir)
            analyzer.plot_measured_time_over_measurements(raw_df, save_dir, optimal_conditions)
            plots_generated.append("scatter plot + measured volume plot + measured time plot")
            
        if include_boxplots:
            analyzer.plot_boxplots(results_df, save_dir)
            plots_generated.append("boxplots")
            
        if include_pairplot:
            analyzer.plot_pairplot(results_df, save_dir)
            plots_generated.append("pairplot")
            
        if include_learning_curves:
            analyzer.plot_learning_curves(results_df, save_dir, metrics=learning_curve_metrics)
            plots_generated.append("learning curves")
            
        if include_improvement:
            analyzer.plot_improvement_summary(results_df, save_dir)
            plots_generated.append("improvement summary")
            
        if include_top_trials:
            best_trials = analyzer.get_top_trials(results_df, save_dir, weight_time=1.0, weight_deviation=0.5, weight_variability=2.0, top_n=3)
            analyzer.plot_top_trial_histograms(best_trials, save_dir)
            plots_generated.append("top trials")
            
        if plots_generated:
            print(f"✅ Generated: {', '.join(plots_generated)}")
        else:
            print("No plots requested")
            
    except Exception as e:
        print(f"Warning: Analysis failed but continuing: {e}")

def flatten_measurements(raw_data: dict) -> pd.DataFrame:
    records = []
    for (replicate, wavelength), series in raw_data.items():
        for well, absorbance in series.items():
            records.append({ 
                'replicate': replicate,
                'well': well,
                'wavelength': int(wavelength),  # Convert string like '590' to int
                'absorbance': absorbance
            })
    return pd.DataFrame(records)

# ================= Vial Management (Non-legacy modes) ==================
# Modes: 'legacy' (do nothing), 'maintain', 'swap'.
# Vial management configuration - set via per-experiment overrides only
_VIAL_MANAGEMENT_MODE_OVERRIDE = None
_VIAL_MANAGEMENT_CONFIG_OVERRIDE = None

# Default vial management configuration
VIAL_MANAGEMENT_DEFAULTS = {
    'source_vial': 'liquid_source_0',
    'measurement_vial': 'measurement_vial_0',
    'waste_vial': 'waste_0',
    'reservoir_index': 0,
    'min_source_ml': 2.5,
    'refill_target_ml': 8.0,
    'max_measurement_ml': 7.0,
}

def set_vial_management(mode: str | None = None, config: dict | None = None):
    """Override vial management mode/config for the current experiment.

    Args:
        mode: 'legacy', 'maintain', or 'swap' (case-insensitive). None leaves existing value.
        config: dict with keys: source_vial, measurement_vial, waste_vial, reservoir_index,
                min_source_ml, refill_target_ml, max_measurement_ml.
    """
    global _VIAL_MANAGEMENT_MODE_OVERRIDE, _VIAL_MANAGEMENT_CONFIG_OVERRIDE
    print(f"[vial-mgmt] Setting vial management: mode={mode}, config={config}")
    
    if mode:
        _VIAL_MANAGEMENT_MODE_OVERRIDE = mode.lower()
    if config:
        # Shallow copy to avoid external mutation side-effects
        _VIAL_MANAGEMENT_CONFIG_OVERRIDE = dict(config)

def _maintain_vials(lash_e, state, cfg):
    src = cfg['source_vial']
    meas = cfg['measurement_vial']
    waste = cfg['waste_vial']
    
    # Empty measurement vial if above threshold (do this first to free up clamp if needed)
    try:
        vol_meas = lash_e.nr_robot.get_vial_info(meas, 'vial_volume')
        if vol_meas is not None and vol_meas >= cfg['max_measurement_ml']:
            lash_e.nr_robot.remove_pipet()
            # Leave a small buffer to avoid rounding errors in multi-transfer operations
            # When the volume gets split into multiple transfers, cumulative rounding can cause
            # the final transfer to try aspirating more than what's actually left
            transfer_volume = vol_meas - 0.05  # Leave 50μL buffer to prevent aspiration errors
            if transfer_volume > 0:
                lash_e.nr_robot.dispense_from_vial_into_vial(meas, waste, transfer_volume, remove_tip=False)
                msg = f"[maintain] Emptied {transfer_volume:.2f} mL from {meas} to {waste}"
                lash_e.logger.info(msg)
                print(f"[LOG] {msg}")
    except Exception as e:
        msg = f"[maintain] empty skipped: {e}"
        lash_e.logger.info(msg)
        print(f"[LOG] {msg}")
    
    # Refill source vial if below threshold
    try:
        vol_src = lash_e.nr_robot.get_vial_info(src, 'vial_volume')
        if vol_src is not None and vol_src < cfg['min_source_ml']:
            top_up = cfg['refill_target_ml'] - vol_src
            if top_up > 0:
                lash_e.nr_robot.remove_pipet()
                
                # Handle clamp conflicts: temporarily return measurement vial if needed
                clamp_vial = lash_e.nr_robot.get_vial_in_location('clamp', 0)
                measurement_returned = False
                if clamp_vial is not None and lash_e.nr_robot.get_vial_info(clamp_vial, 'vial_name') == meas:
                    lash_e.nr_robot.return_vial_home(clamp_vial)
                    measurement_returned = True
                
                # Refill the source vial
                lash_e.nr_robot.dispense_into_vial_from_reservoir(cfg['reservoir_index'], src, top_up, return_home=True)
                
                # Move measurement vial back to clamp if we returned it
                if measurement_returned:
                    lash_e.nr_robot.move_vial_to_location(meas, "clamp", 0)
                
                msg = f"[maintain] Refilled {src} by {top_up:.2f} mL"
                lash_e.logger.info(msg)
                print(f"[LOG] {msg}")
    except Exception as e:
        msg = f"[maintain] refill skipped: {e}"
        lash_e.logger.info(msg)
        print(f"[LOG] {msg}")

def _swap_vials_if_needed(lash_e, state, cfg):
    src = cfg['source_vial']
    meas = cfg['measurement_vial']
    try:
        vol_src = lash_e.nr_robot.get_vial_info(src, 'vial_volume')
        vol_meas = lash_e.nr_robot.get_vial_info(meas, 'vial_volume')
        if vol_src is not None and vol_meas is not None:
            # Debug log: Show volumes before swap evaluation
            msg = f"[swap] Pre-swap volumes: source({src})={vol_src:.2f}mL, measurement({meas})={vol_meas:.2f}mL, min_threshold={cfg['min_source_ml']}mL"
            lash_e.logger.info(msg)
            print(f"[LOG] {msg}")
            
            if vol_src < cfg['min_source_ml']:
                lash_e.nr_robot.remove_pipet()
                
                # Physically swap the vials if measurement vial is in clamp
                meas_location = lash_e.nr_robot.get_vial_info(meas, 'location')
                if meas_location == 'clamp':
                    # Return measurement vial home and move source vial to clamp
                    lash_e.nr_robot.return_vial_home(meas)
                    lash_e.nr_robot.move_vial_to_location(src, "clamp", 0)
                
                # Update configuration and state
                cfg['source_vial'], cfg['measurement_vial'] = meas, src
                state['measurement_vial_name'] = cfg['measurement_vial']
                
                # CRITICAL FIX: Update the global config override so changes persist
                global _VIAL_MANAGEMENT_CONFIG_OVERRIDE
                if not _VIAL_MANAGEMENT_CONFIG_OVERRIDE:
                    _VIAL_MANAGEMENT_CONFIG_OVERRIDE = {}
                _VIAL_MANAGEMENT_CONFIG_OVERRIDE['source_vial'] = cfg['source_vial']
                _VIAL_MANAGEMENT_CONFIG_OVERRIDE['measurement_vial'] = cfg['measurement_vial']
                
                msg = f"[swap] Swapped roles: source->{cfg['source_vial']} measurement->{cfg['measurement_vial']}"
                lash_e.logger.info(msg)
                print(f"[LOG] {msg}")
                print(f"[LOG] Updated global config: source={_VIAL_MANAGEMENT_CONFIG_OVERRIDE['source_vial']} measurement={_VIAL_MANAGEMENT_CONFIG_OVERRIDE['measurement_vial']}")
    except Exception as e:
        msg = f"[swap] skipped: {e}"
        lash_e.logger.info(msg)
        print(f"[LOG] {msg}")

# Track if single mode has been initialized to avoid repeated moves
_SINGLE_MODE_INITIALIZED = False

def _single_vial_mode(lash_e, state, cfg):
    """Single vial mode - use same vial for both source and destination (infinite recycling)"""
    global _SINGLE_MODE_INITIALIZED
    
    try:
        # Set both source and measurement to use the same vial (measurement_vial_0)
        # This aligns with initialize_experiment() which always puts measurement_vial_0 in clamp
        single_vial = 'measurement_vial_0'
        
        # Update configuration to use same vial for both roles
        cfg['source_vial'] = single_vial
        cfg['measurement_vial'] = single_vial
        state['measurement_vial_name'] = single_vial
        
        # Update global config so future lookups get the same vial
        global _VIAL_MANAGEMENT_CONFIG_OVERRIDE
        if not _VIAL_MANAGEMENT_CONFIG_OVERRIDE:
            _VIAL_MANAGEMENT_CONFIG_OVERRIDE = {}
        _VIAL_MANAGEMENT_CONFIG_OVERRIDE['source_vial'] = single_vial
        _VIAL_MANAGEMENT_CONFIG_OVERRIDE['measurement_vial'] = single_vial
        
        # No physical vial moves needed! initialize_experiment() already puts measurement_vial_0 in clamp
        # This eliminates the conflict between startup and single mode initialization
        if not _SINGLE_MODE_INITIALIZED:
            _SINGLE_MODE_INITIALIZED = True
            msg = f"[single] Initialized single vial mode with: {single_vial} (already in clamp from startup)"
        else:
            msg = f"[single] Single mode already active with: {single_vial}"
        
        lash_e.logger.info(msg)
        print(f"[LOG] {msg}")
        
    except Exception as e:
        msg = f"[single] setup failed: {e}"
        lash_e.logger.info(msg)
        print(f"[LOG] {msg}")

def manage_vials(lash_e, state):
    # Only use per-experiment overrides (no environment variables)
    mode = _VIAL_MANAGEMENT_MODE_OVERRIDE
    print(f"[vial-mgmt] mode = {mode}")
    
    # Merge override with defaults
    if _VIAL_MANAGEMENT_CONFIG_OVERRIDE:
        cfg = {**VIAL_MANAGEMENT_DEFAULTS, **_VIAL_MANAGEMENT_CONFIG_OVERRIDE}
    else:
        cfg = VIAL_MANAGEMENT_DEFAULTS
    
    if not mode or mode == 'legacy':
        print(f"[vial-mgmt] Using legacy mode (no vial management)")
        return
    
    # Log activation with configuration keys
    cfg_keys = list(cfg.keys()) if cfg else []
    lash_e.logger.info(f"[vial-mgmt] manage_vials called: mode={mode} cfg_keys={cfg_keys}")
    
    if mode == 'maintain':
        _maintain_vials(lash_e, state, cfg)
    elif mode == 'swap':
        _swap_vials_if_needed(lash_e, state, cfg)
    elif mode == 'single':
        # Always run single mode to ensure configuration is updated
        _single_vial_mode(lash_e, state, cfg)
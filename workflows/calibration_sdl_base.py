# calibration_sdl_base.py
import sys
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    "glycerol": {"density": 1.26, "refill_pipets": True},
    "PEG_Water": {"density": 1.05, "refill_pipets": True},
}

# --- Utility Functions ---
def pipet_and_measure_simulated(volume, params, expected_mass, expected_time):
    time.sleep(0.2)
    
    # More realistic simulation that considers ALL parameters, especially volume-dependent ones
    mass_error_factor = 0.0
    
    # Speed parameters: Faster speeds (lower numbers) may be less accurate but faster
    # Higher speeds (higher numbers) are more accurate but slower
    # No single "optimal" - it's a tradeoff between speed and accuracy
    min_speed_penalty = 0.002  # Penalty for very fast speeds (accuracy issues)
    fast_speed_penalty_aspirate = max(0, (15 - params["aspirate_speed"])) * min_speed_penalty
    fast_speed_penalty_dispense = max(0, (15 - params["dispense_speed"])) * min_speed_penalty
    mass_error_factor += fast_speed_penalty_aspirate + fast_speed_penalty_dispense
    
    # Wait time parameters: Longer waits generally improve accuracy (settling time)
    # Very short waits can cause accuracy issues, very long waits waste time but don't help much
    wait_accuracy_benefit = -0.001  # Slight accuracy improvement with longer waits
    mass_error_factor += max(0, (5 - params["aspirate_wait_time"])) * 0.002  # Penalty for very short waits
    mass_error_factor += max(0, (5 - params["dispense_wait_time"])) * 0.002  # Penalty for very short waits
    mass_error_factor += params["aspirate_wait_time"] * wait_accuracy_benefit  # Small benefit from longer waits
    mass_error_factor += params["dispense_wait_time"] * wait_accuracy_benefit  # Small benefit from longer waits
    
    # VOLUME-DEPENDENT PARAMETERS - These are HIGHLY critical for selective optimization!
    # Make these parameters much more volume-sensitive so optimization is forced
    
    # blowout_vol: bounds [0.0, 0.2] - VERY volume dependent
    # Different optimal values for different volumes to force re-optimization
    if volume <= 0.03:  # Small volumes (0.025 mL)
        optimal_blowout = 0.03  # Very specific optimal value
    elif volume <= 0.06:  # Medium-small volumes (0.05 mL)  
        optimal_blowout = 0.07  # Different optimal value
    elif volume <= 0.15:  # Medium volumes (0.1 mL)
        optimal_blowout = 0.11  # Yet another optimal value
    else:  # Large volumes (0.5 mL)
        optimal_blowout = 0.15  # High optimal value
    
    # STRICT penalty - if you're more than 0.03 away from optimal, big penalty
    blowout_error = np.abs(params["blowout_vol"] - optimal_blowout)
    if blowout_error > 0.03:  # Sharp penalty threshold (adjusted for new range)
        mass_error_factor += blowout_error * 0.4  # Very high penalty
    else:
        mass_error_factor += blowout_error * 0.1  # Small penalty if close
    
    # overaspirate_vol: Also VERY volume-dependent, bounds [0.0, volume*0.75]
    # Different optimal fractions for different volumes
    if volume <= 0.03:  # Small volumes need higher fraction
        optimal_overasp = volume * 0.08  # 8% of volume
    elif volume <= 0.06:  # Medium-small volumes
        optimal_overasp = volume * 0.04  # 4% of volume  
    elif volume <= 0.15:  # Medium volumes
        optimal_overasp = volume * 0.02  # 2% of volume
    else:  # Large volumes need very small fraction
        optimal_overasp = volume * 0.01  # 1% of volume
    
    # STRICT penalty for overaspirate_vol too
    overasp_error = np.abs(params["overaspirate_vol"] - optimal_overasp)
    relative_error = overasp_error / volume  # Error as fraction of volume
    if relative_error > 0.03:  # More than 3% of volume off
        mass_error_factor += relative_error * 0.3  # High penalty
    else:
        mass_error_factor += relative_error * 0.05  # Small penalty if close
    
    # Other parameters
    mass_error_factor += np.abs(params["retract_speed"] - 8) * 0.002  # bounds [1, 15], optimal ~8
    mass_error_factor += np.abs(params["post_asp_air_vol"] - 0.05) * 0.04  # bounds [0, 0.1], optimal ~0.05
    
    # Add base random noise
    mass_error_factor += np.random.normal(0, 0.01)
    
    # REALISTIC SIMULATION: Pipetting typically under-delivers due to surface tension, viscosity
    # Add systematic underdelivery bias based on volume (stronger bias)
    underdelivery_bias = -0.05 - (0.008 * volume)  # 5% + 0.8% per mL systematic underdelivery
    
    # Overaspirate compensation: partially compensates but not perfectly
    overasp_compensation = params["overaspirate_vol"] * 0.7 / volume if volume > 0 else 0  # 70% effectiveness
    
    # Apply realistic scaling - start with the underdelivery bias, then add parameter effects
    total_error = underdelivery_bias + (mass_error_factor * 0.3) + overasp_compensation
    
    # Apply non-linear scaling but bias toward underdelivery
    if total_error > 0:
        # Over-pipetting: rare and capped at ~5%
        final_error = 0.05 * np.tanh(total_error / 0.05)
    else:
        # Under-pipetting: more common and can be up to ~12%
        final_error = -0.12 * np.tanh(-total_error / 0.12)
    
    # Use the final calculated error
    mass_error_factor = final_error
    
    # Generate simulated mass with parameter-dependent error
    simulated_mass = expected_mass * (1 + mass_error_factor)
    simulated_mass = max(simulated_mass, 0)  # Can't have negative mass
    
    # Calculate deviation from expected mass
    deviation = abs(simulated_mass - expected_mass) / expected_mass * 100
       
    # Time simulation: More realistic model
    # Start with a reasonable baseline (8-15 seconds for typical pipetting)
    baseline = 12.0  # Base pipetting time in seconds
    
    # Wait times ALWAYS add to the time (no "optimal" value)
    wait_time_penalty = params["aspirate_wait_time"] + params["dispense_wait_time"]
    
    # Speed penalties: Higher numbers (like 35) are SLOWER, lower numbers (like 10) are FASTER
    # Speed range is [10, 35], with 10 being fastest and 35 being slowest
    # Convert speed to time penalty: higher speed = more time
    aspirate_time_penalty = (params["aspirate_speed"] - 10) * 0.3  # 0.3s per speed unit above 10
    dispense_time_penalty = (params["dispense_speed"] - 10) * 0.3  # 0.3s per speed unit above 10
    
    # Small random variation (±2 seconds)
    noise = np.random.normal(0, 2.0)
    
    # Total time = baseline + wait times + speed penalties + noise
    time_score = baseline + wait_time_penalty + aspirate_time_penalty + dispense_time_penalty + noise
    
    # Enforce realistic bounds: minimum 3 seconds, maximum 150 seconds
    time_score = np.clip(time_score, 3.0, 150.0)
    
    # Ensure no NaN values are returned
    deviation = np.nan_to_num(deviation, nan=15.0)
    time_score = np.nan_to_num(time_score, nan=50.0)
    
    return {"deviation": deviation, "time": time_score, "simulated_mass": simulated_mass}

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

def pipet_and_measure(lash_e, source_vial, dest_vial, volume, params, expected_measurement, expected_time, replicate_count, simulate, raw_path, raw_measurements, liquid, new_pipet_each_time, trial_type="UNKNOWN"):
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
        # Note: blowout_vol removed from aspirate_params since it's only used during dispense
    )
    dispense_params = PipettingParameters(
        dispense_speed=params["dispense_speed"],
        dispense_wait_time=params["dispense_wait_time"],
        air_vol=air_vol,
        blowout_vol=blowout_vol,  # CRITICAL: Add blowout_vol to dispense_params!
    )  

    if simulate:
        # In simulation mode, generate simulated data directly
        simulated_result = pipet_and_measure_simulated(volume, params, expected_measurement, expected_time)
        
        # Get liquid density for volume calculation - FAIL if not found
        if liquid not in LIQUIDS:
            raise ValueError(f"Unknown liquid '{liquid}' - must be one of: {list(LIQUIDS.keys())}")
        if "density" not in LIQUIDS[liquid]:
            raise ValueError(f"No density specified for liquid '{liquid}' in LIQUIDS dictionary")
        liquid_density = LIQUIDS[liquid]["density"]

        # Simple debug for simulation mode
        if simulate:
            print(f"[sim-debug] target_volume_ul={volume*1000:.2f} expected_mass={expected_measurement:.5f}g density={liquid_density} -> expected_vol_from_mass={(expected_measurement/liquid_density)*1000:.2f}µL")
        
        for replicate_idx in range(replicate_count):
            # Generate a slightly different mass for each replicate to simulate real variability
            
            lash_e.nr_robot.aspirate_from_vial(source_vial, volume+over_volume, parameters=aspirate_params)
            measurement = lash_e.nr_robot.dispense_into_vial(dest_vial, volume+over_volume, parameters=dispense_params, measure_weight=True)
            if new_pipet_each_time:
                lash_e.nr_robot.remove_pipet()

            base_mass = simulated_result["simulated_mass"]
            replicate_mass = base_mass + np.random.normal(0, base_mass * 0.02)  # 2% replicate variation
            replicate_mass = max(replicate_mass, 0)  # Can't be negative
            
            # Calculate volume from mass and density
            calculated_volume = replicate_mass / liquid_density

            if simulate and replicate_idx == 0:
                print(f"[sim-debug] first_replicate_mass={replicate_mass:.5f}g calc_volume_ul={calculated_volume*1000:.2f} deviation_pct={simulated_result['deviation']:.2f}")
            
            # Simulate replicate timing (base time + some variation)
            base_time = simulated_result["time"]
            replicate_time = base_time + np.random.normal(0, base_time * 0.1)  # 10% time variation
            replicate_time = max(replicate_time, 0.1)  # Minimum 0.1 seconds
            
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
        
        return simulated_result
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
        
        # Vial management hook - only use per-experiment overrides
        try:
            # Use only the global overrides set by set_vial_management()
            if _VIAL_MANAGEMENT_MODE_OVERRIDE and _VIAL_MANAGEMENT_MODE_OVERRIDE.lower() != "legacy":
                # state may be managed higher level; create minimal state if absent
                _state = {'measurement_vial_name': dest_vial}
                manage_vials(lash_e, _state)
        except Exception as _e:
            lash_e.logger.info(f"[vial-mgmt] skipped: {_e}")

        for replicate_idx in range(replicate_count):
            replicate_start_time = time.time()
            replicate_start = datetime.now().isoformat()
            
            lash_e.nr_robot.aspirate_from_vial(source_vial, volume+over_volume, parameters=aspirate_params)
            measurement = lash_e.nr_robot.dispense_into_vial(dest_vial, volume+over_volume, parameters=dispense_params, measure_weight=True)
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
        return {"deviation": deviation, "time": time_score}

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
            if vol_src < cfg['min_source_ml'] and vol_meas > cfg['min_source_ml']:
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
                
                msg = f"[swap] Swapped roles: source→{cfg['source_vial']} measurement→{cfg['measurement_vial']}"
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
        # Set both source and measurement to use the same vial (liquid_source_0)
        single_vial = cfg.get('source_vial', 'liquid_source_0')
        
        # Update configuration to use same vial for both roles
        cfg['measurement_vial'] = single_vial
        state['measurement_vial_name'] = single_vial
        
        # Update global config so future lookups get the same vial
        global _VIAL_MANAGEMENT_CONFIG_OVERRIDE
        if not _VIAL_MANAGEMENT_CONFIG_OVERRIDE:
            _VIAL_MANAGEMENT_CONFIG_OVERRIDE = {}
        _VIAL_MANAGEMENT_CONFIG_OVERRIDE['source_vial'] = single_vial
        _VIAL_MANAGEMENT_CONFIG_OVERRIDE['measurement_vial'] = single_vial
        
        # Only do physical vial moves on first initialization
        if not _SINGLE_MODE_INITIALIZED:
            # Remove pipet first to avoid conflicts
            lash_e.nr_robot.remove_pipet()
            
            # Ensure the vial is in the clamp for easy access
            clamp_vial = lash_e.nr_robot.get_vial_in_location('clamp', 0)
            if clamp_vial is not None:
                lash_e.nr_robot.return_vial_home(clamp_vial)
            
            vial_location = lash_e.nr_robot.get_vial_info(single_vial, 'location')
            if vial_location != 'clamp':
                lash_e.nr_robot.move_vial_to_location(single_vial, "clamp", 0)
            
            _SINGLE_MODE_INITIALIZED = True
            msg = f"[single] Initialized single vial mode with: {single_vial}"
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
    # Force to console as well in case of handler issues
    print(f"[LOG] [vial-mgmt] manage_vials called: mode={mode} cfg_keys={cfg_keys}")
    
    if mode == 'maintain':
        _maintain_vials(lash_e, state, cfg)
    elif mode == 'swap':
        _swap_vials_if_needed(lash_e, state, cfg)
    elif mode == 'single':
        # Only run single mode setup once, then skip all future calls
        global _SINGLE_MODE_INITIALIZED
        if not _SINGLE_MODE_INITIALIZED:
            _single_vial_mode(lash_e, state, cfg)
        else:
            print(f"[LOG] [single] Skipping - already initialized")
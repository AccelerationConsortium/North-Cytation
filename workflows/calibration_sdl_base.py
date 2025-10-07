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
    
    # Speed parameters (optimal around 15-20, bounds [5, 35])
    mass_error_factor += np.abs(params["aspirate_speed"] - 17) * 0.004
    mass_error_factor += np.abs(params["dispense_speed"] - 17) * 0.004
    
    # Wait time parameters (optimal around 10-15, bounds [0, 30])
    mass_error_factor += np.abs(params["aspirate_wait_time"] - 12) * 0.003
    mass_error_factor += np.abs(params["dispense_wait_time"] - 12) * 0.003
    
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
    
    # IMPROVED: Allow for realistic range of parameter performance
    # Keep the volume-dependent optimization challenge but allow more variation
    
    # Instead of hard clipping, use a softer approach that preserves parameter sensitivity
    # but keeps results in a reasonable range for optimization
    
    # Apply non-linear scaling to prevent extreme outliers but maintain differentiation
    if mass_error_factor > 0:
        # Positive errors (over-pipetting): softer scaling
        mass_error_factor = 0.15 * np.tanh(mass_error_factor / 0.15)  # Max ~15% over
    else:
        # Negative errors (under-pipetting): softer scaling  
        mass_error_factor = -0.12 * np.tanh(-mass_error_factor / 0.12)  # Max ~12% under
    
    # Generate simulated mass with parameter-dependent error
    simulated_mass = expected_mass * (1 + mass_error_factor)
    simulated_mass = max(simulated_mass, 0)  # Can't have negative mass
    
    # Calculate deviation from expected mass
    deviation = abs(simulated_mass - expected_mass) / expected_mass * 100
    
    # Variability: also affected by volume-dependent parameters (but keep it reasonable)
    base_variability = 1.5
    variability = base_variability + blowout_error * 0.3 + overasp_error * 0.2
    variability += np.random.normal(0, 0.3)
    variability = np.clip(variability, 0.5, 6)
    
    # Time: affected by speeds and wait times (keep reasonable)
    time_score = (params["aspirate_wait_time"] + params["dispense_wait_time"]) * 0.6
    time_score += (40 - params["aspirate_speed"]) * 0.3  # Slower aspirate = more time
    time_score += (40 - params["dispense_speed"]) * 0.3  # Slower dispense = more time
    time_score += np.random.normal(25, 8)
    time_score = np.clip(time_score, 10, 70)
    
    # Ensure no NaN values are returned
    deviation = np.nan_to_num(deviation, nan=15.0)
    variability = np.nan_to_num(variability, nan=5.0)
    time_score = np.nan_to_num(time_score, nan=50.0)
    
    return {"deviation": deviation, "variability": variability, "time": time_score, "simulated_mass": simulated_mass}

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

def pipet_and_measure(lash_e, source_vial, dest_vial, volume, params, expected_measurement, expected_time, replicate_count, simulate, raw_path, raw_measurements, liquid, new_pipet_each_time):
    blowout_vol = params.get("blowout_vol", 0.05)  # Default blowout volume
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
        blowout_vol=blowout_vol,  # Use blowout_vol instead of pre_asp_air_vol
    )
    dispense_params = PipettingParameters(
        dispense_speed=params["dispense_speed"],
        dispense_wait_time=params["dispense_wait_time"],
        air_vol=air_vol,
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
        
        for replicate_idx in range(replicate_count):
            # Generate a slightly different mass for each replicate to simulate real variability
            base_mass = simulated_result["simulated_mass"]
            replicate_mass = base_mass + np.random.normal(0, base_mass * 0.02)  # 2% replicate variation
            replicate_mass = max(replicate_mass, 0)  # Can't be negative
            
            # Calculate volume from mass and density
            calculated_volume = replicate_mass / liquid_density
            
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
            
            raw_entry = {
                "volume": volume, 
                "replicate": replicate_idx, 
                "mass": replicate_mass,
                "calculated_volume": calculated_volume,
                "replicate_time": replicate_time,
                "start_time": replicate_start, 
                "end_time": replicate_end, 
                "liquid": liquid, 
                **params
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
            
            raw_entry = {
                "volume": volume, 
                "replicate": replicate_idx, 
                "mass": measurement,
                "calculated_volume": calculated_volume,
                "replicate_time": replicate_time,
                "start_time": replicate_start, 
                "end_time": replicate_end, 
                "liquid": liquid, 
                **params
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
        return {"deviation": deviation, "variability": std_measurement , "time": time_score}

def strip_tuples(d):
    """Convert any (x, None) → x in a flat dict."""
    return {k: (v if not (isinstance(v, tuple) and v[1] is None) else v[0]) for k, v in d.items()}

def save_analysis(results_df, raw_df, save_dir, include_shap=True, include_scatter=True, 
                  include_boxplots=False, include_pairplot=False, include_learning_curves=False, 
                  include_improvement=False, include_top_trials=False, optimal_conditions=None):
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
            plots_generated.append("scatter plot")
            
        if include_boxplots:
            analyzer.plot_boxplots(results_df, save_dir)
            plots_generated.append("boxplots")
            
        if include_pairplot:
            analyzer.plot_pairplot(results_df, save_dir)
            plots_generated.append("pairplot")
            
        if include_learning_curves:
            analyzer.plot_learning_curves(results_df, save_dir)
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
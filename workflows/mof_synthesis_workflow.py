"""
MOF Synthesis Workflow - Cu(NO3)2.2.5H2O + Divanillic Acid (DiVA)
Automated synthesis of Metal-Organic Framework with 2:1 linker:metal ratio.

SYNTHESIS OVERVIEW:
- Metal salt: Cu(NO3)2.2.5H2O (6.25 mM stock in ethanol)
- Linker: Divanillic acid (DiVA) (6.25 mM stock in ethanol) 
- Ratio: 2:1 Linker:Metal
- Reaction time: 1 hour at room temperature with stirring
- Expected color change: brown to teal/green with precipitate formation
- Analysis: UV-Vis spectroscopy on wellplate

WORKFLOW STEPS:
1. Initialize workstation (robot, track, cytation, heater for stirring)
2. Check stock solution vial status
3. Prepare reaction vial with linker solution (DiVA)
4. Add metal solution (Cu(NO3)2.2.5H2O) in 2:1 ratio
5. Stir mixture at room temperature for 1 hour
6. Monitor for color change (brown -> teal/green) and precipitate formation
7. Dispense reaction mixture onto wellplate for analysis
8. Take UV-Vis measurements to detect MOF formation signals
9. Data analysis and reporting
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
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import slack_agent
from master_usdl_coordinator import Lash_E

# ================================================================================
# WORKFLOW CONFIGURATION PARAMETERS
# ================================================================================

# Stock solution concentrations (mM)
LINKER_STOCK_CONC = 6.25  # DiVA concentration in ethanol (mM)
METAL_STOCK_CONC = 6.25   # Cu(NO3)2.2.5H2O concentration in ethanol (mM)

# Stoichiometric ratio
LINKER_TO_METAL_RATIO = 2.0  # 2:1 ratio

# Reaction parameters
REACTION_TIME_HOURS = 1.0  # Stirring time at room temperature
REACTION_TEMP = 25  # Room temperature (°C)
STIRRING_SPEED = 1500  # RPM for mixing

# Analysis parameters
ANALYSIS_WELLS_PER_REACTION = 3  # Triplicate measurements
CYTATION_PROTOCOL_FILE = r"C:\Protocols\MOF_Analysis_UV_Vis.prt"  # UV-Vis protocol for MOF detection

# Sampling parameters
SAMPLING_INTERVAL_MINUTES = 10  # Sample every 10 minutes
TOTAL_SAMPLING_TIME_MINUTES = 60  # 1 hour total sampling time

# Volume parameters (mL)
TOTAL_REACTION_VOLUME = 4.0  # Total reaction volume
WELLPLATE_DISPENSE_VOLUME = 0.200  # Volume per well for analysis

SIMULATE = True  # Set to False for hardware execution

# ================================================================================
# MOF SYNTHESIS WORKFLOW FUNCTION
# ================================================================================

def mof_synthesis_workflow(
    lash_e,
    reaction_vial: str = "reaction_vial_1",
    replicates: int = 3,
    prepare_substock: dict = None
):
    """
    Automated MOF synthesis workflow for Cu(NO3)2.2.5H2O + DiVA system.
    
    Arguments:
        lash_e: Initialized Lash_E coordinator object
        reaction_vial (str): Name of vial to use for reaction (default: "reaction_vial_1")
        replicates (int): Number of replicate wells per time point (default: 3)
        prepare_substock (dict or None): If provided, calls prepare_substock_from_solid() before mixing.
            Required keys: 'vial', 'target_volume_mL', 'molecular_weight_g_per_mol',
                           'target_concentration_mM', 'ethanol_vial'.
            Optional keys: 'powder_channel' (default 0).
    
    Returns:
        dict: Analysis results with UV-Vis data and MOF formation indicators
    """
    
    # ============================================================================
    # STEP 1: SETUP WORKSTATION
    # ============================================================================
    
    # Send startup Slack notification
    if not lash_e.simulate:
        startup_message = f"🧪 MOF Synthesis Workflow Started\n" \
                         f"Reaction: Cu(NO3)2 + DiVA ({LINKER_TO_METAL_RATIO}:1)\n" \
                         f"Vial: {reaction_vial}\n" \
                         f"Replicates: {replicates} per time point\n" \
                         f"Duration: {TOTAL_SAMPLING_TIME_MINUTES} min"
        slack_agent.send_slack_message(startup_message)
    
    lash_e.temp_controller.set_temp(REACTION_TEMP)
    #lash_e.grab_new_wellplate()
    
    # ============================================================================
    # STEP 1b: PREPARE SUBSTOCK FROM SOLID (OPTIONAL)
    # ============================================================================

    if prepare_substock is not None:
        prepare_substock_from_solid(
            lash_e,
            vial=prepare_substock['vial'],
            target_volume_mL=prepare_substock['target_volume_mL'],
            molecular_weight_g_per_mol=prepare_substock['molecular_weight_g_per_mol'],
            target_concentration_mM=prepare_substock['target_concentration_mM'],
            ethanol_vial=prepare_substock['ethanol_vial'],
            powder_channel=prepare_substock.get('powder_channel', 0)
        )

    # ============================================================================
    # STEP 2: PREPARE REACTION MIXTURE
    # ============================================================================
    
    linker_volume_ratio = LINKER_TO_METAL_RATIO / (LINKER_TO_METAL_RATIO + 1)
    metal_volume_ratio = 1 / (LINKER_TO_METAL_RATIO + 1)
    linker_volume = TOTAL_REACTION_VOLUME * linker_volume_ratio
    metal_volume = TOTAL_REACTION_VOLUME * metal_volume_ratio
    
    lash_e.logger.info("Starting MOF synthesis reaction preparation")
    prepare_reaction_mixture(lash_e, reaction_vial, linker_volume, metal_volume)
    
    # ============================================================================
    # STEP 3: REACTION MIXING WITH PERIODIC SAMPLING
    # ============================================================================
    
    sampling_times = list(range(0, TOTAL_SAMPLING_TIME_MINUTES + 1, SAMPLING_INTERVAL_MINUTES))
    all_measurements = []
    well_index = 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output/mof_synthesis") / f"{timestamp}_{reaction_vial}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_synthesis_environment(lash_e, reaction_vial, REACTION_TEMP)
    
    for time_point in sampling_times:
        lash_e.logger.info(f"Time point: {time_point} minutes - taking {replicates} samples from {reaction_vial}")
        
        # Dispense samples to wells
        wells = dispense_samples_to_wells(lash_e, reaction_vial, well_index + 1, replicates, WELLPLATE_DISPENSE_VOLUME)
        well_index += len(wells)
        
        # Collect measurement data
        uv_vis_data = collect_timepoint_data(lash_e, CYTATION_PROTOCOL_FILE, wells, time_point, output_dir, lash_e.simulate)
        
        # Process data for combined dataset
        if uv_vis_data is not None:
            measurements = transpose_well_data(uv_vis_data, wells, time_point, reaction_vial)
            all_measurements.extend(measurements)
        
        # Send Slack update for time point completion
        if not lash_e.simulate:
            progress_msg = f"⏱️ MOF Time Point {time_point} min completed\n" \
                          f"Wells measured: {wells}\n" \
                          f"Progress: {time_point}/{TOTAL_SAMPLING_TIME_MINUTES} min"
            slack_agent.send_slack_message(progress_msg)
        
        # Return to heater and wait (except last time point)
        if time_point < sampling_times[-1]:
            lash_e.nr_robot.move_vial_to_location(reaction_vial, "heater", 0)
            wait_time = SAMPLING_INTERVAL_MINUTES * 60
            lash_e.logger.info(f"  Waiting {SAMPLING_INTERVAL_MINUTES} minutes until next sample...")
            time.sleep(wait_time)
    
    lash_e.logger.info(f"Completed periodic sampling for {reaction_vial}")
    
    # ============================================================================
    # STEP 4: DATA PROCESSING AND CLEANUP
    # ============================================================================
    
    lash_e.logger.info("Analyzing time series spectroscopic data for MOF formation kinetics...")
    combined_file = combine_and_save_data(all_measurements, output_dir, lash_e.simulate)
    if combined_file:
        lash_e.logger.info(f"Saved complete time series: {combined_file}")
    
    # Generate spectral evolution plot
    plot_file = None
    if all_measurements:
        combined_data = pd.concat(all_measurements, ignore_index=True)
        plot_file = plot_mof_spectra(combined_data, output_dir, lash_e.simulate)
        if plot_file:
            lash_e.logger.info(f"Saved spectral evolution plot: {plot_file}")
    
    cleanup_synthesis(lash_e, reaction_vial)
    
    # Export results summary
    results = {
        'timestamp': timestamp,
        'reaction_vial': reaction_vial,
        'replicates_per_timepoint': replicates,
        'linker_metal_ratio': f"{LINKER_TO_METAL_RATIO}:1",
        'sampling_interval_minutes': SAMPLING_INTERVAL_MINUTES,
        'total_sampling_points': len(sampling_times),
        'total_wells_measured': well_index,
        'sampling_schedule': sampling_times,
        'output_folder': str(output_dir),
        'combined_data_file': combined_file,
        'spectral_plot_file': plot_file
    }
    
    lash_e.logger.info("MOF synthesis workflow completed")
    lash_e.logger.info(f"Results saved with timestamp: {timestamp}")
    
    # Send completion Slack notification
    if not lash_e.simulate:
        completion_msg = f"✅ MOF Synthesis Workflow Completed\n" \
                        f"Reaction: {reaction_vial}\n" \
                        f"Total wells: {well_index}\n" \
                        f"Time points: {len(sampling_times)}\n" \
                        f"Output: {output_dir.name}\n" \
                        f"Files: CSV + Spectral Plot"
        slack_agent.send_slack_message(completion_msg)
    
    return results

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def prepare_reaction_mixture(lash_e, reaction_vial: str, linker_volume: float, metal_volume: float):
    """Prepare MOF reaction mixture with calculated stoichiometric volumes."""
    lash_e.logger.info(f"Using {reaction_vial} for MOF synthesis")
    lash_e.logger.info(f"  Linker volume (DiVA): {linker_volume:.3f} mL")
    lash_e.logger.info(f"  Metal volume (Cu(NO3)2): {metal_volume:.3f} mL")
    
    lash_e.nr_robot.dispense_from_vial_into_vial("diva_stock", reaction_vial, linker_volume, liquid="ethanol")
    lash_e.nr_robot.dispense_from_vial_into_vial("copper_nitrate_stock", reaction_vial, metal_volume, liquid="ethanol")
    lash_e.logger.info(f"  Mixed linker and metal solutions in {reaction_vial}")

def setup_synthesis_environment(lash_e, reaction_vial: str, temp: int):
    """Setup heater and stirring for reaction."""
    lash_e.nr_robot.move_vial_to_location(reaction_vial, "heater", 0)
    lash_e.temp_controller.turn_on_stirring()
    lash_e.logger.info(f"Started heating and stirring {reaction_vial} at {temp}°C")

def dispense_samples_to_wells(lash_e, reaction_vial: str, start_well: int, num_replicates: int, volume: float):
    """Dispense samples from reaction vial to wellplate wells."""
    lash_e.nr_robot.move_vial_to_location(reaction_vial, "clamp", 0)
    
    wells = []
    for rep in range(num_replicates):
        well_idx = start_well + rep
        wells.append(well_idx)
        lash_e.nr_robot.aspirate_from_vial(reaction_vial, volume, liquid="ethanol")
        lash_e.nr_robot.dispense_into_wellplate([well_idx], [volume], liquid="ethanol")
    
    return wells

def collect_timepoint_data(lash_e, protocol_file: str, wells: list, time_point: int, output_dir: Path, simulate: bool):
    """Measure wellplate and save raw data file."""
    uv_vis_data = lash_e.measure_wellplate(protocol_file, wells=wells)
    lash_e.logger.info(f"  Measured wells {wells} at {time_point} min")
    
    if uv_vis_data is not None:
        if not simulate:
            output_file = output_dir / f"mof_output_{time_point}min.txt"
            uv_vis_data.to_csv(output_file, sep=',', index=True)
            lash_e.logger.info(f"  Saved time point data: {output_file}")
        else:
            lash_e.logger.info(f"  Simulation mode: Would save data for time point {time_point}")
    
    return uv_vis_data

def transpose_well_data(uv_vis_data, wells: list, time_point: int, reaction_vial: str):
    """Transpose spectral data and add metadata columns."""
    measurements = []
    for rep_idx, well in enumerate(wells):
        well_data = uv_vis_data.iloc[:, well-1:well].T
        well_data.reset_index(drop=True, inplace=True)
        well_data.columns = [f"abs_{int(wl)}nm" for wl in uv_vis_data.index]
        well_data['time_point'] = time_point
        well_data['replicate'] = rep_idx + 1
        well_data['reaction_vial'] = reaction_vial
        measurements.append(well_data)
    return measurements

def combine_and_save_data(all_measurements: list, output_dir: Path, simulate: bool):
    """Combine all time point data and save final CSV."""
    if all_measurements:
        combined_data = pd.concat(all_measurements, ignore_index=True)
        if not simulate:
            combined_csv_path = output_dir / "mof_complete_timeseries.csv"
            combined_data.to_csv(combined_csv_path, index=False)
            return str(combined_csv_path)
    return None

def cleanup_synthesis(lash_e, reaction_vial: str):
    """Return vial and turn off heater/stirring."""
    lash_e.nr_robot.return_vial_to_home(reaction_vial)
    lash_e.temp_controller.turn_off_stirring()
    lash_e.temp_controller.turn_off_heating()
    lash_e.logger.info("Heater and stirring turned off")

def plot_mof_spectra(combined_data: pd.DataFrame, output_dir: Path, simulate: bool):
    """Plot overlaid spectra showing MOF formation over time."""
    if combined_data.empty:
        return None
        
    # Get wavelength columns (abs_XXXnm)
    wavelength_cols = [col for col in combined_data.columns if col.startswith('abs_') and col.endswith('nm')]
    wavelengths = [int(col.replace('abs_', '').replace('nm', '')) for col in wavelength_cols]
    
    # Average across replicates for each time point
    time_averaged = combined_data.groupby('time_point')[wavelength_cols].mean()
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    for time_point in time_averaged.index:
        absorbance_values = time_averaged.loc[time_point, wavelength_cols].values
        plt.plot(wavelengths, absorbance_values, marker='o', markersize=3, 
                linewidth=2, label=f'{time_point} min')
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Absorbance', fontsize=12)
    plt.title('MOF Formation Spectral Evolution\nCu(NO3)2 + DiVA', fontsize=14, fontweight='bold')
    plt.legend(title='Time Point', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if not simulate:
        plot_file = output_dir / "mof_spectral_evolution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        return str(plot_file)
    else:
        plt.close()
        return None


def prepare_substock_from_solid(
    lash_e,
    vial: str,
    target_volume_mL: float,
    molecular_weight_g_per_mol: float,
    target_concentration_mM: float,
    ethanol_vial: str,
    powder_channel: int = 0,
    vortex_time: int = 10
):
    """
    Prepare a stock solution by depositing solid into a vial and dissolving in ethanol.

    Uses target_volume_mL and target_concentration_mM to derive how much solid to aim for,
    then reads the actual dispensed mass and adjusts the ethanol volume accordingly so the
    concentration stays as close as possible to target_concentration_mM.

    Arguments:
        lash_e: Initialized Lash_E coordinator object
        vial (str): Destination vial for the substock
        target_volume_mL (float): Desired final solution volume (mL)
        molecular_weight_g_per_mol (float): Molecular weight of the solid (g/mol)
        target_concentration_mM (float): Desired final concentration (mM)
        ethanol_vial (str): Name of the ethanol source vial
        powder_channel (int): Powder dispenser channel (default 0)
        vortex_time (int): Seconds to vortex after ethanol addition (default 10)

    Returns:
        dict: {'actual_mass_mg': float, 'ethanol_volume_mL': float,
               'actual_concentration_mM': float}
    """
    # Derive target mass from desired volume and concentration
    # mmol = conc_mM * volume_mL / 1000  ->  mass_mg = mmol * MW  [g/mol == mg/mmol]
    target_mass_mg = target_concentration_mM * target_volume_mL / 1000.0 * molecular_weight_g_per_mol

    lash_e.logger.info(f"Preparing substock in {vial}: target {target_volume_mL:.3f} mL at "
                       f"{target_concentration_mM:.2f} mM -> need {target_mass_mg:.2f} mg "
                       f"(MW {molecular_weight_g_per_mol:.2f} g/mol)")

    # --- Step 1: Deposit solid ---
    lash_e.nr_robot.move_vial_to_location(vial, 'clamp', 0)
    lash_e.nr_robot.move_home()

    if lash_e.simulate:
        actual_mass_mg = target_mass_mg
        lash_e.logger.info(f"[SIMULATE] Would dispense {target_mass_mg:.2f} mg solid on channel {powder_channel}")
    else:
        actual_mass_mg = lash_e.powder_dispenser.dispense_powder_mg(
            mass_mg=target_mass_mg, channel=powder_channel
        )

    lash_e.logger.info(f"Solid dispensed: {actual_mass_mg:.3f} mg (target was {target_mass_mg:.2f} mg)")

    # --- Step 2: Recalculate ethanol volume from actual mass ---
    # mmol_dispensed = actual_mass_mg / MW  ->  volume_mL = mmol * 1000 / conc_mM
    mmol_dispensed = actual_mass_mg / molecular_weight_g_per_mol
    ethanol_volume_mL = mmol_dispensed * 1000.0 / target_concentration_mM
    actual_concentration_mM = target_concentration_mM  # by construction

    lash_e.logger.info(f"Adjusted ethanol volume: {ethanol_volume_mL:.3f} mL "
                       f"(target was {target_volume_mL:.3f} mL) -> concentration: {actual_concentration_mM:.3f} mM")

    # --- Step 3: Return vial home, then add ethanol ---
    lash_e.nr_robot.return_vial_home(vial)
    lash_e.nr_robot.dispense_from_vial_into_vial(ethanol_vial, vial, ethanol_volume_mL, liquid="ethanol")

    # --- Step 4: Vortex to dissolve solid ---
    lash_e.logger.info(f"Vortexing {vial} for {vortex_time} s to dissolve solid")
    lash_e.nr_robot.vortex_vial(vial_name=vial, vortex_time=vortex_time)

    lash_e.logger.info(f"Substock ready in {vial}: {actual_mass_mg:.3f} mg dissolved in "
                       f"{ethanol_volume_mL:.3f} mL ethanol at {actual_concentration_mM:.3f} mM")

    return {
        'actual_mass_mg': actual_mass_mg,
        'ethanol_volume_mL': ethanol_volume_mL,
        'actual_concentration_mM': actual_concentration_mM
    }


# ================================================================================
# WORKFLOW EXECUTION
# ================================================================================

if __name__ == "__main__":
    # Initialize Lash_E coordinator
    INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/mof_synthesis_vials.csv"
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, initialize_t8=True, simulate=SIMULATE)
    
    # Run MOF synthesis workflow
    results = mof_synthesis_workflow(lash_e)
    
    print(f"MOF synthesis workflow completed. Results: {results}")
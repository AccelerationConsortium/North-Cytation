# --- cmc_workflow.py ---
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
from datetime import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pipetting_data.pipetting_parameters import PipettingParameters

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/microgel_inputs.csv" #This file contains the status of the vials used for the workflow
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\abs_300_800_sweep.prt" #This is the measurement protocol developed in the Cytation software
LOGGING_FOLDER = "../utoronto_demo/logs/"
simulate = True
enable_logging = True

def fill_water_vial(lash_e):
    vial_max_volume = 8.0
    water_reservoir = 1
    current_water_volume = lash_e.nr_robot.get_vial_info('water', 'vial_volume')
    if current_water_volume < vial_max_volume:
        print(f"Filling water vial from {current_water_volume} mL to {vial_max_volume} mL")
        lash_e.nr_robot.dispense_into_vial_from_reservoir(water_reservoir, 'water', vial_max_volume - current_water_volume, return_home=True)

def dilute_microgel(lash_e, source_vial_name, target_vial_name, initial_concentration, final_concentration, final_volume_mL):
    """
    Dilutes a microgel solution from initial_concentration to final_concentration 
    in a target vial with total final_volume.
    
    Parameters:
    - source_vial_name: str, name of the source microgel vial
    - target_vial_name: str, name of the target vial for the diluted solution
    - initial_concentration: float, mg/mL
    - final_concentration: float, mg/mL
    - final_volume_mL: float, desired final volume in mL
    """

    if final_concentration > initial_concentration:
        raise ValueError("Final concentration cannot be greater than initial concentration.")

    # Volume of concentrated microgel needed
    volume_microgel = (final_concentration / initial_concentration) * final_volume_mL
    # Volume of water needed to dilute
    volume_water = final_volume_mL - volume_microgel

    print(f"Diluting {source_vial_name} ({initial_concentration} mg/mL) into {target_vial_name} "
          f"to get {final_concentration} mg/mL @ {final_volume_mL} mL:")
    print(f"--> Add {volume_microgel:.3f} mL microgel")
    print(f"--> Add {volume_water:.3f} mL water")

    # Add concentrated microgel from source vial
    lash_e.nr_robot.vortex_vial(vial_name=source_vial_name, vortex_time=5)  # Ensure microgel is well mixed
    lash_e.nr_robot.move_vial_to_location(source_vial_name, 'main_8mL_rack', 45)  # Move to pipetting position
    # Use new parameter system with blowout volume
    microgel_params = PipettingParameters(blowout_vol=0.10)
    lash_e.nr_robot.dispense_from_vial_into_vial(source_vial_name=source_vial_name, dest_vial_name=target_vial_name, volume=volume_microgel, parameters=microgel_params)  # Dispense concentrated microgel
    lash_e.nr_robot.remove_pipet()  # Remove pipette to avoid contamination
    lash_e.nr_robot.return_vial_home(source_vial_name)  # Return source vial to its home position

    # Add water from reservoir 
    lash_e.nr_robot.dispense_into_vial_from_reservoir(reservoir_index=1, vial_index=target_vial_name, volume=volume_water, return_home=True)
    lash_e.nr_robot.vortex_vial(vial_name=target_vial_name, vortex_time=5)  # Mix the solution

def calculate_volumes(total_volume_mL,toluidine_blue_mL,concentration_high_mg_mL,concentration_profile_mg_mL,min_pipette_mL=0.01):
    
    available_volume_mL = total_volume_mL - toluidine_blue_mL

    fallback_concs = np.linspace(0.5, 0.001, 1000)
    selected_low_conc = None

    for candidate_low_conc in fallback_concs:
        valid = True
        for c in concentration_profile_mg_mL:
            if c == 0:
                continue
            vol_high = (c / concentration_high_mg_mL) * available_volume_mL
            if vol_high < min_pipette_mL:
                vol_low = (c / candidate_low_conc) * available_volume_mL
                if vol_low < min_pipette_mL:
                    valid = False
                    break
        if valid:
            selected_low_conc = candidate_low_conc
            break

    if selected_low_conc is None:
        raise ValueError("No suitable fallback concentration allows ≥ min pipette volume for all targets.")

    rows = []
    for c in concentration_profile_mg_mL:
        row = {
            "target_concentration_mg_mL": c,
            "water": 0.0,
            "toluidine_blue": toluidine_blue_mL,
            "microgel_solution_concentrated": 0.0,
            "microgel_solution_dilute": 0.0,
        }

        if c == 0:
            row["water"] = available_volume_mL
        else:
            vol_high = (c / concentration_high_mg_mL) * available_volume_mL
            if vol_high >= min_pipette_mL:
                row["microgel_solution_concentrated"] = round(vol_high, 3)
                row["water"] = round(available_volume_mL - vol_high, 3)
            else:
                vol_low = (c / selected_low_conc) * available_volume_mL
                row["microgel_solution_dilute"] = round(vol_low, 3)
                row["water"] = round(available_volume_mL - vol_low, 3)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df, round(selected_low_conc, 4)

def dispense_into_wellplate(lash_e, wellplate_volumes, liquid_source):
    lash_e.nr_robot.move_vial_to_location(liquid_source, 'main_8mL_rack', 45)
    
    # Create parameters for this dispensing operation
    from pipetting_data.pipetting_parameters import PipettingParameters
    params = PipettingParameters(
        blowout_vol=0.10,
        # pipet_back_and_forth=True - this was old batching behavior, will use strategy="auto"
    )
    
    lash_e.nr_robot.dispense_from_vials_into_wellplate(
        wellplate_volumes[[liquid_source]], 
        parameters=params,
        strategy="serial",  # Use serial for full parameter support including blowout
        well_plate_type="48 WELL PLATE", 
        low_volume_cutoff=0.2
    )
    lash_e.nr_robot.return_vial_home(liquid_source)

def analyze_spectral_data(filepath, concentrations, output_prefix="output"):
    # Load data
    df = pd.read_csv(filepath)
    wavelengths = df['Wavelengths']
    spectra_columns = df.columns[1:]

    # Plot all spectra
    plt.figure(figsize=(10, 5))
    for col in spectra_columns:
        plt.plot(wavelengths, df[col], label=col, alpha=0.6)
    plt.title("Absorbance Spectra")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.grid(True)
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_spectra_plot.png")
    plt.close()

    # Detect peaks in specified regions
    peak_regions = {
        "540 nm peak": (520, 560),
        "630 nm peak": (610, 650)
    }

    refined_peaks = []
    for col in spectra_columns:
        for peak_name, (low, high) in peak_regions.items():
            mask = (wavelengths >= low) & (wavelengths <= high)
            sub_df = df.loc[mask, ['Wavelengths', col]]
            max_idx = sub_df[col].idxmax()
            refined_peaks.append({
                "Sample": col,
                "Peak": peak_name,
                "Wavelength": df.loc[max_idx, 'Wavelengths'],
                "Absorbance": df.loc[max_idx, col]
            })

    refined_peaks_df = pd.DataFrame(refined_peaks)
    pivot_df = refined_peaks_df.pivot(index="Sample", columns="Peak", values="Absorbance").reset_index()

    # Add concentration info
    pivot_df["Concentration (mg/mL)"] = concentrations

    # Compute ratio
    pivot_df["630/540 Ratio"] = pivot_df["630 nm peak"] / pivot_df["540 nm peak"]
    pivot_df = pivot_df.sort_values("Concentration (mg/mL)", ascending=False)

    from scipy.optimize import curve_fit, root_scalar

    # Step-edge model function (logistic in log10 space)
    def step_edge_model(conc, x0, k, a, b):
        return a / (1 + np.exp(-k * (np.log10(conc) - np.log10(x0)))) + b

    # Filter non-zero concentrations
    xdata = pivot_df["Concentration (mg/mL)"].values
    ydata = pivot_df["630/540 Ratio"].values
    mask = xdata > 0
    xdata = xdata[mask]
    ydata = ydata[mask]

    # Fit the step-edge model
    p0 = [0.015, 10, 2.5, 0.4]  # Initial guess
    params, _ = curve_fit(step_edge_model, xdata, ydata, p0=p0)

    # Use root finding to estimate concentration where model = 1
    def model_minus_one(c):
        return step_edge_model(c, *params) - 1

    result = root_scalar(model_minus_one, bracket=[1e-4, 0.1], method='brentq')
    conc_at_ratio_1 = result.root if result.converged else None

    # Optional: Plot and save the model fit
    x_fit = np.logspace(np.log10(min(xdata)), np.log10(max(xdata)), 500)
    y_fit = step_edge_model(x_fit, *params)

    plt.figure(figsize=(8, 5))
    plt.plot(xdata, ydata, 'o', label="Observed")
    plt.plot(x_fit, y_fit, '-', label="Step-edge Fit")
    plt.axhline(1, color='red', linestyle='--', label="Ratio = 1")
    if conc_at_ratio_1:
        plt.axvline(conc_at_ratio_1, color='green', linestyle='--',
                    label=f"Model Est. = {conc_at_ratio_1:.4f} mg/mL")
    plt.xscale("log")
    plt.xlabel("Concentration (mg/mL, log scale)")
    plt.ylabel("630/540 Absorbance Ratio")
    plt.title("Step-Edge Model Fit")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_model_fit.png")
    plt.close()

    # Save summary
    with open(f"{output_prefix}_summary.txt", "w") as f:
        f.write(f"Estimated concentration where ratio = 1: {conc_at_ratio_1:.4f} mg/mL\\n")

    return conc_at_ratio_1

with Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate, logging=enable_logging) as lash_e:
    
    lash_e.nr_robot.check_input_file() #Check the status of the input vials
    #lash_e.nr_track.check_input_file() #Check the status of the wellplates (for multiple wellplate assays)

    folder = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f'C:/Users/Imaging Controller/Desktop/microgel/{timestamp}/'

    if not simulate:
        os.makedirs(folder, exist_ok=True) #This is to create a folder for the workflow saves results
        print(f"Folder created at: {folder}")

    #Prepare the experiment
    starting_wp_index = 0 #Ensure we start at the first well of the wellplate
    concentration_profile_mg_mL = [0.12, 0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.001, 0]
    initial_concentration = 4.0 #Initial concentration of the microgel solution in mg/mL
    concentration_profile_mg_mL.sort() #Sort lowest to highest
    wellplate_volumes, dilute_concentration = calculate_volumes(total_volume_mL=1.0,toluidine_blue_mL=0.1,concentration_high_mg_mL=4.0,concentration_profile_mg_mL=concentration_profile_mg_mL)
    volume_dilute_needed = wellplate_volumes['microgel_solution_dilute'].sum() + 1.0 #Total volume of dilute solution needed, plus 1 mL
    wells = range(starting_wp_index, starting_wp_index + len(concentration_profile_mg_mL))

    print(f"Dilute concentration for microgel solution: {dilute_concentration} mg/mL")
    print("To be dispensed (uL):/n", wellplate_volumes)

    '''Preparation for workflow'''
    #lash_e.nr_track.get_new_wellplate()
    fill_water_vial(lash_e)

    '''Start the workflow'''
    #1. Create a diluted microgel solution
    dilute_microgel(lash_e, source_vial_name='microgel_solution_concentrated', target_vial_name='microgel_solution_dilute', initial_concentration=initial_concentration, final_concentration=dilute_concentration, final_volume_mL=volume_dilute_needed)

    #2.Dispense into the wellplate (Separate into 4)
    for liquid_source in ['microgel_solution_concentrated', 'microgel_solution_dilute', 'water', 'toluidine_blue']:
        dispense_into_wellplate(lash_e, wellplate_volumes, liquid_source)

    #Mix the wells using aspirate/dispense
    lash_e.nr_robot.get_pipet("large_tip")
    for well in wells:
        lash_e.nr_robot.mix_well_in_wellplate(well, volume=0.5, well_plate_type="48 WELL PLATE")
    lash_e.nr_robot.remove_pipet()

    #3. Read the wellplate 1x
    results = lash_e.measure_wellplate(protocol_file_path=MEASUREMENT_PROTOCOL_FILE, wells_to_measure=wells, plate_type="48 WELL PLATE", repeats=1)

    #4. Save the results
    if not simulate:
        results_file = os.path.join(folder, "microgel_results.csv")
        results.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")
    else:
        print("[Simulate] Results would be saved here.")

    #5. Analyze the results. Here we would call an analysis function that processes the results and determine the equivalence point
    if not simulate:
        conc_at_ratio_1 = analyze_spectral_data(results_file, concentration_profile_mg_mL, output_prefix=timestamp)
    
    print(lash_e.nr_robot.VIAL_DF)

    if lash_e.nr_track.NR_OCCUPIED:
        lash_e.discard_used_wellplate()
        print("Workflow complete and wellplate discarded")
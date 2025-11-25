"""
Surfactant Substock Creation Workflow
Creates dilution series for all 6 surfactants that can be reused across multiple experiments.

This workflow only needs to be run once to create all substocks.
Individual screening experiments can then reference these pre-made dilutions.
"""
import sys
sys.path.append("../utoronto_demo")
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from master_usdl_coordinator import Lash_E

# WORKFLOW CONSTANTS
SIMULATE = False  # Set to False for actual hardware execution

# Surfactant library with stock concentrations (from cmc_exp_new.py)
SURFACTANT_LIBRARY = {
    "SDS": {
        "full_name": "Sodium Dodecyl Sulfate",
        "category": "anionic",
        "stock_conc": 50,  # mM
    },
    "NaDC": {
        "full_name": "Sodium Docusate", 
        "category": "anionic",
        "stock_conc": 25,  # mM
    },
    "NaC": {
        "full_name": "Sodium Cholate",
        "category": "anionic", 
        "stock_conc": 50,  # mM
    },
    "CTAB": {
        "full_name": "Hexadecyltrimethylammonium Bromide",
        "category": "cationic",
        "stock_conc": 5,  # mM
    },
    "DTAB": {
        "full_name": "Dodecyltrimethylammonium Bromide",
        "category": "cationic",
        "stock_conc": 50,  # mM
    },
    "TTAB": {
        "full_name": "Tetradecyltrimethylammonium Bromide", 
        "category": "cationic",
        "stock_conc": 50,  # mM
    }
}

# Dilution parameters
MIN_CONC_LOG = -4  # 10^-4 mM minimum
MAX_CONC_LOG = 1   # 10^1 mM maximum  
LOG_STEP = 1       # 10^1 step size
FINAL_SUBSTOCK_VOLUME = 6  # mL final volume for each dilution

# File paths
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/surfactant_grid_vials_expanded.csv"

def calculate_achievable_concentrations(surfactant_name, target_concentrations):
    """
    Calculate which target concentrations are achievable given stock concentration.
    
    Args:
        surfactant_name: Name of surfactant (key in SURFACTANT_LIBRARY)
        target_concentrations: List of target dilution concentrations (before 1:1 mixing)
        
    Returns:
        list: Achievable concentrations, others replaced with None
    """
    stock_conc = SURFACTANT_LIBRARY[surfactant_name]["stock_conc"]
    
    achievable = []
    for target in target_concentrations:
        # Target concentration is after 1:1 mixing, so substock needs to be 2x higher
        required_substock_conc = target * 2
        
        if required_substock_conc <= stock_conc:
            achievable.append(target)
        else:
            print(f"WARNING: {surfactant_name} cannot achieve {target:.2e} mM (needs {required_substock_conc:.2e} mM, stock is {stock_conc} mM)")
            achievable.append(None)
    
    return achievable

def create_surfactant_dilutions(lash_e, surfactant_name, target_concentrations):
    """
    Create dilution series for a single surfactant.
    
    Args:
        lash_e: Lash_E coordinator instance
        surfactant_name: Name of surfactant (e.g., "SDS", "TTAB")
        target_concentrations: List of target concentrations (final after 1:1 mixing)
        
    Returns:
        tuple: (dilution_vials, dilution_steps, achievable_concentrations)
    """
    print(f"\\nCreating dilution series for {surfactant_name}...")
    
    stock_conc_mm = SURFACTANT_LIBRARY[surfactant_name]["stock_conc"]
    stock_vial = f"{surfactant_name}_stock"
    
    # Check which concentrations are achievable
    achievable_concs = calculate_achievable_concentrations(surfactant_name, target_concentrations)
    
    dilution_vials = []
    dilution_steps = []
    
    for i, target_conc in enumerate(achievable_concs):
        vial_name = f"{surfactant_name}_dilution_{i}"
        dilution_vials.append(vial_name)
        
        if target_conc is None:
            # Skip this dilution - not achievable
            dilution_steps.append({
                'vial_name': vial_name,
                'target_conc_mm': None,
                'achievable': False,
                'reason': f'Stock concentration ({stock_conc_mm} mM) too low'
            })
            continue
            
        # Calculate dilution for substock (2x final concentration for 1:1 mixing)
        required_substock_conc = target_conc * 2
        
        if i == 0:
            # First dilution from stock
            dilution_factor = stock_conc_mm / required_substock_conc
            source_vial = stock_vial
            source_conc = stock_conc_mm
        else:
            # Serial dilution from previous step
            prev_conc = achievable_concs[i-1] * 2 if achievable_concs[i-1] else None
            if prev_conc and prev_conc > required_substock_conc:
                dilution_factor = prev_conc / required_substock_conc
                source_vial = f"{surfactant_name}_dilution_{i-1}"
                source_conc = prev_conc
            else:
                # Go back to stock if previous dilution not suitable
                dilution_factor = stock_conc_mm / required_substock_conc
                source_vial = stock_vial
                source_conc = stock_conc_mm
        
        # Calculate volumes
        stock_volume = FINAL_SUBSTOCK_VOLUME / dilution_factor  # mL
        water_volume = FINAL_SUBSTOCK_VOLUME - stock_volume     # mL
        
        print(f"  {vial_name}: {target_conc:.2e} mM (substock: {required_substock_conc:.2e} mM)")
        print(f"    From {source_vial} ({source_conc:.2e} mM), {dilution_factor:.1f}x dilution")
        print(f"    Volumes: {stock_volume:.3f} mL + {water_volume:.3f} mL = {FINAL_SUBSTOCK_VOLUME} mL")
        
        # Perform dilution
        if not lash_e.simulate:
            lash_e.nr_robot.move_vial_to_location(source_vial, "clamp", 0)
            lash_e.nr_robot.aspirate(stock_volume, 0)
            lash_e.nr_robot.move_vial_to_location(vial_name, "clamp", 0)
            lash_e.nr_robot.dispense(stock_volume, 0)
            
            # Add water
            lash_e.nr_robot.move_vial_to_location("water", "clamp", 0)
            lash_e.nr_robot.aspirate(water_volume, 1)  # Pump 1 for water
            lash_e.nr_robot.move_vial_to_location(vial_name, "clamp", 0)
            lash_e.nr_robot.dispense(water_volume, 1)
        
        # Record step
        dilution_steps.append({
            'vial_name': vial_name,
            'target_conc_mm': target_conc,
            'substock_conc_mm': required_substock_conc,
            'source_vial': source_vial,
            'source_conc_mm': source_conc,
            'dilution_factor': dilution_factor,
            'stock_volume_ul': stock_volume * 1000,
            'water_volume_ul': water_volume * 1000,
            'total_volume_ml': FINAL_SUBSTOCK_VOLUME,
            'achievable': True
        })
    
    return dilution_vials, dilution_steps, achievable_concs

def create_all_substocks(simulate=True):
    """
    Create dilution series for all surfactants.
    
    Args:
        simulate: If True, run in simulation mode
        
    Returns:
        dict: Complete dilution information for all surfactants
    """
    print("="*60)
    print("SURFACTANT SUBSTOCK CREATION WORKFLOW")
    print("="*60)
    
    # Generate target concentrations
    target_concentrations = [10**(log_conc) for log_conc in range(MIN_CONC_LOG, MAX_CONC_LOG + 1, LOG_STEP)]
    print(f"Target concentrations: {target_concentrations}")
    
    # Initialize workstation
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
    
    # Setup robot
    print("\\nInitializing robot...")
    lash_e.nr_robot.move_home()
    lash_e.nr_robot.home_robot_components()
    lash_e.nr_robot.check_input_file()
    lash_e.nr_robot.prime_reservoir_line(1, 'water')
    
    # Create dilutions for all surfactants
    all_dilution_info = {}
    
    for surfactant_name in SURFACTANT_LIBRARY.keys():
        dilution_vials, dilution_steps, achievable_concs = create_surfactant_dilutions(
            lash_e, surfactant_name, target_concentrations
        )
        
        all_dilution_info[surfactant_name] = {
            'surfactant_info': SURFACTANT_LIBRARY[surfactant_name],
            'target_concentrations': target_concentrations,
            'achievable_concentrations': achievable_concs,
            'dilution_vials': dilution_vials,
            'dilution_steps': dilution_steps
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"../utoronto_demo/output/substock_creation_{timestamp}"
    if simulate:
        output_folder += "_SIMULATION"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Save dilution info as JSON
    dilution_file = os.path.join(output_folder, "all_surfactant_dilutions.json")
    with open(dilution_file, 'w') as f:
        json.dump(all_dilution_info, f, indent=2)
    
    # Create human-readable report
    report_file = os.path.join(output_folder, "substock_creation_report.txt")
    with open(report_file, 'w') as f:
        f.write("SURFACTANT SUBSTOCK CREATION REPORT\\n")
        f.write("=" * 50 + "\\n\\n")
        
        if simulate:
            f.write("NOTE: This is a SIMULATION - no actual pipetting was performed\\n\\n")
        
        for surfactant_name, info in all_dilution_info.items():
            f.write(f"{surfactant_name} ({info['surfactant_info']['full_name']})\\n")
            f.write(f"Category: {info['surfactant_info']['category']}\\n")
            f.write(f"Stock concentration: {info['surfactant_info']['stock_conc']} mM\\n")
            f.write("-" * 40 + "\\n")
            
            for i, step in enumerate(info['dilution_steps']):
                if step['achievable']:
                    f.write(f"Dilution {i}: {step['vial_name']}\\n")
                    f.write(f"  Target: {step['target_conc_mm']:.2e} mM (final after 1:1 mixing)\\n")
                    f.write(f"  Substock: {step['substock_conc_mm']:.2e} mM\\n")
                    f.write(f"  Source: {step['source_vial']} ({step['source_conc_mm']:.2e} mM)\\n")
                    f.write(f"  Volumes: {step['stock_volume_ul']:.0f} uL + {step['water_volume_ul']:.0f} uL\\n")
                else:
                    f.write(f"Dilution {i}: {step['vial_name']} - NOT ACHIEVABLE\\n")
                    f.write(f"  Reason: {step['reason']}\\n")
                f.write("\\n")
            f.write("\\n")
    
    lash_e.nr_robot.move_home()
    print(f"\\nSubstock creation complete!")
    print(f"Results saved to: {output_folder}")
    
    return all_dilution_info

if __name__ == "__main__":
    """
    Run the substock creation workflow.
    This only needs to be done once - substocks can be reused for multiple experiments.
    """
    dilution_info = create_all_substocks(simulate=SIMULATE)
    print("All surfactant substocks created successfully!")
#!/usr/bin/env python3
"""
Debug script to trace CMC control creation and extraction flow.
"""

import numpy as np

# Copy the relevant constants and functions
def get_cmc_concentrations_from_controls(cmc_controls, surfactant_name):
    """Extract target concentrations from CMC control specifications."""
    concentrations = []
    print(f"DEBUG: get_cmc_concentrations_from_controls for {surfactant_name}")
    print(f"DEBUG: Checking {len(cmc_controls)} controls")
    
    for control in cmc_controls:
        print(f"DEBUG: Control surfactant_for_cmc='{control.get('surfactant_for_cmc')}', looking for '{surfactant_name}'")
        if control['surfactant_for_cmc'] == surfactant_name:
            target_conc_a = control.get('target_concentration_a_mm', 0.0)
            target_conc_b = control.get('target_concentration_b_mm', 0.0)
            conc = target_conc_a if target_conc_a > 0 else target_conc_b
            print(f"DEBUG: Found match - target_conc_a={target_conc_a}, target_conc_b={target_conc_b}, using conc={conc}")
            if conc > 0:
                concentrations.append(conc)
        else:
            print(f"DEBUG: No match")
    
    print(f"DEBUG: Returning {len(concentrations)} concentrations: {concentrations}")
    return concentrations

def create_simple_control(surfactant_name, surfactant_a_name, surfactant_b_name, conc):
    """Create a single control for testing."""
    is_surfactant_a = (surfactant_name == surfactant_a_name)
    is_surfactant_b = (surfactant_name == surfactant_b_name)
    
    return {
        'control_type': f'cmc_{surfactant_name}_1',
        'description': f'{surfactant_name} CMC series: {conc:.4f} mM (single surfactant)',
        'target_concentration_a_mm': conc if is_surfactant_a else 0.0,
        'target_concentration_b_mm': conc if is_surfactant_b else 0.0,
        'surfactant_for_cmc': surfactant_name,
        'is_control': True
    }

def test_full_flow():
    print("=== Testing Full CMC Flow ===")
    
    surfactant_a_name = "SDS" 
    surfactant_b_name = "BDDAC"
    
    # Create some test controls for each surfactant
    cmc_controls_a_preview = [
        create_simple_control("SDS", surfactant_a_name, surfactant_b_name, 5.0),
        create_simple_control("SDS", surfactant_a_name, surfactant_b_name, 10.0)
    ]
    
    cmc_controls_b_preview = [
        create_simple_control("BDDAC", surfactant_a_name, surfactant_b_name, 3.0),
        create_simple_control("BDDAC", surfactant_a_name, surfactant_b_name, 8.0)
    ]
    
    print("\n=== Testing SDS Concentration Extraction ===")
    cmc_target_concs_a = get_cmc_concentrations_from_controls(cmc_controls_a_preview, surfactant_a_name)
    print(f"SDS target concentrations: {cmc_target_concs_a}")
    print(f"SDS has concentrations: {bool(cmc_target_concs_a)}")
    
    print("\n=== Testing BDDAC Concentration Extraction ===") 
    cmc_target_concs_b = get_cmc_concentrations_from_controls(cmc_controls_b_preview, surfactant_b_name)
    print(f"BDDAC target concentrations: {cmc_target_concs_b}")
    print(f"BDDAC has concentrations: {bool(cmc_target_concs_b)}")

if __name__ == "__main__":
    test_full_flow()
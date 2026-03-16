#!/usr/bin/env python3
"""
Minimal debug script to investigate why BDDAC CMC controls are failing.
"""

import numpy as np

# Copy the relevant constants and data structures from the module
SURFACTANT_LIBRARY = {
    "SDS": {
        "full_name": "Sodium Dodecyl Sulfate",
        "category": "anionic",
        "stock_conc": 50,  # mM
        "cmc_mm": 8.6,  # mM (t = 10 min)
    },
    "BDDAC": {
        "full_name": "Benzyldimethyldodecylammonium Chloride",
        "category": "cationic",
        "stock_conc": 50,  # mM
        "cmc_mm": 8.4 #mM
    },
}

MIN_CONC = 10**-2  # 0.01 mM minimum concentration for all surfactants
WELL_VOLUME_UL = 200  # uL per well
ADD_BUFFER = False  # Set to False to skip buffer addition
BUFFER_VOLUME_UL = 20  # uL buffer to add per well
PYRENE_VOLUME_UL = 5  # uL pyrene solution per well

def create_cmc_control_series_debug(surfactant_name, surfactant_a_name, surfactant_b_name, num_points=7, concentration_span_factor=3.0):
    """
    Debug version of create_cmc_control_series function.
    """
    controls = []
    
    # DEBUG: Add logging to understand BDDAC issue
    print(f"DEBUG: create_cmc_control_series called for {surfactant_name}")
    print(f"DEBUG: surfactant_a_name={surfactant_a_name}, surfactant_b_name={surfactant_b_name}")
    
    # Get CMC value from library
    if surfactant_name not in SURFACTANT_LIBRARY:
        print(f"DEBUG: {surfactant_name} not in SURFACTANT_LIBRARY")
        return controls  # No CMC controls if surfactant not in library
        
    surfactant_info = SURFACTANT_LIBRARY[surfactant_name]
    if 'cmc_mm' not in surfactant_info:
        print(f"DEBUG: {surfactant_name} has no cmc_mm value")
        return controls  # No CMC controls if no CMC value available
        
    cmc_value = surfactant_info['cmc_mm']
    print(f"DEBUG: {surfactant_name} CMC = {cmc_value} mM")
    
    # Create concentration range around CMC (log scale)
    # Span from CMC/span_factor to CMC*span_factor
    min_conc = cmc_value / concentration_span_factor
    max_conc = cmc_value * concentration_span_factor
    
    # Ensure we don't go below global minimum
    min_conc = max(min_conc, MIN_CONC)
    
    print(f"DEBUG: {surfactant_name} initial range: {min_conc:.4f} to {max_conc:.4f} mM")
    
    # For CMC controls: Total volume = WELL_VOLUME_UL + PYRENE_VOLUME_UL = 205 µL
    # Available volume for surfactant+water depends on buffer setting
    available_volume_ul = WELL_VOLUME_UL - (BUFFER_VOLUME_UL if ADD_BUFFER else 0)  # 200µL (no buffer) or 180µL (with buffer)
    total_well_volume_ul = WELL_VOLUME_UL + PYRENE_VOLUME_UL  # 205 µL total
    
    # Don't exceed what's achievable with available volume (after buffer) at stock concentration
    stock_conc = surfactant_info['stock_conc']
    max_achievable_with_stock = (stock_conc * available_volume_ul) / WELL_VOLUME_UL  # Scale down for buffer space
    max_conc = min(max_conc, max_achievable_with_stock)
    
    print(f"DEBUG: {surfactant_name} stock_conc={stock_conc}, available_volume={available_volume_ul}")
    print(f"DEBUG: {surfactant_name} max_achievable={max_achievable_with_stock:.4f}, final_max={max_conc:.4f}")
    
    if min_conc > max_conc:
        print(f"DEBUG: {surfactant_name} min_conc > max_conc ({min_conc:.4f} > {max_conc:.4f}), returning empty")
        return controls
        
    # Generate CMC-centered concentration series with dense center, sparse edges
    # Create symmetric points around CMC with increasing spacing
    if num_points % 2 == 1:
        # Odd number: include CMC as center point
        half_points = (num_points - 1) // 2
        # Generate points from 0 to 1, then apply stretching function
        raw_spacing = np.linspace(0, 1, half_points + 1)[1:]  # Exclude 0, include 1
        # Apply cubic stretching to increase spacing toward edges
        stretched_spacing = raw_spacing ** 2  # Quadratic stretching
        
        # Map to log scale around CMC
        log_cmc = np.log10(cmc_value)
        log_span = np.log10(concentration_span_factor)
        
        # Create positive and negative sides
        positive_logs = log_cmc + stretched_spacing * log_span
        negative_logs = log_cmc - stretched_spacing * log_span
        
        # Combine: [low ... CMC ... high]
        all_logs = np.concatenate([negative_logs[::-1], [log_cmc], positive_logs])
        concentrations = 10 ** all_logs
    else:
        # Even number: no exact CMC point, symmetric around it
        half_points = num_points // 2
        raw_spacing = np.linspace(0, 1, half_points + 1)[1:]  # Exclude 0
        stretched_spacing = raw_spacing ** 2
        
        log_cmc = np.log10(cmc_value)
        log_span = np.log10(concentration_span_factor)
        
        positive_logs = log_cmc + stretched_spacing * log_span
        negative_logs = log_cmc - stretched_spacing * log_span
        
        all_logs = np.concatenate([negative_logs[::-1], positive_logs])
        concentrations = 10 ** all_logs
    
    # Ensure bounds are respected
    concentrations = np.clip(concentrations, min_conc, max_conc)
    
    print(f"DEBUG: {surfactant_name} generated {len(concentrations)} concentrations: {concentrations}")
    
    # Create control specifications (vial selection happens in create_well_recipe_from_control)
    for i, conc in enumerate(concentrations):
        # Determine which surfactant this is for using current pairing parameters
        is_surfactant_a = (surfactant_name == surfactant_a_name)
        is_surfactant_b = (surfactant_name == surfactant_b_name)
        
        print(f"DEBUG: {surfactant_name} control {i+1}: conc={conc:.4f}, is_a={is_surfactant_a}, is_b={is_surfactant_b}")
        
        controls.append({
            'control_type': f'cmc_{surfactant_name}_{i+1}',
            'description': f'{surfactant_name} CMC series: {conc:.4f} mM (single surfactant)',
            'target_concentration_a_mm': conc if is_surfactant_a else 0.0,
            'target_concentration_b_mm': conc if is_surfactant_b else 0.0,
            'cmc_control': True,
            'single_surfactant': False,  # Include buffer in CMC controls
            'surfactant_for_cmc': surfactant_name,  # Which surfactant this series is for
            'replicate': 1,
            'is_control': True
        })
    
    print(f"DEBUG: {surfactant_name} returning {len(controls)} controls")
    return controls

def test_cmc_controls():
    print("=== Testing CMC Control Creation ===")
    
    # Test parameters from the log
    surfactant_a_name = "SDS" 
    surfactant_b_name = "BDDAC"
    num_points = 8  # CMC_CONTROL_POINTS from config
    
    print(f"Testing SDS (surfactant A)...")
    cmc_controls_a = create_cmc_control_series_debug(surfactant_a_name, surfactant_a_name, surfactant_b_name, num_points)
    print(f"SDS returned {len(cmc_controls_a)} controls\n")
    
    print(f"Testing BDDAC (surfactant B)...")
    cmc_controls_b = create_cmc_control_series_debug(surfactant_b_name, surfactant_a_name, surfactant_b_name, num_points) 
    print(f"BDDAC returned {len(cmc_controls_b)} controls\n")

if __name__ == "__main__":
    test_cmc_controls()
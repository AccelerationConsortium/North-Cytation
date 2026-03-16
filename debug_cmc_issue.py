#!/usr/bin/env python3
"""
Debug script to investigate why BDDAC CMC controls are failing.
"""

import sys
import os

# Add the workflows directory to path to import the module
sys.path.append("workflows")
from surfactant_grid_adaptive_concentrations import create_cmc_control_series, SURFACTANT_LIBRARY

def test_cmc_controls():
    print("=== Testing CMC Control Creation ===")
    
    # Test parameters from the log
    surfactant_a_name = "SDS" 
    surfactant_b_name = "BDDAC"
    num_points = 8  # CMC_CONTROL_POINTS from config
    
    print(f"Testing SDS (surfactant A)...")
    cmc_controls_a = create_cmc_control_series(surfactant_a_name, surfactant_a_name, surfactant_b_name, num_points)
    print(f"SDS returned {len(cmc_controls_a)} controls\n")
    
    print(f"Testing BDDAC (surfactant B)...")
    cmc_controls_b = create_cmc_control_series(surfactant_b_name, surfactant_a_name, surfactant_b_name, num_points) 
    print(f"BDDAC returned {len(cmc_controls_b)} controls\n")
    
    # Check library contents
    print("=== SURFACTANT_LIBRARY Check ===")
    for surf in ["SDS", "BDDAC"]:
        if surf in SURFACTANT_LIBRARY:
            info = SURFACTANT_LIBRARY[surf]
            print(f"{surf}: stock_conc={info.get('stock_conc')}, cmc_mm={info.get('cmc_mm')}")
        else:
            print(f"{surf}: NOT FOUND in library")

if __name__ == "__main__":
    test_cmc_controls()
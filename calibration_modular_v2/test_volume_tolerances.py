#!/usr/bin/env python3
"""
Test script to verify volume tolerance calculations match original system.

This validates our new explicit volume range approach against the original
VOLUME_TOLERANCE_RANGES from calibration_sdl_simplified.py.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ExperimentConfig

def test_tolerance_calculations():
    """Test tolerance calculations for various volumes."""
    config = ExperimentConfig.from_yaml("experiment_config.yaml")
    
    # Test volumes from original VOLUME_TOLERANCE_RANGES
    test_cases = [
        # Format: (volume_ml, expected_tolerance_pct)
        (0.0005, 10.0),  # 0.5μL -> 0-0.999μL range -> 10%
        (0.0009, 10.0),  # 0.9μL -> 0-0.999μL range -> 10%
        (0.005, 5.0),    # 5μL -> 1-19μL range -> 5%
        (0.019, 5.0),    # 19μL -> 1-19μL range -> 5%
        (0.030, 3.0),    # 30μL -> 20-59μL range -> 3%
        (0.059, 3.0),    # 59μL -> 20-59μL range -> 3%
        (0.100, 2.0),    # 100μL -> 60-199μL range -> 2%
        (0.199, 2.0),    # 199μL -> 60-199μL range -> 2%
        (0.500, 1.0),    # 500μL -> 200-1000μL range -> 1%
        (1.000, 1.0),    # 1000μL -> 200-1000μL range -> 1%
    ]
    
    print("Testing Volume Tolerance Calculations")
    print("=" * 50)
    print(f"{'Volume (mL)':<12} {'Expected %':<12} {'Calculated %':<12} {'Match':<8}")
    print("-" * 50)
    
    all_passed = True
    
    for volume_ml, expected_pct in test_cases:
        # Test in non-simulation mode first
        config._config['execution']['simulate'] = False
        tolerances = config.calculate_tolerances_for_volume(volume_ml)
        
        # Convert accuracy tolerance back to percentage
        calculated_pct = (tolerances.accuracy_tolerance_ul / (volume_ml * 1000)) * 100
        
        match = abs(calculated_pct - expected_pct) < 0.01
        if not match:
            all_passed = False
            
        print(f"{volume_ml:<12.3f} {expected_pct:<12.1f} {calculated_pct:<12.1f} {'✓' if match else '✗':<8}")
    
    print("\nTesting Simulation Multipliers")
    print("-" * 50)
    
    # Test simulation mode multipliers
    config._config['execution']['simulate'] = True
    test_volume = 0.050  # 50μL
    tolerances = config.calculate_tolerances_for_volume(test_volume)
    
    # Expected: 50μL in 20-60μL range = 3% base tolerance
    # With simulation multipliers: dev * 2.0, var * 2.0
    expected_accuracy_ul = (test_volume * 1000 * 3.0 / 100) * 2.0  # 3.0μL
    expected_precision_pct = 3.0 * 2.0  # 6.0%
    
    accuracy_match = abs(tolerances.accuracy_tolerance_ul - expected_accuracy_ul) < 0.01
    precision_match = abs(tolerances.precision_tolerance_pct - expected_precision_pct) < 0.01
    
    print(f"50μL simulation accuracy: {tolerances.accuracy_tolerance_ul:.2f}μL (expected {expected_accuracy_ul:.2f}μL) {'✓' if accuracy_match else '✗'}")
    print(f"50μL simulation precision: {tolerances.precision_tolerance_pct:.1f}% (expected {expected_precision_pct:.1f}%) {'✓' if precision_match else '✗'}")
    
    if accuracy_match and precision_match:
        print("\n✅ All tolerance calculations match expected values!")
    else:
        print("\n❌ Some tolerance calculations failed!")
        all_passed = False
    
    return all_passed

if __name__ == "__main__":
    test_tolerance_calculations()
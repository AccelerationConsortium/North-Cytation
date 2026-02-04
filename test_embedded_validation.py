#!/usr/bin/env python3
"""
Test script for embedded calibration validation with quality control
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from master_usdl_coordinator import Lash_E
from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy, validate_reservoir_accuracy

def test_embedded_validation():
    """Test embedded validation with simulation mode"""
    
    # Initialize in simulation mode for testing
    SIMULATE = True
    vial_file = "status/test_vials.csv" 
    
    print("=== Testing Embedded Calibration Validation ===")
    print(f"Simulation Mode: {SIMULATE}")
    
    try:
        # Initialize coordinator
        lash_e = Lash_E(vial_file, simulate=SIMULATE)
        print("✓ Lash_E coordinator initialized")
        
        # Test vial-to-vial validation
        print("\n--- Testing Vial-to-Vial Validation ---")
        vial_results = validate_pipetting_accuracy(
            lash_e=lash_e,
            source_vial="source_vial",
            destination_vial="dest_vial", 
            liquid_type="water",
            volumes_ml=[0.01, 0.02],  # Small test volumes
            replicates=2,  # Quick test
            output_folder="test_output",
            save_raw_data=False  # Skip file output for test
        )
        
        print("✓ Vial-to-vial validation completed")
        print(f"  - R²: {vial_results['overall_stats']['r_squared']:.4f}")
        print(f"  - Mean accuracy: {vial_results['overall_stats']['mean_accuracy_pct']:.2f}%")
        
        # Test reservoir validation
        print("\n--- Testing Reservoir Validation ---")
        reservoir_results = validate_reservoir_accuracy(
            lash_e=lash_e,
            reservoir_index=1,
            target_vial="dest_vial",
            liquid_type="water", 
            volumes_ml=[0.05, 0.1],  # Small test volumes
            replicates=2,  # Quick test
            output_folder="test_output",
            save_raw_data=False  # Skip file output for test
        )
        
        print("✓ Reservoir validation completed")
        print(f"  - R²: {reservoir_results['overall_stats']['r_squared']:.4f}")
        print(f"  - Mean accuracy: {reservoir_results['overall_stats']['mean_accuracy_pct']:.2f}%")
        
        print("\n=== All Tests Passed ===")
        print("Embedded validation with quality control is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedded_validation()
    sys.exit(0 if success else 1)
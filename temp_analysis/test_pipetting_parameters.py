#!/usr/bin/env python3
"""
Test script to demonstrate the new standardized pipetting parameters system.
This shows how the old parameter names are still supported through backward compatibility,
but the new PipettingParameters class provides consistency.
"""

from pipetting_data.pipetting_parameters import PipettingParameters, create_standard_parameters

def test_parameter_standardization():
    """Test the parameter standardization and backward compatibility."""
    
    print("=== Testing PipettingParameters Standardization ===\n")
    
    # 1. Create standard parameters
    print("1. Standard parameters:")
    standard_params = PipettingParameters()  # Default parameters
    print(f"   wait_time_after_aspirate: {standard_params.wait_time_after_aspirate}")
    print(f"   wait_time_after_dispense: {standard_params.wait_time_after_dispense}")
    print(f"   aspirate_speed: {standard_params.aspirate_speed}")
    print(f"   move_to_aspirate: {standard_params.move_to_aspirate}")
    print()
    
    # 1b. Show standard parameter sets
    print("1b. Available standard parameter sets:")
    standard_sets = create_standard_parameters()
    for name, params in standard_sets.items():
        print(f"   {name}: wait_time_after_aspirate={params.wait_time_after_aspirate}")
    print()
    
    # 2. Test parameter overrides
    print("2. Parameter overrides:")
    custom_params = standard_params.copy_with_overrides(
        wait_time_after_aspirate=2.5,
        aspirate_speed=500,
        move_to_aspirate=False
    )
    print(f"   wait_time_after_aspirate: {custom_params.wait_time_after_aspirate}")
    print(f"   aspirate_speed: {custom_params.aspirate_speed}")
    print(f"   move_to_aspirate: {custom_params.move_to_aspirate}")
    print()
    
    # 3. Test merging parameters
    print("3. Merging parameters:")
    fast_params = PipettingParameters(
        aspirate_speed=800,
        dispense_speed=800,
        wait_time_after_aspirate=0.5,
        wait_time_after_dispense=0.5
    )
    
    merged_params = standard_params.merge(fast_params)
    print(f"   aspirate_speed (from fast_params): {merged_params.aspirate_speed}")
    print(f"   wait_time_after_aspirate (from fast_params): {merged_params.wait_time_after_aspirate}")
    print(f"   move_to_aspirate (from standard): {merged_params.move_to_aspirate}")
    print()
    
    # 4. Show parameter consistency across methods
    print("4. Parameter consistency demonstration:")
    print("   Before: Different parameter names across methods")
    print("     aspirate_from_vial(wait_time=1)")
    print("     pipet_aspirate(settling_time=1)")
    print("     dispense_into_vial(initial_move=True)")
    print("     dispense_into_wellplate(wait_time=1)")
    print()
    print("   After: Consistent parameter names")
    print("     aspirate_from_vial(parameters=PipettingParameters(wait_time_after_aspirate=1))")
    print("     pipet_aspirate(parameters=PipettingParameters(wait_time_after_aspirate=1))")
    print("     dispense_into_vial(parameters=PipettingParameters(move_to_dispense=True))")
    print("     dispense_into_wellplate(parameters=PipettingParameters(wait_time_after_dispense=1))")
    print()
    
    # 5. Show backward compatibility
    print("5. Backward compatibility:")
    print("   Old code still works with deprecated parameters:")
    print("     aspirate_from_vial(source_vial, 0.1, wait_time=2)  # Still works!")
    print("   But new code can use standardized parameters:")
    print("     params = PipettingParameters(wait_time_after_aspirate=2)")
    print("     aspirate_from_vial(source_vial, 0.1, parameters=params)  # New way!")
    print()
    
    print("=== Parameter Standardization Complete! ===")
    print("\nBenefits:")
    print("- Consistent parameter names across all pipetting methods")
    print("- Easy parameter reuse and customization")
    print("- Backward compatibility with existing code")
    print("- Clear separation of concerns (speed, timing, movement, etc.)")
    print("- Type safety and IDE autocompletion")

if __name__ == "__main__":
    test_parameter_standardization()
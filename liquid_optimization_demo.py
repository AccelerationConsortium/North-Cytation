#!/usr/bin/env python3
"""
Test liquid-specific parameter optimization integration

This script demonstrates the new intelligent parameter resolution system:
defaults → liquid-calibrated → user overrides
"""

# Example usage patterns for the new liquid parameter integration

def demo_usage_patterns():
    """Demonstrate different usage patterns with liquid-specific parameters"""
    
    # Example 1: Basic usage with liquid-specific optimization
    print("Example 1: Basic liquid-specific parameters")
    print("lash_e.nr_robot.aspirate_from_vial('water_vial', 0.1, liquid='water')")
    print("# System will: Use defaults → Apply water calibration → No user overrides")
    print()
    
    # Example 2: User overrides on top of liquid calibration  
    print("Example 2: User overrides with liquid calibration")
    print("from pipetting_data.pipetting_parameters import PipettingParameters")
    print("custom_params = PipettingParameters(aspirate_speed=15)  # Override speed")
    print("lash_e.nr_robot.aspirate_from_vial('ethanol_vial', 0.05, parameters=custom_params, liquid='ethanol')")
    print("# System will: Use defaults → Apply ethanol calibration → Override aspirate_speed=15")
    print()
    
    # Example 3: Backward compatibility - no liquid specified
    print("Example 3: Backward compatibility (no changes needed)")
    print("lash_e.nr_robot.aspirate_from_vial('generic_vial', 0.2)")
    print("# System will: Use defaults → No calibration → No user overrides")
    print()
    
    # Example 4: Full workflow with liquid-specific parameters
    print("Example 4: Complete workflow with liquid optimization")
    print("# Transfer water with optimized parameters")
    print("lash_e.nr_robot.dispense_from_vial_into_vial('water_source', 'reaction_vial', 0.1, liquid='water')")
    print("# Mix ethanol with optimized parameters")  
    print("lash_e.nr_robot.mix_vial('ethanol_vial', 0.05, repeats=5, liquid='ethanol')")
    print("# Wellplate operations with calibrated parameters")
    print("lash_e.nr_robot.pipet_from_wellplate(0, 0.02, liquid='buffer', aspirate=True)")
    print()

def demo_calibration_benefits():
    """Show the benefits of liquid-specific calibration"""
    
    print("Calibration Benefits:")
    print("- Water: Slower aspirate speeds to prevent bubbles")
    print("- Ethanol: Faster speeds with longer wait times for evaporation")
    print("- Viscous liquids: Slower speeds, longer wait times")
    print("- Volatile solvents: Minimal air gaps, faster dispensing")
    print("- Automated parameter lookup based on volume and liquid type")
    print("- Graceful fallback if calibration data unavailable")
    print()

if __name__ == "__main__":
    print("=== Liquid-Specific Parameter Optimization Demo ===")
    print()
    
    demo_usage_patterns()
    demo_calibration_benefits()
    
    print("Implementation Summary:")
    print("✅ Added PipettingWizard import to North_Safe.py")
    print("✅ Added _get_optimized_parameters() helper method")
    print("✅ Updated 8 liquid handling methods with liquid parameter")
    print("✅ Implemented intelligent parameter hierarchy")
    print("✅ Maintained backward compatibility")
    print("✅ Added graceful error handling for missing calibration")
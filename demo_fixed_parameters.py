#!/usr/bin/env python3
"""
Demo script for the fixed parameters feature in calibration_sdl_simplified.py

This demonstrates how to run calibration experiments with fixed parameters.
"""

import sys
import os
sys.path.append("workflows")

def demo_fixed_parameters():
    """Demonstrate the fixed parameters feature with different scenarios."""
    
    print("🎯 FIXED PARAMETERS FEATURE DEMO")
    print("="*60)
    
    # Import the workflow function
    from calibration_sdl_simplified import run_simplified_calibration_workflow
    
    print("\n📋 SCENARIO 1: Standard optimization (no fixed parameters)")
    print("   - All parameters will be optimized")
    print("   - This is the default behavior")
    
    # We'll just show the config summary, not run the full workflow
    from calibration_sdl_simplified import reset_config_to_defaults, get_current_config_summary
    
    reset_config_to_defaults()
    get_current_config_summary()
    
    print("\n" + "="*60)
    print("\n📋 SCENARIO 2: Water experiment with fixed parameters")
    print("   - post_asp_air_vol fixed at 0.1 mL")
    print("   - Both wait times fixed at 0 seconds")
    print("   - Only remaining parameters will be optimized")
    
    # Simulate what would happen with fixed parameters
    import calibration_sdl_simplified as workflow
    workflow.FIXED_PARAMETERS = {
        'post_asp_air_vol': 0.1,
        'aspirate_wait_time': 0,
        'dispense_wait_time': 0
    }
    get_current_config_summary()
    
    print("\n📝 To run this configuration, you would call:")
    print("""
optimal_conditions, save_dir = run_simplified_calibration_workflow(
    vial_mode="legacy",
    liquid="water",
    simulate=True,
    volumes=[0.05, 0.025],
    fixed_parameters={
        'post_asp_air_vol': 0.1,  # Fixed at 0.1 mL
        'aspirate_wait_time': 0,  # No aspirate wait time
        'dispense_wait_time': 0   # No dispense wait time
    }
)""")
    
    print("\n" + "="*60)
    print("\n📋 SCENARIO 3: Glycerol experiment with only one fixed parameter")
    print("   - Only retract_speed fixed at 5.0")
    print("   - All other parameters will be optimized")
    
    workflow.FIXED_PARAMETERS = {'retract_speed': 5.0}
    get_current_config_summary()
    
    print("\n📝 To run this configuration, you would call:")
    print("""
optimal_conditions, save_dir = run_simplified_calibration_workflow(
    vial_mode="legacy",
    liquid="glycerol",
    simulate=True,
    volumes=[0.05, 0.025, 0.01],
    fixed_parameters={
        'retract_speed': 5.0  # Fixed retract speed
    }
)""")
    
    print("\n" + "="*60)
    print("\n📋 USAGE EXAMPLES:")
    print("""
# Example 1: Fast water experiments (no wait times)
run_simplified_calibration_workflow(
    liquid="water",
    fixed_parameters={
        'aspirate_wait_time': 0,
        'dispense_wait_time': 0
    }
)

# Example 2: Conservative air volume handling
run_simplified_calibration_workflow(
    liquid="ethanol", 
    fixed_parameters={
        'post_asp_air_vol': 0.05,  # Fixed small air volume
        'retract_speed': 3.0       # Slow, careful retraction
    }
)

# Example 3: Focus only on speeds and volumes
run_simplified_calibration_workflow(
    liquid="glycerol",
    fixed_parameters={
        'aspirate_wait_time': 0,
        'dispense_wait_time': 0,
        'retract_speed': 10.0,
        'post_asp_air_vol': 0.0    # No air volume
    }
)

# Example 4: Fix everything except overaspirate (volume compensation only)
run_simplified_calibration_workflow(
    liquid="water",
    fixed_parameters={
        'aspirate_speed': 25,
        'dispense_speed': 25,
        'aspirate_wait_time': 0,
        'dispense_wait_time': 0,
        'retract_speed': 8.0,
        'blowout_vol': 0.02,
        'post_asp_air_vol': 0.0
        # overaspirate_vol will still be optimized
    }
)""")
    
    print(f"\n✅ Fixed parameters feature is ready to use!")
    print(f"📖 The system will automatically:")
    print(f"   - Remove fixed parameters from optimization")
    print(f"   - Use fixed values throughout the experiment")
    print(f"   - Save fixed parameters in the configuration")
    print(f"   - Show parameter allocation in the logs")

if __name__ == "__main__":
    demo_fixed_parameters()
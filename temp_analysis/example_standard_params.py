"""
Example of how to use the standard parameter sets
"""

from pipetting_data.pipetting_parameters import PipettingParameters, create_standard_parameters

def show_standard_parameters():
    """Show how to use the standard parameter sets"""
    
    print("📋 Available Standard Parameter Sets:")
    print("=" * 50)
    
    # Get all standard parameter sets
    standards = create_standard_parameters()
    
    for name, params in standards.items():
        print(f"\n🔧 {name}:")
        print(f"   aspirate_speed: {params.aspirate_speed}")
        print(f"   dispense_speed: {params.dispense_speed}")
        print(f"   aspirate_wait_time: {params.aspirate_wait_time}")
        print(f"   dispense_wait_time: {params.dispense_wait_time}")
        print(f"   pre_asp_air_vol: {params.pre_asp_air_vol}")
        print(f"   blowout_vol: {params.blowout_vol}")
    
    print("\n" + "=" * 50)
    print("💡 Usage Examples:")
    print()
    
    # Example 1: Use a standard set directly
    print("1. Use standard parameters directly:")
    slow_params = standards['slow_careful']
    print(f"   slow_params = standards['slow_careful']")
    print(f"   robot.aspirate_from_vial('vial_1', 0.5, parameters=slow_params)")
    print()
    
    # Example 2: Modify a standard set
    print("2. Modify a standard set:")
    custom_slow = standards['slow_careful'].copy_with_overrides(aspirate_wait_time=5.0)
    print(f"   custom = standards['slow_careful'].copy_with_overrides(aspirate_wait_time=5.0)")
    print(f"   robot.aspirate_from_vial('vial_1', 0.5, parameters=custom)")
    print(f"   → aspirate_wait_time: {custom_slow.aspirate_wait_time} (was {slow_params.aspirate_wait_time})")
    print()
    
    # Example 3: For workflows
    print("3. In calibration workflows:")
    print("   # Instead of building params manually:")
    print("   params = standards['precise_small_volume']")
    print("   aspirate_params = params.copy_with_overrides(pre_asp_air_vol=0.01)")
    print("   dispense_params = params.copy_with_overrides(measure_weight=True)")

def workflow_example():
    """Show how the workflow could be simplified using standard parameters"""
    
    print("\n🚀 Workflow Simplification Example:")
    print("=" * 50)
    
    standards = create_standard_parameters()
    
    # Old way (what calibration_sdl_base.py was doing)
    print("\n❌ OLD WAY (complex parameter building):")
    print("""
    aspirate_kwargs = {
        "aspirate_speed": params["aspirate_speed"],
        "wait_time": params["aspirate_wait_time"],
        "retract_speed": params["retract_speed"],
        "pre_asp_air_vol": pre_air,
        "post_asp_air_vol": post_air,
    }
    """)
    
    # New way (what it could be)
    print("✅ NEW WAY (using standard parameters):")
    print("""
    # Option A: Use standard set + overrides
    base_params = standards['precise_small_volume']
    aspirate_params = base_params.copy_with_overrides(
        pre_asp_air_vol=pre_air,
        post_asp_air_vol=post_air
    )
    
    # Option B: Create custom for specific experiment
    experiment_params = PipettingParameters(
        aspirate_speed=3,
        aspirate_wait_time=2.0,
        pre_asp_air_vol=0.005
    )
    """)

if __name__ == "__main__":
    show_standard_parameters()
    workflow_example()
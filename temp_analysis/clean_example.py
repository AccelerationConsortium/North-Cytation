"""
Example showing the clean, simple approach after eliminating verbose mapping
"""

from pipetting_data.pipetting_parameters import PipettingParameters

def example_usage():
    """Show how the cleaned up system works"""
    
    print("✨ Clean Parameter Usage Examples")
    print("=" * 50)
    
    # Example 1: Use all defaults
    print("\n1. Using all defaults:")
    defaults = PipettingParameters()
    print(f"   aspirate_wait_time: {defaults.aspirate_wait_time}")
    print(f"   dispense_wait_time: {defaults.dispense_wait_time}")
    print(f"   blowout_vol: {defaults.blowout_vol}")
    print("   → nr_robot.dispense_into_vial('vial_1', 0.5)  # Uses all defaults")
    
    # Example 2: Override specific parameters
    print("\n2. Custom parameters for slow, careful pipetting:")
    careful = PipettingParameters(
        aspirate_wait_time=3.0,
        dispense_wait_time=1.0,
        blowout_vol=0.1,
        pre_asp_air_vol=0.05
    )
    print(f"   aspirate_wait_time: {careful.aspirate_wait_time}")
    print(f"   dispense_wait_time: {careful.dispense_wait_time}")
    print(f"   blowout_vol: {careful.blowout_vol}")
    print("   → nr_robot.dispense_into_vial('vial_1', 0.5, parameters=careful)")
    
    # Example 3: Quick override
    print("\n3. Quick parameter override:")
    quick_override = defaults.copy_with_overrides(dispense_wait_time=0.0)
    print(f"   dispense_wait_time: {quick_override.dispense_wait_time} (was {defaults.dispense_wait_time})")
    print("   → fast_params = defaults.copy_with_overrides(dispense_wait_time=0.0)")
    print("   → nr_robot.dispense_into_vial('vial_1', 0.5, parameters=fast_params)")
    
    print("\n" + "=" * 50)
    print("🎯 Key Benefits:")
    print("   ✅ No verbose parameter mapping code")
    print("   ✅ Parameters are just defaults - override when needed")
    print("   ✅ Methods have clean, minimal signatures")
    print("   ✅ Easy to understand and use")
    print("   ✅ No confusing 'overrides' dictionaries")

if __name__ == "__main__":
    example_usage()
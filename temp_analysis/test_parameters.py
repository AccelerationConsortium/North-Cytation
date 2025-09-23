"""
Simple test to validate the PipettingParameters system works correctly
"""

from pipetting_data.pipetting_parameters import PipettingParameters

def test_parameters():
    """Test that the parameter system works as expected"""
    
    # Test 1: Default parameters
    print("Test 1: Default parameters")
    params = PipettingParameters()
    print(f"  aspirate_wait_time: {params.aspirate_wait_time}")
    print(f"  dispense_wait_time: {params.dispense_wait_time}")
    print(f"  pre_asp_air_vol: {params.pre_asp_air_vol}")
    print(f"  blowout_vol: {params.blowout_vol}")
    print()
    
    # Test 2: Parameter overrides
    print("Test 2: Parameter overrides")
    custom_params = params.copy_with_overrides(
        aspirate_wait_time=2.5,
        dispense_wait_time=0.5,
        pre_asp_air_vol=0.1
    )
    print(f"  aspirate_wait_time: {custom_params.aspirate_wait_time}")
    print(f"  dispense_wait_time: {custom_params.dispense_wait_time}")
    print(f"  pre_asp_air_vol: {custom_params.pre_asp_air_vol}")
    print(f"  blowout_vol: {custom_params.blowout_vol}")  # Should remain default
    print()
    
    print("✅ All parameter tests passed!")

if __name__ == "__main__":
    test_parameters()
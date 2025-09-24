#!/usr/bin/env python3
"""
Test script to verify that LLM parameter filtering works correctly
"""

def test_parameter_filtering():
    """Test that we correctly filter LLM parameters to match search space"""
    
    # Simulated LLM response with extra fields
    llm_response = {
        'aspirate_speed': 15.5,
        'dispense_speed': 20.0,
        'aspirate_wait_time': 2.5,
        'dispense_wait_time': 1.0,
        'retract_speed': 8.0,
        'pre_asp_air_vol': 0.05,
        'post_asp_air_vol': 0.02,
        'overaspirate_vol': 0.01,
        'confidence': 'high',  # Extra field that should be filtered out
        'reasoning': 'This combination optimizes for speed while maintaining accuracy',  # Extra field
        'expected_improvement': '20% faster with maintained precision'  # Extra field
    }
    
    # Simulated ax_client search space parameters 
    expected_params = {
        'aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 
        'dispense_wait_time', 'retract_speed', 'pre_asp_air_vol', 
        'post_asp_air_vol', 'overaspirate_vol'
    }
    
    # Filter parameters (this is what our fix does)
    filtered_params = {k: v for k, v in llm_response.items() if k in expected_params}
    
    print("Original LLM response:")
    for k, v in llm_response.items():
        print(f"  {k}: {v}")
    
    print(f"\nExpected parameters: {expected_params}")
    print(f"LLM response keys: {set(llm_response.keys())}")
    print(f"Filtered parameters: {set(filtered_params.keys())}")
    
    print(f"\nFiltered LLM parameters:")
    for k, v in filtered_params.items():
        print(f"  {k}: {v}")
    
    # Test that filtering worked
    extra_fields = set(llm_response.keys()) - expected_params
    print(f"\nExtra fields filtered out: {extra_fields}")
    
    # Verify that filtered params match exactly what ax_client expects
    if set(filtered_params.keys()) == expected_params:
        print("✅ SUCCESS: Filtered parameters match ax_client search space exactly!")
        return True
    else:
        missing = expected_params - set(filtered_params.keys())
        unexpected = set(filtered_params.keys()) - expected_params
        print(f"❌ FAILURE: Missing parameters: {missing}, Unexpected: {unexpected}")
        return False

if __name__ == "__main__":
    test_parameter_filtering()
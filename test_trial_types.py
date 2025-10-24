#!/usr/bin/env python3
"""
Test script to verify that trial_type is being set correctly in raw measurements
"""
import sys
sys.path.append("workflows")

from calibration_sdl_base import pipet_and_measure
from master_usdl_coordinator import Lash_E
import tempfile
import os

# Test parameters
test_params = {
    "aspirate_speed": 10,
    "dispense_speed": 10, 
    "aspirate_wait_time": 5,
    "dispense_wait_time": 5,
    "retract_speed": 8,
    "blowout_vol": 0.05,
    "post_asp_air_vol": 0.02,
    "overaspirate_vol": 0.001
}

# Test each trial type
test_cases = [
    ("SCREENING", "Test screening trial"),
    ("OPTIMIZATION", "Test optimization trial"), 
    ("PRECISION", "Test precision trial"),
    ("OVERVOLUME_ASSAY", "Test overvolume trial")
]

print("🧪 Testing trial_type assignment in raw measurements...")
print("=" * 60)

# Setup simulation
vial_file = "status/calibration_vials_short.csv"
lash_e = Lash_E(vial_file, simulate=True, initialize_biotek=False)

for trial_type, description in test_cases:
    print(f"\n📋 {description} (trial_type='{trial_type}')")
    
    raw_measurements = []
    
    # Create temporary file for this test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        # Call pipet_and_measure with the specific trial_type
        result = pipet_and_measure(
            lash_e=lash_e,
            source_vial="source_vial", 
            dest_vial="measurement_vial",
            volume=0.1,  # 100 µL
            params=test_params,
            expected_measurement=0.1,
            expected_time=30,
            replicate_count=2,  # Test with 2 replicates
            simulate=True,
            raw_path=temp_path,
            raw_measurements=raw_measurements,
            liquid="water",
            new_pipet_each_time=False,
            trial_type=trial_type
        )
        
        # Check the results
        print(f"   ✅ Generated {len(raw_measurements)} measurement records")
        
        # Verify each measurement has the correct trial_type
        for i, measurement in enumerate(raw_measurements):
            actual_trial_type = measurement.get('trial_type', 'MISSING')
            if actual_trial_type == trial_type:
                print(f"   ✅ Replicate {i}: trial_type = '{actual_trial_type}' ✓")
            else:
                print(f"   ❌ Replicate {i}: Expected '{trial_type}', got '{actual_trial_type}'")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
    finally:
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass

print(f"\n{'='*60}")
print("🎉 Test completed! Check results above.")
print("\nNow when you run the full workflow, each trial should have the correct type:")
print("  • SCREENING trials during initial parameter exploration")  
print("  • OPTIMIZATION trials during parameter refinement")
print("  • PRECISION trials during final validation")
print("  • OVERVOLUME_ASSAY trials during overvolume calibration")
#!/usr/bin/env python3
"""
Test External Data Loading Pipeline
==================================

This script tests whether the external data loading system correctly:
1. Loads external_calibration_data.csv 
2. Creates trial results with proper statistics
3. Validates that manual measurements can compete for optimal selection

Run this to verify the pipeline without running full calibration.
"""

import sys
from pathlib import Path

# Add calibration_modular_v2 to path
sys.path.insert(0, str(Path(__file__).parent / "calibration_modular_v2"))

# Change directory to calibration_modular_v2 for relative imports
import os
original_dir = os.getcwd()
os.chdir(Path(__file__).parent / "calibration_modular_v2")

try:
    from external_data import ExternalDataLoader
    from config_manager import ExperimentConfig
finally:
    # Change back to original directory
    os.chdir(original_dir)

import pandas as pd

def test_external_data_pipeline():
    """Test the complete external data loading pipeline."""
    
    print("="*60)
    print("TESTING EXTERNAL DATA LOADING PIPELINE")
    print("="*60)
    
    try:
        # 1. Load configuration
        print("\n1. Loading experiment configuration...")
        config = ExperimentConfig("calibration_modular_v2/experiment_config.yaml")
        
        # Check if external data is enabled
        external_enabled = config.is_external_data_enabled()
        external_path = config.get_external_data_path()
        
        print(f"   External data enabled: {external_enabled}")
        print(f"   External data path: {external_path}")
        
        if not external_enabled:
            print("   ❌ External data is disabled in config!")
            return False
            
        # 2. Test ExternalDataLoader
        print("\n2. Initializing ExternalDataLoader...")
        loader = ExternalDataLoader(config)
        
        print(f"   Data loaded successfully: {loader.has_valid_data()}")
        if loader.has_valid_data():
            print(f"   Number of external data rows: {len(loader.data)}")
            
            # Show the loaded data
            print("\n   External Data Contents:")
            print(loader.data.to_string(index=False))
        else:
            print("   ❌ No valid external data loaded!")
            return False
            
        # 3. Generate screening trials
        print("\n3. Generating screening trials from external data...")
        target_volume = 0.05  # 50 uL
        
        trials = loader.generate_screening_trials(target_volume_ml=target_volume)
        
        if trials:
            print(f"   ✅ Generated {len(trials)} screening trials")
            
            # Show trial details
            for i, trial in enumerate(trials):
                print(f"\n   Trial {i+1}:")
                print(f"     Trial ID: {trial.trial_id}")
                print(f"     Parameters: {len(trial.parameters)} parameters")
                print(f"     Measurements: {len(trial.measurements)} measurements")
                print(f"     Mean Volume: {trial.mean_volume_ml:.6f} mL")
                print(f"     Accuracy: {trial.accuracy_pct:.3f}%")
                print(f"     Precision (CV): {trial.precision_pct:.3f}%")
                print(f"     Duration: {trial.mean_time_s:.3f}s")
                print(f"     Quality: {trial.quality_evaluation}")
                
                # Show parameter values
                print("     Parameter Values:")
                for param_name, param_value in trial.parameters.items():
                    print(f"       {param_name}: {param_value}")
                    
        else:
            print("   ❌ No screening trials generated!")
            return False
            
        # 4. Test Ax integration format
        print("\n4. Testing Ax optimizer integration...")
        
        # Check if trials have proper format for Ax
        for i, trial in enumerate(trials):
            # Verify required fields
            required_fields = ['trial_id', 'parameters', 'mean_volume_ml', 'accuracy_pct', 'precision_pct']
            missing_fields = [field for field in required_fields if not hasattr(trial, field)]
            
            if missing_fields:
                print(f"   ❌ Trial {i+1} missing fields: {missing_fields}")
                return False
                
            # Verify parameter format matches what Ax expects
            param_keys = set(trial.parameters.keys())
            expected_keys = {
                'overaspirate_vol', 'aspirate_speed', 'dispense_speed', 'retract_speed',
                'aspirate_wait_time', 'dispense_wait_time', 'pre_asp_air_vol', 
                'post_asp_air_vol', 'blowout_vol'
            }
            
            if not expected_keys.issubset(param_keys):
                missing_params = expected_keys - param_keys
                print(f"   ❌ Trial {i+1} missing parameters: {missing_params}")
                return False
                
        print("   ✅ All trials have proper Ax integration format")
        
        # 5. Summary
        print("\n" + "="*60)
        print("EXTERNAL DATA PIPELINE TEST RESULTS")
        print("="*60)
        print(f"✅ Configuration loaded successfully")
        print(f"✅ External data enabled and file found")  
        print(f"✅ {len(loader.data)} external data entries loaded")
        print(f"✅ {len(trials)} screening trials generated")
        print(f"✅ All trials have proper Ax optimizer format")
        print(f"✅ Manual measurements ready to compete for optimal selection!")
        
        print(f"\n🎯 CONCLUSION: Your external data pipeline is working correctly!")
        print(f"   When you run optimization, it will use these {len(trials)} trials")
        print(f"   instead of random SOBOL trials for kickstart.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR in external data pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_external_data_pipeline()
    sys.exit(0 if success else 1)
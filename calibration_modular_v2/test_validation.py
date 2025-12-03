#!/usr/bin/env python3
"""
Test script for the validation system.
Runs a quick validation test to ensure the system is working properly.
"""

import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration_modular_v2 import ExperimentConfig
from calibration_modular_v2.run_validation import ValidationRunner


def create_test_optimal_conditions():
    """Create a test optimal_conditions.csv file for validation testing."""
    # Create test data similar to calibration results
    volumes_ml = [0.010, 0.020, 0.030, 0.050, 0.075, 0.100]  # 10-100 uL
    
    data = []
    for vol in volumes_ml:
        # Generate realistic parameters based on volume
        overaspirate = max(0.002, vol * 0.05)  # 2-5 uL overaspirate
        aspirate_speed = 50 + vol * 200  # Speed increases with volume
        dispense_speed = 30 + vol * 150
        
        row = {
            'volume_target_ml': vol,
            'overaspirate_vol': overaspirate,
            'aspirate_speed': aspirate_speed,
            'dispense_speed': dispense_speed,
            'accuracy_score': 0.95 + np.random.normal(0, 0.02),  # Realistic accuracy
            'precision_score': 0.90 + np.random.normal(0, 0.03)   # Realistic precision
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def create_test_config(optimal_conditions_file: str) -> str:
    """Create a test experiment_config.yaml for validation."""
    config_content = f"""
experiment:
  name: "validation_test"
  description: "Test run for validation system"
  
hardware:
  protocol_name: "simulation"  # Use simulation for testing
  
liquid:
  name: "water"
  
calibration:
  target_volumes_ml: [0.010, 0.050, 0.100]
  replicates: 3
  
validation:
  volumes_ml: [0.015, 0.025, 0.040, 0.080]  # Different volumes from calibration
  replicates_per_volume: 3
  optimal_conditions_file: "{optimal_conditions_file}"
  output_directory: "validation"
  generate_plots: true
  success_criteria:
    max_deviation_pct: 5.0   # 5% accuracy requirement
    max_cv_pct: 10.0         # 10% precision requirement
"""
    return config_content


def test_validation_system():
    """Test the validation system with simulated data."""
    print("🧪 Testing Validation System")
    print("=" * 40)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test optimal conditions file
            print("📊 Creating test calibration results...")
            optimal_conditions = create_test_optimal_conditions()
            optimal_conditions_file = temp_path / "optimal_conditions.csv"
            optimal_conditions.to_csv(optimal_conditions_file, index=False)
            print(f"   Created: {optimal_conditions_file}")
            print(f"   Volumes: {optimal_conditions['volume_target_ml'].tolist()}")
            
            # Create test configuration
            print("⚙️ Creating test configuration...")
            config_content = create_test_config(str(optimal_conditions_file))
            config_file = temp_path / "experiment_config.yaml"
            with open(config_file, 'w') as f:
                f.write(config_content)
            print(f"   Created: {config_file}")
            
            # Load configuration
            print("🔧 Loading configuration...")
            config = ExperimentConfig.from_yaml(str(config_file))
            print(f"   Protocol: {config.get_protocol_name()}")
            print(f"   Simulation: {config.is_simulation()}")
            
            # Run validation
            print("🚀 Running validation...")
            validator = ValidationRunner(config)
            results = validator.run_validation()
            
            # Display results
            print("📈 Validation Results:")
            analysis = results['analysis']
            if 'summary' in analysis:
                summary = analysis['summary']
                print(f"   Volumes tested: {summary['total_volumes_tested']}")
                print(f"   Volumes passed: {summary['volumes_passed']}")
                print(f"   Pass rate: {summary['overall_pass_rate']:.1%}")
                print(f"   Results saved to: {results['output_dir']}")
                
                if summary['overall_pass_rate'] >= 0.8:
                    print("✅ Test PASSED")
                    return True
                else:
                    print("❌ Test FAILED")
                    return False
            else:
                print("❌ Test FAILED - no analysis results")
                return False
                
    except Exception as e:
        print(f"❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_validation_system()
    if success:
        print("\\n🎉 Validation system is working correctly!")
        print("\\n📝 To use validation in your experiments:")
        print("   1. Configure validation section in experiment_config.yaml")
        print("   2. Run: python calibration_modular_v2/run_validation.py")
        print("   3. Check validation/ directory for results")
    else:
        print("\\n💥 Validation system test failed!")
        print("   Check error messages and fix issues before using")
    
    sys.exit(0 if success else 1)
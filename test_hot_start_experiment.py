#!/usr/bin/env python3
"""
Hot Start Calibration Experiment

This script demonstrates how to use the unified dataset for hot starting
the Bayesian optimization model. This is an EXPERIMENTAL configuration
that loads prior knowledge from previous experiments.

Usage:
    python test_hot_start_experiment.py

What this does:
- Loads previous calibration data from unified_dataset_water.csv
- Uses this data to "pre-train" the Bayesian optimizer
- Should converge faster than starting from scratch
- Compares performance with and without hot starting
"""

import sys
import os
sys.path.append('.')

# Import the main calibration workflow
from workflows.calibration_sdl_simplified import *

def test_hot_start_experiment():
    """
    Run a calibration experiment using hot start from unified dataset.
    
    This modifies the global configuration to use external data loading,
    then runs a standard calibration to see the benefits.
    """
    
    print("🔥 HOT START CALIBRATION EXPERIMENT")
    print("="*50)
    print("This experiment uses prior data to jump-start optimization")
    print()
    
    # Show original config
    print("📋 ORIGINAL CONFIG:")
    get_current_config_summary()
    
    # Modify global config for hot start experiment
    global USE_EXTERNAL_DATA, EXTERNAL_DATA_PATH, EXTERNAL_DATA_LIQUID_FILTER
    
    print(f"\n🔧 ENABLING HOT START...")
    USE_EXTERNAL_DATA = True
    EXTERNAL_DATA_PATH = "pipetting_data/unified_dataset_water.csv"
    EXTERNAL_DATA_LIQUID_FILTER = "water"
    
    print(f"   📁 External data: {EXTERNAL_DATA_PATH}")
    print(f"   🧪 Liquid filter: {EXTERNAL_DATA_LIQUID_FILTER}")
    
    # Show modified config
    print(f"\n📋 HOT START CONFIG:")
    get_current_config_summary()
    
    # Test if the data file exists
    if not os.path.exists(EXTERNAL_DATA_PATH):
        print(f"\n❌ ERROR: External data file not found: {EXTERNAL_DATA_PATH}")
        print("   Make sure the unified dataset is in the correct location")
        return False
    
    # Test loading external data for a specific volume
    test_volume = 0.05  # 50μL
    print(f"\n🧪 TESTING EXTERNAL DATA LOADING for {test_volume*1000:.0f}μL...")
    
    external_results = load_external_calibration_data(test_volume, "water")
    
    if external_results:
        print(f"   ✅ Successfully loaded {len(external_results)} records")
        
        # Show some statistics
        deviations = [r['deviation'] for r in external_results]
        times = [r['time'] for r in external_results]
        
        print(f"   📊 Performance range:")
        print(f"      Deviation: {min(deviations):.1f}% - {max(deviations):.1f}%")
        print(f"      Time: {min(times):.1f}s - {max(times):.1f}s")
        print(f"      Best deviation: {min(deviations):.1f}%")
        
        # Show a few example records
        print(f"\n   📝 Example records:")
        for i, result in enumerate(external_results[:3]):
            print(f"      {i+1}. Dev={result['deviation']:.1f}%, Time={result['time']:.1f}s, "
                  f"Asp={result['aspirate_speed']}, Disp={result['dispense_speed']}")
    else:
        print(f"   ❌ No external data loaded")
        return False
    
    print(f"\n🎯 HOT START EXPERIMENT READY!")
    print(f"   The next calibration run will use {len(external_results)} prior experiments")
    print(f"   This should converge much faster than starting from scratch")
    
    # Ask if user wants to run full calibration
    print(f"\n❓ Run full hot start calibration experiment? (y/N): ", end="")
    response = input().strip().lower()
    
    if response in ['y', 'yes']:
        print(f"\n🚀 Starting hot start calibration...")
        
        try:
            # Run the main calibration workflow
            # This will now use the external data we configured
            main()
        except KeyboardInterrupt:
            print(f"\n⏹️  Experiment interrupted by user")
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
        finally:
            # Reset configuration to defaults
            reset_to_default_config()
    else:
        print(f"\n⏹️  Experiment setup complete but not run")
        print(f"   To run hot start calibration later:")
        print(f"   1. Set USE_EXTERNAL_DATA = True")
        print(f"   2. Set EXTERNAL_DATA_PATH = 'pipetting_data/unified_dataset_water.csv'") 
        print(f"   3. Run main() calibration")
    
    return True

def reset_to_default_config():
    """Reset global configuration back to defaults."""
    global USE_EXTERNAL_DATA, EXTERNAL_DATA_PATH, EXTERNAL_DATA_LIQUID_FILTER
    
    print(f"\n🔄 Resetting to default configuration...")
    USE_EXTERNAL_DATA = DEFAULT_USE_EXTERNAL_DATA
    EXTERNAL_DATA_PATH = DEFAULT_EXTERNAL_DATA_PATH  
    EXTERNAL_DATA_LIQUID_FILTER = DEFAULT_EXTERNAL_DATA_LIQUID_FILTER
    
    print(f"   ✅ Configuration reset to defaults")

def compare_with_without_hot_start():
    """
    Advanced experiment: Compare calibration performance with and without hot starting.
    
    This would run two calibrations:
    1. Cold start (traditional SOBOL screening)  
    2. Hot start (using unified dataset)
    
    And compare convergence speed and final performance.
    """
    print(f"\n🔬 COMPARISON EXPERIMENT: Hot Start vs Cold Start")
    print(f"   This would run calibration twice and compare results:")
    print(f"   1. Cold start: Traditional SOBOL screening (5 initial trials)")
    print(f"   2. Hot start: Using unified dataset ({len(load_external_calibration_data(0.05, 'water'))} prior trials)")
    print(f"   📊 Metrics: Convergence speed, final performance, measurement efficiency")
    print(f"   ⚠️  This is not implemented yet - would need experimental framework")

if __name__ == "__main__":
    print("🧪 Hot Start Calibration Testing")
    print()
    
    # Test the basic hot start functionality  
    success = test_hot_start_experiment()
    
    if success:
        print(f"\n✅ Hot start experiment setup successful!")
        print(f"\nYour unified dataset contains rich prior knowledge that can:")
        print(f"  🚀 Skip initial exploration phase")
        print(f"  📈 Start optimization from known good regions") 
        print(f"  ⚡ Converge to optimal parameters faster")
        print(f"  🎯 Reduce total measurements needed")
    else:
        print(f"\n❌ Hot start experiment setup failed")
        print(f"   Check that unified_dataset_water.csv exists in pipetting_data/")
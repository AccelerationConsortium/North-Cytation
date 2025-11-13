#!/usr/bin/env python3
"""
Test SOBOL parameter diversity in screening phase.
"""

import sys
import os
sys.path.append('calibration_modular_v2')

from calibration_modular_v2.config_manager import ExperimentConfig
from calibration_modular_v2.experiment import CalibrationExperiment

def test_sobol_diversity():
    """Test that SOBOL generates diverse parameters for screening."""
    print("🎲 Testing SOBOL Parameter Diversity")
    print("=" * 50)
    
    # Load default config
    config_path = "calibration_modular_v2/multi_volume_calibration_test.yaml"
    config = ExperimentConfig(config_path)
    
    # Create experiment in simulation mode  
    experiment = CalibrationExperiment(config, simulate=True)
    
    print("\n🎯 Generating 5 screening trials with SOBOL:")
    print("-" * 45)
    
    screening_params = []
    for i in range(5):
        params = experiment._generate_screening_parameters(0.05, trial_idx=i)  # 50μL target
        screening_params.append(params)
        print(f"Trial {i+1}: speed={params.plunger_speed:.3f}, pos={params.mix_position_from_bottom:.3f}, "
              f"asp_wait={params.aspirate_wait_time_s:.3f}")
    
    # Check for diversity
    print("\n📊 Diversity Analysis:")
    print("-" * 20)
    
    speeds = [p.plunger_speed for p in screening_params]
    positions = [p.mix_position_from_bottom for p in screening_params]
    wait_times = [p.aspirate_wait_time_s for p in screening_params]
    
    print(f"Plunger speeds: min={min(speeds):.3f}, max={max(speeds):.3f}, range={max(speeds)-min(speeds):.3f}")
    print(f"Mix positions: min={min(positions):.3f}, max={max(positions):.3f}, range={max(positions)-min(positions):.3f}")  
    print(f"Wait times: min={min(wait_times):.3f}, max={max(wait_times):.3f}, range={max(wait_times)-min(wait_times):.3f}")
    
    # Check if parameters are actually different
    all_same_speed = len(set(f"{s:.6f}" for s in speeds)) == 1
    all_same_pos = len(set(f"{p:.6f}" for p in positions)) == 1
    all_same_wait = len(set(f"{w:.6f}" for w in wait_times)) == 1
    
    if all_same_speed and all_same_pos and all_same_wait:
        print("❌ WARNING: All parameters are identical - SOBOL not working!")
    else:
        print("✅ SUCCESS: Parameters show SOBOL diversity")
        
    return not (all_same_speed and all_same_pos and all_same_wait)

if __name__ == "__main__":
    success = test_sobol_diversity()
    exit(0 if success else 1)
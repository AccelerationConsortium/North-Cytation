#!/usr/bin/env python3
"""
Test SOBOL parameter diversity directly.
"""
import sys
sys.path.append('calibration_modular_v2')

# Import the specific components we need
from calibration_modular_v2.bayesian_recommender import AxBayesianOptimizer

def test_sobol_diversity():
    """Test that SOBOL generates diverse parameters."""
    print("🎲 Testing SOBOL Parameter Diversity")
    print("=" * 50)
    
    # Create a SOBOL optimizer directly
    optimizer = AxBayesianOptimizer(
        parameter_names=['plunger_speed', 'mix_position_from_bottom', 'aspirate_wait_time_s'],
        parameter_bounds={
            'plunger_speed': [1.0, 20.0],
            'mix_position_from_bottom': [0.1, 0.9], 
            'aspirate_wait_time_s': [0.1, 2.0]
        },
        fixed_params=None,
        volume_dependent_only=False
    )
    
    print("\n🎯 Generating 5 SOBOL suggestions:")
    print("-" * 45)
    
    suggestions = []
    for i in range(5):
        params = optimizer.suggest_parameters()
        suggestions.append(params)
        print(f"Trial {i+1}: speed={params.plunger_speed:.3f}, pos={params.mix_position_from_bottom:.3f}, "
              f"wait={params.aspirate_wait_time_s:.3f}")
    
    # Check for diversity
    print("\n📊 Diversity Analysis:")
    print("-" * 20)
    
    speeds = [p.plunger_speed for p in suggestions]
    positions = [p.mix_position_from_bottom for p in suggestions]
    wait_times = [p.aspirate_wait_time_s for p in suggestions]
    
    print(f"Plunger speeds: min={min(speeds):.3f}, max={max(speeds):.3f}, range={max(speeds)-min(speeds):.3f}")
    print(f"Mix positions: min={min(positions):.3f}, max={max(positions):.3f}, range={max(positions)-min(positions):.3f}")  
    print(f"Wait times: min={min(wait_times):.3f}, max={max(wait_times):.3f}, range={max(wait_times)-min(wait_times):.3f}")
    
    # Check if parameters are actually different
    all_same_speed = len(set(f"{s:.6f}" for s in speeds)) == 1
    all_same_pos = len(set(f"{p:.6f}" for p in positions)) == 1
    all_same_wait = len(set(f"{w:.6f}" for w in wait_times)) == 1
    
    if all_same_speed and all_same_pos and all_same_wait:
        print("❌ WARNING: All parameters are identical - SOBOL not working!")
        return False
    else:
        print("✅ SUCCESS: Parameters show SOBOL diversity")
        return True

if __name__ == "__main__":
    success = test_sobol_diversity()
    exit(0 if success else 1)